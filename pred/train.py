import random
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
from dataset import VideoPlayDataset, collate_fn
from pred.model import CompactPredictor

def asymmetricHuberLoss(delta=1.0, beta=1.3):
    """
    创建一个可调用的非对称 Huber 损失函数。

    参数：
        delta (float): Huber 损失的 delta 参数。
        beta (float): 控制负误差惩罚的系数。

    返回：
        callable: 可调用的损失函数。
    """
    def loss_function(input, target):
        error = input - target
        abs_error = torch.abs(error)

        linear_loss = abs_error - 0.5 * delta
        quadratic_loss = 0.5 * error**2

        loss = torch.where(abs_error < delta, quadratic_loss, linear_loss)
        loss = torch.where(error < 0, beta * loss, loss)

        return torch.mean(loss)

    return loss_function

def train(model, dataloader, device, epochs=100):
    writer = SummaryWriter(f'./pred/runs/play_predictor_{time.strftime("%Y%m%d_%H%M")}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                  total_steps=len(dataloader)*30)
    # Huber loss
    criterion = asymmetricHuberLoss(delta=1.0, beta=2.1)
    
    model.train()
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            global_step += 1
            
            if global_step % 100 == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step)
            if batch_idx % 50 == 0:
                # Monitor gradients
                grad_norms = [
                    torch.norm(p.grad).item() 
                    for p in model.parameters() if p.grad is not None
                ]
                writer.add_scalar('Grad/Norm', sum(grad_norms)/len(grad_norms), global_step)
                
                # Monitor parameter values
                param_means = [torch.mean(p.data).item() for p in model.parameters()]
                writer.add_scalar('Params/Mean', sum(param_means)/len(param_means), global_step)

                samples_count = len(targets)
                good = 0
                for r in range(samples_count):
                    r = random.randint(0, samples_count-1)
                    t = float(torch.exp2(targets[r])) - 1
                    o = float(torch.exp2(outputs[r])) - 1
                    d = features[r].cpu().numpy()[0]
                    speed = np.exp2(features[r].cpu().numpy()[8]) / 6
                    time_diff = np.exp2(d) / 3600
                    inc = speed * time_diff
                    model_error = abs(t - o)
                    reg_error = abs(inc - t)
                    if model_error < reg_error:
                        good += 1
                #print(f"{t:07.1f} | {o:07.1f} | {d:07.1f} | {inc:07.1f} | {good/samples_count*100:.1f}%")
                writer.add_scalar('Train/WinRate', good/samples_count, global_step)
            
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(dataloader):.4f}")
    
    writer.close()
    return model

if __name__ == "__main__":
    device = 'mps'
    
    # Initialize dataset and model
    dataset = VideoPlayDataset('./data/pred', './data/pred/publish_time.csv', 'short')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    
    # Get feature dimension
    sample = next(iter(dataloader))
    input_size = sample['features'].shape[1]
    
    model = CompactPredictor(input_size).to(device)
    trained_model = train(model, dataloader, device, epochs=18)
    
    # Save model
    torch.save(trained_model.state_dict(), f"./pred/checkpoints/model_{time.strftime('%Y%m%d_%H%M')}.pt")
