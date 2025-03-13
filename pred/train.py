import random
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
from dataset import VideoPlayDataset, collate_fn
from pred.model import CompactPredictor

def train(model, dataloader, device, epochs=100):
    writer = SummaryWriter(f'./pred/runs/play_predictor_{time.strftime("%Y%m%d_%H%M")}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                  total_steps=len(dataloader)*epochs)
    criterion = torch.nn.MSELoss()
    
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
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            global_step += 1
            
            if global_step % 100 == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step)
            if batch_idx % 50 == 0:
                # 监控梯度
                grad_norms = [
                    torch.norm(p.grad).item() 
                    for p in model.parameters() if p.grad is not None
                ]
                writer.add_scalar('Grad/Norm', sum(grad_norms)/len(grad_norms), global_step)
                
                # 监控参数值
                param_means = [torch.mean(p.data).item() for p in model.parameters()]
                writer.add_scalar('Params/Mean', sum(param_means)/len(param_means), global_step)

                samples_count = len(targets)
                r = random.randint(0, samples_count-1)
                t = float(torch.exp2(targets[r])) - 1
                o = float(torch.exp2(outputs[r])) - 1
                d = features[r].cpu().numpy()[0]
                speed = np.exp2(features[r].cpu().numpy()[2])
                time_diff = np.exp2(d) / 3600
                inc = speed * time_diff
                model_error = abs(t - o)
                reg_error = abs(inc - t)
                print(f"{t:07.1f} | {o:07.1f} | {d:07.1f} | {inc:07.1f} | {model_error < reg_error}")
            
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(dataloader):.4f}")
    
    writer.close()
    return model

if __name__ == "__main__":
    device = 'mps'
    
    # 初始化数据集和模型
    dataset = VideoPlayDataset('./data/pred', './data/pred/publish_time.csv')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    
    # 获取特征维度
    sample = next(iter(dataloader))
    input_size = sample['features'].shape[1]
    
    model = CompactPredictor(input_size).to(device)
    trained_model = train(model, dataloader, device, epochs=30)
    
    # 保存模型
    torch.save(trained_model.state_dict(), 'play_predictor.pth')