import numpy as np
from torch.utils.data import DataLoader
from model import MultiTaskWrapper, VideoPlayPredictor
import torch
import torch.nn.functional as F
from dataset import VideoPlayDataset, collate_fn

def train(model, dataloader, epochs=100, device='mps'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    steps = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # movel whole batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # 前向传播
            pred = model(batch)

            y_play = batch['y_play']

            real = np.expm1(y_play.cpu().detach().numpy())
            yhat = np.expm1(pred.cpu().detach().numpy())
            print("real", [int(real[0][0]), int(real[1][0])])
            print("yhat", [int(yhat[0][0]), int(yhat[1][0])], [float(pred.cpu().detach().numpy()[0][0]), float(pred.cpu().detach().numpy()[1][0])])

            # 计算加权损失
            weights = torch.log1p(batch['forecast_span'])  # 时间越长权重越低
            loss_per_sample = F.huber_loss(pred, y_play, reduction='none')
            loss = (loss_per_sample * weights).mean()

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            steps += 1

            print(f"Epoch {epoch+1} | Step {steps} | Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f}")

# 初始化模型
device = 'mps'
model = MultiTaskWrapper(VideoPlayPredictor())
model = model.to(device)

data_dir = './data/pred'
publish_time_path = './data/pred/publish_time.csv'
dataset = VideoPlayDataset(
    data_dir=data_dir,
    publish_time_path=publish_time_path,
    min_seq_len=2,    # 至少2个历史点
    max_seq_len=350,  # 最多350个历史点
    min_forecast_span=60,    # 预测跨度1分钟到
    max_forecast_span=86400 * 10 # 10天
)
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,  # 使用自定义collate函数
)

# 开始训练
train(model, dataloader, epochs=20, device=device)