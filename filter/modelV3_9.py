import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoClassifierV3_9(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=648, output_dim=3):
        super().__init__()
        self.num_channels = 4
        self.channel_names = ['title', 'description', 'tags', 'author_info']
        
        # 可学习温度系数
        self.temperature = nn.Parameter(torch.tensor(1.7))
        
        # 带约束的通道权重（使用Sigmoid替代Softmax）
        self.channel_weights = nn.Parameter(torch.ones(self.num_channels))
        
        # 增强的非线性层
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * self.num_channels, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, output_dim)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                # 使用ReLU的初始化参数（GELU的近似）
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # 修改这里
                
                # 或者使用Xavier初始化（更适合通用场景）
                # nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                
                nn.init.zeros_(layer.bias)


    def forward(self, input_texts, sentence_transformer):
        # 合并文本进行批量编码
        all_texts = [text for channel in self.channel_names for text in input_texts[channel]]
        
        # 冻结的文本编码
        with torch.no_grad():
            embeddings = torch.tensor(
                sentence_transformer.encode(all_texts),
                device=next(self.parameters()).device
            )
        
        # 分割并加权通道特征
        split_sizes = [len(input_texts[name]) for name in self.channel_names]
        channel_features = torch.split(embeddings, split_sizes, dim=0)
        channel_features = torch.stack(channel_features, dim=1)
        
        # 自适应通道权重（Sigmoid约束）
        weights = torch.sigmoid(self.channel_weights)  # [0,1]范围
        weighted_features = channel_features * weights.unsqueeze(0).unsqueeze(-1)
        
        # 特征拼接
        combined = weighted_features.view(weighted_features.size(0), -1)
        
        return self.fc(combined)

    def get_channel_weights(self):
        """获取各通道权重（带温度调节）"""
        return torch.softmax(self.channel_weights / self.temperature, dim=0).detach().cpu().numpy()
    

class AdaptiveRecallLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.8, gamma=2.0, fp_penalty=0.5):
        """
        Args:
            class_weights (torch.Tensor): 类别权重
            alpha (float): 召回率调节因子（0-1）
            gamma (float): Focal Loss参数
            fp_penalty (float): 类别0假阳性惩罚强度
        """
        super().__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        self.fp_penalty = fp_penalty

    def forward(self, logits, targets):
        # 基础交叉熵损失
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        
        # Focal Loss组件
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 召回率增强（对困难样本加权）
        class_mask = F.one_hot(targets, num_classes=len(self.class_weights))
        class_weights = (self.alpha + (1 - self.alpha) * pt.unsqueeze(-1)) * class_mask
        recall_loss = (class_weights * focal_loss.unsqueeze(-1)).sum(dim=1)
        
        # 类别0假阳性惩罚
        probs = F.softmax(logits, dim=1)
        fp_mask = (targets != 0) & (torch.argmax(logits, dim=1) == 0)
        fp_loss = self.fp_penalty * probs[:, 0][fp_mask].pow(2).sum()
        
        # 总损失
        total_loss = recall_loss.mean() + fp_loss / len(targets)
        
        return total_loss