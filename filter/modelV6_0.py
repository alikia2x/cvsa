import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoClassifierV6_0(nn.Module):
    def __init__(self, embedding_dim=256, seq_length=1024, hidden_dim=512, output_dim=3):
        super().__init__()
        self.num_channels = 4
        self.channel_names = ['title', 'description', 'tags', 'author_info']
        
        # CNN特征提取层
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(self.num_channels, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # 第二层卷积
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # 第三层卷积
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        
        # 计算卷积后的特征维度
        self.feature_dim = self._get_conv_output_size(seq_length, embedding_dim)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _get_conv_output_size(self, seq_length, embedding_dim):
        # 用于计算卷积输出尺寸
        x = torch.zeros(1, self.num_channels, seq_length, embedding_dim)
        x = self.conv_layers(x)
        return x.view(1, -1).size(1)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, channel_features: torch.Tensor):
        """
        输入格式: [batch_size, num_channels, seq_length, embedding_dim]
        输出格式: [batch_size, output_dim]
        """
        # CNN特征提取
        conv_features = self.conv_layers(channel_features)
        
        # 展平特征
        flat_features = conv_features.view(conv_features.size(0), -1)
        
        # 全连接层分类
        return self.fc(flat_features)

# 损失函数保持不变
class AdaptiveRecallLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.8, gamma=2.0, fp_penalty=0.5):
        super().__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        self.fp_penalty = fp_penalty

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        class_mask = F.one_hot(targets, num_classes=len(self.class_weights))
        class_weights = (self.alpha + (1 - self.alpha) * pt.unsqueeze(-1)) * class_mask
        recall_loss = (class_weights * focal_loss.unsqueeze(-1)).sum(dim=1)
        
        probs = F.softmax(logits, dim=1)
        fp_mask = (targets != 0) & (torch.argmax(logits, dim=1) == 0)
        fp_loss = self.fp_penalty * probs[:, 0][fp_mask].pow(2).sum()
        
        total_loss = recall_loss.mean() + fp_loss / len(targets)
        return total_loss