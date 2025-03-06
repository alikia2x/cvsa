import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoClassifierV6_0(nn.Module):
    def __init__(self, embedding_dim=256, seq_length=1024, hidden_dim=512, output_dim=3):
        super().__init__()
        self.num_channels = 3
        self.channel_names = ['title', 'description', 'tags']
        
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

            # 全局平均池化层
            # 输出形状为 [batch_size, 256, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        
        # 全局池化后的特征维度固定为 256
        self.feature_dim = 256
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
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
        
        # 展平特征（全局池化后形状为 [batch_size, 256, 1, 1]）
        flat_features = conv_features.view(conv_features.size(0), -1)  # [batch_size, 256]
        
        # 全连接层分类
        return self.fc(flat_features)