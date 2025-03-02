import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoClassifierV3_12(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=648):
        super().__init__()
        self.num_channels = 4
        self.channel_names = ['title', 'description', 'tags', 'author_info']
        
        # 可学习温度系数
        self.temperature = nn.Parameter(torch.tensor(1.7))
        
        # 带约束的通道权重（使用Sigmoid替代Softmax）
        self.channel_weights = nn.Parameter(torch.ones(self.num_channels))
        
        # 第一个二分类器：0 vs 1/2
        self.first_classifier = nn.Sequential(
            nn.Linear(embedding_dim * self.num_channels, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, 2)  # 输出为2类：0 vs 1/2
        )
        
        # 第二个二分类器：1 vs 2
        self.second_classifier = nn.Sequential(
            nn.Linear(embedding_dim * self.num_channels, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, 2)  # 输出为2类：1 vs 2
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.first_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        
        for layer in self.second_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, channel_features: torch.Tensor):
        """
        输入格式: [batch_size, num_channels, embedding_dim]
        输出格式: [batch_size, output_dim]
        """
        # 自适应通道权重（Sigmoid约束）
        weights = torch.sigmoid(self.channel_weights)  # [0,1]范围
        weighted_features = channel_features * weights.unsqueeze(0).unsqueeze(-1)
        
        # 特征拼接
        combined = weighted_features.view(weighted_features.size(0), -1)
        
        # 第一个二分类器：0 vs 1/2
        first_output = self.first_classifier(combined)
        first_probs = F.softmax(first_output, dim=1)
        
        # 第二个二分类器：1 vs 2
        second_output = self.second_classifier(combined)
        second_probs = F.softmax(second_output, dim=1)
        
        # 合并结果
        final_probs = torch.zeros(channel_features.size(0), 3).to(channel_features.device)
        final_probs[:, 0] = first_probs[:, 0]  # 类别0的概率
        final_probs[:, 1] = first_probs[:, 1] * second_probs[:, 0]  # 类别1的概率
        final_probs[:, 2] = first_probs[:, 1] * second_probs[:, 1]  # 类别2的概率
        
        return final_probs

    def get_channel_weights(self):
        """获取各通道权重（带温度调节）"""
        return torch.softmax(self.channel_weights / self.temperature, dim=0).detach().cpu().numpy()
