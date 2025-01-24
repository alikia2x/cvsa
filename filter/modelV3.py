import torch
import torch.nn as nn

class VideoClassifierV3(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=256, output_dim=3):
        super().__init__()
        self.num_channels = 4
        self.channel_names = ['title', 'description', 'tags', 'author_info']
        
        # 改进1：带温度系数的通道权重（比原始固定权重更灵活）
        self.channel_weights = nn.Parameter(torch.ones(self.num_channels))
        self.temperature = 2.0  # 可调节的平滑系数
        
        # 改进2：更稳健的全连接结构
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * self.num_channels, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 改进3：输出层初始化
        nn.init.xavier_uniform_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, input_texts, sentence_transformer):
        # 合并所有通道文本进行批量编码
        all_texts = [text for channel in self.channel_names for text in input_texts[channel]]
        
        # 使用SentenceTransformer生成嵌入（保持冻结）
        with torch.no_grad():
            task = "classification"
            embeddings = torch.tensor(
                sentence_transformer.encode(all_texts, task=task),
                device=next(self.parameters()).device
            )
        
        # 分割嵌入并加权
        split_sizes = [len(input_texts[name]) for name in self.channel_names]
        channel_features = torch.split(embeddings, split_sizes, dim=0)
        channel_features = torch.stack(channel_features, dim=1)  # [batch, 4, 1024]
        
        # 改进4：带温度系数的softmax加权
        weights = torch.softmax(self.channel_weights / self.temperature, dim=0)
        weighted_features = channel_features * weights.unsqueeze(0).unsqueeze(-1)
        
        # 拼接特征
        combined = weighted_features.view(weighted_features.size(0), -1)
        
        # 全连接层
        return self.fc(combined)

    def get_channel_weights(self):
        """获取各通道权重（带温度调节）"""
        return torch.softmax(self.channel_weights / self.temperature, dim=0).detach().cpu().numpy()