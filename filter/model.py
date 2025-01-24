import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class VideoClassifier(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=256, output_dim=3):
        super().__init__()
        self.num_channels = 4
        self.channel_names = ['title', 'description', 'tags', 'author_info']
        
        # 通道权重参数（可学习）
        self.channel_weights = nn.Parameter(torch.ones(self.num_channels))
        
        # 全连接层
        self.fc1 = nn.Linear(embedding_dim * self.num_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_texts, sentence_transformer):
        # 各通道特征提取
        channel_features = []
        for i, name in enumerate(self.channel_names):
            # 获取当前通道的批量文本
            batch_texts = input_texts[name]
            
            # 使用SentenceTransformer生成嵌入
            embeddings = torch.tensor(sentence_transformer.encode(batch_texts))
            channel_features.append(embeddings)
        
        # 将通道特征堆叠并加权
        channel_features = torch.stack(channel_features, dim=1)  # [batch_size, num_channels, embedding_dim]
        channel_weights = torch.softmax(self.channel_weights, dim=0)
        weighted_features = channel_features * channel_weights.unsqueeze(0).unsqueeze(-1)
        
        # 拼接所有通道特征
        combined_features = weighted_features.view(weighted_features.size(0), -1)  # [batch_size, num_channels * embedding_dim]
        
        # 全连接层
        x = torch.relu(self.fc1(combined_features))
        output = self.fc2(x)
        output = self.log_softmax(output)
        return output

    def get_channel_weights(self):
        """获取各通道的权重（用于解释性分析）"""
        return torch.softmax(self.channel_weights, dim=0).detach().cpu().numpy()
