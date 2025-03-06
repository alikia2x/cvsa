import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoClassifierV6_1(nn.Module):
    def __init__(self, embedding_dim=256, seq_length=1024, hidden_dim=256, output_dim=3, num_heads=4):
        super().__init__()
        self.num_channels = 3
        self.channel_names = ['title', 'description', 'tags']
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim  # 每个通道处理后的特征维度
        
        # 通道独立处理模块（每个通道独立的Transformer编码器）
        self.channel_processors = nn.ModuleList()
        for _ in range(self.num_channels):
            self.channel_processors.append(
                nn.Sequential(
                    # 自注意力层
                    nn.MultiheadAttention(
                        embed_dim=embedding_dim,
                        num_heads=num_heads,
                        dropout=0.1
                    ),
                    # 层归一化和前馈网络
                    nn.LayerNorm(embedding_dim),
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                )
            )
        
        # 通道权重（可学习，Sigmoid约束）
        self.channel_weights = nn.Parameter(torch.ones(self.num_channels))
        
        # 全连接层（扩展维度）
        self.fc = nn.Sequential(
            nn.Linear(self.num_channels * hidden_dim, 1024),  # 拼接后的特征维度
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(512, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化（Xavier初始化）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                # 初始化MultiheadAttention的参数（输入投影和输出投影）
                for name, param in m.named_parameters():
                    if "in_proj" in name or "out_proj" in name:
                        if "weight" in name:
                            nn.init.xavier_uniform_(param)
                        elif "bias" in name:
                            nn.init.zeros_(param)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
    
    def forward(self, channel_features: torch.Tensor):
        """
        输入格式: [batch_size, num_channels, seq_length, embedding_dim]
        输出格式: [batch_size, output_dim]
        """
        batch_size = channel_features.size(0)
        processed_channels = []
        
        for c in range(self.num_channels):
            # 提取当前通道的特征 [B, S, E]
            c_data = channel_features[:, c]
            # 转置为 [S, B, E] 以适配MultiheadAttention
            c_data = c_data.permute(1, 0, 2)
            
            # 通道独立处理
            x = c_data
            for layer in self.channel_processors[c]:
                if isinstance(layer, nn.MultiheadAttention):
                    # 自注意力层需要显式提供键、值
                    x = layer(x, x, x)[0]
                else:
                    x = layer(x)
            # 转回 [B, S, hidden_dim]
            x = x.permute(1, 0, 2)
            # 全局池化（序列维度平均）
            pooled = x.mean(dim=1)  # [B, hidden_dim]
            processed_channels.append(pooled)
        
        # 堆叠通道特征 [B, C, hidden_dim]
        processed_channels = torch.stack(processed_channels, dim=1)
        
        # 应用通道权重（Sigmoid约束）
        weights = torch.sigmoid(self.channel_weights).unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
        weighted_features = processed_channels * weights  # [B, C, hidden_dim]
        
        # 拼接所有通道特征
        combined = weighted_features.view(batch_size, -1)  # [B, C*hidden_dim]
        
        # 全连接层分类
        return self.fc(combined)