import torch
import torch.nn as nn

class VideoClassifierV6_3(nn.Module):
    def __init__(self, embedding_dim=72, hidden_dim=256, output_dim=3, num_heads=4, num_layers=2):
        super().__init__()
        self.num_channels = 3
        self.channel_names = ['title', 'description', 'tags']
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 通道独立处理模块（每个通道独立的Transformer编码器）
        self.channel_processors = nn.ModuleList()
        for _ in range(self.num_channels):
            layers = []
            # 首先将输入维度转换为hidden_dim
            layers.extend([
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim)
            ])
            # 添加num_layers层的Transformer块
            for _ in range(num_layers):
                layers.extend([
                    # 自注意力层（使用hidden_dim作为embed_dim）
                    nn.MultiheadAttention(
                        embed_dim=hidden_dim,  # 修改为hidden_dim
                        num_heads=num_heads,
                        dropout=0.1
                    ),
                    nn.LayerNorm(hidden_dim),
                    # 前馈网络部分
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                ])
            self.channel_processors.append(nn.Sequential(*layers))
        
        # 通道权重（可学习，Sigmoid约束）
        self.channel_weights = nn.Parameter(torch.ones(self.num_channels))
        
        # 全连接层（扩展维度）
        self.fc = nn.Sequential(
            nn.Linear(self.num_channels * hidden_dim, 1024),
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
            c_data = channel_features[:, c].permute(1, 0, 2)  # 转为 [S, B, E]

            # 通道独立处理
            x = c_data
            for layer in self.channel_processors[c]:
                if isinstance(layer, nn.MultiheadAttention):
                    # 自注意力层需要显式提供键、值
                    x = layer(x, x, x)[0]
                else:
                    x = layer(x)

            # 转换回 [B, S, hidden_dim] 并全局平均池化
            x = x.permute(1, 0, 2)
            pooled = x.mean(dim=1)
            processed_channels.append(pooled)
        
        # 堆叠通道特征
        processed_channels = torch.stack(processed_channels, dim=1)

        # 应用通道权重（Sigmoid约束）
        weights = torch.sigmoid(self.channel_weights).view(1, -1, 1)
        weighted_features = processed_channels * weights

        # 拼接所有通道特征
        combined = weighted_features.view(batch_size, -1)
        
        return self.fc(combined)