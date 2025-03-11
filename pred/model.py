import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    """时间特征编码模块"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.norm = nn.LayerNorm(5)
        
        # 时间特征编码（适配新的5维时间特征）
        self.time_encoder = nn.Sequential(
            nn.Linear(5, 64),  # 输入维度对应x_time_feat的5个特征
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, embed_dim)
        )
        
    def forward(self, time_feat):
        """
        time_feat: 时间特征 (batch, seq_len, 5)
        """
        time_feat = self.norm(time_feat)  # 应用归一化
        return self.time_encoder(time_feat)


class MultiScaleEncoder(nn.Module):
    """多尺度特征编码器"""
    def __init__(self, input_dim, d_model, nhead, conv_kernels=[3, 7, 23]):
        super().__init__()
        self.d_model = d_model
        
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, d_model, kernel_size=k, padding=k//2),
                nn.GELU(),
            ) for k in conv_kernels
        ])

        # 添加 LayerNorm 到单独的列表中
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in conv_kernels])
        
        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, 
                nhead,
                dim_feedforward=d_model*4,
                batch_first=True # 修改为batch_first
            ), 
            num_layers=4
        )
        
        # 特征融合层
        self.fusion = nn.Linear(d_model*(len(conv_kernels)+1), d_model)

    def forward(self, x, padding_mask=None):
        """
        x: 输入特征 (batch, seq_len, input_dim)
        padding_mask: 填充掩码 (batch, seq_len)
        """
        
        # 卷积分支处理
        conv_features = []
        x_conv = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        for i, branch in enumerate(self.conv_branches):
            feat = branch(x_conv)  # 输出形状 (batch, d_model, seq_len)
            # 手动转置并应用 LayerNorm
            feat = feat.permute(0, 2, 1)  # (batch, seq_len, d_model)
            feat = self.layer_norms[i](feat)  # 应用 LayerNorm
            conv_features.append(feat)
        
        # Transformer分支处理
        trans_feat = self.transformer(
            x, 
            src_key_padding_mask=padding_mask
        )  # (batch, seq_len, d_model)
        
        # 特征拼接与融合
        combined = torch.cat(conv_features + [trans_feat], dim=-1)
        fused = self.fusion(combined)  # (batch, seq_len, d_model)
        
        return fused

class VideoPlayPredictor(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.d_model = d_model
        
        # 特征嵌入
        self.time_embed = TimeEmbedding(embed_dim=64)
        self.base_embed = nn.Linear(1 + 64, d_model)  # 播放量 + 时间特征
        
        # 编码器
        self.encoder = MultiScaleEncoder(d_model, d_model, nhead)
        
        # 时间感知预测头
        self.forecast_head = nn.Sequential(
            nn.Linear(2 * d_model + 1, d_model * 4),  # 关键修改：输入维度为 2*d_model +1
            nn.GELU(),
            nn.Linear(d_model * 4, 1),
            nn.ReLU()  # 确保输出非负
        )
        
        # 上下文提取器
        self.context_extractor = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'forecast_head' in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(p, gain=1e-2)  # 缩小初始化范围
                elif 'bias' in name:
                    nn.init.constant_(p, 0.0)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_play, x_time_feat, padding_mask, forecast_span):
        """
        x_play: 历史播放量 (batch, seq_len)
        x_time_feat: 时间特征 (batch, seq_len, 5)
        padding_mask: 填充掩码 (batch, seq_len)
        forecast_span: 预测时间跨度 (batch, 1)
        """
        batch_size = x_play.size(0)
        
        # 时间特征编码
        time_emb = self.time_embed(x_time_feat)  # (batch, seq_len, 64)
        
        # 基础特征拼接
        base_feat = torch.cat([
            x_play.unsqueeze(-1),  # (batch, seq_len, 1)
            time_emb
        ], dim=-1)  # (batch, seq_len, 1+64)
        
        # 投影到模型维度
        embedded = self.base_embed(base_feat)  # (batch, seq_len, d_model)
        
        # 编码特征
        encoded = self.encoder(embedded, padding_mask)  # (batch, seq_len, d_model)
        
        # 提取上下文
        context, _ = self.context_extractor(encoded)  # (batch, seq_len, d_model*2)
        context = context.mean(dim=1)  # (batch, d_model*2)
        
        # 融合时间跨度特征
        span_feat = torch.log1p(forecast_span) / 10  # 归一化
        combined = torch.cat([
            context,
            span_feat
        ], dim=-1)  # (batch, d_model*2 + 1)
        
        # 最终预测
        pred = self.forecast_head(combined)  # (batch, 1)
        
        return pred

class MultiTaskWrapper(nn.Module):
    """适配新数据结构的封装"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, batch):
        return self.model(
            batch['x_play'],
            batch['x_time_feat'],
            batch['padding_mask'],
            batch['forecast_span']
        )
