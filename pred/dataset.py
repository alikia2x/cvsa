import os
import json
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime
import torch

class VideoPlayDataset(Dataset):
    def __init__(self, data_dir, publish_time_path, 
                 min_seq_len=6, max_seq_len=200,
                 min_forecast_span=60, max_forecast_span=604800):
        """
        改进后的数据集类，支持非等间隔时间序列
        :param data_dir: JSON文件目录
        :param publish_time_path: 发布时间CSV路径
        :param min_seq_len: 最小历史数据点数
        :param max_seq_len: 最大历史数据点数
        :param min_forecast_span: 最小预测时间跨度（秒）
        :param max_forecast_span: 最大预测时间跨度（秒）
        """
        self.data_dir = data_dir
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.min_forecast_span = min_forecast_span
        self.max_forecast_span = max_forecast_span
        self.series_dict = self._load_and_process_data(data_dir, publish_time_path)
        self.valid_series = self._generate_valid_series()

    def _load_and_process_data(self, data_dir, publish_time_path):
        # 加载发布时间数据
        publish_df = pd.read_csv(publish_time_path)
        publish_df['published_at'] = pd.to_datetime(publish_df['published_at'])
        publish_dict = dict(zip(publish_df['aid'], publish_df['published_at']))

        # 加载并处理JSON数据
        series_dict = {}
        for filename in os.listdir(data_dir):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                for item in json_data:
                    aid = item['aid']
                    if aid not in publish_dict:
                        continue
                    
                    # 计算相对时间
                    added_time = datetime.fromtimestamp(item['added'])
                    published_time = publish_dict[aid]
                    rel_time = (added_time - published_time).total_seconds()
                    
                    # 按视频组织数据
                    if aid not in series_dict:
                        series_dict[aid] = {
                            'abs_time': [],
                            'rel_time': [],
                            'play_count': []
                        }
                    
                    series_dict[aid]['abs_time'].append(item['added'])
                    series_dict[aid]['rel_time'].append(rel_time)
                    series_dict[aid]['play_count'].append(item['view'])
        
        # 按时间排序并计算时间间隔
        for aid in series_dict:
            # 按时间排序
            sorted_idx = np.argsort(series_dict[aid]['abs_time'])
            for key in ['abs_time', 'rel_time', 'play_count']:
                series_dict[aid][key] = np.array(series_dict[aid][key])[sorted_idx]
            
            # 计算时间间隔特征
            abs_time_arr = series_dict[aid]['abs_time']
            time_deltas = np.diff(abs_time_arr, prepend=abs_time_arr[0])
            series_dict[aid]['time_delta'] = time_deltas
        
        return series_dict

    def _generate_valid_series(self):
        # 生成有效数据序列
        valid_series = []
        for aid in self.series_dict:
            series = self.series_dict[aid]
            n_points = len(series['play_count'])
            
            # 过滤数据量不足的视频
            if n_points < self.min_seq_len + 1:
                continue
                
            valid_series.append({
                'aid': aid,
                'length': n_points,
                'abs_time': series['abs_time'],
                'rel_time': series['rel_time'],
                'play_count': series['play_count'],
                'time_delta': series['time_delta']
            })
        return valid_series

    def __len__(self):
        return sum(s['length'] - self.min_seq_len for s in self.valid_series)

    def __getitem__(self, idx):
        # 随机选择视频序列
        series = random.choice(self.valid_series)
        max_start = series['length'] - self.min_seq_len - 1
        start_idx = random.randint(0, max_start)
        
        # 动态确定历史窗口长度
        seq_len = random.randint(self.min_seq_len, min(self.max_seq_len, series['length'] - start_idx - 1))
        end_idx = start_idx + seq_len
        
        # 提取历史窗口特征
        hist_slice = slice(start_idx, end_idx)
        x_play = np.log1p(series['play_count'][hist_slice])
        x_abs_time = series['abs_time'][hist_slice]
        x_rel_time = series['rel_time'][hist_slice]
        x_time_delta = series['time_delta'][hist_slice]
        
        # 生成预测目标（动态时间跨度）
        forecast_span = random.randint(self.min_forecast_span, self.max_forecast_span)
        target_time = x_abs_time[-1] + forecast_span
        
        # 寻找实际目标点（处理数据间隙）
        future_times = series['abs_time'][end_idx:]
        future_plays = series['play_count'][end_idx:]
        
        # 找到第一个超过目标时间的点
        target_idx = np.searchsorted(future_times, target_time)
        if target_idx >= len(future_times):
            # 若超出数据范围，取最后一个点
            y_play = future_plays[-1] if len(future_plays) > 0 else x_play[-1]
            actual_span = future_times[-1] - x_abs_time[-1] if len(future_times) > 0 else self.max_forecast_span
        else:
            y_play = future_plays[target_idx]
            actual_span = future_times[target_idx] - x_abs_time[-1]

        y_play_val = np.log1p(y_play)
        
        # 构造时间相关特征
        time_features = np.stack([
            x_abs_time,
            x_rel_time,
            x_time_delta,
            np.log1p(x_time_delta),  # 对数变换处理长尾分布
            (x_time_delta > 3600).astype(float)  # 间隔是否大于1小时
        ], axis=-1)
        
        return {
            'x_play': torch.FloatTensor(x_play),
            'x_time_feat': torch.FloatTensor(time_features),
            'y_play': torch.FloatTensor([y_play_val]),
            'forecast_span': torch.FloatTensor([actual_span])
        }

def collate_fn(batch):
    """动态填充处理"""
    max_len = max(item['x_play'].shape[0] for item in batch)
    
    padded_batch = {
        'x_play': [],
        'x_time_feat': [],
        'y_play': [],
        'forecast_span': [],
        'padding_mask': []
    }
    
    for item in batch:
        seq_len = item['x_play'].shape[0]
        pad_len = max_len - seq_len
        
        # 填充播放量数据
        padded_play = torch.cat([
            item['x_play'],
            torch.zeros(pad_len)
        ])
        padded_batch['x_play'].append(padded_play)
        
        # 填充时间特征
        padded_time_feat = torch.cat([
            item['x_time_feat'],
            torch.zeros(pad_len, item['x_time_feat'].shape[1])
        ])
        padded_batch['x_time_feat'].append(padded_time_feat)
        
        # 创建padding mask
        mask = torch.cat([
            torch.ones(seq_len),
            torch.zeros(pad_len)
        ])
        padded_batch['padding_mask'].append(mask.bool())
        
        # 其他字段
        padded_batch['y_play'].append(item['y_play'])
        padded_batch['forecast_span'].append(item['forecast_span'])
    
    # 转换为张量
    padded_batch['x_play'] = torch.stack(padded_batch['x_play'])
    padded_batch['x_time_feat'] = torch.stack(padded_batch['x_time_feat'])
    padded_batch['y_play'] = torch.stack(padded_batch['y_play'])
    padded_batch['forecast_span'] = torch.stack(padded_batch['forecast_span'])
    padded_batch['padding_mask'] = torch.stack(padded_batch['padding_mask'])
    
    return padded_batch
