# dataset.py
import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import datetime

class VideoPlayDataset(Dataset):
    def __init__(self, data_dir, publish_time_path, max_future_days=7):
        self.data_dir = data_dir
        self.max_future_seconds = max_future_days * 86400
        self.series_dict = self._load_and_process_data(publish_time_path)
        self.valid_series = [s for s in self.series_dict.values() if len(s['abs_time']) > 1]
        self.feature_windows = [3600, 3*3600, 6*3600, 24*3600, 3*24*3600, 7*24*3600, 60*24*3600]

    def _extract_features(self, series, current_idx, target_idx):
        """提取增量特征"""
        current_time = series['abs_time'][current_idx]
        current_play = series['play_count'][current_idx]
        dt = datetime.datetime.fromtimestamp(current_time)
        # 时间特征
        time_features = [
            (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400, (dt.weekday() * 24 + dt.hour) / 168,
            np.log2(max(current_time - series['create_time'],1))
        ]
        
        # 窗口增长特征（增量）
        growth_features = []
        for window in self.feature_windows:
            prev_time = current_time - window
            prev_idx = self._get_nearest_value(series, prev_time, current_idx)
            if prev_idx is not None:
                time_diff = current_time - series['abs_time'][prev_idx]
                play_diff = current_play - series['play_count'][prev_idx]
                scaled_diff = play_diff / (time_diff / window) if time_diff > 0 else 0.0
            else:
                scaled_diff = 0.0
            growth_features.append(np.log2(max(scaled_diff,1)))
        
        time_diff = series['abs_time'][target_idx] - series['abs_time'][current_idx]

        return [np.log2(max(time_diff,1))] + [np.log2(current_play + 1)] + growth_features + time_features

    def _load_and_process_data(self, publish_time_path):
        # 加载发布时间数据
        publish_df = pd.read_csv(publish_time_path)
        publish_df['published_at'] = pd.to_datetime(publish_df['published_at'])
        publish_dict = dict(zip(publish_df['aid'], publish_df['published_at']))
        series_dict = {}
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.json'):
                continue
            with open(os.path.join(self.data_dir, filename), 'r') as f:
                data = json.load(f)
                if 'code' in data:
                    continue
                for item in data:
                    aid = item['aid']
                    published_time = pd.to_datetime(publish_dict[aid]).timestamp()
                    if aid not in series_dict:
                        series_dict[aid] = {
                            'abs_time': [],
                            'play_count': [],
                            'create_time': published_time
                        }
                    series_dict[aid]['abs_time'].append(item['added'])
                    series_dict[aid]['play_count'].append(item['view'])
        return series_dict

    def __len__(self):
        return 100000  # 使用虚拟长度实现无限采样

    def _get_nearest_value(self, series, target_time, current_idx):
        """获取指定时间前最近的数据点"""
        min_diff = float('inf')
        for i in range(current_idx + 1, len(series['abs_time'])):
            diff = abs(series['abs_time'][i] - target_time)
            if diff < min_diff:
                min_diff = diff
            else:
                return i - 1
        return len(series['abs_time']) - 1

    def __getitem__(self, idx):
        series = random.choice(self.valid_series)
        current_idx = random.randint(0, len(series['abs_time'])-2)
        target_idx = random.randint(max(0, current_idx-10), current_idx)
        
        # 提取特征
        features = self._extract_features(series, current_idx, target_idx)

        # 目标值：未来播放量增量
        current_play = series['play_count'][current_idx]
        target_play = series['play_count'][target_idx]
        target_delta = max(target_play - current_play, 0)  # 增量
        
        return {
            'features': torch.FloatTensor(features),
            'target': torch.log2(torch.FloatTensor([target_delta]) + 1)  # 输出增量
        }

def collate_fn(batch):
    return {
        'features': torch.stack([x['features'] for x in batch]),
        'targets': torch.stack([x['target'] for x in batch])
    }