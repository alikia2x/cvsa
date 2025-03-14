import os
import json
import random
import bisect
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import datetime

class VideoPlayDataset(Dataset):
    def __init__(self, data_dir, publish_time_path, term='long', seed=42):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.data_dir = data_dir
        self.series_dict = self._load_and_process_data(publish_time_path)
        self.valid_series = [s for s in self.series_dict.values() if len(s['abs_time']) > 1]
        self.term = term
        # Set time window based on term
        self.time_window = 1000 * 24 * 3600 if term == 'long' else 7 * 24 * 3600
        MINUTE = 60
        HOUR = 3600
        DAY = 24 * HOUR

        if term == 'long':
            self.feature_windows = [
                1 * HOUR,
                6 * HOUR,
                1 *DAY,
                3 * DAY,
                7 * DAY,
                30 * DAY,
                100 * DAY
            ]
        else:
            self.feature_windows = [
                ( 15 * MINUTE,  0 * MINUTE),
                ( 40 * MINUTE,  0 * MINUTE),
                ( 1 * HOUR,  0 * HOUR),
                ( 2 * HOUR,  1 * HOUR),
                ( 3 * HOUR,  2 * HOUR),
                ( 3 * HOUR,  0 * HOUR),
                #( 6 * HOUR,  3 * HOUR),
                ( 6 * HOUR,  0 * HOUR),
                (18 * HOUR, 12 * HOUR),
                #( 1 * DAY,   6 * HOUR),
                ( 1 * DAY,   0 * DAY),
                #( 2 * DAY,   1 * DAY),
                ( 3 * DAY,   0 * DAY),
                #( 4 * DAY,   1 * DAY),
                ( 7 * DAY,   0 * DAY)
            ]

    def _extract_features(self, series, current_idx, target_idx):
        current_time = series['abs_time'][current_idx]
        current_play = series['play_count'][current_idx]
        dt = datetime.datetime.fromtimestamp(current_time)

        if self.term == 'long':
            time_features = [
                np.log2(max(current_time - series['create_time'], 1))
            ]
        else:
            time_features = [
                (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400,
                (dt.weekday() * 24 + dt.hour) / 168,
                np.log2(max(current_time - series['create_time'], 1))
            ]
        
        growth_features = []
        if self.term == 'long':
            for window in self.feature_windows:
                prev_time = current_time - window
                prev_idx = self._get_nearest_value(series, prev_time, current_idx)
                if prev_idx is not None:
                    time_diff = current_time - series['abs_time'][prev_idx]
                    play_diff = current_play - series['play_count'][prev_idx]
                    scaled_diff = play_diff / (time_diff / window) if time_diff > 0 else 0.0
                else:
                    scaled_diff = 0.0
                growth_features.append(np.log2(max(scaled_diff, 1)))
        else:
            for window_start, window_end in self.feature_windows:
                prev_time_start = current_time - window_start
                prev_time_end = current_time - window_end  # window_end is typically 0
                prev_idx_start = self._get_nearest_value(series, prev_time_start, current_idx)
                prev_idx_end = self._get_nearest_value(series, prev_time_end, current_idx)
                if prev_idx_start is not None and prev_idx_end is not None:
                    time_diff = series['abs_time'][prev_idx_end] - series['abs_time'][prev_idx_start]
                    play_diff = series['play_count'][prev_idx_end] - series['play_count'][prev_idx_start]
                    scaled_diff = play_diff / (time_diff / (window_start - window_end)) if time_diff > 0 else 0.0
                else:
                    scaled_diff = 0.0
                growth_features.append(np.log2(max(scaled_diff, 1)))
        
        time_diff = series['abs_time'][target_idx] - current_time
        return [np.log2(max(time_diff, 1))] + [np.log2(current_play + 1)] + growth_features + time_features

    def _load_and_process_data(self, publish_time_path):
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
        # Sort each series by absolute time
        for aid in series_dict:
            sorted_indices = sorted(range(len(series_dict[aid]['abs_time'])),
                                key=lambda k: series_dict[aid]['abs_time'][k])
            series_dict[aid]['abs_time'] = [series_dict[aid]['abs_time'][i] for i in sorted_indices]
            series_dict[aid]['play_count'] = [series_dict[aid]['play_count'][i] for i in sorted_indices]
        return series_dict

    def __len__(self):
        return 100000  # Virtual length for sampling

    def _get_nearest_value(self, series, target_time, current_idx):
        times = series['abs_time']
        pos = bisect.bisect_right(times, target_time, 0, current_idx + 1)
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos <= current_idx:
            candidates.append(pos)
        if not candidates:
            return None
        closest_idx = min(candidates, key=lambda i: abs(times[i] - target_time))
        return closest_idx

    def __getitem__(self, _idx):
        while True:
            series = random.choice(self.valid_series)
            if len(series['abs_time']) < 2:
                continue
            current_idx = random.randint(0, len(series['abs_time']) - 2)
            current_time = series['abs_time'][current_idx]
            max_target_time = current_time + self.time_window
            candidate_indices = []
            for j in range(current_idx + 1, len(series['abs_time'])):
                if series['abs_time'][j] > max_target_time:
                    break
                candidate_indices.append(j)
            if not candidate_indices:
                continue
            target_idx = random.choice(candidate_indices)
            break
        current_play = series['play_count'][current_idx]
        target_play = series['play_count'][target_idx]
        target_delta = max(target_play - current_play, 0)
        return {
            'features': torch.FloatTensor(self._extract_features(series, current_idx, target_idx)),
            'target': torch.log2(torch.FloatTensor([target_delta]) + 1)
        }

def collate_fn(batch):
    return {
        'features': torch.stack([x['features'] for x in batch]),
        'targets': torch.stack([x['target'] for x in batch])
    }