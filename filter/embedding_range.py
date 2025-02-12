import json
import torch
import random
from embedding import prepare_batch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

file_path = './data/filter/model_predicted.jsonl'

class Dataset:
    def __init__(self, file_path):
        all_examples = self.load_data(file_path)
        self.examples = all_examples

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def __getitem__(self, idx):
        end_idx = min((idx + 1) * self.batch_size, len(self.examples))
        texts = {
            'title': [ex['title'] for ex in self.examples[idx * self.batch_size:end_idx]],
            'description': [ex['description'] for ex in self.examples[idx * self.batch_size:end_idx]],
            'tags': [",".join(ex['tags']) for ex in self.examples[idx * self.batch_size:end_idx]],
            'author_info': [ex['author_info'] for ex in self.examples[idx * self.batch_size:end_idx]]
        }
        return texts

    def __len__(self):
        return len(self.examples)

    def get_batch(self, idx, batch_size):
        self.batch_size = batch_size
        return self.__getitem__(idx)
    
total = 600000
batch_size = 512
batch_num = total // batch_size
dataset = Dataset(file_path)
arr_len = batch_size * 4 * 1024
sample_rate = 0.1
sample_num = int(arr_len * sample_rate)

data = np.array([])
for i in tqdm(range(batch_num)):
    batch = dataset.get_batch(i, batch_size)
    batch = prepare_batch(batch, device="cpu")
    arr = batch.flatten().numpy()
    sampled = np.random.choice(arr.shape[0], size=sample_num, replace=False)
    data = np.concatenate((data, arr[sampled]), axis=0) if data.size else arr[sampled]
    if i % 10 == 0:
        np.save('embedding_range.npy', data)
np.save('embedding_range.npy', data)