import json
import torch
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def split_and_save_data(file_path, test_size=0.1, eval_size=0.1, random_state=42):
    """分割数据并保存test、train和eval集"""
    # 读取原始数据
    with open(file_path, 'r', encoding='utf-8') as f:
        all_examples = [json.loads(line) for line in f]
    
    # 检查test文件是否存在
    test_file = os.path.join(os.path.dirname(file_path), 'test.jsonl')
    
    if not os.path.exists(test_file):
        # 如果test文件不存在，分割test数据并保存
        train_eval_examples, test_examples = train_test_split(
            all_examples, test_size=test_size, random_state=random_state
        )
        
        # 保存test集
        with open(test_file, 'w', encoding='utf-8') as f:
            for example in test_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
    else:
        # 如果test文件存在，直接读取test数据
        with open(test_file, 'r', encoding='utf-8') as f:
            test_examples = [json.loads(line) for line in f]
        
        # 剩余的作为train_eval_examples
        test_ids = set(json.dumps(example, sort_keys=True) for example in test_examples)
        train_eval_examples = [example for example in all_examples if json.dumps(example, sort_keys=True) not in test_ids]
    
    # 分割train和eval数据
    train_examples, eval_examples = train_test_split(
        train_eval_examples, test_size=eval_size, random_state=random_state
    )
    
    # 保存train集
    train_file = os.path.join(os.path.dirname(file_path), 'train.jsonl')
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # 保存eval集
    eval_file = os.path.join(os.path.dirname(file_path), 'eval.jsonl')
    with open(eval_file, 'w', encoding='utf-8') as f:
        for example in eval_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    return train_examples, eval_examples, test_examples

class MultiChannelDataset(Dataset):
    def __init__(self, file_path, max_length=128, mode='train', test_size=0.1, eval_size=0.1, random_state=42):
        self.max_length = max_length
        self.mode = mode
        
        # 检查train、eval和test文件是否存在
        train_file = os.path.join(os.path.dirname(file_path), 'train.jsonl')
        eval_file = os.path.join(os.path.dirname(file_path), 'eval.jsonl')
        test_file = os.path.join(os.path.dirname(file_path), 'test.jsonl')
        
        if os.path.exists(train_file) and os.path.exists(eval_file) and os.path.exists(test_file):
            # 如果文件存在，直接读取对应的数据
            if self.mode == 'train':
                with open(train_file, 'r', encoding='utf-8') as f:
                    self.examples = [json.loads(line) for line in f]
            elif self.mode == 'eval':
                with open(eval_file, 'r', encoding='utf-8') as f:
                    self.examples = [json.loads(line) for line in f]
            elif self.mode == 'test':
                with open(test_file, 'r', encoding='utf-8') as f:
                    self.examples = [json.loads(line) for line in f]
            else:
                raise ValueError("Invalid mode. Choose from 'train', 'eval', or 'test'.")
        else:
            # 如果文件不存在，执行分割并保存文件
            train_examples, eval_examples, test_examples = split_and_save_data(
                file_path, test_size, eval_size, random_state
            )
            
            # 根据mode选择对应的数据
            if self.mode == 'train':
                self.examples = train_examples
            elif self.mode == 'eval':
                self.examples = eval_examples
            elif self.mode == 'test':
                self.examples = test_examples
            else:
                raise ValueError("Invalid mode. Choose from 'train', 'eval', or 'test'.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 处理tags（将数组转换为空格分隔的字符串）
        tags_text = ",".join(example['tags'])
        
        # 返回文本字典
        texts = {
            'title': example['title'],
            'description': example['description'],
            'tags': tags_text,
            'author_info': example['author_info']
        }
        
        return {
            'texts': texts,  # 文本字典
            'label': torch.tensor(example['label'], dtype=torch.long)  # 标签
        }