import os
import json
import random
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from modelV3_4 import VideoClassifierV3_4
from sentence_transformers import SentenceTransformer
import sys
import tty
import termios
from db_utils import fetch_entry_data, parse_entry_data

DATABASE_PATH = "./data/main.db"
BATCH_SIZE = 50  # 动态加载批次大小

class LabelingSystem:
    def __init__(self):
        # 初始化模型
        self.model = VideoClassifierV3_4()
        self.model.load_state_dict(torch.load('./filter/checkpoints/best_model_V3.8.pt'))
        self.model.eval()
        self.sentence_transformer = SentenceTransformer("Thaweewat/jina-embedding-v3-m2v-1024")
        
        # 数据相关
        self.existing_entries = self._load_existing_entries()
        self.existing_aids = set(entry['aid'] for entry in self.existing_entries)
        self.candidate_pool = []
        self.history = []
        self.current_index = -1  # -1表示未开始
        
        # 初始化第一批数据
        self._load_more_candidates()

    def _save_entry(self, entry):
        """保存或更新条目"""
        # 查找是否已存在
        existing_index = next((i for i, e in enumerate(self.existing_entries) 
                             if e['aid'] == entry['aid']), None)
        
        # 更新或添加条目
        if existing_index is not None:
            self.existing_entries[existing_index] = entry
        else:
            self.existing_entries.append(entry)
        
        # 重写整个文件
        with open("./data/filter/real_test.jsonl", "w") as fp:
            for entry in self.existing_entries:
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _load_existing_entries(self):
        """加载已有条目"""
        if not os.path.exists("./data/filter/real_test.jsonl"):
            return []
        
        with open("./data/filter/real_test.jsonl", "r") as fp:
            return [json.loads(line) for line in fp]

    def _load_more_candidates(self):
        """动态加载更多候选数据"""
        with open('data/filter/model_predicted.jsonl', 'r') as fp:
            new_candidates = []
            for line in fp:
                entry = json.loads(line)
                if entry['aid'] not in self.existing_aids:
                    new_candidates.append(entry['aid'])
        
        # 随机打乱后取批次
        random.shuffle(new_candidates)
        self.candidate_pool.extend(new_candidates[:BATCH_SIZE])
        del new_candidates[:BATCH_SIZE]  # 释放内存

    def _get_entry_details(self, aid):
        """获取条目详细信息并预测模型标签"""
        # 获取元数据
        title, description, tags, author_info, url = parse_entry_data(
            fetch_entry_data(DATABASE_PATH, aid)
        )
        
        # 模型预测
        with torch.no_grad():
            logits = self.model(
                input_texts={
                    "title": [title],
                    "description": [description],
                    "tags": [" ".join(tags)],
                    "author_info": [author_info]
                },
                sentence_transformer=self.sentence_transformer
            )
            model_label = torch.argmax(logits, dim=1).item()
        
        return {
            'aid': aid,
            'title': title,
            'description': description,
            'tags': tags,
            'author_info': author_info,
            'url': url,
            'model_label': model_label,
            'user_label': None
        }

    def _display_entry(self, entry):
        """显示条目信息"""
        os.system("clear")
        print(f"AID: {entry['aid']}")
        print(f"URL: {entry['url']}")
        print(f"Title: {entry['title']}")
        print(f"Tags: {', '.join(entry['tags'])}")
        print(f"Author Info: {entry['author_info']}")
        print(f"Description: {entry['description']}")
        print(f"\nModel Prediction: {entry['model_label']}")
        if entry['user_label'] is not None:
            print(f"Your Label: {entry['user_label']}")

    def run(self):
        while True:
            # 处理当前条目
            if self.current_index < 0:
                self.current_index = 0
                
            if self.current_index >= len(self.history):
                if not self.candidate_pool:
                    self._load_more_candidates()
                    if not self.candidate_pool:
                        print("\nAll entries processed!")
                        return
                
                # 处理新条目
                aid = self.candidate_pool.pop(0)
                entry = self._get_entry_details(aid)
                self.history.append(entry)
                self.current_index = len(self.history) - 1

            current_entry = self.history[self.current_index]
            self._display_entry(current_entry)

            # 获取用户输入
            print("\nLabel (0/1/2, s=skip, ←↑/→↓=nav, q=quit): ", end="", flush=True)
            cmd = getch().lower()

            # 处理导航命令
            if cmd in ['left', 'up']:
                self.current_index = max(0, self.current_index - 1)
            elif cmd in ['right', 'down']:
                self.current_index += 1
            elif cmd in ('0', '1', '2'):
                current_entry['human'] = int(cmd)
                self._save_entry(current_entry)
                self.current_index += 1  # 自动前进
            elif cmd == 's':
                self.current_index += 1  # 跳过
            elif cmd == 'q':
                return

def getch():
    """支持方向键检测的输入函数"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            seq = sys.stdin.read(2)
            return {'[A': 'up', '[B': 'down', '[C': 'right', '[D': 'left'}.get(seq, 'unknown')
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    labeling_system = LabelingSystem()
    labeling_system.run()