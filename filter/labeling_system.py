import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import os
import json
import random
import sqlite3
import torch
import sys
import tty
import termios
from sentence_transformers import SentenceTransformer
from db_utils import fetch_entry_data, parse_entry_data
from modelV3_4 import VideoClassifierV3_4

class LabelingSystem:
    def __init__(self, mode='model_testing', database_path="./data/main.db", 
                 output_file="./data/filter/labeled_data.jsonl", model_path=None,
                 start_from=None, batch_size=50):
        # 基础配置
        self.mode = mode
        self.database_path = database_path
        self.output_file = output_file
        self.batch_size = batch_size
        self.start_from = start_from
        
        # 模型相关初始化
        self.model = None
        self.sentence_transformer = None
        if self.mode == 'model_testing':
            self.model = VideoClassifierV3_4()
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            self.sentence_transformer = SentenceTransformer("Thaweewat/jina-embedding-v3-m2v-1024")

        # 数据存储相关
        self.candidate_pool = []
        self.history = []
        self.current_index = -1  # -1表示未开始
        self.label_counts = {0: 0, 1: 0, 2: 0}
        self.existing_entries = self._load_existing_entries()
        self.existing_aids = set(entry['aid'] for entry in self.existing_entries)

        # 初始化候选池
        self._load_more_candidates()

    def _load_existing_entries(self):
        """加载已有标注数据"""
        if not os.path.exists(self.output_file):
            return []
            
        entries = []
        with open(self.output_file, "r") as fp:
            for line in fp:
                entry = json.loads(line)
                entries.append(entry)
                # 统计已有标注
                if 'label' in entry and self.mode == "labeling":
                    self.label_counts[entry['label']] += 1
                elif 'human' in entry and self.mode == "model_testing":
                    self.label_counts[entry['human']] += 1
        return entries

    def _load_more_candidates(self):
        """动态加载更多候选数据"""
        if self.mode == 'model_testing':
            # 从模型预测文件加载
            with open('data/filter/model_predicted.jsonl', 'r') as fp:
                new_candidates = []
                for line in fp:
                    entry = json.loads(line)
                    if entry['aid'] not in self.existing_aids:
                        new_candidates.append(entry['aid'])
        else:
            # 从数据库随机加载
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            query = "SELECT aid FROM bili_info_crawl WHERE status = 'success'"
            params = ()
            if self.start_from is not None:
                query += " AND timestamp >= ?"
                params = (self.start_from,)
            cursor.execute(query, params)
            aids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            new_candidates = [aid for aid in aids if aid not in self.existing_aids]
        
        random.shuffle(new_candidates)
        self.candidate_pool.extend(new_candidates[:self.batch_size])

    def _get_entry_details(self, aid):
        """获取条目详细信息"""
        # 获取元数据
        title, description, tags, author_info, url = parse_entry_data(
            fetch_entry_data(self.database_path, aid)
        )
        
        entry = {
            'aid': aid,
            'title': title,
            'description': description,
            'tags': tags,
            'author_info': author_info,
            'url': url
        }
        
        # 模型预测
        if self.mode == 'model_testing':
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
                entry['model'] = torch.argmax(logits, dim=1).item()
                
        return entry

    def _save_entry(self, entry):
        """保存标注结果"""
        
        # 更新已有数据
        existing_index = next((i for i, e in enumerate(self.existing_entries) 
                             if e['aid'] == entry['aid']), None)
        if existing_index is not None:
            self.existing_entries[existing_index] = entry
        else:
            self.existing_entries.append(entry)
        
        # 写入文件
        with open(self.output_file, "w") as fp:
            for entry in self.existing_entries:
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _display_entry(self, entry):
        """显示条目信息"""
        os.system("clear")
        
        # 计算全局统计
        global_total = sum(self.label_counts.values())
        global_ratios = {k: v/global_total if global_total > 0 else 0 
                        for k, v in self.label_counts.items()}
        
        # 计算当前会话统计
        session_counts = self._get_session_counts()
        session_total = sum(session_counts.values())
        session_ratios = {k: v/session_total if session_total > 0 else 0
                        for k, v in session_counts.items()}
        
        # 显示统计信息
        print("Global Stats:")
        print(f"Count: {', '.join(str(count) for count in self.label_counts.values())} | "
            f"Ratios: {', '.join(f'{ratio * 100:.2f}%' for ratio in global_ratios.values())}")

        print("\nSession Stats:")
        print(f"Count: {', '.join(str(count) for count in session_counts.values())} | "
            f"Ratios: {', '.join(f'{ratio * 100:.2f}%' for ratio in session_ratios.values())}")

        # 显示条目信息
        print("\n" + "="*50)
        print(f"AID: {entry['aid']}")
        print(f"URL: {entry['url']}")
        print(f"Title: {entry['title']}")
        print(f"Tags: {', '.join(entry['tags'])}")
        print(f"Author Info: {entry['author_info']}")
        print(f"Description: {entry['description']}")
        
        # 显示模式相关信息
        if self.mode == 'model_testing':
            print(f"\nModel Prediction: {entry.get('model', 'N/A')}")
            if 'human' in entry and entry['human'] is not None:
                print(f"Your Label: {entry['human']}")
        else:
            if 'label' in entry and entry['label'] is not None:
                print(f"Your Label: {entry['label']}")

    def _get_session_counts(self):
        """获取当前会话的标注统计"""
        session_counts = {0: 0, 1: 0, 2: 0}
        
        # 遍历历史记录
        for entry in self.history:
            if self.mode == 'model_testing':
                if 'human' in entry and entry['human'] is not None:
                    session_counts[entry['human']] += 1
            else:
                if 'label' in entry and entry['label'] is not None:
                    session_counts[entry['label']] += 1
                    
        return session_counts

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
            cmd = self._get_input().lower()

            # 处理命令
            if cmd in ['left', 'up']:
                self.current_index = max(0, self.current_index - 1)
            elif cmd in ['right', 'down']:
                self.current_index += 1
            elif cmd in ('0', '1', '2'):
                label = int(cmd)
                if self.mode == 'model_testing':
                    current_entry['human'] = label
                else:
                    current_entry['label'] = label
                self.label_counts[label] += 1
                self._save_entry(current_entry)
                self.current_index += 1
            elif cmd == 's':
                self.current_index += 1
            elif cmd == 'q':
                return

    def _get_input(self):
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
