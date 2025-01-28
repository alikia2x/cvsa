import sqlite3
import json
import random
import os
import sys
import tty
import termios
import argparse
from db_utils import fetch_entry_data, parse_entry_data

# 数据库路径
DATABASE_PATH = "./data/main.db"
# 输出文件路径
OUTPUT_FILE = "./data/filter/labeled_data.jsonl"
def getch():
    """
    获取单个字符输入，无需回车
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def fetch_random_aids(db_path, num_entries=10, start_from=None):
    """
    从数据库中随机抽取指定数量的aid
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 构建查询语句
    query = "SELECT aid FROM bili_info_crawl WHERE status = 'success'"
    if start_from is not None:
        query += " AND timestamp >= ?"
        cursor.execute(query, (start_from,))
    else:
        cursor.execute(query)
    
    aids = [row[0] for row in cursor.fetchall()]
    # 随机抽取指定数量的aid
    if len(aids) > num_entries:
        aids = random.sample(aids, num_entries)
    conn.close()
    return aids

def label_entries(db_path, aids):
    """
    标注工具：展示条目信息，等待用户输入标签
    """
    labeled_data = []
    label_counts = {0: 0, 1: 0, 2: 0}
    for aid in aids:
        data = fetch_entry_data(db_path, aid)
        title, description, tags, author_info, url = parse_entry_data(data)
        if not title:  # 如果解析失败，跳过
            continue
        # 展示信息
        os.system("clear")
        print(f"Count: {label_counts[0]}, {label_counts[1]}, {label_counts[2]}")
        print(f"AID: {aid}")
        print(f"URL: {url}")
        print(f"Title: {title}")
        print(f"Tags: {', '.join(tags)}")
        print(f"Author Info: {author_info}")
        print(f"Description: {description}")
        # 等待用户输入
        while True:
            print("Label (0 or 1 or 2, s to skip, q to quit): ", end="", flush=True)
            label = getch().lower()
            if label in ["0", "1", "2", "s", "q"]:
                break
            print("\nInvalid input. Please enter 0, 1, 2, s or q.")
        if label == "s":  # 跳过
            continue
        if label == "q":  # 退出
            break
        # 更新计数
        label_counts[int(label)] += 1
        # 保存标注结果
        labeled_data.append({
            "aid": aid,
            "title": title,
            "description": description,
            "tags": tags,
            "author_info": author_info,
            "label": int(label)
        })
    return labeled_data

def save_labeled_data(labeled_data, output_file):
    """
    将标注结果保存到文件
    """
    with open(output_file, "a", encoding="utf-8") as f:
        for entry in labeled_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Label entries from the database.")
    parser.add_argument("--output", type=str, help="Output file path for labeled data.")
    parser.add_argument("--startFrom", type=float, help="Minimum timestamp for fetching entries.")
    args = parser.parse_args()

    # 使用提供的输出文件路径或默认值
    output_file = args.output if args.output else OUTPUT_FILE
    start_from = args.startFrom

    # 从数据库中随机抽取aid
    aids = fetch_random_aids(DATABASE_PATH, num_entries=1000, start_from=start_from)
    # 标注工具
    labeled_data = label_entries(DATABASE_PATH, aids)
    # 保存标注结果
    if labeled_data:
        save_labeled_data(labeled_data, output_file)
        print(f"Labeled data saved to {output_file}")
    else:
        print("No data labeled.")

if __name__ == "__main__":
    main()