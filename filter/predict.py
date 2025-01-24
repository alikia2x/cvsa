import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sqlite3
import json
import torch
from modelV3_4 import VideoClassifierV3_4
from sentence_transformers import SentenceTransformer

# 数据库配置
DATABASE_PATH = "./data/main.db"
OUTPUT_FILE = "./data/filter/model_predicted.jsonl"
BATCH_SIZE = 128  # 批量处理的大小

def fetch_all_aids(conn):
    """获取数据库中所有符合条件的aid"""
    cursor = conn.cursor()
    cursor.execute("SELECT aid FROM bili_info_crawl WHERE status = 'success'")
    aids = [row[0] for row in cursor.fetchall()]
    return aids

def fetch_entry_data(conn, aid):
    """获取单个条目的原始数据"""
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM bili_info_crawl WHERE aid = ?", (aid,))
    d = cursor.fetchone()
    data = d[0] if d else None
    return data

def parse_entry_data(data):
    """解析原始数据为结构化信息"""
    try:
        obj = json.loads(data)
        title = obj["View"]["title"]
        description = obj["View"]["desc"]
        tags = [tag["tag_name"] for tag in obj["Tags"] 
               if tag["tag_type"] in ["old_channel", "topic"]]
        author_info = f"{obj['Card']['card']['name']}: {obj['Card']['card']['sign']}"
        return title, description, tags, author_info
    except (KeyError, json.JSONDecodeError) as e:
        print(f"解析错误: {e}")
        return None, None, None, None

def initialize_model():
    """初始化模型和文本编码器"""
    model = VideoClassifierV3_4()
    model.load_state_dict(torch.load('./filter/checkpoints/best_model_V3.8.pt', map_location=torch.device('cpu')))
    model.eval()
    
    st_model = SentenceTransformer("Thaweewat/jina-embedding-v3-m2v-1024")
    return model, st_model

def predict_batch(model, st_model, batch_data):
    """批量执行预测"""
    with torch.no_grad():
        input_texts = {
            "title": [entry["title"] for entry in batch_data],
            "description": [entry["description"] for entry in batch_data],
            "tags": [" ".join(entry["tags"]) for entry in batch_data],
            "author_info": [entry["author_info"] for entry in batch_data]
        }
        logits = model(input_texts=input_texts, sentence_transformer=st_model)
        return torch.argmax(logits, dim=1).tolist()

def process_entries():
    """主处理流程"""
    # 初始化模型
    model, st_model = initialize_model()
    
    # 获取数据库连接
    conn = sqlite3.connect(DATABASE_PATH)
    
    # 获取所有aid
    aids = fetch_all_aids(conn)
    print(f"总需处理条目数: {len(aids)}")

    # 批量处理并保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as output:
        batch_data = []
        for idx, aid in enumerate(aids, 1):
            try:
                # 获取并解析数据
                raw_data = fetch_entry_data(conn, aid)
                if not raw_data:
                    continue
                    
                title, desc, tags, author = parse_entry_data(raw_data)
                if not title:
                    continue

                # 构造预测输入
                entry = {
                    "aid": aid,
                    "title": title,
                    "description": desc,
                    "tags": tags,
                    "author_info": author
                }
                batch_data.append(entry)

                # 当达到批量大小时进行预测
                if len(batch_data) >= BATCH_SIZE:
                    predictions = predict_batch(model, st_model, batch_data)
                    for entry, prediction in zip(batch_data, predictions):
                        output.write(json.dumps({
                            "aid": entry["aid"],
                            "title": entry["title"],
                            "description": entry["description"],
                            "tags": entry["tags"],
                            "author_info": entry["author_info"],
                            "label": prediction
                        }, ensure_ascii=False) + "\n")
                    batch_data = []  # 清空批量数据

                # 进度显示
                if idx % 100 == 0:
                    print(f"已处理 {idx}/{len(aids)} 条...")
                    
            except Exception as e:
                print(f"处理aid {aid} 时出错: {str(e)}")

        # 处理剩余的条目
        if batch_data:
            predictions = predict_batch(model, st_model, batch_data)
            for entry, prediction in zip(batch_data, predictions):
                output.write(json.dumps({
                    "aid": entry["aid"],
                    "title": entry["title"],
                    "description": entry["description"],
                    "tags": entry["tags"],
                    "author_info": entry["author_info"],
                    "label": prediction
                }, ensure_ascii=False) + "\n")

    # 关闭数据库连接
    conn.close()

if __name__ == "__main__":
    process_entries()
    print("预测完成，结果已保存至", OUTPUT_FILE)
