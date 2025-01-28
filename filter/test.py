import os, json
import random
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import torch
from modelV3_4 import VideoClassifierV3_4
from sentence_transformers import SentenceTransformer
from tag import getch

def predict(json_input):
    # 加载模型
    model = VideoClassifierV3_4()
    model.load_state_dict(torch.load('./filter/checkpoints/best_model_V3.8.pt'))
    model.eval()
    
    # 加载SentenceTransformer
    sentence_transformer = SentenceTransformer("Thaweewat/jina-embedding-v3-m2v-1024")

    input_texts = {
        "title": [json_input["title"]],
        "description": [json_input["description"]],
        "tags": [" ".join(json_input["tags"])],
        "author_info": [json_input["author_info"]]
    }
    
    # 预测
    with torch.no_grad():
        logits = model(
            input_texts=input_texts,
            sentence_transformer=sentence_transformer
        )
        pred = torch.argmax(logits, dim=1).item()
            
    return pred

if __name__ == "__main__":
    with open('data/filter/model_predicted.jsonl', 'r') as fp:
        data = [json.loads(line) for line in fp.readlines()]
    sampled = random.sample(data, min(200, len(data)))
    test_data = []
    for sample in sampled:
        label = sample['label']
        os.system("clear")
        print(f"AID: {sample['aid']}")
        print(f"Title: {sample['title']}")
        print(f"Tags: {', '.join(sample['tags'])}")
        print(f"Author Info: {sample['author_info']}")
        print(f"Description: {sample['description']}")
        # 等待用户输入
        while True:
            print("Label (0 or 1 or 2, s to skip, q to quit): ", end="", flush=True)
            real_label = getch().lower()
            if real_label in ["0", "1", "2", "s", "q"]:
                break
            print("\nInvalid input. Please enter 0, 1, 2, s or q.")
        if real_label == "s":  # 跳过
            continue
        if real_label == "q":  # 退出
            break
        test_data.append({
            "aid": sample['aid'],
            "title": sample['title'],
            "tags": sample['tags'],
            "author_info": sample['author_info'],
            "description": sample['description'],
            "model": label,
            "human": int(real_label)
        })
    
    with open("./data/filter/real_test.jsonl", "a") as fp:
        fp.writelines([json.dumps(item, ensure_ascii=False) + "\n" for item in test_data])