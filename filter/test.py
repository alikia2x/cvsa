import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import torch
from modelV3_4 import VideoClassifierV3_4
from sentence_transformers import SentenceTransformer

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
    # 示例用法
    sample_input = {"title": "", "description": "", "tags": ["",""], "author_info": "xx: yy"}
    
    result = predict(sample_input)
    print(f"预测结果: {result}")
