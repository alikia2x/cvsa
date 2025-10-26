import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="CVSA ML API", version="1.0.0")

# Global variables for models
tokenizer = None
classifier_model = None

class ClassificationRequest(BaseModel):
    title: str
    description: str
    tags: str
    aid: int = None

class ClassificationResponse(BaseModel):
    label: int
    probabilities: List[float]
    aid: int = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool

def load_models():
    """Load the tokenizer and classifier models"""
    global tokenizer, classifier_model
    
    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
        
        # Load classifier model
        logger.info("Loading classifier model...")
        from model_config import VideoClassifierV3_15
        
        model_path = "../../model/akari/3.17.pt"
        classifier_model = VideoClassifierV3_15()
        classifier_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        classifier_model.eval()
        
        logger.info("All models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return False

def softmax(logits: np.ndarray) -> np.ndarray:
    """Apply softmax to logits"""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def get_jina_embeddings_1024(texts: List[str]) -> np.ndarray:
    """Get Jina embeddings using tokenizer and ONNX-like processing"""
    if tokenizer is None:
        raise ValueError("Tokenizer not loaded")
    
    import onnxruntime as ort
    
    session = ort.InferenceSession("../../model/embedding/model.onnx")
    
    encoded_inputs = tokenizer(
        texts,
        add_special_tokens=False,  # 关键：不添加特殊token（与JS一致）
        return_attention_mask=False,
        return_tensors=None  # 返回原生Python列表，便于后续处理
    )
    input_ids = encoded_inputs["input_ids"]  # 形状: [batch_size, seq_len_i]（每个样本长度可能不同）
    
    # 2. 计算offsets（与JS的cumsum逻辑完全一致）
    # 先获取每个样本的token长度
    lengths = [len(ids) for ids in input_ids]
    # 计算累积和（排除最后一个样本）
    cumsum = []
    current_sum = 0
    for l in lengths[:-1]:  # 只累加前n-1个样本的长度
        current_sum += l
        cumsum.append(current_sum)
    # 构建offsets：起始为0，后面跟累积和
    offsets = [0] + cumsum  # 形状: [batch_size]
    
    # 3. 展平input_ids为一维数组
    flattened_input_ids = []
    for ids in input_ids:
        flattened_input_ids.extend(ids)  # 直接拼接所有token id
    flattened_input_ids = np.array(flattened_input_ids, dtype=np.int64)
    
    # 4. 准备ONNX输入（与JS的tensor形状保持一致）
    inputs = {
        "input_ids": ort.OrtValue.ortvalue_from_numpy(flattened_input_ids),
        "offsets": ort.OrtValue.ortvalue_from_numpy(np.array(offsets, dtype=np.int64))
    }
    
    # 5. 运行模型推理
    outputs = session.run(None, inputs)
    embeddings = outputs[0]  # 假设第一个输出是embeddings，形状: [batch_size, embedding_dim]
    
    return torch.tensor(embeddings, dtype=torch.float32).numpy()

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    success = load_models()
    if not success:
        logger.error("Failed to load models during startup")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = tokenizer is not None and classifier_model is not None
    return HealthResponse(
        status="healthy" if models_loaded else "models_not_loaded",
        models_loaded=models_loaded
    )

@app.post("/classify", response_model=ClassificationResponse)
async def classify_video(request: ClassificationRequest):
    """Classify a video based on title, description, and tags"""
    try:
        if tokenizer is None or classifier_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Get embeddings for each channel
        texts = [request.title, request.description, request.tags]
        embeddings = get_jina_embeddings_1024(texts)
        
        # Prepare input for classifier (batch_size=1, channels=3, embedding_dim=1024)
        channel_features = torch.tensor(embeddings).unsqueeze(0)  # [1, 3, 1024]
        
        # Run inference
        with torch.no_grad():
            logits = classifier_model(channel_features)
            probabilities = softmax(logits.numpy()[0])
            predicted_label = int(np.argmax(probabilities))
        
        logger.info(f"Classification completed for aid {request.aid}: label={predicted_label}")
        
        return ClassificationResponse(
            label=predicted_label,
            probabilities=probabilities.tolist(),
            aid=request.aid
        )
        
    except Exception as e:
        logger.error(f"Classification error for aid {request.aid}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify_batch")
async def classify_video_batch(requests: List[ClassificationRequest]):
    """Classify multiple videos in batch"""
    try:
        if tokenizer is None or classifier_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        results = []
        for request in requests:
            try:
                # Get embeddings for each channel
                texts = [request.title, request.description, request.tags]
                embeddings = get_jina_embeddings_1024(texts)
                
                # Prepare input for classifier
                channel_features = torch.tensor(embeddings).unsqueeze(0)
                
                # Run inference
                with torch.no_grad():
                    logits = classifier_model(channel_features)
                    probabilities = softmax(logits.numpy()[0])
                    predicted_label = int(np.argmax(probabilities))
                
                results.append({
                    "aid": request.aid,
                    "label": predicted_label,
                    "probabilities": probabilities.tolist()
                })
                
            except Exception as e:
                logger.error(f"Batch classification error for aid {request.aid}: {str(e)}")
                results.append({
                    "aid": request.aid,
                    "label": -1,
                    "probabilities": [],
                    "error": str(e)
                })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Batch classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8544)