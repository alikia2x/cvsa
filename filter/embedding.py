import numpy as np
import torch
from model2vec import StaticModel


def prepare_batch(batch_data, device="cpu"):
    """
    将输入的 batch_data 转换为模型所需的输入格式 [batch_size, num_channels, embedding_dim]。
    
    参数:
        batch_data (dict): 输入的 batch 数据，格式为 {
            "title": [text1, text2, ...],
            "description": [text1, text2, ...],
            "tags": [text1, text2, ...]
        }
        device (str): 模型运行的设备（如 "cpu" 或 "cuda"）。
    
    返回:
        torch.Tensor: 形状为 [batch_size, num_channels, embedding_dim] 的张量。
    """
    # 1. 对每个通道的文本分别编码
    channel_embeddings = []
    model = StaticModel.from_pretrained("./model/embedding_1024/")
    for channel in ["title", "description", "tags"]:
        texts = batch_data[channel]  # 获取当前通道的文本列表
        embeddings = torch.from_numpy(model.encode(texts)).to(torch.float32).to(device)  # 编码为 [batch_size, embedding_dim]
        channel_embeddings.append(embeddings)
    
    # 2. 将编码结果堆叠为 [batch_size, num_channels, embedding_dim]
    batch_tensor = torch.stack(channel_embeddings, dim=1)  # 在 dim=1 上堆叠
    return batch_tensor

import onnxruntime as ort

def prepare_batch_per_token(session, tokenizer, batch_data, max_length=1024):
    """
    将输入的 batch_data 转换为模型所需的输入格式 [batch_size, num_channels, seq_length, embedding_dim]。

    参数:
        batch_data (dict): 输入的 batch 数据，格式为 {
            "title": [text1, text2, ...],
            "description": [text1, text2, ...],
            "tags": [text1, text2, ...]
        }
        max_length (int): 最大序列长度。

    返回:
        torch.Tensor: 形状为 [batch_size, num_channels, max_length, embedding_dim] 的张量。
    """

    batch_size = len(batch_data["title"])
    batch_tensor = torch.zeros(batch_size, 3, max_length, 256)
    for i in range(batch_size):
        channel_embeddings = torch.zeros((3, 1024, 256))
        for j, channel in enumerate(["title", "description", "tags"]):
            # 获取当前通道的文本
            text = batch_data[channel][i]  
            encoded_inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors='np')

            # embeddings: [max_length, embedding_dim]
            embeddings = torch.zeros((1024, 256))
            for idx, token in enumerate(encoded_inputs['input_ids'][0]):
                inputs = {
                    "input_ids": ort.OrtValue.ortvalue_from_numpy(np.array([token])),
                    "offsets": ort.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64))
                }
                output = session.run(None, inputs)[0]
                embeddings[idx] =  torch.from_numpy(output)
            channel_embeddings[j] = embeddings
        batch_tensor[i] = channel_embeddings

    return batch_tensor