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
            "tags": [text1, text2, ...],
            "author_info": [text1, text2, ...]
        }
        device (str): 模型运行的设备（如 "cpu" 或 "cuda"）。
    
    返回:
        torch.Tensor: 形状为 [batch_size, num_channels, embedding_dim] 的张量。
    """
    # 1. 对每个通道的文本分别编码
    channel_embeddings = []
    model = StaticModel.from_pretrained("./model/embedding_1024/")
    for channel in ["title", "description", "tags", "author_info"]:
        texts = batch_data[channel]  # 获取当前通道的文本列表
        embeddings = torch.from_numpy(model.encode(texts)).to(torch.float32).to(device)  # 编码为 [batch_size, embedding_dim]
        channel_embeddings.append(embeddings)
    
    # 2. 将编码结果堆叠为 [batch_size, num_channels, embedding_dim]
    batch_tensor = torch.stack(channel_embeddings, dim=1)  # 在 dim=1 上堆叠
    return batch_tensor

import onnxruntime as ort
from transformers import AutoTokenizer
from itertools import accumulate

def prepare_batch_per_token(batch_data, max_length=1024):
    """
    将输入的 batch_data 转换为模型所需的输入格式 [batch_size, num_channels, seq_length, embedding_dim]。

    参数:
        batch_data (dict): 输入的 batch 数据，格式为 {
            "title": [text1, text2, ...],
            "description": [text1, text2, ...],
            "tags": [text1, text2, ...],
            "author_info": [text1, text2, ...]
        }
        max_length (int): 最大序列长度。

    返回:
        torch.Tensor: 形状为 [batch_size, num_channels, seq_length, embedding_dim] 的张量。
    """
    # 初始化 tokenizer 和 ONNX 模型
    tokenizer = AutoTokenizer.from_pretrained("alikia2x/jina-embedding-v3-m2v-1024")
    session = ort.InferenceSession("./model/embedding_256/onnx/model.onnx")

    # 1. 对每个通道的文本分别编码
    channel_embeddings = []
    for channel in ["title", "description", "tags", "author_info"]:
        texts = batch_data[channel]  # 获取当前通道的文本列表

        # Step 1: 生成 input_ids 和 offsets
        # 对每个文本单独编码，保留原始 token 长度
        encoded_inputs = [tokenizer(text, truncation=True, max_length=max_length, return_tensors='np') for text in texts]

        # 提取每个文本的 input_ids 长度（考虑实际的 token 数量）
        input_ids_lengths = [len(enc["input_ids"][0]) for enc in encoded_inputs]

        # 生成 offsets: [0, len1, len1+len2, ...]
        offsets = list(accumulate([0] + input_ids_lengths[:-1]))  # 累积和，排除最后一个长度

        # 将所有 input_ids 展平为一维数组
        flattened_input_ids = np.concatenate([enc["input_ids"][0] for enc in encoded_inputs], axis=0).astype(np.int64)

        # Step 2: 构建 ONNX 输入
        inputs = {
            "input_ids": ort.OrtValue.ortvalue_from_numpy(flattened_input_ids),
            "offsets": ort.OrtValue.ortvalue_from_numpy(np.array(offsets, dtype=np.int64))
        }

        # Step 3: 运行 ONNX 模型
        embeddings = session.run(None, inputs)[0]  # 假设输出名为 "embeddings"

        # Step 4: 将输出重塑为 [batch_size, seq_length, embedding_dim]
        # 注意：这里假设 ONNX 输出的形状是 [total_tokens, embedding_dim]
        # 需要根据实际序列长度重新分组
        batch_size = len(texts)
        embeddings_split = np.split(embeddings, np.cumsum(input_ids_lengths[:-1]))
        padded_embeddings = []
        for emb, seq_len in zip(embeddings_split, input_ids_lengths):
            # 对每个序列填充到 max_length
            if seq_len > max_length:
                # 如果序列长度超过 max_length，截断
                emb = emb[:max_length]
                pad_length = 0
            else:
                # 否则填充到 max_length
                pad_length = max_length - seq_len

            # 填充到 [max_length, embedding_dim]
            padded = np.pad(emb, ((0, pad_length), (0, 0)), mode='constant')
            padded_embeddings.append(padded)

        # 确保所有填充后的序列形状一致
        embeddings_tensor = torch.tensor(np.stack(padded_embeddings), dtype=torch.float32)
        channel_embeddings.append(embeddings_tensor)

    # 2. 将编码结果堆叠为 [batch_size, num_channels, seq_length, embedding_dim]
    batch_tensor = torch.stack(channel_embeddings, dim=1)
    return batch_tensor