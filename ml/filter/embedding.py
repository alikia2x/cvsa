from typing import List
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer

# 初始化 tokenizer 和 ONNX 模型（全局缓存）
_tokenizer = None
_onnx_session = None

def _get_tokenizer_and_session():
    """获取全局缓存的 tokenizer 和 ONNX session"""
    global _tokenizer, _onnx_session
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
    if _onnx_session is None:
        _onnx_session = ort.InferenceSession("../model/embedding/model.onnx")
    return _tokenizer, _onnx_session


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
    
    title_embeddings = get_jina_embeddings_1024(batch_data['title'])
    desc_embeddings = get_jina_embeddings_1024(batch_data['description'])
    tags_embeddings = get_jina_embeddings_1024(batch_data['tags'])

    return torch.stack([title_embeddings, desc_embeddings, tags_embeddings], dim=1).to(device)


def get_jina_embeddings_1024(texts: List[str]) -> np.ndarray:
    """Get Jina embeddings using tokenizer and ONNX-like processing"""
    [tokenizer, session] = _get_tokenizer_and_session()

    encoded_inputs = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None,  # 返回原生Python列表，便于后续处理
    )
    input_ids = encoded_inputs[
        "input_ids"
    ]  # 形状: [batch_size, seq_len_i]（每个样本长度可能不同）

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
        "offsets": ort.OrtValue.ortvalue_from_numpy(np.array(offsets, dtype=np.int64)),
    }

    # 5. 运行模型推理
    outputs = session.run(None, inputs)
    embeddings = outputs[
        0
    ]  # 假设第一个输出是embeddings，形状: [batch_size, embedding_dim]

    return torch.tensor(embeddings, dtype=torch.float32)


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
        from itertools import accumulate
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