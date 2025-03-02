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
    # 1. 对每个通道的文本分别编码
    channel_embeddings = []
    model = StaticModel.from_pretrained("./model/embedding_256/")
    for channel in ["title", "description", "tags", "author_info"]:
        texts = batch_data[channel]  # 获取当前通道的文本列表
        # 使用tokenizer将文本转换为tokens
        encoded_input = model.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        with torch.no_grad():
            model_output = model.model(**encoded_input)
        # 提取最后一个隐藏层的结果
        embeddings = model_output.last_hidden_state.to(torch.float32) # 将embeddings 放在指定device上
        channel_embeddings.append(embeddings)

    # 2. 将编码结果堆叠为 [batch_size, num_channels, seq_length, embedding_dim]
    batch_tensor = torch.stack(channel_embeddings, dim=1)
    return batch_tensor