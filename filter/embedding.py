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
    model = StaticModel.from_pretrained("./model/embedding/")
    for channel in ["title", "description", "tags", "author_info"]:
        texts = batch_data[channel]  # 获取当前通道的文本列表
        embeddings = torch.from_numpy(model.encode(texts)).to(torch.float32).to(device)  # 编码为 [batch_size, embedding_dim]
        channel_embeddings.append(embeddings)
    
    # 2. 将编码结果堆叠为 [batch_size, num_channels, embedding_dim]
    batch_tensor = torch.stack(channel_embeddings, dim=1)  # 在 dim=1 上堆叠
    return batch_tensor