import torch

from modelV3_10 import VideoClassifierV3_10
from modelV3_9 import VideoClassifierV3_9


def convert_checkpoint(original_model, new_model):
    """转换原始checkpoint到新结构"""
    state_dict = original_model.state_dict()
    
    # 直接复制所有参数（因为结构保持兼容）
    new_model.load_state_dict(state_dict)
    return new_model

# 使用示例
original_model = VideoClassifierV3_9()
new_model = VideoClassifierV3_10()

# 加载原始checkpoint
original_model.load_state_dict(torch.load('./filter/checkpoints/best_model_V3.9.pt'))

# 转换参数
converted_model = convert_checkpoint(original_model, new_model)

# 保存转换后的模型
torch.save(converted_model.state_dict(), './filter/checkpoints/best_model_V3.10.pt')