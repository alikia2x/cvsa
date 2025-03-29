from safetensors import safe_open
from safetensors.torch import save_file
import torch

# 配置路径
model_path = "./model/embedding/model.safetensors"
save_path = "./model/embedding/int8_model.safetensors"

# 加载原始嵌入层
with safe_open(model_path, framework="pt") as f:
    embeddings_tensor = f.get_tensor("embeddings")

# 计算极值
min_val = torch.min(embeddings_tensor)
max_val = torch.max(embeddings_tensor)

# 计算量化参数
scale = (max_val - min_val) / 255  # int8 的范围是 256 个值（-128 到 127）

# 将浮点数映射到 int8 范围
int8_tensor = torch.round((embeddings_tensor - min_val) / scale).to(torch.int8) - 128

# 确保与原张量形状一致
assert int8_tensor.shape == embeddings_tensor.shape

# 保存映射后的 int8 张量
save_file({"embeddings": int8_tensor}, save_path)

# 输出反映射公式
print("int8 反映射公式：")
m = min_val.item()
am = abs(min_val.item())
sign = "-" if m < 0 else "+"
print(f"int8_tensor = (int8_value + 128) × {scale.item()} {sign} {am}")

print("int8 映射完成！")