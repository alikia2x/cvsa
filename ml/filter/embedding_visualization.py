import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load("1.npy")

# 绘制直方图，获取频数
n, bins, patches = plt.hist(data, bins=32, density=False, alpha=0.7, color='skyblue')

# 计算数据总数
total_data = len(data)

# 将频数转换为频率
frequencies = n / total_data

# 计算统计信息
max_val = np.max(data)
min_val = np.min(data)
std_dev = np.std(data)

# 设置图形属性
plt.title('Frequency Distribution Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 重新绘制直方图，使用频率作为高度
plt.cla()  # 清除当前坐标轴上的内容
plt.bar([(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)], frequencies, width=[bins[i+1]-bins[i] for i in range(len(bins)-1)], alpha=0.7, color='skyblue')

# 在柱子上注明频率值
for i in range(len(patches)):
    plt.text(bins[i]+(bins[i+1]-bins[i])/2, frequencies[i], f'{frequencies[i]:.2e}', ha='center', va='bottom', fontsize=6)

# 在图表一角显示统计信息
stats_text = f"Max: {max_val:.6f}\nMin: {min_val:.6f}\nStd: {std_dev:.4e}"
plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
         ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

# 设置 x 轴刻度对齐柱子边界
plt.xticks(bins, fontsize = 6)

# 显示图形
plt.show()