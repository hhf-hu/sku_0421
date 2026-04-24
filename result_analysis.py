import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
df = pd.read_csv('new_model_top5_predictions.csv')  # 请将文件名替换为您的实际文件名

# 数据预处理：将字符串格式的列表转换为实际的列表
df['topk_similarities'] = df['topk_similarities'].apply(eval)

# 提取数据
query_indices = df['query_index'].values
similarities = df['topk_similarities'].values

# 将相似度数据转换为numpy数组，每列代表一条折线
similarities_array = np.array(similarities)

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制五条折线
colors = ['blue', 'red', 'green', 'orange', 'purple']
labels = ['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5']

for i in range(5):
    plt.plot(query_indices, similarities_array[:, i],
             color=colors[i], label=labels[i], marker='o', markersize=3, linewidth=2)

# 设置图形属性
plt.xlabel('Query Index', fontsize=12)
plt.ylabel('Similarity Score', fontsize=12)
plt.title('Top-5 Similarities for Each Query', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 如果query_index太多，可以调整x轴显示
if len(query_indices) > 20:
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 可选：显示统计信息
print("数据统计信息：")
for i in range(5):
    print(f"{labels[i]}: 平均值={similarities_array[:, i].mean():.4f}, "
          f"最大值={similarities_array[:, i].max():.4f}, "
          f"最小值={similarities_array[:, i].min():.4f}")