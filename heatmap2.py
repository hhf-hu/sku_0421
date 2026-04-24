import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap_with_labels(matrix, coords, top_n=50, cmap='viridis'):
    """
    在热力图上标注数值
    coords: 要标注的坐标列表 [(row1, col1), (row2, col2), ...]
    top_n: 只标注前N个点
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 图1：整个矩阵的热力图
    im1 = axes[0].imshow(matrix, cmap=cmap, aspect='auto')
    axes[0].set_title(f'完整矩阵热力图 ({matrix.shape[0]}×{matrix.shape[1]})')
    plt.colorbar(im1, ax=axes[0])

    # 图2：局部放大并标注
    # 如果有变化点，自动确定放大区域
    if len(coords) > 0:
        # 获取所有变化点的中心区域
        rows = coords[:top_n, 0]
        cols = coords[:top_n, 1]
        center_row = int(rows.mean())
        center_col = int(cols.mean())

        # 确定显示范围
        window = 20  # 显示窗口大小
        row_start = max(0, center_row - window)
        row_end = min(matrix.shape[0], center_row + window)
        col_start = max(0, center_col - window)
        col_end = min(matrix.shape[1], center_col + window)

        sub_matrix = matrix[row_start:row_end, col_start:col_end]

        im2 = axes[1].imshow(sub_matrix, cmap=cmap, aspect='auto')
        axes[1].set_title(f'局部放大区域 ({row_start}:{row_end}, {col_start}:{col_end})')
        plt.colorbar(im2, ax=axes[1])

        # 在放大的区域标注数值
        for idx, (row, col) in enumerate(coords[:top_n]):
            # 检查点是否在显示区域内
            if row_start <= row < row_end and col_start <= col < col_end:
                # 转换为局部坐标
                local_row = row - row_start
                local_col = col - col_start

                # 获取数值
                value = matrix[row, col]

                # 标注数值
                axes[1].text(local_col, local_row, f'{value:.2f}',
                             ha='center', va='center',
                             color='white' if value > np.median(sub_matrix) else 'black',
                             fontsize=8,
                             bbox=dict(boxstyle="round,pad=0.3",
                                       facecolor='red' if value > np.median(sub_matrix) else 'yellow',
                                       edgecolor='black',
                                       alpha=0.7))
    else:
        # 如果没有变化点，显示整个矩阵的一部分
        sub_matrix = matrix[:50, :50]
        im2 = axes[1].imshow(sub_matrix, cmap=cmap, aspect='auto')
        axes[1].set_title('矩阵前50×50区域')
        plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 假设已经有了matrix和coords
    plot_heatmap_with_labels(matrix, coords, top_n=30)