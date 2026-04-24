import matplotlib.pyplot as plt


def plot_heatmap(matrix, cmap='viridis', save_path=None):
    """
    绘制矩阵热力图

    参数:
    matrix: numpy数组
    cmap: 颜色映射，可选值: 'viridis', 'plasma', 'hot', 'coolwarm', 'YlOrRd', 'Reds'等
    save_path: 保存图片的路径
    """
    plt.figure(figsize=(12, 10))

    # 创建热力图
    im = plt.imshow(matrix,
                    cmap=cmap,  # 颜色映射
                    aspect='auto',  # 保持原始纵横比
                    interpolation='nearest')  # 不进行插值

    # 添加颜色条
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('label', rotation=270, labelpad=15)

    # 添加标题和标签
    plt.title(f'similarity matrix heatmap ({matrix.shape[0]}×{matrix.shape[1]})', fontsize=16)
    plt.xlabel('sku', fontsize=12)
    plt.ylabel('sku', fontsize=12)

    # 优化显示
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    # plt.show()

    # 打印一些统计信息
    print(f"矩阵形状: {matrix.shape}")
    print(f"数值范围: {matrix.min():.4f} ~ {matrix.max():.4f}")
    print(f"平均值: {matrix.mean():.4f}")
    print(f"标准差: {matrix.std():.4f}")