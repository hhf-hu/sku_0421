import numpy as np
import matplotlib.pyplot as plt


# 1. 从txt文件读取矩阵
def read_matrix_from_txt(filename):
    """从txt文件读取矩阵数据"""
    with open(filename, 'r') as f:
        # 读取所有行，转换为浮点数
        matrix = []
        for line in f:
            # 处理每行的数据，假设数据以空格或逗号分隔
            row = []
            for item in line.strip().split():
                try:
                    # 尝试转换为浮点数
                    row.append(float(item))
                except:
                    # 如果直接分隔不行，尝试其他分隔符
                    for sep in [',', ';', '\t']:
                        if sep in line:
                            row = [float(x) for x in line.strip().split(sep)]
                            break
            if row:  # 避免空行
                matrix.append(row)

    # 转换为numpy数组
    return np.array(matrix)


# 2. 绘制热力图
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

    plt.show()

    # 打印一些统计信息
    print(f"矩阵形状: {matrix.shape}")
    print(f"数值范围: {matrix.min():.4f} ~ {matrix.max():.4f}")
    print(f"平均值: {matrix.mean():.4f}")
    print(f"标准差: {matrix.std():.4f}")


def read_label_matrix(filename):
    """
    读取包含字符串标签的矩阵文件
    格式示例：['label1', 'label2', 'label3']
    """
    import ast
    import re

    labels_matrix = []

    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue

            # 处理方括号包围的列表格式
            if line.startswith('[') and line.endswith(']'):
                try:
                    # 使用 ast.literal_eval 安全地解析列表
                    row_labels = ast.literal_eval(line)
                    if isinstance(row_labels, list):
                        labels_matrix.append(row_labels)
                    else:
                        print(f"警告: 第 {line_num} 行不是列表格式")
                except:
                    # 如果解析失败，尝试手动处理
                    # 移除两端的方括号
                    content = line[1:-1]
                    # 分割逗号分隔的字符串
                    items = [item.strip().strip("'\"") for item in content.split(',')]
                    labels_matrix.append(items)
            else:
                # 如果不是列表格式，直接按行处理
                labels_matrix.append([line])

    return labels_matrix


def count_specific_label(filename, target_label):
    """
    统计文件中特定标签出现的总次数
    """
    count = 0

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('[') and line.endswith(']'):
                # 移除方括号，分割标签
                content = line[1:-1]
                labels = [item.strip().strip("'\"") for item in content.split(',')]
                # 统计当前行中目标标签的数量
                count += labels.count(target_label)
            elif target_label in line:
                # 整行匹配
                count += 1

    return count


# 3. 使用示例
if __name__ == "__main__":

    # 替换为您的txt文件路径
    txt_file = "/sku/matrix/new_model_matrix.npy"  # 请修改为您的文件路径

    # 使用示例
    filename = 'dinov3_labels.txt'
    labels_matrix = read_label_matrix(filename)
    # ll = labels_matrix[0][464]
    # count = count_specific_label('dinov3_labels.txt', 'snoopy_cupsleeve_sku2_RB')
    try:
        # 读取矩阵
        print("正在读取矩阵数据...")
        # matrix = read_matrix_from_txt(txt_file)
        # matrix = np.load(txt_file)
        simi = np.load(txt_file)

        simi = (np.power(simi, 2))
        # simi = np.exp(simi)

        matrix = simi / np.nanmax(simi)
        print(f"成功读取矩阵，形状: {matrix.shape}")

        # 选择颜色映射（数值越大颜色越深）
        # 常用选项:
        # - 'viridis': 紫色到黄色的渐变（推荐，对色盲友好）
        # - 'plasma': 紫色到黄色的另一种渐变
        # - 'hot': 黑色->红色->黄色->白色
        # - 'YlOrRd': 黄色->橙色->红色
        # - 'Reds': 白色到红色的渐变
        # - 'bone': 黑白渐变
        color_map = 'viridis'  # 您可以更换这个

        # 绘制热力图
        print("正在生成热力图...")
        plot_heatmap(matrix, cmap=color_map, save_path='heatmap1211_new.png')

    except FileNotFoundError:
        print(f"错误：找不到文件 {txt_file}")
        print("请确保文件路径正确！")
    except Exception as e:
        print(f"发生错误: {e}")