import os
import random
import shutil
from pathlib import Path
from collections import defaultdict


def count_images_in_directory(directory):
    """
    统计指定目录及其所有子目录中的图片数量，并包含排名信息

    Args:
        directory (str): 要统计的根目录路径

    Returns:
        dict: 包含各级目录图片数量的统计结果和排名
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    # 存储统计结果
    stats = {
        'total_images': 0,
        'data_folders': defaultdict(lambda: {
            'total_images': 0,
            'subfolder_count': 0,
            'subfolders': defaultdict(int)
        }),
        'rankings': {
            'by_total_images': [],
            'by_subfolder_count': []
        }
    }

    # 遍历目录
    for root, dirs, files in os.walk(directory):
        # 计算当前目录中的图片数量
        image_count = sum(1 for file in files
                          if Path(file).suffix.lower() in image_extensions)

        if image_count > 0:
            # 获取相对路径
            rel_path = os.path.relpath(root, directory)
            path_parts = rel_path.split(os.sep)

            # 如果是data1-1这样的次级目录
            if len(path_parts) == 2:
                data_folder = path_parts[0]  # data1
                sub_folder = path_parts[1]  # data1-1

                stats['data_folders'][data_folder]['total_images'] += image_count
                stats['data_folders'][data_folder]['subfolders'][sub_folder] = image_count
                stats['data_folders'][data_folder]['subfolder_count'] = len(
                    stats['data_folders'][data_folder]['subfolders'])

            # 如果是data1这样的主目录（直接包含图片的情况）
            elif len(path_parts) == 1 and path_parts[0] != '.':
                data_folder = path_parts[0]
                stats['data_folders'][data_folder]['total_images'] += image_count
                stats['data_folders'][data_folder]['subfolders']['root'] = image_count
                stats['data_folders'][data_folder]['subfolder_count'] = len(
                    stats['data_folders'][data_folder]['subfolders'])

        stats['total_images'] += image_count

    # 生成排名
    _generate_rankings(stats)

    return stats
def _generate_rankings(stats):
    """生成各种排名"""
    data_folders = stats['data_folders']

    # 按总图片数量排名
    total_images_ranking = sorted(
        [(folder, info['total_images']) for folder, info in data_folders.items()],
        key=lambda x: x[1],
        reverse=True
    )

    # 按子文件夹数量排名
    subfolder_count_ranking = sorted(
        [(folder, info['subfolder_count']) for folder, info in data_folders.items()],
        key=lambda x: x[1],
        reverse=True
    )

    # 为每个主文件夹内的子文件夹生成排名
    for folder, info in data_folders.items():
        subfolders_ranking = sorted(
            [(subfolder, count) for subfolder, count in info['subfolders'].items()],
            key=lambda x: x[1],
            reverse=True
        )
        info['subfolders_ranking'] = subfolders_ranking

    stats['rankings']['by_total_images'] = total_images_ranking
    stats['rankings']['by_subfolder_count'] = subfolder_count_ranking


def print_statistics_rank(stats):
    """打印统计结果和排名"""
    print(f"总图片数量: {stats['total_images']}")
    print("\n" + "=" * 50)

    # 按总图片数量排名
    print("\n主文件夹排名（按图片数量从多到少）:")
    print("-" * 40)
    for i, (folder, count) in enumerate(stats['rankings']['by_total_images'], 1):
        print(f"{i:2d}. {folder}: {count} 张图片")

    # 按子文件夹数量排名
    print("\n主文件夹排名（按子文件夹数量从多到少）:")
    print("-" * 40)
    for i, (folder, count) in enumerate(stats['rankings']['by_subfolder_count'], 1):
        print(f"{i:2d}. {folder}: {count} 个子文件夹")

    # 详细统计
    print("\n详细统计信息:")
    print("-" * 40)
    for folder, info in stats['rankings']['by_total_images']:
        folder_info = stats['data_folders'][folder]
        print(f"\n{folder}:")
        print(f"  - 总图片数: {folder_info['total_images']}")
        print(f"  - 子文件夹数: {folder_info['subfolder_count']}")

        # 子文件夹排名
        print("  - 子文件夹图片数量排名:")
        for j, (subfolder, count) in enumerate(folder_info['subfolders_ranking'], 1):
            print(f"    {j:2d}. {subfolder}: {count} 张图片")


def print_statistics(stats):
    """
    打印统计结果，把汇总信息放到最后
    """
    print("=" * 60)
    print("图片数量详细统计")
    print("=" * 60)

    # 先打印每个次级目录的详细情况
    for data_folder, data_info in stats['data_folders'].items():
        print(f"\n{data_folder} 的次级目录详情:")
        print("-" * 40)

        for sub_folder, count in data_info['subfolders'].items():
            print(f"  {sub_folder}: {count} 张图片")

    print("\n" + "=" * 60)
    print("最终汇总统计")
    print("=" * 60)

    # 最后打印每个data文件夹的汇总信息
    total_sub = 0
    for data_folder, data_info in stats['data_folders'].items():
        total_sub = total_sub+data_info['subfolder_count']

        print(f"{data_folder}:")
        print(f"  子目录数量: {data_info['subfolder_count']} 个")
        print(f"  总图片数量: {data_info['total_images']} 张")
        print("-" * 30)


    # 打印总计
    print("total sub:",total_sub)
    print(f"\n所有图片总数: {stats['total_images']} 张")
    print("=" * 60)


def main():
    # 指定要统计的目录
    base_directory = "/Users/vincent/workspace/trademark_similar/character/character_data/train/animal"  # 修改为你的实际目录路径

    if not os.path.exists(base_directory):
        print(f"错误：目录 '{base_directory}' 不存在！")
        return

    print(f"正在统计目录: {os.path.abspath(base_directory)}")

    # 统计图片数量
    stats = count_images_in_directory(base_directory)

    # 打印结果
    print_statistics(stats)
    print_statistics_rank(stats)


def move_random_images(source_dir, target_dir, num_images=1):
    """
    从每个子目录中随机移动指定数量的图片到目标目录，保持目录结构

    Args:
        source_dir (str): 源目录路径
        target_dir (str): 目标目录路径
        num_images (int): 每个子目录要移动的图片数量（1或2）
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    # 存储统计信息
    stats = defaultdict(int)
    total_moved = 0

    # 遍历所有子目录
    for root, dirs, files in os.walk(source_dir):
        # 获取当前目录中的图片文件
        image_files = [f for f in files if Path(f).suffix.lower() in image_extensions]

        if not image_files:
            continue

        # 随机选择图片
        # num_images_ =  int(0.2*len(image_files))
        num_to_select = min(num_images, len(image_files))
        selected_images = random.sample(image_files, num_to_select)

        # 获取相对路径
        rel_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, rel_path)

        # 创建目标目录
        os.makedirs(target_path, exist_ok=True)

        # 移动选中的图片
        for img_name in selected_images:
            src_file = os.path.join(root, img_name)
            dst_file = os.path.join(target_path, img_name)

            # 如果目标文件已存在，添加后缀避免冲突
            counter = 1
            while os.path.exists(dst_file):
                name, ext = os.path.splitext(img_name)
                dst_file = os.path.join(target_path, f"{name}_{counter}{ext}")
                counter += 1

            # 移动文件
            shutil.move(src_file, dst_file)
            stats[rel_path] += 1
            total_moved += 1

    return stats, total_moved


def main11():
    # 配置参数
    source_directory = "/Users/vincent/workspace/trademark_similar/character/character_data/dataset/dataset"  # 源目录
    target_directory = "/Users/vincent/workspace/trademark_similar/character/character_data/train/animal"  # 目标目录
    images_per_folder = 18  # 每个子目录移动的图片数量（1或2）

    # 检查源目录是否存在
    if not os.path.exists(source_directory):
        print(f"错误：源目录 '{source_directory}' 不存在！")
        return

    print(f"从目录: {os.path.abspath(source_directory)}")
    print(f"移动每个子目录 {images_per_folder} 张图片到: {os.path.abspath(target_directory)}")
    print("正在处理...")

    # 执行移动操作
    stats, total_moved = move_random_images(source_directory, target_directory, images_per_folder)

    # 打印结果
    print("\n" + "=" * 60)
    print("图片移动完成！")
    print("=" * 60)

    print(f"\n总共移动了 {total_moved} 张图片")
    print("\n各目录移动情况:")
    print("-" * 40)

    for folder, count in stats.items():
        print(f"{folder}: {count} 张图片")

    print(f"\n所有图片已移动到: {os.path.abspath(target_directory)}")

if __name__ == "__main__":
    main()
    # main11()