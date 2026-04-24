import os
import shutil
from pathlib import Path


def copy_first_n_images(src_root='.', dest_root='bad', n=5):
    """
    从每个Unit*文件夹复制前n张图片到bad/Unit*文件夹

    参数:
        src_root: 源文件夹根目录
        dest_root: 目标文件夹根目录
        n: 要复制的图片数量
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif'}

    # 遍历源目录下的所有文件夹
    for item in os.listdir(src_root):
        item_path = os.path.join(src_root, item)

        # 检查是否为Unit开头的文件夹
        if os.path.isdir(item_path) and item.startswith('Unit'):
            # 获取文件夹中所有图片文件
            image_files = []
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in image_extensions:
                        image_files.append(file)

            # 按文件名排序（如果需要按其他方式排序，可以修改这里）
            image_files.sort()

            # 只取前n个文件
            files_to_copy = image_files[:n]

            if not files_to_copy:
                print(f"警告: {item} 中没有找到图片文件")
                continue

            # 创建目标文件夹
            dest_folder = os.path.join(dest_root, item)
            os.makedirs(dest_folder, exist_ok=True)

            # 复制文件
            copied_count = 0
            for filename in files_to_copy:
                src_file = os.path.join(item_path, filename)
                dest_file = os.path.join(dest_folder, filename)

                # 如果目标文件已存在，可以选择跳过或覆盖
                if not os.path.exists(dest_file):
                    shutil.copy2(src_file, dest_file)
                    copied_count += 1
                else:
                    # 如果要覆盖已存在的文件，取消下面的注释
                    # shutil.copy2(src_file, dest_file)
                    # copied_count += 1
                    pass

            print(f"成功从 {item} 复制 {copied_count}/{len(files_to_copy)} 个文件到 {dest_folder}")


if __name__ == "__main__":
    copy_first_n_images()