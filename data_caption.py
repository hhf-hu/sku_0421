import os
import pandas as pd
import random
from pathlib import Path
from collections import defaultdict


def main():
    # 设置路径
    root_dir = "/Users/vincent/workspace/trademark_similar/character_data/train"

    # 询问划分模式
    print("选择划分模式:")
    print("1. 训练集、验证集、测试集共享类别（每个类别出现在所有数据集中）")
    print("2. 训练集、验证集、测试集类别互斥（每个类别只出现在一个数据集中）")

    choice = input("请选择 (1或2，默认1): ").strip()
    if choice == "2":
        share_categories = False
        output_file = "separate_categories.csv"
    else:
        share_categories = True
        output_file = "shared_categories.csv"

    # 支持的图片扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    # 收集数据
    category_data = defaultdict(list)

    print("扫描图片中...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in image_extensions:
                rel_path = file_path.relative_to(root_dir)
                path_parts = str(rel_path).split(os.sep)

                if len(path_parts) >= 2:
                    main_caption = path_parts[0]
                    sub_caption = f"{path_parts[0]}_{path_parts[1]}"

                    category_data[sub_caption].append({
                        'images': str(file_path),
                        'main_captions': main_caption,
                        'sub_category_captions': sub_caption
                    })

    print(f"找到 {sum(len(imgs) for imgs in category_data.values())} 张图片")

    # 划分数据
    train_list, val_list, test_list = [], [], []

    if share_categories:
        # 模式1: 共享类别
        for category_key in sorted(category_data.keys()):
            images = category_data[category_key]
            random.shuffle(images)

            total = len(images)
            train_end = int(total * 0.7)
            val_end = train_end + int(total * 0.1)

            # 训练集
            for img in images[:train_end]:
                img_copy = img.copy()
                img_copy['train/val/test'] = 'train'
                train_list.append(img_copy)

            # 验证集
            for img in images[train_end:val_end]:
                img_copy = img.copy()
                img_copy['train/val/test'] = 'val'
                val_list.append(img_copy)

            # 测试集
            for img in images[val_end:]:
                img_copy = img.copy()
                img_copy['train/val/test'] = 'test'
                test_list.append(img_copy)
    else:
        # 模式2: 类别互斥
        categories = list(category_data.keys())
        random.shuffle(categories)

        total_categories = len(categories)
        train_cat_end = int(total_categories * 0.7)
        val_cat_end = train_cat_end + int(total_categories * 0.15)

        train_categories = categories[:train_cat_end]
        val_categories = categories[train_cat_end:val_cat_end]
        test_categories = categories[val_cat_end:]

        # 训练集
        for category_key in sorted(train_categories):
            for img in category_data[category_key]:
                img_copy = img.copy()
                img_copy['train/val/test'] = 'train'
                train_list.append(img_copy)

        # 验证集
        for category_key in sorted(val_categories):
            for img in category_data[category_key]:
                img_copy = img.copy()
                img_copy['train/val/test'] = 'val'
                val_list.append(img_copy)

        # 测试集
        for category_key in sorted(test_categories):
            for img in category_data[category_key]:
                img_copy = img.copy()
                img_copy['train/val/test'] = 'test'
                test_list.append(img_copy)

    # 合并数据
    all_data = train_list + val_list + test_list
    df = pd.DataFrame(all_data)
    df_shuffled = df.sample(frac=1, random_state=5).reset_index(drop=True)
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n数据已保存到: {output_file}")
    print(f"训练集: {len(train_list)} 张")
    print(f"验证集: {len(val_list)} 张")
    print(f"测试集: {len(test_list)} 张")

    return df


if __name__ == "__main__":
    main()