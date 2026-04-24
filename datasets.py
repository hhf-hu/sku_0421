import os
import pandas as pd

def generate_dataset_info(data_dir, use_main_category_only=False):
    """从文件夹结构生成数据集信息"""
    image_paths = []
    captions = []

    for main_category in os.listdir(data_dir):
        main_category_path = os.path.join(data_dir, main_category)
        if not os.path.isdir(main_category_path):
            continue

        for sub_category in os.listdir(main_category_path):
            sub_category_path = os.path.join(main_category_path, sub_category)
            if not os.path.isdir(sub_category_path):
                continue

            if use_main_category_only:
                label = main_category
            else:
                label = f"{main_category}_{sub_category}"

            for image_file in os.listdir(sub_category_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    rel_path = os.path.join(main_category, sub_category, image_file)
                    image_paths.append(rel_path)
                    captions.append(label)

    return pd.DataFrame({'image': image_paths, 'caption': captions})

character_path = "/Users/vincent/workspace/trademark_similar/character_data/"#"/data1/vincent/datasets/data_gray/"

TRAIN_DATA_DIR = character_path + "train"
TEST_DATA_DIR = character_path + "val"

image_train = [TRAIN_DATA_DIR]
image_test = [TEST_DATA_DIR]
MODEL_NAME = "/data1/vincent/models/apple-DFN5B-CLIP-ViT-H-14-378"
BATCH_SIZE = 8
NUM_EPOCHS = 1000
LEARNING_RATE = 5e-6
SCHEDULER_TYPE = "cosine_with_restarts"
CHECKPOINT_DIR = "ds_dfn5b_checkpoints_sku"
RESUME = True
use_main_category_only = False

# 创建检查点目录
# if local_rank == 0 and not os.path.exists(CHECKPOINT_DIR):
#     os.makedirs(CHECKPOINT_DIR)
#
# # --- 数据加载和准备 ---
# if local_rank == 0:
#     print("正在生成数据集信息...")
#     print(f"正在使用 {world_size} 个GPU进行训练")

train_df = generate_dataset_info(TRAIN_DATA_DIR, use_main_category_only=use_main_category_only)
val_df = generate_dataset_info(TEST_DATA_DIR, use_main_category_only=use_main_category_only)