import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, precision_recall_curve, auc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from transformers import CLIPProcessor, CLIPModel, AutoModel
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def generate_dataset_info(data_dir, use_main_category_only=False):
    """
    从文件夹结构生成数据集信息
    格式: data1/data1-1/image1.jpg -> 标签: "data1" (仅主类别) 或 "data1_data1-1" (完整标签)
    """
    image_paths = []
    captions = []

    # 遍历主类别文件夹 (data1, data2, ...)
    for main_category in os.listdir(data_dir):
        main_category_path = os.path.join(data_dir, main_category)
        if not os.path.isdir(main_category_path):
            continue

        # 遍历子类别文件夹 (data1-1, data1-2, ...)
        for sub_category in os.listdir(main_category_path):
            sub_category_path = os.path.join(main_category_path, sub_category)
            if not os.path.isdir(sub_category_path):
                continue

            # 生成标签
            if use_main_category_only:
                label = main_category  # 仅使用主类别
            else:
                label = f"{main_category}_{sub_category}"  # 使用完整标签

            # 遍历图片文件
            for image_file in os.listdir(sub_category_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    # 相对路径: main_category/sub_category/image_file
                    rel_path = os.path.join(main_category, sub_category, image_file)
                    image_paths.append(rel_path)
                    captions.append(label)

    return pd.DataFrame({'image': image_paths, 'caption': captions})


class DFN5BTestImageCaptionDataset(Dataset):
    def __init__(self, df, image_dirs, processor):
        self.df = df
        self.image_dirs = image_dirs  # 改为接收目录列表
        self.processor = processor
        self.image_paths = df['image'].tolist()
        self.captions = df['caption'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # 尝试在多个目录中查找图片
        found_image = None
        for image_dir in self.image_dirs:
            full_path = os.path.join(image_dir, image_path)
            if os.path.exists(full_path):
                found_image = full_path
                break

        if found_image is None:
            print(f"警告：找不到图片文件 {image_path}，将跳过此样本。")
            return None

        try:
            image = Image.open(found_image).convert("RGB")
        except Exception as e:
            print(f"警告：无法打开图片文件 {found_image}: {e}，将跳过此样本。")
            return None

        caption = self.captions[idx]

        try:
            # 使用CLIP处理器处理图像和文本
            inputs = self.processor(text=caption, images=image, return_tensors="pt", padding="max_length",
                                    truncation=True)

            return {
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "pixel_values": inputs["pixel_values"].squeeze(),
                "image_path": found_image,
                "caption": caption,
                "index": idx  # 添加索引以便后续匹配
            }
        except Exception as e:
            print(f"处理图像或文本时出错: {e}")
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    image_paths = [item["image_path"] for item in batch]
    captions = [item["caption"] for item in batch]
    indices = [item["index"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_paths": image_paths,
        "captions": captions,
        "indices": indices
    }


def evaluate_all_pairs(model, dataloader, device):
    """评估所有文本与所有图片的相似度"""
    model.eval()
    all_image_features = []
    all_text_features = []
    all_indices = []
    all_image_paths = []
    all_captions = []

    # 首先提取所有特征
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            if batch is None:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            # 获取图像和文本的嵌入
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

            # 归一化嵌入
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
            all_indices.extend(batch["indices"])
            all_image_paths.extend(batch["image_paths"])
            all_captions.extend(batch["captions"])

    # 合并所有特征
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)

    # 计算相似度矩阵（使用softmax）
    logit_scale = model.logit_scale.exp().cpu() if hasattr(model, 'logit_scale') else 100.0
    similarity_matrix = logit_scale * torch.matmul(all_text_features, all_image_features.t())
    similarity_matrix = torch.softmax(similarity_matrix, dim=1)
    from sklearn.metrics.pairwise import pairwise_distances

    return similarity_matrix.detach().numpy(), all_indices, all_image_paths, all_captions


def calculate_topk_accuracy(similarity_matrix, captions, topk=2, mode='text_to_image'):
    """计算top-k准确率 - 修改为类别匹配而非索引匹配"""
    n = similarity_matrix.shape[0]
    correct_topk = 0

    if mode == 'text_to_image':
        # 文本检索图片
        for i in range(n):
            text_category = captions[i]
            similarities = similarity_matrix[i]
            topk_indices = np.argsort(similarities)[-topk:][::-1]

            found_correct = False
            for idx in topk_indices:
                if captions[idx] == text_category:
                    found_correct = True
                    break

            if found_correct:
                correct_topk += 1
    else:
        # 图片检索文本 - 转置相似度矩阵
        image_to_text_similarity = similarity_matrix.T
        n_images = image_to_text_similarity.shape[0]

        for i in range(n_images):
            image_category = captions[i]
            similarities = image_to_text_similarity[i]
            topk_indices = np.argsort(similarities)[-topk:][::-1]

            found_correct = False
            for idx in topk_indices:
                if captions[idx] == image_category:
                    found_correct = True
                    break

            if found_correct:
                correct_topk += 1

    accuracy_topk = correct_topk / n
    return accuracy_topk


def calculate_f1_score_topk(similarity_matrix, captions, topk=2, mode='text_to_image'):
    """计算基于topk的F1分数 - 修改为类别匹配而非索引匹配"""
    n = similarity_matrix.shape[0]
    y_true = []
    y_pred = []

    if mode == 'text_to_image':
        # 文本检索图片
        for i in range(n):
            y_true.append(1)  # 正样本
            text_category = captions[i]
            similarities = similarity_matrix[i]
            topk_indices = np.argsort(similarities)[-topk:][::-1]

            found_correct = False
            for idx in topk_indices:
                if captions[idx] == text_category:
                    found_correct = True
                    break

            y_pred.append(1 if found_correct else 0)
    else:
        # 图片检索文本
        image_to_text_similarity = similarity_matrix.T
        n_images = image_to_text_similarity.shape[0]

        for i in range(n_images):
            y_true.append(1)  # 正样本
            image_category = captions[i]
            similarities = image_to_text_similarity[i]
            topk_indices = np.argsort(similarities)[-topk:][::-1]

            found_correct = False
            for idx in topk_indices:
                if captions[idx] == image_category:
                    found_correct = True
                    break

            y_pred.append(1 if found_correct else 0)

    # 计算F1分数
    f1 = f1_score(y_true, y_pred)
    return f1, y_true, y_pred


def calculate_pr_curve_topk(similarity_matrix, captions, topk=2, mode='text_to_image'):
    """计算基于Top-k的PR曲线和AUC - 修改为类别匹配而非索引匹配"""
    n = similarity_matrix.shape[0]
    y_true = []
    y_scores = []

    if mode == 'text_to_image':
        # 文本检索图片
        for i in range(n):
            text_category = captions[i]
            similarities = similarity_matrix[i]
            topk_indices = np.argsort(similarities)[-topk:][::-1]
            topk_scores = similarities[topk_indices]

            contains_positive = False
            for idx in topk_indices:
                if captions[idx] == text_category:
                    contains_positive = True
                    break

            for j, idx in enumerate(topk_indices):
                if contains_positive:
                    y_true.append(1)
                else:
                    y_true.append(1 if captions[idx] == text_category else 0)
                y_scores.append(topk_scores[j])
    else:
        # 图片检索文本
        image_to_text_similarity = similarity_matrix.T
        n_images = image_to_text_similarity.shape[0]

        for i in range(n_images):
            image_category = captions[i]
            similarities = image_to_text_similarity[i]
            topk_indices = np.argsort(similarities)[-topk:][::-1]
            topk_scores = similarities[topk_indices]

            contains_positive = False
            for idx in topk_indices:
                if captions[idx] == image_category:
                    contains_positive = True
                    break

            for j, idx in enumerate(topk_indices):
                if contains_positive:
                    y_true.append(1)
                else:
                    y_true.append(1 if captions[idx] == image_category else 0)
                y_scores.append(topk_scores[j])

    # 计算PR曲线
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    return precision, recall, pr_auc, y_true, y_scores


def calculate_pr_curve(similarity_matrix, captions, mode='text_to_image'):
    """计算完整的PR曲线和AUC（使用总样本的一半作为正负样本）- 修改为类别匹配而非索引匹配"""
    if mode == 'image_to_text':
        similarity_matrix = similarity_matrix.T

    n = similarity_matrix.shape[0]
    y_true = []
    y_scores = []

    # 收集所有正样本对和负样本对
    positive_pairs = []
    negative_pairs = []

    for i in range(n):
        for j in range(n):
            # 如果图片类别与文本类别相同，则为正样本
            if captions[i] == captions[j]:
                positive_pairs.append(similarity_matrix[i, j])
            else:
                negative_pairs.append(similarity_matrix[i, j])

    # 计算需要采样的数量（总样本的一半）
    total_samples = n * n
    sample_size = total_samples // 2

    # 确保正负样本数量均衡（各占一半）
    print("len(positive_pairs)", len(positive_pairs), "len(negative_pairs)：", len(negative_pairs))
    positive_sample_size = min(sample_size // 2, len(positive_pairs))
    negative_sample_size = min(sample_size // 2, len(positive_pairs))
    print("positive_sample_size:", positive_sample_size, "negative_sample_size:", negative_sample_size, "sample_size:",
          sample_size)

    # 如果正样本或负样本不足，调整采样数量
    if positive_sample_size + negative_sample_size < sample_size:
        sample_size = positive_sample_size + negative_sample_size

    # 随机采样正负样本
    random.seed(42)  # 设置随机种子以确保可重复性

    if positive_pairs:
        sampled_positive = random.sample(positive_pairs, positive_sample_size)
    else:
        sampled_positive = []

    if negative_pairs:
        sampled_negative = random.sample(negative_pairs, negative_sample_size)
    else:
        sampled_negative = []

    # 添加正样本
    for score in sampled_positive:
        y_true.append(1)
        y_scores.append(score)

    # 添加负样本
    for score in sampled_negative:
        y_true.append(0)
        y_scores.append(score)

    # 计算PR曲线
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    return precision, recall, pr_auc, y_true, y_scores


def plot_pr_curve(precision, recall, pr_auc, model_name, output_path):
    """绘制PR曲线并保存"""
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compare_models_pr_curve(new_model_similarity, old_model_similarity, captions, output_dir, topk=None,
                            mode='text_to_image'):
    """比较两个模型的PR曲线"""
    if topk is None:
        # 完整PR曲线比较
        new_precision, new_recall, new_pr_auc, _, _ = calculate_pr_curve(new_model_similarity, captions, mode=mode)
        old_precision, old_recall, old_pr_auc, _, _ = calculate_pr_curve(old_model_similarity, captions, mode=mode)
        title_suffix = f' ({mode.replace("_", " ").title()})'
        filename_suffix = f'_{mode}'
    else:
        # Top-k PR曲线比较
        new_precision, new_recall, new_pr_auc, _, _ = calculate_pr_curve_topk(new_model_similarity, captions, topk,
                                                                              mode=mode)
        old_precision, old_recall, old_pr_auc, _, _ = calculate_pr_curve_topk(old_model_similarity, captions, topk,
                                                                              mode=mode)
        title_suffix = f' (Top-{topk}, {mode.replace("_", " ").title()})'
        filename_suffix = f'_top{topk}_{mode}'

    # 绘制对比图
    plt.figure(figsize=(12, 8))
    plt.plot(new_recall, new_precision, label=f'New Model (AUC = {new_pr_auc:.4f})', color='blue', linewidth=6)
    plt.plot(old_recall, old_precision, label=f'Old Model (AUC = {old_pr_auc:.4f})', color='red', linewidth=6, )
    # linestyle='--')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'PR Curve Comparison: New Model vs Old Model{title_suffix}', fontsize=14)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存对比图
    comparison_path = os.path.join(output_dir, f"pr_curve_comparison{filename_suffix}.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"PR曲线对比图已保存: {comparison_path}")
    print(f"新模型 PR-AUC: {new_pr_auc:.4f}")
    print(f"旧模型 PR-AUC: {old_pr_auc:.4f}")

    return new_pr_auc, old_pr_auc


def load_model(model_path, model_name, device):
    """加载模型"""
    print(f"正在加载模型: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device)

    # 检查是否是目录（包含safetensors分片文件）
    if os.path.isdir(model_path):
        model_dir = model_path
        # 检查目录中是否有safetensors文件
        safetensors_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]

        if safetensors_files:
            print(f"找到 {len(safetensors_files)} 个safetensors分片文件")
            try:
                # 直接从包含safetensors文件的目录加载模型
                model = AutoModel.from_pretrained(model_dir).to(device)
                print(f"已从目录加载safetensors分片模型: {model_dir}")
                return model
            except Exception as e:
                print(f"从目录加载safetensors模型失败: {e}")
                # 如果失败，尝试使用transformers的safetensors支持
                try:
                    from transformers import modeling_utils
                    model = AutoModel.from_pretrained(model_dir, use_safetensors=True).to(device)
                    print(f"使用safetensors加载成功: {model_dir}")
                    return model
                except Exception as e2:
                    print(f"使用safetensors加载也失败: {e2}")
                    print("尝试使用预训练模型...")
                    model = AutoModel.from_pretrained(model_name).to(device)
                    return model
        else:
            # 目录中没有safetensors文件，尝试常规加载
            print("目录中未找到safetensors文件，尝试常规加载...")
            try:
                model = AutoModel.from_pretrained(model_dir).to(device)
                print(f"已从目录加载模型: {model_dir}")
                return model
            except Exception as e:
                print(f"从目录加载失败: {e}")
                print("使用预训练模型...")
                model = AutoModel.from_pretrained(model_name).to(device)
                return model
    else:
        # 如果提供的路径不是目录，使用预训练模型
        print(f"提供的模型路径不是目录: {model_path}，使用预训练模型")
        model = AutoModel.from_pretrained(model_name).to(device)
        return model


def save_topk_predictions_format(similarity_matrix, image_paths, captions, output_path, topk=2, mode='text_to_image'):
    """按照指定格式保存Top-k预测结果到CSV文件 - 修改为类别匹配而非索引匹配"""
    if mode == 'image_to_text':
        similarity_matrix = similarity_matrix.T

    n = similarity_matrix.shape[0]
    results = []

    if mode == 'text_to_image':
        # 文本检索图片
        for text_index in range(n):
            text_category = captions[text_index]
            similarities = similarity_matrix[text_index]
            topk_indices = np.argsort(similarities)[-topk:][::-1]
            topk_scores = similarities[topk_indices].tolist()

            topk_image_files = [os.path.basename(image_paths[idx]) for idx in topk_indices]
            topk_categories = [captions[idx] for idx in topk_indices]

            is_correct = False
            for idx in topk_indices:
                if captions[idx] == text_category:
                    is_correct = True
                    break

            true_image_index = text_index
            true_image_file = os.path.basename(image_paths[true_image_index])

            result = {
                'mode': 'text_to_image',
                'query_index': text_index,
                'query_category': text_category,
                'true_target_index': true_image_index,
                'true_target_file': true_image_file,
                'topk_indices': json.dumps(topk_indices.tolist()),
                'topk_files': json.dumps(topk_image_files),
                'topk_categories': json.dumps(topk_categories),
                'topk_similarities': json.dumps(topk_scores),
                'is_correct': is_correct
            }
            results.append(result)
    else:
        # 图片检索文本
        for image_index in range(n):
            image_category = captions[image_index]
            similarities = similarity_matrix[image_index]
            topk_indices = np.argsort(similarities)[-topk:][::-1]
            topk_scores = similarities[topk_indices].tolist()

            topk_text_categories = [captions[idx] for idx in topk_indices]

            is_correct = False
            for idx in topk_indices:
                if captions[idx] == image_category:
                    is_correct = True
                    break

            true_text_index = image_index
            true_text_category = captions[true_text_index]

            result = {
                'mode': 'image_to_text',
                'query_index': image_index,
                'query_category': image_category,
                'true_target_index': true_text_index,
                'true_target_category': true_text_category,
                'topk_indices': json.dumps(topk_indices.tolist()),
                'topk_categories': json.dumps(topk_text_categories),
                'topk_similarities': json.dumps(topk_scores),
                'is_correct': is_correct
            }
            results.append(result)

    # 创建DataFrame并保存
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Top-{topk} {mode} 预测结果已保存到: {output_path}")

    return df_results


def test_model(use_main_category_only=False):
    """主测试函数"""
    # 配置参数
    character_path = "/data1/vincent/datasets/data/"
    # IMAGE_DIR1 = character_path + "data_930/image_9"
    # IMAGE_DIR2 = character_path + "data_930"
    TEST_DATA_DIR = character_path + "test"

    # TEST_CSV_PATH1 = character_path + "data_930/test.csv"
    # TEST_CSV_PATH2 = character_path + "data_930/test_0930.csv"
    # test_dirs = [TEST_DATA_DIR, IMAGE_DIR1, IMAGE_DIR2]

    test_dirs = [TEST_DATA_DIR]
    # 修改模型路径为DeepSpeed检查点目录
    NEW_MODEL_PATH = "/data1/vincent/sku/ds_dfn5b_checkpoints_sku/output_dir/"
    OLD_MODEL_NAME = "/data1/vincent/models/apple-DFN5B-CLIP-ViT-H-14-378"

    BATCH_SIZE = 16
    TOPK = 2

    # 根据标签类型设置输出目录
    label_type = "main" if use_main_category_only else "full"
    output_dir = f"dfn5b_evaluation_results_character_image_to_image_{label_type}"
    os.makedirs(output_dir, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载处理器
    processor = CLIPProcessor.from_pretrained(OLD_MODEL_NAME)

    # 生成数据集信息
    print("正在生成数据集信息...")
    test_df = generate_dataset_info(TEST_DATA_DIR, use_main_category_only=use_main_category_only)
    # test_df2 = pd.read_csv(TEST_CSV_PATH1)
    # test_df2 = test_df2[['image', 'caption']]
    # test_df3 = pd.read_csv(TEST_CSV_PATH2)
    # test_df3 = test_df3[['image', 'caption']]

    # test_df = pd.concat([test_df1, test_df2, test_df3], ignore_index=True)
    # test_df = pd.concat([test_df1, test_df2], ignore_index=True)
    print(f"测试集大小: {len(test_df)}")

    # 创建数据集和数据加载器
    test_dataset = DFN5BTestImageCaptionDataset(test_df, test_dirs, processor)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 加载新模型（使用修改后的加载函数）
    new_model = load_model(NEW_MODEL_PATH, OLD_MODEL_NAME, device)

    # 加载旧模型
    old_model = CLIPModel.from_pretrained(OLD_MODEL_NAME).to(device)

    # 评估新模型
    print("正在评估新模型...")
    new_similarity_matrix, indices, image_paths, captions = evaluate_all_pairs(new_model, test_dataloader, device)

    # 评估旧模型
    print("正在评估旧模型...")
    old_similarity_matrix, _, _, _ = evaluate_all_pairs(old_model, test_dataloader, device)

    # 定义评估模式
    modes = ['text_to_image', 'image_to_text']

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"评估模式: {mode.replace('_', ' ').title()}")
        print(f"{'=' * 60}")

        mode_dir = os.path.join(output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        # 计算准确率
        print("计算准确率...")
        new_accuracy_topk = calculate_topk_accuracy(new_similarity_matrix, captions, topk=TOPK, mode=mode)
        old_accuracy_topk = calculate_topk_accuracy(old_similarity_matrix, captions, topk=TOPK, mode=mode)

        print(f"新模型 Top-{TOPK} {mode} 准确率: {new_accuracy_topk:.4f}")
        print(f"旧模型 Top-{TOPK} {mode} 准确率: {old_accuracy_topk:.4f}")

        # 计算F1分数
        print("计算F1分数...")
        for i in range(5):
            TOPK = 1 + i
            new_f1, new_y_true, new_y_pred = calculate_f1_score_topk(new_similarity_matrix, captions, topk=TOPK, mode=mode)
            old_f1, old_y_true, old_y_pred = calculate_f1_score_topk(old_similarity_matrix, captions, topk=TOPK, mode=mode)

            print(f"新模型 Top-{TOPK} {mode} F1分数: {new_f1:.4f}")
            print(f"旧模型 Top-{TOPK} {mode} F1分数: {old_f1:.4f}")

        # 计算PR曲线和AUC
        print("计算PR曲线...")
        new_precision_topk, new_recall_topk, new_pr_auc_topk, _, _ = calculate_pr_curve_topk(new_similarity_matrix,
                                                                                             captions,
                                                                                             topk=TOPK, mode=mode)
        old_precision_topk, old_recall_topk, old_pr_auc_topk, _, _ = calculate_pr_curve_topk(old_similarity_matrix,
                                                                                             captions,
                                                                                             topk=TOPK, mode=mode)

        print(f"新模型 Top-{TOPK} {mode} PR-AUC: {new_pr_auc_topk:.4f}")
        print(f"旧模型 Top-{TOPK} {mode} PR-AUC: {old_pr_auc_topk:.4f}")

        # 绘制并保存PR曲线
        new_pr_path = os.path.join(mode_dir, f"new_model_pr_curve_top{TOPK}.png")
        plot_pr_curve(new_precision_topk, new_recall_topk, new_pr_auc_topk, f"New Model (Top-{TOPK}, {mode})",
                      new_pr_path)

        old_pr_path = os.path.join(mode_dir, f"old_model_pr_curve_top{TOPK}.png")
        plot_pr_curve(old_precision_topk, old_recall_topk, old_pr_auc_topk, f"Old Model (Top-{TOPK}, {mode})",
                      old_pr_path)

        # 比较两个模型的PR曲线
        print("比较两个模型的PR曲线...")
        new_pr_auc, old_pr_auc = compare_models_pr_curve(new_similarity_matrix, old_similarity_matrix, captions,
                                                         mode_dir,
                                                         topk=TOPK, mode=mode)

        # 计算完整PR曲线（使用总样本的一半作为正负样本）
        print("计算完整PR曲线（总样本的一半作为正负样本）...")
        new_precision_full, new_recall_full, new_pr_auc_full, _, _ = calculate_pr_curve(
            new_similarity_matrix, captions, mode=mode
        )
        old_precision_full, old_recall_full, old_pr_auc_full, _, _ = calculate_pr_curve(
            old_similarity_matrix, captions, mode=mode
        )

        print(f"新模型完整PR-AUC (均衡采样): {new_pr_auc_full:.4f}")
        print(f"旧模型完整PR-AUC (均衡采样): {old_pr_auc_full:.4f}")

        # 绘制并保存完整PR曲线
        new_pr_full_path = os.path.join(mode_dir, f"new_model_pr_curve_full_balanced.png")
        plot_pr_curve(new_precision_full, new_recall_full, new_pr_auc_full, f"New Model (Balanced, {mode})",
                      new_pr_full_path)

        old_pr_full_path = os.path.join(mode_dir, f"old_model_pr_curve_full_balanced.png")
        plot_pr_curve(old_precision_full, old_recall_full, old_pr_auc_full, f"Old Model (Balanced, {mode})",
                      old_pr_full_path)

        # 比较完整PR曲线
        print("比较两个模型的完整PR曲线（均衡采样）...")
        plt.figure(figsize=(12, 8))
        plt.plot(new_recall_full, new_precision_full, label=f'New Model (AUC = {new_pr_auc_full:.4f})', color='blue',
                 linewidth=6)
        plt.plot(old_recall_full, old_precision_full, label=f'Old Model (AUC = {old_pr_auc_full:.4f})', color='red',
                 linewidth=6, )
        # linestyle='--')
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('PR Curve Comparison: New Model vs Old Model (Balanced Sampling)', fontsize=16)
        plt.legend(loc='lower left', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        comparison_full_path = os.path.join(output_dir, f"pr_curve_comparison_full_balanced_{mode}.png")
        plt.savefig(comparison_full_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"完整PR曲线对比图已保存: {comparison_full_path}")

        # 保存预测结果
        print("保存预测结果...")
        new_predictions_path = os.path.join(mode_dir, f"new_model_top{TOPK}_predictions.csv")
        save_topk_predictions_format(new_similarity_matrix, image_paths, captions, new_predictions_path, topk=TOPK,
                                     mode=mode)

        old_predictions_path = os.path.join(mode_dir, f"old_model_top{TOPK}_predictions.csv")
        save_topk_predictions_format(old_similarity_matrix, image_paths, captions, old_predictions_path, topk=TOPK,
                                     mode=mode)

        # 保存评估结果到文本文件
        results_path = os.path.join(mode_dir, f"evaluation_results_top{TOPK}.txt")
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(f"数据集大小: {len(test_df)}\n")
            f.write(f"标签类型: {'仅主类别' if use_main_category_only else '完整标签'}\n")
            f.write(f"评估模式: {mode.replace('_', ' ').title()}\n")
            f.write(f"Top-K: {TOPK}\n")
            f.write(f"PR曲线样本: 均衡采样（总样本的一半）\n\n")
            f.write(f"新模型 Top-{TOPK} 准确率: {new_accuracy_topk:.4f}\n")
            f.write(f"旧模型 Top-{TOPK} 准确率: {old_accuracy_topk:.4f}\n\n")
            f.write(f"新模型 Top-{TOPK} F1分数: {new_f1:.4f}\n")
            f.write(f"旧模型 Top-{TOPK} F1分数: {old_f1:.4f}\n\n")
            f.write(f"新模型 Top-{TOPK} PR-AUC: {new_pr_auc_topk:.4f}\n")
            f.write(f"旧模型 Top-{TOPK} PR-AUC: {old_pr_auc_topk:.4f}\n\n")
            f.write(f"新模型完整PR-AUC (均衡采样): {new_pr_auc_full:.4f}\n")
            f.write(f"旧模型完整PR-AUC (均衡采样): {old_pr_auc_full:.4f}\n")

        print(f"评估结果已保存到: {results_path}")


if __name__ == "__main__":
    # 测试使用完整标签
    print("=" * 50)
    print("测试使用完整标签...")
    print("=" * 50)
    test_model(use_main_category_only=True)

    # # 测试仅使用主类别
    # print("\n" + "=" * 50)
    # print("测试仅使用主类别...")
    # print("=" * 50)
    # test_model(use_main_category_only=True)
