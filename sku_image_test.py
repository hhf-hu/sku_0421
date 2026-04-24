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
from transformers import CLIPProcessor, CLIPModel
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
    """评估所有图片与所有图片的相似度"""
    model.eval()
    all_image_features = []
    all_indices = []
    all_image_paths = []
    all_captions = []

    # 首先提取所有图片特征
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取图片特征"):
            if batch is None:
                continue

            pixel_values = batch["pixel_values"].to(device)

            # 获取图像的嵌入
            image_outputs = model.vision_model(pixel_values=pixel_values)
            image_features = model.visual_projection(image_outputs.last_hidden_state[:, 0, :])

            # 归一化嵌入
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_image_features.append(image_features.cpu())
            all_indices.extend(batch["indices"])
            all_image_paths.extend(batch["image_paths"])
            all_captions.extend(batch["captions"])

    # 合并所有特征
    all_image_features = torch.cat(all_image_features, dim=0)

    # 计算图片相似度矩阵（使用softmax）
    logit_scale = model.logit_scale.exp().cpu() if hasattr(model, 'logit_scale') else 100.0
    similarity_matrix = logit_scale * torch.matmul(all_image_features, all_image_features.t())

    # 将对角线元素设置为负无穷，避免图片与自身匹配
    similarity_matrix = similarity_matrix - torch.diag(torch.ones(similarity_matrix.shape[0]) * float('inf'))

    similarity_matrix = torch.softmax(similarity_matrix, dim=1)

    return similarity_matrix.detach().numpy(), all_indices, all_image_paths, all_captions


def calculate_topk_accuracy(similarity_matrix, captions, topk=2):
    """计算图片到图片检索的top-k准确率，并统计正确和错误样本"""
    n = similarity_matrix.shape[0]
    correct_topk = 0
    correct_indices = []  # 存储正确样本的索引
    incorrect_indices = []  # 存储错误样本的索引

    for i in range(n):
        query_category = captions[i]
        similarities = similarity_matrix[i]

        # 排除自身（相似度最高的是自己）
        topk_indices = np.argsort(similarities)[-topk:][::-1]

        found_correct = False
        for idx in topk_indices:
            if captions[idx] == query_category:
                found_correct = True
                break

        if found_correct:
            correct_topk += 1
            correct_indices.append(i)
        else:
            incorrect_indices.append(i)

    print("correct_indices:",len(correct_indices))
    print("incorrect_indices",len(incorrect_indices))
    accuracy_topk = correct_topk / n
    return accuracy_topk


def calculate_f1_score_topk(similarity_matrix, captions, topk=2):
    """计算基于topk的F1分数 - 图片到图片检索"""
    n = similarity_matrix.shape[0]
    y_true = []
    y_pred = []

    for i in range(n):
        y_true.append(1)  # 正样本
        query_category = captions[i]
        similarities = similarity_matrix[i]
        topk_indices = np.argsort(similarities)[-topk:][::-1]

        found_correct = False
        for idx in topk_indices:
            if captions[idx] == query_category:
                found_correct = True
                break

        y_pred.append(1 if found_correct else 0)

    # 计算F1分数
    f1 = f1_score(y_true, y_pred)
    return f1, y_true, y_pred


def calculate_pr_curve_topk(similarity_matrix, captions, topk=2):
    """计算基于Top-k的PR曲线和AUC - 图片到图片检索"""
    n = similarity_matrix.shape[0]
    y_true = []
    y_scores = []

    for i in range(n):
        query_category = captions[i]
        similarities = similarity_matrix[i]
        topk_indices = np.argsort(similarities)[-topk:][::-1]
        topk_scores = similarities[topk_indices]

        contains_positive = False
        for idx in topk_indices:
            if captions[idx] == query_category:
                contains_positive = True
                break

        for j, idx in enumerate(topk_indices):
            if contains_positive:
                y_true.append(1)
            else:
                y_true.append(1 if captions[idx] == query_category else 0)
            y_scores.append(topk_scores[j])

    # 计算PR曲线
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    return precision, recall, pr_auc, y_true, y_scores


def calculate_pr_curve(similarity_matrix, captions):
    """计算完整的PR曲线和AUC（使用总样本的一半作为正负样本）- 图片到图片检索"""
    n = similarity_matrix.shape[0]
    y_true = []
    y_scores = []

    # 收集所有正样本对和负样本对
    positive_pairs = []
    negative_pairs = []

    for i in range(n):
        for j in range(n):
            if i == j:  # 跳过自身比较
                continue

            # 如果图片类别相同，则为正样本
            if captions[i] == captions[j]:
                positive_pairs.append(similarity_matrix[i, j])
            else:
                negative_pairs.append(similarity_matrix[i, j])

    # 计算需要采样的数量（总样本的一半）
    total_samples = n * (n - 1)  # 排除自身比较
    sample_size = total_samples // 2

    # 确保正负样本数量均衡（各占一半）
    print("len(positive_pairs)",len(positive_pairs),"len(negative_pairs)：",len(negative_pairs))
    positive_sample_size = min(sample_size // 2, len(positive_pairs))
    negative_sample_size = min(sample_size // 2, len(positive_pairs))
    print("positive_sample_size:",positive_sample_size,"negative_sample_size:",negative_sample_size,"sample_size:",sample_size)


    # 如果正样本或负样本不足，调整采样数量
    if positive_sample_size + negative_sample_size < sample_size:
        sample_size = positive_sample_size + negative_sample_size

    # 随机采样正负样本
    random.seed(36)  # 设置随机种子以确保可重复性

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


def compare_models_pr_curve(new_model_similarity, old_model_similarity, captions, output_dir, topk=None):
    """比较两个模型的PR曲线 - 图片到图片检索"""
    if topk is None:
        # 完整PR曲线比较
        new_precision, new_recall, new_pr_auc, _, _ = calculate_pr_curve(new_model_similarity, captions)
        old_precision, old_recall, old_pr_auc, _, _ = calculate_pr_curve(old_model_similarity, captions)
        title_suffix = ' (Image-to-Image)'
        filename_suffix = '_image_to_image'
    else:
        # Top-k PR曲线比较
        new_precision, new_recall, new_pr_auc, _, _ = calculate_pr_curve_topk(new_model_similarity, captions, topk)
        old_precision, old_recall, old_pr_auc, _, _ = calculate_pr_curve_topk(old_model_similarity, captions, topk)
        title_suffix = f' (Top-{topk}, Image-to-Image)'
        filename_suffix = f'_top{topk}_image_to_image'

    # 绘制对比图
    plt.figure(figsize=(12, 8))
    plt.plot(new_recall, new_precision, label=f'New Model (AUC = {new_pr_auc:.4f})', color='blue', linewidth=6)
    plt.plot(old_recall, old_precision, label=f'Old Model (AUC = {old_pr_auc:.4f})', color='red', linewidth=6)
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

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # print(checkpoint)
        state_dict = checkpoint['state_dict']

        # 处理状态字典键名（移除可能的'module.'前缀）
        if any(key.startswith('module.') for key in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        print(f"已加载模型权重从: {model_path}")
    else:
        print(f"警告: 未找到模型文件 {model_path}，使用预训练权重")
        model = CLIPModel.from_pretrained(model_name).to(device)

    return model


def save_topk_predictions_format(similarity_matrix, image_paths, captions, output_path, topk=2):
    """按照指定格式保存Top-k预测结果到CSV文件 - 图片到图片检索"""
    n = similarity_matrix.shape[0]
    results = []

    for query_index in range(n):
        query_category = captions[query_index]
        query_image = os.path.basename(image_paths[query_index])

        similarities = similarity_matrix[query_index]
        topk_indices = np.argsort(similarities)[-topk:][::-1]
        topk_scores = similarities[topk_indices].tolist()

        topk_image_files = [os.path.basename(image_paths[idx]) for idx in topk_indices]
        topk_categories = [captions[idx] for idx in topk_indices]

        is_correct = False
        for idx in topk_indices:
            if captions[idx] == query_category:
                is_correct = True
                break

        result = {
            'query_index': query_index,
            'query_image': query_image,
            'query_category': query_category,
            'topk_indices': json.dumps(topk_indices.tolist()),
            'topk_images': json.dumps(topk_image_files),
            'topk_categories': json.dumps(topk_categories),
            'topk_similarities': json.dumps(topk_scores),
            'is_correct': is_correct
        }
        results.append(result)

    # 创建DataFrame并保存
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Top-{topk} 图片到图片预测结果已保存到: {output_path}")

    return df_results


def test_model(use_main_category_only=False):
    """主测试函数 - 图片到图片检索"""
    # 配置参数
    TEST_DATA_DIR = "./character_data/test_F"  # 测试数据目录
    IMAGE_DIR = "./data/image_9"
    TEST_CSV_PATH = "./data/test.csv"  # 修改为测试集路径
    test_dirs = [TEST_DATA_DIR, IMAGE_DIR]  # 改为目录列表
    # NEW_MODEL_PATH = "dfn5b_checkpoints_character/model_best_similarity.pth.tar"
    NEW_MODEL_PATH = "/home/greaterheat/trademark_similar/save_models/model_best_similarity0917.pth.tar"
    OLD_MODEL_NAME = "apple/DFN5B-CLIP-ViT-H-14-378"
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
    # test_df2 = pd.read_csv(TEST_CSV_PATH)
    # test_df2 = test_df2[['image', 'caption']]
    # test_df = pd.concat([test_df1, test_df2], ignore_index=True)
    print(f"测试集大小: {len(test_df)}")

    # 创建数据集和数据加载器
    test_dataset = DFN5BTestImageCaptionDataset(test_df, test_dirs, processor)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 加载新模型
    new_model = load_model(NEW_MODEL_PATH, OLD_MODEL_NAME, device)

    # 加载旧模型
    old_model = CLIPModel.from_pretrained(OLD_MODEL_NAME).to(device)

    # 评估新模型
    print("正在评估新模型...")
    new_similarity_matrix, indices, image_paths, captions = evaluate_all_pairs(new_model, test_dataloader, device)

    # 评估旧模型
    print("正在评估旧模型...")
    old_similarity_matrix, _, _, _ = evaluate_all_pairs(old_model, test_dataloader, device)

    print(f"\n{'=' * 60}")
    print(f"评估模式: Image-to-Image Retrieval")
    print(f"{'=' * 60}")

    # 计算准确率
    print("计算准确率...")
    new_accuracy_topk = calculate_topk_accuracy(new_similarity_matrix, captions, topk=TOPK)
    old_accuracy_topk = calculate_topk_accuracy(old_similarity_matrix, captions, topk=TOPK)

    print(f"新模型 Top-{TOPK} 图片到图片检索准确率: {new_accuracy_topk:.4f}")
    print(f"旧模型 Top-{TOPK} 图片到图片检索准确率: {old_accuracy_topk:.4f}")

    # 计算F1分数
    print("计算F1分数...")
    new_f1, new_y_true, new_y_pred = calculate_f1_score_topk(new_similarity_matrix, captions, topk=TOPK)
    old_f1, old_y_true, old_y_pred = calculate_f1_score_topk(old_similarity_matrix, captions, topk=TOPK)

    print(f"新模型 Top-{TOPK} 图片到图片检索 F1分数: {new_f1:.4f}")
    print(f"旧模型 Top-{TOPK} 图片到图片检索 F1分数: {old_f1:.4f}")

    # 计算PR曲线和AUC
    print("计算PR曲线...")
    new_precision_topk, new_recall_topk, new_pr_auc_topk, _, _ = calculate_pr_curve_topk(new_similarity_matrix,
                                                                                         captions,
                                                                                         topk=TOPK)
    old_precision_topk, old_recall_topk, old_pr_auc_topk, _, _ = calculate_pr_curve_topk(old_similarity_matrix,
                                                                                         captions,
                                                                                         topk=TOPK)

    print(f"新模型 Top-{TOPK} 图片到图片检索 PR-AUC: {new_pr_auc_topk:.4f}")
    print(f"旧模型 Top-{TOPK} 图片到图片检索 PR-AUC: {old_pr_auc_topk:.4f}")

    # 绘制并保存PR曲线
    new_pr_path = os.path.join(output_dir, f"new_model_pr_curve_top{TOPK}.png")
    plot_pr_curve(new_precision_topk, new_recall_topk, new_pr_auc_topk, f"New Model (Top-{TOPK})", new_pr_path)

    old_pr_path = os.path.join(output_dir, f"old_model_pr_curve_top{TOPK}.png")
    plot_pr_curve(old_precision_topk, old_recall_topk, old_pr_auc_topk, f"Old Model (Top-{TOPK})", old_pr_path)

    # 比较两个模型的PR曲线
    print("比较两个模型的PR曲线...")
    new_pr_auc, old_pr_auc = compare_models_pr_curve(new_similarity_matrix, old_similarity_matrix, captions,
                                                     output_dir, topk=TOPK)

    # 计算完整PR曲线（使用总样本的一半作为正负样本）
    print("计算完整PR曲线（总样本的一半作为正负样本）...")
    new_precision_full, new_recall_full, new_pr_auc_full, _, _ = calculate_pr_curve(new_similarity_matrix, captions)
    old_precision_full, old_recall_full, old_pr_auc_full, _, _ = calculate_pr_curve(old_similarity_matrix, captions)

    print(f"新模型完整PR-AUC (均衡采样): {new_pr_auc_full:.4f}")
    print(f"旧模型完整PR-AUC (均衡采样): {old_pr_auc_full:.4f}")

    # 绘制并保存完整PR曲线
    new_pr_full_path = os.path.join(output_dir, f"new_model_pr_curve_full_balanced.png")
    plot_pr_curve(new_precision_full, new_recall_full, new_pr_auc_full, "New Model (Balanced)", new_pr_full_path)

    old_pr_full_path = os.path.join(output_dir, f"old_model_pr_curve_full_balanced.png")
    plot_pr_curve(old_precision_full, old_recall_full, old_pr_auc_full, "Old Model (Balanced)", old_pr_full_path)

    # 比较完整PR曲线
    print("比较两个模型的完整PR曲线（均衡采样）...")
    plt.figure(figsize=(12, 8))
    plt.plot(new_recall_full, new_precision_full, label=f'New Model (AUC = {new_pr_auc_full:.4f})', color='blue',
             linewidth=6)
    plt.plot(old_recall_full, old_precision_full, label=f'Old Model (AUC = {old_pr_auc_full:.4f})', color='red',
             linewidth=6)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('PR Curve Comparison: New Model vs Old Model (Balanced Sampling)', fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_full_path = os.path.join(output_dir, f"pr_curve_comparison_full_balanced.png")
    plt.savefig(comparison_full_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"完整PR曲线对比图已保存: {comparison_full_path}")

    # 保存预测结果
    print("保存预测结果...")
    new_predictions_path = os.path.join(output_dir, f"new_model_top{TOPK}_predictions.csv")
    save_topk_predictions_format(new_similarity_matrix, image_paths, captions, new_predictions_path, topk=TOPK)

    old_predictions_path = os.path.join(output_dir, f"old_model_top{TOPK}_predictions.csv")
    save_topk_predictions_format(old_similarity_matrix, image_paths, captions, old_predictions_path, topk=TOPK)

    # 保存评估结果到文本文件
    results_path = os.path.join(output_dir, f"evaluation_results_top{TOPK}.txt")
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"数据集大小: {len(test_df)}\n")
        f.write(f"标签类型: {'仅主类别' if use_main_category_only else '完整标签'}\n")
        f.write(f"评估模式: Image-to-Image Retrieval\n")
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
    test_model(use_main_category_only=False)

    # # 测试仅使用主类别
    # print("\n" + "=" * 50)
    # print("测试仅使用主类别...")
    # print("=" * 50)
    # test_model(use_main_category_only=True)
