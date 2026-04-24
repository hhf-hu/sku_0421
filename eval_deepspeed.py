import torch
import deepspeed
from sklearn.metrics import f1_score, precision_recall_curve, auc
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import numpy as np
import shutil
from model_deepspeed import save_checkpoint, load_checkpoint


# 计算PR曲线和AUC
def calculate_pr_curve(similarity_matrix):
    """计算PR曲线和AUC"""
    if similarity_matrix.size == 0:
        return 0.0

    n = similarity_matrix.shape[0]
    y_true = []
    y_scores = []

    # 确保矩阵是方阵或至少行数不超过列数
    min_dim = min(n, similarity_matrix.shape[1])

    for i in range(min_dim):
        for j in range(min_dim):
            y_true.append(1 if i == j else 0)
            y_scores.append(similarity_matrix[i, j])

    # 计算PR曲线
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    return pr_auc


def evaluate(model, dataloader, device, world_size=1):
    # 获取模型（可能是DeepSpeed引擎）
    if hasattr(model, 'module'):
        model_to_eval = model.module
    else:
        model_to_eval = model

    model_to_eval.eval()
    all_preds = []
    all_labels = []
    all_cosine_similarities = []
    all_similarity_matrices = []  # 存储所有批次的相似度矩阵

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            if batch is None:
                continue

            # 将批次数据移动到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            # 获取图像和文本的嵌入
            outputs = model_to_eval(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # 归一化嵌入
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # 计算余弦相似度
            cosine_similarities = (image_embeds * text_embeds).sum(dim=1)
            all_cosine_similarities.extend(cosine_similarities.cpu().numpy())

            # 计算相似度矩阵
            logits_per_image = torch.matmul(image_embeds, text_embeds.t())
            similarity_matrix = torch.softmax(logits_per_image, dim=1)

            # 存储相似度矩阵用于PR曲线计算
            all_similarity_matrices.append(similarity_matrix.cpu())

            # 标签是对角线上的元素
            labels = torch.arange(logits_per_image.shape[0]).to(device)
            preds = logits_per_image.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_labels:
        return 0.0, "", 0.0, 0.0

    # 合并所有相似度矩阵
    if all_similarity_matrices:
        full_similarity_matrix = torch.cat([matrix.view(-1, matrix.shape[-1]) for matrix in all_similarity_matrices],
                                           dim=0)
    else:
        full_similarity_matrix = torch.tensor([])

    # 在多卡训练时，需要收集所有进程的结果
    if world_size > 1:
        all_preds_gathered = [None] * world_size
        all_labels_gathered = [None] * world_size
        all_similarity_matrices_gathered = [None] * world_size
        all_cosine_similarities_gathered = [None] * world_size

        # 使用DeepSpeed的分布式通信
        torch.distributed.all_gather_object(all_preds_gathered, all_preds)
        torch.distributed.all_gather_object(all_labels_gathered, all_labels)
        torch.distributed.all_gather_object(all_similarity_matrices_gathered, full_similarity_matrix.numpy())
        torch.distributed.all_gather_object(all_cosine_similarities_gathered, all_cosine_similarities)

        if torch.distributed.get_rank() == 0:  # 只在主进程计算指标
            all_preds_combined = []
            all_labels_combined = []
            all_cosine_similarities_combined = []

            for preds, labels, cosine_sims in zip(all_preds_gathered, all_labels_gathered,
                                                  all_cosine_similarities_gathered):
                all_preds_combined.extend(preds)
                all_labels_combined.extend(labels)
                all_cosine_similarities_combined.extend(cosine_sims)

            # 计算F1分数和分类报告
            f1 = f1_score(all_labels_combined, all_preds_combined, average='macro')
            cls_rep = classification_report(all_labels_combined, all_preds_combined, target_names=None, zero_division=0)

            # 计算平均余弦相似度
            avg_cosine_similarity = sum(all_cosine_similarities_combined) / len(all_cosine_similarities_combined)

            # 计算PR-AUC
            try:
                combined_similarity_matrix = np.concatenate(all_similarity_matrices_gathered, axis=0)
                pr_auc = calculate_pr_curve(combined_similarity_matrix)
            except Exception as e:
                print(f"计算PR-AUC时出错: {e}")
                pr_auc = 0.0
        else:
            f1, cls_rep, avg_cosine_similarity, pr_auc = 0.0, "", 0.0, 0.0
    else:
        # 单卡计算
        f1 = f1_score(all_labels, all_preds, average='macro')
        cls_rep = classification_report(all_labels, all_preds, target_names=None, zero_division=0)
        avg_cosine_similarity = sum(all_cosine_similarities) / len(all_cosine_similarities)

        # 计算PR-AUC
        try:
            pr_auc = calculate_pr_curve(full_similarity_matrix.numpy())
        except Exception as e:
            print(f"计算PR-AUC时出错: {e}")
            pr_auc = 0.0

    return f1, cls_rep, avg_cosine_similarity, pr_auc