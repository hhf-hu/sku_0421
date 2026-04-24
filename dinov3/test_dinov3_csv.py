import os
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from safetensors import safe_open
import numpy as np
from accelerate import load_checkpoint_and_dispatch
from tqdm.auto import tqdm
import pandas as pd
from PIL import Image
import traceback

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型路径
model_dir = "/data1/vincent/sku/dinov3/ds_dinov3_checkpoints-0113-2/output_dir"
model_name = "/data1/vincent/models/facebook-dinov3-vith16plus-pretrain-lvd1689m"
matrix_name = "dinov3-vith16plus-pretrain-lvd1689m_similarity_jojo_data-0113"
feature_dim = 1280


# 1. 首先定义与训练时相同的模型结构
class DINOv3ForContrastive(torch.nn.Module):
    def __init__(self, model_name, num_classes=None, feature_dim=2048):
        super().__init__()

        # 加载预训练的DINOv3模型
        self.backbone = AutoModel.from_pretrained(model_name)

        # 获取特征维度
        hidden_size = self.backbone.config.hidden_size

        # 分类头（可选）
        if num_classes is not None:
            self.classifier = torch.nn.Linear(hidden_size, num_classes)
        else:
            self.classifier = None

        # 特征投影头（用于对比学习）
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, feature_dim),
            torch.nn.LayerNorm(feature_dim)
        )

        # 相似度计算的温度参数（可学习）
        self.temperature = torch.nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, pixel_values):
        # 获取图像特征
        outputs = self.backbone(pixel_values=pixel_values)

        # 使用[CLS] token作为图像表示
        features = outputs.last_hidden_state[:, 0, :]

        # 投影特征（用于对比学习）
        projected_features = self.projection_head(features)
        projected_features = torch.nn.functional.normalize(projected_features, dim=-1)

        # 分类（如果启用）
        if self.classifier is not None:
            logits = self.classifier(features)
        else:
            logits = None

        return {
            "features": features,
            "projected_features": projected_features,
            "logits": logits,
            "temperature": self.temperature
        }


# 检查模型目录
print(f"Checking model directory: {model_dir}")
files = os.listdir(model_dir)
print("Files in directory:")
for f in sorted(files):
    print(f"  {f}")

# 加载处理器
print("\nLoading image processor...")
processor = AutoImageProcessor.from_pretrained(model_name, )
# size={"height": 512, "width": 512},

# 创建模型实例
print("Creating model instance...")

model = DINOv3ForContrastive(
    model_name=model_name,
    num_classes=2,
    feature_dim=feature_dim
)

# 加载分片的safetensors权重
print("Loading safetensors weights...")

# 检查是否有分片权重文件
safetensor_files = [f for f in files if f.endswith('.safetensors')]
if safetensor_files:
    print(f"Found {len(safetensor_files)} safetensor files")

    try:
        # 使用load_checkpoint_and_dispatch加载分片权重
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=model_dir,
            # device_map="auto",
            # dtype=torch.float32
        )
        print("Successfully loaded weights with load_checkpoint_and_dispatch")
    except Exception as e:
        print(f"Error loading with load_checkpoint_and_dispatch: {e}")
        print("Trying manual loading...")

        # 手动加载权重
        try:
            # 首先将模型移动到设备
            model = model.to(device)

            # 加载权重文件
            state_dict = {}
            for file in safetensor_files:
                file_path = os.path.join(model_dir, file)
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)

            # 加载权重
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded weights manually")
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")

            if missing_keys:
                print("First 10 missing keys:", missing_keys[:10])
            if unexpected_keys:
                print("First 10 unexpected keys:", unexpected_keys[:10])

        except Exception as e2:
            print(f"Error loading weights manually: {e2}")
            print("Using pretrained model only...")
            model = model.to(device)
else:
    print("No safetensor files found, using pretrained model only")
    model = model.to(device)

# 确保模型在正确的设备上
model = model.to(device)
model.eval()
print(f"Model is on device: {next(model.parameters()).device}")


def extract_embeddings_batch(image_paths, batch_size=4, use_projection=True):
    """批量提取嵌入特征"""
    all_embeddings = []

    # 获取模型所在的设备
    model_device = next(model.parameters()).device
    print(f"Model is on device: {model_device}")

    # 添加进度条
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    pbar = tqdm(total=num_batches, desc="Extracting embeddings", unit="batch")

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]

        try:
            images = []
            valid_indices = []

            for j, p in enumerate(batch_paths):
                # 尝试加载图像
                try:
                    if isinstance(p, str) and os.path.exists(p):
                        img = load_image(p)
                        images.append(img)
                        valid_indices.append(j)
                    else:
                        print(f"Warning: Image path does not exist: {p}")
                        continue
                except Exception as e:
                    print(f"Error loading image {p}: {e}")
                    continue

            if not images:
                print("No valid images in this batch, skipping...")
                pbar.update(1)
                continue

            # 处理图像
            inputs = processor(images=images, return_tensors="pt")

            # 确保输入数据在正确的设备和dtype上
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            # 确保像素值在正确的dtype上（与模型权重匹配）
            if hasattr(model.backbone.config, 'torch_dtype'):
                target_dtype = model.backbone.config.torch_dtype
                if target_dtype:
                    inputs["pixel_values"] = inputs["pixel_values"].to(target_dtype)

            # 前向传播
            with torch.no_grad():
                outputs = model(pixel_values=inputs["pixel_values"])

                if use_projection:
                    emb = outputs["projected_features"]
                else:
                    emb = outputs["features"]

                emb = F.normalize(emb, dim=-1)

            all_embeddings.append(emb.cpu())
            pbar.update(1)

        except Exception as e:
            print(f"Error processing batch {i // batch_size}: {e}")
            traceback.print_exc()
            pbar.update(1)
            continue

    pbar.close()

    if all_embeddings:
        return torch.cat(all_embeddings, dim=0)
    else:
        print("Warning: No embeddings were extracted!")
        return torch.tensor([])


def load_test_data_from_csv(csv_path, use_main_category_only=True):
    """从CSV文件加载测试数据"""
    print(f"Loading test data from CSV: {csv_path}")

    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        print(f"CSV loaded successfully. Total rows: {len(df)}")

        # 筛选测试集数据
        test_df = df[df['train/val/test'].str.strip().str.lower() == 'test'].copy()
        print(f"Test set rows: {len(test_df)}")

        if len(test_df) == 0:
            print("Warning: No test data found in CSV!")
            return [], [], {}

        # 准备数据
        image_paths = []
        labels = []

        for idx, row in test_df.iterrows():
            # 直接从images列获取完整路径
            full_path = row['images']

            # 检查文件是否存在
            if os.path.exists(full_path):
                image_paths.append(full_path)

                # 根据参数选择标签
                if use_main_category_only:
                    label = row['main_captions']
                else:
                    label = row['sub_category_captions']
                labels.append(label)
            else:
                print(f"Warning: Image file not found: {full_path}")

        if len(image_paths) == 0:
            print("Warning: No valid image files found!")
            return [], [], {}

        # 创建标签到ID的映射
        unique_labels = sorted(set(labels))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}

        # 将标签转换为ID
        label_ids = [label_to_id[label] for label in labels]

        print(f"\nTest dataset summary:")
        print(f"  Total images: {len(image_paths)}")
        print(f"  Unique labels: {len(unique_labels)}")
        print(f"  Label mapping: {label_to_id}")

        # 显示前几个样本
        print(f"\nFirst 5 samples:")
        for i in range(min(5, len(image_paths))):
            print(f"  {i + 1}. {os.path.basename(image_paths[i])} -> {labels[i]} (ID: {label_ids[i]})")

        return image_paths, label_ids, label_to_id, id_to_label

    except Exception as e:
        print(f"Error loading CSV file: {e}")
        traceback.print_exc()
        return [], [], {}, {}


def compute_similarity_matrix(embeddings, block_size=256):
    """
    计算相似度矩阵（余弦相似度）
    """
    if embeddings.numel() == 0:
        print("Error: No embeddings to compute similarity matrix")
        return None

    N = embeddings.size(0)
    print(f"Computing similarity matrix for {N} images...")

    # 确保embeddings是归一化的
    embeddings = F.normalize(embeddings, dim=-1)

    # 初始化相似度矩阵
    sim_matrix = torch.zeros((N, N), dtype=torch.float32)

    # 将embeddings移动到GPU（如果可用）
    if torch.cuda.is_available():
        embeddings_gpu = embeddings.to(device)
    else:
        embeddings_gpu = embeddings

    # 分块计算相似度
    num_blocks = (N + block_size - 1) // block_size
    pbar = tqdm(total=num_blocks, desc="Computing similarity", unit="block")

    for i in range(0, N, block_size):
        end_i = min(i + block_size, N)
        emb_i = embeddings_gpu[i:end_i]

        # 计算相似度
        with torch.no_grad():
            sim = emb_i @ embeddings_gpu.T  # [block_size, N]

        sim_matrix[i:end_i] = sim.cpu()

        # 更新进度条
        pbar.update(1)
        pbar.set_postfix({"current": f"{end_i}/{N}"})

    pbar.close()
    return sim_matrix


def predict_labels(sim_matrix, true_labels, k=5):
    """
    基于相似度矩阵进行预测（k最近邻）

    参数:
        sim_matrix: 相似度矩阵 (N x N)
        true_labels: 真实标签列表 (长度N)
        k: 近邻数量

    返回:
        predictions: 预测标签列表
        similarities: 与最近邻的相似度列表
    """
    N = len(true_labels)
    predictions = []
    similarities = []

    print(f"\nPredicting labels using {k}-nearest neighbors...")

    for i in range(N):
        # 获取第i个样本与其他样本的相似度
        similarities_i = sim_matrix[i].clone()

        # 排除自身（相似度为1）
        similarities_i[i] = -1  # 设置为负数，确保不会被选为最近邻

        # 找到top-k相似度的索引
        top_k_values, top_k_indices = torch.topk(similarities_i, k=k)

        # 获取这些近邻的标签
        neighbor_labels = [true_labels[idx] for idx in top_k_indices.tolist()]

        # 投票选择最频繁的标签作为预测
        from collections import Counter
        label_counter = Counter(neighbor_labels)
        predicted_label = label_counter.most_common(1)[0][0]

        # 存储预测结果和平均相似度
        predictions.append(predicted_label)
        similarities.append(top_k_values.mean().item())

    return predictions, similarities


def save_predictions_to_csv(image_paths, true_labels, predictions, similarities,
                            id_to_label, output_dir=".", filename="predictions.csv"):
    """
    保存预测结果到CSV文件
    """
    os.makedirs(output_dir, exist_ok=True)

    # 准备数据
    data = []
    for i in range(len(image_paths)):
        true_label_name = id_to_label[true_labels[i]]
        pred_label_name = id_to_label[predictions[i]]
        is_correct = "Yes" if true_labels[i] == predictions[i] else "No"

        data.append({
            'image_path': image_paths[i],
            'file_name': os.path.basename(image_paths[i]),
            'similarity': f"{similarities[i]:.6f}",
            'true_label': true_label_name,
            'predicted_label': pred_label_name,
            'is_correct': is_correct,
            'true_label_id': true_labels[i],
            'predicted_label_id': predictions[i]
        })

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 保存到CSV
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\nPredictions saved to: {output_path}")
    print(f"Total predictions: {len(df)}")
    print(f"Correct predictions: {df['is_correct'].value_counts().get('Yes', 0)}")
    print(f"Incorrect predictions: {df['is_correct'].value_counts().get('No', 0)}")

    # 计算准确率
    accuracy = (df['is_correct'] == 'Yes').sum() / len(df)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    return df, accuracy


def print_statistics(sim_matrix, labels, predictions=None):
    """打印统计信息"""
    if sim_matrix is None:
        print("No similarity matrix to analyze")
        return

    print("\n" + "=" * 50)
    print("Similarity Matrix Statistics:")
    print("=" * 50)

    print(f"Matrix shape: {sim_matrix.shape}")
    print(f"Min similarity: {sim_matrix.min():.4f}")
    print(f"Max similarity: {sim_matrix.max():.4f}")
    print(f"Mean similarity: {sim_matrix.mean():.4f}")
    print(f"Std similarity: {sim_matrix.std():.4f}")

    # 对角线统计
    diag_values = sim_matrix.diagonal()
    print(f"\nDiagonal statistics:")
    print(f"  Mean: {diag_values.mean():.4f} (should be ~1.0)")
    print(f"  Std: {diag_values.std():.4f}")
    print(f"  Min: {diag_values.min():.4f}")
    print(f"  Max: {diag_values.max():.4f}")

    # 同类样本相似度统计
    if labels and len(labels) > 0:
        labels_tensor = torch.tensor(labels)
        same_class_mask = (labels_tensor.unsqueeze(0) == labels_tensor.unsqueeze(1)).float()
        same_class_mask.fill_diagonal_(0)  # 排除自身

        num_same_pairs = same_class_mask.sum().item()
        if num_same_pairs > 0:
            same_class_similarities = sim_matrix * same_class_mask
            avg_intra_similarity = same_class_similarities.sum().item() / num_same_pairs

            print(f"\nIntra-class similarity:")
            print(f"  Number of same-class pairs: {int(num_same_pairs)}")
            print(f"  Average intra-class similarity: {avg_intra_similarity:.4f}")

    # 如果提供了预测结果，打印混淆矩阵
    if predictions is not None:
        from sklearn.metrics import confusion_matrix, classification_report
        print(f"\n" + "=" * 50)
        print("Prediction Performance:")
        print("=" * 50)

        # 计算混淆矩阵
        cm = confusion_matrix(labels, predictions)
        print(f"Confusion Matrix:\n{cm}")

        # 打印分类报告
        unique_labels = sorted(set(labels))
        print(f"\nClassification Report:")
        print(classification_report(labels, predictions,
                                    target_names=[f"Class_{i}" for i in unique_labels]))


def save_results(sim_matrix, image_paths, labels, embeddings, predictions_df, accuracy,
                 output_dir=".", label_to_id=None, matrix_prefix="similarity"):
    """保存所有结果到文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 创建结果目录
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\nSaving results...")

    # 保存相似度矩阵
    if sim_matrix is not None:
        similarity_np = sim_matrix.numpy()

        # 保存为文本文件
        txt_path = os.path.join(results_dir, f'{matrix_prefix}.txt')
        np.savetxt(txt_path, similarity_np, fmt='%.6f')
        print(f"  Similarity matrix (text): {txt_path}")

        # 保存为numpy文件
        npy_path = os.path.join(results_dir, f'{matrix_prefix}.npy')
        np.save(npy_path, similarity_np)
        print(f"  Similarity matrix (numpy): {npy_path}")

    # 保存其他结果
    results = {
        'image_paths': image_paths,
        'labels': labels,
        'embeddings': embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings,
        'label_mapping': label_to_id,
        'accuracy': accuracy
    }

    torch_path = os.path.join(results_dir, f'{matrix_prefix}_results.pt')
    torch.save(results, torch_path)
    print(f"  Full results: {torch_path}")

    # 保存预测结果总结
    summary_path = os.path.join(results_dir, 'test_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Test Results Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total images: {len(image_paths)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Correct predictions: {(predictions_df['is_correct'] == 'Yes').sum()}\n")
        f.write(f"Incorrect predictions: {(predictions_df['is_correct'] == 'No').sum()}\n")
        f.write(f"\nLabel mapping:\n")
        for label, idx in label_to_id.items():
            f.write(f"  {idx}: {label}\n")

    print("Results saved successfully!")


# 主程序
if __name__ == "__main__":
    # 配置文件路径
    CSV_PATH = "jojo_data1231.csv"
    use_main_category_only = True  # 与训练时保持一致
    K_NEIGHBORS = 5  # K近邻的K值

    print("=" * 60)
    print("DINOv3 Test Script - With Predictions")
    print("=" * 60)

    # 1. 加载测试数据
    print("\n[1/5] Loading test data from CSV...")
    image_paths, labels, label_to_id, id_to_label = load_test_data_from_csv(
        CSV_PATH,
        use_main_category_only=use_main_category_only
    )

    if not image_paths:
        print("Error: No test images loaded. Exiting...")
        exit(1)

    # 2. 提取特征嵌入
    print(f"\n[2/5] Extracting embeddings for {len(image_paths)} images...")

    # 根据图像数量调整batch_size
    if len(image_paths) > 1000:
        batch_size = 8
    elif len(image_paths) > 500:
        batch_size = 16
    else:
        batch_size = 32

    embeddings = extract_embeddings_batch(
        image_paths,
        batch_size=batch_size,
        use_projection=True  # 使用投影特征（训练时使用的）
    )

    if embeddings.numel() == 0:
        print("Error: Failed to extract embeddings. Exiting...")
        exit(1)

    print(f"Embeddings shape: {embeddings.shape}")

    # 3. 计算相似度矩阵
    print(f"\n[3/5] Computing similarity matrix...")
    sim_matrix = compute_similarity_matrix(embeddings)

    if sim_matrix is None:
        print("Error: Failed to compute similarity matrix. Exiting...")
        exit(1)

    # 4. 进行预测
    print(f"\n[4/5] Making predictions using {K_NEIGHBORS}-NN...")
    predictions, similarities = predict_labels(sim_matrix, labels, k=K_NEIGHBORS)

    # 5. 保存预测结果到CSV
    print(f"\n[5/5] Saving predictions to CSV...")
    predictions_df, accuracy = save_predictions_to_csv(
        image_paths=image_paths,
        true_labels=labels,
        predictions=predictions,
        similarities=similarities,
        id_to_label=id_to_label,
        output_dir=".",
        filename=f"predictions_{matrix_name}.csv"
    )

    # 6. 保存所有结果
    save_results(
        sim_matrix=sim_matrix,
        image_paths=image_paths,
        labels=labels,
        embeddings=embeddings,
        predictions_df=predictions_df,
        accuracy=accuracy,
        output_dir=".",
        label_to_id=label_to_id,
        matrix_prefix=matrix_name
    )

    # 7. 打印统计信息
    print_statistics(sim_matrix, labels, predictions)

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print(f"Predictions saved to: predictions_{matrix_name}.csv")
    print("=" * 60)