import os
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from transformers.image_utils import load_image
from safetensors import safe_open
import numpy as np
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型路径 - 这里应该是包含多个safetensors文件的目录
model_dir = "/data1/vincent/sku/dinov3/ds_accelerator_checkpoints-1223/latest/output_dir"


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
            torch.nn.Linear(hidden_size, feature_dim),
            torch.nn.LayerNorm(feature_dim)
        )

        # 相似度计算的温度参数（可学习）
        self.temperature = torch.nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, pixel_values=None, **kwargs):
        # 获取图像特征
        if pixel_values is not None:
            outputs = self.backbone(pixel_values=pixel_values)
        else:
            # 处理其他输入格式
            outputs = self.backbone(**kwargs)

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


# 2. 首先检查模型目录中的文件
print(f"Checking model directory: {model_dir}")
files = os.listdir(model_dir)
print("Files in directory:")
for f in sorted(files):
    print(f"  {f}")

# 3. 加载处理器
print("\nLoading image processor...")
processor = AutoImageProcessor.from_pretrained("/data1/vincent/models/facebook-dinov3-vit7b16-pretrain-sat493m")

# 4. 创建模型实例
print("Creating model instance...")
with init_empty_weights():
    model = DINOv3ForContrastive(
        model_name="/data1/vincent/models/facebook-dinov3-vit7b16-pretrain-sat493m",  # 基础模型
        num_classes=None,  # 根据你的训练设置调整
        feature_dim=2048  # 根据你的训练设置调整
    )

# 5. 加载分片的safetensors权重
print("Loading safetensors weights...")
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=model_dir,  # safetensors 目录
    device_map="auto",
    dtype=torch.float16
)


# 10. 定义辅助函数
def extract_embeddings_batch(image_paths, batch_size=4, use_projection=True):
    """
    批量提取图像嵌入
    """
    all_embeddings = []

    # 获取模型所在的设备
    model_device = next(model.parameters()).device
    print(f"Model is on device: {model_device}")

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(image_paths) - 1) // batch_size + 1}")

        # 加载图像
        images = [load_image(p) for p in batch_paths]

        # 预处理
        inputs = processor(images=images, return_tensors="pt")

        # 将输入数据移动到模型所在的设备
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            # 前向传播
            outputs = model(pixel_values=inputs["pixel_values"])

            # 选择使用原始特征还是投影特征
            if use_projection:
                emb = outputs["projected_features"]
            else:
                emb = outputs["features"]

            # L2归一化
            emb = F.normalize(emb, dim=-1)

        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)


def collect_images(root_dir, exts=(".jpg", ".png", ".jpeg", ".JPG", ".JPEG")):
    """收集图像路径和标签"""
    image_paths = []
    labels = []

    for root, dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(exts):
                full_path = os.path.join(root, fname)

                # 用 test 下的一级目录作为 label
                rel = os.path.relpath(full_path, root_dir)
                label = rel.split(os.sep)[0]

                image_paths.append(full_path)
                labels.append(label)

    return image_paths, labels


def compute_similarity_matrix(embeddings, block_size=256):
    """
    计算相似度矩阵（余弦相似度）

    Args:
        embeddings: [N, D] 嵌入矩阵
        block_size: 分块计算大小（避免内存不足）
    """
    N = embeddings.size(0)
    sim_matrix = torch.zeros((N, N), dtype=torch.float32)

    # 确保embeddings是归一化的
    embeddings = F.normalize(embeddings, dim=-1)

    print(f"Computing similarity matrix for {N} images...")

    # 将embeddings移动到GPU（如果可用）
    if torch.cuda.is_available():
        embeddings_gpu = embeddings.to(device)
    else:
        embeddings_gpu = embeddings

    # 分块计算相似度
    for i in range(0, N, block_size):
        end_i = min(i + block_size, N)
        emb_i = embeddings_gpu[i:end_i]

        # 计算与其他所有图像的相似度
        with torch.no_grad():
            sim = emb_i @ embeddings_gpu.T  # [block_size, N]

        sim_matrix[i:end_i] = sim.cpu()

        print(f"  Block {i}-{end_i} / {N} completed")

    return sim_matrix


def save_results(sim_matrix, image_paths, labels, embeddings, output_dir="."):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 转换为numpy
    similarity_np = sim_matrix.numpy()

    # 保存相似度矩阵
    np.savetxt(os.path.join(output_dir, 'dinov3_similarity.txt'), similarity_np, fmt='%.6f')
    np.save(os.path.join(output_dir, 'dinov3_similarity.npy'), similarity_np)

    # 保存完整结果
    torch.save(
        {
            "similarity_matrix": sim_matrix,
            "image_paths": image_paths,
            "labels": labels,
            "embeddings": embeddings,
        },
        os.path.join(output_dir, "dinov3_results.pt")
    )

    print(f"Results saved to {output_dir}")


# 11. 主程序
if __name__ == "__main__":
    root_dir = "/data1/vincent/datasets/data-1210/test"

    # 收集图像
    print("\n" + "=" * 50)
    print("Collecting images...")
    print("=" * 50)
    image_paths, labels = collect_images(root_dir)
    print(f"Found {len(image_paths)} images")
    print(f"Unique labels: {sorted(set(labels))}")

    # 提取嵌入（使用投影特征，这是对比学习训练时使用的）
    print("\n" + "=" * 50)
    print("Extracting embeddings...")
    print("=" * 50)

    # 根据图像数量调整batch_size
    if len(image_paths) > 1000:
        batch_size = 8
    else:
        batch_size = 16

    try:
        embeddings = extract_embeddings_batch(
            image_paths,
            batch_size=batch_size,
            use_projection=True  # 使用投影特征
        )
        print(f"Embeddings shape: {embeddings.shape}")

        # 计算相似度矩阵
        print("\n" + "=" * 50)
        print("Computing similarity matrix...")
        print("=" * 50)
        sim_matrix = compute_similarity_matrix(embeddings)

        # 保存结果
        print("\n" + "=" * 50)
        print("Saving results...")
        print("=" * 50)
        save_results(sim_matrix, image_paths, labels, embeddings, output_dir=".")

        # 可选：打印一些统计信息
        print("\n" + "=" * 50)
        print("Statistics:")
        print("=" * 50)
        print(f"Similarity matrix shape: {sim_matrix.shape}")
        print(f"Min similarity: {sim_matrix.min():.4f}")
        print(f"Max similarity: {sim_matrix.max():.4f}")
        print(f"Mean similarity: {sim_matrix.mean():.4f}")

        # 对角线应该是1（自相似度）
        diag_values = sim_matrix.diagonal()
        print(f"Diagonal mean (should be ~1.0): {diag_values.mean():.4f}")
        print(f"Diagonal std: {diag_values.std():.4f}")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)