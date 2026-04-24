import os
import pandas as pd
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_scheduler, AutoImageProcessor
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_recall_curve, auc
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import deepspeed
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'


# 生成数据集信息的函数（返回标签ID）
def generate_dataset_info(data_dir, use_main_category_only=False):
    """从文件夹结构生成数据集信息"""
    image_paths = []
    labels = []
    label_to_id = {}
    current_id = 0

    for main_category in os.listdir(data_dir):
        main_category_path = os.path.join(data_dir, main_category)
        if not os.path.isdir(main_category_path):
            continue

        for sub_category in os.listdir(main_category_path):
            sub_category_path = os.path.join(main_category_path, sub_category)
            if not os.path.isdir(sub_category_path):
                continue

            if use_main_category_only:
                label_name = main_category
            else:
                label_name = f"{main_category}_{sub_category}"

            if label_name not in label_to_id:
                label_to_id[label_name] = current_id
                current_id += 1

            label_id = label_to_id[label_name]

            for image_file in os.listdir(sub_category_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    rel_path = os.path.join(main_category, sub_category, image_file)
                    image_paths.append(rel_path)
                    labels.append(label_id)

    return pd.DataFrame({'image': image_paths, 'label': labels}), label_to_id


# 自定义数据集类（只返回图像和标签）
class CustomImageDataset(Dataset):
    def __init__(self, df, image_dirs, processor, mode='train'):
        self.df = df
        self.image_dirs = image_dirs if isinstance(image_dirs, list) else [image_dirs]
        self.processor = processor
        self.image_paths = df['image'].tolist()
        self.labels = df['label'].tolist()
        self.mode = mode
        # 训练集使用数据增强，验证/测试集不使用
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=60, expand=True),
                transforms.RandomHorizontalFlip(p=0.5),
                # 确保输出大小一致，如果需要的话
                # transforms.Resize((224, 224)),  # 根据你的模型调整尺寸
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_rel_path = self.image_paths[idx]
        label = self.labels[idx]

        image = None
        for image_dir in self.image_dirs:
            image_path = os.path.join(image_dir, image_rel_path)
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                    break
                except Exception as e:
                    print(f"警告：无法打开图片文件 {image_path}: {e}")
                    continue

        if image is None:
            print(f"错误：在所有目录中都找不到图片 {image_rel_path}，将跳过此样本。")
            return None

        if self.transform is not None:
            image = self.transform(image)
        # 只处理图像（DINOv3是纯视觉模型）
        inputs = self.processor(images=image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    return torch.utils.data.dataloader.default_collate(batch)


# 自定义DINOv3模型，输出图像特征和分类
class DINOv3ForContrastive(torch.nn.Module):
    def __init__(self, model_name, num_classes=None, feature_dim=2048):
        super().__init__()
        from transformers import AutoModel

        # 加载预训练的DINOv3模型
        self.backbone = AutoModel.from_pretrained(model_name)

        # 获取特征维度
        hidden_size = self.backbone.config.hidden_size

        # 分类头
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
        # print(outputs)

        # 使用[CLS] token作为图像表示
        features = outputs.last_hidden_state[:, 0, :]
        # print(features)

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


# 使用cross_entropy的对比损失函数
def image_image_contrastive_loss_with_ce(projected_features, labels, temperature=0.07):
    """
    使用cross_entropy计算图像之间的对比损失
    Args:
        projected_features: 归一化的特征向量 [batch_size, feature_dim]
        labels: 图像标签 [batch_size]
        temperature: 温度参数
    """
    batch_size = projected_features.size(0)
    device = projected_features.device

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(projected_features, projected_features.T)  # [batch_size, batch_size]

    # 应用温度缩放
    similarity_matrix = similarity_matrix / temperature

    # 创建标签：每个样本的正样本是它自己（对角线）
    # 这在对比学习中很常见，将每个样本视为自己的正样本
    target_labels = torch.arange(batch_size, device=device)

    # 使用交叉熵损失
    loss = torch.nn.functional.cross_entropy(similarity_matrix, target_labels)

    return loss


def margin_contrastive_loss(projected_features, labels, temperature=0.07, margin=0.2):
    """
    带间隔的对比损失：相同类别的样本作为正样本，不同类别的样本添加间隔
    Args:
        projected_features: 归一化的特征向量 [batch_size, feature_dim]
        labels: 图像标签 [batch_size]
        temperature: 温度参数
        margin: 间隔大小（用于增大正负样本之间的差异）
    """
    batch_size = projected_features.size(0)
    device = projected_features.device

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(projected_features, projected_features.T)  # [batch_size, batch_size]
    # print(projected_features)

    # 应用温度缩放
    similarity_matrix = similarity_matrix / temperature

    # 创建正样本掩码（相同标签的样本是正样本）
    labels_expanded = labels.unsqueeze(0)
    positive_mask = (labels_expanded == labels_expanded.T).float()  # [batch_size, batch_size]

    # 创建负样本掩码
    negative_mask = 1 - positive_mask

    # 方法1：直接在logits上应用间隔
    # 对正样本的logits增加间隔（使其更容易被识别为正样本）
    # 对负样本的logits减少间隔（使其更容易被识别为负样本）
    margin_matrix = torch.zeros_like(similarity_matrix)
    margin_matrix[positive_mask.bool()] = -margin  # 正样本增加间隔
    margin_matrix[negative_mask.bool()] = margin  # 负样本减少间隔
    margin_matrix.fill_diagonal_(margin * 2)  # 自身作为最强正样本

    # 应用间隔
    margin_logits = similarity_matrix + margin_matrix

    # 使用交叉熵损失
    # 对于每个样本i，正样本集作为目标
    # 我们需要将多标签分类问题转换为多分类问题
    # 简单做法：每个样本最相似的正样本作为目标
    max_similarity, target_indices = torch.max(margin_logits * positive_mask, dim=1)

    # 使用交叉熵损失
    loss = torch.nn.functional.cross_entropy(margin_logits, target_indices)

    return loss


def margin_contrastive_loss_multi_positive(
        projected_features,
        labels,
        temperature=0.07,
        margin=0.2
):
    device = projected_features.device
    batch_size = projected_features.size(0)

    # 归一化（非常重要）
    # projected_features = F.normalize(projected_features, dim=1)

    # 相似度矩阵
    logits = torch.matmul(projected_features, projected_features.T) / temperature

    # 标签 mask
    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.T).float().to(device)

    # 去掉自身
    self_mask = torch.eye(batch_size, device=device)
    positive_mask = positive_mask - self_mask
    negative_mask = 1 - positive_mask - self_mask

    # 加 margin
    logits = logits - margin * positive_mask
    logits = logits + margin * negative_mask

    # 数值稳定
    logits = logits - logits.max(dim=1, keepdim=True)[0]

    exp_logits = torch.exp(logits)

    # 分母：所有非自身
    denom = exp_logits * (1 - self_mask)
    denom = denom.sum(dim=1, keepdim=True)

    # 分子：所有正样本
    pos_exp = exp_logits * positive_mask
    pos_sum = pos_exp.sum(dim=1)  # TODO: pos_sum = pos_exp.mean(dim=1)

    if False:
        pos_sum = pos_sum / (torch.sum(positive_mask, dim=1) + 0.0001)

    # 避免没有正样本的情况
    valid = pos_sum > 0

    loss = -torch.log(pos_sum[valid] / denom[valid].squeeze(1))
    loss = torch.sqrt(torch.mean(loss ** 2))
    return loss


# 更高级的对比损失：考虑相同类别的样本作为正样本
def supervised_contrastive_loss(projected_features, labels, temperature=0.07):
    """
    监督对比损失：相同类别的样本作为正样本
    Args:
        projected_features: 归一化的特征向量 [batch_size, feature_dim]
        labels: 图像标签 [batch_size]
        temperature: 温度参数
    """
    batch_size = projected_features.size(0)
    device = projected_features.device

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(projected_features, projected_features.T)  # [batch_size, batch_size]

    # 应用温度缩放
    similarity_matrix = similarity_matrix / temperature

    # 创建正样本掩码（相同标签的样本是正样本，包括自己）
    labels = labels.unsqueeze(0)
    positive_mask = (labels == labels.T).float()  # [batch_size, batch_size]

    # 将对角线权重设为1，其他相同类别样本权重设为0.3（可调节）
    positive_mask.fill_diagonal_(1.0)

    # 创建目标概率分布
    # 对于每个样本i，正样本的logits应该大，负样本的logits应该小
    # 我们可以使用softmax来计算概率分布
    exp_similarity = torch.exp(similarity_matrix)

    # 计算分母：所有样本的exp相似度之和
    sum_exp = torch.sum(exp_similarity, dim=1, keepdim=True)  # [batch_size, 1]

    # 计算概率
    probabilities = exp_similarity / sum_exp  # [batch_size, batch_size]

    # 计算损失：-log(正样本的概率加权和)
    positive_probs = probabilities * positive_mask
    sum_positive_probs = torch.sum(positive_probs, dim=1)  # [batch_size]

    # 避免log(0)
    sum_positive_probs = torch.clamp(sum_positive_probs, min=1e-8)

    loss = -torch.log(sum_positive_probs).mean()

    return loss


# 评估函数
def evaluate(model, dataloader, device, world_size, local_rank):
    """评估模型性能"""
    model.eval()
    all_true_labels = []
    all_predicted_labels = []
    all_features = []

    # 添加进度条
    if local_rank == 0:
        eval_progress_bar = tqdm(dataloader, desc="评估中", leave=False)
    else:
        eval_progress_bar = dataloader

    with torch.no_grad():
        for batch in eval_progress_bar:
            if batch is None:
                continue

            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            outputs = model(pixel_values)

            # 如果是分类任务
            if outputs["logits"] is not None:
                logits = outputs["logits"]
                predicted = torch.argmax(logits, dim=1)
                all_predicted_labels.extend(predicted.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())

            # 保存特征用于相似度计算
            features = outputs["projected_features"]
            all_features.append(features.cpu())

    # 收集所有进程的结果
    if world_size > 1:
        gathered_features = [None] * world_size
        dist.all_gather_object(gathered_features, all_features)

        gathered_true_labels = [None] * world_size
        dist.all_gather_object(gathered_true_labels, all_true_labels)

        gathered_pred_labels = [None] * world_size
        dist.all_gather_object(gathered_pred_labels, all_predicted_labels)

        if dist.get_rank() != 0:
            return 0.0, "", 0.0, 0.0

        # 合并所有进程的结果
        all_features = [item for sublist in gathered_features for batch in sublist for item in batch]
        all_true_labels = [item for sublist in gathered_true_labels for item in sublist]
        all_predicted_labels = [item for sublist in gathered_pred_labels for item in sublist]

    if not all_true_labels:
        return 0.0, "No data", 0.0, 0.0

    # 计算分类F1分数
    if all_predicted_labels:
        f1 = f1_score(all_true_labels, all_predicted_labels, average='macro', zero_division=0)
        cls_rep = classification_report(all_true_labels, all_predicted_labels, zero_division=0)
    else:
        f1 = 0.0
        cls_rep = "No predictions"

    # 计算特征相似度
    avg_similarity = 0.0
    pr_auc = 0.0

    if all_features and len(all_features) > 1:
        features_tensor = torch.stack(all_features)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features_tensor, features_tensor.T)

        # 计算同类样本的平均相似度
        labels_tensor = torch.tensor(all_true_labels)
        same_class_mask = (labels_tensor.unsqueeze(0) == labels_tensor.unsqueeze(1)).float()
        same_class_mask.fill_diagonal_(0)  # 排除自身

        num_same_class = same_class_mask.sum().item()
        if num_same_class > 0:
            same_class_similarities = similarity_matrix * same_class_mask
            avg_similarity = same_class_similarities.sum().item() / num_same_class

        # 计算PR-AUC
        try:
            # 将相似度矩阵展平
            y_scores = similarity_matrix.flatten().numpy()

            # 创建二元标签（1表示同类，0表示不同类）
            y_true = same_class_mask.flatten().numpy()

            # 计算PR-AUC
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)
        except Exception as e:
            print(f"PR-AUC计算失败: {e}")
            pr_auc = 0.0

    return f1, cls_rep, avg_similarity, pr_auc


# 检查点函数（保持不变）
def save_model_checkpoint(model, checkpoint_dir, tag, epoch, best_metrics, save_torch_model=False):
    """保存检查点"""
    if dist.is_initialized():
        save_flag = torch.tensor([1], device=torch.cuda.current_device())
        dist.broadcast(save_flag, src=0)

    client_state = {
        'epoch': epoch,
        'best_f1': best_metrics['best_f1'],
        'best_loss': best_metrics['best_loss'],
        'best_pr_auc': best_metrics['best_pr_auc'],
        'best_similarity': best_metrics['best_similarity']
    }

    model.save_checkpoint(checkpoint_dir, tag=tag, client_state=client_state)

    if dist.get_rank() == 0:
        latest_file = os.path.join(checkpoint_dir, tag, "latest")
        try:
            with open(latest_file, 'w') as f:
                f.write("")
        except Exception as e:
            print(f"创建latest文件失败: {e}")

    if dist.is_initialized():
        dist.barrier()


def load_model_checkpoint(checkpoint_dir, model, tag=None):
    """加载检查点"""
    if tag is None:
        tag = "epoch_latest"

    if tag:
        checkpoint_path = os.path.join(checkpoint_dir, tag)
        if dist.get_rank() == 0:
            print(f"尝试加载检查点: {checkpoint_path}")
            print(f"目录存在: {os.path.exists(checkpoint_path)}")
            if os.path.exists(checkpoint_path):
                print(f"目录内容: {os.listdir(checkpoint_path)}")

        try:
            load_path, client_state = model.load_checkpoint(checkpoint_path)
            if load_path is not None:
                if dist.get_rank() == 0:
                    print(f"成功加载检查点: {load_path}")
                start_epoch = client_state.get('epoch', 0) + 1
                best_metrics = {
                    'best_f1': client_state.get('best_f1', 0.0),
                    'best_loss': client_state.get('best_loss', float('inf')),
                    'best_pr_auc': client_state.get('best_pr_auc', 0.0),
                    'best_similarity': client_state.get('best_similarity', 0.0)
                }
                return start_epoch, best_metrics
            else:
                if dist.get_rank() == 0:
                    print("DeepSpeed加载返回的load_path为None")
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"加载DeepSpeed检查点失败: {e}")
                import traceback
                traceback.print_exc()

    if dist.get_rank() == 0:
        print("将使用初始epoch和指标开始训练")
    return 0, {
        'best_f1': 0.0,
        'best_loss': float('inf'),
        'best_pr_auc': 0.0,
        'best_similarity': 0.0
    }


# 主执行逻辑
if __name__ == "__main__":
    # --- 初始化分布式训练 ---
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # --- 配置参数 ---
    character_path = "/Users/vincent/workspace/sku/data/jojo_test"#"/data1/vincent/datasets/data1210/"
    TRAIN_DATA_DIR = character_path
    TEST_DATA_DIR = character_path

    image_train = [TRAIN_DATA_DIR]
    image_test = [TEST_DATA_DIR]

    MODEL_NAME = "/Users/vincent/workspace/sku/facebook-dinov3-vith16plus-pretrain-lvd1689m"#"/data1/vincent/models/facebook-dinov3-vith16plus-pretrain-lvd1689m"
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 5e-5
    SCHEDULER_TYPE = "cosine_with_restarts"
    CHECKPOINT_DIR = "ds_dinov3_checkpoints-1226-1"
    RESUME = True
    use_main_category_only = True

    # 创建检查点目录
    if local_rank == 0 and not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # --- 数据加载和准备 ---
    if local_rank == 0:
        print("正在生成数据集信息...")
        print(f"正在使用 {world_size} 个GPU进行训练")

    train_df, label_mapping = generate_dataset_info(TRAIN_DATA_DIR, use_main_category_only=use_main_category_only)
    val_df, _ = generate_dataset_info(TEST_DATA_DIR, use_main_category_only=use_main_category_only)

    if local_rank == 0:
        print(f"训练集大小: {len(train_df)}")
        print(f"验证集大小: {len(val_df)}")
        print(f"类别数量: {len(label_mapping)}")
        print(f"标签映射: {label_mapping}")

    # 加载DINOv3的图像处理器
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME,size={"height": 512, "width": 512},)

    # 创建数据集和数据加载器
    train_dataset = CustomImageDataset(train_df, image_train, processor)
    val_dataset = CustomImageDataset(val_df, image_test, processor)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # --- 加载模型 ---
    num_classes = len(label_mapping)
    model = DINOv3ForContrastive(MODEL_NAME, num_classes=num_classes, feature_dim=1024)
    model = model.to(device)

    # 准备优化器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)

    num_warmup_steps = int(0.02 * num_training_steps)
    scheduler = get_scheduler(
        SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # DeepSpeed 配置
    ds_config = {
        "train_batch_size": BATCH_SIZE * world_size,  # 总批处理大小
        "train_micro_batch_size_per_gpu": BATCH_SIZE,  # 每个GPU的批处理大小
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 3,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
        "bf16": {
            "enabled": True,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": False
    }

    # 初始化 DeepSpeed 引擎
    model, optimizer, _, scheduler = deepspeed.initialize(
        args=None,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config
    )

    # 初始化最佳指标
    best_metrics = {
        'best_f1': 0.0,
        'best_loss': float('inf'),
        'best_pr_auc': 0.0,
        'best_similarity': 0.0
    }
    start_epoch = 0

    # 恢复训练
    if RESUME:
        start_epoch, best_metrics = load_model_checkpoint(CHECKPOINT_DIR, model, tag="epoch_latest")
        if local_rank == 0:
            print(f"从epoch {start_epoch + 1}恢复训练")
            print(f"恢复的最佳指标: F1={best_metrics['best_f1']:.4f}, "
                  f"Loss={best_metrics['best_loss']:.8f}, "
                  f"PR-AUC={best_metrics['best_pr_auc']:.4f}, "
                  f"Similarity={best_metrics['best_similarity']:.4f}")

    # 广播起始epoch和最佳指标到所有进程
    if world_size > 1:
        start_epoch_tensor = torch.tensor([start_epoch], device=device)
        dist.broadcast(start_epoch_tensor, src=0)
        start_epoch = start_epoch_tensor.item()

        best_metrics_tensor = torch.tensor([
            best_metrics['best_f1'],
            best_metrics['best_loss'],
            best_metrics['best_pr_auc'],
            best_metrics['best_similarity']
        ], device=device)
        dist.broadcast(best_metrics_tensor, src=0)
        best_metrics = {
            'best_f1': best_metrics_tensor[0].item(),
            'best_loss': best_metrics_tensor[1].item(),
            'best_pr_auc': best_metrics_tensor[2].item(),
            'best_similarity': best_metrics_tensor[3].item()
        }

    # --- 训练循环 ---
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_classification_loss = 0.0
    num_batches = 0
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)

        if local_rank == 0:
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [训练]")
        else:
            progress_bar = train_dataloader

        for batch in progress_bar:
            if batch is None:
                continue

            pixel_values = batch["pixel_values"].to(device)
            # print(pixel_values.size())
            labels = batch["label"].to(device)

            outputs = model(pixel_values)
            projected_features = outputs["projected_features"]
            temperature = outputs["temperature"].clamp(min=0.01, max=1.0)  # 限制温度范围

            # 选择对比损失函数（两种方式可选）
            # 方式1：使用cross_entropy的简单对比损失（每个样本的正样本是自己）
            # contrastive_loss = image_image_contrastive_loss_with_ce(projected_features, labels, temperature)

            # 方式2：监督对比损失（相同类别的样本都是正样本）
            contrastive_loss = margin_contrastive_loss_multi_positive(projected_features, labels, temperature,
                                                                      margin=0.001)

            # 如果有分类头，可以加上分类损失
            classification_loss = 0.0
            if outputs["logits"] is not None:
                logits = outputs["logits"]
                classification_loss = torch.nn.functional.cross_entropy(logits, labels)
                # print(f"分类损失: {classification_loss.item():.4f}")
                # 组合损失
                loss = contrastive_loss + 0.1 * classification_loss  # 可以调整权重
            else:
                loss = contrastive_loss

            model.backward(loss)
            model.step()

            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss.item()
            if classification_loss > 0:
                total_classification_loss += classification_loss.item()
            num_batches += 1

            current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
            if local_rank == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.8f}",
                    'contrastive': f"{contrastive_loss.item():.8f}",
                    'cls_loss': f"{classification_loss.item():.8f}" if classification_loss > 0 else "0.0000",
                    'temp': f"{temperature.item():.4f}",
                    'lr': f"{current_lr:.2e}"
                })

        avg_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_contrastive_loss = total_contrastive_loss / num_batches if num_batches > 0 else float('inf')
        avg_classification_loss = total_classification_loss / num_batches if num_batches > 0 and total_classification_loss > 0 else 0.0

        if local_rank == 0:
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            print(f"平均训练损失: {avg_train_loss:.8f}")
            print(f"平均对比损失: {avg_contrastive_loss:.8f}")
            if avg_classification_loss > 0:
                print(f"平均分类损失: {avg_classification_loss:.8f}")
            print("--- 正在评估模型 ---")

        f1, cls_rep, avg_cosine_similarity, pr_auc = evaluate(model, val_dataloader, device, world_size, local_rank)

        if local_rank == 0:
            print(f"F1分数 (Macro): {f1:.4f}")
            print(f"PR-AUC: {pr_auc:.4f}")
            print(f"平均同类相似度: {avg_cosine_similarity:.4f}")

            is_best_f1 = f1 > best_metrics['best_f1']
            is_best_loss = avg_train_loss < best_metrics['best_loss']
            is_best_pr_auc = pr_auc > best_metrics['best_pr_auc']
            is_best_similarity = avg_cosine_similarity > best_metrics['best_similarity']

            if is_best_f1:
                best_metrics['best_f1'] = f1
            if is_best_loss:
                best_metrics['best_loss'] = avg_train_loss
            if is_best_pr_auc:
                best_metrics['best_pr_auc'] = pr_auc
            if is_best_similarity:
                best_metrics['best_similarity'] = avg_cosine_similarity

        # 保存检查点
        if world_size > 1:
            save_decisions = torch.zeros(5, dtype=torch.bool, device=device)
            if local_rank == 0:
                save_decisions[0] = True
                save_decisions[1] = bool(is_best_f1)
                save_decisions[2] = bool(is_best_loss)
                save_decisions[3] = bool(is_best_pr_auc)
                save_decisions[4] = bool(is_best_similarity)

            dist.broadcast(save_decisions, src=0)
            save_regular, save_best_f1, save_best_loss, save_best_pr_auc, save_best_similarity = save_decisions.tolist()
        else:
            save_regular = True
            save_best_f1 = is_best_f1
            save_best_loss = is_best_loss
            save_best_pr_auc = is_best_pr_auc
            save_best_similarity = is_best_similarity

        latest_tag = "epoch_latest"

        if save_best_f1:
            save_model_checkpoint(model, CHECKPOINT_DIR, "best_f1", epoch, best_metrics, save_torch_model=False)
            if local_rank == 0:
                print(f"最佳F1检查点已保存")

        if save_best_loss:
            save_model_checkpoint(model, CHECKPOINT_DIR, "best_loss", epoch, best_metrics, save_torch_model=False)
            if local_rank == 0:
                print(f"最佳损失检查点已保存")

        if save_best_pr_auc:
            save_model_checkpoint(model, CHECKPOINT_DIR, "best_pr_auc", epoch, best_metrics, save_torch_model=False)
            if local_rank == 0:
                print(f"最佳PR-AUC检查点已保存")

        if save_best_similarity:
            save_model_checkpoint(model, CHECKPOINT_DIR, "best_similarity", epoch, best_metrics, save_torch_model=False)
            if local_rank == 0:
                print(f"最佳相似度检查点已保存")

        if save_regular:
            save_model_checkpoint(model, CHECKPOINT_DIR, latest_tag, epoch, best_metrics, save_torch_model=False)
            if local_rank == 0:
                print(f"最新检查点已保存: {latest_tag}")

        if local_rank == 0:
            print(f"当前最佳 F1: {best_metrics['best_f1']:.4f}, "
                  f"最佳损失: {best_metrics['best_loss']:.8f}, "
                  f"最佳PR-AUC: {best_metrics['best_pr_auc']:.4f}, "
                  f"最佳相似度: {best_metrics['best_similarity']:.4f}")
            print("=" * 50)

    dist.destroy_process_group()