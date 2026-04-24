import os
import pandas as pd
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import CLIPProcessor, CLIPModel, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_recall_curve, auc
from tqdm.auto import tqdm
from sklearn.metrics import classification_report

import deepspeed

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5,6,7'


# 从CSV文件生成数据集信息的函数（已修改）
def generate_dataset_info_from_csv(csv_path, use_main_category_only=True):
    """从CSV文件生成数据集信息"""
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 初始化列表
    train_image_paths = []
    train_captions = []
    val_image_paths = []
    val_captions = []
    test_image_paths = []
    test_captions = []

    # 遍历每一行
    for idx, row in df.iterrows():
        image_path = row['images']

        # 根据参数选择标签
        if use_main_category_only:
            label = row['main_captions']
        else:
            label = row['sub_category_captions']

        split = row['train/val/test'].strip().lower()

        if split == 'train':
            train_image_paths.append(image_path)
            train_captions.append(label)
        elif split == 'val':
            val_image_paths.append(image_path)
            val_captions.append(label)
        elif split == 'test':
            test_image_paths.append(image_path)
            test_captions.append(label)
        else:
            print(f"警告：第{idx + 1}行的split值为'{split}'，不是'train', 'val'或'test'，已跳过")

    # 创建DataFrame
    train_df = pd.DataFrame({'image': train_image_paths, 'caption': train_captions})
    val_df = pd.DataFrame({'image': val_image_paths, 'caption': val_captions})
    test_df = pd.DataFrame({'image': test_image_paths, 'caption': test_captions})

    return train_df, val_df, test_df,train_captions


# 自定义数据集类
class CustomImageCaptionDataset_multi(Dataset):
    def __init__(self, df, processor):
        """
        初始化数据集
        Args:
            df: 包含'image'和'caption'列的DataFrame，'image'列是完整的图片路径
            processor: CLIP处理器
        """
        self.df = df
        self.processor = processor
        self.image_paths = df['image'].tolist()
        self.captions = df['caption'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"警告：无法打开图片文件 {image_path}: {e}")
            return None

        inputs = self.processor(text=caption, images=image, return_tensors="pt", padding="max_length", truncation=True)

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "pixel_values": inputs["pixel_values"].squeeze(),
            "caption": caption
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    captions = [item.pop('caption') for item in batch]
    collated_batch = torch.utils.data.dataloader.default_collate(batch)
    collated_batch['captions'] = captions
    return collated_batch


# 评估函数
def evaluate(model, dataloader, device, world_size, local_rank):
    """评估模型性能"""
    model.eval()
    local_similarity_matrices = []
    local_captions = []

    # 添加进度条
    if local_rank == 0:
        eval_progress_bar = tqdm(dataloader, desc="评估中", leave=False)
    else:
        eval_progress_bar = dataloader

    with torch.no_grad():
        for batch in eval_progress_bar:
            if batch is None:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            captions = batch["captions"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            similarity_matrix = outputs.logits_per_image
            local_similarity_matrices.append(similarity_matrix.cpu())
            local_captions.extend(captions)

            # 更新进度条描述（仅在rank 0上）
            if local_rank == 0:
                eval_progress_bar.set_postfix(processed_batches=len(local_similarity_matrices))

    if world_size > 1:
        gathered_similarity_matrices = [None] * world_size
        dist.all_gather_object(gathered_similarity_matrices, local_similarity_matrices)

        gathered_captions = [None] * world_size
        dist.all_gather_object(gathered_captions, local_captions)

        if dist.get_rank() != 0:
            return 0.0, "", 0.0, 0.0

        all_similarity_matrices = [item for sublist in gathered_similarity_matrices for item in sublist]
        all_captions = [item for sublist in gathered_captions for item in sublist]
    else:
        all_similarity_matrices = local_similarity_matrices
        all_captions = local_captions

    if not all_similarity_matrices:
        return 0.0, "No data", 0.0, 0.0

    all_true_labels = []
    all_predicted_labels = []
    all_diagonals = []
    y_true_flat = []
    y_scores_flat = []

    caption_offset = 0
    for matrix in all_similarity_matrices:
        batch_size = matrix.shape[0]
        if batch_size == 0:
            continue

        true_labels_batch = all_captions[caption_offset: caption_offset + batch_size]
        all_true_labels.extend(true_labels_batch)

        predicted_indices_batch = torch.argmax(matrix, dim=1)
        predicted_labels_batch = [true_labels_batch[i] for i in predicted_indices_batch]
        all_predicted_labels.extend(predicted_labels_batch)

        all_diagonals.append(torch.diag(matrix))

        probs = torch.softmax(matrix, dim=1)
        true_binary = torch.eye(batch_size).bool()
        y_scores_flat.append(probs.flatten())
        y_true_flat.append(true_binary.flatten())

        caption_offset += batch_size

    if not all_true_labels:
        return 0.0, "No processed labels", 0.0, 0.0

    f1 = f1_score(all_true_labels, all_predicted_labels, average='macro', zero_division=0)
    cls_rep = classification_report(all_true_labels, all_predicted_labels, zero_division=0)
    avg_cosine_similarity = torch.cat(all_diagonals).mean().item()

    try:
        y_true_np = torch.cat(y_true_flat).numpy()
        y_scores_np = torch.cat(y_scores_flat).numpy()
        precision, recall, _ = precision_recall_curve(y_true_np, y_scores_np)
        pr_auc = auc(recall, precision)
    except Exception:
        pr_auc = 0.0

    return f1, cls_rep, avg_cosine_similarity, pr_auc


# 检查点函数
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
    CSV_PATH = "/Users/vincent/workspace/trademark_similar/character_data/data_info.csv"  # 修改为你的CSV文件路径
    MODEL_NAME = "/data1/vincent/models/apple-DFN5B-CLIP-ViT-H-14-378"
    BATCH_SIZE = 8
    NUM_EPOCHS = 1000
    LEARNING_RATE = 5e-5
    SCHEDULER_TYPE = "cosine_with_restarts"
    CHECKPOINT_DIR = "ds_dfn5b_checkpoints_sku"
    RESUME = True
    USE_MAIN_CATEGORY_ONLY = True  # True: 使用main_captions, False: 使用sub_category_captions

    '''
        Scheduler types:
           - "linear" = get_linear_schedule_with_warmup
           - "cosine" = get_cosine_schedule_with_warmup
           - "cosine_with_restarts" = get_cosine_with_hard_restarts_schedule_with_warmup
           - "polynomial" = get_polynomial_decay_schedule_with_warmup
           - "constant" =  get_constant_schedule
           - "constant_with_warmup" = get_constant_schedule_with_warmup
           - "inverse_sqrt" = get_inverse_sqrt_schedule
           - "reduce_lr_on_plateau" = get_reduce_on_plateau_schedule
           - "cosine_with_min_lr" = get_cosine_with_min_lr_schedule_with_warmup
           - "warmup_stable_decay" = get_wsd_schedule
    '''

    # 创建检查点目录
    if local_rank == 0 and not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # --- 数据加载和准备 ---
    if local_rank == 0:
        print("正在从CSV文件加载数据集信息...")
        print(f"正在使用 {world_size} 个GPU进行训练")
        print(f"使用{'主类别' if USE_MAIN_CATEGORY_ONLY else '子类别'}作为标签")

    # 从CSV文件加载数据集
    train_df, val_df, test_df = generate_dataset_info_from_csv(CSV_PATH, use_main_category_only=USE_MAIN_CATEGORY_ONLY)

    if local_rank == 0:
        print(f"训练集大小: {len(train_df)}")
        print(f"验证集大小: {len(val_df)}")
        print(f"测试集大小: {len(test_df)}")
        if len(test_df) > 0:
            print("注意：测试集已加载但当前代码中未使用")

        # 显示一些示例标签
        unique_labels = train_df['caption'].unique()
        print(f"训练集中共有 {len(unique_labels)} 个唯一标签")
        if len(unique_labels) <= 20:  # 如果标签数量不多，打印所有标签
            print(f"标签列表: {sorted(unique_labels)}")
        else:
            print(f"前10个标签: {sorted(unique_labels)[:10]}")

    # 加载CLIP处理器
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # 创建数据集和数据加载器
    train_dataset = CustomImageCaptionDataset_multi(train_df, processor)
    val_dataset = CustomImageCaptionDataset_multi(val_df, processor)

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
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)

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

    # 修改后的 DeepSpeed 配置
    ds_config = {
        "train_batch_size": BATCH_SIZE * world_size,
        "train_micro_batch_size_per_gpu": BATCH_SIZE,
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
        "fp16": {
            "enabled": True,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
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
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        num_batches = 0

        if local_rank == 0:
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [训练]")
        else:
            progress_bar = train_dataloader

        for batch in progress_bar:
            if batch is None:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            logits_img_per_img = outputs.logits_img_per_img

            labels = torch.arange(logits_per_image.size(0)).to(device)

            loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
            loss_img = torch.nn.functional.cross_entropy(logits_img_per_img, labels)

            lmd = 1
            loss = (loss_i + loss_t + lmd * loss_img) / 3

            model.backward(loss)
            model.step()

            total_loss += loss.item()
            num_batches += 1

            current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
            if local_rank == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{current_lr:.2e}"
                })

        avg_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        if local_rank == 0:
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} 平均训练损失: {avg_train_loss:.8f}")
            print("--- 正在评估模型 ---")

        f1, cls_rep, avg_cosine_similarity, pr_auc = evaluate(model, val_dataloader, device, world_size, local_rank)

        if local_rank == 0:
            print(f"F1分数 (Macro): {f1:.4f}")
            print(f"PR-AUC: {pr_auc:.4f}")
            print(f"平均余弦相似度: {avg_cosine_similarity:.4f}")

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

    dist.destroy_process_group()