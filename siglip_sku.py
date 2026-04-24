import os
import pandas as pd
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, AutoModel, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_recall_curve, auc
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import numpy as np
import shutil
import deepspeed

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


# 生成数据集信息的函数（保持不变）
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


# 自定义数据集类（修改为适应SigLIP处理器）
class CustomImageCaptionDataset_multi(Dataset):
    def __init__(self, df, image_dirs, processor):
        self.df = df
        self.image_dirs = image_dirs if isinstance(image_dirs, list) else [image_dirs]
        self.processor = processor
        self.image_paths = df['image'].tolist()
        self.captions = df['caption'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_rel_path = self.image_paths[idx]
        caption = self.captions[idx]

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

        # SigLIP处理方式 - 添加max_length限制
        try:
            inputs = self.processor(
                text=[caption],
                images=image,
                return_tensors="pt",
                padding=True,
                max_length=64,  # 添加最大长度限制
                truncation=True  # 启用截断
            )

            item = {
                "input_ids": inputs["input_ids"].squeeze(0),
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "caption": caption
            }

            if "attention_mask" in inputs:
                item["attention_mask"] = inputs["attention_mask"].squeeze(0)
            else:
                item["attention_mask"] = torch.ones_like(item["input_ids"])

        except Exception as e:
            print(f"处理样本时出错: {e}")
            return None

        return item


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    captions = [item.pop('caption') for item in batch]

    # 动态获取所有键
    keys = batch[0].keys()
    collated_batch = {}

    for key in keys:
        if key == 'caption':
            continue

        # 专门处理文本相关的变长序列
        if key in ['input_ids', 'attention_mask']:
            # 找到最大长度
            max_len = max(item[key].shape[0] for item in batch)
            padded_tensors = []

            for item in batch:
                tensor = item[key]
                current_len = tensor.shape[0]

                if current_len < max_len:
                    # 计算需要padding的长度
                    pad_size = max_len - current_len

                    if key == 'input_ids':
                        # 使用pad_token_id进行padding (SigLIP通常使用0)
                        pad_value = 0
                    else:  # attention_mask
                        pad_value = 0  # attention_mask的padding通常用0

                    # 创建padding张量
                    if len(tensor.shape) == 1:
                        pad_tensor = torch.full((pad_size,), pad_value, dtype=tensor.dtype, device=tensor.device)
                    else:
                        # 如果是2D或更高维，需要更复杂的padding
                        pad_tensor = torch.full((pad_size, tensor.shape[1]), pad_value, dtype=tensor.dtype,
                                                device=tensor.device)

                    # 拼接原始张量和padding
                    padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
                    padded_tensors.append(padded_tensor)
                else:
                    padded_tensors.append(tensor)

            collated_batch[key] = torch.stack(padded_tensors)

        else:
            # 其他张量正常stack
            try:
                collated_batch[key] = torch.stack([item[key] for item in batch])
            except Exception as e:
                print(f"处理键 {key} 时出错: {e}")
                # 如果stack失败，尝试其他方式或跳过
                continue

    collated_batch['captions'] = captions
    return collated_batch


# 评估函数（修改为适应SigLIP）
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
            pixel_values = batch["pixel_values"].to(device)
            captions = batch["captions"]

            # 检查是否有attention_mask
            if "attention_mask" in batch:
                attention_mask = batch["attention_mask"].to(device)
            else:
                attention_mask = None

            # SigLIP前向传播
            if attention_mask is not None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            else:
                outputs = model(input_ids=input_ids, pixel_values=pixel_values)

            logits_per_image = outputs.logits_per_image

            # SigLIP输出需要sigmoid处理
            similarity_matrix = torch.sigmoid(logits_per_image)
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

        # SigLIP使用sigmoid而不是softmax
        probs = matrix  # 已经是sigmoid输出
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


# SigLIP损失函数
def siglip_loss(logits_per_image, logits_per_text, labels=None):
    """SigLIP的sigmoid损失函数"""
    # 对角线应该是正样本，其他是负样本
    batch_size = logits_per_image.shape[0]

    # 创建目标：对角线为1，其他为0
    targets = torch.eye(batch_size, device=logits_per_image.device)

    # 图像到文本的损失
    loss_i = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_per_image, targets, reduction='mean'
    )

    # 文本到图像的损失
    loss_t = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_per_text, targets, reduction='mean'
    )

    return (loss_i + loss_t) / 2


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
    character_path = "/media/nvme/workspace/vincent/character/character_data/"
    IMAGE_DIR1 = character_path + "data_930/image_9"
    IMAGE_DIR2 = character_path + "data_930"
    # IMAGE_DIR3 = "./data/data_flower"
    TRAIN_DATA_DIR = character_path + "train"
    # TEST_DATA_DIR = character_path + "test"
    # TRAIN_DATA_DIR = "./character_data/train"
    # TEST_DATA_DIR = "./character_data/test"
    TRAIN_CSV_PATH1 = character_path + "data_930/train.csv"
    # TEST_CSV_PATH1 = character_path + "data_930/test.csv"
    TRAIN_CSV_PATH2 = character_path + "data_930/train_0930.csv"
    TEST_CSV_PATH2 = character_path + "data_930/val_0930.csv"
    image_train = [TRAIN_DATA_DIR, IMAGE_DIR1, IMAGE_DIR2]
    image_test = [IMAGE_DIR1, IMAGE_DIR2]
    MODEL_NAME = "/media/nvme/workspace/vincent/models/siglip2-giant-opt-patch16-384"  # 改为SigLIP模型
    BATCH_SIZE = 8  # 减小批次大小以防内存问题
    NUM_EPOCHS = 1000
    LEARNING_RATE = 5e-6
    SCHEDULER_TYPE = "cosine_with_restarts"
    CHECKPOINT_DIR = "ds_siglip_checkpoints_character"  # 修改检查点目录名
    RESUME = True
    use_main_category_only = True

    # 创建检查点目录
    if local_rank == 0 and not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # --- 数据加载和准备 ---
    if local_rank == 0:
        print("正在生成数据集信息...")
        print(f"正在使用 {world_size} 个GPU进行训练")
        print(f"使用模型: {MODEL_NAME}")

    train_df1 = generate_dataset_info(TRAIN_DATA_DIR, use_main_category_only=use_main_category_only)
    # val_df1 = generate_dataset_info(TEST_DATA_DIR, use_main_category_only=use_main_category_only)
    train_df2 = pd.read_csv(TRAIN_CSV_PATH1)
    # val_df2 = pd.read_csv(TEST_CSV_PATH1)
    train_df2 = train_df2[['image', 'caption']]
    # val_df2 = val_df2[['image', 'caption']]

    train_df3 = pd.read_csv(TRAIN_CSV_PATH2)
    val_df3 = pd.read_csv(TEST_CSV_PATH2)
    train_df3 = train_df3[['image', 'caption']]
    val_df3 = val_df3[['image', 'caption']]

    train_df = pd.concat([train_df1, train_df2, train_df3], ignore_index=True)
    val_df = pd.concat([val_df3], ignore_index=True)

    if local_rank == 0:
        print(f"训练集大小: {len(train_df)}")
        print(f"验证集大小: {len(val_df)}")

    # 加载SigLIP处理器
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # 创建数据集和数据加载器
    train_dataset = CustomImageCaptionDataset_multi(train_df, image_train, processor)
    val_dataset = CustomImageCaptionDataset_multi(val_df, image_test, processor)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,  # 减少worker数量
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

    # --- 加载SigLIP模型 ---
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

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
        "train_batch_size": BATCH_SIZE * world_size,
        "train_micro_batch_size_per_gpu": BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True
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
            pixel_values = batch["pixel_values"].to(device)

            # 检查是否有attention_mask
            if "attention_mask" in batch:
                attention_mask = batch["attention_mask"].to(device)
            else:
                attention_mask = None

            # SigLIP前向传播
            if attention_mask is not None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            else:
                outputs = model(input_ids=input_ids, pixel_values=pixel_values)

            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            # 使用SigLIP的sigmoid损失函数
            loss = siglip_loss(logits_per_image, logits_per_text)

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
