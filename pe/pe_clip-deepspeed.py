import os
import pandas as pd
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import get_scheduler
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_recall_curve, auc
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import open_clip

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


# 自定义数据集类（保持不变接口，只替换内部对 processor 的使用）
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
        for d in self.image_dirs:
            image_path = os.path.join(d, image_rel_path)
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"警告：无法打开图片文件 {image_path}: {e}")
                    continue
                break

        if image is None:
            print(f"错误：在所有目录中都找不到图片 {image_rel_path}，将跳过此样本。")
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
    if len(batch) == 0:
        return None
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    captions = [item["caption"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "captions": captions
    }


# 评估函数（修改：使用模型直接计算）
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

            # 使用模型的forward方法直接计算
            outputs = model(pixel_values, input_ids)
            similarity_matrix = outputs.logits_per_image.cpu()

            local_similarity_matrices.append(similarity_matrix)
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

class ProcessorShim:
    def __init__(self, preprocess, device=None):
        self.preprocess = preprocess
        self.device = device

    def __call__(self, text, images, return_tensors="pt", padding=True, truncation=True):
        # images: PIL Image
        # text: string or list of strings
        if isinstance(images, (list, tuple)):
            img_tensors = [self.preprocess(img) for img in images]
            pixel_values = torch.stack(img_tensors)
        else:
            pixel_values = self.preprocess(images).unsqueeze(0)
        # tokenize text using open_clip.tokenize (returns LongTensor)
        if isinstance(text, (list, tuple)):
            input_ids = open_clip.tokenize(list(text))
        else:
            input_ids = open_clip.tokenize([text])
        # attention mask: nonzero tokens
        attention_mask = (input_ids != 0).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}

class OpenCLIPWrapper(torch.nn.Module):
    def __init__(self, openclip_model):
        super().__init__()
        self.model = openclip_model
        # open_clip model stores logit_scale as parameter or buffer
        try:
            self.logit_scale = self.model.logit_scale
        except AttributeError:
            # create a parameter if not present
            self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))
            self.model.logit_scale = self.logit_scale

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        # input_ids: (batch, seq_len) or torch.LongTensor
        # pixel_values: images tensor (batch, C, H, W)
        device = pixel_values.device if pixel_values is not None else next(self.model.parameters()).device
        # encode images
        image_features = self.model.encode_image(pixel_values)
        if input_ids is not None:
            # open_clip expects token ids as LongTensor on device
            text_input = input_ids.to(device)
            text_features = self.model.encode_text(text_input)
        else:
            text_features = None
        # normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if text_features is not None:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
        else:
            logits_per_image = None
            logits_per_text = None
        # mimic transformers output object with attributes
        class Out:
            pass
        out = Out()
        out.logits_per_image = logits_per_image
        out.logits_per_text = logits_per_text
        return out
# 3. 主执行逻辑（修改后的版本）
if __name__ == "__main__":
    # --- 初始化分布式训练 ---
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # --- 配置参数 ---
    character_path = "/data1/vincent/datasets/data_gray/"  # "/Users/vincent/workspace/trademark_similar/character_data"#"/data1/vincent/datasets/data_gray/"

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
    use_main_category_only = True

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
        print("正在生成数据集信息...")
        print(f"正在使用 {world_size} 个GPU进行训练")

    train_df = generate_dataset_info(TRAIN_DATA_DIR, use_main_category_only=use_main_category_only)
    val_df = generate_dataset_info(TEST_DATA_DIR, use_main_category_only=use_main_category_only)

    if local_rank == 0:
        print(f"训练集大小: {len(train_df)}")
        print(f"验证集大小: {len(val_df)}")

    # 使用open_clip加载模型和预处理
    if local_rank == 0:
        print(f"正在加载模型: {MODEL_NAME}")

    # 加载模型
    model, preprocess = open_clip.create_model_from_pretrained(f"local-dir:{MODEL_NAME}")
    processor = ProcessorShim(preprocess)
    model = OpenCLIPWrapper(model)

    # 获取tokenizer
    tokenizer = open_clip.get_tokenizer(f"local-dir:{MODEL_NAME}")

    # 将模型移到GPU
    model = model.to(device)

    # 创建数据集和数据加载器
    train_dataset = CustomImageCaptionDataset_multi(train_df, image_train, processor)
    val_dataset = CustomImageCaptionDataset_multi(val_df, image_test, processor)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, collate_fn=collate_fn)

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

    # 修改后的 DeepSpeed 配置 - 禁用混合精度以解决类型不匹配问题
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
        # 暂时禁用混合精度训练
        "fp16": {
            "enabled": False  # 先禁用，后续再启用
        },
        "bf16": {
            "enabled": False
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

            # 使用模型的forward方法
            outputs = model(pixel_values, input_ids)

            # 计算对比损失
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            labels = torch.arange(logits_per_image.size(0)).to(device)
            loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2

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