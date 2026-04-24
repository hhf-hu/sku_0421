import os
import pandas as pd
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import CLIPProcessor, CLIPModel, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_recall_curve, auc
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import numpy as np
import shutil
from model import save_checkpoint,load_checkpoint
from eval import evaluate
import torchvision.transforms as transforms



os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 生成数据集信息的函数
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


# 1. 自定义数据集类
class CustomImageCaptionDataset(Dataset):
    def __init__(self, df, image_dir, processor):
        self.df = df
        self.image_dir = image_dir
        self.processor = processor
        self.image_paths = df['image'].tolist()
        self.captions = df['caption'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"警告：找不到图片文件 {image_path}，将跳过此样本。")
            return None

        caption = self.captions[idx]

        # 使用CLIP处理器来准备图片和文本
        inputs = self.processor(text=caption, images=image, return_tensors="pt", padding="max_length", truncation=True)

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "pixel_values": inputs["pixel_values"].squeeze(),
        }

class CustomImageCaptionDataset_multi(Dataset):
    def __init__(self, df, image_dirs, processor,transform=None):
        self.df = df
        self.image_dirs = image_dirs if isinstance(image_dirs, list) else [image_dirs]
        self.processor = processor
        self.image_paths = df['image'].tolist()
        self.captions = df['caption'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_rel_path = self.image_paths[idx]
        caption = self.captions[idx]

        # 尝试在多个目录中查找图片
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

        # 应用图片变换（如果有）
        if self.transform:
            image = self.transform(image)
        # 使用CLIP处理器来准备图片和文本
        inputs = self.processor(text=caption, images=image, return_tensors="pt", padding="max_length", truncation=True)

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "pixel_values": inputs["pixel_values"].squeeze(),
        }

# 修正后的 collate_fn
def collate_fn(batch):
    # 过滤掉数据加载过程中返回None的样本
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    # 使用默认的collate函数来堆叠样本
    return torch.utils.data.dataloader.default_collate(batch)


# 3. 主执行逻辑
if __name__ == "__main__":
    # --- 初始化分布式训练 ---
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # --- 配置参数 ---
    IMAGE_DIR = "./data/image_9"
    TRAIN_DATA_DIR = "./character_data/train"  # 训练数据目录
    TEST_DATA_DIR = "./character_data/test"  # 测试数据目录
    TRAIN_CSV_PATH = "./data/train.csv"  # 修改为训练集路径
    TEST_CSV_PATH = "./data/test.csv"  # 修改为测试集路径
    image_train = [TRAIN_DATA_DIR, IMAGE_DIR]
    image_test = [TEST_DATA_DIR, IMAGE_DIR]
    MODEL_NAME = "apple/DFN5B-CLIP-ViT-H-14-378"
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-5
    SCHEDULER_TYPE = "cosine_with_restarts"
    CHECKPOINT_DIR = "dfn5b_checkpoints_character"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pth.tar")
    BEST_F1_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_best_f1.pth.tar")
    BEST_LOSS_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_best_loss.pth.tar")
    BEST_PRAUC_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_best_pr_auc.pth.tar")
    BEST_SIMILARITY_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_best_similarity.pth.tar")
    RESUME = True  # 是否从检查点恢复训练
    use_main_category_only = False #是否使用主标签训练

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


    # 从文件夹结构生成数据集信息
    train_df1 = generate_dataset_info(TRAIN_DATA_DIR,use_main_category_only=use_main_category_only)
    val_df1 = generate_dataset_info(TEST_DATA_DIR,use_main_category_only=use_main_category_only)
    train_df2 = pd.read_csv(TRAIN_CSV_PATH)
    val_df2 = pd.read_csv(TEST_CSV_PATH)  # 这里使用测试集作为验证集
    train_df2 = train_df2[['image', 'caption']]
    val_df2 = val_df2[['image', 'caption']]
    train_df = pd.concat([train_df1, train_df2], ignore_index=True)
    val_df = pd.concat([val_df1, val_df2], ignore_index=True)


    if local_rank == 0:
        print(f"训练集大小: {len(train_df)}")
        print(f"验证集大小: {len(val_df)}")
        print(f"类别数量: {len(train_df['caption'].unique())}")
        print("示例数据:")
        print(train_df.head())

    # 加载CLIP处理器
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # 创建数据集和数据加载器
    # 定义图片预处理变换
    image_transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # 调整图片大小，根据模型输入尺寸调整
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，增强数据
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),  # 转换为Tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    # train_dataset = CustomImageCaptionDataset(train_df, TRAIN_DATA_DIR, processor)
    # val_dataset = CustomImageCaptionDataset(val_df, TEST_DATA_DIR, processor)
    train_dataset = CustomImageCaptionDataset_multi(train_df, image_train, processor,transform=image_transform)
    val_dataset = CustomImageCaptionDataset_multi(val_df, image_test, processor,transform=image_transform)

    # 使用DistributedSampler
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

    # 使用带warmup的学习率调度
    num_warmup_steps = int(0.02 * num_training_steps)  # 2%的warmup
    scheduler = get_scheduler(
        SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 使用DistributedDataParallel包装模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    start_epoch = 0
    best_f1 = 0.0
    best_loss = float('inf')
    best_pr_auc = 0.0
    best_similarity = 0.0  # 新增：最佳相似度

    # --- 评估微调前的模型 ---
    # if local_rank == 0 and start_epoch == 0:
    #     print("\n--- 正在评估原始模型 ---")
    #     f1_before, cls_rep, avg_cosine_similarity, pr_auc_before = evaluate(model, val_dataloader, device, world_size)
    #     print(f"微调前的F1分数 (Macro): {f1_before:.4f}")
    #     print(f"微调前的PR-AUC: {pr_auc_before:.4f}")
    #     print(cls_rep)
    #     print(f"平均余弦相似度: {avg_cosine_similarity:.4f}")
    #     best_pr_auc = pr_auc_before
    #     best_similarity = avg_cosine_similarity  # 新增：初始化最佳相似度

    # 恢复训练
    if RESUME and local_rank == 0:
        start_epoch, best_f1, best_loss, best_pr_auc = load_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler,
                                                                       device, use_ddp=True)
        start_epoch += 1  # 从下一个epoch开始

    # 确保所有进程同步
    if world_size > 1:
        dist.barrier()

    # --- 训练循环 ---
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        num_batches = 0

        # 训练进度条
        if local_rank == 0:
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [训练]")
        else:
            progress_bar = train_dataloader

        for batch in progress_bar:
            if batch is None:
                continue

            optimizer.zero_grad()

            # 将批次数据移动到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

            # 计算对比损失
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            # 创建标签：对角线为1，其余为0
            labels = torch.arange(logits_per_image.size(0)).to(device)

            # 图像到文本和文本到图像的损失
            loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2

            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if local_rank == 0:
                progress_bar.set_postfix(loss=loss.item())

        # 计算平均训练损失
        avg_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # 评估模型
        if local_rank == 0:
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} 平均训练损失: {avg_train_loss:.8f}")
            print("--- 正在评估模型 ---")

        f1, cls_rep, avg_cosine_similarity, pr_auc = evaluate(model, val_dataloader, device, world_size)

        if local_rank == 0:
            print(f"F1分数 (Macro): {f1:.4f}")
            print(f"PR-AUC: {pr_auc:.4f}")
            print(cls_rep)
            print(f"平均余弦相似度: {avg_cosine_similarity:.4f}")

            # 检查是否为最佳模型
            is_best_f1 = f1 > best_f1
            is_best_loss = avg_train_loss < best_loss
            is_best_pr_auc = pr_auc > best_pr_auc
            is_best_similarity = avg_cosine_similarity > best_similarity  # 新增：检查是否为最佳相似度

            if is_best_f1:
                best_f1 = f1
            if is_best_loss:
                best_loss = avg_train_loss
            if is_best_pr_auc:
                best_pr_auc = pr_auc
            if is_best_similarity:  # 新增：更新最佳相似度
                best_similarity = avg_cosine_similarity

            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_f1': best_f1,
                'best_loss': best_loss,
                'best_pr_auc': best_pr_auc,
                'best_similarity': best_similarity  # 新增：保存最佳相似度
            }

            save_checkpoint(checkpoint, is_best_f1, is_best_loss, is_best_pr_auc, is_best_similarity,  # 修改：添加is_best_similarity参数
                           CHECKPOINT_PATH, BEST_F1_MODEL_PATH, BEST_LOSS_MODEL_PATH, BEST_PRAUC_MODEL_PATH, BEST_SIMILARITY_MODEL_PATH)  # 修改：添加BEST_SIMILARITY_MODEL_PATH参数

            print(f"当前最佳 F1: {best_f1:.4f}, 最佳损失: {best_loss:.8f}, 最佳PR-AUC: {best_pr_auc:.4f}, 最佳相似度: {best_similarity:.4f}")  # 修改：添加最佳相似度输出

    # 清理分布式进程组
    dist.destroy_process_group()