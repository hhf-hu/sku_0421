import os
import pandas as pd
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import get_scheduler
from torch.optim import AdamW, lr_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error, r2_score, recall_score, classification_report
import shutil

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

from datasets_process import CustomImageCaptionDataset, collate_fn
from evaluate import evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7,8'


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

# 保存检查点函数 - 修改为支持保存最佳F1、最佳损失和最佳PR-AUC模型
def save_checkpoint(state, is_best_f1, is_best_loss, is_best_pr_auc, filename,
                    best_f1_filename, best_loss_filename, best_pr_auc_filename):
    torch.save(state, filename)
    if is_best_f1:
        shutil.copyfile(filename, best_f1_filename)
    if is_best_loss:
        shutil.copyfile(filename, best_loss_filename)
    if is_best_pr_auc:
        shutil.copyfile(filename, best_pr_auc_filename)


# 修正的加载检查点函数 - 处理DDP前缀问题
def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, use_ddp=True):
    if os.path.isfile(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)

        # 处理状态字典的键名不匹配问题
        state_dict = checkpoint['state_dict']

        if use_ddp:
            # 如果当前使用DDP但检查点没有DDP前缀，需要添加前缀
            if not any(key.startswith('module.') for key in state_dict.keys()):
                # 添加module.前缀
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = 'module.' + k if not k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict
        else:
            # 如果当前不使用DDP但检查点有DDP前缀，需要移除前缀
            if any(key.startswith('module.') for key in state_dict.keys()):
                # 移除module.前缀
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '') if k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict

        # 加载状态字典
        model.load_state_dict(state_dict)

        # 加载优化器和调度器状态
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint.get('best_f1', 0.0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        best_pr_auc = checkpoint.get('best_pr_auc', 0.0)  # 新增最佳PR-AUC

        print(
            f"从 epoch {start_epoch} 继续训练，最佳 F1: {best_f1:.4f}, 最佳损失: {best_loss:.4f}, 最佳PR-AUC: {best_pr_auc:.4f}")
        return start_epoch, best_f1, best_loss, best_pr_auc
    else:
        print(f"未找到检查点: {checkpoint_path}，从头开始训练")
        return 0, 0.0, float('inf'), 0.0


# 3. 主执行逻辑
if __name__ == "__main__":
    # --- 初始化分布式训练 ---
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # --- 配置参数 ---
    # IMAGE_DIR = "./data/image_9"
    # TRAIN_CSV_PATH = "./data/train.csv"  # 修改为训练集路径
    # TEST_CSV_PATH = "./data/test.csv"  # 修改为测试集路径
    # --- 配置参数 ---
    character_path = "/data1/vincent/datasets/data_gray/"  # "/Users/vincent/workspace/trademark_similar/character_data"#"/data1/vincent/datasets/data_gray/"

    TRAIN_DATA_DIR = character_path + "train"
    TEST_DATA_DIR = character_path + "val"

    image_train = [TRAIN_DATA_DIR]
    image_test = [TEST_DATA_DIR]
    MODEL_NAME = "/data1/vincent/models/facebook-PE-Core-G14-448/PE-Core-G14-448.pt"
    BATCH_SIZE = 2
    NUM_EPOCHS = 1000
    LEARNING_RATE = 5e-6
    SCHEDULER_TYPE = "cosine_with_restarts"
    CHECKPOINT_DIR = "ds_dfn5b_checkpoints_sku"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pth.tar")
    BEST_F1_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_best_f1.pth.tar")
    BEST_LOSS_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_best_loss.pth.tar")
    BEST_PRAUC_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_best_pr_auc.pth.tar")
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

    # --- 加载PE-Core模型和处理器 ---
    if local_rank == 0:
        print("正在加载PE-Core模型...")
        print(f"正在使用 {world_size} 个GPU进行训练")

    # 加载模型
    model = pe.CLIP.from_config(name='PE-Core-G14-448',checkpoint_path=MODEL_NAME, pretrained=True).to(device)

    # 获取预处理函数和tokenizer
    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)

    # 准备优化器
    # optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 数据加载和准备 ---
    if local_rank == 0:
        print("正在加载数据...")
    # 分别加载训练集和测试集
    # train_df = pd.read_csv(TRAIN_CSV_PATH)
    # val_df = pd.read_csv(TEST_CSV_PATH)  # 这里使用测试集作为验证集
    # 从文件夹结构生成数据集信息
    train_df = generate_dataset_info(TRAIN_DATA_DIR, use_main_category_only=use_main_category_only)
    val_df = generate_dataset_info(TEST_DATA_DIR, use_main_category_only=use_main_category_only)

    # 创建数据集和数据加载器
    # train_dataset = CustomImageCaptionDataset(train_df, IMAGE_DIR, preprocess, tokenizer)
    # val_dataset = CustomImageCaptionDataset(val_df, IMAGE_DIR, preprocess, tokenizer)
    train_dataset = CustomImageCaptionDataset(train_df, TRAIN_DATA_DIR, preprocess, tokenizer)
    val_dataset = CustomImageCaptionDataset(val_df, TEST_DATA_DIR, preprocess, tokenizer)

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

    # 配置学习率调度器
    num_training_steps = NUM_EPOCHS * len(train_dataloader)

    optimizer = AdamW(model.parameters(),
                      lr=LEARNING_RATE,
                      weight_decay=0.01,  # 添加权重衰减
                      betas=(0.9, 0.999),  # 明确的beta参数
                      eps=1e-8)

    # 使用带warmup的学习率调度
    num_warmup_steps = int(0.02 * num_training_steps)  # 10%的warmup
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
    best_pr_auc = 0.0  # 初始化最佳PR-AUC

    # --- 评估微调前的模型 ---
    if local_rank == 0 and start_epoch == 0:
        print("\n--- 正在评估原始模型 ---")
        f1_before, cls_rep, avg_cosine_similarity, pr_auc_before = evaluate(model, val_dataloader, device, world_size)
        print(f"微调前的F1分数 (Macro): {f1_before:.4f}")
        print(f"微调前的PR-AUC: {pr_auc_before:.4f}")
        print(cls_rep)
        print(f"平均余弦相似度: {avg_cosine_similarity:.4f}")
        best_pr_auc = pr_auc_before  # 初始化最佳PR-AUC

    # 恢复训练
    if RESUME and local_rank == 0:
        start_epoch, best_f1, best_loss, best_pr_auc = load_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler,
                                                                       device, use_ddp=True)
        start_epoch += 1  # 从下一个epoch开始

    # 广播开始epoch到所有进程
    start_epoch_tensor = torch.tensor([start_epoch], device=device)
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = start_epoch_tensor.item()

    # --- 微调模型 ---
    if local_rank == 0:
        print(f"\n--- 开始微调模型 (从 epoch {start_epoch} 开始) ---")

    # 训练循环
    if local_rank == 0:
        progress_bar = tqdm(range(start_epoch * len(train_dataloader), num_training_steps))
    epoch_losses = []

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None:
                continue

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)

            # 前向传播
            with torch.autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
                image_features, text_features, logit_scale = model(pixel_values, input_ids)

                # 计算对比损失
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # 计算相似度矩阵
                logits_per_image = logit_scale * torch.matmul(image_features, text_features.t())
                logits_per_text = logits_per_image.t()

                # 创建标签
                labels = torch.arange(logits_per_image.shape[0]).to(device)

                # 计算对比损失
                loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels, label_smoothing=0.02)
                loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels, label_smoothing=0.02)
                loss = (loss_img + loss_txt) * 100 / 2

            epoch_loss += loss.item()
            num_batches += 1

            # 反向传播
            loss.backward()
            optimizer.step()

            if SCHEDULER_TYPE != "reduce_on_plateau":
                scheduler.step()

            optimizer.zero_grad()

            if local_rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.8f}, LR: {current_lr:.2e}")

        # 计算平均epoch loss（需要跨所有GPU同步）
        avg_epoch_loss = torch.tensor(epoch_loss / num_batches if num_batches > 0 else 0).to(device)
        dist.all_reduce(avg_epoch_loss, op=dist.ReduceOp.SUM)
        avg_epoch_loss = avg_epoch_loss.item() / world_size
        epoch_losses.append(avg_epoch_loss)

        if SCHEDULER_TYPE == "reduce_on_plateau":
            scheduler.step(avg_epoch_loss)

        # 每2个epoch或在最后一个epoch评估并保存模型
        if (epoch + 1) % 2 == 0 or epoch == NUM_EPOCHS - 1:
            if local_rank == 0:
                print(f"Epoch {epoch + 1} 完成，平均损失: {avg_epoch_loss:.8f}")
                print("正在评估模型...")

            f1_after, cls_rep, avg_cosine_similarity, pr_auc_after = evaluate(model, val_dataloader, device, world_size=1)

            if local_rank == 0:
                print(f"Epoch {epoch + 1} 评估结果:")
                print(f"F1分数: {f1_after:.4f}")
                print(f"PR-AUC: {pr_auc_after:.4f}")
                print(f"平均余弦相似度: {avg_cosine_similarity:.4f}")
                print(f"分类报告: {cls_rep}")

                # 检查是否为最佳F1模型
                is_best_f1 = f1_after > best_f1
                if is_best_f1:
                    best_f1 = f1_after
                    print(f"新的最佳F1分数: {best_f1:.4f}")

                # 检查是否为最佳损失模型
                is_best_loss = avg_epoch_loss < best_loss
                if is_best_loss:
                    best_loss = avg_epoch_loss
                    print(f"新的最佳损失: {best_loss:.8f}")

                # 检查是否为最佳PR-AUC模型
                is_best_pr_auc = pr_auc_after > best_pr_auc
                if is_best_pr_auc:
                    best_pr_auc = pr_auc_after
                    print(f"新的最佳PR-AUC: {best_pr_auc:.4f}")

                # 保存检查点
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'best_f1': best_f1,
                    'best_loss': best_loss,
                    'best_pr_auc': best_pr_auc,  # 保存最佳PR-AUC
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }

                # 保存检查点，同时考虑F1、损失和PR-AUC
                save_checkpoint(checkpoint, is_best_f1, is_best_loss, is_best_pr_auc,
                                CHECKPOINT_PATH, BEST_F1_MODEL_PATH, BEST_LOSS_MODEL_PATH, BEST_PRAUC_MODEL_PATH)
                print(f"检查点已保存到 {CHECKPOINT_PATH}")

                if is_best_f1:
                    print(f"最佳F1模型已保存到 {BEST_F1_MODEL_PATH}")
                if is_best_loss:
                    print(f"最佳损失模型已保存到 {BEST_LOSS_MODEL_PATH}")
                if is_best_pr_auc:
                    print(f"最佳PR-AUC模型已保存到 {BEST_PRAUC_MODEL_PATH}")

    # --- 最终评估 ---
    if local_rank == 0:
        print("\n--- 最终评估 ---")
        # 加载最佳PR-AUC模型进行最终评估
        if os.path.exists(BEST_PRAUC_MODEL_PATH):
            # 创建临时模型实例来加载最佳权重
            best_pr_auc_model = pe.CLIP.from_config(MODEL_NAME, pretrained=False).to(device)
            best_checkpoint = torch.load(BEST_PRAUC_MODEL_PATH, map_location=device,weights_only=False)

            # 处理状态字典键名
            state_dict = best_checkpoint['state_dict']
            if any(key.startswith('module.') for key in state_dict.keys()):
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '') if k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict

            best_pr_auc_model.load_state_dict(state_dict)
            best_pr_auc_model = best_pr_auc_model.to(device)
            print("已加载最佳PR-AUC模型进行最终评估")

            # 评估最佳PR-AUC模型
            f1_pr_auc, cls_rep_pr_auc, avg_cosine_similarity_pr_auc, pr_auc_final = evaluate(best_pr_auc_model,
                                                                                             val_dataloader, device,
                                                                                             world_size=1)
            print(f"最佳PR-AUC模型的F1分数: {f1_pr_auc:.4f}")
            print(f"最佳PR-AUC: {pr_auc_final:.4f}")
            print(f"分类报告: {cls_rep_pr_auc}")
            print(f"平均余弦相似度: {avg_cosine_similarity_pr_auc:.4f}")

        # 也可以加载最佳F1模型和最佳损失模型进行评估
        if os.path.exists(BEST_F1_MODEL_PATH):
            best_f1_model = pe.CLIP.from_config(MODEL_NAME, pretrained=False).to(device)
            best_f1_checkpoint = torch.load(BEST_F1_MODEL_PATH, map_location=device,weights_only=False)

            # 处理状态字典键名
            state_dict = best_f1_checkpoint['state_dict']
            if any(key.startswith('module.') for key in state_dict.keys()):
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '') if k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict

            best_f1_model.load_state_dict(state_dict)
            best_f1_model = best_f1_model.to(device)
            print("已加载最佳F1模型进行评估")

            f1_final, cls_rep_final, avg_cosine_similarity_final, pr_auc_f1 = evaluate(best_f1_model, val_dataloader,
                                                                                       device, world_size=1)
            print(f"最佳F1模型的F1分数: {f1_final:.4f}")
            print(f"最佳F1模型的PR-AUC: {pr_auc_f1:.4f}")

        if os.path.exists(BEST_LOSS_MODEL_PATH):
            best_loss_model = pe.CLIP.from_config(MODEL_NAME, pretrained=False).to(device)
            best_loss_checkpoint = torch.load(BEST_LOSS_MODEL_PATH, map_location=device,weights_only=False)

            # 处理状态字典键名
            state_dict = best_loss_checkpoint['state_dict']
            if any(key.startswith('module.') for key in state_dict.keys()):
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '') if k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict

            best_loss_model.load_state_dict(state_dict)
            best_loss_model = best_loss_model.to(device)
            print("已加载最佳损失模型进行评估")

            f1_loss_model, cls_rep_loss, avg_cosine_similarity_loss, pr_auc_loss_model = evaluate(best_loss_model,
                                                                                                  val_dataloader,
                                                                                                  device, world_size=1)
            print(f"最佳损失模型的F1分数: {f1_loss_model:.4f}")
            print(f"最佳损失模型的PR-AUC: {pr_auc_loss_model:.4f}")

    # 清理分布式进程
    dist.destroy_process_group()