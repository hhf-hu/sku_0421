"""
DINOv3 Patch-Level Difference Detection
完整实现：以patch为单位检测两张图片的差异区域
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import AutoImageProcessor, AutoModel
import cv2
import random


# ==================== 配置部分 ====================
class Config:
    """配置类"""

    def __init__(self):
        # 模型配置
        self.model_name = "facebook/dinov2-base"  # 可以使用 dinov3-small, dinov3-base 等
        self.patch_size = 16  # DINOv3的标准patch大小
        self.num_classes = 2  # 0: 相同, 1: 不同

        # 数据配置
        self.data_root = "/path/to/your/dataset"  # 修改为你的数据集路径
        self.image_size = (224, 224)  # 输入图像尺寸
        self.train_split = 0.7
        self.val_split = 0.15
        self.test_split = 0.15

        # 训练配置
        self.batch_size = 8
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.min_lr = 1e-6
        self.patience = 10  # early stopping patience

        # 损失函数配置
        self.pos_weight = 3.0  # 正样本权重（差异patch通常较少）
        self.focal_gamma = 2.0  # 焦点损失的gamma参数
        self.dice_weight = 0.3  # Dice损失的权重

        # 模型保存
        self.save_dir = "./patch_diff_models"
        self.experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 其他
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42

    def setup_paths(self):
        """创建必要的目录"""
        self.model_save_dir = os.path.join(self.save_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.model_save_dir, "checkpoints")
        self.log_dir = os.path.join(self.model_save_dir, "logs")
        self.result_dir = os.path.join(self.model_save_dir, "results")

        for path in [self.model_save_dir, self.checkpoint_dir, self.log_dir, self.result_dir]:
            os.makedirs(path, exist_ok=True)

        return self


# ==================== 数据集类 ====================
class PatchDiffDataset(Dataset):
    """Patch级差异检测数据集"""

    def __init__(self, data_dir, processor, image_size=(224, 224),
                 patch_size=16, is_train=True, transform=None):
        """
        初始化数据集

        Args:
            data_dir: 数据集目录
            processor: 图像处理器
            image_size: 图像大小
            patch_size: patch大小
            is_train: 是否为训练集
            transform: 数据增强变换
        """
        self.data_dir = data_dir
        self.processor = processor
        self.image_size = image_size
        self.patch_size = patch_size
        self.is_train = is_train
        self.transform = transform

        # 计算patch网格
        self.grid_h = image_size[0] // patch_size
        self.grid_w = image_size[1] // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # 收集数据样本
        self.samples = self._load_samples()

        print(f"加载数据集: {data_dir}")
        print(f"样本数量: {len(self.samples)}")
        print(f"Patch网格: {self.grid_h}x{self.grid_w}")
        print(f"总patch数: {self.num_patches}")

    def _load_samples(self):
        """加载所有样本"""
        samples = []

        # 遍历数据集目录
        for pair_dir in sorted(os.listdir(self.data_dir)):
            pair_path = os.path.join(self.data_dir, pair_dir)

            if not os.path.isdir(pair_path):
                continue

            # 检查必要的文件是否存在
            img1_path = os.path.join(pair_path, "image1.jpg")
            img2_path = os.path.join(pair_path, "image2.jpg")
            label_path = os.path.join(pair_path, "patch_labels.json")

            if all(os.path.exists(p) for p in [img1_path, img2_path, label_path]):
                try:
                    with open(label_path, 'r') as f:
                        label_info = json.load(f)

                    samples.append({
                        'img1_path': img1_path,
                        'img2_path': img2_path,
                        'label_info': label_info,
                        'pair_id': pair_dir
                    })
                except Exception as e:
                    print(f"加载样本 {pair_dir} 失败: {e}")

        return samples

    def _create_patch_labels(self, label_info):
        """从标注信息创建patch标签"""
        # 初始化全0标签
        patch_labels = np.zeros(self.num_patches, dtype=np.float32)

        # 方法1: 使用differences列表
        if 'differences' in label_info:
            for diff_info in label_info['differences']:
                patch_id = diff_info.get('patch_id')
                if patch_id is not None and 0 <= patch_id < self.num_patches:
                    patch_labels[patch_id] = 1.0

        # 方法2: 使用patch_mask矩阵
        elif 'patch_mask' in label_info:
            patch_mask = np.array(label_info['patch_mask'])
            patch_labels = patch_mask.flatten()

        # 方法3: 使用patch_position
        elif 'patch_positions' in label_info:
            for pos_info in label_info['patch_positions']:
                row = pos_info.get('row')
                col = pos_info.get('col')
                if row is not None and col is not None:
                    patch_id = row * self.grid_w + col
                    if 0 <= patch_id < self.num_patches:
                        patch_labels[patch_id] = 1.0

        return patch_labels

    def _load_and_preprocess_image(self, image_path):
        """加载和预处理图像"""
        img = Image.open(image_path).convert('RGB')

        # 调整大小
        if img.size != self.image_size:
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)

        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        img1 = self._load_and_preprocess_image(sample['img1_path'])
        img2 = self._load_and_preprocess_image(sample['img2_path'])

        # 数据增强（仅对训练集）
        if self.is_train and self.transform is not None:
            seed = torch.randint(0, 2 ** 32, (1,)).item()

            torch.manual_seed(seed)
            img1 = self.transform(img1)

            torch.manual_seed(seed)
            img2 = self.transform(img2)

        # 使用processor处理图像
        inputs1 = self.processor(images=img1, return_tensors="pt")["pixel_values"].squeeze(0)
        inputs2 = self.processor(images=img2, return_tensors="pt")["pixel_values"].squeeze(0)

        # 创建patch标签
        patch_labels = self._create_patch_labels(sample['label_info'])
        patch_labels = torch.from_numpy(patch_labels).long()

        return {
            'image1': inputs1,
            'image2': inputs2,
            'patch_labels': patch_labels,
            'pair_id': sample['pair_id'],
            'image1_path': sample['img1_path'],
            'image2_path': sample['img2_path']
        }


# ==================== 数据增强 ====================
class PatchDiffTransform:
    """专门为patch差异检测设计的数据增强"""

    def __init__(self, augment_prob=0.5):
        self.augment_prob = augment_prob
        from torchvision import transforms as T

        # 定义增强变换
        self.color_jitter = T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )

        self.random_affine = T.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        )

        self.random_horizontal_flip = T.RandomHorizontalFlip(p=0.5)
        self.random_vertical_flip = T.RandomVerticalFlip(p=0.3)

    def __call__(self, img):
        """应用数据增强"""
        import random

        # 随机应用增强
        if random.random() < self.augment_prob:
            img = self.color_jitter(img)

        if random.random() < self.augment_prob:
            img = self.random_affine(img)

        if random.random() < self.augment_prob:
            img = self.random_horizontal_flip(img)

        if random.random() < self.augment_prob * 0.6:
            img = self.random_vertical_flip(img)

        return img


# ==================== 模型架构 ====================
class PatchAttentionModule(nn.Module):
    """Patch注意力模块"""

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x):
        # x: [batch_size, num_patches, hidden_dim]
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.norm(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm(x + ffn_out)

        return x


class DINOv3PatchDiffModel(nn.Module):
    """基于DINOv3的Patch级差异检测模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 加载DINOv3骨干网络
        self.backbone = AutoModel.from_pretrained(config.model_name)
        self.hidden_size = self.backbone.config.hidden_size

        # 冻结骨干网络（可选）
        # self._freeze_backbone()

        # 特征提取层数
        self.num_layers_to_extract = 4

        # 多尺度特征融合
        self.feature_fusion = nn.ModuleDict({
            'layer1': nn.Linear(self.hidden_size * 2, self.hidden_size),
            'layer2': nn.Linear(self.hidden_size * 2, self.hidden_size),
            'layer3': nn.Linear(self.hidden_size * 2, self.hidden_size),
            'layer4': nn.Linear(self.hidden_size * 2, self.hidden_size),
        })

        # 注意力模块
        self.patch_attention = PatchAttentionModule(self.hidden_size)

        # 上下文编码器（考虑空间关系）
        self.context_encoder = nn.Sequential(
            nn.Conv2d(self.hidden_size, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Patch分类头
        self.classifier = nn.Sequential(
            nn.Linear(128 + 1, 64),  # +1 for similarity feature
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _freeze_backbone(self):
        """冻结骨干网络参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("已冻结DINOv3骨干网络")

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_patch_features(self, pixel_values):
        """提取多尺度patch特征"""
        with torch.set_grad_enabled(self.training):
            outputs = self.backbone(pixel_values, output_hidden_states=True)

            all_features = []
            # 获取不同层的特征
            layer_indices = [1, 4, 8, 12]  # 提取不同深度的特征

            for idx in layer_indices:
                if idx < len(outputs.hidden_states):
                    hidden_state = outputs.hidden_states[idx]
                    # 获取patch tokens（排除CLS token）
                    patch_tokens = hidden_state[:, 1:, :]
                    all_features.append(patch_tokens)

            return all_features

    def compute_patch_similarity(self, feats1, feats2):
        """计算patch级相似度特征"""
        # 计算余弦相似度
        similarity = F.cosine_similarity(feats1, feats2, dim=-1)

        # 计算差异特征（1 - 相似度）
        diff_feature = 1.0 - similarity

        return similarity, diff_feature

    def forward(self, pixel_values1, pixel_values2):
        """
        前向传播

        Args:
            pixel_values1: 第一张图片 [B, C, H, W]
            pixel_values2: 第二张图片 [B, C, H, W]

        Returns:
            dict: 包含预测结果和中间特征
        """
        batch_size = pixel_values1.shape[0]

        # 1. 提取多尺度特征
        feats1_list = self.extract_patch_features(pixel_values1)
        feats2_list = self.extract_patch_features(pixel_values2)

        # 2. 多尺度特征融合
        fused_features_list = []
        for i, (feats1, feats2) in enumerate(zip(feats1_list, feats2_list)):
            # 特征拼接和融合
            combined = torch.cat([feats1, feats2], dim=-1)
            fused = self.feature_fusion[f'layer{i + 1}'](combined)
            fused_features_list.append(fused)

        # 3. 多尺度特征聚合
        if len(fused_features_list) > 1:
            # 加权求和
            weights = torch.softmax(torch.randn(len(fused_features_list)), dim=0).to(pixel_values1.device)
            aggregated_features = sum(w * f for w, f in zip(weights, fused_features_list))
        else:
            aggregated_features = fused_features_list[0]

        # 4. Patch注意力
        attended_features = self.patch_attention(aggregated_features)

        # 5. 计算相似度特征
        similarity, diff_feature = self.compute_patch_similarity(
            feats1_list[-1], feats2_list[-1]
        )

        # 6. 空间上下文编码
        num_patches = attended_features.shape[1]
        grid_size = int(np.sqrt(num_patches))

        spatial_features = attended_features.permute(0, 2, 1).reshape(
            batch_size, -1, grid_size, grid_size
        )
        contextual_features = self.context_encoder(spatial_features)

        # 7. 准备分类特征
        contextual_features_flat = contextual_features.flatten(2).permute(0, 2, 1)

        # 结合相似度特征
        combined_features = torch.cat([
            contextual_features_flat,
            diff_feature.unsqueeze(-1)
        ], dim=-1)

        # 8. Patch分类
        patch_logits = self.classifier(combined_features)

        return {
            'patch_logits': patch_logits,  # [B, num_patches, num_classes]
            'patch_similarity': similarity,  # [B, num_patches]
            'diff_feature': diff_feature,  # [B, num_patches]
            'attended_features': attended_features,  # [B, num_patches, hidden_size]
            'contextual_features': contextual_features,  # [B, 128, grid_size, grid_size]
            'grid_size': grid_size
        }


# ==================== 损失函数 ====================
class PatchDiffLoss(nn.Module):
    """Patch级差异检测的复合损失函数"""

    def __init__(self, pos_weight=2.0, focal_gamma=2.0, dice_weight=0.3):
        super().__init__()
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight

        # 基础交叉熵损失（带类别权重）
        weight = torch.tensor([1.0, pos_weight])
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def focal_loss(self, logits, targets):
        """焦点损失：关注困难样本"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.focal_gamma
        return (focal_weight * ce_loss).mean()

    def dice_loss(self, pred_logits, targets):
        """Dice损失：改善类别不平衡"""
        pred_probs = F.softmax(pred_logits, dim=-1)[:, :, 1]  # 正类概率
        pred_probs = pred_probs.flatten()
        targets = targets.flatten().float()

        intersection = (pred_probs * targets).sum()
        dice_score = (2. * intersection + 1e-6) / (pred_probs.sum() + targets.sum() + 1e-6)
        return 1.0 - dice_score

    def consistency_loss(self, patch_sim, patch_logits):
        """一致性损失：相似度低的patch应该有更高的差异概率"""
        diff_probs = F.softmax(patch_logits, dim=-1)[:, :, 1]

        # 相似度和差异概率应该负相关
        correlation = torch.corrcoef(torch.stack([
            patch_sim.flatten(),
            diff_probs.flatten()
        ]))[0, 1]

        # 我们希望correlation接近-1
        consistency = (correlation + 1.0) ** 2
        return consistency

    def forward(self, outputs, targets):
        """
        计算总损失

        Args:
            outputs: 模型输出字典
            targets: patch标签 [B, num_patches]
        """
        patch_logits = outputs['patch_logits']
        patch_similarity = outputs['patch_similarity']

        batch_size, num_patches, _ = patch_logits.shape

        # 重塑为2D
        logits_flat = patch_logits.view(-1, 2)
        targets_flat = targets.view(-1)

        # 1. 交叉熵损失
        ce_loss = self.ce_loss(logits_flat, targets_flat)

        # 2. 焦点损失
        focal_loss = self.focal_loss(logits_flat, targets_flat)

        # 3. Dice损失
        dice_loss = self.dice_loss(patch_logits, targets)

        # 4. 一致性损失（可选）
        consistency_loss = self.consistency_loss(patch_similarity, patch_logits)

        # 5. 组合损失
        total_loss = (
                ce_loss +
                0.5 * focal_loss +
                self.dice_weight * dice_loss +
                0.1 * consistency_loss
        )

        return {
            'total': total_loss,
            'ce': ce_loss,
            'focal': focal_loss,
            'dice': dice_loss,
            'consistency': consistency_loss
        }


# ==================== 训练器类 ====================
class PatchDiffTrainer:
    """训练器类"""

    def __init__(self, config, model, train_loader, val_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = config.device
        self.model.to(self.device)

        # 损失函数
        self.criterion = PatchDiffLoss(
            pos_weight=config.pos_weight,
            focal_gamma=config.focal_gamma,
            dice_weight=config.dice_weight
        )

        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_lr
        )

        # Early stopping
        self.patience = config.patience
        self.best_val_loss = float('inf')
        self.counter = 0

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }

        # 创建实验目录
        config.setup_paths()

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for batch in pbar:
            # 移动到设备
            image1 = batch['image1'].to(self.device)
            image2 = batch['image2'].to(self.device)
            patch_labels = batch['patch_labels'].to(self.device)

            # 前向传播
            outputs = self.model(image1, image2)
            loss_dict = self.criterion(outputs, patch_labels)
            loss = loss_dict['total']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 统计
            total_loss += loss.item()

            # 收集预测用于计算指标
            with torch.no_grad():
                pred_probs = F.softmax(outputs['patch_logits'], dim=-1)[:, :, 1]
                predictions = (pred_probs > 0.5).long()

                all_preds.append(predictions.cpu())
                all_targets.append(patch_labels.cpu())

            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'ce': loss_dict['ce'].item(),
                'focal': loss_dict['focal'].item()
            })

        # 计算平均损失和指标
        avg_loss = total_loss / len(self.train_loader)

        # 计算指标
        all_preds = torch.cat(all_preds).flatten().numpy()
        all_targets = torch.cat(all_targets).flatten().numpy()

        metrics = self.compute_metrics(all_preds, all_targets)

        return avg_loss, metrics

    @torch.no_grad()
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]")
        for batch in pbar:
            # 移动到设备
            image1 = batch['image1'].to(self.device)
            image2 = batch['image2'].to(self.device)
            patch_labels = batch['patch_labels'].to(self.device)

            # 前向传播
            outputs = self.model(image1, image2)
            loss_dict = self.criterion(outputs, patch_labels)
            loss = loss_dict['total']

            # 统计
            total_loss += loss.item()

            # 收集预测
            pred_probs = F.softmax(outputs['patch_logits'], dim=-1)[:, :, 1]
            predictions = (pred_probs > 0.5).long()

            all_preds.append(predictions.cpu())
            all_targets.append(patch_labels.cpu())

            pbar.set_postfix({'loss': loss.item()})

        # 计算平均损失和指标
        avg_loss = total_loss / len(self.val_loader)

        # 计算指标
        all_preds = torch.cat(all_preds).flatten().numpy()
        all_targets = torch.cat(all_targets).flatten().numpy()

        metrics = self.compute_metrics(all_preds, all_targets)

        return avg_loss, metrics

    def compute_metrics(self, predictions, targets):
        """计算评估指标"""
        # 基础指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='binary', zero_division=0
        )

        # 准确率
        accuracy = (predictions == targets).mean()

        # IoU (Jaccard Index)
        intersection = ((predictions == 1) & (targets == 1)).sum()
        union = ((predictions == 1) | (targets == 1)).sum()
        iou = intersection / (union + 1e-6)

        # 类别统计
        pos_ratio = targets.mean()
        pred_pos_ratio = predictions.mean()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'pos_ratio': pos_ratio,
            'pred_pos_ratio': pred_pos_ratio,
            'num_samples': len(targets)
        }

    def save_checkpoint(self, epoch, val_loss, val_metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'history': self.history,
            'config': self.config.__dict__
        }

        # 保存常规检查点
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{epoch + 1:03d}.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config.model_save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到: {best_path}")

    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.config.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def plot_training_history(self):
        """绘制训练历史图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # 损失曲线
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # F1分数
        train_f1 = [m['f1'] for m in self.history['train_metrics']]
        val_f1 = [m['f1'] for m in self.history['val_metrics']]
        axes[0, 1].plot(epochs, train_f1, label='Train F1')
        axes[0, 1].plot(epochs, val_f1, label='Val F1')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 准确率
        train_acc = [m['accuracy'] for m in self.history['train_metrics']]
        val_acc = [m['accuracy'] for m in self.history['val_metrics']]
        axes[0, 2].plot(epochs, train_acc, label='Train Accuracy')
        axes[0, 2].plot(epochs, val_acc, label='Val Accuracy')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # IoU
        train_iou = [m['iou'] for m in self.history['train_metrics']]
        val_iou = [m['iou'] for m in self.history['val_metrics']]
        axes[1, 0].plot(epochs, train_iou, label='Train IoU')
        axes[1, 0].plot(epochs, val_iou, label='Val IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].set_title('IoU (Jaccard Index)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 正样本比例
        train_pos = [m['pos_ratio'] for m in self.history['train_metrics']]
        train_pred_pos = [m['pred_pos_ratio'] for m in self.history['train_metrics']]
        axes[1, 1].plot(epochs, train_pos, label='Actual Positive Ratio')
        axes[1, 1].plot(epochs, train_pred_pos, label='Predicted Positive Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Positive Ratio')
        axes[1, 1].set_title('Positive Ratio Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 学习率
        if self.history['learning_rates']:
            axes[1, 2].plot(epochs, self.history['learning_rates'])
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_title('Learning Rate Schedule')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, 'training_history.png'), dpi=150)
        plt.show()

    def train(self):
        """主训练循环"""
        print(f"开始训练，共 {self.config.num_epochs} 个epoch")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print(f"设备: {self.device}")

        for epoch in range(self.config.num_epochs):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'=' * 60}")

            # 训练
            train_loss, train_metrics = self.train_epoch(epoch)

            # 验证
            val_loss, val_metrics = self.validate_epoch(epoch)

            # 学习率调整
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # 更新历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(current_lr)

            # 打印结果
            print(f"\n训练结果:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  学习率: {current_lr:.6f}")

            print(f"\n训练指标:")
            print(f"  准确率: {train_metrics['accuracy']:.4f}")
            print(f"  F1分数: {train_metrics['f1']:.4f}")
            print(f"  Precision: {train_metrics['precision']:.4f}")
            print(f"  Recall: {train_metrics['recall']:.4f}")
            print(f"  IoU: {train_metrics['iou']:.4f}")

            print(f"\n验证指标:")
            print(f"  准确率: {val_metrics['accuracy']:.4f}")
            print(f"  F1分数: {val_metrics['f1']:.4f}")
            print(f"  Precision: {val_metrics['precision']:.4f}")
            print(f"  Recall: {val_metrics['recall']:.4f}")
            print(f"  IoU: {val_metrics['iou']:.4f}")

            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1

            self.save_checkpoint(epoch, val_loss, val_metrics, is_best)

            # Early stopping
            if self.counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # 保存最终模型
        final_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config.__dict__
        }
        final_path = os.path.join(self.config.model_save_dir, 'final_model.pth')
        torch.save(final_checkpoint, final_path)

        # 保存历史和绘图
        self.save_history()
        self.plot_training_history()

        print(f"\n训练完成！最佳验证损失: {self.best_val_loss:.4f}")
        print(f"模型保存在: {self.config.model_save_dir}")


# ==================== 推理类 ====================
class PatchDiffPredictor:
    """推理类"""

    def __init__(self, model_path, device=None):
        # 加载配置和模型
        checkpoint = torch.load(model_path, map_location='cpu')
        config_dict = checkpoint['config']

        # 创建配置对象
        self.config = Config()
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # 创建模型
        self.model = DINOv3PatchDiffModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 加载图像处理器
        self.processor = AutoImageProcessor.from_pretrained(self.config.model_name)

        print(f"加载模型: {model_path}")
        print(f"使用设备: {self.device}")
        print(f"图像尺寸: {self.config.image_size}")
        print(f"Patch大小: {self.config.patch_size}")

    def preprocess_images(self, img1_path, img2_path):
        """预处理图像"""
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # 记录原始尺寸
        self.original_size = img1.size

        # 调整到模型输入尺寸
        img1_resized = img1.resize(self.config.image_size, Image.Resampling.LANCZOS)
        img2_resized = img2.resize(self.config.image_size, Image.Resampling.LANCZOS)

        # 预处理
        inputs1 = self.processor(img1_resized, return_tensors="pt")["pixel_values"]
        inputs2 = self.processor(img2_resized, return_tensors="pt")["pixel_values"]

        return {
            'image1': img1,
            'image2': img2,
            'image1_resized': img1_resized,
            'image2_resized': img2_resized,
            'inputs1': inputs1,
            'inputs2': inputs2
        }

    @torch.no_grad()
    def predict(self, img1_path, img2_path, threshold=0.5):
        """预测差异"""
        # 预处理
        preprocessed = self.preprocess_images(img1_path, img2_path)

        # 移动到设备
        inputs1 = preprocessed['inputs1'].to(self.device)
        inputs2 = preprocessed['inputs2'].to(self.device)

        # 前向传播
        outputs = self.model(inputs1, inputs2)

        # 获取预测结果
        patch_logits = outputs['patch_logits']
        patch_probs = F.softmax(patch_logits, dim=-1)[0, :, 1].cpu().numpy()  # 差异概率
        patch_pred = (patch_probs > threshold).astype(np.uint8)

        # 获取相似度
        patch_sim = outputs['patch_similarity'][0].cpu().numpy()

        # 网格信息
        grid_size = outputs['grid_size']
        patch_grid = patch_pred.reshape(grid_size, grid_size)
        prob_grid = patch_probs.reshape(grid_size, grid_size)
        sim_grid = patch_sim.reshape(grid_size, grid_size)

        # 计算统计信息
        num_diff_patches = patch_pred.sum()
        total_patches = len(patch_pred)
        diff_percentage = num_diff_patches / total_patches * 100

        # 创建可视化
        visualization = self.create_visualization(
            preprocessed['image1'],
            preprocessed['image2'],
            patch_grid,
            prob_grid,
            sim_grid,
            threshold
        )

        return {
            'patch_predictions': patch_pred,
            'patch_probabilities': patch_probs,
            'patch_similarity': patch_sim,
            'patch_grid': patch_grid,
            'prob_grid': prob_grid,
            'sim_grid': sim_grid,
            'grid_size': grid_size,
            'num_diff_patches': num_diff_patches,
            'total_patches': total_patches,
            'diff_percentage': diff_percentage,
            'visualization': visualization,
            'original_size': self.original_size,
            'processed_size': self.config.image_size
        }

    def create_visualization(self, img1, img2, patch_grid, prob_grid, sim_grid, threshold):
        """创建可视化结果"""
        img1_np = np.array(img1)
        img2_np = np.array(img2)

        grid_h, grid_w = patch_grid.shape

        # 计算patch在原始图像中的位置
        orig_h, orig_w = img1_np.shape[:2]
        patch_h = orig_h // grid_h
        patch_w = orig_w // grid_w

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. 原始图像对比
        axes[0, 0].imshow(img1_np)
        axes[0, 0].set_title("Image 1 (Original)")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(img2_np)
        axes[0, 1].set_title("Image 2 (Original)")
        axes[0, 1].axis('off')

        # 2. Patch预测网格
        im1 = axes[0, 2].imshow(patch_grid, cmap='binary', vmin=0, vmax=1)
        axes[0, 2].set_title(f"Patch Predictions\n(Threshold={threshold})")
        axes[0, 2].set_xlabel("White: Different, Black: Same")
        plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # 3. Patch概率热图
        im2 = axes[1, 0].imshow(prob_grid, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[1, 0].set_title("Patch Difference Probabilities")
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 4. Patch相似度热图
        im3 = axes[1, 1].imshow(sim_grid, cmap='RdYlBu', vmin=0, vmax=1)
        axes[1, 1].set_title("Patch Similarities")
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # 5. 差异区域叠加（图片1）
        axes[1, 2].imshow(img1_np)
        # 绘制差异patch
        for i in range(grid_h):
            for j in range(grid_w):
                if patch_grid[i, j] == 1:
                    rect = mpatches.Rectangle(
                        (j * patch_w, i * patch_h),
                        patch_w, patch_h,
                        linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
                    )
                    axes[1, 2].add_patch(rect)
        axes[1, 2].set_title("Difference Regions (Red boxes)")
        axes[1, 2].axis('off')

        plt.suptitle("Patch-Level Difference Detection Results", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    def save_results(self, results, output_dir):
        """保存预测结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存可视化图像
        if results['visualization'] is not None:
            vis_path = os.path.join(output_dir, "difference_visualization.png")
            results['visualization'].savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close(results['visualization'])
            print(f"可视化图像保存到: {vis_path}")

        # 保存预测数据
        data_path = os.path.join(output_dir, "patch_predictions.npz")
        np.savez_compressed(
            data_path,
            patch_predictions=results['patch_predictions'],
            patch_probabilities=results['patch_probabilities'],
            patch_similarity=results['patch_similarity'],
            patch_grid=results['patch_grid'],
            prob_grid=results['prob_grid'],
            sim_grid=results['sim_grid']
        )
        print(f"预测数据保存到: {data_path}")

        # 保存统计信息
        stats_path = os.path.join(output_dir, "statistics.txt")
        with open(stats_path, 'w') as f:
            f.write("Patch Difference Detection Results\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Grid Size: {results['grid_size']}x{results['grid_size']}\n")
            f.write(f"Total Patches: {results['total_patches']}\n")
            f.write(f"Different Patches: {results['num_diff_patches']}\n")
            f.write(f"Difference Percentage: {results['diff_percentage']:.2f}%\n\n")

            f.write(f"Original Image Size: {results['original_size']}\n")
            f.write(f"Processed Image Size: {results['processed_size']}\n\n")

            f.write("Patch Statistics:\n")
            f.write(f"  Min Probability: {results['patch_probabilities'].min():.4f}\n")
            f.write(f"  Max Probability: {results['patch_probabilities'].max():.4f}\n")
            f.write(f"  Mean Probability: {results['patch_probabilities'].mean():.4f}\n")
            f.write(f"  Min Similarity: {results['patch_similarity'].min():.4f}\n")
            f.write(f"  Max Similarity: {results['patch_similarity'].max():.4f}\n")
            f.write(f"  Mean Similarity: {results['patch_similarity'].mean():.4f}\n")

        print(f"统计信息保存到: {stats_path}")


# ==================== 数据标注工具 ====================
class PatchAnnotator:
    """交互式patch标注工具"""

    def __init__(self, patch_size=16):
        self.patch_size = patch_size
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.patch_mask = None
        self.img1 = None
        self.img2 = None
        self.grid_h = None
        self.grid_w = None

    def annotate(self, img1_path, img2_path, output_dir):
        """交互式标注"""
        # 加载图像
        self.img1 = cv2.imread(img1_path)
        self.img2 = cv2.imread(img2_path)
        self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)

        # 检查图像尺寸是否一致
        if self.img1.shape != self.img2.shape:
            print("警告：图像尺寸不一致，将调整第二张图像")
            self.img2 = cv2.resize(self.img2, (self.img1.shape[1], self.img1.shape[0]))

        h, w = self.img1.shape[:2]
        self.grid_h = h // self.patch_size
        self.grid_w = w // self.patch_size

        # 初始化patch掩码
        self.patch_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)

        # 创建图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 8))

        # 显示图像
        self.ax1.imshow(self.img1)
        self.ax1.set_title("Image 1 - Click to mark differences")
        self.ax2.imshow(self.img2)
        self.ax2.set_title("Image 2")

        # 绘制网格
        self._draw_grid()

        # 显示当前标注
        self._update_display()

        # 绑定点击事件
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        # 添加指令
        print("\n标注指南:")
        print("  1. 在Image 1上点击patch来标记/取消标记差异")
        print("  2. 按 's' 键保存标注")
        print("  3. 按 'q' 键退出")

        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.tight_layout()
        plt.show()

        # 保存标注
        if output_dir:
            self._save_annotation(output_dir, img1_path, img2_path)

    def _draw_grid(self):
        """绘制patch网格"""
        for i in range(self.grid_h + 1):
            y = i * self.patch_size
            self.ax1.axhline(y=y, color='yellow', alpha=0.3, linewidth=0.5)
            self.ax2.axhline(y=y, color='yellow', alpha=0.3, linewidth=0.5)

        for j in range(self.grid_w + 1):
            x = j * self.patch_size
            self.ax1.axvline(x=x, color='yellow', alpha=0.3, linewidth=0.5)
            self.ax2.axvline(x=x, color='yellow', alpha=0.3, linewidth=0.5)

    def _update_display(self):
        """更新显示"""
        # 清除之前的矩形
        for patch in self.ax1.patches + self.ax2.patches:
            patch.remove()

        # 绘制当前标注
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                if self.patch_mask[i, j] == 1:
                    rect1 = mpatches.Rectangle(
                        (j * self.patch_size, i * self.patch_size),
                        self.patch_size, self.patch_size,
                        linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
                    )
                    rect2 = mpatches.Rectangle(
                        (j * self.patch_size, i * self.patch_size),
                        self.patch_size, self.patch_size,
                        linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
                    )
                    self.ax1.add_patch(rect1)
                    self.ax2.add_patch(rect2)

        self.fig.canvas.draw()

    def _on_click(self, event):
        """处理点击事件"""
        if event.inaxes == self.ax1:
            # 计算点击的patch坐标
            patch_x = int(event.xdata // self.patch_size)
            patch_y = int(event.ydata // self.patch_size)

            # 确保坐标在范围内
            if 0 <= patch_x < self.grid_w and 0 <= patch_y < self.grid_h:
                # 切换标注状态
                self.patch_mask[patch_y, patch_x] = 1 - self.patch_mask[patch_y, patch_x]

                # 更新显示
                self._update_display()

                print(f"标记patch ({patch_y}, {patch_x}) - 状态: {self.patch_mask[patch_y, patch_x]}")

    def _on_key(self, event):
        """处理键盘事件"""
        if event.key == 's':
            # 保存标注
            print("保存标注...")
            # 这里可以添加保存逻辑
        elif event.key == 'q':
            # 退出
            print("退出标注")
            plt.close(self.fig)

    def _save_annotation(self, output_dir, img1_path, img2_path):
        """保存标注"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存图像
        cv2.imwrite(f"{output_dir}/image1.jpg", cv2.cvtColor(self.img1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/image2.jpg", cv2.cvtColor(self.img2, cv2.COLOR_RGB2BGR))

        # 创建标注信息
        annotation = {
            "image_size": [self.img1.shape[1], self.img1.shape[0]],
            "patch_size": self.patch_size,
            "grid_shape": [self.grid_h, self.grid_w],
            "patch_mask": self.patch_mask.tolist(),
            "differences": []
        }

        # 添加具体的差异信息
        diff_count = 0
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                if self.patch_mask[i, j] == 1:
                    patch_id = i * self.grid_w + j
                    annotation["differences"].append({
                        "patch_id": patch_id,
                        "patch_position": [i, j],
                        "coordinates": [
                            j * self.patch_size,
                            i * self.patch_size,
                            (j + 1) * self.patch_size,
                            (i + 1) * self.patch_size
                        ],
                        "confidence": 1.0
                    })
                    diff_count += 1

        # 保存标注文件
        with open(f"{output_dir}/patch_labels.json", "w") as f:
            json.dump(annotation, f, indent=2)

        print(f"\n标注完成！")
        print(f"保存到: {output_dir}")
        print(f"标注了 {diff_count} 个差异patch")

        return annotation


# ==================== 主程序 ====================
def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_data_loaders(config):
    """创建数据加载器"""
    # 加载图像处理器
    processor = AutoImageProcessor.from_pretrained(config.model_name)

    # 创建完整数据集
    full_dataset = PatchDiffDataset(
        data_dir=config.data_root,
        processor=processor,
        image_size=config.image_size,
        patch_size=config.patch_size,
        is_train=True,
        transform=PatchDiffTransform()  # 数据增强
    )

    # 划分数据集
    dataset_size = len(full_dataset)
    train_size = int(config.train_split * dataset_size)
    val_size = int(config.val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # 更新数据集的is_train属性
    for dataset in [val_dataset, test_dataset]:
        for idx in range(len(dataset)):
            dataset.dataset.is_train = False

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    return train_loader, val_loader, test_loader


def train_model():
    """训练模型"""
    print("=" * 60)
    print("DINOv3 Patch-Level Difference Detection")
    print("=" * 60)

    # 设置配置
    config = Config()

    # 设置随机种子
    setup_seed(config.seed)

    # 创建数据加载器
    train_loader, val_loader, _ = create_data_loaders(config)

    # 创建模型
    model = DINOv3PatchDiffModel(config)

    # 创建训练器
    trainer = PatchDiffTrainer(config, model, train_loader, val_loader)

    # 开始训练
    trainer.train()

    return trainer


def test_model(model_path):
    """测试模型"""
    print("=" * 60)
    print("模型测试")
    print("=" * 60)

    # 创建预测器
    predictor = PatchDiffPredictor(model_path)

    # 测试图片对
    test_pairs = [
        ("/path/to/test/image1.jpg", "/path/to/test/image2.jpg"),
        # 添加更多测试对
    ]

    for img1_path, img2_path in test_pairs:
        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
            print(f"跳过不存在的图片对: {img1_path}, {img2_path}")
            continue

        print(f"\n分析图片对:")
        print(f"  图片1: {img1_path}")
        print(f"  图片2: {img2_path}")

        # 预测
        results = predictor.predict(img1_path, img2_path, threshold=0.5)

        # 打印结果
        print(f"\n分析结果:")
        print(f"  网格大小: {results['grid_size']}x{results['grid_size']}")
        print(f"  总patch数: {results['total_patches']}")
        print(f"  差异patch数: {results['num_diff_patches']}")
        print(f"  差异百分比: {results['diff_percentage']:.2f}%")
        print(f"  平均相似度: {results['patch_similarity'].mean():.4f}")

        # 保存结果
        output_dir = f"./results/{os.path.basename(img1_path).split('.')[0]}_vs_{os.path.basename(img2_path).split('.')[0]}"
        predictor.save_results(results, output_dir)

        # 显示可视化
        plt.show()

    print("\n测试完成！")


def create_annotation():
    """创建标注"""
    print("=" * 60)
    print("交互式标注工具")
    print("=" * 60)

    # 图片路径
    img1_path = input("输入第一张图片路径: ").strip()
    img2_path = input("输入第二张图片路径: ").strip()
    output_dir = input("输入输出目录: ").strip()

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("错误：图片文件不存在")
        return

    # 创建标注器
    annotator = PatchAnnotator(patch_size=16)
    annotator.annotate(img1_path, img2_path, output_dir)


def main():
    """主函数"""
    print("DINOv3 Patch-Level Difference Detection System")
    print("=" * 60)
    print("选择操作:")
    print("  1. 训练模型")
    print("  2. 测试模型")
    print("  3. 创建标注")
    print("  4. 使用预训练模型进行推理")
    print("=" * 60)

    choice = input("请输入选择 (1-4): ").strip()

    if choice == "1":
        train_model()
    elif choice == "2":
        model_path = input("输入模型路径: ").strip()
        if os.path.exists(model_path):
            test_model(model_path)
        else:
            print("错误：模型文件不存在")
    elif choice == "3":
        create_annotation()
    elif choice == "4":
        # 使用预训练模型进行推理
        model_path = input("输入模型路径: ").strip()
        img1_path = input("输入第一张图片路径: ").strip()
        img2_path = input("输入第二张图片路径: ").strip()

        if not os.path.exists(model_path):
            print("错误：模型文件不存在")
            return

        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
            print("错误：图片文件不存在")
            return

        predictor = PatchDiffPredictor(model_path)
        results = predictor.predict(img1_path, img2_path)

        # 显示结果
        plt.show()

        # 询问是否保存
        save = input("是否保存结果? (y/n): ").strip().lower()
        if save == 'y':
            output_dir = input("输入保存目录: ").strip()
            predictor.save_results(results, output_dir)
    else:
        print("无效选择")


if __name__ == "__main__":
    main()