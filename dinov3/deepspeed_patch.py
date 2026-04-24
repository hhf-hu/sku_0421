"""
DINOv3 Patch-Level Difference Detection with DeepSpeed
完整实现：以patch为单位检测两张图片的差异区域，支持分布式训练
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
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import deepspeed
from transformers import get_scheduler


# ==================== 配置部分 ====================
class DeepSpeedConfig:
    """DeepSpeed配置类"""

    def __init__(self):
        # 模型配置
        self.model_name = "/data1/vincent/models/facebook-dinov3-vith16plus-pretrain-lvd1689m"
        self.patch_size = 16
        self.num_classes = 2

        # 数据配置
        self.data_root = "/data1/vincent/sku/dinov3/patch_data"
        self.image_size = (224, 224)
        self.train_split = 0.9
        self.val_split = 0.1
        self.test_split = 0

        # 训练配置
        self.batch_size_per_gpu = 2  # 每个GPU的批处理大小
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.min_lr = 1e-6

        # 损失函数配置
        self.pos_weight = 3.0
        self.focal_gamma = 2.0
        self.dice_weight = 0.3
        self.contrastive_weight = 0.1  # 对比损失权重
        # 分布式配置（自动设置）
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # DeepSpeed配置
        self.deepspeed_config = {
            "train_batch_size": self.batch_size_per_gpu * self.world_size,  # 总批处理大小
            "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,  # 每个GPU的批处理大小
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": 3,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
            },
            "bf16": {
                "enabled": True,

            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": self.weight_decay
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.learning_rate,
                    "warmup_num_steps": 5,  # 这里可以保持auto
                    # "total_num_steps": "auto"     # 这里可以保持auto
                }
            },
            "gradient_clipping": 1.0,
            "steps_per_print": 10,
            "wall_clock_breakdown": False
        }

        # 检查点配置
        self.checkpoint_dir = "./deepspeed_checkpoints"
        self.save_interval = 1000  # 保存间隔步数

        # 其他配置
        self.num_workers = 4
        self.seed = 42



    def setup_paths(self):
        """创建必要的目录"""
        if dist.get_rank() == 0:  # 只在主进程中创建目录
            self.experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.model_save_dir = os.path.join(self.checkpoint_dir, self.experiment_name)
            self.log_dir = os.path.join(self.model_save_dir, "logs")

            for path in [self.model_save_dir, self.log_dir]:
                os.makedirs(path, exist_ok=True)

            # 保存配置
            config_path = os.path.join(self.model_save_dir, "config.json")
            with open(config_path, 'w') as f:
                config_dict = {k: v for k, v in self.__dict__.items()
                               if not k.startswith('_') and not callable(v)}
                json.dump(config_dict, f, indent=2, default=str)

            print(f"实验目录: {self.model_save_dir}")

        return self


# ==================== 模型架构（添加对比学习头）====================
class DINOv3PatchDiffModelWithContrastive(nn.Module):
    """基于DINOv3的Patch级差异检测模型，支持对比学习"""

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

        # Patch注意力模块
        self.patch_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=4,
            batch_first=True
        )

        # 上下文编码器
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

        # 对比学习投影头
        self.contrastive_projection = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128)  # 对比特征维度
        )
        # 添加适配层，将200个patches转换为196个
        self.patch_adapter = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        # 初始化权重
        self._init_weights()

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
        outputs = self.backbone(pixel_values, output_hidden_states=True)

        all_features = []
        layer_indices = [1, 4, 8, 12]

        for idx in layer_indices:
            if idx < len(outputs.hidden_states):
                hidden_state = outputs.hidden_states[idx]
                patch_tokens = hidden_state[:, 1:, :]  # 200个patches

                # 固定patch数量为196 (14*14)
                num_patches = 196
                if patch_tokens.shape[1] > num_patches:
                    # 如果原始patch数大于196，截取前196个
                    adjusted_tokens = patch_tokens[:, :num_patches, :]
                else:
                    # 如果不够，可能需要插值或填充
                    adjusted_tokens = patch_tokens

                all_features.append(adjusted_tokens)

        return all_features

    def forward(self, pixel_values1, pixel_values2):
        """前向传播"""
        batch_size = pixel_values1.shape[0]

        # 1. 提取多尺度特征
        feats1_list = self.extract_patch_features(pixel_values1)
        feats2_list = self.extract_patch_features(pixel_values2)

        # 2. 多尺度特征融合
        fused_features_list = []
        for i, (feats1, feats2) in enumerate(zip(feats1_list, feats2_list)):
            combined = torch.cat([feats1, feats2], dim=-1)
            fused = self.feature_fusion[f'layer{i + 1}'](combined)
            fused_features_list.append(fused)

        # 3. 多尺度特征聚合
        if len(fused_features_list) > 1:
            weights = torch.softmax(torch.randn(len(fused_features_list)), dim=0).to(pixel_values1.device)
            aggregated_features = sum(w * f for w, f in zip(weights, fused_features_list))
        else:
            aggregated_features = fused_features_list[0]

        # 4. Patch注意力
        attended_features, _ = self.patch_attention(aggregated_features, aggregated_features, aggregated_features)

        # 5. 计算相似度特征
        similarity = F.cosine_similarity(feats1_list[-1], feats2_list[-1], dim=-1)
        diff_feature = 1.0 - similarity

        # 6. 根据实际patch数量计算网格
        num_patches = attended_features.shape[1]  # 200
        hidden_dim = attended_features.shape[2]  # 1280

        # 找到最接近平方数的网格
        # 对于200个patches，可能的网格形状：
        # 10×20 = 200 (10行，20列)
        # 8×25 = 200
        # 5×40 = 200
        # 4×50 = 200
        # 2×100 = 200

        # 选择最接近正方形的网格
        # 计算所有可能的因数对
        factors = []
        for i in range(1, int(np.sqrt(num_patches)) + 1):
            if num_patches % i == 0:
                factors.append((i, num_patches // i))

        # 选择最接近正方形的一对
        if factors:
            # 选择高宽比最接近1的
            best_ratio = float('inf')
            best_grid = factors[0]
            for h, w in factors:
                ratio = max(h / w, w / h)  # 高宽比，越接近1越好
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_grid = (h, w)

            grid_h, grid_w = best_grid
        else:
            # 如果没有找到整数因数，使用近似值
            grid_h = int(np.sqrt(num_patches))
            grid_w = int(np.ceil(num_patches / grid_h))

        # print(f"实际patch数量: {num_patches}, 使用网格: {grid_h}x{grid_w}")

        # 7. 空间上下文编码
        # 确保能够正确reshape
        spatial_features = attended_features.permute(0, 2, 1).reshape(
            batch_size, hidden_dim, grid_h, grid_w
        )

        contextual_features = self.context_encoder(spatial_features)

        # 8. 准备分类特征
        contextual_features_flat = contextual_features.flatten(2).permute(0, 2, 1)

        # 结合相似度特征
        combined_features = torch.cat([
            contextual_features_flat,
            diff_feature.unsqueeze(-1)
        ], dim=-1)

        # 9. Patch分类
        patch_logits = self.classifier(combined_features)

        # 10. 对比学习特征
        contrastive_features = torch.cat([feats1_list[-1], feats2_list[-1]], dim=-1)
        contrastive_projected = self.contrastive_projection(contrastive_features)
        contrastive_projected = F.normalize(contrastive_projected, dim=-1)

        return {
            'patch_logits': patch_logits,
            'patch_similarity': similarity,
            'diff_feature': diff_feature,
            'contrastive_features': contrastive_projected,
            'temperature': self.temperature.clamp(min=0.01, max=1.0),
            'grid_h': grid_h,
            'grid_w': grid_w
        }


# ==================== 对比学习损失函数 ====================
def supervised_contrastive_loss(projected_features, patch_labels, temperature=0.07):
    """
    监督对比损失：相同patch状态的样本作为正样本
    Args:
        projected_features: 投影后的特征 [batch_size * num_patches, feature_dim]
        patch_labels: patch标签 [batch_size * num_patches]
        temperature: 温度参数
    """
    batch_size = projected_features.size(0)
    device = projected_features.device

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(projected_features, projected_features.T)
    similarity_matrix = similarity_matrix / temperature

    # 创建正样本掩码（相同标签的样本是正样本）
    labels = patch_labels.unsqueeze(0)
    positive_mask = (labels == labels.T).float()
    positive_mask.fill_diagonal_(0)  # 排除自身

    # 创建目标概率分布
    exp_similarity = torch.exp(similarity_matrix)

    # 计算分母：所有样本的exp相似度之和
    sum_exp = torch.sum(exp_similarity, dim=1, keepdim=True)

    # 计算概率
    probabilities = exp_similarity / sum_exp

    # 计算损失：-log(正样本的概率加权和)
    positive_probs = probabilities * positive_mask
    sum_positive_probs = torch.sum(positive_probs, dim=1)

    # 避免log(0)和没有正样本的情况
    sum_positive_probs = torch.clamp(sum_positive_probs, min=1e-8)

    loss = -torch.log(sum_positive_probs).mean()

    return loss


# ==================== 复合损失函数 ====================
class PatchDiffContrastiveLoss(nn.Module):
    """Patch级差异检测的复合损失函数（含对比学习）"""

    def __init__(self, pos_weight=2.0, focal_gamma=2.0, dice_weight=0.3, contrastive_weight=0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.contrastive_weight = contrastive_weight

        # 基础交叉熵损失 - 不在这里创建权重张量
        # 在forward方法中动态创建
        self.ce_loss = None
        self.weight = None

    def _ensure_weight_on_device(self, device):
        """确保权重在正确的设备上"""
        if self.weight is None or self.weight.device != device:
            self.weight = torch.tensor([1.0, self.pos_weight], device=device)
            self.ce_loss = nn.CrossEntropyLoss(weight=self.weight)

    def focal_loss(self, logits, targets):
        """焦点损失"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.focal_gamma
        return (focal_weight * ce_loss).mean()

    def dice_loss(self, pred_logits, targets):
        """Dice损失"""
        pred_probs = F.softmax(pred_logits, dim=-1)[:, :, 1]
        pred_probs = pred_probs.flatten()
        targets = targets.flatten().float()

        intersection = (pred_probs * targets).sum()
        dice_score = (2. * intersection + 1e-6) / (pred_probs.sum() + targets.sum() + 1e-6)
        return 1.0 - dice_score

    def forward(self, outputs, targets):
        """计算总损失"""
        patch_logits = outputs['patch_logits']
        contrastive_features = outputs['contrastive_features']
        temperature = outputs['temperature']

        batch_size, num_patches, _ = patch_logits.shape

        # 重塑为2D
        logits_flat = patch_logits.view(-1, 2)
        targets_flat = targets.view(-1)

        # 1. 交叉熵损失 - 动态创建权重张量，确保与输入相同的数据类型
        weight = torch.tensor([1.0, self.pos_weight],
                              device=logits_flat.device,
                              dtype=logits_flat.dtype)  # 添加 dtype 参数
        ce_loss = F.cross_entropy(logits_flat, targets_flat, weight=weight)

        # 2. 焦点损失
        focal_loss = self.focal_loss(logits_flat, targets_flat)

        # 3. Dice损失
        dice_loss = self.dice_loss(patch_logits, targets)

        # 4. 对比学习损失
        contrastive_loss = supervised_contrastive_loss(
            contrastive_features.view(-1, contrastive_features.size(-1)),
            targets_flat,
            temperature
        )

        # 5. 组合损失
        total_loss = (
                ce_loss +
                0.5 * focal_loss +
                self.dice_weight * dice_loss +
                self.contrastive_weight * contrastive_loss
        )

        return {
            'total': total_loss,
            'ce': ce_loss,
            'focal': focal_loss,
            'dice': dice_loss,
            'contrastive': contrastive_loss
        }


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

# ==================== DeepSpeed训练器类 ====================
class DeepSpeedPatchDiffTrainer:
    """使用DeepSpeed的训练器类"""

    def __init__(self, config):
        self.config = config
        self.local_rank = config.local_rank
        self.world_size = config.world_size

        # 设置设备
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        # 初始化分布式
        # if self.world_size > 1:
        #     os.environ['MASTER_ADDR'] = 'localhost'  # 或者设置实际的主机地址
        #     os.environ['MASTER_PORT'] = '29500'      # 选择一个可用的端口

        #     dist.init_process_group(
        #     backend='nccl',
        #     init_method='env://',
        #     world_size=self.world_size,
        #     rank=self.local_rank
        #     )
            # dist.init_process_group(backend='nccl')


        # 设置随机种子
        self._setup_seed(config.seed)

        # 创建实验目录
        if dist.get_rank() == 0:
            config.setup_paths()

        # 创建数据加载器
        self.train_loader, self.val_loader = self._create_data_loaders()

        # 创建模型
        self.model = DINOv3PatchDiffModelWithContrastive(config)

        # 创建损失函数
        self.criterion = PatchDiffContrastiveLoss(
            pos_weight=config.pos_weight,
            focal_gamma=config.focal_gamma,
            dice_weight=config.dice_weight,
            contrastive_weight=config.contrastive_weight
        )

        # 计算训练步数
        num_training_steps = config.num_epochs * len(self.train_loader)
        num_warmup_steps = int(0.1 * num_training_steps)

        # 更新DeepSpeed配置
        config.deepspeed_config['train_micro_batch_size_per_gpu'] = config.batch_size_per_gpu
        config.deepspeed_config['optimizer']['params']['lr'] = config.learning_rate
        config.deepspeed_config['optimizer']['params']['weight_decay'] = config.weight_decay
        config.deepspeed_config['scheduler']['params']['warmup_min_lr'] = config.min_lr
        config.deepspeed_config['scheduler']['params']['warmup_max_lr'] = config.learning_rate
        config.deepspeed_config['scheduler']['params']['warmup_num_steps'] = num_warmup_steps
        # config.deepspeed_config['scheduler']['params']['total_num_steps'] = num_training_steps

        # 初始化DeepSpeed
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=None,
            model=self.model,
            model_parameters=self.model.parameters(),
            config=config.deepspeed_config
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }

        if dist.get_rank() == 0:
            print(f"使用 {self.world_size} 个GPU进行训练")
            print(f"训练集大小: {len(self.train_loader.dataset)}")
            print(f"验证集大小: {len(self.val_loader.dataset)}")

    def _setup_seed(self, seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_data_loaders(self):
        """创建数据加载器"""
        from torchvision import transforms

        # 加载图像处理器
        processor = AutoImageProcessor.from_pretrained(self.config.model_name)

        # 数据增强
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

        # 创建数据集
        train_dataset = PatchDiffDataset(
            data_dir=self.config.data_root,
            processor=processor,
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            is_train=True,
            transform=train_transform
        )

        val_dataset = PatchDiffDataset(
            data_dir=self.config.data_root,
            processor=processor,
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            is_train=False,
            transform=None
        )

        # 划分数据集
        dataset_size = len(train_dataset)
        train_size = int(self.config.train_split * dataset_size)
        val_size = int(self.config.val_split * dataset_size)

        # 注意：这里需要确保所有进程使用相同的划分
        torch.manual_seed(self.config.seed)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # 创建分布式采样器
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size_per_gpu,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size_per_gpu,
            sampler=val_sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model_engine.train()
        if hasattr(self.train_loader, 'sampler') and self.train_loader.sampler is not None:
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0
        all_preds = []
        all_targets = []

        if dist.get_rank() == 0:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")
        else:
            pbar = self.train_loader

        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            image1 = batch['image1'].to(self.device)
            image2 = batch['image2'].to(self.device)
            patch_labels = batch['patch_labels'].to(self.device)

            # 前向传播
            outputs = self.model_engine(image1, image2)
            loss_dict = self.criterion(outputs, patch_labels)
            loss = loss_dict['total']

            # DeepSpeed反向传播
            self.model_engine.backward(loss)
            self.model_engine.step()

            # 收集统计信息
            total_loss += loss.item()

            # 收集预测用于计算指标
            with torch.no_grad():
                patch_logits = outputs['patch_logits']
                patch_probs = F.softmax(patch_logits, dim=-1)[:, :, 1]
                predictions = (patch_probs > 0.5).long()

                all_preds.append(predictions.cpu())
                all_targets.append(patch_labels.cpu())

            # 在主进程中更新进度条
            if dist.get_rank() == 0:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'ce': loss_dict['ce'].item(),
                    'contrastive': loss_dict['contrastive'].item()
                })
            # print(111111)

            # # 保存检查点
            # if batch_idx % self.config.save_interval == 0 and dist.get_rank() == 0:
            #     self.save_checkpoint(epoch, batch_idx)

        # 收集所有进程的结果
        if self.world_size > 1:
            total_loss_tensor = torch.tensor([total_loss], device=self.device)
            dist.all_reduce(total_loss_tensor)
            avg_loss = total_loss_tensor.item() / (len(self.train_loader) * self.world_size)

            # 收集预测和标签
            gathered_preds = [None] * self.world_size
            gathered_targets = [None] * self.world_size
            dist.all_gather_object(gathered_preds, all_preds)
            dist.all_gather_object(gathered_targets, all_targets)

            if dist.get_rank() == 0:
                all_preds = [item for sublist in gathered_preds for batch in sublist for item in batch.flatten()]
                all_targets = [item for sublist in gathered_targets for batch in sublist for item in batch.flatten()]
        else:
            avg_loss = total_loss / len(self.train_loader)
            all_preds = torch.cat(all_preds).flatten().numpy() if all_preds else []
            all_targets = torch.cat(all_targets).flatten().numpy() if all_targets else []

        # 计算指标
        metrics = {}
        if all_preds and all_targets and dist.get_rank() == 0:
            metrics = self.compute_metrics(all_preds, all_targets)

        return avg_loss, metrics

    @torch.no_grad()
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model_engine.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        if dist.get_rank() == 0:
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]")
        else:
            pbar = self.val_loader

        for batch in pbar:
            # 移动到设备
            image1 = batch['image1'].to(self.device)
            image2 = batch['image2'].to(self.device)
            patch_labels = batch['patch_labels'].to(self.device)

            # 前向传播
            outputs = self.model_engine(image1, image2)
            loss_dict = self.criterion(outputs, patch_labels)
            loss = loss_dict['total']

            # 收集统计信息
            total_loss += loss.item()

            # 收集预测
            patch_logits = outputs['patch_logits']
            patch_probs = F.softmax(patch_logits, dim=-1)[:, :, 1]
            predictions = (patch_probs > 0.5).long()

            all_preds.append(predictions.cpu())
            all_targets.append(patch_labels.cpu())

            if dist.get_rank() == 0:
                pbar.set_postfix({'loss': loss.item()})

        # 收集所有进程的结果
        if self.world_size > 1:
            total_loss_tensor = torch.tensor([total_loss], device=self.device)
            dist.all_reduce(total_loss_tensor)
            avg_loss = total_loss_tensor.item() / (len(self.val_loader) * self.world_size)

            # 收集预测和标签
            gathered_preds = [None] * self.world_size
            gathered_targets = [None] * self.world_size
            dist.all_gather_object(gathered_preds, all_preds)
            dist.all_gather_object(gathered_targets, all_targets)

            if dist.get_rank() == 0:
                all_preds = [item for sublist in gathered_preds for batch in sublist for item in batch.flatten()]
                all_targets = [item for sublist in gathered_targets for batch in sublist for item in batch.flatten()]
        else:
            avg_loss = total_loss / len(self.val_loader)
            all_preds = torch.cat(all_preds).flatten().numpy() if all_preds else []
            all_targets = torch.cat(all_targets).flatten().numpy() if all_targets else []

        # 计算指标
        metrics = {}
        if all_preds and all_targets and dist.get_rank() == 0:
            metrics = self.compute_metrics(all_preds, all_targets)

        return avg_loss, metrics

    def compute_metrics(self, predictions, targets):
        """计算评估指标"""
        if len(predictions) == 0 or len(targets) == 0:
            return {}

        # 确保predictions和targets是numpy数组
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        if isinstance(targets, list):
            targets = np.array(targets)

        # 如果仍然是布尔值或其他类型，进行转换
        if hasattr(predictions, 'dtype') and predictions.dtype == bool:
            predictions = predictions.astype(np.int32)
        if hasattr(targets, 'dtype') and targets.dtype == bool:
            targets = targets.astype(np.int32)

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
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'iou': float(iou),
            'pos_ratio': float(pos_ratio),
            'pred_pos_ratio': float(pred_pos_ratio),
            'num_samples': len(targets)
        }

    def save_checkpoint(self, epoch, step):
        """保存检查点"""
        # if dist.get_rank() == 0:
            # DeepSpeed保存检查点
        tag = f"epoch_{epoch + 1}_step_{step}"
        self.model_engine.save_checkpoint(
                save_dir=self.config.checkpoint_dir,
                tag=tag,
                client_state={
                    'epoch': epoch,
                    'step': step,
                    'history': self.history,
                    'config': self.config.__dict__
                }
            )
        print(f"检查点保存到: {os.path.join(self.config.checkpoint_dir, tag)}")

    def save_history(self):
        """保存训练历史"""
        # if dist.get_rank() == 0:
        history_path = os.path.join(self.config.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def train(self):
        """主训练循环"""
        if dist.get_rank() == 0:
            print(f"开始训练，共 {self.config.num_epochs} 个epoch")

        for epoch in range(self.config.num_epochs):
            if dist.get_rank() == 0:
                print(f"\n{'=' * 60}")
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                print(f"{'=' * 60}")

            # 训练
            train_loss, train_metrics = self.train_epoch(epoch)

            # 验证
            val_loss, val_metrics = self.validate_epoch(epoch)

            # 更新历史
            if dist.get_rank() == 0:
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_metrics'].append(train_metrics)
                self.history['val_metrics'].append(val_metrics)

                # 获取当前学习率
                if hasattr(self.optimizer, 'param_groups'):
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.history['learning_rates'].append(current_lr)

                # 打印结果
                print(f"\n训练结果:")
                print(f"  训练损失: {train_loss:.4f}")
                print(f"  验证损失: {val_loss:.4f}")

                if train_metrics:
                    print(f"\n训练指标:")
                    print(f"  准确率: {train_metrics['accuracy']:.4f}")
                    print(f"  F1分数: {train_metrics['f1']:.4f}")
                    print(f"  Precision: {train_metrics['precision']:.4f}")
                    print(f"  Recall: {train_metrics['recall']:.4f}")
                    print(f"  IoU: {train_metrics['iou']:.4f}")

                if val_metrics:
                    print(f"\n验证指标:")
                    print(f"  准确率: {val_metrics['accuracy']:.4f}")
                    print(f"  F1分数: {val_metrics['f1']:.4f}")
                    print(f"  Precision: {val_metrics['precision']:.4f}")
                    print(f"  Recall: {val_metrics['recall']:.4f}")
                    print(f"  IoU: {val_metrics['iou']:.4f}")

            # 保存最终模型检查点

            self.save_checkpoint(epoch, "final")

        # 保存历史
        self.save_history()

        # 清理分布式训练
        if self.world_size > 1:
            dist.destroy_process_group()


        print(f"\n训练完成！")


# ==================== 推理类（使用DeepSpeed加载）====================
class DeepSpeedPatchDiffPredictor:
    """使用DeepSpeed加载的推理类"""

    def __init__(self, checkpoint_dir, tag=None, device=None):
        """
        加载DeepSpeed检查点

        Args:
            checkpoint_dir: 检查点目录
            tag: 检查点标签（如'epoch_1_final'），如果为None则加载最新
            device: 推理设备
        """
        # 加载配置
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        self.config = DeepSpeedConfig()
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # 创建模型
        self.model = DINOv3PatchDiffModelWithContrastive(self.config)

        # DeepSpeed初始化（使用最小的配置）
        ds_config = {
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {"stage": 0},
            "fp16": {"enabled": False},
        }

        self.model_engine, _, _, _ = deepspeed.initialize(
            args=None,
            model=self.model,
            config=ds_config
        )

        # 加载检查点
        if tag is None:
            # 查找最新检查点
            checkpoints = [d for d in os.listdir(checkpoint_dir)
                           if os.path.isdir(os.path.join(checkpoint_dir, d))]
            tag = sorted(checkpoints)[-1] if checkpoints else None

        if tag:
            self.model_engine.load_checkpoint(
                checkpoint_dir,
                tag=tag,
                load_optimizer_states=False,
                load_lr_scheduler_states=False
            )

        self.model_engine.eval()

        # 加载图像处理器
        self.processor = AutoImageProcessor.from_pretrained(self.config.model_name)

        print(f"加载模型: {checkpoint_dir}/{tag}")
        print(f"使用设备: {self.device}")

    # 其他推理方法保持与原始Predictor相同
    # ...（省略，与原始PatchDiffPredictor类似）


# ==================== 主训练函数 ====================
def train_with_deepspeed():
    """使用DeepSpeed进行训练"""
    print("=" * 60)
    print("DINOv3 Patch-Level Difference Detection with DeepSpeed")
    print("=" * 60)

    # 设置配置
    config = DeepSpeedConfig()

    # 创建训练器
    trainer = DeepSpeedPatchDiffTrainer(config)

    # 开始训练
    trainer.train()

    return trainer


# ==================== 主程序 ====================
def main():
    """主函数"""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # 设置CUDA设备
    torch.cuda.set_device(local_rank)

    # 初始化分布式（只在多GPU时）
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            device_id=local_rank
        )
    print("DINOv3 Patch-Level Difference Detection System with DeepSpeed")
    train_with_deepspeed()


if __name__ == "__main__":
    # 注意：使用DeepSpeed训练时，需要通过命令行启动：


    main()