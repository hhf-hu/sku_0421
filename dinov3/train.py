# accelerate launch train.py
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
import torch.nn as nn
from tqdm.auto import tqdm

from dataset import HierarchicalDataset
from model import DINOv3ForClassification

# ================= 配置 =================
MODEL_NAME = "/data1/vincent/models/facebook-dinov3-vit7b16-pretrain-sat493m"
DATA_ROOT = "/data1/vincent/datasets/data1210/train/"
FINE_GRAINED = False

BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-5

SAVE_EVERY_EPOCHS = 1
OUTPUT_DIR = "./checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =======================================
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
dataset = HierarchicalDataset(DATA_ROOT, processor, fine_grained=FINE_GRAINED)

def main():
    # -------- DeepSpeed ZeRO-3 ----------
    ds_plugin = DeepSpeedPlugin(
        zero_stage=3,
        # 如果显存紧张再开
        # offload_optimizer_device="cpu",
        # offload_param_device="cpu",
    )

    accelerator = Accelerator(
        mixed_precision="fp16",
        deepspeed_plugin=ds_plugin
    )

    # -------- Data ----------
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    dataset = HierarchicalDataset(DATA_ROOT, processor, fine_grained=FINE_GRAINED)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # -------- Model ----------
    model = DINOv3ForClassification(
        MODEL_NAME,
        num_classes=len(dataset.label_names)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    # ================= Train =================
    model.train()

    for epoch in range(EPOCHS):

        if accelerator.is_main_process:
            pbar = tqdm(
                total=len(dataloader),
                desc=f"Epoch [{epoch+1}/{EPOCHS}]",
                leave=True
            )

        for step, (pixel_values, labels) in enumerate(dataloader):
            pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
            labels = labels.to(accelerator.device, non_blocking=True)

            outputs = model(pixel_values)
            loss = criterion(outputs, labels)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.is_main_process:
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        if accelerator.is_main_process:
            pbar.close()

        accelerator.wait_for_everyone()

        # ========== 保存模型 ==========
        if (epoch + 1) % SAVE_EVERY_EPOCHS == 0:
            save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")

            if accelerator.is_main_process:
                print(f"\n💾 Saving checkpoint to {save_path}")

            # 方式一（推荐）：保存完整训练状态（支持 resume）
            accelerator.save_state(save_path)

            # 方式二（可选）：只存模型权重（推理用）
            # unwrapped_model = accelerator.unwrap_model(model)
            # unwrapped_model.save_pretrained(
            #     save_path,
            #     is_main_process=accelerator.is_main_process,
            #     save_function=accelerator.save
            # )

            accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
