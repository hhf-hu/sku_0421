import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from safetensors import safe_open
import numpy as np
from accelerate import load_checkpoint_and_dispatch
from tqdm.auto import tqdm
import pandas as pd
from PIL import Image
import traceback
import json

def create_patch_annotations(img1_path, img2_path, patch_size=16, output_dir="annotations"):
    """交互式创建patch标注的工具"""
    import cv2
    from matplotlib import patches as mpatches

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    h, w = img1.shape[:2]
    grid_h = h // patch_size
    grid_w = w // patch_size

    # 创建空的标注矩阵
    patch_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    ax1.imshow(img1)
    ax1.set_title("Image 1 - Click to mark differences")
    ax2.imshow(img2)
    ax2.set_title("Image 2")

    # 绘制网格
    for i in range(grid_h + 1):
        ax1.axhline(y=i * patch_size, color='yellow', alpha=0.3, linewidth=0.5)
        ax2.axhline(y=i * patch_size, color='yellow', alpha=0.3, linewidth=0.5)
    for j in range(grid_w + 1):
        ax1.axvline(x=j * patch_size, color='yellow', alpha=0.3, linewidth=0.5)
        ax2.axvline(x=j * patch_size, color='yellow', alpha=0.3, linewidth=0.5)

    def onclick(event):
        if event.inaxes == ax1:
            # 计算点击的patch坐标
            patch_x = int(event.xdata // patch_size)
            patch_y = int(event.ydata // patch_size)

            # 切换标注状态
            patch_mask[patch_y, patch_x] = 1 - patch_mask[patch_y, patch_x]

            # 更新显示
            update_display()

    def update_display():
        # 清除之前的矩形
        for patch in ax1.patches + ax2.patches:
            patch.remove()

        # 绘制当前标注
        for i in range(grid_h):
            for j in range(grid_w):
                if patch_mask[i, j] == 1:
                    rect1 = mpatches.Rectangle(
                        (j * patch_size, i * patch_size),
                        patch_size, patch_size,
                        linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
                    )
                    rect2 = mpatches.Rectangle(
                        (j * patch_size, i * patch_size),
                        patch_size, patch_size,
                        linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
                    )
                    ax1.add_patch(rect1)
                    ax2.add_patch(rect2)

        plt.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

    # 保存标注
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    cv2.imwrite(f"{output_dir}/image1.jpg", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{output_dir}/image2.jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

    # 保存标注信息
    annotation = {
        "image_size": [w, h],
        "patch_size": patch_size,
        "grid_shape": [grid_h, grid_w],
        "patch_mask": patch_mask.tolist(),
        "differences": []
    }

    # 添加具体的差异信息
    diff_count = 0
    for i in range(grid_h):
        for j in range(grid_w):
            if patch_mask[i, j] == 1:
                patch_id = i * grid_w + j
                annotation["differences"].append({
                    "patch_id": patch_id,
                    "patch_position": [i, j],
                    "coordinates": [
                        j * patch_size,
                        i * patch_size,
                        (j + 1) * patch_size,
                        (i + 1) * patch_size
                    ],
                    "confidence": 1.0
                })
                diff_count += 1

    with open(f"{output_dir}/patch_labels.json", "w") as f:
        json.dump(annotation, f, indent=2)

    print(f"标注完成！标注了 {diff_count} 个差异patch")
    return annotation


img1 = "/Users/vincent/workspace/sku/data/kitty/Unit0001/IMG_0938.JPG"
img2 = "/Users/vincent/workspace/sku/data/kitty/Unit0002/IMG_0009.JPG"
create_patch_annotations(img1, img2)