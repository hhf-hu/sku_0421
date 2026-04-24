import os
from transformers import AutoImageProcessor


from dataset import HierarchicalDataset


# ================= 配置 =================
MODEL_NAME = "/Users/vincent/workspace/trademark_similar/clip_finetuned"
DATA_ROOT = "/Users/vincent/workspace/trademark_similar/character_data/data_characters"
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
print(len(dataset.samples))
print(dataset)