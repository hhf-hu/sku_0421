import os
import pandas as pd

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, precision_recall_curve, auc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from transformers import CLIPProcessor, CLIPModel, AutoModel,AutoProcessor
import random



def sliding_windows(img, win=448, stride=448):
    W, H = img.size
    crops = []
    boxes = []
    for y in range(0, H - win + 1, stride):
        for x in range(0, W - win + 1, stride):
            crop = img.crop((x, y, x + win, y + win))
            crops.append(crop)
            boxes.append((x, y, x + win, y + win))
    return crops, boxes

def load_model(model_path, device):
    model_dir = model_path
    model = AutoModel.from_pretrained(model_dir, use_safetensors=True).to(device)

    return model

model_path = "/Users/vincent/workspace/sku/output_dir"

processor = AutoProcessor.from_pretrained("/Users/vincent/Downloads/dfn5b")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, device)

imgA = Image.open("/Users/vincent/workspace/sku/Unit0001/IMG_0937.JPG").convert("RGB")
imgB = Image.open("/Users/vincent/workspace/sku/Unit0002/IMG_0009.JPG").convert("RGB")

cropsA, boxesA = sliding_windows(imgA)
cropsB, boxesB = sliding_windows(imgB)




def encode_crops(crops):
    inputs = processor(images=crops, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

featA = encode_crops(cropsA)  # [NA, D]
featB = encode_crops(cropsB)

sim = featA @ featB.T
best_sim_A, _ = sim.max(dim=1)
print(best_sim_A.mean().item(), best_sim_A.max().item())
