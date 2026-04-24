import os
from PIL import Image
from torch.utils.data import Dataset


class HierarchicalDataset(Dataset):
    def __init__(self, root_dir, processor, fine_grained=False):
        self.samples = []
        self.labels = []
        self.processor = processor
        self.fine_grained = fine_grained

        label_set = set()

        for sku_name in os.listdir(root_dir):
            sku_path = os.path.join(root_dir, sku_name)
            if not os.path.isdir(sku_path):
                continue

            for unit_name in os.listdir(sku_path):
                unit_path = os.path.join(sku_path, unit_name)
                if not os.path.isdir(unit_path):
                    continue

                label_name = f"{sku_name}_{unit_name}" if fine_grained else sku_name
                label_set.add(label_name)

                for img_name in os.listdir(unit_path):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(unit_path, img_name)
                        self.samples.append((img_path, label_name))

        self.label_names = sorted(list(label_set))
        self.label2id = {label: idx for idx, label in enumerate(self.label_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
        label_id = self.label2id[label_name]
        print(len(label_id))
        return pixel_values, label_id
