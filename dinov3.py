import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "facebook/dinov3-vit7b16-pretrain-sat493m"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

def extract_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    # CLS token
    embedding = outputs.last_hidden_state[:, 0, :]  # [1, hidden_dim]

    # L2 normalize
    embedding = F.normalize(embedding, dim=-1)

    return embedding.squeeze(0)  # [hidden_dim]

# 测试
img1 = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
img2 = load_image("http://images.cocodataset.org/val2017/000000039770.jpg")

emb1 = extract_embedding(img1)
emb2 = extract_embedding(img2)

# cosine similarity
sim = torch.dot(emb1, emb2).item()
print("Cosine similarity:", sim)
