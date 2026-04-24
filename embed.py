import requests

# 128.241.246.80:37001
EMBED_URL = "http://183.131.181.170:30000/encode_images"
img_path = "/Users/vincent/workspace/sku/Unit0001/IMG_0937.JPG"
with open(img_path, "rb") as f:
    files = {"files": (img_path.split("/")[-1], f)}
    response = requests.post(EMBED_URL, files=files, timeout=30)
data = response.json()

embedding = data["results"][0]["features"][0] if "results" in data else data[0]
print(embedding)