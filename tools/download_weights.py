import os
import requests

def download_file(url, save_path):
    print(f"Đang tải model từ: {url}...")
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Đã lưu tại: {save_path}")

# Ví dụ link trực tiếp từ MMDetection hoặc HuggingFace
MODELS = {
    "rt_detr_weights.pth": "https://download.openmmlab.com/mmdetection/v3.0/rt_detr/..." 
}

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    for name, url in MODELS.items():
        download_file(url, os.path.join("checkpoints", name))