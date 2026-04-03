from mmdet.apis import DetInferencer
import cv2
import os

# ===== CONFIG =====
IMAGE_PATH = 'data/images/1.png'
CONFIG_PATH = 'configs/custom.py'
CHECKPOINT = 'work_dir/epoch_30.pth'
OUTPUT_DIR = 'crops'
SCORE_THR = 0.5

# ===== LOAD MODEL =====
inferencer = DetInferencer(
    model=CONFIG_PATH,
    weights=CHECKPOINT,
    device='cpu'
)

# ===== RUN INFERENCE =====
result = inferencer(IMAGE_PATH)

# ===== LOAD IMAGE =====
img = cv2.imread(IMAGE_PATH)

# ===== CREATE OUTPUT DIR =====
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== GET RESULT =====
pred = result['predictions'][0]
bboxes = pred['bboxes']
scores = pred['scores']
labels = pred['labels']

# ===== CLASS NAMES ===== (optional)
classes = ['class1', 'class2', 'class3']

# ===== CROP =====
for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
    if score < SCORE_THR:
        continue

    x1, y1, x2, y2 = map(int, bbox)

    # padding cho đẹp
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.shape[1], x2 + pad)
    y2 = min(img.shape[0], y2 + pad)

    crop = img[y1:y2, x1:x2]

    class_name = classes[label]

    filename = f'{OUTPUT_DIR}/{class_name}_{i}_{score:.2f}.png'
    cv2.imwrite(filename, crop)

    print(f"Saved: {filename}")

print("✅ DONE CROPPING!")