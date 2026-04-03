import os
import json
import cv2

os.environ["FLAGS_use_mkldnn"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"

from mmdet.apis import DetInferencer
from paddleocr import PaddleOCR

# ========================
# CONFIG
# ========================
IMAGE_PATH = 'data/images/1.png'
CONFIG_FILE = 'configs/custom.py'
CHECKPOINT = 'work_dir/epoch_30.pth'

OUTPUT_DIR = 'outputs'
CROP_DIR = os.path.join(OUTPUT_DIR, "crops")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

# ========================
# INIT
# ========================
print("🚀 Loading models...")
detector = DetInferencer(model=CONFIG_FILE, weights=CHECKPOINT, device='cpu')

ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    use_gpu=False,
    show_log=False
)

img = cv2.imread(IMAGE_PATH)
vis_img = img.copy()

# ========================
# CLASS MAP
# ========================
classes = detector.model.dataset_meta['classes']
CLASS_MAP = {i: name for i, name in enumerate(classes)}

# ========================
# OCR HELPERS
# ========================
def extract_text(image):
    result = ocr.ocr(image)
    lines = []

    if result and result[0]:
        for line in result[0]:
            box = line[0]
            text = line[1][0]
            y = box[0][1]
            lines.append((y, text))

    lines = sorted(lines, key=lambda x: x[0])
    return "\n".join([l[1] for l in lines])


def extract_table(image):
    result = ocr.ocr(image)
    rows = []

    if result and result[0]:
        lines = []
        for line in result[0]:
            box = line[0]
            text = line[1][0]
            x = box[0][0]
            y = box[0][1]
            lines.append((x, y, text))

        lines = sorted(lines, key=lambda x: (round(x[1]/15)*15, x[0]))

        current_row = []
        last_y = None

        for x, y, t in lines:
            if last_y is None:
                current_row.append((x, t))
                last_y = y
                continue

            if abs(y - last_y) > 15:
                rows.append([c[1] for c in sorted(current_row)])
                current_row = [(x, t)]
            else:
                current_row.append((x, t))

            last_y = y

        if current_row:
            rows.append([c[1] for c in sorted(current_row)])

    return rows

# ========================
# DETECT
# ========================
print("🔍 Detecting...")
result = detector(IMAGE_PATH, pred_score_thr=0.5)
pred = result['predictions'][0]

json_output = {
    "image": os.path.basename(IMAGE_PATH),
    "total_objects": len(pred['bboxes']),
    "objects": []
}

obj_id = 1

# ========================
# PROCESS
# ========================
for bbox, label, score in zip(pred['bboxes'], pred['labels'], pred['scores']):
    x1, y1, x2, y2 = map(int, bbox)

    if x2 <= x1 or y2 <= y1:
        continue

    crop = img[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        continue

    class_name = CLASS_MAP[label]

    # ========================
    # SAVE CROP
    # ========================
    crop_path = os.path.join(CROP_DIR, f"{obj_id}_{class_name}.png")
    cv2.imwrite(crop_path, crop)

    # ========================
    # OCR
    # ========================
    ocr_content = ""

    if class_name == "note":
        ocr_content = extract_text(crop)

    elif class_name == "table":
        ocr_content = extract_table(crop)

    # ========================
    # JSON OBJECT
    # ========================
    obj = {
        "id": obj_id,
        "class": class_name.capitalize(),
        "confidence": float(score),
        "bbox": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        },
        "crop_path": crop_path,
        "ocr_content": ocr_content
    }

    json_output["objects"].append(obj)

    # ========================
    # VISUALIZE
    # ========================
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis_img, class_name, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    obj_id += 1

# ========================
# SAVE JSON
# ========================
json_path = os.path.join(OUTPUT_DIR, "result.json")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=4, ensure_ascii=False)

# ========================
# SAVE VISUALIZATION
# ========================
vis_path = os.path.join(OUTPUT_DIR, "visualize.png")
cv2.imwrite(vis_path, vis_img)

print("🎉 DONE")
print("📄 JSON:", json_path)
print("🖼 VIS:", vis_path)