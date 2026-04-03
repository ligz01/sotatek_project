import os
import cv2
import uuid

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from mmdet.apis import DetInferencer
from paddleocr import PaddleOCR

# ========================
# INIT APP
# ========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# LOAD MODEL (1 LẦN DUY NHẤT)
# ========================
print("🚀 Loading models...")

detector = DetInferencer(
    model='configs/custom.py',
    weights='work_dir/epoch_30.pth',
    device='cpu'
)

ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    use_gpu=False,
    show_log=False
)

classes = detector.model.dataset_meta['classes']
CLASS_MAP = {i: name for i, name in enumerate(classes)}

# ========================
# OUTPUT DIR
# ========================
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# OCR HELPER
# ========================
def extract_text(image):
    result = ocr.ocr(image)
    texts = []

    if result and result[0]:
        for line in result[0]:
            texts.append(line[1][0])

    return "\n".join(texts)

# ========================
# API: PREDICT
# ========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # save temp image
    temp_name = f"temp_{uuid.uuid4().hex}.png"
    with open(temp_name, "wb") as f:
        f.write(await file.read())

    img = cv2.imread(temp_name)
    vis_img = img.copy()

    result = detector(temp_name, pred_score_thr=0.5)
    pred = result['predictions'][0]

    output = {
        "image": file.filename,
        "objects": []
    }

    obj_id = 1

    for bbox, label, score in zip(pred['bboxes'], pred['labels'], pred['scores']):
        x1, y1, x2, y2 = map(int, bbox)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue

        class_name = CLASS_MAP[label]

        # OCR
        ocr_text = ""
        if class_name in ["note", "table"]:
            ocr_text = extract_text(crop)

        # COLOR theo class
        if class_name == "note":
            color = (0, 255, 0)
        elif class_name == "table":
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        # draw bbox
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis_img,
            f"{class_name} {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

        output["objects"].append({
            "id": obj_id,
            "class": class_name.capitalize(),
            "confidence": float(score),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            },
            "ocr_content": ocr_text
        })

        obj_id += 1

    # save result image
    result_path = os.path.join(
        OUTPUT_DIR,
        f"result_{uuid.uuid4().hex}.png"
    )
    cv2.imwrite(result_path, vis_img)

    return JSONResponse({
        "json": output,
        "image_path": result_path
    })

# ========================
# API: GET IMAGE
# ========================
@app.get("/image")
def get_image(path: str):
    return FileResponse(path)