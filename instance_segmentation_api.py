# 🔧 Python script to run FastAPI segmentation API
import io, logging
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("segmentation_api")

# --- Config ---
MODEL_PATH = "best.pt"
YOLO_CLASSES = [
    'Adjustable spanner','Backsaw','Calipers','Cutting pliers','Drill','Gas wrench',
    'Gun','Hammer','Hand','Handsaw','Needle-nose pliers','Pliers','Ratchet',
    'Screwdriver','Tape measure','Utility knife','Wrench'
]

# --- Schemas ---
class Prediction(BaseModel):
    class_name: str = Field(..., alias="class")
    confidence: float
    polygon: List[List[int]]

    class Config:
        populate_by_name = True  # ✅ Allows using class_name in Python code

class PredictionsResponse(BaseModel):
    predictions: List[Prediction]

# --- Model wrapper ---
class SegModel:
    def __init__(self, path: str, class_names: List[str]):
        self.model = YOLO(path)
        self.class_names = class_names
        logger.info(f"Loaded model from {path}")

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def predict(self, image: Image.Image, conf_thresh=0.25) -> List[Prediction]:
        result = self.model(image)[0]
        masks = result.masks.data.cpu().numpy() if result.masks else []
        confs = result.boxes.conf.cpu().numpy() if result.boxes else []
        classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes else []

        preds = []
        for mask, conf, cls in zip(masks, confs, classes):
            if conf < conf_thresh: continue
            poly = self.mask_to_polygon(mask)
            if not poly: continue
            label = self.class_names[cls] if 0 <= cls < len(self.class_names) else str(cls)
            preds.append(Prediction(class_name=label, confidence=float(conf), polygon=poly))
        return preds

    def mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        mask_u8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return []
        return max(contours, key=cv2.contourArea).reshape(-1, 2).tolist()

# --- FastAPI setup ---
app = FastAPI(
    title="Instance Segmentation API",
    version="1.0.0",
    description="Detect objects and return polygons with class and confidence."
)

@app.on_event("startup")
def init_model():
    global seg_model
    try:
        seg_model = SegModel(MODEL_PATH, YOLO_CLASSES)
    except Exception as e:
        logger.exception("Model load failed.")
        raise RuntimeError("Could not load model.") from e

@app.post("/predict", response_model=PredictionsResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        predictions = seg_model.predict(image)
        return PredictionsResponse(predictions=predictions)
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.exception_handler(HTTPException)
async def handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
