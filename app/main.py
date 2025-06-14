from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from app.utils import configure_logging
from app.model import YOLOModel
from app.schemas import PredictionsResponse
from PIL import Image
import io, logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

YOLO_CLASSES = [
    'Adjustable spanner','Backsaw','Calipers','Cutting pliers','Drill','Gas wrench',
    'Gun','Hammer','Hand','Handsaw','Needle-nose pliers','Pliers','Ratchet',
    'Screwdriver','Tape measure','Utility knife','Wrench'
]

app = FastAPI(
    title="YOLOv8 Segmentation API",
    version="1.0.0",
    description="Detect objects and return polygons, classes, confidences."
)

# Load model once at startup
@app.on_event("startup")
def load_model():
    global yolo_model
    try:
        yolo_model = YOLOModel("best.pt", YOLO_CLASSES)
    except Exception as e:
        logger.exception("Model failed to load")
        raise

@app.post("/predict", response_model=PredictionsResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        preds = yolo_model.predict(img)
        return PredictionsResponse(predictions=preds)
    except ValidationError as ve:
        logger.error("Output validation error: %s", ve)
        raise HTTPException(status_code=422, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
