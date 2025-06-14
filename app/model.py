import logging
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List
from app.schemas import Prediction

logger = logging.getLogger(__name__)

class YOLOModel:
    def __init__(self, model_path: str, class_names: List[str]):
        self.model = YOLO(model_path)
        self.class_names = class_names
        logger.info(f"Loaded YOLO model from {model_path}")

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def predict(self, image: Image.Image, conf_threshold: float = 0.25) -> List[Prediction]:
        results = self.model(image)
        r = results[0]
        preds: List[Prediction] = []

        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()    # shape: [N, H, W]
            confs = r.boxes.conf.cpu().numpy()    # shape: [N]
            classes = r.boxes.cls.cpu().numpy().astype(int)

            for mask, conf, cls in zip(masks, confs, classes):
                if conf < conf_threshold:
                    continue
                poly = self._mask_to_polygon(mask)
                if not poly:
                    continue
                label = (
                    self.class_names[cls]
                    if 0 <= cls < len(self.class_names)
                    else str(cls)
                )
                preds.append(Prediction(class_=label, confidence=float(conf), polygon=poly))
        return preds

    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        return largest.reshape(-1, 2).tolist()
