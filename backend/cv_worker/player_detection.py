import os
from typing import List, Tuple, Dict, Any
import numpy as np

# Try importing ultralytics (installed in the virtualenv)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class BoundingBox:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float, confidence: float, class_id: int):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.confidence = confidence
        self.class_id = class_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "confidence": self.confidence,
            "class_id": self.class_id
        }

class PlayerDetector:
    def __init__(self, model_path: str = "yolov8n.pt", force_cpu: bool = False):
        self.model_path = model_path
        self.force_cpu = force_cpu
        self.model = None
        self.is_mock = True

        # Check if weights exist and YOLO is imported
        if YOLO is not None and os.path.exists(model_path):
            try:
                device = "cpu" if force_cpu else "cuda"
                self.model = YOLO(model_path)
                self.is_mock = False
                print(f"[YOLOv8] Loaded model from {model_path} on {device}")
            except Exception as e:
                print(f"[YOLOv8] Failed to load model: {e}. Falling back to Mock Detector.")
        else:
            print("[YOLOv8] Model weights not found or ultralytics not available. Initializing Mock Detector.")

    def detect_players(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Detects players in the frame.
        Returns a list of BoundingBox objects.
        """
        if self.is_mock or self.model is None:
            # Generate mock bounding boxes representing typical locations on the court:
            # Player 1 (top half): x: 0.4-0.6, y: 0.2-0.4
            # Player 2 (bottom half): x: 0.4-0.6, y: 0.6-0.8
            height, width = frame.shape[0], frame.shape[1]
            box_1 = BoundingBox(
                x_min=width * 0.45,
                y_min=height * 0.25,
                x_max=width * 0.55,
                y_max=height * 0.35,
                confidence=0.98,
                class_id=0  # Person class in COCO
            )
            box_2 = BoundingBox(
                x_min=width * 0.45,
                y_min=height * 0.65,
                x_max=width * 0.55,
                y_max=height * 0.75,
                confidence=0.97,
                class_id=0
            )
            return [box_1, box_2]

        device = "cpu" if self.force_cpu else 0  # 0 for first CUDA device
        results = self.model(frame, device=device, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                # Class 0 is 'person' in standard COCO dataset
                if cls_id == 0:
                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    boxes.append(BoundingBox(x_min, y_min, x_max, y_max, conf, cls_id))
        
        # Sort by confidence and return top 2 detected players (singles match)
        boxes = sorted(boxes, key=lambda x: x.confidence, reverse=True)
        return boxes[:2]
