from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
import cv2
from ultralytics import YOLO

class TrafficLightDetector:
    """
    입력: BGR np.ndarray(H,W,3) 또는 이미지 경로(str)
    출력: [{"class_id":int,"class_name":str,"conf":float,"bbox":[x1,y1,x2,y2]}]
    """
    def __init__(self,
                 weights: Union[str, Path],
                 device: str = "0",
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.7,
                 classes_map: Dict[int, str] = None):
        self.model = YOLO(str(weights))
        self.device = device
        self.conf = conf_thres
        self.iou = iou_thres
        self.classes_map = classes_map or {i: f"class_{i}" for i in range(14)}

    def _to_bgr(self, img):
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_COLOR)
        assert isinstance(img, np.ndarray) and img.ndim == 3, "input must be BGR image or path"
        return img

    def predict(self, img, imgsz: int = 512) -> List[Dict[str, Any]]:
        img = self._to_bgr(img)
        r = self.model.predict(
            source=img,
            device=self.device,
            imgsz=imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )[0]

        out = []
        if r.boxes is None: 
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls   = r.boxes.cls.cpu().numpy().astype(int)
        for (x1,y1,x2,y2), c, k in zip(boxes, confs, cls):
            out.append({
                "class_id": int(k),
                "class_name": self.classes_map.get(int(k), f"class_{int(k)}"),
                "conf": float(c),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
        return out
