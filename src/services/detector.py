"""唯一的 YOLO 模型加载与推理入口，Web 层不接触 Ultralytics。"""
import logging
from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2

from src.services.errors import DetectionError, ModelUnavailable
from src.services.model_metadata import read_model_metadata
from src.utils.validators import read_image

logger = logging.getLogger(__name__)
PALETTE = [(112, 137, 47), (102, 111, 193), (178, 130, 76), (160, 111, 139)]


def choose_device(preference="auto"):
    import torch

    if preference != "auto":
        return preference
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class HelmetDetector:
    def __init__(self, model_path, device="auto", conf=0.25, iou=0.7, imgsz=640, max_pixels=25_000_000):
        path = Path(model_path)
        if not path.is_file():
            raise ModelUnavailable(f"YOLO model not found: {path}\nPlease place your trained model in models/best.pt")
        try:
            from ultralytics import YOLO

            self.device = choose_device(device)
            self.model = YOLO(str(path))
            if self.model.task != "detect":
                raise ModelUnavailable("请提供用于目标检测的 YOLO 权重，当前权重任务不是 detect。")
            self.names = {int(key): str(value) for key, value in self.model.names.items()}
        except ModelUnavailable:
            raise
        except Exception as error:
            raise ModelUnavailable("YOLO 模型加载失败，请检查权重文件和依赖环境。") from error
        self.conf = conf
        self.metadata = read_model_metadata(path)
        self.iou = iou
        self.imgsz = imgsz
        self.max_pixels = max_pixels
        # Ultralytics 的 predictor 可变，串行访问同一实例避免多请求串扰。
        self._lock = Lock()
        logger.info("YOLO loaded once: %s device=%s classes=%s", path, self.device, self.names)

    def detect_frame(self, frame, conf=None):
        try:
            with self._lock:
                started = perf_counter()
                prediction = self.model.predict(
                    source=frame, conf=self.conf if conf is None else conf, iou=self.iou,
                    imgsz=self.imgsz, device=self.device, verbose=False, save=False,
                )[0]
                boxes = prediction.boxes.data.cpu().tolist()
                inference_time = perf_counter() - started
        except Exception as error:
            raise DetectionError("模型推理失败，请检查文件或推理设备配置后重试。") from error
        classes = {name: 0 for name in self.names.values()}
        detections = []
        for x1, y1, x2, y2, confidence, class_id in boxes:
            class_id = int(class_id)
            name = self.names.get(class_id, str(class_id))
            classes[name] = classes.get(name, 0) + 1
            detections.append({"class_id": class_id, "class_name": name,
                               "confidence": round(float(confidence), 4),
                               "bbox": [round(value, 2) for value in (x1, y1, x2, y2)]})
        return {"total": len(detections), "classes": classes, "detections": detections,
                "inference_time": inference_time,
                "mean_confidence": sum(item["confidence"] for item in detections) / len(detections) if detections else None}

    def annotate(self, frame, detections):
        output = frame.copy()
        height, width = output.shape[:2]
        thickness = max(2, round(min(width, height) / 300))
        scale = max(0.45, min(width, height) / 1200)
        for detection in detections:
            x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            color = PALETTE[detection["class_id"] % len(PALETTE)]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            label_y = max(text_height + 10, y1)
            cv2.rectangle(output, (x1, label_y - text_height - 10),
                          (min(width - 1, x1 + text_width + 12), label_y + baseline), color, -1)
            cv2.putText(output, label, (x1 + 6, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), 1, cv2.LINE_AA)
        return output

    def detect_image(self, image_path, result_path=None, conf=None):
        frame = read_image(image_path, self.max_pixels)
        result = self.detect_frame(frame, conf)
        if result_path is not None and not cv2.imwrite(str(result_path), self.annotate(frame, result["detections"])):
            raise DetectionError("结果图片保存失败，请检查存储目录。")
        result["media_info"] = {"width": frame.shape[1], "height": frame.shape[0]}
        return result
