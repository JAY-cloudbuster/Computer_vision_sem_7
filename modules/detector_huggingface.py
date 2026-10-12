"""
Pretrained Modern Deep License Plate Detector from Hugging Face Hub.
Uses YOLOv8 License Plate Detection (ml-debi/yolov8-license-plate-detection) via OpenCV DNN engine.
Ultra-lightweight, zero DLL issues, and runs at 30+ FPS on CPU.
"""

import os
import cv2
import urllib.request
import numpy as np

MODEL_URL = "https://huggingface.co/ml-debi/yolov8-license-plate-detection/resolve/main/best.onnx"
DEFAULT_WEIGHTS = "models/yolov8_plate.onnx"

class HuggingFacePlateDetector:
    def __init__(self, model_path=DEFAULT_WEIGHTS, conf_thresh=0.25, iou_thresh=0.45):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self._ensure_model_exists()
        
        # Load via OpenCV's built-in DNN engine
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"[*] Downloading pretrained Hugging Face detector weights from {MODEL_URL}...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print(f"[OK] Downloaded detector to '{self.model_path}'.")

    def detect(self, image):
        """
        Runs inference on an image (path or BGR ndarray).
        Returns:
            boxes: list of [x, y, w, h] in original pixel coordinates.
            confs: list of float confidence scores.
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        if image is None:
            return [], []

        h0, w0 = image.shape[:2]

        # Prepare 640x640 blob
        blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()  # shape (1, 5, 8400)

        preds = preds[0].transpose()  # shape (8400, 5) -> cx, cy, w, h, conf

        confs = preds[:, 4]
        mask = confs >= self.conf_thresh
        filtered_preds = preds[mask]
        filtered_confs = confs[mask]

        if len(filtered_confs) == 0:
            return [], []

        boxes_xywh = filtered_preds[:, :4]
        # Rescale from 640x640 to original image resolution
        cx = boxes_xywh[:, 0] * (w0 / 640.0)
        cy = boxes_xywh[:, 1] * (h0 / 640.0)
        w = boxes_xywh[:, 2] * (w0 / 640.0)
        h = boxes_xywh[:, 3] * (h0 / 640.0)
        x = cx - w / 2.0
        y = cy - h / 2.0

        # Convert to integer boxes for cv2 NMS
        boxes_for_nms = [[int(x[i]), int(y[i]), int(w[i]), int(h[i])] for i in range(len(x))]
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, filtered_confs.tolist(), self.conf_thresh, self.iou_thresh)

        final_boxes = []
        final_confs = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                bx, by, bw, bh = boxes_for_nms[i]
                bx = max(0, min(bx, w0 - 1))
                by = max(0, min(by, h0 - 1))
                bw = max(1, min(bw, w0 - bx))
                bh = max(1, min(bh, h0 - by))
                final_boxes.append([float(bx), float(by), float(bw), float(bh)])
                final_confs.append(float(filtered_confs[i]))

        return final_boxes, final_confs

    def draw_detections(self, image, boxes, confs=None, color=(0, 140, 255), thickness=2):
        annotated = image.copy()
        for idx, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
            conf_str = f"{confs[idx]:.2f}" if confs is not None and idx < len(confs) else ""
            label = f"HF YOLO {conf_str}".strip()
            cv2.putText(annotated, label, (int(x), max(15, int(y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return annotated
