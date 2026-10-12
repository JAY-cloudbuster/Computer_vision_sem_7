"""
Classical Computer Vision License Plate Detector using OpenCV.
Techniques: Bilateral Filtering, Sobel Horizontal Gradients, Morphological Closing,
Contour Geometric Analysis (Aspect Ratio, Area, Solidity), and Non-Maximum Suppression (NMS).
"""

import cv2
import numpy as np

def non_max_suppression(boxes, scores=None, overlap_thresh=0.3):
    """
    Standard Non-Maximum Suppression (NMS) for bounding boxes.
    boxes format: [[x, y, w, h], ...]
    """
    if len(boxes) == 0:
        return []

    boxes_arr = np.array(boxes, dtype=np.float32)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 0] + boxes_arr[:, 2]
    y2 = boxes_arr[:, 1] + boxes_arr[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    if scores is not None:
        order = np.argsort(scores)[::-1]
    else:
        order = np.argsort(areas)[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
        inds = np.where(iou <= overlap_thresh)[0]
        order = order[inds + 1]

    return [boxes[i] for i in keep]

class ClassicalPlateDetector:
    def __init__(self, min_ar=1.8, max_ar=5.5, min_area_ratio=0.0008, max_area_ratio=0.08):
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def detect(self, image):
        """
        Detects candidate license plate bounding boxes in an image.
        Returns: list of bounding boxes [[x, y, w, h], ...]
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        if image is None:
            return []

        h_img, w_img = image.shape[:2]
        total_area = h_img * w_img

        # 1. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # 3. Morphological Top-Hat & Black-Hat to reveal plate contrast
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, rect_kernel)
        blackhat = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, rect_kernel)
        enhanced = cv2.add(filtered, tophat)
        enhanced = cv2.subtract(enhanced, blackhat)

        # 4. Sobel vertical edge gradient (highlights alphanumeric characters)
        grad_x = cv2.Sobel(enhanced, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        min_val, max_val = np.min(grad_x), np.max(grad_x)
        grad_x = 255 * ((grad_x - min_val) / (max_val - min_val + 1e-6))
        grad_x = grad_x.astype("uint8")

        # 5. Gaussian blur & Otsu thresholding
        blurred = cv2.GaussianBlur(grad_x, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 6. Morphological Closing to consolidate character blobs into a plate rectangle
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        # Clean noise
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.erode(closed, clean_kernel, iterations=1)
        closed = cv2.dilate(closed, clean_kernel, iterations=2)

        # 7. Contour extraction & Geometric Filtering
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        scores = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            aspect_ratio = float(w) / float(h + 1e-5)
            area_ratio = area / float(total_area)

            # Check geometric priors
            if (self.min_ar <= aspect_ratio <= self.max_ar and 
                self.min_area_ratio <= area_ratio <= self.max_area_ratio):
                
                # Check solidity / extent
                contour_area = cv2.contourArea(c)
                extent = contour_area / (float(area) + 1e-5)
                
                if extent > 0.35:
                    candidates.append([x, y, w, h])
                    # Score candidate by aspect ratio proximity to ideal standard plate ratio (~3.2)
                    score = 1.0 - abs(aspect_ratio - 3.2) / 3.2
                    scores.append(score)

        # 8. Apply Non-Maximum Suppression
        nms_boxes = non_max_suppression(candidates, scores, overlap_thresh=0.25)
        return nms_boxes

    def draw_detections(self, image, boxes, color=(0, 255, 0), thickness=2):
        """Draws bounding boxes on a copy of the image."""
        annotated = image.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
            cv2.putText(annotated, "Plate Candidate", (int(x), max(15, int(y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return annotated
