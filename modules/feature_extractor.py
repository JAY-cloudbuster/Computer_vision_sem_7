"""
Feature Extractor for License Plate Candidate Crops.
Extracts 10 Handcrafted Visual Features across Shape, Texture, and Color:
- Shape (4): Aspect Ratio, Solidity, Extent, Compactness
- Texture (3): GLCM Contrast, GLCM Homogeneity, GLCM Energy
- Color (3): Mean Saturation (HSV), Value Std Dev (HSV), Color Uniformity / Entropy
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

FEATURE_NAMES = [
    # Shape
    "aspect_ratio",
    "solidity",
    "extent",
    "compactness",
    # Texture
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    # Color
    "mean_saturation",
    "value_std",
    "color_uniformity"
]

class PlateFeatureExtractor:
    def __init__(self, target_size=(120, 40)):
        self.target_size = target_size  # (W, H)

    def extract(self, crop):
        """
        Extracts 10-dimensional feature vector from a candidate crop (BGR image).
        Returns: np.ndarray of shape (10,)
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

        h_orig, w_orig = crop.shape[:2]
        aspect_ratio = float(w_orig) / float(h_orig + 1e-5)

        # Standardize size for consistent texture & color feature evaluation
        crop_std = cv2.resize(crop, self.target_size)
        gray = cv2.cvtColor(crop_std, cv2.COLOR_BGR2GRAY)

        # ----------------- 1. SHAPE FEATURES (from Otsu binary contour) -----------------
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            c_area = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            perimeter = cv2.arcLength(c, True)

            solidity = float(c_area) / float(hull_area + 1e-5)
            extent = float(c_area) / float(self.target_size[0] * self.target_size[1] + 1e-5)
            compactness = (perimeter ** 2) / (4.0 * np.pi * c_area + 1e-5)
        else:
            solidity = 0.5
            extent = 0.5
            compactness = 1.0

        # ----------------- 2. TEXTURE FEATURES (GLCM) -----------------
        # Quantize grayscale to 16 levels for fast, invariant GLCM computation
        gray_quantized = (gray // 16).astype(np.uint8)
        glcm = graycomatrix(gray_quantized, distances=[1], angles=[0, np.pi/4, np.pi/2], levels=16, symmetric=True, normed=True)

        glcm_contrast = float(np.mean(graycoprops(glcm, 'contrast')))
        glcm_homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
        glcm_energy = float(np.mean(graycoprops(glcm, 'energy')))

        # ----------------- 3. COLOR FEATURES (HSV & Intensity Distribution) -----------------
        hsv = cv2.cvtColor(crop_std, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        mean_saturation = float(np.mean(s_channel)) / 255.0
        value_std = float(np.std(v_channel)) / 255.0

        # Uniformity: Normalized energy of the 1D intensity histogram
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist_norm = hist / (hist.sum() + 1e-5)
        color_uniformity = float(np.sum(hist_norm ** 2))

        features = np.array([
            aspect_ratio,
            solidity,
            extent,
            compactness,
            glcm_contrast,
            glcm_homogeneity,
            glcm_energy,
            mean_saturation,
            value_std,
            color_uniformity
        ], dtype=np.float32)

        return features

    def extract_batch(self, crops):
        return np.array([self.extract(c) for c in crops], dtype=np.float32)
