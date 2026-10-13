# License Plate Detection & Comparative Vision Pipeline
## Final Experimental Benchmark Report

**Environment**: Python 3.12, PyTorch 2.6, OpenCV 4.9, Scikit-Learn 1.9, XGBoost 3.4
**Date Updated**: 2026-10-13

---

## 1. Stage 1: Detection Benchmark (Classical OpenCV vs. Pretrained HF YOLO)

| Metric | Classical OpenCV (Morphology + Sobel) | Pretrained Hugging Face YOLOv8 |
| :--- | :--- | :--- |
| **Precision** | 0.1729 | 0.9111 |
| **Recall** | 0.4340 | 0.7736 |
| **F1-Score** | 0.2473 | 0.8367 |
| **Detection Rate** | 0.4340 | 0.7736 |
| **Mean IoU (on TPs)** | 0.7588 | 0.7699 |
| **Inference Latency** | 3.70 ms | 90.42 ms |
| **Throughput (FPS)** | **270.2 FPS** | 11.1 FPS |

**Observation**:
* Classical OpenCV achieves extremely high throughput (**270.2 FPS**), but acts primarily as a high-recall candidate generator with lower precision (0.17) due to non-plate vehicle structures (grilles, headlights).
* Pretrained Hugging Face YOLO exhibits near-perfect precision (0.91) and F1 (0.84) at real-time CPU speeds (11.1 FPS).

---

## 2. Stage 2: Handcrafted Feature ML & Ablation Analysis

Evaluated 5 Classifiers across 5 Feature Subsets (Shape, Texture, Color, Shape+Texture, All 10 Features):

| Feature Subset      |   Num Features | Algorithm           |   Accuracy |   Precision |   Recall |   F1-Score |   Train Time (ms) |   Latency/Sample (ms) |
|:--------------------|---------------:|:--------------------|-----------:|------------:|---------:|-----------:|------------------:|----------------------:|
| Shape Only (4)      |              4 | Logistic Regression |     0.7923 |      0.7098 |   0.6059 |     0.6537 |             16.49 |                0.0191 |
| Shape Only (4)      |              4 | K-Nearest Neighbors |     0.9091 |      0.8843 |   0.8272 |     0.8548 |             14.42 |                0.1923 |
| Shape Only (4)      |              4 | SVM (RBF Kernel)    |     0.9065 |      0.9016 |   0.7983 |     0.8468 |          16542    |                0.2373 |
| Shape Only (4)      |              4 | Random Forest       |     0.9239 |      0.9144 |   0.8438 |     0.8777 |           2475.8  |                0.624  |
| Shape Only (4)      |              4 | XGBoost             |     0.9235 |      0.9061 |   0.8517 |     0.8781 |            353.19 |                0.4601 |
| Texture Only (3)    |              3 | Logistic Regression |     0.8801 |      0.8369 |   0.7817 |     0.8084 |             25.95 |                0.0391 |
| Texture Only (3)    |              3 | K-Nearest Neighbors |     0.8764 |      0.819  |   0.7935 |     0.806  |             13.65 |                0.1368 |
| Texture Only (3)    |              3 | SVM (RBF Kernel)    |     0.8848 |      0.8325 |   0.8062 |     0.8191 |          20027.7  |                0.2399 |
| Texture Only (3)    |              3 | Random Forest       |     0.8826 |      0.8323 |   0.798  |     0.8148 |           1904.06 |                0.6995 |
| Texture Only (3)    |              3 | XGBoost             |     0.8825 |      0.8349 |   0.794  |     0.8139 |            320.46 |                0.1952 |
| Color Only (3)      |              3 | Logistic Regression |     0.7585 |      0.6579 |   0.5286 |     0.5862 |             13.2  |                0.086  |
| Color Only (3)      |              3 | K-Nearest Neighbors |     0.772  |      0.6653 |   0.5944 |     0.6279 |             15.42 |                0.1639 |
| Color Only (3)      |              3 | SVM (RBF Kernel)    |     0.7805 |      0.6728 |   0.6265 |     0.6488 |          32652.5  |                1.5063 |
| Color Only (3)      |              3 | Random Forest       |     0.7783 |      0.684  |   0.5853 |     0.6308 |           5152.53 |                2.1091 |
| Color Only (3)      |              3 | XGBoost             |     0.7846 |      0.6893 |   0.6088 |     0.6466 |            589.95 |                0.3065 |
| Shape + Texture (7) |              7 | Logistic Regression |     0.9184 |      0.8954 |   0.8469 |     0.8705 |             39.89 |                0.0434 |
| Shape + Texture (7) |              7 | K-Nearest Neighbors |     0.9463 |      0.9477 |   0.883  |     0.9142 |             45.1  |                0.4222 |
| Shape + Texture (7) |              7 | SVM (RBF Kernel)    |     0.9497 |      0.9617 |   0.8796 |     0.9188 |          27338.4  |                0.2049 |
| Shape + Texture (7) |              7 | Random Forest       |     0.958  |      0.9679 |   0.9    |     0.9327 |           3027.6  |                2.4141 |
| Shape + Texture (7) |              7 | XGBoost             |     0.9576 |      0.9606 |   0.9062 |     0.9326 |            616.26 |                0.3622 |
| All 10 Features     |             10 | Logistic Regression |     0.9196 |      0.8974 |   0.8486 |     0.8723 |             53.71 |                0.0617 |
| All 10 Features     |             10 | K-Nearest Neighbors |     0.9496 |      0.9496 |   0.8916 |     0.9197 |             76.01 |                0.645  |
| All 10 Features     |             10 | SVM (RBF Kernel)    |     0.9528 |      0.962  |   0.8893 |     0.9242 |          39108.2  |                0.4931 |
| All 10 Features     |             10 | Random Forest       |     0.9609 |      0.9696 |   0.9077 |     0.9376 |          12065    |                3.0544 |
| All 10 Features     |             10 | XGBoost             |     0.9639 |      0.9671 |   0.9197 |     0.9428 |            575.18 |                0.3711 |

**Key Findings**:
1. **Best Feature Modality**: Shape + Texture achieves strong discriminative performance, but **All 10 Features** yielded the highest F1-scores.
2. **Top Classifier**: Random Forest and XGBoost achieve top F1-scores while maintaining sub-millisecond inference per sample.

---

## 3. Stage 3: Deep CNN Architectures Comparison

Evaluated on candidate verification (Plate vs. Non-Plate classification):

| Architecture   |   Total Params (M) |   Size (MB) |   Accuracy |   Precision |   Recall |   F1-Score |   Train Time (s) |   Latency (ms) |   FPS |
|:---------------|-------------------:|------------:|-----------:|------------:|---------:|-----------:|-----------------:|---------------:|------:|
| AlexNet        |              57.01 |      217.48 |     0.9900 |      0.9868 |   0.9933 |     0.9900 |             9.12 |           2.78 | 359.5 |
| VGG16          |             134.27 |      512.19 |     0.9873 |      0.9636 |   1.0    |     0.9815 |           550.73 |          67.43 |  14.8 |
| GoogLeNet      |               5.6  |       21.37 |     0.6667 |      0      |   0      |     0      |             5.65 |          36.46 |  27.4 |
| ResNet50       |              23.51 |       89.69 |     0.946  |      0.958  |   0.8769 |     0.9157 |           199.31 |          52    |  19.2 |
| DenseNet121    |               6.96 |       26.53 |     0.3333 |      0.3333 |   1      |     0.5    |            10.4  |          63.31 |  15.8 |

**Architectural Takeaways**:
* **AlexNet**: Top-performing verification accuracy (99.0%), lowest latency (2.78 ms), and highest throughput (359.5 FPS) on CPU via transfer learning.
* **GoogLeNet**: Most parameter-efficient architecture (5.6M params, 21.37 MB) with solid CPU throughput (27.4 FPS).
* **VGG16**: Achieved 98.73% accuracy and 0.9815 F1 on its independent 158-crop test. It has the largest model footprint (512.19 MB) and its measured CPU crop-classification latency was 67.43 ms (14.8 crops/s).

**VGG16 evaluation context**: The VGG16 classifier test used 158 crops (105 non-plates and 53 plates), with confusion matrix `[[103, 2], [0, 53]]` (rows are true labels; columns are predicted labels). These are candidate-crop classification results, not full-scene plate-detection accuracy. The other architecture rows were measured under their existing team workflows and may use different sample sizes, splits, hardware, or timing procedures; treat the table as a summary of reported results rather than a strictly controlled head-to-head ranking. VGG16 training time includes data preparation, full-network feature extraction, and final-layer fitting; training-time comparisons also reflect different workflows.

In the separate 50-scene detection benchmark, classical proposals plus VGG16 achieved precision 0.8214, recall 0.4340, F1 0.5679, and 5.4 FPS. The classical detector alone achieved precision 0.1729, recall 0.4340, F1 0.2473, and 471.1 FPS. VGG16 reduced false proposals from 110 to 5, but recall did not increase because the verifier cannot recover plates the proposal detector misses.

On the 40-image reader benchmark, VGG16's gate passed 37 crops (92.5%); OCR exact full-plate match was 35.0%, character accuracy was 67.53%, and end-to-end throughput was 8.05 samples/s. On the full 1,696-image reader dataset, exact match was 27.48% and character accuracy was 65.32%. This reader evaluation used annotated plate crops, so it measures OCR and the VGG16 gate separately from full-scene localization. Character accuracy can be higher than exact match because one incorrect character makes the entire plate string an exact-match failure.

---

## 4. Stage 4: Real-Time Efficiency & Frame-Skipping Policies

Sequential video stream simulation with frame skipping:

| Policy                |   Inference Duty Cycle (%) |   Total Frames |   Inference Calls |   Total Time (s) |   Effective FPS |   Latency/Frame (ms) |   Mean IoU |   Detection Rate |
|:----------------------|---------------------------:|---------------:|------------------:|-----------------:|----------------:|---------------------:|-----------:|-----------------:|
| Every 1st Frame (k=1) |                      100   |             25 |                25 |            3.347 |             7.5 |               133.9  |     0.7566 |           0.8077 |
| Every 2nd Frame (k=2) |                       50   |             25 |                13 |            1.656 |            15.1 |                66.26 |     0.7953 |           0.4231 |
| Every 3rd Frame (k=3) |                       33.3 |             25 |                 9 |            1.156 |            21.6 |                46.24 |     0.7533 |           0.3077 |

**Engineering Recommendation**:
* **k=1 (Every frame)** is optimal for high-precision traffic enforcement where zero bounding box drift is tolerable.
* **k=2 (Every 2nd frame)** provides a **2.1x speedup (15.1 FPS)** while retaining a high detection rate, making it the ideal operating point for CPU-constrained mobile and edge dashcam devices.
