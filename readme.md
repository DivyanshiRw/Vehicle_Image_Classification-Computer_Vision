# Vehicle Image Classification- Computer Vision

## 1. Project Overview

This project implements a vehicle image classification system using PyTorch.

The model classifies images into 12 vehicle categories:

* auto-rickshaw
* bicycle
* bus
* car
* e-rickshaw
* mini-bus
* mini-truck
* motorcycle
* rickshaw
* tractor
* truck
* van

The project includes data exploration, preprocessing, model training, evaluation, model comparison, ONNX export, and ONNX Runtime verification.

---

## 2. Project Structure

```text
CV_task/
│
├── models/
│   └── vehicle_classifier.onnx            (exported ONNX model)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb          (dataset analysis and exploration)
│   └── 02_model_training.ipynb            (preprocessing, training and evaluation)
│
├── outputs/
│   ├── classes.txt                        (class label mapping)
│   ├── loss_curve.png                     (training/validation loss)
│   ├── accuracy_curve.png                 (training/validation accuracy)
│   ├── confusion_matrix.png               (validation confusion matrix)
│   ├── model_comparison.png               (comparison of model experiments)
│   ├── cleaned_train_manifest.csv         (training dataset manifest)
│   ├── cleanvision_train_issues.csv       (detected training data-quality issues)
│   ├── validation_manifest.csv            (validation dataset manifest)
│   └── cleaning_exclusions.csv            (record of excluded images)
│
├── report/                                 (project report)
│
├── src/
│   ├── export_onnx.py                     (exports trained model to ONNX)
│   └── verify_model.py                    (ONNX Runtime inference verification)
│
│
├── readme.md                              (project setup and usage instructions)
└── requirements.txt                       (Python package dependencies)
```

---

## 3. Environment Setup


Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The project was developed and tested using CPU execution.

---

## 4. Dataset

The dataset used during development contains 3,484 images across 12 vehicle categories:

- Training images: 2,719
- Validation images: 765
- Number of classes: 12

The dataset is not included in this repository.

The project was developed using a separate local dataset, and the repository contains only the code, model artifacts, evaluation results, and documentation.

---

## 5. Preprocessing

Images are converted to RGB and resized using letterbox preprocessing to preserve their aspect ratio.

Final input size:

```text
224 × 224 × 3
```

Black padding is added where required to obtain a square image.

The following ImageNet normalization is applied:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

Training augmentation used in one experiment included horizontal flipping, small affine transformations, and color jitter.

---

## 6. Training

The main experiments were performed using pretrained CNN architectures:

* MobileNetV3-Small
* MobileNetV3-Small + augmentation
* EfficientNet-B0

Training used weighted Cross-Entropy Loss to account for class imbalance.

The detailed training procedure and experiment results are available in:

```text
notebooks/02_model_training.ipynb
```

---

## 7. Model Evaluation

The final MobileNetV3-Small validation performance was:

| Metric          |  Score |
| --------------- | -----: |
| Accuracy        | 86.80% |
| Macro Precision | 86.70% |
| Macro Recall    | 89.11% |
| Macro F1        | 87.29% |
| Macro mAP       | 95.04% |

The confusion matrix and training curves are available in the `outputs/` directory.

An additional EfficientNet-B0 experiment achieved a validation accuracy of 88.10% and macro F1 of 89.14%.

MobileNetV3-Small was selected for the final ONNX deployment as a practical balance between classification performance and computational cost in the available CPU environment.

---

## 8. ONNX Deployment

The selected MobileNetV3-Small model was exported to ONNX format and verified using ONNX Runtime.

The exported model is available at:

models/vehicle_classifier.onnx

Model specification:

Input: 3 × 224 × 224
Output: 12 classes
Opset: 18

---

## 9. ONNX Verification

The exported ONNX model was verified using ONNX Runtime to confirm that the model loads correctly and produces predictions with the expected input and output dimensions.

The ONNX model expects preprocessed RGB images of size 224 × 224.

Image preprocessing is not embedded inside the ONNX model and consists of:

- RGB conversion
- aspect-ratio-preserving letterbox resizing
- tensor conversion
- ImageNet normalization

---

## 10. Reproducibility

The dataset is not included in this repository.

The notebooks contain the data exploration, preprocessing, training, and evaluation workflow used during development.

The exported ONNX model is provided for inspection and deployment experimentation.

---

## 11. Notes

* The dataset contains significant class imbalance, particularly for mini-bus, tractor, e-rickshaw, and rickshaw.
* Some visually similar vehicle categories show confusion, particularly among bus, truck, and mini-truck.
* All dataset images were retained for training; identified data-quality issues were documented rather than manually excluded.
- The dataset used during development is not included in the repository.
- Validation metrics reported in this project are based on the development validation set and should not be interpreted as performance on an independent test set.

The project includes data exploration, preprocessing, model training, evaluation, model comparison, ONNX export, and ONNX Runtime verification.