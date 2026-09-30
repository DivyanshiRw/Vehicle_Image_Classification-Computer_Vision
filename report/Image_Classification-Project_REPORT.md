# Vehicle Image Classification- Computer Vision Project

## 1. Abstract

This project develops a deep-learning based vehicle image classification system using PyTorch. The development dataset contains 3,484 images belonging to 12 vehicle categories and has noticeable class imbalance and some data-quality issues. The data was analyzed using CleanVision and appropriate preprocessing was applied using aspect-ratio-preserving letterbox resizing and ImageNet normalization. Three model experiments were performed using MobileNetV3-Small, MobileNetV3-Small with augmentation, and EfficientNet-B0. EfficientNet-B0 achieved the highest validation performance among the tested models, with 88.10% accuracy and 89.14% macro F1-score. However, MobileNetV3-Small achieved 86.80% accuracy and 87.29% macro F1-score while requiring substantially less training time in the available CPU environment. Considering this computational difference, the relatively small validation-performance gap, and the lightweight design of MobileNetV3-Small, it was selected for the final deployment pipeline. The selected model was successfully exported to ONNX and verified using ONNX Runtime.
---

# 2. Data Analysis and Cleaning

## 2.1 Dataset Overview

The development dataset contains separate training and validation sets.

| Dataset    | Number of Images |
| ---------- | ---------------: |
| Training   |            2,719 |
| Validation |              765 |
| Total      |            3,484 |

There are 12 vehicle categories:

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

The class distribution was highly uneven. For example, the training set contained 481 truck images and 448 auto-rickshaw images, while mini-bus had only 12 images and rickshaw had only 24 images.

This imbalance is important because a model can achieve good overall accuracy while performing poorly on classes with fewer examples.

## 2.2 Data Quality Analysis

The images were inspected for common data-quality problems using CleanVision.

The analysis identified issues including:

* possible duplicate or near-duplicate images
* unusual image dimensions or aspect ratios
* some blurry images

The training-set analysis flagged 143 images for one or more quality issues, while the validation-set analysis flagged 34 images.

The dataset also contains legitimate images with small resolutions and unusual aspect ratios. Therefore, an automatic quality flag was not treated as sufficient reason to remove an image.

## 2.3 Cleaning Approach

The identified issues were documented rather than aggressively removing images. Duplicate or near-duplicate images were considered the most suitable candidates for removal, while blurry images and unusual image dimensions or aspect ratios were retained because they may still represent valid real-world samples.

Since the dataset is relatively small, retaining valid but imperfect images also helps avoid unnecessarily reducing the available training data. The cleaning process therefore focused on identifying and documenting potential issues rather than removing images solely based on automatic quality flags.

The detected issues and dataset manifests are saved in the `outputs/` directory.

---

# 3. Data Preprocessing

The following preprocessing pipeline was used.

## 3.1 RGB Conversion

All images were converted to RGB format so that every input had three color channels.

## 3.2 Letterbox Resizing

Images were resized to:

**224 × 224 pixels**

while preserving their original aspect ratio.

Instead of directly stretching an image into a square, the image was first resized proportionally and then padded with black pixels where necessary.

This was chosen because directly stretching vehicle images could distort their shape and potentially make classification harder.

## 3.3 Tensor Conversion

The processed images were converted into PyTorch tensors.

## 3.4 ImageNet Normalization

The pretrained models were initialized using ImageNet weights. Therefore, ImageNet normalization was used:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

## 3.5 Training Augmentation

A separate experiment was performed with additional augmentation:

* random horizontal flip
* small affine transformations
* small translation
* small scale variation
* color jitter

The purpose was to test whether moderate variations in training images could improve generalization.

Validation images were not augmented.

---

# 4. Model Architecture

Three model configurations were evaluated.

## 4.1 MobileNetV3-Small

MobileNetV3-Small pretrained on ImageNet was used as the first model.

The original final classification layer was replaced with a new fully connected layer containing 12 output classes.

MobileNetV3-Small was selected because it is a relatively compact convolutional neural network and provides a practical starting point for image classification when training resources are limited.

## 4.2 MobileNetV3-Small with Augmentation

The same architecture was trained with additional image augmentation.

This experiment was used to determine whether augmentation could improve performance on the available dataset.

## 4.3 EfficientNet-B0

EfficientNet-B0 pretrained on ImageNet was also evaluated.

It provided another modern CNN architecture for comparison and was trained using the same general preprocessing and loss strategy.

The EfficientNet experiment was stopped after 12 epochs because the best validation result had already been obtained earlier in training and later epochs did not improve it.

---

# 5. Training and Experimentation

## 5.1 Training Configuration

The main training configuration was:

| Parameter             | Setting                |
| --------------------- | ---------------------- |
| Framework             | PyTorch                |
| Input size            | 224 × 224              |
| Batch size            | 16                     |
| Optimizer             | AdamW                  |
| Initial learning rate | 0.0001                 |
| Weight decay          | 0.0001                 |
| Loss                  | Weighted Cross-Entropy |
| Pretrained weights    | ImageNet               |
| Device                | CPU                    |

The training dataset was class-imbalanced, so class weights were incorporated into the Cross-Entropy loss.

This gives relatively greater importance to underrepresented classes during training.

## 5.2 Model Experiments

Three main experiments were conducted.

| Experiment                       | Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro mAP |
| -------------------------------- | -------: | --------------: | -----------: | -------: | --------: |
| MobileNetV3-Small                |   86.80% |          86.70% |       89.11% |   87.29% |    95.04% |
| MobileNetV3-Small + Augmentation |   88.10% |          82.70% |       89.73% |   84.96% |    94.89% |
| EfficientNet-B0                  |   88.10% |          88.85% |       89.61% |   89.14% |    95.11% |

The results show that augmentation increased overall accuracy slightly for MobileNetV3-Small, but its macro precision and macro F1 decreased.

EfficientNet-B0 produced the strongest validation metrics among the experiments.

EfficientNet-B0 achieved the best validation performance, while MobileNetV3-Small provided a practical middle ground between classification performance and computational cost. In the available CPU environment, MobileNetV3-Small completed 20 epochs in approximately 12 minutes, compared with approximately 2 hours for 12 epochs of EfficientNet-B0. The validation-performance gap was relatively small, while the difference in training time was substantial. Therefore, considering the lightweight design of MobileNetV3-Small, it was selected for the final deployment pipeline.

---

# 6. Results and Key Findings

## 6.1 Final MobileNetV3-Small Results

The final MobileNetV3-Small model achieved the following results on the development validation set:

| Metric          |     Result |
| --------------- | ---------: |
| Accuracy        | **86.80%** |
| Macro Precision | **86.70%** |
| Macro Recall    | **89.11%** |
| Macro F1-score  | **87.29%** |
| Macro mAP       | **95.04%** |

The complete classification report is available in the training notebook.


## 6.2 Learning Curves

Training and validation loss and accuracy curves were generated for the final MobileNetV3-Small model.

The curves are available at:

```text
outputs/loss_curve.png
outputs/accuracy_curve.png
```

These curves were used to monitor training progress and compare training behavior with validation performance.

## 6.3 Confusion Matrix

The confusion matrix showed that most predictions were concentrated along the diagonal, indicating that the model correctly classified a large portion of the validation images.

Several errors occurred between visually similar vehicle categories, which may be related to similarities in vehicle shape, viewpoint, size, and surrounding environment.

The confusion matrix is saved as:

```text
outputs/confusion_matrix.png
```

## 6.4 Effect of Class Imbalance

The class-wise results also demonstrate the effect of the uneven dataset.

For example, the validation set contained:

* 1 mini-bus image
* 3 tractor images
* 6 e-rickshaw images
* 12 rickshaw images

Therefore, the precision, recall and F1-score for these classes should be interpreted carefully. A single prediction can have a large effect on the metric when the number of samples is very small.

This is one reason macro-level metrics and the confusion matrix were considered along with overall accuracy.

## 6.5 ONNX Verification

After training, the MobileNetV3-Small model was exported to ONNX format.

The exported model was successfully loaded and verified using ONNX Runtime. The model produced the expected input and output dimensions:

```text
Input: 3 × 224 × 224
Output: 12 classes
Opset: 18
```

## 6.6 Key Findings

The main findings from the experiments were:

1. Transfer learning from ImageNet provided a strong starting point for the relatively small dataset.
2. Class-weighted Cross-Entropy was used to give greater importance to underrepresented classes during training.
3. Moderate augmentation improved MobileNet's accuracy from 86.80% to 88.10%, but did not improve macro F1.
4. EfficientNet-B0 achieved the strongest validation metrics among the tested configurations.
5. Visually similar vehicle categories, especially bus, truck and mini-truck, remain challenging.
6. Very small classes make class-specific metrics less stable.
7. The final MobileNet model was successfully exported to ONNX and verified using ONNX Runtime.

---

# 7. Future Work

The following improvements could be explored with more time and computational resources:

### 7.1 Dataset Improvement

* Review and correct suspicious or incorrect labels.
* Remove confirmed duplicate images.
* Collect more samples for underrepresented classes such as mini-bus, tractor, e-rickshaw, and rickshaw.
* Improve the representation of visually similar vehicle categories.

### 7.2 Better Handling of Similar Classes

The confusion matrix showed confusion between classes such as bus and truck, truck and mini-truck, and car and van. More targeted training data and augmentation could help the model distinguish these classes more effectively.

### 7.3 Further Model Tuning

Performance could be improved through systematic tuning of learning rate, batch size, weight decay, class-weighting strategies, augmentation, and backbone freezing. Learning-rate scheduling and early stopping could also be explored.

### 7.4 Model and Deployment Evaluation

Additional lightweight architectures could be tested and compared based on both validation performance and computational cost. Measuring actual inference time and memory usage would also help make a more informed deployment choice.

### 7.5 More Reliable Evaluation

A separate labeled test set would provide a better estimate of performance on unseen data. The test set should be kept separate from model selection and tuning to avoid biased evaluation.

---

# Conclusion

The project successfully developed an end-to-end vehicle image classification pipeline covering data analysis, preprocessing, model experimentation, evaluation, ONNX export and inference verification.

The final MobileNetV3-Small model achieved **86.80% validation accuracy and 87.29% macro F1-score**. EfficientNet-B0 achieved the highest validation metrics among the tested models, while MobileNetV3-Small was selected for the final ONNX deployment as a practical trade-off between validation performance and computational cost.

The results also showed clear limitations caused by class imbalance, visually similar vehicle categories, and potential data-quality issues. These limitations were documented rather than hidden.

The project artifacts include the exported ONNX model, class mapping, notebooks, evaluation plots, requirements file, README, and project report. The development dataset and dataset-specific files are not included in the repository.
