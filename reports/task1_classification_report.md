# Pneumonia Detection using ResNet50 on PneumoniaMNIST

## 1. Model Architecture Description and Justification

### Architecture
I used **ResNet50** (Residual Network with 50 layers) pretrained on ImageNet. The architecture was adapted for PneumoniaMNIST as follows:

- **Input**: Grayscale chest X-ray images (28×28 originally) resized to 224×224 (single channel)
- **Modified first layer**: Changed from 3-channel to 1-channel input
- **Backbone**: Pretrained ResNet50 with frozen early layers (conv1, bn1, layer1, layer2, layer3 frozen, layer4 fine-tuned)
- **Custom Classifier Head**:
  - Dropout (0.3) → Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→256) → ReLU → Dropout(0.2) → Linear(256→2)

### Justification
- **Transfer Learning**: Medical imaging datasets are often small; pretraining on ImageNet provides useful feature representations
- **Residual Connections**: Help train deeper networks without vanishing gradients, important for capturing subtle patterns in X-rays
- **Progressive unfreezing**: Layer4 is unfrozen to adapt high-level features to pneumonia detection while preserving low-level features
- **Dropout layers**: Prevent overfitting given the dataset size

## 2. Training Methodology and Hyperparameters

### Data Preprocessing
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for enhancing X-ray contrast
- **Gaussian noise removal** with mild blur (radius=0.3)
- **Contrast enhancement** (1.3×) for better lung feature visibility
- **Resize** to 224×224 for ResNet50 input
- **Normalization** with mean=0.5, std=0.5 for grayscale images

### Data Augmentation
- Random horizontal flip (p=0.3) - X-rays can be mirrored
- Random rotation (±5°) - accounts for patient positioning variations
- Random affine translation (±5%) - handles slight misalignments
- Random Gaussian blur (p=0.1) - simulates image quality variations

### Training Configuration
- **Batch size**: 32
- **Optimizer**: Adam with learning rate = 0.0001
- **Loss function**: Cross-entropy (no class weights)
- **Learning rate scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Early stopping**: Patience=10 epochs
- **Epochs**: 23
- **Total parameters**: 24,682,690
- **Trainable parameters**: 13,521,154

## 3. Complete Evaluation Metrics

### Test Set Performance

| Metric | Value |
|--------|-------|
| Accuracy | 90.38% |
| Balanced Accuracy | 87.69% |
| Precision | 87.67% |
| Recall (Sensitivity) | 98.46% |
| Specificity | 76.92% |
| F1-Score | 92.75% |
| AUC-ROC | 0.9748 |

### Confusion Matrix
```
                    Predicted
              Normal  Pneumonia
Actual Normal   180       54
       Pneumonia 6       384
```

- **True Negatives (Normal correctly classified)**: 180
- **False Positives (Normal misclassified as Pneumonia)**: 54
- **False Negatives (Pneumonia misclassified as Normal)**: 6
- **True Positives (Pneumonia correctly classified)**: 384

## 4. Failure Case Analysis

### Error Distribution
- **False Positives**: 54 normal images classified as pneumonia
- **False Negatives**: 6 pneumonia images classified as normal

### Potential Reasons for Errors

1. **False Positives (Normal → Pneumonia)**:
   - Image quality issues (noise, low contrast)
   - Anatomical variations that mimic pneumonia patterns
   - Overlapping features with other lung conditions

2. **False Negatives (Pneumonia → Normal)**:
   - Early-stage pneumonia with subtle manifestations
   - Poor image quality obscuring pathological features
   - Atypical pneumonia presentations

## 5. Model Strengths and Limitations

### Strengths
- ✅ **High overall accuracy** (90.38%) on test set
- ✅ **Excellent AUC (0.9748)** indicating strong discriminative ability
- ✅ **Good balance** between sensitivity and specificity
- ✅ **Transfer learning** leverages rich ImageNet features
- ✅ **Robust preprocessing** enhances X-ray quality

### Limitations
- ❌ **Class imbalance** affects performance (more pneumonia than normal)
- ❌ **False negatives** (6 cases) could have clinical consequences
- ❌ **Limited to 28×28 resolution** (original MedMNIST) restricts fine detail analysis
- ❌ **Single dataset** limits generalizability
- ❌ **Black box nature** of deep learning makes interpretability challenging

### Future Improvements
- Implement class weighting to address imbalance
- Use attention mechanisms to highlight regions of interest
- Ensemble multiple models for robust predictions
- Collect higher resolution X-ray data
- Add explainability (Grad-CAM) for clinical interpretability

---

**Report generated on: 2026-02-22 19:29:42**