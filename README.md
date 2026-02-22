# Pneumonia Detection and Report Generation System

## Project Overview

This project implements a comprehensive pipeline for pneumonia detection from chest X-rays using the PneumoniaMNIST dataset (MedMNIST v2). The system combines traditional deep learning classification with state-of-the-art visual language models for automated medical report generation.

### Key Features
- **Dataset**: PneumoniaMNIST from MedMNIST v2 (28x28 grayscale chest X-rays)
- **Classification**: ResNet50 with spatial attention mechanism
- **Imbalance Handling**: Class weights and weighted sampling
- **Report Generation**: MedGemma 1.5 4B IT for automated medical reports
- **Comprehensive Evaluation**: Full classification metrics and qualitative analysis

## Project Structure

```
pneumonia-detection-system/
│
├── data/
│   └── (downloaded PneumoniaMNIST dataset)
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_resnet50_training.ipynb
│   ├── 03_spatial_attention.ipynb
│   └── 04_report_generation.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models/
│   │   ├── resnet50.py
│   │   └── spatial_attention.py
│   ├── train.py
│   ├── evaluate.py
│   └── report_generator.py
│
├── models/
│   ├── resnet50_best_model.pth
│   ├── resnet50_attention_best_model.pth
│   └── resnet50_attention_complete.pth
│
├── reports/
│   ├── task1_classification_report.md
│   └── task2_report_generation.md
│
├── results/
│   ├── generated_reports.json
│   └── sample_predictions/
│
├── requirements.txt
├── README.md
└── .gitignore
```


```bash
# Clone the repository
git clone https://github.com/yourusername/pneumonia-detection-system.git
cd pneumonia-detection-system

# Install dependencies
pip install -r requirements.txt
```

## Requirements
```bash
torch>=2.0.0
torchvision>=0.15.0
medmnist==2.2.3
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
seaborn>=0.11.0
tqdm>=4.64.0
Pillow>=9.0.0
transformers>=4.50.0
accelerate>=0.20.0
scikit-image>=0.19.0
pandas>=1.3.0
```
## Dataset: PneumoniaMNIST (MedMNIST v2)
The PneumoniaMNIST dataset consists of chest X-ray images for binary classification:

Source: MedMNIST v2 collection

Task: Binary classification (Normal vs. Pneumonia)

Image Size: 28×28 pixels (grayscale)

Channels: 1 (grayscale)

Split Sizes:

Training: 4,708 images (imbalanced)

Validation: 524 images

Test: 624 images

Class Distribution
text
Normal: 1,214 images (25.8%)
Pneumonia: 3,494 images (74.2%)
Imbalance Ratio: 1 : 2.88 (Normal : Pneumonia)
##Data Preprocessing
###Medical Image Preprocessing Pipeline
python
class MedicalImagePreprocessing:
    """Specialized preprocessing for chest X-rays"""
    
    def __call__(self, img):
        # Convert to numpy array
        img_array = np.array(img).astype(np.float32)
        
        # CLAHE for contrast enhancement
        from skimage import exposure
        img_array = exposure.equalize_adapthist(img_array / 255.0, clip_limit=0.03)
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Gaussian noise removal
        from PIL import ImageFilter
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        
        # Contrast enhancement
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
        
        return img
Data Augmentation
Augmentation strategies designed specifically for chest X-rays:

python
train_transform = transforms.Compose([
    # Preprocessing
    MedicalImagePreprocessing(),
    transforms.Resize((224, 224)),
    
    # Augmentation
    transforms.RandomHorizontalFlip(p=0.3),      # X-rays can be mirrored
    transforms.RandomRotation(degrees=5),         # Patient positioning variations
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Slight misalignments
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),  # Quality variations
    
    # Normalization
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
Model Architecture
1. Baseline ResNet50
Modified ResNet50 for grayscale input:

python
import torch.nn as nn
import torchvision.models as models

class ResNet50ForPneumonia(nn.Module):
    def __init__(self, num_classes=2, freeze_layers=True):
        super(ResNet50ForPneumonia, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify first layer for grayscale
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if freeze_layers:
            # Freeze early layers
            for name, param in self.resnet50.named_parameters():
                if 'conv1' in name or 'bn1' in name or 'layer1' in name or 'layer2' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        # Custom classifier head
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet50(x)
2. Spatial Attention Module
To help the model focus on relevant regions in chest X-rays:

python
class SpatialAttention(nn.Module):
    """Highlights important spatial regions in feature maps"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average and max pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Generate attention map
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        
        return x * attention
3. ResNet50 with Spatial Attention
python
class ResNet50WithSpatialAttention(nn.Module):
    def __init__(self, num_classes=2, freeze_layers=True):
        super(ResNet50WithSpatialAttention, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify first convolution layer to accept 1 channel
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if freeze_layers:
            # Freeze early layers
            for name, param in self.resnet50.named_parameters():
                if 'conv1' in name or 'bn1' in name or 'layer1' in name or 'layer2' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        # Insert attention modules
        self.attention1 = SpatialAttention(kernel_size=7)  # Coarse attention
        self.attention2 = SpatialAttention(kernel_size=5)  # Mid-level attention
        self.attention3 = SpatialAttention(kernel_size=3)  # Fine attention
        
        # Custom classifier head
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, return_attention=False):
        # Store attention maps if needed
        attention_maps = {}
        
        # Initial layers
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        
        # Layer 1 with attention
        x = self.resnet50.layer1(x)
        x = self.attention1(x)
        if return_attention:
            attention_maps['layer1'] = x
        
        # Layer 2 with attention
        x = self.resnet50.layer2(x)
        x = self.attention2(x)
        if return_attention:
            attention_maps['layer2'] = x
        
        # Layer 3
        x = self.resnet50.layer3(x)
        
        # Layer 4 with attention
        x = self.resnet50.layer4(x)
        x = self.attention3(x)
        if return_attention:
            attention_maps['layer4'] = x
        
        # Global average pooling and classifier
        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet50.fc(x)
        
        if return_attention:
            return x, attention_maps
        return x
Handling Class Imbalance
1. Class Weights
python
# Calculate class weights (inverse frequency)
normal_count = 1214
pneumonia_count = 3494
total_count = normal_count + pneumonia_count

normal_weight = total_count / (2 * normal_count)
pneumonia_weight = total_count / (2 * pneumonia_count)

class_weights = torch.tensor([normal_weight, pneumonia_weight]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
2. Weighted Random Sampler
python
from torch.utils.data import WeightedRandomSampler
import numpy as np

def create_weighted_sampler(dataset):
    """Creates balanced batches"""
    labels = [dataset[i][1] for i in range(len(dataset))]
    # Convert to integer if needed
    labels = [l.item() if hasattr(l, 'item') else l for l in labels]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# Usage
train_sampler = create_weighted_sampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
Training Configuration
python
# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 10

# Optimizer and scheduler
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0

early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
Evaluation Metrics
Classification Metrics
Accuracy: Overall correct predictions

Balanced Accuracy: (Sensitivity + Specificity)/2

Precision: Positive predictive value

Recall (Sensitivity): True positive rate

Specificity: True negative rate

F1-Score: Harmonic mean of precision and recall

AUC-ROC: Area under ROC curve

Confusion Matrix Example
text
                    Predicted
              Normal  Pneumonia
Actual Normal    TN       FP
       Pneumonia  FN       TP
Medical Report Generation with MedGemma
Model Selection: MedGemma 1.5 4B IT
MedGemma was chosen for several reasons:

Medical Specialization: Pre-trained on chest X-rays and medical images

Performance: 89.5% Macro F1 on MIMIC-CXR

Accessibility: Open-source on Hugging Face

Latest Version: January 2026 release with improved capabilities

Setup and Installation
python
from transformers import pipeline
import torch
from PIL import Image

# Load model
model_id = "google/medgemma-1.5-4b-it"
pipe = pipeline(
    "image-text-to-text",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
Prompting Strategies Tested
Based on research:

Basic Prompting: Simple, direct description

Structured Prompting: Organized with sections (Findings, Impression, etc.)

Few-Shot Prompting: Example-based style guidance

Clinical Decision Support: Differential diagnosis focus

Longitudinal Analysis: Comparison with prior studies

Best Performing Strategy: Structured Prompting
python
def generate_structured_report(image):
    system_prompt = """You are an expert radiologist. Provide a structured radiology report with:
1. Examination: Type of study
2. Findings: Detailed observations organized by anatomical structure
3. Impression: Summary and diagnosis
4. Recommendations: Any follow-up suggestions"""

    user_prompt = "Generate a complete radiology report for this chest X-ray following the specified format."
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt}
        ]}
    ]
    
    output = pipe(text=messages, max_new_tokens=500, temperature=0.1)
    return output[0]["generated_text"][-1]["content"]
Report Generation Pipeline
python
def batch_generate_reports(images, output_file="generated_reports.json"):
    """Generate reports for multiple images"""
    import json
    from datetime import datetime
    
    results = []
    
    for i, img_data in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}")
        
        # Generate report
        report = generate_structured_report(img_data['image'])
        
        # Store result
        result = {
            "image_id": img_data['name'],
            "true_label": img_data.get('label', 'unknown'),
            "generated_report": report,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results
Results
Classification Performance (ResNet50 + Spatial Attention)
Metric	Baseline ResNet50	With Attention	Improvement
Accuracy	85.2%	87.8%	+2.6%
Balanced Acc	82.1%	85.4%	+3.3%
Sensitivity	91.3%	92.7%	+1.4%
Specificity	73.0%	78.1%	+5.1%
AUC-ROC	0.892	0.914	+0.022
Report Generation Quality
Agreement with Ground Truth: ~85% on clear cases

Clinical Terminology: Excellent use of radiological vocabulary

Structured Output: Successfully follows prompted format

Failure Cases: Typically occur with poor image quality or subtle findings

Usage Examples
1. Training the Model
bash
# Train baseline ResNet50
python src/train.py --model resnet50 --epochs 30 --batch-size 32 --use-class-weights

# Train with spatial attention
python src/train.py --model resnet50_attention --epochs 30 --batch-size 32 --use-class-weights
2. Evaluation
bash
# Evaluate model
python src/evaluate.py --model-path models/resnet50_attention_best_model.pth

# Generate confusion matrix and ROC curve
python src/evaluate.py --model-path models/resnet50_attention_best_model.pth --visualize --output-dir results/
3. Report Generation
bash
# Generate report for single image
python src/report_generator.py --image path/to/xray.jpg --output report.txt --strategy structured

# Batch processing
python src/report_generator.py --batch --input-dir test_images/ --output-dir reports/ --strategy structured
4. Complete Pipeline
bash
# Run end-to-end pipeline
python run_pipeline.py \
    --data pneumoniamnist \
    --train \
    --evaluate \
    --generate-reports \
    --use-class-weights \
    --model resnet50_attention
Visualizations
Training Graphs
<img width="1789" height="490" alt="resnetTrainingGraph" src="https://github.com/user-attachments/assets/7452c0cb-5e16-4216-abe8-f2ed2206dfd5" />

Attention Maps

<img width="1568" height="1573" alt="atten" src="https://github.com/user-attachments/assets/1116a9c6-1200-4646-858d-670deb68154f" />

Confusion Matrix

<img width="1345" height="490" alt="resnetPerformanceTestData" src="https://github.com/user-attachments/assets/97e06eed-9570-439c-85d9-09688e5a6f93" />

ROC Curve

<img width="706" height="552" alt="roc" src="https://github.com/user-attachments/assets/8c67caeb-4709-4840-8716-b3fbce0ccc2b" />

Key Findings
Spatial Attention Improves Performance: +2.6% accuracy and +5.1% specificity

<img width="1189" height="495" alt="resenet50Vsattentation" src="https://github.com/user-attachments/assets/5e590908-7c50-45a4-8e04-461f2b58a269" />
ights Essential: Mitigates imbalance impact

Structured Prompting Best: Most clinically useful reports
Miss Classification cases
<img width="1533" height="788" alt="Failure Cases" src="https://github.com/user-attachments/assets/ce03d50a-99a4-4933-b032-e1cbe2adec5b" />
on Maps Interpretable: Model focuses on lung regions

VLM Complements CNN: Different error patterns, good for cross-validation

Future Work
Ensemble Methods: Combine multiple models for robust predictions

Grad-CAM Explanations: Add heatmap visualizations

Fine-tune MedGemma: On domain-specific radiology reports

Multi-label Classification: Detect multiple conditions

Deployment API: FastAPI endpoint for inference

DICOM Support: Handle standard medical imaging format

Contributing
Contributions are welcome! Please read our contributing guidelines and submit pull requests.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
MedMNIST Team: For the excellent dataset

Google Research: For MedGemma model

PyTorch Team: For deep learning framework

Hugging Face: For transformer libraries

Contact
For questions or collaborations, please open an issue or contact [your-email@example.com].

Last Updated: February 2026
Version: 1.0.0
