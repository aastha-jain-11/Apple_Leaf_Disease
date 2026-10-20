# Apple Leaf Disease Classification and Severity Grading Using Deep Learning

An end-to-end deep learning framework for **apple leaf disease classification, lesion segmentation, and quantitative severity estimation** using ConvNeXt V2 and SegFormer.

The project aims to identify common apple leaf diseases, estimate the affected leaf area, and assign severity grades to support disease assessment in precision agriculture.

## Project Overview

The system combines:

- **ConvNeXt V2 Base** — four-class disease classification
- **SegFormer (MiT-B0)** — semantic segmentation of leaf and lesion regions
- **Image processing** — lesion area and lesion-count extraction
- **Regression models** — continuous severity estimation
- **Severity grading** — conversion of severity estimates into four grades

## Disease Classes

| Class | Description |
|---|---|
| Apple Scab | Fungal disease producing dark lesions |
| Black Rot | Disease associated with dark necrotic regions |
| Cedar Apple Rust | Fungal disease producing orange or yellow lesions |
| Healthy | Leaves without visible disease symptoms |

## System Architecture

```text
Apple Leaf Image
       |
       +-----------------------+
       |                       |
       v                       v
ConvNeXt V2 Base          SegFormer
Disease Classification   Leaf and Lesion Segmentation
       |                       |
Disease Label             Feature Extraction
                               |
                        Severity Regression
                               |
                        Severity Grade
```

## Dataset

### Classification dataset

The final classification dataset contains **3,164 unique images** after duplicate removal.

| Split | Images |
|---|---:|
| Training | 2,214 |
| Validation | 475 |
| Test | 475 |
| **Total** | **3,164** |

### Severity dataset

The manually annotated severity subset contains **693 images**.

| Class | Images |
|---|---:|
| Apple Scab | 199 |
| Black Rot | 195 |
| Cedar Rust | 200 |
| Healthy | 99 |
| **Total** | **693** |

Segmentation masks use three pixel classes: background (0), leaf tissue (1), and lesion (2).

The stratified severity split contains 485 training, 104 validation, and 104 test images.

## Methodology

### Severity Index

Ground-truth severity is calculated from manually annotated masks:

\[
SI = 100 \times
\frac{N_{\mathrm{lesion}}}
{N_{\mathrm{leaf}} + N_{\mathrm{lesion}}}
\]

Background pixels are excluded. Healthy leaves have a severity index of 0%.

### Regression Experiments

| Model | Features |
|---|---|
| R0 | Lesion area percentage |
| R1 | Lesion area percentage + lesion count |
| R2 | Area percentage + count + lesion-size statistics |

### Severity Grades

| Grade | Severity Index |
|---|---|
| Grade 0 | 0–5% |
| Grade 1 | >5–15% |
| Grade 2 | >15–30% |
| Grade 3 | >30–100% |

These are experimental, dataset-defined thresholds and require independent validation before being treated as a standardized plant-pathology scale.

## Latest Results

### Disease Classification

ConvNeXt V2 Base (`C_FULL`) achieved the following results on the 475-image test set.

| Metric | Result |
|---|---:|
| Accuracy | 99.58% |
| Macro Precision | 99.48% |
| Macro Recall | 99.80% |
| Macro F1-score | 99.64% |

### Semantic Segmentation

| Metric | Result |
|---|---:|
| Background Dice | 0.9919 |
| Leaf Dice | 0.9714 |
| Lesion Dice | 0.7110 |
| Mean Dice | 0.8914 |
| Mean IoU | 0.8266 |

Lesion segmentation remains the main segmentation challenge.

### Severity Regression

Results on the 104-image test split. MAE and RMSE are measured in severity-index percentage points.

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| R0 — Area only | 3.7248 | 10.1333 | 0.5997 |
| R1 — Area + count | **2.4238** | **5.4596** | **0.8838** |
| R2 — Area + count + size | 2.4776 | 5.7123 | 0.8728 |

R1 currently performs best on the reported test metrics.

### Four-Grade Severity Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 85.58% |
| Macro F1-score | 0.7862 |
| Weighted F1-score | 0.8521 |
| Cohen's Kappa | 0.7583 |
| Grade MAE | 0.1538 |

Grade 2 is the weakest category in the current test results. Grade 3 has only four test samples, so its score should be interpreted cautiously.

**Evaluation note:** Before reporting the severity results as definitive held-out performance, verify that SegFormer was trained without the 104 test images and that segmentation evaluation used the same stratified test split.

## Technologies Used

| Component | Technology |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Classification | ConvNeXt V2 |
| Segmentation | SegFormer |
| Model Libraries | timm, Transformers |
| Image Processing | OpenCV, Pillow |
| Data Processing | NumPy, Pandas |
| Machine Learning | scikit-learn |
| GPU Acceleration | NVIDIA CUDA |
| Annotation | LabelMe |

## Project Structure

```text
apple_disease_project/
├── src/
│   └── severity/
│       ├── labelme_to_masks.py
│       ├── calculate_severity_index.py
│       ├── create_severity_stratified_split.py
│       ├── train_segformer.py
│       ├── evaluate_segmentation.py
│       ├── extract_severity_features.py
│       ├── train_severity_regression_stratified.py
│       └── evaluate_severity_grades_stratified.py
├── models/
│   ├── convnext/
│   └── segformer/
├── outputs/
│   ├── classification/
│   └── severity/
├── requirements.txt
├── .gitignore
└── README.md
```

*This structure summarizes the main pipeline; update it to match the files committed to the repository. Datasets and model checkpoints may need to be downloaded or generated separately.*

## Installation

```powershell
git clone <repository-url>
cd apple_disease_project

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a compatible CUDA-enabled PyTorch installation for GPU training.

## Running the Severity Pipeline

Run commands from the repository root in this order:

```powershell
python src\severity\labelme_to_masks.py
python src\severity\calculate_severity_index.py
python src\severity\create_severity_stratified_split.py
python src\severity\train_segformer.py
python src\severity\evaluate_segmentation.py
python src\severity\extract_severity_features.py
python src\severity\train_severity_regression_stratified.py
python src\severity\evaluate_severity_grades_stratified.py
```

If annotations or splits change, rerun the dependent stages so all outputs correspond to the same dataset version.

## Limitations

- Lesion segmentation is less accurate than background and leaf segmentation.
- Severity-grade performance is affected by class imbalance, particularly in Grades 2 and 3.
- Predicted severity depends on annotation quality and lesion-boundary accuracy.
- The current thresholds require independent validation.
- Results from one dataset do not guarantee generalization to other orchards, cultivars, cameras, or environmental conditions.

## Project Team

- Anushka Chaudhary
- Aastha Jain
- Sai Gurunath Tandel

**Project Guide:** Dr. Suganthi K  
**Institution:** Vellore Institute of Technology (VIT)

## References

1. Bi et al. (2025). *Apple leaf disease severity grading based on deep learning and the DRL-Watershed algorithm.* Scientific Reports. [DOI: 10.1038/s41598-025-15246-8](https://doi.org/10.1038/s41598-025-15246-8)
2. Kumar et al. (2025). *A Novel Approach for Classification and Detection of Apple Leaf Disease Using Enhanced RBVT-Net With Transfer Learning and YOLOv7.* IEEE Access. [DOI: 10.1109/ACCESS.2025.3596451](https://doi.org/10.1109/ACCESS.2025.3596451)
3. Dong et al. (2024). *Apple Leaf Disease Diagnosis Based on Knowledge Distillation and Attention Mechanism.* IEEE Access. [DOI: 10.1109/ACCESS.2024.3397329](https://doi.org/10.1109/ACCESS.2024.3397329)



---

*Developed as an undergraduate deep learning research project focused on apple leaf disease classification and quantitative severity assessment.*
