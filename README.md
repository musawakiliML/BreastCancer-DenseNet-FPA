# DenseNet-FPA: Integrating DenseNet and Flower Pollination Algorithm for Breast Cancer Histopathology Image Classification

---

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

This repository contains code for the paper, "DenseNet-FPA: Integrating DenseNet and Flower Pollination Algorithm for Breast Cancer Histopathology Image Classification, Musa Adamu Wakili, Harisu Abdullahi Shehu, Mahdi
Abdollahi, Badamasi Imam Ya’U, Md Haidar Sharif, Huseyin Kusetogullari", which is currently under review.

## 🚀 Goal

The goal is to improve classification performance and model generalizability by combining:

- ✅ DenseNet as a powerful image feature extractor
- 🌸 FPA as a nature-inspired metaheuristic for selecting the most informative features
- 🤖 Evaluation on BreakHis and BACH datasets trained on selected features

---

## Installation and Usage

1. Clone this repository:

    ```bash
        git clone https://github.com/musawakiliML/Breastcancer-DenseNet-FPA.git
        cd Breastcancer-DenseNet-FPA
    ```

2. Install Dependencies

    ```bash
        pip install -r requirements.txt
    ```

3. Run the main.py script

    ```python
        python3 main.py
    ```

---

## 📂 Folder Structure

``` markdown
    breastcancer_densenet_fpa/
    ├── data/                   # dataloader function for your data
    ├── models/                 # models class and methods(DenseNet and FPA implementation)
    ├── main.py                 # Full training pipeline
    ├── requirements.txt        # Project libraries
    └── README.md               # Information about the project
```

---

## 📊 Dataset

**BreakHis (Breast Cancer Histopathological Database)**  

- URL: [https://www.kaggle.com/datasets/ambarish/breakhis](https://www.kaggle.com/datasets/ambarish/breakhis)
- Classes: Benign (A, F, PT, TA) and Malignant (DC, LC, MC, PC)
- Images: 7,909 PNGs at 40x, 100x, 200x, 400x magnifications

> Note: This project uses selected magnifications and applies class-wise image augmentation and resizing to (224, 224).

**BACH (Breast Cancer Histology images)**  

- URL: <https://www.kaggle.com/datasets/truthisneverlinear/bach-breast-cancer-histology-images/data>
- Classes: Benign, In Situ, Invasive, and Normal
- Images: 400 Tif

---

## 🔍 Key Components

### Data Preprocessing

- Stratified train/val/test split (Benign vs Malignant)
- Augmentation: Rotation, flipping

### Feature Extraction

- DenseNet (pretrained on ImageNet)
- Output: High-dimensional feature vectors

### Feature Selection

- Flower Pollination Algorithm (FPA)
- Goal: Reduce feature space while preserving accuracy

### Classification

- DenseNet classifier
- Evaluation: Accuracy, Precision, Recall, F1 Score

---

## Citation

If you use this code in your research, please cite our paper:

```markdown
    @article{yourcitation,
    title={DenseNet-FPA: Integrating DenseNet and Flower Pollination Algorithm for Breast Cancer Histopathology Image Classification},
    author={Musa Adamu Wakili, Harisu Abdullahi Shehu, Mahdi Abdollahi, Badamasi Imam Ya’U, Md Haidar Sharif, Huseyin Kusetogullari},
    journal={IEEE Access},
    volume={Access-2025-18278},
    number={Y},
    pages={Z},
    year={2025},
    publisher={IEEE}
    }
```

## Note

Contributions to the repository are welcome, and any questions can be sent to musaadamuw@gmail.com.

We appreciate your interest and hope that this code proves valuable in your research endeavors.

Best regards,

The Authors