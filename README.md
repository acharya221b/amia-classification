# AMIA Thoracic Disease Classification using ResNet50 and DenseNet121

This repository contains the Jupyter Notebooks and supplementary materials for the "AMIA Thoracic Disease Classification" project, as detailed in the accompanying report. The goal of this project is to develop and compare deep learning models for multi-class, multi-label classification of 14 common thoracic abnormalities from chest X-ray images.

This automated system aims to provide a valuable "second opinion" for radiologists, helping to improve diagnostic accuracy and efficiency.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Methodology](#-methodology)
    - [Data Preprocessing](#data-preprocessing)
    - [Model 1: Transfer Learning with ResNet50](#model-1-transfer-learning-with-resnet50)
    - [Model 2: MONAI DenseNet121](#model-2-monai-densenet121)
4. [Results](#-results)
    - [ResNet50 Performance](#resnet50-performance)
    - [DenseNet121 Performance](#densenet121-performance)
5. [Setup and Usage](#-setup-and-usage)
6. [File Structure](#-file-structure)
7. [Future Improvements](#-future-improvements)
8. [References](#-references)

---

## 🔭 Project Overview

The project tackles the challenge of classifying thoracic diseases from chest radiographs sourced from the AMIA dataset. Since a single X-ray can exhibit multiple abnormalities, this is framed as a **multi-label classification task**. We explore two powerful Convolutional Neural Network (CNN) architectures: **ResNet50** and **DenseNet121**, leveraging transfer learning and the MONAI framework for medical imaging.

The notebook documents the entire pipeline, from data exploration and preprocessing to model training, evaluation, and a comparative analysis of the results.

## 💾 Dataset

The project uses the **AMIA Dataset**, which is a subset of the larger VinBig Chest X-ray dataset.

- **Source**: [AMIA Public Challenge 2024 on Kaggle](https://www.kaggle.com/competitions/amia-public-challenge-2024)
- **Content**: The dataset consists of thousands of chest X-ray scans annotated by experienced radiologists.
- **Labels**: There are 14 distinct thoracic abnormalities and a "No finding" class, making a total of 15 unique classes for the multi-label classification task.

## 🛠️ Methodology

### Data Preprocessing
Before training, the data undergoes several preprocessing steps:
- **Image Rescaling**: All DICOM images are converted to PNG and rescaled to a uniform size of **1024x1024 pixels**.
- **Class Annotations**: The raw annotations (bounding boxes per finding) are processed to create a **one-hot encoded vector** for each image, representing the presence or absence of each of the 15 classes.
- **Data Augmentation**: We use the **Albumentations** library to apply transformations like random cropping, horizontal flipping, and brightness adjustments. This helps generate new training samples, improve model generalization, and reduce overfitting.

### Model 1: Transfer Learning with ResNet50
Our first approach utilizes a pre-trained ResNet50 model.
- **Architecture**: A 50-layer deep CNN known for its residual connections that mitigate the vanishing gradient problem.
- **Transfer Learning**:
    1. The first convolutional layer was modified to accept single-channel (grayscale) images instead of 3-channel RGB images.
    2. The final fully connected layer was replaced with a new one having 15 output units (for our classes).
    3. We initially froze the pre-trained layers and trained only the modified layers, then unfroze the entire model for fine-tuning.
- **Training Details**:
    - **Loss Function**: `BCEWithLogitsLoss`, which is suitable for multi-label classification.
    - **Optimizer**: Adam with a learning rate of `1e-5`.
    - **Epochs**: 10 epochs.

### Model 2: MONAI DenseNet121
Our second model is a DenseNet121 architecture from the **MONAI** framework, which is specifically designed for medical imaging tasks.
- **Architecture**: DenseNet is unique for its dense connectivity, where each layer is connected to every other layer in a feed-forward fashion. This promotes feature reuse and improves information flow.
- **Training Details**:
    1. The model was initialized to accept 1 input channel and produce 15 output channels.
    2. We used the same `BCEWithLogitsLoss` function and Adam optimizer with a `1e-5` learning rate.
    - **Epochs**: 7 epochs.

## 📊 Results

### ResNet50 Performance
The ResNet50 model showed rapid initial improvement, but its performance plateaued quickly.
- **Accuracy**: Both training and validation accuracy stagnated at around **87%-88%**.
- **Observations**: The model did not appear to overfit, but it also failed to generalize further with more training. This suggests it may have converged to a local optimum.

![ResNet50 Accuracy and Loss](https://github.com/user/repo/raw/main/results/resnet_results.png)

### DenseNet121 Performance
The DenseNet121 model demonstrated more stable and effective learning.
- **Accuracy**: The model achieved a final accuracy of **90.2%** with a loss of 0.2838 after 7 epochs.
- **Observations**: The learning process was steady, with no signs of overfitting. The dense architecture's feature reuse seemed beneficial. Given the low learning rate, more training epochs could potentially yield even better results.

![DenseNet121 Accuracy and Loss](https://github.com/user/repo/raw/main/results/monai_results.png)
---

## ⚙️ Setup and Usage

To replicate this project, follow the steps below.

### 1. Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Familiarity with PyTorch

### 2. Installation
**a. Clone the repository:**
```bash
git clone https://github.com/acharya221b/amia-classification.git
cd amia-classification
```
**b. Install Required Packages**
```bash
pip install -r requirements.txt
```
**c. Download the Dataset**

1.  Download the AMIA dataset from the **[Kaggle competition page](https://www.kaggle.com/competitions/amia-public-challenge-2024)**.
2.  Unzip the downloaded files and place them in a `data/` directory within your project's root folder. The final structure should match the one described in the [File Structure](#-file-structure) section below.

### 3. Running the Notebook

Launch Jupyter Notebook or JupyterLab and open the following files. You can then run the cells sequentially to execute the entire analysis and model training pipeline.

```bash
jupyter notebook monai-amia.ipynb
jupyter notebook resnet-amia.ipynb
```
## 📁 File Structure

Your project directory should be organized as follows:

```
amia-classification/
│
├── monai-amia.ipynb              # Monai Jupyter Notebook
├── resnet-amia.ipynb             # Resnet Jupyter Notebook
├── data/                         # Directory for the dataset
│   ├── train/                    # Training images
│   ├── test/                     # Testing images
│   ├── train.csv                 # Training annotations
│   └── ...                       # Other dataset files
│
├── results/                       # Directory for storing result plots
│   ├── resnet_results.png
│   └── densenet_results.png
│
├── requirements.txt              # List of Python dependencies
└── README.md                     # This file
```

### 💡 Future Improvements

*   **Learning Rate Scheduler**: Implement a scheduler (e.g., `ReduceLROnPlateau`) to dynamically adjust the learning rate, which could help the model fine-tune better as it nears convergence.
*   **Regularization**: Add techniques like dropout or weight decay to the DenseNet model to prevent potential overfitting in longer training runs and improve generalization.
*   **More Augmentation**: Enhance the diversity of the training data by applying a wider range of data augmentation techniques.
*   **Longer Training**: Train the DenseNet121 model for more epochs to allow it to reach its full potential.

### 📚 References

> **AMIA Dataset on Kaggle**: `https://www.kaggle.com/competitions/amia-public-challenge-2024`
>
> **Data augmentation using Albumentation**: `https://albumentations.ai/docs/`
>
> **MONAI Official Documentation**: `https://docs.monai.io/en/stable/networks.html`