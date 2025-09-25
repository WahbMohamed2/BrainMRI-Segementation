# U-Net for Brain MRI Segmentation

This repository provides a PyTorch implementation of the [U-Net architecture](https://arxiv.org/pdf/1505.04597), applied to the [LGG Brain MRI Segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).  
It includes the model, training and evaluation workflows, Jupyter notebooks for experiments, and pretrained weights.

---

## Features

- U-Net implementation in PyTorch (`src/model.py`)
- Training and validation pipelines with BCE + Dice loss
- Reproducible experiments provided as Jupyter notebooks
- Utility functions for data loading, metrics, and visualization (`utils/`)
- Pretrained checkpoint (`best_model.pth`, tracked via Git LFS)

---

## Project Structure

├── src/
│ ├── model.py # U-Net implementation
│ ├── data_loader.py # Dataset / dataloader utilities
│ ├── main.py # (placeholder) training entry point
│ ├── UNet.ipynb # Model architecture exploration
│ └── brain-mri-semantic-segmentation-u-net-pytorch.ipynb # Full training/eval
│
├── utils/ # Metrics, plotting, helpers
├── assets/
│ ├── assets1.png/ # Ground truth samples
│ └── assets2.png/ # Predicted mask samples
│
├── requirements.txt
├── .gitattributes # Git LFS config for large files
├── .gitignore
└── README.md

yaml
Copy code

---

## Installation

Clone the repository and install dependencies:

```bash
git clone git@github.com:WahbMohamed2/BrainMRI-Segementation.git
cd BrainMRI-Segementation
pip install -r requirements.txt
Dataset
The project uses the LGG Brain MRI Segmentation dataset:
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

Download the dataset from Kaggle.

Place it under a data/ directory at the project root.

Update paths in notebooks or scripts as needed.

Usage
Option 1: Jupyter notebooks
Run UNet.ipynb or brain-mri-semantic-segmentation-u-net-pytorch.ipynb for model definition, training, and evaluation.

Option 2: Training script (planned)
src/main.py is the placeholder for a CLI training entry point. Future work will add arguments for training and inference.

Results
Validation performance (sample run):

Loss: 0.1923

Accuracy: 98.57%

Example outputs (Ground Truth vs Prediction):

Ground Truth	Prediction
	

Loss Function
The training uses a combined loss function to address class imbalance and segmentation quality:

Binary Cross Entropy (BCE)

Dice loss

total_loss = 0.7 * BCE + 0.3 * Dice

References
U-Net: Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, 2015

Dataset: LGG Brain MRI Segmentation (Kaggle)

License
MIT License. Free to use, modify, and distribute.
