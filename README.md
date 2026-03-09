# Customer Churn Prediction Neural Network

<hr>

![](https://img.shields.io/badge/python-3.12-lightblue) ![](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![](https://img.shields.io/badge/Licence-MIT-lightgray) ![](https://img.shields.io/badge/status-Release-darkgreen) ![](https://img.shields.io/badge/pipeline-passed-green) ![](https://img.shields.io/badge/testing-passing-green)

This repository implements a production-ready **PyTorch Neural Network (Multi-Layer Perceptron)** designed to predict customer churn based on numerical and categorical customer features.

## Technology Stack

- **Deep Learning Framework**: [PyTorch](https://pytorch.org/) (Custom `nn.Module` creation, backward propagation with `BCEWithLogitsLoss`)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
- **Preprocessing**: [Scikit-Learn](https://scikit-learn.org/) (`StandardScaler` and `OneHotEncoder` via `ColumnTransformer`)
- **Execution & Tracking**: Python 3.12, `tqdm` for training progress bars.

## Project Structure

```text
├── data/
│   ├── customer_churn_dataset-training-master.csv
│   └── customer_churn_dataset-testing-master.csv
├── models/                      # Saved trained models
├── src/                         # Source code
│   ├── config.py                # Hyperparameters and paths
│   ├── dataset.py               # Custom PyTorch Dataset and Preprocessing
│   ├── model.py                 # PyTorch Neural Network module
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── inference.py             # Inference on new data
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the Model**:
   ```bash
   python src/train.py
   ```
   This script handles the preprocessing of the target variables and features, trains the PyTorch model, and saves the best weights to `models/churn_model.pth`.

2. **Evaluate the Model**:
   ```bash
   python src/evaluate.py
   ```
   Evaluates the saved model on the testing dataset, printing classification metrics (Accuracy, F1-Score, ROC-AUC).

3. **Inference**:
   See `src/inference.py` for how to load and use the model in production predicting new, raw records.
