# 🔧 Gradient Descent from Scratch

This project is a **from-scratch implementation** of a linear regression model trained using various types of **gradient descent algorithms**. It offers flexibility in choosing the optimization method, regularization, and loss functions, all without relying on machine learning libraries like scikit-learn.

## 📌 Project Highlights

- Implemented gradient descent variants:
  - **Full Batch Gradient Descent**
  - **Stochastic Gradient Descent (SGD)**
  - **Momentum-based Descent**
  - **Adam Optimizer**
- Custom loss functions:
  - **Mean Squared Error (MSE)**
  - **Log-Cosh**
- Optional **L2 regularization (Ridge)** via parameter `mu`

## 🚗 Use Case: Car Price Prediction

The implemented linear model is applied to a **real-world regression problem** — predicting car prices based on a provided dataset.

Key steps in the pipeline include:
- Data preprocessing (feature scaling, encoding, etc.)
- Exploratory data analysis (EDA)
- Model training and evaluation

## 🔍 Hyperparameter Tuning & Experiments

The project includes a series of experiments aimed at optimizing model performance:
- Tuning the **learning rate (`lambda`)**
- Selecting an optimal **batch size** for stochastic descent
- Investigating the effect of **regularization strength (`mu`)**

## 📊 Results & Analysis

The training results are analyzed to compare:
- Convergence speed and stability of each optimizer
- Sensitivity to batch size in SGD
- Effectiveness of regularization
- Accuracy and runtime across different configurations

Plots and metrics help illustrate the trade-offs between training time, generalization, and model complexity.

## 🛠️ Tech Stack

- Python 3
- NumPy / Pandas
- Jupyter Notebook
- Matplotlib / Seaborn

