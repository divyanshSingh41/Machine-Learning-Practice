# 📊 Logistic Regression from Scratch

A hands-on Python notebook implementing **Logistic Regression** from scratch — to build deep intuition for how the algorithm works.

---
## 📋 Overview
This notebook covers the fundamentals of Logistic Regression, a supervised machine learning algorithm used for **binary classification** tasks (e.g., spam/not spam, disease/no disease). Despite its name, it predicts *categories*, not continuous values.

The core idea: instead of outputting a direct label, Logistic Regression predicts the **probability** that an input belongs to a class using the **sigmoid (logistic) function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

---
## 📁 Contents

| Section | Description |
|---|---|
| **Logistic Regression (Batch GD)** | Full implementation using Gradient Descent, trained and tested on synthetic data |
| **Logistic Regression (SGD Style)** | Perceptron-style implementation using Stochastic Gradient Descent |
| **Decision Boundary Visualization** | Visual plot of the learned decision boundary |
| **Train/Test Split Evaluation** | Model evaluation on a held-out test set |
 
---

## 🛠️ Tech Stack
 
- Python 3
- NumPy
- Matplotlib
- scikit-learn *(only for dataset generation, train/test split, and accuracy scoring)*
 
---

## 🧠 Implementations

### 1. Logistic Regression — Batch Gradient Descent
 
```python
class LogisticReg:
    def __init__(self, lr=0.01, epochs=1000):
        ...
    def sigmoid(self, z): ...
    def fit(self, X, y): ...
    def predict(self, X): ...
```

- Updates weights using the **full dataset** at every step
- Achieved **99% accuracy** on a synthetic 2-class dataset

### 2. Logistic Regression — Stochastic Gradient Descent (Perceptron Style)

```python
class LogisticRegSGD:
    def __init__(self, lr, epochs):
        ...
    def sigmoid(self, z): ...
    def fit(self, X, y): ...
    def predict(self, X_test): ...
```

- Picks a **random point** each epoch and updates weights proportionally to the error
- Uses sigmoid activation instead of a hard step function
- Achieved **100% test accuracy** on held-out data

---

## 📌 Key Concepts Covered
 
- Sigmoid function and probability estimation
- Binary Cross-Entropy loss (implicit via gradient computation)
- Weight update rule via Gradient Descent
- Stochastic vs Batch Gradient Descent
- Decision boundary visualization
 
---

