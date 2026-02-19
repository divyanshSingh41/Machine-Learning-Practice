# 📌 K-Nearest Neighbors (KNN) From Scratch
## 📖 Overview
KNN is a supervised machine learning algorithm used for both classification and regression tasks. It is a non-parametric and instance-based learning method.

In this project, the focus is on understanding the algorithm mathematically and implementing it step by step.

## 🎯 Objectives
- Understand how distance-based learning works
- Implement KNN without using built-in ML libraries
- Visualize decision boundaries
- Experiment with different values of K
- Compare performance with Scikit-learn implementation

## 🛠️ Technologies Used
- Python
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 🧠 Algorithm Intuition
KNN works based on a simple idea:
- Choose the value of K
- Calculate the distance between the new data point and all training points
- Select the K nearest neighbors
- For classification → Take majority vote
- For regression → Take the average
- Distance metric used:
-     Euclidean Distance
-     distance = √ Σ (x1 - x2)²


