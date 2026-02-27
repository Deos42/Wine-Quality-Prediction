# Wine Quality Prediction using Multi-Layer Perceptron (MLP) from Scratch
## Overview
The model is trained to predict wine quality scores based on physicochemical properties.
This project implements a **Multi-Layer Perceptron (MLP)** neural network entirely from scratch using NumPy to predict wine quality (regression task).

The dataset used:
- Red Wine Quality Dataset
- White Wine Quality Dataset
- Combined into a single dataset

Total Features: **11**  
Target: **Wine Quality Score**

This project demonstrates:
- Manual forward propagation
- Manual backpropagation (chain rule)
- Gradient descent optimization
- Full neural network implementation without deep learning libraries


---

## 1. Methodology

The project implements the fundamental components of a neural network without the use of high-level deep learning frameworks.

### Data Preprocessing
* **Integration**: Red and white wine datasets were concatenated into a single dataset.
* **Feature Scaling**: Input features were normalized using `StandardScaler` to ensure stable gradients during training.
* **Split**: The data was divided into an 80% training set and a 20% test set.

Input Features: **11**
Output: **1 (regression)**

### Network Architecture
The model utilizes a feedforward architecture with two hidden layers:
* **Input Layer**: 11 features.
* **Hidden Layer 1**: 64 neurons with **ReLU** activation.
* **Hidden Layer 2**: 32 neurons with **ReLU** activation.
* **Output Layer**: 1 neuron with a **linear** activation for regression output.



### Training and Optimization
* **Weight Initialization**: Weights were initialized with small random values ($0.01$) to prevent symmetry and saturation.
* **Loss Function**: Mean Squared Error (MSE) was used to measure prediction accuracy.
* **Backpropagation**: Gradients were manually calculated using the chain rule to update weights and biases.
* **Hyperparameters**: The model was trained for 4,000 epochs with a learning rate of $0.05$.

---

## 2. Results

The model demonstrated successful convergence, significantly reducing error from the initial epoch.

### Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Initial Training MSE** | 34.5774 |
| **Final Training MSE** | 0.4325 |
| **Final Test MSE** | **0.4861** |

### Sample Predictions
The model provides continuous quality score predictions that closely align with the actual integer ratings:

| Actual Quality | Predicted Value |
| :--- | :--- |
| 8.0 | 6.26 |
| 5.0 | 4.96 |
| 7.0 | 6.96 |
| 6.0 | 5.51 |

### Observations

- Model converges steadily over 4000 epochs.
- Standardization significantly improves training stability.
- Two hidden layers allow nonlinear regression modeling.
- Performance depends heavily on learning rate tuning.


---

## Key Learnings

- Implemented complete MLP without TensorFlow/PyTorch.
- Derived and coded backpropagation manually.
- Understood gradient flow through multiple hidden layers.
- Gained strong intuition for optimization in neural networks.


---

## Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn (for preprocessing only)
- Jupyter Notebook


---

## Future Improvements

- Mini-batch gradient descent
- Adam optimizer
- L2 Regularization
- Early stopping
- Deeper architectures
- Convert to modular class-based implementation
