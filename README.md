# 🧠 Simple CNN from Scratch (NumPy)

This project demonstrates how to build and train a **Convolutional
Neural Network (CNN)** entirely from scratch using **NumPy**, without
using any deep learning frameworks like TensorFlow or PyTorch.\
It trains a small CNN on the **MNIST handwritten digit dataset** (28×28
grayscale images).

------------------------------------------------------------------------

## 📘 Overview

This project helps you understand the **core building blocks** of a CNN,
including:

-   Convolutional layers (3×3 filters)
-   Max Pooling layers (2×2)
-   Softmax output layer (10 classes)
-   Manual forward and backward propagation
-   Gradient updates via simple stochastic gradient descent (SGD)

Everything --- from convolution to backpropagation --- is implemented
manually in **pure NumPy**, so you can see *exactly how CNNs learn*
under the hood.

------------------------------------------------------------------------

## 🧩 Architecture

``` mermaid
graph TD
    A["Input Image (28×28)"] --> B["Conv3x3 Layer (8 Filters)"]
    B --> C["ReLU Activation"]
    C --> D["MaxPool2 Layer (2×2)"]
    D --> E["Flatten"]
    E --> F["Softmax Layer (10 Classes)"]
    F --> G["Output Probabilities"]
```

### Layer Details

  ------------------------------------------------------------------------
  Layer       Parameters         Output Shape         Description
  ----------- ------------------ -------------------- --------------------
  Conv3x3     8 filters (3×3)    8 × 26 × 26          Extracts local
                                                      spatial features

  ReLU        \-                 8 × 26 × 26          Adds non-linearity

  MaxPool2    2×2                8 × 13 × 13          Reduces spatial
                                                      dimensions

  Flatten     \-                 1352                 Converts 3D tensor →
                                                      1D vector

  Softmax     Input: 1352,       (10,)                Produces class
              Output: 10                              probabilities
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## ⚙️ Forward & Backward Pass Cycle

``` mermaid
sequenceDiagram
    participant Input
    participant Conv3x3
    participant ReLU
    participant MaxPool2
    participant Softmax
    participant Loss

    Input->>Conv3x3: Forward (convolution)
    Conv3x3->>ReLU: Apply activation
    ReLU->>MaxPool2: Downsample (2×2)
    MaxPool2->>Softmax: Flatten + feed forward
    Softmax->>Loss: Compute cross-entropy loss
    Loss-->>Softmax: Backpropagate error
    Softmax-->>MaxPool2: Gradients wrt input
    MaxPool2-->>ReLU: Pass gradients through mask
    ReLU-->>Conv3x3: Update filters (SGD)
```

------------------------------------------------------------------------

## 🧠 Training Flow

1.  **Load MNIST dataset** (automatically from TensorFlow or NumPy).\
2.  Initialize layers: `Conv3x3`, `MaxPool2`, `Softmax`.
3.  For each image:
    -   Forward pass → compute prediction
    -   Compute loss and accuracy
    -   Backward pass → update weights
4.  Plot training loss after every few hundred steps.

Example logs:

    Epoch 1/3
    [Step 100] Avg Loss=2.18 | Accuracy=45%
    [Step 200] Avg Loss=2.01 | Accuracy=62%
    [Step 300] Avg Loss=1.65 | Accuracy=70%

------------------------------------------------------------------------

## 📈 Training Loss Curve

At the end of training, a plot will automatically show **training loss
vs. steps**.\
A downward trend indicates that the model is learning correctly.

------------------------------------------------------------------------
# 🧮 Key Math Concepts

### 🧩 Convolution

``` math
(I * K)(x, y) = \sum_{i=0}^{2}\sum_{j=0}^{2} I_{x+i,\,y+j} \, K_{i,j}
```

Slides a 3×3 kernel (filter) across the input image to extract local
features.

------------------------------------------------------------------------

### ⚡ ReLU Activation

``` math
f(x) = \max(0, x)
```

Applies non-linearity by zeroing out negative values.

------------------------------------------------------------------------

### 🌀 Max Pooling

``` math
P_{i,j} = \max_{(m,n)\,\in\,R_{i,j}} X_{m,n}
```

Downsamples feature maps by taking the maximum value in each 2×2 region.

------------------------------------------------------------------------

### 🔢 Softmax

``` math
S_i = rac{e^{z_i}}{\sum_j e^{z_j}}
```

Converts logits into normalized probabilities for each class.

------------------------------------------------------------------------

### 📉 Cross-Entropy Loss

``` math
L = -\log(p_{   ext{true}})
```

Measures how far predicted probabilities are from the true class.


------------------------------------------------------------------------

## 🧰 Dependencies

-   Python 3.8+
-   NumPy
-   Matplotlib
-   TensorFlow *(only used to fetch MNIST)*

Install dependencies:

``` bash
pip install numpy matplotlib tensorflow
```

If you want to make it **framework-free**, replace TensorFlow loading
with:

``` python
from sklearn.datasets import fetch_openml
```

------------------------------------------------------------------------

## 🧪 Experiment Ideas

-   🔢 Change number of filters (8 → 16 or 32)
-   🔁 Train for more epochs (3--5)
-   ⚙️ Adjust learning rate (0.005 → 0.01)
-   🧩 Add extra convolutional layers
-   🔄 Compare performance with a Keras CNN

------------------------------------------------------------------------

