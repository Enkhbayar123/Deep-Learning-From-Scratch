# Deep Learning From Scratch

A Python implementation of Deep Learning models (MLP and CNN) built entirely from first principles using **NumPy**. This repository avoids using the auto-differentiation engines of high-level frameworks (like PyTorch or TensorFlow) to build the models, demonstrating a fundamental understanding of the mathematics behind neural networks—including manual backpropagation, chain rule derivation, and tensor operations.

## 📂 Project Structure

The repository contains two interactive Jupyter Notebooks applied to the MNIST digit recognition task:

### 1\. `NeuralNetworkFromScratch.ipynb` (MLP)

A standard fully connected feedforward network implementation.

  * **Architecture:** Input (784) $\to$ Hidden (128, ReLU) $\to$ Output (10, Softmax).
  * **Optimization:** Full-Batch Gradient Descent.
  * **Performance:** Achieves **\~92.6% accuracy** on the validation set after 1000 iterations.
  * **Data Source:** Designed to ingest Kaggle's `digit-recognizer` CSV format.

### 2\. `ConvolutionalNeuralNetworkFromScratch.ipynb` (CNN)

A modular, object-oriented implementation of a Convolutional Neural Network.

  * **Architecture:** Conv2D (8 filters, $3\times3$) $\to$ ReLU $\to$ MaxPooling ($2\times2$) $\to$ Flatten $\to$ Dense (Softmax).
  * **Design:** Implements a base `Layer` class with modular `forward()` and `backward()` methods for every component.
  * **Optimization:** Mini-Batch Stochastic Gradient Descent (SGD).
  * **Data Source:** Uses `tensorflow.keras.datasets.mnist` solely for convenient data downloading/loading.

## 🚀 Features Implemented

### Core Mechanics

  * **Matrix Operations:** Efficient vectorized forward/backward passes using NumPy broadcasting.
  * **Optimization Algorithms:** \* Gradient Descent (for MLP)
      * Stochastic Gradient Descent (SGD) (for CNN)

### Custom Layers (built from scratch)

  * **Dense (Fully Connected):** Implements weights/biases initialization and linear transformation.
  * **Conv2D:** Handles 4D tensor manipulation ($N, H, W, C$), padding, striding, and gradient propagation through filters.
  * **MaxPooling:** Implements downsampling and gradient routing (passing gradients only to the "winning" max pixels).
  * **Flatten:** Reshapes tensors between convolutional and dense layers.

### Activation & Loss

  * **ReLU:** Rectified Linear Unit and its derivative.
  * **Softmax:** Stable implementation handling numerical overflow.
  * **Categorical Cross-Entropy:** Loss function with numerical stability optimizations.

## 🧮 The Mathematics

The core of this project is the manual implementation of **Backpropagation**.

  * **Dense Layers:** Implements the chain rule for matrix multiplication:
    $$\frac{\partial E}{\partial W} = X^T \cdot \frac{\partial E}{\partial Y}$$
  * **Convolutional Layers:** Implements "Transposed Convolution" logic to propagate gradients from the output map back to the input map and the filters.
  * **Pooling Layers:** Implements a mask-based backward pass, where gradients flow back only to the indices that were selected during the forward max-pooling step.

## 🛠️ Installation & Usage

### Prerequisites

You will need Python 3 and the following libraries:

```bash
pip install numpy pandas matplotlib
```

*Note: The CNN notebook requires `tensorflow` only to download the MNIST dataset conveniently.*

### Running the Projects

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/Deep-Learning-From-Scratch.git
    ```

2.  **Run the MLP:**

      * Open `NeuralNetworkFromScratch.ipynb`.
      * Ensure `train.csv` (from Kaggle Digit Recognizer) is in the expected path (or update the `pd.read_csv` path).
      * Run all cells to train the model and generate predictions.

3.  **Run the CNN:**

      * Open `ConvolutionalNeuralNetworkFromScratch.ipynb`.
      * Run all cells. The notebook will automatically download the MNIST dataset and begin training the custom CNN class.
