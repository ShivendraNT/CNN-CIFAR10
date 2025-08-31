# CNN-CIFAR10

A simple Convolutional Neural Network (CNN) built with PyTorch to classify images from the CIFAR-10 dataset.


## 📌 Project Overview
This project implements a CNN from scratch using PyTorch to classify 32×32 RGB images into 10 categories:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

The model is trained on the CIFAR-10 training set and evaluated on accuracy per class.

## ⚙️ Model Architecture
The network follows a LeNet-inspired architecture:
Conv1 → 3 input channels, 6 output channels, kernel 5×5
ReLU + MaxPool (2×2)
Conv2 → 6 input channels, 16 output channels, kernel 5×5
ReLU + MaxPool (2×2)
Fully Connected 1 → 120 units
Fully Connected 2 → 84 units
Fully Connected 3 (Output Layer) → 10 units (CIFAR-10 classes)

## 🛠️ Tech Stack
Python 3.10+
PyTorch
Torchvision
Matplotlib (for visualization)
NumPy

## 🚀 Installation & Setup
Clone this repo:
git clone https://github.com/your-username/CNN-CIFAR10.git
cd CNN-CIFAR10
Install dependencies:
pip install torch torchvision matplotlib numpy

## ▶️ Training the Model
Run the script:
python CNN-CIFAR10.py
During training, the model learns on CIFAR-10 dataset.
After training, the script evaluates model accuracy overall and per class.

## 📊 Sample Output
Finished training
Accuracy : 52.34 %
Accuracy of plane : 55.67 %
Accuracy of car   : 64.10 %
Accuracy of bird  : 40.32 %
Accuracy of cat   : 38.90 %
Accuracy of deer  : 46.50 %
Accuracy of dog   : 48.77 %
Accuracy of frog  : 60.12 %
Accuracy of horse : 59.23 %
Accuracy of ship  : 65.89 %
Accuracy of truck : 58.76 %
(Accuracy may vary depending on epochs, optimizer, and hyperparameters.)

## 📈 Improvements to Try
Train for more epochs (default is 4)
Use Adam optimizer instead of SGD
Add Dropout / BatchNorm for better generalization
Experiment with deeper CNNs (e.g., ResNet, VGG)

## 📂 Project Structure
CNN-CIFAR10/
│── data/               # CIFAR-10 dataset (auto-downloaded)
│── CNN-CIFAR10.py      # Main training + evaluation script
│── README.md           # Project documentation


## ✨ Results & Learnings
The CNN achieves ~50–60% accuracy in just a few epochs.
Performance can be significantly improved with deeper networks and data augmentation.
