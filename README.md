# Self-Pruning Neural Network

This repository contains the solution for the Tredence Analytics placement assignment to build a Self-Pruning Neural Network.

## Overview
The project features a Convolutional Neural Network (CNN) architecture to extract image features from the CIFAR-10 dataset, followed by custom `PrunableLinear` dense layers. In these custom layers, each weight is associated with a learnable gate parameter. The network uses an **L1 Sparsity Penalty** to push unnecessary gate values to exactly 0, dynamically pruning the dense layers during training.

## Key Features
- **CNN Feature Extraction**: Utilizes `Conv2d` and `BatchNorm2d` layers to achieve a strong baseline accuracy on CIFAR-10.
- **Custom Prunable Layer**: Dense weights are scaled dynamically using `sigmoid(gate_scores)`.
- **L1 Sparsity Regularization**: A custom loss function penalizes active gates, forcing the network to discover a sparse subnetwork.
- **Trade-off Analysis**: The project trains the model using different sparsity penalties ($\lambda$) to evaluate the trade-off between compression (sparsity) and Test Accuracy.

## Results
By increasing the $\lambda$ penalty, the network successfully prunes up to 99.78% of its connections with a manageable drop in accuracy.
- **$\lambda = 0.0$**: Baseline (0% Sparsity, 53.1% Acc)
- **$\lambda = 0.0001$**: 95.27% Sparsity, 50.9% Acc
- **$\lambda = 0.001$**: 99.78% Sparsity, 44.0% Acc

Please see `report.md` for a full analysis and visual plots of the gate distributions.

## How to Run
1. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib
   ```
2. Run the main training script:
   ```bash
   python train.py
   ```
   This will download the CIFAR-10 dataset, train 3 independent models with different $\lambda$ values, print the results table, and generate the plots.
