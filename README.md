# LSTM Handwritten Digit Recognition

This repository contains a PyTorch implementation of an LSTM (Long Short-Term Memory) model for recognizing handwritten digits using the MNIST dataset. The LSTM network is trained to classify sequences of pixels representing digits and achieve high accuracy in recognizing both training and test samples.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [LSTM Model Architecture](#lstm-model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Code](#code)

## Introduction

Handwritten digit recognition is a classic problem in computer vision and machine learning. This project focuses on using an LSTM neural network to classify sequences of pixel values representing handwritten digits from the MNIST dataset. The LSTM model is designed to capture temporal dependencies in the sequences, making it suitable for this sequential data task.

## Requirements

Make sure you have the following dependencies installed:
- Python (>= 3.6)
- PyTorch (>= 1.8)
- torchvision
- matplotlib
- numpy

**Clone the repository:**
   ```bash
   git clone https://github.com/Avijitma123/lstm-handwritten-digit-recognition.git
   cd lstm-handwritten-digit-recognition