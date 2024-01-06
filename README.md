# LSTM Handwritten Digit Recognition

This repository contains a PyTorch implementation of an LSTM (Long Short-Term Memory) model for recognizing handwritten digits using the MNIST dataset. The LSTM network is trained to classify sequences of pixels representing digits and achieve high accuracy in recognizing both training and test samples.

## Table of Contents

- [Model Definition](#model-definition)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Model Definition

The LSTM model architecture is defined in the `LSTM_23` class within the `model.py` file. It consists of an LSTM layer followed by a fully connected layer. The input to the model is a sequence of pixel values representing a digit image.

```python
# model.py

import torch
import torch.nn as nn

class LSTM_23(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_23, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
```

## Training

The model is trained using the train.py script. Hyperparameters such as learning rate, batch size, and the number of epochs can be adjusted within the script.
```bash
python train.py
```

## Evaluation
The trained model can be evaluated on the test dataset using the evaluate.py script.

```bash
python evaluate.py
```

## Results

After training and evaluating the model, you can expect to see the training loss for each epoch printed to the console. Additionally, the accuracy of the model on both the training and test datasets will be displayed.



