# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import LSTM_23

# Set device (cuda if available, otherwise cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.0001
batch_size = 64
number_epoch = 3

# Load MNIST dataset
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Create LSTM model and move it to the specified device
model = LSTM_23(input_size, hidden_size, num_layers, num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(number_epoch):
    total_loss = 0
    for batch_index, (data, target) in enumerate(train_loader):
        data = data.to(device=device).squeeze(1)
        target = target.to(device=device)

        predicted = model(data)
        loss = criterion(predicted, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch =", epoch + 1, "/", number_epoch, " Loss =", total_loss / (batch_index + 1))
