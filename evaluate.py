# evaluate.py

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import LSTM_23

# Function to check accuracy on the given loader and model
def Check_accuracy(loader, model):
    num_correct = 0
    num_sample = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, prediction = scores.max(1)
            num_correct += (prediction == y).sum().item()
            num_sample += prediction.size(0)
        print("The accuracy: ", (num_correct / num_sample) * 100)

# Load MNIST test dataset
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Load trained model
model = LSTM_23(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load("lstm_model.pth"))

# Evaluate and print training accuracy
print("Training accuracy:")
Check_accuracy(train_loader, model)

# Evaluate and print test accuracy
print("Test accuracy:")
Check_accuracy(test_loader, model)
