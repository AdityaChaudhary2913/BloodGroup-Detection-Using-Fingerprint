import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import nn, optim
from bloodgroup.logger import logging
from bloodgroup.constants import DEVICE

class BloodGroupClassifier(nn.Module):
    def __init__(self):
        super(BloodGroupClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(262144, 512)  # Adjust input size based on image resolution after pooling
        self.fc2 = nn.Linear(512, 8)  # 8 output classes for blood groups
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Forward pass through convolutional layers + pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.pool(x)

        # Flatten the feature maps
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def initialize_model() -> nn.Module:
    logging.info("Initializing model")
    model = BloodGroupClassifier()
    model.to(DEVICE)
    return model

def get_loss_and_optimizer(model: nn.Module) -> Tuple[nn.Module, optim.Optimizer]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    logging.info("Loss function and optimizer initialized")
    return criterion, optimizer