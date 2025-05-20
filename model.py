import torch
import torch.nn as nn

class WineQualityCNN(nn.Module):
    def __init__(self, input_features):
        super(WineQualityCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        conv_output_size = input_features - 2  # adjust based on conv layers
        self.fc1 = nn.Linear(64 * conv_output_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
