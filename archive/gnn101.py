import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(1, 3), stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=1)

        # Dynamically calculate the flattened size
        self.flattened_size = self._get_flattened_size()

        # Fully connected layer
        self.fc1 = nn.Linear(self.flattened_size, 10)  # Use the dynamically calculated size

    def _get_flattened_size(self):
        # Create a dummy input tensor with the same shape as your input
        dummy_input = torch.zeros(1, 3, 1, 30)  # Batch size=1, channels=3, height=1, width=30
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        return x.numel()  # Get the total number of elements in the tensor

    def forward(self, x):
        print("Input shape:", x.shape)
        x = F.relu(self.conv1(x))
        print("After conv1:", x.shape)
        x = F.relu(self.conv2(x))
        print("After conv2:", x.shape)
        x = torch.flatten(x, start_dim=1)
        print("After flatten:", x.shape)
        x = self.fc1(x)
        return x


# Input: batch of 1 matrix of size 3x30
input_tensor = torch.randn(1, 3, 1, 30)  # Batch size=1, 3 channels, height=1, width=30
model = SimpleCNN()
output = model(input_tensor)
print("Output shape:", output.shape)