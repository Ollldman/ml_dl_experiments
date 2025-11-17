import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):
    def __init__(self, inputs, hidden, outputs, stride=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inputs,
                               hidden,
                               kernel_size=3,
                               stride=stride,
                               bias=False,
                               padding='same')
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(hidden,
                               hidden,
                               kernel_size=3,
                               stride=stride,
                               bias=False,
                               padding='same')
        self.conv3 = nn.Conv2d(hidden,
                               outputs,
                               kernel_size=3,
                               stride=stride,
                               bias=False,
                               padding='same')

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += identity
        out = self.relu(out)

        return out
    

class ResidualBlockBN(nn.Module):
    def __init__(self, inputs, outputs, stride=1) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inputs,
                               outputs,
                               kernel_size=3,
                               stride=stride,
                               bias=False,
                               padding='same')
        
        self.bn1 = norm_layer(outputs)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(outputs,
                               outputs,
                               kernel_size=3,
                               stride=stride,
                               bias=False,
                               padding='same')
        
        self.bn2 = norm_layer(outputs)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out