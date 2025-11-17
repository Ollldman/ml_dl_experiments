import torch.nn as nn
from torch import Tensor

class BottleneckBlock(nn.Module):

    def __init__(
        self,
        inputs: int,
        hidden: int,
        outputs: int,
        stride: int = 1,
        bias: bool = False
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inputs, 
                            hidden, 
                            1,
                            stride=stride, 
                            padding="same", 
                            bias=bias)
        self.bn1 = norm_layer(hidden)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden, 
                            hidden, 
                            3,
                            stride=stride, 
                            padding="same", 
                            bias=bias)
        self.bn2 = norm_layer(hidden)
        self.conv3 = nn.Conv2d(hidden, 
                            outputs,
                            1, 
                            stride=stride, 
                            padding="same", 
                            bias=bias)
        self.bn3 = norm_layer(outputs)

    def forward(self, x: Tensor) -> Tensor:
        identify = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += identify
        out = self.relu(out)

        return out

