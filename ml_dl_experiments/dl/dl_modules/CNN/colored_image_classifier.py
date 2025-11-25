import torch.nn as nn
from torch import Tensor
from ml_dl_experiments.dl.dl_modules.CNN.bottleneck_block import BottleneckBlock

class Classificator(nn.Module):

    def __init__(
        self,
        n_classes: int,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU()
            
        # слой макс-пулинга
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.bottleneck = BottleneckBlock(64, 16, 64)
        
        # слой глобального пулинга
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Полносвязная часть в конце
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.bottleneck(out)
        out = self.avgpool(out)
                
        # полносвязная часть для классификации
        out = self.flatten(out)
        out = self.fc(out)
        out = self.softmax(out)

        return out

