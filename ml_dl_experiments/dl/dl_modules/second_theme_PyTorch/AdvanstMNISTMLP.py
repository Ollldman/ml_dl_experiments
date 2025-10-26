
import torch.nn as nn

import torch.nn.init as init

from torchvision.datasets import MNIST


# Класс через ModuleList

class AdvancedMNISTMLPList(nn.Module):
    def __init__(self):
        super().__init__()
        # Используйте nn.Flatten() + nn.ModuleList для:
        #   Linear(784→256), BatchNorm1d(256), ReLU, Dropout(0.5),
        #   Linear(256→128), ReLU,
        #   Linear(128→10)
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList([
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ])
        # Инициализируйте все Linear-слои Xavier’ем
        for l in self.layers:
            if isinstance(l, nn.Linear):
                init.xavier_uniform_(l.weight)
                init.zeros_(l.bias)


    def forward(self, x):
        x = self.flatten(x)
        # Пройдите по self.layers по порядку и примените каждый модуль
        for l in self.layers:
            x = l(x)
        return x
