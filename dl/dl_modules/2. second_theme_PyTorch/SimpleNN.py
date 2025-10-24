
import torch
from torch import nn

tensor = torch.Tensor 

class SimpleNN(nn.Module):
    fc1: nn.Linear
    relu: nn.ReLU
    fc2: nn.Linear

    def __init__(self, in_features: int) -> None:
        # Базовая инициализация nn.Module
        super().__init__()
        self.in_features: int = in_features
        # Линейный слой: принимает тензор размера [..., 10], выдаёт [..., 5]
        self.fc1 = nn.Linear(in_features=self.in_features, out_features=5)
        # Функция активации ReLU
        self.relu = nn.ReLU()
        # Линейный слой: принимает [..., 5], выдаёт [..., 1]
        self.fc2 = nn.Linear(in_features=5, out_features=1)


    def forward(self, x: tensor) -> tensor:
        """
            Args:
                x (torch.Tensor): Input tensor of shape [batch_size, 10]

            Returns:
                torch.Tensor: Output tensor of shape [batch_size, 1]
        """
        # x - тензор размерности [batch_size, 10]
        x = self.fc1(x)          # теперь x → [batch_size, 5]
        # Примените ReLU
        x = self.relu(x)
        # Линейный слой x → [batch_size, 1]      
        x = self.fc2(x)
        return x                 # возвращаем тензор прогнозов


