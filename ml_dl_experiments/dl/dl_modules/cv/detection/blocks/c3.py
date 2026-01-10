import torch

from ml_dl_experiments.dl.dl_modules.\
    cv.detection.blocks.basic_conv import Conv
from ml_dl_experiments.dl.dl_modules.\
    cv.detection.blocks.bottleneck import Bottleneck
    
class C3(torch.nn.Module):
    # Блок C3 из YOLOv5 (CSP Bottleneck with 3 convolutions)
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Инициализация C3.
        Args:
            c1 (int): количество входных каналов.
            c2 (int): количество выходных каналов.
            n (int): количество Bottleneck блоков.
            shortcut (bool): флаг для residual connection в Bottleneck.
            g (int): количество групп.
            e (float): коэффициент расширения.
        """
        super().__init__()
        c_ = int(c2 * e)  # скрытые каналы
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # финальная свертка
        self.m = torch.nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        # Разделение на два пути и конкатенация
        # Путь 1: x -> cv1 -> m
        # Путь 2: x -> cv2
        # Результат: concat(путь1, путь2) -> cv3
        part1 = self.m(self.cv1(x))
        part2 = self.cv2(x)
        return self.cv3(torch.cat((part1, part2), dim=1))