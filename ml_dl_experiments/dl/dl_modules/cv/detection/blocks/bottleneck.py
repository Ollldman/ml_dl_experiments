import torch.nn as nn
from ml_dl_experiments.dl.dl_modules.\
    cv.detection.blocks.basic_conv import Conv

class Bottleneck(nn.Module):
    # Стандартный Bottleneck блок из ResNet
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """
        Инициализация Bottleneck.
        Args:
            c1 (int): количество входных каналов.
            c2 (int): количество выходных каналов.
            shortcut (bool): флаг, указывающий, нужно ли использовать residual connection.
            g (int): количество групп.
            e (float): коэффициент расширения каналов.
        """
        super().__init__()
        c_ = int(c2 * e)  # скрытые каналы
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # Если self.add=True, складываем вход с выходом (residual connection)
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))