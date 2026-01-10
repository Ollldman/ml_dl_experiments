import torch
from ml_dl_experiments.dl.dl_modules.\
    cv.detection.blocks.basic_conv import Conv
from ml_dl_experiments.dl.dl_modules.\
    cv.detection.blocks.bottleneck import Bottleneck
    

class C2f(torch.nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Инициализация C2f.
        Args:
            c1 (int): количество входных каналов.
            c2 (int): количество выходных каналов.
            n (int): количество Bottleneck блоков.
            shortcut (bool): флаг для residual connection в Bottleneck.
            g (int): количество групп.
            e (float): коэффициент расширения.
        """
        super().__init__()
        self.c_ = int(c2 * e)  # скрытые каналы
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1) # Увеличиваем каналы для последующего разделения
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)  # Финальная свёртка
        self.m = torch.nn.ModuleList(Bottleneck(self.c_, self.c_, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        x = self.cv1(x)
        y1, y2 = x.split((self.c_, self.c_), 1)

        outputs = [y1, y2]
        current_y = y2
        for bottleneck_module in self.m:
            current_y = bottleneck_module(current_y)
            outputs.append(current_y)
        concatenated_output = torch.cat(outputs, 1)
        return self.cv2(concatenated_output)