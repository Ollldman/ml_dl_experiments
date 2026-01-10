import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Инициализация свёрточного блока.
        Args:
            c1 (int): количество входных каналов.
            c2 (int): количество выходных каналов.
            k (int): размер ядра свертки (kernel size).
            s (int): шаг свёртки (stride).
            p (int): отступ (padding). Если None, вычисляется автоматически.
            g (int): количество групп (для depth-wise свёрток).
            act (bool): флаг, указывающий, нужно ли применять функцию активации.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None):
    # Автоматический расчёт padding для "same" свёртки
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p