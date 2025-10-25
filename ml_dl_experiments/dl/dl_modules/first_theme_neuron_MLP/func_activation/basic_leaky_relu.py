import numpy as np
from numpy.typing import ArrayLike

def leaky_relu(x: ArrayLike, alpha: float = 0.01) -> np.ndarray:
    """
    LeakyRelu(x) = x if x >= 0 else alpha * x 

    На графике виден «спуск» в отрицательной части.
    Плюсы. Эта функция — простая модификация ReLU. При этом она сохраняет ненулевой градиент при $x<0$ — нейроны не «умирают».

    Минусы. В функции появляется гиперпараметр α. Кроме того, малый α может недостаточно компенсировать нули. 
    """
    x = np.asarray(x)
    return np.where(x >= 0, x, alpha * x)