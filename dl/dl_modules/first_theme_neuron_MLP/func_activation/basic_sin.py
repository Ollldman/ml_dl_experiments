from ctypes import Array
import numpy as np
from numpy.typing import ArrayLike

def sin_f(x: ArrayLike) -> np.ndarray:
    """
    Простая функция для приближения периодических гладких функций
    """
    x = np.asarray(x)
    return np.sin(x)

def sin_derivative(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x)
    return np.cos(x)