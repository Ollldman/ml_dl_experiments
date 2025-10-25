from ctypes import Array
import numpy as np
from numpy.typing import ArrayLike, NDArray

def sin_f(x: ArrayLike) -> NDArray[np.float64]:
    """
    Простая функция для приближения периодических гладких функций
    """
    x = np.asarray(x)
    return np.sin(x)

def sin_derivative(x: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(x)
    return np.cos(x)