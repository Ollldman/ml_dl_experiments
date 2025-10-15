from typing import Iterable
from matplotlib.pylab import ArrayLike
import numpy as np


from func_activation.basic_relu import relu


def neuron(inputs: ArrayLike, weights: ArrayLike, bias: float)-> float:
    # Вычисляем через скалярное произведение вектора весов и входа, прибавляем смещение, возвращаем  единственное число от 0 до +inf
    total = np.dot(inputs, weights) + bias
    # Примените ReLU
    output = relu(total)
    return output