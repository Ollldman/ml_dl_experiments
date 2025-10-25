from matplotlib.pylab import ArrayLike
import numpy as np


from func_activation import relu


def neuron(inputs: ArrayLike, weights: ArrayLike, bias: float)-> float | np.ndarray:
    # Вычисляем через скалярное произведение вектора весов и входа, прибавляем смещение, возвращаем  единственное число от 0 до +inf
    total = np.dot(inputs, weights) + bias
    # Примените ReLU
    output = relu(total)
    return output