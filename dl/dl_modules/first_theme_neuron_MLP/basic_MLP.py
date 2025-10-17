import numpy as np
from typing import Any, List, Sequence
from numpy.typing import ArrayLike, NDArray

from func_activation import (
    sigmoid,
    sigmoid_derivative,
    relu, 
    tanh,
    tanh_derivative,
    sin_f,
    sin_derivative)
from metrics import mse

class MLP:

    def __init__(
            self,
            layer_sizes: Sequence[int], 
            activation: str = 'sigmoid')-> None:
        
        self.layer_sizes: NDArray[np.integer] = np.asarray(layer_sizes, dtype=int)
        self.activation: str = activation

        self.W: List[NDArray[np.float64]] = []      # список матриц весов
        self.b: List[NDArray[np.float64]] = []      # список векторов смещений

        self.Z_list: List[NDArray[np.float64]] = []
        self.A_list: List[NDArray[np.float64]] = []

        self.dW_list: List[NDArray[np.float64]] = []
        self.db_list: List[NDArray[np.float64]] = []

        # В цикле заполнить self.W и self.b случайными параметрами
        for i in range(len(layer_sizes) - 1):
            in_dim: int = int(layer_sizes[i])
            out_dim: int = int(layer_sizes[i + 1])
            self.W.append(np.random.randn(in_dim, out_dim) * 0.1)
            self.b.append(np.zeros((1, out_dim)))

    # Метод forward    
    def forward(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Forward to MLP
        """
        # очищаем списки перед новым вызовом
        self.Z_list, self.A_list = [], []
        # Активация входного слоя - линейна и равна примерам - X
        A:  NDArray[np.float64] = np.asarray(X, dtype=np.float64)
        # Добавляем активацию входного слоя
        self.A_list.append(A)

        for i, (W, b) in enumerate(zip(self.W, self.b)):
            # Высчитываем линейную функцию - скалярное произведение входов на веса + bias
            Z: NDArray[np.float64] = A @ W + b
            self.Z_list.append(Z)
            # Определяем функцию активации пока только для всех слоев, кроме выходного, т.к. там выбор функции влияет на характер решаемой задачи (классификация или регрессия)
            if i < len(self.W) - 1:
                if self.activation == 'sigmoid':
                    A = sigmoid(Z)
                elif self.activation == 'relu':
                    A = relu(Z)
                elif self.activation == 'tanh':
                    A = tanh(Z)
                elif self.activation == 'sin':
                    A = sin_f(Z)
                else:
                    raise ValueError(f'Unknown activation {self.activation}')
            else:
                A = Z  # линейный выход для последнего слоя
            # Добавляем получившуюся активацию
            self.A_list.append(A)

        return A
    
    def forward_with_activations(self, X: ArrayLike) -> List[NDArray[np.float64]]:
        """
            This function return list of forward activations to all layers in MLP
        """
        activations: List[NDArray[np.float64]] = [np.asarray(X, dtype=np.float64)]
        A: NDArray[np.float64] = np.asarray(X, dtype=np.float64)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z: NDArray[np.float64] = A @ W + b
            if i < len(self.W) - 1:
                A = sigmoid(Z)
            else:
                A = Z
            activations.append(A)
        return activations
    
    def backward(self, y_true: NDArray[np.float64]) -> None:
        m: int = y_true.shape[0]  # количество примеров

        # δ на выходном слое (линейный + MSE)
        aL: NDArray[np.float64] = self.A_list[-1]
        # δ^L = (2/m) * (A^L - Y)
        delta: NDArray[np.float64] = 2 * (aL - y_true) / m

        # Градиенты для выходного слоя
        a_prev: NDArray[np.float64] = self.A_list[-2]
        # dW^L = A^{L-1}ᵀ · δ^L
        dW: NDArray[np.float64] = a_prev.T @ delta
        db: NDArray[np.float64] = np.sum(delta, axis=0, keepdims=True)

        self.dW_list = [dW]
        self.db_list = [db]

        # Обратный проход по скрытым слоям (sigmoid)
        # l — индекс текущего слоя (от предпоследнего до первого)
        for l in range(len(self.layer_sizes) - 2, 0, -1):
            z: NDArray[np.float64] = self.Z_list[l - 1]
            a_prev = self.A_list[l - 1]
            W_next: NDArray[np.float64] = self.W[l]  # веса слоя l+1

            # δ^l = (δ^{l+1} · W^{l+1}ᵀ) * σ'(z^l)
            if self.activation == 'sigmoid':
                delta = (delta @ W_next.T) * sigmoid_derivative(z)
            elif self.activation == 'tanh':
                delta = (delta @ W_next.T) * tanh_derivative(z)
            elif self.activation == 'sin':
                 delta = (delta @ W_next.T) * sin_derivative(z)

            # Градиенты для слоя l
            dW = a_prev.T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            # Вставляем в начало, чтобы порядок совпадал с W и b
            self.dW_list.insert(0, dW)
            self.db_list.insert(0, db)


    def update_params(self, lr: np.float64) -> None:
        """Обновление весов и смещений в цикле
        Через градиентный спуск"""
        for i in range(len(self.W)):
            self.W[i] -= lr * self.dW_list[i]
            self.b[i] -= lr * self.db_list[i]