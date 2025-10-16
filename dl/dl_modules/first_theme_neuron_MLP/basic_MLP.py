import numpy as np
from typing import List, Sequence
from numpy.typing import ArrayLike, NDArray

from func_activation import sigmoid, relu, tanh

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

        # В цикле заполнить self.W и self.b случайными параметрами
        for i in range(len(layer_sizes) - 1):
            in_dim: int = int(layer_sizes[i])
            out_dim: int = int(layer_sizes[i + 1])
            self.W.append(np.random.randn(in_dim, out_dim) * 0.1)
            self.b.append(np.zeros((1, out_dim)))

    # Метод forward    
    def forward(self, X: ArrayLike):
        """
        Forward to MLP
        """
        # очищаем списки перед новым вызовом
        self.Z_list = []
        self.A_list = []
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