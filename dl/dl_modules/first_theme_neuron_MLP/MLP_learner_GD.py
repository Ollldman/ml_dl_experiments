from numpy.typing import NDArray, ArrayLike
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Any, Sequence

from basic_MLP import MLP
from metrics import mse


class MLP_learner_GD():
    def __init__(self,
                X: ArrayLike, 
                y: ArrayLike, 
                model: str = MLP.__name__,
                activation: str = 'sigmoid',
                loss_function: str = 'mse',
                layer_sizes: Sequence[int] = [1, 30, 1], 
                learning_rate: np.float64 = np.float64(0.01), 
                epochs: int = 100, 
                epoch_output: int = 10, 
                batch_size: int = 16) -> None:
        self.layer_sizes: Sequence[int] = layer_sizes
        self.lr: np.float64 = np.float64(learning_rate)
        self.epochs: int = epochs
        self.epoch_output: int = epoch_output
        self.batch_size: int = batch_size
        self.loss_history: List[Union[np.floating[Any], np.complexfloating[Any, Any]]] = []
        self.activation: str = activation
        self.X: NDArray[np.float64] = np.asarray(X, dtype=np.float64)
        self.y: NDArray[np.float64] = np.asarray(y, dtype=np.float64)
        self.model = model
        
        if loss_function == 'mse':
            self.loss_function = loss_function
        else:
            raise ValueError(f'This loss function is not exist now: {loss_function}')
        
        if self.model == 'MLP':
            try:
                self.estimator = MLP(
                    layer_sizes=self.layer_sizes, 
                    activation=self.activation)
            except Exception as e:
                raise ValueError(f'Warning by creating estimator model: {e}')
        
    def learn_model(self, 
                    GD_type: str = 'mini-batch_GD'):
        if GD_type == 'mini-batch_GD':
            for epoch in range(1, self.epochs+1):
                # Перетасовываем, для избежания быстрого переобучения
                perm: NDArray[np.long] = np.random.permutation(len(self.X))
                # Выделяем только те части, которые попали от permutation
                X_sh, y_sh = self.X[perm], self.y[perm]
                # mini-batch learning:
                for start in range(0, len(self.X), self.batch_size):
                    xb: NDArray[np.float64] = X_sh[start : start + self.batch_size]
                    yb: NDArray[np.float64] = y_sh[start : start + self.batch_size]
                    # Прямой проход
                    preds: NDArray[np.float64] = self.estimator.forward(xb)
                    # Обратный проход
                    self.estimator.backward(yb)
                    # Обновление параметров
                    self.estimator.update_params(self.lr)
                # Прямой проход на всех данных необходим для вычисления потерь
                full_pred: NDArray[np.float64] = self.estimator.forward(self.X)
                # Запомниаем текущие потери
                if self.loss_function == 'mse':
                    self.loss_history.append(mse(self.y, full_pred))
                else:
                    break
                # Будем выводить статистику обучения только каждые 50 epoch
                if epoch % self.epoch_output == 0:
                    print(f"Epoch {epoch:3d}, loss={self.loss_history[-1]:.4f}")
        else:
            raise ValueError(f'This GD is not exist in that module: {GD_type}')

    def visualise_learning_process(self) -> None:
        """
        Пока есть реализация только для 
        """
        # Визуализация
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(self.loss_history); plt.title(f"{self.loss_function} по эпохам"); plt.xlabel("Эпоха"); plt.grid(True)
        plt.subplot(1,2,2)
        plt.scatter(self.X[:, 0], self.y, label="Целевое распределение", c='green', alpha=.5)
        plt.plot(self.X[:, 0], self.estimator.forward(self.X), label="MLP", c='r')
        plt.legend(); plt.grid(True); plt.title(f"Model {self.model} estimation")
        plt.show()

    def get_learned_estimator(self) -> MLP:
        return self.estimator
    
