from numpy.typing import NDArray, ArrayLike
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Any, Sequence

from basic_MLP import MLP
from metrics import (
    mae,
    mse,
    huber,
    log_loss,
    categorical_cross_entropy
)


class MLP_learner_GD():
    def __init__(self,
                X: ArrayLike, 
                y: ArrayLike, 
                model: str = "MLP",
                activation: str = 'sigmoid',
                loss_function_name: str = 'mse',
                layer_sizes: Sequence[int] = [1, 30, 1], 
                learning_rate: np.float64 = np.float64(0.01), 
                epochs: int = 100, 
                epoch_output: int = 10, 
                batch_size: int = 16) -> None:
        """
        При выборе гиперпараметров стоит ориентироваться на компромисс «скорость-стабильность»:

        Шаг обучения (learning rate). Большой шаг ускоряет обучение, но может вызвать расходимость. Маленький шаг делает обучение медленным. Часто применяют адаптивные алгоритмы (Adam, RMSProp), которые позволяют использовать большие шаги без сильных колебаний.

        Размер батча. Обычно выбирают батч средней величины (например, 32 или 64) и настраивают шаг. При росте батча часто приходится увеличивать и шаг обучения. Помните, что слишком малый батч может «скакать» вокруг минимума, а слишком крупный — тратить лишние ресурсы.

        Мониторинг обучения. Визуализируйте поведение модели по эпохам: строите графики функции потерь или метрик качества на тренировочном и валидационном наборах. Резкое ухудшение метрик может указывать на слишком большой шаг или на переобучение. Графики позволяют увидеть шум и стабильность.

        :param: batch_size: int
        
        Вот что полезно учитывать, выбирая размер батча:

        Сходимость. При малом батче алгоритм делает много быстрых шагов (больше обновлений за эпоху), из-за чего может потребовать больше эпох для полной сходимости. Крупный батч обновляет параметры реже, но шаги более уверенные.

        Обобщающая способность. Малые батчи (вплоть до SGD) обычно дают лучшие результаты на тесте из-за регуляризации шума, крупные батчи могут хуже обобщать (generalization gap).

        Шумность. Шум градиента обратно пропорционален размеру батча. Малый батч — большой шум, большой батч — малый шум.

        :param: loss_function_name: str

        Using this function names:

        mae,

        mse,

        huber,

        categorical_cross_entropy
        """
        self.layer_sizes: Sequence[int] = layer_sizes
        self.lr: np.float64 = np.float64(learning_rate)
        self.epochs: int = epochs
        self.epoch_output: int = epoch_output
        self.batch_size: int = batch_size
        self.loss_history: List[Union[np.floating[Any], np.complexfloating[Any, Any]]] = []
        self.grad_norms: List[Union[np.floating[Any], np.complexfloating[Any, Any]]] = []
        self.activation: str = activation
        self.X: NDArray[np.float64] = np.asarray(X, dtype=np.float64)
        self.y: NDArray[np.float64] = np.asarray(y, dtype=np.float64)
        self.model = model
        if loss_function_name == 'mse':
            self.loss_function_name = loss_function_name
        else:
            raise ValueError(f'This loss function is not exist now: {loss_function_name}')
        
        if self.model == 'MLP':
            try:
                self.estimator = MLP(
                    layer_sizes=self.layer_sizes, 
                    activation=self.activation)
            except Exception as e:
                raise ValueError(f'Warning by creating estimator model: {e}')
        
    def _loss_function(self, 
                      preds: NDArray[np.float64],
                      y_true: NDArray[np.float64]) -> np.floating[Any] | np.complexfloating[Any, Any]:
        if self.loss_function_name == 'mse':
            return mse(y_pred=preds, y_true=y_true)
        elif self.loss_function_name == 'mae':
            return mae(y_pred=preds, y_true=y_true)
        elif self.loss_function_name == 'huber':
            return huber(y_pred=preds, y_true=y_true)
        elif self.loss_function_name == 'categorical_cross_entropy':
            return categorical_cross_entropy(y_pred=preds, y_true=y_true)
        else:
            raise ValueError(f'This loss function is not found: {self.loss_function_name}')

        
    def learn_model(self, 
                    GD_type: str = 'mini-batch_GD',
                    update_params_type: str = 'SGD',
                    learning_rate_strategy: str | None=None,
                    decay_factor: float | None=None,
                    k_for_decay_step: int | None=None):
        """
        Совет по выбору стратегии (GD_type): при малых данных до нескольких тысяч можно использовать полный GD (batch_GD) для стабильности. На очень больших данных и в онлайн-режиме применяют SGD (stohastic gradient descent) или мини-батч (mini-batch_GD). mini-batch_GD считается самым распространённым для нейросетей благодаря балансу «шум/вычислительность».

        Специфика обновления параметров модели (update_params_type):
        Если модель не сходится или переобучается, можно попробовать поменять оптимизатор: например, заменить SGD на Adam или наоборот, а потом сравнить результаты их работы.
        """
        self.GDtype: str = GD_type
        self.update_params_type: str = update_params_type
        if GD_type == 'mini-batch_GD':
            self._mini_batch_gd(learning_rate_strategy,
                                decay_factor,
                                k_for_decay_step)

        elif GD_type == 'batch_GD':
            self._batch_GD(learning_rate_strategy,
                      decay_factor,
                      k_for_decay_step)

        elif GD_type == 'SGD':
            self._SGD(learning_rate_strategy,
                      decay_factor,
                      k_for_decay_step)
        else:
            raise ValueError(f'This GD is not exist in that module: {GD_type}')

    def visualise_learning_process(self) -> None:
        """
            Пока строяться графики на матрице 2 * 2:

            1.Зависимость функции потерь от эпохи
            
            2.Зависимость норм градиентов от шага обучения

            3.Предсказание и факт - X[0] от Y 

            4.Пустой график
        """
        # Визуализация
        plt.figure(figsize=(12,8))
        plt.subplot(2,2,1)
        plt.plot(self.loss_history); plt.title(f"{self.loss_function_name} по эпохам"); plt.xlabel("Эпоха"); plt.grid(True)
        # Правая панель: ||grad||, оранжевым
        plt.subplot(2, 2, 2)
        plt.plot(self.grad_norms, linestyle='-', color='tab:orange')
        plt.title(f"{self.GDtype}: Норма градиента vs шаг GD")
        plt.xlabel("Шаг обучения")
        plt.ylabel("||grad||")
        plt.grid(True)
        # Апроксимация
        plt.subplot(2,2,3)
        plt.scatter(self.X[:, 0], self.y, label="Целевое распределение", c='green', alpha=.5)
        plt.plot(self.X[:, 0], self.estimator.forward(self.X), label="MLP", c='r')
        plt.legend(); plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Model {self.model} estimation with GD type: {self.GDtype}")
        # Выводим результат
        plt.tight_layout()
        plt.show()

    def get_learned_estimator(self) -> MLP:
        return self.estimator
    
    #######################
    #    GRADIEND         #
    #    descents         #
    #                     #
    #######################
    def _mini_batch_gd(self,
                       learning_rate_strategy: str | None = None,
                       decay_factor: float | None = None,
                       k_for_decay_step: int | None = None,
                       k_for_exp_fall: float| None = None,
                       k_for_one_on_t: float | None = None,
                       patience: int| None = None,
                       wait: int | None = None,
                       X_val = None,
                       Y_val = None) -> None:
            
            """
            learning_rate_strategy from ["step_decay", "exp_fall"(exponential_fall)]
            """
            print(f"Using Mini-batch GD with learning rate update strategy: {learning_rate_strategy}")
            for epoch in range(1, self.epochs+1):
                if learning_rate_strategy == "exp_fall" and k_for_exp_fall:
                    self.lr *= np.exp(-k_for_exp_fall * epoch)
                elif learning_rate_strategy == "one_on_t" and k_for_one_on_t:
                    self.lr /= (1 + k_for_one_on_t * epoch)

                # Перетасовываем, для избежания быстрого переобучения
                perm: NDArray[np.long] = np.random.permutation(len(self.X))
                # Выделяем только те части, которые попали от permutation
                X_sh, y_sh = self.X[perm], self.y[perm]
                # mini-batch learning:
                for start in range(0, len(self.X), self.batch_size):
                    xb: NDArray[np.float64] = X_sh[start : start + self.batch_size]
                    yb: NDArray[np.float64] = y_sh[start : start + self.batch_size]
                    # Прямой проход
                    _ = self.estimator.forward(xb)
                    # Обратный проход
                    self.estimator.backward(yb)
                    grads: NDArray[np.float64] = np.concatenate(
                        [g.ravel() for g in self.estimator.dW_list] + 
                        [f.ravel() for f in self.estimator.db_list])
                    self.grad_norms.append(np.linalg.norm(grads))
                    # Обновление параметров
                    self.estimator.update_params(self.lr)
                # Прямой проход на всех данных необходим для вычисления потерь
                full_pred: NDArray[np.float64] = self.estimator.forward(self.X)
                # Запомниаем текущие потери
                self.loss_history.append(
                    self._loss_function(y_true=self.y, preds=full_pred))
                # Будем выводить статистику обучения только каждые N epoch
                # Примем во внимание возможные стратегии обновления шага обучения
                if learning_rate_strategy == 'step_decay'\
                    and k_for_decay_step \
                    and decay_factor:

                    if epoch % k_for_decay_step == 0:
                        self.lr *= decay_factor

                if epoch % self.epoch_output == 0:
                    print(f"Epoch {epoch:3d}, loss={self.loss_history[-1]:.4f}")
                
                if \
                    learning_rate_strategy == 'reduce_on_plateu'\
                    and patience \
                    and wait \
                    and X_val \
                    and Y_val:
                    best_val_loss = np.inf
                    val_preds = self.estimator.forward(X_val)
                    val_loss = self._loss_function(val_preds, Y_val)  # например, MSE
                    print(f"Epoch {epoch}: val_loss = {val_loss:.4f}, lr = {self.lr:.5f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            self.lr *= 0.5
                            wait = 0
                            print(f"  → Plateau reached. New lr = {self.lr:.5f}") 



    def _batch_GD(self,
                   learning_rate_strategy: str | None = None,
                    decay_factor: float | None = None,
                    k_for_decay_step: int | None = None,
                    k_for_exp_fall: float| None = None,
                    k_for_one_on_t: float | None = None,
                    patience: int| None = None,
                    wait: int | None = None,
                    X_val = None,
                    Y_val = None) -> None:
        for epoch in range(1, self.epochs+1):
            if learning_rate_strategy == "exp_fall" and k_for_exp_fall:
                    self.lr *= np.exp(-k_for_exp_fall * epoch)
            elif learning_rate_strategy == "one_ont_t" and k_for_one_on_t:
                self.lr /= (1 + k_for_one_on_t * epoch)

            preds = self.estimator.forward(self.X)
            loss = self._loss_function(preds=preds, y_true=self.y)
            self.loss_history.append(loss)

            self.estimator.backward(self.y)
            # Градиент смещения и весов объединяем
            grads: NDArray[np.float64] = np.concatenate(
                [g.ravel() for g in self.estimator.dW_list] + 
                [f.ravel() for f in self.estimator.db_list])
            # Норма градиента в список
            self.grad_norms.append(np.linalg.norm(grads))

            self.estimator.update_params(self.lr)
            # Примем во внимание возможные стратегии обновления шага обучения
            if learning_rate_strategy == 'step_decay'\
                and k_for_decay_step \
                and decay_factor:

                if epoch % k_for_decay_step == 0:
                    self.lr *= decay_factor
            if epoch % self.epoch_output == 0:
                print(f"Epoch {epoch}: loss={loss:.4f}, ||grad||={self.grad_norms[-1]:.4f}")
            
            if \
                learning_rate_strategy == 'reduce_on_plateu'\
                and patience \
                and wait \
                and X_val \
                and Y_val:
                best_val_loss = -np.inf
                val_preds = self.estimator.forward(X_val)
                val_loss = self._loss_function(val_preds, Y_val)  # например, MSE
                print(f"Epoch {epoch}: val_loss = {val_loss:.4f}, lr = {self.lr:.5f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        self.lr *= 0.5
                        wait = 0
                        print(f"  → Plateau reached. New lr = {self.lr:.5f}")
    
    def _SGD(self,
            learning_rate_strategy: str | None = None,
            decay_factor: float | None = None,
            k_for_decay_step: int | None = None,
            k_for_exp_fall: float| None = None,
            k_for_one_on_t: float | None = None,
            patience: int| None = None,
            wait: int | None = None,
            X_val = None,
            Y_val = None) -> None:
        for epoch in range(1, self.epochs+1):
            if learning_rate_strategy == "exp_fall" and k_for_exp_fall:
                    self.lr *= np.exp(-k_for_exp_fall * epoch)
            elif learning_rate_strategy == "one_ont_t" and k_for_one_on_t:
                self.lr /= (1 + k_for_one_on_t * epoch)
            # Перемешайте индексы наблюдений
            perm = np.random.permutation(len(self.X))
            for i in perm:
                # Получите xi и yi из X и Y
                xi, yi = self.X[i:1 + i], self.y[i: i+1]
                # Прямой и обратный проход внутри цикла
                self.estimator.forward(xi)
                self.estimator.backward(yi)
                grads = np.concatenate(
                    [g.ravel() for g in self.estimator.dW_list] +
                    [g.ravel() for g in self.estimator.db_list])
                self.grad_norms.append(np.linalg.norm(grads))
                self.estimator.update_params(self.lr)
            
            if learning_rate_strategy == 'step_decay'\
                    and k_for_decay_step \
                    and decay_factor:

                    if epoch % k_for_decay_step == 0:
                        self.lr *= decay_factor
            loss = self._loss_function(self.estimator.forward(self.X), self.y)
            self.loss_history.append(loss)
            if epoch % self.epoch_output == 0:
                print(f"Epoch {epoch}: loss={loss:.4f}, ||grad||={self.grad_norms[-1]:.4f}")
            if \
                learning_rate_strategy == 'reduce_on_plateu'\
                and patience \
                and wait \
                and X_val \
                and Y_val:
                best_val_loss = -np.inf
                val_preds = self.estimator.forward(X_val)
                val_loss = self._loss_function(val_preds, Y_val)  # например, MSE
                print(f"Epoch {epoch}: val_loss = {val_loss:.4f}, lr = {self.lr:.5f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        self.lr *= 0.5
                        wait = 0
                        print(f"  → Plateau reached. New lr = {self.lr:.5f}")
