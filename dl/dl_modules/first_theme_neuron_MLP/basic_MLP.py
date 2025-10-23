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
            momentum: np.float64 | None = None, 
            beta: np.float64 | None = None,
            beta_two: np.float64 | None = None,
            weight_decay: np.float64 | None = None,
            activation: str = 'sigmoid')-> None:
        # Архитектура сети
        self.layer_sizes: NDArray[np.integer] = np.asarray(layer_sizes, dtype=int)
        # Функция активации
        self.activation: str = activation
        # список матриц весов
        self.W: List[NDArray[np.float64]] = []
        # список векторов смещений
        self.b: List[NDArray[np.float64]] = []
        # Список градиентов векторов весов
        self.dW_list: List[NDArray[np.float64]] = []
        # Список градиентов вектора смещений
        self.db_list: List[NDArray[np.float64]] = []
        # Скорость обновления весов каждого слоя
        self.velocity_W: List[NDArray[np.float64]] = []  
        # Скорость обновления смещений каждого слоя
        self.velocity_b: List[NDArray[np.float64]] = []
        # момент импульса для обновления весов-смещений
        self.momentum: np.float64  | None = momentum if momentum else None
        # Сумма квадратов граиента весов
        self.G_W: List[NDArray[np.float64]] = []
        # Сумма квадратов градиента смещений
        self.G_b: List[NDArray[np.float64]] = []
        # Экспоненциально накопленнае скользащее среднее квадрата граиента весов слоя
        self.S_W: List[NDArray[np.float64]] = []
        # Экспоненциально накопленнае скользащее среднее квадрата граиента смещений слоя
        self.S_b: List[NDArray[np.float64]] = []
        # Скользящее среднее квадрата градиента:
        self.V_W: List[NDArray[np.float64]] = []
        self.V_b: List[NDArray[np.float64]] = []
        # Коэффициент затухания:
        self.beta: np.float64 | None = beta if beta else None
        self.beta_two: np.float64 | None = beta_two if beta_two else None
        self.weight_decay: np.float64 | None = weight_decay if weight_decay else None
        # список net-inputs каждого слоя
        self.Z_list: List[NDArray[np.float64]] = [] 
        # список результатов активации каждого слоя
        self.A_list: List[NDArray[np.float64]] = [] 
        # Коэффициент для деления на ноль:
        self.epsilon: float = 10e-8
        self.iterations: int = 0

        # Инициализация весов и смещений сети
        # В цикле заполнить self.W и self.b случайными параметрами
        for i in range(len(layer_sizes) - 1):
            in_dim: int = int(layer_sizes[i])
            out_dim: int = int(layer_sizes[i + 1])
            self.W.append(np.random.randn(in_dim, out_dim) * 0.1)
            self.b.append(np.zeros((1, out_dim)))
            #  Инициализируем скорости весов и смещений для обновления через момент
            if self.momentum:  
                self.velocity_W.append(np.zeros((in_dim, out_dim), dtype=np.float64))
                # Скорость обновления смещений каждого слоя
                self.velocity_b.append(np.zeros((1, len(self.b)), dtype=np.float64))
            if self.beta:
                # Скорость обновления весов каждого слоя
                self.S_W.append(np.zeros((in_dim, out_dim), dtype=np.float64))
                # Скорость обновления смещений каждого слоя
                self.S_b.append(np.zeros((1, len(self.b)), dtype=np.float64))
                if self.beta_two:
                    # Моменты для смещений и весов
                    self.V_W.append(np.zeros((in_dim, out_dim), dtype=np.float64))
                    self.V_b.append(np.zeros((1, len(self.b)), dtype=np.float64))
            

    def _activation(
            self,
            Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Возвращает функцию активации из доступных
        """
        if self.activation == 'sigmoid':
            return sigmoid(Z)
        elif self.activation == 'relu':
            return relu(Z)
        elif self.activation == 'tanh':
            return tanh(Z)
        elif self.activation == 'sin':
            return sin_f(Z)
        else:
            raise ValueError(f'Unknown activation {self.activation}')
        
    def _activation_derivative(
            self,
            z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Возвращает производную функции активации из доступных
        """
        if self.activation == 'sigmoid':
            return sigmoid_derivative(z)
        elif self.activation == 'tanh':
            return tanh_derivative(z)
        elif self.activation == 'sin':
            return sin_derivative(z)
        else:
            raise ValueError(f'Unknown activation {self.activation}')

    # Метод forward    
    def forward(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Forward to MLP
        При первом вызове заполняет значения Z и активаций для каждого слоя.

        Возвращает первое предсказание. (активацию на последнем слое)

        Последующие запуски используются для получения промежуточных предсказаний,
        для анализа функции потерь в процессе обучения.

        :return: y_pred: NDArray[np.float64], предсказание модели
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
                A = self._activation(Z)
            else:
                A = Z  # линейный выход для последнего слоя
            # Добавляем получившуюся активацию
            self.A_list.append(A)
        return A
      
    def backward(self, y_true: NDArray[np.float64]) -> None:
        """
        Реализация алгоритма `backpropogation` и обратного распространения ошибки.

        Используется цепное правило для определения `delta` на каждом слое нейросети.

        `delta` используется для расчета градиентов векторов весов и смещений для каждого слоя.

        (Сейчас добавлена дробь `2 / m`  для стабилизации delta в случае большого батча при инициализации на входном слое.)
        """
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
            delta = (delta @ W_next.T) * self._activation_derivative(z)

            # Градиенты для слоя l
            dW = a_prev.T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            # Вставляем в начало, чтобы порядок совпадал с W и b
            self.dW_list.insert(0, dW)
            self.db_list.insert(0, db)


    def update_params(self, lr: np.float64) -> None:
        """Обновление весов и смещений в цикле
        Через градиентный спуск
        Есди в модель передан параметр momentum,
        то используется обновление скоростей весов и смещения через момент инерции

        SGD с momentum прост и хорошо подходит, когда можно тщательно настроить обучение. Метод известен как «конвейер» для задач компьютерного зрения. 

        """
        for i in range(len(self.W)):
            if self.momentum:
                self.velocity_W[i] = self.momentum * \
                    self.velocity_W[i] - lr * self.dW_list[i]
                self.velocity_b[i] = self.momentum * \
                    self.velocity_b[i] - lr * self.db_list[i]
                self.b[i] += self.velocity_W[i]
                self.W[i] += self.velocity_b[i]
            else:
                self.W[i] -= lr * self.dW_list[i]
                self.b[i] -= lr * self.db_list[i]

    def update_adagrad(self, lr: np.float64) -> None:
        """
        AdaGrad (Adaptive Gradient)
        ---------------------------
        `Идея`:  уменьшать шаг для тех параметров, у которых накоплен большой градиент, и оставлять большой шаг там, где градиенты редкие.

        AdaGrad очень прост, но «умирает» из-за неограниченного накопления.

        Adagrad в современных нейросетях используют редко — из-за «затухания» шага.

        КАК ЭТО РАБОТАЕТ:
        -----------------
        Если по wi​ часто проходят большие градиенты, то si​ быстро растёт. 
        Знаменатель дроби становится большим, и эффективный шаг η/si​ уменьшается.

        Если градиенты редкие или маленькие, то si​ остаётся малым, а значит, шаг остаётся близким к η.

        Однако если si​ только растёт со временем и никогда не обнуляется, то шаг будет неуклонно падать и может стать слишком малым.
        """
        for i in range(len(self.W)):
            # аккумулируем сумму квадратов градиентов
            self.G_W[i] += self.dW_list[i] ** 2  
            self.G_b[i] += self.db_list[i] ** 2
            # рассчитываем адаптивный шаг
            adj_lr_w = lr / (np.sqrt(self.G_W[i]) + self.epsilon)
            adj_lr_b = lr / (np.sqrt(self.G_b[i]) + self.epsilon)
            # обновляем параметры с адаптивным шагом
            self.W[i] -= adj_lr_w * self.dW_list[i]
            self.b[i] -= adj_lr_b * self.db_list[i] 


    def update_rmsprop(self, lr: np.float64) -> None:
        """
        RMSProp: модификация AdaGrad
        ----------------------------

        RMSProp добавляет «забывание» старых градиентов, благодаря чему шаг остаётся адаптивным и вдобавок не «умирает».

        RMSProp аналогичен Adam по адаптивности, но без компенсирования смещения моментов. Благодаря этому он чуть дешевле в вычислениях и проще в настройке. 

        Идея: не накапливать квадраты градиентов до бесконечности, а использовать скользящее (экспоненциальное) среднее, чтобы шаг не «умирал». 

        Необходим коэффициент затухания  beta (Например 0.9)

        По умолчанию списки инициализируеются нулями при создании модели.
        """
        if self.beta:
            for i in range(len(self.W)):
                # обновляем скользящее среднее квадрата градиента
                self.S_W[i] = self.beta * self.S_W[i] + (1 - self.beta) * (self.dW_list[i] ** 2)
                self.S_b[i] = self.beta * self.S_b[i] + (1 - self.beta) * (self.db_list[i] ** 2)
                # вычисляем адаптивный шаг
                adj_lr_w = lr / (np.sqrt(self.S_W[i]) + self.epsilon)
                adj_lr_b = lr / (np.sqrt(self.S_b[i]) + self.epsilon)
                # обновляем параметры
                self.W[i] -= adj_lr_w * self.dW_list[i]
                self.b[i] -= adj_lr_b * self.db_list[i] 
        else:
            self.update_params(lr)


    def update_adam(self, lr: np.float64) -> None:
        """
        Adam сочетает импульс и RMSProp, автоматически подбирая шаг на каждый параметр.

        Adam популярнее в исследованиях и NLP, так как автоматически подстраивает шаги и требует меньше ручной настройки. В NLP встречаются очень разреженные градиенты: слов в тексте миллионы, но большинство слов для конкретного батча «неактивны». Adam адаптирует шаг для редких и частых параметров автоматически. 
        """
        if self.beta and self.beta_two:
            self.iterations += 1
            for i in range(len(self.W)):
                # обновление первого момента (скользящее среднее градиента)
                self.V_W[i] = self.beta * self.V_W[i] + (1 - self.beta) * self.dW_list[i]
                self.V_b[i] = self.beta * self.V_b[i] + (1 - self.beta) * self.db_list[i]
                # обновление второго момента (скользящее среднее квадрата градиента)
                self.S_W[i] = self.beta_two * self.S_W[i] + (1 - self.beta_two) * (self.dW_list[i] ** 2)
                self.S_b[i] = self.beta_two * self.S_b[i] + (1 - self.beta_two) * (self.db_list[i] ** 2)
                # коррекция смещения моментов (bias correction)
                V_corr_w = self.V_W[i] / (1 - self.beta ** self.iterations)
                V_corr_b = self.V_b[i] / (1 - self.beta ** self.iterations)
                S_corr_w = self.S_W[i] / (1 - self.beta_two ** self.iterations)
                S_corr_b = self.S_b[i] / (1 - self.beta_two ** self.iterations)
                # обновляем параметры
                self.W[i] -= lr * V_corr_w / (np.sqrt(S_corr_w) + self.epsilon)
                self.b[i] -= lr * V_corr_b / (np.sqrt(S_corr_b) + self.epsilon) 
        else:
            self.update_params(lr)

    def update_adamw(self, lr: np.float64) -> None:
        """
        AdamW дополнительно аккуратно выносит регуляризацию в отдельный член, улучшая обобщение.

        AdamW — стандарт для трансформеров и современных архитектур, где важна регуляризация. 
        """
        if self.beta and self.beta_two and self.weight_decay:
            self.iterations += 1
            for i in range(len(self.W)):
                self.W[i] -= lr * self.weight_decay * self.W[i]
                # обновление первого момента (скользящее среднее градиента)
                self.V_W[i] = self.beta * self.V_W[i] + (1 - self.beta) * self.dW_list[i]
                self.V_b[i] = self.beta * self.V_b[i] + (1 - self.beta) * self.db_list[i]
                # обновление второго момента (скользящее среднее квадрата градиента)
                self.S_W[i] = self.beta_two * self.S_W[i] + (1 - self.beta_two) * (self.dW_list[i] ** 2)
                self.S_b[i] = self.beta_two * self.S_b[i] + (1 - self.beta_two) * (self.db_list[i] ** 2)
                # коррекция смещения моментов (bias correction)
                V_corr_w = self.V_W[i] / (1 - self.beta ** self.iterations)
                V_corr_b = self.V_b[i] / (1 - self.beta ** self.iterations)
                S_corr_w = self.S_W[i] / (1 - self.beta_two ** self.iterations)
                S_corr_b = self.S_b[i] / (1 - self.beta_two ** self.iterations)
                # обновляем параметры
                self.W[i] -= lr * V_corr_w / (np.sqrt(S_corr_w) + self.epsilon)
                self.b[i] -= lr * V_corr_b / (np.sqrt(S_corr_b) + self.epsilon) 
        else:
            self.update_params(lr)

    def forward_with_activations(self, X: ArrayLike) -> List[NDArray[np.float64]]:
        """
            This function return list of forward activations to all layers in MLP
            
            Не используется для обычной работы и обучения и создан для анализа!!!!
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