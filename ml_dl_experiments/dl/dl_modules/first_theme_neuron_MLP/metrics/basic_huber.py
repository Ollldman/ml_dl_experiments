import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Union


def huber(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    delta: np.float64 = np.float64(1.)
) -> Union[np.floating[Any], np.complexfloating[Any, Any]]:
    """
    Compute the Huber Loss (MAE and MSE composition) between true and predicted values.

    Для ошибок меньше `delta` действует квадратичный штраф, а для крупных — линейный. Это даёт плавность около нуля и одновременно уменьшает влияние выбросов
    
    Когда применять.
    ----------
    Если в данных есть выбросы, но вы хотите сохранить гладкую скорость изменения функции потерь для мелких ошибок.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    
    delta : np.float64 number задаёт порог, где поведение переходит из квадратичного (MSE-like) в линейное (MAE-like)

    Returns
    -------
    huber : np.floating | np.complexfloating
        a scalar of type float32 or float64,
        matching the precision of the input arrays.

    Raises
    ------
    ValueError
        If shapes of `y_true` and `y_pred` do not match.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Input arrays must have the same shape. Got {y_true.shape} and {y_pred.shape}.")
    
    diff: ArrayLike = y_pred - y_true
    small: NDArray[np.bool] = np.abs(diff) <= delta
    squared: ArrayLike = 0.5 * diff**2
    linear: ArrayLike = delta * (np.abs(diff) - 0.5 * delta)
    return np.mean(np.where(small, squared, linear))