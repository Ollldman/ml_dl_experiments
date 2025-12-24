import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Union


def log_loss(
    y_true: ArrayLike,
    p: ArrayLike,
) -> Union[np.floating[Any], np.complexfloating[Any, Any]]:
    """
    Compute the Binary Cross-Entropy between true and sigmoid(z).

    Для бинарной классификации модель выдаёт логиты z, из которых через сигмоиду получаем вероятность класса 1: p=σ(z). Binary CE измеряет «достоверность» этих вероятностей через энтропию (логарифмическую меру неопределённости)
    
    Когда применять.
    ----------
    В любых бинарных задачах классификации — от логистической регрессии до двоичных нейросетей.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    p : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    

    Returns
    -------
    log_loss : np.floating | np.complexfloating
        a scalar of type float32 or float64,
        уверенность модели в классификации.

    Raises
    ------
    ValueError
        If shapes of `y_true` and `p` do not match.
    """
    y_true = np.asarray(y_true)
    eps: np.float64 = np.float64(1e-15)
    p = np.clip(p, eps, 1-eps)
    if y_true.shape != p.shape:
        raise ValueError(f"Input arrays must have the same shape. Got {y_true.shape} and {p.shape}.")
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1-p))