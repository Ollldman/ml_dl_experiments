import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Union


def categorical_cross_entropy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> Union[np.floating[Any], np.complexfloating[Any, Any]]:
    """
    Compute the Categorical Cross-Entropy between true and softmax(z).

    В многоклассовой классификации на выходном слое часто стоит softmax, дающий вектор вероятностей ypred по K классам. Функция измеряет разницу между этим вектором и истинным one-hot ytrue:
    
    Когда применять.
    ----------
    Во всех нейросетях для многоклассовых задач — классификация изображений, текста и т. д.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    

    Returns
    -------
    categorical_cross_entropy : np.floating | np.complexfloating
        a scalar of type float32 or float64,
        уверенность модели в классификации классов.

    Raises
    ------
    ValueError
        If shapes of `y_true` and `y_pred` do not match.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps: np.float64 = np.float64(1e-15)
    p = np.clip(y_pred, eps, 1-eps)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Input arrays must have the same shape. Got {y_true.shape} and {p.shape}.")
    return -np.mean(np.sum(y_true * np.log(p), axis=1))