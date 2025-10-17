import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Any, Union


def mse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> Union[np.floating[Any], np.complexfloating[Any, Any]]:
    """
    Compute the Mean Squared Error (MSE) between true and predicted values.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    mse : np.floating | np.complexfloating
        The mean squared error; a scalar of type float32 or float64,
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

    return np.mean((y_true - y_pred) ** 2)