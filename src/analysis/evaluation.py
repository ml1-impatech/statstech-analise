"""Métricas no mesmo formato para todos os modelos."""

import numpy as np
from numpy.typing import ArrayLike
from statstech.metrics.classification import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    macro_f1,
    precision_score,
    recall_score,
)
from statstech.metrics.regression import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def evaluate_regressor(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Calcula as métricas de regressão de um modelo.

    Parameters
    ----------
    y_true : array-like de formato (n_amostras,)
        Valores reais do alvo.

    y_pred : array-like de formato (n_amostras,)
        Valores previstos pelo modelo.

    Returns
    -------
    metrics : dict
        Dicionário com as chaves "rmse", "mae" e "r2", nessa ordem.
    """
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def evaluate_classifier(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Calcula as métricas de classificação de um modelo.

    As classes são as que aparecem em y_true ou em y_pred, em ordem
    crescente. Para cada classe k, o dicionário tem a precisão, a
    revocação e o F1 de k contra as demais.

    Parameters
    ----------
    y_true : array-like de formato (n_amostras,)
        Classes reais.

    y_pred : array-like de formato (n_amostras,)
        Classes previstas pelo modelo.

    Returns
    -------
    metrics : dict
        Dicionário com as chaves "accuracy" e "f1_macro" e, para cada
        classe k, "precision_k", "recall_k" e "f1_k", nessa ordem.

    Warns
    -----
    UserWarning
        Se alguma classe nunca for prevista ou nunca aparecer em
        y_true: a métrica de denominador zero vale 0 (ver statstech).
    """

    y_true = np.asarray(y_true).tolist()
    y_pred = np.asarray(y_pred).tolist()
    classes = np.unique(y_true + y_pred).tolist()
    cm = confusion_matrix(y_true, y_pred, classes)
    metrics = {}
    metrics["accuracy"] = accuracy_score(cm)
    metrics["f1_macro"] = macro_f1(cm)

    for index, label in enumerate(classes):
        metrics[f"precision_{label}"] = precision_score(cm, index)
        metrics[f"recall_{label}"] = recall_score(cm, index)
        metrics[f"f1_{label}"] = f1_score(cm, index)

    return metrics
