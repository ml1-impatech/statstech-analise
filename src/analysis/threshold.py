"""Previsão com limiar para uma classe de interesse.

Em vez de prever sempre a classe mais provável, marca a classe positiva
(na CTG, o patológico) sempre que a probabilidade dela passa de um
limiar. O limiar é escolhido no treino, por validação cruzada, para
garantir um recall mínimo.
"""

import numpy as np
from numpy.typing import ArrayLike


THRESHOLDS = np.round(np.linspace(0.05, 0.95, 91), 2)

def predict_with_threshold(
    proba: ArrayLike, classes: ArrayLike, positive: int | str, threshold: float
) -> np.ndarray:
    """Prevê a classe mais provável, trocada por positive acima do limiar.

    O limiar só acrescenta previsões de positive, onde positive já é a
    classe mais provável, ela é mantida mesmo com probabilidade abaixo
    de threshold.

    Parameters
    ----------
    proba : array-like de formato (n_amostras, n_classes)
        Probabilidade de cada classe, com as colunas na ordem de classes.
    classes : array-like de formato (n_classes,)
        Rótulos das classes, na ordem das colunas de proba.
    positive : int ou str
        Classe de interesse (na CTG, 2 = patológico).
    threshold : float
        Limiar: onde proba de positive for maior ou igual a ele, a
        previsão é positive.

    Returns
    -------
    pred : ndarray de formato (n_amostras,)
        Classe prevista para cada amostra.

    Raises
    ------
    ValueError
        Se positive não estiver em classes.
    """
    classes = np.asarray(classes)
    proba = np.asarray(proba)
    positive_idx = np.flatnonzero(classes == positive)
    if len(positive_idx) == 0:
        raise ValueError("positive não está em classes")
    positive_idx = positive_idx[0]
    base_pred = classes[np.argmax(proba, axis=1)]
    pred = np.where(proba[:, positive_idx] >= threshold, positive, base_pred)
    return pred


def threshold_curve(
    proba: ArrayLike, y: ArrayLike, positive: int | str, thresholds: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula o recall e a precisão de positive para cada limiar.

    As colunas de proba devem seguir a ordem das classes ordenadas de y,
    como no cross_val_predict com method="predict_proba".

    Parameters
    ----------
    proba : array-like de formato (n_amostras, n_classes)
        Probabilidades fora da amostra de cada classe.
    y : array-like de formato (n_amostras,)
        Classes verdadeiras.
    positive : int ou str
        Classe de interesse.
    thresholds : array-like de formato (n_limiares,)
        Limiares a avaliar.

    Returns
    -------
    recall : ndarray de formato (n_limiares,)
        Fração das amostras de positive previstas como positive.
    precision : ndarray de formato (n_limiares,)
        Fração das previsões de positive que são positive de verdade.
        Vale NaN quando nenhuma amostra é prevista como positive.

    Raises
    ------
    ValueError
        Se o número de colunas de proba for diferente do número de
        classes em y.
    """
    y = np.asarray(y)
    proba = np.asarray(proba)
    classes = np.unique(y)
    if proba.shape[1] != len(classes):
        raise ValueError(
            "o número de classes em y é diferente do número de colunas de proba"
        )
    recall = np.empty(len(thresholds))
    precision = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        pred = predict_with_threshold(proba, classes, positive, t)
        recall[i] = np.mean(pred[y == positive] == positive)
        if np.any(pred == positive):
            precision[i] = np.mean(y[pred == positive] == positive)
        else:
            precision[i] = np.nan
    return recall, precision


def choose_threshold(
    proba: ArrayLike, y: ArrayLike, positive: int | str, min_recall: float = 0.9
) -> float:
    """Escolhe o maior limiar que garante um recall mínimo de positive.

    Testa os limiares 0.05, 0.06, ..., 0.95 e devolve o maior com recall
    de pelo menos min_recall. O maior preserva o máximo de precisão.

    Parameters
    ----------
    proba : array-like de formato (n_amostras, n_classes)
        Probabilidades fora da amostra de cada classe, com as colunas na
        ordem das classes ordenadas de y.
    y : array-like de formato (n_amostras,)
        Classes verdadeiras.
    positive : int ou str
        Classe de interesse.
    min_recall : float, default=0.9
        Recall mínimo exigido para positive.

    Returns
    -------
    threshold : float
        Maior limiar com recall de pelo menos min_recall.

    Raises
    ------
    ValueError
        Se nenhum limiar atingir min_recall.
    """
    thresholds = np.round(THRESHOLDS)
    recall, _ = threshold_curve(proba, y, positive, thresholds)
    max_t = 0
    for t, r in zip(thresholds, recall):
        if r >= min_recall:
            max_t = t
    if max_t == 0:
        raise ValueError(
            f"O recall pedido foi {min_recall}, o máximo "
            f"encontrado foi {recall.max():.3f}"
        )
    return max_t
