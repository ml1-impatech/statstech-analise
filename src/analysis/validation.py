"""Divisões e previsões por validação cruzada."""

import numpy as np

from analysis.config import SEED, TEST_SIZE

def group_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separa treino e teste sorteando grupos inteiros, não linhas.

    Segmentos do mesmo registro são parecidos entre si. Se um fosse para
    o treino e outro para o teste, o resultado ficaria inflado, como um
    paciente repetido. Aqui todas as linhas de um grupo vão para o mesmo
    lado.

    Parameters
    ----------
    X : ndarray of shape (n_amostras, n_features)
        Matriz de dados.
    y : ndarray of shape (n_amostras,)
        Rótulos.
    groups : ndarray of shape (n_amostras,)
        Grupo de cada linha (por exemplo, o registro de origem).
    test_size : float, default=TEST_SIZE
        Fração dos grupos sorteados para o teste.
    random_state : int, default=SEED
        Semente do gerador de números aleatórios.

    Returns
    -------
    X_train, X_test : ndarray
        Linhas dos grupos de treino e de teste.
    y_train, y_test : ndarray
        Rótulos correspondentes.

    Notes
    -----
    A fração de linhas no teste não é exatamente test_size, porque os
    grupos têm tamanhos diferentes. A proporção das classes também pode
    diferir entre treino e teste, e deve ser conferida depois da
    divisão.
    """

    pass