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

    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y e groups devem ter o mesmo número de linhas.")
    if not 0 < test_size < 1:
        raise ValueError("test_size deve estar entre 0 e 1.")

    grupos_unicos = np.unique(groups)
    n_grupos = len(grupos_unicos)
    n_teste = round(test_size * n_grupos)
    n_teste = min(
        max(n_teste, 1), n_grupos - 1
    )  # pelo menos 1 grupo no teste, pelo menos 1 no treino

    rng = np.random.default_rng(random_state)
    grupos_teste = rng.choice(grupos_unicos, size=n_teste, replace=False)
    mascara_teste = np.isin(groups, grupos_teste)

    X_train, X_test = X[~mascara_teste], X[mascara_teste]
    y_train, y_test = y[~mascara_teste], y[mascara_teste]

    print(
        f"treino: {len(X_train)} linhas, "
        f"{len(np.unique(groups[~mascara_teste]))} registros, "
        f"classes {np.bincount(y_train, minlength=3)}"
    )
    print(
        f"teste:  {len(X_test)} linhas, "
        f"{len(np.unique(groups[mascara_teste]))} registros, "
        f"classes {np.bincount(y_test, minlength=3)}"
    )

    return X_train, X_test, y_train, y_test
