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

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    n_test = round(test_size * n_groups)
    # pelo menos 1 grupo no teste, pelo menos 1 no treino
    n_test = min(max(n_test, 1), n_groups - 1)  

    rng = np.random.default_rng(random_state) 
    test_groups = rng.choice(unique_groups, size=n_test, replace=False)
    test_mask = np.isin(groups, test_groups)

    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    print(f"treino: {len(X_train)} linhas, "
          f"{len(np.unique(groups[~test_mask]))} registros, "
          f"classes {np.bincount(y_train, minlength=3)}")
    print(f"teste:  {len(X_test)} linhas, "
          f"{len(np.unique(groups[test_mask]))} registros, "
          f"classes {np.bincount(y_test, minlength=3)}")

    return X_train, X_test, y_train, y_test


class StratifiedGroupKFold:
    """K folds que sorteiam registros inteiros e equilibram as classes.
    
    Parameters
    ----------
    groups : array-like de formato (n_amostras,)
        Grupo de cada linha, na mesma ordem das linhas de X e y
        passados ao split.
    n_splits : int, default=10
        Quantidade de folds. Deve ser pelo menos 2 e no máximo o
        número de grupos.
    random_state : int, default=SEED
        Semente do sorteio. Ela desempata grupos de mesmo peso e folds
        igualmente bons.
    
    Raises
    ------
    ValueError
        Se n_splits for menor que 2 ou maior que o número de grupos.
    """
    
    def __init__(
        self, groups: np.ndarray, n_splits: int = 10, random_state: int = SEED
    ) -> None:
        self.groups = np.asarray(groups)
        self.n_splits = n_splits
        self.random_state = random_state
        n_groups = len(np.unique(self.groups))
        if not 2 <= n_splits <= n_groups:
            raise ValueError(
                f"n_splits deve estar entre 2 e o número de grupos ({n_groups})."
            )
        
    def split(self, X: np.ndarray, y: np.ndarray | None = None):
        """Gera os índices de treino e de validação dde cada fold.
            
        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de entrada. 
        y : array-like de formato (n_amostras,)
            Classes, usadas para equilibrar os folds.
            
        Yields
        ------
        train_idx : ndarray
            Índices das linhas de treino do fold.
        test_idx : ndarray
            Índices das linhas de validação do fold.
            
        Raises
        ------
        Se X (ou y) não tiver o mesmo número de linhas que groups.
        """
        n = len(X)
        if n != len(self.groups):
            raise ValueError( f"X tem {n} linhas e groups tem {len(self.groups)}.")
        if y is not None and len(y) != n:
            raise ValueError(f"X tem {n} linhas e y tem {len(y)}.")
        
        unique_groups, group_idx = np.unique(self.groups, return_inverse=True)
        _, class_idx = np.unique(np.zeros(n) if y is None else y, return_inverse=True)
        
        counts = np.zeros((len(unique_groups), class_idx.max() + 1))
        np.add.at(counts, (group_idx, class_idx), 1)
        share = counts / counts.sum(axis=0)
        
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(unique_groups))
        order = order[np.argsort(-share[order].sum(axis=1), kind="stable")]
        
        fold_share = np.zeros((self.n_splits, share.shape[1]))
        group_fold = np.empty(len(unique_groups), dtype=int)
        for g in order: 
            cost = ((fold_share + share[g]) ** 2).sum(axis=1)
            target = rng.choice(np.flatnonzero(np.isclose(cost, cost.min())))
            group_fold[g] = target
            fold_share[target] += share[g]
        
        row_fold = group_fold[group_idx]
        indices = np.arange(n)
        for k in range(self.n_splits):
            is_val = row_fold == k
            yield indices[~is_val], indices[is_val]
            
            