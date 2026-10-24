"""Salvar e carregar as bases já preparadas, em arquivos .npz (C2)."""

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis.config import PROCESSED

SCRIPTS = {
    "ctg": "prepare_ctg.py",
    "enem": "prepare_enem.py",
    "enem_onehot": "prepare_enem.py",
}


def save_dataset(
    name: str,
    X_train: ArrayLike,
    X_test: ArrayLike,
    y_train: ArrayLike,
    y_test: ArrayLike,
    feature_names: list[str],
) -> Path:
    """Salva treino e teste de uma base em data/processed/<name>.npz.

    Parameters
    ----------
    name : str
        Nome da base, sem extensão (por exemplo, "ctg", "enem" ou
        "enem_onehot").
    X_train : array-like de formato (n_treino, n_features)
        Preditores do conjunto de treino.
    X_test : array-like de formato (n_teste, n_features)
        Preditores do conjunto de teste.
    y_train : array-like de formato (n_treino,)
        Alvo do conjunto de treino.
    y_test : array-like de formato (n_teste,)
        Alvo do conjunto de teste.
    feature_names : list de str
        Nome de cada coluna de X, na mesma ordem das colunas.

    Returns
    -------
    path : Path
        Caminho do arquivo salvo.

    Raises
    ------
    ValueError
        Se X não for bidimensional ou y não for unidimensional.
        Se X_train e X_test tiverem números de colunas diferentes, se
        esse número for diferente de len(feature_names) ou se X e y de
        um mesmo conjunto tiverem números de linhas diferentes.
    """
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    feature_names = np.array(feature_names, dtype=str)
    if X_train.ndim != 2:
        raise ValueError("X_train deve ser bidimensional.")
    if y_train.ndim != 1:
        raise ValueError("y_train deve ser unidimensional.")
    if X_test.ndim != 2:
        raise ValueError("X_test deve ser bidimensional.")
    if y_test.ndim != 1:
        raise ValueError("y_test deve ser unidimensional.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"X_train tem {X_train.shape[1]} colunas e "
            f"X_test tem {X_test.shape[1]} colunas."
        )
    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            f"X tem {X_train.shape[1]} colunas mas "
            f"features_name tem {len(feature_names)} nomes."
        )
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train tem {X_train.shape[0]} linhas e "
            f"y_train tem {y_train.shape[0]} linhas."
        )
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"X_test tem {X_test.shape[0]} linhas e "
            f"y_test tem {y_test.shape[0]} linhas."
        )

    path = PROCESSED / f"{name}.npz"

    np.savez_compressed(
        path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )

    return path


def load_dataset(name: str) -> dict[str, NDArray]:
    """Carrega uma base salva por save_dataset.

    Parameters
    ----------
    name : str
        Nome da base, sem extensão (o mesmo usado em save_dataset).

    Returns
    -------
    data : dict
        Dicionário com as chaves "X_train", "X_test", "y_train",
        "y_test" e "features_name". feature_names é um ndarray de str.

    Raises
    ------
    FileNotFoundError
        Se data/processed/<name>.npz não existir. A mensagem diz qual
        script de preparação gera o arquivo.
    """
    path = PROCESSED / f"{name}.npz"

    script = SCRIPTS.get(name, "o script de preparação")

    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Rode antes: python scripts/{script}")

    with np.load(path) as npz:
        data = {key: npz[key] for key in npz.files}
    return data
