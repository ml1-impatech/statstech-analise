import numpy as np
import pytest

from analysis.validation import group_train_test_split


@pytest.fixture
def dados():
    g = np.array(["a", "a", "b", "b", "b", "c", "d", "d"])
    X = np.arange(8).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 2, 0, 0])
    return X, y, g

def test_sem_vazamento(dados):
    X, y, g = dados
    X_train, X_test, _, _ = group_train_test_split(
        X, y, g, test_size=0.5, random_state=0
    )
    grupos_treino = g[X_train[:, 0]]
    grupos_teste = g[X_test[:, 0]]
    assert set(grupos_treino) & set(grupos_teste) == set()


def test_cobre_tudo(dados):
    X, y, g = dados
    X_train, X_test, _, _ = group_train_test_split(X, y, g, test_size=0.5)
    todos = np.sort(np.concatenate((X_train[:, 0], X_test[:, 0])))
    np.testing.assert_array_equal(todos, np.arange(len(X)))


def test_reprodutivel(dados):
    X, y, g = dados
    a = group_train_test_split(X, y, g, random_state=0)
    b = group_train_test_split(X, y, g, random_state=0)
    np.testing.assert_array_equal(a[0], b[0])

def test_erros(dados):
    X, y, g = dados
    with pytest.raises(ValueError):
        group_train_test_split(X, y[:-1], g)
    with pytest.raises(ValueError):
        group_train_test_split(X, y, g[:-1])
    for tamanho in (0, 1):
        with pytest.raises(ValueError):
            group_train_test_split(X, y, g, test_size=tamanho)