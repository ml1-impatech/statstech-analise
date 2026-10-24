"""Testes de src/analysis/io.py (C2)."""

import numpy as np
import pytest

import analysis.io
from analysis.io import load_dataset, save_dataset


@pytest.fixture(autouse=True)
def temporary_processed(tmp_path, monkeypatch):
    """Faz todos os testes gravarem numa pasta temporária."""
    monkeypatch.setattr(analysis.io, "PROCESSED", tmp_path)
    return tmp_path


def make_data():
    """Treino 5x2, teste 3x2 e nomes de duas colunas."""
    X_train = np.arange(10.0).reshape(5, 2)
    X_test = np.arange(100.0, 106.0).reshape(3, 2)
    y_train = np.arange(5.0)
    y_test = np.arange(3.0)
    return X_train, X_test, y_train, y_test, ["a", "b"]


def test_round_trip():
    """Arrays e nomes carregados são iguais aos salvos."""
    X_train, X_test, y_train, y_test, names = make_data()
    save_dataset("toy", X_train, X_test, y_train, y_test, names)
    data = load_dataset("toy")
    np.testing.assert_array_equal(data["X_train"], X_train)
    np.testing.assert_array_equal(data["X_test"], X_test)
    np.testing.assert_array_equal(data["y_train"], y_train)
    np.testing.assert_array_equal(data["y_test"], y_test)
    assert data["feature_names"].tolist() == names


def test_load_returns_plain_dict():
    """load_dataset devolve um dict comum, não um NpzFile."""
    X_train, X_test, y_train, y_test, names = make_data()
    path = save_dataset("toy", X_train, X_test, y_train, y_test, names)
    data = load_dataset("toy")
    path.unlink()
    assert type(data) is dict
    assert set(data) == {"X_train", "X_test", "y_train", "y_test", "feature_names"}


def test_feature_names_mismatch_raises():
    """Erro claro quando len(feature_names) não bate com as colunas de X."""
    X_train, X_test, y_train, y_test, _ = make_data()
    with pytest.raises(ValueError, match="feature_names"):
        save_dataset("toy", X_train, X_test, y_train, y_test, ["a"])


def test_train_test_columns_mismatch_raises():
    """Erro quando X_train e X_test têm números de colunas diferentes."""
    X_train, _, y_train, y_test, names = make_data()
    X_test = np.ones((3, 3))
    with pytest.raises(ValueError, match="X_test"):
        save_dataset("toy", X_train, X_test, y_train, y_test, names)


def test_missing_file_raises():
    """FileNotFoundError ao carregar uma base que não foi salva."""
    with pytest.raises(FileNotFoundError, match="prepare_ctg"):
        load_dataset("ctg")
