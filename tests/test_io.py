"""Testes de src/analysis/io.py (C2)."""

import pytest

pytestmark = pytest.mark.skip(reason="C2: a implementar")


def test_round_trip():
    """Arrays e nomes carregados são iguais aos salvos."""


def test_load_returns_plain_dict():
    """load_dataset devolve um dict comum, não um NpzFile."""


def test_feature_names_mismatch_raises():
    """Erro claro quando len(feature_names) não bate com as colunas de X."""


def test_train_test_columns_mismatch_raises():
    """Erro quando X_train e X_test têm números de colunas diferentes."""


def test_missing_file_raises():
    """FileNotFoundError ao carregar uma base que não foi salva."""
