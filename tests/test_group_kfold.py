import numpy as np
import pytest

from analysis.validation import StratifiedGroupKFold


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), rng.integers(1, 6, size=40))
    group_label = np.array([0] * 25 + [1] * 10 + [2] * 5)
    y = group_label[groups]
    X = np.zeros((len(y), 2))
    return X, y, groups


def test_whole_group_in_same_fold(data):
    X, y, g = data
    for train_idx, val_idx in StratifiedGroupKFold(g, n_splits=5).split(X, y):
        assert set(g[train_idx]) & set(g[val_idx]) == set()


def test_each_row_validated_once(data):
    X, y, g = data
    val = np.concatenate([v for _, v in StratifiedGroupKFold(g, 5).split(X, y)])
    np.testing.assert_array_equal(np.sort(val), np.arange(len(X)))


def test_every_class_in_every_fold(data):
    X, y, g = data
    for _, val_idx in StratifiedGroupKFold(g, n_splits=5).split(X, y):
        assert set(y[val_idx]) == {0, 1, 2}


def test_reproducible(data):
    X, y, g = data
    a = [v for _, v in StratifiedGroupKFold(g, 5, random_state=1).split(X, y)]
    b = [v for _, v in StratifiedGroupKFold(g, 5, random_state=1).split(X, y)]
    for val_a, val_b in zip(a, b):
        np.testing.assert_array_equal(val_a, val_b)
