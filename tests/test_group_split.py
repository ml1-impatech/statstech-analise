import numpy as np
import pytest

from analysis.validation import group_train_test_split


@pytest.fixture
def data():
    groups = np.array(["a", "a", "b", "b", "b", "c", "d", "d"])
    X = np.arange(8).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 2, 0, 0])
    return X, y, groups


def test_no_leakage(data):
    X, y, groups = data
    X_train, X_test, _, _ = group_train_test_split(
        X, y, groups, test_size=0.5, random_state=0
    )
    train_groups = groups[X_train[:, 0]]
    test_groups = groups[X_test[:, 0]]
    assert set(train_groups) & set(test_groups) == set()


def test_covers_all_rows(data):
    X, y, groups = data
    X_train, X_test, _, _ = group_train_test_split(
        X, y, groups, test_size=0.5
    )
    all_indices = np.sort(np.concatenate((X_train[:, 0], X_test[:, 0])))
    np.testing.assert_array_equal(all_indices, np.arange(len(X)))


def test_reproducible(data):
    X, y, groups = data
    first = group_train_test_split(X, y, groups, random_state=0)
    second = group_train_test_split(X, y, groups, random_state=0)
    np.testing.assert_array_equal(first[0], second[0])


def test_errors(data):
    X, y, groups = data
    with pytest.raises(ValueError):
        group_train_test_split(X, y[:-1], groups)
    with pytest.raises(ValueError):
        group_train_test_split(X, y, groups[:-1])
    for size in (0, 1):
        with pytest.raises(ValueError):
            group_train_test_split(X, y, groups, test_size=size)