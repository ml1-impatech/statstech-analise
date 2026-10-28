import numpy as np
import pytest
 
from analysis.threshold import (
    choose_threshold,
    predict_with_threshold,
    threshold_curve,)

# Exemplo pequeno, com contas feitas à mão. Classe positiva: 2.
CLASSES = [0, 1, 2]
Y = np.array([0, 0, 1, 2, 2, 2])
PROBA = np.array(
    [
        [0.80, 0.10, 0.10],  # mais provável: 0
        [0.50, 0.20, 0.30],  # mais provável: 0, p(2) = 0.3
        [0.20, 0.60, 0.20],  # mais provável: 1
        [0.10, 0.10, 0.80],  # mais provável: 2
        [0.45, 0.15, 0.40],  # mais provável: 0, p(2) = 0.4
        [0.60, 0.20, 0.20],  # mais provável: 0, p(2) = 0.2
    ]
)
 
# testes de predict_with_threshold
 
def test_predict_with_threshold_hand_example():
    pred = predict_with_threshold(PROBA, CLASSES, 2, 0.3)
    np.testing.assert_array_equal(pred, [0, 2, 1, 2, 2, 0])
 
def test_threshold_only_adds_positive():
    pred = predict_with_threshold([[0.30, 0.25, 0.45]], CLASSES, 2, 0.6)
    np.testing.assert_array_equal(pred, [2])

def test_predict_with_threshold_string_labels():
    pred = predict_with_threshold(PROBA, ["n", "s", "p"], "p", 0.3)
    np.testing.assert_array_equal(pred, ["n", "p", "s", "p", "p", "n"])
 
def test_positive_not_in_classes():
    with pytest.raises(ValueError, match="positive"):
        predict_with_threshold(PROBA, CLASSES, 5, 0.5)

# testes de threshold_curve 
 
@pytest.mark.parametrize(
    "t, recall, precision",
    [
        (0.9, 1 / 3, 1.0),  # só a linha 3 (argmax)
        (0.4, 2 / 3, 1.0),  # linhas 3 e 4
        (0.3, 2 / 3, 2 / 3),  # linhas 1, 3 e 4
        (0.2, 1.0, 3 / 5),  # linhas 1, 2, 3, 4 e 5
    ],
)
def test_threshold_curve_hand_example(t, recall, precision):
    r, p = threshold_curve(PROBA, Y, 2, [t])
    np.testing.assert_allclose(r, [recall])
    np.testing.assert_allclose(p, [precision])
 
def test_precision_nan_when_nothing_predicted_positive():
    proba = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.6, 0.1, 0.3]])
    r, p = threshold_curve(proba, [0, 1, 2], 2, [0.5])
    np.testing.assert_allclose(r, [0.0])
    assert np.isnan(p[0])