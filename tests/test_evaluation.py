"""Testes de src/analysis/evaluation.py"""

import numpy as np
import pytest

from analysis.evaluation import evaluate_classifier, evaluate_regressor

# Exemplos pequenos, calculados à mão
Y_TRUE_CLASS = [0, 0, 1, 1, 2, 2]
Y_PRED_CLASS = [0, 1, 1, 1, 2, 0]
Y_TRUE_REG = [1.0, 2.0, 3.0]
Y_PRED_REG = [1.0, 2.0, 4.0]


def test_regressor_hand_made_example():
    result = evaluate_regressor(Y_TRUE_REG, Y_PRED_REG)
    assert result == pytest.approx({"rmse": np.sqrt(1 / 3), "mae": 1 / 3, "r2": 1 / 2})


def test_regressor_keys_in_fixed_order():
    result = evaluate_regressor(Y_TRUE_REG, Y_PRED_REG)
    assert list(result) == ["rmse", "mae", "r2"]


def test_classifier_general_metrics():
    result = evaluate_classifier(Y_TRUE_CLASS, Y_PRED_CLASS)
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["f1_macro"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_classifier_per_class_metrics():
    result = evaluate_classifier(Y_TRUE_CLASS, Y_PRED_CLASS)
    assert result == pytest.approx(
        {
            "accuracy": 4 / 6,
            "f1_macro": (0.5 + 0.8 + 2 / 3) / 3,
            "precision_0": 1 / 2,
            "recall_0": 1 / 2,
            "f1_0": 1 / 2,
            "precision_1": 2 / 3,
            "recall_1": 1.0,
            "f1_1": 0.8,
            "precision_2": 1.0,
            "recall_2": 1 / 2,
            "f1_2": 2 / 3,
        }
    )


def test_classifier_keys_in_fixed_order():
    result = evaluate_classifier(Y_TRUE_CLASS, Y_PRED_CLASS)
    assert list(result) == [
        "accuracy",
        "f1_macro",
        "precision_0",
        "recall_0",
        "f1_0",
        "precision_1",
        "recall_1",
        "f1_1",
        "precision_2",
        "recall_2",
        "f1_2",
    ]


def test_classifier_accepts_numpy_arrays():
    result_list = evaluate_classifier(Y_TRUE_CLASS, Y_PRED_CLASS)
    result_nparray = evaluate_classifier(np.array(Y_TRUE_CLASS), np.array(Y_PRED_CLASS))
    assert result_nparray == pytest.approx(result_list)


def test_classifier_text_labels():
    y_true = ["normal", "suspeito", "normal"]
    y_pred = ["normal", "normal", "patologico"]
    result = evaluate_classifier(y_true, y_pred)
    assert "precision_normal" in result
    assert "precision_0" not in result


def test_classifier_class_only_in_predictions():
    y_true = [0, 1, 1]
    y_pred = [0, 1, 2]
    with pytest.warns(UserWarning):
        result = evaluate_classifier(y_true, y_pred)
    assert {"precision_2", "recall_2", "f1_2"} <= set(result)
    assert result["precision_2"] == 0.0
