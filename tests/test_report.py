"""Testes de src/analysis/report.py"""

import csv
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")  # desenhar sem abrir a janela
import matplotlib.pyplot as plt
import pytest

import analysis.report as report
from analysis.report import plot_search, save_figure, save_table


def make_search(alphas, means, best_index):
    """Busca falsa com um hiperparâmetro (model__alpha) e desvio 0.01."""
    results = [
        {"params": {"model__alpha": a}, "mean_score": m, "std_score": 0.01}
        for a, m in zip(alphas, means)
    ]
    return SimpleNamespace(cv_results_=results, best_index_=best_index)


def make_net_search():
    """Busca falsa com dois hiperparâmetros; a melhor é a última linha."""
    return SimpleNamespace(
        cv_results_=[
            {
                "params": {
                    "model__hidden_layer_sizes": (32,),
                    "model__learning_rate": 0.001,
                },
                "mean_score": 72.0,
                "std_score": 1.0,
            },
            {
                "params": {
                    "model__hidden_layer_sizes": (32,),
                    "model__learning_rate": 0.01,
                },
                "mean_score": 70.1,
                "std_score": 1.0,
            },
            {
                "params": {
                    "model__hidden_layer_sizes": (64, 32),
                    "model__learning_rate": 0.001,
                },
                "mean_score": 71.5,
                "std_score": 1.0,
            },
            {
                "params": {
                    "model__hidden_layer_sizes": (64, 32),
                    "model__learning_rate": 0.01,
                },
                "mean_score": 69.4,
                "std_score": 1.0,
            },
        ],
        best_index_=3,
    )


def test_save_table_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "TABLES", tmp_path)
    rows = [{"modelo": "Ridge", "rmse": 70.1}, {"modelo": "Lasso", "rmse": 70.4}]
    path = save_table(rows, "teste")
    assert path == tmp_path / "teste.csv"
    with open(path, encoding="utf-8") as f:
        read = list(csv.DictReader(f))
    assert read == [
        {"modelo": "Ridge", "rmse": "70.1"},
        {"modelo": "Lasso", "rmse": "70.4"},
    ]


def test_save_table_header_order(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "TABLES", tmp_path)
    rows = [{"modelo": "Ridge", "rmse": 70.1, "mae": 55.0, "r2": 0.31}]
    path = save_table(rows, "teste")
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        assert header == "modelo,rmse,mae,r2\n"


def test_save_table_empty_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "TABLES", tmp_path)
    with pytest.raises(ValueError):
        save_table([], "vazia")
    assert not (tmp_path / "vazia.csv").exists()


def test_save_figure_creates_png(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "FIGURES", tmp_path)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    path = save_figure(fig, "teste")
    assert path == tmp_path / "teste.png"
    assert path.exists()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    plt.close(fig)


def test_plot_search_single_param():
    search = make_search([0.1, 1, 10], [0.80, 0.85, 0.83], best_index=1)
    ax = plot_search(search, "model__alpha")
    assert list(ax.lines[0].get_xdata()) == [0.1, 1, 10]
    assert list(ax.lines[0].get_ydata()) == [0.80, 0.85, 0.83]
    assert list(ax.lines[-1].get_xdata()) == [1]
    assert list(ax.lines[-1].get_ydata()) == [0.85]
    assert ax.get_xlabel() == "alpha"
    plt.close(ax.figure)


def test_plot_search_lower_is_better():
    search = make_search([1, 10, 100], [72.0, 69.8, 70.5], best_index=1)
    ax = plot_search(search, "model__alpha")
    assert list(ax.lines[-1].get_xdata()) == [10]
    assert list(ax.lines[-1].get_ydata()) == [69.8]
    plt.close(ax.figure)


def test_plot_search_fixes_other_params():
    search = make_net_search()
    ax = plot_search(search, "model__learning_rate")
    assert list(ax.lines[0].get_xdata()) == [0.001, 0.01]
    assert list(ax.lines[0].get_ydata()) == [71.5, 69.4]
    assert list(ax.lines[-1].get_xdata()) == [0.01]
    assert list(ax.lines[-1].get_ydata()) == [69.4]
    plt.close(ax.figure)


def test_plot_search_non_numeric_values():
    search = make_net_search()
    ax = plot_search(search, "model__hidden_layer_sizes")
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert list(ax.lines[0].get_xdata()) == [0, 1]
    list(ax.lines[0].get_ydata()) == [70.1, 69.4]
    assert labels == ["(32,)", "(64, 32)"]
    assert list(ax.lines[-1].get_xdata()) == [1]
    assert list(ax.lines[-1].get_ydata()) == [69.4]
    plt.close(ax.figure)


def test_plot_search_log_scale():
    search = make_search([0.1, 1, 10], [0.80, 0.85, 0.83], best_index=1)
    ax = plot_search(search, "model__alpha", log_x=True)
    assert ax.get_xscale() == "log"
    plt.close(ax.figure)
    ax = plot_search(search, "model__alpha")
    assert ax.get_xscale() == "linear"
    plt.close(ax.figure)


def test_plot_search_log_with_zero_raises():
    search = make_search([0, 0.1, 1], [0.80, 0.85, 0.83], best_index=1)
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="positivos"):
        plot_search(search, "model__alpha", log_x=True)
    assert plt.get_fignums() == before
