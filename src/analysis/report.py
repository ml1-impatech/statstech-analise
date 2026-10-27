"""Tabelas e figuras do relatório, salvas em results/ ."""

import csv
from pathlib import Path
from typing import Any
from numbers import Real

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from statstech.model_selection import GridSearchCV

from analysis.config import FIGURES, TABLES


def save_table(rows: list[dict[str, Any]], name: str) -> Path:
    """Salva uma tabela em CSV na pasta de tabelas.

    Cada dicionário de ``rows`` vira uma linha. O cabeçalho vem das
    chaves do primeiro dicionário, na mesma ordem.

    Parameters
    ----------
    rows : list of dict
        Linhas da tabela. Todos os dicionários devem ter as mesmas
        chaves.
    name : str
        Nome do arquivo, sem extensão. Use o prefixo da base, por
        exemplo ``"ctg_modelos"``.

    Returns
    -------
    path : Path
        Caminho do arquivo salvo, ``TABLES / f"{name}.csv"``.

    Raises
    ------
    ValueError
        Se ``rows`` estiver vazia.

    Notes
    -----
    Um arquivo com o mesmo name é sobrescrito.
    """
    if not rows:
        raise ValueError("rows não pode ser uma lista vazia.")
    path = TABLES / f"{name}.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return path


def save_figure(fig: Figure, name: str) -> Path:
    """Salva uma figura em PNG, com 200 dpi, na pasta de figuras.

    Parameters
    ----------
    fig : Figure
        Figura do matplotlib já desenhada.
    name : str
        Nome do arquivo, sem extensão. Use o prefixo da base, por
        exemplo ``"enem_knn_k"``.

    Returns
    -------
    path : Path
        Caminho do arquivo salvo, ``FIGURES / f"{name}.png"``.

    Notes
    -----
    A margem branca em volta da figura é cortada
    (``bbox_inches="tight"``). Um arquivo com o mesmo name é
    sobrescrito.
    """
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def plot_search(
    search: GridSearchCV,
    param_name: str,
    ax: Axes | None = None,
    log_x: bool = False,
) -> Axes:
    """Desenha a nota da validação cruzada contra um hiperparâmetro.

    Lê ``search.cv_results_`` e marca cada combinação com a nota média
    dos folds e uma barra de erro de um desvio padrão. A melhor
    combinação (``search.best_index_``) é destacada.

    Se a grade tem mais de um hiperparâmetro, os outros ficam fixos no
    valor da melhor combinação: só entram no gráfico as linhas que
    diferem dela apenas em ``param_name``.

    Parameters
    ----------
    search : GridSearchCV
        Busca já ajustada.
    param_name : str
        Nome do hiperparâmetro no eixo x, como aparece na grade, por
        exemplo ``"model__n_neighbors"``.
    ax : Axes or None, default=None
        Eixos onde desenhar. Se None, cria uma figura nova.
    log_x : bool, default=False
        Se True, usa escala logarítmica no eixo x. Útil para grades
        como ``np.logspace(-2, 4, 13)``.

    Returns
    -------
    ax : Axes
        Eixos com o gráfico, para ajustar rótulos ou salvar com
        ``save_figure(ax.figure, ...)``.

    Raises
    ------
    ValueError
        Se ``log_x=True`` e algum valor de ``param_name`` não for um
        número positivo.

    Notes
    -----
    Valores que não são números, como ``hidden_layer_sizes=(64, 32)``,
    são desenhados em posições igualmente espaçadas, com o valor
    escrito no eixo x.
    """
    best = search.cv_results_[search.best_index_]
    best_params = best["params"]

    xs, means, stds = [], [], []
    for result in search.cv_results_:
        params = result["params"]

        # Os outros hiperparâmetros precisam ser iguais aos do melhor.
        keep = True
        for name in best_params:
            if name == param_name:
                continue
            if params[name] != best_params[name]:
                keep = False
        if not keep:
            continue

        xs.append(params[param_name])
        means.append(result["mean_score"])
        stds.append(result["std_score"])

    numeric = all(isinstance(x, Real) for x in xs)

    # Conferir a entrada antes de criar a figura.
    if log_x and (not numeric or min(xs) <= 0):
        raise ValueError(
            "A escala log só mostra números positivos: os valores de "
            f"{param_name} são {xs}."
        )

    if ax is None:
        _, ax = plt.subplots()

    if numeric:
        positions = xs
        best_x = best_params[param_name]
    else:
        # Tuplas ou textos: uma posição para cada valor, na ordem da grade.
        positions = list(range(len(xs)))
        best_x = xs.index(best_params[param_name])
        ax.set_xticks(positions)
        ax.set_xticklabels([str(x) for x in xs])

    ax.errorbar(positions, means, yerr=stds, marker="o", capsize=4)
    ax.plot(
        best_x,
        best["mean_score"],
        marker="*",
        markersize=15,
        color="C1",
        linestyle="none",
        label="melhor",
    )

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(param_name.split("__")[-1])
    ax.set_ylabel("nota média (validação cruzada)")
    ax.legend()
    return ax
