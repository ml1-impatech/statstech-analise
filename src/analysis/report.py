"""Tabelas e figuras do relatório, salvas em results/ ."""

import csv
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

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
    Um arquivo com o mesmo nome é sobrescrito.
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
    (``bbox_inches="tight"``). Um arquivo com o mesmo nome é
    sobrescrito.
    """
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path