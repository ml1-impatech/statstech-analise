"""Configuração única da análise: semente, tamanhos e caminhos.

Todo script e notebook importa daqui (``from analysis.config import SEED``).
Ninguém escreve a semente ou um caminho solto no código.
"""

from pathlib import Path

# Semente de todos os sorteios (divisão, folds, bootstrap, redes).
SEED = 42

# Fração dos dados reservada para o teste final.
TEST_SIZE = 0.2

# Tamanho da subamostra de participantes do ENEM.
ENEM_SAMPLE_SIZE = 50_000

# Raiz do repositório: este arquivo fica em src/analysis/config.py.
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
STYLE = RESULTS / "style.mplstyle"

for _folder in (RAW, PROCESSED, FIGURES, TABLES):
    _folder.mkdir(parents=True, exist_ok=True)
