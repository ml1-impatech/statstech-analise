if __name__ == "__main__":
    pass

#2.126 linhas lidas, 12 duplicatas nas medidas removidas, 2.114 finais, todas as duplicatas dentro do mesmo registro e com o mesmo NSP

import numpy as np, pandas as pd
from pathlib import Path
#from analysis.config import SEED, TEST_SIZE, RAW
#from analysis.io import save_dataset # C2 (só quando existir)

COLUNAS_VAZAMENTO = ["CLASS", "A", "B", "C", "D", "E", "AD", "DE", "LD", "FS", "SUSP", "LBE", "DR"]
IDENTIFICADORES   = ["FileName", "Date", "SegFile", "b", "e"]
CONTAGENS         = ["AC", "FM", "UC", "DL", "DS", "DP"]

def load_ctg(path):
    """Lê CTG.xls; devolve X, y, names, groups.

    2126 linhas lidas, 12 duplicatas nas medidas removidas, 2114 finais.
    """
    df = pd.read_excel(path, sheet_name="Raw Data")
    df = df.dropna(subset=["NSP"])

    medidas = df.drop(columns=IDENTIFICADORES)
    duplicadas = medidas.duplicated()
    print(f"duplicatas nas medidas: {duplicadas.sum()}")
    df = df.loc[~duplicadas].copy()

    y = (df["NSP"] - 1).to_numpy().astype(int)
    groups = df["FileName"].to_numpy()
    duracao = df["e"] - df["b"]
    for coluna in CONTAGENS:
        df[coluna] = df[coluna] / duracao

    remover = COLUNAS_VAZAMENTO + IDENTIFICADORES + ["NSP"]
    if (df["Width"] == df["Max"] - df["Min"]).all():
        remover.append("Width")
    df = df.drop(columns=remover)

    names = df.columns.tolist()
    X = df.to_numpy(dtype=float)
    return X, y, names, groups

if __name__ == "__main__":
    X, y, names, groups = load_ctg(RAW / "CTG.xls")
    conferir: X sem NaN, 3 classes em y, nenhuma coluna de vazamento em names
    (T2) separar treino e teste por grupo
    imprimir formato e contagem por classe em treino e teste
    save_dataset("ctg", X_train, X_test, y_train, y_test, names)