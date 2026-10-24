import numpy as np, pandas as pd
from pathlib import Path
from analysis.config import SEED, TEST_SIZE, RAW
from analysis.io import save_dataset

COLUNAS_VAZAMENTO = ["CLASS", "A", "B", "C", "D", "E", "AD", "DE", "LD", "FS", "SUSP", "LBE", "DR"]
IDENTIFICADORES   = ["FileName", "Date", "SegFile", "b", "e"]
CONTAGENS         = ["AC", "FM", "UC", "DL", "DS", "DP"]

def load_ctg(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Lê a planilha da cardiotocografia (UCI) e prepara X, y e os grupos.

    Lê a aba "Raw Data", descarta as linhas sem NSP e as duplicatas
    (comparadas só nas medidas, sem os identificadores do registro),
    converte as contagens em taxa por segundo e remove as colunas que
    entregam a classe ou que não são medidas.

    Parameters
    ----------
    path : Path
        Caminho do arquivo CTG.xls.

    Returns
    -------
    X : ndarray of shape (n_amostras, n_features)
        Medidas do exame. Contagens (AC, FM, UC, DL, DS, DP) divididas
        pela duração do segmento, e - b, em segundos.
    y : ndarray of shape (n_amostras,)
        Classe dada pelos obstetras: 0 normal, 1 suspeito, 2 patológico
        (NSP - 1).
    names : list of str
        Nome de cada coluna de X, na mesma ordem.
    groups : ndarray of shape (n_amostras,)
        FileName do registro de origem. Vários segmentos vêm do mesmo
        registro.

    Notes
    -----
    Das 2126 linhas lidas, 12 duplicatas nas medidas são removidas,
    todas dentro do mesmo registro e com o mesmo NSP, restando 2114.
    Width é removida porque Width == Max - Min em todas as linhas, o que
    deixaria a covariância do LDA singular. As colunas de outra
    classificação dos especialistas (CLASS, A a SUSP e outras) são
    removidas por vazamento.

    References
    ----------
    UCI Machine Learning Repository, Cardiotocography.
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
    assert X.shape == (2114, 20)
    assert not np.isnan(X).any()
    assert set(np.unique(y)) == {0, 1, 2}
    assert not any(name in COLUNAS_VAZAMENTO for name in names)
    print("exames por classe (normal, suspeito, patológico):", np.bincount(y))
    print("registros distintos:", len(np.unique(groups)))
    
    #save_dataset("ctg", X_train, X_test, y_train, y_test, names)