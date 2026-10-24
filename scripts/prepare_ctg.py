def load_ctg(path):
    """Lê CTG.xls; devolve X, y, names, groups."""
    pass

if __name__ == "__main__":
    pass

#2.126 linhas lidas, 12 duplicatas nas medidas removidas, 2.114 finais, todas as duplicatas dentro do mesmo registro e com o mesmo NSP

import numpy, pandas, Path
from analysis.config import SEED, TEST_SIZE, RAW de analysis.config
from analysis.io import save_dataset # C2 (só quando existir)

COLUNAS_VAZAMENTO = [CLASS, A, B, C, D, E, AD, DE, LD, FS, SUSP, LBE, DR]
IDENTIFICADORES   = [FileName, Date, SegFile, b, e]
CONTAGENS         = [AC, FM, UC, DL, DS, DP]

def load_ctg(path):
    df = ler aba "Raw Data" de path
    df = tirar linhas onde NSP é vazio                  # esperado: 2126

    medidas = df sem as colunas IDENTIFICADORES
    imprimir quantas linhas duplicadas existem em medidas
    df = manter só as linhas não duplicadas              # esperado: 2114

    y = NSP - 1 (inteiros 0, 1, 2)
    groups = FileName
    duracao = e - b
    for coluna in CONTAGENS:
        df[coluna] = df[coluna] / duracao

    if Width == Max - Min em todas as linhas:
        adicionar Width à lista de remoção
    remover de df: NSP, IDENTIFICADORES, COLUNAS_VAZAMENTO, Width

    names = lista com os nomes das colunas que sobraram
    X = df como array numpy float
    devolver X, y, names, groups

if rodando como script:
    X, y, names, groups = load_ctg(RAW / "CTG.xls")
    conferir: X sem NaN, 3 classes em y, nenhuma coluna de vazamento em names
    (T2) separar treino e teste por grupo
    imprimir formato e contagem por classe em treino e teste
    save_dataset("ctg", X_train, X_test, y_train, y_test, names)