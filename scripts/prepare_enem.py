from pathlib import Path

import pandas as pd

from analysis.config import ENEM_SAMPLE_SIZE, PROCESSED, RAW, SEED

score_columns = [
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
]

presence_columns = [
    "TP_PRESENCA_CN",
    "TP_PRESENCA_CH",
    "TP_PRESENCA_LC",
    "TP_PRESENCA_MT",
]

other_columns = [
    "TP_ST_CONCLUSAO",
    "IN_TREINEIRO",
    "TP_STATUS_REDACAO",
    "TP_ESCOLA",
    "SG_UF_PROVA",
    "TP_FAIXA_ETARIA",
    "TP_SEXO",
    "TP_COR_RACA",
]

questionnaire_columns = [f"Q{i:03d}" for i in range(1, 26)]


def read_enem(path: Path, columns: list[str], chunksize: int = 500_000) -> pd.DataFrame:
    """
    Lê e filtra os microdados do ENEM em blocos e seleciona uma amostra.

    Parameters
    ----------
    path : Path
        Caminho para o arquivo CSV dos microdados do ENEM.
    columns : list[str]
        Colunas que serão carregadas do arquivo.
    chunksize : int, default=500_000
        Número de linhas lidas por bloco.

    Returns
    -------
    pd.DataFrame
        Amostra aleatória dos participantes que passaram pelos filtros.

    Raises
    ------
    ValueError
        Se nenhum bloco for lido ou nenhum participante passar pelos filtros.

    Notes
    -----
    São mantidos os participantes que estiveram presentes nas quatro provas
    objetivas, não são treineiros, atendem ao critério de conclusão do Ensino
    Médio definido por TP_ST_CONCLUSAO == 2 e possuem notas nas cinco provas.
    O tamanho da amostra e a semente aleatória são definidos em
    analysis.config.
    """
    chunks = pd.read_csv(
        path,
        sep=";",
        encoding="latin-1",
        usecols=columns,
        chunksize=chunksize,
    )
    filtered_chunks = []

    for chunk in chunks:
        presence_filter = (chunk[presence_columns] == 1).all(axis=1)
        trainee_filter = chunk["IN_TREINEIRO"] == 0
        completion_filter = chunk["TP_ST_CONCLUSAO"] == 2
        score_filter = chunk[score_columns].notna().all(axis=1)

        filter_mask = (
            presence_filter & trainee_filter & completion_filter & score_filter
        )

        filtered_chunks.append(chunk.loc[filter_mask])

    if not filtered_chunks:
        raise ValueError("Nenhum bloco foi lido do arquivo.")

    filtered_df = pd.concat(filtered_chunks, ignore_index=True)

    if filtered_df.empty:
        raise ValueError("Nenhum participante passou pelos filtros.")

    print(f"Participantes após os filtros: {len(filtered_df)}")

    sample_df = filtered_df.sample(
        n=min(ENEM_SAMPLE_SIZE, len(filtered_df)),
        random_state=SEED,
    )

    return sample_df


if __name__ == "__main__":
    columns = score_columns + presence_columns + other_columns + questionnaire_columns

    sample_df = read_enem(
        RAW / "enem" / "MICRODADOS_ENEM_2023.csv",
        columns,
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)

    sample_df.to_csv(
        PROCESSED / "enem_amostra.csv",
        index=False,
    )

    print("Amostra salva com sucesso!")
