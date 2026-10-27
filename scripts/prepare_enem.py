from pathlib import Path

import numpy as np
import pandas as pd
from statstech.preprocessing import OneHotEncoder

from analysis.config import ENEM_SAMPLE_SIZE, PROCESSED, RAW, SEED
from analysis.enem_codes import (
    ASSET_COLUMNS,
    ASSET_LEVELS,
    EDUCATION_LEVELS,
    EDUCATION_UNKNOWN,
    INCOME_MIDPOINTS,
    OCCUPATION_GROUPS,
    OCCUPATION_UNKNOWN,
    REGION_BY_UF,
)

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


def letter_scale(letters: pd.Series, mapping: dict[str, int]) -> np.ndarray:
    """
    Troca cada letra pelo número do mapa.

    Parameters
    ----------
    letters: pd.Series
      Coluna do questionário, com as letras das respostas.
    mapping: dict[str, int]
      Mpa de letra para número (ver analysis.enem_codes)

    Returns
    -------
    np.ndarray
      Um numero por participante. Letras que nao estao no mapa (como 'nao sei') viram NaN
    """
    return letters.map(mapping).to_numpy(dtype=float)


def encode_enem(
    df: pd.DataFrame, ordinal: bool = True
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Transforma a maostra do enem em matriz de entradas X e akvo y

    Parameters
    ----------
    df: pd.DataFrame
      Amostra preparada por read_enem.
    ordinal: bool, default=True
      se True, escolaridade e renda entram como escala numerica

    Returns
    -------
    X: np.ndarray
      Matriz de entradas, uma linha por participante
    y: np.ndarray
      Media das cinco notas de cada participante
    names: list[str]
      Nome de cada coluna de X

    Notes
    -----
    Participantes com redacao anulada ou com outro problema (TP_STATUS_REDACAO diferente de 1) sao excluidos: nesses casos a nota da redacao é 0, o que reduziria a media por um motivo que nao ede o conhecimento do aluno.
    """
    df = df[df["TP_STATUS_REDACAO"] == 1]
    y = df[score_columns].mean(axis=1).to_numpy()

    features: dict[str, np.ndarray] = {}

    education = [("Q001", "father_education"), ("Q002", "mother_education")]
    for column, name in education:
        features[name] = letter_scale(df[column], EDUCATION_LEVELS)
        features[f"{name}_unknown"] = (df[column] == EDUCATION_UNKNOWN).to_numpy(
            dtype=float
        )

    occupation = [("Q003", "father_occupation"), ("Q004", "mother_occupation")]
    for column, name in occupation:
        features[name] = letter_scale(df[column], OCCUPATION_GROUPS)
        features[f"{name}_unknown"] = (df[column] == OCCUPATION_UNKNOWN).to_numpy(
            dtype=float
        )

    household_size = df["Q005"].to_numpy(dtype=float)
    income = df["Q006"].map(INCOME_MIDPOINTS).to_numpy(dtype=float)
    features["household_size"] = household_size
    features["log_income_per_capita"] = np.log1p(income / household_size)

    for column, name in ASSET_COLUMNS.items():
        features[name] = letter_scale(df[column], ASSET_LEVELS)

    categorical = {
        "school": df["TP_ESCOLA"],
        "region": df["SG_UF_PROVA"].map(REGION_BY_UF),
        "sex": df["TP_SEXO"],
        "race": df["TP_COR_RACA"],
        "age_group": df["TP_FAIXA_ETARIA"],
    }
    encoder = OneHotEncoder(drop="first")
    encoded = encoder.fit_transform(
        np.column_stack(
            [values.to_numpy().astype(str) for values in categorical.values()]
        )
    )
    column = 0
    for j, variable in enumerate(categorical):
        for category in encoder.categories_[j][1:]:
            features[f"{variable}_{category}"] = encoded[:, column]
            column += 1

    X = np.column_stack(list(features.values()))
    return X, y, list(features)


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
