from pathlib import Path

import numpy as np
import pandas as pd
from statstech.decomposition import PCA
from statstech.model_selection import train_test_split
from statstech.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler

from analysis.config import ENEM_SAMPLE_SIZE, PROCESSED, RAW, SEED, TEST_SIZE
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
from analysis.io import save_dataset

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
    Transforma a amostra do ENEM em matriz de entradas X e alvo y.

    Parameters
    ----------
    df : pd.DataFrame
        Amostra preparada por read_enem.
    ordinal : bool, default=True
        Se True, a escolaridade dos pais e a renda entram como escala
        numérica. Se False, entram como categorias em one-hot.

    Returns
    -------
    X : np.ndarray
        Matriz de entradas, uma linha por participante.
    y : np.ndarray
        Média das cinco notas de cada participante.
    names : list[str]
        Nome de cada coluna de X.

    Notes
    -----
    Participantes com redação anulada ou com outro problema
    (TP_STATUS_REDACAO diferente de 1) são excluídos: nesses casos a nota da
    redação é 0, o que reduziria a média por um motivo que não mede o
    conhecimento do aluno.

    Com ordinal=False, Q001, Q002 e Q006 viram colunas 0/1, uma por letra
    ("não sei" é uma categoria como as outras), e saem a escolaridade em
    escala e a renda per capita. O número de pessoas na casa continua.
    """
    df = df[df["TP_STATUS_REDACAO"] == 1]
    y = df[score_columns].mean(axis=1).to_numpy()

    features: dict[str, np.ndarray] = {}

    if ordinal:
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
    features["household_size"] = household_size
    if ordinal:
        income = df["Q006"].map(INCOME_MIDPOINTS).to_numpy(dtype=float)
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
    if not ordinal:
        categorical["father_education"] = df["Q001"]
        categorical["mother_education"] = df["Q002"]
        categorical["income_range"] = df["Q006"]

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


def add_asset_index(
    X_train: np.ndarray,
    X_test: np.ndarray,
    cols: list[int],
    n_components: int = 1,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA]:
    """
    Troca as colunas de bens por um índice de bens calculado com PCA.

    Parameters
    ----------
    X_train : np.ndarray
        Entradas de treino, sem NaN.
    X_test : np.ndarray
        Entradas de teste, sem NaN.
    cols : list[int]
        Posições das colunas de bens em X_train e X_test.
    n_components : int, default=1
        Número de componentes do índice.

    Returns
    -------
    X_train : np.ndarray
        Treino sem as colunas de bens e com as componentes no fim.
    X_test : np.ndarray
        Teste sem as colunas de bens e com as componentes no fim.
    scaler : StandardScaler
        Padronizador ajustado nas colunas de bens do treino.
    pca : PCA
        PCA ajustado nas colunas de bens padronizadas do treino.

    Notes
    -----
    O padronizador e o PCA aprendem só com o treino; o teste apenas é
    transformado. Assim o teste não influencia o índice.
    """
    scaler = StandardScaler().fit(X_train[:, cols])
    pca = PCA(n_components=n_components).fit(scaler.transform(X_train[:, cols]))

    keep = [i for i in range(X_train.shape[1]) if i not in cols]
    train_index = pca.transform(scaler.transform(X_train[:, cols]))
    test_index = pca.transform(scaler.transform(X_test[:, cols]))

    X_train_new = np.column_stack([X_train[:, keep], train_index])
    X_test_new = np.column_stack([X_test[:, keep], test_index])
    return X_train_new, X_test_new, scaler, pca


def split_and_prepare(
    df: pd.DataFrame, ordinal: bool = True, n_components: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], PCA]:
    """
    Codifica, separa treino e teste e cria o índice de bens.

    Parameters
    ----------
    df : pd.DataFrame
        Amostra preparada por read_enem.
    ordinal : bool, default=True
        Versão da codificação (ver encode_enem).
    n_components : int, default=1
        Número de componentes do índice de bens.

    Returns
    -------
    X_train, X_test : np.ndarray
        Entradas de treino e de teste, sem NaN.
    y_train, y_test : np.ndarray
        Alvo de treino e de teste.
    names : list[str]
        Nome de cada coluna de X_train e X_test.
    pca : PCA
        PCA do índice de bens, ajustado no treino.

    Notes
    -----
    O teste é separado antes de imputar e de ajustar o PCA. A imputação (mediana)
    e o índice de bens aprendem só com o treino, para o teste não vazar.
    """
    X, y, names = encode_enem(df, ordinal=ordinal)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    imputer = SimpleImputer(strategy="median").fit(X_train)
    X_train, X_test = imputer.transform(X_train), imputer.transform(X_test)

    asset_cols = [names.index(name) for name in ASSET_COLUMNS.values()]
    X_train, X_test, _, pca = add_asset_index(X_train, X_test, asset_cols, n_components)

    kept = [name for i, name in enumerate(names) if i not in asset_cols]
    new_names = kept + [f"asset_index_{k + 1}" for k in range(n_components)]
    return X_train, X_test, y_train, y_test, new_names, pca


if __name__ == "__main__":
    sample_path = PROCESSED / "enem_amostra.csv"

    if not sample_path.exists():
        columns = (
            score_columns + presence_columns + other_columns + questionnaire_columns
        )
        sample_df = read_enem(RAW / "enem" / "MICRODADOS_ENEM_2023.csv", columns)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(sample_path, index=False)
        print("Amostra salva com sucesso!")

    sample_df = pd.read_csv(sample_path)

    for name, ordinal in [("enem", True), ("enem_onehot", False)]:
        X_train, X_test, y_train, y_test, names, _ = split_and_prepare(
            sample_df, ordinal=ordinal
        )
        path = save_dataset(name, X_train, X_test, y_train, y_test, names)
        print(f"{name}: treino {X_train.shape}, teste {X_test.shape} -> {path}")
