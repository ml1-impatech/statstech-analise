from pathlib import Path


import pandas as pd

def read_enem(
    path: Path,
    columns: list[str],
    chunksize: int = 500_000) -> pd.DataFrame:
    chunks = pd.read_csv(
        path,
        sep=";",
        encoding="latin-1",
        usecols=columns,
        chunksize=chunksize,
    )
    filtered_chunks = []

    for chunk in chunks:
        presence_columns = [
            "TP_PRESENCA_CN",
            "TP_PRESENCA_CH",
            "TP_PRESENCA_LC",
            "TP_PRESENCA_MT",
        ]

        presence_filter = (chunk[presence_columns] == 1).all(axis=1)
        trainee_filter = chunk["IN_TREINEIRO"] == 0
        completion_filter = chunk["TP_ST_CONCLUSAO"] == 2
        
        score_columns = [
            "NU_NOTA_CN",
            "NU_NOTA_CH",
            "NU_NOTA_LC",
            "NU_NOTA_MT",
            "NU_NOTA_REDACAO",
        ]

        score_filter = chunk[score_columns].notna().all(axis=1)
        filter_mask = (
            presence_filter
            & trainee_filter
            & completion_filter
            & score_filter
        )

    filtered_df = pd.concat(filtered_chunks, ignore_index=True)

    if filtered_df.empty:
        raise ValueError("Nenhum participante passou pelos filtros.")

    print(f"Participantes após os filtros: {len(filtered_df)}")

    sample_df = filtered_df.sample(
        n=min(ENEM_SAMPLE_SIZE, len(filtered_df)),
        random_state=SEED,
    )

    return sample_df

