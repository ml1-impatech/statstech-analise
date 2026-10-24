import zipfile
import io
import hashlib
import urllib.request
from pathlib import Path

from analysis.config import RAW

URL = (
    "https://download.inep.gov.br/microdados/"
    "microdados_enem_2023.zip"
)

ZIP_PATH = RAW / "microdados_enem_2023.zip"
DEST = RAW / "enem"
CSV_INTERNO = (
    "microdados_enem_2023/DADOS/MICRODADOS_ENEM_2023.csv"
)
CHUNK_SIZE = 1024 * 1024 

def download(url: str, dest: Path, chunk_size: int = CHUNK_SIZE) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"Arquivo já existe: {dest}")
        return dest
    
    temporario = dest.with_suffix(dest.suffix + ".part")

    try:
        with urllib.request.urlopen(url, timeout=60) as resposta:
            with temporario.open("wb") as arquivo:
                while True:
                    bloco = resposta.read(chunk_size)
                    if not bloco:
                        break
                    arquivo.write(bloco)

        
        temporario.replace(dest)

    finally:
        if temporario.exists():
            temporario.unlink()

    print(f"Download concluído: {dest}")
    return dest


def sha256(path: Path, chunk: int = CHUNK_SIZE) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as arquivo:
        while True:
            bloco = arquivo.read(chunk)
            if not bloco:
                break
            digest.update(bloco)

    return digest.hexdigest()

def check_csv_columns(zip_file: zipfile.ZipFile, filename: str) -> None:
    with zip_file.open(filename) as file:
        text_file = io.TextIOWrapper(file, encoding="latin-1")
        header = text_file.readline().rstrip("\r\n")

    columns = {column.strip() for column in header.split(";")}
    required_columns = {"NU_NOTA_MT", "Q006"}
    missing_columns = required_columns - columns

    if missing_columns:
        raise ValueError(
            f"The CSV file {filename} is missing columns: "
            f"{sorted(missing_columns)}"
        )