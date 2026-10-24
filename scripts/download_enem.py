import zipfile
import io
import hashlib
import urllib.request
from pathlib import Path
import tempfile

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

def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"Arquivo já existe: {dest}")
        return dest

    temporario = dest.with_suffix(dest.suffix + ".part")

    try:
        with urllib.request.urlopen(url, timeout=60) as resposta:
            with temporario.open("wb") as arquivo:
                while True:
                    bloco = resposta.read(chunk)
                    if not bloco:
                        break
                    arquivo.write(bloco)


        temporario.replace(dest)

    finally:
        if temporario.exists():
            temporario.unlink()

    print(f"Download concluído: {dest}")
    return dest

def sha256(path: Path, chunk: int = 1 << 20) -> str:
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
        with io.TextIOWrapper(file, encoding="latin-1") as text_file:
            header = text_file.readline().rstrip("\r\n")

    columns = {column.strip() for column in header.split(";")}
    required_columns = {"NU_NOTA_MT", "Q006"}
    missing_columns = required_columns - columns

    if missing_columns:
        raise ValueError(
            f"O arquivo csv {filename} não contém as colunas: "
            f"{sorted(missing_columns)}"
        )
    print(f"Cabeçalho validado: {filename}")

def extract_csv(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)

    dictionary_internal = (
        "microdados_enem_2023/"
        "DICIONÁRIO/Dicionário_Microdados_Enem_2023.xlsx"
    )
    files = (CSV_INTERNO, dictionary_internal)

    with zipfile.ZipFile(zip_path) as zip_file:
        filenames = zip_file.namelist()

        for filename in files:
            if filename not in filenames:
                raise FileNotFoundError(
                    f"Arquivo não encontrado no ZIP: {filename}"
                )

        check_csv_columns(zip_file, CSV_INTERNO)

        for filename in files:
            dest_file = dest / Path(filename).name

            if dest_file.exists():
                if filename != CSV_INTERNO:
                    print(f"Arquivo já extraído: {dest_file}")
                    continue

                with dest_file.open(
                    "r", encoding="latin-1"
                ) as file:
                    header = file.readline().rstrip("\r\n")

                columns = {column.strip() for column in header.split(";")}

                if {"NU_NOTA_MT", "Q006"}.issubset(columns):
                    print(f"Arquivo já extraído e válido: {dest_file}")
                    continue

                print("Cabeçalho inválido. Extraindo novamente...")

            with tempfile.TemporaryDirectory(dir=dest) as temp_dir:
                zip_file.extract(filename, temp_dir)

                extracted_file = Path(temp_dir) / filename
                extracted_file.replace(dest_file)

            print(f"Arquivo extraído: {dest_file}")

    return dest / Path(CSV_INTERNO).name

def main() -> None:
    zip_path = download(URL, ZIP_PATH)
    print(f"SHA-256: {sha256(zip_path)}")

    csv_path = extract_csv(zip_path, DEST)
    print(f"CSV disponível em: {csv_path}")


if __name__ == "__main__":
    main()
