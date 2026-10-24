"""
Baixa e prepara os microdados do ENEM 2023.

O script:
1. Baixa o arquivo ZIP do INEP, caso ele ainda não exista localmente.
2. Calcula e exibe o SHA-256 do arquivo.
3. Verifica se o CSV contém as colunas NU_NOTA_MT e Q006.
4. Extrai o CSV de microdados e o dicionário para data/raw/enem/.

Os microdados são armazenados localmente e não devem ser versionados no Git.
"""

import hashlib
import io
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from analysis.config import RAW

URL = "https://download.inep.gov.br/microdados/microdados_enem_2023.zip"

ZIP_PATH = RAW / "microdados_enem_2023.zip"
DEST = RAW / "enem"
CSV_INTERNO = "microdados_enem_2023/DADOS/MICRODADOS_ENEM_2023.csv"


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """
    Baixa um arquivo em blocos, caso ele ainda não exista no destino.

    Parameters
    ----------
    url : str
        Endereço do arquivo a ser baixado.
    dest : Path
        Caminho onde o arquivo será salvo.
    chunk : int, default=1048576
        Quantidade de bytes lidos por bloco.

    Returns
    -------
    Path
        Caminho do arquivo baixado ou já existente.
    """

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"Arquivo já existe: {dest}")
        return dest

    temporario = dest.with_suffix(dest.suffix + ".part")

    try:
        with(
            urllib.request.urlopen(url, timeout=60) as resposta,
            temporario.open("wb") as arquivo,
        ):
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
    """
    Calcula o hash SHA-256 de um arquivo lendo-o em blocos.

    Parameters
    ----------
    path : Path
        Caminho do arquivo cujo hash será calculado.
    chunk : int, default=1048576
        Quantidade de bytes lidos por bloco.

    Returns
    -------
    str
        Hash SHA-256 em representação hexadecimal.
    """

    digest = hashlib.sha256()

    with path.open("rb") as arquivo:
        while True:
            bloco = arquivo.read(chunk)
            if not bloco:
                break
            digest.update(bloco)

    return digest.hexdigest()


def check_csv_columns(zip_file: zipfile.ZipFile, filename: str) -> None:
    """
    Verifica se o cabeçalho do CSV contém as colunas obrigatórias.

    Parameters
    ----------
    zip_file : zipfile.ZipFile
        Arquivo ZIP que contém o CSV.
    filename : str
        Caminho interno do CSV dentro do ZIP.

    Raises
    ------
    ValueError
        Se alguma das colunas NU_NOTA_MT ou Q006 estiver ausente.
    """

    with (
        zip_file.open(filename) as file,
        io.TextIOWrapper(file, encoding="latin-1") as text_file,
    ):
        header = text_file.readline().rstrip("\r\n")

    columns = {column.strip() for column in header.split(";")}
    required_columns = {"NU_NOTA_MT", "Q006"}
    missing_columns = required_columns - columns

    if missing_columns:
        raise ValueError(
            f"O arquivo csv {filename} não contém as colunas: {sorted(missing_columns)}"
        )
    print(f"Cabeçalho validado: {filename}")


def extract_csv(zip_path: Path, dest: Path) -> Path:
    """
    Extrai o CSV de microdados e o dicionário do ENEM para o destino.

    Antes da extração, verifica a presença dos arquivos esperados no ZIP
    e valida o cabeçalho do CSV. Se o CSV já existir e tiver as colunas
    obrigatórias, evita extraí-lo novamente.

    Parameters
    ----------
    zip_path : Path
        Caminho do arquivo ZIP dos microdados.
    dest : Path
        Diretório onde os arquivos serão extraídos.

    Returns
    -------
    Path
        Caminho do CSV de microdados extraído.

    Raises
    ------
    FileNotFoundError
        Se algum dos arquivos esperados não estiver no ZIP.
    ValueError
        Se o CSV não contiver as colunas obrigatórias.
    """

    dest.mkdir(parents=True, exist_ok=True)

    dictionary_internal = (
        "microdados_enem_2023/DICIONÁRIO/Dicionário_Microdados_Enem_2023.xlsx"
    )
    files = (CSV_INTERNO, dictionary_internal)

    with zipfile.ZipFile(zip_path) as zip_file:
        filenames = zip_file.namelist()

        for filename in files:
            if filename not in filenames:
                raise FileNotFoundError(f"Arquivo não encontrado no ZIP: {filename}")

        check_csv_columns(zip_file, CSV_INTERNO)

        for filename in files:
            dest_file = dest / Path(filename).name

            if dest_file.exists():
                if filename != CSV_INTERNO:
                    print(f"Arquivo já extraído: {dest_file}")
                    continue

                with dest_file.open("r", encoding="latin-1") as file:
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
