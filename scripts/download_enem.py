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
