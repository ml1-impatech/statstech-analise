# statstech-analise

Aplicação da biblioteca [statstech](https://github.com/ml1-impatech/statstech)
em dois conjuntos de dados reais. Parte II do projeto final de Aprendizado de
Máquina I (IMPA Tech, 2026.2).

| Base | Tarefa | Pergunta |
|---|---|---|
| ENEM 2023 | Regressão | Quanto da nota o contexto familiar e a escola explicam? |
| Cardiotocografia (CTG) | Classificação em 3 classes | Dá para separar exames normais, suspeitos e patológicos só com as medidas automáticas? |

## Como reproduzir

Requer Python 3.11 ou mais recente.

```bash
git clone https://github.com/ml1-impatech/statstech-analise.git
cd statstech-analise
pip install -r requirements.txt
python scripts/download_enem.py
python scripts/prepare_ctg.py
python scripts/prepare_enem.py
```

Depois, os notebooks na ordem: exploração, modelos e avaliação final.

O `requirements.txt` instala a statstech presa numa versão e instala o código
de `src/analysis` em modo editável.

## Microdados do ENEM 2023

**Fonte:** [Microdados do ENEM — INEP](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem).

**Arquivo:** `microdados_enem_2023.zip`

**Localização do ZIP:** `data/raw/microdados_enem_2023.zip`

**SHA-256:** `19512b7e29c9c8e94055439669722081341caa275c245a6c9d2d559c5d13a0a`

O script `scripts/download_enem.py` baixa os microdados, verifica se as colunas `NU_NOTA_MT` e `Q006` estão presentes no mesmo CSV e extrai o arquivo de dados e o dicionário para `data/raw/enem/`.

Para obter os dados manualmente, acesse a página de microdados do INEP, selecione a edição de 2023 e coloque o arquivo ZIP em `data/raw/`. 

## Preparação dos dados do ENEM

O script `scripts/prepare_enem.py` prepara os microdados do ENEM 2023 para as análises de regressão do projeto.

A leitura é realizada em blocos (*chunks*), utilizando `usecols` para carregar apenas as colunas necessárias e reduzir o consumo de memória.

São aplicados os seguintes filtros:
- Presença nas quatro provas objetivas: Ciências da Natureza, Ciências Humanas, Linguagens e Matemática.
- Exclusão dos participantes identificados como treineiros.
- Seleção dos participantes conforme o critério de conclusão do Ensino Médio (`TP_ST_CONCLUSAO == 2`).
- Presença de notas válidas nas cinco provas, incluindo a Redação.

Após a filtragem, o script informa a quantidade de participantes restantes. Na execução registrada, **1.050.863 participantes** passaram pelos filtros. Em seguida, é selecionada uma amostra aleatória de até 50.000 participantes, com reprodutibilidade garantida pela semente definida em `src/analysis/config.py`.

A amostra é salva em `data/processed/enem_amostra.csv`.

## Estrutura

```
statstech-analise/
├── requirements.txt       dependências, com a statstech presa numa versão
├── pyproject.toml         instala src/analysis
├── data/
│   ├── raw/               CTG.xls versionado; microdados do ENEM fora do git
│   └── processed/         ctg.npz, enem.npz, enem_onehot.npz (fora do git)
├── scripts/               download e preparação das bases
├── src/analysis/
│   ├── config.py          semente, tamanhos e caminhos
│   ├── io.py              salvar e carregar as bases preparadas
│   ├── validation.py      divisão em treino e teste por registro
│   ├── evaluation.py      métricas padronizadas para todos os modelos
│   ├── report.py          tabelas e figuras em results/
│   ├── threshold.py       limiar de decisão da CTG
│   └── enem_codes.py      mapas do questionário do ENEM
├── tests/                 testes de src/analysis com pytest
├── notebooks/             exploração, modelos e avaliação final de cada base
├── results/
│   ├── figures/
│   ├── tables/
│   └── style.mplstyle     estilo comum das figuras
└── slides/
```

## Convenções

- **Semente e caminhos** vêm sempre de `analysis.config`:

  ```python
  from analysis.config import SEED, TEST_SIZE, PROCESSED, FIGURES
  ```

- **Figuras** começam com o estilo comum:

  ```python
  import matplotlib.pyplot as plt
  from analysis.config import STYLE

  plt.style.use(STYLE)
  ```

- **Depois da preparação**, os notebooks só carregam as bases com
  `analysis.io.load_dataset` e usam statstech, NumPy e matplotlib.
- **Código usado por mais de um notebook** vai para `src/analysis`, com
  docstring NumPy em português, type hints e teste.
- **Cada notebook tem uma frente responsável**, para evitar conflitos de merge.
- **Fluxo no git:** `main` protegida, uma branch por tarefa
  (`comum/c3-embrulho`, `ctg/t4-modelos`, ...) e Pull Request com uma revisão.

## Testes

```bash
pytest
```

## Autores

| Nome | GitHub |
|---|---|
| Daniela Moraes | [daniela-mmoraes](https://github.com/daniela-mmoraes) |
| Dulce Costa | [dulce-costa](https://github.com/dulce-costa) |
| Julia Fonseca | [JuliaQuiuquiF](https://github.com/JuliaQuiuquiF) |
| Lucas Silva | [lucas-impatech](https://github.com/lucas-impatech) |
| Natália Sousa | [NataliaBrandaodeSousa](https://github.com/NataliaBrandaodeSousa) |
| Manuela Ronconi | [manuronconi](https://github.com/manuronconi) |
| Maria Clara Fernandes | [mclaracarvalhoo](https://github.com/mclaracarvalhoo) |
| Maria Izabelle Silva | [izabelle02](https://github.com/izabelle02) |

## Licença

MIT. Veja o arquivo [`LICENSE`](LICENSE).
