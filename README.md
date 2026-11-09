# statstech-analise

Aplicação da biblioteca [statstech](https://github.com/ml1-impatech/statstech)
em dois conjuntos de dados reais. Parte II do projeto final de Aprendizado de
Máquina I (IMPA Tech, 2026.2).

| Base | Tarefa | Pergunta |
|---|---|---|
| ENEM 2023 | Regressão | Quanto da nota o contexto familiar e a escola explicam? |
| Cardiotocografia (CTG) | Classificação em 3 classes | Dá para separar exames normais, suspeitos e patológicos só com as medidas automáticas? |

## Resultados

**ENEM (regressão).** Com validação cruzada no treino, a Ridge com codificação one-hot teve o menor erro (RMSE 72,5 pontos) e foi a escolhida. No teste, ela confirmou o resultado: RMSE 72,3 e R² 0,33, contra 88,4 de prever sempre a média. A rede neural empatou com ela (diferença de 0,13 ponto, IC 95% de −0,05 a 0,29). O contexto socioeconômico explica cerca de um terço da variação das notas, e de forma quase aditiva. Usaríamos a Ridge com one-hot: erra o mesmo que a rede, ajusta em segundos e é interpretável.

**CTG (classificação).** Com validação cruzada por gestante, a regressão logística teve o maior F1 macro (0,78) e foi a escolhida. No teste, F1 macro de 0,77 e AUC de 0,97 para o patológico. Com o limiar de 0,11, escolhido por validação cruzada, o modelo encontra 25 dos 27 patológicos do teste (recall 0,93), ao custo de mais alarmes falsos (precisão 0,36).

**Problemas encontrados.** Na CTG, cada gestante tem vários exames parecidos, e dividir por exame inflava as notas (o KNN parecia ótimo com k = 1); a solução foi validar por gestante. No ENEM, o KNN sofre com variáveis quase todas discretas, e o download dos microdados falhou várias vezes.

As tabelas completas estão em `results/tables/` e as figuras em `results/figures/`. A discussão completa está nos slides (`slides/`).

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

Depois, os notebooks na ordem:

- CTG: `ctg_exploration` → `ctg_models` → `ctg_final`
- ENEM: `enem_exploration_graficos` → `enem_exploration_kmeans` → `enem_linear` → `enem_knn` → `enem_rede` → `enem_final`

Se o download do ENEM falhar com `connection reset`, rode o script de novo: ele usa o `curl` com novas tentativas e continua de onde parou. Também é possível baixar o ZIP pelo navegador e colocá-lo em `data/raw/microdados_enem_2023.zip`.

O `requirements.txt` instala a statstech presa numa versão e instala o código
de `src/analysis` em modo editável.

## Microdados do ENEM 2023

**Fonte:** [Microdados do ENEM — INEP](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem).

**Arquivo:** `microdados_enem_2023.zip`

**Localização do ZIP:** `data/raw/microdados_enem_2023.zip`

**SHA-256:** `19512b7e29c9c8e940554396969722081341caa275c245a6c9d2d559c5d13a0a`

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

Em seguida, o script codifica as respostas do questionário (em escalas ordinais em `enem.npz` e em colunas 0/1 em `enem_onehot.npz`) e separa treino (80%) e teste (20%) **antes** de imputar os valores faltantes (pela mediana do treino) e de ajustar o PCA que resume os 19 itens de bens num índice. Assim, o teste não influencia nenhuma dessas etapas.

## Preparação dos dados da CTG

O script `scripts/prepare_ctg.py` lê `data/raw/CTG.xls` (versionado no repositório) e:
- remove as colunas com a classificação dos especialistas (CLASS, A a SUSP, LBE, DR), que entregariam a resposta;
- remove 12 linhas duplicadas, todas dentro do mesmo registro;
- converte as contagens (acelerações, contrações, desacelerações) em taxa por segundo;
- separa treino e teste **por gestante** (`FileName`), para que exames da mesma gestante não fiquem nos dois lados.

Restam 2.114 exames de 352 gestantes: 1.755 no treino e 359 no teste.

## Estrutura

```
statstech-analise/
├── requirements.txt       dependências, com a statstech presa numa versão
├── pyproject.toml         instala src/analysis
├── package.json           Delta, para gerar os slides
├── data/
│   ├── raw/               CTG.xls versionado; microdados do ENEM fora do git
│   └── processed/         ctg.npz, enem.npz, enem_onehot.npz (fora do git)
├── scripts/               download e preparação das bases
├── src/analysis/
│   ├── config.py          semente, tamanhos e caminhos
│   ├── io.py              salvar e carregar as bases preparadas
│   ├── validation.py      divisão treino/teste e validação cruzada por gestante (CTG)
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
  `analysis.io.load_dataset` e usam statstech, NumPy e matplotlib (pandas aparece na exploração e nas tabelas do ENEM, e o scikit-learn só na comparação com a rede da statstech, no `enem_rede`).
- **Código usado por mais de um notebook** vai para `src/analysis`, com
  docstring NumPy em português, type hints e teste.
- **Cada notebook tem uma frente responsável**, para evitar conflitos de merge.

## Slides

A apresentação está em `slides/apresentacao.dlt`, escrita em [Delta](https://github.com/rbribeiro/delta). Para gerar o HTML (requer Node.js 20 ou mais recente):

```bash
npm install
npm run slides
```

O resultado é `slides/apresentacao.html`, que abre em qualquer navegador.

## Testes

```bash
python -m pytest
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

## Uso de IA

Seguindo as políticas internas da instituição, o Claude (Anthropic) foi utilizado como ferramenta de consulta para explicações de conceitos e dos resultados, revisão de código e verificação de vazamento de dados, rascunhos de docstrings, testes e trechos dos slides, e apoio com o git. As decisões de análise, a escolha dos modelos e a interpretação dos resultados foram do grupo, e todo o código foi revisado e executado pelos integrantes.

## Licença

MIT. Veja o arquivo [`LICENSE`](LICENSE).
