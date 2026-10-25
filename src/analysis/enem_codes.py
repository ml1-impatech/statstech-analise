"""Mapas das letras do questionário socioeconômico do ENEM 2023.

Os códigos vêm do dicionário de variáveis do INEP (data/raw/enem/Dicionário_Microdados_Enem_2023.xlsx).
"""

# Escolariddade do pai (Q001) e da mãe (Q002), da mneor para o maior.
# A letra 'H" (nao sei) fica de fora da escala: virar NaN.
EDUCATION_LEVELS = {
    "A": 0,  # Nunca estudou
    "B": 1,  # Não completou o 4/5 ano
    "C": 2,  # Compeltou o 4/5 ano, mas nao o 8/9 ano
    "D": 3,  # Completou o 8/9 ano, mas nao o EM
    "E": 4,  # Completou o EM, mas nao a faculdade
    "F": 5,  # Completou a faculdade, mas nao a pos-graduacao
    "G": 6,  # Completou a pos
}
EDUCATION_UNKNOWN = "H"

# Ocupacao do pai (Q003) e da mae (Q004), do grupo 1 ao grupo 5.
# A letra 'F' (nao sei) fica fora da escala: NAN
OCCUPATION_GROUPS = {
    "A": 0,  # grupo 1:agro, pesca, extrativismo
    "B": 1,  # grupo 2: domestico, comercio, portaria
    "C": 2,  # grupo 3: operarios, oficios, motoristas
    "D": 3,  # grupo 4: professor, tecnicos, gerentes, pequenos empresarios
    "E": 4,  # grupo 5: profissões de nivel superior, grandes empresarios
}
OCCUPATION_UNKNOWN = "F"

# Renda familiar mensal (Q006): ṕonto medio de cada faixa, em reais.
# A faixa 'Q' (accima de 26.400,00) nao tem limite de cima
INCOME_MIDPOINTS = {
    "A": 0.0,
    "B": 660.0,
    "C": 1650.0,
    "D": 2310.0,
    "E": 2970.0,
    "F": 3630.0,
    "G": 4620.0,
    "H": 5940.0,
    "I": 7260.0,
    "J": 8580.0,
    "K": 9900.0,
    "L": 11220.0,
    "M": 12540.0,
    "N": 14520.0,
    "O": 17820.0,
    "P": 23100.0,
    "Q": 29700.0,
}

# Bens da residência (Q007 a Q025). Em todas as perguntas, "A" é "Não" e as
# letras seguintes são a quantidade ("B" = um ou "Sim", "C" = dois, ...).
ASSET_LEVELS = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

# Coluna do questionário e nome usado em X, na ordem das perguntas.
ASSET_COLUMNS = {
    "Q007": "domestic_worker",
    "Q008": "bathrooms",
    "Q009": "bedrooms",
    "Q010": "cars",
    "Q011": "motorcycles",
    "Q012": "refrigerators",
    "Q013": "freezers",
    "Q014": "washing_machines",
    "Q015": "dryers",
    "Q016": "microwaves",
    "Q017": "dishwashers",
    "Q018": "vacuum_cleaner",
    "Q019": "televisions",
    "Q020": "dvd_player",
    "Q021": "cable_tv",
    "Q022": "mobile_phones",
    "Q023": "landline",
    "Q024": "computers",
    "Q025": "internet",
}
