from connect import rg  # Conexão com o Argilla
import spacy
import re
import pandas as pd
from tqdm import tqdm

# Carrega o modelo de idioma do spaCy
nlp = spacy.load("en_core_web_sm")

# Carrega os dados e seleciona uma amostra
df = pd.read_csv('IMDB Dataset.csv')
df = df.sample(5000, random_state=42)  # Garante replicabilidade

# Função para remover tags HTML
def remove_html_tags(text):
    return re.sub(r'<[^>]+>', '', text)

# Função para tokenizar o texto usando spaCy
def tokenize_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]  # Lista de tokens
    return tokens

# Remove as tags HTML
df['cleaned_review'] = df['review'].apply(remove_html_tags)

# Tokeniza os textos
tqdm.pandas(desc="Tokenizando os reviews")
df['tokens'] = df['cleaned_review'].progress_apply(tokenize_text)

df.to_csv('IMDB_Dataset_Tokens.csv', index=False,sep=';')

labels = [
    'PESSOA',
    'FILME',
    'CATEGORIA',
    'ORGANIZACAO',
    'LOCAL',
    'TEMPO',
    'DATA',
]

settings = rg.Settings(
    guidelines="IMDB Dataset parac reconhecimento de entidades",
    fields=[
        rg.TextField(
            name="text",
            title="Text",
            use_markdown=False,
        ),
    ],
    questions=[
        rg.SpanQuestion(
            name="span_label",
            field="text",
            labels=labels,
            title="Classifique os tokens de acordo com as categorias especific",
            allow_overlapping=False,
        )
    ],
)

dataset = rg.Dataset(
    name="IMDB Dataset NER",
    settings=settings,
)
dataset.create()
records = [rg.Record(fields={"text": " ".join(row["tokens"])}) for row in df.to_dict(orient="records")]

dataset.records.log(records)
