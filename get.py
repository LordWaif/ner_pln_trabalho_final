from connect import rg, client
import regex as re
from collections import Counter
from typing import List, Dict, Tuple
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import numpy as np

def load_argilla_data(dataset_name: str) -> List[Dict]:
    dataset = client.datasets(dataset_name)
    status_filter = rg.Query(filter=rg.Filter(("response.status", "==", "submitted")))
    return dataset.records(status_filter).to_list(flatten=True)

def get_iob_tag_for_token(token_start: int, token_end: int, ner_spans: List[Dict]) -> str:
    for span in ner_spans:
        if token_start >= span["start"] and token_end <= span["end"]:
            if token_start == span["start"]:
                return f"B-{span['label']}"
            else:
                return f"I-{span['label']}"
    return "O"

def extract_ner_tags(text: str, responses: List[Dict]) -> List[str]:
    tokens = re.split(r"(\s+)", text)
    ner_tags = []
    current_position = 0
    
    for token in tokens:
        if token.strip():
            token_start = current_position
            token_end = current_position + len(token)
            tag = get_iob_tag_for_token(token_start, token_end, responses)
            ner_tags.append(tag)
        current_position += len(token)
    
    return ner_tags

def process_records(submitted_records: List[Dict]) -> Tuple[List[List[str]], List[List[str]]]:
    tokens = []
    ner_tags = []
    
    for record in submitted_records:
        tags = extract_ner_tags(record["text"], record["span_label.responses"][0])
        tks = record["text"].split()
        if len(tks) == len(tags):  # Garante que tokens e tags têm o mesmo tamanho
            tokens.append(tks)
            ner_tags.append(tags)
    
    return tokens, ner_tags

def create_label_mappings(ner_tags: List[List[str]]) -> Tuple[Dict[int, str], Dict[str, int]]:
    labels = sorted(set([item for sublist in ner_tags for item in sublist]))
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: id_ for id_, label in id2label.items()}
    return id2label, label2id

def analyze_entity_distribution(ner_tags: List[List[str]]) -> Dict[str, int]:
    entity_counts = Counter()
    for sentence_tags in ner_tags:
        entity_counts.update(tag for tag in sentence_tags if tag != "O")
    return dict(entity_counts)

def simple_split(
    tokens: List[List[str]],
    ner_tags: List[List[str]],
    train_size: float = 0.75,
    val_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, Dataset]:
    """
    Realiza uma divisão simples dos dados, tentando manter entidades raras no conjunto de treino.
    """
    # Conta a frequência das entidades em cada sentença
    sentence_entity_counts = []
    for tags in ner_tags:
        counts = Counter(tag for tag in tags if tag != "O")
        sentence_entity_counts.append(len(counts))
    
    # Primeira divisão: treino e resto
    indices = np.arange(len(tokens))
    train_size_adjusted = int(len(tokens) * train_size)
    
    # Embaralha os índices
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    # Separa os conjuntos
    train_idx = indices[:train_size_adjusted]
    temp_idx = indices[train_size_adjusted:]
    
    # Divide o resto entre validação e teste
    val_size_adjusted = int(len(temp_idx) * (val_size / (1 - train_size)))
    val_idx = temp_idx[:val_size_adjusted]
    test_idx = temp_idx[val_size_adjusted:]
    
    # Mapear tags para IDs
    _, label2id = create_label_mappings(ner_tags)
    mapped_ner_tags = [[label2id[label] for label in tag_seq] for tag_seq in ner_tags]
    
    # Criar datasets
    def create_subset(indices):
        return Dataset.from_dict({
            "tokens": [tokens[i] for i in indices],
            "ner_tags": [mapped_ner_tags[i] for i in indices]
        })
    
    return DatasetDict({
        "train": create_subset(train_idx),
        "validation": create_subset(val_idx),
        "test": create_subset(test_idx)
    })

def main(dataset_name: str = "IMDB Dataset NER"):
    # Carregar e processar dados
    submitted_records = load_argilla_data(dataset_name)
    tokens, ner_tags = process_records(submitted_records)
    
    print(f"\nTotal de exemplos: {len(tokens)}")
    
    # Analisar distribuição inicial
    print("\nDistribuição inicial de entidades:")
    initial_dist = analyze_entity_distribution(ner_tags)
    for entity, count in sorted(initial_dist.items()):
        print(f"{entity}: {count}")
    
    # Realizar divisão
    dataset_dict = simple_split(tokens, ner_tags)
    
    # Salvar mapeamentos de labels
    id2label, label2id = create_label_mappings(ner_tags)
    pd.to_pickle({"id2label": id2label, "label2id": label2id}, "label_mappings.pkl")
    
    # Salvar o dataset completo em formato pickle
    records_df = pd.DataFrame({
        "tokens": tokens,
        "ner_tags": [[label2id[tag] for tag in tags] for tags in ner_tags]
    })
    records_df.to_pickle('records_df_ner.pkl')
    
    return dataset_dict, id2label, label2id

if __name__ == "__main__":
    dataset_dict, id2label, label2id = main()
    
    # Imprimir estatísticas
    print("\nTamanho dos conjuntos:")
    for split, dataset in dataset_dict.items():
        print(f"{split}: {len(dataset)} exemplos")