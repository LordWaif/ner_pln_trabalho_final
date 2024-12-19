import pandas as pd
import numpy as np
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
import joblib
from typing import List, Dict, Tuple
import json
from seqeval.metrics import classification_report

def load_data() -> Tuple[Dict, pd.DataFrame]:
    """
    Carrega dados salvos e mapeamentos de labels.
    """
    # Carrega os dados processados
    records_df = pd.read_pickle('records_df_ner.pkl')
    
    # Carrega os mapeamentos de labels
    label_mappings = pd.read_pickle('label_mappings.pkl')
    
    return label_mappings, records_df

def word2features(sent: List[str], i: int) -> Dict[str, str]:
    """
    Extrai features para uma palavra em uma posição específica.
    """
    word = sent[i]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    
    # Features da palavra anterior
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence
    
    # Features da próxima palavra
    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True  # End of sentence
    
    return features

def sent2features(sent: List[str]) -> List[Dict[str, str]]:
    """
    Extrai features para todas as palavras em uma sentença.
    """
    return [word2features(sent, i) for i in range(len(sent))]

def prepare_data(
    records_df: pd.DataFrame,
    label_mappings: Dict
) -> Tuple[List[List[Dict]], List[List[str]]]:
    """
    Prepara dados para treinamento do CRF.
    """
    # Converte IDs de volta para labels
    id2label = label_mappings['id2label']
    
    X = [sent2features(s) for s in records_df['tokens']]
    # y = [[id2label[tag] for tag in tags] for tags in records_df['ner_tags']]
    y = records_df['ner_tags']
    
    return X, y

def train_crf(X_train: List[List[Dict]], y_train: List[List[str]]) -> CRF:
    """
    Treina modelo CRF com os parâmetros otimizados.
    """
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    crf.fit(X_train, y_train)
    return crf

def evaluate_model(
    crf: CRF,
    X_test: List[List[Dict]],
    y_test: List[List[str]]
) -> None:
    """
    Avalia o modelo e imprime métricas.
    """
    y_pred = crf.predict(X_test)
    
    # Gera relatório detalhado
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Calcula métricas por label
    print("\nMétricas por Label:")
    report = flat_classification_report(y_test, y_pred)
    print(report)

def save_model(crf: CRF, label_mappings: Dict) -> None:
    """
    Salva o modelo e mapeamentos necessários.
    """
    # Salva o modelo
    joblib.dump(crf, 'ner_crf_model.joblib')
    
    # Salva os mapeamentos em formato JSON para fácil acesso
    with open('label_mappings.json', 'w') as f:
        json.dump(label_mappings, f, indent=2)

def main():
    # Carrega dados
    print("Carregando dados...")
    label_mappings, records_df = load_data()
    
    # Prepara features e labels
    print("Preparando dados...")
    X, y = prepare_data(records_df, label_mappings)
    
    # Define índices para treino e teste (70-30)
    n_samples = len(X)
    n_train = int(0.7 * n_samples)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    
    print(f"Tamanho do conjunto de treino: {len(X_train)}")
    print(f"Tamanho do conjunto de teste: {len(X_test)}")
    
    # Treina o modelo
    print("\nTreinando modelo CRF...")
    crf = train_crf(X_train, y_train)
    
    # Avalia o modelo
    print("\nAvaliando modelo...")
    evaluate_model(crf, X_test, y_test)
    
    # Salva o modelo
    print("\nSalvando modelo...")
    save_model(crf, label_mappings)
    
    return crf, label_mappings

if __name__ == "__main__":
    crf, label_mappings = main()