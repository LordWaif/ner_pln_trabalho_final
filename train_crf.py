import pandas as pd
import numpy as np
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
import joblib
from typing import List, Dict, Tuple
import json,spacy
from seqeval.metrics import classification_report
N = 3
def load_data() -> Tuple[Dict, pd.DataFrame]:
    """
    Carrega dados salvos e mapeamentos de labels.
    """
    # Carrega os dados processados
    records_df = pd.read_pickle('records_df_ner.pkl')
    
    # Carrega os mapeamentos de labels
    label_mappings = pd.read_pickle('label_mappings.pkl')
    
    return label_mappings, records_df

import regex as re
def word2features(sent: List[str], doc, i: int, N: int) -> Dict[str, str]:
    """
    Extrai features para uma palavra em uma posição específica, considerando uma janela deslizante de comprimento N.
    """
    word = sent[i]
    
    features = {
        'word.lower()': word.lower(),
        'word': word,
        'word.lemma()': doc[i].lemma_,
        'word.len()': len(word),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.isAlpha()': word.isalpha(),
        # Features POS
        'pos': doc[i].pos_,
        'tag': doc[i].tag_,
        'dep': doc[i].dep_,
        'isStop': doc[i].is_stop,
        'has.number': bool(re.search(r'\d', word)),
        'is.year': bool(re.match(r'\d{4}', word))
    }
    
    # Features das palavras na janela deslizante
    for n in range(1, N + 1):
        if i - n >= 0:
            word_prev = sent[i - n]
            features.update({
                f'-{n}:word.lower()': word_prev.lower(),
                f'-{n}:word': word_prev,
                f'-{n}:word.lemma()': doc[i - n].lemma_,
                f'-{n}:word.len()': len(word_prev),
                f'-{n}:word.istitle()': word_prev.istitle(),
                f'-{n}:word.isupper()': word_prev.isupper(),
                f'-{n}:word.isAlpha()': word_prev.isalpha(),
                f'-{n}:pos': doc[i - n].pos_,
                f'-{n}:tag': doc[i - n].tag_,
                f'-{n}:dep': doc[i - n].dep_,
                f'-{n}:isStop': doc[i - n].is_stop,
            })
        else:
            features[f'BOS-{n}'] = True  # Beginning of sentence
        
        if i + n < len(sent):
            word_next = sent[i + n]
            features.update({
                f'+{n}:word.lower()': word_next.lower(),
                f'+{n}:word': word_next,
                f'+{n}:word.lemma()': doc[i + n].lemma_,
                f'+{n}:word.len()': len(word_next),
                f'+{n}:word.istitle()': word_next.istitle(),
                f'+{n}:word.isupper()': word_next.isupper(),
                f'+{n}:word.isAlpha()': word_next.isalpha(),
                f'+{n}:pos': doc[i + n].pos_,
                f'+{n}:tag': doc[i + n].tag_,
                f'+{n}:dep': doc[i + n].dep_,
                f'+{n}:isStop': doc[i + n].is_stop,
            })
        else:
            features[f'EOS+{n}'] = True  # End of sentence
    
    return features

def sent2features(sent: List[str]) -> List[Dict[str, str]]:
    """
    Extrai features para todas as palavras em uma sentença.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(" ".join(sent))
    return [word2features(sent,doc, i,N) for i in range(len(sent))]

def prepare_data(
    records_df: pd.DataFrame,
    label_mappings: Dict
) -> Tuple[List[List[Dict]], List[List[str]]]:
    """
    Prepara dados para treinamento do CRF.
    """
    # records_df = records_df.sample(5)
    # Converte IDs de volta para labels
    id2label = label_mappings['id2label']
    # TODO Ponto interessante
    X = [sent2features(s) for s in records_df['tokens']]
    # X = records_df['tokens'].to_list()
    y = [[id2label[tag] for tag in tags] for tags in records_df['ner_tags']]
    # y = records_df['ner_tags'].to_list()
    
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
    report_det = classification_report(y_test, y_pred, output_dict=True)
    report_det = pd.DataFrame(report_det).transpose()
    report_det.to_csv('report_det.csv')

    # Calcula métricas por label
    print("\nMétricas por Label:")
    report = flat_classification_report(y_test, y_pred)
    report_dict = flat_classification_report(y_test, y_pred, output_dict=True)
    report_dict = pd.DataFrame(report_dict).transpose()
    report_dict.to_csv('report_dict.csv')
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
    n_train = int(0.8 * n_samples)
    
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