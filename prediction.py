import joblib
import json
from typing import List, Dict

def load_model_and_mappings():
    """
    Carrega o modelo treinado e mapeamentos.
    """
    crf = joblib.load('ner_crf_model.joblib')
    with open('label_mappings.json', 'r') as f:
        label_mappings = json.load(f)
    return crf, label_mappings

def predict_entities(text: str, crf: CRF, tokenize: bool = True) -> List[Dict]:
    """
    Prevê entidades em um texto.
    
    Args:
        text: Texto para análise
        crf: Modelo CRF treinado
        tokenize: Se True, tokeniza o texto. Se False, assume que texto já é uma lista de tokens
    
    Returns:
        Lista de dicionários com entidades encontradas
    """
    # Tokenização básica se necessário
    if tokenize:
        tokens = text.split()
    else:
        tokens = text
    
    # Extrai features
    features = sent2features(tokens)
    
    # Faz a predição
    predictions = crf.predict([features])[0]
    
    # Formata resultado
    entities = []
    current_entity = None
    
    for i, (token, label) in enumerate(zip(tokens, predictions)):
        if label != 'O':
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token,
                    'label': label[2:],
                    'start': i,
                    'tokens': [token]
                }
            elif label.startswith('I-') and current_entity:
                current_entity['tokens'].append(token)
                current_entity['text'] += f" {token}"
        elif current_entity:
            entities.append(current_entity)
            current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

# Exemplo de uso
if __name__ == "__main__":
    # Carrega modelo
    crf, label_mappings = load_model_and_mappings()
    
    # Exemplo de texto
    texto = "João trabalha na Microsoft em São Paulo"
    
    # Faz predição
    entities = predict_entities(texto, crf)
    
    # Mostra resultados
    print("\nEntidades encontradas:")
    for entity in entities:
        print(f"- {entity['text']} ({entity['label']})")