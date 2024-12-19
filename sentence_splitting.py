import spacy
from typing import List, Dict, Tuple
import regex as re
from connect import rg, client
from tqdm import tqdm

class NERDataProcessor:
    def __init__(self):
        # Carrega modelo spaCy para divisão de sentenças
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
        self.nlp.enable_pipe("senter")  # Habilita apenas o componente de sentenças
        
    def split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Divide o texto em sentenças e retorna cada sentença com seus índices.
        
        Returns:
            Lista de tuplas (texto_sentenca, inicio, fim)
        """
        doc = self.nlp(text)
        return [(sent.text, sent.start_char, sent.end_char) for sent in doc.sents]
    
    def adjust_spans_for_sentence(
        self, 
        sentence_start: int,
        sentence_end: int,
        spans: List[Dict]
    ) -> List[Dict]:
        """
        Ajusta as posições das entidades para a nova sentença.
        """
        adjusted_spans = []
        
        for span in spans:
            # Verifica se a entidade está dentro da sentença
            if (span["start"] >= sentence_start and 
                span["end"] <= sentence_end):
                
                # Ajusta os índices relativos à sentença
                adjusted_span = span.copy()
                adjusted_span["start"] = span["start"] - sentence_start
                adjusted_span["end"] = span["end"] - sentence_start
                adjusted_spans.append(adjusted_span)
                
        return adjusted_spans
    
    def process_argilla_records(self, dataset_name: str) -> List[Dict]:
        """
        Processa registros do Argilla, dividindo em sentenças.
        """
        # Carrega dados do Argilla
        dataset = client.datasets(dataset_name)
        status_filter = rg.Query(filter=rg.Filter(("response.status", "==", "submitted")))
        submitted = dataset.records(status_filter).to_list(flatten=True)
        
        processed_records = []
        
        print("Processando registros e dividindo em sentenças...")
        for record in tqdm(submitted):
            # Divide o texto em sentenças
            sentences = self.split_into_sentences(record["text"])
            
            # Processa cada sentença
            for sent_text, sent_start, sent_end in sentences:
                # Ajusta as posições das entidades para a sentença
                adjusted_spans = self.adjust_spans_for_sentence(
                    sent_start,
                    sent_end,
                    record["span_label.responses"][0]
                )
                
                # Só adiciona a sentença se ela contiver entidades
                if adjusted_spans:
                    processed_records.append({
                        "text": sent_text,
                        "original_start": sent_start,
                        "original_end": sent_end,
                        "original_id": record["id"],
                        "spans": adjusted_spans
                    })
        
        return processed_records
    
    def extract_ner_tags(self, text: str, spans: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Extrai tokens e tags NER para uma sentença.
        """
        tokens = text.split()
        current_position = 0
        ner_tags = []
        
        for token in tokens:
            token_start = text.find(token, current_position)
            token_end = token_start + len(token)
            
            tag = "O"
            for span in spans:
                if token_start >= span["start"] and token_end <= span["end"]:
                    if token_start == span["start"]:
                        tag = f"B-{span['label']}"
                    else:
                        tag = f"I-{span['label']}"
                    break
            
            ner_tags.append(tag)
            current_position = token_end
            
        return tokens, ner_tags
    
    def create_training_data(self, processed_records: List[Dict]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Cria dados de treinamento a partir dos registros processados.
        """
        all_tokens = []
        all_tags = []
        
        print("Criando dados de treinamento...")
        for record in tqdm(processed_records):
            tokens, tags = self.extract_ner_tags(record["text"], record["spans"])
            if len(tokens) == len(tags):  # Verificação de segurança
                all_tokens.append(tokens)
                all_tags.append(tags)
        
        return all_tokens, all_tags

def main():
    processor = NERDataProcessor()
    
    # Processa os dados
    processed_records = processor.process_argilla_records("IMDB Dataset NER")
    
    print(f"\nTotal de sentenças processadas: {len(processed_records)}")
    
    # Cria dados de treinamento
    tokens, ner_tags = processor.create_training_data(processed_records)
    
    print(f"\nTotal de sentenças para treinamento: {len(tokens)}")
    
    # Salva os dados processados
    import pandas as pd
    
    records_df = pd.DataFrame({
        "tokens": tokens,
        "ner_tags": ner_tags
    })
    
    records_df.to_pickle('records_df_ner.pkl')
    
    # Análise básica
    print("\nEstatísticas básicas:")
    sentence_lengths = [len(t) for t in tokens]
    print(f"Comprimento médio das sentenças: {sum(sentence_lengths)/len(sentence_lengths):.1f} tokens")
    print(f"Comprimento máximo: {max(sentence_lengths)} tokens")
    print(f"Comprimento mínimo: {min(sentence_lengths)} tokens")
    
    # Contagem de entidades
    entity_counts = {}
    for tags in ner_tags:
        for tag in tags:
            if tag != "O":
                entity_counts[tag] = entity_counts.get(tag, 0) + 1
    
    print("\nContagem de entidades:")
    for tag, count in sorted(entity_counts.items()):
        print(f"{tag}: {count}")

if __name__ == "__main__":
    main()