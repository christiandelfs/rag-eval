#!/usr/bin/env python
# encoding: utf-8

import os
#import gc
import urllib

import numpy as np
import pandas as pd

from openai import OpenAI

import pyarrow as pa
import pyarrow.parquet as pq

import concurrent.futures

"""
DatasetProcessor

Klasse mit Methoden zur Erzeugung von Datensätzen und Splits
"""
class DatasetProcessor(object):
    def __init__(self):
        self.df_doc_processed = None
        # Embedding-Modell
        self.model_embedding = None
        # OpenAI API credentials
        self.oai_client_embedding = None
    
    #def __init__(self, openai_api, model_embedding):
    #    self.df_doc_processed = None
    #    # Embedding-Modell
    #    self.model_embedding = model_embedding
    #    
    #    # OpenAI API credentials
    #    # os.environ.get("OPENAI_API_KEY")
    #    self.oai_client_embedding = OpenAI(
    #        api_key=openai_api['key'],
    #        base_url=openai_api['base'],
    #    )

    """
    OpenAI-Client zuweisen
    """
    def set_openai_client(self, api_key, base_url, model_embedding):
        self.model_embedding = model_embedding
        
        # OpenAI API credentials
        # os.environ.get("OPENAI_API_KEY")
        self.oai_client_embedding = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
    """
    Dokument laden

    Prüfen, ob Datei lokal bereits vorhanden ist, andernfalls herunterladen
    """
    @staticmethod
    def get_file(doc_id: str, path: str):#self, 
        filename = f'{path}{doc_id}.pdf'

        # wenn Datei/PDF noch nicht vorhanden, PDF herunterladen
        if not os.path.isfile(filename):
            try:
                urllib.request.urlretrieve(f'https://arxiv.org/pdf/{doc_id}', filename=filename)
            except Exception as e:
                print(f"Exception occurred: {e}")
    
    """
    Dataset split

    Datensatz im angegebenen Verhältnis in zwei Sets splitten
    """
    @staticmethod
    def dataset_split(X, y, fraction_test, seed=42):#self, 
        # Seed setzen
        np.random.seed(seed)
        # Maske mit Zufallszahlen erstellen
        mask = np.random.choice([True, False], size=X.shape[0], p=[fraction_test, 1 - fraction_test])
        
        return X.loc[~mask], X.loc[mask], y.loc[~mask], y.loc[mask]

    """
    Cross validation Splits erstellen

    Für eine bestimmte Anzahl Splits eine entsprechende Teilmenge per Zufall aus dem Datensatz wählen (Bsp.: 5 Splits, 5 mal 1/5)
    """
    @staticmethod
    def dataset_cross_validation(df, num_splits=1, seed=42):
        # Seed setzen
        np.random.seed(seed)
        # Splits erstellen
        splits = []
        if num_splits > 0:
            fraction = 1 / num_splits
            # Ziehen mit Zurücklegen
            for i in range(num_splits):
                splits.append(np.random.choice([True, False], size=df.shape[0], p=[fraction, 1 - fraction]))
        return splits

    """
    Methode zur Erzeugung von Embedding für Block eines Dokuments
    """
    def create_document_embeddings(self, idx_block):
        responses_embeddings = self.oai_client_embedding.embeddings.create(
            input=self.df_doc_processed.loc[idx_block, 'text'],
            model=self.model_embedding
        )

        responses_embeddings_truncated = responses_embeddings.data[0].embedding#[:512]
        #responses_embeddings_truncated_normalized = responses_embeddings_truncated / np.linalg.norm(responses_embeddings_truncated)
        
        self.df_doc_processed.at[idx_block, 'embedding'] = responses_embeddings_truncated#_normalized

    """
    Merkmale erzeugen

    Einzelne vorverarbeitete Dataframes der PDF-Dokumente einlesen und Merkmale wie Anzahl Zeichen+Wörter sowie Embedding-Vektoren für Text-Blöcke erzeugen
    """
    def process_document(self, doc_id: str, path: str, reset=False):
        # Dataframe mit vorverarbeiteten Daten abrufen
        filename = f'{path}pdfs_df/{doc_id}.parquet'
        filename_processed = f'{path}pdfs_df_process/{doc_id}.parquet'
        
        # Wenn parquet-File mit Dataframe der Vorverarbeiteten Daten für Dokument bereits vorhanden 
        if os.path.isfile(filename_processed) and reset == False:
            self.df_doc_processed = pq.read_table(filename_processed).to_pandas()
        else:
        # Wenn parquet-File mit Dataframe der Vorverarbeiteten Daten für Dokument bereits vorhanden 
        #if os.path.isfile(filename):
            # parquet-Datei einlesen
            self.df_doc_processed = pq.read_table(filename).to_pandas()
            
            self.df_doc_processed['text_replace'] = self.df_doc_processed['text'].str.replace(r'\p{P}+', '', regex=True).str.replace(r'[\n\t\s]+', ' ', regex=True)
            # Anzahl Zeichen/Block 
            self.df_doc_processed['text_len'] = self.df_doc_processed['text'].str.len()
            # Anzahl Wörter/Block 
            self.df_doc_processed['word_count'] = self.df_doc_processed['text_replace'].str.split(' ').str.len()
            # Anzahl eindeutiger Wörter/Block 
            self.df_doc_processed['word_distinct_count'] = self.df_doc_processed['text_replace'].apply(lambda col: len(set(col.split(' '))))
            # Verhältnis Anzahl eindeutiger Wörter zu Gesamtzahl Wörter
            self.df_doc_processed['word_distinct_count_ratio'] = self.df_doc_processed['word_distinct_count'] / self.df_doc_processed['word_count']
            
            # Kapitelnummerierungen am Anfang aller Segementgrenzen entfernen
            #self.df_doc_processed.loc[df_doc_processed['boundary'] == True, 'text'] = self.df_doc_processed.loc[df_doc_processed['boundary'] == True, 'text'].str.replace(r'^((\d+|[A-Z]{1,3})(\.|\s))+', '', regex=True)

            # Embeddings für Blöcke des Dokuments/Dataframes erzeugen
            
            # Achtung!!!
            # Beim Erzeugen der Embeddings für die Blöcke eines Dokuments als Stapel wurde beobachtet, dass für einen Block an der gleichen Stelle in der Liste Embeddings mit unterschiedlichen Werten erzeugt wurden
            # Reihenfolge der Liste der Ausgabe des Modells bei Batch-Verarbeitung entspricht scheinbar nicht der Reihenfolge der Liste der Eingabe
            # Das Modell sollte deterministisch sein und zu einer Eingabe bei jedem Aufruf die gleiche Ausgabe liefern
            # Eine Reproduzierbarkeit der Ergebnisse ist nicht möglich, daher wird eine einzelne Erzeugung der Embedding für jeden Block eines Dokuments durchgeführt und zur Erhöhung der Performance des Prozesses Multi-Threading verwendet

            embeddings = self.oai_client_embedding.embeddings.create(
                input=self.df_doc_processed['text'],
                model=self.model_embedding
            )

            self.df_doc_processed['embedding'] = [embeddings.data[i].embedding for i in range(len(embeddings.data))]
        
            #print(responses_embeddings.data[0].embedding[0:10])
            #print(sum(responses_embeddings.data[0].embedding))

            # Threading verwenden
            # Spalte "embedding" hinzufügen, Pandas Series vom Typ "object" für Listen mit Embeddings
            #self.df_doc_processed['embedding'] = pd.Series(dtype='object')
    
            #with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            #    result = executor.map(self.create_document_embeddings, self.df_doc_processed.index)

            df_doc_processed_table = pa.Table.from_pandas(self.df_doc_processed)
            pq.write_table(df_doc_processed_table, filename_processed)

        return self.df_doc_processed

    """
    Dataset für Modell-Training erstellen

    Einzelne vorverarbeitete Dataframes aller PDF-Dokumente mit Merkmalen in Dictionary einlesen und Dataframe erstellen

    Merkmale des aktuellen und folgenden Abschnitt verwenden
    """
    @staticmethod
    def dataset_create(path, pdf_urls, embedding_dimension=768):#dataset_create_current_next
        # Dictionary mit Merkmalen/Spalten für Dataframe
        data = {'text_len': [], 'word_count': [], 'word_distinct_count_ratio': [], 'text_len_next': [], 'word_count_next': [], 'word_distinct_count_ratio_next': [], 'boundary': []}#, 'text_distinct_len': []

        # Keys für Komponenten der Embedding-Vektoren hinzufügen
        data = dict(data, **{f'c{i}': [] for i in range(embedding_dimension)})
        data = dict(data, **{f'n{i}': [] for i in range(embedding_dimension)})

        # über Dokumente iterieren
        for idx_doc in range(pdf_urls.shape[0]):
            doc_id = pdf_urls.iloc[idx_doc].name
            print(doc_id)

            filename = f'{path}pdfs_df_process/{doc_id}.parquet'

            # Wenn parquet-File mit Dataframe der Vorverarbeiteten Daten für Dokument bereits vorhanden 
            #if os.path.isfile(filename):
            df_doc_processed = pq.read_table(filename).to_pandas()#, memory_map=True

            # Einträge für die kein Embedding-Vektor vorhanden ist entfernen
            df_doc_processed = df_doc_processed.drop(df_doc_processed.loc[df_doc_processed['embedding'].isna()].index)

            data['text_len'].extend(df_doc_processed.iloc[:-1]['text_len'].tolist())
            data['word_count'].extend(df_doc_processed.iloc[:-1]['word_count'].tolist())
            data['word_distinct_count_ratio'].extend(df_doc_processed.iloc[:-1]['word_distinct_count_ratio'].tolist())
            data['text_len_next'].extend(df_doc_processed.iloc[1:]['text_len'].tolist())
            data['word_count_next'].extend(df_doc_processed.iloc[1:]['word_count'].tolist())
            data['word_distinct_count_ratio_next'].extend(df_doc_processed.iloc[1:]['word_distinct_count_ratio'].tolist())
            data['boundary'].extend(df_doc_processed.iloc[:-1]['boundary'].tolist())
            
            # Embedding-Vektoren kürzen
            df_doc_processed_embedding = df_doc_processed['embedding'].apply(pd.Series).loc[:, :embedding_dimension - 1]

            # Embedding-Vektoren kürzen und normieren
            #df_doc_processed_embedding = df_doc_processed['embedding'].apply(lambda x: x[:embedding_dimension] / np.linalg.norm(x[:embedding_dimension])).apply(pd.Series).loc[:, :embedding_dimension - 1]

            # Werte der Embedding-Vektoren des aktuellen und folgenden Abschnitts dem Dictionary hinzufügen
            for i in range(embedding_dimension):
                # Komponenten der Vektoren spaltenweise hinzufügen
                data[f'c{i}'].extend(np.float16(df_doc_processed_embedding.loc[:, i].tolist())[:-1])
                data[f'n{i}'].extend(np.float16(df_doc_processed_embedding.loc[:, i].tolist())[1:])

        return pd.DataFrame.from_dict(data)#.convert_dtypes()

    """
    Dataset für Modell-Training erstellen

    Einzelne vorverarbeitete Dataframes aller PDF-Dokumente mit Merkmalen in Dictionary einlesen und Dataframe erstellen

    Merkmale des aktuellen und vorherigen Abschnitt verwenden
    """
    """
    @staticmethod
    def dataset_create_current_previous(path, pdf_urls, embedding_dimension=384):
        # Dictionary mit Merkmalen/Spalten für Dataframe
        data = {'text_len': [], 'word_count': [], 'word_distinct_count_ratio': [], 'text_len_prev': [], 'word_count_prev': [], 'word_distinct_count_ratio_prev': [], 'boundary': []}

        # Keys für Komponenten der Embedding-Vektoren hinzufügen
        data = dict(data, **{f'c{i}': [] for i in range(embedding_dimension)})
        data = dict(data, **{f'n{i}': [] for i in range(embedding_dimension)})

        # über Dokumente iterieren
        for idx_doc in range(pdf_urls.shape[0]):
            doc_id = pdf_urls.iloc[idx_doc].name
            print(doc_id)

            # Dataframe der Vorverarbeiteten Daten für Dokument laden
            df_doc_processed = pq.read_table(f'{path}pdfs_df_process/{doc_id}.parquet').to_pandas()#, memory_map=True

            # Einträge, für welche kein Embedding-Vektor vorhanden ist entfernen
            df_doc_processed = df_doc_processed.drop(df_doc_processed.loc[df_doc_processed['embedding'].isna()].index)

            data['text_len'].extend(df_doc_processed.iloc[1:]['text_len'].tolist())
            data['word_count'].extend(df_doc_processed.iloc[1:]['word_count'].tolist())
            data['word_distinct_count_ratio'].extend(df_doc_processed.iloc[1:]['word_distinct_count_ratio'].tolist())
            data['text_len_prev'].extend(df_doc_processed.iloc[:-1]['text_len'].tolist())
            data['word_count_prevt'].extend(df_doc_processed.iloc[:-1]['word_count'].tolist())
            data['word_distinct_count_ratio_prev'].extend(df_doc_processed.iloc[:-1]['word_distinct_count_ratio'].tolist())
            data['boundary'].extend(df_doc_processed.iloc[1:]['boundary'].tolist())

            df_doc_processed_embedding = df_doc_processed['embedding'].apply(pd.Series).loc[:, :embedding_dimension - 1]

            # Werte der Embedding-Vektoren des aktuellen und vorhergehenden Abschnitts dem Dictionary hinzufügen
            for i in range(embedding_dimension):
                data[f'c{i}'].extend(np.float16(df_doc_processed_embedding.loc[:, i].tolist())[1:])
                data[f'n{i}'].extend(np.float16(df_doc_processed_embedding.loc[:, i].tolist())[:-1])

        return pd.DataFrame.from_dict(data)
    """

    """
    Dataset für Modell-Training erstellen

    Einzelne vorverarbeitete Dataframes aller PDF-Dokumente mit Merkmalen in Dictionary einlesen und Dataframe erstellen

    Merkmale des aktuellen, vorherigen und folgenden Abschnitt verwenden
    """
    """
    @staticmethod
    def dataset_create_current_previous_next(path, pdf_urls, embedding_dimension=384):
        # Dictionary mit Merkmalen/Spalten für Dataframe
        data = {'text_len': [], 'word_count': [], 'word_distinct_count_ratio': [], 'text_len_prev': [], 'word_count_prev': [], 'word_distinct_count_ratio_prev': [], 'text_len_next': [], 'word_count_next': [], 'word_distinct_count_ratio_next': [], 'boundary': []}

        # Keys für Komponenten der Embedding-Vektoren hinzufügen
        data = dict(data, **{f'c{i}': [] for i in range(embedding_dimension)})
        data = dict(data, **{f'p{i}': [] for i in range(embedding_dimension)})
        data = dict(data, **{f'n{i}': [] for i in range(embedding_dimension)})

        # über Dokumente iterieren
        for idx_doc in range(pdf_urls.shape[0]):
            doc_id = pdf_urls.iloc[idx_doc].name
            print(doc_id)

            # Dataframe der Vorverarbeiteten Daten für Dokument laden
            df_doc_processed = pq.read_table(f'{path}pdfs_df_process/{doc_id}.parquet').to_pandas()#, memory_map=True

            # Einträge, für welche kein Embedding-Vektor vorhanden ist entfernen
            df_doc_processed = df_doc_processed.drop(df_doc_processed.loc[df_doc_processed['embedding'].isna()].index)

            data['text_len'].extend(df_doc_processed.iloc[1:-1]['text_len'].tolist())
            data['word_count'].extend(df_doc_processed.iloc[1:-1]['word_count'].tolist())
            data['word_distinct_count_ratio'].extend(df_doc_processed.iloc[1:-1]['word_distinct_count_ratio'].tolist())
            data['text_len_prev'].extend(df_doc_processed.iloc[:-2]['text_len'].tolist())
            data['word_count_prev'].extend(df_doc_processed.iloc[:-2]['word_count'].tolist())
            data['word_distinct_count_ratio_prev'].extend(df_doc_processed.iloc[:-2]['word_distinct_count_ratio'].tolist())
            data['text_len_next'].extend(df_doc_processed.iloc[2:]['text_len'].tolist())
            data['word_count_next'].extend(df_doc_processed.iloc[2:]['word_count'].tolist())
            data['word_distinct_count_ratio_next'].extend(df_doc_processed.iloc[2:]['word_distinct_count_ratio'].tolist())
            data['boundary'].extend(df_doc_processed.iloc[1:-1]['boundary'].tolist())

            df_doc_processed_embedding = df_doc_processed['embedding'].apply(pd.Series).loc[:, :embedding_dimension - 1]

            # Werte der Embedding-Vektoren des aktuellen, vorhergehenden und folgenden Abschnitts dem Dictionary hinzufügen
            for i in range(embedding_dimension):
                data[f'c{i}'].extend(np.float16(df_doc_processed_embedding.loc[:, i].tolist())[1:-1])
                data[f'p{i}'].extend(np.float16(df_doc_processed_embedding.loc[:, i].tolist())[:-2])
                data[f'n{i}'].extend(np.float16(df_doc_processed_embedding.loc[:, i].tolist())[2:])

        return pd.DataFrame.from_dict(data)
    """