#!/usr/bin/env python
# encoding: utf-8

import os
import json

import regex as re

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

import pymupdf

"""
PDFProcessor

Klasse mit Methoden zur Verarbeitung der PDF-Dokumente
"""
class PDFProcessor(object):
    #def __init__(self):
    
    """
    doc_to_dataframe
    
    PDF-Dokument mit pymupdf elementweise einlesen und als Pandas DataFrame zurückgeben
    """
    @staticmethod
    def doc_to_dataframe(path: str):
        doc = pymupdf.open(path)
        # Dictionary zur Speicherung der Element-Eigenschaften und Texte der Dokumente
        data = {"block_type": [], "size":[], "text":[], "bbox_block":[], "bbox_span":[], "color":[], "font":[], "page_index":[], "block_index":[], "block_index_page":[], "line_index":[], "span_index":[]}
        
        # Über alle Seiten eines PDF-Dokuments iterieren
        for idx_page in range(0,len(doc)):
            page = doc[idx_page]
            idx_block_page = 0
            # https://pymupdf.readthedocs.io/en/latest/vars.html
            blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE|pymupdf.TEXT_PRESERVE_IMAGES)["blocks"]#|pymupdf.TEXT_DEHYPHENATE
            # Über alle Bläcke der Seite iterieren
            for idx_block in range(len(blocks)):
                block = blocks[idx_block]
                # Text Elemente
                if block['type'] == 0:
                    # Über alle Zeilen innerhalb eines Blocks iterieren
                    for idx_line in range(len(block['lines'])):
                        line = block['lines'][idx_line]
                        # Über alle Text-Elemente innerhalb einer Zeile iterieren
                        for idx_span in range(len(line['spans'])):
                            span = line['spans'][idx_span]
                            # text prüfen, länge > 0 (getrimmt)
                            if len(span["text"].strip()) > 0:
                                data["size"].append(span['size'])
                                data["text"].append(span['text'])
                                data["bbox_block"].append(block['bbox'])
                                data["bbox_span"].append(span['bbox'])
                                data["color"].append(span['color'])
                                data["font"].append(span['font'])
                                data["page_index"].append(idx_page)
                                data["block_index"].append(idx_block)
                                data["block_index_page"].append(idx_block_page)
                                data["block_type"].append(block['type'])
                                data["line_index"].append(idx_line)
                                data["span_index"].append(idx_span)
                # andere/nicht Text Elemente (z.B. Bild)
                else:
                    data["size"].append(0.0)
                    data["text"].append('')
                    data["bbox_block"].append(block['bbox'])
                    data["bbox_span"].append((0.0, 0.0, 0.0, 0.0))
                    data["color"].append(0)
                    data["font"].append('')
                    data["page_index"].append(idx_page)
                    data["block_index"].append(idx_block)
                    data["block_index_page"].append(idx_block_page)
                    data["block_type"].append(block['type'])
                    data["line_index"].append(0)
                    data["span_index"].append(0)
                idx_block_page = idx_block_page + 1
        
        # Pandas Dataframe aus Elementen erstellen
        df_doc = pd.DataFrame.from_dict(data).convert_dtypes()

        return df_doc

    """
    pdf_pages_to_image

    Seiten eines PDF-Dokument mit pymupdf in Grafik umwandeln
    """
    @staticmethod
    def pdf_pages_to_image(doc_id: str, path: str, reset=False):
        if reset == True:
            image_path = f'{path}images/{doc_id}/'
    
            # wenn Verzeichnis zur Speicherung der Bilder nicht vorhanden, Verzeichnis anlegen
            if not os.path.isdir(image_path):
                try:
                    os.mkdir(image_path)
                except Exception as e:
                    print(f"Exception occurred: {e}")
                    #continue
            
            # PDF mit pymupdf öffnen
            doc = pymupdf.open(f'{path}pdfs/{doc_id}.pdf')
            
            zoom = 1.5#2.0 # 2x = ~144 DPI
            mat = pymupdf.Matrix(zoom, zoom)
        
            # Seiten des PDFs in Bilder umwandeln
            for page in doc:#range(1, len(doc)):
                filename_image = f'{path}images/{doc_id}/page-{page.number}.png'
                # Wenn Bild von Seite des Dokuments noch nicht vorhanden, Bilder erstellen
                #if not os.path.isfile(filename_image):
                pix = page.get_pixmap(matrix=mat)#dpi=300
                pix.save(filename_image)

    """
    Preprocessing basic

    Dataframes mit Rohdaten der einzelnen PDF-Dokumente aus Funktion doc_to_dataframe einlesen und verarbeiten

    Einfache Vorverarbeitungsschritte durchführen und finale Dataframes für einzelne PDF-Dokumente erstellen
    """
    @staticmethod
    def process_dataframe(df_doc: pd.DataFrame, doc_id: str, path: str, reset=False):
        filename = f'{path}pdfs_df/{doc_id}.parquet'
        clean = True

        # Wenn parquet-File mit Dataframe der Vorverarbeiteten Daten für Dokument bereits vorhanden 
        if os.path.isfile(filename) and reset == False:
            #df_doc_group_page_index_block_index = pd.read_parquet(filename)
            df_doc_group_page_index_block_index = pq.read_table(filename).to_pandas()
        else:
            # Text-Blöcke entfernen, welche über einem Bild/einer Grafik liegen
            # Problematisch bei (großflächigen) Hintergrundbildern
            """df_doc['image_text'] = False
            # Cross-join zwischen Bild-Blöcken und Text-Blöcken
            df_doc_cross = df_doc.loc[df_doc['block_type'] == 1].reset_index().merge(df_doc.loc[df_doc['block_type'] == 0].reset_index(), how="cross")#drop=True
            # Einträge selektieren, welche sich auf der gleichen Seite befinden
            df_doc_cross = df_doc_cross.loc[df_doc_cross['page_index_x'] == df_doc_cross['page_index_y']]
            
            #df_doc = df_doc.drop(df_doc_cross.loc[
            #    (df_doc_cross['bbox_block_y'].str[0] >= df_doc_cross['bbox_block_x'].str[0]) & 
            #    (df_doc_cross['bbox_block_y'].str[1] >= df_doc_cross['bbox_block_x'].str[1]) & 
            #    (df_doc_cross['bbox_block_y'].str[2] >= df_doc_cross['bbox_block_x'].str[2]) & 
            #    (df_doc_cross['bbox_block_y'].str[3] >= df_doc_cross['bbox_block_x'].str[3]), 'index_y'])
        
            df_doc.loc[df_doc_cross.loc[
                (
                    (df_doc_cross['bbox_block_y'].str[0] >= df_doc_cross['bbox_block_x'].str[0]) & 
                    (df_doc_cross['bbox_block_y'].str[0] <= df_doc_cross['bbox_block_x'].str[2])
                ) | (
                    (df_doc_cross['bbox_block_y'].str[2] >= df_doc_cross['bbox_block_x'].str[0]) & 
                    (df_doc_cross['bbox_block_y'].str[2] <= df_doc_cross['bbox_block_x'].str[2])
                ) | (
                    (df_doc_cross['bbox_block_y'].str[1] >= df_doc_cross['bbox_block_x'].str[1]) & 
                    (df_doc_cross['bbox_block_y'].str[1] <= df_doc_cross['bbox_block_x'].str[3])
                ) | (
                    (df_doc_cross['bbox_block_y'].str[3] >= df_doc_cross['bbox_block_x'].str[1]) & 
                    (df_doc_cross['bbox_block_y'].str[3] <= df_doc_cross['bbox_block_x'].str[3])
                ), 'index_y'], 'image_text'] = True"""
            
            # Blöcke mit Type Text (0) selektieren (Bilder etc. ausschließen)
            df_doc = df_doc.loc[df_doc['block_type'] == 0]
            #df_doc = df_doc.loc[(df_doc['block_type'] == 0) & (df_doc['image_text'] == False)]

            # Leerzeichen einfügen
            # der Text eines Elements kann fortlaufend, ohne Leerzeichen zwischen den Zeichen/Wörtern zurückgegeben werden
            # Leerzeichen werden über Berechnung der horizontalen Distanz zwischen aufeinanderfolgenden Elementen hinzugefügt
            
            # horizontale Distanz zwischen aufeinanderfolgenden Elementen berechnen
            df_doc['distance'] = (df_doc['bbox_span'].str[0] - df_doc.shift()['bbox_span'].str[2]).round()#.drop("index", axis=1)#.apply(np.ceil)
            # Werte für distance < 0.0 oder NAN auf 0.0 setzen
            df_doc.loc[(df_doc['distance'] < 0.0) | (df_doc['distance'].isna()), 'distance'] = 0.0
        
            # Leerzeichen an Anfang aller Elemente mit distance > 0.0 anfügen
            df_doc.loc[df_doc['distance'] > 1.0, 'text'] = ' ' + df_doc.loc[df_doc['distance'] > 1.0, 'text']
            # Leerzeichen an Anfang des ersten Elements einer Zeile anfügen
            df_doc.loc[df_doc['span_index'] == 0, 'text'] = ' ' + df_doc.loc[df_doc['span_index'] == 0, 'text']
        
            # Blöcke entfernen, für welche die Breite geringer als die Höhe ist (vertikale Elemente)
            #df_doc = df_doc.drop(df_doc.loc[df_doc['bbox_block'].str[2] - df_doc['bbox_block'].str[0] < df_doc['bbox_block'].str[3] - df_doc['bbox_block'].str[1]].index).reset_index(drop=True)
        
            # Elemente nach Spalten "page_index" und "block_index" gruppieren und Inhalte/Werte der Spalte "text" mit Leerzeichen verbinden
            df_doc_group_page_index_block_index = df_doc.groupby(['page_index', 'block_index_page']).agg({'bbox_block': 'max', 'size': 'max', 'text': lambda x: ''.join(x)}).reset_index()
            #df_doc_group_page_index_block_index = df_doc.groupby(['page_index', 'block_index_page'])['text'].agg(''.join).reset_index()#block_index_page

            # Schriftgröße auf Ganzzahl aufrunden
            df_doc_group_page_index_block_index['size_int'] = df_doc_group_page_index_block_index['size'].apply(lambda x: np.ceil(x))
            #df_doc_group_page_index_block_index.assign(size_int=lambda x: np.ceil(x['size']))
            
            # drei oder mehr Zeilenumbrüche durch zwei Zeilenumbrüche ersetzen
            #df_corpus['section'] = df_corpus['section'].str.replace(r'\n{3,}', '\n\n', regex=True)
        
            # Zwei oder mehr Leerzeichen durch ein Leerzeichen ersetzen
            df_doc_group_page_index_block_index['text'] = df_doc_group_page_index_block_index['text'].str.replace(r'\s{2,}', ' ', regex=True)
            
            # '- ' durch '' ersetzen
            df_doc_group_page_index_block_index['text'] = df_doc_group_page_index_block_index['text'].str.replace('- ', '')
            
            # Leerzeichen am Anfang und Ende der Spalte text entfernen
            df_doc_group_page_index_block_index['text'] = df_doc_group_page_index_block_index['text'].str.strip(' ')#-
        
            # Anzahl Wörter
            df_doc_group_page_index_block_index['word_count'] = df_doc_group_page_index_block_index['text'].str.split(' ').str.len()
        
            ###
            # Behandlung Kopf- Fußzeilenelemente
            # Blöcke entfernen, welche auf allen oder jeder zweiten Seite vorkommen
            
            # Elemente der ersten Seite eines Dokuments ausschließen
            df_doc_group_page_index_block_index_page = df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index['page_index'] != 0]
        
            # Zahl+Leerzeichen am Anfang und Ende der Blöcke entfernen
            df_doc_group_page_index_block_index_page['text'] = df_doc_group_page_index_block_index_page['text'].str.replace(r'^\d+\s+', '', regex=True).str.replace(r'\s+\d+$', '', regex=True)
            
            # Blöcke entfernen, welche auf jeder Seite vorkommen
            # Bsp.
            # Dokument: 2402.03953v4
            # SSN 2379-5980 (online)\nDOI 10.5195/LEDGER.201X.X
            # LEDGER VOL X (201X) 1-5
            
            # Blöcke entfernen, welche auf jeder zweiten geraden oder ungeraden Seite vorkommen
            # Bsp.
            # 2411.00713v1
            # JONAS BLESSING, MICHAEL KUPPER, AND ALESSANDRO SGARABOTTOLO
            # RISK-BASED PRICES
            # erste Seite ausschließen
            # Cross-Join Dataframe
            
            # Alle Blöcke des Dataframe mit allen Blöcken verbinden (Cross-Join)
            df_doc_group_page_index_block_index_merge = df_doc_group_page_index_block_index_page.merge(df_doc_group_page_index_block_index_page, how='cross')
            
            # Blöcke auf selber Seite ausschließen
            #df_doc_group_page_index_block_index_merge = df_doc_group_page_index_block_index_merge.loc[df_doc_group_page_index_block_index_merge['page_index_x'] != df_doc_group_page_index_block_index_merge['page_index_y']]
            
            # Blöcke mit gleichen Werten auf unterschiedlichen Seiten selektieren
            df_doc_group_page_index_block_index_merge = df_doc_group_page_index_block_index_merge.loc[df_doc_group_page_index_block_index_merge['text_x'] == df_doc_group_page_index_block_index_merge['text_y']]
            
            # Dataframe nach Spalte text gruppieren und eindeutigen Nummern der Seiten mit Vorkommen ermitteln
            df_doc_group_page_index_block_index_merge_group = df_doc_group_page_index_block_index_merge.groupby(by=['text_x'])['page_index_y'].unique().reset_index()
        
            # Alle Blöcke selektieren, für welche der text nicht auf allen oder jeder zweiten Seite vorkommt
            df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.loc[
                ~df_doc_group_page_index_block_index['text'].str.replace(r'^\d+\s+', '', regex=True).str.replace(r'\s+\d+$', '', regex=True).isin(
                    df_doc_group_page_index_block_index_merge_group.loc[(
                        # Vorkommen auf allen Seiten prüfen
                        (df_doc_group_page_index_block_index_merge_group['page_index_y'].apply(tuple) == tuple(np.arange(start=1,stop=df_doc['page_index'].nunique(), step=1).tolist())) | 
                        # Vorkommen auf jeder Zweiten Seite, beginnend ab Seite 1 prüfen
                        (df_doc_group_page_index_block_index_merge_group['page_index_y'].apply(tuple) == tuple(np.arange(start=1,stop=df_doc['page_index'].nunique(), step=2).tolist())) | 
                        # Vorkommen auf jeder Zweiten Seite, beginnend ab Seite 2 prüfen
                        (df_doc_group_page_index_block_index_merge_group['page_index_y'].apply(tuple) == tuple(np.arange(start=2,stop=df_doc['page_index'].nunique(), step=2).tolist()))
                    ), 'text_x']
                )
            ].reset_index(drop=True)
        
            ### 
            # Blöcke des PDF-Dokuments entsprechend der strukutrierten Referenz-Daten als Segmentgrenzen markieren
            df_doc_group_page_index_block_index['boundary'] = False
        
            # struktuierte Dokument-Daten aus Korpus laden
            try:
                file = open(f'{path}corpus/{doc_id}.json', 'r')
                data = json.load(file)
                file.close()
            except Exception as e:
                print(f"Exception occurred: {e}")
        
            # Pandas Datafrome mit Dokument-Abschnitten aus Korpus erstellen
            df_corpus = pd.DataFrame({'section':pd.Series([section['text'] for section in data['sections']], dtype=pd.StringDtype)})

            # Muster "^#+\s" (Markdown-Headings)
            re_md_heading = re.compile(r'^#+\s')

            # Inhalt der Abschnitte nach Zeilenumbrüchen spliten
            # für jeden Abschnitt aus Liste Einträge mit Vorkommen des Musters re_md_heading selektieren und durch '' ersetzen
            df_corpus_first_line = df_corpus['section'].str.split('\n').apply(lambda x: [re_md_heading.sub('', s) for s in x if re_md_heading.match(s)]).explode()
        
            # keine Buchstaben und Leerzeichen
            re_ws = re.compile(r'[^\w\s]')
            # ersetzen ^[A-Z]\.\s[0-9]+ durch ^[A-Z]\.)([0-9]+ (Leerzeichen entfernen)
            # Bsp.
            # Dokument: 2410.08709v3
            # Referenz: A. 1 Surrogate of distillation loss
            # Ziel: A.1 Surrogate of distillation loss
            re_wpd = re.compile(r'(^[A-Z]\.)\s([0-9]+)')
        
            # über alle Abschnitte aus Korpus iterieren
            for boundary in df_corpus_first_line:
                # Übereinstimmung zwischen Abschnitten aus Korpus und PDF-Dokument ermitteln
                # Zeile in Dataframe für PDF-Dokument als Segmentgrenze markieren
                # Alles was nicht Buchstabe, Zahl, Unterstrich oder Leerzeichen ist ersetzen durch ''
                # Bsp.:
                # Dokument: 2411.00713v1
                # 3. Agent’s preferences and indifference pricing
                # Referenz: ' - Apostrophe (Ascii)
                # Ziel: ’ - Right Single Quotation Mark (Unicode)
                df_doc_group_page_index_block_index.loc[
                    df_doc_group_page_index_block_index['text'].str.replace(r'[^\w\s]', '', regex=True)
                    .str.strip()
                    .str.contains(r'^' + re.escape(re_ws.sub('', re_wpd.sub(r'\1\2', boundary)).strip()), regex=True, case=False), 'boundary'
                ] = True
        
            # Blöcke mit erkannten Grenzen aus Referenz selektieren, welche Suchmuster für Grenzen entsprechen
            df_doc_group_page_index_block_index_boundary = (
                (
                    (df_doc_group_page_index_block_index['text'].str.contains(r'^((\d+|[A-Z]{1,3})(\.|\s))+', regex=True, case=False)) | 
                    (df_doc_group_page_index_block_index['text'].str.contains(r'^(Abstract|Introduction|Contents|Appendix|References|Conclusion|Acknowledgment|Acknowledgement)', regex=True, case=False))#$
                ) & 
                (df_doc_group_page_index_block_index['boundary'] == True)
            )
        
            # Blöcke mit erkannten Grenzen aus Referenz selektieren, welche Suchmuster für Grenzen nicht entsprechen
            df_doc_group_page_index_block_index_boundary_neg = (
                ~(
                    (df_doc_group_page_index_block_index['text'].str.contains(r'^((\d+|[A-Z]{1,3})(\.|\s))+', regex=True, case=False)) | 
                    (df_doc_group_page_index_block_index['text'].str.contains(r'^(Abstract|Introduction|Contents|Appendix|References|Conclusion|Acknowledgment|Acknowledgement)', regex=True, case=False))
                ) & 
                (df_doc_group_page_index_block_index['boundary'] == True)
            )

            # Markierung als Grenze für Blöcke mit erkannter Grenze aus Referenz, welche Suchmuster für Grenzen nicht entsprechen, auf False setzen
            df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index_boundary_neg, 'boundary'] = False

            # Dokumente ausschließen, für die Grenzen nicht abgeglichen werden konnten
            
            # Wenn nicht (
            # Anzahl erkannter/markierter Grenzen größer 4
            # und Anzahl erkannter Grenzen größer Anzahl nicht erkannter Grenzen
            # )
            # weitere Verarbeitung abbrechen
            if not (
                df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index_boundary].shape[0] > 4 and 
                df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index_boundary].shape[0] > df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index_boundary_neg].shape[0]
            ):
                #continue
                #return None
                clean = False
            
            #print(f'{df_corpus_first_line.shape[0] - df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index['boundary'] == True].shape[0]} - {df_corpus_first_line.shape[0]} - {df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index['boundary'] == True].shape[0]}')
        
            # Cleanup
            # arxiv-Label entfernen
            df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.drop(df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index['text'].str.strip(' ').str.contains(r'^arxiv\:\d{4}\.\d+v\d+', case=False, regex=True)].index).reset_index(drop=True)
            # ersten Block entfernen (Titel)
            #df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.drop(0).reset_index(drop=True)
            
            # Seitennummerierungen behandeln
            # Alle Blöcke selektieren, für welche der Wert der Spalte text keine Zahl ist
            #df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.loc[~df_doc_group_page_index_block_index['text'].str.strip().str.match(r'^\d+$')].reset_index(drop=True)#(df_doc_group_page_index_block_index['block_index_page'] > 0) & (
            
            # Ersten Block einer Seite, welcher eine Zahl enthält, entfernen
            df_doc_group_page_index_block_index_first = df_doc_group_page_index_block_index.loc[
                df_doc_group_page_index_block_index.groupby(['page_index'])['block_index_page']
                .idxmin()
                .reset_index()['block_index_page']
            ]
            df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.drop(
                df_doc_group_page_index_block_index_first.loc[
                    df_doc_group_page_index_block_index_first['text'].str.strip()
                    .str.match(r'^\d+$')
                ].index
            ).reset_index(drop=True)
        
            # Letzten Block einer Seite, welcher eine Zahl enthält, entfernen
            df_doc_group_page_index_block_index_last = df_doc_group_page_index_block_index.loc[
                df_doc_group_page_index_block_index.groupby(['page_index'])['block_index_page']
                .idxmax()
                .reset_index()['block_index_page']
            ]
            df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.drop(
                df_doc_group_page_index_block_index_last.loc[
                    df_doc_group_page_index_block_index_last['text'].str.strip()
                    .str.match(r'^\d+$')
                ].index
            ).reset_index(drop=True)
        
            # block_index_page für erstes Element jeder Seite auf 0 setzen
            #df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index.groupby(by=['page_index'])['block_index_page'].head(1).index, 'block_index_page'] = 0#df_doc_group_page_index_block_index_page
            df_doc_group_page_index_block_index.loc[
                df_doc_group_page_index_block_index.groupby(by=['page_index'])['block_index_page']
                .idxmin()
                .reset_index(drop=True)
                .values, 'block_index_page'
            ] = 0#df_doc_group_page_index_block_index_page
            
            # Fußnoten behandeln
            
            # Alle Blöcke selektieren, für welche der Werte der Spalte text nicht mit einer Zahl, einem mathematischen Symbol oder Punctuation other beginnt und kein Leerzeichen oder Punkt folgt
            #pattern = re.compile(r'^[\d\p{Sm}\p{Po}]+(?![\s\.])')#^(\d+|\p{Sm}|\p{Po})(?!(\s|\.))
            
            #df_doc_group_page_index_block_index_select = df_doc_group_page_index_block_index.loc[
            #    (df_doc_group_page_index_block_index['boundary'] == False) & 
            #    (df_doc_group_page_index_block_index['text'].str.strip().str.contains(r'^[\d\p{Sm}\p{Po}]+', regex=True))#, flags=re.UNICODE
            #]
            
            #df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.drop(df_doc_group_page_index_block_index_select.loc[~df_doc_group_page_index_block_index_select['text'].str[1:].str.contains(r'^[\s\.\p{Sm}]', regex=True)].index).reset_index(drop=True)#
        
            #df_doc_group_page_index_block_index['footnote'] = False
            
            # Index des letzten Blocks jeder Seite ermitteln
            df_doc_group_page_index_last_block_index = df_doc_group_page_index_block_index.groupby(['page_index'])['block_index_page'].idxmax().reset_index().sort_values(['page_index'],ascending=False)['block_index_page'].values
            df_page_last_block = df_doc_group_page_index_block_index.loc[df_doc_group_page_index_last_block_index]
        
            # Kleinste Schriftgröße aus selektierten Blöcken ermitteln
            df_page_last_block_size_min = df_page_last_block['size_int'].min()
            #df_page_last_block_size_min = df_page_last_block.loc[df_page_last_block['text'].str.contains(r'^[\d\p{Sm}\p{Po}]+', regex=True), 'size_int'].min()
        
            # Letzten Block einer Seite mit vorhergehendem Block verbinden, solange Seitennummer und Schriftgröße übereinstimmen
            for i in df_doc_group_page_index_last_block_index:
                it = 0
        
                # Solage index größer-gleich 0
                # und Seitennummer des aktuellen Blocks gleich der Seitennummer des vorhergehenden Blocks
                # und Schriftgröße des aktuellen Blocks gleich der Schriftgröße des vorhergehenden Blocks 
                # und Schriftgröße des aktuellen Blocks gleich der minimalen Schriftgröße
                while (
                    i - it - 1 >= 0 and 
                    df_doc_group_page_index_block_index.loc[i - it, 'page_index'] == df_doc_group_page_index_block_index.loc[i - it - 1, 'page_index'] and 
                    df_doc_group_page_index_block_index.loc[i - it, 'size_int'] == df_doc_group_page_index_block_index.loc[i - it - 1, 'size_int'] and 
                    df_doc_group_page_index_block_index.loc[i - it, 'size_int'] == df_page_last_block_size_min
                ):
                    df_doc_group_page_index_block_index.loc[i - it - 1, 'text']+= ' ' + df_doc_group_page_index_block_index.loc[i - it, 'text']
                    df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.drop(i - it)#index=
                    it+= 1
        
            df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.reset_index(drop=True)
        
            # Index des letzten Block jeder Seite ermitteln
            # Blöcke nach Seiten nach Seitennummern gruppieren und Block mit der höchsten Nummer wählen
            df_doc_group_page_index_last_block_index = df_doc_group_page_index_block_index.groupby(['page_index'])['block_index_page'].idxmax().reset_index().sort_values(['page_index'],ascending=False)['block_index_page'].values
            #df_page_last_block = df_doc_group_page_index_block_index.loc[df_doc_group_page_index_last_block_index]
        
            # Kleinste Schriftgröße aus selektierten Blöcken ermitteln
            #df_page_last_block_size_min = df_page_last_block.loc[df_page_last_block['text'].str.contains(r'^[\d\p{Sm}\p{Po}]+', regex=True), 'size_int'].min()
        
            # Suchmuster für Beginn Fußnoten
            re_footnote = re.compile(r'^[\d\p{Sm}\p{Po}]+')
        
            # Letzten Block einer Seite entfernen
            for i in df_doc_group_page_index_last_block_index:
                it = 0
                page_index = df_doc_group_page_index_block_index.loc[i - it, 'page_index']
        
                # Solage Seitennummer des aktuellen Blocks gleich der Seitennummer des letzten Blocks
                # und Schriftgröße des aktuellen Blocks gleich der minimalen Schriftgröße
                # und text entspricht Suchmuster für Fußnoten
                while (
                    df_doc_group_page_index_block_index.loc[i - it, 'page_index'] == page_index and 
                    df_doc_group_page_index_block_index.loc[i - it, 'size_int'] == df_page_last_block_size_min and 
                    re_footnote.match(df_doc_group_page_index_block_index.loc[i - it, 'text'])
                ):
                    page_index = df_doc_group_page_index_block_index.loc[i - it, 'page_index']
                    df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.drop(i - it)#index=
                    #df_doc_group_page_index_block_index.loc[i - it, 'footnote'] = True
                    it+= 1
        
            #df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index['footnote'] == False]
            df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.reset_index(drop=True)
            
            # Zusammenfügen von Blöcken
            
            # Dataframes df_doc.shift() und df_doc mergen
            df_doc_shift = df_doc_group_page_index_block_index.shift().merge(df_doc_group_page_index_block_index, how='inner', left_index=True, right_index=True)
        
            # Aufeinander foglende Blöcke mit weniger als 8 Wörtern selektieren
            df_doc_shift_word_blocks = (
                (df_doc_shift['word_count_x'] < 8) & 
                (df_doc_shift['boundary_x'] == False) & 
                (df_doc_shift['word_count_y'] < 8) & 
                (df_doc_shift['boundary_y'] == False)# & 
                #(df_doc_shift['size_int_x'] == df_doc_shift['size_int_y'])
            )
        
            # Iterativer Prozess für aufeinander folgende Blöcke
            # Absteigend nach Seitennummer und Block-Index sortieren und text zusammenfügen
            for i in df_doc_shift.loc[df_doc_shift_word_blocks].sort_values(['page_index_x', 'block_index_page_x'], ascending=False).index:
                if i + 1 < df_doc_shift.shape[0] - 1:
                    df_doc_shift.loc[i, 'text_x']+= ' ' + df_doc_shift.loc[i + 1, 'text_x']
                    #df_doc_shift = df_doc_shift.drop(i + 1)
        
            #df_doc_shift = df_doc_shift.reset_index(drop=True)
            
            # Folgende Blöcke löschen
            df_doc_shift = df_doc_shift.drop(df_doc_shift.loc[
                df_doc_shift_word_blocks & 
                (df_doc_shift.index < df_doc_shift.shape[0] - 1)
            ].index + 1).reset_index(drop=True)
            
            # Behandlung mehrspaltiger Layouts
            
            # Zusammenfügen der Werte der Spalte text wenn Koordinate left der Bounding-Box des folgenden Blocks größer als right des aktuellen Blocks
            df_doc_shift_concat_column_last_first_block = (
                (df_doc_shift['boundary_x'] == False) & 
                (df_doc_shift['boundary_y'] == False) & 
                #~(df_doc_shift['text_y'].str.contains(r'^((\d+|[A-Z]{1,3})(\.|\s))+', regex=True)) & 
                (df_doc_shift['bbox_block_y'].str[0] > df_doc_shift['bbox_block_x'].str[2])# & 
                #(df_doc_shift['size_int_x'] == df_doc_shift['size_int_y'])
            )
            
            df_doc_shift.loc[df_doc_shift_concat_column_last_first_block, 'text_x']+= ' ' + df_doc_shift.loc[df_doc_shift_concat_column_last_first_block, 'text_y']
        
            # Folgenden Block löschen
            df_doc_shift = df_doc_shift.drop(df_doc_shift.loc[
                df_doc_shift_concat_column_last_first_block & 
                (df_doc_shift.index < df_doc_shift.shape[0] - 1)
            ].index + 1).reset_index(drop=True)
            
            # Zusammenfügen der Werte der Spalte text des letzten Abschnitts/Blocks einer Seite mit dem ersten Abschnitt der folgenden Seite bei gleicher Schriftgröße
            # Erste Seite, Segmentgrenzen überspringen
            df_doc_shift_concat_page_last_first_block = (
                (df_doc_shift['page_index_y'] > 0) & 
                (df_doc_shift['boundary_x'] == False) & 
                (df_doc_shift['boundary_y'] == False) & 
                #~(df_doc_shift['text_y'].str.contains(r'^((\d+|[A-Z]{1,3})(\.|\s))+', regex=True)) & 
                (df_doc_shift['block_index_page_y'] == 0) & 
                (df_doc_shift['size_int_x'] == df_doc_shift['size_int_y'])
            )
        
            # Iterativer Prozess bei Abschnitten über mehrere Seiten
            for i in df_doc_shift.loc[df_doc_shift_concat_page_last_first_block].sort_values(['page_index_x', 'block_index_page_x'], ascending=False).index:
                if i + 1 < df_doc_shift.shape[0] - 1:
                    df_doc_shift.loc[i, 'text_x']+= ' ' + df_doc_shift.loc[i + 1, 'text_x']
                    #df_doc_shift = df_doc_shift.drop(i + 1)
        
            #df_doc_shift = df_doc_shift.reset_index(drop=True)
            
            # Folgenden Block löschen
            df_doc_shift = df_doc_shift.drop(df_doc_shift.loc[
                df_doc_shift_concat_page_last_first_block & 
                (df_doc_shift.index < df_doc_shift.shape[0] - 1)
            ].index + 1).reset_index(drop=True)
            
            # Zeilen und Spalten des Dataframe df_doc_shift für weitere Verarbeitung selektieren
            df_doc_group_page_index_block_index = (
                df_doc_shift.iloc[1:][['text_x', 'boundary_x']]#, 'page_index_x', 'block_index_page_x'
                .reset_index(drop=True)
                .rename({'text_x': 'text', 'boundary_x': 'boundary'}, axis=1)#, 'page_index_x': 'page_index', 'block_index_page_x': 'block_index_page'
            )
        
            # Behandlung � (Unicode replacement character \uFFFD, z.B. Summen- oder Produktzeichen), führt zu Zeilenumbruch/neuem Block
            
            # Iterativer Prozess bei mehreren aufeinander folgenden Blöcken, welche zusammenhängen
            # Reverse order (von unten nach oben), text des vorherigen Blocks an 
            df_doc_group_page_index_block_index_max = df_doc_group_page_index_block_index.index.max()
            for i in reversed(df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index['text'].str.contains(u'\uFFFD$')].index.tolist()):
                if i < df_doc_group_page_index_block_index_max:
                    df_doc_group_page_index_block_index.loc[i, 'text']+= ' ' + df_doc_group_page_index_block_index.loc[i + 1, 'text']
                    df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.drop(i + 1)
        
            df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.reset_index(drop=True)
        
            #print(df_doc_group_page_index_block_index['text'].agg('\n\n'.join))

            # Dataframe als parquet speichern
            #df_doc_group_page_index_block_index.astype({'block_index':'int64', 'page_index': 'int64'}).to_parquet(filename, compression='gzip')
            df_doc_group_page_index_block_index_table = pa.Table.from_pandas(df_doc_group_page_index_block_index)
            pq.write_table(df_doc_group_page_index_block_index_table, filename)

        return clean, df_doc_group_page_index_block_index

    """
    Preprocessing basic

    Dataframes mit Rohdaten der einzelnen PDF-Dokumente aus Funktion doc_to_dataframe einlesen und verarbeiten

    Einfache Vorverarbeitungsschritte durchführen und finale Dataframes für einzelne PDF-Dokumente erstellen
    """
    """
    @staticmethod
    def process_dataframe_basic(df_doc: pd.DataFrame, doc_id: str, path: str, reset=False):
        filename = f'{path}pdfs_df/{doc_id}.parquet'
        clean = True

        # Wenn parquet-File mit Dataframe der Vorverarbeiteten Daten für Dokument bereits vorhanden 
        if os.path.isfile(filename) and reset == False:
            #df_doc_group_page_index_block_index = pd.read_parquet(filename)
            df_doc_group_page_index_block_index = pq.read_table(filename).to_pandas()
        else:
            # Blöcke mit Type Text (0) selektieren (Bilder ausnehmen)
            df_doc = df_doc.loc[df_doc['block_type'] == 0]
            #df_doc = df_doc.loc[(df_doc['block_type'] == 0) & (df_doc['image_text'] == False)]
        
            # Leerzeichen am Anfang und Ende der Spalte text entfernen
            #df_doc['text'] = df_doc['text'].str.strip(' ')

            # Leerzeichen einfügen
            # der Text eines Elements kann fortlaufend, ohne Leerzeichen zwischen den Zeichen/Wörtern zurückgegeben werden
            # Leerzeichen werden über Berechnung der horizontalen Distanz zwischen aufeinanderfolgenden Elementen hinzugefügt
            
            # horizontale Distanz zwischen aufeinanderfolgenden Elementen berechnen
            df_doc['distance'] = (df_doc['bbox_span'].str[0] - df_doc.shift()['bbox_span'].str[2]).round()#.drop("index", axis=1)#.apply(np.ceil)
            # Werte für distance < 0.0 oder NAN auf 0.0 setzen
            df_doc.loc[(df_doc['distance'] < 0.0) | (df_doc['distance'].isna()), 'distance'] = 0.0
        
            # Leerzeichen an Anfang aller Elemente mit distance > 1.0 anfügen
            df_doc.loc[df_doc['distance'] > 1.0, 'text'] = ' ' + df_doc.loc[df_doc['distance'] > 1.0, 'text']
            #df_doc.loc[(df_doc['distance'] > 0.0) & (df_doc['text'].str.len() > 1), 'text'] = df_doc.loc[(df_doc['distance'] > 0.0) & (df_doc['text'].str.len() > 1), 'text'] + ' '
            # Leerzeichen an Anfang des ersten Elements einer Zeile anfügen
            df_doc.loc[df_doc['span_index'] == 0, 'text'] = ' ' + df_doc.loc[df_doc['span_index'] == 0, 'text']
        
            # Blöcke entfernen, für welche die Breite geringer als die Höhe ist (vertikale Elemente)
            #df_doc = df_doc.drop(df_doc.loc[df_doc['bbox_block'].str[2] - df_doc['bbox_block'].str[0] < df_doc['bbox_block'].str[3] - df_doc['bbox_block'].str[1]].index).reset_index(drop=True)
        
            # Elemente nach Spalten "page_index" und "block_index" gruppieren und Inhalte/Werte der Spalte text mit Leerzeichen verbinden
            df_doc_group_page_index_block_index = df_doc.groupby(['page_index', 'block_index_page']).agg({'bbox_block': 'max', 'size': 'max', 'text': lambda x: ''.join(x)}).reset_index()
            #df_doc_group_page_index_block_index = df_doc.groupby(['page_index', 'block_index_page'])['text'].agg(''.join).reset_index()#block_index_page

            # Schriftgröße auf Ganzzahl aufrunden
            df_doc_group_page_index_block_index['size_int'] = df_doc_group_page_index_block_index['size'].apply(lambda x: np.ceil(x))
            #df_doc_group_page_index_block_index.assign(size_int=lambda x: np.ceil(x['size']))
            
            # drei oder mehr Zeilenumbrüche durch zwei Zeilenumbrüche ersetzen
            #df_corpus['section'] = df_corpus['section'].str.replace(r'\n{3,}', '\n\n', regex=True)
        
            # Zwei oder mehr Leerzeichen durch ein Leerzeichen ersetzen
            df_doc_group_page_index_block_index['text'] = df_doc_group_page_index_block_index['text'].str.replace(r'\s{2,}', ' ', regex=True)
            
            # '- ' durch '' ersetzen
            df_doc_group_page_index_block_index['text'] = df_doc_group_page_index_block_index['text'].str.replace('- ', '')
            
            # Leerzeichen am Anfang und Ende der Spalte text entfernen
            df_doc_group_page_index_block_index['text'] = df_doc_group_page_index_block_index['text'].str.strip(' ')#-
        
            # Anzahl Wörter
            df_doc_group_page_index_block_index['word_count'] = df_doc_group_page_index_block_index['text'].str.split(' ').str.len()
        
            ###
            # Behandlung Kopf- Fußzeilenelemente
            # Blöcke entfernen, welche auf allen oder jeder zweiten Seite vorkommen
            # Erste Seite ausschließen
            df_doc_group_page_index_block_index_page = df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index['page_index'] != 0]
        
            # Zahl+Leerzeichen am Anfang und Ende der Blöcke entfernen
            df_doc_group_page_index_block_index_page['text'] = df_doc_group_page_index_block_index_page['text'].str.replace(r'^\d+\s+', '', regex=True).str.replace(r'\s+\d+$', '', regex=True)
            
            # Blöcke entfernen, welche auf jeder Seite vorkommen
            # Bsp.
            # Dokument: 2402.03953v4
            # SSN 2379-5980 (online)\nDOI 10.5195/LEDGER.201X.X
            # LEDGER VOL X (201X) 1-5
            
            # Blöcke entfernen, welche auf jeder zweiten geraden oder ungeraden Seite vorkommen
            # Bsp.
            # 2411.00713v1
            # JONAS BLESSING, MICHAEL KUPPER, AND ALESSANDRO SGARABOTTOLO
            # RISK-BASED PRICES
            # erste Seite ausschließen
            # Cross-Join Dataframe
            
            # Alle Blöcke des Dataframe mit allen Blöcken verbinden (Cross-Join)
            df_doc_group_page_index_block_index_merge = df_doc_group_page_index_block_index_page.merge(df_doc_group_page_index_block_index_page, how='cross')
            # Blöcke auf selber Seite ausschließen
            #df_doc_group_page_index_block_index_merge = df_doc_group_page_index_block_index_merge.loc[df_doc_group_page_index_block_index_merge['page_index_x'] != df_doc_group_page_index_block_index_merge['page_index_y']]
            # Blöcke mit gleichen Werten auf unterschiedlichen Seiten selektieren
            df_doc_group_page_index_block_index_merge = df_doc_group_page_index_block_index_merge.loc[df_doc_group_page_index_block_index_merge['text_x'] == df_doc_group_page_index_block_index_merge['text_y']]
            # Dataframe nach text gruppieren, eindeutigen Zahlen der Seiten mit Vorkommen ermitteln
            df_doc_group_page_index_block_index_merge_group = df_doc_group_page_index_block_index_merge.groupby(by=['text_x'])['page_index_y'].unique().reset_index()
        
            # Alle Blöcke selektieren, für welche der text nicht auf allen oder jeder zweiten Seite vorkommt
            df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.loc[
                ~df_doc_group_page_index_block_index['text'].str.replace(r'^\d+\s+', '', regex=True).str.replace(r'\s+\d+$', '', regex=True).isin(
                    df_doc_group_page_index_block_index_merge_group.loc[(
                        # Vorkommen auf allen Seiten
                        (df_doc_group_page_index_block_index_merge_group['page_index_y'].apply(tuple) == tuple(np.arange(start=1,stop=df_doc['page_index'].nunique(), step=1).tolist())) | 
                        # Vorkommen auf jeder Zweiten Seite, beginnend ab Seite 1
                        (df_doc_group_page_index_block_index_merge_group['page_index_y'].apply(tuple) == tuple(np.arange(start=1,stop=df_doc['page_index'].nunique(), step=2).tolist())) | 
                        # Vorkommen auf jeder Zweiten Seite, beginnend ab Seite 2
                        (df_doc_group_page_index_block_index_merge_group['page_index_y'].apply(tuple) == tuple(np.arange(start=2,stop=df_doc['page_index'].nunique(), step=2).tolist()))
                    ), 'text_x']
                )
            ].reset_index(drop=True)
        
            ### 
            # Segmentgrenzen in Dokumenten mit Referenz-Daten setzen
            # Blöcke des PDF-Dokuments entsprechend der strukutrierten Referenz-Daten als Segmentgrenzen markieren
            df_doc_group_page_index_block_index['boundary'] = False
        
            # struktuierte Dokument-Daten aus Korpus laden
            try:
                file = open(f'{path}corpus/{doc_id}.json', 'r')
                data = json.load(file)
                file.close()
            except Exception as e:
                print(f"Exception occurred: {e}")
        
            # Pandas Datafrome mit Dokument-Abschnitten aus Korpus erstellen
            df_corpus = pd.DataFrame({'section':pd.Series([section['text'] for section in data['sections']], dtype=pd.StringDtype)})

            # Muster "^#+\s" (Markdown-Headings)
            re_md_heading = re.compile(r'^#+\s')

            # Inhalt der Abschnitte nach Zeilenumbrüchen spliten
            # für jeden Abschnitt aus Liste Einträge mit Vorkommen des Musters re_md_heading selektieren und Muster durch '' ersetzen
            df_corpus_first_line = df_corpus['section'].str.split('\n').apply(lambda x: [re_md_heading.sub('', s) for s in x if re_md_heading.match(s)]).explode()
        
            #re_formula = re.compile(r'\$\\.*\$')
            re_ws = re.compile(r'[^\w\s]')
            # ersetzen ^[A-Z]\.\s[0-9]+ durch ^[A-Z]\.)([0-9]+ (Leerzeichen entfernen)
            # Bsp.
            # Dokument: 2410.08709v3
            # Referenz: A. 1 Surrogate of distillation loss
            # Ziel: A.1 Surrogate of distillation loss
            re_wpd = re.compile(r'(^[A-Z]\.)\s([0-9]+)')
        
            # über alle Abschnitte aus Korpus iterieren
            for boundary in df_corpus_first_line:
                # Übereinstimmung zwischen Abschnitten aus Korpus und PDF-Dokument ermitteln
                # Zeile in Dataframe für PDF-Dokument als Segmentgrenze markieren
                #print(f'!{boundary.strip()}!')
                #print(f'!{re_ws.sub(r'', re_wpd.sub(r'\1\2', boundary)).strip()}!')
                #print(f'!{df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index['text'].str.replace(r'[^\w\s]', '', regex=True).str.strip().str.contains(r'^' + re.escape(re_ws.sub('', re_wpd.sub(r'\1\2', boundary)).strip()), regex=True, case=False), 'text']}!')#.apply(lambda x: x.encode("ascii", "ignore").decode('ascii'))#.apply(lambda x: re.escape(x))
                # Alles was nicht Buchstabe, Zahl, Unterstrich oder Leerzeichen ist ersetzen durch ''
                # Bsp.:
                # Dokument: 2411.00713v1
                # 3. Agent’s preferences and indifference pricing
                # Referenz: ' - Apostrophe (Ascii)
                # Ziel: ’ - Right Single Quotation Mark (Unicode)
                df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index['text'].str.replace(r'[^\w\s]', '', regex=True).str.strip().str.contains(r'^' + re.escape(re_ws.sub('', re_wpd.sub(r'\1\2', boundary)).strip()), regex=True, case=False), 'boundary'] = True#re_formula.sub('', boundary).strip(), regex=True)
        
            # Blöcke mit erkannten Grenzen aus Referenz selektieren, welche Suchmuster für Grenzen entsprechen
            df_doc_group_page_index_block_index_boundary = (
                (
                    (df_doc_group_page_index_block_index['text'].str.contains(r'^((\d+|[A-Z]{1,3})(\.|\s))+', regex=True, case=False)) | 
                    (df_doc_group_page_index_block_index['text'].str.contains(r'^(Abstract|Introduction|Contents|Appendix|References|Conclusion|Acknowledgment|Acknowledgement)', regex=True, case=False))#$
                ) & 
                (df_doc_group_page_index_block_index['boundary'] == True)
            )
        
            # Blöcke mit erkannten Grenzen aus Referenz selektieren, welche Suchmuster für Grenzen nicht entsprechen
            df_doc_group_page_index_block_index_boundary_neg = (
                ~(
                    (df_doc_group_page_index_block_index['text'].str.contains(r'^((\d+|[A-Z]{1,3})(\.|\s))+', regex=True, case=False)) | 
                    (df_doc_group_page_index_block_index['text'].str.contains(r'^(Abstract|Introduction|Contents|Appendix|References|Conclusion|Acknowledgment|Acknowledgement)', regex=True, case=False))
                ) & 
                (df_doc_group_page_index_block_index['boundary'] == True)
            )
        
            # Markierung als Grenze für Blöcke mit erkannten Grenzen aus Referenz, welche Suchmuster für Grenzen nicht entsprechen, auf False setzen
            df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index_boundary_neg, 'boundary'] = False

            # Dokumente ausschließen
            # Dokumente für die keine bzw. nur wenige Grenzen erkannt wurden ausschließen
            # Wenn nicht (
            # Anzahl erkannter/markierter Grenzen größer 4
            # und Anzahl erkannter Grenzen größer Anzahl nicht erkannter Grenzen
            # )
            # weitere Verarbeitung abbrechen
            if not (
                df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index_boundary].shape[0] > 4 and 
                df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index_boundary].shape[0] > df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index_boundary_neg].shape[0]
            ):
                #continue
                #return None
                clean = False
            
            #print(f'{df_corpus_first_line.shape[0] - df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index['boundary'] == True].shape[0]} - {df_corpus_first_line.shape[0]} - {df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index['boundary'] == True].shape[0]}')
        
            # Cleanup
            # arxiv-Label entfernen
            df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.drop(df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index['text'].str.strip(' ').str.contains(r'^arxiv\:\d{4}\.\d+v\d+', case=False, regex=True)].index).reset_index(drop=True)
            # ersten Block entfernen (Titel)
            #df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.drop(0).reset_index(drop=True)
            
            # Seitennummerierungen behandeln
            # Alle Blöcke selektieren, für welche der Wert der Spalte text keine Zahl ist
            #df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.loc[~df_doc_group_page_index_block_index['text'].str.strip().str.match(r'^\d+$')].reset_index(drop=True)#(df_doc_group_page_index_block_index['block_index_page'] > 0) & (
            
            # Ersten Block einer Seite, welcher eine Zahl enthält, entfernen
            df_doc_group_page_index_block_index_first = df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index.groupby(['page_index'])['block_index_page'].idxmin().reset_index()['block_index_page']]
            df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.drop(df_doc_group_page_index_block_index_first.loc[df_doc_group_page_index_block_index_first['text'].str.strip().str.match(r'^\d+$')].index).reset_index(drop=True)
        
            # Letzten Block einer Seite, welcher eine Zahl enthält, entfernen
            df_doc_group_page_index_block_index_last = df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index.groupby(['page_index'])['block_index_page'].idxmax().reset_index()['block_index_page']]
            df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.drop(df_doc_group_page_index_block_index_last.loc[df_doc_group_page_index_block_index_last['text'].str.strip().str.match(r'^\d+$')].index).reset_index(drop=True)
        
            # block_index_page für erstes Element jeder Seite auf 0 setzen
            #df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index.groupby(by=['page_index'])['block_index_page'].head(1).index, 'block_index_page'] = 0#df_doc_group_page_index_block_index_page
            df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index.groupby(by=['page_index'])['block_index_page'].idxmin().reset_index(drop=True).values, 'block_index_page'] = 0#df_doc_group_page_index_block_index_page
        
            # Behandlung � (Unicode replacement character \uFFFD, z.B. Summen- oder Produktzeichen), führt zu Zeilenumbruch/neuem Block
            # Iterativer Prozess bei mehreren aufeinander folgenden Blöcken, welche zusammenhängen
            # Reverse order (von unten nach oben), text des aktuellen Blocks an vorherigen Block anfügen
            df_doc_group_page_index_block_index_max = df_doc_group_page_index_block_index.index.max()
            for i in reversed(df_doc_group_page_index_block_index.loc[df_doc_group_page_index_block_index['text'].str.contains(u'\uFFFD$')].index.tolist()):
                if i < df_doc_group_page_index_block_index_max:
                    df_doc_group_page_index_block_index.loc[i, 'text']+= ' ' + df_doc_group_page_index_block_index.loc[i + 1, 'text']
                    df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.drop(i + 1)
        
            df_doc_group_page_index_block_index = df_doc_group_page_index_block_index.reset_index(drop=True)
        
            #print(df_doc_group_page_index_block_index['text'].agg('\n\n'.join))

            # Dataframe als parquet speichern
            #df_doc_group_page_index_block_index.astype({'block_index':'int64', 'page_index': 'int64'}).to_parquet(filename, compression='gzip')
            df_doc_group_page_index_block_index_table = pa.Table.from_pandas(df_doc_group_page_index_block_index)
            pq.write_table(df_doc_group_page_index_block_index_table, filename)

        return clean, df_doc_group_page_index_block_index
    """