import config

import asyncio

import os
import json

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

import plotly.express as px
import plotly.graph_objects as go

from autorubric import Rubric, Criterion, LLMConfig, DataItem, RubricDataset, evaluate, EvalRunner, EvalConfig, EvalResult
from autorubric.graders import CriterionGrader
from autorubric.meta import evaluate_rubric_standalone

#%env APPHUBAI_API_KEY=FycFTLpaJr7ILwMb78i63bhQOMrOUrr2
#%env LOCAL_API_KEY=c017TkMCsbF

###
# Daten laden

# Answers
answers = pd.read_json(f'{config.dataset["path"]}answers.json', orient='index', dtype_backend='numpy_nullable').rename(columns={0: "answer"})
# Queries
queries = pd.read_json(f'{config.dataset["path"]}queries.json', orient='index', dtype_backend='numpy_nullable')
# Relations
qrels = pd.read_json(f'{config.dataset["path"]}qrels.json', orient='index', dtype_backend='numpy_nullable')

# PDF URLs
#pdf_urls = pd.read_json(f'{config.dataset["path"]}pdf_urls.json', orient='index', dtype_backend='numpy_nullable').rename(columns={0: "pdf_urls"})

pdf_urls_clean = pd.read_json(f'{config.dataset["path"]}pdf_urls_clean.json', orient='index', dtype_backend='numpy_nullable')#.rename(columns={0: "pdf_urls"})
pdf_urls = pdf_urls_clean.loc[pdf_urls_clean.values]

# Fragen selektieren

# Fragen und Antworten über Index joinen
queries_answers = queries.merge(answers, how='inner', left_index=True, right_index=True)

# Anzahl Wörter der Fragen 
queries_answers['answer_replace'] = queries_answers['answer'].str.replace(r'\p{P}+', '', regex=True).str.replace(r'[\n\t\s]+', ' ', regex=True)
queries_answers['answer_word_count'] = queries_answers['answer_replace'].str.split(' ').str.len()

# Fragen mit source=text|text-table und answer_word_count>=8
queries_text_text_table = queries_answers.loc[((queries_answers['source'] == 'text') | (queries_answers['source'] == 'text-table')) & (queries_answers['answer_word_count'] >= 8)]#.sample(n=samples, axis=0, random_state=42)

# 1/4 der Fragen selektieren
queries_text_text_table = queries_text_text_table.sample(n=round(queries_text_text_table.shape[0] / 4), axis=0, random_state=42)

# Fragen ausschließen, welche sich auf ausgeschlossene Dokumente beziehen

# Beziehung zwischen Frage und Dokument über qrels
# qrels/Dokumente für alle Fragen wählen
qrels_select = qrels.loc[queries_text_text_table.index]
# qrels ausschließen, für die Dokumente (doc_id) nicht in pdf_urls (index)
qrels_select = qrels.loc[qrels_select.loc[~qrels_select['doc_id'].isin(pdf_urls.index)].index]

queries_text_text_table_select = queries_text_text_table.loc[~queries_text_text_table.index.isin(qrels_select.index)]

###
# Rubric Kriterien definieren

rubric = Rubric.from_dict([
    {'weight': 5.0, 'requirement': 'Is the question answered factually correctly based on the reference answer?'},
    {'weight': 5.0, 'requirement': 'Compared to the reference answer, the model answer contains no internal contradictions.'},
    {
        'weight': 10.0,
        'requirement': 'Does the model answer cover all factual information relative to the reference answer? The absence of a single piece of information is considered a failure to meet the criterion.',
        'scale_type': 'ordinal',
        'options': [
            {'label': 'The model answer does not contain any of the information.', 'value': 0.0},
            {'label': 'The model answer contains at least one piece of information.', 'value': 0.5},
            {'label': 'The model answer contains all the information.', 'value': 1.0},
        ]
    },
])

###
# Evaluierung

# gewähltes Generator-LLM
model_select = 1
model_name = config.models['generation'][model_select][config.models['generation'][model_select].index('/') + 1:]

#retrieval = f'block-size512_overlap0.5'
retrieval = 'classification_overlap0.5'
#retrieval = 'baseline'

# Modell-Antworten laden
answers_model = pq.read_table(f'{config.dataset["path"]}answers_model_{model_name}_retrieval_{retrieval}.parquet').to_pandas()

###
# Datensatz erstellen
###
dataset = RubricDataset(
    rubric=rubric
)

# Einträge zum Datensatz hinzufügen
for i in range(queries_text_text_table_select.shape[0]):
    dataset.add_item(
        description=f'answer {queries_text_text_table_select.iloc[i].name}',
        submission=answers_model.loc[queries_text_text_table_select.iloc[i].name]['answer_model'], 
        reference_submission=queries_text_text_table_select.iloc[i]['answer'],
        prompt=queries_text_text_table_select.iloc[i]['query']
    )

# gewähltes Judge-LLM
judge_model_select = 0
judge_model_name = config.models['judge'][judge_model_select][config.models['judge'][judge_model_select].index('/') + 1:config.models['judge'][judge_model_select].index(':')]

# LLM-Judge konfigurieren
llm_config = LLMConfig(
    #model=f'local/{config.models['generation'][1 - model_select]}',
    model=f'apphubai/{config.models['judge'][judge_model_select]}',
    temperature=0.1,
    #cache_enabled=True,
    max_parallel_requests=1,
    retry_min_wait=2,
    retry_max_wait=5
)

grader = CriterionGrader(
    llm_config=llm_config,
    #aggregation='weighted',#unanimous#weighted#any
    #system_prompt="Users posed questions on various topics and areas related to scientific publications, and a large language model generated answers to them. The task involves evaluating the language model's output. To this end, various binary and ordinal criteria were defined to assess the quality of aspects such as the correctness, consistency und completeness of the generated answers. The ordinal criteria include a set of options indicating the degree of alignment; when evaluating a specific aspect, the option that best describes it must be selected.",
)

# Datensatz laden
#dataset = RubricDataset.from_file('./rag-eval/dataset.json')

###
# Evaluierung
###
"""result = await evaluate(dataset, grader, show_progress=True)

print(f"Evaluated {result.successful_items}/{result.total_items}")
print(f"Throughput: {result.timing_stats.items_per_second:.2f} items/s")
print(f"Total cost: ${result.total_completion_cost or 0:.4f}")"""

# Evaluation-Runner Konfiguration
eval_config = EvalConfig(
    experiment_name=f'{model_name}_retrieval_{retrieval}_{judge_model_name}',
    experiments_dir='./rag-eval',
    max_concurrent_items=1,
    show_progress=True,
)

runner = EvalRunner(dataset=dataset, grader=grader, config=eval_config)
#result = await runner.run()

# Resume
#runner = EvalRunner(dataset=dataset, grader=grader, config=config)
#result = await runner.run()

# Load results later
#result = EvalResult.from_experiment(f'./rag-eval/{model_name}_retrieval_{retrieval}_{judge_model_name}')

async def main():
    result = await runner.run()
    return result

result = asyncio.run(main())