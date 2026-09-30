# Setup


Anweisungen zur Einrichtung der Umgebungen für die Durchführung der verschiedenen Untersuchungen und Ausführung/Bereitstellung der benötigten Komponenten wie den Sprach- und Embedding-Modellen

## Datensatz vectara/open_ragbench


Hugging-Face Dataset via git clonen


git clone git@hf.co:datasets/vectara/open_ragbench


## Datensatz pdfqa/pdfQA-Annotations


Hugging-Face Dataset via git clonen


git clone git@hf.co:datasets/pdfqa/pdfQA-Annotations


## Sprachmodelle


Umgebung zur Bereitstellung der LLMs über Rest-API mit vLLM


### Python venv


python3 -m venv /home/chris/venv/vllm
source venv/vllm/bin/activate
pip install vllm>=0.16.0


### vllm serve


#### mistralai/Ministral-8B-Instruct-2410


vllm serve mistralai/Ministral-8B-Instruct-2410 \
--tokenizer_mode mistral \
--config_format mistral \
--load\_format mistral \
--host 192.168.5.133 \
--port 8001 \
--gpu-memory-utilization 0.8 \
--api-key "oA#-c84mE" \
--max-model-len 19072


#### meta-llama/Llama-3.1-8B-Instruct


vllm serve meta-llama/Llama-3.1-8B-Instruct \
--host 192.168.5.133 \
--port 8001 \
--gpu-memory-utilization 0.8 \
--api-key "oA#-c84mE" \
--max-model-len 21504


## Embedding-Model


Umgebung zur Bereitstellung der Embedding-Modelle über Rest-API mit vLLM


### Python venv


python3 -m venv /home/chris/venv/vllm
source venv/vllm/bin/activate


pip install vllm>=0.16.0


### vllm serve


#### nvidia/llama-nemotron-embed-1b-v2


vllm serve nvidia/llama-nemotron-embed-1b-v2 \
--trust-remote-code \
--host 192.168.5.133 \
--port 8002 \
--gpu-memory-utilization 0.1 \
--api-key "oA#-c84mE"


#### intfloat/multilingual-e5-large


vllm serve intfloat/multilingual-e5-large \
--dtype bfloat16 \
--host 192.168.5.133 \
--port 8002 \
--gpu-memory-utilization 0.1 \
--api-key "oA#-c84mE"


## Redis stack server


Anweisungen zur Ausführung der Redis In-Memory DB als Vector-Store als Docker-Container
https://hub.docker.com/r/redis/redis-stack-server
Docker-Container als root ausführen
su


### Dateien kopieren (Initial)


mkdir /opt/docker-redis-stack/redis-stack-data
docker run -d -p 6379:6379 -e REDIS\_ARGS="--requirepass 3cAC7q4dp" --restart always --name redis-stack redis/redis-stack-server:latest
docker cp redis-stack:/data /opt/docker-redis-stack/redis-stack-data


### Server starten


docker run -d -p 6379:6379 -v "./redis-stack-data:/data" -e REDIS\_ARGS="--requirepass 3cAC7q4dp" --restart always --name redis-stack redis/redis-stack-server:latest


### Container logs


docker exec -it redis-stack /bin/bash
docker logs -f -n 10 redis-stack


### Redis-CLI


https://redis.io/docs/latest/commands/
docker exec -it redis-stack redis-cli


AUTH 3cAC7q4dp
SELECT 0
KEYS \*
GET document:


Indizes löschen


FLUSHALL SYNC


Index erzeugen


FT.CREATE idx:document ON JSON PREFIX 1 document: SCHEMA $.document AS document TEXT NOSTEM $.page AS page NUMERIC $.content AS content TEXT $.content\_embeddings AS vector VECTOR FLAT 6 TYPE FLOAT16 DIM 2048 DISTANCE\_METRIC COSINE


FT.\_LIST
FT.INFO idx:document


## Redis Index


Umgebung für Indexzierung der Dokumente/Kontext-Informationen im Redis Vector-Store


### Python venv


python3 -m venv /home/chris/venv/redis
source venv/redis/bin/activate
pip install notebook ipywidgets pandas redis pyarrow openai torch pyarrow datasets transformers


## Masked-Language-Modell


Umgebung für Fine-Tuning des Masked-Language-Modells


### Python venv


python3 -m venv /home/chris/venv/mlm
source venv/mlm/bin/activate
pip install notebook ipywidgets pandas transformers datasets evaluate accelerate torch scikit-learn pymupdf plotly openai


## Datenanalyse+Preprocessing


Umgebung für Durchführung von Datensatz-Analysen und Preprocessing


### conda Environment (Windows)


conda create --name dski python=3.11.2
conda activate dski
pip install notebook ipywidgets pandas pymupdf openai plotly regex pyarrow redis statsmodels datasets scipy


## Segmentierung pymupdf4llm+docling


Umgebung zur Durchführung der Segmentierung der Dokumente mit den Paketen pymupdf4llm+docling


### Python venv


python3 -m venv /home/chris/venv/package
source venv/package/bin/activate
pip install notebook ipywidgets pandas openai plotly pymupdf4llm docling regex pyarrow


###  Tesseract OCR


https://github.com/tesseract-ocr/tessdata
Dateien deu.traineddata etc. in lokales Verzeichnis kopieren und Umgebungsvariable "%env TESSDATA\_PREFIX=" für Pfad in Jupyter-Notebook setzen


### ibm-granite docling OCR/Document conversion


vllm serve ibm-granite/granite-docling-258M  
--host 192.168.5.133  
--port 8003  
--max-num-seqs 512  
--max-num-batched-tokens 8192  
--enable-chunked-prefill  
--gpu-memory-utilization 0.2


## Segmentierung Model (Random Forest etc.)


Umgebung für Training der Klassifikationsmodelle zur Segmentierung der Dokumente mit baumbasierten Verfahren wie Random Forest etc.


### Python venv


python3 -m venv /home/chris/venv/model
source venv/model/bin/activate
pip install notebook ipywidgets pandas regex pyarrow pymupdf plotly openai scikit-learn scikit-optimize lightgbm xgboost


## Model (Neural Networks)


Umgebung für Training der Klassifikationsmodelle zur Segmentierung der Dokumente mit Neuronalen Netzwerken


### Python venv


python3 -m venv /home/chris/venv/model\_nn
source venv/model\_nn/bin/activate
pip install notebook ipywidgets pandas regex pyarrow pymupdf plotly openai torch keras scikit-learn


## BERTScore


Umgebung zur Messung der Ähnlichkeiten zwischen Modell- und Referenz-Antworten mit Hilfe des BERTScore


### Python venv


python3 -m venv /home/chris/venv/bert
source venv/bert/bin/activate
pip install notebook ipywidgets numpy plotly torch transformers scipy


## AutoRubric


https://autorubric.org/
Umgebung zur Durchführung der automatisierten Evaluierung der Modell-Antworten mit AutoRubric


## conda Environment (Windows)


conda create --name autorubric
conda activate autorubric
pip install notebook ipywidgets pandas pyarrow plotly scikit-learn autorubric

