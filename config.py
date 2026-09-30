# import os

dataset =  {
    'path': 'vectara_open_ragbench/open_ragbench/pdf/arxiv/'
}

redis_client = {
    'host': '192.168.5.133',
    'port': 6379,
    'password': '3cAC7q4dp'
}

models = {
    'generation': [
        'mistralai/Ministral-8B-Instruct-2410',
        'meta-llama/Llama-3.1-8B-Instruct'
    ],
    'embedding': [
        'nvidia/llama-nemotron-embed-1b-v2',
        'intfloat/multilingual-e5-large'
    ],
    'judge': [
        'unsloth/gemma-4-31B-it-qat-GGUF:UD-Q4_K_XL',
        'unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M'
    ]
}

openai_client = {
    'api_key': 'oA#-c84mE',#os.environ.get("OPENAI_API_KEY")
    'chat': {
        'base_url': 'http://192.168.5.133:8001/v1'
    },
    'embedding': {
        'base_url': 'http://192.168.5.133:8002/v1'
    }
}