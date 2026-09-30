#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

"""
BERTScore

Implementierung der BERTScore-Metrik
"""
class BERTScore:
    def __init__(self,
                 #model_name="google-bert/bert-base-uncased",
                 model_name="google-bert/bert-large-uncased",
                 device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    """
    _encode

    Embeddings zu Eingabe erzeugen
    """
    @torch.no_grad()
    def _encode(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)

        return embeddings[inputs["attention_mask"].squeeze(0).bool()]

    """
    def encode_decode(self, text):
        inputs = self.tokenizer(
            text,
            #return_tensors="pt",
            truncation=True,
            max_length=512
        )

        return self.tokenizer.convert_ids_to_tokens(inputs.input_ids)#, skip_special_tokens=True
    """

    """
    _normalize

    Werte eines Embedding-Vektors normalisieren
    """
    @staticmethod
    def _normalize(embeddings):
        return F.normalize(embeddings, p=2, dim=-1)

    """
    score

    Score (Precision+Recall) für Kandidaten- und Referenz-Text berechnen

    Embedding-Vektoren normalisieren und Matrix-Multiplikation durchführen
    """
    def score(self, candidate, reference):

        cand = self._normalize(self._encode(candidate))
        ref = self._normalize(self._encode(reference))

        similarity = torch.matmul(cand, ref.T)

        precision = similarity.max(dim=1).values.mean().item()
        recall = similarity.max(dim=0).values.mean().item()

        return precision, recall

    """
    similarity

    Vector-Similarity für Kandidaten- und Referenz-Text berechnen
    
    Embedding-Vektoren normalisieren und Matrix-Multiplikation durchführen
    """
    def similarity(self, candidate, reference):

        cand = self._normalize(self._encode(candidate))
        ref = self._normalize(self._encode(reference))

        return torch.matmul(cand, ref.T)