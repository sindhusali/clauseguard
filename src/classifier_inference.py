"""
ClauseGuard — Week 2, Part A
Load the LoRA adapter trained in Week 1 and use it to classify clause text
at inference time. This is imported by rag_agent.py — you shouldn't need to
run this file directly, but it has a __main__ block for a quick sanity check.
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE_MODEL_NAME = "roberta-base"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MAX_LENGTH = 256

_tokenizer = None
_model = None
_id_to_label = None


def _load_once():
    """Lazy-load the model once per process — classifying many clauses
    shouldn't reload weights from disk every time."""
    global _tokenizer, _model, _id_to_label

    if _model is not None:
        return

    with open(os.path.join(ADAPTER_DIR, "label_mapping.json")) as f:
        raw_mapping = json.load(f)
    _id_to_label = {int(k): v for k, v in raw_mapping.items()}
    num_labels = len(_id_to_label)

    _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=num_labels
    )
    _model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    _model.eval()


def classify_clause(text: str) -> dict:
    """Returns {'label': str, 'confidence': float} for a single clause."""
    _load_once()

    inputs = _tokenizer(
        text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    top_id = int(torch.argmax(probs).item())
    return {
        "label": _id_to_label[top_id],
        "confidence": round(float(probs[top_id]), 4),
    }


def classify_clauses_batch(texts: list[str]) -> list[dict]:
    """Same as classify_clause but for a list — used when tagging every
    retrieved chunk in one pass instead of one-by-one."""
    return [classify_clause(t) for t in texts]


if __name__ == "__main__":
    test_clauses = [
        "Either party may terminate this Agreement upon 30 days written notice.",
        "This Agreement shall be governed by the laws of the State of Delaware.",
        "The parties acknowledge that time is of the essence.",
    ]
    for clause in test_clauses:
        result = classify_clause(clause)
        print(f"[{result['label']} ({result['confidence']:.2%})] {clause}")
