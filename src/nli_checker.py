"""
ClauseGuard — Week 3, Part A
Hallucination/faithfulness checker using a standard NLI model trained on
MNLI + FEVER + ANLI. FEVER in particular is a fact-verification dataset
(claim vs. evidence passage) — much closer to our actual task (does this
generated sentence hold up against this contract clause) than a model
trained only on short SNLI-style sentences.

This uses plain `transformers` AutoModelForSequenceClassification — no
custom repo code, no trust_remote_code flag, no CrossEncoder wrapper. Two
earlier versions of this file used models that turned out to be fragile
(one had a domain mismatch, the other had custom code that broke against a
newer transformers version) — this one is intentionally boring and stable.

Model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (free, runs on CPU,
~370MB download the first time you use it).
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MAX_LENGTH = 512

_tokenizer = None
_model = None
_id_to_label = None


def _load_once():
    global _tokenizer, _model, _id_to_label
    if _model is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.eval()
    # Read the label mapping straight from the model's own config instead of
    # hardcoding an assumed order — this is what broke us once already.
    _id_to_label = _model.config.id2label


def check_entailment(premise: str, hypothesis: str) -> dict:
    """Does `premise` support `hypothesis`?
    Returns {'label': 'entailment'|'neutral'|'contradiction', 'scores': {...}}
    """
    _load_once()
    inputs = _tokenizer(premise, hypothesis, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)

    label_scores = {_id_to_label[i].lower(): float(probs[i]) for i in range(len(probs))}
    top_label = max(label_scores, key=label_scores.get)
    return {"label": top_label, "scores": label_scores}


def check_entailment_batch(pairs: list[tuple[str, str]]) -> list[dict]:
    """Batched version — much faster than calling check_entailment in a loop."""
    _load_once()
    premises = [p[0] for p in pairs]
    hypotheses = [p[1] for p in pairs]

    inputs = _tokenizer(
        premises, hypotheses, truncation=True, max_length=MAX_LENGTH,
        padding=True, return_tensors="pt",
    )
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    results = []
    for row in probs:
        label_scores = {_id_to_label[i].lower(): float(row[i]) for i in range(len(row))}
        top_label = max(label_scores, key=label_scores.get)
        results.append({"label": top_label, "scores": label_scores})
    return results


def is_supported_claim(clause_text: str, claim_sentence: str) -> bool:
    """True = the claim is grounded in the clause (entailment).
    False = contradiction or neutral (unsupported / possibly hallucinated)."""
    result = check_entailment(clause_text, claim_sentence)
    return result["label"] == "entailment"


if __name__ == "__main__":
    premise = "Either party may terminate this Agreement upon 30 days written notice."

    print("Should be entailment:")
    print(check_entailment(premise, "The agreement can be ended with 30 days notice."))

    print("\nShould be contradiction:")
    print(check_entailment(premise, "Neither party may ever terminate this agreement."))

    print("\nShould be neutral:")
    print(check_entailment(premise, "The agreement requires arbitration for disputes."))
