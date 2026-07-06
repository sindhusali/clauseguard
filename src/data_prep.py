"""
ClauseGuard — Step 1
Turn raw CUAD into a clause-level classification dataset.
Run this in Google Colab (needs GPU-free but internet access to Hugging Face).
"""

import re
import random
import pandas as pd
from datasets import load_dataset

RANDOM_SEED = 42
MAX_NEGATIVES_PER_CONTRACT = 3
MIN_CLAUSE_CHARS = 30
MAX_CLAUSE_CHARS = 1200

random.seed(RANDOM_SEED)
QUESTION_CATEGORY_RE = re.compile(r'related to\s+"([^"]+)"')


def extract_category(question: str) -> str:
    match = QUESTION_CATEGORY_RE.search(question)
    return match.group(1).strip() if match else "UNKNOWN"


def sample_negative_paragraphs(context, positive_spans, n):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", context) if p.strip()]
    candidates = [
        p for p in paragraphs
        if MIN_CLAUSE_CHARS <= len(p) <= MAX_CLAUSE_CHARS
        and not any(p in span or span in p for span in positive_spans)
    ]
    random.shuffle(candidates)
    return candidates[:n]


def main():
    print("Downloading CUAD (theatticusproject/cuad-qa) from Hugging Face Hub...")
    dataset = datasets.load_dataset("theatticusproject/cuad-qa")
    train_split = dataset["train"]

    rows = []
    seen_contexts_for_negatives = {}

    for example in train_split:
        question = example["question"]
        context = example["context"]
        category = extract_category(question)
        answer_texts = example["answers"]["text"]

        context_key = hash(context)
        if context_key not in seen_contexts_for_negatives:
            seen_contexts_for_negatives[context_key] = {"context": context, "positives": []}

        for span in answer_texts:
            span = span.strip()
            if MIN_CLAUSE_CHARS <= len(span) <= MAX_CLAUSE_CHARS:
                rows.append({"text": span, "label": category})
                seen_contexts_for_negatives[context_key]["positives"].append(span)

    print(f"Collected {len(rows)} positive examples across "
          f"{len(set(r['label'] for r in rows))} categories.")

    negative_rows = []
    for entry in seen_contexts_for_negatives.values():
        negs = sample_negative_paragraphs(entry["context"], entry["positives"], MAX_NEGATIVES_PER_CONTRACT)
        negative_rows.extend({"text": n, "label": "NONE"} for n in negs)

    print(f"Sampled {len(negative_rows)} NONE-class negatives.")

    all_rows = rows + negative_rows
    random.shuffle(all_rows)
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["text"]).reset_index(drop=True)

    TOP_N_CATEGORIES = 15
    top_categories = df[df.label != "NONE"]["label"].value_counts().head(TOP_N_CATEGORIES).index.tolist()
    df = df[df.label.isin(top_categories + ["NONE"])].reset_index(drop=True)

    print("\nFinal class distribution:")
    print(df["label"].value_counts())

    df.to_csv("../data/clauseguard_classification_dataset.csv", index=False)
    print(f"\nSaved {len(df)} rows to ../data/clauseguard_classification_dataset.csv")


if __name__ == "__main__":
    main()