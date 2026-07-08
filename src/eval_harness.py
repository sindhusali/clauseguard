"""
ClauseGuard — Week 3, Part B
The eval harness. This is what turns "I noticed it behaves well on a couple
of examples" into a real, reportable number.

For every question in eval_questions.json, this runs the RAG agent TWICE —
once with the strict citation-or-decline system prompt, once with a loose
baseline prompt — and scores both runs using the NLI checker from
nli_checker.py. The result is a before/after comparison you can actually
quote and defend.

Scoring logic:
- For "answerable" questions: split the answer into sentences, and for each
  one, check whether ANY of the retrieved supporting clauses entail it (NLI).
  If the best label across all clauses is "contradiction" or "neutral", that
  sentence is flagged as unsupported (a hallucination candidate).
- For "unanswerable" questions: check whether the answer actually declined
  to answer (keyword heuristic) rather than inventing a clause.

Run this after Week 2's rag_agent.py is working and you've indexed nda_001.
"""

import json
import re
import time
import statistics
from datetime import datetime

from rag_agent import answer_question
from nli_checker import check_entailment_batch

EVAL_QUESTIONS_PATH = "eval_questions.json"
RESULTS_PATH_TEMPLATE = "eval_results_{mode}.json"

# Simple, transparent heuristic for "did the model decline instead of
# guessing" — not a hidden model, just plain keyword matching so you can
# explain exactly how this works in an interview.
DECLINE_PHRASES = [
    "does not mention", "does not address", "not addressed", "not mentioned",
    "no mention of", "does not specify", "does not contain", "not specified",
    "cannot determine", "not explicitly stated", "not explicitly state",
    "no information", "does not include", "do not mention", "unable to determine",
    "not provided in", "does not appear to",
]


def split_into_sentences(text: str) -> list[str]:
    # Good enough for short LLM answers — not meant for arbitrary text.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 8]


def answer_declines(answer_text: str) -> bool:
    lower = answer_text.lower()
    return any(phrase in lower for phrase in DECLINE_PHRASES)


def strip_citation_markers(sentence: str) -> str:
    """Remove '(Page N)' style citations before NLI scoring. The citation
    text itself isn't a factual claim from the clause — leaving it in
    confuses the NLI model into treating '(Page 2)' as unsupported content,
    which unfairly penalizes exactly the citation behavior we want."""
    return re.sub(r"\(Page\s*\d+\)", "", sentence).strip()


def score_answerable_question(answer_text: str, supporting_clauses: list[dict]) -> dict:
    """For questions expected to be answerable: check whether each factual
    sentence in the answer is entailed by the combined retrieved context.

    Two deliberate exclusions from scoring, both correcting for false
    positives found during real runs of this harness:
    1. Sentences that hedge/decline ("does not clearly define...") are
       meta-statements about missing information, not factual claims a
       clause can "entail" — scoring them as hallucinations penalizes
       exactly the honest behavior we want.
    2. Entailment is checked against the COMBINED text of all retrieved
       clauses (not one clause at a time), since a legitimate answer often
       synthesizes multiple clauses — no single clause should be expected
       to "entail" a sentence that correctly draws on several.
    """
    sentences = split_into_sentences(answer_text)
    if not sentences or not supporting_clauses:
        return {"total_sentences": 0, "supported": 0, "hallucinated": 0, "detail": []}

    combined_context = "\n\n".join(c["text"] for c in supporting_clauses)

    detail = []
    supported_count = 0
    hallucinated_count = 0
    skipped_hedges = 0

    factual_sentences = []
    for sentence in sentences:
        cleaned_sentence = strip_citation_markers(sentence)
        if len(cleaned_sentence) < 8:
            continue
        if answer_declines(cleaned_sentence):
            skipped_hedges += 1
            continue  # epistemic hedge, not a factual claim — don't score it
        factual_sentences.append(cleaned_sentence)

    if factual_sentences:
        pairs = [(combined_context, s) for s in factual_sentences]
        results = check_entailment_batch(pairs)
        for sentence, result in zip(factual_sentences, results):
            is_supported = result["label"] == "entailment"
            if is_supported:
                supported_count += 1
            else:
                hallucinated_count += 1
            detail.append({
                "sentence": sentence,
                "best_label": result["label"],
                "entailment_score": round(result["scores"]["entailment"], 3),
            })

    return {
        "total_sentences": len(detail),
        "supported": supported_count,
        "hallucinated": hallucinated_count,
        "skipped_hedges": skipped_hedges,
        "detail": detail,
    }


def run_eval(strict: bool) -> dict:
    with open(EVAL_QUESTIONS_PATH) as f:
        questions = json.load(f)

    mode_label = "strict (citation-or-decline)" if strict else "loose (baseline)"
    print(f"\n{'='*70}\nRunning eval — {mode_label}\n{'='*70}")

    per_question_results = []
    total_sentences = 0
    total_hallucinated = 0
    unanswerable_total = 0
    unanswerable_correct_declines = 0

    for q in questions:
        print(f"  [{q['id']}] {q['question'][:70]}...")
        result = answer_question(q["question"], q["contract_id"], strict=strict)
        time.sleep(1)  # be polite to the free-tier rate limit

        entry = {
            "id": q["id"],
            "question": q["question"],
            "expected": q["expected"],
            "answer": result["answer"],
        }

        if q["expected"] == "answerable":
            score = score_answerable_question(result["answer"], result["supporting_clauses"])
            entry.update(score)
            total_sentences += score["total_sentences"]
            total_hallucinated += score["hallucinated"]

        elif q["expected"] == "unanswerable":
            declined = answer_declines(result["answer"])
            entry["correctly_declined"] = declined
            unanswerable_total += 1
            if declined:
                unanswerable_correct_declines += 1

        per_question_results.append(entry)

    hallucination_rate = (total_hallucinated / total_sentences) if total_sentences else None
    decline_accuracy = (unanswerable_correct_declines / unanswerable_total) if unanswerable_total else None

    summary = {
        "mode": "strict" if strict else "loose",
        "run_at": datetime.now().isoformat(),
        "total_answerable_sentences_checked": total_sentences,
        "hallucinated_sentences": total_hallucinated,
        "hallucination_rate": round(hallucination_rate, 4) if hallucination_rate is not None else None,
        "unanswerable_questions": unanswerable_total,
        "correct_declines": unanswerable_correct_declines,
        "decline_accuracy": round(decline_accuracy, 4) if decline_accuracy is not None else None,
        "per_question": per_question_results,
    }

    out_path = RESULTS_PATH_TEMPLATE.format(mode=summary["mode"])
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved detailed results to {out_path}")
    return summary


def print_comparison(strict_summary: dict, loose_summary: dict):
    print(f"\n{'='*70}\nBEFORE / AFTER COMPARISON\n{'='*70}")
    print(f"{'Metric':<35}{'Loose (baseline)':<20}{'Strict (ours)':<20}")
    print(f"{'-'*70}")
    print(f"{'Hallucination rate':<35}"
          f"{loose_summary['hallucination_rate']:<20}"
          f"{strict_summary['hallucination_rate']:<20}")
    print(f"{'Correct-decline rate':<35}"
          f"{loose_summary['decline_accuracy']:<20}"
          f"{strict_summary['decline_accuracy']:<20}")

    if loose_summary["hallucination_rate"] and strict_summary["hallucination_rate"] is not None:
        reduction = loose_summary["hallucination_rate"] - strict_summary["hallucination_rate"]
        pct = (reduction / loose_summary["hallucination_rate"]) * 100 if loose_summary["hallucination_rate"] else 0
        print(f"\nHallucination rate reduced by {reduction:.1%} absolute "
              f"({pct:.0f}% relative) with the citation-or-decline constraint.")

    print("\nThis is the number for your resume bullet — quote it exactly as measured.")


def main():
    loose_summary = run_eval(strict=False)
    strict_summary = run_eval(strict=True)
    print_comparison(strict_summary, loose_summary)


if __name__ == "__main__":
    main()
