"""
ClauseGuard — Week 4, Part A
Risk scoring engine. Takes the clauses your Week 1 classifier has already
tagged with a category + confidence, and turns that into a single 0-100
risk score with a breakdown of what contributed to it.

Important honesty note, same spirit as the compliance-framing caveat from
the project plan: these weights are a heuristic YOU are choosing based on
general contract-risk intuition, not a certified legal risk model. Frame it
that way in any report output and in your resume — "heuristic risk scoring",
not "legal risk assessment."
"""

# Rough, defensible weights: categories more likely to create real financial/
# legal exposure score higher. These are opinionated but explainable — you
# should be able to justify each one if asked in an interview.
RISK_WEIGHTS = {
    "Cap On Liability": 25,
    "Ip Ownership Assignment": 22,
    "Exclusivity": 20,
    "Minimum Commitment": 18,
    "Revenue/Profit Sharing": 18,
    "Rofr/Rofo/Rofn": 16,
    "Anti-Assignment": 14,
    "Insurance": 12,
    "Audit Rights": 10,
    "Post-Termination Services": 10,
    "License Grant": 10,
    "Termination For Convenience": 15,
    "Expiration Date": 5,
    "Governing Law": 5,
    "Parties": 2,
    "NONE": 0,
}

MAX_REASONABLE_SCORE = 100


def compute_risk_score(clauses: list[dict]) -> dict:
    """clauses: list of dicts with 'risk_category' and 'risk_confidence' keys
    (this is exactly the shape rag_agent.py's retrieve_and_classify returns).

    Returns a 0-100 score plus a breakdown of which categories contributed
    how much, so the report can show its work instead of just a number.
    """
    contributors = []
    raw_total = 0

    for clause in clauses:
        category = clause.get("risk_category", "NONE")
        confidence = clause.get("risk_confidence", 0)
        weight = RISK_WEIGHTS.get(category, 5)  # unseen category -> small default weight, not zero

        points = round(weight * confidence, 1)
        if category != "NONE" and points > 0:
            contributors.append({
                "category": category,
                "weight": weight,
                "confidence": confidence,
                "points": points,
                "page": clause.get("page_number"),
            })
        raw_total += points

    # Scale so a handful of high-weight categories doesn't blow past 100 —
    # simple cap rather than a normalized formula, kept deliberately simple
    # and easy to explain.
    score = min(round(raw_total), MAX_REASONABLE_SCORE)

    contributors.sort(key=lambda c: c["points"], reverse=True)

    return {
        "score": score,
        "severity": _severity_label(score),
        "contributors": contributors,
    }


def _severity_label(score: int) -> str:
    if score >= 70:
        return "HIGH"
    if score >= 35:
        return "MODERATE"
    if score > 0:
        return "LOW"
    return "MINIMAL"


if __name__ == "__main__":
    # Quick sanity check with fabricated example clauses
    example_clauses = [
        {"risk_category": "Cap On Liability", "risk_confidence": 0.91, "page_number": 3},
        {"risk_category": "Exclusivity", "risk_confidence": 0.78, "page_number": 2},
        {"risk_category": "NONE", "risk_confidence": 0.99, "page_number": 1},
    ]
    result = compute_risk_score(example_clauses)
    print(f"Score: {result['score']}/100 ({result['severity']})")
    for c in result["contributors"]:
        print(f"  - {c['category']} (page {c['page']}): {c['points']} pts "
              f"(weight {c['weight']} x confidence {c['confidence']:.0%})")
