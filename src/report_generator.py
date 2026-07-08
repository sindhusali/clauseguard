"""
ClauseGuard — Week 4, Part B (updated for verified clause-number citations)
"""

import sys
from datetime import datetime

from rag_agent import answer_question
from risk_scoring import compute_risk_score


def generate_report(question: str, contract_id: str) -> dict:
    result = answer_question(question, contract_id, strict=True)
    risk = compute_risk_score(result["supporting_clauses"])

    report = {
        "generated_at": datetime.now().isoformat(),
        "contract_id": contract_id,
        "question": question,
        "answer": result["answer"],
        "risk_score": risk["score"],
        "risk_severity": risk["severity"],
        "risk_contributors": risk["contributors"],
        "supporting_clauses": result["supporting_clauses"],
    }
    return report


def format_report_as_text(report: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("CLAUSEGUARD — EXPLAINABILITY REPORT")
    lines.append("=" * 70)
    lines.append(f"Contract: {report['contract_id']}")
    lines.append(f"Question: {report['question']}")
    lines.append("")
    lines.append(f"ANSWER:\n{report['answer']}")
    lines.append("")
    lines.append(f"RISK SCORE: {report['risk_score']}/100 ({report['risk_severity']})")

    if report["risk_contributors"]:
        lines.append("\nRisk contributors:")
        for c in report["risk_contributors"]:
            lines.append(
                f"  - {c['category']}: +{c['points']} pts "
                f"(page {c['page']}, {c['confidence']:.0%} classifier confidence)"
            )
    else:
        lines.append("\nNo notable risk categories detected in the retrieved clauses.")

    lines.append("\nSupporting evidence:")
    for c in report["supporting_clauses"]:
        clause_str = f", Clause {c['clause_number']}" if c.get("clause_number") else ""
        lines.append(
            f"  - Page {c['page_number']}{clause_str} | {c['risk_category']} "
            f"({c['risk_confidence']:.0%}) | similarity {c['similarity']}"
        )
        lines.append(f"    \"{c['text'][:150]}...\"")

    lines.append("\n" + "-" * 70)
    lines.append("Note: heuristic risk scoring, not a substitute for legal review.")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print('Usage: python report_generator.py <contract_id> "your question here"')
        return

    contract_id = sys.argv[1]
    question = " ".join(sys.argv[2:])

    report = generate_report(question, contract_id)
    print(format_report_as_text(report))


if __name__ == "__main__":
    main()
