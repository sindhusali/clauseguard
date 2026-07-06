"""
ClauseGuard — Week 2, Part B
The actual RAG agent: retrieves relevant clauses from ChromaDB, tags each
with the Week 1 LoRA classifier, then asks an LLM to answer the user's
question using ONLY those clauses — citing page numbers, and explicitly
declining if the contract doesn't address the question.

This citation-or-decline constraint is what Week 3's eval harness will
measure (hallucination rate before/after this constraint exists).

Requires an Anthropic API key: https://console.anthropic.com/settings/keys
Set it as an environment variable before running:
    Windows (cmd):  set ANTHROPIC_API_KEY=your-key-here
    Or put it in a .env file (see .env.example) and this script will load it.
"""

import os
import sys
import json
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions

from classifier_inference import classify_clauses_batch

load_dotenv()

COLLECTION_NAME = "clauseguard_contracts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"   # free-tier friendly; swap for llama-3.1-8b-instant if you want faster/cheaper
TOP_K = 5

SYSTEM_PROMPT = """You are ClauseGuard, a contract-analysis assistant.

Rules you must follow exactly:
1. Answer ONLY using the clauses provided in the context below. Do not use outside knowledge of contract law.
2. Every claim in your answer must cite the page number of the clause it came from, like this: (Page 2).
3. If the provided clauses do not contain enough information to answer the question, say so explicitly instead of guessing. Do not fabricate a clause that isn't shown to you.
4. Be concise. Answer in 2-4 sentences unless the question requires a list.
"""


def get_chroma_collection(persist_path="../chroma_store"):
    client = chromadb.PersistentClient(path=persist_path)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve_and_classify(collection, question: str, contract_id: str, n_results: int = TOP_K):
    """Retrieve top-k relevant clauses for this contract, then tag each
    with its risk category using the Week 1 LoRA classifier."""
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        where={"contract_id": contract_id},
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return []

    classifications = classify_clauses_batch(docs)

    clauses = []
    for doc, meta, dist, cls in zip(docs, metas, distances, classifications):
        clauses.append({
            "text": doc,
            "page_number": meta["page_number"],
            "similarity": round(1 - dist, 3),
            "risk_category": cls["label"],
            "risk_confidence": cls["confidence"],
        })
    return clauses


def build_context_block(clauses: list[dict]) -> str:
    lines = []
    for i, c in enumerate(clauses, start=1):
        lines.append(
            f"[Clause {i} — Page {c['page_number']} — Risk category: {c['risk_category']} "
            f"({c['risk_confidence']:.0%} confidence)]\n{c['text']}"
        )
    return "\n\n".join(lines)


def ask_llm(question: str, clauses: list[dict]) -> str:
    client = Groq()  # reads GROQ_API_KEY from environment

    context_block = build_context_block(clauses)
    user_message = f"""Contract clauses retrieved for this question:

{context_block}

Question: {question}"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def answer_question(question: str, contract_id: str, persist_path: str = "../chroma_store") -> dict:
    collection = get_chroma_collection(persist_path)
    clauses = retrieve_and_classify(collection, question, contract_id)

    if not clauses:
        return {
            "question": question,
            "answer": "No indexed clauses found for this contract_id. "
                      "Did you run vectorstore_setup.py on it first?",
            "supporting_clauses": [],
        }

    answer = ask_llm(question, clauses)
    return {
        "question": question,
        "answer": answer,
        "supporting_clauses": clauses,
    }


def main():
    if len(sys.argv) < 3:
        print('Usage: python rag_agent.py <contract_id> "your question here"')
        print('Example: python rag_agent.py nda_001 "Can either party end this agreement early?"')
        return

    contract_id = sys.argv[1]
    question = " ".join(sys.argv[2:])

    result = answer_question(question, contract_id)

    print(f"\nQuestion: {result['question']}\n")
    print(f"Answer: {result['answer']}\n")
    print("Supporting clauses:")
    for c in result["supporting_clauses"]:
        print(f"  - Page {c['page_number']} | {c['risk_category']} "
              f"({c['risk_confidence']:.0%}) | similarity {c['similarity']}")
        print(f"    \"{c['text'][:120]}...\"")


if __name__ == "__main__":
    main()
