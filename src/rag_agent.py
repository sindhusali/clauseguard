"""
ClauseGuard — Week 2/3 (updated for verified clause-number citations)
The RAG agent: retrieves relevant clauses from ChromaDB, tags each with the
Week 1 LoRA classifier, then asks an LLM to answer the user's question using
ONLY those clauses — citing page AND clause number, with both now coming
from verified metadata (not the LLM guessing a clause number from context,
which previously produced occasional wrong numbers).
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
from groq import Groq, RateLimitError
import chromadb
from chromadb.utils import embedding_functions

from classifier_inference import classify_clauses_batch

load_dotenv()

COLLECTION_NAME = "clauseguard_contracts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
TOP_K = 5

STRICT_SYSTEM_PROMPT = """You are ClauseGuard, a contract-analysis assistant.

Rules you must follow exactly:
1. Answer ONLY using the clauses provided in the context below. Do not use outside knowledge of contract law.
2. Every claim in your answer must cite EXACTLY the page and clause number given to you in the context, in this
   format: (Page 2, Clause 5). Use ONLY the page/clause numbers shown in the context — never guess or infer a
   clause number yourself. If a clause has no clause number listed, cite the page only, like (Page 1).
3. If the provided clauses do not contain enough information to answer the question, say so explicitly instead of guessing. Do not fabricate a clause that isn't shown to you.
4. Be concise. Answer in 2-4 sentences unless the question requires a list.
"""

LOOSE_SYSTEM_PROMPT = """You are a helpful contract-analysis assistant.
Answer the user's question about their contract as helpfully as you can,
using the clauses provided as context. Be concise.
"""


def get_chroma_collection(persist_path="../chroma_store"):
    client = chromadb.PersistentClient(path=persist_path)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve_and_classify(collection, question: str, contract_id: str, n_results: int = TOP_K):
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
        raw_clause_number = meta.get("clause_number", 0)
        clauses.append({
            "text": doc,
            "page_number": meta["page_number"],
            "clause_number": raw_clause_number if raw_clause_number else None,
            "similarity": round(1 - dist, 3),
            "risk_category": cls["label"],
            "risk_confidence": cls["confidence"],
        })
    return clauses


def build_context_block(clauses: list[dict]) -> str:
    lines = []
    for i, c in enumerate(clauses, start=1):
        citation = f"Page {c['page_number']}"
        if c.get("clause_number"):
            citation += f", Clause {c['clause_number']}"
        lines.append(
            f"[Clause {i} — {citation} — Risk category: {c['risk_category']} "
            f"({c['risk_confidence']:.0%} confidence)]\n{c['text']}"
        )
    return "\n\n".join(lines)


def ask_llm(question: str, clauses: list[dict], strict: bool = True, max_retries: int = 3) -> str:
    client = Groq()

    context_block = build_context_block(clauses)
    user_message = f"""Contract clauses retrieved for this question:

{context_block}

Question: {question}"""

    system_prompt = STRICT_SYSTEM_PROMPT if strict else LOOSE_SYSTEM_PROMPT

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait_seconds = 30
            print(f"  Rate limit hit, waiting {wait_seconds}s before retry ({attempt + 1}/{max_retries})...")
            time.sleep(wait_seconds)


def answer_question(question: str, contract_id: str, persist_path: str = "../chroma_store",
                     strict: bool = True) -> dict:
    collection = get_chroma_collection(persist_path)
    clauses = retrieve_and_classify(collection, question, contract_id)

    if not clauses:
        return {
            "question": question,
            "answer": "No indexed clauses found for this contract_id. "
                      "Did you run vectorstore_setup.py on it first?",
            "supporting_clauses": [],
        }

    answer = ask_llm(question, clauses, strict=strict)
    return {
        "question": question,
        "answer": answer,
        "supporting_clauses": clauses,
    }


def main():
    if len(sys.argv) < 3:
        print('Usage: python rag_agent.py <contract_id> "your question here"')
        return

    contract_id = sys.argv[1]
    question = " ".join(sys.argv[2:])

    result = answer_question(question, contract_id)

    print(f"\nQuestion: {result['question']}\n")
    print(f"Answer: {result['answer']}\n")
    print("Supporting clauses:")
    for c in result["supporting_clauses"]:
        clause_str = f", Clause {c['clause_number']}" if c.get("clause_number") else ""
        print(f"  - Page {c['page_number']}{clause_str} | {c['risk_category']} "
              f"({c['risk_confidence']:.0%}) | similarity {c['similarity']}")
        print(f"    \"{c['text'][:120]}...\"")


if __name__ == "__main__":
    main()
