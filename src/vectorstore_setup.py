"""
ClauseGuard — Step 4
Embed clause chunks into ChromaDB for retrieval.
Runs locally.
"""

import json
import sys
import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "clauseguard_contracts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_client(persist_path="../chroma_store"):
    return chromadb.PersistentClient(path=persist_path)


def get_or_create_collection(client):
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def index_chunks(collection, chunks_json_path, contract_id):
    with open(chunks_json_path) as f:
        chunks = json.load(f)

    collection.add(
        ids=[f"{contract_id}_{c['chunk_id']}" for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"contract_id": contract_id, "page_number": c["page_number"],
             "paragraph_index_on_page": c["paragraph_index_on_page"]}
            for c in chunks
        ],
    )
    print(f"Indexed {len(chunks)} chunks from {chunks_json_path} into '{COLLECTION_NAME}'.")


def query(collection, question, n_results=3):
    results = collection.query(query_texts=[question], n_results=n_results)
    print(f"\nTop {n_results} clauses for: \"{question}\"\n")
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"- (page {meta['page_number']}, similarity={1 - dist:.3f}) {doc[:150]}")
    return results


def main():
    if len(sys.argv) < 3:
        print("Usage: python vectorstore_setup.py ../sample_contracts/contract_chunks.json contract_001")
        return

    chunks_path, contract_id = sys.argv[1], sys.argv[2]
    client = get_client()
    collection = get_or_create_collection(client)
    index_chunks(collection, chunks_path, contract_id)

    QUESTION = "Can either party end this agreement early?"
    query(collection, QUESTION)


if __name__ == "__main__":
    main()