"""
ClauseGuard — Week 4, Part C (updated for verified clause-number citations)
"""

import os
import tempfile
import hashlib
import json
import streamlit as st

from pdf_chunking import chunk_contract
from vectorstore_setup import get_client, get_or_create_collection, index_chunks
from risk_scoring import compute_risk_score
from rag_agent import answer_question

st.set_page_config(page_title="ClauseGuard", page_icon="📄", layout="wide")

st.title("📄 ClauseGuard")
st.caption(
    "Upload a contract, ask questions in plain English, get grounded answers "
    "with page citations and a heuristic risk score. Not a substitute for legal review."
)

if "indexed_contract_id" not in st.session_state:
    st.session_state.indexed_contract_id = None
if "contract_name" not in st.session_state:
    st.session_state.contract_name = None


def contract_id_from_filename(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()[:12]


with st.sidebar:
    st.header("1. Upload a contract")
    uploaded_file = st.file_uploader("Contract PDF", type=["pdf"])

    if uploaded_file is not None:
        contract_id = contract_id_from_filename(uploaded_file.name)

        if st.session_state.indexed_contract_id != contract_id:
            with st.spinner("Parsing and indexing contract..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                clauses = chunk_contract(tmp_path)
                clauses_as_dicts = [
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "page_number": c.page_number,
                        "paragraph_index_on_page": c.paragraph_index_on_page,
                        "clause_number": c.clause_number,
                    }
                    for c in clauses
                ]

                tmp_json_path = tmp_path.replace(".pdf", "_chunks.json")
                with open(tmp_json_path, "w") as f:
                    json.dump(clauses_as_dicts, f)

                client = get_client()
                collection = get_or_create_collection(client)
                index_chunks(collection, tmp_json_path, contract_id)

                os.unlink(tmp_path)
                os.unlink(tmp_json_path)

            st.session_state.indexed_contract_id = contract_id
            st.session_state.contract_name = uploaded_file.name
            st.success(f"Indexed {len(clauses)} clauses from {uploaded_file.name}")
        else:
            st.info(f"Already indexed: {uploaded_file.name}")

    st.divider()
    st.caption(
        "⚠️ Heuristic risk scoring based on a fine-tuned classifier — "
        "not a certified legal risk assessment. Always have a qualified "
        "professional review real contracts."
    )

st.header("2. Ask a question")

if st.session_state.indexed_contract_id is None:
    st.info("Upload a contract PDF in the sidebar to get started.")
else:
    question = st.text_input(
        "Your question",
        placeholder="e.g. Can either party terminate this agreement early?",
    )

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving clauses, classifying risk, and generating answer..."):
            result = answer_question(question, st.session_state.indexed_contract_id, strict=True)
            risk = compute_risk_score(result["supporting_clauses"])

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Supporting clauses")
            for c in result["supporting_clauses"]:
                clause_label = f", Clause {c['clause_number']}" if c.get("clause_number") else ""
                with st.expander(
                    f"Page {c['page_number']}{clause_label} — {c['risk_category']} "
                    f"({c['risk_confidence']:.0%} confidence, similarity {c['similarity']})"
                ):
                    st.write(c["text"])

        with col2:
            st.subheader("Risk score")
            severity_colors = {"HIGH": "🔴", "MODERATE": "🟠", "LOW": "🟡", "MINIMAL": "🟢"}
            st.metric(
                label=f"{severity_colors.get(risk['severity'], '')} {risk['severity']}",
                value=f"{risk['score']}/100",
            )

            if risk["contributors"]:
                st.caption("Contributing factors")
                for c in risk["contributors"]:
                    st.write(f"**{c['category']}**: +{c['points']} pts (page {c['page']})")
            else:
                st.caption("No notable risk categories detected in retrieved clauses.")
