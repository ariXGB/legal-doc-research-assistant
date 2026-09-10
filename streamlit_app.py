import streamlit as st
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

from app.ingestion import extract_text_from_pdf, clean_pages, chunk_pages
from app.embeddings import add_documents, load_index
from app.retriever import retrieve
from app.generation import generate_answer
from app.history import add_turn, clear_history

st.set_page_config(page_title="Legal Document Assistant", layout="wide")

if "index" not in st.session_state:
    index, chunks = load_index()
    st.session_state.index = index
    st.session_state.chunks = chunks

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Legal Document Research Assistant")
st.caption("Upload legal PDFs and ask questions grounded in their content. Not a substitute for legal advice.")

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Process Documents") and uploaded_files:
        os.makedirs("data/documents", exist_ok=True)
        all_pages = []

        with st.spinner("Extracting and processing documents..."):
            for uploaded_file in uploaded_files:
                # Sanitize the client-supplied filename to prevent path traversal
                # (e.g. "../../something") from escaping data/documents.
                safe_name = os.path.basename(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')} {uploaded_file.name}.pdf")
                save_path = os.path.join("data/documents", safe_name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    pages = extract_text_from_pdf(save_path)
                    pages = clean_pages(pages)
                    all_pages.extend(pages)
                except ValueError as e:
                    st.error(f"{safe_name}: {e}")

            if all_pages:
                new_chunks = chunk_pages(all_pages)
                # Append to whatever is already indexed instead of replacing it,
                # so previously processed PDFs stay searchable.
                index, chunks = add_documents(
                    new_chunks,
                    existing_index=st.session_state.index,
                    existing_chunks=st.session_state.chunks,
                )
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.success(
                    f"Processed {len(uploaded_files)} document(s), {len(new_chunks)} new chunks indexed "
                    f"({len(chunks)} total)."
                )

    if st.button("Clear Chat History"):
        st.session_state.history = clear_history()
        st.rerun()

if st.session_state.index is None:
    st.info("Upload and process at least one PDF to get started.")
else:
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])

    question = st.chat_input("Ask a question about your documents...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                results = retrieve(question, st.session_state.index, st.session_state.chunks)
                response = generate_answer(question, results, chat_history=st.session_state.history)

            st.write(response["answer"])

            if response["formatted_sources"]:
                with st.expander("Sources"):
                    for line in response["formatted_sources"]:
                        st.write("-", line)

        st.session_state.history = add_turn(st.session_state.history, question, response["answer"])