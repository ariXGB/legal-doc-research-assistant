import pdfplumber
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No such file: {pdf_path}")

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page_number": page_num,
                    "text": text,
                    "source": os.path.basename(pdf_path)
                })

    if not pages:
        raise ValueError(f"No extractable text found in {pdf_path}. It may be a scanned/image PDF.")

    return pages


def extract_text_from_folder(folder_path):
    all_pages = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            try:
                pages = extract_text_from_pdf(full_path)
                all_pages.extend(pages)
            except ValueError as e:
                print(f"Skipping {filename}: {e}")
    return all_pages

def clean_text(text):
    text = re.sub(r"-\n(?=[a-z])", "", text)      # de-hyphenate words broken across lines
    text = re.sub(r"\n{2,}", "\n", text)           # collapse multiple blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)         # collapse extra spaces
    text = text.strip()
    return text


def clean_pages(pages):
    for page in pages:
        page["text"] = clean_text(page["text"])
    return pages


def chunk_pages(pages, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    chunk_id = 0
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page_number": page["page_number"],
                "source": page["source"]
            })
            chunk_id += 1

    return chunks