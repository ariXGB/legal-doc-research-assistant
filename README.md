# Legal Document Research Assistant (RAG-based Chatbot)

## Overview

This project is a chatbot that lets you upload legal PDF documents and ask questions about them in plain English. Instead of reading through pages of legal text to find one clause, you can just ask a question and the app will find the relevant part of the document and explain it to you, along with the exact page it came from.

I built this project to learn how **RAG (Retrieval-Augmented Generation)** actually works, instead of just using a hosted chatbot API and calling it a "project." Legal documents (agreements, contracts, terms and conditions, etc.) are usually long, written in dense language, and hard to search through manually. This project tries to solve that by combining document search with an LLM, so answers are based only on what is actually written in the uploaded document — not on the model's general knowledge.

This is **not** a legal advice tool. It only explains what the uploaded documents say. Every answer ends with a disclaimer reminding the user of this.

## Features

- Upload one or more PDF files directly from the browser (via Streamlit)
- Automatically extracts and cleans text from PDFs
- Splits documents into smaller chunks for better search accuracy
- Converts chunks into embeddings and stores them in a local vector index
- Answers questions using only the content retrieved from the uploaded documents (grounded generation)
- Shows the exact source (filename + page number) for every answer
- Keeps a short chat history so follow-up questions have context
- Runs completely locally — no data is sent to an external cloud LLM, since it uses Ollama running on the same machine

## Technologies Used

| Category | Technology |
|---|---|
| Language | Python |
| UI Framework | Streamlit |
| PDF Text Extraction | pdfplumber |
| Text Chunking | LangChain (`RecursiveCharacterTextSplitter`) |
| Embeddings Model | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (Facebook AI Similarity Search) |
| LLM (Answer Generation) | Ollama (local LLM server, e.g. `llama3.1:8b`) |
| Environment Config | python-dotenv |

No traditional database (like MySQL/MongoDB) is used in this project. Document text and chunk metadata are stored as JSON, and vector embeddings are stored using FAISS's own index file.

## How It Works

The basic workflow of this application is:

1. **Upload** – User uploads one or more PDF files through the Streamlit sidebar.
2. **Extract** – Text is extracted from each page of the PDF using `pdfplumber`.
3. **Clean** – Extracted text is cleaned up (removing broken line breaks, extra spaces, etc.) since PDF text extraction is usually messy.
4. **Chunk** – Cleaned text is split into smaller overlapping chunks (~800 characters each). This is done because embedding an entire document as one block loses accuracy — smaller chunks give more precise search results.
5. **Embed** – Each chunk is converted into a vector (a list of numbers representing its meaning) using a Sentence-Transformers model.
6. **Store** – These vectors are stored in a FAISS index, along with metadata (source file, page number) in a JSON file.
7. **Ask** – When the user asks a question, the question is also converted into a vector, and FAISS finds the most similar chunks from the index (this is the "Retrieval" part of RAG).
8. **Generate** – The retrieved chunks are passed to a local LLM (via Ollama) along with a system prompt that tells it to answer *only* using the given context. This is the "Generation" part of RAG.
9. **Display** – The answer is shown in the chat, along with the sources it was based on.

This is what makes it a **RAG** system — instead of the LLM answering from its own training data (which could be outdated or made up), it answers based on documents retrieved specifically for that question.

## Project Structure

```
legal-rag-chatbot/
│
├── app/
│   ├── ingestion.py       # Extracts and cleans text from PDFs, splits into chunks
│   ├── embeddings.py      # Generates embeddings, builds/updates the FAISS index
│   ├── retriever.py       # Searches the FAISS index for relevant chunks
│   ├── generation.py      # Builds the prompt and calls the local LLM (Ollama)
│   ├── history.py         # Manages chat history (keeps last few turns)
│   └── streamlit_app.py   # Main Streamlit app - UI and app flow
│
├── data/
│   └── documents/         # Uploaded PDF files are saved here
│
├── vectorstore/
│   ├── index.faiss        # FAISS vector index (created after processing documents)
│   └── metadata.json      # Chunk text + source + page number for each vector
│
├── requirements.txt
└── README.md
```

### What the important files do

- **`ingestion.py`** – Reads PDF files page by page, cleans the extracted text (fixes hyphenated words split across lines, removes extra whitespace), and splits it into overlapping chunks.
- **`embeddings.py`** – Loads the Sentence-Transformers model, converts chunks into vectors, and saves/loads the FAISS index. Also handles adding new documents to an existing index without losing previously uploaded ones.
- **`retriever.py`** – Takes a user's question, embeds it, and searches FAISS for the closest matching chunks (using cosine similarity, with a minimum similarity threshold so irrelevant chunks are filtered out).
- **`generation.py`** – Contains the system prompt that restricts the LLM to only use retrieved context, formats the final prompt, and calls the local Ollama server.
- **`history.py`** – Simple functions to store and clear the last few question-answer pairs for context in follow-up questions.
- **`streamlit_app.py`** – The main entry point. Handles file uploads, triggers document processing, and renders the chat interface.

## Installation

### Prerequisites

- Python 3.10+ installed
- [Ollama](https://ollama.com) installed and running locally
- A pulled Ollama model (this project is configured for `llama3.1:8b` by default)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/legal-rag-chatbot.git
cd legal-rag-chatbot
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

3. Install the required dependencies:
```bash
pip install streamlit pdfplumber faiss-cpu sentence-transformers langchain-text-splitters requests python-dotenv
```

4. Pull the LLM model in Ollama (only needs to be done once):
```bash
ollama pull llama3.1:8b
```

5. Make sure Ollama is running in the background:
```bash
ollama serve
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

2. Open the local URL shown in the terminal (usually `http://localhost:8501`).

3. In the sidebar, upload one or more PDF files and click **"Process Documents."**

4. Once processing is done, type a question in the chat box at the bottom, for example:
   > "What is the notice period mentioned in this agreement?"

5. The chatbot will reply with an answer along with the source document and page number it used. You can click **"Sources"** to expand and see them.

6. Use **"Clear Chat History"** in the sidebar to reset the conversation.

## Dataset / Input Data

This project does not use a fixed/pre-existing dataset. It works on whatever PDF documents the user uploads at runtime. It was tested using sample legal documents like agreements and terms-of-service PDFs found online.

## Model / Algorithm

This project does not train any machine learning model from scratch. It uses two **pre-trained** models:

1. **Embedding model** – `all-MiniLM-L6-v2` (from Sentence-Transformers). This model converts text into a 384-dimension vector that represents its meaning. Similar sentences end up having vectors that are close to each other, which is what makes semantic search possible.
2. **LLM for answering** – A local LLM served through Ollama (e.g. `llama3.1:8b`). This is only used for generating the final answer text based on the context it's given — not for search or retrieval.

The retrieval itself uses **cosine similarity** (via FAISS `IndexFlatIP` with normalized vectors) to compare the question's vector with all stored chunk vectors and return the top matches above a similarity threshold.

## Results

Since this is a retrieval + generation system and not a classification/prediction model, there is no accuracy percentage to report. The system was manually tested by uploading sample PDF documents and checking whether:
- The retrieved chunks were actually relevant to the question asked
- The answer stayed grounded in the retrieved text (no hallucinated clauses)
- The correct source page number was shown for each answer

No formal benchmark or evaluation dataset was used to measure retrieval accuracy — this would be a good addition for future work.

## Limitations

- Only works with text-based PDFs. Scanned/image PDFs without embedded text won't work, since there's no OCR step.
- Runs on a local LLM through Ollama, so answer quality and speed depend on the model size and the machine's hardware.
- No duplicate-detection — uploading the same PDF twice will index it twice.
- No login/authentication system — it is built for single-user local use, not multiple users at once.
- Chat history is not saved permanently; it resets when the Streamlit session ends.
- Retrieval quality has only been checked manually, not measured with any formal evaluation metric.

## Future Improvements

- Add OCR support (e.g. using Tesseract) for scanned PDFs
- Prevent duplicate documents from being indexed twice
- Add a way to remove a specific document from the index
- Store chat history in a database so it persists across sessions
- Add basic user authentication for multi-user use
- Add a small evaluation set to measure retrieval accuracy
- Option to switch between different local models or a hosted API

## Learning Outcomes

Building this project helped me understand:

- How a RAG (Retrieval-Augmented Generation) pipeline actually works end-to-end, instead of just calling an LLM API directly
- How text embeddings represent meaning, and how vector similarity search works using FAISS
- How to work with PDF text extraction and why cleaning is necessary before processing raw text
- How prompt design (system prompts, context injection) affects how grounded and reliable an LLM's answers are
- Building a functional UI using Streamlit and managing state across user interactions (`st.session_state`)
- Structuring a Python project into separate modules instead of one large script
- Debugging integration issues across files (for example, keeping function return types consistent between modules)
- Using Git for version control while developing the project

## Conclusion

This project is a working example of a Retrieval-Augmented Generation system built for a legal document use case. It shows how document search and LLMs can be combined to answer questions based strictly on uploaded content rather than the model's general knowledge. It's not meant to replace legal professionals — it's meant to make it easier to search and understand long legal documents, and it helped me learn how practical RAG applications are built.

## Author

**[Your Name Here]**
BCA Graduate
[Your Email / GitHub / LinkedIn — optional]
