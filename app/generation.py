import httpx
from ollama import ResponseError
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "gemma4:31b-cloud"

SYSTEM_PROMPT = """You are a legal document research assistant. You help users understand the content of legal documents they have uploaded.

Rules you must follow strictly:
1. Answer ONLY using the information in the provided context. Do not use outside knowledge of law.
2. If the context does not contain enough information to answer, say clearly: "The provided documents do not contain enough information to answer this question."
3. Never invent, guess, or fabricate clauses, section numbers, dates, or citations that are not in the context.
4. When you state a fact, it must be traceable to the context given.
5. You may explain legal terminology in simpler language, but clearly separate explanation from what the document actually states.
6. You are not a lawyer and must not give legal advice or tell the user what they should do. You may only explain what the documents say.
"""


_llm = None
_chain = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0,       # deterministic, grounded answers over creative ones
            timeout=120,         # matches the original requests.post timeout
        )
    return _llm


def get_chain():
    """
    Builds the LCEL pipeline: prompt -> model -> output parser.
    """
    global _chain
    if _chain is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{user_message}"),
        ])
        _chain = prompt | get_llm() | StrOutputParser()
    return _chain


def build_context_block(retrieved_chunks):
    blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        blocks.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page_number']}]\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def call_ollama(user_message):
    """
    Runs the LCEL chain (prompt | llm | StrOutputParser) and handles errors from Ollama.
    """
    chain = get_chain()

    try:
        answer = chain.invoke({"user_message": user_message})
    except (httpx.ConnectError, ConnectionRefusedError):
        raise RuntimeError(
            "Could not reach Ollama. Make sure it's installed and running (`ollama serve` "
            f"or the Ollama app), and that you've pulled the model with `ollama pull {MODEL_NAME}`."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            "Ollama took too long to respond (120s timeout). The model may be too large "
            "for this machine, or still loading — try again in a moment."
        )
    except ResponseError as e:
        raise RuntimeError(
            f"Ollama returned an error ({e}). Confirm the model is pulled: "
            f"`ollama pull {MODEL_NAME}`."
        )

    if not answer:
        raise RuntimeError("Unexpected response from Ollama (empty message content).")
    return answer


def generate_answer(question, retrieved_chunks, chat_history=None):
    if not retrieved_chunks:
        return {
            "answer": "The provided documents do not contain enough information to answer this question.\n\nThis is an informational summary of the uploaded documents, not legal advice.",
            "sources": [],
            "formatted_sources": []
        }

    context_block = build_context_block(retrieved_chunks)

    history_text = ""
    if chat_history:
        recent = chat_history[-3:]
        history_text = "\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in recent)
        history_text = f"Previous conversation:\n{history_text}\n\n"

    user_message = f"""{history_text}Context from uploaded documents:

{context_block}

Question: {question}

Answer the question using only the context above."""

    answer_text = call_ollama(user_message)

    sources = [
        {"source": c["source"], "page_number": c["page_number"], "score": c["score"]}
        for c in retrieved_chunks
    ]

    return {
        "answer": answer_text,
        "sources": sources,
        "formatted_sources": format_sources(sources)
    }


def format_sources(sources):
    seen = {}
    for s in sources:
        key = (s["source"], s["page_number"])
        if key not in seen or s["score"] > seen[key]["score"]:
            seen[key] = s

    unique_sources = sorted(seen.values(), key=lambda x: (x["source"], x["page_number"]))

    return [
        f"{s['source']} — Page {s['page_number']} (relevance: {s['score']:.2f})"
        for s in unique_sources
    ]
