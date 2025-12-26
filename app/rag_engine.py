# app/rag_engine.py
import os
from typing import Any, Dict, List
import re

from duckduckgo_search import DDGS
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Returns a list of {title, href, body} using DuckDuckGo.
    """
    results: List[Dict[str, Any]] = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            # duckduckgo-search обычно возвращает ключи: title, href, body
            results.append(
                {
                    "title": r.get("title", "").strip(),
                    "href": (r.get("href") or r.get("url") or "").strip(),
                    "body": (r.get("body") or r.get("snippet") or "").strip(),
                }
            )

    # чистим пустые
    results = [x for x in results if x["href"] or x["title"] or x["body"]]
    return results


def rewrite_query(question: str) -> str:
    q = question.strip()

    # если спрашивают про RAG, но не уточняют AI/LLM — дописываем контекст
    has_rag = re.search(r"\brag\b", q, flags=re.IGNORECASE) is not None
    has_ai_context = re.search(
        r"retrieval|augmented|generation|llm|ai|langchain|vector|embedding|rag\s*system",
        q,
        flags=re.IGNORECASE,
    ) is not None

    if has_rag and not has_ai_context:
        q = q + " Retrieval-Augmented Generation RAG LLM AI"
    return q


def _format_sources(results: List[Dict[str, Any]]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title") or f"Source {i}"
        body = r.get("body") or ""
        href = r.get("href") or ""
        lines.append(f"[{i}] {title}\n{body}\nURL: {href}".strip())
    return "\n\n".join(lines)


def answer_question(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    RAG from web search -> LLM.
    Returns: {answer: str, sources: [...]}
    """
    sources = web_search(question, max_results=top_k)
    sources_text = _format_sources(sources)

    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    llm = ChatOllama(model=model, temperature=0.2)  # Ollama chat model integration :contentReference[oaicite:1]{index=1}

    system = (
        "Ты RAG-ассистент. Отвечай только опираясь на Search results. "
        "Если данных недостаточно — скажи честно, что не знаешь. "
        "Если используешь факт из источника — ставь ссылку в формате [1], [2]."
    )

    user = (
        f"Вопрос: {question}\n\n"
        f"Search results:\n{sources_text}\n\n"
        "Сформулируй короткий и ясный ответ на русском. "
        "Обязательно добавь цитаты [1], [2] там, где используешь факты."
    )

    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    answer = (msg.content or "").strip()

    return {"answer": answer, "sources": sources}



import re

def rewrite_query(question: str) -> str:
    """
    Если пользователь пишет 'RAG', но не уточняет AI/LLM контекст,
    дописываем "Retrieval-Augmented Generation" чтобы поиск шёл в правильную тему.
    """
    q = (question or "").strip()

    has_rag = re.search(r"\brag\b", q, flags=re.IGNORECASE) is not None
    has_ai_context = re.search(
        r"retrieval|augmented|generation|llm|ai|langchain|vector|embedding|rag\s*system",
        q,
        flags=re.IGNORECASE,
    ) is not None

    if has_rag and not has_ai_context:
        q = q + " Retrieval-Augmented Generation RAG LLM AI"
    return q


def answer_question(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    RAG from web search -> LLM.
    Returns: {answer: str, sources: [...], search_query: str}
    """
    search_q = rewrite_query(question)

    sources = web_search(search_q, max_results=top_k)

    # Если вопрос про RAG — отсекаем источники, которые явно не про AI
    if re.search(r"\brag\b", question or "", re.IGNORECASE):
        filtered = []
        for s in sources:
            blob = f"{s.get('title','')} {s.get('body','')} {s.get('href','')}".lower()
            if any(k in blob for k in ["retrieval", "augmented", "generation", "llm", "langchain", "embedding", "vector"]):
                filtered.append(s)
        # используем фильтр только если он не пустой
        if filtered:
            sources = filtered

    sources_text = _format_sources(sources)

    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    llm = ChatOllama(model=model, temperature=0.2)

    system = (
        "Ты RAG-ассистент. Отвечай только опираясь на Search results. "
        "Если данных недостаточно — скажи честно, что не знаешь. "
        "Если используешь факт из источника — ставь ссылку в формате [1], [2]. "
        "Игнорируй источники, которые не относятся к теме вопроса."
    )

    user = (
        f"Вопрос: {question}\n\n"
        f"Search results:\n{sources_text}\n\n"
        "Сформулируй короткий и ясный ответ на русском. "
        "Обязательно добавь цитаты [1], [2] там, где используешь факты."
    )

    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    answer = (msg.content or "").strip()

    return {"answer": answer, "sources": sources, "search_query": search_q}
