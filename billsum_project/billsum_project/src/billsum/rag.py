"""RAG sans document fourni par l'utilisateur : recherche en direct sur les
sites officiels ivoiriens (web_search.py, Google Custom Search restreint),
extraction du contenu des pages retournées, puis génération d'une réponse
citant les sources (Gemini, backend Vertex AI).
"""
from concurrent.futures import ThreadPoolExecutor

from . import web_search


def _fetch_source(hit: dict) -> dict:
    try:
        text = web_search.fetch_page_text(hit["link"])
    except Exception:
        text = ""
    if not text.strip():
        text = hit["snippet"]
    return {"title": hit["title"], "link": hit["link"], "text": text}


def answer_question(query: str, num_results: int = 5, backend: str = "vertex") -> dict:
    """Cherche sur le web (sites officiels) et génère une réponse citant les sources.

    Seul le backend "vertex" (Gemini fine-tuné, instruction-tuned) est supporté :
    le modèle T5 local est fine-tuné pour la synthèse (préfixe "summarize:") et
    ne sait pas répondre à une question ouverte.
    """
    hits = web_search.search(query, num_results=num_results)
    if not hits:
        return {
            "answer": "Aucune source officielle trouvée pour répondre à cette question.",
            "sources": [],
        }

    # Les pages sont récupérées en parallèle (I/O réseau, indépendantes les
    # unes des autres) : en séquentiel, jusqu'à 5 pages x 15s de timeout
    # pouvaient bloquer 75s avant même l'appel au modèle.
    with ThreadPoolExecutor(max_workers=len(hits)) as pool:
        sources = list(pool.map(_fetch_source, hits))

    if backend != "vertex":
        raise ValueError(
            "Seul le backend 'vertex' est supporté pour la génération de réponses RAG."
        )

    from .vertex_backend import answer_with_web_sources

    answer = answer_with_web_sources(query, sources)
    return {"answer": answer, "sources": sources}
