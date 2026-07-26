"""Recherche web restreinte à des sites officiels ivoiriens (Brave Search
API) et extraction du contenu des pages retournées.

Configuration requise (variable d'environnement) :
- BRAVE_API_KEY : clé API Brave Search (https://api.search.brave.com)

On ajoute un filtre `site:` explicite dans la requête pour se limiter à ces
domaines (Brave ne propose pas de moteur personnalisé pré-restreint comme
Google CSE).
"""
import os

OFFICIAL_SITES = [
    "jo.gouv.ci",                  # Journal Officiel de Côte d'Ivoire
    "sgg.gouv.ci",                 # Secrétariat Général du Gouvernement (textes de loi)
    "assnat.ci",                   # Assemblée Nationale de Côte d'Ivoire
    "gouv.ci",                     # Portail officiel du gouvernement
    "conseil-constitutionnel.ci",  # Conseil constitutionnel
]

_API_KEY_ENV = "BRAVE_API_KEY"
_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class WebSearchNotConfigured(RuntimeError):
    pass


def _get_credentials():
    api_key = os.environ.get(_API_KEY_ENV)
    if not api_key:
        raise WebSearchNotConfigured(
            "Recherche web non configurée : définissez la variable d'environnement "
            f"{_API_KEY_ENV} (clé API Brave Search, https://api.search.brave.com)."
        )
    return api_key


def search(query: str, num_results: int = 5, sites=None) -> list:
    """Recherche Brave restreinte aux sites officiels.

    Retourne une liste de {title, link, snippet}.
    """
    import requests

    api_key = _get_credentials()
    sites = sites or OFFICIAL_SITES
    site_filter = " OR ".join(f"site:{s}" for s in sites)
    full_query = f"({site_filter}) {query}"

    resp = requests.get(
        _BRAVE_ENDPOINT,
        params={"q": full_query, "count": min(num_results, 20)},
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("web", {}).get("results", [])
    return [
        {"title": it.get("title", ""), "link": it.get("url", ""), "snippet": it.get("description", "")}
        for it in items
    ]


def fetch_page_text(url: str, max_chars: int = 6000) -> str:
    """Récupère et nettoie le texte principal d'une page (HTML ou PDF)."""
    import requests

    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        from io import BytesIO
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(resp.content))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    else:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n")

    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text[:max_chars]
