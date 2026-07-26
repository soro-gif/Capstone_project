"""Couverture des obligations légales — cœur juridique du projet.

1. Extraction des obligations de la source (patterns déontiques, délais, montants).
2. Vérification sémantique de leur présence dans le résumé (embeddings + seuil).
3. coverage = obligations retrouvées / obligations totales.

Adaptable au contexte OHADA/francophone : changer OBLIGATION_PATTERNS et
passer l'embedder en multilingue (ex. paraphrase-multilingual-MiniLM-L12-v2).
"""
import re

# Patterns d'obligations (droit US)
OBLIGATION_PATTERNS = [
    r"\b(?:shall|must|may not|shall not|is required to|are required to|"
    r"is prohibited|shall be liable|is entitled to)\b",
    r"\bwithin\s+\d+\s+(?:days|months|years|business days)\b",  # délais
    r"\bnot\s+later\s+than\b",
    r"\$\s?\d[\d,]*(?:\.\d+)?",  # montants
    r"\bpenalty\b|\bfine\b|\bsanction\b",
]

# Patterns d'obligations FR/OHADA (verbes déontiques, délais, montants FCFA/euros)
OBLIGATION_PATTERNS_FR = [
    r"\b(?:doit|doivent|est tenu de|sont tenus de|ne peut pas|ne peuvent pas|"
    r"est interdit|il est interdit|est passible de|a l'obligation de|"
    r"a le droit de|sont soumis à|sous peine de)\b",
    r"\bdans un délai de\s+\d+\s+(?:jours|mois|ans|heures)\b",
    r"\bau plus tard\b",
    r"\d[\d\s]*(?:francs\s*cfa|fcfa|€|euros?)\b",
    r"\bamende\b|\bpénalité\b|\bsanction\b",
]

_obl_re = re.compile("|".join(OBLIGATION_PATTERNS), re.IGNORECASE)
_obl_re_fr = re.compile("|".join(OBLIGATION_PATTERNS_FR), re.IGNORECASE)

_embedder = None
_embedder_name = None


def _get_embedder(device="cpu", lang="en"):
    """lang='fr' charge un embedder multilingue (OHADA/francophone)."""
    global _embedder, _embedder_name
    name = (
        "paraphrase-multilingual-MiniLM-L12-v2" if lang == "fr" else "all-MiniLM-L6-v2"
    )
    if _embedder is None or _embedder_name != name:
        from sentence_transformers import SentenceTransformer
        from ..hf import retry_closed_hf_client

        _embedder = retry_closed_hf_client(
            lambda: SentenceTransformer(name, device=device)
        )
        _embedder_name = name
    return _embedder


def _sentences(text):
    import nltk

    return [s.strip() for s in nltk.sent_tokenize(text or "") if s.strip()]


def extract_obligations(text, lang="en"):
    """Phrases de la source contenant une obligation."""
    pattern = _obl_re_fr if lang == "fr" else _obl_re
    return [s for s in _sentences(text) if pattern.search(s)]


def obligation_coverage(source, summary, threshold=0.55, device="cpu", lang="en",
                        return_details=False):
    """Fraction des obligations de la source présentes dans le résumé.

    Retourne None si aucune obligation détectée (exclu de la moyenne).
    Si return_details=True, retourne aussi la liste des obligations couvertes/omises
    (utile pour signaler les sections omises à l'utilisateur).
    """
    from sentence_transformers import util as st_util

    obligations = extract_obligations(source, lang=lang)
    if not obligations:
        return (None, {"covered": [], "omitted": []}) if return_details else None
    summ_sents = _sentences(summary)
    if not summ_sents:
        result = 0.0
        details = {"covered": [], "omitted": obligations}
        return (result, details) if return_details else result

    embedder = _get_embedder(device, lang=lang)
    ob_emb = embedder.encode(obligations, convert_to_tensor=True, show_progress_bar=False)
    su_emb = embedder.encode(summ_sents, convert_to_tensor=True, show_progress_bar=False)
    sim = st_util.cos_sim(ob_emb, su_emb)
    best = sim.max(dim=1).values
    covered_mask = best >= threshold
    covered = covered_mask.sum().item()
    result = covered / len(obligations)
    if not return_details:
        return result
    details = {
        "covered": [o for o, ok in zip(obligations, covered_mask.tolist()) if ok],
        "omitted": [o for o, ok in zip(obligations, covered_mask.tolist()) if not ok],
    }
    return result, details


def mean_coverage(sources, summaries, threshold=0.55, device="cpu", lang="en"):
    vals = [
        obligation_coverage(s, p, threshold, device, lang=lang)
        for s, p in zip(sources, summaries)
    ]
    vals = [v for v in vals if v is not None]
    return round(100 * sum(vals) / len(vals), 2) if vals else None
