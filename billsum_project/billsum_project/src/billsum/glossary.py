"""Glossaire juridique FR -> langage clair, pour le registre CITOYEN."""
import re

GLOSSARY = {
    "nonobstant": "malgré",
    "en vigueur": "applicable",
    "abrogé": "annulé",
    "décret": "texte officiel du gouvernement",
    "promulgation": "publication officielle de la loi",
    "sanction pécuniaire": "amende",
    "personne morale": "entreprise ou organisation",
    "personne physique": "un individu",
    "ayant droit": "bénéficiaire",
    "de plein droit": "automatiquement",
    "à peine de nullité": "sinon ce n'est pas valable",
    "in fine": "à la fin",
    "susmentionné": "cité plus haut",
    "codifié": "inscrit dans le code de loi",
    "dispositif": "les mesures prévues",
    "recours contentieux": "action en justice",
}

_pattern = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in sorted(GLOSSARY, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def simplify_for_citizen(text: str) -> str:
    """Remplace le jargon juridique par des équivalents en langage clair."""

    def repl(m):
        match_str = m.group()
        replacement = GLOSSARY[match_str.lower()]
        if match_str.istitle() or match_str[0].isupper():
            return replacement.capitalize()
        return replacement

    return _pattern.sub(repl, text or "")


def glossary_terms_used(text: str):
    """Termes du glossaire présents dans le texte (pour annexe pédagogique)."""
    found = set(m.group().lower() for m in _pattern.finditer(text or ""))
    return {t: GLOSSARY[t] for t in found}
