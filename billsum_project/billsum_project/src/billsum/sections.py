"""Découpage d'un texte de loi en sections pour la traçabilité de couverture."""
import re

_SECTION_RE = re.compile(
    r"(?m)^\s*(?:TITRE|Titre|CHAPITRE|Chapitre|Article|ARTICLE|Section|SECTION)\s+"
    r"(?:[0-9IVXLCDM]+(?:er)?|\w+)\b"
)


def split_sections(text: str):
    """Découpe le texte en (titre_section, contenu). Fallback : un seul bloc."""
    text = text or ""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [("Texte entier", text)]

    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = m.group().strip()
        body = text[start:end].strip()
        sections.append((header, body))
    return sections
