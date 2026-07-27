"""Interprétation d'articles de loi en langage clair, par public.

Contrairement au résumé (compression du texte), l'interprétation explique le
sens, les obligations/droits et les implications pratiques de chaque article,
article par article. Utilise exclusivement le backend Gemini (Vertex AI) :
le T5 local n'est pas instruction-tuned et ne peut pas suivre ce type de
consigne.
"""
from dataclasses import dataclass, field

from .sections import split_sections
from .vertex_backend import interpret_article_with_vertex


@dataclass
class ArticleInterpretation:
    header: str
    interpretation: str


@dataclass
class InterpretationResult:
    audience: str
    articles: list = field(default_factory=list)  # list[ArticleInterpretation]


def interpret_for_audience(text: str, audience: str) -> InterpretationResult:
    audience = audience.upper()
    sections = split_sections(text)
    articles = [
        ArticleInterpretation(
            header=header,
            interpretation=interpret_article_with_vertex(body, audience),
        )
        for header, body in sections
    ]
    return InterpretationResult(audience=audience, articles=articles)
