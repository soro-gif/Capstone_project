"""Contrôleur de style par public : longueur, niveau de détail, registre.

Le modèle T5-small-billsum-fr n'est pas instruction-tuned : le contrôle par
audience se fait donc via (1) la longueur de génération, (2) un préfixe de
tâche, et (3) un post-traitement (simplification lexicale pour le citoyen).
"""
from dataclasses import dataclass

from .glossary import simplify_for_citizen


@dataclass
class AudienceProfile:
    name: str
    prefix: str          # préfixe injecté avant le texte source
    max_new_tokens: int   # contrôle de longueur
    min_new_tokens: int
    simplify: bool        # applique le glossaire (registre citoyen)


AUDIENCES = {
    "JURISTE": AudienceProfile(
        name="JURISTE",
        prefix="summarize: ",
        max_new_tokens=400,
        min_new_tokens=20,
        simplify=False,
    ),
    "DIRIGEANT": AudienceProfile(
        name="DIRIGEANT",
        prefix="summarize: ",
        max_new_tokens=180,
        min_new_tokens=15,
        simplify=False,
    ),
    "CITOYEN": AudienceProfile(
        name="CITOYEN",
        prefix="summarize: ",
        max_new_tokens=120,
        min_new_tokens=10,
        simplify=True,
    ),
}


def get_profile(audience: str) -> AudienceProfile:
    key = audience.upper()
    if key not in AUDIENCES:
        raise ValueError(f"Public inconnu: {audience}. Choix: {list(AUDIENCES)}")
    return AUDIENCES[key]


def postprocess(text: str, profile: AudienceProfile) -> str:
    return simplify_for_citizen(text) if profile.simplify else text
