"""Backend de génération via le modèle Gemini fine-tuné (Vertex AI).

Alternative au modèle local T5-small-billsum-fr : appelle l'endpoint Vertex
AI issu du tuning supervisé (scripts/run_vertex_finetune.py). Le contrôle
par audience se fait via l'instruction système (le modèle est instruction-
tuned, contrairement à T5) et une limite de tokens de sortie.

Utilise le SDK unifié google-genai en mode Vertex AI plutôt que le SDK
vertexai.generative_models (déprécié), et désactive le raisonnement interne
("thinking") du modèle de base gemini-2.5-flash : sans ça, les tokens de
thinking consomment tout le budget max_output_tokens et tronquent la
réponse visible avant qu'elle ne soit produite.
"""
VERTEX_PROJECT = "project-12661c0f-93ef-4036-bb0"
VERTEX_LOCATION = "us-central1"
VERTEX_ENDPOINT = "projects/937895719900/locations/us-central1/endpoints/890169561649774592"

_SYSTEM_INSTRUCTIONS = {
    "JURISTE": (
        "Tu es un assistant qui résume des projets de loi de façon précise et "
        "complète, à destination d'un juriste. Conserve toutes les obligations "
        "légales, exceptions et références précises. Ne simplifie pas le vocabulaire."
    ),
    "DIRIGEANT": (
        "Tu es un assistant qui résume des projets de loi à destination d'un "
        "dirigeant d'entreprise. Sois concis, concentre-toi sur les impacts "
        "opérationnels et obligations concrètes, sans jargon juridique inutile."
    ),
    "CITOYEN": (
        "Tu es un assistant qui résume des projets de loi en langage simple pour "
        "un citoyen sans formation juridique. Utilise des phrases courtes et évite "
        "le jargon technique."
    ),
}

_MAX_TOKENS = {"JURISTE": 500, "DIRIGEANT": 250, "CITOYEN": 180}

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(
            vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION
        )
    return _client


def generate_with_vertex(text: str, audience: str) -> str:
    from google.genai import types

    audience = audience.upper()
    client = _get_client()
    system = _SYSTEM_INSTRUCTIONS.get(audience, _SYSTEM_INSTRUCTIONS["JURISTE"])
    prompt = f"{system}\n\nTexte du projet de loi :\n{text}"
    response = client.models.generate_content(
        model=VERTEX_ENDPOINT,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=_MAX_TOKENS.get(audience, 400),
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text.strip()
