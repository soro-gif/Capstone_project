"""Pipeline d'inférence multi-publics : audience + longueur + traçabilité.

Utilise le modèle déjà fine-tuné `sorolamoussa/t5-small-billsum-fr`
(https://huggingface.co/sorolamoussa/t5-small-billsum-fr) : pas de
réentraînement, uniquement du contrôle au moment de la génération et de la
vérification post-hoc (couverture des obligations + hallucination).
"""
from dataclasses import dataclass, field

from .audience import get_profile, postprocess
from .sections import split_sections
from .metrics.obligations import obligation_coverage
from .metrics.hallucination import hallucination_rate

FR_MODEL_NAME = "sorolamoussa/t5-small-billsum-fr"

_model = None
_tokenizer = None
_model_name = None


def load_model(model_name: str = FR_MODEL_NAME, device: str = "cpu"):
    global _model, _tokenizer, _model_name
    if _model is None or _model_name != model_name:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from .hf import retry_closed_hf_client

        _tokenizer, _model = retry_closed_hf_client(
            lambda: (
                AutoTokenizer.from_pretrained(model_name),
                AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval(),
            )
        )
        _model_name = model_name
    return _model, _tokenizer


@dataclass
class SummaryResult:
    audience: str
    summary: str
    covered_sections: list = field(default_factory=list)
    omitted_sections: list = field(default_factory=list)
    obligation_coverage: float = None
    hallucination_rate: float = None
    unsupported_sentences: list = field(default_factory=list)
    glossary_used: dict = field(default_factory=dict)


def _clean_output(text: str) -> str:
    """Nettoie les artefacts courants du modèle en tête de résumé (ex. ': ' parasite)."""
    import re
    text = text.strip()
    # Le modèle répète parfois le séparateur ':' du préfixe en début de sortie
    text = re.sub(r'^[:\s]+', '', text)
    return _dedupe_repeated_sentences(text.strip())


def _dedupe_repeated_sentences(text: str) -> str:
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) >= 2:
        def normalize(sentence: str) -> str:
            return re.sub(r"[^a-z0-9àâäéèêëîïôöùûüç]+", " ", sentence.lower()).strip()

        first_norm = normalize(sentences[0])
        second_norm = normalize(sentences[1])
        if first_norm and second_norm and (
            first_norm.endswith(second_norm) or second_norm.endswith(first_norm)
        ):
            return " ".join([sentences[0]] + sentences[2:])
    return text


def _generate_single_chunk(model, tokenizer, text, profile, device, max_source_len=512):
    import torch

    enc = tokenizer(
        profile.prefix + text,
        max_length=max_source_len,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=profile.max_new_tokens,
            min_new_tokens=profile.min_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=5,
            repetition_penalty=1.15,
            early_stopping=True,
            length_penalty=1.0,
        )
    raw = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return _clean_output(raw)


def _get_section_content(header: str, body: str) -> str:
    """Extrait le contenu utile d'une section en retirant la ligne d'en-tête répétée.

    Retourne une chaîne vide si la section ne contient pas de contenu propre
    (ex. section TITRE qui n'a pas d'articles entre le header et le suivant).
    """
    import re

    lines = [l for l in body.split("\n") if l.strip()]
    if not lines:
        return ""
    # La première ligne est souvent l'en-tête repris : "TITRE I - ...", "Article 1"
    first = lines[0].strip()
    if re.match(r"^(TITRE|CHAPITRE|ARTICLE|SECTION)\s+\w+", first, re.IGNORECASE):
        # Retirer la ligne d'en-tête ; retourner "" si rien ne suit (section titre vide)
        return "\n".join(lines[1:]).strip()
    return body


def _generate(model, tokenizer, text, profile, device, max_source_len=512):
    """Génère un résumé complet, section par section."""
    sections = split_sections(text)

    # Texte sans sections identifiables : chemin direct
    if len(sections) <= 1:
        return _generate_single_chunk(model, tokenizer, text, profile, device, max_source_len=max_source_len)

    # Résumé indépendant de chaque section (contenu sans en-tête répété)
    section_summaries = []
    for header, body in sections:
        content = _get_section_content(header, body)
        if not content.strip():
            continue
        sec_sum = _generate_single_chunk(
            model, tokenizer, content, profile, device, max_source_len=max_source_len
        )
        sec_sum = sec_sum.strip()
        if sec_sum:
            # Capitaliser la première lettre de chaque résumé de section
            sec_sum = sec_sum[0].upper() + sec_sum[1:]
            section_summaries.append(sec_sum)

    if not section_summaries:
        summary = _generate_single_chunk(model, tokenizer, text, profile, device, max_source_len=max_source_len)
    else:
        combined = " ".join(section_summaries)
        comb_tokens = tokenizer(combined, truncation=False, return_tensors="pt").input_ids.shape[1]
        # Si la concaténation reste courte, on la garde telle quelle (couverture maximale).
        # Sinon, une passe de synthèse finale pour réduire.
        if comb_tokens <= 1024:
            summary = combined
        else:
            summary = _generate_single_chunk(model, tokenizer, combined, profile, device, max_source_len=max_source_len)

    return summary


def _section_coverage(sections, summary, device, lang="fr", threshold=0.4):
    """Pour chaque section, vérifie si son contenu est représenté dans le résumé
    (similarité sémantique max entre les phrases de la section et du résumé).
    Sert à signaler les sections omises, indépendamment des patterns d'obligation.
    """
    from sentence_transformers import util as st_util
    from .metrics.obligations import _get_embedder, _sentences

    summ_sents = _sentences(summary)
    if not summ_sents:
        return [], [h for h, _ in sections]

    embedder = _get_embedder(device, lang=lang)
    su_emb = embedder.encode(summ_sents, convert_to_tensor=True, show_progress_bar=False)

    covered, omitted = [], []
    for header, body in sections:
        sec_sents = _sentences(body)
        if not sec_sents:
            continue
        sec_emb = embedder.encode(sec_sents, convert_to_tensor=True, show_progress_bar=False)
        sim = st_util.cos_sim(sec_emb, su_emb)
        best = sim.max().item()
        (covered if best >= threshold else omitted).append(header)
    return covered, omitted


def summarize_for_audience(text: str, audience: str, model=None, tokenizer=None,
                           device="cpu", model_name: str = FR_MODEL_NAME,
                           check_factuality: bool = True,
                           backend: str = "t5") -> SummaryResult:
    """Résumé pour un public donné, avec traçabilité des sections et
    vérification anti-hallucination. Ne produit pas d'interprétation
    juridique non sourcée : le résumé provient uniquement du modèle
    entraîné sur le texte source, et les sections/obligations non
    retrouvées dans le résumé sont explicitement signalées.

    backend: "t5" (modèle local sorolamoussa/t5-small-billsum-fr, par section)
             ou "vertex" (Gemini fine-tuné sur Vertex AI, document entier).
    """
    profile = get_profile(audience)

    if backend == "vertex":
        from .vertex_backend import generate_with_vertex

        raw_summary = generate_with_vertex(text, audience)
    else:
        if model is None or tokenizer is None:
            model, tokenizer = load_model(model_name, device)
        raw_summary = _generate(model, tokenizer, text, profile, device)

    summary = postprocess(raw_summary, profile)

    sections = split_sections(text)
    covered, omitted = _section_coverage(sections, summary, device)

    cov, cov_details = obligation_coverage(
        text, summary, threshold=0.5, device=device, lang="fr", return_details=True
    )

    unsupported = []
    hrate = None
    if check_factuality:
        hrate, unsupported = hallucination_rate(
            text, summary, device=device, lang="fr", return_details=True
        )

    from .glossary import glossary_terms_used

    return SummaryResult(
        audience=profile.name,
        summary=summary,
        covered_sections=covered,
        omitted_sections=omitted,
        obligation_coverage=cov,
        hallucination_rate=hrate,
        unsupported_sentences=unsupported,
        glossary_used=glossary_terms_used(raw_summary) if profile.simplify else {},
    )


def generate_report(text: str, audiences=("JURISTE", "CITOYEN"), device="cpu",
                    model_name: str = FR_MODEL_NAME) -> dict:
    """Résumé exécutif (JURISTE) + résumé citoyen (CITOYEN) par défaut,
    avec traçabilité des sections couvertes/omises pour chacun.
    """
    model, tokenizer = load_model(model_name, device)
    results = {}
    for aud in audiences:
        results[aud] = summarize_for_audience(
            text, aud, model=model, tokenizer=tokenizer, device=device,
            model_name=model_name,
        )
    return results
