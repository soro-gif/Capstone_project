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


def _generate_single_chunk(model, tokenizer, text, profile, device, max_source_len=512,
                           num_beams=4, no_repeat_ngram_size=5, repetition_penalty=1.15,
                           length_penalty=1.0):
    import torch

    enc = tokenizer(
        profile.prefix + text,
        max_length=max_source_len,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Un extrait court (ex. un seul article bref) ne justifie pas un résumé de
    # min_new_tokens fixe : forcer une longueur minimale au-delà de ce que le
    # texte source permet pousse le modèle à halluciner du contenu hors sujet
    # pour "remplir". On adapte donc ce plancher à la longueur de l'entrée,
    # sans toucher à max_new_tokens (early_stopping laisse le modèle s'arrêter
    # naturellement, et max_new_tokens reste le levier de différenciation par public).
    source_len = enc["input_ids"].shape[1]
    min_new_tokens = min(profile.min_new_tokens, max(3, source_len // 2))

    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=profile.max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            early_stopping=True,
            length_penalty=length_penalty,
        )
    raw = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return _clean_output(raw)


def _get_section_content(header: str, body: str) -> str:
    """Extrait le contenu utile d'une section en retirant l'en-tête répété.

    `body` commence toujours par `header` (voir `split_sections`), qu'il soit
    suivi d'un saut de ligne ("Article 1\nLe texte...") ou, cas le plus
    fréquent, directement du texte sur la même ligne ("Article 1er. Le
    texte..."). On retire donc uniquement le préfixe `header`, jamais la
    ligne entière : la retirer entièrement effacerait aussi le contenu réel
    quand il suit l'en-tête sans saut de ligne.

    Retourne une chaîne vide si la section ne contient pas de contenu propre
    au-delà de l'en-tête (ex. section TITRE qui n'a pas d'articles entre le
    header et le suivant).
    """
    import re

    body = body.strip()
    rest = body.removeprefix(header)
    rest = re.sub(r"^[\s.:\-–—]+", "", rest)
    return rest.strip()


def _generate(model, tokenizer, text, profile, device, max_source_len=512, **gen_kwargs):
    """Génère un résumé complet, section par section."""
    sections = split_sections(text)

    # Texte sans sections identifiables : chemin direct
    if len(sections) <= 1:
        return _generate_single_chunk(model, tokenizer, text, profile, device, max_source_len=max_source_len, **gen_kwargs)

    # Résumé indépendant de chaque section (contenu sans en-tête répété)
    section_summaries = []
    for header, body in sections:
        content = _get_section_content(header, body)
        if not content.strip():
            continue
        sec_sum = _generate_single_chunk(
            model, tokenizer, content, profile, device, max_source_len=max_source_len, **gen_kwargs
        )
        sec_sum = sec_sum.strip()
        if sec_sum:
            # Capitaliser la première lettre de chaque résumé de section
            sec_sum = sec_sum[0].upper() + sec_sum[1:]
            section_summaries.append(sec_sum)

    if not section_summaries:
        summary = _generate_single_chunk(model, tokenizer, text, profile, device, max_source_len=max_source_len, **gen_kwargs)
    else:
        combined = " ".join(section_summaries)
        comb_tokens = tokenizer(combined, truncation=False, return_tensors="pt").input_ids.shape[1]
        # Si la concaténation reste courte, on la garde telle quelle (couverture maximale).
        # Sinon, une passe de synthèse finale pour réduire.
        if comb_tokens <= 1024:
            summary = combined
        else:
            summary = _generate_single_chunk(model, tokenizer, combined, profile, device, max_source_len=max_source_len, **gen_kwargs)

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
                           check_omissions: bool = True,
                           backend: str = "t5",
                           num_beams: int = 4, no_repeat_ngram_size: int = 5,
                           repetition_penalty: float = 1.15,
                           length_penalty: float = 1.0) -> SummaryResult:
    """Résumé pour un public donné, avec traçabilité des sections et
    vérification anti-hallucination. Ne produit pas d'interprétation
    juridique non sourcée : le résumé provient uniquement du modèle
    entraîné sur le texte source, et les sections/obligations non
    retrouvées dans le résumé sont explicitement signalées.

    backend: "t5" (modèle local sorolamoussa/t5-small-billsum-fr, par section)
             ou "vertex" (Gemini fine-tuné sur Vertex AI, document entier).

    num_beams/no_repeat_ngram_size/repetition_penalty/length_penalty :
    paramètres de decoding, exposés pour permettre des essais comparatifs
    (voir scripts/run_eval.py) sans devoir modifier le code.
    """
    profile = get_profile(audience)

    if backend == "vertex":
        from .vertex_backend import generate_with_vertex

        raw_summary = generate_with_vertex(text, audience)
    else:
        if model is None or tokenizer is None:
            model, tokenizer = load_model(model_name, device)
        raw_summary = _generate(
            model, tokenizer, text, profile, device,
            num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty, length_penalty=length_penalty,
        )

    summary = postprocess(raw_summary, profile)

    covered, omitted = [], []
    cov = None
    if check_omissions:
        sections = split_sections(text)
        covered, omitted = _section_coverage(sections, summary, device)
        cov, _cov_details = obligation_coverage(
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
                    model_name: str = FR_MODEL_NAME, **gen_kwargs) -> dict:
    """Résumé exécutif (JURISTE) + résumé citoyen (CITOYEN) par défaut,
    avec traçabilité des sections couvertes/omises pour chacun.
    """
    model, tokenizer = load_model(model_name, device)
    results = {}
    for aud in audiences:
        results[aud] = summarize_for_audience(
            text, aud, model=model, tokenizer=tokenizer, device=device,
            model_name=model_name, **gen_kwargs,
        )
    return results
