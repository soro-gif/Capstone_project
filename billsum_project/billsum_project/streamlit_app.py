"""Générateur de résumés législatifs multi-publics — interface Streamlit.

Utilise le modèle pré-entraîné sorolamoussa/t5-small-billsum-fr pour produire
un résumé exécutif (juriste), un résumé dirigeant et/ou un résumé citoyen
d'un projet de loi, avec traçabilité des sections et détection d'hallucinations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import streamlit as st

from billsum.extract import extract_text
from billsum.interpret import interpret_for_audience
from billsum.multipublic import FR_MODEL_NAME, load_model, summarize_for_audience
from billsum.rag import answer_question
from billsum.web_search import OFFICIAL_SITES, WebSearchNotConfigured

st.set_page_config(page_title="IvoireLoi AI", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #2E7D32 !important;
        color: #FFFFFF !important;
        border-color: #2E7D32 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #256428 !important;
        border-color: #256428 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

AUDIENCE_ICONS = {
    "JURISTE": ":material/gavel:",
    "DIRIGEANT": ":material/business_center:",
    "CITOYEN": ":material/groups:",
}


@st.cache_resource(show_spinner="Chargement du modèle...")
def get_model(model_name: str):
    return load_model(model_name, device="cpu")


st.title("IvoireLoi AI")
st.subheader("Rendre les articles de loi accessibles et compréhensibles à tous")
st.caption(f"Modèle : {FR_MODEL_NAME}")

tab_summary, tab_rag = st.tabs([":material/auto_awesome: Résumés", ":material/search: Recherche (RAG)"])

with st.sidebar:
    st.subheader("Paramètres")
    mode = st.radio(
        "Mode",
        ["Résumé", "Interprétation"],
        index=0,
        help="Résumé : condense le texte. Interprétation : explique le sens, "
             "les obligations/droits et les implications de chaque article.",
    )
    if mode == "Résumé":
        backend_label = st.radio(
            "Modèle",
            ["T5 local (rapide)", "Gemini fine-tuné (Vertex AI)"],
            index=0,
        )
        backend = "vertex" if backend_label.startswith("Gemini") else "t5"
    else:
        backend = "vertex"
        st.caption("L'interprétation utilise le modèle Gemini (Vertex AI).")
    audience = st.pills(
        "Public",
        ["JURISTE", "DIRIGEANT", "CITOYEN"],
        selection_mode="single",
        default="JURISTE",
    )
    audiences = [audience] if audience else []
    check_factuality = False
    if mode == "Résumé":
        check_factuality = st.toggle(
            "Vérifier les hallucinations (NLI)", value=False,
            help="Vérifie que chaque phrase du résumé est bien soutenue par le texte source. "
                 "Télécharge un modèle NLI multilingue au premier lancement.",
        )

with tab_summary:
    st.session_state.setdefault("bill_text", "")

    uploaded_file = st.file_uploader(
        "Importer un document (Word, PDF ou texte)",
        type=["pdf", "docx", "txt"],
    )
    if uploaded_file is not None and uploaded_file.file_id != st.session_state.get("last_upload_id"):
        try:
            st.session_state["bill_text"] = extract_text(uploaded_file)
            st.session_state["last_upload_id"] = uploaded_file.file_id
        except ValueError as e:
            st.error(str(e))

    with st.form("bill_form", border=False):
        text = st.text_area(
            "Texte du projet de loi",
            height=280,
            placeholder="Collez ici le texte intégral du projet de loi (FR), ou importez un document ci-dessus.",
            key="bill_text",
        )
        submit_label = "Générer les résumés" if mode == "Résumé" else "Interpréter les articles"
        submitted = st.form_submit_button(submit_label, icon=":material/auto_awesome:", type="primary")

    if submitted:
        if not text.strip():
            st.warning("Veuillez coller un texte de loi avant de générer un résumé.")
        elif not audiences:
            st.warning("Sélectionnez au moins un public.")
        elif mode == "Interprétation":
            tabs = st.tabs([f"{a.capitalize()}" for a in audiences])
            for tab, audience in zip(tabs, audiences):
                with tab:
                    try:
                        with st.spinner(f"Interprétation pour le public {audience.lower()}..."):
                            result = interpret_for_audience(text, audience)
                    except Exception as e:
                        st.error(
                            "Échec de l'appel au modèle Gemini (Vertex AI). "
                            "Vérifiez l'authentification (`gcloud auth application-default login`) "
                            f"et votre connexion réseau.\n\nDétail : {e}"
                        )
                        continue

                    for art in result.articles:
                        with st.container(border=True):
                            st.markdown(f"**{art.header}**")
                            st.markdown(art.interpretation)
        else:
            try:
                model, tokenizer = (None, None) if backend == "vertex" else get_model(FR_MODEL_NAME)
            except Exception as e:
                st.error(f"Impossible de charger le modèle T5 local : {e}")
                st.stop()

            tabs = st.tabs([f"{a.capitalize()}" for a in audiences])
            for tab, audience in zip(tabs, audiences):
                with tab:
                    try:
                        with st.spinner(f"Génération du résumé {audience.lower()}..."):
                            result = summarize_for_audience(
                                text, audience, model=model, tokenizer=tokenizer,
                                device="cpu", check_factuality=check_factuality,
                                backend=backend,
                            )
                    except Exception as e:
                        if backend == "vertex":
                            st.error(
                                "Échec de l'appel au modèle Gemini fine-tuné (Vertex AI). "
                                "Vérifiez l'authentification (`gcloud auth application-default login`), "
                                "que l'endpoint est bien déployé, et votre connexion réseau.\n\n"
                                f"Détail : {e}"
                            )
                        else:
                            st.error(f"Échec de la génération du résumé {audience.lower()} : {e}")
                        continue

                    st.markdown(result.summary)

                    with st.container(horizontal=True):
                        if result.obligation_coverage is not None:
                            st.metric(
                                "Couverture des obligations",
                                f"{result.obligation_coverage * 100:.0f}%",
                            )
                        if result.hallucination_rate is not None:
                            st.metric(
                                "Taux d'hallucination",
                                f"{result.hallucination_rate * 100:.0f}%",
                            )
                        st.metric("Sections couvertes", len(result.covered_sections))
                        st.metric("Sections omises", len(result.omitted_sections))

                    if result.omitted_sections:
                        with st.container(border=True):
                            st.markdown(":material/warning: **Sections omises**")
                            st.write(", ".join(result.omitted_sections))

                    if result.unsupported_sentences:
                        with st.expander("Phrases non sourcées par le texte original"):
                            for s in result.unsupported_sentences:
                                st.markdown(f"- {s}")

                    if result.glossary_used:
                        with st.expander("Glossaire appliqué (registre citoyen)"):
                            for term, definition in result.glossary_used.items():
                                st.markdown(f"- **{term}** → {definition}")

with tab_rag:
    st.caption(
        "Posez une question, sans fournir de document. La réponse est générée "
        "par le modèle Gemini fine-tuné (Vertex AI) à partir d'une recherche en "
        "direct sur des sites officiels ivoiriens, avec citation des sources."
    )
    with st.expander("Sites interrogés"):
        st.write(", ".join(OFFICIAL_SITES))

    question = st.text_input(
        "Posez une question sur la loi ivoirienne",
        placeholder="Ex : Quel est le délai légal pour se mettre en conformité avec la protection des données ?",
    )
    rag_submitted = st.button("Rechercher", icon=":material/search:", type="primary")

    if rag_submitted:
        if not question.strip():
            st.warning("Veuillez saisir une question.")
        else:
            try:
                with st.spinner("Recherche sur les sites officiels et génération de la réponse..."):
                    result = answer_question(question, num_results=5, backend="vertex")
            except WebSearchNotConfigured as e:
                st.error(str(e))
            except (TypeError, ValueError) as e:
                st.error(f"Erreur interne lors de la recherche RAG : {e}")
            except Exception as e:
                st.error(
                    "Échec de la recherche web ou de l'appel au modèle Gemini fine-tuné "
                    "(Vertex AI). Vérifiez la clé Google Custom Search, l'authentification "
                    "(`gcloud auth application-default login`), que l'endpoint est bien "
                    f"déployé, et votre connexion réseau.\n\nDétail : {e}"
                )
            else:
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources citées", expanded=True):
                        for s in result["sources"]:
                            st.markdown(f"**[{s['title']}]({s['link']})**")
                            st.caption(s["text"][:500] + ("…" if len(s["text"]) > 500 else ""))
                            st.markdown("---")
