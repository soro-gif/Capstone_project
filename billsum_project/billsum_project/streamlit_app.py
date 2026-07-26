"""Générateur de résumés législatifs multi-publics — interface Streamlit.

Utilise le modèle pré-entraîné sorolamoussa/t5-small-billsum-fr pour produire
un résumé exécutif (juriste), un résumé dirigeant et/ou un résumé citoyen
d'un projet de loi, avec traçabilité des sections et détection d'hallucinations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from billsum.extract import extract_text
from billsum.multipublic import FR_MODEL_NAME, load_model, summarize_for_audience

st.set_page_config(page_title="Résumés législatifs multi-publics", layout="wide")

AUDIENCE_ICONS = {
    "JURISTE": ":material/gavel:",
    "DIRIGEANT": ":material/business_center:",
    "CITOYEN": ":material/groups:",
}


@st.cache_resource(show_spinner="Chargement du modèle...")
def get_model(model_name: str):
    return load_model(model_name, device="cpu")


st.title("Résumés législatifs multi-publics")
st.caption(f"Modèle : {FR_MODEL_NAME}")

with st.sidebar:
    st.subheader("Paramètres")
    backend_label = st.radio(
        "Modèle",
        ["T5 local (rapide)", "Gemini fine-tuné (Vertex AI)"],
        index=0,
    )
    backend = "vertex" if backend_label.startswith("Gemini") else "t5"
    audience = st.pills(
        "Public",
        ["JURISTE", "DIRIGEANT", "CITOYEN"],
        selection_mode="single",
        default="JURISTE",
    )
    audiences = [audience] if audience else []
    check_factuality = st.toggle(
        "Vérifier les hallucinations (NLI)", value=False,
        help="Vérifie que chaque phrase du résumé est bien soutenue par le texte source. "
             "Télécharge un modèle NLI multilingue au premier lancement.",
    )

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
    submitted = st.form_submit_button("Générer les résumés", icon=":material/auto_awesome:", type="primary")

if submitted:
    if not text.strip():
        st.warning("Veuillez coller un texte de loi avant de générer un résumé.")
    elif not audiences:
        st.warning("Sélectionnez au moins un public.")
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
