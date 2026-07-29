"""Générateur de résumés législatifs multi-publics — interface Streamlit.

Utilise le modèle pré-entraîné sorolamoussa/t5-small-billsum-fr pour produire
un résumé exécutif (juriste), un résumé dirigeant et/ou un résumé citoyen
d'un projet de loi, avec traçabilité des sections et détection d'hallucinations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if sys.platform == "win32":
    # Silence un ConnectionResetError bénin (WinError 10054) que l'event loop
    # asyncio Proactor de Windows lève quand une connexion est coupée
    # brutalement (rechargement à chaud, onglet fermé) : sans impact sur le
    # fonctionnement de l'app, juste du bruit dans les logs.
    from asyncio.proactor_events import _ProactorBasePipeTransport

    _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

    def _call_connection_lost(self, exc):
        try:
            _orig_call_connection_lost(self, exc)
        except ConnectionResetError:
            pass

    _ProactorBasePipeTransport._call_connection_lost = _call_connection_lost

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import streamlit as st

from billsum.extract import extract_text
from billsum.interpret import interpret_for_audience
from billsum.library import delete_entry, list_entries, save_entry
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

KIND_STYLE = {
    "Résumé": (":material/summarize:", "blue"),
    "Interprétation": (":material/menu_book:", "violet"),
    "Recherche": (":material/travel_explore:", "green"),
}


@st.cache_resource(show_spinner="Chargement du modèle...")
def get_model(model_name: str):
    return load_model(model_name, device="cpu")


st.title("IvoireLoi AI")
st.subheader("L'IA au service du droit ivoirien : résumés et interprétations multi-publics de projets de loi")
st.caption("Résumer votre article de loi avec IvoireLoi AI")

tab_summary, tab_rag, tab_library = st.tabs([
    ":material/auto_awesome: Résumer un article de loi",
    ":material/search: Rechercher un article de loi",
    ":material/bookmark: Bibliothèque",
])

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
        # Le backend T5 local reste disponible dans le code (summarize_for_audience,
        # backend="t5") mais est masqué de l'interface : seul Gemini (Vertex AI) est proposé.
        backend = "vertex"
    else:
        backend = "vertex"
        st.caption("Interpréter l'article de loi.")
    audience = st.pills(
        "Public",
        ["JURISTE", "DIRIGEANT", "CITOYEN"],
        selection_mode="single",
        default="JURISTE",
    )
    audiences = [audience] if audience else []
    check_factuality = False
    check_omissions = True
    if mode == "Résumé":
        check_factuality = st.toggle(
            "Vérifier les hallucinations (NLI)", value=False,
            help="Vérifie que chaque phrase du résumé est bien soutenue par le texte source. "
                 "Télécharge un modèle NLI multilingue au premier lancement.",
        )
        check_omissions = st.toggle(
            "Détecter les sections omises", value=True,
            help="Compare chaque section du texte source au résumé (embeddings sémantiques) "
                 "pour signaler les sections non couvertes. Désactiver accélère la génération.",
        )

with tab_summary:
    st.session_state.setdefault("bill_text", "")
    st.session_state.setdefault("interp_results", {})
    st.session_state.setdefault("summary_results", {})

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
        submit_label = "Générer le résumé" if mode == "Résumé" else "Interpréter l'article"
        submitted = st.form_submit_button(submit_label, icon=":material/auto_awesome:", type="primary")

    # Les résultats sont calculés à la soumission puis stockés dans st.session_state :
    # les boutons "Enregistrer" ci-dessous déclenchent eux-mêmes un rerun (submitted
    # redevient False), donc l'affichage doit survivre à ce rerun pour que le clic
    # sur "Enregistrer" reste utile.
    if submitted:
        if not text.strip():
            st.warning("Veuillez coller un texte de loi avant de générer un résumé.")
        elif not audiences:
            st.warning("Sélectionnez au moins un public.")
        elif mode == "Interprétation":
            for aud in audiences:
                try:
                    with st.spinner(f"Interprétation pour le public {aud.lower()}..."):
                        result = interpret_for_audience(text, aud)
                except Exception as e:
                    st.error(
                        "Échec de l'appel au modèle Gemini (Vertex AI). "
                        "Vérifiez l'authentification (`gcloud auth application-default login`) "
                        f"et votre connexion réseau.\n\nDétail : {e}"
                    )
                    continue
                st.session_state["interp_results"][aud] = {"result": result, "text": text}
        else:
            try:
                model, tokenizer = (None, None) if backend == "vertex" else get_model(FR_MODEL_NAME)
            except Exception as e:
                st.error(f"Impossible de charger le modèle T5 local : {e}")
                st.stop()

            for aud in audiences:
                try:
                    with st.spinner(f"Génération du résumé {aud.lower()}..."):
                        result = summarize_for_audience(
                            text, aud, model=model, tokenizer=tokenizer,
                            device="cpu", check_factuality=check_factuality,
                            check_omissions=check_omissions,
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
                        st.error(f"Échec de la génération du résumé {aud.lower()} : {e}")
                    continue
                st.session_state["summary_results"][aud] = {"result": result, "text": text}

    if mode == "Interprétation":
        relevant = [(a, st.session_state["interp_results"][a]) for a in audiences if a in st.session_state["interp_results"]]
        if relevant:
            tabs = st.tabs([a.capitalize() for a, _ in relevant])
            for tab, (aud, data) in zip(tabs, relevant):
                with tab:
                    result, source_text = data["result"], data["text"]
                    for art in result.articles:
                        with st.container(border=True):
                            st.markdown(f"**{art.header}**")
                            st.markdown(art.interpretation)

                    if st.button(
                        "Enregistrer dans la bibliothèque", icon=":material/bookmark_add:",
                        key=f"save_interp_{aud}",
                    ):
                        content = "\n\n".join(
                            f"**{art.header}**\n{art.interpretation}" for art in result.articles
                        )
                        save_entry("Interprétation", aud, content, source_text)
                        st.toast("Enregistré dans la bibliothèque.", icon=":material/bookmark:")
    else:
        relevant = [(a, st.session_state["summary_results"][a]) for a in audiences if a in st.session_state["summary_results"]]
        if relevant:
            tabs = st.tabs([a.capitalize() for a, _ in relevant])
            for tab, (aud, data) in zip(tabs, relevant):
                with tab:
                    result, source_text = data["result"], data["text"]
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

                    if st.button(
                        "Enregistrer dans la bibliothèque", icon=":material/bookmark_add:",
                        key=f"save_summary_{aud}",
                    ):
                        save_entry("Résumé", aud, result.summary, source_text)
                        st.toast("Enregistré dans la bibliothèque.", icon=":material/bookmark:")

with tab_rag:
    st.caption("Posez une question, l'IA cherche la réponse directement sur les sites officiels ivoiriens.")
    with st.container(horizontal=True):
        for site in OFFICIAL_SITES:
            st.badge(site, icon=":material/language:", color="green")

    st.session_state.setdefault("rag_result", None)

    with st.form("rag_form", border=False):
        question = st.text_input(
            "Votre question",
            placeholder="Ex : Quel est le délai légal pour se mettre en conformité avec la protection des données ?",
            label_visibility="collapsed",
        )
        rag_submitted = st.form_submit_button("Rechercher", icon=":material/search:", type="primary")

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
                st.session_state["rag_result"] = {"result": result, "question": question}

    stored_rag = st.session_state["rag_result"]
    if stored_rag:
        result, source_question = stored_rag["result"], stored_rag["question"]
        st.divider()
        with st.container(border=True):
            st.markdown(f":material/gavel: **Réponse à :** _{source_question}_")
            st.markdown(result["answer"])

            if result["sources"]:
                st.caption(f":material/link: {len(result['sources'])} source(s) citée(s)")
                for s in result["sources"]:
                    with st.expander(s["title"]):
                        st.caption(s["link"])
                        st.write(s["text"][:500] + ("…" if len(s["text"]) > 500 else ""))

            if st.button("Enregistrer dans la bibliothèque", icon=":material/bookmark_add:", key="save_rag"):
                save_entry("Recherche", "—", result["answer"], source_question)
                st.toast("Enregistré dans la bibliothèque.", icon=":material/bookmark:")
    elif not rag_submitted:
        st.info(":material/travel_explore: Posez une question ci-dessus pour lancer une recherche.")

with tab_library:
    entries = list_entries()
    if not entries:
        with st.container(border=True):
            st.markdown(":material/bookmark: **Votre bibliothèque est vide**")
            st.caption(
                "Les résumés, interprétations et recherches que vous enregistrez "
                "apparaîtront ici pour être retrouvés plus tard."
            )
    else:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.metric("Éléments enregistrés", len(entries))
            kind_filter = st.pills(
                "Filtrer",
                ["Tous"] + sorted({e.kind for e in entries}),
                selection_mode="single",
                default="Tous",
            )
            search = st.text_input(
                "Rechercher", placeholder="Rechercher un titre...", label_visibility="collapsed",
            )

        visible = [
            e for e in entries
            if (not kind_filter or kind_filter == "Tous" or e.kind == kind_filter)
            and (not search or search.lower() in e.title.lower())
        ]

        if not visible:
            st.caption("Aucun élément ne correspond à ce filtre.")

        for entry in visible:
            icon, color = KIND_STYLE.get(entry.kind, (":material/description:", "gray"))
            with st.container(border=True):
                header_col, action_col = st.columns([5, 1], vertical_alignment="center")
                with header_col:
                    st.markdown(f"**{entry.title}**")
                    with st.container(horizontal=True):
                        st.badge(entry.kind, icon=icon, color=color)
                        if entry.audience and entry.audience != "—":
                            st.badge(
                                entry.audience.capitalize(),
                                icon=AUDIENCE_ICONS.get(entry.audience, ":material/person:"),
                                color="gray",
                            )
                        st.caption(entry.created_at)
                with action_col:
                    if st.button("", icon=":material/delete:", key=f"delete_{entry.id}", help="Supprimer"):
                        delete_entry(entry.id)
                        st.rerun()

                with st.expander("Voir le contenu"):
                    st.markdown(entry.content)
                    if entry.source_excerpt:
                        st.caption(f":material/description: Source : {entry.source_excerpt}…")
