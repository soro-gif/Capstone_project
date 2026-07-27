"""Extraction de texte depuis des fichiers Word (.docx) et PDF (.pdf)."""


def extract_text(uploaded_file) -> str:
    """uploaded_file : objet type st.UploadedFile (a .name et lit des bytes)."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        text = _extract_pdf(uploaded_file)
    elif name.endswith(".docx"):
        text = _extract_docx(uploaded_file)
    elif name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore").strip()
    else:
        raise ValueError(f"Format non supporté: {name} (attendu : .pdf, .docx, .txt)")

    if not text:
        raise ValueError(
            f"Aucun texte n'a pu être extrait de « {uploaded_file.name} » "
            "(PDF probablement scanné/image sans couche texte, ou document vide)."
        )
    return text


def _extract_pdf(uploaded_file) -> str:
    from pypdf import PdfReader

    reader = PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _extract_docx(uploaded_file) -> str:
    from docx import Document

    doc = Document(uploaded_file)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()
