import streamlit as st
import os
import xml.etree.ElementTree as ET

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(page_title="Chat Wiki", page_icon="💬", layout="centered")
st.title("💬 Chat avec ton Wiki")

# ── Clé API Anthropic ─────────────────────────────────────────────────────────
if "ANTHROPIC_API_KEY" in st.secrets:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
else:
    api_key = st.sidebar.text_input("🔑 Clé API Anthropic (Claude)", type="password")

if not api_key:
    st.info("👈 Entre ta clé API Claude dans la barre latérale pour commencer.")
    st.stop()

os.environ["ANTHROPIC_API_KEY"] = api_key

# ── Fonctions utilitaires ─────────────────────────────────────────────────────
def extract_text_from_xml(file) -> list[Document]:
    """Parse le XML et retourne une liste de Documents LlamaIndex."""
    tree = ET.parse(file)
    root = tree.getroot()

    def strip_ns(tag):
        return tag.split("}")[-1] if "}" in tag else tag

    def get_all_text(element):
        texts = []
        if element.text and element.text.strip():
            texts.append(element.text.strip())
        for child in element:
            texts.extend(get_all_text(child))
            if child.tail and child.tail.strip():
                texts.append(child.tail.strip())
        return texts

    page_tags = ["page", "article", "entry", "item", "document", "doc"]
    pages_found = []
    for elem in root.iter():
        if strip_ns(elem.tag).lower() in page_tags:
            pages_found.append(elem)

    documents = []
    if pages_found:
        for i, page in enumerate(pages_found):
            text = " ".join(get_all_text(page))
            if text.strip():
                documents.append(Document(text=text, metadata={"source": f"page_{i+1}"}))
    else:
        all_text = " ".join(get_all_text(root))
        documents.append(Document(text=all_text, metadata={"source": "wiki"}))

    return documents


@st.cache_resource(show_spinner="⏳ Construction de l'index (1-3 min)...")
def build_index(_documents):
    """Construit l'index vectoriel — mis en cache pour éviter de reconstruire à chaque interaction."""
    Settings.llm = Anthropic(
        model="claude-sonnet-4-5",
        api_key=api_key,
        max_tokens=2048
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return VectorStoreIndex.from_documents(_documents)


@st.cache_resource(show_spinner=False)
def get_chat_engine(_index):
    return _index.as_chat_engine(
        chat_mode="condense_plus_context",
        verbose=False,
        system_prompt=(
            "Tu es un assistant expert sur ce wiki. "
            "Réponds en français, de façon claire et synthétique. "
            "Base-toi uniquement sur le contenu du wiki fourni."
        ),
    )


# ── Upload du fichier XML ─────────────────────────────────────────────────────
uploaded_file = st.sidebar.file_uploader("📂 Upload ton fichier XML", type=["xml"])

if not uploaded_file:
    st.info("👈 Upload ton fichier XML dans la barre latérale pour commencer.")
    st.stop()

# ── Construction de l'index ───────────────────────────────────────────────────
with st.spinner("📖 Lecture du XML..."):
    documents = extract_text_from_xml(uploaded_file)
    st.sidebar.success(f"✅ {len(documents)} pages chargées")

index = build_index(tuple(documents))
chat_engine = get_chat_engine(index)

# ── Interface de chat ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Pose ta question sur le wiki..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            response = chat_engine.chat(prompt)
            answer = str(response)
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

if st.session_state.messages:
    if st.sidebar.button("🔄 Nouvelle conversation"):
        st.session_state.messages = []
        chat_engine.reset()
        st.rerun()
