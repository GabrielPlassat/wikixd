import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import anthropic
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="FabMob Wiki Chatbot",
    page_icon="🚗",
    layout="centered"
)

# ── Chargement des ressources (mis en cache) ──────────────────────────────────
@st.cache_resource
def charger_ressources():
    # Wiki
    with open("fabmob_embeddings_openai_light.pkl", "rb") as f:
        data_wiki = pickle.load(f)

    # Retours experience (optionnel)
    try:
        with open("fabmob_experiences.pkl", "rb") as f:
            data_exp = pickle.load(f)
        df_exp  = data_exp["df"]
        emb_exp = data_exp["embeddings"].astype(np.float32)
    except FileNotFoundError:
        df_exp  = None
        emb_exp = None

    anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    openai_client    = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    return (data_wiki["df"], data_wiki["embeddings"].astype(np.float32),
            df_exp, emb_exp, anthropic_client, openai_client)

df, embeddings, df_exp, emb_exp, anthropic_client, openai_client = charger_ressources()

# ── Fonctions ─────────────────────────────────────────────────────────────────
def recherche_hybride(question, top_n=8):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    )
    vecteur = np.array([response.data[0].embedding])

    mots = [m.lower() for m in re.findall(r'\w+', question) if len(m) > 3]
    def score_kw(texte):
        t = str(texte).lower()
        return sum(1 for m in mots if m in t) / max(len(mots), 1)

    # Recherche dans le wiki
    scores_sem_wiki = cosine_similarity(vecteur, embeddings)[0]
    scores_kw_wiki  = df['texte_embedding'].apply(score_kw).values
    scores_wiki     = 0.5 * scores_sem_wiki + 0.5 * scores_kw_wiki
    top_wiki = df.iloc[np.argsort(scores_wiki)[::-1][:top_n]].copy()
    top_wiki['score'] = scores_wiki[np.argsort(scores_wiki)[::-1][:top_n]]

    # Recherche dans les retours experience (si disponible)
    if df_exp is not None and emb_exp is not None:
        scores_sem_exp = cosine_similarity(vecteur, emb_exp)[0]
        scores_kw_exp  = df_exp['texte_embedding'].apply(score_kw).values
        scores_exp     = 0.5 * scores_sem_exp + 0.5 * scores_kw_exp
        top_exp = df_exp.iloc[np.argsort(scores_exp)[::-1][:3]].copy()
        top_exp['score'] = scores_exp[np.argsort(scores_exp)[::-1][:3]]
        # Fusionner et trier
        res = pd.concat([top_wiki, top_exp]).sort_values('score', ascending=False).head(top_n)
    else:
        res = top_wiki.head(top_n)

    return res


def construire_contexte(pages_df):
    blocs = []
    for _, row in pages_df.iterrows():
        blocs.append(
            f"### {row['title']} [{row['categorie']}]\n"
            f"{row['texte_embedding']}\n"
            f"URL : {row['url']}"
        )
    return "\n\n---\n\n".join(blocs)


SYSTEM_PROMPT = (
    "Tu es un assistant expert de la Fabrique des Mobilités (FabMob). "
    "La FabMob anime une communauté de projets, véhicules, acteurs et communs open source "
    "autour de la mobilité durable. "
    "Tu réponds UNIQUEMENT à partir des pages wiki FabMob fournies en contexte. "
    "Si l'information est absente du contexte, dis-le clairement. "
    "Tu cites les titres des pages sources entre crochets [Titre] avec leur URL. "
    "Tu réponds en français de façon claire et structurée."
)


def repondre(question, historique_messages):
    pages = recherche_hybride(question, top_n=8)
    contexte = construire_contexte(pages)

    # Construire les messages avec l'historique
    messages = []
    for msg in historique_messages[:-1]:  # tout sauf le dernier (la question en cours)
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Ajouter la question avec le contexte
    messages.append({
        "role": "user",
        "content": f"Contexte wiki FabMob :\n\n{contexte}\n\n---\n\nQuestion : {question}"
    })

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=messages
    )
    return response.content[0].text, pages


# ── Interface Streamlit ───────────────────────────────────────────────────────
st.title("🚗 FabMob Wiki Chatbot")
st.caption(
    f"Base de connaissance : {len(df)} pages wiki · "
    f"{df['categorie'].nunique()} catégories · "
    "Propulsé par Claude + OpenAI"
)

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

# Bouton reset
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🔄 Nouvelle conversation"):
        st.session_state.messages = []
        st.session_state.sources = []
        st.rerun()

# Afficher l'historique de la conversation
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Afficher les sources pour les messages assistant
        if msg["role"] == "assistant" and i // 2 < len(st.session_state.sources):
            src_idx = i // 2
            if src_idx < len(st.session_state.sources):
                with st.expander("📚 Pages wiki consultées"):
                    sources = st.session_state.sources[src_idx]
                    for _, row in sources.iterrows():
                        st.markdown(
                            f"**{row['score']:.2f}** · [{row['title']}]({row['url']}) "
                            f"· *{row['categorie']}*"
                        )

# Zone de saisie
if question := st.chat_input("Posez votre question sur la mobilité durable..."):

    # Ajouter la question à l'historique
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans le wiki..."):
            reponse, pages = repondre(question, st.session_state.messages)

        st.markdown(reponse)

        with st.expander("📚 Pages wiki consultées"):
            for _, row in pages.iterrows():
                st.markdown(
                    f"**{row['score']:.2f}** · [{row['title']}]({row['url']}) "
                    f"· *{row['categorie']}*"
                )

    # Sauvegarder dans l'historique
    st.session_state.messages.append({"role": "assistant", "content": reponse})
    st.session_state.sources.append(pages)
