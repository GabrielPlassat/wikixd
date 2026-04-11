import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import anthropic
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FabMob Wiki Chatbot",
    page_icon="🚗",
    layout="wide"
)

# ── Chargement des ressources ─────────────────────────────────────────────────
@st.cache_resource
def charger_ressources():
    with open("fabmob_embeddings_openai_light.pkl", "rb") as f:
        data_wiki = pickle.load(f)

    try:
        with open("fabmob_experiences.pkl", "rb") as f:
            data_exp = pickle.load(f)
        df_exp  = data_exp["df"]
        emb_exp = data_exp["embeddings"].astype(np.float32)
    except FileNotFoundError:
        df_exp  = None
        emb_exp = None

    try:
        with open("fabmob_forum.pkl", "rb") as f:
            data_forum = pickle.load(f)
        df_forum  = data_forum["df"]
        emb_forum = data_forum["embeddings"].astype(np.float32)
    except FileNotFoundError:
        df_forum  = None
        emb_forum = None

    anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    openai_client    = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    return (data_wiki["df"], data_wiki["embeddings"].astype(np.float32),
            df_exp, emb_exp, df_forum, emb_forum,
            anthropic_client, openai_client)

df, embeddings, df_exp, emb_exp, df_forum, emb_forum, anthropic_client, openai_client = charger_ressources()

# ── Fonctions ─────────────────────────────────────────────────────────────────
def recherche_hybride(question, top_n=8):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    )
    vecteur = np.array([response.data[0].embedding])

    mots = [m.lower() for m in re.findall(r"\w+", question) if len(m) > 3]
    def score_kw(texte):
        t = str(texte).lower()
        return sum(1 for m in mots if m in t) / max(len(mots), 1)

    scores_sem = cosine_similarity(vecteur, embeddings)[0]
    scores_kw  = df["texte_embedding"].apply(score_kw).values
    scores     = 0.5 * scores_sem + 0.5 * scores_kw
    top_idx    = np.argsort(scores)[::-1][:top_n]
    sources    = [df.iloc[top_idx].copy()]
    sources[-1]["score"] = scores[top_idx]

    if df_exp is not None and emb_exp is not None:
        s_sem = cosine_similarity(vecteur, emb_exp)[0]
        s_kw  = df_exp["texte_embedding"].apply(score_kw).values
        s     = 0.5 * s_sem + 0.5 * s_kw
        top   = df_exp.iloc[np.argsort(s)[::-1][:3]].copy()
        top["score"] = s[np.argsort(s)[::-1][:3]]
        sources.append(top)

    if df_forum is not None and emb_forum is not None:
        s_sem = cosine_similarity(vecteur, emb_forum)[0]
        s_kw  = df_forum["texte_embedding"].apply(score_kw).values
        s     = 0.5 * s_sem + 0.5 * s_kw
        top   = df_forum.iloc[np.argsort(s)[::-1][:2]].copy()
        top["score"] = s[np.argsort(s)[::-1][:2]]
        sources.append(top)

    return pd.concat(sources).sort_values("score", ascending=False).head(top_n)


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
    "Tu réponds UNIQUEMENT à partir des sources FabMob fournies en contexte. "
    "Si l'information est absente du contexte, dis-le clairement. "
    "Tu structures TOUJOURS ta réponse en distinguant la provenance de chaque information : "
    "- 📖 Wiki FabMob : pages du wiki (véhicules, acteurs, projets, communs...) "
    "- 💬 Forum XD : discussions du forum Extrême Défi "
    "- 📊 Retours utilisateurs : retours d'expérience des véhicules "
    "Si plusieurs sources abordent le même sujet, croise-les et signale convergences ou divergences. "
    "Tu cites les titres entre crochets [Titre] avec leur URL quand disponible. "
    "Tu réponds en français de façon claire et structurée."
)

SYSTEM_PROMPT_CONSEILS = (
    "Tu es un assistant de la Fabrique des Mobilités. "
    "À partir de la question posée et des pages wiki trouvées, génère 2 ou 3 conseils courts "
    "pour encourager la personne à contribuer à la communauté FabMob. "
    "Chaque conseil doit : "
    "1. Suggérer de compléter une page wiki spécifique (donne le titre exact de la page) "
    "   si des informations semblent manquantes, OU "
    "2. Suggérer de poster sur le forum dans un topic pertinent (donne le nom du topic) "
    "   si la question mérite une discussion communautaire. "
    "Format de réponse STRICT — retourne uniquement un JSON : "
    "[{'type': 'wiki' ou 'forum', 'message': 'texte court du conseil', "
    "'url': 'URL de la page wiki ou du topic forum', 'label': 'Compléter le wiki' ou 'Poster sur le forum'}] "
    "Maximum 3 conseils. Réponds UNIQUEMENT avec le JSON, sans texte avant ou après."
)


def generer_conseils(question, pages):
    contexte_titres = "\n".join([
        f"- {r['title']} ({r['categorie']}) : {r['url']}"
        for _, r in pages.iterrows()
    ])
    prompt = (
        f"Question de l'utilisateur : {question}\n\n"
        f"Pages trouvées :\n{contexte_titres}"
    )
    try:
        resp = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=SYSTEM_PROMPT_CONSEILS,
            messages=[{"role": "user", "content": prompt}]
        )
        import json
        texte = resp.content[0].text.strip()
        conseils = json.loads(texte)
        return conseils
    except Exception:
        return []


def repondre(question, historique_messages):
    pages = recherche_hybride(question, top_n=8)
    contexte = construire_contexte(pages)
    messages = []
    for msg in historique_messages[:-1]:
        messages.append({"role": msg["role"], "content": msg["content"]})
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


# ── Interface ─────────────────────────────────────────────────────────────────
# Colonnes : chat (large) | sidebar conseils (étroite)
col_chat, col_side = st.columns([3, 1])

with col_chat:
    st.title("🚗 FabMob Wiki Chatbot")
    st.caption(
        f"Base de connaissance : {len(df)} pages wiki · "
        f"{df['categorie'].nunique()} catégories · "
        "Propulsé par Claude + OpenAI"
    )

    # Liens rapides
    st.markdown(
        "🔗 [Wiki FabMob](https://wikixd.fabmob.io) &nbsp;|&nbsp; "
        "💬 [Forum XD](https://forum.fabmob.io/c/25) &nbsp;|&nbsp; "
        "📊 [eXtrême Défi](https://xd.ademe.fr)",
        unsafe_allow_html=True
    )
    st.divider()

    # Initialisation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = []
    if "conseils" not in st.session_state:
        st.session_state.conseils = []

    # Bouton reset
    if st.button("🔄 Nouvelle conversation"):
        st.session_state.messages = []
        st.session_state.sources  = []
        st.session_state.conseils = []
        st.rerun()

    # Historique
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                idx = i // 2
                if idx < len(st.session_state.sources):
                    with st.expander("📚 Sources consultées"):
                        for _, row in st.session_state.sources[idx].iterrows():
                            icone = {"Forum": "💬", "Experience": "📊"}.get(row["categorie"], "📖")
                            lien = f"[{row['title']}]({row['url']})" if row["url"] else row["title"]
                            st.markdown(f"{icone} **{row['score']:.2f}** · {lien} · *{row['categorie']}*")

    # Zone de saisie
    if question := st.chat_input("Posez votre question sur la mobilité durable..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours..."):
                reponse, pages = repondre(question, st.session_state.messages)
                conseils = generer_conseils(question, pages)

            st.markdown(reponse)
            with st.expander("📚 Sources consultées"):
                for _, row in pages.iterrows():
                    icone = {"Forum": "💬", "Experience": "📊"}.get(row["categorie"], "📖")
                    lien = f"[{row['title']}]({row['url']})" if row["url"] else row["title"]
                    st.markdown(f"{icone} **{row['score']:.2f}** · {lien} · *{row['categorie']}*")

        st.session_state.messages.append({"role": "assistant", "content": reponse})
        st.session_state.sources.append(pages)
        st.session_state.conseils = conseils
        st.rerun()

# ── Colonne latérale : liens + conseils ───────────────────────────────────────
with col_side:
    st.markdown("### 🔗 Ressources FabMob")
    st.markdown(
        "- [🌐 Wiki FabMob](https://wikixd.fabmob.io)\n"
        "- [💬 Forum XD](https://forum.fabmob.io/c/25)\n"
        "- [🚗 eXtrême Défi](https://xd.ademe.fr)\n"
        "- [📋 Contribuer au wiki](https://wikixd.fabmob.io/wiki/Aide:Premiers_pas)"
    )
    st.divider()

    st.markdown("### 💡 Contribuer à la communauté")

    if st.session_state.get("conseils"):
        for conseil in st.session_state.conseils:
            icone = "📝" if conseil.get("type") == "wiki" else "💬"
            label = conseil.get("label", "En savoir plus")
            msg   = conseil.get("message", "")
            url   = conseil.get("url", "")
            st.info(f"{icone} {msg}")
            if url:
                st.markdown(f"[→ {label}]({url})")
    else:
        st.caption(
            "Posez une question pour recevoir des suggestions "
            "personnalisées pour contribuer au wiki ou au forum."
        )
    st.divider()
    st.caption("💡 Conseil : posez des questions directes.\nEx : *'Quelles sont les caractéristiques du Biro ?'*")
