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
    page_title="ChatBot XD Mobilité",
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

    # Ressources CSV/PDF/Lexique (optionnel)
    try:
        with open("fabmob_ressources.pkl", "rb") as f:
            data_res = pickle.load(f)
        df_res  = data_res["df"]
        emb_res = data_res["embeddings"].astype(np.float32)
    except FileNotFoundError:
        df_res  = None
        emb_res = None

    # Lexique pour enrichir le system prompt
    try:
        with open("lexique_pour_prompt.txt", "r") as f:
            lexique = f.read()
    except FileNotFoundError:
        lexique = ""

    anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    openai_client    = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    return (data_wiki["df"], data_wiki["embeddings"].astype(np.float32),
            df_exp, emb_exp, df_forum, emb_forum,
            df_res, emb_res, lexique,
            anthropic_client, openai_client)
df, embeddings, df_exp, emb_exp, df_forum, emb_forum, df_res, emb_res, lexique, anthropic_client, openai_client = charger_ressources()

# ── Fonctions ─────────────────────────────────────────────────────────────────
def recherche_hybride(question, top_n=8, categories_filtre=None):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    )
    vecteur = np.array([response.data[0].embedding])

    mots = [m.lower() for m in re.findall(r"\w+", question) if len(m) > 3]
    def score_kw(texte):
        t = str(texte).lower()
        return sum(1 for m in mots if m in t) / max(len(mots), 1)

    # Filtre catégorie sur le wiki si demandé
    if categories_filtre:
        mask = df["categorie"].isin(categories_filtre)
        df_f = df[mask].reset_index(drop=True)
        emb_f = embeddings[df[mask].index]
    else:
        df_f = df
        emb_f = embeddings
    scores_sem = cosine_similarity(vecteur, embeddings)[0]
    scores_kw  = df["texte_embedding"].apply(score_kw).values
    scores     = 0.5 * scores_sem + 0.5 * scores_kw
    top_idx    = np.argsort(scores)[::-1][:top_n]
    sources    = [df.iloc[top_idx].copy()]
    sources[-1]["score"] = scores[top_idx]

    if df_exp is not None and emb_exp is not None:
    scores_sem = cosine_similarity(vecteur, emb_f)[0]
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

    # Recherche dans les ressources CSV/PDF
    if df_res is not None and emb_res is not None:
        s_sem = cosine_similarity(vecteur, emb_res)[0]
        s_kw  = df_res["texte_embedding"].apply(score_kw).values
        s     = 0.5 * s_sem + 0.5 * s_kw
        top   = df_res.iloc[np.argsort(s)[::-1][:3]].copy()
        top["score"] = s[np.argsort(s)[::-1][:3]]
        sources.append(top)

    return pd.concat(sources).sort_values("score", ascending=False).head(top_n)


def construire_contexte(pages_df):
    blocs = []
    for _, row in pages_df.iterrows():
        cat = row['categorie']
        url = row['url']
        if cat == 'Forum' and url:
            entete = f"### {row['title']} [Forum XD] — Lien direct : {url}"
        elif cat == 'Experience' and url:
            entete = f"### {row['title']} [Retours utilisateurs] — Source : {url}"
        else:
            entete = f"### {row['title']} [{cat}]"
            if url:
                entete += f" — URL : {url}"
        blocs.append(f"{entete}\n{row['texte_embedding']}")
    return "\n\n---\n\n".join(blocs)


# Le lexique est injecté dynamiquement dans repondre() — voir ci-dessous
SYSTEM_PROMPT_BASE = (
    "Tu es un assistant expert de la communauté XD Mobilité. "
    "La communauté XD Mobilité anime des projets, véhicules, acteurs et communs open source "
    "autour de la mobilité durable et de l'eXtrême Défi ADEME. "
    "Tu réponds UNIQUEMENT à partir des sources fournies en contexte. "
    "Si l'information est absente du contexte, dis-le clairement. "
    "GESTION DES DATES — date du jour : 11/04/2026. "
    "Lorsque tu mentionnes un événement, une résidence, un salon ou toute date précise : "
    "- si la date est antérieure à aujourd'hui, précise que c'est un événement passé "
    "  (ex : la résidence XD du 4 au 8 novembre 2024 (passée)) "
    "- si la date est postérieure à aujourd'hui, présente-la normalement "
    "- en cas de doute sur le statut d'une date, indique-le explicitement "
    "Ne jamais présenter un événement passé comme futur ou en cours. "
    "Tu structures TOUJOURS ta réponse en distinguant la provenance de chaque information : "
    "- Wiki : pages du wiki (véhicules, acteurs, projets, communs...) "
    "- Forum XD : discussions du forum — tu DOIS toujours inclure le lien cliquable [Titre](URL) "
    "- Retours utilisateurs : retours d'expérience des véhicules "
    "Si plusieurs sources abordent le même sujet, croise-les et signale convergences ou divergences. "
    "Tu cites les titres entre crochets [Titre] avec leur URL quand disponible. "
    "Tu réponds en français de façon claire et structurée."
)

SYSTEM_PROMPT_CONSEILS = (
    "Tu es un assistant de la communauté XD Mobilité. "
    "À partir de la question posée et des pages wiki trouvées, génère 2 ou 3 conseils courts "
    "pour encourager la personne à contribuer à la communauté XD Mobilité. "
    "Chaque conseil doit : "
    "1. Suggérer de compléter une page wiki spécifique (donne le titre exact de la page) "
    "   si des informations semblent manquantes, OU "
    "2. Suggérer de poster sur le forum Extrême Défi dans un topic pertinent (donne le nom du topic) "
    "   si la question mérite une discussion communautaire. "
    "   L'URL du forum est toujours https://forum.fabmob.io/c/extreme-defi/ "
    "   sauf si tu connais un topic précis dans les pages trouvées, auquel cas utilise son URL exacte. "
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
    # Adapter top_n et catégorie selon la question
    import re as _re

    # Détecter si la question demande une liste/catalogue
    _nb = _re.search(r"\b(\d+)\b", question)
    _nb_demande = int(_nb.group(1)) if _nb else 0
    _q = question.lower()
    _mots_liste = ["décris", "liste", "cite", "donne", "montre", "quels sont", "présente"]
    _mots_veh   = ["véli", "veli", "vehicule", "véhicule", "tricycle", "quadricycle", "cargo", "prototype"]
    _mots_proj  = ["projet", "equipe", "équipe", "acteur", "commun"]
    _est_liste  = any(m in _q for m in _mots_liste) and _nb_demande > 1

    if _est_liste and any(m in _q for m in _mots_veh):
        # MODE CATALOGUE : requête directe sur df, pas d'embedding
        n = min(_nb_demande if _nb_demande > 0 else 20, 50)
        pages = df[df["categorie"] == "Vehicule"].head(n).copy()
        pages["score"] = 1.0
    elif _est_liste and any(m in _q for m in _mots_proj):
        # MODE CATALOGUE projets
        n = min(_nb_demande if _nb_demande > 0 else 20, 50)
        cat = "Projet" if "projet" in _q else "Equipe" if "équipe" in _q or "equipe" in _q else "Acteur"
        pages = df[df["categorie"] == cat].head(n).copy()
        pages["score"] = 1.0
    else:
        # MODE NORMAL : recherche hybride
        top_n = min(_nb_demande, 20) if _nb_demande > 1 else 8
        _cat = ["Vehicule"] if any(m in _q for m in _mots_veh) else None
        pages = recherche_hybride(question, top_n=top_n, categories_filtre=_cat)

        _cat = ["Vehicule"]
    contexte = construire_contexte(pages)
    messages = []
    for msg in historique_messages[:-1]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"Contexte wiki FabMob :\n\n{contexte}\n\n---\n\nQuestion : {question}"
    })
    # Enrichir le system prompt avec le lexique si disponible
    system = SYSTEM_PROMPT_BASE
    if lexique:
        system += (
            " LEXIQUE XD MOBILITÉ — utilise ces définitions pour interpréter "
            "les termes techniques dans les questions et réponses : "
            + lexique[:4000]
        )
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=system,
        messages=messages
    )
    return response.content[0].text, pages


# ── Interface ─────────────────────────────────────────────────────────────────
# Colonnes : chat (large) | sidebar conseils (étroite)
col_chat, col_side = st.columns([3, 1])

with col_chat:
    st.title("🚗 ChatBot XD Mobilité")
    st.caption("Informations issues du wiki, forum et site 30vélis")

    # Liens rapides
    st.markdown(
        "🔗 [Wiki FabMob](https://wikixd.fabmob.io) &nbsp;|&nbsp; "
        "💬 [Forum XD](https://forum.fabmob.io/c/extreme-defi/) &nbsp;|&nbsp; "
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

    # Message d'accueil au démarrage
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Bonjour ! Je suis le ChatBot XD Mobilité. "
                "Je peux répondre à vos questions sur les véhicules intermédiaires, "
                "les projets, les acteurs et les communs de la FabMob.\n\n"
                "💡 **Conseil** : posez des questions directes pour de meilleurs résultats.\n"
                "Exemple : *Quelles sont les caractéristiques d'Acticycle ?*"
            )

    # Historique
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                idx = i // 2
                if idx < len(st.session_state.sources):
                    with st.expander("📚 Sources consultées"):
                        for _, row in st.session_state.sources[idx].iterrows():
                            icone = {"Forum": "💬", "Experience": "📊", "PDF_Bilan": "📋", "Lexique": "📚", "CSV_Synthese": "📊", "CSV_XD1": "📊", "CSV_XD2": "📊", "CSV_Recap": "📊", "CSV_Evenements": "📅"}.get(row["categorie"], "📖")
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
                    icone = {"Forum": "💬", "Experience": "📊", "PDF_Bilan": "📋", "Lexique": "📚", "CSV_Synthese": "📊", "CSV_XD1": "📊", "CSV_XD2": "📊", "CSV_Recap": "📊", "CSV_Evenements": "📅"}.get(row["categorie"], "📖")
                    lien = f"[{row['title']}]({row['url']})" if row["url"] else row["title"]
                    st.markdown(f"{icone} **{row['score']:.2f}** · {lien} · *{row['categorie']}*")

        st.session_state.messages.append({"role": "assistant", "content": reponse})
        st.session_state.sources.append(pages)
        st.session_state.conseils = conseils
        st.rerun()

    # Conseils contribution — affichés en bas après chaque réponse
    if st.session_state.get("conseils"):
        st.divider()
        st.markdown("#### 💡 Contribuer à la communauté")
        cols = st.columns(len(st.session_state.conseils))
        for j, conseil in enumerate(st.session_state.conseils):
            with cols[j]:
                icone = "📝" if conseil.get("type") == "wiki" else "💬"
                msg   = conseil.get("message", "")
                url   = conseil.get("url", "")
                label = conseil.get("label", "En savoir plus")
                st.info(f"{icone} {msg}")
                if url:
                    st.markdown(f"[→ {label}]({url})")


# ── Colonne latérale : liens + conseils ───────────────────────────────────────
with col_side:
    st.markdown("### 🔗 Ressources FabMob")
    st.markdown(
        "- [🌐 Wiki FabMob](https://wikixd.fabmob.io)\n"
        "- [💬 Forum XD](https://forum.fabmob.io/c/extreme-defi/)\n"
        "- [🚗 eXtrême Défi](https://xd.ademe.fr)\n"
        "- [📊 Expérimentations 30vélis](https://30veli.fabmob.io/)\n"
        "- [📋 Contribuer au wiki](https://wikixd.fabmob.io/wiki/Aide:Premiers_pas)"
    )

