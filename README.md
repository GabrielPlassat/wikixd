# FabMob Wiki Chatbot

Chatbot conversationnel sur la base de connaissance du wiki de la Fabrique des Mobilités.

## Déploiement sur Streamlit Cloud

### 1. Préparer le repo GitHub

Créez un repo public et uploadez ces 3 fichiers :
- `app.py`
- `requirements.txt`
- `fabmob_embeddings_openai.pkl` (généré depuis le notebook Colab)

> ⚠️ Si le fichier `.pkl` dépasse 100 Mo, utilisez Git LFS.

### 2. Déployer sur Streamlit Cloud

1. Allez sur [share.streamlit.io](https://share.streamlit.io)
2. Connectez votre compte GitHub
3. Cliquez **New app** → sélectionnez votre repo → `app.py`
4. Cliquez **Advanced settings** → **Secrets** et ajoutez :

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
OPENAI_API_KEY = "sk-..."
```

5. Cliquez **Deploy**

### 3. Structure du repo

```
votre-repo/
├── app.py
├── requirements.txt
└── fabmob_embeddings_openai.pkl
```

## Mise à jour de la base de connaissance

Pour mettre à jour avec un nouveau XML wiki :
1. Relancez le notebook `FabMob_Chatbot_OpenAI.ipynb` (blocs 4a → 4b → 4c)
2. Téléchargez le nouveau `fabmob_embeddings_openai.pkl`
3. Remplacez le fichier dans le repo GitHub
4. Streamlit Cloud redéploie automatiquement
