# ChatBot XD Mobilité

Informations issues du wiki, forum et site 30vélis

Propulsé par **Claude (Anthropic)** et **OpenAI Embeddings**.

🌐 **Wikixd** : https://wikixd.fabmob.io  
💬 **Forum XD** : https://forum.fabmob.io/c/extreme-defi/
📊 **Expérimentations 30vélis** : https://30veli.fabmob.io  
🚗 **eXtrême Défi** : https://xd.ademe.fr

---

## Architecture générale

```
┌─────────────────────────────────────────────────────┐
│  Interface Streamlit (app.py)                       │
│  • Chat avec historique de conversation             │
│  • Colonne droite : liens + conseils contribution   │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Moteur de recherche hybride                        │
│  50% sémantique (OpenAI) + 50% mots-clés            │
└────────────────────┬────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
┌─────────┐   ┌──────────┐   ┌──────────────┐
│  Wiki   │   │  Forum   │   │  Retours     │
│  FabMob │   │  XD      │   │  utilisateurs│
│ ~3959p  │   │ 30 topics│   │  30vélis CSV │
└─────────┘   └──────────┘   └──────────────┘
```

---

## Sources de données

| Source | Fichier pkl | Catégories | Volume |
|--------|------------|------------|--------|
| Wiki FabMob | `fabmob_embeddings_openai_light.pkl` | Vehicule, Equipe, Projet, Evénement, Commun, Acteur, Actioncollterr, Cas usage, Référentiel, XD | ~3959 pages |
| Forum Discourse | `fabmob_forum.pkl` | Forum (catégorie XD, ID 25) | 30 topics |
| Retours expérience | `fabmob_experiences.pkl` | Experience (agrégé par véhicule) | CSV 30vélis |

---

## Fichiers du projet

### Repo GitHub (Streamlit Cloud)
```
votre-repo/
├── app.py                              ← Application Streamlit
├── requirements.txt                    ← Dépendances Python
├── README.md                           ← Ce fichier
├── fabmob_embeddings_openai_light.pkl  ← Embeddings wiki
├── fabmob_experiences.pkl              ← Embeddings retours expérience
└── fabmob_forum.pkl                    ← Embeddings forum
```

### Notebooks Google Colab

| Notebook | Rôle | Génère |
|----------|------|--------|
| `FabMob_Wiki_API.ipynb` | Exploration du wiki via l'API MediaWiki | — |
| `FabMob_Chatbot_OpenAI.ipynb` | **Principal** — wiki complet | `fabmob_embeddings_openai_light.pkl` |
| `FabMob_Experiences_Embeddings.ipynb` | Retours utilisateurs depuis CSV 30vélis | `fabmob_experiences.pkl` |
| `FabMob_Forum_Embeddings.ipynb` | Discussions forum Discourse XD | `fabmob_forum.pkl` |

---

## Déploiement sur Streamlit Cloud

### 1. Clés API nécessaires

| Service | Variable | Usage | Où créer |
|---------|----------|-------|----------|
| Anthropic | `ANTHROPIC_API_KEY` | Génération des réponses (Claude) | https://console.anthropic.com |
| OpenAI | `OPENAI_API_KEY` | Calcul des embeddings | https://platform.openai.com/api-keys |
| Discourse | `DISCOURSE_API_KEY` + `DISCOURSE_USERNAME` | Lecture du forum (Colab uniquement) | https://forum.fabmob.io/admin/api/keys |

> La clé Discourse : permissions **Read-only**, Single User (votre compte admin).

### 2. Générer les fichiers pkl (Google Colab)

Ajoutez les secrets dans Colab via l'icône 🔑 (panneau gauche) avant de lancer.

**Wiki** — `FabMob_Chatbot_OpenAI.ipynb` :
```
Blocs : 1 → 2 → 3 → 4a → 4b → 4c → télécharger fabmob_embeddings_openai_light.pkl
```
Le bloc 4b nécessite d'uploader le XML du wiki.
Export XML : `https://wikixd.fabmob.io/wiki/Spécial:Exporter`

**Retours expérience** — `FabMob_Experiences_Embeddings.ipynb` :
```
Uploadez 30veli_export_experiences.csv → Blocs : 1 → 2 → 3 → 4 → télécharger fabmob_experiences.pkl
```

**Forum** — `FabMob_Forum_Embeddings.ipynb` :
```
Blocs : 1 → 2 → 3 → 4 → télécharger fabmob_forum.pkl
```

### 3. Déployer

1. Créez un repo GitHub public avec tous les fichiers
2. Allez sur [share.streamlit.io](https://share.streamlit.io)
3. **New app** → votre repo → `app.py`
4. **Advanced settings → Secrets** :

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
OPENAI_API_KEY = "sk-..."
```

5. **Deploy**

> ⚠️ Si les fichiers `.pkl` dépassent 25 Mo, réduire avant upload :
> ```python
> data['embeddings'] = data['embeddings'].astype(np.float16)
> with open('fabmob_embeddings_openai_light.pkl', 'wb') as f:
>     pickle.dump(data, f)
> ```

---

## Sessions Colab suivantes

Une fois le `.pkl` généré, ne recalculez pas tout à chaque fois.

Dans `FabMob_Chatbot_OpenAI.ipynb` :
1. Blocs **1 → 2 → 3**
2. Décommentez et lancez le **bloc 4d** (recharger le pkl)
3. Blocs **5 → 6** (moteur + chatbot)

---

## Mise à jour de la base de connaissance

| Source | Fréquence conseillée | Action |
|--------|---------------------|--------|
| Wiki | À chaque export XML significatif | Relancer blocs 4a→4b→4c, remplacer le pkl |
| Forum | Mensuelle | Relancer `FabMob_Forum_Embeddings.ipynb` entièrement |
| Retours expérience | À chaque nouveau CSV | Relancer `FabMob_Experiences_Embeddings.ipynb` entièrement |

Streamlit Cloud redéploie automatiquement après chaque push GitHub.

---

## Fonctionnement de la recherche

**Recherche hybride** à chaque question :

```
score_final = 0.5 × score_sémantique (OpenAI text-embedding-3-small)
            + 0.5 × score_mots_clés (% de mots de la question dans le texte)
```

Pages retournées par question :
- Top 8 wiki
- Top 3 retours expérience (si `fabmob_experiences.pkl` présent)
- Top 2 forum (si `fabmob_forum.pkl` présent)

Fusion et tri par score décroissant → envoyé à Claude comme contexte.

---

## Structure des catégories wiki

| Catégorie | Modèle wiki | Champs clés pour l'embedding |
|-----------|-------------|------------------------------|
| Vehicule | `{{Vehicule}}` | modeleveh, fabricant, infogene, typeveh, avancement, autono, dossier_veh |
| Equipe | `{{equipe}}` | description, vehicule_equipe, dossier_narra, dossier_ecosys |
| Projet | `{{Project}}` | shortDescription, description, challenge, Develop |
| Evénement | `{{event}}` | name, description, startDate, Location, organizer |
| Commun | `{{Ressource}}` | shortDescription, description, Tags, from, challenge |
| Acteur | `{{acteur}}` | shortDescription, description, pays d'implantation, villes |
| Actioncollterr | `{{Actioncollterr}}` | description, Tags, Theme |
| Cas usage | `{{Cas usage}}` | description, Tags, Theme |
| Référentiel | `{{Connaissance}}` | shortDescription, description, theme |
| XD | (catégorie transverse) | shortDescription, description, Tags, Theme |

---

## Coûts estimés

| Action | Coût estimé |
|--------|-------------|
| Génération embeddings wiki (~3959 pages) | ~0.02 € |
| Génération embeddings forum (30 topics) | < 0.01 € |
| Génération embeddings expériences | < 0.01 € |
| Embedding d'une question | ~0.0001 € |
| Réponse Claude (1500 tokens max) | ~0.005 € |

---

## Interface Streamlit

```
┌─────────────────────────────────┬──────────────────┐
│  🚗 FabMob Wiki Chatbot         │ 🔗 Ressources    │
│  [liens rapides]                │ • Wiki FabMob    │
│─────────────────────────────────│ • Forum XD       │
│                                 │ • eXtrême Défi   │
│  [historique de conversation]   │ • 30vélis        │
│                                 │ • Contribuer     │
│  [sources consultées ▼]         │──────────────────│
│                                 │ 💡 Contribuer à  │
│  > Posez votre question...      │ la communauté    │
│                                 │ (conseils        │
│                     [🔄 Reset]  │ dynamiques)      │
└─────────────────────────────────┴──────────────────┘
```

Les réponses distinguent les sources :
- 📖 **Wiki FabMob** — pages du wiki
- 💬 **Forum XD** — discussions avec lien direct vers le topic
- 📊 **Retours utilisateurs** — données agrégées 30vélis

---

## Contribuer à FabMob

- **Wiki** : https://wikixd.fabmob.io/wiki/Aide:Premiers_pas
- **Forum** : https://forum.fabmob.io
- **GitHub FabMob** : https://github.com/fabmob
