# Projet 4 — Assistant IA Multi-Provider

## Objectif du projet

Créer une application web qui permet d'interagir avec plusieurs fournisseurs d'IA (Claude, ChatGPT, Grok, DeepSeek, Hugging Face) via une seule interface.

Ce projet enseigne :
- L'intégration de plusieurs API d'IA
- La création d'une API backend flexible avec FastAPI
- Le design d'une interface frontend réactive
- La gestion des clés API de manière sécurisée
- La conteneurisation Docker

## Architecture

```
projet4-assistant/
├── backend/
│   ├── app.py              # Application FastAPI principale
│   ├── requirements.txt    # Dépendances Python
│   ├── Dockerfile          # Image Docker du backend
│   └── .env.example        # Template des variables d'environnement
├── frontend/
│   └── index.html          # Interface web complète
├── docker-compose.yml      # Orchestration des services
└── README.md               # Ce fichier
```

## Prérequis

- Python 3.11+
- Docker & Docker Compose (optionnel, pour la conteneurisation)
- Clés API pour les services désirés

## Installation locale

### 1. Cloner le projet et installer les dépendances

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configurer les clés API

Créer un fichier `.env` à partir du template :

```bash
cp .env.example .env
```

Remplir les clés API dans le fichier `.env` selon les services que vous souhaitez utiliser.

### 3. Lancer le backend

```bash
python -m uvicorn app:app --reload --port 8000
```

Le backend démarre sur `http://localhost:8000`

### 4. Ouvrir l'interface frontend

Ouvrir `frontend/index.html` dans un navigateur web.

## Configuration des clés API

### Claude (Anthropic)
1. Aller sur https://console.anthropic.com
2. Créer un compte ou se connecter
3. Naviguer vers API Keys
4. Générer une nouvelle clé
5. Copier la clé dans `.env` : `ANTHROPIC_API_KEY=sk-ant-...`

**Note** : Remplacer la clé par votre clé réelle. Les clés commencent par `sk-ant-`.

### ChatGPT (OpenAI)
1. Aller sur https://platform.openai.com/account/api-keys
2. Créer un compte ou se connecter
3. Créer une nouvelle clé API
4. Copier la clé dans `.env` : `OPENAI_API_KEY=sk-...`

**Note** : Vous devez avoir des crédits disponibles. Gratuit les 3 premiers mois.

### Grok (X AI)
1. Aller sur https://console.x.ai
2. Créer un compte
3. Créer une clé API
4. Copier la clé dans `.env` : `XAI_API_KEY=...`

**Note** : Service actuellement en bêta. Gratuit avec limite de requêtes.

### DeepSeek
1. Aller sur https://platform.deepseek.com
2. Créer un compte ou se connecter
3. Générer une clé API
4. Copier la clé dans `.env` : `DEEPSEEK_API_KEY=sk-...`

**Note** : Très économique. Très bon rapport performance/prix.

### Hugging Face
1. Aller sur https://huggingface.co/settings/tokens
2. Créer un compte ou se connecter
3. Créer un token d'accès (read)
4. Copier le token dans `.env` : `HUGGINGFACE_API_KEY=hf_...`

**Note** : Gratuit. Modèle utilisé : Llama 3.1 8B (exécution rapide).

## Utilisation

1. Sélectionner un provider dans la dropdown
2. Taper un message dans la zone de texte
3. Appuyer sur "Envoyer" ou Entrée
4. Voir la réponse du provider dans le chat

## Endpoints de l'API

### GET /health
Vérifier l'état du serveur.

**Réponse:**
```json
{"status": "ok"}
```

### GET /providers
Liste tous les providers disponibles.

**Réponse:**
```json
{
  "providers": ["claude", "chatgpt", "grok", "deepseek", "huggingface"],
  "configured": ["claude", "chatgpt"]
}
```

### POST /chat
Envoyer un message à un provider.

**Requête:**
```json
{
  "message": "Bonjour, comment ça va?",
  "provider": "claude"
}
```

**Réponse:**
```json
{
  "response": "Bonjour! Je vais bien, merci de demander...",
  "provider": "claude",
  "usage": {
    "input_tokens": 15,
    "output_tokens": 25
  }
}
```

**Erreurs possibles:**
- 400 : Provider non configuré ou message vide
- 500 : Erreur API du provider

## Déploiement avec Docker

### Build et lancer les services

```bash
docker-compose up --build
```

Le frontend est accessible via un serveur HTTP simple exposé sur le port 8080.
Le backend est exposé sur le port 8000.

### Arrêter les services

```bash
docker-compose down
```

## Problèmes courants

### "API key not configured for provider X"
Vérifier que :
1. La clé est présente dans le fichier `.env`
2. Le nom de la variable correspond à celui attendu
3. Le backend a été redémarré après modification du `.env`

### "Provider not found" dans l'interface
Vérifier que le provider est bien répertorié dans le dropdown HTML.

### CORS error
Vérifier que le backend est en cours d'exécution et accessible sur le port 8000.

### "Could not authenticate with the API provider"
Vérifier que :
1. La clé API est valide et non expirée
2. Le fournisseur de services n'a pas changé le format de l'API
3. Les crédits sont disponibles (pour les services payants)

## Structure du backend

### `app.py`

Le fichier principal contient :

1. **Configuration FastAPI** : CORS, routes, configuration globale
2. **Modèles Pydantic** : Définition des requêtes/réponses
3. **Fonctions de chat** : Une pour chaque provider
4. **Routes** : `/health`, `/providers`, `/chat`

Chaque fonction de chat :
- Valide les inputs
- Appelle l'API du provider via `httpx`
- Gère les erreurs
- Retourne une réponse structurée

### `requirements.txt`

Dépendances Python minimales :
- `fastapi` : Framework web asynchrone
- `uvicorn` : Serveur ASGI
- `httpx` : Client HTTP asynchrone
- `python-dotenv` : Gestion des variables d'environnement
- `pydantic` : Validation des données

## Structure du frontend

### `index.html`

Contient :

1. **HTML** : Structure du chat
2. **CSS** : Design moderne type ChatGPT/Claude
3. **JavaScript** : Logique d'interaction et appels API

Features :
- Provider selector avec couleurs
- Message bubbles (user/assistant)
- Loading animation
- Error handling
- Responsive design
- Keyboard shortcuts

## Extensibilité

Pour ajouter un nouveau provider :

### Backend (app.py)

1. Ajouter la variable d'environnement
2. Créer une fonction `chat_<provider>()` asynchrone
3. Ajouter le provider au dictionnaire de routing

### Frontend (index.html)

1. Ajouter une option au select des providers
2. Ajouter les couleurs CSS correspondantes
3. Tester l'intégration

Exemple d'ajout de Mistral :

**Backend:**
```python
async def chat_mistral(message: str) -> dict:
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise ValueError("MISTRAL_API_KEY not configured")
    # ... implémentation similaire aux autres providers
```

**Frontend:**
```html
<option value="mistral">🟡 Mistral</option>
```

## Ressources pédagogiques

- FastAPI: https://fastapi.tiangolo.com/
- Httpx: https://www.python-httpx.org/
- Anthropic API: https://docs.anthropic.com/
- OpenAI API: https://platform.openai.com/docs/
- Docker: https://docs.docker.com/

## Évaluation

Le projet sera évalué sur :

1. **Fonctionnalité** (40%)
   - Tous les providers fonctionnent
   - Interface responsive
   - Gestion d'erreurs correcte

2. **Code quality** (30%)
   - Code lisible et commenté
   - Structure organisée
   - Pas d'API keys en dur

3. **Design** (20%)
   - Interface attrayante
   - UX fluide
   - Distinction visuelle des providers

4. **Documentation** (10%)
   - README clair
   - Code commenté
   - Instructions de setup

## Livrable

Soumettre sur Moodle :
- Repository Git (ou archive zip)
- Fichier `.env.example` rempli (SANS les clés réelles)
- Screenshot de l'interface en fonctionnement
- Rapport court (1-2 pages) expliquant les choix techniques

## Astuces

- Utiliser le Network tab du navigateur pour déboguer les requêtes
- Les API keys de test/démo sont souvent disponibles gratuitement
- Tester d'abord avec une simple requête depuis curl
- Vérifier les limites de taux (rate limits) des APIs
- Documenter les problèmes rencontrés et solutions trouvées

## Support

Pour toute question :
- Vérifier la documentation officielle de chaque fournisseur
- Consulter les logs du backend (`python -m uvicorn app:app --reload`)
- Vérifier les erreurs dans la console navigateur (F12)
