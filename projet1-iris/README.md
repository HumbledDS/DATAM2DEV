# Projet 1 — Prédicteur Iris

## Objectif
Développer une application web complète de classification d'espèces d'Iris en utilisant un modèle ML (Random Forest) avec une API FastAPI et une interface utilisateur moderne.

**Compétences visées :**
- Entraînement d'un modèle de machine learning (scikit-learn)
- Création d'une API REST (FastAPI)
- Intégration frontend-backend (fetch API)
- Containerisation Docker
- Orchestration multi-conteneurs (docker-compose)

---

## Architecture du Projet

```
projet1-iris/
├── backend/
│   ├── train_model.py          # Script d'entraînement du modèle
│   ├── app.py                  # API FastAPI
│   ├── requirements.txt         # Dépendances Python
│   ├── Dockerfile              # Configuration Docker pour le backend
│   └── iris_model.pkl          # Modèle entraîné (généré)
├── frontend/
│   ├── index.html              # Interface utilisateur
│   └── nginx.conf              # Configuration nginx (généré)
└── docker-compose.yml          # Orchestration des services
```

---

## Description des Composants

### Backend (FastAPI)

L'API backend expose les endpoints suivants :

#### `GET /health`
Vérifie que le service est opérationnel.

**Réponse :**
```json
{ "status": "healthy", "model_loaded": true }
```

#### `POST /predict`
Prédit la classe d'Iris basée sur 4 features.

**Payload (JSON) :**
```json
{
  "sepal_length": 5.5,
  "sepal_width": 3.5,
  "petal_length": 1.3,
  "petal_width": 0.3
}
```

**Réponse :**
```json
{
  "prediction": "setosa",
  "probabilities": {
    "setosa": 0.98,
    "versicolor": 0.02,
    "virginica": 0.00
  }
}
```

**Paramètres :**
- `sepal_length` : longueur du sépale (4.3 - 7.9 cm)
- `sepal_width` : largeur du sépale (2.0 - 4.4 cm)
- `petal_length` : longueur du pétale (1.0 - 6.9 cm)
- `petal_width` : largeur du pétale (0.1 - 2.5 cm)

### Frontend (HTML5)

Interface responsive avec :
- 4 sliders pour saisir les paramètres des fleurs
- Affichage en temps réel des valeurs
- Bouton "Prédire" pour appeler l'API
- Zone de résultats avec les probabilités (barres visuelles)
- Gestion des états de chargement (spinner)
- Design moderne avec gradients et cartes

---

## Instructions d'Exécution

### Prérequis

- Docker et Docker Compose installés
- Port 8000 (backend) et 5000 (frontend) disponibles

### Démarrage Local (sans Docker)

#### 1. Entraîner le modèle
```bash
cd backend
pip install -r requirements.txt
python train_model.py
```

Cela génère `iris_model.pkl`.

#### 2. Lancer l'API
```bash
python app.py
```

Le backend démarre sur `http://localhost:8000`.

#### 3. Lancer le frontend
Ouvrez directement `frontend/index.html` dans un navigateur, ou servez-le avec un serveur HTTP :

```bash
# Avec Python 3
python -m http.server 5000 --directory frontend
```

Accédez à `http://localhost:5000`.

### Démarrage avec Docker Compose

```bash
docker-compose up --build
```

- Backend : `http://localhost:8000`
- Frontend : `http://localhost:5000`
- API docs (Swagger) : `http://localhost:8000/docs`

Pour arrêter :
```bash
docker-compose down
```

---

## Fichiers Importants

### `backend/train_model.py`
Script qui :
1. Charge le dataset Iris depuis scikit-learn
2. Entraîne un modèle Random Forest
3. Sauvegarde le modèle en fichier `.pkl`
4. Affiche des métriques d'évaluation (précision, rapport de classification)

### `backend/app.py`
API FastAPI qui :
1. Charge le modèle à la démarrage
2. Expose l'endpoint `/health`
3. Expose l'endpoint `/predict` avec validation Pydantic
4. Active CORS pour les requêtes cross-origin
5. Fournit la documentation Swagger automatique

### `backend/requirements.txt`
Dépendances Python nécessaires :
- `fastapi` : framework web asynchrone
- `uvicorn` : serveur ASGI
- `scikit-learn` : machine learning
- `joblib` : sérialisation de modèles
- `pydantic` : validation de schémas
- `python-multipart` : parsing de formulaires (optionnel mais recommandé)

### `frontend/index.html`
Page HTML complète avec :
- HTML5 sémantique
- CSS moderne (gradients, flexbox, animations)
- JavaScript vanilla pour l'appel API
- Gestion des erreurs et états de chargement
- Design responsive (mobile-friendly)

---

## Tâches Étudiants (Optionnel)

Pour approfondir le projet, les étudiants peuvent :

1. **Améliorer le modèle** : tester d'autres algorithmes (SVM, XGBoost, KNN)
2. **Ajouter de la persistance** : stocker l'historique des prédictions en base de données
3. **Déployer en production** : utiliser AWS/GCP/Heroku
4. **Améliorer le frontend** : ajouter des graphiques (Chart.js), un historique des requêtes
5. **Ajouter l'authentification** : JWT tokens pour sécuriser l'API
6. **Optimiser le modèle** : hyper-paramètres tuning, cross-validation

---

## Dépannage

### Le modèle ne se charge pas
Vérifiez que `backend/iris_model.pkl` existe. Si non, exécutez `python train_model.py`.

### Erreur CORS (frontend ne peut pas appeler backend)
Vérifiez que `app.py` configure CORS correctement avec `allow_origins=["*"]`.

### Port déjà utilisé
Changez les ports dans `docker-compose.yml` ou utilisez :
```bash
lsof -i :8000  # Trouver le processus sur port 8000
kill -9 <PID>
```

### Frontend affiche une erreur de connexion
Assurez-vous que le backend tourne sur `http://localhost:8000`. Vérifiez dans la console du navigateur (F12).

---

## Ressources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Iris Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [Docker Documentation](https://docs.docker.com/)
- [MDN Web Docs - Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)

---

**Auteur :** Cours IA-ML-DL
**Date :** 2026
**Niveau :** M2 DEV
