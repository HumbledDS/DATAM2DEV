# Index — Projet 1: Prédicteur Iris

## Pour Commencer

1. **Lire d'abord** : [QUICK_START.md](QUICK_START.md) (5 minutes)
   - Démarrage rapide avec Docker ou local
   - Endpoints API avec exemples curl

2. **Ensuite lire** : [README.md](README.md) (documentation complète)
   - Objectifs du projet
   - Architecture détaillée
   - Instructions complètes
   - Dépannage

## Documents de Référence

- [STRUCTURE.txt](STRUCTURE.txt) - Vue d'ensemble technique du projet
- [LEARNING_OUTCOMES.md](LEARNING_OUTCOMES.md) - Compétences acquises et auto-évaluation
- [INDEX.md](INDEX.md) - Ce fichier

## Structure du Projet

```
projet1-iris/
├── README.md                    ← Lire d'abord (complet)
├── QUICK_START.md              ← Pour démarrer vite
├── LEARNING_OUTCOMES.md        ← Résultats d'apprentissage
├── STRUCTURE.txt               ← Architecture technique
├── docker-compose.yml          ← Orchestration
├── backend/
│   ├── train_model.py          ← Entraînement ML
│   ├── app.py                  ← API FastAPI
│   ├── requirements.txt         ← Dépendances
│   └── Dockerfile              ← Containerisation
└── frontend/
    ├── index.html              ← Interface web
    └── nginx.conf              ← Serveur web
```

## Démarrage Rapide

### Option 1: Docker (Recommandé)
```bash
docker-compose up --build
# Frontend:  http://localhost:5000
# Backend:   http://localhost:8000
# API Docs:  http://localhost:8000/docs
```

### Option 2: Local
```bash
# Terminal 1
cd backend
pip install -r requirements.txt
python train_model.py
python app.py

# Terminal 2
cd frontend
python -m http.server 5000
```


## Fichiers Clés

### Backend
- **train_model.py** : Entraîne Random Forest sur Iris dataset
- **app.py** : API FastAPI avec /health et /predict
- **requirements.txt** : Dépendances (fastapi, scikit-learn, etc.)
- **Dockerfile** : Build Docker multi-stage

### Frontend
- **index.html** : Interface web complète (HTML + CSS + JS)
  - 4 sliders pour les features
  - Fetch API pour appels backend
  - Design moderne et responsive
  - Gestion des erreurs

### Deployment
- **docker-compose.yml** : Orchestration (backend + frontend)
- **frontend/nginx.conf** : Configuration serveur web

## Endpoints API

### GET /health
Vérifier l'état du service
```bash
curl http://localhost:8000/health
```

### POST /predict
Prédire une classe Iris
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.5,
    "sepal_width": 3.5,
    "petal_length": 1.3,
    "petal_width": 0.3
  }'
```

## Dépannage

### Backend ne démarre pas
→ Voir [README.md](README.md#dépannage)

### Frontend ne se connecte pas au backend
→ Vérifier que le backend est sur http://localhost:8000

### Port déjà utilisé
→ Modifier les ports dans docker-compose.yml

## Prochaines Étapes

Voir [LEARNING_OUTCOMES.md](LEARNING_OUTCOMES.md#extensions-possibles) pour :
- Modifications simples (CSS)
- Nouvelles features (historique, export)
- Améliorations techniques (BD, auth)
- Déploiement production (AWS, Heroku)

## Ressources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Iris Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [Docker Documentation](https://docs.docker.com/)
- [MDN Web Docs](https://developer.mozilla.org/)

---

**Projet créé pour M2 DEV - EFREI Paris**

