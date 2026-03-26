# Résumé des enseignements — Projet Prédicteur Iris

## Compétences 

### Machine Learning (ML)
- **Classification** : Utiliser Random Forest pour la classification multi-classe
- **Entraînement** : Train/test split, évaluation (accuracy, précision, recall)
- **Sérialisation** : Sauvegarder/charger des modèles avec joblib
- **Dataset** : Travailler avec le dataset Iris (4 features, 3 classes)

### Backend (API REST)
- **FastAPI** : Framework web moderne et performant
- **REST API** : Design d'endpoints RESTful (/health, /predict)
- **Validation** : Pydantic pour la validation des données
- **CORS** : Autoriser les requêtes cross-origin du frontend
- **Documentation** : OpenAPI/Swagger auto-généré
- **Error Handling** : Gestion robuste des erreurs HTTP

### Frontend (HTML/CSS/JavaScript)
- **HTML5** : Sémantique moderne (sections, labels, inputs)
- **CSS** : Gradients, flexbox, animations, responsive design
- **JavaScript** : Fetch API, event listeners, gestion d'état
- **UX** : Sliders, spinners, barres de probabilité animées
- **Accessibilité** : Contraintes min/max, labels explicites

### DevOps & Deployment
- **Docker** : Containerisation, multi-stage builds, optimisation d'images
- **docker-compose** : Orchestration multi-conteneurs, networks, health checks
- **nginx** : Serveur web, gzip compression, static file serving
- **Port forwarding** : Exposition de ports, communication inter-conteneurs

---

## Architecture & Design Patterns

### Architecture 3-Tiers
```
Presentation Layer (Frontend)
    ↓ HTTP/JSON
Application Layer (FastAPI Backend)
    ↓ Python/ML
Data Layer (Iris Model)
```

### Patterns Utilisés
- **Model-View-Controller** (MVC) : Séparation frontend/backend
- **REST API** : Stateless communication
- **Pydantic Validation** : Type safety et validation côté serveur
- **Health Check** : Monitoring de service
- **Containerization** : Isolation et portabilité

---

## Points Clés Pédagogiques

### 1. Machine Learning
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X)
```

**Concept** : Random Forest agrège plusieurs arbres de décision pour améliorer la robustesse

### 2. API REST
```python
@app.post("/predict")
async def predict(features: IrisFeatures):
    # Validation automatique avec Pydantic
    # Prédiction du modèle
    # Retour des probabilités
    return PredictionResponse(...)
```

**Concept** : API stateless exposant une logique métier via HTTP

### 3. Frontend-Backend Communication
```javascript
const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
});
```

**Concept** : Fetch API pour appels asynchrones du frontend vers le backend

### 4. Containerization
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

**Concept** : Isolation et portabilité de l'application

---

## Flux de Données Complet

```
1. FRONTEND (Browser)
   ↓ User ajuste 4 sliders
   ↓ Click "Prédire"
   
2. JAVASCRIPT
   ↓ Récupère les valeurs
   ↓ Prépare JSON payload
   ↓ Appel fetch → POST /predict
   
3. RÉSEAU HTTP
   ↓ Request: POST /predict
   ↓ Body: {"sepal_length": 5.5, ...}
   
4. BACKEND (FastAPI)
   ↓ Reçoit la requête
   ↓ Valide avec Pydantic
   ↓ Appelle model.predict()
   ↓ Calcule probabilities
   
5. RÉPONSE HTTP
   ↓ Body: {"prediction": "setosa", "probabilities": {...}}
   
6. FRONTEND (JavaScript)
   ↓ Parse JSON
   ↓ Affiche résultat
   ↓ Anime barres de probabilité
```

---


---

## Bonnes Pratiques à Retenir

### Backend
✓ Toujours valider les inputs (Pydantic)
✓ Documenter l'API (OpenAPI/Swagger)
✓ Gérer les erreurs correctement (HTTP status codes)
✓ Activer CORS si frontend ≠ backend
✓ Tester les endpoints (curl, Postman)

### Frontend
✓ Utiliser fetch pour les appels asynchrones
✓ Afficher des spinners pendant le loading
✓ Gérer les erreurs utilisateur
✓ Faire du design responsive (mobile-first)
✓ Vérifier la santé du backend

### DevOps
✓ Utiliser Docker pour la portabilité
✓ docker-compose pour multi-conteneurs
✓ Health checks pour la résilience
✓ Volumes pour le développement
✓ Logs structurés pour le debug

---

## Ressources d'Apprentissage

### Official Documentation
- [FastAPI](https://fastapi.tiangolo.com/)
- [scikit-learn](https://scikit-learn.org/)
- [Docker](https://docs.docker.com/)
- [MDN Web Docs](https://developer.mozilla.org/)

### Concepts Clés
- Random Forest : Ensemble learning, bagging
- REST API : Stateless communication, HTTP methods
- CORS : Cross-origin requests, security
- Docker : Containerization, image optimization

### Outils Recommandés
- **API Testing** : Postman, Insomnia, curl
- **Code Editor** : VSCode, PyCharm, WebStorm
- **Container Tools** : Docker Desktop, Docker Compose
- **Browser DevTools** : Console, Network, Elements

---

