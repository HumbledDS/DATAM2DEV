# Démarrage Rapide — Prédicteur Iris

## 5 minutes pour démarrer

### Option 1: Docker Compose (Recommandé)

```bash
# Aller dans le dossier du projet
cd projet1-iris

# Démarrer tous les services
docker-compose up --build

# Le projet est prêt!
# Frontend:  http://localhost:5000
# Backend:   http://localhost:8000
# API Docs:  http://localhost:8000/docs
```

### Option 2: Démarrage Local (sans Docker)

**Terminal 1 - Backend:**
```bash
cd backend
pip install -r requirements.txt
python train_model.py
python app.py
# Le backend est maintenant sur http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
python -m http.server 5000
# Le frontend est maintenant sur http://localhost:5000
```

---

## Architecture Simplifiée

```
Navigateur (http://localhost:5000)
    ↓
    │ HTML/CSS/JS vanilla
    │ (sliders, fetch API)
    ↓
Frontend (nginx sur port 80)
    ↓
    │ HTTP POST /predict
    │ JSON: sepal_length, sepal_width, petal_length, petal_width
    ↓
Backend (FastAPI sur port 8000)
    ├─ Charge le modèle Iris
    ├─ Prédiction (Random Forest)
    └─ Retourne: {prediction, probabilities}
```

---

## API Endpoints

### 1. Vérifier la santé du service

```bash
curl http://localhost:8000/health
```

Réponse:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Faire une prédiction

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

Réponse:
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

---

## Fichiers Importants

| Fichier | Rôle |
|---------|------|
| `README.md` | Documentation complète |
| `backend/train_model.py` | Entraîne le modèle ML |
| `backend/app.py` | Expose l'API FastAPI |
| `frontend/index.html` | Interface web |
| `docker-compose.yml` | Lance tous les services |

---

## Dépannage

### "Cannot connect to backend"
→ Assurez-vous que le backend est démarré:
```bash
cd backend && python app.py
```

### "Model not found"
→ Entraînez le modèle:
```bash
cd backend && python train_model.py
```

### Port déjà utilisé
→ Changez le port dans `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Backend sur 8001 au lieu de 8000
  - "5001:80"    # Frontend sur 5001 au lieu de 5000
```

---

## Prochaines Étapes (Optionnel)

1. **Améliorer le modèle**: Tester SVM, KNN, XGBoost
2. **Ajouter une base de données**: PostgreSQL + SQLAlchemy
3. **Authentification**: JWT tokens sur l'API
4. **Frontend avancé**: Ajouter historique, graphiques (Chart.js)
5. **Déployer**: AWS, Heroku, DigitalOcean

---



