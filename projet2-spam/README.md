# Projet 2 — Détecteur de Spam

## Objectif

Construire une application web complète pour détecter les emails spam à l'aide du machine learning. Ce projet intègre un modèle NLP (Natural Language Processing) entraîné avec TF-IDF et Logistic Regression dans une application web fullstack avec FastAPI (backend) et HTML/CSS/JavaScript (frontend).

## Architecture

```
projet2-spam/
├── README.md
├── docker-compose.yml
├── backend/
│   ├── app.py              # API FastAPI
│   ├── train_model.py      # Script d'entraînement
│   ├── requirements.txt
│   ├── Dockerfile
│   └── models/             # Sera créé après l'entraînement
│       ├── vectorizer.pkl
│       └── model.pkl
└── frontend/
    └── index.html          # Interface utilisateur
```

## Compétences Couvertes

- **Machine Learning**: TF-IDF vectorization, Logistic Regression
- **NLP**: Feature extraction, Text preprocessing
- **Backend**: FastAPI, CORS, REST API
- **Frontend**: HTML5, CSS3, JavaScript vanilla
- **DevOps**: Docker, Docker Compose

## Démarrage Rapide

### Avec Docker (Recommandé)

```bash
docker-compose up --build
```

Puis accédez à `http://localhost:8000` dans votre navigateur.

### Sans Docker

1. **Entraîner le modèle**:
```bash
cd backend
python train_model.py
```

2. **Installer les dépendances**:
```bash
pip install -r requirements.txt
```

3. **Lancer l'API**:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

4. **Ouvrir l'interface**:
```bash
cd ../frontend
# Servir index.html sur http://localhost:8000
# Utilisez un serveur simple: python -m http.server 8000
```

## Utilisation

1. **Copier un email** ou du texte dans la zone de texte
2. **Cliquer sur "Analyser"**
3. **Consulter le résultat**:
   - 🟢 **HAM (Légitime)**: Email sûr
   - 🔴 **SPAM**: Email détecté comme spam
   - **Confiance**: Score de probabilité (0-100%)

## Exemples d'Emails pour Tester

### Spam (À copier/coller)

```
Subject: URGENT! Claim your FREE CASH NOW!!!

Congratulations! You have been selected to receive $5,000 in free money! Click here immediately to claim your prize. Do not share this with anyone. Limited time offer - expires in 1 hour. This is a one-time opportunity!

Act now: www.totallylegalsite.ru/claim-money?token=xyz
```

### Ham (À copier/coller)

```
Hi John,

I hope this email finds you well. I wanted to follow up on our meeting last Thursday about the Q2 project roadmap. Could you please send me the updated timeline when you get a chance?

Looking forward to hearing from you.

Best regards,
Sarah
```

## Détails Techniques

### Modèle ML

- **Algorithme**: Logistic Regression
- **Vectorisation**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Dataset d'entraînement**: 20 exemples (10 spam, 10 ham)
- **Séparation**: 80% train, 20% test
- **Métrique**: Accuracy

### API Endpoints

#### POST `/predict`

**Requête**:
```json
{
  "text": "Your email content here"
}
```

**Réponse (Spam)**:
```json
{
  "prediction": "SPAM",
  "confidence": 0.92,
  "message": "Cet email semble être du spam."
}
```

**Réponse (Ham)**:
```json
{
  "prediction": "HAM",
  "confidence": 0.87,
  "message": "Cet email semble légitime."
}
```

#### GET `/`

Sert le fichier `index.html`

## Structure du Code Backend

### `train_model.py`

1. Crée un dataset d'entraînement d'exemples d'emails
2. Vectorise le texte avec TF-IDF
3. Entraîne un modèle Logistic Regression
4. Évalue les performances sur un ensemble de test
5. Sauvegarde le vectorizer et le modèle en pickle

### `app.py`

1. Initialise FastAPI
2. Configure CORS pour permettre les requêtes du frontend
3. Charge le vectorizer et le modèle au démarrage
4. Expose l'endpoint `/predict` pour les prédictions
5. Sert le fichier HTML statique

## Améliorations Possibles

- **Dataset plus large**: Utiliser le [UCI Spam Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Modèle plus avancé**: SVM, Random Forest, ou Neural Networks
- **Deep Learning**: LSTM ou BERT pour les embeddings
- **Frontend**: Ajouter historique des analyses, visualisations
- **Validation**: Ajouter des métriques de performance (precision, recall, F1)
- **Preprocessing avancé**: Lemmatization, stopwords removal, etc.

## Livrables Attendus

1. ✅ Modèle entraîné sauvegardé
2. ✅ API fonctionnelle testable
3. ✅ Interface utilisateur intuitive
4. ✅ Application dockerisée
5. ✅ Documentation complète

## Ressources

- [Scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [UCI Spam Datasets](https://archive.ics.uci.edu/ml/)

## Notes pour les Étudiants

- Le modèle initial est entraîné sur un petit dataset pour la démonstration
- N'hésitez pas à améliorer le dataset ou le modèle
- Testez avec vos propres emails (attention aux données personnelles)
- Documentez toute modification apportée au code

---

**Durée estimée du projet**: 4-6 heures
**Difficulté**: Intermédiaire (★★★☆☆)
