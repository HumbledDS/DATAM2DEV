"""
app.py - Application FastAPI pour la détection de spam

API REST avec endpoint POST /predict pour classifier les emails en spam/ham.
Le modèle et le vectorizer sont chargés au démarrage de l'application.
"""

import os
import pickle
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Configuration
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Vérifier que les modèles existent
if not MODELS_DIR.exists():
    raise FileNotFoundError(
        f"Le répertoire 'models' n'existe pas. "
        f"Veuillez exécuter train_model.py d'abord."
    )

VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
MODEL_PATH = MODELS_DIR / "model.pkl"

if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Les fichiers du modèle n'existent pas. "
        f"Veuillez exécuter train_model.py d'abord."
    )


# Schémas Pydantic
class PredictionRequest(BaseModel):
    """Schéma pour une requête de prédiction"""
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Click here to win a free iPhone now! Limited offer!"
            }
        }


class PredictionResponse(BaseModel):
    """Schéma pour une réponse de prédiction"""
    prediction: str
    confidence: float
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "SPAM",
                "confidence": 0.92,
                "message": "Cet email semble être du spam."
            }
        }


# Initialiser l'application FastAPI
app = FastAPI(
    title="Spam Detector API",
    description="API de détection de spam utilisant TF-IDF et Logistic Regression",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Charger le modèle et le vectorizer au démarrage
@app.on_event("startup")
def load_model():
    """Charge le modèle et le vectorizer au démarrage de l'application"""
    global vectorizer, model

    try:
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        print(f"✓ Vectorizer chargé depuis {VECTORIZER_PATH}")

        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Modèle chargé depuis {MODEL_PATH}")
    except Exception as e:
        print(f"✗ Erreur lors du chargement des modèles: {e}")
        raise


# Variables globales pour le modèle et le vectorizer
vectorizer = None
model = None


# Routes API

@app.get("/", tags=["Frontend"])
async def get_frontend():
    """Sert la page HTML du frontend"""
    html_path = FRONTEND_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Le fichier frontend/index.html n'a pas été trouvé"
        )
    return FileResponse(html_path)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_spam(request: PredictionRequest) -> PredictionResponse:
    """
    Prédit si un texte est du spam ou du ham.

    Args:
        request: Contient le texte à analyser

    Returns:
        PredictionResponse: Prédiction (SPAM/HAM), confiance et message explicatif

    Raises:
        HTTPException: Si le texte est vide ou si une erreur se produit
    """

    # Validation du texte
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Le texte ne peut pas être vide"
        )

    try:
        # Vectoriser le texte
        text_vectorized = vectorizer.transform([request.text])

        # Prédire la classe
        prediction_label = model.predict(text_vectorized)[0]

        # Obtenir les probabilités
        probabilities = model.predict_proba(text_vectorized)[0]

        # Extraire la confiance pour la classe prédite
        confidence = float(probabilities[prediction_label])

        # Convertir le label en texte
        prediction_text = "SPAM" if prediction_label == 1 else "HAM"

        # Créer un message explicatif
        if prediction_label == 1:
            message = (
                f"Cet email semble être du spam (confiance: {confidence*100:.1f}%). "
                "Soyez prudent avec ce message."
            )
        else:
            message = (
                f"Cet email semble légitime (confiance: {confidence*100:.1f}%). "
                "Il devrait être sûr d'interagir avec."
            )

        return PredictionResponse(
            prediction=prediction_text,
            confidence=confidence,
            message=message
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


@app.get("/health", tags=["Health"])
async def health_check():
    """Vérifie que l'API est opérationnelle"""
    return {
        "status": "healthy",
        "model_loaded": vectorizer is not None and model is not None
    }


# Info des modèles
@app.get("/model-info", tags=["Info"])
async def get_model_info():
    """Retourne des informations sur le modèle"""
    return {
        "model_type": "Logistic Regression",
        "vectorizer_type": "TF-IDF",
        "max_features": 1000,
        "ngram_range": (1, 2),
        "classes": ["HAM", "SPAM"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
