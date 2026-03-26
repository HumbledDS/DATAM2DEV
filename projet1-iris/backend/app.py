#!/usr/bin/env python3
"""
API FastAPI pour le prédicteur Iris
Expose les endpoints /health et /predict
"""

import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict
import uvicorn

# ============================================================================
# Configuration de l'application FastAPI
# ============================================================================

app = FastAPI(
    title="Prédicteur Iris API",
    description="API de classification d'espèces d'Iris avec Machine Learning",
    version="1.0.0"
)

# ============================================================================
# Configuration CORS (Cross-Origin Resource Sharing)
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser les requêtes de n'importe quelle origine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Chargement du modèle
# ============================================================================

# Chemins des fichiers du modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'iris_model.pkl')
CLASSES_PATH = os.path.join(os.path.dirname(__file__), 'iris_classes.pkl')

# Charger le modèle et les classes
model = None
class_names = None

def load_model():
    """Charger le modèle et les noms de classes au démarrage"""
    global model, class_names

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Modèle non trouvé: {MODEL_PATH}\n"
            "Veuillez exécuter 'python train_model.py' d'abord."
        )

    if not os.path.exists(CLASSES_PATH):
        raise RuntimeError(
            f"Fichier de classes non trouvé: {CLASSES_PATH}\n"
            "Veuillez exécuter 'python train_model.py' d'abord."
        )

    model = joblib.load(MODEL_PATH)
    class_names = joblib.load(CLASSES_PATH)
    print(f"[INFO] Modèle chargé: {MODEL_PATH}")
    print(f"[INFO] Classes: {class_names}")

# ============================================================================
# Schémas Pydantic pour validation
# ============================================================================

class IrisFeatures(BaseModel):
    """Schéma pour les features d'une fleur Iris"""
    sepal_length: float = Field(
        ...,
        ge=4.0,
        le=8.0,
        description="Longueur du sépale en cm (4.0 - 8.0)"
    )
    sepal_width: float = Field(
        ...,
        ge=2.0,
        le=4.5,
        description="Largeur du sépale en cm (2.0 - 4.5)"
    )
    petal_length: float = Field(
        ...,
        ge=1.0,
        le=7.0,
        description="Longueur du pétale en cm (1.0 - 7.0)"
    )
    petal_width: float = Field(
        ...,
        ge=0.1,
        le=2.5,
        description="Largeur du pétale en cm (0.1 - 2.5)"
    )

    class Config:
        """Configuration du schéma"""
        json_schema_extra = {
            "example": {
                "sepal_length": 5.5,
                "sepal_width": 3.5,
                "petal_length": 1.3,
                "petal_width": 0.3
            }
        }

class HealthResponse(BaseModel):
    """Schéma pour la réponse de santé"""
    status: str
    model_loaded: bool

class PredictionResponse(BaseModel):
    """Schéma pour la réponse de prédiction"""
    prediction: str
    probabilities: Dict[str, float]

# ============================================================================
# Événements du cycle de vie
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Appelé au démarrage de l'application"""
    print("[INFO] Démarrage de l'API Prédicteur Iris...")
    try:
        load_model()
        print("[INFO] API prête à recevoir les requêtes")
    except RuntimeError as e:
        print(f"[ERREUR] Impossible de charger le modèle: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Appelé à l'arrêt de l'application"""
    print("[INFO] Arrêt de l'API Prédicteur Iris")

# ============================================================================
# Endpoints
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérifier l'état du service",
    tags=["Status"]
)
async def health_check():
    """
    Endpoint pour vérifier que l'API est opérationnelle.

    Retourne:
    - status: "healthy" si tout va bien
    - model_loaded: True si le modèle est chargé
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prédire l'espèce d'Iris",
    tags=["Prediction"]
)
async def predict(features: IrisFeatures):
    """
    Endpoint pour prédire l'espèce d'une fleur Iris.

    Args:
    - sepal_length: Longueur du sépale (cm)
    - sepal_width: Largeur du sépale (cm)
    - petal_length: Longueur du pétale (cm)
    - petal_width: Largeur du pétale (cm)

    Retourne:
    - prediction: L'espèce prédite (setosa, versicolor, virginica)
    - probabilities: Probabilités pour chaque classe
    """

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Modèle non chargé. Veuillez relancer l'API."
        )

    # Construire le vecteur de features
    X = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]

    # Faire la prédiction
    prediction_idx = model.predict(X)[0]
    prediction_class = class_names[prediction_idx]

    # Obtenir les probabilités
    probabilities_array = model.predict_proba(X)[0]
    probabilities = {
        class_names[i]: float(prob)
        for i, prob in enumerate(probabilities_array)
    }

    return PredictionResponse(
        prediction=prediction_class,
        probabilities=probabilities
    )

@app.get(
    "/",
    tags=["Info"]
)
async def root():
    """Endpoint racine avec lien vers la documentation"""
    return {
        "message": "Bienvenue à l'API Prédicteur Iris",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
        "health": "/health"
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
