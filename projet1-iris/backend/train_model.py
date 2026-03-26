#!/usr/bin/env python3
"""
Script d'entraînement du modèle Iris
Entraîne un Random Forest sur le dataset Iris et le sauvegarde en fichier pkl
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def train_iris_model():
    """
    Entraîne un modèle Random Forest sur le dataset Iris
    et le sauvegarde dans iris_model.pkl
    """

    print("="*60)
    print("Entraînement du modèle Iris avec Random Forest")
    print("="*60)

    # Charger le dataset Iris
    print("\n1. Chargement du dataset Iris...")
    iris = load_iris()
    X = iris.data  # Features: sepal_length, sepal_width, petal_length, petal_width
    y = iris.target  # Target: 0=setosa, 1=versicolor, 2=virginica

    print(f"   - Nombre d'échantillons: {X.shape[0]}")
    print(f"   - Nombre de features: {X.shape[1]}")
    print(f"   - Classe de target: {iris.target_names}")
    print(f"   - Distribution des classes: {[sum(y==i) for i in range(3)]}")

    # Diviser en train/test
    print("\n2. Division train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   - Ensemble d'entraînement: {X_train.shape[0]} samples")
    print(f"   - Ensemble de test: {X_test.shape[0]} samples")

    # Entraîner le modèle
    print("\n3. Entraînement du Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("   - Modèle entraîné avec succès")

    # Évaluation
    print("\n4. Évaluation du modèle...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   - Précision (Accuracy): {accuracy:.4f}")

    print("\n   - Rapport de classification:")
    print(classification_report(
        y_test, y_pred,
        target_names=iris.target_names,
        digits=4
    ))

    print("\n   - Matrice de confusion:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Feature importance
    print("\n5. Importance des features:")
    for name, importance in zip(iris.feature_names, model.feature_importances_):
        print(f"   - {name}: {importance:.4f}")

    # Sauvegarder le modèle
    print("\n6. Sauvegarde du modèle...")
    model_path = os.path.join(os.path.dirname(__file__), 'iris_model.pkl')
    joblib.dump(model, model_path)
    print(f"   - Modèle sauvegardé: {model_path}")

    # Sauvegarder aussi les noms de classes
    classes_path = os.path.join(os.path.dirname(__file__), 'iris_classes.pkl')
    joblib.dump(iris.target_names, classes_path)
    print(f"   - Classes sauvegardées: {classes_path}")

    print("\n" + "="*60)
    print("Entraînement terminé avec succès!")
    print("="*60)

    return model, iris.target_names

if __name__ == "__main__":
    train_iris_model()
