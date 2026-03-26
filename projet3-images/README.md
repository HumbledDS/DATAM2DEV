# Projet 3 — Classificateur d'Images (Deep Learning)

## Objectifs du Projet

Ce projet vise à intégrer un modèle de Deep Learning pré-entraîné (MobileNetV2) dans une application web full-stack pour la classification d'images en temps réel.

### Compétences Visées
- Intégration d'un modèle IA pré-entraîné (TensorFlow/Keras)
- Architecture client-serveur avec FastAPI
- Upload et traitement d'images côté serveur
- Communication asynchrone frontend-backend
- Containerisation avec Docker et Docker Compose
- Interface utilisateur moderne avec drag-and-drop

## Contexte Technique

### Modèle Utilisé
- **Architecture** : MobileNetV2 (léger et rapide)
- **Jeu d'entraînement** : ImageNet (1000 classes)
- **Poids pré-entraînés** : Téléchargés automatiquement lors du premier démarrage
- **Avantage** : Pas d'entraînement nécessaire, modèle prêt à utiliser

### Stack Technique
- **Frontend** : HTML5, CSS3, JavaScript vanilla
- **Backend** : Python 3.11, FastAPI, TensorFlow/Keras
- **Déploiement** : Docker et Docker Compose
- **Modèle ML** : TensorFlow/Keras MobileNetV2

## Structure du Projet

```
projet3-images/
├── README.md                    # Ce fichier
├── docker-compose.yml          # Orchestration des services
├── frontend/
│   └── index.html              # Interface utilisateur complète
└── backend/
    ├── app.py                  # Application FastAPI
    ├── Dockerfile              # Image Docker backend
    └── requirements.txt        # Dépendances Python
```

## Installation et Démarrage

### Prérequis
- Docker et Docker Compose installés
- Connexion Internet (pour télécharger le modèle MobileNetV2)
- Environ 150 Mo d'espace disque pour le modèle

### Étapes de Démarrage

1. **Clonez ou téléchargez le projet**
   ```bash
   cd projet3-images
   ```

2. **Lancez l'application avec Docker Compose**
   ```bash
   docker-compose up --build
   ```

   Les services démarrent dans cet ordre :
   - Backend FastAPI sur `http://localhost:8000`
   - Frontend sur `http://localhost:3000`
   - Le modèle MobileNetV2 est téléchargé automatiquement

3. **Accédez à l'interface**
   Ouvrez votre navigateur et allez à `http://localhost:3000`

4. **Pour arrêter l'application**
   ```bash
   docker-compose down
   ```

### Alternatives (sans Docker)

Si vous préférez exécuter directement sur votre machine :

**Backend (Terminal 1)**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

**Frontend (Terminal 2)**
```bash
# Servez le frontend avec un serveur HTTP simple
cd frontend
python -m http.server 3000
```

## Utilisation de l'Application

### Interface Utilisateur

1. **Zone de dépôt d'image**
   - Glissez-déposez une image (drag-and-drop)
   - Ou cliquez pour sélectionner un fichier
   - Formats acceptés : JPEG, PNG, WebP, GIF

2. **Aperçu de l'image**
   - L'image sélectionnée s'affiche automatiquement
   - Redimensionnée et prévisualisée avant classification

3. **Classification**
   - Cliquez sur "Classifier" pour envoyer l'image au serveur
   - Attend la réponse du modèle

4. **Résultats**
   - Top 5 prédictions affichées
   - Classe (label en anglais)
   - Probabilité (en %) avec barre de progression visuelle
   - Résultats triés par confiance décroissante

### Exemples d'Images à Tester

Le modèle ImageNet reconnaît 1000 classes, notamment :
- Animaux : chat, chien, oiseau, éléphant, lion, etc.
- Objets : voiture, vélo, téléphone, ordinateur, chaise, etc.
- Paysages : montagne, plage, forêt, rue, etc.
- Aliments : pizza, hamburger, café, etc.

## Architecture de l'Application

### Backend (FastAPI)

```
GET  /health              → Vérifier la disponibilité du service
POST /predict             → Classifier une image
```

**Endpoint `/predict`**
- **Entrée** : Form data avec fichier image (multipart/form-data)
- **Traitement** :
  1. Réception du fichier image
  2. Redimensionnement à 224×224 pixels
  3. Normalisation (valeurs entre 0 et 1)
  4. Inférence avec MobileNetV2
  5. Décodage des résultats avec labels ImageNet
- **Sortie** : JSON avec top 5 prédictions
  ```json
  {
    "predictions": [
      {"class": "golden_retriever", "probability": 0.856},
      {"class": "Labrador_retriever", "probability": 0.102},
      ...
    ]
  }
  ```

### Frontend (HTML5 + JavaScript)

**Fonctionnalités principales** :
- Drag-and-drop avec feedback visuel
- Sélection de fichier via clic
- Prévisualisation d'image
- Envoi asynchrone au backend (fetch API)
- Affichage des résultats avec barres de probabilité
- Design responsive et thème sombre
- Gestion des erreurs et états de chargement

## Guide de Développement

### Étendre le Projet

#### 1. Ajouter un Nouveau Modèle
Modifiez `backend/app.py` pour charger un autre modèle TensorFlow/Keras :
```python
# Exemples d'alternatives
model = tf.keras.applications.ResNet50(weights='imagenet')  # ResNet50
model = tf.keras.applications.InceptionV3(weights='imagenet')  # InceptionV3
model = tf.keras.applications.EfficientNetB0(weights='imagenet')  # EfficientNet
```

#### 2. Ajouter des Métadonnées
Créez un fichier `backend/classes.json` avec des descriptions :
```json
{
  "golden_retriever": "Golden Retriever - Race de chien amical et loyal",
  "cat": "Chat - Animal domestique carnivore"
}
```

#### 3. Optimiser les Performances
- Ajouter la mise en cache des résultats (Redis)
- Implémenter la pagination pour les résultats
- Ajouter la quantification du modèle

#### 4. Améliorations Frontend
- Historique des classifications
- Export des résultats en JSON/CSV
- Mode sombre/clair personnalisé
- Support du webcam en temps réel

### Dépannage

**Le modèle ne se télécharge pas**
- Vérifiez votre connexion Internet
- Vérifiez l'espace disque disponible (min. 150 Mo)
- Les logs de démarrage du container indiquent la progression

**Erreur 413 "Payload Too Large"**
- Compressez votre image avant l'upload
- Réduisez la taille du fichier (< 10 Mo)

**Lenteur de classification**
- C'est normal pour les premières requêtes
- Le modèle se réchauffe progressivement
- Temps typique : 500 ms à 2 secondes

**Erreur de CORS**
- Vérifiez que le backend est accessible
- Vérifiez les ports 8000 (backend) et 3000 (frontend)

## Ressources d'Apprentissage

### Concepts Fondamentaux
- **Transfer Learning** : Utiliser un modèle pré-entraîné
- **ImageNet** : Base de données de 14 millions d'images avec 1000 classes
- **MobileNetV2** : Architecture optimisée pour mobile et edge computing

### Documentation Officielle
- [TensorFlow Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Docker Compose](https://docs.docker.com/compose/)

### Tutoriels Recommandés
1. Comprendre la classification d'images avec CNN
2. Transfer learning avec TensorFlow
3. Déploiement de modèles ML en production

## Évaluation du Projet

### Critères de Succès
- ✓ Backend fonctionne et charge le modèle correctement
- ✓ Endpoint `/predict` retourne des résultats valides
- ✓ Frontend affiche les images et résultats correctement
- ✓ Drag-and-drop fonctionne sans erreur
- ✓ Top 5 prédictions affichées avec probabilités
- ✓ Docker Compose lance tous les services sans erreur
- ✓ Interface responsive et usable

### Points Bonus
- Affichage d'une confidence score avec couleur (rouge/orange/vert)
- Historique des classifications
- Support pour plusieurs images simultanées
- Optimisation des performances
- Tests unitaires du backend
- Documentation améliorée

## Questions Fréquemment Posées

**Q: Comment le modèle reconnaît-il les images ?**
R: MobileNetV2 extrait des caractéristiques visuelles (formes, textures, couleurs) et les compare aux 1000 classes ImageNet pour assigner une probabilité à chacune.

**Q: Pourquoi le modèle se trompe-t-il parfois ?**
R: Le modèle ImageNet est optimisé pour ces 1000 classes. Pour des domaines spécialisés, un entraînement custom serait nécessaire.

**Q: Puis-je entraîner mon propre modèle ?**
R: Oui, mais ce n'est pas l'objectif de ce projet. Consultez les ressources d'apprentissage.

**Q: Combien d'images puis-je traiter en même temps ?**
R: Une à la fois avec cette architecture. Pour le batch processing, modifiez le backend.

## Auteurs et Licence

Ce projet est fourni à titre éducatif pour le cours IA/ML/DL de l'EFREI Paris.

---

**Bon développement ! 🚀**

Pour toute question, consultez les logs Docker :
```bash
docker-compose logs -f backend
docker-compose logs -f frontend
```
