# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /app

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances dans une venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="EFREI Paris M2 DEV"
LABEL description="API FastAPI pour le prédicteur Iris - Classification de fleurs"

WORKDIR /app

# Copier la venv depuis le stage précédent
COPY --from=builder /opt/venv /opt/venv

# Copier l'application
COPY train_model.py .
COPY app.py .

# Définir la venv comme PATH
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Exposer le port
EXPOSE 8000

# Entraîner le modèle au démarrage, puis lancer l'API
CMD ["sh", "-c", "python train_model.py && python app.py"]
