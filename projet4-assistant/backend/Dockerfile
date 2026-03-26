# Multi-stage build pour optimiser la taille de l'image
FROM python:3.11-slim as builder

WORKDIR /app

# Installer les dépendances de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage final
FROM python:3.11-slim

WORKDIR /app

# Créer un utilisateur non-root pour des raisons de sécurité
RUN useradd -m -u 1000 appuser

# Copier les dépendances depuis le builder
COPY --from=builder /root/.local /home/appuser/.local

# Copier le code application
COPY app.py .
COPY .env.example .env.example

# Mettre à jour le PATH
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Changer l'utilisateur
USER appuser

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
