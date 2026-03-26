"""
Projet 4 - Assistant IA Multi-Provider
FastAPI backend pour intégration de multiples fournisseurs d'IA
"""

import os
import json
from typing import Optional
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Initialiser FastAPI
app = FastAPI(
    title="Assistant IA Multi-Provider",
    description="API pour interagir avec plusieurs fournisseurs d'IA",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Modèles Pydantic
# ============================================================================

class ChatRequest(BaseModel):
    """Requête de chat"""
    message: str = Field(..., min_length=1, max_length=4096)
    provider: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    """Réponse de chat"""
    response: str
    provider: str
    timestamp: str
    usage: Optional[dict] = None


class HealthResponse(BaseModel):
    """Réponse health check"""
    status: str
    timestamp: str


class ProvidersResponse(BaseModel):
    """Réponse liste des providers"""
    all_providers: list[str]
    configured_providers: list[str]


# ============================================================================
# Configuration des API Keys
# ============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Providers disponibles
ALL_PROVIDERS = ["claude", "chatgpt", "grok", "deepseek", "huggingface"]

# Providers configurés (ayant une clé API)
CONFIGURED_PROVIDERS = {
    "claude": ANTHROPIC_API_KEY is not None,
    "chatgpt": OPENAI_API_KEY is not None,
    "grok": XAI_API_KEY is not None,
    "deepseek": DEEPSEEK_API_KEY is not None,
    "huggingface": HUGGINGFACE_API_KEY is not None,
}


# ============================================================================
# Fonctions de Chat par Provider
# ============================================================================

async def chat_claude(message: str) -> dict:
    """
    Appel à l'API Claude (Anthropic)

    Documentation: https://docs.anthropic.com/messages/guide
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not configured")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            },
            timeout=30.0
        )

        if response.status_code != 200:
            error_detail = response.text
            try:
                error_detail = response.json().get("error", {}).get("message", error_detail)
            except:
                pass
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Claude API error: {error_detail}"
            )

        data = response.json()

        return {
            "response": data["content"][0]["text"],
            "usage": {
                "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                "output_tokens": data.get("usage", {}).get("output_tokens", 0),
            }
        }


async def chat_chatgpt(message: str) -> dict:
    """
    Appel à l'API ChatGPT (OpenAI)

    Documentation: https://platform.openai.com/docs/api-reference/chat/create
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                "max_tokens": 1024
            },
            timeout=30.0
        )

        if response.status_code != 200:
            error_detail = response.text
            try:
                error_detail = response.json().get("error", {}).get("message", error_detail)
            except:
                pass
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenAI API error: {error_detail}"
            )

        data = response.json()

        return {
            "response": data["choices"][0]["message"]["content"],
            "usage": {
                "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
            }
        }


async def chat_grok(message: str) -> dict:
    """
    Appel à l'API Grok (X AI)

    Documentation: https://docs.x.ai/docs
    """
    if not XAI_API_KEY:
        raise ValueError("XAI_API_KEY not configured")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-beta",
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                "max_tokens": 1024
            },
            timeout=30.0
        )

        if response.status_code != 200:
            error_detail = response.text
            try:
                error_detail = response.json().get("error", {}).get("message", error_detail)
            except:
                pass
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Grok API error: {error_detail}"
            )

        data = response.json()

        return {
            "response": data["choices"][0]["message"]["content"],
            "usage": {
                "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
            }
        }


async def chat_deepseek(message: str) -> dict:
    """
    Appel à l'API DeepSeek

    Documentation: https://platform.deepseek.com/api-docs
    """
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not configured")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                "max_tokens": 1024
            },
            timeout=30.0
        )

        if response.status_code != 200:
            error_detail = response.text
            try:
                error_detail = response.json().get("error", {}).get("message", error_detail)
            except:
                pass
            raise HTTPException(
                status_code=response.status_code,
                detail=f"DeepSeek API error: {error_detail}"
            )

        data = response.json()

        return {
            "response": data["choices"][0]["message"]["content"],
            "usage": {
                "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
            }
        }


async def chat_huggingface(message: str) -> dict:
    """
    Appel à l'API Hugging Face Inference

    Documentation: https://huggingface.co/docs/api-inference
    """
    if not HUGGINGFACE_API_KEY:
        raise ValueError("HUGGINGFACE_API_KEY not configured")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct",
            headers={
                "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "inputs": message,
                "parameters": {
                    "max_length": 1024,
                    "temperature": 0.7,
                }
            },
            timeout=60.0
        )

        if response.status_code != 200:
            error_detail = response.text
            try:
                error_detail = response.json().get("error", error_detail)
            except:
                pass
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Hugging Face API error: {error_detail}"
            )

        data = response.json()

        # Hugging Face retourne une liste de résultats
        if isinstance(data, list) and len(data) > 0:
            response_text = data[0].get("generated_text", "")
            # Retirer le prompt du texte généré
            if response_text.startswith(message):
                response_text = response_text[len(message):].strip()
        else:
            response_text = str(data)

        return {
            "response": response_text,
            "usage": {
                "input_tokens": 0,  # HF n'expose pas l'usage
                "output_tokens": 0,
            }
        }


# Routing des providers
PROVIDER_HANDLERS = {
    "claude": chat_claude,
    "chatgpt": chat_chatgpt,
    "grok": chat_grok,
    "deepseek": chat_deepseek,
    "huggingface": chat_huggingface,
}


# ============================================================================
# Routes
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Vérifier l'état du serveur

    Retourne le statut du serveur et l'timestamp actuel.
    """
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/providers", response_model=ProvidersResponse)
async def list_providers() -> ProvidersResponse:
    """
    Lister les providers disponibles

    Retourne la liste de tous les providers et celle des providers configurés.
    """
    configured = [
        provider for provider, is_configured in CONFIGURED_PROVIDERS.items()
        if is_configured
    ]

    return ProvidersResponse(
        all_providers=ALL_PROVIDERS,
        configured_providers=configured
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Envoyer un message à un provider

    Paramètres:
    - message: Le message à envoyer (max 4096 caractères)
    - provider: Le provider à utiliser (claude, chatgpt, grok, deepseek, huggingface)

    Retourne:
    - response: La réponse du provider
    - provider: Le provider utilisé
    - timestamp: Timestamp de la réponse
    - usage: Statistiques de tokens (si disponibles)
    """

    # Valider que le message n'est pas vide
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Valider que le provider existe
    if request.provider not in ALL_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{request.provider}' not found. Available: {', '.join(ALL_PROVIDERS)}"
        )

    # Valider que le provider est configuré
    if not CONFIGURED_PROVIDERS.get(request.provider, False):
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{request.provider}' is not configured. Missing API key."
        )

    # Récupérer le handler du provider
    handler = PROVIDER_HANDLERS.get(request.provider)
    if not handler:
        raise HTTPException(status_code=500, detail="Internal error: handler not found")

    try:
        # Appeler le provider
        result = await handler(request.message)

        return ChatResponse(
            response=result["response"],
            provider=request.provider,
            timestamp=datetime.utcnow().isoformat(),
            usage=result.get("usage")
        )

    except ValueError as e:
        # Erreur de configuration
        raise HTTPException(status_code=500, detail=str(e))
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Provider request timeout")
    except Exception as e:
        # Erreur générique
        raise HTTPException(status_code=500, detail=f"Error calling provider: {str(e)}")


# ============================================================================
# Événements du serveur
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Événement au démarrage du serveur"""
    print("=" * 60)
    print("Assistant IA Multi-Provider - Backend")
    print("=" * 60)
    print(f"Tous les providers: {ALL_PROVIDERS}")
    configured = [p for p, c in CONFIGURED_PROVIDERS.items() if c]
    print(f"Providers configurés: {configured if configured else 'None'}")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
