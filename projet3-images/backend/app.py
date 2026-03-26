import os
import logging
from io import BytesIO
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="API pour la classification d'images avec MobileNetV2",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the model
model = None

@app.on_event("startup")
async def load_model():
    """
    Load the pre-trained MobileNetV2 model at startup.
    This function runs once when the server starts.
    """
    global model
    try:
        logger.info("Starting to load MobileNetV2 model...")
        # Load pre-trained MobileNetV2 weights from ImageNet
        # weights='imagenet' downloads automatically on first run (~100MB)
        model = MobileNetV2(weights='imagenet', include_top=True)
        logger.info("MobileNetV2 model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running and model is loaded.
    """
    if model is None:
        return {"status": "loading", "message": "Model is still loading"}
    return {
        "status": "healthy",
        "message": "Service is running and ready for predictions",
        "model": "MobileNetV2"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded image using MobileNetV2.

    Args:
        file: Image file (JPEG, PNG, WebP, GIF, etc.)

    Returns:
        JSON with top 5 predictions including class name and probability

    Raises:
        HTTPException: If model is not loaded, file is invalid, or processing fails
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    try:
        # Read the uploaded file
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="File is empty")

        # Open image with PIL
        try:
            img = Image.open(BytesIO(contents))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Accepted formats: JPEG, PNG, WebP, GIF"
            )

        # Convert to RGB if necessary (handle RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize image to 224x224 (MobileNetV2 input size)
        img = img.resize((224, 224), Image.Resampling.LANCZOS)

        # Convert PIL image to numpy array
        img_array = keras_image.img_to_array(img)

        # Add batch dimension (model expects batches)
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the image (normalize pixel values)
        # MobileNetV2 expects values in range [-1, 1]
        img_array = preprocess_input(img_array)

        # Make prediction
        logger.info(f"Processing image: {file.filename}")
        predictions = model.predict(img_array, verbose=0)

        # Decode predictions to get class names
        # top=5 returns top 5 predictions
        decoded_preds = decode_predictions(predictions, top=5)

        # Extract results from first (and only) image in batch
        results = decoded_preds[0]

        # Format response
        predictions_list = []
        for class_id, class_name, probability in results:
            predictions_list.append({
                "class": class_name,
                "probability": float(probability),
                "percentage": round(float(probability) * 100, 2)
            })

        logger.info(f"Prediction complete for {file.filename}")

        return {
            "status": "success",
            "filename": file.filename,
            "predictions": predictions_list
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Image Classification API",
        "version": "1.0.0",
        "model": "MobileNetV2",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "docs": "/docs"
        },
        "usage": "Send a POST request to /predict with an image file"
    }

if __name__ == "__main__":
    import uvicorn
    # Run the server with: python app.py
    # Or use: uvicorn app:app --reload
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
