from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from utils import mask_pii, clean_text
from typing import List, Dict, Optional

app = FastAPI(
    title="Email Classification API",
    description="Classifies emails and masks PII (Personal Identifiable Information)",
    version="1.0.0"
)

# --- Models ---
class PIIEntity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class ClassificationResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[PIIEntity]
    masked_email: str
    category_of_the_email: str

class EmailRequest(BaseModel):
    email_body: str

# --- Load ML Artifacts ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.joblib")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✓ Model and vectorizer loaded successfully")
except Exception as e:
    print(f"! Loading failed: {str(e)}")
    model = None
    vectorizer = None

# --- Endpoints ---
@app.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify email and mask PII",
    response_description="Processed email with classification results",
    tags=["Email Processing"]
)
async def classify(request: EmailRequest):
    """
    Processes an email through:
    1. PII Masking
    2. Text Cleaning
    3. Classification
    """
    if not model or not vectorizer:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable - ML models not loaded"
        )
    
    try:
        # 1. Get raw input
        email_text = request.email_body
        
        # 2. Mask PII 
        masked, entities = mask_pii(email_text)
        
        # 3. Clean text
        cleaned = clean_text(masked)
        
        # 4. Vectorize and predict
        X = vectorizer.transform([cleaned])
        category = model.predict(X)[0]
        
        return {
            "input_email_body": email_text,
            "list_of_masked_entities": [
                {
                    "position": [ent.start, ent.end],
                    "classification": ent.label,
                    "entity": ent.text
                } for ent in entities
            ],
            "masked_email": masked,
            "category_of_the_email": category
        }
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get(
    "/health",
    summary="Service health check",
    tags=["Monitoring"]
)
async def health_check():
    """Verify if the service is ready to handle requests"""
    return {
        "ready": bool(model and vectorizer),
        "model_type": type(model).__name__ if model else None,
        "features": vectorizer.get_feature_names_out().tolist()[:5] if vectorizer else None,
        "status": "operational" if model and vectorizer else "degraded"
    }