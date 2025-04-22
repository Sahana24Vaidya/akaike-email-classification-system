from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import joblib
from utils import mask_pii, clean_text
import uvicorn
import json

app = FastAPI()

# Load model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Input format
class EmailInput(BaseModel):
    subject: str
    body: str

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "🚀 Email Classifier API is running!"}

# Main prediction endpoint
@app.post("/classify")
async def classify_email(email: EmailInput):
    try:
        combined_text = f"{email.subject} {email.body}"
        
        # Mask PII
        masked_text, entities = mask_pii(combined_text)

        # Clean for classification
        cleaned_text = clean_text(masked_text)

        # Vectorize
        X = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(X)[0]
        confidence = max(model.predict_proba(X)[0])

        # Prepare response
        response = {
            "input_email_body": f"{email.subject} {email.body}",
            "list_of_masked_entities": [
                {"position": [e.start, e.end], "classification": e.label, "entity": e.text}
                for e in entities
            ],
            "masked_email": masked_text,
            "category_of_the_email": prediction
        }

        # Return response with pretty printed JSON
        return JSONResponse(content=json.dumps(response, indent=4))

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)