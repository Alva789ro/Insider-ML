from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

import uvicorn
from inference import load_xgbInsider, predict, prepare_data

app = FastAPI(title="Finance Insider Trading - XGBoost Model API")

# Pydantic model describing input
class InsiderTransaction(BaseModel):
    Data: str

# Global or app state variable to hold loaded model
MODEL = None

@app.on_event("startup")
def startup_event():
    global MODEL
    # Load model from local file or from S3
    MODEL = load_xgbInsider()

@app.get("/")
def read_root():
    return {"message": "Insider Trading Model API is up and running!"}

@app.post("/insider_effect_predict")
def insider_effect_predict(transaction: InsiderTransaction):
    """
    Currently only predicts 1 day into the future
    """
    # Extract input features from the transaction object
    input_data = prepare_data(transaction.Data)
    result = predict(MODEL, input_data)
    return {
        "status": "success",
        "message": "Prediction completed successfully",
        "data": {
            "prediction": result,
            "details": "The prediction indicates %s effect" % (result)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)