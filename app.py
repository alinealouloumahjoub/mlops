
# ===== 1. Import the libraries =====
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ===== 2. Create the application =====
# The title is displayed on the /docs page.
app = FastAPI(title="Churn API")


# ===== 3. Load the champion model only once =====
# Reloading the model for every request would make the API slower.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)


# ===== 4. Describe and validate the expected data =====
class Customer(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# ===== 5. Route /health: check that the API is running =====
# GET: retrieves information without sending a request body.
@app.get("uvicorn app:app --port 8000/health")
def health():
    return {"status": "ok"}


# ===== 6. Route /predict: receive a customer and return a prediction =====
# POST: sends customer data to the API.
@app.post("/predict")
def predict(customer: Customer):
    # a) Convert the customer into a one-row DataFrame.
    row = pd.DataFrame([customer.model_dump()])

    # b) Convert the subscription category into 0/1 columns.
    row = pd.get_dummies(row)

    # c) Keep the model's expected columns in the correct order.
    # Missing columns are filled with 0.
    row = row.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # d) Make the prediction: 1 = leaves, 0 = stays.
    result = int(model.predict(row)[0])

    # e) Return the prediction and a message.
    return {
        "churn": result,
        "message": (
            "The customer will leave"
            if result == 1
            else "The customer stays"
        )
    }
