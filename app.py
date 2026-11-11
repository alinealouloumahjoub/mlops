
# ===== 1. Import the libraries =====

import os
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ===== 2. Create the application =====

app = FastAPI(title="Churn API")


# ===== 3. Read the configuration from environment variables =====

# Nothing is hard-coded.
# The same code works on a computer, in a single container,
# or in Docker Compose. Only the variables change.

# os.getenv("NAME", "value"):
# Returns the environment variable or the default value if it does not exist.

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/churn-model@champion")


# ===== 4. Load the champion ONLY ONCE, at startup =====

# MODEL_URI can be a Registry address (models:/...)
# or a simple folder (model).

mlflow.set_tracking_uri(TRACKING_URI)

model = mlflow.sklearn.load_model(MODEL_URI)

print(f"Model loaded from: {MODEL_URI}")


# ===== 5. Describe the expected data (validated automatically) =====

class Customer(BaseModel):
    age: int  # Whole number
    salaire: float  # Decimal number (salary)

    abonnement: Literal[
        "Basique",
        "Standard",
        "Premium"
    ]  # Three possible subscription values

    anciennete: float  # Seniority / tenure
    consommation: float  # Usage


# ===== 6. Route /health: check that the API is alive =====

@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 7. Route /predict: receive a customer and return the prediction =====

@app.post("/predict")
def predict(customer: Customer):

    # a) Convert the received customer into a one-row DataFrame
    row = pd.DataFrame([customer.model_dump()])

    # b) Convert the categorical column "abonnement" into dummy columns
    #    (as during model training)
    row = pd.get_dummies(row)

    # c) Keep the model's expected columns in the correct order.
    #    Missing columns are filled with 0.
    row = row.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # d) Make the prediction:
    #    1 = customer leaves
    #    0 = customer stays
    result = int(model.predict(row)[0])

    return {
        "churn": result,
        "message": (
            "The customer will leave"
            if result == 1
            else "The customer stays"
        )
    }
