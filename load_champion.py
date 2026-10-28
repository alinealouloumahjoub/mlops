import mlflow
import mlflow.sklearn  # To reload a scikit-learn model
import pandas as pd


# ===== 1. Connect to the same database as Lab 3 =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ===== 2. Load the champion model using its alias =====
# We do not need to know the model's version or run ID.
model = mlflow.sklearn.load_model("models:/churn-model@champion")

print("Model loaded:", type(model).__name__)
print("Expected variables:", list(model.feature_names_in_))


# ===== 3. Create a new customer =====
# The subscription columns are already encoded as Boolean values.
# Here, the customer has a Standard subscription.
customer = pd.DataFrame([
    {
        "age": 35,
        "salaire": 2500,
        "anciennete": 1,
        "consommation": 10,
        "abonnement_Premium": False,
        "abonnement_Standard": True,
    }
])


# ===== 4. Select the expected features in the correct order =====
customer = customer[list(model.feature_names_in_)]


# ===== 5. Predict whether the customer will leave =====
# 1 = the customer will leave
# 0 = the customer will stay
prediction = model.predict(customer)[0]

print(
    "Will the customer leave? ->",
    "YES" if prediction == 1 else "NO"
)
