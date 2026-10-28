
# ===== 1. Import the libraries =====
import mlflow
from mlflow import MlflowClient


# ===== 2. Connect to the same database as Lab 3 =====
# The mlflow.db file stores all our experiment runs.
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ===== 3. Find the best run (same logic as in Lab 3) =====
# Sort runs by F1 score in descending order.
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# The first row corresponds to the run with the highest F1 score.
best = runs.iloc[0]
run_id = best["run_id"]

print(
    "Best run:",
    best["tags.mlflow.runName"],
    "| id:",
    run_id[:8],
    "..."
)


# ===== 4. Register the model in the Model Registry =====
# Register the model saved in the best run.
# "churn-model" is the registered model's name.
result = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)

print(
    "Model registered: churn-model | version",
    result.version
)


# ===== 5. Assign the "champion" alias to this version =====
client = MlflowClient()

client.set_registered_model_alias(
    "churn-model",
    "champion",
    result.version
)

print(
    "Alias 'champion' -> version",
    result.version
)
