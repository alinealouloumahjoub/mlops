import mlflow
from mlflow import MlflowClient


# ===== 1. Connect to the MLflow database =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()


# ===== 2. Find the run "trial-2" =====
# "trial-2" is the weakest model from Lab 3.
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'trial-2'"
)

run_id = runs.iloc[0]["run_id"]


# ===== 3. Register the model as a new version =====
# This creates a new version of "churn-model".
v = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)

print("New version created:", v.version)


# ===== 4. Get the current champion version =====
current = client.get_model_version_by_alias(
    "churn-model",
    "champion"
)

print("Current champion: version", current.version)


# ===== 5. Move the champion alias to the new version =====
client.set_registered_model_alias(
    "churn-model",
    "champion",
    v.version
)

print(
    "After the move: champion -> version",
    client.get_model_version_by_alias(
        "churn-model",
        "champion"
    ).version
)


# ===== 6. Restore the original champion version =====
client.set_registered_model_alias(
    "churn-model",
    "champion",
    current.version
)

print(
    "After going back: champion -> version",
    client.get_model_version_by_alias(
        "churn-model",
        "champion"
    ).version
)