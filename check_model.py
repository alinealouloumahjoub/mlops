
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

# 1. List experiments
print("\n===== EXPERIMENTS =====")

experiments = client.search_experiments()

for exp in experiments:
    print("Experiment:", exp.name, "| ID:", exp.experiment_id)

# 2. List registered models
print("\n===== REGISTERED MODELS =====")

models = client.search_registered_models()

for model in models:
    print("Model:", model.name)
