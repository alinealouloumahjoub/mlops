
# ===== 1. Import the libraries =====

import mlflow
import mlflow.sklearn


# ===== 2. Connect to the Registry database (the one of Lab 4) =====

mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ===== 3. Load the champion from the Registry =====

model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)


# ===== 4. Save it to an ordinary folder named "model" =====

# serialization_format="cloudpickle":
# Uses the same format as in Labs 3 and 4.
# (The default format of MLflow 3 is refused at loading time.)

# This folder will be copied into the Docker image.
# The image will no longer need access to the Registry.

mlflow.sklearn.save_model(
    sk_model=model,
    path="model",
    serialization_format="cloudpickle"
)


# ===== 5. Confirmation =====

print("Champion exported to the model/ folder")
