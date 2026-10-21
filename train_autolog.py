
import pandas as pd
import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ===== 1. Load the data =====
df = pd.read_csv("churn_data.csv")


# ===== 2. Prepare the data =====
X = pd.get_dummies(
    df[["age", "salaire", "abonnement", "anciennete", "consommation"]],
    drop_first=True
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    df["churn"],
    test_size=0.2,
    random_state=42
)


# ===== 3. Choose the MLflow experiment =====
mlflow.set_experiment("churn-prediction")


# ===== 4. Enable automatic logging =====
# MLflow automatically records the model parameters, metrics,
# and the model itself.
mlflow.sklearn.autolog(serialization_format="cloudpickle")


# ===== 5. Train the model and record the run =====
with mlflow.start_run(run_name="trial-auto"):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=42
    )

    # Training triggers automatic logging
    model.fit(X_train, y_train)

    # Compute the accuracy on the test data
    print("Test score:", model.score(X_test, y_test))
