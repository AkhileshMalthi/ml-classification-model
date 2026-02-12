# Implement a training script that:
# Calls load_and_preprocess_data.
# Initializes an MLflow run (with mlflow.start_run(run_name='<run_name>'):).
# Initializes and trains a scikit-learn classification model.
# Logs model parameters using mlflow.log_param().
# Calculates and logs evaluation metrics (accuracy, precision, recall, f1-score) using mlflow.log_metric().
# Logs the trained scikit-learn model using mlflow.sklearn.log_model(model, "model", registered_model_name="ClassificationModel").
# Generates and logs a confusion matrix plot as an artifact.
# Logs the fitted scaler object as an artifact (e.g., mlflow.log_artifact("scaler.pkl", "preprocessing_artifacts")).
# Registers the best-performing model to the MLflow Model Registry.

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os
from src.data_processor import load_and_preprocess_data

def train_model(dataset_name='iris', C=1.0, penalty='l2', random_state=42):
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(dataset_name, random_state=random_state)

    # Set up MLflow
    mlflow.set_experiment(f"{dataset_name}_classification")
    with mlflow.start_run(run_name=f"{dataset_name}_logreg") as run:
        # Initialize and train model
        model = LogisticRegression(C=C, penalty=penalty, random_state=random_state, max_iter=1000)
        model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("C", C)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("random_state", random_state)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        plt.title("Confusion Matrix")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close(fig)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # Save and log scaler
        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing_artifacts")
        os.remove(scaler_path)

        # Log and register model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="ClassificationModel"
        )

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    train_model(dataset_name='iris')
