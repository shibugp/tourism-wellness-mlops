# train.py
# Production model training script for execution in the CI/CD pipeline.
# Reads split data from Hugging Face Hub, runs GridSearchCV with MLflow
# tracking, and registers the best model to the Hugging Face Model Hub.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
import mlflow.sklearn
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

HF_USERNAME  = os.getenv("HF_USERNAME")
DATASET_REPO = f"{HF_USERNAME}/tourism-package-prediction"
MODEL_REPO   = f"{HF_USERNAME}/tourism-wellness-model"
MODEL_FNAME  = "best_tourism_wellness_model.joblib"

api = HfApi(token=os.getenv("HF_TOKEN"))

mlflow.set_tracking_uri("file:///tmp/mlflow")
mlflow.set_experiment("mlops-training-experiment")

print("Loading split datasets from Hugging Face Hub...")
X_train = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
X_test  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
y_train = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").squeeze()
y_test  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").squeeze()
print(f"Training set: {X_train.shape} | Test set: {X_test.shape}")

param_grid = {
    "n_estimators":      [100, 200],
    "max_depth":         [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}

base_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

with mlflow.start_run(run_name="RF_Production_Run"):

    grid_search = GridSearchCV(
        base_model, param_grid, cv=3,
        scoring="f1", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(run_name=f"run_{i+1}", nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("cv_mean_f1", results["mean_test_score"][i])

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("test_accuracy",  acc)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall",    recall)
    mlflow.log_metric("test_f1",        f1)
    mlflow.log_metric("test_roc_auc",   roc_auc)
    mlflow.log_metric("best_cv_f1",     grid_search.best_score_)
    mlflow.sklearn.log_model(best_model, "best_rf_model")

    print(f"Best parameters:  {grid_search.best_params_}")
    print(f"Test Accuracy:    {acc:.4f}")
    print(f"Test F1:          {f1:.4f}")
    print(f"Test ROC-AUC:     {roc_auc:.4f}")

joblib.dump(best_model, MODEL_FNAME)
print(f"Model saved: {MODEL_FNAME}")

try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"Model repository {MODEL_REPO} exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)
    print(f"Model repository {MODEL_REPO} created.")

api.upload_file(
    path_or_fileobj=MODEL_FNAME,
    path_in_repo=MODEL_FNAME,
    repo_id=MODEL_REPO,
    repo_type="model",
)
print(f"Model registered at: https://huggingface.co/{MODEL_REPO}")
