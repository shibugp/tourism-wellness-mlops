# data_register.py
# Verifies the tourism dataset repository exists on Hugging Face Dataset Hub.
# All downstream pipeline stages load data directly from this centralised location.

from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

HF_USERNAME  = os.getenv("HF_USERNAME")
DATASET_NAME = "tourism-package-prediction"
REPO_ID      = f"{HF_USERNAME}/{DATASET_NAME}"
REPO_TYPE    = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repository {REPO_ID} exists. Data already registered.")
except RepositoryNotFoundError:
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False)
    print(f"Dataset repository {REPO_ID} created.")

print(f"Data registration complete: https://huggingface.co/datasets/{REPO_ID}")
