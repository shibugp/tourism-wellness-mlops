# prep.py
# Loads the raw dataset from Hugging Face, applies cleaning and encoding,
# performs a stratified train/test split, and uploads the resulting files
# back to the Hugging Face Dataset Hub.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
import os
import json

HF_USERNAME  = os.getenv("HF_USERNAME")
REPO_ID      = f"{HF_USERNAME}/tourism-package-prediction"
DATASET_PATH = f"hf://datasets/{REPO_ID}/tourism.csv"

api = HfApi(token=os.getenv("HF_TOKEN"))

print("Loading dataset from Hugging Face Hub...")
df = pd.read_csv(DATASET_PATH)
print(f"Shape: {df.shape}")

print("\nTarget variable distribution:")
print(df["ProdTaken"].value_counts())
class_ratio = round(
    df["ProdTaken"].value_counts()[0] / df["ProdTaken"].value_counts()[1], 2
)
print(f"Class imbalance ratio: {class_ratio}:1")

print("\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values detected.")

print("\nRemoving non-predictive columns: Unnamed: 0, CustomerID")
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True)

print("\nApplying data quality corrections...")
before_gender = df["Gender"].value_counts().to_dict()
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
after_gender = df["Gender"].value_counts().to_dict()
print(f"  Gender before correction: {before_gender}")
print(f"  Gender after correction:  {after_gender}")

before_marital = df["MaritalStatus"].value_counts().to_dict()
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
after_marital = df["MaritalStatus"].value_counts().to_dict()
print(f"  MaritalStatus before consolidation: {before_marital}")
print(f"  MaritalStatus after consolidation:  {after_marital}")

print("\nApplying Label Encoding to categorical columns...")
categorical_cols = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation"
]

encoding_map = {}
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
    encoding_map[col] = {label: int(idx) for idx, label in enumerate(le.classes_)}
    print(f"  {col}: {encoding_map[col]}")

with open("tourism_project/model_building/encoding_map.json", "w") as f:
    json.dump(encoding_map, f, indent=2)
print("\nEncoding map saved to: tourism_project/model_building/encoding_map.json")

TARGET_COL = "ProdTaken"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
print(f"\nFeature matrix: {X.shape} | Target vector: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape} | Test set: {X_test.shape}")
print(f"Train class distribution: {dict(y_train.value_counts())}")
print(f"Test class distribution:  {dict(y_test.value_counts())}")

X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)
print("\nSplit files saved locally.")

print("\nUploading files to Hugging Face Hub...")
for file_path in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"  Uploaded: {file_path}")

api.upload_file(
    path_or_fileobj="tourism_project/model_building/encoding_map.json",
    path_in_repo="encoding_map.json",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("  Uploaded: encoding_map.json")
print(f"\nData preparation complete.")
print(f"All files available at: https://huggingface.co/datasets/{REPO_ID}")
