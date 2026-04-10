# hosting.py
# Uploads the deployment folder contents to the Hugging Face Space.

from huggingface_hub import HfApi
import os

HF_USERNAME  = os.getenv("HF_USERNAME")
SPACE_REPO  = f"{HF_USERNAME}/tourism-wellness-app"

api = HfApi(token=os.getenv("HF_TOKEN"))

print(f"Uploading deployment files to Space: {SPACE_REPO}")

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)

print(f"Deployment complete.")
print(f"Application URL: https://huggingface.co/spaces/{SPACE_REPO}")
