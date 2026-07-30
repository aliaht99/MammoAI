#!/usr/bin/env python3
"""
Deploy MammoAI to a Hugging Face Space (free, permanent, always-on).

Prerequisites (one time, done by you in a terminal — your token never leaves
your machine):

    1. Create a free account at https://huggingface.co/join
    2. Create a WRITE token at https://huggingface.co/settings/tokens
    3. huggingface-cli login        # paste the token when prompted

Then just run:

    python deploy_hf.py

It creates the Space (if needed), uploads only the files the app needs
(~7 MB — never the DICOM data or CNN checkpoints), and prints the live URL.
"""
from pathlib import Path
import sys

from huggingface_hub import HfApi, whoami

ROOT = Path(__file__).parent
SPACE_NAME = "MammoAI"

# Only what the app actually needs at runtime. Everything else — the 152 GB
# DICOM manifest, the 521 MB CNN checkpoints, notebooks — stays out.
ALLOW = [
    "mammo_doctor.py",
    "app.py",
    "requirements.txt",
    "Dockerfile",
    "LICENSE",
    "models/**",
    "results/**",
    "demo_images/**",
    "src/**",
    "stage2_cnn/*.py",
]
IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"]


def main():
    try:
        user = whoami()["name"]
    except Exception:
        sys.exit("Not logged in. Run:  huggingface-cli login")

    repo_id = f"{user}/{SPACE_NAME}"
    api = HfApi()

    print(f"▶ Creating / reusing Space: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",   # HF retired the native Streamlit SDK
        exist_ok=True,
    )

    print("▶ Uploading app files (~7 MB)…")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(ROOT),
        allow_patterns=ALLOW,
        ignore_patterns=IGNORE,
        commit_message="Deploy MammoAI research demo",
    )

    print("▶ Uploading Space README (title card + usage)…")
    api.upload_file(
        repo_id=repo_id,
        repo_type="space",
        path_or_fileobj=str(ROOT / "README_HF.md"),
        path_in_repo="README.md",
        commit_message="Add Space card",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print("\n✅ Deployed.")
    print(f"   Space page : {url}")
    print(f"   Direct app : https://{user.lower()}-{SPACE_NAME.lower()}.hf.space")
    print("\n   First build takes ~3-5 minutes. Watch progress on the Space page.")


if __name__ == "__main__":
    main()
