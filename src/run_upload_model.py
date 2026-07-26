from pathlib import Path

from huggingface_hub import HfApi

from config.env import settings

checkpoint_path = Path("model-output/final")

if not settings.hf_token:
    raise RuntimeError("Falta HF_TOKEN en .env")
if not settings.hf_model_repo_id:
    raise RuntimeError("Falta HF_MODEL_REPO_ID en .env (ej. tu-usuario/nombre-del-modelo)")

api = HfApi(token=settings.hf_token)
api.create_repo(settings.hf_model_repo_id, private=False, exist_ok=True)
api.upload_folder(
    repo_id=settings.hf_model_repo_id,
    folder_path=str(checkpoint_path),
    commit_message="Upload model",
)

print(f"Subido a https://huggingface.co/{settings.hf_model_repo_id}")