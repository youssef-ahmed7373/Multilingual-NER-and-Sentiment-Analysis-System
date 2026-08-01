import os
from pathlib import Path

from transformers import AutoModelForTokenClassification, AutoTokenizer

from configs.core import config


def download_from_hub(repo_id: str):
    # 1. Resolve local destination directory layout safely
    base_output_dir = Path(config.ner_model_settings.output_dir)
    final_local_path = base_output_dir.parent / "final"
    
    print(f"🔄 Downloading assets from Hugging Face Hub: '{repo_id}'...")
    
    # 2. Download weights from the cloud into memory
    model = AutoModelForTokenClassification.from_pretrained(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    
    # 3. Secure folder generation path alignment
    final_local_path.mkdir(parents=True, exist_ok=True)
    
    # 4. Save both components using save_pretrained()
    print(f"💾 Saving components to disk: {final_local_path}")
    model.save_pretrained(str(final_local_path))      # <- Changed from save_model
    tokenizer.save_pretrained(str(final_local_path))
    
    print(f"🟢 Model is fully operational locally and saved at: {final_local_path.resolve()}")

if __name__ == "__main__":
    # Replace this string with your real Hugging Face Hub repository identifier
    YOUR_HUB_REPO_ID = config.ner_model_settings.model_finetuned
    
    download_from_hub(repo_id=YOUR_HUB_REPO_ID)



