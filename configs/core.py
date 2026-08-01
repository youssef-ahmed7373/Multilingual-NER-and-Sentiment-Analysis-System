import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel

# =====================================================================
# 1. PATH CONFIGURATION & SETTINGS
# =====================================================================
# Automatically resolves the directory layout where core.py lives
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
print(PACKAGE_ROOT)
CONFIG_FILE_PATH = PACKAGE_ROOT / 'config.yaml'


# =====================================================================
# 2. PYDANTIC VALIDATION SCHEMAS
# =====================================================================
class AppConfig(BaseModel):
    package_name: str
    version: str

class NERModelConfig(BaseModel):
    model_checkpoint: str
    model_finetuned: str
    max_length: int
    languages: List[str]
    train_samples: int
    val_samples: int
    test_samples: int
    output_dir: str
    raw_data_dir: str
    processed_data_dir: str
    num_epochs: int
    batch_size_train: int
    batch_size_eval: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    fp16: bool
    label_list: List[str]

    @property
    def id2label(self) -> Dict[int, str]:
        return {i: label for i, label in enumerate(self.label_list)}

    @property
    def label2id(self) -> Dict[str, int]:
        return {label: i for i, label in enumerate(self.label_list)}

class SentimentModelConfig(BaseModel):
    model_checkpoint: str
    model_finetuned: str
    max_length: int
    languages: List[str]
    train_samples: int
    val_samples: int
    test_samples: int
    output_dir: str
    raw_data_dir: str
    processed_data_dir: str
    num_epochs: int
    batch_size_train: int
    batch_size_eval: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    fp16: bool
    label_list: List[str]

    @property
    def id2label(self) -> Dict[int, str]:
        return {i: label for i, label in enumerate(self.label_list)}

    @property
    def label2id(self) -> Dict[str, int]:
        return {label: i for i, label in enumerate(self.label_list)}

class Config(BaseModel):
    """The master wrapper object mapping fields together."""
    app_config: AppConfig
    ner_model_settings: NERModelConfig
    sentiment_model_settings: SentimentModelConfig


# =====================================================================
# 3. CORE MANAGEMENT FUNCTIONS
# =====================================================================
def find_config_file() -> Path:
    """Checking configuration file exists or not."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise FileNotFoundError(f"Config yaml file not found at path: {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> dict:
    """Parse yaml file containing your pipeline properties configuration."""
    if not cfg_path:
        cfg_path = find_config_file()
        
    if cfg_path.is_file():
        with open(cfg_path, 'r') as conf_file:
            parsed_config = yaml.safe_load(conf_file)
            return parsed_config
            
    raise OSError(f"Didn't find config file at path {CONFIG_FILE_PATH}")


def create_and_validate_config(parsed_config: Optional[dict] = None) -> Config:
    """Run structural Pydantic validation checks on parsed config parameters."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
        
    # Instantiate fields by unpacking specific block segments 
    _config = Config(
        app_config=AppConfig(**parsed_config["app_metadata"]),
        ner_model_settings=NERModelConfig(**parsed_config["ner_model_settings"]),
        sentiment_model_settings=SentimentModelConfig(**parsed_config["sentiment_model_settings"])
    )
    return _config


# =====================================================================
# 4. INSTANCE GENERATION EXPORT
# =====================================================================
# This generates the validated config automatically when imported
config = create_and_validate_config()