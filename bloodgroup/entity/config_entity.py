from dataclasses import dataclass
import os
from bloodgroup.constants import *

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.ARTIFACTS_DIR: str = os.path.join(os.getcwd(), DATA_INGESTION_ARTIFACTS_DIR)
        self.DATA_INGESTION_UNZIPED_ARTIFACTS_DIR: str = os.path.join(self.ARTIFACTS_DIR, DATA_INGESTION_UNZIPED_ARTIFACTS_DIR)
        self.SOURCE_URL: str = SOURCE_URL
        self.CREDENTIAL: str = CREDENTIAL
        self.KAGGLE_CREDENTIAL_DIR: str = KAGGLE_CREDENTIAL_DIR

@dataclass
class PrepareBaseModelConfig:
    def __init__(self):
        self.ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, PREPARE_BASE_MODEL_ARTIFACTS_DIR)
        self.BASE_MODEL_PATH: str = os.path.join(self.ARTIFACTS_DIR, BASE_MODEL_PATH)
        self.UPDATED_BASE_MODEL_PATH: str = os.path.join(self.ARTIFACTS_DIR, UPDATED_BASE_MODEL_PATH)

@dataclass
class PrepareCallbacksConfig:
    def __init__(self):
        self.ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, PREPARE_CALLBACKS_ARTIFACTS_DIR)
        self.TENSORBOARD_LOG_DIR: str = os.path.join(self.ARTIFACTS_DIR, TENSORBOARD_LOG_DIR)
        self.CHECKPOINT_DIR: str = os.path.join(self.ARTIFACTS_DIR, CHECKPOINT_DIR)

@dataclass
class TrainingConfig:
    def __init__(self):
        self.ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, TRAINING_ARTIFACTS_DIR)
        self.TRAINED_MODEL_PATH: str = os.path.join(self.ARTIFACTS_DIR, TRAINED_MODEL_PATH)
        self.EPOCHS: int = EPOCHS
        self.BATCH_SIZE: int = BATCH_SIZE
        self.LEARNING_RATE: float = LEARNING_RATE
