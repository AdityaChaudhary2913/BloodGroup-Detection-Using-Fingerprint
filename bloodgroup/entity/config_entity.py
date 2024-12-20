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
class DataTransformationConfig:
    def __init__(self):
        self.ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRAIN_LOADER_PATH: str = os.path.join(self.ARTIFACTS_DIR, TRAIN_LOADER)
        self.VAL_LOADER_PATH: str = os.path.join(self.ARTIFACTS_DIR, VAL_LOADER)
        self.TEST_LOADER_PATH: str = os.path.join(self.ARTIFACTS_DIR, TEST_LOADER)
        
@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.BEST_MODEL_PATH: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        self.BEST_MODEL_NAME: str = BEST_MODEL_NAME
        self.EPOCHS: int = EPOCHS
        self.PATIENCE: int = PATIENCE
