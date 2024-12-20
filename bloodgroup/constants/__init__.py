import os
from datetime import datetime

# Common constants
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)

# Data ingestion constants

DATA_INGESTION_ARTIFACTS_DIR = os.path.join("artifacts", "DataIngestionArtifacts")
DATA_INGESTION_UNZIPED_ARTIFACTS_DIR = "dataset_blood_group"
SOURCE_URL = "https://www.kaggle.com/datasets/rajumavinmar/finger-print-based-blood-group-dataset/data"
CREDENTIAL = "rajumavinmar/finger-print-based-blood-group-dataset"
KAGGLE_CREDENTIAL_DIR = "/Users/adityachaudhary/Desktop/Important Projects/Data Science/BloodGroup-Detection-Using-Fingerprint/.kaggle"

# Prepare base model constants
PREPARE_BASE_MODEL_ARTIFACTS_DIR = "PrepareBaseModelArtifacts"
BASE_MODEL_PATH = "base_model.h5"
UPDATED_BASE_MODEL_PATH = "base_model_updated.h5"

# Prepare callbacks constants
PREPARE_CALLBACKS_ARTIFACTS_DIR = "PrepareCallbacksArtifacts"
TENSORBOARD_LOG_DIR = "tensorboard_log_dir"
CHECKPOINT_DIR = "checkpoint_dir"

# Training constants
TRAINING_ARTIFACTS_DIR = "TrainingArtifacts"
TRAINED_MODEL_PATH = "model.h5"

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001