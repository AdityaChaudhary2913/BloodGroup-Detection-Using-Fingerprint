from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    data_dir: str

@dataclass
class PrepareBaseModelArtifacts:
    base_model_path: str
    updated_base_model_path: str

@dataclass
class PrepareCallbacksArtifacts:
    tensorboard_log_dir: str
    checkpoint_dir: str

@dataclass
class TrainingArtifacts:
    trained_model_path: str
