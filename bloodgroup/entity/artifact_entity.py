from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    data_dir: str
    
@dataclass
class DataTransformationArtifacts:
    train_loader_path: str
    val_loader_path: str
    test_loader_path: str
    
@dataclass
class ModelTrainerArtifacts:
    best_model_path: str
    validation_accuracy: float
