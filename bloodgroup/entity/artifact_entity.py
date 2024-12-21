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
    final_model_after_training_path: str
    validation_accuracy: float
    final_model_after_evaluation_path: str
    test_loader_path: str
    
@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool
