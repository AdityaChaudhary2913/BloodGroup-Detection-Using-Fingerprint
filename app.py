from bloodgroup.pipeline.training_pipeline import TrainPipeline
from bloodgroup.pipeline.prediction_pipeline import PredictionPipeline

if __name__ == "__main__":
    train = TrainPipeline()
    train.run_pipeline()
    predict = PredictionPipeline()
    predict.run_pipeline("/Users/adityachaudhary/Desktop/Important Projects/Data Science/BloodGroup-Detection-Using-Fingerprint/artifacts/DataIngestionArtifacts/dataset_blood_group/O-/cluster_7_5958.BMP") 