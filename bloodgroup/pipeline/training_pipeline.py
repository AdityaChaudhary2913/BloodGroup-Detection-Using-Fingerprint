import sys
import time
from zipfile import Path
from bloodgroup.logger import logging
from bloodgroup.exception import CustomException
from bloodgroup.components.data_ingestion import DataIngestion
from bloodgroup.components.data_transformation import DataTransformation
from bloodgroup.components.model_trainer import ModelTrainer
# from bloodgroup.components.model_evaluation import ModelEvaluation
# from bloodgroup.components.model_pusher import ModelPusher

from bloodgroup.entity.config_entity import (DataIngestionConfig,
                                        DataTransformationConfig,
                                        ModelTrainerConfig,
                                    #    ModelEvaluationConfig,
                                    #    ModelPusherConfig
                                       )

from bloodgroup.entity.artifact_entity import (DataIngestionArtifacts,
                                            DataTransformationArtifacts,
                                            ModelTrainerArtifacts,
#                                          ModelEvaluationArtifacts,
#                                          ModelPusherArtifacts
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        # self.model_evaluation_config =ModelEvaluationConfig()
        # self.model_pusher_config = ModelPusherConfig()
        
    def start_data_ingestion(self) -> Path:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logging.info("Getting the data from Kaggle")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from Kaggle")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e    

    def start_data_transformation(self, data_ingestion_artifacts = DataIngestionArtifacts) -> DataTransformationArtifacts:
        logging.info("Entered the start_data_transformation method of TrainPipeline class")
        try:
            data_transformation = DataTransformation(data_ingestion_artifacts = data_ingestion_artifacts,data_transformation_config=self.data_transformation_config)
            data_transformation_artifacts = data_transformation.initiate_data_transformation()
            logging.info("Exited the start_data_transformation method of TrainPipeline class")
            return data_transformation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def start_model_trainer(self, data_transformation_artifacts: DataTransformationArtifacts) -> ModelTrainerArtifacts:
        logging.info("Entered the start_model_trainer method of TrainPipeline class")
        try:
            model_trainer = ModelTrainer(data_transformation_artifacts=data_transformation_artifacts,model_trainer_config=self.model_trainer_config)
            model_trainer_artifacts = model_trainer.initiate_model_training()
            logging.info("Exited the start_model_trainer method of TrainPipeline class")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys) 

    # def start_model_evaluation(self, model_trainer_artifacts: ModelTrainerArtifacts, data_transformation_artifacts: DataTransformationArtifacts) -> ModelEvaluationArtifacts:
    #     logging.info("Entered the start_model_evaluation method of TrainPipeline class")
    #     try:
    #         model_evaluation = ModelEvaluation(data_transformation_artifacts = data_transformation_artifacts,
    #                                             model_evaluation_config=self.model_evaluation_config,
    #                                             model_trainer_artifacts=model_trainer_artifacts)
    #         model_evaluation_artifacts = model_evaluation.initiate_model_evaluation()
    #         logging.info("Exited the start_model_evaluation method of TrainPipeline class")
    #         return model_evaluation_artifacts
    #     except Exception as e:
    #         raise CustomException(e, sys) from e
        
    # def start_model_pusher(self,) -> ModelPusherArtifacts:
    #     logging.info("Entered the start_model_pusher method of TrainPipeline class")
    #     try:
    #         model_pusher = ModelPusher(model_pusher_config=self.model_pusher_config)
    #         model_pusher_artifact = model_pusher.initiate_model_pusher()
    #         logging.info("Initiated the model pusher")
    #         logging.info("Exited the start_model_pusher method of TrainPipeline class")
    #         return model_pusher_artifact
    #     except Exception as e:
    #         raise CustomException(e, sys) from e

    def run_pipeline(self):
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            print("\n\nStarted Data Ingestion - Approx Duration: 1 minute\n")
            start_time = time.time()
            data_ingestion_artifacts = self.start_data_ingestion()
            end_time = time.time()
            ingestion_duration = (end_time - start_time) / 60
            print(f"Ended Data Ingestion - Duration: {ingestion_duration:.2f} minutes\n\n")
            
            print("Started Data Transformation - Approx Duration: 5 minute\n")
            start_time = time.time()
            data_transformation_artifacts = self.start_data_transformation(data_ingestion_artifacts=data_ingestion_artifacts)
            end_time = time.time()
            transformation_duration = (end_time - start_time) / 60
            print(f"Ended Data Transformation - Duration: {transformation_duration:.2f} minutes\n\n")
            
            print("Started Model Training - Approx Duration: 3 hour\n")
            start_time = time.time()
            model_trainer_artifacts = self.start_model_trainer(data_transformation_artifacts=data_transformation_artifacts)
            end_time = time.time()
            training_duration = ((end_time - start_time) / 60)
            print(f"Ended Model Training - Duration: {training_duration:.2f} minutes\n\n")
            
            # print("Started Model Evaluation - Approx Duration: 1 hour\n")
            # start_time = time.time()
            # model_evaluation_artifacts = self.start_model_evaluation(model_trainer_artifacts=model_trainer_artifacts,data_transformation_artifacts=data_transformation_artifacts)
            # end_time = time.time()
            # evaluation_duration = ((end_time - start_time) / 60)
            # print(f"Ended Model Evalution - Duration: {evaluation_duration:.2f} minutes\n\n")
            
            # if not model_evaluation_artifacts.is_model_accepted:
            #     raise Exception("Trained model is not better than the best model")
            
            # print("Started Model Pushing - Approx Duration: 1 hour\n")
            # start_time = time.time()
            # self.start_model_pusher()
            # end_time = time.time()
            # pushing_duration = ((end_time - start_time) / 60)
            # print(f"Ended Model Pushing - Duration: {pushing_duration:.2f} minutes\n\n")
            
            # print(f"Total Training Pipeline - Duration: {pushing_duration + evaluation_duration + training_duration + transformation_duration + ingestion_duration:.2f} minutes\n\n")

            logging.info("Exited the run_pipeline method of TrainPipeline class") 
        except Exception as e:
            raise CustomException(e, sys) from e
