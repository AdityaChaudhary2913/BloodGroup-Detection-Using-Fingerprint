import os
import sys
import torch
from bloodgroup.logger import logging
from bloodgroup.exception import CustomException
from bloodgroup.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts
from bloodgroup.constants import DEVICE, FINAL_MODEL_PATH


class NewModelEvaluation:
    def __init__(self, model_trainer_artifacts: ModelTrainerArtifacts):
        self.model_trainer_artifacts = model_trainer_artifacts

    def load_model(self, model_path: str):
        try:
            model = torch.load(model_path)
            model.to(DEVICE)
            logging.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate_model(self, model, val_loader):
        try:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            accuracy = correct / total * 100
            logging.info(f"Model evaluation accuracy: {accuracy:.2f}%")
            print(f"Model evaluation accuracy: {accuracy:.2f}%")
            return accuracy
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        try:
            logging.info("Initiating model evaluation")
            
            val_loader = torch.load(self.model_trainer_artifacts.test_loader_path)

            # Load the new trained model
            trained_model_path = self.model_trainer_artifacts.best_model_path
            trained_model = self.load_model(trained_model_path)
            new_model_accuracy = self.evaluate_model(trained_model, val_loader)

            # Path to the existing best model
            final_model_dir = self.model_trainer_artifacts.final_model_path

            # Check if a final model exists
            if os.path.exists(final_model_dir):
                logging.info("Final model exists, loading for comparison")
                final_model = self.load_model(final_model_dir)
                final_model_accuracy = self.evaluate_model(final_model, val_loader)

                logging.info(f"New model accuracy: {new_model_accuracy:.2f}%")
                logging.info(f"Final model accuracy: {final_model_accuracy:.2f}%")

                # Replace the final model if the new model performs better
                if new_model_accuracy > final_model_accuracy:
                    torch.save(trained_model.state_dict(), final_model_dir)
                    logging.info("New model is better and has replaced the final model")
                    is_model_accepted = True
                else:
                    logging.info("New model did not outperform the final model")
                    is_model_accepted = False
            else:
                # If no final model exists, save the new model as the final model
                os.makedirs(FINAL_MODEL_PATH, exist_ok=True)
                torch.save(trained_model.state_dict(), FINAL_MODEL_PATH)
                logging.info("No existing final model. New model saved as the final model")
                is_model_accepted = True

            # Return evaluation artifacts
            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            logging.info("Model evaluation completed")
            print("Model evaluation completed")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
