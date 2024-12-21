import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from bloodgroup.logger import logging
from bloodgroup.constants import FINAL_MODEL_NAME, CLASSES, IMAGE_SIZE, FINAL_MODEL_PATH, DEVICE
from bloodgroup.exception import CustomException
from bloodgroup.components.model_creater import initialize_model
from bloodgroup.components.data_transformation import DataTransformation
from bloodgroup.entity.config_entity import DataTransformationConfig
from bloodgroup.entity.artifact_entity import DataIngestionArtifacts


class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join(FINAL_MODEL_PATH, FINAL_MODEL_NAME)
        self.device = DEVICE
        self.classes = CLASSES 
        self.data_transformation = DataTransformation(
            data_transformation_config=DataTransformationConfig,
            data_ingestion_artifacts=DataIngestionArtifacts
        )
    
    def load_model(self):
        try:
            model = initialize_model()
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise CustomException(e, sys) from e

    def preprocess_image(self, image_path):
        try:
            logging.info(f"Preprocessing image: {image_path}")
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            image = Image.open(image_path)
            image = transform(image)
            image = image.unsqueeze(0)  # Add batch dimension
            return image
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, model, image):
        try:
            # Perform prediction
            with torch.no_grad():
                outputs = model(image)
                _, predicted_class = torch.max(outputs, 1)
                predicted_label = self.classes[predicted_class.item()]
                print(f"Prediction: {predicted_label}")
                return predicted_label
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, image_path):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            # Load the trained model
            model = self.load_model()
            
            # Preprocess the image
            image = self.preprocess_image(image_path).to(self.device)
            
            predicted_class = self.predict(model, image)
            
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_class
        except Exception as e:
            raise CustomException(e, sys) from e
