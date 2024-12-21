import os
import torch
from bloodgroup.logger import logging
from bloodgroup.entity.config_entity import ModelTrainerConfig
from bloodgroup.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts
from bloodgroup.constants import DEVICE
from bloodgroup.components.model_creater import initialize_model, get_loss_and_optimizer

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifacts: DataTransformationArtifacts):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifacts = data_transformation_artifacts

    def train_epoch(self, model, train_loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        logging.info(f"Training loss: {epoch_loss}")
        print(f"\nTraining loss: {epoch_loss}\n")
        return epoch_loss

    def validate_epoch(self, model, val_loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_loss = running_loss / len(val_loader)
        val_accuracy = correct / total * 100
        logging.info(f"Validation loss: {val_loss}, Accuracy: {val_accuracy:.2f}%")
        print(f"\nValidation loss: {val_loss}, Accuracy: {val_accuracy:.2f}%\n")
        return val_loss, val_accuracy

    def train_and_validate(self, model, train_loader, val_loader, criterion, optimizer):
        best_accuracy = 0.0
        best_loss = float('inf')
        patience_counter = 0
        
        best_model_dir = self.model_trainer_config.BEST_MODEL_PATH
        os.makedirs(best_model_dir, exist_ok=True)
        best_model_path = os.path.join(best_model_dir, self.model_trainer_config.BEST_MODEL_NAME)
        
        for epoch in range(self.model_trainer_config.EPOCHS):
            logging.info(f"Epoch {epoch + 1}/{self.model_trainer_config.EPOCHS}")
            print(f"\nEpoch {epoch + 1}/{self.model_trainer_config.EPOCHS}\n")
            self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_accuracy = self.validate_epoch(model, val_loader, criterion)
            if val_loss < best_loss:
                best_loss = val_loss
                best_accuracy = val_accuracy
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"Saved best model with validation loss: {best_loss:.4f}, accuracy: {best_accuracy:.2f}%")
                print(f"\nSaved best model with validation loss: {best_loss:.4f}, accuracy: {best_accuracy:.2f}%\n")
            else:
                patience_counter += 1
                logging.info(f"No improvement in validation loss for {patience_counter} consecutive epochs")
                print(f"\nNo improvement in validation loss for {patience_counter} consecutive epochs\n")
            if patience_counter >= self.model_trainer_config.PATIENCE:
                logging.info("Early stopping triggered")
                print("\nEarly stopping triggered\n")
                break
        return best_accuracy

    def initiate_model_training(self):
        try:
            logging.info("Starting model training pipeline")
            
            torch.serialization.add_safe_globals([torch.utils.data.dataloader.DataLoader])

            # Load DataLoaders
            train_loader = torch.load(self.data_transformation_artifacts.train_loader_path)
            val_loader = torch.load(self.data_transformation_artifacts.val_loader_path)

            # Initialize model, loss function, and optimizer
            model = initialize_model()
            criterion, optimizer = get_loss_and_optimizer(model)

            # Train and validate
            best_accuracy = self.train_and_validate(model, train_loader, val_loader, criterion, optimizer)
            
            # Save final model after training
            final_model_dir = self.model_trainer_config.FINAL_MODEL_PATH
            os.makedirs(final_model_dir, exist_ok=True)
            final_model_path = os.path.join(final_model_dir, self.model_trainer_config.FINAL_MODEL_NAME)
            torch.save(model.state_dict(), final_model_path)
            logging.info(f"Final model saved at: {final_model_path}")
            print(f"\nFinal model saved at: {final_model_path}\n")

            # Save training artifacts
            model_trainer_artifacts = ModelTrainerArtifacts(
                best_model_path=self.model_trainer_config.BEST_MODEL_PATH,
                validation_accuracy=best_accuracy,
                final_model_path=final_model_path
            )
            logging.info(f"Model training completed with best accuracy: {best_accuracy:.2f}%")
            print(f"\nModel training completed with best accuracy: {best_accuracy:.2f}%\n")
            return model_trainer_artifacts
        except Exception as e:
            raise Exception(f"Error in model training pipeline: {str(e)}") from e
