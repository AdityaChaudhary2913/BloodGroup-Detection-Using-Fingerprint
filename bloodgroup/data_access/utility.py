import os
import torch
from bloodgroup.logger import logging
from bloodgroup.constants import DEVICE

def save_model(model: torch.nn.Module, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(model.state_dict(), file_path)
        logging.info(f"Model saved successfully at {file_path}")
    except Exception as e:
        raise Exception(f"Error saving model: {str(e)}") from e

def load_model(model: torch.nn.Module, file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No model found at {file_path}")
        model.load_state_dict(torch.load(file_path, map_location=DEVICE))
        model.to(DEVICE)
        logging.info(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}") from e
