import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from bloodgroup.logger import logging
import matplotlib.pyplot as plt
from typing import Tuple
from bloodgroup.entity.config_entity import DataTransformationConfig
from bloodgroup.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts
from bloodgroup.constants import IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifacts: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    class BloodGroupDataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.image_paths = []
            self.labels = []

            # Map class names to numerical labels
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(data_dir)))}

            # Load all image paths and their corresponding labels
            for cls_name in os.listdir(data_dir):
                class_dir = os.path.join(data_dir, cls_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[cls_name])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            label = self.labels[idx]

            # Load and preprocess the image
            image = Image.open(img_path).convert("L")
            if self.transform:
                image = self.transform(image)

            return image, label

    def get_train_transforms(self):
        return transforms.Compose([
            transforms.RandomRotation(degrees=15),  # Rotate images ±15 degrees
            transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
            transforms.RandomVerticalFlip(p=0.5),    # 50% chance to flip vertically
            transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8, 1.0)),  # Random crop
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness and contrast
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])

    def get_normalization_transforms(self):
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])

    def prepare_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # Create datasets with appropriate transformations
        train_transform = self.get_train_transforms()
        val_transform = self.get_normalization_transforms()

        dataset = self.BloodGroupDataset(self.data_ingestion_artifacts.data_dir, transform=train_transform)

        # Split dataset into train, validation, and test sets
        train_size = int(0.7 * len(dataset))  # 70% for training
        val_size = int(0.15 * len(dataset))   # 15% for validation
        remaining_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, remaining_size])

        # Apply validation transform to validation and test datasets
        val_dataset.dataset.transform = val_transform
        test_dataset.dataset.transform = val_transform

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        return train_loader, val_loader, test_loader

    def save_dataloaders(self, train_loader, val_loader, test_loader):
        try:
            os.makedirs(self.data_transformation_config.ARTIFACTS_DIR, exist_ok=True)

            train_path = self.data_transformation_config.TRAIN_LOADER_PATH
            val_path = self.data_transformation_config.VAL_LOADER_PATH
            test_path = self.data_transformation_config.TEST_LOADER_PATH
            
            torch.save(train_loader, train_path)
            torch.save(val_loader, val_path)
            torch.save(test_loader, test_path)

            logging.info(f"Saved train loader at {train_path}")
            logging.info(f"Saved validation loader at {val_path}")
            logging.info(f"Saved test loader at {test_path}")

        except Exception as e:
            raise Exception(f"Error saving dataloaders: {str(e)}") from e

    def initiate_data_transformation(self):
        try:
            logging.info("Starting data preprocessing pipeline")
            train_loader, val_loader, test_loader = self.prepare_dataloaders()
            self.save_dataloaders(train_loader, val_loader, test_loader)

            data_transformation_artifacts = DataTransformationArtifacts(
                train_loader_path=self.data_transformation_config.TRAIN_LOADER_PATH,
                val_loader_path=self.data_transformation_config.VAL_LOADER_PATH,
                test_loader_path=self.data_transformation_config.TEST_LOADER_PATH
            )

            logging.info("Data preprocessing pipeline completed")
            return data_transformation_artifacts
        except Exception as e:
            raise Exception(f"Error in data transformation pipeline: {str(e)}") from e

    @staticmethod
    def visualize_augmentations(dataset, num_images=5):
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        for i in range(num_images):
            image, label = dataset[i]
            axes[i].imshow(image.permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Unnormalize for visualization
            axes[i].set_title(f"Class: {label}")
            axes[i].axis("off")
        plt.show()