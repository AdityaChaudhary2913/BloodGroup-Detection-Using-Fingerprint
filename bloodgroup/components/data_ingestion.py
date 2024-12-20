import os
import zipfile
from bloodgroup.logger import logging
from kaggle.api.kaggle_api_extended import KaggleApi
from bloodgroup.entity.config_entity import DataIngestionConfig
from bloodgroup.entity.artifact_entity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def download_data(self):
        try:
            if not os.path.exists(self.data_ingestion_config.ARTIFACTS_DIR):
                os.makedirs(self.data_ingestion_config.ARTIFACTS_DIR, exist_ok=True)
                os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser(self.data_ingestion_config.KAGGLE_CREDENTIAL_DIR)
                api = KaggleApi()
                api.authenticate()
                print(f"Downloading data from {self.data_ingestion_config.SOURCE_URL}...")
                logging.info(f"Downloading data from {self.data_ingestion_config.SOURCE_URL}...")
                api.dataset_download_files(self.data_ingestion_config.CREDENTIAL, path=self.data_ingestion_config.ARTIFACTS_DIR, unzip=True)
                print(f"Data downloaded to {self.data_ingestion_config.DATA_INGESTION_UNZIPED_ARTIFACTS_DIR}")
                logging.info(f"Data downloaded to {self.data_ingestion_config.DATA_INGESTION_UNZIPED_ARTIFACTS_DIR}")
            else:
                print(f"Data already exists at {self.data_ingestion_config.DATA_INGESTION_UNZIPED_ARTIFACTS_DIR}")
            return self.data_ingestion_config.DATA_INGESTION_UNZIPED_ARTIFACTS_DIR
        except Exception as e:
            raise Exception(f"Error in downloading data: {str(e)}")

    def initiate_data_ingestion(self):
        try:
            data_dir = self.download_data()
            return DataIngestionArtifacts(data_dir=data_dir)
        except Exception as e:
            raise Exception(f"Error in initiating data ingestion: {str(e)}")