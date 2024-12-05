import os
from mlProject import logger
from mlProject.entity.config_entity import DataTransformationConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

class DataTransformation:
    def __init__(self, config):
        self.config = config

    def train_test_spliting(self):
        # Load data
        data = pd.read_csv(self.config.data_path)

        # Drop the first two columns
        data = data.drop(columns=data.columns[:2])  # Dropping the first two columns

        # Encode categorical features
        categorical_columns = data.select_dtypes(include=['object']).columns  # Adjust if needed
        label_encoders = {}
        for column in categorical_columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
            label_encoders[column] = le  # Save encoder for later use

        # Save encoders using joblib for future use
        encoder_save_path = os.path.join(self.config.root_dir, "label_encoders.joblib")
        joblib.dump(label_encoders, encoder_save_path)
        logger.info(f"Saved label encoders to {encoder_save_path}")

        logger.info("Encoded categorical features")
        logger.info(f"Categorical columns: {list(categorical_columns)}")

        # Split into train and test
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Save to CSV
        train_data.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test_data.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Split data into train and test")
        logger.info(f"Train data: {train_data.shape}")
        logger.info(f"Test data: {test_data.shape}")
