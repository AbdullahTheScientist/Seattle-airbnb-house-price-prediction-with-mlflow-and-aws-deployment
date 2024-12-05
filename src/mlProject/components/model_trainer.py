from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject import logger
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop([self.config.target_column], axis=1)
        Y_train = train_data[[self.config.target_column]].values.ravel()

        X_test = test_data.drop([self.config.target_column], axis=1)
        Y_test = test_data[[self.config.target_column]].values.ravel()

        rf = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            max_features=self.config.max_features,
            random_state=self.config.random_state
        )

        rf.fit(X_train, Y_train)

        joblib.dump(rf, os.path.join(self.config.root_dir, self.config.model_name))
        logger.info(f"Model saved at {os.path.join(self.config.root_dir, self.config.model_name)}")