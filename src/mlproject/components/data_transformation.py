import os
from src.mlproject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
from src.mlproject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def preprocess_data(self,data):
        
        data['Date'] = pd.to_datetime(data['Date'])
        data['Month'] = data['Date'].dt.month
        data.drop('Date', axis=1, inplace=True)
        categorical_features = ['Product Category', 'Region', 'Payment Method', 'Product Name']
        for feature in categorical_features:
            le = preprocessing.LabelEncoder()
            data[feature] = le.fit_transform(data[feature])
        return data
    
    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        # Preprocess data df_preprocessed
        data_preprocessed = self.preprocess_data(data.drop('Total Revenue', axis=1))
        data_preprocessed['Total Revenue'] = data['Total Revenue']

        # Split the data into training and test sets (0.75, 0.25) split
        train, test = train_test_split(data_preprocessed, test_size=0.2, random_state=42)

        # Save the split data to CSV files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        # Log the shapes of the split data
        logger.info("Split data into training and test sets")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")

        print(train.shape)
        print(test.shape)
