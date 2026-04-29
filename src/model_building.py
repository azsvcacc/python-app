import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Logging Setup
# -----------------------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# -----------------------------
# Load Data
# -----------------------------
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise


# -----------------------------
# Train Model
# -----------------------------
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Mismatch in X and y samples")

        # 🔹 Hardcoded parameters (instead of YAML)
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        logger.debug('Training model with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')

        return clf

    except Exception as e:
        logger.error('Error during training: %s', e)
        raise


# -----------------------------
# Save Model
# -----------------------------
def save_model(model, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

        logger.debug('Model saved to %s', file_path)

    except Exception as e:
        logger.error('Error saving model: %s', e)
        raise


# -----------------------------
# Main Function
# -----------------------------
def main():
    try:
        train_data = load_data('./data/processed/train_tfidf.csv')

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train)

        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Pipeline failed: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()