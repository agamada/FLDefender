import numpy as np
import os
import sys
import random
import pandas as pd
from dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 40
dir_path = "UCIHAR/"

LABEL_MAP = {
    'WALKING': 0,
    'WALKING_UPSTAIRS': 1,
    'WALKING_DOWNSTAIRS': 2,
    'SITTING': 3,
    'STANDING': 4,
    'LAYING': 5,
}


def load_har_data(dir_path):
    """Load UCI HAR data from Kaggle CSV files."""
    rawdata_path = os.path.join(dir_path, "rawdata")
    train_csv = os.path.join(rawdata_path, "train.csv")
    test_csv = os.path.join(rawdata_path, "test.csv")

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"Please place train.csv and test.csv in {rawdata_path}/\n"
            "You can download from Kaggle: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones"
        )

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Extract features (all columns except 'subject' and 'Activity')
    feature_cols = [c for c in train_df.columns if c not in ('subject', 'Activity')]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['Activity'].map(LABEL_MAP).values

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['Activity'].map(LABEL_MAP).values

    return X_train, y_train, X_test, y_test


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Load UCI HAR data from CSV
    X_train, y_train, X_test, y_test = load_har_data(dir_path)

    # Combine train and test sets (same as other datasets in this project)
    dataset_image = np.concatenate([X_train, X_test], axis=0)
    dataset_label = np.concatenate([y_train, y_test], axis=0)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')
    print(f'Feature dimension: {dataset_image.shape[1]}')
    print(f'Total samples: {len(dataset_label)}')

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=6, p=0.5)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
