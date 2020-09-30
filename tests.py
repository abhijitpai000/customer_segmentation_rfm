"""Basic I/O Test Cases to check if modules and packages are working as expected"""

__author__ = "Abhijit Pai"
__email__ = "abhijitpai000@gmail.com"

# Importing Modules.
from src.preprocess import clean_data
from src.features import add_features
from src.models import train_model

import pandas as pd


def test_data_cleaning():
    # Checks preprocess.clean_data()
    clean_df = clean_data(raw_file_name="OnlineRetail.csv", save_data=False)
    assert clean_df.shape == (337342, 5), "Data Cleaning Failed"


def test_added_features():
    # Checks features.add_features()
    features_df = pd.read_csv("datasets/preprocessed.csv")
    assert features_df.shape == (337342, 5), "Adding Features Failed"


def test_final_data():
    # Checks model.train_model()
    final = pd.read_csv("datasets/rfm_scores.csv")
    assert final.shape == (3835, 10), "RFM Scores dataset failed"
