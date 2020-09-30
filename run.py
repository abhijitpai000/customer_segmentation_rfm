"""Run Source Code through command line"""

__author__ = "Abhijit Pai"
__email__ = "abhijitpai000@gmail.com"


# Imports from modules in codebase 'src'.
from src.preprocess import clean_data
from src.features import add_features
from src.models import train_model


if __name__ == '__main__':
    clean_df = clean_data(raw_file_name="OnlineRetail.csv")
    features_df = add_features(clean_df)
    final_df = train_model(features_df)








