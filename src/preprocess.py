"""Module for pre-processing data saving to .csv in 'datasets' package"""

__author__ = "Abhijit Pai"
__email__ = "abhijitpai000@gmail.com"

from pathlib import Path
import pandas as pd


def _clean_to_csv(dataframe):
    """
    Private Module to save pre-processed dataframe to .csv in 'datasets' package.
    """
    preprocessed = dataframe.copy()
    file_path = Path.cwd() / "datasets/preprocessed.csv"
    preprocessed.to_csv(file_path, index=False)
    return


def clean_data(raw_file_name, save_data=True):
    """
    Pre-processes data frame and save preprocessed.csv in datasets module.

    Cleaning Steps:
    1. Extracts United Kingdom 2011 data.
    2. Drop Missing CustomerIDs
    3. Drops Negative Unit Price values
    4. Drops Invoice Number, Stock Code & Description columns

    :parameter:
        raw_file_name: .csv file name.

        save_data: bool, default=True.
            Saves cleaned dataframe to 'preprocessed.csv' in datasets package.
            NOTE: Admin access is required to save file.

    :return: Pre-processed data frame.
    """
    retail = pd.read_csv(f"datasets/{raw_file_name}",
                         encoding='ISO-8859-1',
                         parse_dates=['InvoiceDate'])

    # Drop InvoiceNo, StockCode, Description.
    retail = retail.drop(['InvoiceNo', 'StockCode', 'Description'], axis=1)

    # Extract United Kingdom 2011 data.
    retail = retail.query('Country == "United Kingdom" and InvoiceDate.dt.year == 2011')

    # Dropping rows with Missing CustomerID
    retail = retail.dropna(axis=0)

    # Drop Negative UnitPrice values.
    retail = retail.query('UnitPrice >= 0')

    if save_data:
        try:
            _clean_to_csv(retail)  # Saves to csv in 'datasets' package.
        except PermissionError:
            print("Error: Permission to save csv file denied. Admin Access is required")

    return retail
