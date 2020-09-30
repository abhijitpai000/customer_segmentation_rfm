"""Trains KMeans Model, Computes RFM Score and saves pkl version"""

__author__ = "Abhijit Pai"
__email__ = "abhijitpai000@gmail.com"


from src.features import recency, frequency, monetary_value
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def _train_to_joblib(recency_model, frequency_model, monetary_model):
    """
    Private function to save joblib models.
    """
    model_path = Path.cwd() / "models/"
    try:
        joblib.dump(recency_model, model_path/"recency_model.pkl")
        joblib.dump(frequency_model, model_path/"frequency_model.pkl")
        joblib.dump(monetary_model, model_path/"monetary_model.pkl")
    except PermissionError:
        print("Error Permission to dump pickle models denied. Admin Access is required")
    return


def _df_to_csv(dataframe):
    rfm_scores = dataframe.copy()
    file_path = Path.cwd() / "datasets/rfm_scores.csv"
    try:
        rfm_scores.to_csv(file_path, index=False)
    except PermissionError:
        print("Error: Permission to dump rfm_scores.csv denied. Admin Access is required")
    return


def sort_clusters(dataframe, estimator, new_field_name, ascending=True):
    """
    Sorts clusters based on order specified by 'ascending'.

    :parameters:
        dataframe: dataframe with KMeans clusters.

        estimator: KMeans model that was fit to corresponding 'cluster_field'

        new_field_name: str, field name to sorted clusters.

        ascending: bool, default=True.
            Higher Cluster Number will be assigned to Higher values.

    :returns: dataframe with new clusters.
    """

    if ascending:
        indexes = np.argsort(estimator.cluster_centers_.sum(axis=1))  # ascending sort.
    if not ascending:
        indexes = np.argsort(-(estimator.cluster_centers_.sum(axis=1)))  # descending sort.

    # Look up array to swap cluster values.
    lookup = np.zeros_like(indexes)
    lookup[indexes] = np.arange(len(indexes))

    # New Clusters.
    new_clusters = pd.DataFrame(lookup[estimator.labels_],
                                columns=[new_field_name],
                                index=dataframe.index)

    new_clusters = new_clusters + 1  # Adding 1 for RFM Score scale (1-5)
    output = dataframe.join(new_clusters, how='outer')
    return output


def train_model(dataframe, n_clusters=5, save_model=True):
    """
    Trains KMeans Model with 'n_clusters' for 'feature_field'.

    :parameters:
        dataframe: Pandas data frame.

        n_clusters: int, default=5.
            Number of clusters.

        save_model: bool, default=True.
            saves joblib models.

    :return: Data frame with RecencyScore, FrequencyScore, MonetaryValueScore and RFM Score.
    """
    # Recency.
    basedata, r_kmeans = recency(dataframe, model=True, n_clusters=n_clusters)
    # Frequency.
    basedata, f_kmeans = frequency(basedata, model=True, n_clusters=n_clusters)
    # Monetary Value.
    basedata, m_kmeans = monetary_value(basedata, model=True, n_clusters=n_clusters)

    # Restructuring Clusters.
    r_basedata = sort_clusters(basedata, r_kmeans, 'RecencyScore', ascending=False)
    f_basedata = sort_clusters(r_basedata, f_kmeans, 'FrequencyScore', ascending=True)
    sorted_df = sort_clusters(f_basedata, m_kmeans, 'MonetaryValueScore', ascending=True)

    if save_model:
        # Saving final_df to .csv.
        _df_to_csv(sorted_df)

        # Save to joblib.
        _train_to_joblib(r_kmeans, f_kmeans, m_kmeans)

    return sorted_df


