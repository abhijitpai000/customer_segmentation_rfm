"""Computes Recency, Frequency & Monetary Value based on Purchase History"""

__author__ = "Abhijit Pai"
__email__ = "abhijitpai000@gmail.com"


from sklearn.cluster import KMeans
import pandas as pd


def recency(dataframe, model=False, n_clusters=None):
    """
    Computes Recency = (Closing Transaction Date - Recent Transaction Date) + 1

    model, n_clusters: When True, Trains a KMeans Cluster.
    """
    if not model:
        closing_purchase = dataframe['InvoiceDate'].max()

        recent_purchase = dataframe.groupby('CustomerID')[['InvoiceDate']].max()
        recent_purchase = recent_purchase.reset_index()
        recent_purchase.columns = ['CustomerID', 'RecentInvoiceDate']

        recent_purchase['Recency'] = closing_purchase - recent_purchase['RecentInvoiceDate']
        recent_purchase['Recency'] = recent_purchase['Recency'].dt.days
        recent_purchase['Recency'] = recent_purchase['Recency'] + 1

        output = recent_purchase[['CustomerID', 'Recency']]
        return output

    elif model:
        r_kmeans = KMeans(n_clusters=n_clusters, max_iter=1000, random_state=0)
        r_kmeans.fit(dataframe[['Recency']])
        dataframe['RecencyClusters'] = r_kmeans.predict(dataframe[['Recency']])
        return dataframe, r_kmeans


def frequency(dataframe, model=False, n_clusters=None):
    """
    Computes Frequency = Number of purchases.

    model, n_clusters: When True, Trains a KMeans Cluster.
    """
    if not model:
        freq_purchase = dataframe.groupby('CustomerID')[['InvoiceDate']].count()
        freq_purchase.reset_index(inplace=True)
        freq_purchase.columns = ['CustomerID', 'Frequency']
        return freq_purchase

    elif model:
        f_kmeans = KMeans(n_clusters=n_clusters, max_iter=1000, random_state=0)
        f_kmeans.fit(dataframe[['Frequency']])
        dataframe['FrequencyClusters'] = f_kmeans.predict(dataframe[['Frequency']])
        return dataframe, f_kmeans


def monetary_value(dataframe, model=False, n_clusters=None):
    """
    Computes Monetary Value = Revenue Generated (Unit Price x Quantity)

    model, n_clusters: When True, Trains a KMeans Cluster.
    """
    if not model:
        dataframe['Revenue'] = dataframe['UnitPrice'] * dataframe['Quantity']
        revenue = dataframe.groupby('CustomerID')[['Revenue']].mean()
        revenue.reset_index(inplace=True)
        revenue.columns = ['CustomerID', 'MonetaryValue']
        return revenue

    elif model:
        m_kmeans = KMeans(n_clusters=n_clusters, max_iter=1000, random_state=0)
        m_kmeans.fit(dataframe[['MonetaryValue']])
        dataframe['MonetaryValueClusters'] = m_kmeans.predict(dataframe[['MonetaryValue']])
        return dataframe, m_kmeans


def add_features(dataframe):
    """
    Adds Recency, Frequency and Revenue Features to data frame.

    Recency = (Closing Transaction Date - Recent Transaction Date) + 1
    Frequency = Number of purchases
    Monetary Value = Max Revenue Generated (Unit Price x Quantity)

    :parameter:
        dataframe: pandas dataframe.

    :return: data frame with CustomerID, Recency, Frequency and MonetaryValue features.
    """

    # Base data with unique customer IDs.
    customers = pd.DataFrame(dataframe.CustomerID.unique(),
                             columns=['CustomerID'])

    # Recency.
    recency_data = recency(dataframe)
    customers = customers.merge(recency_data[['CustomerID', 'Recency']], on='CustomerID')  # Merging with base data

    # Frequency.
    freq_data = frequency(dataframe)
    customers = customers.merge(freq_data[['CustomerID', 'Frequency']], on='CustomerID')  # Merging with base data

    # MonetaryValue.
    monetary_value_data = monetary_value(dataframe)
    customers = customers.merge(monetary_value_data[['CustomerID', 'MonetaryValue']], on='CustomerID')
    return customers
