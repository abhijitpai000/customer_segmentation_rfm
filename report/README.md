# Customer Segmentation - KMeans Clustering based on RFM


**Overview:**


Recency-Frequency-Monetary Value (RFM) is a quantitative technique used to analyze & group customers based on their purchase history by assigning numerical values (1 to K) for customer groups. In this study, I computed RFM for each customer based on their purchases (using definitions below), then applied the K-Means clustering algorithm, which aims to partition N-observations into K-clusters by assigning each observation a random cluster number (0 to K), based on nearest 'Means'.

Using this randomly generated cluster numbers (0 to 4) as scores, I re-structured it to a scale of 1 to 5 and assigned the customer group that is recent, frequent, and generates high monetary value a higher score.


* **Recency:** Days since the last transcation. 
* **Frequency:** Transactions Count.
* **Monetary Value:** Average Monetary Value Generated. 



**Data Source :**

To perform this study I used a [Online Retail](https://www.kaggle.com/vijayuv/onlineretail) dataset from Kaggle.



**Table of Contents**


1. [Introduction](#introduction)
2. [EDA](#eda)
3. [KMeans Clustering](#kmeans)
    1. [RFM Scores with Segments](#rfm_scores)


```python
# Setting Git Clone Path as Current Working Directory (cwd).

import os
FILE_PATH = "Git\Clone\Path"
os.chdir(FILE_PATH)   # Changes cwd
os.getcwd()   # Prints cwd
```

# Introduction <a name="introduction"></a>

**Reproduce Code:**

To ensure reproducibility of this study, I put together a python package stored in 'src' directory of the GitHub Repo.

**Codebase Structure**

| Module | Function | Description |
| :--- | :--- | :--- |
| datasets | load_raw() | Loads .csv file stored in '../src/datasets/OnlineRetail.csv'
| preprocess | clean_data() | Performs pre-processing and stores data frame in '../src/datasets/preprocessed.csv'
| features | add_features() | Computes Recency, Frequency and Monetary Values for each customer.
| models | train_model() | trains k-means model saves model in '../src/pickle_models/..', and final rfm_scores as .csv in '../src/datasets/rfm_scores.csv'
| plots | generate_plots | Generates Distribution, Cat and Scatter Plots.


```python
from src.preprocess import clean_data
from src.features import add_features
from src.models import train_model
from src.plots import generate_plots

import pandas as pd
```


```python
retail_raw = pd.read_csv("datasets/OnlineRetail.csv",
                         encoding='ISO-8859-1',
                         parse_dates=['InvoiceDate'])
```


```python
retail_raw.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.55</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.75</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>



# EDA <a name="eda"></a>

**Insights**
* Country: 90% of data points are from United Kingdom.
* Quantity and UnitPrice: Negative Values observed.
    * Based on the 'Description', it seems the Negative values are for return sales.
    * Only 2 rows have negative UnitPrice.
* CustomerID: ~25% of of rows have missing IDs.

**Actions - clean_data():** Based on the EDA Insights following actions will be performed.
* Extracting 'United Kingdom - 2011' data only
* Drop two rows with Negative Unit Price.
* Drop rows with missing CustomerIDs
* Drops 'InvoiceNo', 'StockCode' and 'Description' features from dataset.


```python
# Loading pre-processed data into 'retail' dataframe.

retail = clean_data(raw_file_name="OnlineRetail.csv", save_data=True)
```


```python
retail.shape
```




    (337342, 5)




```python
retail.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42481</th>
      <td>10</td>
      <td>2011-01-04 10:00:00</td>
      <td>1.95</td>
      <td>13313.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>42482</th>
      <td>25</td>
      <td>2011-01-04 10:00:00</td>
      <td>0.42</td>
      <td>13313.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>42483</th>
      <td>25</td>
      <td>2011-01-04 10:00:00</td>
      <td>0.42</td>
      <td>13313.0</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>



## Recency, Frequency and Monetary value for each customer.

**Actions - add_features():** Computes the following using 'InvoiceDate' of each customer transaction.

* Recency = (Closing Transaction Date - Recent Transaction Date) + 1
* Frequency = Number of purchases
* Monetary Value = Max Revenue Generated (Unit Price x Quantity)


```python
# Loading dataset with features to 'customers' dataframe.

customers = add_features(retail)
```


```python
customers.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Recency</th>
      <th>Frequency</th>
      <th>MonetaryValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13313.0</td>
      <td>22</td>
      <td>78</td>
      <td>19.940000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18097.0</td>
      <td>8</td>
      <td>102</td>
      <td>24.611078</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16656.0</td>
      <td>22</td>
      <td>76</td>
      <td>107.098421</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers.shape
```




    (3835, 4)




```python
# Generating Distribution Plots.

generate_plots(customers, plot_type="distribution")
```


![png](output_14_0.png)


# K-Means Clustering for Recency, Frequency, Monetary Value <a name="kmeans"></a>


**Actions - train_model():** Computes the following.
* Trains K-Means models.
* Re-structures K-Means cluster numbers, ensuring "higher the better"
* Computes Aggregated RFM Score for each customer.
* Categorizes RFM Score into 3 Segments: low_score, medium_score & high_score


```python
customers = train_model(customers, n_clusters=5, save_model=True)
```


```python
customers.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Recency</th>
      <th>Frequency</th>
      <th>MonetaryValue</th>
      <th>RecencyClusters</th>
      <th>FrequencyClusters</th>
      <th>MonetaryValueClusters</th>
      <th>RecencyScore</th>
      <th>FrequencyScore</th>
      <th>MonetaryValueScore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13313.0</td>
      <td>22</td>
      <td>78</td>
      <td>19.940000</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18097.0</td>
      <td>8</td>
      <td>102</td>
      <td>24.611078</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16656.0</td>
      <td>22</td>
      <td>76</td>
      <td>107.098421</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plotting Recency, Frequency and Monetary Value Scores.

generate_plots(customers, plot_type="cat_plot")
```


    <Figure size 360x288 with 0 Axes>



![png](output_18_1.png)



    <Figure size 360x288 with 0 Axes>



![png](output_18_3.png)



    <Figure size 360x288 with 0 Axes>



![png](output_18_5.png)


**End**
