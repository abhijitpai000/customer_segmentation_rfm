# Customer Segmentation: KMeans Clustering based on RFM
**Overview:**

Recency-Frequency-Monetary Value (RFM) is a quantitative technique used to analyze & group customers based on their purchase history by assigning numerical values (1 to K) for customer groups. In this study, I computed RFM for each customer based on their purchases (using definitions below), then applied the K-Means clustering algorithm, which aims to partition N-observations into K-clusters by assigning each observation a random cluster number (0 to K), based on nearest 'Means'.

Using this randomly generated cluster numbers (0 to 4) as scores, I re-structured it to a scale of 1 to 5 and assigned the customer group that is recent, frequent, and generates high monetary value a higher score.

Lastly, computed an aggregated 'RFM Score' by averaging R, F, and M scores, where customers with high 'RFM Score' are recent, frequent, and generate high monetary value.

* **Recency:** Days since the last transcation. 
* **Frequency:** Maximum purchases.
* **Monetary Value:** Total Monetary Value Generated. 
* **RFM Score** ( RecencyScore + FrequencyScore + MonetaryValueScore ) / 3

**Applications:**
* This approach can be utilized to quantitatively understand Customer base.
* Implement Personalized Marketing strategies to customer groups.

**Data Source:**

To perform this study I used [Online Retail](https://www.kaggle.com/vijayuv/onlineretail) dataset.

**Customer Groups: RFM Scores:**
<p float="left">
  <img src="https://github.com/abhijitpai000/customer_segmentation_rfm/blob/master/figures/recencyscore.png" width="250" />
  <img src="https://github.com/abhijitpai000/customer_segmentation_rfm/blob/master/figures/frequencyscore.png" width="250" /> 
  <img src="https://github.com/abhijitpai000/customer_segmentation_rfm/blob/master/figures/monetaryvaluescore.png" width="250" />
</p>

## Final Report & Package Walk-Through

To reproduce this study, use modules in 'src' directory of this repo. (setup instructions below) and walk-through of the package is presented in the [final report](https://github.com/abhijitpai000/customer_segmentation_rfm/tree/master/report)

## Setup instructions

#### Creating Python environment

This repository has been tested on Python 3.7.6.

1. Cloning the repository:

`git clone https://github.com/abhijitpai000/customer_segmentation_rfm.git`

2. Navigate to the git clone repository.

`cd customer_segmentation_rfm`

3. Install [virtualenv](https://pypi.org/project/virtualenv/)

`pip install virtualenv`

`virtualenv rfm`

4. Activate it by running:

`rfm/Scripts/activate`

5. Install project requirements by using:

`pip install -r requirements.txt`

**Note**
To use load_raw() function, please ensure the raw .csv file is placed in '../src/datasets' directory.


