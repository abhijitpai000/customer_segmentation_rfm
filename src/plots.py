"""Module to Generate Plots"""

import matplotlib.pyplot as plt
import seaborn as sns


def generate_plots(dataframe, plot_type="distribution"):
    """
    Generates the following Plots.

    - Distribution Plots
    - Cat Plot for Scores.
    - Scatterplot for RFM Score.

    :parameters:
        dataframe: Pandas dataframe.

        plot_type: str, default="distribution"
            distribution: Distribution plot for Recency, Frequency, Monetary Value of purchases.
            cat_plot: Cat plot for Recency, Frequency, Monetary Value Scores computed by KMeans.
    """
    if plot_type == "distribution":
        plt.figure(figsize=(20, 6))

        plt.subplot(131)
        sns.distplot(dataframe['Recency'])
        plt.title('Recency Distribution')

        plt.subplot(132)
        plt.title('Frequency Distribution')
        sns.distplot(dataframe['Frequency'], bins=20)

        plt.subplot(133)
        plt.title('Monetary Value Distribution')
        sns.distplot(dataframe['MonetaryValue'])
        plt.suptitle("Distributions")

    elif plot_type == "cat_plot":
        # Recency.
        plt.figure(figsize=(5, 4));
        plt.title
        sns.catplot(x='RecencyScore', y='Recency', data=dataframe, palette='pastel');

        # Frequency.
        plt.figure(figsize=(5, 4));
        sns.catplot(x='FrequencyScore', y='Frequency', data=dataframe, palette='pastel');

        # Monetary Value.
        plt.figure(figsize=(5, 4));
        sns.catplot(x='MonetaryValueScore', y='MonetaryValue', data=dataframe, palette='pastel');
    return
