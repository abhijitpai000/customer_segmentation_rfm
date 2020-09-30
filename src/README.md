## Documentation

| Module | Function | Description | Parameters | Yields | Returns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| preprocess | clean_data()	 | Performs pre-processing | raw_file_name, save_data=True | preprocessed.csv | preprocessed_df
| features	 | add_features()	 | Computes Recency, Frequency and Monetary Values for each customer. | dataframe | -- | customers_rfm_df
| models	 | train_model()	 | trains k-means model |dataframe, n_clusters=5, save_model=True| rfm_scores.csv, recency.pkl, frequency.pkl & monetaryvalue.pkl | rfm_scores.
| plots	 | generate_plots()	 | Generates Distribution, Cat and Scatter Plots. | dataframe, plot_type="distribution" | -- | ---
                                                                                  
