## Documentation

| Module | Function | Description | Parameters | Yields | Returns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| datasets | 	load_raw() | Loads Raw data | data_file_name | -- | dataframe. 
| preprocess | clean_data()	 | Performs pre-processing | dataframe, save_data=True | preprocessed.csv | preprocessed_df
| features	 | add_features()	 | Computes Recency, Frequency and Monetary Values for each customer. | dataframe | -- | customers_rfm_df
| models	 | train_model()	 | trains k-means model |dataframe, n_clusters=5, save_model=True| rfm_scores.csv, recency.pkl, frequency.pkl & monetaryvalue.pkl | rfm_scores.
| plots	 | generate_plots()	 | Generates Distribution, Cat and Scatter Plots. | dataframe, plot_type="distribution" | -- | ---
                                                                                  
