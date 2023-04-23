# Preprocessing for the IoT23 and UCI datasets can be found in the source code for the SVM
# IoT-23:
    - encoded categorical variables with sklearn's LabelEncoder class
    - dropped irrelevant features (uid)
    - dropped features where values were only found within a miniscule amount of datapoints (service, duration, orig_bytes, etc.) as
      to prevent any influence they would have on the model
    - of the original ~60 million datapoints, ~76,000 were selected to train/test the model

# UCI:
    - no additional
    - out of the original ~70 million datapoints, ~90,000 were selected to train/test the model
    - all data points were from the Danmini Smart Doorbell data
    - the separate .csv files (ack.csv, benign_traffic.csv, etc.) were combined into a single .csv files