import numpy as np
import pandas as pd
import networkx as nx
import dill
from sklearn.preprocessing import LabelBinarizer
from feature_extraction.features import get_feature_extractor


# Read dataframe.
df = pd.read_csv('./data/trip_data/sampled.csv')

# Extract target variable values.
pickup_time = pd.to_datetime(df["lpep_pickup_datetime"], format='%Y-%m-%d %H:%M:%S')
dropoff_time = pd.to_datetime(df["Lpep_dropoff_datetime"], format='%Y-%m-%d %H:%M:%S')
target = ((dropoff_time - pickup_time).dt.total_seconds()/60.0).to_numpy()

# Set any NaN values to -1.
df.iloc[np.where(np.isnan(df['VendorID']))[0], df.columns.get_loc('VendorID')] = -1
df.iloc[np.where(np.isnan(df['RateCodeID']))[0], df.columns.get_loc('RateCodeID')] = -1
df.iloc[np.where(np.isnan(df['Payment_type']))[0], df.columns.get_loc('Payment_type')] = -1
df.iloc[np.where(np.isnan(df['Trip_type ']))[0], df.columns.get_loc('Trip_type ')] = -1
df.iloc[np.where(np.isnan(df['Ehail_fee']))[0], df.columns.get_loc('Ehail_fee')] = -1

# Get one-hot encoders for categorical features.
encoders = {
        'VendorID' : LabelBinarizer().fit(df['VendorID']),
        'Store_and_fwd_flag' : LabelBinarizer().fit(df['Store_and_fwd_flag']),
        'RateCodeID' : LabelBinarizer().fit(df['RateCodeID']),
        'Payment_type' : LabelBinarizer().fit(df['Payment_type']),
        'Trip_type ' : LabelBinarizer().fit(df['Trip_type '])
}

# Parse list of features to extract.
with open('./features.txt', 'r') as f:
        features_to_extract = list(filter(lambda x: x != '' and x[0] != '#', 
            map(lambda x: x.strip(), f.readlines())))

# Get feature extraction function that extract features from a given sample.
feature_extractor = get_feature_extractor(features_to_extract, encoders)


# Extract features and get data matrix.
data = np.vstack([feature_extractor(sample[1]) for sample in df.iterrows()])

# Save constructed features and target variable values.
np.save('./data/features_data.npy', data)
np.save('./data/features_target.npy', target)

# Save feature extractor.
with open('./data/feature_extractor.p', 'wb') as f:
    dill.dump(feature_extractor, f)

