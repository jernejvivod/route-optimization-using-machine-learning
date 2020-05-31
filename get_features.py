import numpy as np
import pandas as pd
import networkx as nx
import dill
from sklearn.preprocessing import LabelBinarizer
from feature_extraction.features import get_feature_extractor


# Read dataframe.
df = pd.read_csv('./data/trip_data/sampled.csv')

# Add column for merge with weather data.
merge_key = pd.to_datetime(df["lpep_pickup_datetime"], format='%Y-%m-%d %H:%M:%S').dt.strftime('%d-%m-%Y')
merge_key.name = 'key'
df = pd.concat((df, merge_key), axis=1)

# Parse weather data frame and pre-process.
df_weather = pd.read_csv('./data/weather_data/weather_data_nyc_centralpark_2016.csv')
df_weather['precipitation'] = df_weather['precipitation'].replace('T', 0.01).astype(float)
df_weather['snow fall'] = df_weather['snow fall'].replace('T', 0.01).astype(float)
df_weather['snow depth'] = df_weather['snow depth'].replace('T', 1).astype(float)
df_weather['date'] = pd.to_datetime(df_weather['date'], format='%d-%m-%Y').dt.strftime('%d-%m-%Y')

# Merge weather dataset.
df = pd.merge(df, df_weather, left_on='key', right_on='date')

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
with open('./features.txt', 'r') as f1, open('./features_all.txt', 'w') as f2:
    features_to_extract = list(filter(lambda x: x != '' and x[0] != '#', 
        map(lambda x: x.strip(), f1.readlines())))
    extracted_features_names_all = features_to_extract.copy()
    
    if 'pickup-datetime' in features_to_extract:
        index = extracted_features_names_all.index('pickup-datetime')
        extracted_features_names_all[index+1:index+1] = ['pickup-month', 'pickup-day', 
                'pickup-weekday', 'pickup-hour', 'pickup-minute', 'pickup-second']
        extracted_features_names_all.remove('pickup-datetime')
    if 'vendor-id' in features_to_extract:
        to_insert = ['vendor-id-' + str(idx) for idx in range(int(np.ceil(np.log2(len(encoders['VendorID'].classes_)))))]
        index = extracted_features_names_all.index('vendor-id')
        extracted_features_names_all[index+1:index+1] = to_insert
        extracted_features_names_all.remove('vendor-id')
    if 'store-and-fwd-flag' in features_to_extract:
        to_insert = ['store-and-fwd-flag' + str(idx) for idx in range(int(np.ceil(np.log2(len(encoders['Store_and_fwd_flag'].classes_)))))]
        index = extracted_features_names_all.index('store-and-fwd-flag')
        extracted_features_names_all[index+1:index+1] = to_insert
        extracted_features_names_all.remove('store-and-fwd-flag')
    if 'rate-code-id' in features_to_extract:
        to_insert = ['rate-code-id-' + str(idx) for idx in range(int(np.ceil(np.log2(len(encoders['RateCodeID'].classes_)))))]
        index = extracted_features_names_all.index('rate-code-id')
        extracted_features_names_all[index+1:index+1] = to_insert
        extracted_features_names_all.remove('rate-code-id')
    if 'payment-type' in features_to_extract:
        to_insert = ['payment-type-' + str(idx) for idx in range(int(np.ceil(np.log2(len(encoders['Payment_type'].classes_)))))]
        index = extracted_features_names_all.index('payment-type')
        extracted_features_names_all[index+1:index+1] = to_insert
        extracted_features_names_all.remove('payment-type')
    if 'trip-type' in features_to_extract:
        to_insert = ['trip-type-' + str(idx) for idx in range(int(np.ceil(np.log2(len(encoders['Trip_type '].classes_)))))]
        index = extracted_features_names_all.index('trip-type')
        extracted_features_names_all[index+1:index+1] = to_insert
        extracted_features_names_all.remove('trip-type')
    f2.write('\n'.join(extracted_features_names_all))

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

