import numpy as np
import pandas as pd
import geopy.distance as geodesic_distance

def get_feature_extractor(features, encoders):
    """
    Get function for extracting features from a single sample.

    Args:
        features (list): List of features to extract
        encoders (dict): Dictionary mapping feature names to
        fitted Scikit-Learn one-hot encoders.

    Returns:
        (numpy.ndarray): Array containing the values of extracted
        feature.
    """

    def get_feature(sample, feature):
        if feature == 'vendor-id':

            # Get encoding of Vendor's ID.
            return encoders['VendorID'].transform([sample['VendorID']])[0]

        elif feature == 'pickup-datetime':

            # Get month, day, day of week, hours, minutes, seconds.
            datetime = pd.to_datetime(sample['lpep_pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
            return np.array([datetime.month, datetime.day, datetime.weekday(), datetime.hour, datetime.minute, datetime.second])

        elif feature == 'store-and-fwd-flag':

            # Get encoding of Store_and_fwd_flag value.
            return encoders['Store_and_fwd_flag'].transform([sample['Store_and_fwd_flag']])[0]

        elif feature == 'rate-code-id':
            
            # Get encoding of RateCodeID value.
            return encoders['RateCodeID'].transform([sample['RateCodeID']])[0]

        elif feature == 'pickup-longitude':
            
            # Get pickup longitude.
            return np.array([sample["Pickup_longitude"]])

        elif feature == 'pickup-latitude':

            # Get pickup latitude.
            return np.array([sample["Pickup_latitude"]])

        elif feature == 'dropoff-longitude':
            
            # Get dropoff longitude.
            return np.array([sample['Dropoff_longitude']])

        elif feature == 'dropoff-latitude':

            # Get dropoff latitude.
            return np.array([sample['Dropoff_latitude']])

        elif feature == 'longitude-difference':
            
            # Compute longitude difference between pickup and dropoff.
            return np.array([sample['Dropoff_longitude'] - sample['Pickup_longitude']])

        elif feature == 'latitude-difference':

            # Compute latitude difference between pickup and dropoff.
            return np.array([sample['Dropoff_latitude'] - sample['Pickup_longitude']])

        elif feature == 'geodesic-dist':

            # Compute geodesic distance of trip.
            return np.array([geodesic_distance.distance((sample['Pickup_latitude'], sample['Pickup_longitude']), 
                (sample['Dropoff_latitude'], sample['Dropoff_longitude'])).km])

        elif feature == 'passenger-count':
            
            # Get passenger count feature.
            return np.array([sample['Passenger_count']])

        elif feature == 'trip-distance':
            
            # Get trip distance feature.
            return np.array([sample['Trip_distance']])

        elif feature == 'fare-amount':
            
            # Get fare amount feature.
            return np.array([sample['Fare_amount']])

        elif feature == 'extra':
            
            # Get extra feature.
            return np.array([sample['Extra']])

        elif feature == 'mta-tax':
            
            # Get MTA tax feature.
            return np.array([sample['MTA_tax']])

        elif feature == 'tip-amount':
            
            # Get tip amount feature.
            return np.array([sample['Tip_amount']])


        elif feature == 'tolls-amount':
            
            # Get tolls amount feature.
            return np.array([sample['Tolls_amount']])

        elif feature == 'ehail-fee':

            # Get ehail fee feature.
            return np.array([sample['Ehail_fee']])

        elif feature == 'improvement-surcharge':
        
            # Get improvement surcharge feature.
            return np.array([sample['improvement_surcharge']])

        elif feature == 'total-amount':
        
            # Get tolls amount feature.
            return np.array([sample['Total_amount']])

        elif feature == 'payment-type':
            
            # Get payment type feature.
            return np.array([sample['Payment_type']])

        elif feature == 'trip-type':

            # Get trip type feature.
            return np.array([sample['Trip_type ']])


    def feature_extractor(features, sample):
        """
        Function for extracting specified features from a specified sample.

        Args:
            features (list): List of features to extract
            sample (object): Row of a Pandas dataframe returned by the iterrows() iterator.

        Returns:
            (numpy.ndarray): Array of computed features.

        """
        return np.hstack([get_feature(sample, feature) for feature in features])

    
    # Return function that takes a sample and extracts specified features.
    return (lambda sample: feature_extractor(features, sample))

