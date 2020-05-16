import numpy as np
import pandas as pd
import dill
import joblib
import geopy.distance as geodesic_dist
from collections import OrderedDict


def get_dist_func(network, which='learned', prediction_model='rf'):
    """
    Get distance function for measuring distances between nodes
    in network. The distance function accepts indices of two
    nodes and returns the distance.

    Args:
        network (object): Networkx representation of the network.
        which (str): Type of distance function to return.
        prediction_model (str): The prediction model to use.

    Returns:
        (function): The distance function used to measure
        the distance between the nodes.
    """


    ### DISTANCE FUNCTIONS ###
    def dist_func_geodesic(n1, n2):

        # Get coordinates and compute geodesic distance.
        coor1 = network.nodes[n1]['latlon']
        coor2 = network.nodes[n2]['latlon']
        return geodesic_dist.distance(coor1, coor2).km

    def dist_func_learned(n1, n2):

        # Get coordinates.
        coor1 = network.nodes[n1]['latlon']
        coor2 = network.nodes[n2]['latlon']

        # If using pre-set trip parameters.
        trip_params = {
                'VendorID' : 2,
                'lpep_pickup_datetime' : '2016-01-11 12:46:22',
                'Store_and_fwd_flag' : 'N',
                'RateCodeID' : 1,
                'Pickup_longitude' : coor1[1],
                'Pickup_latitude' : coor1[0],
                'Dropoff_longitude' : coor2[1] ,
                'Dropoff_latitude' : coor2[0],
                'Passenger_count' : 1,
                'Trip_distance' : 0.621371 * 6371 * (abs(2 * np.arctan2(np.sqrt(np.square(np.sin((abs(coor2[1] - coor1[1]) * np.pi / 180) / 2))), 
                                  np.sqrt(1-(np.square(np.sin((abs(coor2[1] - coor1[1]) * np.pi / 180) / 2)))))) + \
                                     abs(2 * np.arctan2(np.sqrt(np.square(np.sin((abs(coor2[1] - coor1[1]) * np.pi / 180) / 2))), 
                                  np.sqrt(1-(np.square(np.sin((abs(coor2[0] - coor1[0]) * np.pi / 180) / 2))))))),
                'Fare_amount' : 9.0,
                'Extra' : 0.5,
                'MTA_tax' : 0.5,
                'Tip_amount' : 0.0,
                'Tolls_amount' : 0.0,
                'Ehail_fee' : -1,
                'improvement_surcharge' : 0.3,
                'Total_amount' : 11.16,
                'Payment_type' : 2.0,
                'Trip_type ' : 1.0,
            }

        
        # Get input for feature extractor and use feature extractor
        # to get processed feature vector.
        feature_df = pd.Series(trip_params)
        feature_vec = feature_extractor(feature_df)

        # Use model to make prediction.
        return model.predict(feature_vec[np.newaxis])[0]


    ##########################

    # Return specified distance function.
    if which == 'geodesic':
        return dist_func_geodesic
    elif which == 'learned':
        model = joblib.load('./models/' + prediction_model + '.joblib')
        with open('./data/feature_extractor.p', 'rb') as f:
            feature_extractor = dill.load(f)

        return dist_func_learned

