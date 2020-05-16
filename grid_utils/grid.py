import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
import random
import gmplot
from geopy import distance


def geodesic_dist(p1, p2):
    """
    Compute geodesic distance between points
    p1 and p2 described by latitude and longitude.

    Args:
        p1 (numpy.ndarray): The first point
        p2 (numpy.ndarray): The second point
    """
    return distance.distance(p1, p2).m


def group_distance(g1, g2, method='average-linkage'):
    """
    Compute distance between the specified groups using specified method and
    metric.

    Args:
        g1 (list): The first group
        g2 (list): The second group
        method (str): The method to use (single-linkage, complete-linkage or average-linkage)

    Returns:
        (float): The evaluated distance between the groups.
    """
    
    # Compute distance between groups using specified metric and method.
    return np.mean(np.ravel(cdist(np.vstack(g1), np.vstack(g2), metric='cityblock')))


def get_groups(data, delta_condition):
    """
    Merge close nodes into clusters using agglomerative procedure.

    Args:
        data (np.ndarray): Data points represented as a numpy array
        delta_condition (float): Distance limit for considering nodes to
        be part of same cluster.

    Returns:
        (list): List of clusters represented as numpy arrays.
    """
    
    # Initialize list for storing the groups.
    groups = []
    
    # Go over all data-points.
    for data_idx in range(data.shape[0]):

        # Initialize list for storing indices of merged groups.
        to_remove = []

        # Consider the next point as a single group.
        group_nxt = [data[data_idx, :]]

        # Go over all existing groups.
        for idx, group in enumerate(groups):
            
            # Compute distance to next group.
            dist = group_distance(group_nxt, group)

            # If distance below set threshold, merge groups.
            if dist < delta_condition:
                group_nxt = group + group_nxt

                # Add index of merged group to be removed later.
                to_remove.append(idx)
        
        # Remove groups that were merged.
        for rem_idx in sorted(to_remove, reverse=True):
            del groups[rem_idx] 
        to_remove = []

        # Append next found group to list of groups.
        groups.append(group_nxt)

    # Stack data points in groups into numpy arrays.
    return list(map(np.vstack, groups))


def get_medoids(groups):
    """
    Get medoids of found groups and stack them
    into a numpy array.

    Args:
        groups (list): List of groups

    Returns:
        (numpy.ndarray): Array of found medoids.
    """

    # Initialize list for found medoids.
    medoids = []

    # Go over groups and compute medoids.
    for group in groups:
        idx_min = np.argmin(np.sum(squareform(pdist(group, metric='cityblock')), axis=0))
        medoids.append(group[idx_min, :])
    
    # Stack medoids into numpy array.
    return np.vstack(medoids)


def get_grid(n_samples=10000, min_dist=10):
    """
    Get grid of points using the sample and cluster process.

    Args:
        n_samples (int): Number of samples to use in the process
        min_dist (float): Distance limit for considering nodes to
        be part of same cluster.

    Returns:
        (numpy.ndarray): Spatial points forming the grid as well as the corresponding
        sample indices.
    """

    # Parse list of latitude and longitude values and join.
    df = pd.read_csv('./data/trip_data/sampled.csv')
    lat_1 = df['Pickup_latitude'].to_numpy()
    lon_1 = df['Pickup_longitude'].to_numpy()
    lat_2 = df['Dropoff_latitude'].to_numpy()
    lon_2 = df['Dropoff_longitude'].to_numpy()
    lat_all = np.hstack((lat_1, lat_2))
    lon_all = np.hstack((lon_1, lon_2))
    data = np.vstack((lat_all, lon_all)).T

    # Sample spatial points for grid generation using specified sample size.
    sample_indices = random.sample(range(data.shape[0]), n_samples)
    node_sample = data[sample_indices, :]

    # Join nodes in clusters and find medoids.
    clusters = get_groups(node_sample, min_dist)
    nodes_filtered = get_medoids(clusters)
    return nodes_filtered


def draw_grid(nodes, map_center=(40.7143528, -74.0059731, 13), save_path='grid_vis.html'):
    """
    Draw grid using computed nodes.

    Args:
        nodes (numpy.ndarray): Data points to plot
        map_center (tuple): Latitute, Longitude and zoom used for initial map view
        save_path (str): Path for the visualization
    """
    
    # Initialize map and plot points.
    gmap = gmplot.GoogleMapPlotter(*map_center)
    gmap.scatter(nodes[:, 0], nodes[:, 1], '#FF0000', size=35, marker=False)
    gmap.draw(save_path)


