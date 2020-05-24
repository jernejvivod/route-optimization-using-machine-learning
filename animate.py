import numpy as np
import networkx as nx
import geoplotlib
from geoplotlib.core import BatchPainter, GeoplotlibApp
from geoplotlib.colors import colorbrewer
from geoplotlib.utils import epoch_to_str, BoundingBox, read_csv
from geoplotlib.layers import BaseLayer
from geopy.geocoders import Nominatim
import argparse

class AnimatedProcess(BaseLayer):

    def __init__(self, network, edgelists, show_addresses=False, save_frames=False):

        # Set network and edge lists.
        self.network = network
        self.edgelists = edgelists
        self.num_frames = len(self.edgelists)

        # Set flag specifying whether to save frames.
        self.save_frames = save_frames

        # Set flag specifying whether to show addresses.
        self.show_addresses = show_addresses
        
        # Initialize state counter.
        self.count = 0
        
        # Initialize geolocator for retrieving addresses.
        geolocator = Nominatim(user_agent='test')

        # Set coordinates of nodes and get addresses.
        self.node_coordinates = np.empty((2, network.number_of_nodes()), dtype=float)
        self.node_addresses = []
        for idx, node in enumerate(network.nodes()):
            self.node_coordinates[:, idx] = np.array(self.network.nodes[node]['latlon'])
            if self.show_addresses:
                address = geolocator.reverse(self.node_coordinates[:, idx]).address
                self.node_addresses.append(address[:address.index(',', address.index(',') + 1)])


    def invalidate(self, proj):
        self.x, self.y = proj.lonlat_to_screen(self.node_coordinates[1, :], self.node_coordinates[0, :])
        

    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        
        # Prepare edges for next frame.
        self.edge_src = np.empty((2, len(self.edgelists[self.count])), dtype=float)
        self.edge_dst = np.empty((2, len(self.edgelists[self.count])), dtype=float)
        for idx, edge in enumerate(self.edgelists[self.count]):
            self.edge_src[:, idx] = self.network.nodes[edge[0]]['latlon']
            self.edge_dst[:, idx] = self.network.nodes[edge[1]]['latlon']
        self.edge_src_trans_x, self.edge_src_trans_y = proj.lonlat_to_screen(self.edge_src[1, :], self.edge_src[0, :])
        self.edge_dst_trans_x, self.edge_dst_trans_y = proj.lonlat_to_screen(self.edge_dst[1, :], self.edge_dst[0, :])
        
        # Initialize painter, plot nodes and addresses.
        self.painter = BatchPainter()
        self.painter.points(self.x, self.y, point_size=10, rounded=True)
        if self.show_addresses:
            self.painter.labels(self.x, self.y, self.node_addresses, font_size=10, anchor_x='left')

        # Plot edges.
        self.painter.set_color([255, 0, 0])
        self.painter.lines(self.edge_src_trans_x, self.edge_src_trans_y, self.edge_dst_trans_x, self.edge_dst_trans_y, width=1)
        
        # Draw and increment counter.
        self.painter.batch_draw()
        if self.count < len(self.edgelists) - 1:
            self.count += 1
            self.count = self.count % self.num_frames
            print("count: {0}/{1}".format(self.count, self.num_frames))
        
        # If saving frames.
        if self.save_frames:
            GeoplotlibApp.screenshot(f'./results/animation_frames/{self.count}.png')


if __name__ == '__main__':

    ### PARSE ARGUMENTS ###
    parser = argparse.ArgumentParser(description='Animate lists of edge lists.')
    parser.add_argument('--network-path', type=str, required=True, help='Path to the network corresponding to the edge list')
    parser.add_argument('--edgelist-path', type=str, required=True, help='Path to the edge list')
    parser.add_argument('--show-addresses', action='store_true', help='Show addresses corresponding to the nodes')
    parser.add_argument('--save-frames', action='store_true', help='Save animation frames')
    args = parser.parse_args()
    #######################
    
    # Parse network.
    network = nx.read_gpickle(args.network_path)

    # Parse edgelist.
    edgelists = np.load(args.edgelist_path)
    
    # Add animation layer, set bounding box and show.
    geoplotlib.add_layer(AnimatedProcess(network, edgelists, show_addresses=args.show_addresses, save_frames=args.save_frames))
    geoplotlib.set_bbox(BoundingBox(north=40.897994, west=-73.199040, south=40.595581, east=-74.55040))
    geoplotlib.show()

