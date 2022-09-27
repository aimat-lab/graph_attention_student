from typing import Dict, Union, List, Tuple

import numpy as np

"""
This is a specific list consisting of exactly three float values in the range [0, 1]. The entire list thus
represents an RGB color value, where the corresponding elements denote the intensity of the colors
red green and blue respectively.
"""
RgbList = Union[List[float], Tuple[float, float, float]]

"""
This is an alias which specifies the main graph representation which is used throughout this application.
A graph is generally represented as a dict structure with very key value pairs.

The following keys are always expected to be present:

- ``node_indices``: A (N, ) array of integers representing the unique node indices
- ``node_attributes``: A (N, K) array of float node features
- ``node_adjacency``: A (N, N) array which is the adjacency matrix for the graph. (Binary entries)
- ``edge_indices``: A (M, 2) array, which is essentially a list of all edges (specified as two int node
  indices) without any particular order
- ``edge_attributes``: A (M, ) array of edge weights

The following keys are *optionally* present / expected for certain tasks:

- ``node_positions``: A (N, 2) array of float positions for the nodes on a 2D plane. These are used to place
  the nodes into a matplotlib coordinate system.
- ``node_coordinates``: A (N, 2) array of float positions for the nodes within a corresponding *image*. These
  values are the pixel coordinates in the final image!
- ``node_importances``: A (N, ) array of float values determining node importance values (=explanations)
- ``edge_importances``: A (M, ) array of float values determining edge importance values (=explanations). The
  values are assigned to the edges as they are ordered in edge_indices!
"""
GraphDict = Dict[str, np.ndarray]
