import typing as t

import numpy as np

"""
This is a specific list consisting of exactly three float values in the range [0, 1]. The entire list thus
represents an RGB color value, where the corresponding elements denote the intensity of the colors
red green and blue respectively.
"""
RgbList = t.Union[t.List[float], t.Tuple[float, float, float]]

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
GraphDict = t.Dict[str, np.ndarray]

"""
A dictionary which contains the metadata corresponding to an element of the dataset. This metadata dict 
contains all the relevant information about the representation of a graph as well as additional information 
about the prediction target for example.

There are some keys which all the metadata dicts should have in common, but theoretically any additional 
fields can be added for special purpose annotations of different datasets.
"""
MetadataDict = t.Dict[str, t.Union[GraphDict, str, float, t.Any]]

"""
This is a special dictionary structure which represents a loaded "eye tracking dataset". An eye tracking 
dataset is a special format for a graph dataset, where the whole dataset is represented by a folder in the 
file system. Within this folder, every element is represented by two files: A JSON file which contains the 
graph representation and the metadata and a PNG file which visualizes the graph in a domain specific way.

The keys of this dict are the *names* of the elements. Those are also used as the file names for both the 
JSON and the PNG files in the folder structure. The corresponding values are again dicts which contain the 
following fields:

- ``index``: An integer for the order in which the elements were loaded from the filesystem. NOTE that this 
  index may differ between operating systems and dev environments as it depends on how the files are 
  sorted in the folder.
- ``name``: A string for the name of the element, which is determined by the file name
- ``image_path``: The absolute string path to the image file that contains the visualization of the 
  corresponding graph element
- ``metadata``: A dictionary which contains the metadata about the element as well as the graph 
  representation of the element itself. Metadata for example contains information about the target value 
  of the element and a possible dataset split as well as usually a canonical index within the dataset
"""
EtDatasetDict = t.Dict[str, t.Union[str, dict]]
