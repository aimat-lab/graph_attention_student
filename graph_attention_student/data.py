import os
import json
from typing import List, Tuple
from collections import defaultdict

import numpy as np
from kgcnn.utils.data import ragged_tensor_from_nested_numpy


def process_graph_dataset(dataset: List[dict],
                          test_indices: List[int]
                          ) -> Tuple[tuple, tuple, tuple, tuple]:
    indices = list(range(len(dataset)))
    train_indices = [index for index in indices if index not in test_indices]

    labels_train = [dataset[i]['graph_labels'] for i in train_indices]
    labels_test = [dataset[i]['graph_labels'] for i in test_indices]
    nodes_train = [dataset[i]['node_attributes'] for i in train_indices]
    nodes_test = [dataset[i]['node_attributes'] for i in test_indices]
    edges_train = [dataset[i]['edge_attributes'] for i in train_indices]
    edges_test = [dataset[i]['edge_attributes'] for i in test_indices]
    edge_indices_train = [dataset[i]['edge_indices'] for i in train_indices]
    edge_indices_test = [dataset[i]['edge_indices'] for i in test_indices]
    node_importances_train = [dataset[i]['node_importances'] for i in train_indices]
    node_importances_test = [dataset[i]['node_importances'] for i in test_indices]
    edge_importances_train = [dataset[i]['edge_importances'] for i in train_indices]
    edge_importances_test = [dataset[i]['edge_importances'] for i in test_indices]

    # The train scores
    xtrain = (
        ragged_tensor_from_nested_numpy(nodes_train),
        ragged_tensor_from_nested_numpy(edges_train),
        ragged_tensor_from_nested_numpy(edge_indices_train)
    )

    xtest = (
        ragged_tensor_from_nested_numpy(nodes_test),
        ragged_tensor_from_nested_numpy(edges_test),
        ragged_tensor_from_nested_numpy(edge_indices_test)
    )

    # The importance scores
    ytrain = (
        # ragged_tensor_from_nested_numpy(labels_train),
        np.array(labels_train),
        ragged_tensor_from_nested_numpy(node_importances_train),
        ragged_tensor_from_nested_numpy(edge_importances_train)
    )

    ytest = (
        # ragged_tensor_from_nested_numpy(labels_test),
        np.array(labels_test),
        ragged_tensor_from_nested_numpy(node_importances_test),
        ragged_tensor_from_nested_numpy(edge_importances_test)
    )

    return xtrain, ytrain, xtest, ytest


def load_eye_tracking_dataset(folder_path: str) -> List[dict]:
    dataset_map = defaultdict(dict)
    for root, dirs, files in os.walk(folder_path):

        for file_name in files:
            name, extension = file_name.split('.')
            file_path = os.path.join(root, file_name)

            if extension in ['png']:
                dataset_map[name]['image_path'] = file_path
                dataset_map[name]['name'] = name

            if extension in ['json']:
                dataset_map[name]['metadata_path'] = file_path
                with open(file_path, mode='r') as json_file:
                    metadata = json.load(json_file)

                # This check fixes a bug, which can happen if there are other JSON files present in the
                # folder and these most likely do not define a similar structure which would cause a
                # KeyError here if we just assumed they would
                if isinstance(metadata, dict) and 'graph' in metadata:
                    dataset_map[name]['metadata'] = metadata
                    # At this point there is already a field "graph" in metadata but the values in that
                    # dict are just lists after being loaded from the JSON file, so we need to convert
                    # them all into numpy arrays (this is actually important!) here
                    dataset_map[name]['graph'] = {key: np.array(value)
                                                  for key, value in metadata['graph'].items()}

        break

    # Now, if we just pass a folder to this function, then there is a real possibility that this folder
    # contains other json or image files as well and we dont want these in our result, which is why we
    # filter out any elements which does not contain both(!) the image and metadata file path fields!
    return [data
            for data in dataset_map.values()
            if 'metadata_path' in data and 'image_path' in data]


def load_solubility_aqsoldb():
    pass