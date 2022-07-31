import os
import json
from typing import List
from collections import defaultdict

import numpy as np


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