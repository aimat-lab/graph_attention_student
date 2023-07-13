import os
import sys
import csv
import time
import json
import random
import orjson
import logging
import tempfile
import typing as t
from copy import copy
from typing import List, Tuple, Optional, Dict, Callable
from collections import defaultdict

import cairosvg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import visual_graph_datasets.typing as tv
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from kgcnn.data.moleculenet import MoleculeNetDataset
from rdkit import Chem
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)

import graph_attention_student.typing as tc
from graph_attention_student.util import NULL_LOGGER
from graph_attention_student.util import graph_dict_to_list_values
from graph_attention_student.util import update_nested_dict
from graph_attention_student.visualization import draw_extended_colors_graph
from graph_attention_student.visualization import draw_edge_black
from graph_attention_student.visualization import draw_node_color_scatter


class NumericJsonEncoder(json.JSONEncoder):

    def default(self, o: t.Any) -> t.Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.generic):
            return o.item()
        else:
            return super(NumericJsonEncoder, self).default(o)


def numpy_to_native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.generic):
        return value.item()
    else:
        return value


def tensors_from_graphs(graph_list: t.List[tv.GraphDict]
                        ) -> t.Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
    """
    Given a list of GraphDicts, this function will convert that into a tuple of ragged tensors which can
    directly be used as the input to a MEGAN model.
    """
    return (
        ragged_tensor_from_nested_numpy([graph['node_attributes'] for graph in graph_list]),
        ragged_tensor_from_nested_numpy([graph['edge_attributes'] for graph in graph_list]),
        ragged_tensor_from_nested_numpy([graph['edge_indices'] for graph in graph_list]),
    )
    
def mock_importances_from_graphs(graphs: t.List[tv.GraphDict],
                                 num_channels: int,
                                 ) -> t.Tuple[tf.RaggedTensor, tf.RaggedTensor]:
    return (
        ragged_tensor_from_nested_numpy([np.zeros(shape=(len(graph['node_attributes']), num_channels)) for graph in graphs]),
        ragged_tensor_from_nested_numpy([np.zeros(shape=(len(graph['edge_attributes']), num_channels)) for graph in graphs]),
    )


def process_index_data_map(index_data_map: t.Dict[int, dict],
                           insert_empty_importances: bool = True,
                           importance_channels: int = 1,
                           ) -> t.Tuple[list, list]:
    """
    Given the ``index_data_map`` which was loaded from a visual graph dataset, this function will return a
    tuple of two values: The first is the list of integer indices of the all the dataset elements and the
    second element is the list of graph dicts representing each corresponding element.

    :param index_data_map: A dict whose keys are the integer indices of the dataset elements and the values
        the corresponding dictionaries containing all the relevant data about those elements. The format
        of this data structure should be exactly as returned by the "load_visual_graph_dataset" function.
    :param insert_empty_importances: boolean flag, if true empty (zero) numpy arrays will be inserted into
        each graph dict for the "node_importances" and "edge_importances" field.
    :param importance_channels: The integer number of importance channels to be used for the insertion of
        the zero importance arrays.

    :returns: A tuple of two lists.
    """
    dataset_length = len(index_data_map)
    dataset_indices = []
    dataset = [None for _ in range(max(index_data_map.keys()) + 1)]
    for index, data in index_data_map.items():
        g = data['metadata']['graph']

        if insert_empty_importances:
            num_nodes = len(g['node_indices'])
            g['node_importances'] = np.zeros(shape=(num_nodes, importance_channels), dtype=float)

            num_edges = len(g['edge_indices'])
            g['edge_importances'] = np.zeros(shape=(num_edges, importance_channels), dtype=float)

        dataset[index] = g
        dataset_indices.append(index)

    return dataset_indices, dataset


def process_graph_dataset(dataset: t.List[dict],
                          test_indices: t.Optional[t.List[int]] = None,
                          train_indices: t.Optional[List[int]] = None,
                          use_importances: bool = True,
                          use_graph_attributes: bool = False,
                          ) -> t.Tuple[tuple, tuple, tuple, tuple]:
    """
    Given a list ``dataset`` of GraphDict objects representing the elements of a dataset, this function will
    firstly split these elements according to the given ``train_indices`` and ``test_indices`` and also turn
    this list of graph objects into the appropriate RaggedTensors which can be used to train a
    tensorflow model.
    """
    indices = list(range(len(dataset)))

    # It has to be possible to determine the dataset by either the train or the test indices because now
    # there are cases were we need to be explicitly sure to have an exact number of train elements
    # that has to be divisible by the batch size
    if train_indices is not None and test_indices is None:
        test_indices = [index for index in indices if index not in train_indices]
    elif test_indices is not None and train_indices is None:
        train_indices = [index for index in indices if index not in test_indices]

    labels_train = [dataset[i]['graph_labels'] for i in train_indices]
    labels_test = [dataset[i]['graph_labels'] for i in test_indices]
    nodes_train = [dataset[i]['node_attributes'] for i in train_indices]
    nodes_test = [dataset[i]['node_attributes'] for i in test_indices]
    edges_train = [dataset[i]['edge_attributes'] for i in train_indices]
    edges_test = [dataset[i]['edge_attributes'] for i in test_indices]
    edge_indices_train = [dataset[i]['edge_indices'] for i in train_indices]
    edge_indices_test = [dataset[i]['edge_indices'] for i in test_indices]

    x_train = (
        ragged_tensor_from_nested_numpy(nodes_train),
        ragged_tensor_from_nested_numpy(edges_train),
        ragged_tensor_from_nested_numpy(edge_indices_train)
    )

    x_test = (
        ragged_tensor_from_nested_numpy(nodes_test),
        ragged_tensor_from_nested_numpy(edges_test),
        ragged_tensor_from_nested_numpy(edge_indices_test)
    )

    # Optionally it is also possible to use graph attributes, these are features which describe the graph
    # as a whole rather than its individual elements (nodes / edges). This is however not part of every
    # dataset by default because not all graph neural networks are able to deal with this appropriately.
    # So we also only add this as the fourth element of the X tuple, if the corresponding flag is given
    if use_graph_attributes:
        graph_attributes_train = [dataset[i]['graph_attributes'] for i in train_indices]
        graph_attributes_test = [dataset[i]['graph_attributes'] for i in test_indices]

        x_train = (
            *x_train,
            tf.convert_to_tensor(graph_attributes_train)
        )
        x_test = (
            *x_test,
            tf.convert_to_tensor(graph_attributes_test),
        )

    y_train = (
          np.array(labels_train, dtype=np.single),
    )

    y_test = (
        np.array(labels_test, dtype=np.single),
    )

    # Optionally (although in this project this is the true most of the time) it is possible to load
    # node and edge importances (explanations) as additional training targets from the datasets as well.
    if use_importances:
        node_importances_train = [dataset[i]['node_importances'] for i in train_indices]
        node_importances_test = [dataset[i]['node_importances'] for i in test_indices]
        edge_importances_train = [dataset[i]['edge_importances'] for i in train_indices]
        edge_importances_test = [dataset[i]['edge_importances'] for i in test_indices]

        y_train = (
            *y_train,
            ragged_tensor_from_nested_numpy(node_importances_train),
            ragged_tensor_from_nested_numpy(edge_importances_train)
        )

        y_test = (
            *y_test,
            ragged_tensor_from_nested_numpy(node_importances_test),
            ragged_tensor_from_nested_numpy(edge_importances_test)
        )

    return x_train, y_train, x_test, y_test


# == DEPRECATED ==

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
                    metadata = json.loads(json_file.read())

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


def load_eye_tracking_dataset_dict(dataset_path: str,
                                   ratio: float = 1.0,
                                   logger: logging.Logger = NULL_LOGGER,
                                   log_step: int = 1000,
                                   sanitize_dict: bool = False) -> tc.EtDatasetDict:
    """

    *25.10.2022*
    Recently I was confronted with a new dataset that has about half a million entries, and this required
    a serious restructuring of this method due to higher requirements of computational and storage
    efficiency.
    Previously the function worked by iterating the os.listdir() list directly and then checking each
    file's extension and only the correct extensions were added to the dict. This was a more robust
    solution but it required a longer primary loop, which is why I switched to the solution with the
    set of file names now.

    :param dataset_path:
    :param logger:
    :param log_step:
    :param sanitize_dict:
    :return:
    """
    logger.info(f'determining the file names...')
    files = os.listdir(dataset_path)
    names = set()
    for file_name in files:
        name, extension = file_name.split('.')
        if extension in ['png', 'json']:
            png_path = os.path.join(dataset_path, f'{name}.png')
            json_path = os.path.join(dataset_path, f'{name}.json')
            if os.path.exists(png_path) and os.path.exists(json_path):
                names.add(name)

    if ratio < 1.0:
        names = random.sample(names, k=int(len(names) * ratio))

    num_names = len(names)
    num_files = len(files)
    logger.info(f'loading eye tracking dataset from folder {dataset_path} with {num_files} files...')

    start_time = time.time()
    dataset_map = {}
    for c, name in enumerate(names):

        # The dataset canonically represents each element either as png (graph visualization) or
        # json (metadata) if any other files contaminate the folder we do a skip here because afterwards
        # is some code we want to execute for either json or png
        dataset_map[name] = {}
        dataset_map[name]['name'] = name

        # ~ image
        image_path = os.path.join(dataset_path, f'{name}.png')
        dataset_map[name]['image_path'] = image_path

        # ~ metadata
        metadata_path = os.path.join(dataset_path, f'{name}.json')
        dataset_map[name]['metadata_path'] = metadata_path
        dataset_map[name]['index'] = c
        with open(metadata_path, mode='rb') as file:
            content = file.read()
            metadata = orjson.loads(content)
            dataset_map[name]['metadata'] = metadata

        if isinstance(metadata, dict) and 'graph' in metadata:
            dataset_map[name]['metadata'] = metadata
            # At this point there is already a field "graph" in metadata but the values in that
            # dict are just lists after being loaded from the JSON file, so we need to convert
            # them all into numpy arrays (this is actually important!) here
            dataset_map[name]['metadata']['graph'] = {key: np.array(value)
                                                      for key, value in metadata['graph'].items()}

        if c % log_step == 0:
            elapsed_time = time.time() - start_time
            time_per_element = elapsed_time / (c + 1)
            remaining_time = time_per_element * (num_names - c)
            logger.info(f' * loaded ({c}/{num_names})'
                        f' - name: {name}'
                        f' - elapsed time: {elapsed_time:.1f}s ({elapsed_time/3600:.1f}hrs)'
                        f' - remaining time: {remaining_time:.1f}s ({remaining_time/3600:.1f}hrs)'
                        f' - dict overhead: {sys.getsizeof(dataset_map)/1024**2:.2f}MB')

    # It can happen that a dataset is damaged. Maybe it wasn't properly created or maybe there was a problem
    # when copy-pasting or something. It is possible that there is only one of the two required files for
    # an element. In this case there is this optional post-processing which removes such faulty entries
    # from the dict.
    if sanitize_dict:
        logger.info('sanitizing incomplete entries in dataset dict...')
        keys = list(dataset_map.keys())
        for key in keys:
            if 'metadata' not in dataset_map[key] or 'image_path' not in dataset_map[key]:
                del dataset_map[key]

    return dataset_map


# == CHEMISTRY RELATED ======================================================================================

def create_molecule_eye_tracking_dataset(molecule_infos: Dict[str, dict],
                                         dest_path: str,
                                         image_width: int = 900,
                                         image_height: int = 900,
                                         set_attributes_kwargs: dict = {},
                                         logger: logging.Logger = NULL_LOGGER,
                                         log_step: int = 10):
    # First of all, we need to create a temporary folder in which we are going to complete all the
    # intermediate file operations.
    logger.info(f'starting creation of molecule eye tracking dataset with {len(molecule_infos)} elements...')
    with tempfile.TemporaryDirectory() as temp_path:

        logger.info('Writing temporary csv file...')
        # First step is to create a temporary csv file containing the given information about the molecules
        # smiles
        csv_path = os.path.join(temp_path, 'smiles.csv')
        with open(csv_path, mode='w') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=['id', 'smiles'], delimiter=',', quotechar='"')
            csv_writer.writeheader()
            for mol_id, mol_info in molecule_infos.items():
                csv_writer.writerow({'id': mol_id, 'smiles': mol_info['smiles']})
            logger.info(f'csv file written to: {csv_path}')

        logger.info('parsing csv file with kgcnn...')
        # With this CSV file we can now instantiate a MoleculeNetDataset instance and process all
        # the molecules
        moleculenet = MoleculeNetDataset(
            file_name=os.path.basename(csv_path),
            data_directory=os.path.dirname(csv_path),
            dataset_name='temp',
            verbose=0
        )
        moleculenet.prepare_data(
            overwrite=True,
            smiles_column_name='smiles',
            add_hydrogen=True,
            make_conformers=False,
            optimize_conformer=False
        )
        moleculenet.read_in_memory(
            label_column_name='smiles',
            add_hydrogen=False,
            has_conformers=False
        )

        # Then we setup the additional callbacks for the property assignment. These functions can be passed
        # to the dataset instance and they are invoked for every molecule object instance in the dataset.
        # That will be the place at which we create the molecule images using RDKit

        dataset_path = dest_path

        def node_coordinates_cb(mg, ds):
            mol = Chem.MolFromSmiles(str(ds['smiles']))
            #mol = mg.mol

            mol_drawer = MolDraw2DSVG(image_width, image_height)
            mol_drawer.SetLineWidth(3)
            mol_drawer.DrawMolecule(mol)
            mol_drawer.FinishDrawing()
            svg_string = mol_drawer.GetDrawingText()
            png_path = os.path.join(dataset_path, str(ds['id']) + '.png')
            cairosvg.svg2png(
                bytestring=svg_string.encode(),
                write_to=png_path,
                output_height=image_height,
                output_width=image_width,
            )

            node_coordinates = []
            for point in [mol_drawer.GetDrawCoords(i) for i, _ in enumerate(mol.GetAtoms())]:
                node_coordinates.append([
                    point.x,
                    image_height - point.y
                ])

            return np.array(node_coordinates)

        def node_indices_cb(mg, ds):
            return np.array(list(range(len(mg.mol.GetAtoms()))))

        kwargs = set_attributes_kwargs
        if 'additional_callbacks' not in kwargs:
            kwargs['additional_callbacks'] = {}

        logger.info('creating custom properties in moleculenet dataset...')
        kwargs['additional_callbacks']['node_coordinates'] = node_coordinates_cb
        kwargs['additional_callbacks']['node_indices'] = node_indices_cb
        kwargs['additional_callbacks']['id'] = lambda mg, ds: str(ds['id'])
        moleculenet.set_attributes(**kwargs)

        # 12.10.2022: Added this section in response to a bug which would cause the program to break if
        # there were some corrupted elements in the original dataset.
        logger.info(f'cleaning moleculenet dataset with {len(moleculenet)} elements...')
        moleculenet.clean(['id', 'node_attributes', 'edge_indices'])
        logger.info(f'dataset length after cleaning: {len(moleculenet)}')

        dataset_length = len(moleculenet)
        logger.info('start writing output files...')
        # Now that the dataset is prepared and even the images are created, the only thing left to do is
        # to create the metadata json file for each element
        for c, g in enumerate(moleculenet):
            mol_id = str(g['id'])
            mol_info = molecule_infos[mol_id]
            json_path = os.path.join(dataset_path, mol_id + '.json')

            # Creating the node_adjacency matrix which is an important part of the GrapDict representation
            length = len(g['node_indices'])
            g['node_adjacency'] = np.zeros(shape=(length, length))
            for i, j in g['edge_indices']:
                g['node_adjacency'][i, j] = 1

            # Adding empty importance tensors here because it will be required that these keys generally
            # exist in further processing steps.
            g['node_importances'] = np.zeros(shape=(g['node_indices'].shape[0], 1))
            g['edge_importances'] = np.zeros(shape=(g['edge_indices'].shape[0], 1))

            metadata = {
                **mol_info,
                'image_width': image_height,
                'image_height': image_height,
                'graph': {k: numpy_to_native(v) for k, v in g.items()}
            }

            with open(json_path, mode='wb') as json_file:
                content = orjson.dumps(metadata)
                json_file.write(content)

            if c % log_step == 0:
                logger.info(f'* ({c}/{dataset_length}) {mol_id}')


def eye_tracking_dataset_from_moleculenet_dataset(moleculenet: MoleculeNetDataset,
                                                  dest_path: str,
                                                  set_attributes_kwargs: t.Dict[str, t.Any],
                                                  smiles_column_name: str = 'smiles',
                                                  clean_keys: t.List[str] = ['edge_attributes'],
                                                  image_width: int = 900,
                                                  image_height: int = 900,
                                                  return_dataset: bool = True,
                                                  logger: logging.Logger = NULL_LOGGER,
                                                  log_step: int = 1000
                                                  ) -> tc.EtDatasetDict:
    dataset_length = len(moleculenet)

    logger.info(f'setting moleculenet attributes...')
    kwargs = {'additional_callbacks': {
        'smiles': lambda mg, ds: str(ds[smiles_column_name])
    }}
    # The way that this function works is that the default options for 'additional_callbacks' are the ones
    # defined above here, but IF the arg "set_attributes_kwargs" also contains an "additional_callbacks"
    # dict, then it is even possible to override the default options we define here, which is probably
    # especially useful for defining custom indexing and naming schemes for the elements.
    kwargs = update_nested_dict(kwargs, set_attributes_kwargs)
    moleculenet.set_attributes(**kwargs)

    logger.info(f'cleaning moleculenet with keys: {clean_keys}')
    moleculenet.clean(clean_keys)

    logger.info(f'generating dataset files...')
    start_time = time.time()
    dataset_map = {}
    for c, d in enumerate(moleculenet):
        index = d['index'] if 'index' in d else c
        name = d['name'] if 'name' in d else str(c)
        smiles = d['smiles']

        # ~ Render the image
        # Technically we are being a bit redundant here:
        mol = Chem.MolFromSmiles(str(smiles))
        # mol = mg.mol

        mol_drawer = MolDraw2DSVG(image_width, image_height)
        mol_drawer.SetLineWidth(3)
        mol_drawer.DrawMolecule(mol)
        mol_drawer.FinishDrawing()
        svg_string = mol_drawer.GetDrawingText()
        image_path = os.path.join(dest_path, name + '.png')
        cairosvg.svg2png(
            bytestring=svg_string.encode(),
            write_to=image_path,
            output_height=image_height,
            output_width=image_width,
        )

        node_coordinates = []
        for point in [mol_drawer.GetDrawCoords(i) for i, _ in enumerate(mol.GetAtoms())]:
            node_coordinates.append([
                point.x,
                image_height - point.y
            ])

        node_coordinates = np.array(node_coordinates)
        d['node_coordinates'] = np.array(node_coordinates)

        # "numpy_to_native" will convert any kind of numpy array to a native datatype such as a list of
        # floats for example. We need to do this to be able to turn it into JSON later.
        graph: tc.GraphDict = {key: numpy_to_native(value)
                               for key, value in d.items()
                               if isinstance(value, np.ndarray) or isinstance(value, np.generic)}

        metadata = {
            'name': name,
            'index': index,
            'image_width': image_width,
            'image_height': image_height,
            'graph': graph
        }
        if 'target' in d:
            metadata['target'] = d['target']

        metadata_path = os.path.join(dest_path, f'{name}.json')
        with open(metadata_path, mode='w') as file:
            content = json.dumps(metadata, cls=NumericJsonEncoder)
            file.write(content)

        if return_dataset:
            dataset_map[name] = {
                'index': index,
                'image_path': image_path,
                'metadata_path': metadata_path,
                'metadata': metadata,
            }

        if c % log_step == 0:
            elapsed_time = time.time() - start_time
            time_per_element = elapsed_time / (c+1)
            remaining_time = time_per_element * (dataset_length - c+1)
            logger.info(f' * created ({c}/{dataset_length})'
                        f' - index: {index} - name: {name}'
                        f' - elapsed time: {elapsed_time:.1f}s ({elapsed_time/3600:.1f}hrs)'
                        f' - remaining time: {remaining_time/3600:.1f}hrs')

    return dataset_map


# == TEXT GRAPHS ============================================================================================

def load_text_graph_dataset(folder_path: str,
                            logger: Optional[logging.Logger] = None,
                            log_interval: int = 25) -> Dict[str, dict]:
    start_time = time.time()
    dataset_map = defaultdict(dict)
    for root, dirs, files in os.walk(folder_path):

        length = len(files)
        for index, file_name in enumerate(files):
            name, extension = file_name.split('.')
            file_path = os.path.join(root, file_name)

            if extension in ['txt']:
                dataset_map[name]['name'] = name
                dataset_map[name]['text_path'] = file_path

            if extension in ['json']:
                dataset_map[name]['metadata_path'] = file_path
                with open(file_path, mode='r') as json_file:
                    metadata = json.loads(json_file.read())

                # This check fixes a bug, which can happen if there are other JSON files present in the
                # folder and these most likely do not define a similar structure which would cause a
                # KeyError here if we just assumed they would
                if isinstance(metadata, dict) and 'graph' in metadata:
                    dataset_map[name]['metadata'] = metadata

                    # We need to convert the graph which is part of the metadata dict such that it's values
                    # are no longer just lists, but numpy arrays instead
                    graph = {k: np.array(v)
                             for k, v in dataset_map[name]['metadata']['graph'].items()
                             if k not in ['node_adjacency']}
                    del dataset_map[name]['metadata']['graph']
                    dataset_map[name]['metadata']['graph'] = graph

            if logger and index % log_interval == 0:
                logger.info(f'({index}/{length}) loaded element {name:<10} - '
                            f'elapsed time: {time.time() - start_time:.2f} seconds')

        break

    # Now, if we just pass a folder to this function, then there is a real possibility that this folder
    # contains other json or image files as well and we dont want these in our result, which is why we
    # filter out any elements which does not contain both(!) the image and metadata file path fields!
    for key in dataset_map.keys():
        data = dataset_map[key]
        if not ('metadata_path' in data and 'text_path' in data):
            del dataset_map[key]

    return dataset_map


# == DATASET GENERATION =====================================================================================

# Secondary graph can be in a subgraph relation towards the primary graph, such that the primary graph is
# allowed to have additional local connections and the results will still be True. That does NOT go the
# other way around though!
def graphs_locally_similar(primary_graph: dict,
                           primary_index: int,
                           secondary_graph: dict,
                           secondary_index: int,
                           consider_edges: bool = False,
                           similarity_func: Callable = lambda a1, a2: np.all(a1 == a2),
                           primary_visited_indices: List[int] = [],
                           secondary_visited_indices: List[int] = []
                           ) -> Tuple[np.ndarray, np.ndarray]:

    if not similarity_func(primary_graph['node_attributes'][primary_index],
                           secondary_graph['node_attributes'][secondary_index]):
        raise StopIteration('The given initial nodes are not similar!')

    primary_visited_edges = [1
                             for i in primary_graph['node_indices']
                             if (primary_graph['node_adjacency'][primary_index, i] or
                                 primary_graph['node_adjacency'][i, primary_index])
                             and i in primary_visited_indices]

    secondary_visited_edges = [1
                               for j in secondary_graph['node_indices']
                               if (secondary_graph['node_adjacency'][secondary_index, j] or
                                   secondary_graph['node_adjacency'][j, secondary_index])
                               and j in secondary_visited_indices]
    if len(primary_visited_edges) != len(secondary_visited_edges):
        raise StopIteration('Not the same number of edges into the set of already visited nodes')

    #print(f'ENTER === primary: {primary_index} --- secondary: {secondary_index}')
    primary_node_count = primary_graph['node_indices'].shape[0]
    primary_node_mask = np.zeros(shape=(primary_node_count, ), dtype=bool)
    primary_adjacency_mask = np.zeros(shape=(primary_node_count, primary_node_count), dtype=bool)

    for j in secondary_graph['node_indices']:

        secondary_outgoing = secondary_graph['node_adjacency'][secondary_index, j]
        secondary_ingoing = secondary_graph['node_adjacency'][j, secondary_index]
        #print(f'j: {j}   primary ignore {primary_ignore_indices} secondary ignore: {secondary_ignore_indices} ({secondary_ingoing}, {secondary_outgoing})')
        if any([j in secondary_visited_indices,
                not secondary_ingoing and not secondary_outgoing]):
            continue

        secondary_node_attributes = secondary_graph['node_attributes'][j]

        # By default we will assume that such a node does not exist in the local neighborhood of the primary
        # graph and have to be proven wrong first by inspecting each one of them
        exists = False
        for i in primary_graph['node_indices']:

            primary_outgoing = bool(primary_graph['node_adjacency'][primary_index, i])
            primary_ingoing = bool(primary_graph['node_adjacency'][i, primary_index])
            if any([i == primary_index,
                    i in primary_visited_indices,
                    not (primary_ingoing and secondary_ingoing),
                    not (primary_outgoing and secondary_outgoing)]):
                continue

            primary_node_attributes = primary_graph['node_attributes'][i]

            nodes_similar = similarity_func(primary_node_attributes, secondary_node_attributes)
            if nodes_similar:
                #print(f'j: {j} - i: {i} (similar: {nodes_similar})  primary ignore: {primary_ignore_indices}')
                try:
                    _node_mask, _adjacency_mask = graphs_locally_similar(
                        primary_graph=primary_graph,
                        primary_index=i,
                        secondary_graph=secondary_graph,
                        secondary_index=j,
                        similarity_func=similarity_func,
                        primary_visited_indices=primary_visited_indices + [primary_index],
                        secondary_visited_indices=secondary_visited_indices + [secondary_index]
                    )
                    exists = True

                    primary_node_mask = np.logical_or(primary_node_mask, _node_mask)
                    primary_node_mask[primary_index] = True
                    primary_node_mask[i] = True

                    primary_adjacency_mask = np.logical_or(primary_adjacency_mask, _adjacency_mask)
                    primary_adjacency_mask[primary_index][i] = primary_outgoing
                    primary_adjacency_mask[i][primary_index] = primary_ingoing
                    break

                except StopIteration as e:
                    pass

        # If we have not found a similar node at the end of the iteration, then there is none within the
        # primary graph and we can declare the two graphs dissimilar altogether
        if not exists:
            raise StopIteration(f'Graphs not similar! No similarity for secondary node #{j} found within '
                                f'local neighborhood of primary graph node #{primary_index}.')

    #print(f'EXIT  === primary: {primary_index} --- secondary: {secondary_index}')
    return primary_node_mask, primary_adjacency_mask


def contains_subgraph(subgraph: dict,
                      graph: dict,
                      similarity_func: Callable = lambda a1, a2: np.all(a1 == a2),
                      subgraph_index: int = 0
                      ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    # First, we take the first node of the subgraph and iterate through the entire main graph to see
    # if there even are nodes which match those attributes
    possible_seed_indices = []
    for node_index in graph['node_indices']:
        similar = similarity_func(graph['node_attributes'][node_index],
                                  subgraph['node_attributes'][subgraph_index])
        if similar:
            possible_seed_indices.append(node_index)

    # The list "possible_seed_indices" now contains all the node indices of the main graph, which are
    # possibly also part of the given subgraph structure. Now, for every one of those nodes we explore the
    # local neighborhood of the main graph and the subgraph in parallel to see if they are indeed similar.
    similar = False
    node_mask = None
    adjacency_mask = None
    for node_index in possible_seed_indices:
        try:
            node_mask, adjacency_mask = graphs_locally_similar(
                primary_graph=graph,
                primary_index=node_index,
                secondary_graph=subgraph,
                secondary_index=subgraph_index,
                similarity_func=similarity_func
            )
            similar = True
        except StopIteration:
            pass

    return similar, node_mask, adjacency_mask


def create_colors_eye_tracking_dataset(graph_infos: Dict[str, dict],
                                       dest_path: str,
                                       image_width: int = 900,
                                       image_height: int = 900,
                                       layout_cb: Callable[[nx.Graph], List[int]] =
                                            nx.layout.kamada_kawai_layout,
                                       draw_node_cb: Callable[[plt.Axes, tc.GraphDict, int], None] =
                                            draw_node_color_scatter,
                                       draw_edge_cb: Callable[[plt.Axes, tc.GraphDict, int, int], None] =
                                            draw_edge_black,
                                       dpi: int = 100,
                                       image_extension: str = 'png'
                                       ):
    # First of all we need to make sure that the given dest_path is actually a directory as it should be
    assert os.path.exists(dest_path), (f'The destination path for the eye tracking dataset {dest_path} does '
                                       f'not exist! Make sure to create the folder!')
    assert os.path.isdir(dest_path), (f'The destination path for the eye tracking dataset "{dest_path}" '
                                      f'is not a directory, but it has to be!')

    # For every element in the given dict we create one eye tracking dataset element, which consists of two
    # files: An image which is a visual representation of the graph itself which will be shown to the user
    # and a metadata json file for each image.
    for key, info in graph_infos.items():
        g = info['graph']
        name = str(key)

        # First of all we will generate the eye tracking image file which provides a visual representation
        # of the graph
        image_name = f'{name}.{image_extension}'
        image_path = os.path.join(dest_path, image_name)
        #print(image_path)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(image_width / dpi, image_height / dpi))
        draw_extended_colors_graph(
            ax=ax,
            g=g,
            layout_cb=layout_cb,
            draw_node_cb=draw_node_cb,
            draw_edge_cb=draw_edge_cb,
        )
        ax.axis('off')
        fig.savefig(fname=image_path, dpi=dpi)
        g['node_coordinates'] = np.array([ax.transData.transform(pos) for pos in g['node_positions']])
        plt.close(fig)

        # Then we create the metadata json file
        metadata_name = f'{name}.json'
        metadata_path = os.path.join(dest_path, metadata_name)
        metadata = {
            **info,
            'image_width': image_width,
            'image_height': image_height,
            'image_path': f'./{image_name}'
        }
        # At this point the 'graph' which is contained in the passed information is still in the normal
        # representation where all the values are numpy arrays. This will cause problems for the json
        # serialization, which is why we use this helper method which converts all the values into nested
        # list representations of those arrays.
        metadata['graph'] = graph_dict_to_list_values(metadata['graph'])
        with open(metadata_path, mode='wb') as json_file:
            content = json.dumps(metadata)
            json_file.write(content)




def generate_colors_graph(node_count: int,
                          additional_edge_count: int,
                          colors: List[Tuple[float, float, float]],
                          color_weights: Optional[List[float]] = None,) -> Dict[str, np.ndarray]:
    # If no explicit weights for the random color choice are provided, we assume that all colors should
    # occur with equal probability.
    if color_weights is None:
        color_weights = [1] * len(colors)

    node_indices = list(range(node_count))
    node_attributes = [] + random.choices(colors, weights=color_weights, k=1)
    edge_indices = []

    node_indices_inserted = [0]
    node_indices_remaining = list(range(1, node_count))
    random.shuffle(node_indices_remaining)

    # ~ Generating basic graph structure
    while len(node_indices_remaining) != 0:
        node_1 = random.choice(node_indices_inserted)
        node_2 = node_indices_remaining.pop()

        # Generating the color attributes for the newly inserted node
        node_attributes += random.choices(colors, weights=color_weights, k=1)

        # Adding an undirected edge between the two nodes to the edge list (= two directed edges)
        edge_indices += [(node_1, node_2), (node_2, node_1)]

        # In the end we also have to actually add the node to the list of inserted nodes
        node_indices_inserted.append(node_2)

    # ~ Adding the additional edges
    inserted_edge_count = 0
    while inserted_edge_count != additional_edge_count:
        node_1, node_2 = random.sample(node_indices, 2)

        # If no edge between these two nodes already exists, we add one
        if (node_1, node_2) not in edge_indices and (node_2, node_1) not in edge_indices:
            edge_indices += [(node_1, node_2), (node_2, node_1)]
            inserted_edge_count += 1

    # ~ Creating edge weights and adjancency matrix
    edge_attributes = np.ones(shape=(len(edge_indices), 1))
    node_adjacency = [[1 if (i, j) in edge_indices else 0 for i in node_indices] for j in node_indices]

    # ~ Converting to numpy arrays
    return {
        'node_indices':             np.array(node_indices, dtype=np.int32),
        'node_attributes':          np.array(node_attributes, dtype=np.float32),
        'node_adjacency':           np.array(node_adjacency, dtype=np.int32),
        'edge_indices':             np.array(edge_indices, dtype=np.int32),
        'edge_attributes':          np.array(edge_attributes, dtype=np.float32)
    }


class ExtendedColorsGraphGenerator:
    """
    This class can be used to randomly generate graphs of the so called "EXTENDED COLORS" type.

    Extended Colors
    ---------------
    This type of graphs is mainly identified by the fact that each node of the graph is assigned a color as
    it's main attribute. The first three values of the node attribute vectors will always define the [0, 1]
    RGB values which define the color of that node. It is additionally called "extended" because there is
    also the possibility to generate additional attributes for the graphs besides those colors. This could
    for example be an additional binary attribute which determines if the node is a circle or a square.
    All in all, this kind of dataset is focused on rich & intuitive visual representations of graphs, as it's
    purpose is to be presented to human participants in a visual classification task.

    Generation Procedure
    --------------------
    The generation procedure is a seeding process. The most simple version works like this: A fixed number
    of nodes has to be provided. All of them are initialized as "not inserted", a random node is determined
    as seed node and inserted. From that point onwards a random node of the "not inserted" is chosen and
    inserted as neighbor to one randomly chosen already "inserted" node. This is repeated until all nodes
    are inserted and it results in a "tree" graph. To make cycles possible an additional given number of
    additional edges is inserted between random pairs of nodes afterwards as well. Node properties like the
    color and edge properties as well are chosen according to the callback functions which are passed to the
    generation procedure for that purpose.

    There is also the possibility to use entire graphs as the seeds for the generation process. Multiple
    graphs can be passed to the constructor in the format of a list of GraphDict's. If multiple seeding
    subgraph structures are provided, multiple non connected trees will be grown from those at first and
    then in the end connected by the additional edges.

    Example
    -------

    .. code-block:: python

        import random
        from gnn_student_teacher.data import ExtendedColorsGraphGenerator

        generator = ExtendedColorsGraphGenerator(
            node_count=15,
            additional_edge_count=2,
            color_attributes_cb=lambda: random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            additional_attributes_cb=lambda: [],
            edge_attributes_cb=lambda a, b: [1]
        )

    """
    def __init__(self,
                 node_count: int,
                 additional_edge_count: int,
                 color_attributes_cb: Callable[[tc.GraphDict, int], list],
                 additional_attributes_cb: Callable[[tc.GraphDict, int], List[float]],
                 edge_attributes_cb: Callable[[tc.GraphDict, int, int], List[float]],
                 edge_valid_cb: Callable[[tc.GraphDict, int, int], bool] = lambda g, i, j: True,
                 seed_graphs: List[tc.GraphDict] = [],
                 is_directed: bool = False,
                 prevent_edges_in_seed_graphs: bool = True):
        assert additional_edge_count >= len(seed_graphs) - 1, \
            (f'The "additional_edge_count" ({additional_edge_count}) has to be at least as high as the '
             f'number of provided seed graphs - 1 ({len(seed_graphs) - 1}) to ensure connectivity!')

        self.node_count = node_count
        self.additional_edge_count = additional_edge_count
        self.color_attributes_cb = color_attributes_cb
        self.additional_attributes_cb = additional_attributes_cb
        self.edge_attributes_cb = edge_attributes_cb
        self.edge_valid_cb = edge_valid_cb
        self.seed_graphs = seed_graphs
        self.is_directed = is_directed
        self.prevent_edges_in_seed_graphs = prevent_edges_in_seed_graphs

        # ~ Computed properties
        # These properties about the graph can be derived from the information we are given

        # This is the number of edges which are contributed by the seed graphs. If there are no seed
        # graphs then we set that to -1 because then in that case the graph is grown from a single seed
        # node, which in itself does not contribute an edge to the graph which is actually removing one
        # edge under the assumption we make that in the graph growing process every node contributes one
        # edge to the node
        self.seed_node_count = sum(len(g['node_indices']) for g in seed_graphs)
        if seed_graphs:
            self.seed_edge_count = sum(len(g['edge_indices']) for g in seed_graphs)
        else:
            if self.is_directed:
                self.seed_edge_count = -1
            else:
                self.seed_edge_count = -2
        # undirected graphs will be assumed to just be directed graphs where nodes are connected by two
        # edges pointing both ways.
        if self.is_directed:
            self.edge_count = self.seed_edge_count + \
                              (self.node_count - self.seed_node_count) + self.additional_edge_count
        else:
            self.edge_count = self.seed_edge_count + \
                              ((self.node_count - self.seed_node_count) + self.additional_edge_count) * 2

        # ~ Graph properties
        # these following values are the array-like data structures which have to be constructed and which
        # directly represent the graph that is to be generated
        self.node_indices: List[int] = list(range(node_count))
        self.node_attributes: List[List[float]] = []
        self.node_adjacency: List[List[int]] = []
        self.edge_indices: List[List[int]] = []
        self.edge_attributes: List[List[float]] = []

        self.graph: Dict[str, list] = {}

        # ~ Generation properties
        # These properties are helpers which are needed for the construction process

        self.node_indices_inserted: List[int] = []
        self.node_indices_remaining: List[int] = []
        # The keys of this dict will be the indices of the seed graphs in the given list, the values will
        # be mappings which map the index internal to the seed graph to an index of this graph.
        self.seed_graph_index_maps: Dict[int, Dict[int, int]] = {}
        # This is a list where the index is the node index and the value is the index of the seed graph
        # from which that node was grown.
        self.seed_graph_association: List[int] = []
        self.seed_graph_indices: List[Optional[int]] = []
        # To keep track of the index where to insert the next edge element into edge_indices and
        # edge_attributes
        self.current_edge_index = 0

    def reset(self):
        # node_indices never get modified thats why they dont need to get reset here
        # We initialize all the property lists here with essentially empty items because during the
        # construction process we want to use indiced assignment at certain positions and that is not
        # possible with empty lists
        self.node_attributes = [[] for _ in range(self.node_count)]
        self.node_adjacency = [[0 for _ in range(self.node_count)] for _ in self.node_indices]
        self.edge_indices = [[] for _ in range(self.edge_count)]
        self.edge_attributes = [[] for _ in range(self.edge_count)]

        # Resetting the computation state
        self.node_indices_remaining = copy(self.node_indices)
        # We shuffle the node index list here already so that we get random node choices by simply popping
        # the first element in the other steps of the construction process.
        random.shuffle(self.node_indices_remaining)
        self.node_indices_inserted = []
        self.seed_graph_association = [0 for _ in range(self.node_count)]
        self.seed_graph_indices = [-1 for _ in range(self.node_count)]
        self.current_edge_index = 0

        self.graph = {
            'node_indices': self.node_indices,
            'node_attributes': self.node_attributes,
            'edge_indices': self.edge_indices,
            'edge_attributes': self.edge_attributes,
            'seed_graph_indices': self.seed_graph_indices
        }

    def generate(self) -> tc.GraphDict:
        self.reset()

        # ~ Seeding
        # The whole generation procedure is a seeding method. The graph starts out with a seed center and
        # the rest of the graph is then "grown" around that. It is possible to define subgraphs which act
        # as seeds. If that is not the case, then one node is chosen at random to be the seed
        if len(self.seed_graphs) == 0:
            self.insert_seed_node()
        else:
            self.insert_seed_graphs()

        # ~ Growing
        self.grow_nodes()
        self.add_edges()

        self.node_adjacency = self.node_adjacency_from_edge_indices(self.node_count, self.edge_indices)

        return {
            'node_indices':         np.array(self.node_indices, dtype=np.int32),
            'node_attributes':      np.array(self.node_attributes, dtype=np.float32),
            'node_adjacency':       np.array(self.node_adjacency, dtype=np.int32),
            'edge_indices':         np.array(self.edge_indices, dtype=np.int32),
            'edge_attributes':      np.array(self.edge_attributes, dtype=np.float32),
            'seed_graph_indices':   np.array(self.seed_graph_indices, dtype=np.int32),
        }

    def insert_seed_graphs(self):
        for graph_index, g in enumerate(self.seed_graphs):
            seed_size = len(g['node_indices'])
            seed_indices = random.sample(self.node_indices_remaining, k=seed_size)
            # This is a mapping whose keys are the node indices local to the seed graph and the values are
            # the corresponding values are the node indices chosen for this graph to be generated
            seed_index_map = dict(zip(g['node_indices'], seed_indices))
            self.seed_graph_index_maps[graph_index] = seed_index_map

            for seed_index, index in seed_index_map.items():
                self.node_attributes[index] = g['node_attributes'][seed_index]
                self.seed_graph_association[index] = graph_index
                self.seed_graph_indices[index] = graph_index

                # Technically the node indices are now inserted, now we need to establish the corresponding
                # edge connections according to the seed graph.
                for (i, j), edge_attributes in zip(g['edge_indices'], g['edge_attributes']):
                    if seed_index == i:
                        edge = [seed_index_map[i], seed_index_map[j]]
                        self.edge_indices[self.current_edge_index] = edge
                        self.edge_attributes[self.current_edge_index] = edge_attributes
                        self.current_edge_index += 1

                # We now have to remove this node from the list of remaining nodes and also have to add it
                # to the list of inserted nodes
                self.node_indices_remaining.remove(index)
                self.node_indices_inserted.append(index)

    def create_single_node_attributes(self, index: int) -> List[float]:
        color_attributes = self.color_attributes_cb(self.graph, index)
        additional_attributes = self.additional_attributes_cb(self.graph, index)
        return color_attributes + additional_attributes

    def create_single_edge_attributes(self, i: int, j: int) -> List[float]:
        return self.edge_attributes_cb(self.graph, i, j)

    def insert_seed_node(self):
        index = self.node_indices_remaining.pop(0)
        self.node_attributes[index] = self.create_single_node_attributes(0)
        self.node_indices_inserted.append(index)

    def insert_edge(self, i: int, j: int) -> None:
        if self.is_directed:
            edges = [[i, j]]
        else:
            edges = [[i, j], [j, i]]

        for k, l in edges:
            self.edge_indices[self.current_edge_index] = [k, l]
            self.edge_attributes[self.current_edge_index] = self.create_single_edge_attributes(k, l)
            self.current_edge_index += 1

    def grow_nodes(self):
        while len(self.node_indices_remaining) > 0:
            index = self.node_indices_remaining.pop(0)

            # now we choose a random already inserted node to attach this node to
            anchor_index = random.choice(self.node_indices_inserted)

            self.node_attributes[index] = self.create_single_node_attributes(anchor_index)
            self.seed_graph_association[index] = self.seed_graph_association[anchor_index]
            self.insert_edge(index, anchor_index)
            self.node_indices_inserted.append(index)

    def add_edges(self):
        # This method is supposed to add additional edges to the grown graph, because the base growing
        # procedure will always be a tree (no cycles).
        # The most important part of this will be to connect the multiple non-connected graphs grown from
        # potentially multiple seed graphs. All additional edges beyond that will be inserted between
        # random nodes.
        additional_edge_count = self.additional_edge_count
        if len(self.seed_graphs) > 1:
            for graph_index in range(len(self.seed_graphs) - 1):
                if (len(self.seed_graphs[graph_index]['node_indices']) == 0 or
                        len(self.seed_graphs[graph_index + 1]['node_indices']) == 0):
                    continue
                # We choose a random node from both subgraph seeded clusters of nodes and insert an edge
                # there
                i = random.choice([k for k, gi in enumerate(self.seed_graph_association)
                                   if gi == graph_index and self.seed_graph_indices[k] < 0])
                j = random.choice([k for k, gi in enumerate(self.seed_graph_association)
                                   if gi == graph_index + 1])
                self.insert_edge(i, j)
                additional_edge_count -= 1

        while additional_edge_count > 0:
            i, j = random.sample(self.node_indices_inserted, k=2)

            # This first condition here makes sure that the edge is only inserted when that exact same
            # edge does not already exist.
            edge_index_tuples = [tuple(e) for e in self.edge_indices]
            edge_exist = ((i, j) in edge_index_tuples or
                          (j, i) in edge_index_tuples)
            # The second condition makes sure that if the prevent_edges_in_seed_graphs flag is set that we
            # do not insert an additional edge between two nodes of the seed graph, as that would kind of
            # disturb the subgraph structure
            seed_cond = (self.seed_graph_indices[i] == -1 or
                         self.seed_graph_indices[j] == -1 or
                         not self.prevent_edges_in_seed_graphs)

            # It is additionally possible to pass a callback to the constructor of this class which takes a
            # possible edge as arguments and then depending on some criteria returns the boolean value of
            # whether this edge is to be considered a valid insertion or not
            valid_cond = self.edge_valid_cb(self.graph, i, j)

            if not edge_exist and seed_cond and valid_cond:
                self.insert_edge(i, j)
                additional_edge_count -= 1

    @classmethod
    def node_adjacency_from_edge_indices(cls,
                                         node_count: int,
                                         edge_indices: List[List[int]]) -> List[List[int]]:
        node_adjacency = [[0 for _ in range(node_count)] for _ in range(node_count)]
        for i, j in edge_indices:
            node_adjacency[i][j] = 1

        return node_adjacency
