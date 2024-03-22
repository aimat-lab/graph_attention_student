"""
Utility methods
"""
import os
import csv
import pathlib
import tempfile
import logging
import subprocess
import shutil
import typing as t
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from collections import defaultdict

import numpy as np
import jinja2 as j2
import matplotlib.colors as mcolors
from nltk.tokenize import word_tokenize
from scipy.spatial import distance

PATH = pathlib.Path(__file__).parent.absolute()
VERSION_PATH = os.path.join(PATH, 'VERSION')

DATASETS_FOLDER = os.path.join(PATH, 'datasets')
EXAMPLES_FOLDER = os.path.join(PATH, 'examples')
TEMPLATES_FOLDER = os.path.join(PATH, 'templates')
EXPERIMENTS_PATH = os.path.join(PATH, 'experiments')
ASSETS_PATH = os.path.join(PATH, 'assets')

TEMPLATE_ENV = j2.Environment(
    loader=j2.FileSystemLoader(TEMPLATES_FOLDER),
    autoescape=j2.select_autoescape(),
)
TEMPLATE_ENV.globals.update(**{
    'zip': zip,
    'int': int,
    'enumerate': enumerate
})

NULL_LOGGER = logging.Logger('NULL')
NULL_LOGGER.addHandler(logging.NullHandler())

# These are the default tokens which will be removed from a string during the conversion of text to
# graph
SANITIZE_TOKENS = ['', ',', '.', '\'s', '"', '-', '?', '!', '/', '(', ')', '_', '--', ';', '``']
#SANITIZE_TOKENS += ['the', 'be', 'to', 'of', 'and', 'a', 'an', 'in', 'that', 'it', 'you', 'me', 'i', 'at']


def get_experiment_path(name: str) -> str:
    """
    Returns the absolute path to the experiment with the given ``name``

    :returns: the string absolute path
    """
    return os.path.join(EXPERIMENTS_PATH, name)


def get_version():
    with open(VERSION_PATH) as file:
        return file.read().replace(' ', '').replace('\n', '')


def update_nested_dict(original: dict, extension: dict):
    result = original.copy()
    for key, value in extension.items():
        if isinstance(value, dict) and key in result:
            result[key] = update_nested_dict(result[key], value)
        else:
            result[key] = value

    return result


class PathDict:

    def __init__(self,
                 dtype: type = str,
                 data: t.Optional[dict] = None):
        self.dtype = dtype
        self.data = {}
        if data is not None:
            self.data = data

    def __getitem__(self, item):
        keys = item.split('/')
        current_dict = self.data
        for current_key in keys:
            current_key = self.dtype(current_key)
            current_dict = current_dict[current_key]

        return current_dict

    def __setitem__(self, key, value):
        keys = key.split('/')
        current_dict = self.data
        for i, current_key in enumerate(keys):
            current_key = self.dtype(current_key)
            if current_key not in current_dict:
                current_dict[current_key] = {}

            if i < len(keys) - 1:
                current_dict = current_dict[current_key]

        current_dict[current_key] = value

# == GENERAL GRAPH OPERATIONS ===============================================================================

def export_metadatas_csv(metadatas: t.List[dict],
                         path: str = int,
                         ) -> t.List[dict]:
    """
    Given a list of visual graph metadata dicts ``metadatas`` this function will export the basic information 
    about those graphs to a CSV file with the given absolute string ``path``.
    
    :param metadatas: The list of visual graph dataset metadata dictionaries to be exported
    :param path: The absolute string path at which to create the csv file
    
    :returns: Returns a list of row dictionaries, which contain the exact information that was exported to 
        the csv file. 
    """
    # We dont want to export ALL the information in the metadata dict to the CSV file. That would be way 
    # too much information, since that would include the full graph represenation as well. So in this 
    # first step we select a few key entries from the metadata dict to then actually save to the csv
    rows: t.List[dict] = []
    for metadata in metadatas:
        
        # 23.01.24
        # This is for backwards compatibility of datasets. Both of these entries refer to the same thing - 
        # the string representation of the graph - but at some point the name for this field was changed 
        # from "value" to "repr"
        if 'repr' in metadata:
            value = metadata['repr']
        elif 'value' in metadata:
            value = metadata['value']
        
        # Some entries have to be in the metadata dict as per the definition of the visual graph dataset 
        # format, but some values are optional and might not be there - for those we add placeholders 
        # initially and then check for each individual entry if they exist within the dict structure.
        row = {
            'index':    metadata['index'],
            'name':     None,
            'value':    value,
            'out_true': metadata['graph']['graph_labels'],
            'out_pred': None,
        }
        
        if 'name' in metadata:
            row['name'] = metadata['name']
        
        if 'graph_output' in metadata['graph']:
            row['out_pred'] = metadata['graph']['graph_output']

        rows.append(row)

    # Now in the next step we can export those row dicts as the actual csv file.
    with open(path, mode='w') as file:
        
        writer = csv.DictWriter(file, fieldnames=['index', 'name', 'value', 'out_true', 'out_pred'])
        writer.writeheader()
        
        for row in rows:
            writer.writerow(row)

    return rows


def node_adjacency_sliding_window(node_indices: List[int],
                                  window_size: int,
                                  do_self_loops: bool,
                                  ) -> List[List[int]]:
    # First we initialize the adjacency matrix
    node_adjacency = [[0 for _ in node_indices] for _ in node_indices]
    for i in node_indices:
        for j in node_indices:
            if abs(i - j) <= window_size and (i != j or do_self_loops):
                node_adjacency[i][j] = 1

    return node_adjacency


def edge_indices_from_adjacency(node_adjacency: List[List[int]]
                                ) -> List[List[Tuple[int, int]]]:
    node_indices = list(range(len(node_adjacency)))
    return [[i, j] for i in node_indices for j in node_indices if node_adjacency[i][j]]


def graph_dict_to_list_values(g: dict) -> dict:
    """
    Given a GraphDict ``g``, this function will make sure that all of the values of that dict are converted
    to lists. So usually the values of a GraphDict are numpy arrays, but we also want to be able to serialize
    these data structures as JSON strings and that is not possible with numpy arrays, instead we need the
    native representation as nested lists for this.

    :param GraphDict g: The graph representation to be converted

    :returns GraphDict: All of the values of this dict will be nested lists instead of numpy arrays.
    """
    result = {}
    for key, value in g.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        else:
            result[key] = value

    return result


def graph_dict_to_numpy_values(g: dict) -> dict:
    """
    Given a GraphDict ``g``, this function will make sure that all the values are converted to numpy arrays.
    When loading graph structures from JSON files, the values are still represented as nested lists, but for
    various operations on GraphDicts, the values need to be numpy arrays and this function makes sure all
    the values are properly converted.

    :param GraphDict g: The graph representation to be converted

    :returns GraphDict: All value of this dict will be numpy arrays instead of nested lists.
    """
    result = {}
    for key, value in g.items():
        result[key] = np.array(value)

    return result


def importance_absolute_similarity(importances_true: list,
                                   importances_pred: list,
                                   normalize: bool = True
                                   ) -> float:
    importances_true = np.array(importances_true)
    importances_pred = np.array(importances_pred)

    if normalize:
        norm = mcolors.Normalize(vmin=0, vmax=np.max(importances_pred))
        importances_pred = np.vectorize(norm)(importances_pred)

    return 1 - np.mean(np.abs(importances_true - importances_pred))


def mask_distance(y_true: np.ndarray, y_pred: np.ndarray, cutoff=0.1) -> float:
    # return distance.canberra(y_true + epsilon, y_pred + epsilon) / len(y_true)
    y_pred_mod = np.copy(y_pred)
    y_pred_mod[y_pred < cutoff] = 0
    return distance.canberra(y_true, y_pred_mod) / len(y_true)

    y_true_inv = 1 - y_true
    y_pred_inv = 1 - y_pred
    return distance.cosine(y_true_inv, y_pred_inv)


def importance_canberra_similarity(importances_true: list,
                                   importances_pred: list,
                                   normalize: bool = True
                                   ) -> Optional[float]:
    # Make sure that we are working with numpy arrays
    importances_true = np.array(importances_true)
    importances_pred = np.array(importances_pred)

    if normalize:
        norm = mcolors.Normalize(vmin=0, vmax=np.max(importances_pred))
        importances_pred = np.vectorize(norm)(importances_pred)

    if len(importances_pred.shape) == 1:
        dist = mask_distance(importances_true, importances_pred)

    if len(importances_pred.shape) == 2:
        distances = [mask_distance(importances_true[:, k], importances_pred[:, k])
                     for k in range(importances_pred.shape[1])]
        dist = np.mean(distances)

    return 1 - dist


def normalize_importances_fidelity(importances: np.ndarray,
                                   fidelities: t.List[float],
                                   delta: float = 0.01
                                   ) -> np.ndarray:
    """
    This function can be used to normalize an ``importances`` array (either node or edge importances) of
    the shape (N, K) according to the ``fidelities`` of the corresponding importance channels (K, ).
    This function will scale the importances to fit the relative ratio of the fidelities.

    :returns: np array of shape (N, K)
    """
    # We add a small delta value here to avoid a division by zero error.
    max_value = max(fidelities) + delta
    for channel_index, value in enumerate(fidelities):
        importances[:, channel_index] *= (value + delta) / max_value

    return importances


def array_normalize(array: np.ndarray
                    ) -> np.ndarray:
    """
    Normalizes all the values of the given ``array`` into a range between 0 and 1 based on the values 
    maximum and minimum values of the array.
    
    :param array: The array to be normalized
    
    :returns: The normalized array 
    """
    norm = mcolors.Normalize(vmin=np.min(array), vmax=np.max(array))
    return np.vectorize(norm)(array)


def binary_threshold(array: np.ndarray,
                     threshold: float,
                     ) -> np.ndarray:
    """
    Applies a binary threshold to the given ``array``. All values that are greater than the given 
    ``threshold`` will be set to 1, all other values will be set to 0.
    
    :param array: The array to be thresholded
    :param threshold: The threshold value
    
    :returns: The thresholded binary array
    """
    binary = np.zeros_like(array)
    binary[array > threshold] = 1
    return binary


# https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
def binary_threshold_k(array: np.array,
                       k: int,
                       inverse: bool = False
                       ) -> np.array:
    indices = np.argpartition(array, -k)[-k:]
    if inverse:
        binary = np.ones_like(array)
        binary[indices] = 0
    else:
        binary = np.zeros_like(array)
        binary[indices] = 1

    return binary


def fidelity_from_deviation(dev: np.ndarray,
                            dataset_type: t.Literal['classification', 'regression']
                            ) -> np.ndarray:
    """
    Given the leave-one-out deviation array ``dev`` and the string identifier for the ``dataset_type``, this 
    function returns the fidelity array.
    
    :param dev: The leave-one-out deviation array with the shape (num_outputs, num_channels)
    :param dataset_type: The string identifier for the dataset type. This can be either "classification" or
        "regression".
    :returns: The fidelity array with the shape (num_channels, )
    """
    
    # For classification tasks, the number of channels has to be equal to the number of possible classes 
    # (== number of outputs). And in this case the fidelity depends on the deviation that a channel causes 
    # to the logits of it's own class (the output of the same index), which is the diagonal of the deviation
    # matrix.
    if dataset_type == 'classification':
        fidelity = np.diag(dev)
        
    # For regression tasks, the number of channels is usually 2, where the first channel is the negative 
    # and the second channel is the positive channel. Also the number of outputs can only be 1 in this case.
    # The fidelity just indicates the absolute influence we need to invert the sign of the negative channel's 
    # deviation value.
    elif dataset_type == 'regression':
        fidelity = np.array([-dev[0, 0], dev[0, 1]])    

    else:
        raise ValueError(f'Unknown dataset type: {dataset_type} - only "classification" and "regression" are allowed.')

    return fidelity


# == NATURAL LANGUAGE PROCESSING ============================================================================


def load_glove_map(path: str,
                   line_buffer_size: int = 10
                   ) -> Dict[str, list]:
    """
    Given the ``path`` to a file of pre-trained transformations of a GLOVE word to vector model, this
    function will load that file into memory as a defaultdict. The keys of this dict are the string tokens
    and the values are lists with the corresponding numeric vectors. The default case will return a
    zeros array with the same dimensions.

    :param str path: The absolute path to the glove txt file to be used
    :param int line_buffer_size: The number of lines to read from the file at a time. The reading is done
        in a buffered manner because the glove files can get extremely large.
    :return: A dict which maps string tokens to lists of numbers
    """
    glove_map = defaultdict(list)

    num_dimensions = None
    line_index = 0
    with open(path, mode='r') as file:

        lines = file.readlines(line_buffer_size)
        while lines:

            for line in lines:
                token, *values = line.split(' ')
                values = [float(value) for value in values]
                glove_map[token.lower()] = values

                if num_dimensions is None:
                    num_dimensions = len(values)
                elif num_dimensions != len(values):
                    raise ValueError(f'Line {line_index} has a different dimension than previously: '
                                     f'{num_dimensions} != {len(values)}')

                line_index += 1

            lines = file.readlines(line_buffer_size)

    glove_map.default_factory = lambda: [0] * num_dimensions

    return glove_map


def token_dict_to_graph(token_dict: Dict[int, Any],
                        window_size: int = 2,
                        do_self_loops: bool = False
                        ) -> dict:
    g = defaultdict(list)
    for node_index, (token_index, token_attributes) in enumerate(token_dict.items()):
        g['node_indices'].append(node_index)
        g['token_indices'].append(token_index)
        g['node_attributes'].append(token_attributes)

    # Now that we basically already have all of the nodes, we need to connect them up with edges
    # according to the specified window size.
    g['node_adjacency'] = node_adjacency_sliding_window(
        node_indices=g['node_indices'],
        window_size=window_size,
        do_self_loops=do_self_loops
    )

    # Based on the adjacency matrix we can then create the list of edge index tuples and the edge attributes
    # weights, which are just all constant ones in this case
    for i in g['node_indices']:
        for j in g['node_indices']:
            if g['node_adjacency'][i][j]:
                g['edge_indices'].append([i, j])
                g['edge_attributes'].append([1.])

    return dict(g)


graph_from_token_dict = token_dict_to_graph


def text_to_graph(text: str,
                  glove_map: Dict[str, list],
                  sanitize_tokens: list = SANITIZE_TOKENS,
                  window_size: int = 2,
                  do_self_loops: bool = False,
                  include_node_strings: bool = False,
                  ) -> Dict[int, np.ndarray]:
    # ~ 1. Convert string to token dict
    token_list: List[str] = word_tokenize(text)
    token_dict: Dict[int, str] = {i: t.lower()
                                  for i, t in enumerate(token_list)
                                  if t.lower() not in sanitize_tokens}

    # ~ 2. Convert string tokens to vectors with GLOVE
    embedding_dict: Dict[int, np.ndarray] = {i: glove_map[t] for i, t in token_dict.items()}
    g = graph_from_token_dict(
        embedding_dict,
        window_size=window_size,
        do_self_loops=do_self_loops
    )

    if include_node_strings:
        g['node_strings'] = list(token_dict.values())

    # ~ 3. Convert graph properties to np arrays
    g = {k: np.array(v) for k, v in g.items()}

    return g


def latex_table_element_mean(values: List[float],
                             template_name: str = 'table_element_mean.tex.j2',
                             vertical: bool = True,
                             raw: bool = False) -> str:
    if raw:
        mean, std = values
    else:
        mean = np.mean(values)
        std = np.std(values)

    template = TEMPLATE_ENV.get_template(template_name)
    return template.render(
        mean=mean,
        std=std,
        vertical=vertical
    )


def latex_table_element_median(values: List[float],
                               upper_quantile: float = 0.75,
                               lower_quantile: float = 0.25,
                               include_variance: bool = True,
                               template_name: str = 'table_element_median.tex.j2') -> str:
    median = np.median(values)
    upper = np.quantile(values, upper_quantile)
    lower = np.quantile(values, lower_quantile)

    template = TEMPLATE_ENV.get_template(template_name)
    return template.render(
        median=median,
        upper=upper,
        lower=lower,
        include_variance=include_variance
    )


def latex_table(column_names: List[str],
                rows: List[Union[List[float], str]],
                content_template_name: str = 'table_content.tex.j2',
                table_template_name: str = 'table.tex.j2',
                list_element_cb: Callable[[List[float]], str] = latex_table_element_mean,
                prefix_lines: List[str] = [],
                caption: str = '',
                ) -> Tuple[str, str]:

    # ~ Pre Processing the row elements into strings
    string_rows = []
    for row_index, row in enumerate(rows):
        string_row = []
        for element in row:
            if isinstance(element, str):
                string_row.append(element)
            if isinstance(element, list) or isinstance(element, np.ndarray):
                string = list_element_cb(element)
                string_row.append(string)

        string_rows.append(string_row)

    alignment = ''.join(['c' for _ in column_names])

    # ~ Rendering the latex template(s)

    content_template = TEMPLATE_ENV.get_template(content_template_name)
    content = content_template.render(rows=string_rows)

    table_template = TEMPLATE_ENV.get_template(table_template_name)
    table = table_template.render(
        alignment=alignment,
        column_names=column_names,
        content=content,
        header='\n'.join(prefix_lines),
        caption=caption,
    )

    return content, table


def render_latex(kwargs: dict,
                 output_path: str,
                 template_name: str = 'article.tex.j2'
                 ) -> None:
    with tempfile.TemporaryDirectory() as temp_path:
        # First of all we need to create the latex file on which we can then later invoke "pdflatex"
        template = TEMPLATE_ENV.get_template(template_name)
        latex_string = template.render(**kwargs)
        latex_file_path = os.path.join(temp_path, 'main.tex')
        with open(latex_file_path, mode='w') as file:
            file.write(latex_string)

        # Now we invoke the system "pdflatex" command
        command = (f'pdflatex  '
                   f'-interaction=nonstopmode '
                   f'-output-format=pdf '
                   f'-output-directory={temp_path} '
                   f'{latex_file_path} ')
        proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise ChildProcessError(f'pdflatex command failed! Maybe pdflatex is not properly installed on '
                                    f'the system? Error: {proc.stdout.decode()}')

        # Now finally we copy the pdf file - currently in the temp folder - to the final destination
        pdf_file_path = os.path.join(temp_path, 'main.pdf')
        shutil.copy(pdf_file_path, output_path)


# == CUSTOM EXCEPTIONS ======================================================================================


class DatasetError(Exception):
    pass

