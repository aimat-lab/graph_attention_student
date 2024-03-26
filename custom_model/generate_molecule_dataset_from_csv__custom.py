import os
import typing as t

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from visual_graph_datasets.util import EXPERIMENTS_PATH

# == SOURCE PARAMETERS ==
# These parameters determine how to handle the source CSV file of the dataset. There exists the possibility
# to define a file from the local system or to download a file from the VGD remote file share location.
# In this section one also has to determine, for example, the type of the source dataset (regression, 
# classification) and provide the names of the relevant columns in the CSV file.

# :param FILE_SHARE_PROVIDER:
#       The vgd file share provider from which to download the CSV file to be used as the source for the VGD
#       conversion. 
FILE_SHARE_PROVIDER: str = 'main'
# :param CSV_FILE_NAME:
#       The name of the CSV file to be used as the source for the dataset conversion.
#       This may be one of the following two things:
#       1. A valid absolute file path on the local system pointing to a CSV file to be used as the source for
#       the VGD conversion
#       2. A valid relative path to a CSV file stashed on the given vgd file share provider which will be
#       downloaded first and then processed.
CSV_FILE_NAME: str = 'source/benzene_solubility.csv'
# :param INDEX_COLUMN_NAME:
#       (Optional) this may define the string name of the CSV column which contains the integer index
#       associated with each dataset element. If this is not given, then integer indices will be randomly
#       generated for each element in the final VGD
INDEX_COLUMN_NAME: t.Optional[str] = None
# :param INDICES_BLACKLIST_PATH:
#       Optionally it is possible to define the path to a file which defines the blacklisted indices for the 
#       dataset. This file should contain a list of integers, where each integer represents the index of an
#       element which should be excluded from the final dataset. The file should be a normal TXT file where each 
#       integer is on a new line.
#       The indices listed in that file will be immediately skipped during processing without even loading the 
#       the molecule.
INDICES_BLACKLIST_PATH: t.Optional[str] = None
# :param SMILES_COLUMN_NAME:
#       This has to be the string name of the CSV column which contains the SMILES string representation of
#       the molecule.
SMILES_COLUMN_NAME: str = 'SMILES'
# :param TARGET_TYPE:
#       This has to be the string name of the type of dataset that the source file represents. The valid 
#       options here are "regression" and "classification"
TARGET_TYPE: str = 'regression'  # 'classification'
# :param TARGET_COLUMN_NAMES:
#       This has to be a list of string column names within the source CSV file, where each name defines 
#       one column that contains a target value for each row. In the regression case, this may be multiple 
#       different regression targets for each element and in the classification case there has to be one 
#       column per class.
TARGET_COLUMN_NAMES: t.List[str] = ['LogS']
# :param SPLIT_COLUMN_NAMES:
#       The keys of this dictionary are integers which represent the indices of various train test splits. The
#       values are the string names of the columns which define those corresponding splits. It is expected that
#       these CSV columns contain a "1" if that corresponding element is considered as part of the training set
#       of that split and "0" if it is part of the test set.
#       This dictionary may be empty and then no information about splits will be added to the dataset at all.
SPLIT_COLUMN_NAMES: t.Dict[int, str] = {
}
# :param SUBSET:
#       Optional. This can be used to set a number of elements after which to terminate the processing procedure. 
#       If this is None, the whole dataset will be processed. This feature can be useful if only a certain 
#       part of the datase should be processed or for testing reasons for example.
SUBSET: t.Optional[int] = None

# == PROCESSING PARAMETERS ==
# These parameters control the processing of the raw SMILES into the molecule representations with RDKit
# and then finally the conversion into the graph dict representation.

# :param UNDIRECTED_EDGES_AS_TWO:
#       If this flag is True, the undirected edges which make up molecular graph will be converted into two
#       opposing directed edges. Depends on the downstream ML framework to be used.
UNDIRECTED_EDGES_AS_TWO: bool = True
# :param USE_NODE_COORDINATES:
#       If this flag is True, the coordinates of each atom will be calculated for each molecule and the resulting
#       3D coordinate vector will be added as a separate property of the resulting graph dict.
USE_NODE_COORDINATES: bool = True
# :param GRAPH_METADATA_CALLBACKS:
#       This is a dictionary that can be use to define additional information that should be extracted from the 
#       the csv file and to be transferred to the metadata dictionary of the visual graph dataset elements.
#       The keys of this dict should be the string names that the properties will then have in the final metadata 
#       dictionary. The values should be callback functions with two parameters: "mol" is the rdkit molecule object 
#       representation of each dataset element and "data" is the corresponding dictionary containing all the 
#       values from the csv file indexed by the names of the columns. The function itself should return the actual 
#       data to be used for the corresponding custom property. 
GRAPH_METADATA_CALLBACKS = {
    'name': lambda mol, data: data['smiles'],
    'smiles': lambda mol, data: data['smiles'],
}


# == DATASET PARAMETERS ==
# These parameters control aspects of the visual graph dataset creation process. This for example includes 
# the dimensions of the graph visualization images to be created or the name of the visual graph dataset 
# that should be given to the dataset folder.

# :param DATASET_CHUNK_SIZE:
#       This number will determine the chunking of the dataset. Dataset chunking will split the dataset
#       elements into multiple sub folders within the main VGD folder. Especially for larger datasets
#       this should increase the efficiency of subsequent IO operations.
#       If this is None then no chunking will be applied at all and everything will be placed into the
#       top level folder.
DATASET_CHUNK_SIZE: t.Optional[int] = 10_000
# :param DATASET_NAME:
#       The name given to the visual graph dataset folder which will be created.
DATASET_NAME: str = 'dataset'
# :param IMAGE_WIDTH:
#       The width molecule visualization PNG image
IMAGE_WIDTH: int = 1000
# :param IMAGE_HEIGHT:
#       The height of the molecule visualization PNG image
IMAGE_HEIGHT: int = 1000

# == EVALUATION PARAMETERS ==
# These parameters control the evaluation process which included the plotting of the dataset statistics 
# after the dataset has been completed for example.

# :param EVAL_LOG_STEP:
#       The number of iterations after which to print a log message
EVAL_LOG_STEP = 100
# :param NUM_BINS:
#       The number of bins to use for the histogram 
NUM_BINS = 10
# :param PLOT_COLOR:
#       The color to be used for the plots
PLOT_COLOR = 'gray'


# == EXPERIMENT PARAMETERS ==

__DEBUG__ = True
experiment = Experiment.extend(
    os.path.join(EXPERIMENTS_PATH, 'generate_molecule_dataset_from_csv.py'),
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

experiment.run_if_main()
