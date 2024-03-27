==============================================
🤖 Training a MEGAN Model for a Custom Dataset
==============================================

This file will discuss the required steps to apply the MEGAN model to a custom dataset. Generally, the easiest way to do 
this is to use the already existing experimentation code in this package and simply tune the parameters of these experiment 
scripts to your specific needs.

Overall, the process can be summerized into the following two steps:

1. Convert the custom dataset from it's original format into a *visual graph dataset*. This is a special dataset format
   from which the model can be easily trained.
2. Configure the model hyperparameters and train the model on the visual graph dataset created in the previous step. 


🧪 PyComex - Computational Experiments
======================================

Generally, all the experimentation code is based on the PyComex_ micro-framework. PyComex is a simple and lightweight 
framework that simplifies many aspects of developing, executing and managing computational experiments. This section 
introduces some of its core aspects which are necessary for the subsequent sections. 
For more detailed information, please visit the PyComex package: https://github.com/the16thpythonist/pycomex

In PyComex, each individual experiment should be self-contained as its own Python module. These modules can be executed 
to execute the corresponding experiment. Each experiment will automatically create a new folder containing all the 
experiment artifacts in a subfolder of the ``results`` folder. An example folder structure may look like this:

.. code-block:: text

    .
    └── experiments/
        ├── results/
        │   ├── first_experiment/
        │   │   └── debug/
        │   │       ├── experiment_data.json
        │   │       ├── experiment_meta.json
        │   │       ├── experiment_log.txt
        │   │       └── plot.png
        │   └── second_experiment
        ├── first_experiment.py
        └── second_experiemnt.py


Experiments are designed to follow the DRY (dont repeat yourself) principle. When creating a new experiment, which is 
simply a small deviation of an already existing experiment, the code doesn't have to be copied, but can be reused directly 
by using *experiment inheritance*. A sub-experiment can be defined like this:

.. code-block:: python

    from pycomex.functional.experiment import Experiment
    from pycomex.utils import folder_path, file_namespace

    # The default parameters of the base path can be overwritten simply be assigning the 
    # parameter variables new values at the beginning of the module.
    SOURCE_PATH: str = 'path/to/new/source/file.txt'

    experiment = Experiment.extend(
        'base_experiment.py',
        base_path=folder_path(__file__),
        namespace=file_namespace(__file__),
        glob=globals()
    )

    # Besides parameter values, it is also possible to overwrite certain functionality
    # in the base experiment by overwriting the corresponding hooks.
    # In this example, whenever the "load_source" hook function would be invoked in the 
    # base experiment file, the functionality would be replaced by the custom code that 
    # is defined here in the sub-experiment.
    @experiment.hook('load_source', default=False, replace=True)
    def load_source(e: Experiment) -> str:

        # The parameter values from the top of the file can accessed as an instance attribute 
        # of the experiment instance, which is passed as a default parameter to each 
        # hook function.
        with open(e.SOURCE_PATH, 'r') as file:
            return file.read()

    # This special method has to be called at the top-most level of the module
    # so that the experiment code actually gets executed when executing the script.
    experiment.run_if_main()


Specifically, this experiment inheritance can be used to extend the already existing functionality for the 
model training and only overwriting specific parameters and functionality that is required for each specific 
use case.

⚗️ Converting SMILES based Datasets for Molecular Property Prediction
====================================================================

Since the application of MEGAN for molecular property prediction for a SMILES-based dataset is the most common use case, the 
following elaborations will use this as an example to demonstrate the required steps to train a custom MEGAN model.

Most often a dataset for molecular property prediction will be given as a CSV file which contains the representations of the 
source molecules in their SMILES string representation in one column, and the corresponding target values in another column.
The following example illustrates how such a source CSV file may look like:

.. code-block:: csv

    smiles,logP
    CCO,0.2
    CCN,0.3
    CCC,0.5
    CC(=O)O,0.8
    CC(=O)N,0.7
    C1CC1,0.6
    ...

Whenever the dataset is given in this CSV format, the pre-defined ``generate_molecule_dataset_from_csv.py`` experiment 
can be used to conveniently convert this CSV format into a visual graph dataset. In essence, one has to create a new 
sub-experiment module that inherits from this base experiment and modify the corresponding experiment parameters that 
provide the necessary information about the source dataset. This sub-experiment can then be executed to generate the 
visual graph dataset format.

.. code-block:: python

    import os
    from pycomex.functional.experiment import Experiment
    from pycomex.utils import folder_path, file_namespace
    from visual_graph_datasets.util import EXPERIMENTS_PATH

    # == CUSTOMIZE PARAMETERS ==

    # Insert absolute path to your own CSV file
    CSV_FILE_PATH: str = 'path/to/file.csv'
    # Insert name of the column that contains the SMILES representation
    SMILES_COLUMN_NAME: str = 'smiles'
    # Insert name of the columns that contain the target values
    TARGET_COLUMN_NAMES: t.List[str] = ['class_0', 'class_1']
    # Define the type of the dataset / task
    TARGET_TYPE: str = 'classification' # or 'regression'
    # The name of the dataset in the artifacts folder  
    DATASET_NAME: str = 'dataset'

    # == INHERIT EXPERIMENT ==

    experiment = Experiment.extend(
        os.path.join(EXPERIMENTS_PATH, 'generate_molecule_dataset_from_csv.py'),
        base_path=folder_path(__file__),
        namespace=file_namespace(__file__),
        glob=globals()
    )
    experiment.run_if_main()


.. note::

    For a classification dataset, there should be as many target columns as there are classes in the dataset. 
    The corresponding values in these columns should be 0/1 values indicating if a molecule belongs to that class 
    or not. For regression problems, the single target column should contain the raw float property values.

After executing this sub-experiment, a new visual graph dataset will be created in the artifacts subfolder of the 
``results`` folder. The absolute path to this dataset folder will be required in the training step.


🤖 Training the MEGAN Model
===========================

Assuming that a new visual graph dataset was successfully generated in the previous step, this section elaborates 
how to train a new MEGAN model based on this dataset.

To train, one has to create a new sub-experiment module that inherits from the ``vgd_torch__megan.py`` base experiment
like this:

.. code-block:: python

    import os
    import typing as t
    from pycomex.functional.experiment import Experiment
    from pycomex.utils import folder_path, file_namespace
    from graph_attention_student.util import EXPERIMENTS_PATH

    # == CUSTOMIZE PARAMETERS ==

    # Insert absolute path to the recently created visual graph dataset folder
    VISUAL_GRAPH_DATASET: str = 'path/to/visual/graph/dataset/folder'
    # Define the type of the dataset / task
    DATASET_TYPE: str = 'classification' # or 'regression'
    # The number of randomly chosen elements from the dataset to act as the test set
    NUM_TEST: int = 1000

    # The number of hidden units in the message passing layers of the network
    UNITS: t.List[int] = [64, 64, 64]
    # The numbers of hidden units in the dense projection networks
    PROJECTION_UNITS: t.List[int] = [64, 128, 258]
    # The number of hidden units in the final prediction network
    # NOTE: The last value must be equal to the number of target values in the dataset!
    FINAL_UNITS: t.List[int] = [32, 1]
    # The number of explanation channels.
    # regression: always 2 (positive and negative) - classification: number of classes
    NUM_CHANNELS: int = 2
    # For regression tasks, this value should be set to the median target value of the 
    # dataset (defines the reference point of what is considered "negative" and "positive")
    REGRESSION_REFERENCE: t.Optional[float] = None

    # == INHERIT EXPERIMENT ==

    experiment = Experiment.extend(
        os.path.join(EXPERIMENTS_PATH, 'vgd_torch__megan.py'),
        base_path=folder_path(__file__),
        namespace=file_namespace(__file__),
        glob=globals()
    )
    experiment.run_if_main()


❓ FAQs
=======

This section will answer some common questions that may arise during the process of training a custom MEGAN model.

What if I need to customize additional aspects not listed here?
---------------------------------------------------------------

In this case, a good first step is to read the read through the base experiment files that are used as the basis of 
of the specific sub-experiments:

- https://github.com/aimat-lab/graph_attention_student/blob/master/graph_attention_student/experiments/vgd_torch__megan.py
- https://github.com/aimat-lab/graph_attention_student/blob/master/graph_attention_student/experiments/vgd_torch.py
- https://github.com/aimat-lab/visual_graph_datasets/blob/master/visual_graph_datasets/experiments/generate_molecule_dataset_from_csv.py

These files define a lot more parameters than the ones that are presented in this guide. Chances are, that you'll already 
find a parameter for the customization you have in mind. If not, the next best option would be to look at what kinds of *hooks* are 
used in these base experiments. As a first measure, it'd make sense to overwrite one of the hooks to achieve the desired 
functionality.

Only if None of these options are possible, you can modify the base experiment files directly to implement the desired functionality.
This measure is discouraged, however, because these custom modifications will conflict with future updates to the base experiment files. 
If the changes are implemented purely by overwriting parameters or hooks in a sub-experiment, the chances are high that these will maintain 
compatible even if the base experiments are upaded in future version of the ``graph_attention_student`` or ``visual_graph_datasets`` packages.

.. _Pycomex: https://github.com/the16thpythonist/pycomex/tree/master