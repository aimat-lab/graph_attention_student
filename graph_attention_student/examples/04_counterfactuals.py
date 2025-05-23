"""
This example showcases how the ``vgd_counterfactuals`` library can be used to generate and visualize
counterfactuals for the predictions of a MEGAN model.

**WHAT ARE COUNTERFACTUALS**

Counterfactuals are generally a type of explanation for a specific prediction of a model. A counterfactual
is an input element which is very similar to the original input, but at the same time causes the largest
deviation of the model's prediction. One could say they are "counter examples".

In the context of the MEGAN graph neural networks, counterfactuals will consist of slightly modified graphs
which maximize a customizable distance w.r.t. to the model prediction. What kind of modifications of the
input graphs are "valid" can also be customized and heavily depends on the application domain.

**HOW IT WORKS**

Generally, counterfactuals are generated with the ``CounterfactualGenerator`` class of the
``vgd_counterfactuals`` library. To do that, an instance of that class needs to be constructed with the
following arguments:

- ``processing`` - An instance of the specific Processing object instance related to the subject visual
  graph dataset. This object instance contains the functionality to transform the domain-specific graph
  representation, such as a SMILES code, into the full graph representation.
- ``model`` - The actual model instance to be explained from. This model needs to implement the
  ``PredictGraphMixin`` interface!
- ``neighborhood_func`` - A function which takes the domain specific representation of an element as the
  argument and returns a list of all elements in the immediate neighborhood of it (in terms of valid graph
  edit operations). This function will have to be custom implemented for each application domain, since
  all of them differ w.r.t. what counts as valid graph edits.
- ``distance_func`` - A function which receives two input arguments, the model prediction of the original
  element and the prediction of a possible counterfactual. Should return a single float value that quantifies
  how far apart the two predictions are.
  The generation process will maximize this distance measure.
"""
import os
import pathlib
import tempfile

import tensorflow.keras as ks
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.util import dynamic_import
from vgd_counterfactuals import CounterfactualGenerator
from vgd_counterfactuals.generate.molecules import get_neighborhood
from vgd_counterfactuals.visualization import create_counterfactual_pdf

from graph_attention_student.keras import CUSTOM_OBJECTS

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')
MODEL_PATH = os.path.join(ASSETS_PATH, 'aqsoldb_model')

# ! NOTE: To run the example locally, you will have to download the corresponding visual graph dataset and
# insert the local path here.
VISUAL_GRAPH_DATASET_PATH = '/media/ssd/.visual_graph_datasets/datasets/aqsoldb'
# Insert the SMILES code for whose prediction to generate the counterfactuals here.
SMILES = 'CCCCC=CC'

__DEBUG__ = True


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):

    # ~ Loading the dataset
    # One thing which the counterfactuals need is the "Processing" instance of the visual graph dataset
    # which implements the conversion of the domain specific representation (SMILES) to the full graph
    # representation.
    metadata_map, index_data_map = load_visual_graph_dataset(
        path=e.VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=1000,
        metadata_contains_index=True,
    )
    module = dynamic_import(metadata_map['process_path'])
    processing = module.processing

    # ~ Loading the model
    # The central requirement of the counterfactual generation is the model itself. We load a model from
    # persistent memory here so we don't have to go through the lengthy process of training one every time.

    # https://www.tensorflow.org/guide/keras/save_and_serialize
    assert os.path.exists(MODEL_PATH), 'The given MODEL_PATH does not exist!'
    assert os.path.isdir(MODEL_PATH), 'The given MODEL_PATH is not a folder as it should be!'
    e.log('loading the model...')
    with ks.utils.custom_object_scope(CUSTOM_OBJECTS):
        model = ks.models.load_model(e.MODEL_PATH)

    # ~ Defining the distance function
    def distance_func(original, modified):
        # Both "original" and "modified" are the outputs generated by the "model", which in this case is
        # a MEGAN model.  That means that the outputs will be tuples of the following format:
        # (prediction, node_importances, edge_importances).
        # In this example we would like the counterfactuals to only increase the regression value. Since
        # the counterfactual generator tries to maximize the distance we do NOT use the abs() function here
        # to achieve that desired result.
        return float(modified[0] - original[0])

    # ~ Counterfactual Generation
    # The generator object needs to be constructed with all the previously mentioned components. After it
    # is constructed, it can be used for as many independent counterfactual generation processes as needed.
    generator = CounterfactualGenerator(
        model=model,
        processing=processing,
        neighborhood_func=get_neighborhood,
        distance_func=distance_func
    )

    # The counterfactual generation process will actually already generate the graph representations and the
    # visualizations for the top k counterfactuals in the format of a visual graph dataset folder.
    # So here we create a temporary folder where we save these temporary visual graph dataset elements into
    with tempfile.TemporaryDirectory() as path:

        e.log('generating the counterfactuals...')
        cf_index_data_map = generator.generate(
            original=e.SMILES,
            path=path,
            k_results=10,  # number of elements with the biggest deviation to create the vgd elements for
        )

        # ~ Visualizing the results
        # These results can be visualized rather easily through the vgd_counterfactuals library as well.
        # The only additional step we need to do here is to create a new visual graph dataset element for
        # the original SMILES codes as well, since that is not included in the generated dataset of
        # counterfactuals.
        # The function below will create a PDF file that visualizes the original element on the first page
        # and all the top k counterfactuals on the following pages ordered by their distance.
        pdf_path = os.path.join(e.path, 'counterfactuals.pdf')
        counterfactual_elements = list(cf_index_data_map.values())
        original_element = generator.create(e.SMILES, path, index='-1')
        create_counterfactual_pdf(
            counterfactual_elements=counterfactual_elements,
            counterfactual_labels=[f'Prediction: {element["metadata"]["prediction"][0]}'
                                   for element in counterfactual_elements],
            original_element=original_element,
            original_label=f'Prediction: {original_element["metadata"]["prediction"][0]}',
            output_path=pdf_path,
        )


experiment.run_if_main()
