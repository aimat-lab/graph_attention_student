import os

import numpy as np
from visual_graph_datasets.data import load_visual_graph_dataset

from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.models.megan import Megan

from .util import ASSETS_PATH

MOCK_PATH = os.path.join(ASSETS_PATH, 'mock')


def test_megan_basically_works():
    _, index_data_map = load_visual_graph_dataset(MOCK_PATH)
    dataset_length = len(index_data_map)

    num_channels = 2
    num_targets = 2
    model = Megan(
        units=[3, 3, 3],
        final_units=[num_targets],
        final_activation='softmax',
        importance_channels=num_channels
    )

    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    predictions = model.predict_graphs(graphs)

    # The least we can expect is that there is one prediction for each element
    assert len(predictions) == dataset_length
    # But then on a more detailed basis we need to check of the shapes of the explanations are correct!
    for prediction, graph in zip(predictions, graphs):
        out, ni, ei = prediction
        assert len(out) == 2
        assert len(ni) == len(graph['node_indices'])
        assert len(ei) == len(graph['edge_indices'])

    deviations = model.leave_one_out_deviations(
        graphs
    )
    assert isinstance(deviations, np.ndarray)
    assert deviations.shape == (len(graphs), num_channels, num_targets)
