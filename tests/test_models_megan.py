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
    model = Megan(
        units=[3, 3, 3],
        final_units=[2],
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

    fidelities = model.calculate_fidelity(
        graphs,
        channel_funcs=[
            lambda org, mod: float(org[0] - mod[0]),
            lambda org, mod: float(org[1] - mod[1])
        ]
    )
    assert isinstance(fidelities, np.ndarray)
    assert fidelities.shape == (len(graphs), num_channels)
