"""
This module contains all the fidelity-related utility functions.
"""
import typing as t

import visual_graph_datasets.typing as tv
from visual_graph_datasets.models import PredictGraphMixin
from kgcnn.data.utils import ragged_tensor_from_nested_numpy

from graph_attention_student.util import PathDict
from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.models.megan import Megan


def leave_one_out_analysis(model: Megan,
                           graphs: t.List[tv.GraphDict],
                           num_targets: int,
                           num_channels: int,
                           func: t.Callable[[int, int, int, float], None] = None,
                           ) -> dict:
    """
    Given a MEGAN-like model and a set of input graphs, this function will perform the leave-one-out
    analysis for all the importance channels. This analysis works as follows: At first the model will be
    queried with the given input graphs as-is to get the original predictions then for every importance
    channel of the model, that channel will be masked out during a new prediction inference. And for every
    pairing of importance channel and value from the output vector, the resulting deviation of that will
    be calculated.

    This method will return all these results in the form of a 3-layer nested dictionary "results":
    results[element_index][target_index][channel_index]. The values will be float values of the difference
    original - masked.

    :param model: A Megan-like model in the sense that it supports a channel-masked prediction.
    :param graphs: A list GraphDicts to be used as the input
    :param num_targets: The integer number of target values that the model produces as output
    :param num_channels: The integer number of importance channels which the model employs
    :param func: An optional callable function which can be used to inject some custom code execution
        for each computed deviation value.

    :returns: dict with "shape" (num_elements, num_targets, num_channels).
    """
    results = PathDict(dtype=int)
    x = tensors_from_graphs(graphs)

    # First of all we need to create the original predictions of the model, on which we even want to
    # base the fidelity / deviation analysis!
    y_org = model(x, training=False)
    out_org, _, _ = [v.numpy() for v in y_org]

    # First of all we need to create all the deviations
    for channel_index in range(num_channels):
        base_mask = [float(channel_index != i) for i in range(num_channels)]
        mask = [[base_mask for _ in graph['node_indices']] for graph in graphs]
        mask_tensor = ragged_tensor_from_nested_numpy(mask)
        y_mod = model(
            x,
            training=False,
            node_importances_mask=mask_tensor,
        )
        out_mod, _, _ = [v.numpy() for v in y_mod]
        for index, out in enumerate(out_mod):
            for target_index in range(num_targets):
                deviation = float(out_org[index][target_index] - out_mod[index][target_index])

                results[f'{index}/{target_index}/{channel_index}'] = deviation
                if func is not None:
                    func(index, target_index, channel_index, deviation)

    return results.data
