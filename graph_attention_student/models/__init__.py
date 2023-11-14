"""
This file mainly exists to provide the CUSTOM_OBJECTS global dict. This dict will be needed when a model
from this package is to be loaded from a persistent representation, this dictionary has to be passed as
an argument to a keras ``custom_object_scope`` context:

.. code-block:: python

    import tensorflow.keras as ks
    from graph.attention_student.keras import CUSTOM_OBJECTS

    with ks.util.custom_object_scope(CUSTOM_OBJECTS):
        model = ks.models.load_model('model/path')

"""
import tensorflow.keras as ks

from graph_attention_student.training import NoLoss
from graph_attention_student.training import ExplanationLoss

from graph_attention_student.models.megan import *
from graph_attention_student.models.gradient import *
from graph_attention_student.models.gnes import *
from graph_attention_student.models.gnnx import *
from graph_attention_student.models.baseline import *


CUSTOM_OBJECTS = {
    'Megan': Megan,
    'Megan2': Megan2,
    'NoLoss': NoLoss,
    'ExplanationLoss': ExplanationLoss,
    'GCNModel': GCNModel,
}


def load_model(path: str):
    with ks.utils.custom_object_scope(CUSTOM_OBJECTS):
        return ks.models.load_model(path)
