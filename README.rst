=======================
Graph Attention Student
=======================

This repository contains the minimal implementation for the graph attention student model which was used
to conduct the graph student teacher analysis for explanation quality.

Installation
============

Clone the repository from github:

.. code-block:: console

    git clone https://github.com/aimat-lab/graph_attention_student.git

Then in the main folder run a ``pip install``:

.. code-block:: console

    cd graph_attention_student
    pip3 install .

Afterwards, you can check the install by invoking the CLI:

.. code-block:: console

    python3 -m graph_attention_student.cli --version
    python3 -m graph_attention_student.cli --help

Usage
=====

The main focus of the repository is the ``MultiChannelAttentionStudent`` class which is a
keras model built on the `kgcnn`_ library, which can be used for graph learning.

The model can be used like this:

.. code-block:: python

    import tensorflow as tf
    import tensorflow.keras as ks
    from graph_attention_student.models import MultiChannelAttentionStudent

    model = MultiChannelAttentionStudent(
        units=[10, 7, 5],
        # Example for a 3-class classification problem. Regression also possible with some additions
        importance_channels=3,
        final_activation='softmax'
    )

    # The model output is actually a three tuple: (prediction, node_importances, edge_importances).
    # This allows the importances to be trained in a supervised fashion. If we don't want that, we can simply
    # supply the NoLoss function instead.
    model.compile(
        loss=[ks.losses.CategoricalCrossentropy(), NoLoss(), NoLoss()],
        loss_weights=[1, 0, 0],
        optimizer=ks.optimizers.Adam(0.01)
    )

    # model.fit() ...

Check out `examples/solubility_regression.py`_ for a detailed example of how the model can be used for a
regression task.

.. _kgcnn: https://github.com/aimat-lab/gcnn_keras
.. _examples/solubility_regression.py: https://github.com/aimat-lab/graph_attention_student/tree/master/graph_attention_student/examples/solubility_regression.py

Main Idea
=========

The main idea of this model is to provide a self-explaining graph neural network which employs multiple
explanation channels, instead of just one as it is commonly done. It is possible to construct the network
with a given amount of explanation channels. In the basic form each channel will then contribute one value
to the main prediction of the network. So for a 3-class classification problem, there should be 3
explanation channels where each channel produces the probability value for one respective class. Regression
tasks are also possible by adding an additional layer on top of that. Aside from the main prediction, the
network also returns a tensor of ``node_importances`` and ``edge_importances``, which assign each node of
the graph as many importance values as there are explanation channels. So basically each channel produces
it's own node / edge importance mask, which consists of values between 0 and 1. These values indicate how
important the corresponding node / edge was for the outcome of that particular channel.

By also producing these node and edge importance tensors as outputs, it is possible to train them in a
supervised manner. This has been primarily used to implement the student teacher analysis, which determines
the quality of explanations.

Architecturally, the core of the network consists of multiple `GATv2`_ layers. Each layer consists of as many
attention heads as there are explanation channels. Each head maintains it's own set of edge attention
coefficients. These attention coefficients are reduced along the number of layers to obtain the edge
importances. The node importances are produced by an additional dense layer acting on the final node
embeddings which is produced by the final GAT layer. The final node embeddings are then globally pooled into
graph embeddings, on top of which each explanation channel defines it's own dense output network to produce
the final prediction value.


.. _`GATv2`: https://github.com/tech-srl/how_attentive_are_gats