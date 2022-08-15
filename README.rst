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
    from graph_attention_student.training import NoLoss
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

Network Architecture
====================

.. image:: ./architecture.png
    :width: 800
    :alt: Architecture Overview

The main idea of this model is to provide a self-explaining graph neural network which employs multiple
explanation channels, instead of just one as it is commonly done. It is possible to construct the network
with a predefined amount of explanation channels. This is mainly meant for multi-class graph classification
problems. The idea is to have as many explanation channels as there are classes, so that every class
has *it's own explanation*.

Aside from the main prediction, the
network also returns a tensor of ``node_importances`` and ``edge_importances``, which assign each node of
the graph as many importance values as there are explanation channels. So basically each channel produces
it's own node / edge importance mask, which consists of values between 0 and 1. These values indicate how
important the corresponding node / edge was for the outcome of that particular channel.

Architecturally, the core of the network consists of multiple `GATv2`_ layers. Each layer consists of as many
attention heads as there are explanation channels. Each head maintains it's own set of edge attention
coefficients. These attention coefficients are reduced along the number of layers to obtain the edge
importances. The node importances are produced by an additional dense layer acting on the final node
embeddings which is produced by the final GAT layer. The final node embeddings are then globally pooled into
graph embeddings. Actually there will be as many graph embedding vectors as there are explanation channels:
The final node embeddings are weighted-pooled with each separate channel's ``node_importances``. All those
graph embeddings are then concat together and passed into a final network of dense layers to produce the
final prediction target.

.. note::

    Aside from the actual prediction, the network returns the tensor of all ``node_importances``and all
    ``edge_importances``. These explanatory importance values are produced by fully differentiable paths,
    which means that it is also possible to train the network to imitate a dataset of existing explanations,
    by adding additional explanation-supervising loss terms.

.. _`GATv2`: https://github.com/tech-srl/how_attentive_are_gats

Examples
========

The following examples show some results achieved with the network.

RB-Motifs Dataset
-----------------

This is a synthetic dataset, which basically consists of randomly generated graphs with nodes of different
colors. Some of the graphs contain special sub-graph motifs, which are either blue-heavy or red-heavy
structures. The blue-heavy sub-graphs contribute a certain negative value to the overall value of the graph,
while red-heavy structures contain a certain positive value.

This way, every graph has a certain value associated with it, which is between -5 and 5. The network was
trained to predict this value for each graph.

This shows the explanations for an example prediction of the network. For the regression task, the left
channel explains low values while the right channel explains high values. The network correctly identified
one of the special negative motif to be a chain of 4 blue nodes and one of the special positive motifs to
be a triangle of 2 red nodes and 1 green node.

.. image:: rb_motifs_example.png
    :width: 800
    :alt: Rb-Motifs Example

Aquaous Solubility Dataset
--------------------------

Another...

.. image:: solubility_example.png


