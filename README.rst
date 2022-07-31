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

     graph_attention_student --version
    graph_attention_student --help

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

.. _examples/solubility_regression.py: https://github.com/aimat-lab/graph_attention_student/tree/master/graph_attention_student/examples/solubility_regression.py

