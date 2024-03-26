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

======================================
🧪 PyComex - Computational Experiments
======================================

Generally, all the experimentation code is based on the PyComex_ micro-framework. PyComex is a simple and lightweight 
framework that simplifies many aspects of developing, executing and managing computational experiments. This section 
introduces some of its core aspects which are necessary for the subsequent sections. 
For more detailed information, please visit the PyComex package: https://github.com/the16thpythonist/pycomex

In PyComex, each individual experiment should be self-contained as its own Python module. These modules can be executed 


====================================================================
⚗️ Converting SMILES based Datasets for Molecular Property Prediction
====================================================================

Since the application of MEGAN for molecular property prediction for a SMILES-based dataset is the most common use case, the 
following elaborations will use this as an example to demonstrate the required steps to train a custom MEGAN model.

Most often a dataset for molecular property prediction will be given as a CSV file which contains the representations of the 
source molecules in their SMILES string representation in one column, and the corresponding target values in another column.
The following example illustrates how such a source CSV file may look like:

.. code-block: csv

    smiles,logP
    CCO,0.2
    CCN,0.3
    CCC,0.5
    CC(=O)O,0.8
    CC(=O)N,0.7
    C1CC1,0.6
    ...



=======
❓ FAQs
=======

This section will answer some common questions that may arise during the process of training a custom MEGAN model.

Where is the actual code?
=========================

Pass


.. _Pycomex: https://github.com/the16thpythonist/pycomex/tree/master