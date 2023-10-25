"""
This example contains the code from the "Quickstart" section of the README.rst file.
This code illustrates how to load a MEGAN model and use it to perform a prediction 
for a molecular graph given in SMILES format
"""
import os
import typing as t

from visual_graph_datasets.util import dynamic_import
from graph_attention_student.utils import ASSETS_PATH
from graph_attention_student.models import load_model

# We want to predict the water solubility for the molecule represented as this SMILES code
SMILES = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'

# Loading the model
model_path = os.path.join(ASSETS_PATH, 'models', 'aqsoldb')
model = load_model(model_path)

# For the inference we have to convert the SMILES string into the proper molecular graph
module = dynamic_import(os.path.join(model_path, 'process.py'))
processing = module.processing
graph = processing.process(SMILES)

# THe model outputs the node and edge explanation masks directly alongside the main target value prediction
out_pred, ni_pred, ei_pred = model.predict_graphs([graph])[0]
print(f'Solubility: {out_pred}')