"""
This example contains the code from the "Quickstart" section of the README.rst file.
This code illustrates how to load a MEGAN model and use it to perform a prediction 
for a molecular graph given in SMILES format
"""
import os
import numpy as np

from graph_attention_student.utils import EXAMPLES_PATH, load_processing
from graph_attention_student.torch.megan import Megan

# We want to predict the water solubility for the molecule represented as this SMILES code
SMILES = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'

MODEL_PATH = os.path.join(EXAMPLES_PATH, 'assets', 'aqsoldb.ckpt')
model = Megan.load(MODEL_PATH)

# For the inference we have to convert the SMILES string into the proper molecular graph
PROCESSING_PATH = os.path.join(EXAMPLES_PATH, 'assets', 'aqsoldb_process.py')
processing = load_processing(PROCESSING_PATH)

# THe model outputs the node and edge explanation masks directly alongside the main target value prediction
graph = processing.process(SMILES)
info: dict[str, np.ndarray] = model.forward_graph(graph)

print(f'\npredicted water solubility: {info["graph_output"][0]:.2f}')