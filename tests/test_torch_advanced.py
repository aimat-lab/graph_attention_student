import os
import tempfile

import weasyprint as wp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.data import VisualGraphDatasetReader
from graph_attention_student.testing import model_from_processing
from graph_attention_student.torch.advanced import explain_graph_split
from graph_attention_student.torch.advanced import explain_graph_joint
from graph_attention_student.torch.advanced import megan_prediction_report
from graph_attention_student.torch.advanced import explain_value

from .util import ARTIFACTS_PATH


def test_explain_value_basically_works():
    """
    The ``explain_value`` function should directly visualize the importances of the different explanation
    channels given a domain specific string representatation value, a model and a processing instance.
    """
    processing = MoleculeProcessing()
    model = model_from_processing(
        processing=processing,
        num_outputs=1,
        num_channels=2,
        prediction_mode='regression',
    )
    
    fig, axs, info, graph = explain_value(
        value='C1=CC=CC=C1CCN',
        model=model,
        processing=processing,
        strategy='joint',
    )
    
    # saving the figure for visual inspection
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_explain_value_basically_works.pdf')
    fig.savefig(fig_path)


def test_explain_graph_joint_basically_works():
    """
    The ``explain_graph_joint`` function should directly visualize the importances of the different
    explanation channels given a graph and a model in a single plot using different color maps to 
    indicate the different explanation channels.
    """
    processing = MoleculeProcessing()
    model = model_from_processing(
        processing=processing,
        num_outputs=1,
        num_channels=2,
        prediction_mode='regression',
    )
    
    with tempfile.TemporaryDirectory() as path:
        
        # first of all we need to create a new visual graph element (consisting of the graph itself
        # and the visualization image)
        processing.create(
            value='C1=CC=CC=C1CCN',
            index='0',
            output_path=path,
        )
        data = VisualGraphDatasetReader.read_element(path=path, name='0')
        
        fig, axs, info = explain_graph_joint(
            graph=data['metadata']['graph'],
            image_path=data['image_path'],
            model=model,
        )
        
        # first return is the overall figure object
        assert isinstance(fig, plt.Figure)
        # second is a list of all the individual axes objects which are part of that figure
        # in the order of the explanation channels
        assert isinstance(axs, list)
        assert len(axs) == 1
        # last is the info dictionary which contains the results of the model forward pass
        assert isinstance(info, dict)
        assert 'graph_output' in info
    
    # Save the figure for visual inspection
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_explain_graph_joint_basically_works.pdf')
    fig.savefig(fig_path)
    
    
def test_explain_graph_joint_with_custom_color_maps():
    """
    The ``explain_graph_joint`` function should also work when passing in color maps objects instead 
    of the names of the color maps.
    """
    processing = MoleculeProcessing()
    model = model_from_processing(
        processing=processing,
        num_outputs=1,
        num_channels=2,
        prediction_mode='regression',
    )
    
    with tempfile.TemporaryDirectory() as path:
        
        # first of all we need to create a new visual graph element (consisting of the graph itself
        # and the visualization image)
        processing.create(
            value='C1=CC=CC=C1CCN',
            index='0',
            output_path=path,
        )
        data = VisualGraphDatasetReader.read_element(path=path, name='0')
        
        # custom color maps
        cmap_1 = LinearSegmentedColormap.from_list('cmap_1', ['red', 'green'])
        cmap_2 = LinearSegmentedColormap.from_list('cmap_2', ['blue', 'yellow'])
        
        fig, axs, info = explain_graph_joint(
            graph=data['metadata']['graph'],
            image_path=data['image_path'],
            model=model,
            color_maps=[cmap_1, cmap_2]
        )
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(axs, list)
        assert len(axs) == 1
        
    # save the figure for visual inspection
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_explain_graph_joint_with_custom_color_maps.pdf')
    fig.savefig(fig_path)
    


def test_explain_graph_split_basically_works():
    """
    ``explain_graph_split`` should directly visualize the importances of the different explanation 
    channels given a graph and a model into seperate plots in the same figure.
    """
    processing = MoleculeProcessing()
    model = model_from_processing(
        processing=processing,
        num_outputs=1,
        num_channels=2,
        prediction_mode='regression',
    )
    
    with tempfile.TemporaryDirectory() as path:
        
        # first of all we need to create a new visual graph element (consisting of the graph itself
        # and the visualization image)
        processing.create(
            value='CCCC=CCCN',
            index='0',
            output_path=path,
        )
        data = VisualGraphDatasetReader.read_element(path=path, name='0')
                
        fig, axs, info = explain_graph_split(
            graph=data['metadata']['graph'],
            image_path=data['image_path'],
            model=model,
        )
        
        # first return is the overall figure object
        assert isinstance(fig, plt.Figure)
        # second is a list of all the individual axes objects which are part of that figure
        # in the order of the explanation channels
        assert isinstance(axs, list)
        assert len(axs) == model.num_channels
        # last is the info dictionary which contains the results of the model forward pass
        assert isinstance(info, dict)
        assert 'graph_output' in info
    
    # Save the figure for visual inspection
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_explain_graph_split_basically_works.pdf')
    fig.savefig(fig_path)
    
    
def test_megan_prediction_report_basically_works():
    """
    The ``megan_prediction_report`` function should create a html report for MEGAN predictions.
    """
    processing = MoleculeProcessing()
    model = model_from_processing(
        processing=processing,
        num_outputs=1,
        num_channels=2,
        prediction_mode='regression',
    )
    
    # saving the png file for visual inspection
    pdf_path = os.path.join(ARTIFACTS_PATH, 'test_megan_prediction_report_basically_works.pdf')
    megan_prediction_report(
        value='C1=CC=CC=C1CCN',
        model=model,
        processing=processing,
        output_path=pdf_path,
    )