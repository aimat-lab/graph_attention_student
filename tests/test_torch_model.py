import pytest
import os
import tempfile

import torch

from graph_attention_student.testing import set_environ
from graph_attention_student.torch.model import AbstractGraphModel


# Simple mock implementation of the AbstractGraphModel interface for the testing of the 
# general functionality of the base class.
class GraphModel(AbstractGraphModel):
    
    def __init__(self, units=10, **hparams: dict):
        AbstractGraphModel.__init__(self)
        
        self.hparams.update(hparams)
        self.lay = torch.nn.Linear(units, units)
        
    
class TestAbstractGraphModel:
    
    def test_saving_works(self):
        """
        AbstractGraphModel.save should save the model to the given path as a torch archive file.
        """
        with tempfile.TemporaryDirectory() as path:
            
            params = {'param1': 'test', 'param2': 10}
            model = GraphModel(**params)
            
            model_path = os.path.join(path, 'model.ckpt')
            model.save(model_path)
            
            assert os.path.exists(model_path)
            
    def test_saving_loading_works(self):
        """
        It should be possible to reconstruct a model in memory from a previously saved model file 
        using the AbstractGraphModel.load method with the model file path.
        """
        with tempfile.TemporaryDirectory() as path:
            
            params = {'param1': 'test', 'param2': 10}
            model = GraphModel(**params)
            
            # ~ saving the model
            model_path = os.path.join(path, 'model.ckpt')
            model.save(model_path)
            
            assert os.path.exists(model_path)
            
            # ~ loading the model
            model = GraphModel.load(model_path)
            
            assert isinstance(model, GraphModel)
            assert model.hparams == params
            
    def test_loading_incorrect_version_warning(self):
        """
        Whenever there is an error during the loading of the model and the version of the model
        does not match the version of the package then a warning should be raised which informs 
        the user about the mismatch.
        """
        with tempfile.TemporaryDirectory() as path:
            
            # We save the model with a non-standard number of units for the internal layer module 
            # so that this will cause an exception later on during the loading of the model.
            with set_environ({'GRAPH_ATTENTION_STUDENT_VERSION_OVERWRITE': '0.0.0'}):  
                params = {'param1': 'test', 'param2': 10}
                model = GraphModel(units=20, **params)
                model_path = os.path.join(path, 'model.ckpt')
                model.save(model_path)

            # when trying to load the model there should be an error now because there is 
            with pytest.raises(Exception) as exc:
                model = GraphModel.load(model_path)
                assert 'WARNING' in str(exc)
                assert 'version' in str(exc)