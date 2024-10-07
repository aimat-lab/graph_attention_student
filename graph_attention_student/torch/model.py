import typing as t

import torch
import numpy as np
import visual_graph_datasets.typing as tv
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_attention_student.utils import get_version
from graph_attention_student.torch.data import data_list_from_graphs


class AbstractGraphModel(pl.LightningModule):
    """
    This is the abstract base class for implementing a graph property prediction model.
    
    **PREDICT GRAPHS**
    
    This abstract base class implements the method "predict_graphs" which is a convenience wrapper around the 
    function of the model. The method simply receives a list of B graphs in the GraphDict format and returns 
    a numpy array with the shape (B, O) where O is the number of model outputs, that assigns the models predictions 
    to each of the given input graphs in the same order.

    This function is implemented agnostic of the actual network implementation - so long as that implementation 
    conforms to the interface of the "forward" function that is defined by this abstract base class and which will
    be explained in the next section.
    
    **ARBITRARY FORWARD RETURNS**
    
    This abstract base class enforces a certain interface for the forward, which dictates that the forward method 
    receives a pytorch_geometric ``Data`` instance as input and returns a *dictionary* object. The keys of this 
    dictionary should be some string identifier and the values are supposed to be torch tensor objects that are the 
    result of the models forward pass. The one entry that this dict HAS TO contain is "graph_output" which is supposed 
    to be the (B * V, O). The entry with this key will be necessary to compute the main prediction loss.
    
    In addition to that, the returned dict may contain arbitrary additional tensors that may want to be returned as 
    side results of the prediction process, such as the intermediate graph embedding representations or explanation 
    masks.
    
    When adding additional tensors to be returned as part of the dictionary, the following naming scheme has to be used:
    - starting with the prefix "graph" for tensors of the shape (B, ...)
    - starting with the prefix "node" for tensors of the shape (B * V, ...)
    - starting with the prefix "edge" for tensors of the shape (B * E, ...)
    
    The class also implements the method "forward_graphs" which is a convenience method that can be used to obtain the 
    various model outputs for a set of graphs given as a list of GraphDict instances. The method takes a list of graph 
    dicts as an input and returns a list of corresponding output dicts. Each output dict will contain numpy arrays 
    for that particular graph with all attributes that conform to the previously described naming scheme!
    
    B - batch dimensions, number of graphs
    V - number of nodes
    E - number of edges
    O - output dimension, number of outputs that the network generates for each graph
    """
    
    # :attr BATCH_SIZE: 
    #       This is the default batch size that is being used in all the methods for the model INFERENCE
    #       tasks.
    BATCH_SIZE: int = 1_000
    
    def forward(self, data: Data) -> dict:
        """
        The forward model needs to be implemented by every specific child class implementation. This method 
        receives the ``torch_geometric.data.Data`` instance that describes a batch of graphs and is supposed 
        to return a dictionary containing various fields that provide information about the prediction of the 
        model.
        
        This dictionary has to contain the following fields:
        - graph_output: A tensor of the shape (B, O) which assigns the actual output prediction vector to each of the 
            input graphs.
        
        This dictionary *OPTIONALLY* may additionally contain the following fields:
        - node_importance: A tensor of node attributional explanations of the shape (B * V, K) 
        - edge_importance: A tensor of edge attributional explanations of the shape (B * E, K)
        """
        raise NotImplementedError('Please implement the "forward" method for the custom model')
    
    def forward_graphs(self,
                       graphs: t.List[tv.GraphDict],
                       batch_size: int = 1_000,
                       ) -> t.List[dict]:
        """
        Given a list ``graphs`` of graph dict objects, this method runs the forward pass of the model for 
        all of them. Returns a list of dictionaries that contain all the outputs of the forward method.
        
        The forward implementation of the model itself returns a dictionary with string keys and torch 
        tensor values, since each forward pass can produce multiple outputs.
        
        This forward computation is batched with the given ``batch_size``. This number of elements will 
        be converted into a batch tensor and processed by the network at the same time.
        
        :param graphs: A list of graph dictionary instances - each representing a single graph for which 
            the prediction should be generated.
        :param batch_size: The number of graphs to predict at the same time.
        
        :returns: A list of dictionaries. Each dict has several string keys which are descriptive names 
            of different outputs produced by the model. The values are numpy arrays that contain the 
            actual values of the outputs.
        """
        loader = self._loader_from_graphs(graphs, batch_size)
        
        # This will be the data structure that holds all the results of the inference process. Each element 
        # in this list will be a dictionary holding all the information for one graph in the given list of 
        # graphs - having the same order as that list.
        results: t.List[dict] = []
        for data in loader:
            # This is the actual size of the CURRENT batch. Usually this will be the same as the given batch 
            # size, but if the number of graphs is not exactly divisible by that number, it CAN be different!
            _batch_size = np.max(data.batch.numpy()) + 1
            
            # This line ultimately invokes the "forward" method of the class which returns a dictionary structure 
            # that contains all the various bits of information about the prediction process.
            info: dict = self.forward(data)
                
            for index in range(_batch_size):
                
                node_mask = (data.batch == index)
                edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
                
                result = {}
                for key, value in info.items():
                    
                    # Here we apply a bit of magic: Depending on the prefix of the string name of the attribute 
                    # we are going to dynamically treat the corresponding values (=tensors) differently, since 
                    # the names tell us in regard to what graph element those values are defined. For example for 
                    # "graph" global properties we do not have to do any different processing, we can simply get 
                    # the element with the correct index from the tensor. However, for node and edge based attributes 
                    # we first have to construct the appropriate access mask to aggregate the correct values from 
                    # the tensor.
                    
                    if key.startswith('graph'):
                        result[key] = value[index].detach().numpy()
                        
                    elif key.startswith('node'):
                        array = value[node_mask].detach().numpy()
                        result[key] = array
                        
                    elif key.startswith('edge'):
                        array = value[edge_mask].detach().numpy()
                        result[key] = array
                    
                results.append(result)
                
        return results
    
    def forward_graph(self,
                      graph: tv.GraphDict
                      ) -> dict:
        """
        Performs the forward pass of the model for a single ``graph`` dict instance. Returns the info 
        dict with all the model outputs.
        
        :returns: dict
        """
        return self.forward_graphs([graph], batch_size=1)[0]
    
    def predict_graphs(self, 
                       graphs: t.List[tv.GraphDict], 
                       batch_size: int = BATCH_SIZE,
                       ) -> np.ndarray:
        """
        Given a list ``graphs`` of B graph dicts, this method will return the numpy array of the 
        network's output predictions with the shape (B, O) where O is the output dimension.
        
        :param graphs: A list of graphs for which to generate the predictions.
        
        :returns: numpy array of shape (B, O)
        """
        results = self.forward_graphs(
            graphs=graphs,
            batch_size=batch_size,
        )
        
        predictions = [result['graph_output'] for result in results]
        return np.stack(predictions, axis=0)
    
    def predict_graph(self,
                      graph: tv.GraphDict
                      ) -> np.ndarray:
        """
        Given a single graph representation, this method will return the network's output prediction for 
        that graph.
        
        If there is more than one graph to be predicted, this method is NOT recommended. Instead, the 
        predict_graphs method should be used - which is able to use the performance gain of properly batching 
        the input graphs for the model inference
        
        :param graph: The graph for which to generate the prediction.
        
        :returns: numpy array of shape (output_dim, )
        """
        info = self.forward_graphs(
            graphs=[graph],
            batch_size=1,
        )[0]

        prediction: np.ndarray = info['graph_output']
        return prediction
    
    @classmethod
    def load(cls, path: str) -> 'AbstractGraphModel':
        """
        Loads the model from a persistent CKPT path at the given absolute ``path``. Returns the 
        reconstructed model instance.
        
        :returns: model instance
        """
        try:
            model = cls.load_from_checkpoint(path)
            model.eval()
            return model
            
        except Exception as exc:
            # Even if we can't load the model itself directly we can load the state dict and the hyperparameters.
            # One of the most common reasons for a problem with the model loading is that the model was exported 
            # with a prior version of the package and the current version has changed the model architecture in 
            # a backward-incompatible way. In this case, we give a meaningful error message to inform the user 
            # that downgrading the package might be required.
            info = torch.load(path)
            current_version = get_version()
            
            # The first possibility is that the exported model doesnt even contain the model version information 
            # since that was only added at a later version as well. So we first check for that and give a slightly 
            # more generic error message.
            if 'version' not in info:
                message = (
                    f'EXCEPTION: {str(exc)}\n\n'
                    f'WARNING: The package version ({current_version}) does not match the older model '
                    f'version that was used to create the model. This is most likely the source of the '
                    f'problem. You might try to downgrade the model version.'
                )
                raise Exception(message) from exc
            
            model_version = info['version']            

            # If there is a model version we can give a more detailed message where we inform the user about the
            # specific version with which the model was exported.
            if model_version != current_version:
                message = (
                    f'EXCEPTION: {str(exc)}\n\n'
                    f'WARNING: The package version ({current_version}) does not match the model '
                    f'version ({model_version}) that was used to create the model. This is most likely '
                    f'the source of the problem! To load the model please try to downgrade the package '
                    f'version accordingly: "pip install graph-attention-student=={model_version}"'
                )
                raise Exception(message) from exc

            raise exc
    
    def save(self, path: str) -> None:
        """
        Saves the model as a persistent file to the disk at the given ``path``. The file will be a torch
        ckpt file which is in essence a zipped archive that contains the model's state dictionary and the 
        hyperparameters that were used to create the model. Based on this information, the model can later 
        be reconstructed and loaded.
        
        :param path: The absolute file path of the file to which the model should be saved.
        
        :returns: None
        """
        torch.save({
            'state_dict': self.state_dict(),
            'hyper_parameters': self.hparams,
            'pytorch-lightning_version': pl.__version__,
            # 31.08.24: Adding the version of the package to the saved model so that 
            #           during loading we can provide this information to the user in case 
            #           the loading process fails due to a version mismatch...
            'version': get_version(),
        }, path)
    
    def _loader_from_graphs(self,
                            graphs: t.List[tv.GraphDict], 
                            batch_size: int
                            ) -> DataLoader:
        """
        Based on a list ``graphs`` of graph dict objects and a given ``batch_size``, this method constructs 
        a ``torch_geometric.loader.DataLoader`` instance which can be used to make batched predictions for 
        the model.
        
        :param graphs: A list of the graph dict instances to use for a model inference
        :param batch_size: The number of graphs to predict at the same time.
        
        :returns: a DataLoader instance.
        """
        data_list = data_list_from_graphs(graphs)
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        return loader