import os
import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import pytorch_lightning as pl
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, MaxAggregation

import graph_attention_student.typing as tv
from graph_attention_student.utils import get_version
from graph_attention_student.torch.utils import torch_gauss
from graph_attention_student.torch.model import AbstractGraphModel
from graph_attention_student.torch.layers import (
    MessagePassing,
    MaskExpansionLayer
)

NAME_ACTIVATION_MAP: t.Dict[str, t.Callable] = {
    None:           nn.Identity,
    'linear':       nn.Identity,
    'relu':         nn.ReLU,
    'silu':         nn.SiLU,
    'leaky_relu':   nn.LeakyReLU,
    'tanh':         nn.Tanh,
}


class AbstractAttentionLayer(MessagePassing):
    """
    This is the abstract base class that can be used to create custom (MEGAN-compatible) graph attention layers.
    The defining property of such a graph attention layer is that it uses some sort of attention mechanism during 
    the message passing - so that the layer produces a set of attention weights for EDGE in the graph. 
    The "forward" method of this class is therefore expected to not just return the next node feature vector, 
    but instead to return a tuple (node_embedding, edge_attention_logits) where edge_attention_logits is a (E, 1) 
    that contains one attention LOGIT value for each edge.
    
    :param in_dim: the integer number of features in the input node feature vector
    :param out_dim: the integer number of features that the output node feature vector is supposed to have
    :param edge_dim: the integer number of features that the edge feature vector has
    """
    def __init__(self, 
                 in_dim: int,
                 out_dim: int,
                 edge_dim: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
    
    def forward(self,
                **kwargs
                ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        This method has to return a tuple (node_embedding, edge_attention_logit) where node_embedding is 
        an array of the shape (B * V, out) and edge_attention_logit is an array of shape (B * V, 1) which 
        assigns an edge attention LOGIT value to each of the graph edges.
        """
        raise NotImplementedError('A graph attention layer must implement the "forward" method '
                                  'to return a tuple of the new node attributes and edge attentions')
    

class MultiHeadAttention(nn.Module):
    
    AGGREGATION_MAP: t.Dict[str, t.Callable] = {
        'sum':      torch.sum,
        'mean':     torch.mean,
        'max':      torch.amax,
        'min':      torch.amin,   
    }
    
    """
    This class can be used to join multiple implementations of the ``AbstractAttentionLayer`` interface together 
    into a single multi-head prediction layer.
    
    An instance can be instantiated from a list of AbstractAttentionLayer instances which should be combined into
    parallel prediction heads. Depending on the specific aggregation function, the individual node embeddings of 
    these layers are aggregated into a single node embedding vector which is then returned by the forward method.
    This obviously means that all the input and output dimensions of the individual attention layers that are 
    given to this class have to be the same.
    
    Besides returning the node embedding array, the forward method returns an ``edge_attention_logit`` tensor of 
    the shape (B * E, K) which contains the K attention LOGIT values for each edge in the graphs and where K is 
    the number of individual attention layers used to construct the multi-head.
    
    B - batch dimension
    V - number of nodes
    E - number of edges
    K - number of prediction heads
    """
    def __init__(self,
                 layers: t.List[AbstractAttentionLayer],
                 activation: str = 'leaky_relu',
                 aggregation: str = 'sum',
                 residual: bool = True,
                 ):
        super().__init__()
        
        assert aggregation in self.AGGREGATION_MAP, (
            f'The given aggregation "{aggregation}" is not one of the valid keywords {list(self.AGGREGATION_MAP.keys())}'
        )
        
        self.layers = nn.ModuleList(layers)
        
        self.activation = activation
        self.lay_act = NAME_ACTIVATION_MAP[activation]()
        
        self.aggregation = aggregation
        self.aggregate = self.AGGREGATION_MAP[aggregation]
        
        self.residual = residual
        
    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_index: torch.Tensor,
                **kwargs,
                ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the previous node embedding tensor ``x``, the edge embedding tensor ``edge_attr`` and the 
        tensor of edge indices ``edge_index``, this method will perform one round of attention message 
        passing and return the tuple (node_embedding, edge_attention_logit).
        
        ``node_embedding`` is the tensor of new node embeddings with the shape (B * V, out)
        
        ``edge_attention_logit`` is the tensor of shape (B * E, K) which assigns K attention logits 
        to each edge of the graphs - where K is the number of individual attention heads.
        
        :param x: The previous node embedding tensor (B * E, in)
        :param edge_attr: The edge embeddign tensor (B * E, edge)
        :param edge_index: The tensor of edge indices (B * E, 2)
        
        :returns: Tuple (node_embedding, edge_attention_logit)
        """
        # In these two lists we will store the results of the individual attention heads outputs.
        # Each head will produce the transformed node features vector and the edge attention weights.
        node_embeddings: t.List[torch.Tensor] = []
        alphas: t.List[torch.Tensor] = []
        
        for lay in self.layers:
            # node_embedding: (B * V, out)
            # alpha: (B * E, 1)
            node_embedding, alpha = lay(x, edge_attr, edge_index)
            #node_embedding = self.lay_act(node_embedding)
            
            node_embeddings.append(node_embedding)
            alphas.append(alpha)
    
        # node_embeddings: (B * V, out, K)
        node_embeddings = torch.stack(node_embeddings, dim=-1)
        
        # node_embeddings: (B * V, out)
        node_embeddings = self.aggregate(node_embeddings, dim=-1)
        
        if self.residual:
            node_embeddings = node_embeddings + x
        
        # alphas: (B * E, K)
        alphas = torch.cat(alphas, dim=-1)
    
        return node_embeddings, alphas
    
    
class ParallelHeadAttention(nn.Module):
    
    def __init__(self,
                 layers: t.List[AbstractAttentionLayer],
                 activation: str = 'leaky_relu',
                 residual: bool = True,
                 ):
        super().__init__()
        
        self.layers = nn.ModuleList(layers)
        
        self.activation = activation
        self.lay_act = NAME_ACTIVATION_MAP[activation]()
        
        self.residual = residual
        
    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_index: torch.Tensor,
                **kwargs,
                ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the previous node embedding tensor ``x``, the edge embedding tensor ``edge_attr`` and the 
        tensor of edge indices ``edge_index``, this method will perform one round of attention message 
        passing and return the tuple (node_embedding, edge_attention_logit).
        
        ``node_embedding`` is the tensor of new node embeddings with the shape (B * V, out)
        
        ``edge_attention_logit`` is the tensor of shape (B * E, K) which assigns K attention logits 
        to each edge of the graphs - where K is the number of individual attention heads.
        
        :param x: The previous node embedding tensor (B * E, in)
        :param edge_attr: The edge embeddign tensor (B * E, edge)
        :param edge_index: The tensor of edge indices (B * E, 2)
        
        :returns: Tuple (node_embedding, edge_attention_logit)
        """
        # In these two lists we will store the results of the individual attention heads outputs.
        # Each head will produce the transformed node features vector and the edge attention weights.
        node_embeddings: t.List[torch.Tensor] = []
        alphas: t.List[torch.Tensor] = []
        
        for channel_index, lay in enumerate(self.layers):
            
            # node_embeddings: (B * V, num_features_out)
            node_embedding = x[:, :, channel_index]
            # alpha: (B * E, 1)
            node_embedding, alpha = lay(node_embedding, edge_attr, edge_index)
            
            node_embeddings.append(node_embedding)
            alphas.append(alpha)
    
        # node_embeddings: (B * V, num_features_out, num_channels)
        node_embeddings = torch.stack(node_embeddings, dim=-1)
        
        # alphas: (B * E, num_channels)
        alphas = torch.cat(alphas, dim=-1)
    
        return node_embeddings, alphas


class GraphAttentionLayer(AbstractAttentionLayer):
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 edge_dim: int,
        ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            edge_dim=edge_dim,
            aggr='sum',
        )
        
        self.lay_linear = nn.Linear(
            in_features=in_dim,
            out_features=out_dim,
        )
        
        self.lay_attention = nn.Linear(
            in_features=(2 * in_dim) + edge_dim,
            out_features=1,
        )
        
        self.lay_act = nn.LeakyReLU()
        
        self.lay_transform = nn.Linear(
            in_features=out_dim+in_dim,
            out_features=out_dim,
        )
        
        # We will use this instance property to transport the attention weights from the class' 
        # "message" method to the "forward" method. So during the "message" method we will actually
        # assign a tensor value to this attribute.
        self._attention: t.Optional[torch.Tensor] = None
        self._attention_logits: t.Optional[torch.Tensor] = None
    
    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_index: torch.Tensor,
                **kwargs):
        
        # At first we apply a linear transformation on the input node features themselves so that they 
        # have the dimension which will later also be the output dimension.
        # node_embedding: (B * V, out)
        # node_embedding = self.lay_linear(x)
        
        self._attention = None
        self._attention_logits = None
        # node_embedding: (B * V, out)
        node_embedding = self.propagate(
            edge_index,
            x=x, 
            edge_attr=edge_attr
        )
        
        node_embedding = self.lay_act(node_embedding)
        node_embedding = self.lay_transform(torch.cat([node_embedding, x], axis=-1))
        
        return node_embedding, self._attention_logits
    
    def message(self,
                x_i, x_j,
                edge_attr,
                ) -> torch.Tensor:
        
        # message: (B * E, 2*out + edge)
        message = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        
        # _attention: (B * E, 1)
        self._attention_logits = self.lay_attention(message)
        self._attention = F.sigmoid(self._attention_logits)
        
        #return self._attention * self.lay_message(message)
        return self._attention * self.lay_linear(x_j)


class SumNeighbors(MessagePassing):
    
    def __init__(self,
                 aggr: str = 'sum',
                 **kwargs,
                 ):
        super().__init__(aggr=aggr, **kwargs)
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        
        return self.propagate(
            edge_index,
            x=x,
        )
    
    def message(self,
                x_j: torch.Tensor,
                ) -> torch.Tensor:
        return x_j


class GraphAttentionLayerV2(AbstractAttentionLayer):
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 edge_dim: int,
                 hidden_dim: int = 128,
                 bn_momentum: float = 0.1,
                 ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            edge_dim=edge_dim,
            aggr='sum',
        )
        self.message_dim = (3 * in_dim) + edge_dim
        
        self.lay_message = nn.Sequential(
            nn.Linear(in_features=self.message_dim, 
                      out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=bn_momentum),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim,
                      out_features=out_dim),
            nn.BatchNorm1d(out_dim, momentum=bn_momentum),
            nn.LeakyReLU(),
        )
        
        self.lay_attention = nn.Sequential(
            nn.Linear(in_features=self.message_dim, 
                      out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=bn_momentum),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim,
                      out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=bn_momentum),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim,
                      out_features=1),
        )
        
        self.lay_act = nn.LeakyReLU()
        
        self.lay_transform = nn.Sequential(
            nn.Linear(in_features=out_dim+in_dim,
                      out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=bn_momentum),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim,
                      out_features=out_dim),
            nn.BatchNorm1d(out_dim, momentum=bn_momentum),
            nn.LeakyReLU(),
        )
        
        self.lay_neighbors = SumNeighbors()
        
        # We will use this instance property to transport the attention weights from the class' 
        # "message" method to the "forward" method. So during the "message" method we will actually
        # assign a tensor value to this attribute.
        self._attention: t.Optional[torch.Tensor] = None
        self._attention_logits: t.Optional[torch.Tensor] = None
    
    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_index: torch.Tensor,
                **kwargs):
        
        # At first we apply a linear transformation on the input node features themselves so that they 
        # have the dimension which will later also be the output dimension.
        # node_embedding: (B * V, out)
        # node_embedding = self.lay_linear(x)
        
        self._attention = None
        self._attention_logits = None
        
        x_neighbor = self.lay_neighbors(x, edge_index)
        
        # node_embedding: (B * V, out)
        node_embedding = self.propagate(
            edge_index,
            x=x,
            x_neighbor=x_neighbor,
            edge_attr=edge_attr
        )
        
        # node_embedding = self.lay_act(node_embedding)
        node_embedding = self.lay_transform(torch.cat([node_embedding, x], axis=-1))
        node_embedding += x
        
        return node_embedding, self._attention_logits
    
    def message(self,
                x_i, x_j,
                x_neighbor_i, x_neighbor_j,
                edge_attr,
                ) -> torch.Tensor:
        
        # message: (B * E, 2*out + edge)
        # message = torch.cat([x_i, x_j, x_conv_i, x_conv_j, edge_attr], dim=-1)
        #message = torch.cat([x_j, edge_attr], dim=-1)
        #message = torch.cat([x_i, x_j, edge_attr], dim=-1)
        message = torch.cat([x_i, x_j, x_neighbor_j, edge_attr], dim=-1)
        
        # _attention: (B * E, 1)
        self._attention_logits = self.lay_attention(message)
        self._attention = F.sigmoid(self._attention_logits)
        
        return self._attention * self.lay_message(message)


class GraphAttentionLayerV3(AbstractAttentionLayer):
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 edge_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 3,
                 ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            edge_dim=edge_dim,
            aggr='sum',
        )
        
        self.message_layers = nn.ModuleList()
        for n in range(num_heads):
            lay = nn.Sequential(
                nn.Linear(in_features=(2 * in_dim) + edge_dim,
                          out_features=hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(in_features=hidden_dim,
                          out_features=hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(in_features=hidden_dim,
                          out_features=out_dim),
                nn.BatchNorm1d(out_dim),
            )
            self.message_layers.append(lay)

        
        self.lay_attention = nn.Sequential(
            nn.Linear(in_features=(2 * in_dim) + edge_dim, 
                      out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim,
                      out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim,
                      out_features=1),
        )
        
        self.lay_act = nn.LeakyReLU()
        
        self.lay_transform = nn.Sequential(
            nn.Linear(in_features=out_dim +in_dim,
                      out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim,
                      out_features=out_dim),
            nn.BatchNorm1d(out_dim),  
        )
        
        # We will use this instance property to transport the attention weights from the class' 
        # "message" method to the "forward" method. So during the "message" method we will actually
        # assign a tensor value to this attribute.
        self._attention: t.Optional[torch.Tensor] = None
        self._attention_logits: t.Optional[torch.Tensor] = None
    
    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_index: torch.Tensor,
                **kwargs):
        
        # At first we apply a linear transformation on the input node features themselves so that they 
        # have the dimension which will later also be the output dimension.
        # node_embedding: (B * V, out)
        # node_embedding = self.lay_linear(x)
        
        self._attention = None
        self._attention_logits = None
        # node_embedding: (B * V, out)
        node_embedding = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr
        )
        
        node_embedding = self.lay_act(node_embedding)
        node_embedding = self.lay_transform(torch.cat([node_embedding, x], axis=-1))
        #node_embedding = self.lay_transform(torch.cat([node_embedding, x], axis=-1))
        node_embedding = F.tanh(node_embedding)
        
        return node_embedding, self._attention_logits
    
    def message(self,
                x_i, x_j,
                edge_attr,
                ) -> torch.Tensor:
        
        # message: (B * E, 2*out + edge)
        message = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # _attention: (B * E, 1)
        self._attention_logits = self.lay_attention(message)
        self._attention = F.sigmoid(self._attention_logits)
        
        results = []
        for lay in self.message_layers:
            result = lay(torch.cat([x_i, x_j, edge_attr], dim=-1))
            results.append(result)
        
        results = torch.stack(results, dim=-1)
        result = torch.mean(results, dim=-1)
        
        return self._attention * result


class Megan(AbstractGraphModel):
    """
    **SHAPE DOCUMENTATION**
    
    In the context of this class, the following abbreviations are used:
    
    - B: The number of graphs in one batche
    - V: The number of nodes in a graph
    - E: The number of edges in a graph
    - N_l: The dimensionality of the node feature vector after layer l
    - D: The dimensionality of the graph embedding of a single channel
    - M: The dimensionality of the edge feature vector
    - K: The number of importance channels
    
    :param node_dim: the integer number of features of the input node features vectors
    :param edge_dim: the integer number of features of the input edge feature vectors
    :param units: A list of integers, where each entry defines one layer in the attention-based graph
        encoder part of the network. The integer number defines the number of hidden units.
    :param importance_mode: A string identifier which defines what kind of task to be solved w.r.t 
        the model's explanation masks.
    :param importance_factor: A float factor that is multiplied with the explanation co-training loss
    :param num_channels: The integer number of explanation channels. For a regression task this has 
        to be ==2, for a classification task this has to equal to the number of possible classes.
    :param importance_units: A list of integers, where each entry defines one layer in the dense network 
        that predicts the node attention mask from the node embedding vector. The integer number 
        defines the number of hidden units.
    :param importance_offset: A float integer that influences the behavior of the explanation co-training
        procedure. In an approximative sense, this parameter influences how much of the graph will be 
        covered by the explanations. Lower values tend to produce more focused explanations, while higher 
        values may result in more expansive explanations.
    :param projection_units: A list of integers, where each entry defines one layer in the dense 
        networks that project the graph embedding into the graph projection. The integer number 
        defines the number of hidden units. The last integer in this list will determine the 
        dimensionality of the graph embeddings for each channel. Every explanation channel, will 
        have it's own projection network.
    :param sparsity_factor: A float factor that is multiplied with the sparsity regularization loss.
        The higher this factor the more the explanations will be promoted to be sparse (less nodes 
        being highlighted as important)
    :param final_units: A list of integers, where each entry defines one layer in the final prediction 
        MLP of the network. Each integer entry defines the hidden layers size of the corresponding layer.
        The last value in this list determines the output shape of the entire network and therefore has 
        to match the number of target values in the dataset.
    :param prediction_mode: A string identifier that determines what kind of task the network should 
        be used for - regression or classification. This determines the loss function that will be used 
        for the training of the prediction output.
    :param use_bias: A boolean flag of whether to use bias terms in the networks.
    :param learning_rate: A float value which determines the learning rate during the gradient descent 
        optimization of the network.
    """
    
    # :attr IMPORTANCE_MODES:
    #       This attribute defines the possible values for the "importance_mode" parameter of the model
    #       that parameter determines how the explanation co-training procedure is realized during the 
    #       models training_step.
    #       For None, the explanation co-training is disabled completely.
    IMPORTANCE_MODES = [None, 'regression', 'classification']
    # :attr PREDICTION_MODES:
    #       This attribute defines the possible values for the "prediction_mode" parameter of the model
    #       that determines whether the model is supposed to solve a regression or a classification task
    PREDICTION_MODES = ['regression', 'classification', 'bce']
    
    def __init__(self,
                 # encoder-related
                 node_dim: int = 3,
                 edge_dim: int = 1,
                 units: t.List[int] = [16, 16, 16],
                 hidden_units: int = 128,
                 encoder_dropout_rate: float = 0.0,
                 layer_version: t.Literal['v1', 'v2'] = 'v2',
                 # explanation-related
                 importance_units: t.List[int] = [16, ],
                 projection_units: t.List[int] = [],
                 num_channels: int = 2,
                 importance_mode: t.Optional[str] = None,
                 importance_factor: float = 0.0,
                 importance_offset: float = 0.1,
                 importance_target: t.Literal['node', 'edge'] = 'node',
                 regression_reference: float = 0.0,
                 regression_margin: float = 0.0,
                 sparsity_factor: float = 0.0,
                 attention_aggregation: t.Literal['sum', 'min', 'max'] = 'sum',
                 normalize_embedding: bool = True,
                 fidelity_factor: float = 0.0,
                 # contrastive representation related
                 contrastive_factor: float = 0.0,
                 contrastive_noise: float = 0.1,
                 contrastive_temp: float = 1.0,
                 contrastive_beta: float = 1.0,
                 contrastive_tau: float = 0.1,
                 contrastive_units: int = 2048,
                 # prediction-related
                 final_units: t.List[int] = [16, 1],
                 final_dropout_rate: float = 0.0,
                 prediction_mode: t.Literal['regression', 'classification', 'bce'] = 'regression',
                 prediction_factor: float = 1.0,
                 use_bias: bool = True,
                 # classification only
                 class_weights: t.Optional[list] = None,
                 output_norm: t.Optional[float] = None,
                 label_smoothing: float = 0.0,
                 # training-related
                 learning_rate: float = 1e-3,
                 bn_momentum: float = 0.5,
                 ):
        pl.LightningModule.__init__(self)
        
        # The last integer value in the list of final_units determines the output dimension of the network aka how 
        # many graph properties the network will predict at the same time.
        self.out_dim = final_units[-1]
        self.importance_factor = importance_factor
        
        # ~ validating the parameters
        # There are some parameters whose values we want to validate before starting to construct the 
        # the instance, because choosing the incorrect values for some parameters will lead to uninformative errors 
        # down the line.
        assert importance_mode in self.IMPORTANCE_MODES, f'importance_mode has to be one of {self.IMPORTANCE_MODES}'
        assert prediction_mode in self.PREDICTION_MODES, f'prediction_mode has to be one of {self.PREDICTION_MODES}'
        if self.importance_factor > 0 and prediction_mode == 'regression':
            assert num_channels == 2, 'for regression explanations, num_channels must be 2 (negative & positive)!'
        if self.importance_factor > 0 and prediction_mode == 'classification':
            assert num_channels == self.out_dim, 'for classification explanations, num_channels must be number of outputs!'
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.units = units
        self.hidden_units = hidden_units 
        self.use_bias = use_bias
        self.layer_version = layer_version
        
        self.importance_units = importance_units
        self.projection_units = projection_units
        self.importance_mode = importance_mode
        self.num_channels = num_channels
        self.importance_factor = importance_factor
        self.importance_offset = importance_offset
        self.importance_target = importance_target
        self.regression_margin = regression_margin
        self.sparsity_factor = sparsity_factor
        self.normalize_embedding = normalize_embedding
        self.attention_aggregation = attention_aggregation
        self.fidelity_factor = fidelity_factor
        
        self.contrastive_factor = contrastive_factor
        self.contrastive_noise = contrastive_noise
        self.contrastive_temp = contrastive_temp
        self.contrastive_beta = contrastive_beta
        self.contrastive_tau = contrastive_tau
        self.contrastive_units = contrastive_units
        
        self.prediction_mode = prediction_mode
        self.prediction_factor = prediction_factor
        self.final_untis = final_units
        self.output_norm = output_norm
        self.label_smoothing = label_smoothing
        
        self.learning_rate = learning_rate

        # The "hparams" attribute that a pl.LightningModule has anyways. This dict is used to store the hyper 
        # parameter configuration of the module so that it can be serialized and saved when creating a persistent 
        # model checkpoint file. When later loading that checkpoint file, these hyperparameters will then be used 
        # as the parameters of the constructor to re-construct the model object instance. So the string key names 
        # in this dict have to match the names of the constructor parameters.
        self.hparams.update({
            'node_dim':                 node_dim,
            'edge_dim':                 edge_dim,
            'units':                    units,
            'hidden_units':             hidden_units,
            'encoder_dropout_rate':     encoder_dropout_rate,
            'layer_version':            layer_version,
            'importance_units':         importance_units,
            'importance_offset':        importance_offset,
            'importance_target':        importance_target,
            'projection_units':         projection_units,
            'normalize_embedding':      normalize_embedding,
            'attention_aggregation':    attention_aggregation,
            'fidelity_factor':          fidelity_factor,
            'contrastive_factor':       contrastive_factor,
            'contrastive_noise':        contrastive_noise,
            'contrastive_temp':         contrastive_temp,
            'contrastive_beta':         contrastive_beta,
            'contrastive_tau':          contrastive_tau,
            'contrastive_units':        contrastive_units,
            'final_units':              final_units,  
            'final_dropout_rate':       final_dropout_rate,
            'num_channels':             num_channels,
            'prediction_mode':          prediction_mode,
            'regression_reference':     regression_reference,
            'regression_margin':        regression_margin,
            'label_smoothing':          label_smoothing,
            'class_weights':            class_weights,
            'output_norm':              output_norm,
        })

        # ~ Graph encoder layers
        # The following section sets up all the layers for the graph encoder part of the network. This part 
        # consists of multiple multi-head attention layers that at the end produce the final node embedding 
        # representation
        
        # This layer will be applied to node input representations to embedd them into a higher dimensional 
        # space before passing that node embedding vector to the graph encoder layer for the actual message 
        # passing.
        self.lay_embedd = nn.Linear(
            in_features=node_dim,
            out_features=units[0]
        )
        
        # ~ explanation approximation layers
        
        if self.importance_target == 'node':
            in_features_ = node_dim
        elif self.importance_target == 'edge':
            in_features_ = 2 * node_dim + edge_dim
            
        self.lay_transform_1 = nn.Linear(
            in_features=in_features_,
            out_features=16,
            bias=True,
        )
        self.lay_transform_2 = nn.Linear(
            in_features=16,
            out_features=1,
            bias=False,
        )
        
        self.encoder_layers = nn.ModuleList()
        prev_features = units[0]
        # These are the activations that will be used on the result of the muli-head encoding. The important 
        # thing here is that we use a linear / identity activation for the very last layer since we dont want 
        # to restrict the expressiveness of the values which will ultimately become the graph embedding.
        activations = ['leaky_relu' for _ in units]
        activations[-1] = 'linear'
        for num_features, act in zip(units, activations):
            # Each layer in the encoder is a multi-head attention layer, where the number of parallel heads is 
            # also defined as the number of explanation "channels". The idea is that each of these channels captures 
            # different explanations - according to their pre-defined behavior.

            # 04.07.2024
            # Added the switch condition for the layer version. v2 is currently the newest version which contains 
            # some improvements over the v1 version, but we still want to keep the option to use the v1 version for 
            # compatibility.

            if layer_version == 'v1':
                layer_func = lambda: GraphAttentionLayer(
                    in_dim=prev_features,
                    out_dim=num_features,
                    edge_dim=edge_dim,
                )
                
            elif layer_version == 'v2':
                layer_func = lambda: GraphAttentionLayerV2(
                    in_dim=prev_features,
                    out_dim=num_features,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_units,
                    bn_momentum=bn_momentum,
                )
                
            elif layer_version == 'v3':
                layer_func = lambda: GraphAttentionLayerV3(
                    in_dim=prev_features,
                    out_dim=num_features,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_units,
                    num_heads=3,
                )
                
            lay = ParallelHeadAttention(
                [layer_func() for _ in range(num_channels)],
                activation=act,
            )
            
            prev_features = num_features
            self.encoder_layers.append(lay)
            
        # The last number of units in the graph encoder part determines the embedding dimension
        self.embedding_dim = prev_features
        
        # 13.04.24
        # Added the option to apply dropout to the message passing part of the encoder. This dropout will be 
        # applied to the node embeddings after each layer of the encoder (only during training). This dropout
        # will mask certain elements of these node embeddings from being propagated further through the network 
        # therefore promoting the network to learn more robust representations.
        self.lay_dropout_encoder = nn.Dropout(p=encoder_dropout_rate)
            
        # At the end of the graph encoder, the node embeddings for all the nodes are aggregated into one final 
        # graph embedding vector by doing an attention-weighted sum.
        self.lay_pool = SumAggregation()
        self.lay_pool_mean = MeanAggregation()
        self.lay_pool_edge = MaxAggregation()
        self.lay_pool_importance = SumAggregation()
            
        # ~ Importance/Attention Layers
        # The attention values for the final attention-weighted aggregation of the nodes is calculated from 
        # the node embeddings using an MLP. The layers of which are set up in the following section
        
        self.importance_layers = nn.ModuleList()
        prev_features = self.embedding_dim
        for num_features in importance_units:
            lay = nn.Linear(
                in_features=prev_features,
                out_features=num_features,
            )
            prev_features = num_features
            self.importance_layers.append(lay)
            
        # The last layer of the importance MLP has to match the number of importance channels since 
        # we want to generate one attention weight for each of channel.
        self.importance_layers.append(nn.Linear(
            in_features=prev_features,
            out_features=num_channels,
        ))
        
        # ~ Graph embedding projection
        # After the graph embeddigns were created as an attention-weighted sum of the node embeddings, 
        # that graph representation is additionally subjected to a projection MLP. The important 
        # part in this 
        
        self.channel_projection_layers = nn.ModuleList()
        for k in range(num_channels):
            
            layers = nn.ModuleList()
            prev_features = self.embedding_dim
            for num_features in projection_units:
                
                lay = nn.Sequential(
                    nn.Linear(in_features=prev_features,
                              out_features=num_features),
                    #nn.BatchNorm1d(num_features, momentum=bn_momentum),
                )
                prev_features = num_features
                layers.append(lay)
            
            self.channel_projection_layers.append(layers)
            
        self.embedding_dim = prev_features
            
        # ~ Dense Prediction layers
        
        self.dense_layers = nn.ModuleList()
        prev_features = self.embedding_dim * num_channels
        # self.lay_dense_norm = nn.BatchNorm1d(prev_features, momentum=bn_momentum)
        for num_features in final_units:
            
            lay = nn.Linear(
                in_features=prev_features,
                out_features=num_features,
            )
                
            prev_features = num_features
            self.dense_layers.append(lay)
            
        # 17.04.2024
        # In this variable we store the dimension of the target vector which is ultimately predicted 
        # as the main output of the model. For a classification task, for example, this is the number 
        # of possible classes.
        self.target_dim: int = final_units[-1]
            
        self.lay_act = nn.ELU()
        
        self.lay_mask_expansion = MaskExpansionLayer()
        
        self.projection_layers = nn.ModuleList()
        for k in range(num_channels):
            lay = nn.Linear(
                in_features=self.embedding_dim,
                out_features=contrastive_units,
            )
            self.projection_layers.append(lay)
            
        self.lay_final_dropout = nn.Dropout(p=final_dropout_rate)
        
        # Note that the the final activation (aka output activation) does NOT change depending on the 
        # prediction mode being regression or classification - the output activation will be the identify 
        # (aka linear) function in either case. This is intuitively correct for the regression case, but
        # for the multi-class classification case one would expect the softmax function. However, the model 
        # will output the classification LOGITS and the loss function will apply the softmax function 
        # internally.
        # This is important to keep in mind because a forward pass of the model will return the classification
        # logits and not the class probabilities. If the class probabilities are needed, the softmax function
        # has to be applied manually after the forward pass.
        self.lay_act_final = nn.Identity()
                
        # ~ regression mean
        self.n_samples = torch.ones((1, ))
        self.running_mean = torch.zeros((final_units[-1], ))
        self.regression_reference = torch.nn.Parameter(
            torch.zeros((final_units[-1], )),
            requires_grad=False,
        )
        
        # ~ Training Loss
        # Since this model supports both regression and classification, the loss function will have to be 
        # setup accordingly.
        # For a regression task, we use the mean squared error (MSE) loss function.
        # For a classification task, we use the cross entropy loss function.
        
        # 02.07.2024
        # Changed the "reduction" of all the prediction losses to "none" and then applying the mean reduction
        # over the individual losses manually in the train_step to optionally support sample weights for 
        # the individual samples in the batch.
        
        if self.prediction_mode == 'regression':
            self.loss_pred = nn.MSELoss(
                reduction='none'
            )
            
        elif self.prediction_mode == 'classification':
            # 13.04.24
            # Added the "class_weights" parameter as an additional argument during the construction of the 
            # cross entropy loss. This parameter can be used to assign different weights to the different 
            # classes that may appear during training.
            self.loss_pred = nn.CrossEntropyLoss(
                # label_smoothing=label_smoothing,
                weight=torch.tensor(class_weights) if class_weights is not None else None,
                reduction='none',
            )
            
        elif self.prediction_mode == 'bce':
            # 14.06.24
            # In some edge cases we don't actually want / cant use a multi-class classification interpretion 
            # of the prediction task, but actually want to predict mutliple binary classification tasks at 
            # the same time (allow two outputs to be active).
            # In this case we can do BCE loss on every output independently.
            self.loss_pred = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(class_weights) if class_weights is not None else None,
                reduction='none',
            )
            
    def forward(self, 
                data: Data, 
                node_mask: t.Optional[torch.Tensor] = None,
                node_importance_overwrite: t.Optional[torch.Tensor] = None,
                node_feature_mask: t.Optional[torch.Tensor] = None,
                stop_importance_grad: bool = False,
                ) -> t.Dict[str, torch.Tensor]:
    
        node_input, edge_input, edge_index = data.x, data.edge_attr, data.edge_index
        
        node_input = torch.where(torch.isinf(node_input), torch.zeros_like(node_input), node_input) # workaround for infinity values
        
        # node_embedding: (B * V, num_features_0)
        node_embedding = self.lay_embedd(node_input)
        # node_embedding: (B * V, num_features_0, num_channels)
        node_embedding = torch.stack([node_embedding for _ in range(self.num_channels)], axis=-1)
        
        node_embeddings = []
        
        # In this list we are going to store all the edge attention tensors "alpha" for the 
        # individual layers.
        alphas: t.List[nn.Module] = []

        for lay in self.encoder_layers:
            
            # Each layer of the graph encoder part is supposed to be based on the ``AbstractAttentionLayer``
            # base class, which determines that it must return a tuple (node_embedding, alpha) where node_embedding
            # is the new node embedding vector that was created by the layer's main transformation and alpha is a 
            # vector of edge attention LOGITS that was used during this transformation.
            
            # node_embedding: (B * V, N)
            # alpha: (B * E, K)
            node_embedding, alpha = lay(
                x=node_embedding, 
                edge_index=edge_index,
                edge_attr=edge_input, 
                return_attention_weights=True
            )
            # applying the dropout to the node embedding after each layer
            # this will NOT change the shape of the embedding.
            node_embedding = self.lay_dropout_encoder(node_embedding)
            
            node_embeddings.append(node_embedding)
            alphas.append(alpha)
            
        # edge_importance: (B * E, K, L)
        edge_importance = torch.stack(alphas, dim=-1)
        edge_importance = F.sigmoid(edge_importance)
  
        # 03.04.2024
        # Previously the edge importance was aggregated by summing up the individual attention values
        # across the layers. However, I think this may not be the optimal way to aggregate the attention
        # for every possible dataset. Therefore added the option to configure the method in which the 
        # attention values are aggregated.
        # The question of whether there is one universally best way to aggregate the attention values
        # is still open. However, currently the "min" aggregation seems the most promising for generating 
        # the most sparse explanation masks and the lowest explanation loss.
        if self.attention_aggregation == 'sum':
            edge_importance = torch.sum(edge_importance, dim=-1)
        elif self.attention_aggregation == 'mean':
            edge_importance = torch.mean(edge_importance, dim=-1)
        elif self.attention_aggregation == 'max':
            edge_importance = torch.amax(edge_importance, dim=-1)
        elif self.attention_aggregation == 'min':
            edge_importance = torch.amin(edge_importance, dim=-1)
        
        # edge_importance: (B * E, K)
        # After the aggregation we have reduced the shape of this tensor by one dimension - the last 
        # dimension which previously was the number of message passing layers. No matter which aggregation
        # was used, the values are still technically attention LOGITS. To now calculate the actual attention
        # values we need to apply the sigmoid function to obtain values in the range of [0, 1].
        #edge_importance = F.sigmoid(edge_importance)
        #edge_importance = softmax(edge_importance, data)
        
        # edge_importance_pooled: (B * V, K)
        edge_importance_pooled = 0.5 * (
            self.lay_pool_edge(edge_importance, edge_index[0]) + 
            self.lay_pool_edge(edge_importance, edge_index[1])   
        )
        # edge_importance_pooled = self.lay_pool_edge(edge_importance, edge_index[1])
            
        # ~ importance masks / explanations
        node_importance = edge_importance_pooled
        
        # TODO
        if stop_importance_grad:
            node_importance = node_importance.detach()
            edge_importance = edge_importance.detach()
        
        if node_mask is not None:
            node_importance = node_importance * node_mask
            
        if node_importance_overwrite is not None:
            node_importance = node_importance_overwrite
            
        # ~ importance node masking
        # One key step in the MEGAN architecture ist that the attention weights are not simply the edge 
        # attention values during the message passing steps, but at the same time these attention values 
        # are also being used in the global aggregation step as the weights of the global weighted 
        # pooling operation. Specifically, we have one attention weight for every node and every channel 
        # which is a value between 0 and 1. In the global sum aggregation step that turns the individual 
        # node embeddigns into one single graph embedding, the node features of a node are multiplied by 
        # the according value. Therefore, if a value is small ~0 then that node DOES NOT contribute 
        # information towards the channel's graph embedding. In other words, each channel's graph embedding 
        # (approximately) only contains information about the highlighted subgraphs!
            
        # ~ normalization & thresholding, 09.07.24
        # The node importance is just the edge importance (attention values) mapped to the nodes via a 
        # mean message aggregation function. Previously, we just used this raw value to multiply with 
        # the node embeddings. This has the problem that the rest of the network can over time adapt 
        # the the overall scale of the importance values. So even if all the importance values are 
        # actually really low values on an absolute scale, the node embeddings can just generally become 
        # larger by that same factor which can lead to misleading interpretations.
        
        # Here we normalize the node importances w.r.t. to the highest node importance value found 
        # in each graph (across all channels). So after the normalization, the highest value is guaranteed 
        # to be 1.0 and all other values are lower than that but maintain its relative scale. We also 
        # apply a thresholding where all values below 0.25 are effectively set to zero. This helps 
        # to reduce the noise in the graph explanation.
         
        # We need the scatter functionality to find the maximum value across a graph!
        # max_values: (B, K)
        max_values = torch_scatter.scatter_max(node_importance, data.batch, dim=0)[0]
        # max_values: (B, )
        max_values = torch.amax(max_values, dim=-1, keepdim=True)
        max_values = torch.where(max_values < 0.01, torch.ones_like(max_values) * 1e9, max_values)
        # node_importance_norm: (B * V, K)
        node_importance_norm = node_importance / max_values[data.batch]
        # An important detail here is that we dont actually set the values below 0.25 to 0 but rather just 
        # scale them down by a certaion factor. This is important to maintain a proper gradient also 
        # through that computational branch!
        node_importance_norm = torch.where(node_importance_norm < 0.1, node_importance_norm * 0.01, node_importance_norm)
            
        # ~ graph embedding & projection
        # As previously introduced, the graph embedding is calculated as the result of an 
        # importanced-weighted sum of the node embeddings. This is done for each channel separately.
        # Therefore the result of the global aggregation process is a set of K different graph embedding 
        # vectors.
        # In a second step, these graph embeddings are individually *projected*. This means that the 
        # graph embedding for each channel is mapped into a (normally) higher dimensional space 
        # with a feed forward network. Each channel thereby has it's own projection network. This is 
        # done to allow the different channels to develop different representations and to further 
        # reduce the correlation between them (which exists because of the shared node embeddings). 
        # The final projected graph embedding is additionally L2-normalized aka projected onto a 
        # high-dimensional unit sphere. This is done to support the vector product as a similarity
        # measure between the graph embeddings.
            
        graph_embedding_channels = []
        for k, layers in enumerate(self.channel_projection_layers):
            # node_embedding: (B * V, D)
            node_embedding_ = node_embedding[:, :, k]
            #node_embedding_ = torch.sum(node_embedding, dim=-1)
            
            node_embedding_ = node_embedding_ * node_importance_norm[:, k].unsqueeze(-1)
            if node_feature_mask is not None:
                node_embedding_ *= node_feature_mask[:, k].unsqueeze(-1)
            
            graph_embedding_ = self.lay_pool(node_embedding_, data.batch)

            # The following layers implement the projection feed forward network for 
            # the individual channels.
            if len(layers) != 0:
                
                for lay in layers[:-1]:
                    graph_embedding_ = lay(graph_embedding_)
                    graph_embedding_ = self.lay_act(graph_embedding_)
                    
                    graph_embedding_ = self.lay_dropout_encoder(graph_embedding_)
            
                graph_embedding_ = layers[-1](graph_embedding_)
            
            if self.normalize_embedding:
                # F.normalize will apply a transformation on the embedding so that all the embedding 
                # vectors have a constant norm == 1. This makes it possible to use the vector product
                # directly as the cosine similarity measure between two vectors.
                graph_embedding_ = F.normalize(graph_embedding_)
                
            graph_embedding_channels.append(graph_embedding_)
            
        # graph_embedding: (B, D, K)
        graph_embedding = torch.stack(graph_embedding_channels, dim=-1)
        
        # output: (B, D * K)
        output = torch.cat(graph_embedding_channels, dim=-1)
        
        # ~ final prediction network
        # At the end of the projection step, the result is a set of K D-dimensional graph embedding 
        # vectors. These are then concatenated into one single vector of shape (B, D * K) and passed
        # as input to the final prediction feed-forward network, which will transform that vector 
        # to the final output shape.
        
        #output = self.lay_dense_norm(output)
        for lay in self.dense_layers[:-1]:
            output = lay(output)    
            output = self.lay_act(output)
            
            output = self.lay_final_dropout(output)
            
        # output: (B, O)
        output = self.dense_layers[-1](output)
        
        # We basically want to use the regression reference as a preset for the output 
        # bias of the prediction.
        if self.prediction_mode == 'regression':
            output += self.regression_reference
            
        # 16.04.24
        # For classification models there is the option to define an output normalization. In this 
        # case the output logits will be l2 normalized aka projected onto a unit sphere with the 
        # radius defined by "self.output_norm"
        # This type of logits normalization is supposed to address the problem of model overconfidence
        # by using this output norm, a model can (a) only reach a certain maxium confidence level and 
        # (b) the confidence level will be distributed more evenly across the classes since the class 
        # is now only determined by the angle of the logit vector and no longer by it's length.
        if self.output_norm:
            output = self.output_norm * F.normalize(output, dim=-1)
            # print(output)
            
        return {
            'graph_output': output,
            'graph_embedding': graph_embedding,
            'node_embedding': node_embedding,
            'node_importance': node_importance,
            'node_importance_norm': node_importance_norm,
            'edge_importance': edge_importance,
        }
        
    def training_step(self, data: Data, batch_idx):
        
        batch_size = np.max(data.batch.cpu().numpy()) + 1
        
        # out_pred: (B, O)
        out_true = data.y.view(-1, self.out_dim)
        
        # 07.08.24
        # ~ updating running mean
        # The most recent changes to the explanation training routine replaced the global use of the 
        # regression reference by just splitting each batch according to it's median locally. However, 
        # the regression reference is still important for the actual output of the model - if the model 
        # output is not symmetrically centered around the mean value, this can affect the fidelity calculation 
        # therefore we implement the regression reference as a running mean over all the batches here.
        
        if self.prediction_mode == 'regression':

            batch_size_ = torch.amax(data.batch) + 1
            batch_mean = torch.mean(out_true, dim=0)
            
            self.running_mean += (batch_mean - self.running_mean) * (batch_size_ / self.n_samples)
            self.n_samples += batch_size_
            
            self.regression_reference.data.copy_(self.running_mean)
        
        # Conforming to the AbstractGraphModel, the "forward" method that is being invoked here 
        # returns a dictionary structure, whose values are torch tensor object that are somehow 
        # produced by the forward pass of the model.
        info: dict = self(data)
        
        # out_pred: (B, O)
        out_pred = info['graph_output']
        
        # graph_embedding: (B, D)
        graph_embedding = info['graph_embedding']
        
        # In the classification case the model outputs the classification LOGITS and it has to 
        # stay like that, but for the loss calculation we need the class proabilities which is 
        # why we apply the softmax function here.
        if self.prediction_mode == 'classification':
            
            # 17.04.24
            # If the label smoothing parameter is set, we will apply the label smoothing to the
            # target values. This will make the target values more robust against overfitting and
            # will promote the model to learn more generalizable features.
            # The idea is to use soft targets instead of the hard one-hot target vectors.
            # so a target of [1, 0] for example would become [0.9, 0.1] for a label smoothing.
            if self.label_smoothing:
                out_true = (1 - self.label_smoothing) * out_true + self.label_smoothing / self.target_dim
                
        if self.prediction_mode == 'bce':
            
            if self.label_smoothing:
                out_true = (1 - self.label_smoothing) * out_true + self.label_smoothing
        
        # ~ prediction loss
        loss_pred = self.loss_pred(out_pred, out_true)
        if hasattr(data, 'train_weight'):
            loss_pred = (loss_pred * data.train_weight.unsqueeze(-1)).mean()
        else:
            loss_pred = loss_pred.mean()
        
        self.log(
            'loss_pred', self.prediction_factor * loss_pred,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
        )
        
        # ~ explanaition co-training
        # The explanation co-training procedure will try to approximately solve the prediction task that is 
        # given to the network purely through a sub-graph identification based on each individual channel's 
        # explanation masks. So this procedure will promote the explanation masks to form in such a way that 
        # they by themselves will already be maximally informative towards solving the main prediction problem 
        # without taking the actual features into account.
        
        loss_expl = 0.0
        if self.importance_factor != 0:
            # The "training_explanation" method will calculate the loss for the explanation co-training 
            # procedure.
            loss_expl = self.training_explanation(
                data=data,
                info=info,
                batch_size=batch_size,
            )
        
        self.log(
            'loss_expl', loss_expl,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
        )
        
        # ~ explanation regularization
        
        # Here we regularize the explanation masks to be more sparse by penalizing the L1 norm of those 
        # explanation masks, which will lead those node/edge values to become closer to 0, which are not 
        # absolutely needed to maintain either the prediction or the explanation performance.
        
        # loss_spar = (
        #     torch.mean(torch.abs(info['node_importance'])) +
        #     torch.mean(torch.abs(info['edge_importance']))
        # )
        
        # loss_spar = -(
        #     #torch.mean(torch.log(info['edge_importance'] + 1e-8))
        #     torch.mean(torch.log(1 - info['edge_importance'] + 1e-9)) +
        #     torch.mean(torch.log(1 - info['node_importance'] + 1e-9))
        # )
        
        #loss_spar = (info['node_importance_norm'] - 0.5).pow(2).mean()
        ni = info['node_importance_norm']
        loss_spar = torch.mean(torch.abs(ni))
        
        self.log(
            'loss_spar', loss_spar,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
        )
        
        # ~ contrastive loss
        # The contrastive loss is applied to improve the *representation learning*. More specifically we 
        # want to learn semantic representations of the explained subgraphs. So in each channel's graph 
        # embedding space we want to promote that similiarly structured sub-graph explanations are 
        # grouped together in clusters.
        
        loss_cont = 0.0
        if self.contrastive_factor != 0:
            # The "training_representation" method will calculate the loss for the contrastive 
            # representation learning of the graph embeddings.
            loss_cont = self.training_representation(
                data=data,
                info=info,
                batch_size=batch_size,
            )
        
        self.log(
            'loss_cont', loss_cont,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
        )
        
        # ~ fidelity loss
        
        loss_fid = 0.0
        if self.fidelity_factor != 0:
            loss_fid = self.training_fidelity(
                data=data,
                info=info,
                batch_size=batch_size,
            )
            
        self.log(
            'loss_fid', loss_fid,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
        )
        
        # ~ adding the various loss terms
        # In this section we are simply accumulating the overall loss term as a combination of all the individual loss 
        # terms that were previously constructed.
        
        loss = (
            # loss from the primary output predictions
            self.prediction_factor * loss_pred
            # loss from the explanation co-training approximation
            + self.importance_factor * loss_expl
            # loss to encourage sparsity of explanations
            + self.sparsity_factor * loss_spar
            # contrastive loss to encourage the formation of semantic embeddings
            + self.contrastive_factor * loss_cont
            # fidelity loss to encourage positive fidelity values
            + self.fidelity_factor * loss_fid
        )
                    
        return loss
    
    def to(self, device, *args, **kwargs):
        
        if self.prediction_mode == 'regression':
            self.n_samples = self.n_samples.to(device)
            self.running_mean = self.running_mean.to(device)
            
        return super().to(device, *args, **kwargs)
    
    def training_representation(self,
                                data: Data,
                                info: dict,
                                batch_size: int
                                ) -> float:
        """
        This method implements the calculation of the SimCLR method for the explanation 
        representation learning objective.
        Given the input ``data`` representation, this method will return the corresponding loss value.
        
        The general idea behind SimCLR is to learn a semantic representation of the input elements in an 
        unsupervised fashion by jointly maximizing the similarity between an element and it's positive 
        samples (==semantically similar elements) and maximizing the similarity between an element and 
        its negative sampels (==semantically dissimilar elements). The main challange with this approach 
        being to obtain suitable positive and negative samples without supervised knowledge.
        
        In this specific case, the subject of the representatation learning will be each individual 
        explanation channel's explanation. 
        For the negative samples, we follow the common framework 
        of simply declaring all other samples in a given batch as the negative samples, since the 
        probability of them being semantically dissimiliar is higher than them being similar for a 
        moderatbly high number of expected distinct explanations.
        For the positive samples we apply a data augmentation where all non-explained nodes are 
        masked during the model prediction, therefore promoting the model to find a semantic 
        representation of the explained nodes only.
        
        :returns: a single loss value in a torch.Tensor
        """
        loss_cont = 0.0
        
        # info = self(data, stop_importance_grad=True)
        
        # ~ positive samples / data augmentation
        # As a first step we need to construct the data augmentation to obtain the positive 
        # samples for the constrastive term.
        # The general idead for the data augementation is that we want to mask out all the 
        # information that is not explained - so all the nodes that are NOT selected in 
        # each channels explanation mask. So by maximizing the similarity between that 
        # augemented versions embedding and the original embeddings, we promote the 
        # formation of a representation that encodes exactly the information that is 
        # contained within the explanation mask.
        
        # For this we need a way to determine which nodes are actually part of the explanation
        # and which are not - in essence we want to binarize the explanation masks. 
        # This is done by normalizing the importance values to a 0, 1 range and then applying 
        # a threshold.
        
        # node_importance: (B * V, K)
        node_importance = info['node_importance']
        # node_importance_norm: (B * V, K)
        max_values = torch_scatter.scatter_max(node_importance, data.batch, dim=0)[0]
        node_importance_norm = node_importance / max_values[data.batch]
        
        # pooled_node_importance: (B, K)
        pooled_node_importance = self.lay_pool_importance(node_importance, data.batch)
        # is_empty: (B, K)
        # This tensor is a binary mask on the 
        is_empty = (pooled_node_importance < self.importance_offset).float()
        
        # In the SimCLR framework we need 2 augmented views in total, so here we create the binarized 
        # feature masks but with two slightly different thresholds to capture kind of a multi-level 
        # information about the explanation
        node_importance_bin_1 = (node_importance_norm > 0.8).float()
        node_importance_bin_2 = (node_importance_norm > 0.6).float()
        
        # node_importance_bin_1 = self.lay_mask_expansion(
        #     mask=node_importance_bin_1,
        #     edge_index=data.edge_index,
        # )
        
        # node_importance_bin_2 = self.lay_mask_expansion(
        #     mask=node_importance_bin_2,
        #     edge_index=data.edge_index,
        # )
        
        node_importance_1 = node_importance + torch_gauss(list(node_importance.size()), mean=0, std=self.contrastive_noise).to(self.device)
        node_importance_2 = node_importance + torch_gauss(list(node_importance.size()), mean=0, std=self.contrastive_noise).to(self.device)
        
        # To now create the corresponding graph embedding vectors for the data augmentations we have
        # to actually query the model again with those augementations as addditional arguments.
        info_1 = self(
            data, 
            node_importance_overwrite=node_importance_1, 
            node_feature_mask=node_importance_bin_1,
            stop_importance_grad=False,
        )
        info_2 = self(
            data, 
            node_importance_overwrite=node_importance_2, 
            node_feature_mask=node_importance_bin_2,
            stop_importance_grad=False,
        )
        
        # The negative samples are realized as simply assuming that all other samples of a current batch 
        # are the negative samples for each element. The easiest technical implementation is to calculate 
        # a quadratic similarity matrix of all elements with all elements and then later reduce this.
        # However, this matrix will also contain the diagonal entries of an elements similarity with itself
        # we dont want to affect those entries which is why we construct this mask to ignore them.
        # (0, 1, 1, ...)
        # (1, 0, 1, ...)
        mask = torch.ones((batch_size, 2 * batch_size), dtype=bool).to(self.device)
        for i in range (batch_size):
            mask[i, i] = 0
            mask[i, i + batch_size] = 0
        mask = torch.cat([mask, mask], dim=0)
    
        # Essentially we want each channel to develop it's own independent embedding space, so here we 
        # iterate over all the channels and calculate the constrastive loss contribution for each channel.
        for k in range(self.num_channels):
            
            # is_empty_k: (B, )
            is_empty_k = torch.cat([is_empty[:, k], is_empty[:, k]], dim=0)
            
            # NOTE: Looking at the similarity computations, one can see that these are realized as simple 
            # vector multiplications between the embeddings. However, the graph embeddings are inherently 
            # L2-normalized - in this special case the multiplication of two vectors is equal to their 
            # cosine similarity!
            
            # graph_embedding_k = info['graph_embedding'][:, :, k]
            # graph_embedding_1 = info_1['graph_embedding'][:, :, k]
            # graph_embedding_2 = info_2['graph_embedding'][:, :, k]
            
            lay_proj = self.projection_layers[k]
            graph_embedding_k = F.normalize(lay_proj(info['graph_embedding'][:, :, k]))
            graph_embedding_1 = F.normalize(lay_proj(info_1['graph_embedding'][:, :, k]))
            graph_embedding_2 = F.normalize(lay_proj(info_2['graph_embedding'][:, :, k]))
            
            # graph_embedding_all = (2 * B, D)
            graph_embedding_all = torch.cat([graph_embedding_k, graph_embedding_k], dim=0)
            
            sim_neg = (graph_embedding_all.unsqueeze(0) * graph_embedding_all.unsqueeze(1)).sum(dim=-1)
            sim_neg_exp = torch.exp(sim_neg / self.contrastive_temp)
            # sim_neg_exp: (2 * B, 2 * B)
            sim_neg_exp = sim_neg_exp.masked_select(mask).view(2 * batch_size, -1)
            
            # sim_pos_1: (B, )
            sim_pos_1 = (graph_embedding_1 * graph_embedding_k).sum(dim=-1)
            sim_pos_1_exp = torch.exp(sim_pos_1 / self.contrastive_temp)
            # sim_pos_2: (B, )
            sim_pos_2 = (graph_embedding_2 * graph_embedding_k).sum(dim=-1)
            sim_pos_2_exp = torch.exp(sim_pos_2 / self.contrastive_temp)
            sim_pos = (graph_embedding_1 * graph_embedding_2).sum(dim=-1)
            sim_pos_exp = torch.exp(sim_pos / self.contrastive_temp)
            # sim_pos_exp: (2 *B, 1)
            sim_pos_exp = torch.cat([
                torch.stack([sim_pos_exp, sim_pos_1_exp], dim=-1).sum(dim=-1),
                torch.stack([sim_pos_exp, sim_pos_2_exp], dim=-1).sum(dim=-1),
            ], dim=0)
            #sim_pos_exp[is_empty_k > 0.5] = 1.0
            
            # In this section the actual constrastive loss aka the InfoNCE loss is calculated. However, this is 
            # not just the simple loss term that is used in SimCLR, but also uses an additional debiasing method 
            # that was proposed in a different paper.
            N = batch_size * 2 - 2
            imp = (self.contrastive_beta * sim_neg_exp.log()).exp()
            reweight_neg = (imp * sim_neg_exp).sum(dim=-1) / imp.mean(dim=-1)
            #Neg = (-N*self.contrastive_tau*sim_pos_exp + sim_neg_exp.sum(dim=-1)) / (1-self.contrastive_tau)
            Neg = (-N*self.contrastive_tau*sim_pos_exp + reweight_neg) / (1-self.contrastive_tau)
            Neg = torch.clamp(Neg, min=N*np.e**(-1/self.contrastive_temp))
            #Neg = sim_neg_exp.sum(dim=-1)
            
            loss_cont += (1 / self.num_channels) * torch.mean(-torch.log((sim_pos_exp) / (sim_pos_exp + Neg)))
            
        return loss_cont
    
    def training_fidelity(self,
                          data: Data,
                          info: dict,
                          batch_size: int,
                          ) -> float:
        """
        This method implements the fidelity training step. Based on the current Data instance ``data`` and the 
        ``info`` dictionary that was generated by the forward pass of the model. This method will calculate the 
        fidelity based loss and return it as a float value.
        
        The fidelity loss defines a loss values that promotes each channel's associated fidelity value to be 
        generally positive. A positive fidelity value indicates that a channel's explanation actually affects the 
        final prediction outcome in the same manner as the channel's intended interpretation. For example, for the 
        "positive" explanation channel, it should also actually have a positive contribution towards the predicted 
        regression value.
    
        This is done by running an additional forward pass for each channel where that channel's explanation is 
        masked from the prediction by setting all the attention values to zero. These modified forward results 
        are used to calculate the leave-one-out deviation difference regarding the original model output. 
        Depending on the task (regression or classification) the fidelity value is then calculated from those 
        deviation values and a loss is defined to promote positive fidelity values.
        
        :param data: The original torch Data instance with the input data for the model 
        :param info: The dict structure containing the model output for the original forward pass
        :param batch_size: The integer number of elements in the current batch.
        
        :returns: The float fidelity value
        """
        
        # Here we construct the leave-one-out deviation matrix. Essentially for every explanation channel 
        # of the model we mask out that explanation channel by setting all of its attention values to zero
        # and then do another forward pass with this modified input. The deviation between the original 
        # prediction and that modified prediction gives us an indication of how much that explanation channel 
        # contributes to the overall prediction. Specifically we can determine that effect for each of the 
        # output values (in the case of classification there will be multiple outputs).
        # We do this in the face of the thing.
        deviations = []
        for k in range(self.num_channels):
            mask_out = torch.ones_like(info['node_importance'], dtype=torch.float32)
            mask_out[:, k] = 0.0
            
            info_out = self.forward(
                data,
                node_mask=mask_out,
            )    
            deviations.append(info['graph_output'] - info_out['graph_output'])
            
        # deviations: (B, C, K)
        deviations = torch.stack(deviations, dim=-1)
            
        # fidelity: (B, K)
        # Unlike the raw deviations, which is a matrix for the outputs and the channels, the fidelity is a single 
        # value that is associated with each channel. The larger each channel's fidelity value, the higher the 
        # contribution of that channel towards the outcome of the prediction.
        # Most importantly, the fidelity is supposed to be a positive value. A positive fidelity value indicates
        # that the explanation highlighted by a channel actually acts in the same direction as that channel's
        # pre-determined interpretation.
        
        # For regression problems the fidelity is simple since the positive channel's contribution is already a 
        # positive values. Consequently we only need to invert the negative channels (channel 0) value to 
        # promote that to be canonically negative.
        if self.prediction_mode == 'regression':
            fidelity = deviations.squeeze()
            fidelity[:, 0] *= -1.0
            
        # For classification the fidelity is mainly defined through the diagonal elements of the deviation matrix.
        # Each channel corresponds to one output class. Therefore, each channel should positively influence it's 
        # corresponding class. The explanation depicted in that channel should generally increase the class logits
        # while decreasing (off-diagonal) all the other logits.
        elif self.prediction_mode == 'classification' or self.prediction_mode == 'bce':
            # diagonal
            fidelity = torch.diagonal(deviations, dim1=-2, dim2=-1) 
            # off-diagonal
            fidelity -= torch.sum((1.0 - torch.eye(self.num_channels)).to(self.device).unsqueeze(0) * deviations, dim=-2)
    
        return torch.mean(F.relu(-fidelity)) + 0.1 * torch.mean(torch.square(fidelity))
    
    def training_explanation(self, 
                             data: Data,
                             info: dict,
                             batch_size: int
                             ) -> float:
        """
        This method implements the explanation co-training loss that guides the explanation masks to 
        develop according to the pre-defined interpretations of the explanation channels.
        Given the current ``data`` instance and the resulting ``info`` of the model forward pass, 
        this method will calculate the explanation loss.
        
        :returns: A torch Tensor with a single loss value
        """
        
        loss_expl = 0.0
        
        if self.importance_target == 'node':
            
            # -- node-level explanation approximation --
            
            # node_importance: (B * V, K)
            # for each *node* these are attention values in the range [0, 1]
            node_importance = info['node_importance']
            
            max_values = torch_scatter.scatter_max(node_importance, data.batch, dim=0)[0]
            max_values = torch.amax(max_values, dim=-1, keepdim=True)
            max_values = torch.where(max_values < 0.01, torch.ones_like(max_values) * 1e9, max_values)
            #print(max_values.shape, edge_importance.shape)
            node_importance = node_importance / max_values[data.batch]
            
            # node_importance_norm: (B * V, K)
            # max_values = torch_scatter.scatter_max(node_importance, data.batch, dim=0)[0] + 1e-8
            # node_importance = node_importance / max_values[data.batch]
            
            # node_transformed: (B * V, K)
            node_transformed = F.sigmoid(self.lay_transform_2(self.lay_transform_1(data.x).relu())) + self.importance_offset
            node_transformed = node_transformed * node_importance
            
            pooled_importance = self.lay_pool_importance(node_transformed, data.batch)
        
        elif self.importance_target == 'edge':
        
            # -- edge-level explanation approximation --
            
            # edge_importance: (B * E, K)
            edge_importance = info['edge_importance']
            
            # edge_input: (B * E, M + 2N)
            edge_input = torch.cat([data.edge_attr, data.x[data.edge_index[0]], data.x[data.edge_index[1]]], dim=-1)
            
            max_values = torch_scatter.scatter_max(edge_importance, data.batch[data.edge_index[0]], dim=0)[0]
            max_values = torch.amax(max_values, dim=-1, keepdim=True)
            max_values = torch.where(max_values < 0.01, torch.ones_like(max_values) * 1e9, max_values)
            #print(max_values.shape, edge_importance.shape)
            edge_importance = edge_importance / max_values[data.batch[data.edge_index[0]]]
            
            #edge_importance = torch.where(edge_importance < 0.1, edge_importance * 0.01, edge_importance)
            
            # edge_transformed: (B * E, K)
            edge_transformed = F.sigmoid(self.lay_transform_2(self.lay_transform_1(edge_input).relu()) - 5) + self.importance_offset
            edge_transformed = torch.ones_like(edge_transformed, device=self.device) * self.importance_offset
            edge_transformed = edge_transformed * edge_importance
            
            pooled_importance = self.lay_pool_importance(edge_transformed, data.batch[data.edge_index[0]])
        
        # ~ constructing the approximation
        
        # out_pred: (B, O)
        out_pred = info['graph_output']
        # out_pred: (B, O)
        out_true = data.y.view(out_pred.shape)
        
        if self.importance_mode == 'regression':
            
            # Here we actually deviate from the original MEGAN implementation a little bit. In the original 
            # the explanation co-training loss for regression tasks is again an MSE loss that tries to interpolate 
            # the exact ground truth value from the pooled mask. However, it turns out that casting this into a 
            # classification problem and using BCE loss works better (cleaner explanation masks).
            # So here we construct the "true" labels simply as a binary decision problem of samples being either 
            # "positive" or "negative".
            
            #regression_reference = self.regression_reference
            regression_reference = torch.mean(out_true)
            regression_lo = torch.quantile(out_true, 0.5 - self.regression_margin)
            regression_hi = torch.quantile(out_true, 0.5 + self.regression_margin)
            #print(regression_lo, regression_hi, regression_reference)

            values_true = torch.cat([
                # (out_true < (regression_reference - self.regression_margin)), 
                # (out_true > (regression_reference + self.regression_margin)),
                out_true < regression_lo,
                out_true > regression_hi,
            ], 
            axis=1).float()
            
            # values_pred: (B, K)
            #values_pred = torch.sigmoid(scaling * (pooled_importance - offset))
            values_pred = torch.tanh(0.1 * pooled_importance)
            values_true *= 0.9
            
            loss_expl = F.binary_cross_entropy(values_pred, values_true, )
            
        elif self.importance_mode == 'classification':
            
            # The classification case is quite simple in that we want each channels pooled importances to predict 
            # each possible class as a binary classification problem using a BCE loss instead of a multi-class 
            # classification problem.
            
            # values_true: (B, K)
            values_true = out_true
            # values_pred: (B, K)
            #values_pred = torch.sigmoid(scaling * (pooled_importance - offset))
            values_pred = torch.tanh(0.1 * pooled_importance)
            values_true = values_true * 0.9
            
            loss_expl = F.binary_cross_entropy(values_pred, values_true)
                        
        return loss_expl
    
    def configure_optimizers(self):
        """
        This method returns the optimizer to be used for the training of the model.
        
        :returns: A ``torch.optim.Optimizer`` instance
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.NAdam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def leave_one_out_deviations(self,
                                 graphs: t.List[tv.GraphDict],
                                 batch_size: int = 10_000,
                                 ) -> np.ndarray:
        """
        Given a list of graph dict representations ``graphs``, this method will construct the 
        leave one out deviation data structure that contains the fidelity values for each graph 
        in the list and each explanation channel individually.
        
        The array returned by this method has the shape (B, O, K) where B is the number of graphs, 
        O the output dimension of the network and K the number of explanations channels used 
        by the network. 
        
        :param graphs: A list of all graph dict representations of graphs for which to calculate 
            the deviations
        :param batch_size: The number of graphs to query the model with at the same time.
        
        :returns: numpy array of shape (B, O, K)
        """
        result = np.zeros((len(graphs), self.out_dim, self.num_channels,))
        loader = self._loader_from_graphs(graphs, batch_size=batch_size)
        
        index = 0
        # In the first instance we need to iterate over all the batches that were consructed from the 
        # data loader. All data instances contain the iformation of batch_size graphs in the flattened 
        # form
        for data in loader:
            
            # This is the actual number of graphs in the current batch. Note that this will most likely 
            # be == batch_size, but does not have to be when there is a number < batch_size left over at 
            # the end!
            _batch_size = np.max(data.batch.numpy()) + 1
            
            # This is the baseline for the deviation computation. This is the unmodified prediction of 
            # the model.
            # out_pred: (_batch_size, O) numpy
            out_pred = self.forward(data)['graph_output'].detach().numpy()
            
            # Then the whole point is that we need to calculate the fidelity for each channel 
            # independently. So for each channel we mask out that channel and then compute the updated 
            # predictions with that mask. This difference is the leave_one_out deviation.
            for channel_index in range(self.num_channels):
                node_mask = torch.ones((data.x.shape[0], self.num_channels)).to(self.device)
                node_mask[:, channel_index] = 0.0
                
                # out_mod: (_batch_size, O) numpy
                out_mod = self.forward(data, node_mask=node_mask)['graph_output'].detach().numpy()
                
                for i in range(_batch_size):
                    result[index + i, :, channel_index] = out_pred[i, :] - out_mod[i, :]
            
            index += _batch_size
                    
        return result