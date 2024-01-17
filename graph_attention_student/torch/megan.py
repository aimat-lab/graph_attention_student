import os
import typing as t
from pytorch_lightning.utilities.types import OptimizerLRScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.data import Data

from graph_attention_student.torch.model import AbstractGraphModel
from graph_attention_student.torch.layers import MultiHeadAttention
from graph_attention_student.torch.layers import GraphAttentionLayer


class Megan(AbstractGraphModel):
    
    # :attr IMPORTANCE_MODES:
    #       This attribute defines the possible values for the "importance_mode" parameter of the model
    #       that parameter determines how the explanation co-training procedure is realized during the 
    #       models training_step.
    #       For None, the explanation co-training is disabled completely.
    IMPORTANCE_MODES = [None, 'regression', 'classification']
    
    """
    
    **NOMENCLATURE**
    
    In the context of this class, the following abbreviations are used:
    
    - B: The number of graphs in one batche
    - V: The number of nodes in a graph
    - E: The number of edges in a graph
    - N_l: The dimensionality of the node feature vector after layer l
    - D: The dimensionality of the graph embedding of a single channel
    - M: The dimensionality of the edge feature vector
    - K: The number of importance channels
    
    """
    def __init__(self,
                 # encoder-related
                 node_dim: int = 3,
                 edge_dim: int = 1,
                 units: t.List[int] = [16, 16, 16],
                 # explanation-related
                 importance_units: t.List[int] = [16, ],
                 projection_units: t.List[int] = [],
                 num_channels: int = 2,
                 importance_mode: t.Optional[str] = None,
                 importance_factor: float = 0.0,
                 regression_reference: float = 0.0,
                 sparsity_factor: float = 0.0,
                 # prediction-related
                 final_units: t.List[int] = [16, 1],
                 prediction_mode: str = 'regression',
                 use_bias: bool = True,
                 # training-related
                 learning_rate: float = 1e-3,
                 ):
        pl.LightningModule.__init__(self)
        
        # ~ validating the parameters
        # There are some parameters whose values we want to validate before starting to construct the 
        # the instance, because choosing the incorrect values for some parameters will lead to uninformative errors 
        # down the line.
        assert importance_mode in self.IMPORTANCE_MODES, f'importance_mode has to be value from {self.IMPORTANCE_MODES}'
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.units = units
        self.use_bias = use_bias
        
        self.importance_units = importance_units
        self.projection_units = projection_units
        self.importance_mode = importance_mode
        self.num_channels = num_channels
        self.importance_factor = importance_factor
        self.regression_reference = regression_reference
        self.sparsity_factor = sparsity_factor
        
        self.prediction_mode = prediction_mode
        self.final_untis = final_units
        
        self.learning_rate = learning_rate

        # We need to update this dict here so that we can update it later.
        self.hparams.update({
            'node_dim':                 node_dim,
            'edge_dim':                 edge_dim,
            'units':                    units,
            'importance_units':         importance_units,
            'projection_units':         projection_units,
            'final_units':              final_units,  
            'prediction_mode':          prediction_mode,
        })

        # ~ Graph Message passing layers
        
        self.encoder_layers = nn.ModuleList()
        prev_features = node_dim
        for num_features in units:
            lay = MultiHeadAttention([
                GraphAttentionLayer(
                    in_dim=prev_features,
                    out_dim=num_features,
                    edge_dim=edge_dim,
                )
                for _ in range(num_channels)
            ])
            prev_features = num_features
            self.encoder_layers.append(lay)
            
        # The last number of units in the graph encoder part determines the embedding dimension
        self.embedding_dim = prev_features
            
        self.lay_pool = SumAggregation()
        self.lay_pool_importance = SumAggregation()
            
        # ~ Importance Attention Layers
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
        
        self.channel_projection_layers = nn.ModuleList()
        for k in range(num_channels):
            
            layers = nn.ModuleList()
            prev_features = self.embedding_dim
            for num_features in projection_units:
                lay = nn.Linear(
                    in_features=prev_features,
                    out_features=num_features,
                )
                prev_features = num_features
                layers.append(lay)
            
            self.channel_projection_layers.append(layers)
            
        self.embedding_dim = prev_features
            
        # ~ Dense Prediction layers
        
        self.dense_layers = nn.ModuleList()
        prev_features = self.embedding_dim * num_channels
        for num_features in final_units:
            lay = nn.Linear(
                in_features=prev_features,
                out_features=num_features,
                # bias=use_bias,
            )
            prev_features = num_features
            self.dense_layers.append(lay)
            
        self.lay_act = nn.SiLU()
        
        self.lay_act_final = nn.Identity()
        if self.prediction_mode == 'regression':
            self.loss_pred = nn.MSELoss()
            
        elif self.prediction_mode == 'classification':
            self.loss_pred = nn.CrossEntropyLoss(label_smoothing=0.1)
            
    def forward(self, data: Data):
    
        node_input, edge_input, edge_index = data.x, data.edge_attr, data.edge_index
        
        # node_embedding: (B * V, N_0)
        node_embedding = node_input
        
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
            
            node_embeddings.append(node_embedding)
            alphas.append(alpha)
            
        # node_embeddings = torch.stack(node_embeddings, axis=-1)
        # node_embedding = torch.amax(node_embeddings, axis=-1)
            
        edge_importance = torch.cat([alpha.unsqueeze(-1) for alpha in alphas], dim=-1)
        # edge_importance: (B * E, K)
        edge_importance = torch.sum(edge_importance, dim=-1)
        edge_importance = F.sigmoid(edge_importance)
        
        # edge_importance_pooled: (B * V, K)
        edge_importance_pooled = 0.5 * (
            self.lay_pool(edge_importance, edge_index[0]) + 
            self.lay_pool(edge_importance, edge_index[1])   
        )
            
        # ~ importance masks / explanations
        node_importance = node_embedding
        for lay in self.importance_layers[:-1]:
            node_importance = lay(node_importance)
            node_importance = self.lay_act(node_importance)
            
        # node_importance: (B * V, K) - attention values in [0, 1]
        node_importance = self.importance_layers[-1](node_importance)
        node_importance = torch.sigmoid(node_importance)
        
        node_importance = node_importance * edge_importance_pooled
            
        # node_embedding: (B * V, D)
        # node_embedding_channels: (B * V, K, D)
        node_embedding_channels = node_embedding.unsqueeze(1) * node_importance.unsqueeze(2)
            
        graph_embedding_channels = []
        #for k in range(self.num_channels):
        for k, layers in enumerate(self.channel_projection_layers):
            node_embedding_ = node_embedding
            
            node_embedding_ = node_embedding_ * node_importance[:, k].unsqueeze(-1)
            graph_embedding_ = self.lay_pool(node_embedding_, data.batch)

            # The layers that are referred to here are the layers that are 
            if len(layers) != 0:
                
                for lay in layers[:-1]:
                    graph_embedding_ = lay(graph_embedding_)
                    graph_embedding_ = self.lay_act(graph_embedding_)
            
                graph_embedding_ = layers[-1](graph_embedding_)
                
            graph_embedding_ = F.tanh(graph_embedding_)
            graph_embedding_channels.append(graph_embedding_)
            
        # graph_embedding: (B, D, K)
        graph_embedding = torch.stack(graph_embedding_channels, dim=-1)
        
        # output: (B, D * K)
        output = torch.cat(graph_embedding_channels, dim=-1)
        
        for lay in self.dense_layers[:-1]:
            output = lay(output)    
            output = self.lay_act(output)
            
        # output: (B, O)
        output = self.dense_layers[-1](output)
        # output = self.lay_act_final(output)
            
        return {
            'graph_output': output,
            'graph_embedding': graph_embedding,
            'node_embedding': node_embedding,
            'node_importance': node_importance,
            'edge_importance': edge_importance,
        }
        
    def training_step(self, data: Data, batch_idx):
        
        batch_size = np.max(data.batch.cpu().numpy()) + 1
        
        info: dict = self(data)
        
        # out_pred: (B, O)
        out_pred = info['graph_output']
        # out_pred: (B, O)
        out_true = data.y.view(out_pred.shape)
        
        if self.prediction_mode == 'classification':
            out_pred = F.softmax(out_pred, dim=-1)
            out_true = torch.argmax(out_true, dim=-1)
        
        # ~ prediction loss
        loss_pred = self.loss_pred(out_pred, out_true)
        self.log(
            'loss_pred', loss_pred,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        
        # ~ explanaition co-training
        
        loss_expl = self.training_explanation(
            data=data,
            info=info,
            batch_size=batch_size,
        )
        self.log(
            'loss_expl', loss_expl,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        
        # ~ explanation regularization
        
        loss_spar = (
            torch.mean(torch.abs(info['node_importance'])) +
            torch.mean(torch.abs(info['edge_importance']))
        )
        
        # ~ adding the various loss terms
        
        loss = (
            # loss from the primary output predictions
            loss_pred
            # loss from the explanation co-training approximation
            + self.importance_factor * loss_expl
            # loss to encourage sparsity of explanations
            + self.sparsity_factor * loss_spar
        )
        
        return loss
    
    def training_explanation(self, 
                             data: Data,
                             info: dict,
                             batch_size: int
                             ) -> float:
        
        loss_expl = 0.0
        # Here we treat the default case of importance_mode being None, which signals that the explanation 
        # co-training should be disabled completly. In that case we simply return a constant loss of 0 which 
        # will not result in any network gradients and therefore no weight updates.
        if self.importance_mode is None:
            return loss_expl
        
        # ~ aggregating the explanation masks as predicted values
        
        # node_importance: (B * V, K)
        # for each *node* these are attention values in the range [0, 1]
        node_importance = info['node_importance']
        # pooled_importance: (B, K)
        # for each *graph* this is a value in the range [0, V] because sum over the [0, 1] node importances
        pooled_importance = self.lay_pool_importance(node_importance, data.batch)
        
        # ~ constructing the approximation
        
        # out_pred: (B, O)
        out_pred = info['graph_output']
        # out_pred: (B, O)
        out_true = data.y.view(out_pred.shape)
        
        if self.importance_mode == 'regression':
            
            # values_true: (B, 2)
            values_true = torch.cat([
                torch.abs(out_true - self.regression_reference), 
                torch.abs(out_true - self.regression_reference),
            ], 
            axis=1)
            # values_pred: (B, 2)
            values_pred = pooled_importance
            
            # mask: (B, 2)
            mask = torch.cat([
                out_true < self.regression_reference,
                out_true > self.regression_reference,
            ], axis=1).float()
            
            loss_expl = torch.mean(torch.square(values_true - values_pred) * mask)
            
        elif self.importance_mode == 'classification':
            
            # values_true: (B, K)
            values_true = out_true
            # values_pred: (B, K)
            values_pred = torch.sigmoid(3 * (pooled_importance - 0.8))
            
            loss_expl = F.binary_cross_entropy(values_pred, values_true)
                        
        return loss_expl
    
    def configure_optimizers(self):
        """
        This method returns the optimizer to be used for the training of the model.
        
        :returns: A ``torch.optim.Optimizer`` instance
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer