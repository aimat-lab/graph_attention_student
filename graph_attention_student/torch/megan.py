import os
import typing as t
from pytorch_lightning.utilities.types import OptimizerLRScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import pytorch_lightning as pl
import numpy as np
import visual_graph_datasets.typing as tv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn.aggr import MaxAggregation
from torch_geometric.data import Data

from graph_attention_student.torch.utils import torch_uniform
from graph_attention_student.torch.model import AbstractGraphModel
from graph_attention_student.torch.layers import MultiHeadAttention
from graph_attention_student.torch.layers import GraphAttentionLayer
from graph_attention_student.torch.layers import AttentiveFpLayer


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
    PREDICTION_MODES = ['regression', 'classification']
    
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
                 importance_offset: float = 0.8,
                 regression_reference: float = 0.0,
                 sparsity_factor: float = 0.0,
                 # contrastive representation related
                 contrastive_factor: float = 0.0,
                 contrastive_noise: float = 0.1,
                 contrastive_temp: float = 1.0,
                 contrastive_beta: float = 1.0,
                 contrastive_tau: float = 0.1,
                 # prediction-related
                 final_units: t.List[int] = [16, 1],
                 prediction_mode: str = 'regression',
                 use_bias: bool = True,
                 # training-related
                 learning_rate: float = 1e-3,
                 ):
        pl.LightningModule.__init__(self)
        
        # The last integer value in the list of final_units determines the output dimension of the network aka how 
        # many graph properties the network will predict at the same time.
        self.out_dim = final_units[-1]
        
        # ~ validating the parameters
        # There are some parameters whose values we want to validate before starting to construct the 
        # the instance, because choosing the incorrect values for some parameters will lead to uninformative errors 
        # down the line.
        assert importance_mode in self.IMPORTANCE_MODES, f'importance_mode has to be one of {self.IMPORTANCE_MODES}'
        assert prediction_mode in self.PREDICTION_MODES, f'prediction_mode has to be one of {self.PREDICTION_MODES}'
        if prediction_mode == 'regression':
            assert num_channels == 2, 'for regression explanations, num_channels must be 2 (negative & positive)!'
        if prediction_mode == 'classification':
            assert num_channels == self.out_dim, 'for classification explanations, num_channels must be number of outputs!'
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.units = units
        self.use_bias = use_bias
        
        self.importance_units = importance_units
        self.projection_units = projection_units
        self.importance_mode = importance_mode
        self.num_channels = num_channels
        self.importance_factor = importance_factor
        self.importance_offset = importance_offset
        self.regression_reference = regression_reference
        self.sparsity_factor = sparsity_factor
        
        self.contrastive_factor = contrastive_factor
        self.contrastive_noise = contrastive_noise
        self.contrastive_temp = contrastive_temp
        self.contrastive_beta = contrastive_beta
        self.contrastive_tau = contrastive_tau
        
        self.prediction_mode = prediction_mode
        self.final_untis = final_units
        
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
            'importance_units':         importance_units,
            'importance_offset':        importance_offset,
            'projection_units':         projection_units,
            'final_units':              final_units,  
            'num_channels':             num_channels,
            'prediction_mode':          prediction_mode,
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
        
        self.encoder_layers = nn.ModuleList()
        prev_features = units[0]
        # These are the activations that will be used on the result of the muli-head encoding. The important 
        # thing here is that we use a linear / identity activation for the very last layer since we dont want 
        # to restrict the expressiveness of the values which will ultimately become the graph embedding.
        activations = ['relu' for _ in units]
        activations[-1] = 'linear'
        for num_features, act in zip(units, activations):
            # Each layer in the encoder is a multi-head attention layer, where the number of parallel heads is 
            # also defined as the number of explanation "channels". The idea is that each of these channels captures 
            # explanations according to their pre-defined behavior.
            
            lay = MultiHeadAttention([
                GraphAttentionLayer(
                    in_dim=prev_features,
                    out_dim=num_features,
                    edge_dim=edge_dim,
                )
                for _ in range(num_channels)
            ], activation=act)
            # lay = MultiHeadAttention([
            #     AttentiveFpLayer(
            #         in_dim=prev_features,
            #         out_dim=num_features,
            #         edge_dim=edge_dim,
            #     )
            #     for _ in range(num_channels)
            # ])
            
            prev_features = num_features
            self.encoder_layers.append(lay)
            
        # The last number of units in the graph encoder part determines the embedding dimension
        self.embedding_dim = prev_features
            
        # At the end of the graph encoder, the node embeddings for all the nodes are aggregated into one final 
        # graph embedding vector by doing an attention-weighted sum.
        self.lay_pool = SumAggregation()
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
            
        self.lay_act = nn.ReLU()
        
        self.lay_act_final = nn.Identity()
        if self.prediction_mode == 'regression':
            self.loss_pred = nn.MSELoss()
            
        elif self.prediction_mode == 'classification':
            self.loss_pred = nn.CrossEntropyLoss(label_smoothing=0.0)
            
    def forward(self, 
                data: Data, 
                node_mask: t.Optional[torch.Tensor] = None,
                node_feature_mask: t.Optional[torch.Tensor] = None
                ) -> t.Dict[str, torch.Tensor]:
    
        node_input, edge_input, edge_index = data.x, data.edge_attr, data.edge_index
        
        # node_embedding: (B * V, N_0)
        node_embedding = self.lay_embedd(node_input)
        
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
            
        # edge_importance: (B * E, K, L)
        edge_importance = torch.stack(alphas, dim=-1)
        # edge_importance: (B * E, K)
        edge_importance = torch.sum(edge_importance, dim=-1)
        #edge_importance = torch.amin(edge_importance, dim=-1)
        edge_importance = F.sigmoid(edge_importance)
        
        # edge_importance_pooled: (B * V, K)
        edge_importance_pooled = 0.5 * (
            self.lay_pool_edge(edge_importance, edge_index[0]) + 
            self.lay_pool_edge(edge_importance, edge_index[1])   
        )
            
        # ~ importance masks / explanations
        node_importance = node_embedding
        for lay in self.importance_layers[:-1]:
            node_importance = lay(node_importance)
            node_importance = self.lay_act(node_importance)
            
        # node_importance: (B * V, K) - attention values in [0, 1]
        node_importance = self.importance_layers[-1](node_importance)
        node_importance = torch.sigmoid(node_importance)
        
        #node_importance = node_importance * edge_importance_pooled
        node_importance = edge_importance_pooled
        
        if node_mask is not None:
            node_importance = node_importance * node_mask
            
        # node_embedding: (B * V, D)
        # node_embedding_channels: (B * V, K, D)
        node_embedding_channels = node_embedding.unsqueeze(1) * node_importance.unsqueeze(2)
            
        graph_embedding_channels = []
        for k, layers in enumerate(self.channel_projection_layers):
            node_embedding_ = node_embedding
            
            node_embedding_ = node_embedding_ * node_importance[:, k].unsqueeze(-1)
            if node_feature_mask is not None:
                node_embedding_ *= node_feature_mask[:, k].unsqueeze(-1)
            
            graph_embedding_ = self.lay_pool(node_embedding_, data.batch)

            # The layers that are referred to here are the layers that are 
            if len(layers) != 0:
                
                for lay in layers[:-1]:
                    graph_embedding_ = lay(graph_embedding_)
                    graph_embedding_ = self.lay_act(graph_embedding_)
            
                graph_embedding_ = layers[-1](graph_embedding_)
            
            graph_embedding_ = F.normalize(graph_embedding_)
            # F.normalize will apply a transformation on the embedding so that all the embedding 
            # vectors have a constant norm == 1.
            graph_embedding_ = F.normalize(graph_embedding_)
            graph_embedding_channels.append(graph_embedding_)
            
        # graph_embedding: (B, D, K)
        graph_embedding = torch.stack(graph_embedding_channels, dim=-1)
        
        # output: (B, D * K)
        output = torch.cat(graph_embedding_channels, dim=-1)
        # output = torch.sum(graph_embedding, dim=-1)
        
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
        
        # Conforming to the AbstractGraphModel, the "forward" method that is being invoked here 
        # returns a dictionary structure, whose values are torch tensor object that are somehow 
        # produced by the forward pass of the model.
        info: dict = self(data)
        
        # out_pred: (B, O)
        out_pred = info['graph_output']
        # out_pred: (B, O)
        out_true = data.y.view(out_pred.shape)
        # graph_embedding: (B, D)
        graph_embedding = info['graph_embedding']
        
        # In the classification case the model outputs the classification LOGITS and it has to 
        # stay like that, but for the loss calculation we need the class proabilities which is 
        # why we apply the softmax function here.
        if self.prediction_mode == 'classification':
            out_pred = F.softmax(out_pred, dim=-1)
        
        # ~ prediction loss
        loss_pred = self.loss_pred(out_pred, out_true)
        self.log(
            'loss_pred', loss_pred,
            prog_bar=True,
            on_epoch=True,
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
            on_step=True,
            batch_size=batch_size,
        )
        
        # ~ explanation regularization
        
        # Here we regularize the explanation masks to be more sparse by penalizing the L1 norm of those 
        # explanation masks, which will lead those node/edge values to become closer to 0, which are not 
        # absolutely needed to maintain either the prediction or the explanation performance.
        loss_spar = (
            torch.mean(torch.abs(info['node_importance'])) +
            torch.mean(torch.abs(info['edge_importance']))
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
            on_step=True,
            batch_size=batch_size,
        )
        
        # ~ adding the various loss terms
        # In this section we are simply accumulating the overall loss term as a combination of all the individual loss 
        # terms that were previously constructed.
        
        loss = (
            # loss from the primary output predictions
            loss_pred
            # loss from the explanation co-training approximation
            + self.importance_factor * loss_expl
            # loss to encourage sparsity of explanations
            + self.sparsity_factor * loss_spar
            # contrastive loss to encourage the formation of semantic embeddings
            + self.contrastive_factor * loss_cont
        )
        
        return loss
    
    def training_representation(self,
                                data: Data,
                                info: dict,
                                batch_size: int
                                ) -> float:
        
        loss_cont = 0.0
        
        node_importance = info['node_importance']
        
        max_values = torch_scatter.scatter_max(node_importance, data.batch, dim=0)[0]
        node_importance_norm = node_importance / max_values[data.batch]
        node_importance_bin_1 = (node_importance_norm > 0.6).float()
        node_importance_bin_2 = (node_importance_norm > 0.4).float()
        
        noise = 1.0
        noise_mask = torch.full_like(node_importance, noise).to(self.device)
        
        node_mask_1 = torch.bernoulli(noise_mask).float()
        node_mask_1 = torch_uniform(node_importance.shape, 0.95, 1.05).to(self.device)
        data1 = data.clone()
        # data1.x *= torch.bernoulli(torch.full(data.x.shape, noise)).to(self.device)
        info_1 = self(
            data1, 
            #node_mask=node_mask_1, 
            node_feature_mask=node_importance_bin_1
        )
        
        node_mask_2 = torch.bernoulli(noise_mask).float()
        node_mask_2 = torch_uniform(node_importance.shape, 0.95, 1.05).to(self.device)
        data2 = data.clone()
        # data2.x *= torch.bernoulli(torch.full(data.x.shape, noise)).to(self.device)
        info_2 = self(
            data2, 
            #node_mask=node_mask_2, 
            node_feature_mask=node_importance_bin_2
        )
        
        mask = torch.ones((batch_size, 2 * batch_size), dtype=bool).to(self.device)
        for i in range (batch_size):
            mask[i, i] = 0
            mask[i, i + batch_size] = 0
        mask = torch.cat([mask, mask], dim=0)
    
        for k in range(self.num_channels):
            
            graph_embedding_k = info['graph_embedding'][:, :, k]
            graph_embedding_1 = info_1['graph_embedding'][:, :, k]
            graph_embedding_2 = info_2['graph_embedding'][:, :, k]
            # graph_embedding_1 = info_1['graph_projection'][:, :, k]
            # graph_embedding_2 = info_2['graph_projection'][:, :, k]
            
            graph_embedding_all = torch.cat([graph_embedding_k, graph_embedding_k], dim=0)
            
            sim_neg = (graph_embedding_all.unsqueeze(0) * graph_embedding_all.unsqueeze(1)).sum(dim=-1)
            sim_neg_exp = torch.exp(sim_neg / self.contrastive_temp)
            sim_neg_exp = sim_neg_exp.masked_select(mask).view(2 * batch_size, -1)
            
            sim_pos_1 = (graph_embedding_1 * graph_embedding_k).sum(dim=-1)
            sim_pos_1_exp = torch.exp(sim_pos_1 / self.contrastive_temp)
            sim_pos_2 = (graph_embedding_2 * graph_embedding_k).sum(dim=-1)
            sim_pos_2_exp = torch.exp(sim_pos_2 / self.contrastive_temp)
            sim_pos = (graph_embedding_1 * graph_embedding_2).sum(dim=-1)
            sim_pos_exp = torch.exp(sim_pos / self.contrastive_temp)
            sim_pos_exp = torch.cat([
                torch.stack([sim_pos_exp, sim_pos_1_exp], dim=-1).sum(dim=-1),
                torch.stack([sim_pos_exp, sim_pos_2_exp], dim=-1).sum(dim=-1),
            ], dim=0)
            
            N = batch_size * 2 - 2
            imp = (self.contrastive_beta * sim_neg_exp.log()).exp()
            reweight_neg = (imp * sim_neg_exp).sum(dim=-1) / imp.mean(dim=-1)
            Neg = (-N*self.contrastive_tau*sim_pos_exp + sim_neg_exp.sum(dim=-1)) / (1-self.contrastive_tau)
            Neg = torch.clamp(Neg, min=N*np.e**(-1/self.contrastive_temp))
            # Neg = sim_neg_exp.sum(dim=-1)
            loss_cont += (1 / self.num_channels) * torch.mean(-torch.log(sim_pos_exp / (sim_pos_exp + Neg)))
            
        return loss_cont
    
    def training_explanation(self, 
                             data: Data,
                             info: dict,
                             batch_size: int
                             ) -> float:
        
        loss_expl = 0.0
        
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
            
            """
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
            """
            
            values_true = torch.cat([
                (out_true < self.regression_reference), 
                (out_true > self.regression_reference),
            ], 
            axis=1).float()
            # values_pred: (B, K)
            values_pred = torch.sigmoid(3 * (pooled_importance - self.importance_offset))
            
            loss_expl = F.binary_cross_entropy(values_pred, values_true)
            
        elif self.importance_mode == 'classification':
            
            # values_true: (B, K)
            values_true = out_true
            # values_pred: (B, K)
            values_pred = torch.sigmoid(3 * (pooled_importance - self.importance_offset))
            
            loss_expl = F.binary_cross_entropy(values_pred, values_true)
                        
        return loss_expl
    
    def configure_optimizers(self):
        """
        This method returns the optimizer to be used for the training of the model.
        
        :returns: A ``torch.optim.Optimizer`` instance
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def leave_one_out_deviations(self,
                                 graphs: t.List[tv.GraphDict]
                                 ) -> np.ndarray:
        result = np.zeros((len(graphs), self.num_channels, self.out_dim))
        loader = self._loader_from_graphs(graphs)
        
        for data in loader:
            
            for channel_index in range(self.num_channels):
                node_mask = None
                out_mod = self.predict_graphs(graphs)