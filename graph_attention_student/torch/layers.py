import typing as t

import torch
from torch._tensor import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import GINEConv
from torch_geometric.utils import softmax


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
            node_embedding = self.lay_act(node_embedding)
            
            node_embeddings.append(node_embedding)
            alphas.append(alpha)
    
        # node_embeddings: (B * V, out, K)
        node_embeddings = torch.stack(node_embeddings, dim=-1)
        
        # node_embeddings: (B * V, out)
        node_embeddings = self.aggregate(node_embeddings, dim=-1)
        
        # alphas: (B * E, K)
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
        #node_embedding = self.lay_transform(node_embedding)
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
        
        return self._attention * self.lay_linear(x_j)
    
    
class AttentiveFpLayer(AbstractAttentionLayer):
    
    def __init__(self, 
                 in_dim: int,
                 out_dim: int,
                 edge_dim: int,
                 **kwargs):
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
        
        self.lay_alignment = nn.Linear(
            in_features=(2 * in_dim) + edge_dim,
            out_features=1,
        )
        
        self.lay_recurrent = nn.GRUCell(
            input_size=out_dim,
            hidden_size=in_dim,
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
        
        context = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
        )
        context = F.elu(context)
        
        node_embedding = self.lay_recurrent(context, x)
        return node_embedding, self._attention_logits
        
    def message(self,
                x_i, x_j,
                edge_attr,
                index,
                ptr,
                size_i,
                ):
        
        # aligned: (B * E, out)
        aligned = self.lay_alignment(torch.cat([x_i, x_j, edge_attr], dim=-1))
        #aligned = F.leaky_relu(aligned)
        
        self._attention_logits = aligned
        self._attention = F.sigmoid(self._attention_logits)
        #self._attention = softmax(self._attention_logits, index, ptr, size_i)
        
        return self._attention * self.lay_linear(x_j)
    
    
class MaskExpansionLayer(MessagePassing):
    
    def __init__(self,
                 **kwargs,
                 ) -> None:
        super().__init__(aggr='max', **kwargs)
        
    def forward(self,
                mask: torch.Tensor,
                edge_index: torch.Tensor
                ) -> torch.Tensor:
        
        return self.propagate(
            edge_index,
            mask=mask,
        )
        
    def message(self,
                mask_i, mask_j,
                ) -> torch.Tensor:
        return mask_j