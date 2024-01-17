import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import GINEConv


NAME_ACTIVATION_MAP: t.Dict[str, t.Callable] = {
    None:       lambda: nn.Identity(),
    'linear':   lambda: nn.Identity(),
    'relu':     lambda: nn.ReLU(),
    'silu':     lambda: nn.SiLU(),
}


class AbstractAttentionLayer(MessagePassing):
    
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
        """
        raise NotImplementedError('A graph attention layer must implement the "forward" method '
                                  'to return a tuple of the new node attributes and edge attentions')
    

class MultiHeadAttention(nn.Module):
    
    def __init__(self,
                 layers: t.List[nn.Module],
                 activation: str = 'silu',
                 ):
        super().__init__()
        
        self.layers = nn.ModuleList(layers)
        
        self.lay_act = NAME_ACTIVATION_MAP[activation]()
        
    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_index: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        
        # In these two lists we will store the results of the individual attention heads outputs.
        # Each head will produce the transformed node features vector and the edge attention weights.
        node_embeddings: t.List[torch.Tensor] = []
        alphas: t.List[torch.Tensor] = []
        
        for lay in self.layers:
            node_embedding, alpha = lay(x, edge_attr, edge_index)
            node_embedding = self.lay_act(node_embedding)
            
            node_embeddings.append(node_embedding)
            alphas.append(alpha)
    
        # node_attributes: (B * V, out, K)
        node_embeddings = torch.cat([tens.unsqueeze(-1) for tens in node_embeddings], dim=-1)
        # node_attributes: (B * V, out)
        node_embeddings = torch.sum(node_embeddings, dim=-1)
        #node_embeddings = torch.amax(node_embeddings, dim=-1)
        
        # alphas: (B * V, K)
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
            aggr='add'
        )
        
        self.lay_linear = nn.Linear(
            in_features=in_dim,
            out_features=out_dim,
        )
        
        self.lay_attention = nn.Linear(
            in_features=(2 * out_dim) + edge_dim,
            out_features=1,
        )
        
        self.lay_act = nn.SiLU()
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
        node_embedding = self.lay_linear(x)
        
        self._attention = None
        self._attention_logits = None
        # node_embedding: (B * V, out)
        node_embedding = self.propagate(
            edge_index,
            x=node_embedding, 
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
        
        return self._attention * x_j