import os
import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.layers.modules import DenseEmbedding
from kgcnn.layers.conv.gcn_conv import GCN
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from kgcnn.layers.conv.gin_conv import GINE
from kgcnn.layers.pooling import PoolingNodes

from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.layers import MultiHeadGATV2Layer


class GCNModel(ks.models.Model):
    
    def __init__(self,
                 conv_units: t.List[int],
                 dense_units: t.List[int],
                 pooling_method: str = 'sum',
                 final_activation: str = 'linear',
                 **kwargs):
        ks.models.Model.__init__(self, **kwargs)
        
        self.conv_units = conv_units
        self.dense_units = dense_units
        self.pooling_method = pooling_method
        self.final_activation = final_activation
        
        self.conv_layers = []
        for units in self.conv_units:
            lay = GCN(
                units=units,
                pooling_method='sum',
            )
            self.conv_layers.append(lay)
            
        self.lay_pooling = PoolingNodes(pooling_method=pooling_method)
        
        self.dense_layers = []
        self.dense_activations = ['relu' for _ in dense_units]
        self.dense_activations[-1] = final_activation
        for units, act in zip(dense_units, self.dense_activations):
            lay = DenseEmbedding(
                units=units,
                activation=act
            )
            self.dense_layers.append(lay)
            
    def get_config(self):
        return {
            'conv_units': self.conv_units,
            'dense_units': self.dense_units,
            'pooling_method': self.pooling_method,
            'final_activation': self.final_activation,
        }
        
    def embedd(self,
               inputs: t.Tuple,
               training: bool = False,
               **kwargs):
        node_input, edge_input, edge_indices = inputs
        edge_input = tf.expand_dims(tf.ones_like(edge_indices, dtype=tf.float32)[:, :, 0], axis=-1)
        
        node_embedding = node_input
        for lay in self.conv_layers:
            node_embedding = lay([node_embedding, edge_input, edge_indices])
            
        graph_embedding = self.lay_pooling(node_embedding)
        
        return graph_embedding
        
    def call(self, 
             inputs: t.Tuple, 
             **kwargs):
        
        graph_embedding = self.embedd(inputs)
        
        output = graph_embedding
        for lay in self.dense_layers:
            output = lay(output)
            
        return output
    
    def predict_graphs(self, graphs: t.List[dict]):
        x = tensors_from_graphs(graphs)
        out = self.call(x)
        return out.numpy()
    
    def embedd_graphs(self, graphs: t.List[dict]):
        x = tensors_from_graphs(graphs)
        graph_embeddings = self.embedd(x)
        return graph_embeddings.numpy()
    
    
class GATv2Model(ks.models.Model):
    
    def __init__(self,
                 conv_units: t.List[int],
                 dense_units: t.List[int],
                 pooling_method: str = 'sum',
                 final_activation: str = 'linear',
                 **kwargs):
        ks.models.Model.__init__(self, **kwargs)
        
    def __init__(self,
                 conv_units: t.List[int],
                 dense_units: t.List[int],
                 pooling_method: str = 'sum',
                 final_activation: str = 'linear',
                 **kwargs):
        ks.models.Model.__init__(self, **kwargs)
        
        self.conv_units = conv_units
        self.dense_units = dense_units
        self.pooling_method = pooling_method
        self.final_activation = final_activation
        
        self.conv_layers = []
        for units in self.conv_units:
            lay = MultiHeadGATV2Layer(
                units=units,
                heads=1,
                pooling_method='sum',
                concat_heads=False,
                concat_self=True,
            )
            self.conv_layers.append(lay)
            
        self.lay_pooling = PoolingNodes(pooling_method=pooling_method)
        
        self.dense_layers = []
        self.dense_activations = ['relu' for _ in dense_units]
        self.dense_activations[-1] = final_activation
        for units, act in zip(dense_units, self.dense_activations):
            lay = DenseEmbedding(
                units=units,
                activation=act
            )
            self.dense_layers.append(lay)
            
    def get_config(self):
        return {
            'conv_units': self.conv_units,
            'dense_units': self.dense_units,
            'pooling_method': self.pooling_method,
            'final_activation': self.final_activation,
        }
        
    def embedd(self,
               inputs: t.Tuple,
               training: bool = False,
               **kwargs):
        node_input, edge_input, edge_indices = inputs
        # edge_input = tf.expand_dims(tf.ones_like(edge_indices, dtype=tf.float32)[:, :, 0], axis=-1)
        
        node_embeddings = []
        node_embedding = node_input
        for lay in self.conv_layers:
            node_embedding = lay([node_embedding, edge_input, edge_indices])
            node_embeddings.append(node_embedding)
            
        # jumping skip connections
        # node_embedding = tf.reduce_max(tf.stack(node_embeddings, axis=-1), axis=-1)
            
        graph_embedding = self.lay_pooling(node_embedding)
        
        return graph_embedding
        
    def call(self, 
             inputs: t.Tuple, 
             **kwargs):
        
        graph_embedding = self.embedd(inputs)
        
        output = graph_embedding
        for lay in self.dense_layers:
            output = lay(output)
            
        return output
    
    def predict_graphs(self, graphs: t.List[dict]):
        x = tensors_from_graphs(graphs)
        out = self.call(x)
        return out.numpy()
    
    def embedd_graphs(self, graphs: t.List[dict]):
        x = tensors_from_graphs(graphs)
        graph_embeddings = self.embedd(x)
        return graph_embeddings.numpy()