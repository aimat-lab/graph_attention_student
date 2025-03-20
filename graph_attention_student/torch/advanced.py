import tempfile
import typing as typ

import numpy as np
import matplotlib.pyplot as plt
import visual_graph_datasets.typing as tv
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background

from graph_attention_student.utils import array_normalize
from graph_attention_student.torch.megan import Megan



def explain_value(value: str,
                  model: Megan,
                  processing: ProcessingBase,
                  strategy: typ.Literal['split', 'joint'] = 'split',
                  fig_size: int = 10,
                  radius: int = 50,
                  thickness: int = 20,
                  color: str = 'lightgreen',
                  color_maps: list[str] = ['Reds', 'Blues'],
                  width: int = 1000,
                  height: int = 1000,
                  ) -> tuple[plt.Figure, list[plt.Axes], dict, dict]:
    """
    Given a domain specific string graph representation ``value``, a Megan model instance ``model`` 
    and a processing instance ``processing``, this function will process the given value into a 
    graph structure, perform a model forward pass to obtain the node and edge importance explanations 
    and then visualize these explanations on in a new mpl figure.
    
    Explanations are visualized in the "background" style, where the node importances are visualized 
    as colored circles behind the graph nodes and the edge importances are visualized as colored lines
    behind the graph edges. The opacity/color of these elements is controlled by the actual importance 
    value.
    
    :param value: The domain specific string graph representation. In the visual graph datasets 
        framework, each graph domain has to specify how the graphs can be represented as simple 
        strings. To provide an example, molecular graphs can be represented in the SMILES 
        string format.
    :param model: The Megan model instance to make the prediction and to generate the explanations.
    :param processing: The processing instance that is used to convert the domain specific string 
        into a full graph dict representation and to create the graph visualization image. This 
        processing instance must be the same which was used to train the model!
    :param strategy: The visualization strategy to use. The two supported strategies are 'split' 
        and 'joint'. 'split' will visualize the different explanation channels in separate subplots
        of the same figure, while 'joint' will visualize all the explanation channels in the same
        subplot of the same figure using different colors.
    :param fig_size: The size of the figure in inches.
    :param radius: The radius of the node importance circles.
    :param thickness: The thickness of the edge importance lines.
    :param color: The color of the node importance circles and edge importance lines. This parameter 
        is only used for the 'split' strategy. In this case, the same color is used for all the plots.
    :param color_maps: A list of mpl color maps (either string identifiers or ColorMap objects) which
        determine the color maps to be used for the different explanation channels of the model. 
        The number of color maps therefore has to be the same as the number of explanation channels 
        in the model. This parameter is only used for the 'joint' strategy.
    :param width: The width of the graph visualization image.
    :param height: The height of the graph visualization image.
    
    :returns: A tuple (fig, axs, info, graph)
        - fig: The figure object that contains the visualized explanations.
        - axs: A list of the individual axes objects that contain the visualized explanations.
        - info: The info dict that was returned by the model's forward pass.
        - graph: The graph dict representation of the processed value.
    """
    
    with tempfile.TemporaryDirectory() as path:
        
        processing.create(
            value=value,
            index='0',
            width=width,
            height=height,
            output_path=path,    
        )
        data = VisualGraphDatasetReader.read_element(path=path, name='0')
        graph = data['metadata']['graph']
        
        fig, axs, info = explain_graph(
            graph=graph,
            image_path=data['image_path'],
            model=model,
            strategy=strategy,
            fig_size=fig_size,
            radius=radius,
            thickness=thickness,
            color=color,
            color_maps=color_maps,
        )
        
    return fig, axs, info, graph


def explain_graph(graph: tv.GraphDict,
                  image_path: str,
                  model: Megan,
                  strategy: typ.Literal['split', 'joint'] = 'split',
                  fig_size: int = 10,
                  radius: int = 50,
                  thickness: int = 20,
                  color: str = 'lightgreen',
                  color_maps: list[str] = ['Reds', 'Blues'],
                  ) -> tuple[plt.Figure, list[plt.Axes], dict]:
    """
    Given a graph dict representation ``graph``, an absolute ``image_path`` to the graph's visualization
    and a Megan ``model``, this function performs a forward pass of the given graph to obtain the
    node and edge importance explanations, which are then subsequently plotted on top of the graph's
    visualization.
    
    The function returns a tuple of the figure object, a list of the individual axes objects and
    the info dict that was returned by the model's forward pass.
    
    Explanations are visualized in the "background" style, where the node importances are visualized
    as colored circles behind the graph nodes and the edge importances are visualized as colored lines
    behind the graph edges. The opacity/color of these elements is controlled by the actual importance
    value.
    
    :param graph: The graph dict representation of the graph to be explained. In particular, besides the
        usual node and edge features, this graph dict has to contain the ``node_positions`` key which
        contains the 2D pixel positions of the nodes in the graph visualization image.
    :param image_path: The absolute path to the image that visualizes the given ``graph``.
    :param model: The Megan model instance that is used to explain the given graph.
    :param strategy: The visualization strategy to use. The two supported strategies are 'split'
        and 'joint'. 'split' will visualize the different explanation channels in separate subplots
        of the same figure, while 'joint' will visualize all the explanation channels in the same
        subplot of the same figure using different colors.
    :param fig_size: The size of the figure in inches.
    :param radius: The radius of the node importance circles.
    :param thickness: The thickness of the edge importance lines.
    :param color: The color of the node importance circles and edge importance lines. This parameter
        is only used for the 'split' strategy. In this case, the same color is used for all the plots.
    :param color_maps: A list of mpl color maps (either string identifiers or ColorMap objects) which
        determine the color maps to be used for the different explanation channels of the model.
        The number of color maps therefore has to be the same as the number of explanation channels
        in the model. This parameter is only used for the 'joint' strategy.
        
    :returns: A tuple (fig, axs, info)
        - fig: The figure object that contains the visualized explanations.
        - axs: A list of the individual axes objects that contain the visualized explanations.
        - info: The info dict that was returned by the model's forward pass.
    """
    if strategy == 'split':
        return explain_graph_split(
            graph=graph,
            image_path=image_path,
            model=model,
            fig_size=fig_size,
            radius=radius,
            thickness=thickness,
            color=color,
        )

    if strategy == 'joint':
        return explain_graph_joint(
            graph=graph,
            image_path=image_path,
            model=model,
            fig_size=fig_size,
            radius=radius,
            thickness=thickness,
            color_maps=color_maps,
        )
        
    else:
        raise ValueError(f'Unknown strategy: {strategy}, the only supported strategies are "split" and "joint".')


def explain_graph_joint(graph: tv.GraphDict,
                        image_path: str,
                        model: Megan,
                        fig_size: int = 10,
                        radius: int = 50,
                        thickness: int = 20,
                        color_maps: list[plt.Colormap] = ['Reds', 'Blues'],
                        ) -> tuple[plt.Figure, list[plt.Axes], dict]:
    """
    Given a graph dict representation ``graph``, an absolute ``image_path`` to the graph's visualization
    and a Megan ``model``, this function performs a forward pass of the given graph to obtain the 
    node and edge importance explanations, which are then subsequently plotted on top of the graph's 
    visualization.
    This function in particular will visualize all the explanation channels of the model in the same 
    subplot of the same figure. Each of the different explanation channels will be visualized as a 
    different color. To determine the color associated with each explanation channel, the 
    ``color_maps`` list has to be provided. This list determines the mpl color map to be used 
    in the same order as the model's explanation channels.
    The function returns a tuple of the figure object, a list of the individual axes objects and 
    the info dict that was returned by the model's forward pass.
    
    Explanations are visualized in the "background" style, where the node importances are visualized 
    as colored circles behind the graph nodes and the edge importances are visualized as colored lines
    behind the graph edges. The opacity/color of these elements is controlled by the actual importance 
    value.
    
    :param graph: The graph dict representation of the graph to be explained. In particular, besides the 
        usual node and edge features, this graph dict has to contain the ``node_positions`` key which 
        contains the 2D pixel positions of the nodes in the graph visualization image.
    :param image_path: The absolute path to the image that visualizes the given ``graph``.
    :param model: The Megan model instance that is used to explain the given graph.
    :param fig_size: The size of the figure in inches.
    :param radius: The radius of the node importance circles.
    :param thickness: The thickness of the edge importance lines.
    :param color_maps: A list of mpl color maps (either string identifiers or ColorMap objects) which
        determine the color maps to be used for the different explanation channels of the model. 
        The number of color maps therefore has to be the same as the number of explanation channels 
        in the model.

    :returns: A tuple (fig, axs, info)
        - fig: The figure object that contains the visualized explanations.
        - axs: A list of the individual axes objects that contain the visualized explanations.
        - info: The info dict that was returned by the model's forward pass.
    """

    num_channels = model.num_channels
    
    # First of all we need to make sure that the number of cmaps matches the number of channels 
    # of the model. If this does not match we cant properly create the visualization.
    assert len(color_maps) == model.num_channels, (f'The number of colormaps ({len(color_maps)}) must match the number '
                                                   f'of explanation channels of the model ({num_channels})')
    
    info = model.forward_graph(graph)
    
    node_importances = array_normalize(info['node_importance'])
    edge_importances = array_normalize(info['edge_importance'])
    
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(fig_size, fig_size),
    )
    
    draw_image(
        ax=ax,
        image_path=image_path,
        remove_ticks=True,
    )
    
    for channel_index in range(num_channels):
        
        plot_node_importances_background(
            ax=ax,
            g=graph,
            node_positions=graph['node_positions'],
            node_importances=node_importances[:, channel_index],
            color_map=color_maps[channel_index],
            radius=radius,
        )
        
        plot_edge_importances_background(
            ax=ax,
            g=graph,
            node_positions=graph['node_positions'],
            edge_importances=edge_importances[:, channel_index],
            color_map=color_maps[channel_index],
            thickness=thickness,
        )
        
    return fig, [ax], info
    


def explain_graph_split(graph: tv.GraphDict,
                        image_path: str,
                        model: Megan,
                        fig_size: int = 10,
                        radius: int = 50,
                        thickness: int = 20,
                        color: str = 'lightgreen',
                        ) -> tuple[plt.Figure, list[plt.Axes], dict]:
    """
    Given a graph dict representation ``graph``, an absolute ``image_path`` to the graph's visualization
    and a Megan ``model``, this function performs a forward pass of the given graph to obtain the 
    node and edge importance explanations, which are then subsequently plotted on top of the graph's 
    visualization.
    This function in particular will visualize all the explanation channels of the model as separate 
    subplots of the same figure.
    The function returns a tuple of the figure object, a list of the individual axes objects and 
    the info dict that was returned by the model's forward pass.
    
    Explanations are visualized in the "background" style, where the node importances are visualized 
    as colored circles behind the graph nodes and the edge importances are visualized as colored lines
    behind the graph edges. The opacity/color of these elements is controlled by the actual importance 
    value.
    
    :param graph: The graph dict representation of the graph to be explained. In particular, besides the 
        usual node and edge features, this graph dict has to contain the ``node_positions`` key which 
        contains the 2D pixel positions of the nodes in the graph visualization image.
    :param image_path: The absolute path to the image that visualizes the given ``graph``.
    :param model: The Megan model instance that is used to explain the given graph.
    :param fig_size: The size of the figure in inches.
    :param radius: The radius of the node importance circles.
    :param thickness: The thickness of the edge importance lines.
    :param color: The color of the node importance circles and edge importance lines.

    :returns: A tuple (fig, axs, info)
        - fig: The figure object that contains the visualized explanations.
        - axs: A list of the individual axes objects that contain the visualized explanations.
        - info: The info dict that was returned by the model's forward pass.
    """
    num_channels = model.num_channels
    info = model.forward_graph(graph)
    
    node_importances = array_normalize(info['node_importance'])
    edge_importances = array_normalize(info['edge_importance'])

    fig, rows = plt.subplots(
        ncols=num_channels,
        nrows=1,
        figsize=(fig_size * num_channels, fig_size),
        squeeze=False,
    )
    
    axs: list[plt.Axes] = []
    for channel_index in range(num_channels):
        ax = rows[0][channel_index]
        
        draw_image(
            ax=ax,
            image_path=image_path,
            remove_ticks=True,
        )
        
        plot_node_importances_background(
            ax=ax,
            g=graph,
            node_positions=graph['node_positions'],
            node_importances=node_importances[:, channel_index],
            color=color,
            radius=radius,
        )
        
        plot_edge_importances_background(
            ax=ax,
            g=graph,
            node_positions=graph['node_positions'],
            edge_importances=edge_importances[:, channel_index],
            color=color,
            thickness=thickness,
        )

        ax.set_title(f'Channel {channel_index}')
        axs.append(ax)
        
    return fig, axs, info