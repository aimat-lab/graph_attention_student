"""Advanced utilities for MEGAN model analysis and reporting.

This module provides high-level utilities for generating comprehensive prediction reports
and visualizing explanations from MEGAN (Multi-Explanation Graph Attention Network) models.
It includes functionality for creating PDF reports with predictions, model statistics, and
explanation visualizations, as well as various strategies for visualizing graph explanations.

The main functions in this module are:
- `megan_prediction_report`: Generates a complete PDF report for a single prediction
- `explain_value`: Creates visualizations for explaining model predictions on string inputs
- `explain_graph`: Creates visualizations for explaining model predictions on graph dictionaries
- `explain_graph_joint`: Visualizes all explanation channels in a single plot with different colors
- `explain_graph_split`: Visualizes explanation channels in separate subplots
"""

import os
import time
import tempfile
import typing as typ
from typing import List, Dict, Tuple

import matplotlib as mpl
import weasyprint as wp
import numpy as np
import matplotlib.pyplot as plt
import visual_graph_datasets.typing as tv
import matplotlib.colors as mcolors
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import (
    plot_node_importances_background,
)
from visual_graph_datasets.visualization.importances import (
    plot_edge_importances_background,
)

from graph_attention_student.utils import get_version
from graph_attention_student.utils import TEMPLATE_ENV
from graph_attention_student.utils import array_normalize
from graph_attention_student.torch.megan import Megan
from graph_attention_student.visualization import positive_cmap, negative_cmap


def get_model_size_mb_simple(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024**2)
    return size_mb


def megan_prediction_report(
    value: str,
    model: Megan,
    processing: ProcessingBase,
    figsize: Tuple[int, int] = (10, 10),
    output_path: str = tempfile.tempdir,
    color_maps: List[str] = [negative_cmap, positive_cmap],
    vis_width: int = 1000,
    vis_height: int = 1000,
    unit: typ.Optional[str] = None,
) -> Tuple[str, Dict, Dict]:
    """Generate a comprehensive PDF prediction report for a MEGAN model.

    This function creates a detailed PDF report that includes the model's prediction,
    explanation visualizations, fidelity analysis, and model statistics. The report
    is generated from an HTML template and styled with CSS, then converted to PDF
    using weasyprint.

    The function processes a domain-specific string representation (e.g., SMILES for
    molecules) through the provided processing pipeline, makes a prediction using the
    MEGAN model, generates explanation visualizations, and compiles everything into
    a professional PDF report.

    :param value: Domain-specific string representation of the graph (e.g., SMILES
        string for molecular graphs). This will be processed into a graph structure
        using the provided processing instance.
    :param model: The trained MEGAN model instance used for making predictions and
        generating explanations. The model should be in evaluation mode.
    :param processing: Processing instance that converts the string representation
        to a graph dictionary and creates visualizations. Must be the same processing
        instance used during model training.
    :param figsize: Size of the explanation visualization figure as (width, height)
        tuple in inches. Default is (10, 10).
    :param output_path: Path where the PDF report will be saved. If a directory is
        provided, the report will be saved as 'megan_report.pdf' in that directory.
        If a file path is provided, it will be used as the exact output path.
    :param color_maps: List of matplotlib colormaps to use for visualizing different
        explanation channels. Should match the number of channels in the model.
    :param vis_width: Width in pixels for the graph visualization image. Default 1000.
    :param vis_height: Height in pixels for the graph visualization image. Default 1000.
    :param unit: Optional unit string to display with the prediction value (e.g., 'mg/L',
        'eV'). If None, no unit will be shown.

    :returns: The rendered HTML content as a string. The PDF report is saved to the
        specified output_path.

    :raises: Various exceptions may be raised during processing, model inference,
        visualization creation, or PDF generation. These should be handled by the caller.

    The generated report includes:
    - Prediction value and confidence metrics
    - Model architecture and performance statistics
    - Explanation visualizations showing node and edge importance
    - Fidelity analysis using leave-one-out deviations
    - Statistical summaries of explanation channels
    """
    mpl.style.use('default')
    model.eval()

    # --- model prediction ---
    # At first we need to do perform the model forward pass to obtain the prediction
    # for the given sample. First the model needs to process the given domain specific
    # string representation into a full graph dict representation using the provided
    # processing instance, which is then passed to the model to obtain the prediction
    # and the explanations.
    # We record the start and end time to we can calculate the inference time later on.

    time_start = time.time()
    graph: dict = processing.process(value)
    results: dict = model.forward_graph(graph)
    time_end = time.time()

    # Depending on the prediction mode of the model, we need to extract the actual 
    # prediction result in a different way.
    if model.prediction_mode == 'classification':
        prediction = np.argmax(results['graph_output'])
    elif model.prediction_mode == 'bce':
        prediction = int(results['graph_output'].item() > 0.5)
    elif model.prediction_mode == 'regression':
        prediction = float(results['graph_output'].item())

    # Same goes for the fidelity values, which we calculate using the leave-one-out
    # deviations of the model.
    # leave_one_out: (num_output, num_channels) - provides the deviation in the output
    # when leaving out each of the channels.
    leave_one_out: np.ndarray = model.leave_one_out_deviations([graph])[0]
    if model.prediction_mode == 'regression':
        fidelity: List[float] = [-leave_one_out[0][0], leave_one_out[0][1]]
    else:
        fidelity: List[float] = [leave_one_out[i][i] for i in range(model.num_channels)]

    # --- create visualizations ---
    # The next step is to create the actual visualizations for the report.

    # We need to create some artifcat files and we do that in a temporary directory
    # which is then deleted again after we are done.
    with tempfile.TemporaryDirectory() as path:
        
        processing.create(
            value=value,
            index='0',
            width=vis_width,
            height=vis_height,
            output_path=path,
        )
        data = VisualGraphDatasetReader.read_element(path=path, name='0')
        graph = data['metadata']['graph']

        fig, axs, info = explain_graph(
            graph=graph,
            image_path=data['image_path'],
            model=model,
            strategy='joint',
            fig_size=figsize[0],
            radius=50,
            thickness=20,
            color_maps=color_maps,
        )

        explanation_path = os.path.join(path, 'explanation.png')
        fig.savefig(explanation_path, bbox_inches='tight', dpi=600)
        plt.close(fig)

        # --- creating html ---
        # Finally, we need to create the actual report by using the weasyprint package
        # to render an HTML template into a PDF document. The HTML template is dynamically
        # filled with the relevant information about the model, the prediction and the
        # explanation.

        # Load HTML template
        html_template = TEMPLATE_ENV.get_template('megan_prediction_report.html.j2')
        html_content = html_template.render(
            # Information about the prediction only
            prediction={
                'mode': model.prediction_mode,
                'value': prediction,
                'unit': unit,
            },
            # Information about the explanation
            explanation={
                'image_path': explanation_path,
                'statistics': [
                    {
                        'node': {
                            'min': float(
                                np.min(info['node_importance'][:, channel_index])
                            ),
                            'max': float(
                                np.max(info['node_importance'][:, channel_index])
                            ),
                            'mean': float(
                                np.mean(info['node_importance'][:, channel_index])
                            ),
                        },
                        'edge': {
                            'min': float(
                                np.min(info['edge_importance'][:, channel_index])
                            ),
                            'max': float(
                                np.max(info['edge_importance'][:, channel_index])
                            ),
                            'mean': float(
                                np.mean(info['edge_importance'][:, channel_index])
                            ),
                        },
                    }
                    for channel_index in range(model.num_channels)
                ],
                'contribution': [
                    fidelity[channel_index]
                    for channel_index in range(model.num_channels)
                ],
            },
            # Information about the model
            model={
                'name': model.__class__.__name__,
                'version': get_version(),
                'channels': list(range(model.num_channels)),
                'colors': [mcolors.to_hex(cmap(1.0)) for cmap in color_maps],
                'num_parameters': sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                'size': f'{get_model_size_mb_simple(model):.2f}',
                'inference_time': f'{(time_end - time_start)*1000:.2f}',
            },
        )
        html_doc = wp.HTML(string=html_content)

        # Load CSS stylesheet
        css_template = TEMPLATE_ENV.get_template('megan_prediction_report.css.j2')
        css_content = css_template.render()
        css_doc = wp.CSS(string=css_content)

        # --- saving pdf ---
        
        # If output_path is a directory, we save the report as 'megan_report.pdf' in that directory.
        # Otherwise we assume that the path that is given is supposed to be the actual file path
        # under which the report should be created.
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, 'megan_report.pdf')

        html_doc.write_pdf(output_path, stylesheets=[css_doc])

    return html_content


def explain_value(
    value: str,
    model: Megan,
    processing: ProcessingBase,
    strategy: typ.Literal['split', 'joint'] = 'split',
    fig_size: int = 10,
    radius: int = 50,
    thickness: int = 20,
    color: str = 'lightgreen',
    color_maps: List[str] = [positive_cmap, negative_cmap],
    width: int = 1000,
    height: int = 1000,
) -> Tuple[plt.Figure, List[plt.Axes], Dict, Dict]:
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


def explain_graph(
    graph: tv.GraphDict,
    image_path: str,
    model: Megan,
    strategy: typ.Literal['split', 'joint'] = 'split',
    fig_size: int = 10,
    radius: int = 50,
    thickness: int = 20,
    color: str = 'lightgreen',
    color_maps: List[str] = ['Reds', 'Blues'],
) -> Tuple[plt.Figure, List[plt.Axes], Dict]:
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
        raise ValueError(
            f'Unknown strategy: {strategy}, the only supported strategies are "split" and "joint".'
        )


def explain_graph_joint(
    graph: tv.GraphDict,
    image_path: str,
    model: Megan,
    fig_size: int = 10,
    radius: int = 50,
    thickness: int = 20,
    color_maps: List[str] = ['Reds', 'Blues'],
) -> Tuple[plt.Figure, List[plt.Axes], Dict]:
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
    assert len(color_maps) == model.num_channels, (
        f'The number of colormaps ({len(color_maps)}) must match the number '
        f'of explanation channels of the model ({num_channels})'
    )

    # Set model to evaluation mode to avoid BatchNorm issues with single samples
    model.eval()
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


def explain_graph_split(
    graph: tv.GraphDict,
    image_path: str,
    model: Megan,
    fig_size: int = 10,
    radius: int = 50,
    thickness: int = 20,
    color: str = 'lightgreen',
) -> Tuple[plt.Figure, List[plt.Axes], Dict]:
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
    # Set model to evaluation mode to avoid BatchNorm issues with single samples
    model.eval()
    info = model.forward_graph(graph)

    node_importances = array_normalize(info['node_importance'])
    edge_importances = array_normalize(info['edge_importance'])

    fig, rows = plt.subplots(
        ncols=num_channels,
        nrows=1,
        figsize=(fig_size * num_channels, fig_size),
        squeeze=False,
    )

    axs: List[plt.Axes] = []
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
