from collections import defaultdict
from typing import List, Dict, Callable, Optional
import typing as t

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from imageio.v2 import imread

from graph_attention_student.typing import GraphDict, RgbList


reds_cmap: mcolors.Colormap = mcolors.LinearSegmentedColormap.from_list(
    'reds',
    [
        '#FFFFFF',
        '#FF6F00',
        '#8E0000'
    ]
)


# == MISC. VISUALIZATIONS ==

def plot_regression_fit(values_true: t.Union[np.ndarray, t.List[float]],
                        values_pred: t.Union[np.ndarray, t.List[float]],
                        ax: plt.Axes,
                        num: int = 40,
                        cmap: t.Union[str, mcolors.Colormap] = reds_cmap,
                        plot_reference: bool = True,
                        reference_color: str = 'gray',
                        ):
    """
    Plots a regression plot in the Axes ``ax`` using the true target values ``values_true`` and the
    predicted values ``values_pred``. The plot is in the format of a heatmap, where ``num`` defines the
    granularity of that heat map in x and y direction of the plot.

    It is important that the arrays for the true and predicted values have the same number of elements!

    :param values_true: An array containing the ground truth float target values.
    :param values_pred: An array containing the predicted values which were created by some sort of
        regression model.
    :param ax: The matplotlib Axes on which to draw the plot
    :param num: The integer number which defines the granularity of the heatmap plot. The heatmap is
        discretized in x and y direction using binning with ``num`` bins.
    :param cmap: The matplotlib color map, to be used for the heatmap visualization.
    :param plot_reference: This flag determines if the regression reference line is also to be drawn onto
        the plot. The reference line is a line from the bottom left corner to the top right corner of the
        plot. If all the elements of the regression plot are exactly on that line, it is considered to be a
        perfect regression result.
    :param reference_color: The color to be used for the reference line.

    :return: The binned value map which is the basis for the heatmap.
    """
    # 27.03.2023 - Here we turn the input into a numpy array but more important is the squeeze operation,
    # which fixes a bug, where the function cannot be directly applied to the output of a neural network
    # as that has an additional (redundant) dimension for the output tensor.
    values_true = np.squeeze(np.array(values_true))
    values_pred = np.squeeze(np.array(values_pred))

    min_value = min(np.min(values_true), np.min(values_pred))
    max_value = max(np.max(values_true), np.max(values_pred))
    xs = np.linspace(min_value, max_value, num)
    ys = np.linspace(min_value, max_value, num)
    z, x_edges, y_edges = np.histogram2d(
        x=values_true,
        y=values_pred,
        bins=[xs, ys],
    )

    y, x = np.meshgrid(
        np.linspace(min_value, max_value, num),
        np.linspace(min_value, max_value, num)
    )
    ax.set_xlabel('true values')
    ax.set_xlim([min_value, max_value])
    ax.set_ylabel('predicted values')
    ax.set_ylim([min_value, max_value])
    c = ax.pcolormesh(
        x, y, z,
        linewidth=0,
        cmap=cmap,
        rasterized=True,
    )

    if plot_reference:
        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            color=reference_color,
            zorder=1,
        )

    return c

# == EXPLANATION VISUALIZATION ===

def plot_node_importances(g: dict,
                          node_importances: np.ndarray,
                          node_coordinates: np.ndarray,
                          ax: plt.Axes,
                          radius: float = 30,
                          thickness: float = 4,
                          color='black',
                          vmin: float = 0,
                          vmax: float = 1):
    node_normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for i, (x, y), ni in zip(g['node_indices'], node_coordinates, node_importances):
        circle = plt.Circle(
            (x, y),
            radius=radius,
            lw=thickness,
            color=color,
            fill=False,
            alpha=node_normalize(ni)
        )
        ax.add_artist(circle)


def plot_edge_importances(g: dict,
                          edge_importances: np.ndarray,
                          node_coordinates: np.ndarray,
                          ax: plt.Axes,
                          radius: float = 30,
                          thickness: float = 4,
                          color='black',
                          vmin: float = 0,
                          vmax: float = 1):
    edge_normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for (i, j), ei in zip(g['edge_indices'], edge_importances):
        coords_i = node_coordinates[i]
        coords_j = node_coordinates[j]
        # Here we determine the actual start and end points of the line to draw. Now we cannot simply use
        # the node coordinates, because that would look pretty bad. The lines have to start at the very
        # edge of the node importance circle which we have already drawn (presumably) at this point. This
        # circle is identified by it's radius. So what we do here is we reduce the length of the line on
        # either side of it by exactly this radius. We do this by first calculating the unit vector for that
        # line segment and then moving radius times this vector into the corresponding directions
        diff = (coords_j - coords_i)
        delta = (radius / np.linalg.norm(diff)) * diff
        x_i, y_i = coords_i + delta
        x_j, y_j = coords_j - delta

        ax.plot(
            [x_i, x_j],
            [y_i, y_j],
            color=color,
            lw=thickness,
            alpha=edge_normalize(ei)
        )


def pdf_from_eye_tracking_dataset(eye_tracking_dataset: List[dict],
                                  pdf_path: str,
                                  include_meta: bool = True,
                                  name: str = 'Eye Tracking Dataset'):

    with PdfPages(pdf_path) as pdf:

        is_classification = False
        if include_meta:
            fig, (ax_task, ax_sizes) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
            fig.suptitle(name)

            # This plot is meant to show the label distribution for the classification task
            is_classification = eye_tracking_dataset[0]['metadata']['input_type'] == 'classification'
            if is_classification:
                ax_task.set_title("Label Distribution")
                classes = eye_tracking_dataset[0]['metadata']['classes']
                label_counts = defaultdict(int)
                for data in eye_tracking_dataset:
                    label = data['metadata']['label']
                    label_counts[label] += 1
                xs = list(range(len(label_counts)))
                ax_task.bar(xs, label_counts.values(), color='lightgray')
                ax_task.set_xticks(xs)
                ax_task.set_xticklabels([classes[k] for k in label_counts.keys()])

            # This plot is meant to show the size distribution
            ax_sizes.set_title('Graph Size Distribution')
            sizes = [len(data['metadata']['graph']['node_indices']) for data in eye_tracking_dataset]
            hist, bin_edges = np.histogram(sizes, bins=10)
            xs = list(range(len(hist)))
            ax_sizes.bar(xs, hist, color='lightgray')
            ax_sizes.set_xticks(xs)
            ax_sizes.set_xticklabels([f'({int(bin_edges[i])},{int(bin_edges[i+1])})'
                                      for i in range(len(bin_edges) - 1)])

            pdf.savefig(fig)

        for index, data in enumerate(eye_tracking_dataset):
            g = data['metadata']['graph']
            image_path = data['image_path']
            image = imread(image_path)
            array = np.asarray(image)

            fig, (ax_img, ax_exp) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
            fig.suptitle(data['metadata']['name'] if 'name' in data['metadata'] else f'Element {index}')
            ax_img.imshow(array, extent=(0, array.shape[0], 0, array.shape[1]))
            if is_classification:
                label = classes[data["metadata"]["label"]]
                ax_img.set_title(f'Target: {label}')

            # The second graph is supposed to show the ground truth importances, if they exist
            ax_exp.set_title(f'Ground Truth Importances')
            ax_exp.imshow(array, extent=(0, array.shape[0], 0, array.shape[1]))
            if 'node_importances' in g:
                plot_node_importances(
                    g=g,
                    node_importances=g['node_importances'],
                    node_coordinates=np.array(g['node_coordinates']),
                    ax=ax_exp
                )
            if 'edge_importances' in g:
                plot_edge_importances(
                    g=g,
                    edge_importances=g['edge_importances'],
                    node_coordinates=np.array(g['node_coordinates']),
                    ax=ax_exp
                )

            pdf.savefig(fig)
            plt.close(fig)


def plot_distribution(ax: plt.Axes,
                      values: t.Union[t.List[float], np.ndarray],
                      bins: int,
                      show_mean: bool = True,
                      show_std: bool = True,
                      line_width: int = 2):
    histogram, bins, patches = ax.hist(values, bins=bins)
    mean = float(np.mean(values))
    std = float(np.std(values))

    # ~ plot the mean
    if show_mean:
        ax.axvline(
            x=mean,
            color='black',
            ls='--',
            linewidth=line_width,
            label=f'mean: {mean:.2f}'
        )

    # ~ Plot the standard deviation
    if show_std:
        y_min, y_max = ax.get_ylim()
        y_middle = np.mean([y_min, y_max])
        ax.plot(
            [mean - std, mean + std],
            [y_middle, y_middle],
            color='black',
            ls='-',
            linewidth=line_width,
            label=f'std: {std:.2f}'
        )

    return {
        'histogram': histogram,
        'bins': bins,
        'patches': patches,
        'mean': mean,
        'std': std
    }


# == COLORED GRAPHS VIS. ====================================================================================

def draw_node_color_scatter(ax: plt.Axes,
                            g: GraphDict,
                            i: int,
                            size: int = 500,
                            alpha: float = 1.0,
                            border_width: float = 2,
                            add_label: bool = False,
                            marker: str = 'o') -> None:
    # First of all we will draw a white circle underneath so that we can safely apply an alpha value for
    # the actual color whithout seeing through the edges potentially
    x, y = g['node_positions'][i]
    ax.scatter(
        x, y,
        s=size,
        color='white',
        marker=marker,
        zorder=0
    )

    rgb_list = g['node_attributes'][i][:3]
    color = mcolors.to_hex(rgb_list)
    ax.scatter(
        x, y,
        s=size,
        color=color,
        marker=marker,
        alpha=alpha,
        linewidths=border_width,
        edgecolors='black',
        zorder=1,
    )


def draw_edge_black(ax: plt.Axes,
                    g: GraphDict,
                    i: int,
                    j: int,
                    size: int = 2) -> None:
    x_i, y_i = g['node_positions'][i]
    x_j, y_j = g['node_positions'][j]

    ax.plot(
        [x_i, x_j],
        [y_i, y_j],
        lw=size,
        color='black',
        zorder=-1,
    )


def draw_extended_colors_graph(ax: plt.Axes,
                               g: GraphDict,
                               layout_cb: Callable[[nx.Graph], List[int]] = nx.layout.kamada_kawai_layout,
                               draw_node_cb: Callable[[plt.Axes, GraphDict, int], None] = draw_node_color_scatter,
                               draw_edge_cb: Callable[[plt.Axes, GraphDict, int, int], None] = draw_edge_black,
                               ):
    if 'node_positions' not in g:
        graph = nx.Graph()
        graph.add_nodes_from({i: {} for i in g['node_indices']})
        graph.add_edges_from(g['edge_indices'])

        pos = layout_cb(graph)
        g['node_positions'] = np.array(list(pos.values()))

    for i in g['node_indices']:
        draw_node_cb(ax, g, i)

    for (i, j) in g['edge_indices']:
        draw_edge_cb(ax, g, i, j)

    return ax, graph


# == NATURAL LANGUAGE PROCESSING (NLP) VIS. =================================================================

def fixed_row_layout(g: dict,
                     ncols: int = 5,
                     column_spacing: float = 0.5,
                     row_spacing: float = 0.3,
                     ) -> List[List[float]]:
    node_positions = []

    x, y = 0, 0
    for i, node_index in enumerate(g['node_indices']):
        node_positions.append([x, y])

        if i % ncols == 0 and i != 0:
            x = 0
            y -= row_spacing
        else:
            x += column_spacing

    return node_positions


def plot_text_graph(g: dict,
                    ax: plt.Axes,
                    node_importances: Optional[List[float]] = None,
                    edge_importances: Optional[List[float]] = None,
                    text_offset: float = 0.05,
                    node_size: float = 40,
                    node_color: str = 'black',
                    edge_color: str = 'black',
                    do_edges: bool = True,
                    vmin: float = 0.0,
                    vmax: float = 1.0,
                    ) -> None:
    colors = [
        [1, 1, 1, 1],
        [0, 1, 0, 1],
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list('', colors, 256)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i in g['node_indices']:
        x, y = g['node_positions'][i]

        ax.scatter(
            x, y,
            color='black',
            zorder=1,
            s=node_size
        )

        if node_importances is not None:
            value = norm(node_importances[i])
            text_color = cmap(value)
        else:
            text_color = 'white'

        ax.text(
            x, y - text_offset,
            s=g['node_strings'][i],
            ha='center',
            color=node_color,
            bbox={
                'boxstyle': 'round',
                'pad': 0.5,
                'ec': 'black',
                'fc': text_color,
            },
            zorder=1
        )

    if do_edges:

        edge_blacklist = set()
        for e, (i, j) in enumerate(g['edge_indices']):
            if (i, j) in edge_blacklist:
                continue

            x_i, y_i = g['node_positions'][i]
            x_j, y_j = g['node_positions'][j]
            edge_alpha = 1.0
            if edge_importances is not None:
                edge_alpha = norm(edge_importances[e])

            word_distance = abs(i - j)
            if word_distance == 1 or y_i != y_j:

                ax.plot(
                    [x_i, x_j],
                    [y_i, y_j],
                    color=edge_color,
                    zorder=-2,
                    alpha=edge_alpha,
                )

            elif word_distance > 1:
                arrow = mpatches.FancyArrowPatch(
                    (x_i, y_i),
                    (x_j, y_j),
                    connectionstyle=f'arc3, rad=-{.25 * word_distance}',
                    color=edge_color,
                    zorder=-2,
                    alpha=edge_alpha,
                )
                ax.add_patch(arrow)

            edge_blacklist.add((i, j))
            edge_blacklist.add((j, i))

