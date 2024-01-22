"""
This module contains visualization utility functions.

"""
import os
import logging
import tempfile
import typing as t
from collections import defaultdict
from typing import List, Dict, Callable, Optional

import imageio.v2 as imageio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from imageio.v2 import imread
from umap import AlignedUMAP

from graph_attention_student.utils import NULL_LOGGER
from graph_attention_student.typing import GraphDict, RgbList


reds_cmap: mcolors.Colormap = mcolors.LinearSegmentedColormap.from_list(
    'reds',
    [
        '#FFFFFF',
        '#FF6F00',
        '#8E0000'
    ]
)

def generate_contrastive_colors(num: int) -> t.List[str]:
    """
    Generate a list with a given ``num`` of matplotlib color tuples which have the highest contrast 
    
    :returns: a list of lists where each list contains the RGB float values for a color
    """
    hues = np.linspace(0, 0.9, num)
    colors = mcolors.hsv_to_rgb([[h, 0.7, 0.9] for h in hues])
    return colors


# == MISC. VISUALIZATIONS ==

def plot_embeddings_2d(embeddings: np.ndarray,
                       ax: plt.Axes,
                       color: str = 'black',
                       label: t.Optional[str] = None,
                       x_range: t.Optional[tuple] = None,
                       y_range: t.Optional[tuple] = None,
                       size: int = 8,
                       **kwargs
                       ) -> None:
    if x_range is None:
        x_range = np.min(embeddings[:, 0]), np.max(embeddings[:, 0])
    if y_range is None:
        y_range = np.min(embeddings[:, 1]), np.max(embeddings[:, 1])
    
    x_min, x_max = x_range
    y_min, y_max = y_range

    xs, ys = embeddings[:, 0], embeddings[:, 1]
    
    ax.scatter(
        xs, ys,
        c=color,
        label=label,
        edgecolors='none',
        s=size,
        alpha=0.8,
    )
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def plot_embeddings_3d(embeddings: np.ndarray,
                       ax: plt.Axes,
                       color: str = 'black',
                       scatter_kwargs: dict = {},
                       shadow_color: str = 'lightgray',
                       shadow_alpha: float = 0.1,
                       label: t.Optional[str] = None,
                       x_range: t.Optional[tuple] = None,
                       y_range: t.Optional[tuple] = None,
                       z_range: t.Optional[tuple] = None,
                       size: int = 8,
                       **kwargs
                       ) -> None:
    
    if x_range is None:
        x_range = np.min(embeddings[:, 0]), np.max(embeddings[:, 0])
    if y_range is None:
        y_range = np.min(embeddings[:, 1]), np.max(embeddings[:, 1])
    if z_range is None:
        z_range = np.min(embeddings[:, 2]), np.max(embeddings[:, 2])
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    
    xs, ys, zs = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
    
    ax.scatter(xs, ys, z_min, c=shadow_color, zorder=-10, edgecolors='none', alpha=shadow_alpha, s=size)
    ax.scatter(xs, y_max, zs, c=shadow_color, zorder=-10, edgecolors='none', alpha=shadow_alpha, s=size)
    ax.scatter(x_min, ys, zs, c=shadow_color, zorder=-10, edgecolors='none', alpha=shadow_alpha, s=size)
    
    ax.scatter(
        xs, ys, zs,
        c=color,
        **scatter_kwargs,
        label=label,
        zorder=0,
        edgecolors='none',
        s=size,
    )
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])


def create_embeddings_pdf(embeddings: np.ndarray,
                          output_path: str,
                          title: t.Optional[str | list] = None,
                          colors: t.Optional[list] = None,
                          color_map: t.Optional[mcolors.Colormap] = None,
                          logger: logging.Logger = NULL_LOGGER,
                          num_neighbors: int = 50,
                          num_components: int = 2,
                          metric: str = 'euclidean',
                          umap_kwargs: dict = {},
                          scatter_kwargs: dict = {},
                          animation_path: t.Optional[str] = None,
                          ) -> None:
    """
    This function can be used to visualize a series of vector embeddings as represented by the ``embeddings`` array 
    parameter. This visualization will be done in the form of a multi page PDF file containing one plot per page and 
    one page per plot in the series of embeddings. The pdf will be saved to the absolute string path ``output_path``.
    
    The main point of this function is that it will appropriately visualize a (time) series consisting of multiple 
    different embedding arrays - therefore suitable to visualize for example the evolution of an embedding space over time.
    The ``embeddings`` array should be of the shape (T, N, D) where T is the number of distinct (time) steps; N the number of 
    elements in the dataset and D the embedding dimensionality.
    
    **DIMENSIONALITY**
    
    The function is able to plot 2D and 3D embeddings nativley. If the dimensionality of the embeddings is however larger 
    then a dimensionality reduction method using AlignedUMAP mapper will be applied first to reduce the dimensionality to 
    ``num_components`` dimensions, which can be either 2 or 3.
    
    :params embeddings: array of the shape (T, N, D)
    
    :returns: None
    """
    logger.info(f'creating embedding PDF for the given embeddings of shape {embeddings.shape}')
    
    # 1. (Optional) dimensionality reduction
    # If the embeddings are in a high dimensional space, they can't be visualized effectively, which is why 
    # a dimensionality reduction method is applied in this first step to transform them into a lower dimensional 
    # space.
    # Specifically, the dimensionality reduction method that we are using here is AlignedUMAP. It is important to 
    # use the aligned version here because we want each time step to be somewhat comparable to the previous one!
    
    num_steps = embeddings.shape[0]
    num_elements = embeddings.shape[1]
    num_dimensions = embeddings.shape[2]
    
    if num_dimensions > 3:
        logger.info(f'reducing embedding dimension {num_dimensions} -> {num_components}')

        mapper = AlignedUMAP(
            n_neighbors=num_neighbors,
            n_components=num_components,
            metric=metric,
            alignment_window_size=2,
            alignment_regularisation=1e-3,
            min_dist=0.0,
            repulsion_strength=100.0,
            **umap_kwargs,
        )
        mapped = mapper.fit_transform(
            [embeddings[t, :, :] for t in range(num_steps)],
            # The AlignedUMAP needs one additional parametere besided the actual embeddings, which is this "relations" list.
            # this is supposed to be a list of T-1 dictionaries, which each describe the relationship mapping from one 
            # time step to the next. The values are the INTEGER INDICES of the elements in the (relatively seen) current 
            # embeddings array and the values are the integer indices of the SAME elements within the next time step. 
            # In this case it is very trivial since the order of the elements does not change.
            relations=[{i:i for i in range(num_elements)} for t in range(num_steps - 1)]
        )
        embeddings = np.concatenate([np.expand_dims(arr, axis=0) for arr in mapped], axis=0)
        logger.info(f' * new embeddings shape: {embeddings.shape}')

    # 2. Actually plot the embeddings
    # After the optional dimensionality reduction has been completed we can then actually plot the 
    # embeddings.
    
    num_dimensions = embeddings.shape[2]
    
    # ~ optional title argument
    # optionally, it is possible to pass the "title" argument to the function as well. This may be a single string - in which 
    # case that string will be used as the title for all of the individual pages. The argument may also be a list of strings 
    # in which case each string will be used as the plot title for the page with the same index.
    # In any case, we construct a list of fixed format here so that we dont have to check for edge cases during the loop.
    if isinstance(title, list):
        assert len(title) == num_steps, 'the number of plot titles has to match the number of steps in the embeddings array!'
        titles = title
    elif isinstance(title, str):
        titles = [title for _ in range(num_steps)]
    else:
        titles = ['' for _ in range(num_steps)]
    
    # ~ determine the plot bounds
    x_min, x_max = np.min(embeddings[:, :, 0]), np.max(embeddings[:, :, 0])
    y_min, y_max = np.min(embeddings[:, :, 1]), np.max(embeddings[:, :, 1])
    if num_dimensions == 3:
        z_min, z_max = np.min(embeddings[:, :, 2]), np.max(embeddings[:, :, 2])
    
    axs: t.List[plt.Axes] = []
    logger.info('plotting embeddings...')
    with PdfPages(output_path) as pdf, tempfile.TemporaryDirectory() as tmp_path:
        
        # In this structure we will collect the absolute string paths to the PNG images that act as the individual frames 
        # for the potential animation. These frames will be stored in the tmp_path
        frame_paths: t.List[str] = []
        
        for t in range(num_steps):
            fig = plt.figure(figsize=(10, 10))
            c = 'lightgray' if colors is None else colors[t]
            
            if num_dimensions == 2:
                ax = fig.add_subplot(1, 1, 1)
                ax.scatter(
                    embeddings[t, :, 0], embeddings[t, :, 1],
                    c=c,
                    **scatter_kwargs,
                )
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])
                
            if num_dimensions == 3:
                ax = fig.add_subplot(1, 1, 1, projection='3d')
                xs, ys, zs = embeddings[t, :, 0], embeddings[t, :, 1], embeddings[t, :, 2],
                ax.scatter(
                    xs, ys, zs,
                    c=c,
                    **scatter_kwargs,
                )
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])
                ax.set_zlim([z_min, z_max])
                
                # ax.contour(xs, ys, zs, zdir='z', offset=z_min, cmap='coolwarm')
                ax.scatter(xs, ys, z_min, c='lightgray', zorder=-10, edgecolors=None, alpha=0.5)
                ax.scatter(xs, y_max, zs, c='lightgray', zorder=-10, edgecolors=None, alpha=0.5)
                ax.scatter(x_min, ys, zs, c='lightgray', zorder=-10, edgecolors=None, alpha=0.5)
                
            ax.set_title(titles[t])
                
            # It is optionally possible to provide a color map to this function as well, which will then 
            # be used to display a color bar next to each of the plots.
            if color_map is not None:
                fig.colorbar(color_map, ax=ax)
                
            frame_path = os.path.join(tmp_path, f'frame_{t}.png')
            frame_paths.append(frame_path)
            fig.savefig(frame_path)
            
            pdf.savefig(fig)
            plt.close(fig)
            axs.append(ax)
            logger.info(f' * t = {t} done')
            
        if animation_path is not None:
            
            images = [imageio.imread(path) for path in frame_paths]
            imageio.mimsave(animation_path, images)


def map_arrays_to_colors(arrays: t.List[np.ndarray],
                         cmap: mcolors.Colormap = reds_cmap,
                         ) -> t.Tuple[list, mcolors.Normalize]:
    """
    Maps all the values of the given list of numpy ``arrays`` into a color using the given ``cmap`` color map.
    
    One could argue, that a color map could directly do this anyways. The main point of this function is that 
    all the individual arrays in the given list can be regarded as a time-series of arrays that are somewhat 
    connected and this function will choose the scaling for the color conversion globally. That means that the 
    same color map with the same scaling is used for all the arrays in the given list. The function also returns 
    the Normalize object that was used for this scaling besides the actual converted color values.
    
    :returns: a tuple (colors, norm) where colors is the list of color lists and norm is the matplotlib Normalize 
        instance that was used to transform the values into the normalized color range.
    """
    min_value = np.min([np.min(arr) for arr in arrays])
    max_value = np.max([np.max(arr) for arr in arrays])
    
    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)
    colors = [cmap(norm(arr)) for arr in arrays]

    return colors, norm


def plot_regression_value_distribution(values: np.ndarray,
                                       ax: plt.Axes,
                                       bins: int = 15,
                                       color: str = 'lightgray',
                                       line_color: str = 'black',
                                       ) -> dict:
    counts, bins, patches = ax.hist(values, bins=bins, color=color)
    max_count = np.max(counts)
    counts_normalized = counts / max_count

    for count, edge in zip(counts_normalized, bins):
        ax.text(edge, max_count * 0.02, f'{count:.2f}')

    value_min = np.min(values)
    value_max = np.max(values)
    value_mean = np.mean(values)
    value_std = np.std(values)

    ax.axvline(
        value_mean,
        color=line_color,
        ls='-',
        label=f'avg: {value_mean:.2f}',
    )
    ax.axhline(
        max_count / 2,
        xmin=value_mean - value_std,
        xmax=value_mean + value_std,
        color=line_color,
        ls='--',
        label=f'std: {value_std:.2f}'
    )
    ax.legend()

    return {
        'min': value_min,
        'max': value_max,
        'mean': value_mean,
        'std': value_std,
        'histogram': counts_normalized,
    }


def plot_leave_one_out_analysis(results: dict,
                                num_targets: int,
                                num_channels: int,
                                base_fig_size: int = 6,
                                num_bins: int = 20,
                                x_lim: t.Optional[tuple] = None,
                                ) -> plt.Figure:
    """
    Creates a visualization of ``leave_one_out_analysis`` fidelity results. The function will return the
    matplotlib Figure object which contains the new visualization. That visualization will be split over
    multiple subplots, where the row indicates the index of the output target value and the column indicates
    the importance channel. Each cell in the subplot grid will be a histogram of the deviations recoreded
    for that particular pairing of channel and target.

    :param results: A 3-layer nested dict obtained through the ``leave_one_out_analysis`` function.
    :param num_targets: The int number of outputs the model produces
    :param num_channels: The int number of explanation channels the model employs

    :returns: fig
    """
    fig, rows = plt.subplots(
        ncols=num_targets,
        nrows=num_channels,
        figsize=(base_fig_size * num_targets, base_fig_size * num_channels),
        squeeze=False,
    )

    for channel_index in range(num_channels):

        for target_index in range(num_targets):
            ax = rows[channel_index][target_index]
            
            if isinstance(results, dict):
                values = [data[target_index][channel_index] for data in results.values()]
            elif isinstance(results, np.ndarray):
                values = results[:, target_index, channel_index]
                
            mean = np.mean(values)
            std = np.std(values)

            ax.set_title(f'target: {target_index} - channel: {channel_index}')
            ax.hist(
                values,
                bins=num_bins,
                color='lightgray',
                range=x_lim,
            )
            y_min, y_max = ax.get_ylim()
            ax.vlines(
                x=mean,
                ymin=y_min,
                ymax=y_max,
                colors='black',
                linestyles='dashed',
                label=f'avg: {mean:.3f}'
            )
            ax.hlines(
                y=(y_max - y_min) / 2,
                xmin=mean - std,
                xmax=mean + std,
                colors='black',
                linestyles='solid',
                label=f'std: {std:.3f}',
            )
            ax.legend()

    return fig


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

# == DEPRECATED ===

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

