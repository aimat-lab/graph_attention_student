import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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