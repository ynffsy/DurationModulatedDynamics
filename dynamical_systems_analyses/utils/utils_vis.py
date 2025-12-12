import ipdb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib import gridspec
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import combinations
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from scipy.stats import wilcoxon, mannwhitneyu, sem, t, kruskal, ttest_rel
from statsmodels.stats.multitest import fdrcorrection



# Define a custom format function for “5×10^4” style ticks
def sci_notation_fmt(x, pos):
    """Format tick as M×10^E, e.g. 50000 => '5×10^4'"""
    if x == 0:
        return '0'
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / 10**exponent
    return r'${:.1f}\times 10^{{{}}}$'.format(mantissa, exponent)


def confidence_interval_95_unpaired(array1, array2):

    # Compute 95% CI of mean difference for unpaired data
    # 1. Compute sample means
    mean1 = np.mean(array1)
    mean2 = np.mean(array2)
    mean_diff = mean1 - mean2

    # 2. Compute sample variances
    var1 = np.var(array1, ddof=1)
    var2 = np.var(array2, ddof=1)

    n1 = len(array1)
    n2 = len(array2)

    # 3. Standard error (Welch's formula)
    se_diff = np.sqrt(var1/n1 + var2/n2)

    # 4. Degrees of freedom (Welch–Satterthwaite equation)
    df_numer = (var1/n1 + var2/n2)**2
    df_denom = (var1/n1)**2 / (n1 - 1) + (var2/n2)**2 / (n2 - 1)
    df = df_numer / df_denom  # effective degrees of freedom

    # 5. Critical t-value for a 95% CI
    alpha = 0.05
    t_crit = t.ppf(1 - alpha/2, df)

    # 6. Margin of error
    margin = t_crit * se_diff

    # 7. Confidence interval
    ci_lower = mean_diff - margin
    ci_upper = mean_diff + margin

    print(f"Difference of means: {mean_diff:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")


## Create a custom colormap from two colors
def make_colormap(color1, color2):
    return mcolors.LinearSegmentedColormap.from_list("custom", [color1, color2], N=100)


class Arrow3D(FancyArrowPatch):
    """
    A class to draw 3D arrows on a Matplotlib 3D plot.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """
        Initialize the Arrow3D object with 3D coordinates.

        Parameters:
            xs, ys, zs: Lists or arrays with two elements each, representing the start and end points.
            *args, **kwargs: Additional arguments passed to FancyArrowPatch.
        """
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        """
        Project the 3D coordinates to 2D for rendering.
        """
        xs3d, ys3d, zs3d = self._verts3d

        # Transform the 3D coordinates to 2D projections
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)

        # Update the positions
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        # Return the z-depth of the arrow for proper layering
        return min(zs)


def add_axes(
    ax,
    x_min, x_max,
    y_min, y_max,
    z_min, z_max,
    *,
    frac=20,
    lw=1.2,
    alpha=1.0,
    headsize=5,
    label_fontsize=8,
    labels=("L1", "L2", "L3"),
    show_labels=True,
):

    # -------------------------------------------------- 1. arrow geometry
    rngs       = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    arrow_len  = frac * rngs.min()

    # --- A: keep the tail just *inside* the box --------------------------
    eps       = 1e-6 * rngs.min()           # plenty small
    tail_x    = x_max - eps
    tail_y    = y_max - eps
    tail_z    = z_max - eps

    # (If you’d rather keep the tail exactly on the wall,
    #  set eps = 0 and rely on option B or C instead.)

    def _arrow(dx=0, dy=0, dz=0, text=None):
        arr = Arrow3D(
            [tail_x, tail_x + dx],
            [tail_y, tail_y + dy],
            [tail_z, tail_z + dz],
            mutation_scale=headsize,
            lw=lw,
            arrowstyle="-|>",
            color="black",
            alpha=alpha,
            shrinkA=0,
        )

        # --- C: ignore clipping for this artist --------------------------
        # arr.set_clip_on(False)          # ← uncomment if you like this route

        ax.add_artist(arr)
        if show_labels and text:
            ax.text(
                tail_x + dx * 1.1,
                tail_y + dy * 1.1,
                tail_z + dz * 1.1,
                text,
                fontsize=label_fontsize,
                va="center",
                ha="center",
            )

    _arrow( arrow_len, 0,          0,          labels[0])   # +X
    _arrow( 0,          arrow_len, 0,          labels[1])   # +Y
    _arrow( 0,          0,          arrow_len, labels[2])   # +Z

    # --- B: symmetric margin so nothing touches the walls ---------------
    margin = arrow_len * 0.05          # 5 % looks tidy
    ax.set_xlim(x_min - margin, x_max + arrow_len + margin)
    ax.set_ylim(y_min - margin, y_max + arrow_len + margin)
    ax.set_zlim(z_min - margin, z_max + arrow_len + margin)


def add_color_graded_trajectory(
    ax, 
    points, 
    colormap, 
    linewidth=2,
    alpha=0.8,
    t=None, 
    t_max=None):

    points_  = points.reshape(-1, 1, 2)
    segments = np.concatenate([points_[:-1], points_[1:]], axis=1)

    # Normalize 't' for color mapping
    # Here, instead of linearly spacing colors by segment index,
    # we directly use 't' values associated with each segment for color mapping.
    # We take the average 't' value of the endpoints for each segment.
    if t is None:
        t = np.arange(points_.shape[0])

    t_avg  = (t[:-1] + t[1:]) / 2
    if t_max is not None:
        t_norm = plt.Normalize(t.min(), t_max)

    else:
        t_norm = plt.Normalize(t.min(), t.max())

    # Create a LineCollection, mapping 't_avg' to colors
    lc = LineCollection(
        segments, 
        array=t_avg, 
        cmap=colormap, 
        norm=t_norm, 
        linewidths=linewidth,
        alpha=alpha,
        capstyle='round')

    ax.add_collection(lc)
    return lc


def add_color_graded_trajectory_3D(
    ax, 
    points, 
    colormap, 
    linewidth=2,
    alpha=0.8,
    t=None, 
    t_max=None):

    points_  = points.reshape(-1, 1, 3)
    segments = np.concatenate([points_[:-1], points_[1:]], axis=1)

    # Normalize 't' for color mapping
    # Here, instead of linearly spacing colors by segment index,
    # we directly use 't' values associated with each segment for color mapping.
    # We take the average 't' value of the endpoints for each segment.
    if t is None:
        t = np.arange(points_.shape[0])

    t_avg  = (t[:-1] + t[1:]) / 2
    if t_max is not None:
        t_norm = plt.Normalize(t.min(), t_max)

    else:
        t_norm = plt.Normalize(t.min(), t.max())

    # Create a LineCollection, mapping 't_avg' to colors
    lc = Line3DCollection(
        segments, 
        array=t_avg, 
        cmap=colormap, 
        norm=t_norm, 
        linewidths=linewidth,
        alpha=alpha,
        zorder=1,
        capstyle='round')

    ax.add_collection3d(lc)
    return lc


def add_dynamics_quiver(
    ax, 
    dynamics_mat, 
    dynamics_bias, 
    colormap, 
    n_points,
    x_min,
    x_max,
    y_min,
    y_max):

    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    xy_grid = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.zeros((n_points**2, 0))))
    dx = xy_grid.dot(dynamics_mat.T) + dynamics_bias - xy_grid
    
    ## Calculate the magnitude of velocity vectors
    magnitudes = np.sqrt(dx[:, 0]**2 + dx[:, 1]**2)
    
    ## Normalize magnitudes for color mapping
    norm = plt.Normalize(magnitudes.min(), magnitudes.max())
    
    ## Plotting the quiver plot with color encoding by magnitude
    quiver = ax.quiver(x_grid, y_grid, dx[:, 0], dx[:, 1], color=colormap(norm(magnitudes)))
    
    ## Create colorbar
    return plt.cm.ScalarMappable(norm=norm, cmap=colormap)


def add_dynamics_quiver_double(
    axs, 
    dynamic_params, 
    colormap, 
    n_points,
    x_min,
    x_max,
    y_min,
    y_max):

    x_grids, y_grids, dxs, magnitudes_all = [], [], [], []

    for i in range(2):
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
        xy_grid = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.zeros((n_points**2, 0))))
        dx = xy_grid.dot(dynamic_params[i].weights.T) + dynamic_params[i].bias - xy_grid
    
        ## Calculate the magnitude of velocity vectors
        magnitudes = np.sqrt(dx[:, 0]**2 + dx[:, 1]**2)

        x_grids.append(x_grid)
        y_grids.append(y_grid)
        dxs.append(dx)
        magnitudes_all.append(magnitudes)

    ## Normalize magnitudes for color mapping
    norm = plt.Normalize(
        min(magnitudes_all[0].min(), magnitudes_all[1].min()), 
        max(magnitudes_all[0].max(), magnitudes_all[1].max()))
    
    for i in range(2):

        ## Plotting the quiver plot with color encoding by magnitude
        quiver = axs[i].quiver(x_grids[i], y_grids[i], dxs[i][:, 0], dxs[i][:, 1], color=colormap(norm(magnitudes_all[i])))
    
    ## Create colorbar
    return plt.cm.ScalarMappable(norm=norm, cmap=colormap)


## The vectors can be normalized, in which case the magnitudes are visualized with colormaps. One colormap corresponds to one discrete state.
def add_SLDS_dynamics_quiver(
    ax, 
    model, 
    colors_or_cmaps, 
    n_points,
    x_min,
    x_max,
    y_min,
    y_max,
    scale=200, # Default scale for quiver arrows
    alpha=0.5,
    normalize=False):  

    # scale = min((x_max - x_min), (y_max - y_min)) 

    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    xy_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    try:
        z = np.argmax(xy_grid.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)
    except:
        raise ValueError('Could not get most likely states')

    # Store the quiver instances and normalizations for colorbars
    quivers = []
    color_norms = []

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy_grid.dot(A.T) + b - xy_grid
        zk = z == k
            
        if zk.sum(0) > 0:

            if normalize:
                magnitudes = np.sqrt(dxydt_m[:, 0]**2 + dxydt_m[:, 1]**2)
                dxydt_normalized = dxydt_m / magnitudes[:, np.newaxis] # Normalize the vectors

                norm = plt.Normalize(vmin=magnitudes[zk].min(), vmax=magnitudes[zk].max())
                color_norms.append(norm)

                quiv = ax.quiver(
                    xy_grid[zk, 0], xy_grid[zk, 1],
                    dxydt_normalized[zk, 0], dxydt_normalized[zk, 1],
                    color=colors_or_cmaps[k](norm(magnitudes[zk])),  ## Color based on the magnitude
                    norm=norm,
                    scale=scale,
                    alpha=alpha)
                
                quivers.append(quiv)
                
            else:
                ax.quiver(
                    xy_grid[zk, 0], xy_grid[zk, 1],
                    dxydt_m[zk, 0], dxydt_m[zk, 1],
                    color=colors_or_cmaps[k % len(colors_or_cmaps)],
                    scale=scale,
                    alpha=alpha)
            
    return quivers, color_norms


## The vectors can be normalized, in which case the magnitudes are visualized with colormaps.
def add_LDS_dynamics_quiver(
    ax, 
    model, 
    color_or_cmap, 
    n_points,
    x_min,
    x_max,
    y_min,
    y_max,
    scale=200, # Default scale for quiver arrows
    alpha=0.5,
    normalize=False):

    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    xy_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    A = model.dynamics.As[0]
    b = model.dynamics.bs[0]

    dxydt_m = xy_grid.dot(A.T) + b - xy_grid

    if normalize:
        magnitudes = np.sqrt(dxydt_m[:, 0]**2 + dxydt_m[:, 1]**2)
        dxydt_normalized = dxydt_m / magnitudes[:, np.newaxis] # Normalize the vectors

        norm = plt.Normalize(vmin=magnitudes.min(), vmax=magnitudes.max())

        quiv = ax.quiver(
            xy_grid[:, 0], xy_grid[:, 1],
            dxydt_normalized[:, 0], dxydt_normalized[:, 1],
            color=color_or_cmap(norm(magnitudes)),  ## Color based on the magnitude
            norm=norm,
            scale=scale,
            alpha=alpha)
        
    else:
        ax.quiver(
            xy_grid[:, 0], xy_grid[:, 1],
            dxydt_m[:, 0], dxydt_m[:, 1],
            color=color_or_cmap,
            scale=scale,
            alpha=alpha)
        
    return quiv, norm


## The vectors can be normalized, in which case the magnitudes are visualized with colormaps. One colormap corresponds to one discrete state.
def add_SLDS_dynamics_quiver_3D(
    ax, 
    model, 
    colors_or_cmaps, 
    n_points,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
    length=0.1,
    alpha=0.5,
    normalize=False):  

    x_grid, y_grid, z_grid = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points), np.linspace(z_min, z_max, n_points))
    xyz_grid = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))

    try:
        z = np.argmax(xyz_grid.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)
    except:
        raise ValueError('Could not get most likely states')

    # Store the quiver instances and normalizations for colorbars
    quivers = []
    color_norms = []

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxyzdt_m = xyz_grid.dot(A.T) + b - xyz_grid
        zk = z == k
            
        if zk.sum(0) > 0:

            if normalize:
                magnitudes = np.sqrt(dxyzdt_m[:, 0]**2 + dxyzdt_m[:, 1]**2 + dxyzdt_m[:, 2]**2)
                dxyzdt_normalized = dxyzdt_m / magnitudes[:, np.newaxis] # Normalize the vectors

                norm = plt.Normalize(vmin=magnitudes[zk].min(), vmax=magnitudes[zk].max())
                color_norms.append(norm)

                quiv = ax.quiver(
                    xyz_grid[zk, 0], xyz_grid[zk, 1], xyz_grid[zk, 2],
                    dxyzdt_normalized[zk, 0], dxyzdt_normalized[zk, 1], dxyzdt_normalized[zk, 2],
                    color=colors_or_cmaps[k](norm(magnitudes[zk])),  ## Color based on the magnitude
                    norm=norm,
                    length=length,
                    alpha=alpha,
                    arrow_length_ratio=0.4)
                
                quivers.append(quiv)
                
            else:
                ax.quiver(
                    xyz_grid[zk, 0], xyz_grid[zk, 1], xyz_grid[zk, 2],
                    dxyzdt_m[zk, 0], dxyzdt_m[zk, 1], dxyzdt_m[zk, 2],
                    color=colors_or_cmaps[k % len(colors_or_cmaps)],
                    length=length,
                    alpha=alpha)

    return quivers, color_norms


## The vectors can be normalized, in which case the magnitudes are visualized with colormaps.
def add_LDS_dynamics_quiver_3D(
    ax, 
    model, 
    color_or_cmap, 
    n_points,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
    length=0.1, # Default scale for quiver arrows
    alpha=0.5,
    normalize=False):

    x_grid, y_grid, z_grid = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points), np.linspace(z_min, z_max, n_points))
    xyz_grid = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))

    A = model.dynamics.As[0]
    b = model.dynamics.bs[0]

    dxyzdt_m = xyz_grid.dot(A.T) + b - xyz_grid

    if normalize:
        magnitudes = np.sqrt(dxyzdt_m[:, 0]**2 + dxyzdt_m[:, 1]**2 + dxyzdt_m[:, 2]**2)
        dxyzdt_normalized = dxyzdt_m / magnitudes[:, np.newaxis] # Normalize the vectors

        norm = plt.Normalize(vmin=magnitudes.min(), vmax=magnitudes.max())

        quiv = ax.quiver(
            xyz_grid[:, 0], xyz_grid[:, 1], xyz_grid[:, 2],
            dxyzdt_normalized[:, 0], dxyzdt_normalized[:, 1], dxyzdt_normalized[:, 2],
            color=color_or_cmap(norm(magnitudes)),  ## Color based on the magnitude
            norm=norm,
            length=length,
            alpha=alpha)
        
    else:
        ax.quiver(
            xyz_grid[:, 0], xyz_grid[:, 1], xyz_grid[:, 2],
            dxyzdt_m[:, 0], dxyzdt_m[:, 1], dxyzdt_m[:, 2],
            color=color_or_cmap,
            length=length,
            alpha=alpha)
        
    return quiv, norm


def visualize_SLDS_flow_fields_occupancy_grid(
    ax,
    model,
    colors_or_cmaps,
    continuous_states,
    discrete_states,
    n_discrete_states,
    n_points,
    length=0.1,
    alpha=0.5,
    normalize=False):

    n_trials = len(discrete_states)
    assert n_trials == len(continuous_states)

    ## Build occupancy grids for each discrete state according to the continuous states
    occupancy = np.zeros((n_discrete_states, n_points, n_points, n_points), dtype=bool)

    x_min = np.min([np.min(continuous_states_[..., 0]) for continuous_states_ in continuous_states])
    x_max = np.max([np.max(continuous_states_[..., 0]) for continuous_states_ in continuous_states])
    y_min = np.min([np.min(continuous_states_[..., 1]) for continuous_states_ in continuous_states])
    y_max = np.max([np.max(continuous_states_[..., 1]) for continuous_states_ in continuous_states])
    z_min = np.min([np.min(continuous_states_[..., 2]) for continuous_states_ in continuous_states])
    z_max = np.max([np.max(continuous_states_[..., 2]) for continuous_states_ in continuous_states])

    x_edges = np.linspace(x_min, x_max, n_points)
    y_edges = np.linspace(y_min, y_max, n_points)
    z_edges = np.linspace(z_min, z_max, n_points)

    ## Fill occupancy grid with one side logic
    # for i_trial, (discrete_states_, continuous_states_) in enumerate(zip(discrete_states, continuous_states)):

    #     trial_length = len(discrete_states_)
    #     assert trial_length == len(continuous_states_)

    #     # Extract coordinates
    #     xs_continuous_states_ = continuous_states_[:, 0]
    #     ys_continuous_states_ = continuous_states_[:, 1]
    #     zs_continuous_states_ = continuous_states_[:, 2]

    #     # Find indices along each dimension
    #     x_indices = np.searchsorted(x_edges, xs_continuous_states_) - 1
    #     y_indices = np.searchsorted(y_edges, ys_continuous_states_) - 1
    #     z_indices = np.searchsorted(z_edges, zs_continuous_states_) - 1

    #     # Clip indices to be within [0, n_points - 1]
    #     x_indices = np.clip(x_indices, 0, n_points - 1)
    #     y_indices = np.clip(y_indices, 0, n_points - 1)
    #     z_indices = np.clip(z_indices, 0, n_points - 1)

    #     # Update occupancy
    #     for t in range(trial_length):
    #         s = discrete_states_[t]  # discrete state at time t
    #         xi, yi, zi = x_indices[t], y_indices[t], z_indices[t]
    #         occupancy[s, xi, yi, zi] = True


    ## Fill occupancy grid with both sides logic
    epsilon = 10  # tolerance for considering a point close to an edge

    for i_trial, (discrete_states_, continuous_states_) in enumerate(zip(discrete_states, continuous_states)):

        trial_length = len(discrete_states_)
        assert trial_length == len(continuous_states_)

        # Extract coordinates
        xs = continuous_states_[:, 0]
        ys = continuous_states_[:, 1]
        zs = continuous_states_[:, 2]

        # Find indices along each dimension
        x_indices = np.searchsorted(x_edges, xs) - 1
        y_indices = np.searchsorted(y_edges, ys) - 1
        z_indices = np.searchsorted(z_edges, zs) - 1

        # Clip indices to be within [0, n_points - 1]
        x_indices = np.clip(x_indices, 0, n_points - 1)
        y_indices = np.clip(y_indices, 0, n_points - 1)
        z_indices = np.clip(z_indices, 0, n_points - 1)

        for t in range(trial_length):
            s = discrete_states_[t]
            xi, yi, zi = x_indices[t], y_indices[t], z_indices[t]

            # Always mark the primary cell
            occupancy[s, xi, yi, zi] = True

            # Check proximity to x-direction edges
            if xi < n_points - 1:
                # Upper edge for x-bin is x_edges[xi+1]
                if abs(xs[t] - x_edges[xi+1]) < epsilon:
                    occupancy[s, xi+1, yi, zi] = True
            if xi > 0:
                # Lower edge for x-bin is x_edges[xi]
                # Actually, since we found xi as searchsorted(...) - 1,
                # xi is already the lower bin for xs[t].
                # If you want to consider if it's close to the lower edge, check:
                if abs(xs[t] - x_edges[xi]) < epsilon:
                    # If this edge is also close, you might mark the previous bin if xi > 0
                    occupancy[s, xi-1, yi, zi] = True

            # Check proximity to y-direction edges
            if yi < n_points - 1:
                if abs(ys[t] - y_edges[yi+1]) < epsilon:
                    occupancy[s, xi, yi+1, zi] = True
            if yi > 0:
                if abs(ys[t] - y_edges[yi]) < epsilon:
                    occupancy[s, xi, yi-1, zi] = True

            # Check proximity to z-direction edges
            if zi < n_points - 1:
                if abs(zs[t] - z_edges[zi+1]) < epsilon:
                    occupancy[s, xi, yi, zi+1] = True
            if zi > 0:
                if abs(zs[t] - z_edges[zi]) < epsilon:
                    occupancy[s, xi, yi, zi-1] = True


    ## Visualize the occupancy grids
    # discrete_state_colors = ['red', 'blue']

    # for s in range(n_discrete_states):
    #     # Find the indices of all occupied voxels for state s
    #     xi, yi, zi = np.where(occupancy[s])

    #     # Convert these indices to actual coordinates
    #     # If x_edges, y_edges, z_edges are the edges of the grid, then
    #     # x_edges[i], y_edges[j], z_edges[k] give the coordinates of that voxel's position.
    #     x_coords = x_edges[xi]
    #     y_coords = y_edges[yi]
    #     z_coords = z_edges[zi]

    #     # Scatter plot these points
    #     # alpha=0.5 for semi-transparency so overlapping points can be seen
    #     ax.scatter(x_coords, y_coords, z_coords, color=discrete_state_colors[s], label=f"State {s}", alpha=0.5, s=20)


    ## Visualize the flow fields
    x_grid, y_grid, z_grid = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')
    xyz_grid = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))

    # Store the quiver instances and normalizations for colorbars
    quivers = []
    color_norms = []

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        # Compute flow: dxyz/dt = A*x + b - x
        dxyzdt_m = xyz_grid.dot(A.T) + b - xyz_grid  # shape (n_points^3, 3)

        # Get occupancy mask for state k
        # occupancy[k] shape: (n_points, n_points, n_points)
        # Flatten it to match xyz_grid's shape (n_points^3,)
        zk = occupancy[k].ravel()

        # Check if there are any occupied voxels for this state
        if np.any(zk):
            # We have at least one occupied voxel for this state

            if normalize:
                # Compute magnitudes and normalize
                magnitudes = np.sqrt((dxyzdt_m[:, 0]**2 + dxyzdt_m[:, 1]**2 + dxyzdt_m[:, 2]**2))
                dxyzdt_normalized = np.zeros_like(dxyzdt_m)
                nonzero = magnitudes > 0
                dxyzdt_normalized[nonzero] = dxyzdt_m[nonzero] / magnitudes[nonzero, np.newaxis]

                # For color mapping, use only the occupied points
                occupied_magnitudes = magnitudes[zk]
                norm = plt.Normalize(vmin=occupied_magnitudes.min(), vmax=occupied_magnitudes.max())
                color_norms.append(norm)

                # Draw quiver only for occupied points
                quiv = ax.quiver(
                    xyz_grid[zk, 0], xyz_grid[zk, 1], xyz_grid[zk, 2],
                    dxyzdt_normalized[zk, 0], dxyzdt_normalized[zk, 1], dxyzdt_normalized[zk, 2],
                    color=colors_or_cmaps[k](norm(occupied_magnitudes)),  # color by magnitude
                    norm=norm,
                    length=length,
                    alpha=alpha,
                    arrow_length_ratio=0.4)

                quivers.append(quiv)

            else:
                # No normalization
                ax.quiver(
                    xyz_grid[zk, 0], xyz_grid[zk, 1], xyz_grid[zk, 2],
                    dxyzdt_m[zk, 0], dxyzdt_m[zk, 1], dxyzdt_m[zk, 2],
                    color=colors_or_cmaps[k % len(colors_or_cmaps)],
                    length=length,
                    alpha=alpha
                )

    return quivers, color_norms


def visualize_SLDS_flow_fields(
    ax: Axes3D,
    model,
    colors_or_cmaps,
    continuous_states,
    discrete_states,
    n_discrete_states: int,
    index_spacing: int,
    *,
    vmin=None,
    vmax=None,
    length: float = 0.05,
    alpha: float = 0.5,
    normalize: bool = False,
    reverse: bool = False,
    line_width: float = 0.25,         # ➊ shaft thickness (in points)
    head_ratio: float = 0.2          # ➋ head-to-shaft length ratio
):
    """
    Visualize the flow field induced by each discrete SLDS state using 3-D quivers.

    Parameters
    ----------
    ax : matplotlib 3-D axes
    model : Fitted SLDS object with attributes `dynamics.As` and `dynamics.bs`
    colors_or_cmaps : list of matplotlib colormap objects or RGBA tuples
    continuous_states : list/array of (T_i, 3) continuous latent coordinates
    discrete_states   : list/array of (T_i,) discrete state labels
    n_discrete_states : total number of discrete states in the model
    index_spacing     : keep every *n*th point to thin the quiver density
    vmin, vmax        : per-state magnitude limits for color mapping (optional)
    length            : overall arrow length (world units)
    alpha             : arrow opacity
    normalize         : if True, arrows are unit-length; color encodes magnitude
    reverse           : if True, reverse the order of colors_or_cmaps
    line_width        : **shaft thickness in points**  (default 0.3 – nice and thin)
    head_ratio        : **arrow-head-to-shaft ratio** (default 0.2 – smaller heads)
    """
    # ------------------------------------------------------------------ prep colours
    colors_or_cmaps = colors_or_cmaps[:n_discrete_states]
    if reverse:
        colors_or_cmaps = colors_or_cmaps[::-1]

    quivers, color_norms = [], []

    # ---------------------------------------------------------------- iterate states
    for k in range(n_discrete_states):

        # Gather all coordinates that belong to state k across trials
        coords_k = [
            cs[ds == k]              # slice of shape (n_k_i, 3)
            for ds, cs in zip(discrete_states, continuous_states)
            if np.any(ds == k)
        ]
        if not coords_k:             # no points for this state
            continue
        all_coords = np.vstack(coords_k)[1::index_spacing]   # thin them

        # Remove points too close to origin (optional heuristic)
        all_coords = all_coords[np.linalg.norm(all_coords, axis=1) > 2]
        if all_coords.size == 0:
            continue

        # ------------------------------------------------------ compute flow vectors
        A, b = model.dynamics.As[k], model.dynamics.bs[k]
        dxyzdt = all_coords @ A.T + b - all_coords     # (N, 3)

        # ------------------------------------------------------ normalise if desired
        if normalize:
            mags = np.linalg.norm(dxyzdt, axis=1)
            nonzero = mags > 1e-12
            dxyzdt_unit = np.zeros_like(dxyzdt)
            dxyzdt_unit[nonzero] = dxyzdt[nonzero] / mags[nonzero, None]

            vmin_ = mags.min() if vmin is None else vmin[k]
            vmax_ = mags.max() if vmax is None else vmax[k]
            norm  = plt.Normalize(vmin=vmin_, vmax=vmax_)
            color_norms.append(norm)

            # print(mags)

            # print(norm(mags))

            print(mags.max(), mags.min())

            quiver = ax.quiver(
                all_coords[:, 0], all_coords[:, 1], all_coords[:, 2],
                dxyzdt_unit[:, 0], dxyzdt_unit[:, 1], dxyzdt_unit[:, 2],
                color=colors_or_cmaps[k](norm(mags)),
                norm=norm,
                length=length,
                alpha=alpha,
                linewidths=line_width,          # ➊ shaft thickness
                arrow_length_ratio=head_ratio   # ➋ head size
            )
        else:
            quiver = ax.quiver(
                all_coords[:, 0], all_coords[:, 1], all_coords[:, 2],
                dxyzdt[:, 0], dxyzdt[:, 1], dxyzdt[:, 2],
                color=colors_or_cmaps[k % len(colors_or_cmaps)],
                length=length,
                alpha=alpha,
                linewidths=line_width,          # ➊
                arrow_length_ratio=head_ratio   # ➋
            )

        quivers.append(quiver)

    return quivers, color_norms


def plot_state_boundaries(
    ax, 
    model, 
    n_points,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max):

    Rs = model.transitions.Rs  # Shape: (num_states, 3)
    r  = model.transitions.r   # Shape: (num_states,)

    num_states = Rs.shape[0]
    state_pairs = list(combinations(range(num_states), 2))

    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)
    X, Y = np.meshgrid(x, y)

    for (i, j) in state_pairs:
        # Compute the normal vector of the plane separating states i and j
        normal_vector = Rs[i, :] - Rs[j, :]
        d = r[i] - r[j]

        # If the normal vector is zero, skip this pair
        if np.allclose(normal_vector, 0):
            continue

        # Solve for Z in the plane equation: A*x + B*y + C*z + D = 0
        # Rearranged to z = (-A*x - B*y - D) / C
        A, B, C = normal_vector
        if C == 0:
            continue  # Plane is vertical, handle separately if needed

        Z = (-A * X - B * Y - d) / C

        # Plot the plane
        ax.plot_surface(X, Y, Z, alpha=0.3, color='grey')

    # Set plot limits
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])


def add_GPSLDS_dynamics_quiver(
    ax,
    n_discrete_states,
    n_points,
    x_min,
    x_max,
    y_min,
    y_max,
    kernel,
    kernel_params,
    zs,
    q_u_mu,
    f_mean_function,
    get_most_likely_state_function,
    colors,
    ):

    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, n_points),
        np.linspace(y_min, y_max, n_points),
    )
    xy_grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # generate mean of learned dynamics in gpslds space, then rotate back to true space
    f_mean = f_mean_function(kernel.K, kernel_params, xy_grid, zs, q_u_mu)

    discrete_states = get_most_likely_state_function(kernel.construct_partition, kernel_params, xy_grid)

    for i in range(n_discrete_states):
        discrete_state = (discrete_states == i).astype(bool)
        ax.quiver(
            xy_grid[discrete_state, 0], 
            xy_grid[discrete_state, 1], 
            f_mean[discrete_state, 0], 
            f_mean[discrete_state, 1], 
            color=colors[i])
        

def add_GPSLDS_dynamics_quiver_3D(
    ax,
    n_discrete_states,
    n_points,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
    kernel,
    kernel_params,
    zs,
    q_u_mu,
    f_mean_function,
    get_most_likely_state_function,
    colors,
    ):

    x_grid, y_grid, z_grid = np.meshgrid(
        np.linspace(x_min, x_max, n_points),
        np.linspace(y_min, y_max, n_points),
        np.linspace(z_min, z_max, n_points),
    )
    xyz_grid = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    # generate mean of learned dynamics in gpslds space, then rotate back to true space
    f_mean = f_mean_function(kernel.K, kernel_params, xyz_grid, zs, q_u_mu)

    discrete_states = get_most_likely_state_function(kernel.construct_partition, kernel_params, xyz_grid)

    for i in range(n_discrete_states):
        discrete_state = (discrete_states == i).astype(bool)
        ax.quiver(
            xyz_grid[discrete_state, 0], 
            xyz_grid[discrete_state, 1], 
            xyz_grid[discrete_state, 2],
            f_mean[discrete_state, 0], 
            f_mean[discrete_state, 1], 
            f_mean[discrete_state, 2],
            length=0.01,
            color=colors[i])


def heatmap_with_axis_avgs(
    data, 
    xlabel, 
    ylabel, 
    suptitle, 
    xticklabels, 
    yticklabels,
    cmap, 
    vmin, 
    vmax, 
    save_path=None):

    row_averages = data.mean(axis=1)
    col_averages = data.mean(axis=0)

    ## Create figure
    fig = plt.figure(figsize=(12, 10))  # Adjust figure size as needed

    ## Define GridSpec for layout
    gs = gridspec.GridSpec(2, 3, 
        width_ratios=[5, 1, 0.5], height_ratios=[1, 5], 
        left=0.1, right=0.9, bottom=0.1, top=0.9, 
        wspace=0.05, hspace=0.05)

    ## Main heatmap
    ax0 = plt.subplot(gs[1, 0])
    sns.heatmap(data, ax=ax0, cbar=False, xticklabels=xticklabels, yticklabels=yticklabels, cmap=cmap, vmin=vmin, vmax=vmax)  # Set cbar=False, adding manually later
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    ax0.invert_yaxis()

    ## Horizontal line plot for column averages
    ax1 = plt.subplot(gs[0, 0], sharex=ax0)
    ax1.plot(col_averages, linestyle='-', marker='o', color='blue')
    ax1.set_ylabel('Col Avg')
    plt.setp(ax1.get_xticklabels(), visible=False)

    ## Vertical line plot for row averages
    ax2 = plt.subplot(gs[1, 1], sharey=ax0)
    ax2.plot(row_averages, range(len(row_averages)), linestyle='-', marker='o', color='red')
    ax2.set_xlabel('Row Avg')
    plt.setp(ax2.get_yticklabels(), visible=False)

    ## Create an axis for the color bar to the right of the vertical line plot
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="20%", pad=0.1)  # Adjust size and padding as needed

    ## Generate the color bar
    plt.colorbar(ax0.collections[0], cax=cax)

    fig.suptitle(suptitle)

    # plt.tight_layout()

    plt.show()


## PCA dimensionality reduction method for dynamical systems
##   from https://www.cell.com/cell/pdf/S0092-8674(22)01471-4.pdf
def dynamical_PCA(
    x, 
    C, 
    n_components=None, # If n_components is None, then n_components = n_latent_dims 
    whitening=False):

    ## Stack the trials of continuous states
    if isinstance(x, np.ndarray) and x.ndim == 3:
        n_trials, n_times, n_latent_dims = x.shape
        x_stacked = x.reshape(-1, n_latent_dims)
    elif isinstance(x, list) and all(isinstance(trial, np.ndarray) for trial in x):
        n_trials = len(x)
        n_times = [x_.shape[0] for x_ in x]
        x_stacked = np.vstack(x)
        n_latent_dims = x_stacked.shape[1]

    if n_components is None:
        n_components = n_latent_dims

    # Compute the mean across all samples
    x_stacked_mean = np.mean(x_stacked, axis=0)

    # Center the data
    x_stacked_centered = x_stacked - x_stacked_mean

    if whitening:
        # Compute the covariance matrix
        cov_x = np.cov(x_stacked_centered, rowvar=False)

        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(cov_x)

        # Ensure eigenvalues are non-negative (numerical stability)
        eigvals[eigvals < 0] = 0.0

        # Compute the diagonal matrix of inverse square roots of eigenvalues
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + 1e-10))  # Add small constant to avoid division by zero

        # Compute the whitening matrix
        W = eigvecs @ D_inv_sqrt @ eigvecs.T

        # Adjust the emission matrix C (make sure C is provided)
        W_inv = np.linalg.pinv(W)
        C_prime = C @ W_inv

        # Perform SVD
        U, S, Vt = np.linalg.svd(C_prime, full_matrices=True)
    
    else:
        # Perform SVD
        U, S, Vt = np.linalg.svd(C, full_matrices=True)

    # Construct mapping for mapping dynamics and transition matrices
    S_k = S[:n_components]
    Vt_k = Vt[:n_components, :]

    S_matrix_k = np.diag(S_k)
    P_k = S_matrix_k @ Vt_k
    S_inv_k = np.diag(1.0 / S_k)
    P_inv_k = Vt_k.T @ S_inv_k

    # Compute the reduced transformation matrix T_k
    if whitening:
        T_k = W @ P_inv_k
    else:
        T_k = P_inv_k

    x_double_prime = x_stacked_centered @ T_k

    # Recreate 3D structure
    if isinstance(x, np.ndarray) and x.ndim == 3:
        x_res = x_double_prime.reshape(n_trials, n_times, n_components)
    elif isinstance(x, list) and all(isinstance(trial, np.ndarray) for trial in x):
        x_res = []
        start_idx = 0
        for i in range(n_trials):
            end_idx = start_idx + n_times[i]
            x_res.append(x_double_prime[start_idx:end_idx, :])
            start_idx = end_idx

    return x_res, T_k


def partition_state_segments(discrete_states):
    """
    Partition the discrete state sequence into continuous segments.

    Parameters:
    z : numpy.ndarray
        Discrete state sequence of shape (n_samples,).

    Returns:
    segments : list of tuples
        Each tuple contains (start_idx, end_idx, state).
    """
    segments = []
    n_samples = len(discrete_states)
    if n_samples == 0:
        return segments

    start_idx = 0
    current_state = discrete_states[0]

    for idx in range(1, n_samples):
        if discrete_states[idx] != current_state:
            # State has changed
            end_idx = idx - 1
            segments.append((start_idx, end_idx, current_state))
            start_idx = idx
            current_state = discrete_states[idx]

    # Add the last segment
    segments.append((start_idx, n_samples - 1, current_state))
    return segments


def reduce_dynamical_dimensionality( 
    model, 
    continuous_states,
    n_target_dimensions):

    continuous_states_new, T_k = dynamical_PCA(
        continuous_states, 
        model.emissions.Cs[0], 
        n_components=n_target_dimensions, 
        whitening=True)

    ## Compute inverse Gram matrix from T_k
    G_inv = np.linalg.inv(T_k.T @ T_k)

    dynamics_matrices = model.dynamics.As
    dynamics_biases   = model.dynamics.bs

    n_discrete_states_ = dynamics_matrices.shape[0]

    dynamics_matrices_ = np.zeros((n_discrete_states_, n_target_dimensions, n_target_dimensions))
    dynamics_biases_   = np.zeros((n_discrete_states_, n_target_dimensions))

    for i_discrete_state in range(n_discrete_states_):
        dynamics_matrices_[i_discrete_state] = G_inv @ T_k.T @ dynamics_matrices[i_discrete_state] @ T_k
        dynamics_biases_[i_discrete_state]   = dynamics_biases[i_discrete_state] @ T_k

    model.dynamics.As = dynamics_matrices_
    model.dynamics.bs = dynamics_biases_

    try:
        model.transitions.Rs = model.transitions.Rs @ T_k
    except: 
        print('[WARNING][utils_vis.reduce_dynamical_dimensionality] Model has no transition matrix to transform')

    return continuous_states_new


def reduce_dynamical_dimensionality_( 
    model, 
    continuous_states,
    n_target_dimensions):

    continuous_states_new, T_full = dynamical_PCA(
        continuous_states, 
        model.emissions.Cs[0], 
        n_components=None, 
        whitening=True)

    ## Compute inverse Gram matrix from T_k
    G_inv = np.linalg.inv(T_full.T @ T_full)

    dynamics_matrices = model.dynamics.As
    dynamics_biases   = model.dynamics.bs

    n_discrete_states_ = dynamics_matrices.shape[0]

    dynamics_matrices_ = np.zeros((n_discrete_states_, n_target_dimensions, n_target_dimensions))
    dynamics_biases_   = np.zeros((n_discrete_states_, n_target_dimensions))

    for i_discrete_state in range(n_discrete_states_):
        dynamics_matrices_[i_discrete_state] = (G_inv.T @ T_full.T @ dynamics_matrices[i_discrete_state] @ T_full)[:n_target_dimensions, :n_target_dimensions]
        dynamics_biases_[i_discrete_state]   = (dynamics_biases[i_discrete_state] @ T_full)[:n_target_dimensions]

    model.dynamics.As = dynamics_matrices_
    model.dynamics.bs = dynamics_biases_

    try:
        model.transitions.Rs = (model.transitions.Rs @ T_full)[:, :n_target_dimensions]
    except:
        print('[WARNING][utils_vis.reduce_dynamical_dimensionality_] Model has no transition matrix to transform')

    return continuous_states_new


def reduce_dynamical_dimensionality_by_fitting( 
    model, 
    continuous_states,
    discrete_states,
    n_discrete_states,
    n_target_dimensions):

    continuous_states_new, T_k = dynamical_PCA(
        continuous_states, 
        model.emissions.Cs[0], 
        n_components=n_target_dimensions,
        whitening=True)

    ## Partition continuous states by discrete states
    continuous_states_by_discrete_states_t  = [[] for _ in range(n_discrete_states)] # List of n_discrete_states lists of stacked continuous states
    continuous_states_by_discrete_states_t1 = [[] for _ in range(n_discrete_states)] # List of n_discrete_states lists of stacked continuous states (t+1)

    for i_trial, (continuous_states_, discrete_states_) in enumerate(zip(continuous_states_new, discrete_states)):

        discrete_state_segments = partition_state_segments(discrete_states_)

        for discrete_state_segment in discrete_state_segments:
            start_idx, end_idx, discrete_state = discrete_state_segment

            if end_idx - start_idx < 2:
                continue

            continuous_states_by_discrete_states_t[discrete_state].append(continuous_states_[start_idx : end_idx])
            continuous_states_by_discrete_states_t1[discrete_state].append(continuous_states_[start_idx + 1 : end_idx + 1])

    ## Stack each list of continuous states
    continuous_states_by_discrete_states_t  = [np.vstack(continuous_states_) for continuous_states_ in continuous_states_by_discrete_states_t]
    continuous_states_by_discrete_states_t1 = [np.vstack(continuous_states_) for continuous_states_ in continuous_states_by_discrete_states_t1]

    dynamics_matrices_ = np.zeros((n_discrete_states, n_target_dimensions, n_target_dimensions))
    dynamics_biases_   = np.zeros((n_discrete_states, n_target_dimensions))

    for i_discrete_state in range(n_discrete_states):
            
        X_t  = continuous_states_by_discrete_states_t[i_discrete_state].T
        X_t1 = continuous_states_by_discrete_states_t1[i_discrete_state].T

        # Include bias term
        X_t_aug = np.vstack([X_t, np.ones(X_t.shape[1])])

        params = np.linalg.pinv(X_t_aug.T) @ X_t1.T

        # Solve for [A_reduced | b_reduced] using least squares
        # params, residuals, rank, s = np.linalg.lstsq(X_t_aug.T, X_t1.T, rcond=None)
    
        # Extract A_reduced and b_reduced
        A_reduced = params[:-1, :].T
        b_reduced = params[-1, :].T

        dynamics_matrices_[i_discrete_state] = A_reduced
        dynamics_biases_[i_discrete_state]   = b_reduced

    model.dynamics.As = dynamics_matrices_
    model.dynamics.bs = dynamics_biases_

    try:
        model.transitions.Rs = model.transitions.Rs @ T_k
    except:
        print('[WARNING][utils_vis.reduce_dimensionality] Model has no transition matrix to transform')

    return continuous_states_new


def reorder_discrete_states(discrete_states, n_discrete_states):
    """
    Reorder the discrete states in `discrete_states` such that the state
    that appears earliest across all sequences is relabeled as 0, the
    next earliest as 1, and so on.

    Parameters
    ----------
    discrete_states : list of np.ndarray
        A list of 1D numpy arrays, each containing integer states from
        0 to n_discrete_states-1.
    n_discrete_states : int
        The total number of unique states (0, 1, 2, ..., n_discrete_states-1).

    Returns
    -------
    reordered_discrete_states : list of np.ndarray
        A list of 1D numpy arrays of the same shape as input, but
        with relabeled discrete states.
    old_to_new : dict
        A dictionary mapping the old state labels to the new labels.
    """

    # 1. Accumulate position indices for each state
    # sum_positions[s] will store the sum of indices where state 's' occurs.
    sum_positions = np.zeros(n_discrete_states, dtype=np.float64)
    # count how many times each state appears (to help handle average if needed)
    count_states = np.zeros(n_discrete_states, dtype=int)

    for arr in discrete_states:
        for idx, state in enumerate(arr):
            sum_positions[state] += idx
            count_states[state] += 1

    # (Optional) If you'd like to use average positions:
    # for s in range(n_discrete_states):
    #     if count_states[s] > 0:
    #         sum_positions[s] /= count_states[s]

    # 2. Rank states by their total (or average) position
    # We'll sort by sum_positions; the lower the sum, the earlier on average
    # that state appears.
    states_sorted_by_position = sorted(range(n_discrete_states),
                                       key=lambda s: sum_positions[s])

    # 3. Create a mapping from old labels to new labels
    # The state with the smallest sum_positions becomes 0, next smallest becomes 1, etc.
    reorder = False
    old_to_new = {}
    for new_label, old_state in enumerate(states_sorted_by_position):
        old_to_new[old_state] = new_label
        if new_label != old_state:
            reorder = True

    # 4. Apply that mapping to each array
    reordered_discrete_states = []
    for arr in discrete_states:
        # Vectorized remapping using np.vectorize or list comprehension
        new_arr = np.array([old_to_new[s] for s in arr], dtype=int)
        reordered_discrete_states.append(new_arr)

    return reordered_discrete_states, old_to_new, reorder


def kl_divergence_per_time_point(data, num_classes=None, q=None):
    n_trials, n_times = data.shape
    kl_divergences = np.zeros(n_times)
    
    # Infer number of classes from data if not specified
    if num_classes is None:
        num_classes = int(np.max(data)) + 1  # Assuming classes are integers starting from 0
    
    # Set default expected distribution to uniform if not specified
    if q is None:
        q = np.full(num_classes, 1.0 / num_classes)
    else:
        q = np.array(q)
        if len(q) != num_classes:
            raise ValueError("Length of expected distribution q must match num_classes")
        if not np.isclose(np.sum(q), 1):
            raise ValueError("Expected distribution q must sum to 1")

    for t in range(n_times):
        # Counts of each class at time point t
        counts = np.bincount(data[:, t].astype(np.int64), minlength=num_classes)
        p = counts / counts.sum()
        
        # Compute KL divergence, handling zero probabilities
        kl_div = np.sum(p[p > 0] * np.log2(p[p > 0] / q[p > 0]))
        kl_divergences[t] = kl_div
        
    return kl_divergences


def total_variation_per_time_point(data, num_classes=None, q=None):
    n_trials, n_times = data.shape
    tv_distances = np.zeros(n_times)
    
    # Infer number of classes from data if not specified
    if num_classes is None:
        num_classes = int(np.max(data)) + 1  # Assuming classes are integers starting from 0
    
    # Set default expected distribution to uniform if not specified
    if q is None:
        q = np.full(num_classes, 1.0 / num_classes)
    else:
        q = np.array(q)
        if len(q) != num_classes:
            raise ValueError("Length of expected distribution q must match num_classes")
        if not np.isclose(np.sum(q), 1):
            raise ValueError("Expected distribution q must sum to 1")

    for t in range(n_times):
        # Counts of each class at time point t
        counts = np.bincount(data[:, t].astype(np.int64), minlength=num_classes)
        p = counts / counts.sum()
        
        # Compute total variation distance
        tv = 0.5 * np.sum(np.abs(p - q))
        tv_distances[t] = tv
        
    return tv_distances


def entropy_per_time_point(data, num_classes=None):
    """
    Compute the normalized Shannon entropy at each time point (across trials).
    Normalized so that the entropy is in [0, 1], where 0 = all trials in one state,
    and 1 = perfectly uniform distribution of used states.

    Parameters
    ----------
    data : ndarray, shape (n_trials, n_times)
        Discrete states for each trial x time.
    num_classes : int or None
        Number of possible discrete states (used for bin-counting). If None,
        it is inferred as the max state value + 1 in `data`.

    Returns
    -------
    entropies : ndarray, shape (n_times,)
        Normalized entropy at each time point in [0, 1].
    """
    n_trials, n_times = data.shape
    
    # If user does not specify number of classes, infer from data
    if num_classes is None:
        num_classes = int(data.max()) + 1
    
    entropies = np.zeros(n_times)

    for t in range(n_times):
        # Counts of each class at time point t across trials
        counts = np.bincount(data[:, t].astype(np.int64), minlength=num_classes)
        total = counts.sum()
        
        # Handle edge case: no data at this time point
        if total == 0:
            entropies[t] = 0
            continue

        # Probability distribution
        p = counts / total

        # Raw Shannon entropy
        raw_entropy = -np.nansum(p * np.log(p))

        # Number of states that actually appear
        states_used = np.count_nonzero(counts)
        
        if states_used > 1:
            max_entropy = np.log(num_classes)
            entropies[t] = raw_entropy / max_entropy
        else:
            # If only one state is used, entropy is 0 => normalized is 0
            entropies[t] = 0.0

    return entropies


def entropy_per_trial(data, num_classes=None):
    """
    Compute the normalized Shannon entropy for each trial (over time).
    Normalized so that the entropy is in [0, 1], where 0 = only one state used
    in that trial, 1 = perfectly uniform usage over all states that appear.

    Parameters
    ----------
    data : list or ndarray of shape (n_trials, n_times)
        data[i_trial] = array of states at each time for that trial
    num_classes : int or None
        Number of discrete states. If None, will be inferred from data.

    Returns
    -------
    entropies : ndarray, shape (n_trials,)
        Normalized entropy for each trial in [0, 1].
    """
    n_trials = len(data)
    
    # If user does not specify number of classes, infer from data
    if num_classes is None:
        num_classes = int(data.max()) + 1

    entropies = np.zeros(n_trials)

    for i_trial in range(n_trials):
        trial_states = data[i_trial].astype(np.int64)

        # Counts of each class within this trial
        counts = np.bincount(trial_states, minlength=num_classes)
        total = counts.sum()
        
        if total == 0:
            # Edge case: no timepoints in this trial
            entropies[i_trial] = 0
            continue
        
        p = counts / total
        
        # Raw Shannon entropy
        raw_entropy = -np.nansum(p * np.log(p))

        # Number of states actually used
        states_used = np.count_nonzero(counts)
        
        if states_used > 1:
            max_entropy = np.log(num_classes)
            entropies[i_trial] = raw_entropy / max_entropy
        else:
            entropies[i_trial] = 0.0

    return entropies


def transition_penalty_per_trial(data):
    """
    Compute a transition-based penalty for each trial.
    The penalty is the fraction of time steps where a transition occurs
    (i.e., the number of state changes divided by the total possible transitions).

    Parameters
    ----------
    data : ndarray or list of shape (n_trials, n_timepoints)
        data[i_trial] = array of discrete state labels for each time point.

    Returns
    -------
    penalties : ndarray of shape (n_trials,)
        Transition penalty in [0, 1] for each trial.
        0 = no transitions, 1 = transitions at every possible time step.
    """
    n_trials = len(data)
    penalties = np.zeros(n_trials)

    for i_trial in range(n_trials):
        trial_states = data[i_trial]

        # If there's only 1 time point or an empty trial, no transitions
        if len(trial_states) <= 1:
            penalties[i_trial] = 0.0
            continue

        # Count the number of state changes
        n_transitions = np.sum(trial_states[1:] != trial_states[:-1])

        # Fraction of transitions relative to the maximum possible (which is T-1)
        T = len(trial_states)
        frac_transitions = n_transitions / (T - 1)

        penalties[i_trial] = frac_transitions

    return penalties


def entropy_per_time_point_lumped(data, num_classes=None):
    """
    Compute the lumped entropy at each time point, normalized in [0,1].
    For each time point t:
      1. Find the dominant state (the one that appears the most across trials).
      2. Lump all other states into a single 'other' category.
      3. Compute Shannon entropy of this 2-bin distribution and normalize by ln(2).
         This ensures the result is in [0,1].

    Parameters
    ----------
    data : ndarray of shape (n_trials, n_times)
        Discrete states for each trial x time.
    num_classes : int or None
        Number of possible discrete states (used for bin-counting). If None,
        it is inferred as the max state value + 1 in `data`.

    Returns
    -------
    entropies : ndarray of shape (n_times,)
        The normalized lumped entropy at each time point in [0, 1].
    """
    n_trials, n_times = data.shape
    
    # If user does not specify number of classes, infer from data
    if num_classes is None:
        num_classes = int(data.max()) + 1
    
    entropies = np.zeros(n_times)

    for t in range(n_times):
        # Counts of each class at time point t
        counts = np.bincount(data[:, t].astype(np.int64), minlength=num_classes)
        total = counts.sum()
        
        if total == 0:
            # Edge case: no data at this time point
            entropies[t] = 0
            continue
        
        # Dominant class count
        dominant_count = np.max(counts)
        p_dominant = dominant_count / total
        p_other = 1.0 - p_dominant
        
        # If all trials are in the same state => entropy = 0
        if p_dominant == 0 or p_dominant == 1:
            entropies[t] = 0.0
        else:
            # 2-bin entropy
            raw_entropy = -(p_dominant*np.log(p_dominant) + p_other*np.log(p_other))
            # Normalized by ln(2)
            entropies[t] = raw_entropy / np.log(2.0)

    return entropies


## Custom fill function for 3D waterfall plots
def polygon_under_curve(xs, ys, zs, z_min=0):
    verts = [(x, y, z) for x, y, z in zip(xs, ys, zs)]
    # Connect back to base plane
    verts += [(x, y, z_min) for x, y in zip(xs[::-1], ys[::-1])]
    return [verts]


def extract_turning_points(trial_trajectory, index_min=None, index_max=None):

    ## Compute directional vectors between consecutive time points
    directions = trial_trajectory[1:] - trial_trajectory[:-1]

    ## Compute angles between consecutive directional vectors
    v = directions[:-1]   # shape: (n_times - 2, n_dims)
    w = directions[1:]    # shape: (n_times - 2, n_dims)

    dot_products = np.sum(v * w, axis=1)
    v_norms = np.linalg.norm(v, axis=1)
    w_norms = np.linalg.norm(w, axis=1)

    # Prevent numerical issues by clipping 
    cos_theta = dot_products / (v_norms * w_norms)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angles = np.arccos(cos_theta)  # shape: (n_times - 2,)

    ## Find peaks in the angles
    peaks, _ = find_peaks(angles)

    ## Sort peaks by angle
    peaks = peaks[np.argsort(angles[peaks])]
    angles = np.sort(angles[peaks])

    turning_points = peaks + 1  # Add 1 to account for the shift in indexing
    if index_min is not None and index_max is not None:
        index_filter = (turning_points >= index_min) & (turning_points <= index_max)
        turning_points = turning_points[index_filter]
        angles = angles[index_filter]
    elif index_min is not None:
        index_filter = turning_points >= index_min
        turning_points = turning_points[index_filter]
        angles = angles[index_filter]
    elif index_max is not None:
        index_filter = turning_points <= index_max
        turning_points = turning_points[index_filter]
        angles = angles[index_filter]

    return turning_points, angles


## Find the peak onset and duration of neural speed 
## Peak onset is defined as the time when neural speed crosses below the baseline before the peak
## The baseline is defined as the maximum of neural speed before the start of trial
## Peak duration is defined as the time between the peak onset and the time when neural speed crosses below the half peak value
def compute_neural_speed_peak_onset_and_duration(
    neural_speeds, 
    min_amplitude, 
    time_step, 
    pre_start_time_buffer,
    peak_onset_time_nstd):

    res = {
        'peak_onset_time': [],
        'peak_time':       [],
        'peak_duration':   [],
        'left_base_time':  [],
        'peak_value':      [],
    }

    n_trials = neural_speeds.shape[0]
    for i_trial in range(n_trials):
        neural_speed = neural_speeds[i_trial]

        ## Find the highest peak
        peaks, properties = find_peaks(neural_speed, height=min_amplitude)
        peak_index = peaks[np.argmax(properties['peak_heights'])]

        peak_time = peak_index * time_step - pre_start_time_buffer
        peak_value = neural_speed[peak_index]

        half_peak_value = peak_value * 0.5

        ## Compute peak duration
        left_base = np.where(neural_speed[:peak_index] < half_peak_value)[0]
        if len(left_base) == 0:
            left_base_index = 0
        else:
            left_base_index = left_base[-1]
        
        right_base = np.where(neural_speed[peak_index:] < half_peak_value)[0]
        if len(right_base) == 0:
            right_base_index = len(neural_speed) - 1
        else:
            right_base_index = right_base[0] + peak_index

        left_base_time = left_base_index * time_step - pre_start_time_buffer
        right_base_time = right_base_index * time_step - pre_start_time_buffer

        peak_duration = right_base_time - left_base_time

        ## Compute peak onset
        pre_start_index_buffer = int(pre_start_time_buffer / time_step)
        baseline = neural_speed[:pre_start_index_buffer]

        baseline_mean = baseline.mean()
        baseline_std = baseline.std(ddof=1)
        baseline_threshold = baseline_mean + peak_onset_time_nstd * baseline_std

        # guarantee we find something even if the trace never drops below thr
        onset_idx_candidates = np.where(neural_speed[:peak_index] <= baseline_threshold)[0]
        if len(onset_idx_candidates):
            onset_idx = onset_idx_candidates[-1]
        else:
            onset_idx = 0
        if onset_idx < pre_start_index_buffer:
            onset_idx = pre_start_index_buffer

        peak_onset_time = onset_idx * time_step - pre_start_time_buffer

        ## Store results
        # print(f'Peak onset time: {peak_onset_time:.3f} s | Peak time: {peak_time:.3f} s | Peak duration: {peak_duration:.3f} s | Left base time: {left_base_time:.3f} s | Peak value: {peak_value:.3f} s')

        res['peak_onset_time'].append(peak_onset_time)
        res['peak_time'].append(peak_time)
        res['peak_duration'].append(peak_duration)
        res['left_base_time'].append(left_base_time)
        res['peak_value'].append(peak_value)
    
    print(f'Peak onset time: {np.median(res["peak_onset_time"]):.3f} s | Peak time: {np.median(res["peak_time"]):.3f} s | Peak duration: {np.median(res["peak_duration"]):.3f} s | Left base time: {np.median(res["left_base_time"]):.3f} s | Peak value: {np.median(res["peak_value"]):.3f} s')

    return res


def find_initial_zero_durations(discrete_state_matrix):
    """
    Finds the duration of the initial state 0 in each row of the input 2D array,
    treating -1 as a null state that does not interrupt counting.

    Parameters:
        discrete_state_matrix (np.ndarray): 2D array where each row represents a trial.
                           States can be 0, 1, or -1 (-1 indicates null).

    Returns:
        list: A list of initial durations of state 0 for each row.
    """
    initial_durations = []
    for row in discrete_state_matrix:
        duration = 0
        started_counting = False  # To check if we've started counting initial 0s
        for state in row:
            if state == 0:
                # Start or continue counting initial 0s
                started_counting = True
                duration += 1
            elif state == -1:
                # Ignore null states (-1) without breaking the counting
                if started_counting:
                    continue
            else:
                # Stop counting on encountering a non-0, non--1 state
                break
        initial_durations.append(duration)
    return np.array(initial_durations)


def model_selection(
    results_mean, 
    results_se, 
    ns_states, 
    ns_discrete_states,
    includes_pca=False,
    higher_is_better=False):

    if higher_is_better:
        ## Use best result plus standard error as reference
        res_max = np.max(results_mean)
        i_max, j_max = np.unravel_index(np.argmax(results_mean), results_mean.shape)
        res_max_se = results_se[i_max, j_max]

        ## Get the simplest model that performs better than the reference
        performance_reference = res_max - res_max_se
        qualified_models = np.where((results_mean > performance_reference).T)
    else:
        ## Use best result plus standard error as reference
        res_min = np.min(results_mean)
        i_min, j_min = np.unravel_index(np.argmin(results_mean), results_mean.shape)
        res_min_se = results_se[i_min, j_min]

        ## Get the simplest model that performs better than the reference
        performance_reference = res_min + res_min_se
        qualified_models = np.where((results_mean < performance_reference).T)

    # Select parameters in the top left corner of n_discrete_states x n_continuous_states space
    # Also prioritize selecting model with less discrete states
    qualified_models_index_sum = qualified_models[0] + qualified_models[1]
    best_model_index = np.argmin(qualified_models_index_sum)
    best_model_index = [qualified_models[0][np.argmin(qualified_models_index_sum)], qualified_models[1][np.argmin(qualified_models_index_sum)]]
    
    best_model_n_continuous_states = ns_states[best_model_index[1]]
    if includes_pca:
        best_model_n_discrete_states = ns_discrete_states[best_model_index[0] - 1]
    else:
        best_model_n_discrete_states = ns_discrete_states[best_model_index[0]]

    print(f'Best model: {best_model_n_discrete_states} discrete states, {best_model_n_continuous_states} continuous states')

    # ipdb.set_trace()

    return performance_reference


def analyze_neuron_tuning(
    fr_avg: np.ndarray,          # shape (n_trials,)
    baseline: np.ndarray,        # shape (n_trials,)
    alpha: float = 0.05,
    method: str = "wilcoxon",    # "wilcoxon", "ttest", or "perm"
    two_tailed: bool = True,
):
    """
    Return (is_modulated, p_value, effect_size).
    Effect size is rank-biserial r for Wilcoxon, Cohen's d for t-test,
    or mean-difference / SD for permutation.
    """

    # Preconditions -------------------------------------------------------
    if fr_avg.shape != baseline.shape:
        raise ValueError("fr_avg and baseline must have same shape per-trial")
    if np.allclose(fr_avg, 0):
        return False, 1.0, 0.0

    # Paired differences --------------------------------------------------
    diffs = fr_avg - baseline

    if method == "wilcoxon":
        stat, p = wilcoxon(diffs, alternative="two-sided" if two_tailed else "greater")
        # Rank-biserial correlation
        n_pos = np.sum(diffs > 0)
        n_neg = np.sum(diffs < 0)
        effect = (n_pos - n_neg) / (n_pos + n_neg)
    elif method == "ttest":
        stat, p = ttest_rel(fr_avg, baseline)
        effect = diffs.mean() / diffs.std(ddof=1)  # Cohen's d
    elif method == "perm":
        # Permutation test on signs
        n_perm = 5000
        obs = np.abs(diffs.mean())
        null = np.zeros(n_perm)
        for i in range(n_perm):
            signs = np.random.choice([-1, 1], size=diffs.size)
            null[i] = np.abs((signs * diffs).mean())
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
        effect = obs / diffs.std(ddof=1)
    elif method == 'mwu':
        # Mann-Whitney U test
        stat, p = mannwhitneyu(fr_avg, baseline, alternative='greater')
        # Effect size: rank-biserial correlation
        n_pos = np.sum(fr_avg > baseline)
        n_neg = np.sum(fr_avg < baseline)
        effect = (n_pos - n_neg) / (n_pos + n_neg)
    else:
        raise ValueError(f"Unknown method: {method}")

    return (p < alpha), p, effect


def analyse_neuron_direction_tuning(
        fr_avg: np.ndarray,          # shape (n_trials, ) firing‑rate for *this* neuron in one dir
        baseline: np.ndarray,        # shape (n_trials, ) pre‑start baseline for same neuron
        target_ids: np.ndarray,      # shape (n_trials, ) integers 0…K‑1
        K: int = 8,
        alpha_global: float = 0.05,
        alpha_dir: float   = 0.05,
):
    """
    Returns
    -------
    is_modulated : bool
        Any direction differs from baseline (global test, FDR‑corrected).
    sig_dirs     : ndarray, shape (K,)
        Boolean vector: which of the K directions are significantly > baseline.
    pref_angle   : float
        Firing‑rate‑weighted preferred direction in *degrees* (0–360).
    mod_depth    : float
        Vector strength ρ ∈ [0,1] (1 = perfectly sharp, 0 = flat).
    """
    # --------- collect responses by direction ---------------------------
    dir_bins = [fr_avg[target_ids == k] for k in range(K)]

    # -- Global “any modulation?” Kruskal‑Wallis (robust ANOVA) ----------

    # Handle the case where all firing rates are zero
    if np.all(fr_avg == 0):
        is_modulated = False
        sig_dirs = np.zeros(K, dtype=bool)
        pref_angle = 0.0
        mod_depth = 0.0
        return is_modulated, sig_dirs, pref_angle, mod_depth
    
    H, p_global = kruskal(*dir_bins)
    is_modulated = (p_global < alpha_global)

    # -- Direction‑by‑direction contrasts vs baseline -------------------
    p_each = [
        mannwhitneyu(bin_k, baseline, alternative='greater').pvalue
        for bin_k in dir_bins
    ]
    sig, _ = fdrcorrection(p_each, alpha=alpha_dir)   # BH–FDR
    sig_dirs = sig.astype(bool)

    # --------- Weighted preferred angle --------------------------------
    #  eight compass angles 0°, 45°, …, 315°
    angles_rad = np.deg2rad(np.arange(K) * 360 / K)
    mean_rates = np.array([bin_k.mean() for bin_k in dir_bins])
    vec = np.sum(mean_rates * np.exp(1j * angles_rad))
    pref_angle = (np.degrees(np.angle(vec)) + 360) % 360
    mod_depth  = np.abs(vec) / np.sum(mean_rates)

    return is_modulated, sig_dirs, pref_angle, mod_depth
