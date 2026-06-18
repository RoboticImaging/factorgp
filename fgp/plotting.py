import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

_NODE_S   = 3       # scatter marker size (pt^2)
_ELW      = 0.5     # base edge linewidth for present edges
_ELW_CUT  = 0.25    # linewidth for cut edges (ghosted)
_AX_LW    = 0.35    # spine / axis linewidth (matches rcParams)
_CB_FRAC  = 0.04    # colorbar fraction of axis width
_CB_PAD   = 0.01    # colorbar padding

def draw_field(ax, Xt, Yt, Z, vmin=None, vmax=None,
               cmap='viridis', x_train=None, y_train=None, fig=None):
    '''
    Draws 2D spatial field. Optionally plot training data / sample locations
    
    Args:
        ax   (matplotlib.axes) : axes to draw on
        Xt   (np.array)        : x-coordinates in meshgrid format (N, N)
        Yt   (np.array)        : y-coordinates in meshgrid format (N, N)
        Z    (np.array)        : field values corresponding to meshgrid points (N^2,1)
        vmin (float)           : minimum field value
        vmax (float)           : maximum field value
        cmap (string)          : string identifying matplotlib colourmap
        x_train (np.array)     : x-coordinates of training data/sample locations
        y_train (np.array)     : y-coordinates of training data/sample locations
        fig (matplotlib.figure): figure to draw colourbar on
    '''
    cf = ax.contourf(Xt, Yt, Z.reshape(Xt.shape),
                     levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    if x_train is not None:
        ax.scatter(x_train, y_train, c='w', s=_NODE_S,
                   linewidths=0.2, edgecolors='0.3', zorder=5)
    ax.set_aspect('equal')
    # ax.set_xlim(0, 5); ax.set_ylim(0, 5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(_AX_LW)
    if fig is not None:
        cb = fig.colorbar(cf, ax=ax, fraction=_CB_FRAC, pad=_CB_PAD)
        cb.ax.tick_params(labelsize=3.5, length=1.5, width=0.25)
        cb.outline.set_linewidth(0.25)

def draw_hard_graph(ax, ind, edges, pi_mean, threshold=0.5):
    """
    Draws the graph structure given nodes, edge candidates,
    probability of inclusion, and user-defined inclusion threshold.

    Args:
        ax (matplotlib.axes) : axes to draw on
        ind (np.array)       : inducing points / graph node locations (N,2)
        edges (list)         : list of tuples containing start/end node indices
        pi_mean (np.array)   : array of edge inclusion probabilities
        threshold (float)    : threshold for edge inclusion
    """
    for e_idx, (i, j) in enumerate(edges):
        xi, xj = ind[i], ind[j]
        kept = pi_mean[e_idx] >= threshold
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]],
                color='#2ca02c' if kept else '#d62728',
                lw=_ELW if kept else _ELW_CUT,
                alpha=0.85 if kept else 0.25,
                solid_capstyle='round', zorder=3)
    ax.scatter(ind[:, 0], ind[:, 1],
               c='k', s=_NODE_S, linewidths=0, zorder=5)

def draw_soft_graph(ax, ind, edges, pi_mean, pi_std, fig=None):
    """
    Edges coloured by posterior inclusion probability. 
    Variance indicated with edge width (wider is higher variance)

    Args:
        ax (matplotlib.ax)   : axes to draw on
        ind (np.array)       : inducing point / graph node coordinates (N,2)
        edges (list)         : list of tuples indicating (start, end) node indices
        pi_mean (np.array)   : array of inclusion probabilities
        pi_std (np.array)    : array of variances
        fig (matplotlib.fig) : fig to draw colourbar on.
    """
    p_lo=0.3
    p_hi=0.7
    norm = Normalize(p_lo, p_hi)
    cmap = cm.RdYlGn
    for e_idx, p in enumerate(pi_mean):
        i, j = edges[e_idx]
        xi, xj = ind[i], ind[j]
        p  = float(p)
        s  = float(pi_std[e_idx])
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]],
                color=cmap(norm(p)),
                lw=_ELW_CUT + (_ELW - _ELW_CUT) * p + 0.6 * sOutputs,
                solid_capstyle='round', zorder=3)
    ax.scatter(ind[:, 0], ind[:, 1],
               c='k', s=_NODE_S, linewidths=0, zorder=5)
    if fig is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=_CB_FRAC, pad=_CB_PAD)
        cb.ax.tick_params(labelsize=3.5, length=1.5, width=0.25)
        cb.outline.set_linewidth(0.25)
        cb.set_label('$p(e_{ij}\mid\mathcal{D})$', labelpad=1.5)

def draw_edge_histograms(ax, inducing_pts, edges, pi_samples, pi_map,
                         n_bins=20, threshold=0.5):
    """
    Draws per-edge mini-histograms of posterior inclusion probability samples.

    Each edge gets a bar chart oriented along the edge, coloured green/red
    according to whether pi_map exceeds the threshold.

    Args:
        ax           (matplotlib.axes) : axes to draw on
        inducing_pts (np.array)        : node coordinates (N, 2)
        edges        (list)            : list of (i, j) node-index tuples
        pi_samples   (np.array)        : posterior samples, shape (n_particles, n_edges)
        pi_map       (np.array)        : MAP inclusion probability per edge (n_edges,)
        n_bins       (int)             : number of histogram bins over [0, 1]
        threshold    (float)           : inclusion threshold for bar colour
    """
    for e_idx, (i, j) in enumerate(edges):
        xi, xj = inducing_pts[i], inducing_pts[j]
        dx, dy = xj - xi
        length = np.sqrt(dx**2 + dy**2)
        tx, ty = dx / length, dy / length   # unit tangent along edge
        px, py = -ty, tx                    # unit normal (90° CCW)
        mid = (xi + xj) / 2.0

        pi_s = pi_samples[:, e_idx]
        counts, _ = np.histogram(pi_s, bins=n_bins, range=(0.0, 1.0))
        bar_h = length / n_bins
        colour = '#2ca02c' if pi_map[e_idx] >= threshold else '#d62728'

        for b, cnt in enumerate(counts):
            if cnt == 0:
                continue
            bc = (b + 0.5) / n_bins
            half_len = cnt / len(pi_s)
            t_offset = (bc - 0.5) * length
            cx = mid[0] + t_offset * tx
            cy = mid[1] + t_offset * ty
            corners = np.array([
                [cx + half_len*px - (bar_h/2)*tx, cy + half_len*py - (bar_h/2)*ty],
                [cx + half_len*px + (bar_h/2)*tx, cy + half_len*py + (bar_h/2)*ty],
                [cx - half_len*px + (bar_h/2)*tx, cy - half_len*py + (bar_h/2)*ty],
                [cx - half_len*px - (bar_h/2)*tx, cy - half_len*py - (bar_h/2)*ty],
            ])
            ax.add_patch(mpatches.Polygon(corners, closed=True,
                                          facecolor=colour, edgecolor='none',
                                          alpha=1, zorder=4))

        tick_len = 0.1
        ax.plot([mid[0] - 0.05*px, mid[0] + 0.05*px],
                [mid[1] - tick_len*py, mid[1] + tick_len*py],
                color='#000000', lw=0.8, alpha=0.7, zorder=5)
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]],
                color='#888888', lw=0.4, alpha=0.7,
                solid_capstyle='round', zorder=3)