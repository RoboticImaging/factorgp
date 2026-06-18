import math
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for deterministic cuBLAS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch

from gp_utils import rbf, Matern32Kernel, AK, DKLKernel, SparseGPR
from eval_utils import compute_gradient_rmse, compute_ssim, print_results_table

from fgp.interpolate import fd_interpolate_cuts
from fgp.graph_learning import (
    build_fgp_from_pi,
    learn_graph_structure,
    build_candidate_edges,
)
from fgp.plotting import draw_edge_histograms
from fgp.acquisition_fns import (
    heuristic_scores,
    compute_particle_stats,
    acquisition_scores,
)

# ── AK / DKL kernel settings ──────────────────────────────────────────────────
AK_DIM_HIDDEN  = 10
AK_DIM_OUTPUT  = 10
AK_MIN_LS      = 0.01   # min primitive lengthscale (in MinMax-normalised [-1,1] space)
AK_MAX_LS      = 0.5    # max primitive lengthscale
AK_INIT_NOISE  = 1.0
AK_LR_HYPER    = 0.01
AK_LR_NN       = 0.001
AK_N_OPT       = 200    # gradient steps per active-learning turn

# ── Geometry ───────────────────────────────────────────────────────────────────

_here = os.path.dirname(__file__)
gt_data = np.genfromtxt(os.path.join(_here, './data/gas/gas_building_diffusion.csv'),
                         delimiter=',')

building = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

S, W_wall = 5, 1
row_sizes = [W_wall, S, S, S, W_wall, S, S, S, W_wall, W_wall]
col_sizes = [W_wall, S, S, S, W_wall, S, S, S, S,       W_wall]

maze = np.ones((sum(row_sizes), sum(col_sizes)), dtype=int)
row_offsets = np.cumsum([0] + row_sizes)
col_offsets  = np.cumsum([0] + col_sizes)
for r in range(building.shape[0]):
    for c in range(building.shape[1]):
        if building[r, c] == 0:
            maze[row_offsets[r]:row_offsets[r+1],
                 col_offsets[c]:col_offsets[c+1]] = 0

x_coord = np.linspace(0, gt_data.shape[1], maze.shape[1])
y_coord = np.linspace(0, gt_data.shape[0], maze.shape[0])
X, Y    = np.meshgrid(x_coord, y_coord)
x_bound = x_coord[-1]
y_bound = y_coord[-1]

nx, ny = 11, 10
grid_x = np.linspace(0, gt_data.shape[1], nx)
grid_y = np.linspace(0, gt_data.shape[0], ny)
ind_X, ind_Y = np.meshgrid(grid_x, grid_y)
inducing_pts = np.column_stack([ind_X.ravel(), ind_Y.ravel()])
hx = grid_x[1] - grid_x[0]

edges, _, _ = build_candidate_edges(inducing_pts)
n_nodes   = len(inducing_pts)
NOISE_VAR_FRAC = 0.1
noise_var = np.var(gt_data[maze == 0]) * NOISE_VAR_FRAC
kappa_sq  = 0.01

free_yr, free_xc = np.where(maze == 0)
all_free_cart = np.stack([free_xc, y_bound - free_yr], axis=1)
all_free_gt   = gt_data[free_yr, free_xc]

test_pts = np.column_stack([X.ravel(), Y.ravel()])

# Ground truth image aligned with test_pts grid (for per-step error maps).
# test_pts[i, j] is at Cartesian (x_coord[j], y_coord[i]).
# Cartesian y = y_coord[i] corresponds to maze row ≈ y_bound - y_coord[i].
_yr_for_y = np.clip(
    np.round((y_bound - y_coord) * (gt_data.shape[0] - 1) / y_bound).astype(int),
    0, gt_data.shape[0] - 1,
)
_xc_for_x = np.clip(
    np.round(x_coord * (gt_data.shape[1] - 1) / x_bound).astype(int),
    0, gt_data.shape[1] - 1,
)
gt_img_full = gt_data[np.ix_(_yr_for_y, _xc_for_x)]  # shape == X.shape

_gt_var        = float(all_free_gt.var())
_gt_data_range = float(gt_img_full.max() - gt_img_full.min())


def _eval_image_metrics(mu_img):
    """SSIM and gradient-RMSE of a predicted image against the ground truth."""
    ssim  = compute_ssim(mu_img, gt_img_full, data_range=_gt_data_range)
    grmse = compute_gradient_rmse(mu_img, gt_img_full)
    return ssim, grmse


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_dataset(sampled_rc):
    coords = np.array(sampled_rc)
    cart   = np.stack([coords[:, 1], y_bound - coords[:, 0]], axis=1)
    zs     = gt_data[coords[:, 0], coords[:, 1]].reshape(-1, 1)
    return cart, zs


def run_svgd(cart, zs, n_iter, verbose=False):
    return learn_graph_structure(
        inducing_xs    = inducing_pts,
        train_xs       = cart,
        train_zs       = zs,
        h              = hx,
        n_particles    = SVGD_N_PARTICLES,
        n_iter         = n_iter,
        lr             = SVGD_LR,
        logit_init     = SVGD_LOGIT_INIT,
        log_kappa_mean = np.log(kappa_sq),
        log_kappa_std  = SVGD_LOG_KAPPA_STD,
        snapshot_every = 9999,
        noise_var      = noise_var,
        verbose        = verbose
    )


def build_map_fgp(cart, zs, pi_map, kappa_map):
    return build_fgp_from_pi(
        inducing_xs = inducing_pts,
        train_xs    = cart,
        train_zs    = zs,
        kappa_sq    = kappa_map,
        grid_size   = [nx, ny],
        h           = hx,
        edges       = edges,
        pi_mean     = pi_map,
        threshold   = 0.5,
        noise_var   = noise_var,
        node_prior_lam = NODE_PRIOR_LAM
    )


def active_edges(pi_map):
    return [edges[i] for i in range(len(pi_map)) if pi_map[i] > 0.5]


def compute_particle_std_img(test_pts, final_pi, final_kappa, cart, zs):
    """Predictive std at test_pts via law-of-total-variance over SVGD particles.

    Each particle uses its own W_cut (respecting that particle's active edges).
    Var[f*] = E_k[Var(f*|Q_k, y)] + Var_k[E(f*|Q_k, y)]
            = (1/K) Σ_k (W_k∘W_k) diag(Λ_k⁻¹)   +   Var_k[W_k μ_k]
               aleatoric                               epistemic
    """
    from scipy.sparse.linalg import splu as _splu
    from scipy.sparse import eye as _eye

    M = len(test_pts)
    K = len(final_pi)
    N = len(inducing_pts)

    particle_preds = np.zeros((M, K))
    diag_var_list  = []
    W_sq_list      = []

    for k, (pi_k, kappa_k) in enumerate(zip(final_pi, final_kappa)):
        act_e_k = active_edges(pi_k)
        W_k = fd_interpolate_cuts(test_pts, [nx, ny], hx, edges, act_e_k)

        fgp_k = build_fgp_from_pi(
            inducing_xs    = inducing_pts,
            train_xs       = cart,
            train_zs       = zs,
            kappa_sq       = kappa_k,
            grid_size      = [nx, ny],
            h              = hx,
            edges          = edges,
            pi_mean        = pi_k,
            threshold      = 0.5,
            noise_var      = noise_var,
            node_prior_lam = NODE_PRIOR_LAM,
        )

        mu_k = fgp_k.get_posterior_mean()                         # (N,)
        particle_preds[:, k] = np.asarray(W_k @ mu_k).ravel()

        Lam_k = fgp_k.joint_belief.lam.tocsc()
        try:
            lu_k = _splu(Lam_k)
        except RuntimeError as e:
            if "singular" not in str(e).lower():
                raise
            lu_k = _splu((Lam_k + 1e-6 * _eye(N, format="csc")).tocsc())
        diag_var_list.append(lu_k.solve(np.eye(N)).diagonal())    # (N,)
        W_sq_list.append(W_k.power(2))

    var_epistemic = particle_preds.var(axis=1)                    # (M,)
    var_aleatoric = sum(
        np.asarray(W_sq @ dv).ravel()
        for W_sq, dv in zip(W_sq_list, diag_var_list)
    ) / K

    return np.sqrt(np.maximum(var_aleatoric + var_epistemic, 0.0)).reshape(X.shape)


def eval_rmse(fgp_opt, pi_map):
    W     = fd_interpolate_cuts(all_free_cart, [nx, ny], hx, edges, active_edges(pi_map))
    mu, _ = fgp_opt.query(all_free_cart, W)
    mse   = float(np.mean((mu.ravel() - all_free_gt) ** 2))
    return float(np.sqrt(mse)), float(mse / _gt_var)


def draw_walls(ax):
    for r in range(building.shape[0]):
        for c in range(building.shape[1]):
            if building[r, c] == 1:
                x0 = float(col_offsets[c]);  x1 = float(col_offsets[c+1])
                y0 = y_bound - float(row_offsets[r+1])
                y1 = y_bound - float(row_offsets[r])
                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                             linewidth=0, facecolor='dimgrey', alpha=0.35, zorder=10))


# ── GPR (AK / DKL) helpers ────────────────────────────────────────────────────

def build_matern_gpr(cart, zs):
    """Sparse Matérn-3/2 GPR on the FGP inducing grid (FITC)."""
    kernel = Matern32Kernel(amplitude=1.0, lengthscale=0.3)
    gpr = SparseGPR(x_train=cart, y_train=zs, kernel=kernel,
                    noise=AK_INIT_NOISE, inducing_xs=inducing_pts,
                    lr_hyper=AK_LR_HYPER, lr_nn=0.001,
                    is_normalized=True, x_domain_width=max(x_bound, y_bound))
    gpr.optimize(num_iter=AK_N_OPT, verbose=False)
    return gpr


def build_ak_gpr(cart, zs):
    """Sparse Attentive-Kernel GPR on the FGP inducing grid (FITC)."""
    kernel = AK(
        amplitude    = 1.0,
        lengthscales = np.linspace(AK_MIN_LS, AK_MAX_LS, AK_DIM_OUTPUT),
        dim_input    = 2,
        dim_hidden   = AK_DIM_HIDDEN,
        dim_output   = AK_DIM_OUTPUT,
        kernel_fn    = rbf,  # gas uses RBF; srtm uses matern32 (default)
    )
    gpr = SparseGPR(x_train=cart, y_train=zs, kernel=kernel,
                    noise=AK_INIT_NOISE, inducing_xs=inducing_pts,
                    lr_hyper=AK_LR_HYPER, lr_nn=AK_LR_NN,
                    is_normalized=True, x_domain_width=max(x_bound, y_bound))
    gpr.optimize(num_iter=AK_N_OPT, verbose=False)
    return gpr


def build_dkl_gpr(cart, zs):
    """Sparse DKL GPR on the FGP inducing grid (FITC)."""
    kernel = DKLKernel(
        amplitude   = 1.0,
        lengthscale = 0.5,
        dim_input   = 2,
        dim_hidden  = AK_DIM_HIDDEN,
        dim_output  = AK_DIM_OUTPUT,
        kernel_fn   = rbf,  # gas uses RBF; srtm uses matern32 (default)
    )
    gpr = SparseGPR(x_train=cart, y_train=zs, kernel=kernel,
                    noise=AK_INIT_NOISE, inducing_xs=inducing_pts,
                    lr_hyper=AK_LR_HYPER, lr_nn=AK_LR_NN,
                    is_normalized=True, x_domain_width=max(x_bound, y_bound))
    gpr.optimize(num_iter=AK_N_OPT, verbose=False)
    return gpr


def eval_rmse_gpr(gpr):
    """RMSE and SMSE of a GPR on all free pixels (raw cart coords)."""
    mu, _ = gpr(all_free_cart)
    mse   = float(np.mean((mu.ravel() - all_free_gt) ** 2))
    return float(np.sqrt(mse)), float(mse / _gt_var)


def gpr_acq(gpr, cart):
    """Gaussian entropy acquisition: log(std) at each candidate location."""
    _, std = gpr(cart)
    return np.log(std.ravel() + 1e-12)   # proportional to ½ log(2πe σ²)


def _gpr_kernel_graph_arrays(gpr, kernel_type):
    """Compute AK/DKL kernel-structure data on the FGP inducing grid.

    Returns arrays needed by `_plot_gpr_kernel_graph`:
        ell_per_node  : (N,)  effective lengthscale at each node
        edge_vis      : (E,)  pairwise visibility zᵢᵀzⱼ on 4-connected edges
        inducing_xs   : (N, 2) inducing locations in raw task space
        kg_edges      : list of (i, j) edge pairs
    """
    x_np   = gpr.x_scaler.preprocess(inducing_pts)        # normalise to [-1, 1]²
    x_t    = torch.tensor(x_np, dtype=torch.float64)
    kernel = gpr.kernel
    kernel.eval()

    # 4-connected edge list (same topology as FGP)
    _, kg_nx, kg_ny = build_candidate_edges(inducing_pts)
    kg_edges = []
    for iy in range(kg_ny):
        for ix in range(kg_nx - 1):
            kg_edges.append((iy * kg_nx + ix, iy * kg_nx + ix + 1))
    for iy in range(kg_ny - 1):
        for ix in range(kg_nx):
            kg_edges.append((iy * kg_nx + ix, (iy + 1) * kg_nx + ix))

    with torch.no_grad():
        if kernel_type == 'AK':
            reps         = kernel.get_representations(x_t).numpy()
            ell_per_node = (reps ** 2) @ kernel.lengthscales.numpy()
            # rescale from normalised [-1,1]² space to real task space
            ell_per_node = ell_per_node * (float(np.mean(gpr.x_scaler.data_ptp))
                                           / gpr.x_scaler.ptp)
            vis_mat      = reps @ reps.T
        else:  # DKL
            feats        = kernel.input_warping(x_t).numpy()
            ell_per_node = np.full(x_t.shape[0], kernel.lengthscale.item())
            fn           = feats / np.maximum(
                np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
            vis_mat      = fn @ fn.T

    edge_vis = np.array([vis_mat[i, j] for i, j in kg_edges])
    return ell_per_node, edge_vis, inducing_pts, kg_edges


def _plot_gpr_kernel_graph(ax, gpr, kernel_type, title=""):
    """Draw kernel-structure graph for AK or DKL on the FGP inducing grid.

    Nodes coloured by effective lengthscale (plasma); edges by visibility zᵢᵀzⱼ (viridis).
    """
    ell_per_node, edge_vis, ind_xs, kg_edges = _gpr_kernel_graph_arrays(gpr, kernel_type)

    e_cmap = plt.cm.viridis
    e_vmin, e_vmax = float(edge_vis.min()), float(edge_vis.max())
    if e_vmin == e_vmax:
        e_vmin, e_vmax = e_vmin - 0.1, e_vmax + 0.1
    e_norm = plt.Normalize(e_vmin, e_vmax)

    for (i, j), vis in zip(kg_edges, edge_vis):
        xi, xj = ind_xs[i], ind_xs[j]
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]],
                color=e_cmap(e_norm(vis)), linewidth=1.2,
                solid_capstyle='round', alpha=0.7, zorder=2)
        mx, my = 0.5 * (xi[0] + xj[0]), 0.5 * (xi[1] + xj[1])
        ax.text(mx, my, f"{vis:.2f}", fontsize=3, ha='center', va='center',
                color='black', zorder=4)

    n_cmap = plt.cm.plasma
    n_vmin, n_vmax = float(ell_per_node.min()), float(ell_per_node.max())
    if n_vmin == n_vmax:
        n_vmin, n_vmax = n_vmin - 0.05, n_vmax + 0.05
    n_norm = plt.Normalize(n_vmin, n_vmax)
    sc = ax.scatter(ind_xs[:, 0], ind_xs[:, 1],
                    c=ell_per_node, cmap=n_cmap, norm=n_norm,
                    s=35, zorder=5, linewidths=0.5, edgecolors='white')

    divider = make_axes_locatable(ax)
    cax_n = divider.append_axes("right", size="5%", pad=0.05)
    cax_e = divider.append_axes("right", size="5%", pad=0.55)
    cb_n = plt.colorbar(sc, cax=cax_n)
    cb_n.set_label("ℓ_eff", fontsize=7); cb_n.ax.tick_params(labelsize=6)
    sm_e = plt.cm.ScalarMappable(cmap=e_cmap, norm=e_norm); sm_e.set_array([])
    cb_e = plt.colorbar(sm_e, cax=cax_e)
    cb_e.set_label("vis  zᵢᵀzⱼ", fontsize=7); cb_e.ax.tick_params(labelsize=6)

    ax.set_title(f"{title}  [{kernel_type}]\n"
                 f"vis∈[{e_vmin:.2f},{e_vmax:.2f}]  "
                 f"ℓ∈[{n_vmin:.3f},{n_vmax:.3f}]", fontsize=7)
    draw_walls(ax)
    ax.set_xlim(0, x_bound); ax.set_ylim(0, y_bound)
    ax.set_aspect('equal')


def _plot_fgp_kernel_graph(ax, pi_map, kappa_map, title=""):
    """Draw MAP FGP edges coloured by ℓ = 1/√κ² (plasma)."""
    included = [i for i, p in enumerate(pi_map) if p > 0.5]
    vals  = 1.0 / np.sqrt(np.maximum(kappa_map, 1e-12))
    inc_vals = vals[included] if included else np.array([0.0, 1.0])
    vmin, vmax = inc_vals.min(), inc_vals.max()
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin, vmax)

    for e_idx in included:
        i, j = edges[e_idx]
        xi, xj = inducing_pts[i], inducing_pts[j]
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]],
                color=cmap(norm(vals[e_idx])), linewidth=2, solid_capstyle='round')
        mx, my = 0.5 * (xi[0] + xj[0]), 0.5 * (xi[1] + xj[1])
        ax.text(mx, my, f"{vals[e_idx]:.2f}", fontsize=5, ha='center', va='center',
                color='black', zorder=4,
                bbox=dict(boxstyle='round,pad=0.1', fc='none', ec='none'))

    ax.scatter(inducing_pts[:, 0], inducing_pts[:, 1],
               c='k', s=18, zorder=3, linewidths=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, label="ℓ = 1/√κ²", fraction=0.046, pad=0.04)
    ax.set_title(f"{title}\n{len(included)}/{len(edges)} edges  "
                 f"ℓ∈[{inc_vals.min():.2f},{inc_vals.max():.2f}]", fontsize=7)
    draw_walls(ax)
    ax.set_xlim(0, x_bound); ax.set_ylim(0, y_bound)
    ax.set_aspect('equal')


# ── Experiment config ─────────────────────────────────────────────────────────

SEED        = 0
N_init      = 1
N_budget    = 60
PLOT_STEP_FIGS  = False  # True → save per-step slim figure (mean/std/err + kernel at final step)

SVGD_N_PARTICLES   = 30
SVGD_LR            = 0.05
SVGD_LOGIT_INIT    = 0.2
SVGD_LOG_KAPPA_STD = 1.0
NODE_PRIOR_LAM     = 1e-10

FGP_NAMES = ['heuristic', 'heuristic_field', 'heuristic_graph']
GPR_NAMES = ['matern', 'ak', 'dkl']
STRATEGY_NAMES = FGP_NAMES + GPR_NAMES

# (alpha, beta) for each heuristic ablation variant:
#   heuristic       alpha=1, beta=1  — equal weight (field + graph)
#   heuristic_field alpha=1, beta=0  — field uncertainty only
#   heuristic_graph alpha=0, beta=1  — graph-based exploration only
HEURISTIC_PARAMS = {
    'heuristic':       (1, 1),
    'heuristic_field': (1, 0),
    'heuristic_graph': (0, 1),
}

STRATEGY_LABELS = {
    'joint_mi':        r'$I(y_*;\,Q,f_U\mid D)$',
    'field_mi':        r'$I(y_*;\,f_U\mid Q,D)$',
    'graph_mi':        r'$I(y_*;\,Q\mid D)$',
    'field_eig':       r'Field EIG',
    'heuristic':       r'Heuristic ($\alpha$=1,$\beta$=1)',
    'heuristic_field': r'Heuristic field ($\alpha$=1,$\beta$=0)',
    'heuristic_graph': r'Heuristic graph ($\alpha$=0,$\beta$=1)',
    'matern':          r'Mat\'{e}rn-3/2 (sparse)',
    'ak':              'AK (sparse)',
    'dkl':             'DKL (sparse)',
}
STYLE = {
    'joint_mi':        dict(color='tab:blue',   linestyle='-',            marker='o'),
    'field_mi':        dict(color='tab:orange', linestyle='--',           marker='s'),
    'graph_mi':        dict(color='tab:red',    linestyle='-.',           marker='D'),
    'field_eig':       dict(color='tab:green',  linestyle=(0, (5, 1)),   marker='^'),
    'heuristic':       dict(color='tab:purple', linestyle=(0,(3,1,1,1)), marker='P'),
    'heuristic_field': dict(color='tab:pink',   linestyle='--',           marker='P'),
    'heuristic_graph': dict(color='tab:gray',   linestyle='-.',           marker='P'),
    'matern':          dict(color='tab:olive',  linestyle=':',            marker='x'),
    'ak':              dict(color='tab:brown',  linestyle='-',            marker='v'),
    'dkl':             dict(color='tab:cyan',   linestyle='--',           marker='<'),
}

vmin = float(gt_data.min())
vmax = float(gt_data.max())

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.use_deterministic_algorithms(True)

rng      = np.random.default_rng(SEED)
init_idx = rng.choice(len(free_xc), size=N_init, replace=False)
init_rc  = [(int(free_yr[i]), int(free_xc[i])) for i in init_idx]

state = {
    name: {'sampled_rc': list(init_rc),
           'rmse_hist': [], 'smse_hist': [], 'ssim_hist': [], 'grmse_hist': [],
           'n_hist': []}
    for name in STRATEGY_NAMES
}

# ── Main loop ─────────────────────────────────────────────────────────────────

# Storage for last-step GPR models (used for final kernel-graph plot)
last_gpr = {}
n_iter = 200
for step in range(N_budget + 1):
    print(f"\n── Step {step} ──────────────────────────────────────────")
    step_data = {}

    # ── FGP strategies ────────────────────────────────────────────────────────
    for name in FGP_NAMES:
        s        = state[name]
        cart, zs = build_dataset(s['sampled_rc'])

        final_pi, final_kappa, _, snapshots = run_svgd(
            cart, zs, n_iter,
            verbose=(step == 0 and name == 'joint_mi'),
        )

        arg_max   = int(np.argmax(snapshots[-1][-1]))
        pi_map    = final_pi[arg_max]
        kappa_map = final_kappa[arg_max]
        fgp_opt      = build_map_fgp(cart, zs, pi_map, kappa_map)
        rmse, smse   = eval_rmse(fgp_opt, pi_map)
        s['rmse_hist'].append(rmse)
        s['smse_hist'].append(smse)
        s['n_hist'].append(len(s['sampled_rc']))
        print(f"  {name:12s} | n={len(s['sampled_rc']):3d} | RMSE={rmse:.4f} | SMSE={smse:.4f}")

        # Candidates: unsampled free pixels
        sampled_set = set(s['sampled_rc'])
        cand_mask   = np.array([(int(free_yr[k]), int(free_xc[k])) not in sampled_set
                                 for k in range(len(free_yr))])
        cand_yr   = free_yr[cand_mask]
        cand_xc   = free_xc[cand_mask]
        cand_cart = np.stack([cand_xc, y_bound - cand_yr], axis=1)

        act_e      = active_edges(pi_map)
        W_test_map = fd_interpolate_cuts(test_pts, [nx, ny], hx, edges, act_e)
        mu_fld, _  = fgp_opt.query(test_pts, W_test_map)
        mu_img_fgp = mu_fld.reshape(X.shape)
        _ssim, _grmse = _eval_image_metrics(mu_img_fgp)
        s['ssim_hist'].append(_ssim)
        s['grmse_hist'].append(_grmse)
        if step == N_budget:
            std_img = compute_particle_std_img(test_pts, final_pi, final_kappa, cart, zs)
            err_img = np.abs(mu_img_fgp - gt_img_full)
        else:
            std_img = err_img = None

        if name in HEURISTIC_PARAMS:
            _alpha, _beta = HEURISTIC_PARAMS[name]
            pi_std   = final_pi.std(axis=0)
            acq_cand, _, _ = heuristic_scores(
                cand_cart, fgp_opt, pi_map, pi_std, edges, [nx, ny], hx,
                alpha=_alpha, beta=_beta)
            best_k   = int(np.argmax(acq_cand)) if len(acq_cand) > 0 else None
            free_test_mask = (maze.ravel() == 0)
            acq_arr, gmrf_n, graph_n = heuristic_scores(
                test_pts, fgp_opt, pi_map, pi_std, edges, [nx, ny], hx,
                alpha=_alpha, beta=_beta, norm_mask=free_test_mask)
            E_var_n  = gmrf_n.reshape(X.shape)
            Var_mu_n = graph_n.reshape(X.shape)
            acq_img  = acq_arr.reshape(X.shape)
        else:
            mu_cand, var_cand, css_cand = compute_particle_stats(
                cand_cart, final_pi, final_kappa, cart, zs, edges, [nx, ny], hx, noise_var)
            acq    = acquisition_scores(name, mu_cand, var_cand, noise_var, cross_ssq=css_cand)
            best_k = int(np.argmax(acq)) if len(acq) > 0 else None
            mu_test, var_test, css_test = compute_particle_stats(
                test_pts, final_pi, final_kappa, cart, zs, edges, [nx, ny], hx, noise_var)
            E_var_fld  = var_test.mean(axis=1).reshape(X.shape)
            Var_mu_fld = mu_test.var(axis=1).reshape(X.shape)
            E_var_n    = E_var_fld  / (E_var_fld.max()  + 1e-12)
            Var_mu_n   = Var_mu_fld / (Var_mu_fld.max() + 1e-12)
            acq_test = acquisition_scores(name, mu_test, var_test, noise_var, cross_ssq=css_test)
            acq_img  = acq_test.reshape(X.shape)

        next_cart = cand_cart[best_k] if (best_k is not None and step < N_budget) else None

        step_data[name] = dict(
            is_gpr=False,
            final_pi=final_pi, pi_map=pi_map, kappa_map=kappa_map, rmse=rmse,
            sampled_rc=list(s['sampled_rc']),
            mu_img=mu_img_fgp,
            std_img=std_img, err_img=err_img,
            E_var_n=E_var_n, Var_mu_n=Var_mu_n, acq_img=acq_img,
            next_cart=next_cart,
            cand_yr=cand_yr, cand_xc=cand_xc, best_k=best_k,
        )

    # ── GPR strategies (AK / DKL) ─────────────────────────────────────────────
    for name in GPR_NAMES:
        s        = state[name]
        cart, zs = build_dataset(s['sampled_rc'])

        if name == 'matern':
            gpr = build_matern_gpr(cart, zs)
        elif name == 'ak':
            gpr = build_ak_gpr(cart, zs)
        else:
            gpr = build_dkl_gpr(cart, zs)
        last_gpr[name] = gpr   # keep for kernel-graph plot

        rmse, smse = eval_rmse_gpr(gpr)
        s['rmse_hist'].append(rmse)
        s['smse_hist'].append(smse)
        s['n_hist'].append(len(s['sampled_rc']))
        print(f"  {name:12s} | n={len(s['sampled_rc']):3d} | RMSE={rmse:.4f} | SMSE={smse:.4f}")

        sampled_set = set(s['sampled_rc'])
        cand_mask   = np.array([(int(free_yr[k]), int(free_xc[k])) not in sampled_set
                                 for k in range(len(free_yr))])
        cand_yr   = free_yr[cand_mask]
        cand_xc   = free_xc[cand_mask]
        cand_cart = np.stack([cand_xc, y_bound - cand_yr], axis=1)

        acq_cand = gpr_acq(gpr, cand_cart)
        best_k   = int(np.argmax(acq_cand)) if len(acq_cand) > 0 else None

        mu_fld, std_fld = gpr(test_pts)
        mu_img_gpr = mu_fld.reshape(X.shape)
        _ssim, _grmse = _eval_image_metrics(mu_img_gpr)
        s['ssim_hist'].append(_ssim)
        s['grmse_hist'].append(_grmse)
        if step == N_budget:
            std_img = std_fld.reshape(X.shape)
            err_img = np.abs(mu_img_gpr - gt_img_full)
        else:
            std_img = err_img = None
        acq_test   = gpr_acq(gpr, test_pts)
        next_cart  = cand_cart[best_k] if (best_k is not None and step < N_budget) else None

        step_data[name] = dict(
            is_gpr=True,
            gpr=gpr, rmse=rmse,
            sampled_rc=list(s['sampled_rc']),
            mu_img=mu_img_gpr,
            std_img=std_img, err_img=err_img,
            acq_img=acq_test.reshape(X.shape),
            next_cart=next_cart,
            cand_yr=cand_yr, cand_xc=cand_xc, best_k=best_k,
        )

    # ── Slim figure: heuristic / AK / DKL  (every step) ─────────────────────
    if PLOT_STEP_FIGS:
        _strats      = STRATEGY_NAMES
        _is_final  = (step == N_budget)
        _nrows     = 4 if _is_final else 3
        _ncols     = len(_strats)
        _fig_s, _axes_s = plt.subplots(_nrows, _ncols, figsize=(4.5 * _ncols, 4.5 * _nrows))
        for _col, _name in enumerate(_strats):
            _d   = step_data[_name]
            _src = np.array(_d['sampled_rc'])
            _sx  = _src[:, 1].astype(float)
            _sy  = (y_bound - _src[:, 0]).astype(float)
            _nc  = _d['next_cart']

            def _obs(_ax, _sx=_sx, _sy=_sy, _nc=_nc):
                _ax.scatter(_sx, _sy, c='white', s=40, linewidths=0.5,
                            edgecolors='0.4', zorder=5)
                if _nc is not None:
                    _ax.scatter(*_nc, c='#d62728', s=130, marker='*',
                                zorder=15, edgecolors='k', linewidths=0.4)

            def _frm(_ax):
                draw_walls(_ax)
                _ax.set_xlim(0, x_bound); _ax.set_ylim(0, y_bound)
                _ax.set_aspect('equal'); _ax.set_xticks([]); _ax.set_yticks([])

            # row 0 — predictive mean
            _ax = _axes_s[0, _col]
            _cf = _ax.contourf(X, Y, _d['mu_img'], levels=20,
                               cmap='viridis', vmin=_d['mu_img'].min(), vmax=_d['mu_img'].max())
            _fig_s.colorbar(_cf, ax=_ax, fraction=0.046, pad=0.02)
            if not _d['is_gpr']:
                draw_edge_histograms(_ax, inducing_pts, edges,
                                     _d['final_pi'], _d['pi_map'])
            _obs(_ax); _frm(_ax)
            _ax.set_title(f"{STRATEGY_LABELS[_name]}  RMSE={_d['rmse']:.3f}", fontsize=9)
            if _col == 0: _ax.set_ylabel('Mean', fontsize=9)

            # row 1 — predictive std
            _ax = _axes_s[1, _col]
            _cf2 = _ax.contourf(X, Y, _d['std_img'], levels=20,
                                cmap='viridis', vmin=0, vmax=_d['std_img'].max() + 1e-12)
            _fig_s.colorbar(_cf2, ax=_ax, fraction=0.046, pad=0.02)
            _obs(_ax); _frm(_ax)
            if _col == 0: _ax.set_ylabel('Predictive std', fontsize=9)

            # row 2 — abs error
            _ax = _axes_s[2, _col]
            _cf3 = _ax.contourf(X, Y, _d['err_img'], levels=20,
                                cmap='inferno', vmin=0, vmax=_d['err_img'].max() + 1e-12)
            _fig_s.colorbar(_cf3, ax=_ax, fraction=0.046, pad=0.02)
            _obs(_ax); _frm(_ax)
            if _col == 0: _ax.set_ylabel('Abs error', fontsize=9)

            # row 3 — kernel / graph structure (final step only)
            if _is_final:
                _ax = _axes_s[3, _col]
                if _name in HEURISTIC_PARAMS:
                    _plot_fgp_kernel_graph(
                        _ax, _d['pi_map'], _d['kappa_map'],
                        title=f"FGP graph  (n={len(_d['sampled_rc'])})")
                elif _name == 'matern':
                    _ax.set_visible(False)
                elif _name in last_gpr:
                    _ktype = 'AK' if _name == 'ak' else 'DKL'
                    _plot_gpr_kernel_graph(
                        _ax, last_gpr[_name], _ktype,
                        title=f"{_ktype} kernel  (n={len(_d['sampled_rc'])})")
                if _col == 0: _ax.set_ylabel('Kernel / graph', fontsize=9)

        _fig_s.suptitle(f'Step {step}', fontsize=10, fontweight='bold')
        _fig_s.tight_layout(rect=[0, 0, 1, 0.97])
        _fig_s.savefig(os.path.join(_here, f'gas_step_{step:03d}.png'), dpi=150)
        plt.close(_fig_s)
        print(f"  → saved gas_step_{step:03d}.png")

    # ── Advance: add next sample for each strategy ────────────────────────────
    if step == N_budget:
        break
    for name in STRATEGY_NAMES:
        d = step_data[name]
        if d['best_k'] is not None:
            state[name]['sampled_rc'].append(
                (int(d['cand_yr'][d['best_k']]), int(d['cand_xc'][d['best_k']]))
            )

# ── Final results table ───────────────────────────────────────────────────────

rows = [dict(
            label = name,
            rmse  = state[name]['rmse_hist'][-1],
            smse  = state[name]['smse_hist'][-1],
            grmse = state[name]['grmse_hist'][-1],
            ssim3 = compute_ssim(step_data[name]['mu_img'], gt_img_full,
                                 data_range=_gt_data_range, win_size=3),
        ) for name in STRATEGY_NAMES]
print_results_table(rows, title=f"Gas diffusion  —  Final metrics  (N = {N_budget} samples)")


_n_cols = len(STRATEGY_NAMES)
_fig_f, _axes_f = plt.subplots(3, _n_cols, figsize=(4 * _n_cols, 12))
_fig_f.suptitle(
    f'Gas diffusion — Final predictions (N = {N_budget} samples)',
    fontsize=10, fontweight='bold',
)

for _col, _name in enumerate(STRATEGY_NAMES):
    _d = step_data[_name]
    _src = np.array(_d['sampled_rc'])
    _sx  = _src[:, 1].astype(float)
    _sy  = (y_bound - _src[:, 0]).astype(float)

    def _fin_f(_ax):
        draw_walls(_ax)
        _ax.set_xlim(0, x_bound); _ax.set_ylim(0, y_bound)
        _ax.set_aspect('equal'); _ax.set_xticks([]); _ax.set_yticks([])

    def _pts(_ax, _sx=_sx, _sy=_sy):
        _ax.scatter(_sx, _sy, c='white', s=18, linewidths=0.4,
                    edgecolors='0.4', zorder=5)

    _ax = _axes_f[0, _col]
    _cf = _ax.contourf(X, Y, _d['mu_img'], levels=20, cmap='viridis',
                       vmin=_d['mu_img'].min(), vmax=_d['mu_img'].max())
    _fig_f.colorbar(_cf, ax=_ax, fraction=0.046, pad=0.02)
    _fin_f(_ax); _pts(_ax)
    _ax.set_title(f"{STRATEGY_LABELS[_name]}\nRMSE={_d['rmse']:.3f}", fontsize=8)
    if _col == 0:
        _ax.set_ylabel('Mean prediction', fontsize=8, labelpad=6)

    _ax = _axes_f[1, _col]
    _cf2 = _ax.contourf(X, Y, _d['std_img'], levels=20, cmap='viridis',
                        vmin=0, vmax=_d['std_img'].max() + 1e-12)
    _fig_f.colorbar(_cf2, ax=_ax, fraction=0.046, pad=0.02)
    _fin_f(_ax); _pts(_ax)
    if _col == 0:
        _ax.set_ylabel('Predictive std', fontsize=8, labelpad=6)

    _ax = _axes_f[2, _col]
    _cf3 = _ax.contourf(X, Y, _d['err_img'], levels=20, cmap='inferno',
                        vmin=0, vmax=_d['err_img'].max() + 1e-12)
    _fig_f.colorbar(_cf3, ax=_ax, fraction=0.046, pad=0.02)
    _fin_f(_ax); _pts(_ax)
    if _col == 0:
        _ax.set_ylabel('Abs error', fontsize=8, labelpad=6)

_fig_f.tight_layout(rect=[0, 0, 1, 0.97])
_out_f = os.path.join(_here, 'gas_final_comparison.png')
_fig_f.savefig(_out_f, dpi=150, bbox_inches='tight')
plt.close(_fig_f)
print(f"Figure saved → {_out_f}")

# ── Final kernel / graph structure comparison ─────────────────────────────────
# Rows: 3 FGP heuristic variants + AK  (4 rows × 1 col)
_kg_rows = [
    ('heuristic',       'fgp'),
    ('heuristic_field', 'fgp'),
    ('heuristic_graph', 'fgp'),
    ('ak',              'gpr'),
]
fig_kg, axes_kg = plt.subplots(4, 1, figsize=(7, 7 * 4))
fig_kg.suptitle(
    f'Kernel / graph structure after {N_budget} active sampling steps',
    fontsize=11,
)

for row, (kg_name, kg_type) in enumerate(_kg_rows):
    ax = axes_kg[row]
    if kg_type == 'fgp':
        _d = step_data[kg_name]
        _plot_fgp_kernel_graph(
            ax, _d['pi_map'], _d['kappa_map'],
            title=f"FGP {kg_name}  (n={state[kg_name]['n_hist'][-1]})",
        )
    else:
        if kg_name in last_gpr:
            _plot_gpr_kernel_graph(
                ax, last_gpr[kg_name], 'AK',
                title=f"AK  (n={state[kg_name]['n_hist'][-1]})",
            )

fig_kg.tight_layout()
_out_kg = os.path.join(_here, 'gas_kernel_graph_comparison.png')
fig_kg.savefig(_out_kg, dpi=150, bbox_inches='tight')
plt.close(fig_kg)
print(f"Figure saved → {_out_kg}")
