#!/usr/bin/env python3
"""
Usage
-----
    python run_experiment.py --env N17E073
    python run_experiment.py --env N43W080 --samples 700 --seed 0
"""

import argparse
import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import ticker
from PIL import Image
from scipy.sparse import eye as _speye
from scipy.sparse.linalg import splu as _splu
from skimage import transform

import matplotlib
matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))

from fgp.graph_learning import learn_graph_structure, build_fgp_from_pi
from fgp.acquisition_fns import heuristic_scores
from gpytoolbox import fd_interpolate

from gp_utils import StandardScaler, Matern32Kernel, AK, DKLKernel, SparseGPR
from eval_utils import compute_gradient_rmse, compute_ssim, print_results_table

plt.rcParams["image.origin"]        = "lower"
plt.rcParams["image.cmap"]          = "jet"
plt.rcParams["image.interpolation"] = "gaussian"

# ══════════════════════════════════════════════════════════════════════════════
#  Hyperparameter set up
# ══════════════════════════════════════════════════════════════════════════════

# Experiment
MAX_NUM_SAMPLES  = 700
NUM_INIT_SAMPLES = 1
NOISE_SCALE      = 1.0
EVAL_GRID        = [50, 50]
TASK_EXTENT      = [-10.0, 10.0, -10.0, 10.0]
ENV_EXTENT       = [-11.0, 11.0, -11.0, 11.0]
NUM_CANDIDATES   = 1000
CONTROL_RATE     = 10.0
SENSING_RATE     = 1.0
MAX_LIN_VEL      = 1.0
TOLERANCE        = 0.1

# Shared GP
INIT_NOISE  = 1.0
LR_HYPER    = 0.01
LR_NN       = 0.001
JITTER      = 1e-6
DIM_INPUT   = 2
DIM_HIDDEN  = 10
DIM_OUTPUT  = 10

# Stationary  —  Matern-3/2 FITC
STAT_AMPLITUDE   = 1.0
STAT_LENGTHSCALE = 0.5

# DKL  —  Deep Kernel Learning FITC
DKL_AMPLITUDE   = 1.0
DKL_LENGTHSCALE = 0.5

# AK  —  Attentive Kernel FITC
AK_AMPLITUDE = 1.0
AK_MIN_LS    = 0.01
AK_MAX_LS    = 0.50

# FGP  —  Factor-GP, cold SVGD restart each turn
FGP_GRID_N         = 20
FGP_N_PARTICLES    = 50
FGP_N_ITER         = 100
FGP_LR             = 0.05
FGP_LOG_KAPPA_STD  = 1.0
FGP_GRAPH_RELEARN  = 100   # re-learn graph every N new observations
FGP_HEURISTIC_BETA = 1.0

# Per-dataset init params for reproducing FGP results
# (logit_init, node_prior_lam, log_kappa_mean, seed)
ENV_CONFIG = {
    "N17E073": (0.6,  1.0,      1.5069,  0),
    "N43W080": (0.8,  1.0,     -0.7957,  0),
    "N45W123": (0.6,  9.025e-4, 2.2000,  0),
    "N47W124": (0.6,  9.025e-4, 1.6892,  0),
}

# ── derived FGP geometry ──────────────────────────────────────────────────────
_W             = TASK_EXTENT[1] - TASK_EXTENT[0]
_H_domain      = TASK_EXTENT[3] - TASK_EXTENT[2]
FGP_H          = _W / (FGP_GRID_N - 1)
FGP_OFFSET     = np.array([-TASK_EXTENT[0], -TASK_EXTENT[2]])

_gx = np.linspace(0, _W, FGP_GRID_N)
_gy = np.linspace(0, _H_domain, FGP_GRID_N)
_GX, _GY = np.meshgrid(_gx, _gy)
FGP_INDUCING_XS      = np.column_stack([_GX.ravel(), _GY.ravel()])
FGP_INDUCING_XS_TASK = FGP_INDUCING_XS - FGP_OFFSET
FGP_GRID_SIZE        = [FGP_GRID_N, FGP_GRID_N]

# ══════════════════════════════════════════════════════════════════════════════
#  FGP model
# ══════════════════════════════════════════════════════════════════════════════

class FGPModel:
    def __init__(self, inducing_xs, grid_size, h, offset,
                 x_train, y_train, noise_var,
                 n_particles, n_iter, logit_init, log_kappa_mean, lr,
                 log_kappa_std=1.0, graph_relearn_interval=100,
                 node_prior_lam=1.0, heuristic_beta=1.0):
        self._inducing_xs          = inducing_xs
        self._grid_size            = grid_size
        self._h                    = h
        self._offset               = offset
        self._n_particles          = n_particles
        self._n_iter               = n_iter
        self._lr                   = lr
        self._graph_relearn_interval = graph_relearn_interval
        self._log_kappa_mean       = log_kappa_mean
        self._log_kappa_std        = log_kappa_std
        self._logit_init           = logit_init
        self._node_prior_lam       = node_prior_lam
        self._heuristic_beta       = heuristic_beta

        self._x_train = x_train.copy()
        raw = y_train.reshape(-1, 1).astype(np.float64)
        self._y_scaler = StandardScaler(raw)
        self._y_train  = self._y_scaler.preprocess(raw)
        self._noise_var = noise_var / float(self._y_scaler.scale.ravel()[0]) ** 2

        self._fgp                  = None
        self._last_graph_learn_n   = 0
        self._cached_edges         = None
        self._cached_pi            = None
        self._cached_pi_std        = None
        self._cached_kappa         = None
        self._cached_all_pi        = None
        self._cached_all_kappa     = None
        self._mu_particles         = None
        self._var_particles        = None

    def optimize(self, num_iter=None, verbose=True):
        x_fgp = self._x_train + self._offset
        n     = self.num_train
        needs_relearn = (
            self._cached_edges is None
            or (n - self._last_graph_learn_n) >= self._graph_relearn_interval
        )
        if needs_relearn:
            result = learn_graph_structure(
                inducing_xs    = self._inducing_xs,
                train_xs       = x_fgp,
                train_zs       = self._y_train,
                h              = self._h,
                noise_var      = self._noise_var,
                n_particles    = self._n_particles,
                logit_init     = self._logit_init,
                n_iter         = self._n_iter,
                lr             = self._lr,
                log_kappa_mean = self._log_kappa_mean,
                log_kappa_std  = self._log_kappa_std,
                verbose        = verbose,
                device         = 'cpu',
            )
            final_pi, final_kappa, edges, snapshots = result
            arg_max = int(np.argmax(snapshots[-1][-1]))
            self._cached_edges     = edges
            self._cached_pi        = final_pi[arg_max]
            self._cached_pi_std    = final_pi.std(axis=0)
            self._cached_kappa     = final_kappa[arg_max]
            self._cached_all_pi    = final_pi
            self._cached_all_kappa = final_kappa
            self._last_graph_learn_n = n

        self._fgp = build_fgp_from_pi(
            inducing_xs    = self._inducing_xs,
            train_xs       = x_fgp,
            train_zs       = self._y_train,
            kappa_sq       = self._cached_kappa,
            grid_size      = self._grid_size,
            h              = self._h,
            edges          = self._cached_edges,
            pi_mean        = self._cached_pi,
            threshold      = 0.5,
            noise_var      = self._noise_var,
            node_prior_lam = self._node_prior_lam,
        )

    def __call__(self, x_test):
        x_fgp = x_test + self._offset
        nx, ny = self._grid_size
        W = fd_interpolate(x_fgp, np.array([nx, ny]), self._h)

        if self._mu_particles is not None:
            particle_preds = np.asarray(W @ self._mu_particles.T)
            mean_norm      = particle_preds.mean(axis=1)
            var_epistemic  = particle_preds.var(axis=1)
            W_sq           = W.power(2)
            mean_diag_var  = self._var_particles.mean(axis=0)
            var_aleatoric  = np.asarray(W_sq @ mean_diag_var).ravel()
            var_total      = var_aleatoric + var_epistemic
        else:
            mean_map, cov = self._fgp.query(x_fgp, pre_W=W)
            mean_norm     = np.ravel(mean_map)
            var_total     = np.diag(cov)

        mean = self._y_scaler.postprocess_mean(mean_norm.reshape(-1, 1))
        std  = self._y_scaler.postprocess_std(
            np.sqrt(np.maximum(var_total, 1e-6)).reshape(-1, 1))
        return mean.reshape(-1, 1), std.reshape(-1, 1)

    def get_scores(self, x_test):
        x_fgp = x_test + self._offset
        scores, _, _ = heuristic_scores(
            pts        = x_fgp,
            fgp_opt    = self._fgp,
            pi_map     = self._cached_pi,
            pi_std     = self._cached_pi_std,
            edges      = self._cached_edges,
            grid_shape = self._grid_size,
            beta       = self._heuristic_beta,
            h          = self._h,
        )
        return scores

    def finalize(self):
        """Full SVGD re-optimisation + particle ensemble for epistemic variance."""
        self._cached_edges = None
        self.optimize(verbose=True)

        x_fgp    = self._x_train + self._offset
        mu_list, var_list = [], []
        for pi_k, kappa_k in zip(self._cached_all_pi, self._cached_all_kappa):
            fgp_k = build_fgp_from_pi(
                inducing_xs    = self._inducing_xs,
                train_xs       = x_fgp,
                train_zs       = self._y_train,
                kappa_sq       = kappa_k,
                grid_size      = self._grid_size,
                h              = self._h,
                edges          = self._cached_edges,
                pi_mean        = pi_k,
                threshold      = 0.5,
                noise_var      = self._noise_var,
                node_prior_lam = self._node_prior_lam,
            )
            mu_list.append(fgp_k.get_posterior_mean())
            Lam_k = fgp_k.joint_belief.lam.tocsc()
            N_k   = Lam_k.shape[0]
            try:
                lu_k = _splu(Lam_k)
            except RuntimeError as e:
                if "singular" not in str(e).lower():
                    raise
                lu_k = _splu((_speye(N_k, format="csc") * 1e-6 + Lam_k).tocsc())
            var_list.append(lu_k.solve(np.eye(N_k)).diagonal())

        self._mu_particles  = np.array(mu_list)
        self._var_particles = np.array(var_list)

    def add_data(self, x_new, y_new):
        self._x_train = np.vstack([self._x_train, x_new])
        y_norm = self._y_scaler.preprocess(y_new.reshape(-1, 1).astype(np.float64))
        self._y_train = np.vstack([self._y_train, y_norm])

    def get_data(self):
        return self._x_train.copy(), self._y_scaler.postprocess_mean(self._y_train).copy()

    @property
    def num_train(self):
        return len(self._x_train)

# ══════════════════════════════════════════════════════════════════════════════
#  Environment & robot
# ══════════════════════════════════════════════════════════════════════════════

class GridMap:
    def __init__(self, matrix, extent):
        self.matrix = matrix
        self.extent = extent
        self.num_rows, self.num_cols = matrix.shape
        eps = 1e-4
        self.x_cell = (extent[1] - extent[0]) / self.num_cols + eps
        self.y_cell = (extent[3] - extent[2]) / self.num_rows + eps

    def get(self, xs, ys):
        cols = ((xs - self.extent[0]) / self.x_cell).astype(int)
        rows = ((ys - self.extent[2]) / self.y_cell).astype(int)
        return self.matrix[rows, cols]


class Sonar:
    def __init__(self, rate, env, env_extent, noise_scale):
        self.dt          = 1.0 / rate
        self.env         = GridMap(env, env_extent)
        self.noise_scale = noise_scale

    def sense(self, states, rng=None):
        if states.ndim == 1:
            states = states.reshape(1, -1)
        obs = self.env.get(states[:, 0], states[:, 1])
        if rng is not None:
            obs = rng.normal(loc=obs, scale=self.noise_scale)
        return obs


class DubinsCar:
    def __init__(self, rate):
        self.dt = 1.0 / rate

    @staticmethod
    def _wrap(angle):
        while angle >  np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle

    def step(self, state, action):
        x, y, o = state
        v, w    = action
        state[0] = x + v * np.cos(o) * self.dt
        state[1] = y + v * np.sin(o) * self.dt
        state[2] = self._wrap(o + w * self.dt)
        return state


class USV:
    def __init__(self, init_state, control_rate, max_lin_vel, tolerance,
                 sampling_rate):
        self.state             = init_state
        self.tolerance         = tolerance
        self.max_lin_vel       = max_lin_vel
        self.sampling_locations = []
        self.goal_states       = []
        self._dynamics         = DubinsCar(control_rate)
        self._sampling_dt      = 1.0 / sampling_rate
        self._cum_time         = 0.0

    @property
    def has_goal(self):
        return len(self.goal_states) > 0

    def control(self):
        x, y, o = self.state
        gx, gy  = self.goal_states[0][:2]
        dx, dy  = gx - x, gy - y
        dist    = np.hypot(dx, dy)
        xo = np.cos(o) * dx + np.sin(o) * dy
        yo = -np.sin(o) * dx + np.cos(o) * dy
        lin_vel = self.max_lin_vel * np.tanh(xo)
        ang_vel = 2.0 * np.arctan2(yo, xo)
        return dist, np.array([lin_vel, ang_vel])

    def update(self, dist, action):
        self.state = self._dynamics.step(self.state, action)
        self._cum_time += self._dynamics.dt
        if self._cum_time > self._sampling_dt:
            self.sampling_locations.append(self.state[:2].copy())
            self._cum_time = 0.0
        if self.has_goal and dist < self.tolerance:
            self.goal_states = self.goal_states[1:]

    def commit_data(self):
        x_new = np.vstack(self.sampling_locations)
        self.sampling_locations = []
        return x_new

# ══════════════════════════════════════════════════════════════════════════════
#  Planning
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_entropy(std):
    return 0.5 * np.log(2 * np.pi * np.square(std)) + 0.5


class MyopicPlanning:
    def __init__(self, task_extent, rng, num_candidates, robot):
        self.task_extent    = task_extent
        self.rng            = rng
        self.num_candidates = num_candidates
        self.robot          = robot

    def get(self, model, num_states=1):
        while len(self.robot.sampling_locations) == 0:
            xs = self.rng.uniform(self.task_extent[0], self.task_extent[1],
                                  self.num_candidates)
            ys = self.rng.uniform(self.task_extent[2], self.task_extent[3],
                                  self.num_candidates)
            candidates = np.column_stack((xs, ys))

            if hasattr(model, 'get_scores'):
                info = model.get_scores(candidates)
            else:
                _, std = model(candidates)
                info   = gaussian_entropy(std.ravel())

            diffs       = candidates - self.robot.state[:2]
            dists       = np.hypot(diffs[:, 0], diffs[:, 1])
            normed_dist = (dists - dists.min()) / (dists.max() - dists.min())

            info_ptp = info.max() - info.min()
            if info_ptp < 1e-3 * (np.abs(info).mean() + 1e-12):
                scores = normed_dist
            else:
                scores = (info - info.min()) / info_ptp - normed_dist

            goal = candidates[np.argsort(scores)[-num_states:]]
            self.robot.goal_states.append(goal.ravel())
            while self.robot.has_goal:
                self.robot.update(*self.robot.control())

        return self.robot.commit_data()

# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

class Evaluator:
    def __init__(self, sensor, task_extent, eval_grid):
        self.task_extent = task_extent
        self.eval_grid   = eval_grid
        xmin, xmax, ymin, ymax = task_extent
        nx, ny = eval_grid
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, nx),
                             np.linspace(ymin, ymax, ny))
        self.eval_inputs  = np.column_stack((xx.ravel(), yy.ravel()))
        self.eval_outputs = sensor.sense(self.eval_inputs).reshape(-1, 1)
        self._log2pi      = np.log(2 * np.pi)
        self.rmses, self.smses, self.ssims = [], [], []
        self.mslls, self.nlpds             = [], []

    def eval_prediction(self, model):
        _, y_train = model.get_data()
        mean, std  = model(self.eval_inputs)
        error      = np.fabs(mean - self.eval_outputs)

        mse  = np.mean(np.square(error))
        self.rmses.append(float(np.sqrt(mse)))
        self.smses.append(float(mse / self.eval_outputs.var()))

        ll   = 0.5 * self._log2pi + np.log(std) + 0.5 * (error / std) ** 2
        base = 0.5 * self._log2pi + np.log(y_train.std()) + 0.5 * ((
            self.eval_outputs - y_train.mean()) / y_train.std()) ** 2
        self.mslls.append(float(np.mean(ll - base)))
        self.nlpds.append(float(np.mean(ll)))

        gt_2d   = self.eval_outputs.reshape(self.eval_grid)
        pred_2d = mean.reshape(self.eval_grid)
        dr      = float(gt_2d.max() - gt_2d.min())
        self.ssims.append(compute_ssim(pred_2d, gt_2d, data_range=dr))

        return mean, std, error

# ══════════════════════════════════════════════════════════════════════════════
#  Model factories
# ══════════════════════════════════════════════════════════════════════════════

def _make_stationary():
    def factory(x, y):
        return SparseGPR(
            x_train=x, y_train=y,
            kernel=Matern32Kernel(amplitude=STAT_AMPLITUDE,
                                  lengthscale=STAT_LENGTHSCALE),
            noise=INIT_NOISE, lr_hyper=LR_HYPER, lr_nn=LR_NN, jitter=JITTER,
            inducing_xs=FGP_INDUCING_XS_TASK, x_domain_width=_W,
        )
    return factory


def _make_dkl():
    def factory(x, y):
        return SparseGPR(
            x_train=x, y_train=y,
            kernel=DKLKernel(
                amplitude=DKL_AMPLITUDE, lengthscale=DKL_LENGTHSCALE,
                dim_input=DIM_INPUT, dim_hidden=DIM_HIDDEN, dim_output=DIM_OUTPUT,
            ),
            noise=INIT_NOISE, lr_hyper=LR_HYPER, lr_nn=LR_NN, jitter=JITTER,
            inducing_xs=FGP_INDUCING_XS_TASK, x_domain_width=_W,
        )
    return factory


def _make_ak():
    lengthscales = np.linspace(AK_MIN_LS, AK_MAX_LS, DIM_OUTPUT)
    def factory(x, y):
        return SparseGPR(
            x_train=x, y_train=y,
            kernel=AK(
                amplitude=AK_AMPLITUDE, lengthscales=lengthscales,
                dim_input=DIM_INPUT, dim_hidden=DIM_HIDDEN, dim_output=DIM_OUTPUT,
            ),
            noise=INIT_NOISE, lr_hyper=LR_HYPER, lr_nn=LR_NN, jitter=JITTER,
            inducing_xs=FGP_INDUCING_XS_TASK, x_domain_width=_W,
        )
    return factory


def _make_fgp(logit_init, node_prior_lam, log_kappa_mean):
    def factory(x, y):
        return FGPModel(
            inducing_xs=FGP_INDUCING_XS, grid_size=FGP_GRID_SIZE,
            h=FGP_H, offset=FGP_OFFSET,
            x_train=x, y_train=y,
            noise_var=NOISE_SCALE**2,
            n_particles=FGP_N_PARTICLES, n_iter=FGP_N_ITER,
            logit_init=logit_init,
            log_kappa_mean=log_kappa_mean, log_kappa_std=FGP_LOG_KAPPA_STD,
            lr=FGP_LR,
            graph_relearn_interval=FGP_GRAPH_RELEARN,
            node_prior_lam=node_prior_lam,
            heuristic_beta=FGP_HEURISTIC_BETA,
        )
    return factory

# ══════════════════════════════════════════════════════════════════════════════
#  Experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_one(label, factory, sensor, max_samples, rng_seed):
    print(f"\n{'='*60}")
    print(f"  {label}  (n_init={NUM_INIT_SAMPLES}, max_samples={max_samples})")
    print(f"{'='*60}")

    rng = np.random.RandomState(rng_seed)
    torch.manual_seed(rng_seed)

    x_init = np.column_stack((
        rng.uniform(TASK_EXTENT[0], TASK_EXTENT[1], NUM_INIT_SAMPLES),
        rng.uniform(TASK_EXTENT[2], TASK_EXTENT[3], NUM_INIT_SAMPLES),
    ))
    y_init = sensor.sense(states=x_init, rng=rng).reshape(-1, 1)

    robot     = USV(
        init_state=np.array([x_init[-1, 0], x_init[-1, 1], np.pi / 2],
                            dtype=np.float64),
        control_rate=CONTROL_RATE, max_lin_vel=MAX_LIN_VEL,
        tolerance=TOLERANCE, sampling_rate=SENSING_RATE,
    )
    model     = factory(x_init, y_init)
    evaluator = Evaluator(sensor=sensor, task_extent=TASK_EXTENT, eval_grid=EVAL_GRID)
    strategy  = MyopicPlanning(TASK_EXTENT, rng, NUM_CANDIDATES, robot)

    model.optimize(num_iter=max(model.num_train, 200), verbose=True)
    evaluator.eval_prediction(model)

    all_xs = [x_init.copy()]

    while model.num_train < max_samples:
        x_new = strategy.get(model=model)
        y_new = sensor.sense(x_new, rng).reshape(-1, 1)
        model.add_data(x_new, y_new)
        model.optimize(num_iter=max(len(y_new), 10), verbose=False)
        evaluator.eval_prediction(model)
        all_xs.append(x_new)
        print(f"  n={model.num_train:4d} | RMSE={evaluator.rmses[-1]:.4f} | "
              f"SMSE={evaluator.smses[-1]:.4f} | SSIM={evaluator.ssims[-1]:.4f}")

    if hasattr(model, 'finalize'):
        model.finalize()

    mean, std, error = evaluator.eval_prediction(model)
    xs_all = np.vstack(all_xs)

    return dict(
        label  = label,
        mean   = mean.ravel(),
        std    = std.ravel(),
        error  = error.ravel(),
        gt     = evaluator.eval_outputs.ravel(),
        xs     = xs_all,
        n_init = NUM_INIT_SAMPLES,
        rmse   = evaluator.rmses[-1],
        smse   = evaluator.smses[-1],
        ssim   = evaluator.ssims[-1],
    )

# ══════════════════════════════════════════════════════════════════════════════
#  Extra metrics  (gradient RMSE, SSIM with 3×3 window)
# ══════════════════════════════════════════════════════════════════════════════

def compute_extras(r):
    dr    = float(r['gt'].max() - r['gt'].min())
    gt_2d = r['gt'].reshape(EVAL_GRID)
    pr_2d = r['mean'].reshape(EVAL_GRID)
    return dict(
        grmse = compute_gradient_rmse(pr_2d, gt_2d),
        ssim3 = compute_ssim(pr_2d, gt_2d, data_range=dr, win_size=3),
    )

# ══════════════════════════════════════════════════════════════════════════════
#  Printed table
# ══════════════════════════════════════════════════════════════════════════════

def print_table(results, extras, env_name, max_samples):
    rows = [dict(label=r['label'], rmse=r['rmse'], smse=r['smse'],
                 grmse=ex['grmse'], ssim3=ex['ssim3'])
            for r, ex in zip(results, extras)]
    print_results_table(rows, title=f"{env_name}  —  Final metrics  (N = {max_samples} samples)")

# ══════════════════════════════════════════════════════════════════════════════
#  Figure  (3 rows × 4 cols)
# ══════════════════════════════════════════════════════════════════════════════

def _sci_fmt(cb, vmin, vmax, fs=7):
    absmax = max(abs(float(vmin)), abs(float(vmax)))
    power  = int(np.floor(np.log10(absmax))) if absmax > 0 else 0
    scale  = 10.0 ** power
    cb.formatter = ticker.FuncFormatter(lambda x, _: f'{x / scale:.1f}')
    cb.update_ticks()
    cb.ax.set_title(f'$\\times10^{{{power}}}$', fontsize=fs, pad=2)


def _ishow(ax, values, extent):
    im = ax.imshow(values.reshape(EVAL_GRID), extent=extent)
    ax.add_patch(plt.Rectangle(
        (TASK_EXTENT[0], TASK_EXTENT[2]),
        TASK_EXTENT[1] - TASK_EXTENT[0], TASK_EXTENT[3] - TASK_EXTENT[2],
        linewidth=2, edgecolor='white', alpha=0.8, fill=False,
    ))
    return im


def plot_results(results, env_name, max_samples, out_path):
    n_cols = len(results)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 10),
                             constrained_layout=True)
    fig.suptitle(
        f"{env_name}  (N = {max_samples} samples)",
        fontsize=12,
    )

    rows = [
        ("Mean",      [r['mean']  for r in results], True),
        ("Std",       [r['std']   for r in results], True),
        ("Abs Error", [r['error'] for r in results], False),
    ]

    for row_idx, (row_label, col_vals, overlay) in enumerate(rows):
        for col_idx, (vals, r) in enumerate(zip(col_vals, results)):
            ax = axes[row_idx, col_idx]
            im = _ishow(ax, vals, TASK_EXTENT)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend='neither')
            cb.ax.tick_params(labelsize=6)
            _sci_fmt(cb, *im.get_clim())
            ax.tick_params(labelsize=7)

            if overlay:
                xs, n = r['xs'], r['n_init']
                ax.plot(xs[:n,  0], xs[:n,  1], 'w.', markersize=5, alpha=0.9)
                ax.plot(xs[n:,  0], xs[n:,  1], 'w.', markersize=2, alpha=0.3)

            if row_idx == 0:
                ax.set_title(f"{r['label']}\nRMSE = {r['rmse']:.3f}", fontsize=9)

        axes[row_idx, 0].set_ylabel(row_label, fontsize=9)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out_path}")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global MAX_NUM_SAMPLES

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", required=True, choices=list(ENV_CONFIG),
                        help="SRTM tile identifier")
    parser.add_argument("--samples", type=int, default=MAX_NUM_SAMPLES,
                        help=f"Total active-learning samples (default: {MAX_NUM_SAMPLES})")
    args = parser.parse_args()

    MAX_NUM_SAMPLES = args.samples

    logit_init, node_prior_lam, log_kappa_mean, seed = ENV_CONFIG[args.env]

    img_path = os.path.join(_HERE, ".", "data", "srtm", f"{args.env}.jpg")
    print(f"Loading {args.env} (seed={seed}) …")
    image  = Image.open(img_path).convert("L")
    arr    = np.array(image).astype(np.float64)
    env    = transform.resize(arr, (arr.shape[0] // 10, arr.shape[1] // 10))
    sensor = Sonar(rate=SENSING_RATE, env=env, env_extent=ENV_EXTENT,
                   noise_scale=NOISE_SCALE)

    experiments = [
        ("Stationary", _make_stationary()),
        ("DKL",        _make_dkl()),
        ("AK",         _make_ak()),
        ("FGP",        _make_fgp(logit_init, node_prior_lam, log_kappa_mean)),
    ]
    results = []
    for label, factory in experiments:
        results.append(run_one(label, factory, sensor,
                               max_samples=args.samples, rng_seed=seed))

    extras = [compute_extras(r) for r in results]
    print_table(results, extras, args.env, args.samples)

    out_path = os.path.join(_HERE, f"results_{args.env}_seed{seed}.png")
    plot_results(results, args.env, args.samples, out_path)


if __name__ == "__main__":
    main()
