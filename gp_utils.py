import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def softplus(x):
    return F.softplus(x, 1.0, 20.0) + 1e-6

def inv_softplus(y):
    if torch.any(y <= 0.0):
        raise ValueError("Input to inv_softplus must be positive.")
    _y = y - 1e-6
    return _y + torch.log(-torch.expm1(-_y))

def constraint(free):
    return softplus(free)

def unconstraint(param):
    return inv_softplus(torch.tensor(param, dtype=torch.float64))

# ── Kernel primitives ──────────

_SQRT3 = math.sqrt(3.0)

def rbf(dist, lengthscale):
    return torch.exp(-0.5 * torch.square(dist / lengthscale))

def matern32(dist, lengthscale):
    z = _SQRT3 * dist / lengthscale
    return (1.0 + z) * torch.exp(-z)

def robust_cholesky(cov, jitter=1e-6, num_attempts=3):
    L, info = torch.linalg.cholesky_ex(cov)
    if not torch.any(info):
        return L
    if torch.any(torch.isnan(cov)):
        raise ValueError("NaN in covariance matrix.")
    _cov = cov.clone()
    jitter_prev = 0.0
    for i in range(num_attempts):
        jitter_new = jitter * (10 ** i)
        _cov.diagonal().add_((info > 0).float() * (jitter_new - jitter_prev))
        jitter_prev = jitter_new
        L, info = torch.linalg.cholesky_ex(_cov)
        if not torch.any(info):
            return L
    raise ValueError(f"Matrix not positive-definite after adding {jitter_new:.1e}.")

# ── Scalers ───────────────────────────────────────────────────────────────────

class StandardScaler:
    def __init__(self, values, expected_scale=None):
        if values.ndim != 2:
            raise ValueError("values.shape=(num_samples, num_dims)")
        self.mean  = values.mean(axis=0, keepdims=True)
        self.scale = values.std(axis=0, keepdims=True)
        zero_dims = self.scale <= 0.0
        if np.any(zero_dims):
            fallback = (float(expected_scale) if expected_scale is not None
                        else np.maximum(np.abs(self.mean), 1.0))
            self.scale = np.where(zero_dims, fallback, self.scale)

    def preprocess(self, raw):
        return (raw - self.mean) / self.scale

    def postprocess_mean(self, transformed):
        return transformed * self.scale + self.mean

    def postprocess_std(self, transformed):
        return transformed * self.scale


class MinMaxScaler:
    def __init__(self, values, expected_range=(-1.0, 1.0), domain_width=None):
        self.min = expected_range[0]
        self.max = expected_range[1]
        self.ptp = expected_range[1] - expected_range[0]
        if self.ptp <= 0.0:
            raise ValueError("Expected range must be positive.")
        self.data_min = values.min(axis=0, keepdims=True)
        self.data_max = values.max(axis=0, keepdims=True)
        self.data_ptp = self.data_max - self.data_min
        zero_dims = self.data_ptp <= 0.0
        if np.any(zero_dims):
            if domain_width is not None:
                self.data_min = np.where(
                    zero_dims, self.data_min - float(domain_width) / 2.0, self.data_min)
                self.data_ptp = np.where(zero_dims, float(domain_width), self.data_ptp)
            else:
                self.data_ptp = np.where(zero_dims, self.ptp, self.data_ptp)

    def preprocess(self, raw):
        return (raw - self.data_min) / self.data_ptp * self.ptp + self.min

    def postprocess(self, transformed):
        return (transformed - self.min) / self.ptp * self.data_ptp + self.data_min


class Scaler(torch.nn.Module):
    """Batch-statistics input scaler — used as input warp inside DKLKernel."""
    def __init__(self, lower_bound, upper_bound):
        super().__init__()
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.register_buffer("min_val", torch.tensor(lower_bound))
        self.register_buffer("max_val", torch.tensor(upper_bound))

    def forward(self, x):
        if self.training:
            self.min_val.data = x.min()
            self.max_val.data = x.max()
        else:
            x = x.clamp(self.min_val, self.max_val)
        diff = self.max_val - self.min_val
        x = (x - self.min_val) * (0.95 * (self.upper_bound - self.lower_bound) / diff)
        return x + 0.95 * self.lower_bound

# ── Neural networks ───────────────────────────────────────────────────────────

class TwoHiddenLayerTanhNN(torch.nn.Sequential):
    def __init__(self, dim_input, dim_hidden, dim_output, softmax=True):
        super().__init__()
        self.add_module("linear1",     torch.nn.Linear(dim_input, dim_hidden))
        self.add_module("activation1", torch.nn.Tanh())
        self.add_module("linear2",     torch.nn.Linear(dim_hidden, dim_hidden))
        self.add_module("activation2", torch.nn.Tanh())
        self.add_module("linear3",     torch.nn.Linear(dim_hidden, dim_output))
        if softmax:
            self.add_module("activation3", torch.nn.Softmax(dim=1))

# ── Kernel classes ────────────────────────────────────────────────────────────

class Matern32Kernel(torch.nn.Module):
    def __init__(self, amplitude, lengthscale):
        super().__init__()
        self.__free_amplitude   = Parameter(unconstraint(amplitude))
        self.__free_lengthscale = Parameter(unconstraint(lengthscale))

    def diag(self, x):
        return self.amplitude * torch.ones(x.size(0), 1, dtype=torch.float64)

    def forward(self, x1, x2):
        return self.amplitude * matern32(torch.cdist(x1, x2), self.lengthscale)

    @property
    def amplitude(self):
        return constraint(self.__free_amplitude)

    @property
    def lengthscale(self):
        return constraint(self.__free_lengthscale)


class AK(torch.nn.Module):
    """Attentive Kernel.

    kernel_fn defaults to matern32 (as in active_mapping_srtm).
    active_sampling_gas uses rbf — pass kernel_fn=rbf explicitly there.
    """
    def __init__(self, amplitude, lengthscales, dim_input, dim_hidden,
                 dim_output, kernel_fn=None):
        super().__init__()
        self.__free_amplitude = Parameter(unconstraint(amplitude))
        self.lengthscales = torch.tensor(lengthscales, dtype=torch.float64)
        self._kernel_fn   = kernel_fn if kernel_fn is not None else matern32
        self.nn = TwoHiddenLayerTanhNN(dim_input, dim_hidden, dim_output).double()

    def get_representations(self, x):
        z = self.nn(x)
        return z / z.norm(dim=1, keepdim=True)

    def diag(self, x):
        return self.amplitude * torch.ones(x.size(0), 1, dtype=torch.float64)

    def forward(self, x1, x2):
        dist = torch.cdist(x1, x2)
        r1   = self.get_representations(x1)
        r2   = self.get_representations(x2)
        cov  = sum(torch.outer(r1[:, i], r2[:, i]) * self._kernel_fn(dist, self.lengthscales[i])
                   for i in range(len(self.lengthscales)))
        return self.amplitude * (r1 @ r2.t()) * cov

    @property
    def amplitude(self):
        return constraint(self.__free_amplitude)


class DKLKernel(torch.nn.Module):
    """Deep Kernel Learning.

    kernel_fn defaults to matern32 (as in active_mapping_srtm).
    active_sampling_gas uses rbf — pass kernel_fn=rbf explicitly there.
    """
    def __init__(self, amplitude, lengthscale, dim_input, dim_hidden,
                 dim_output, kernel_fn=None):
        super().__init__()
        self.__free_amplitude   = Parameter(unconstraint(amplitude))
        self.__free_lengthscale = Parameter(unconstraint(lengthscale))
        self._kernel_fn = kernel_fn if kernel_fn is not None else matern32
        self.nn     = TwoHiddenLayerTanhNN(
            dim_input, dim_hidden, dim_output, softmax=False).double()
        self.scaler = Scaler(-1.0, 1.0)

    def diag(self, x):
        return self.amplitude * torch.ones(x.size(0), 1, dtype=torch.float64)

    def input_warping(self, x):
        return self.scaler(self.nn(x))

    def forward(self, x1, x2):
        return self.amplitude * self._kernel_fn(
            torch.cdist(self.input_warping(x1), self.input_warping(x2)),
            self.lengthscale,
        )

    @property
    def amplitude(self):
        return constraint(self.__free_amplitude)

    @property
    def lengthscale(self):
        return constraint(self.__free_lengthscale)

class GPR(torch.nn.Module):
    def __init__(self, x_train, y_train, kernel, noise,
                 lr_hyper=0.01, lr_nn=0.001, jitter=1e-6,
                 is_normalized=True, x_domain_width=None):
        super().__init__()
        self.__free_noise  = Parameter(unconstraint(noise))
        self.is_normalized = is_normalized
        if self.is_normalized:
            self._init_scalers(x_train, y_train, x_domain_width)
        self._set_data(x_train, y_train)
        self.kernel = kernel
        self.jitter = jitter
        self._init_opts(lr_hyper, lr_nn)

    def _init_opts(self, lr_hyper, lr_nn):
        hyper_p, nn_p = [], []
        for name, p in self.named_parameters():
            (nn_p if "nn" in name else hyper_p).append(p)
        self.opt_hyper = torch.optim.Adam(hyper_p, lr=lr_hyper)
        self.opt_nn    = torch.optim.Adam(nn_p, lr=lr_nn) if nn_p else None

    def _init_scalers(self, x, y, x_domain_width=None):
        self.x_scaler = MinMaxScaler(x, expected_range=(-1.0, 1.0),
                                     domain_width=x_domain_width)
        self.y_scaler = StandardScaler(values=y)

    def _set_data(self, x, y):
        if self.is_normalized:
            x = self.x_scaler.preprocess(x)
            y = self.y_scaler.preprocess(y)
        self._x_train = torch.tensor(x, dtype=torch.float64)
        self._y_train = torch.tensor(y, dtype=torch.float64)

    def _common(self):
        K = self.kernel(self._x_train, self._x_train)
        K.diagonal().add_(self.noise)
        L    = robust_cholesky(K, jitter=self.jitter)
        iK_y = torch.cholesky_solve(self._y_train, L, upper=False)
        return L, iK_y

    def compute_loss(self):
        L, iK_y = self._common()
        quad   = torch.sum(self._y_train * iK_y)
        logdet = 2.0 * L.diagonal().log().sum()
        return 0.5 * (quad + logdet + self.num_train * np.log(2 * np.pi))

    def optimize(self, num_iter, verbose=False):
        self.train()
        itr = range(num_iter)
        if verbose:
            from tqdm import tqdm
            itr = tqdm(itr)
        for i in itr:
            self.opt_hyper.zero_grad()
            if self.opt_nn is not None:
                self.opt_nn.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            self.opt_hyper.step()
            if self.opt_nn is not None:
                self.opt_nn.step()
            if verbose:
                itr.set_description(f"Iter {i:03d}  loss {loss.item():.2f}")
        self.eval()

    def forward(self, x_test, noise_free=False):
        if self.is_normalized:
            x_test = self.x_scaler.preprocess(x_test)
        x_t = torch.tensor(x_test, dtype=torch.float64)
        with torch.no_grad():
            L, iK_y = self._common()
            Ksn    = self.kernel(x_t, self._x_train)
            Kss_d  = self.kernel.diag(x_t)
            iL_Kns = torch.linalg.solve_triangular(L, Ksn.t(), upper=False)
            mean   = Ksn @ iK_y
            var    = Kss_d - iL_Kns.square().sum(0).view(-1, 1)
            var.clamp_(min=self.jitter)
            if not noise_free:
                var += self.noise
        mean = mean.numpy()
        std  = var.sqrt().numpy()
        if self.is_normalized:
            mean = self.y_scaler.postprocess_mean(mean)
            std  = self.y_scaler.postprocess_std(std)
        return mean, std

    def add_data(self, x_new, y_new):
        if self.is_normalized:
            x_new = self.x_scaler.preprocess(x_new)
            y_new = self.y_scaler.preprocess(y_new)
        self._x_train = torch.vstack((self._x_train,
                                      torch.tensor(x_new, dtype=torch.float64)))
        self._y_train = torch.vstack((self._y_train,
                                      torch.tensor(y_new, dtype=torch.float64)))

    def get_data(self):
        x = self._x_train.numpy()
        y = self._y_train.numpy()
        if self.is_normalized:
            x = self.x_scaler.postprocess(x)
            y = self.y_scaler.postprocess_mean(y)
        return x, y

    @property
    def num_train(self):
        return self._x_train.size(0)

    @property
    def noise(self):
        return constraint(self.__free_noise)


class SparseGPR(GPR):
    """FITC sparse GP on a fixed inducing-point set."""

    def __init__(self, x_train, y_train, kernel, noise, inducing_xs,
                 lr_hyper=0.01, lr_nn=0.001, jitter=1e-6,
                 is_normalized=True, x_domain_width=None):
        super().__init__(x_train, y_train, kernel, noise,
                         lr_hyper=lr_hyper, lr_nn=lr_nn, jitter=jitter,
                         is_normalized=is_normalized,
                         x_domain_width=x_domain_width)
        _z = self.x_scaler.preprocess(inducing_xs) if is_normalized else inducing_xs
        self._inducing_xs = torch.tensor(_z, dtype=torch.float64)

    def _fitc(self):
        x, z   = self._x_train, self._inducing_xs
        K_mm   = self.kernel(z, z); K_mm.diagonal().add_(self.jitter)
        L_mm   = robust_cholesky(K_mm, jitter=self.jitter)
        K_nm   = self.kernel(x, z)
        K_nn_d = self.kernel.diag(x).squeeze()
        iL_Knm = torch.linalg.solve_triangular(L_mm, K_nm.t(), upper=False)
        Q_nn_d = iL_Knm.square().sum(0)
        lam    = (K_nn_d - Q_nn_d + self.noise.squeeze()).clamp(min=self.jitter)
        K_nm_s = K_nm / lam.sqrt().unsqueeze(1)
        A      = K_mm + K_nm_s.t() @ K_nm_s
        L_A    = robust_cholesky(A, jitter=self.jitter)
        rhs    = K_nm.t() @ (self._y_train / lam.unsqueeze(1))
        alpha  = torch.cholesky_solve(rhs, L_A, upper=False)
        return L_mm, L_A, K_nm, lam, alpha

    def compute_loss(self):
        L_mm, L_A, K_nm, lam, alpha = self._fitc()
        y         = self._y_train
        lam_inv_y = y / lam.unsqueeze(1)
        K_mn_li_y = K_nm.t() @ lam_inv_y
        quadratic = (y * lam_inv_y).sum() - (alpha * K_mn_li_y).sum()
        logdet    = (lam.log().sum()
                     + 2.0 * L_A.diag().log().sum()
                     - 2.0 * L_mm.diag().log().sum())
        return 0.5 * (quadratic + logdet + self.num_train * np.log(2.0 * np.pi))

    def forward(self, x_test, noise_free=False):
        if self.is_normalized:
            x_test = self.x_scaler.preprocess(x_test)
        x_t = torch.tensor(x_test, dtype=torch.float64)
        with torch.no_grad():
            L_mm, L_A, _, _, alpha = self._fitc()
            z         = self._inducing_xs
            K_sm      = self.kernel(x_t, z)
            K_ss_d    = self.kernel.diag(x_t)
            mean      = K_sm @ alpha
            iL_mm_Kms = torch.linalg.solve_triangular(L_mm, K_sm.t(), upper=False)
            iL_A_Kms  = torch.linalg.solve_triangular(L_A,  K_sm.t(), upper=False)
            var = (K_ss_d
                   - iL_mm_Kms.square().sum(0).view(-1, 1)
                   + iL_A_Kms.square().sum(0).view(-1, 1))
            var.clamp_(min=self.jitter)
            if not noise_free:
                var += self.noise
        mean = mean.numpy()
        std  = var.sqrt().numpy()
        if self.is_normalized:
            mean = self.y_scaler.postprocess_mean(mean)
            std  = self.y_scaler.postprocess_std(std)
        return mean, std
