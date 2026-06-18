import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import kv, gamma

def rbf(x1, x2, lengthscale=1):
    '''
    Radial basis function kernel

    Args:
        x1 (np.array)       : array of xs, (n1,d)
        x2 (np.array)       : array of xs, (n2,d)
        lengthscale (float) : the lengthscale

    Returns:
        Kernel matrix shape (n1,n2)
    '''
    if not isinstance(x1, np.ndarray):
        x1 = np.asarray(x1).reshape(-1,1)
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2).reshape(-1,1)
    d = cdist(x1,x2)
    return np.exp(- d **2 / (2 * lengthscale ** 2))

def matern(x1, x2, lengthscale=3, nu=0.5):
    """
    General vectorized Matern kernel for any nu > 0.
    
    Args:
        x1: array-like, shape (n1,) or (n1, d)
        x2: array-like, shape (n2,) or (n2, d)
        lengthscale: float
        nu: float, smoothness parameter > 0
        
    Returns:
        Kernel matrix, shape (n1, n2)
    """
    if not isinstance(x1, np.ndarray):
        x1 = np.asarray(x1).reshape(-1, 1)
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2).reshape(-1, 1)
        
    d = cdist(x1, x2)  # Euclidean distance
    if np.any(d < 1e-12):  # avoid division by zero
        d = np.maximum(d, 1e-12)
    
    factor = np.sqrt(2 * nu) * d / lengthscale
    K = (2**(1 - nu) / gamma(nu)) * (factor**nu) * kv(nu, factor)
    
    # Handle the case where distance is zero (kv diverges at 0)
    K[d < 1e-12] = 1.0
    
    return K
