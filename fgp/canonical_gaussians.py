import numpy as np

class Gaussian:
    """ Class for storing Gaussians in canonical form with common operations
    Inputs
    -----
    prior_eta : n x 1 np.array
    prior_lam : n x n np.array

    """

    def __init__(self, prior_eta, prior_lam):
        self.eta = prior_eta
        self.lam = prior_lam

    def get_moment_form(self):
        try:
            cov = np.linalg.inv(self.lam)
        except:
            cov = np.ones(self.lam.shape) * 1
        mu = cov @ self.eta
        return mu, cov
    
    def product(self, other):
        self.eta += other.eta
        self.lam += other.lam

    def quotient(self, other):
        self.eta -= other.eta
        self.lam -= other.lam

    def get_product(self, other):
        return Gaussian(self.eta + other.eta, self.lam + other.lam)
        
    def get_quotient(self, other):
        return Gaussian(self.eta - other.eta, self.lam - other.lam)
    