import numpy as np
import scipy.linalg
# from fgp.interpolate import linear_interpolation

class ReducedGP:

    def __init__(self, kernel=None, x_synth=None, synth_cov_init=0.01, synth_noise=0.0):
        '''Sparse GP class. For regresssion of scalar functions only.
        Implements KISS-GP now

        Inputs
        ---------
        kernel : function 
            function defining the kernel being used (RBF, matern). This is the full kernel function,
            no approximations
        x_synth : MxN np.array
            all M inducing points of dimension N
        synth_cov_init : float
            initial covariance, deprecated
        synth_noise : float
            measurement noise on inducing point measurements. 'synthetic noise'.
        '''
        self.kernel = kernel
        
        M = x_synth.shape[0]
        Nin = x_synth.shape[1]
        kernel_try = self.kernel(np.zeros((1, Nin)), np.zeros((1, Nin)))
        Nout = kernel_try.shape[0]

        self.x_synth = x_synth
        self.synth_mean = np.zeros((M * Nout, 1))
        # Initialise synthetic measurement covariance using kernel
        self.synth_cov = self.kernel(self.x_synth, self.x_synth)
        self.synth_prior_cov_inv = np.linalg.inv(self.synth_cov)
        self.synth_noise = synth_noise

        self.Nin = Nin
        self.Nout = Nout

    def fuse(self, x_meas, y_meas, y_meas_cov=0.01):
        '''
        Function to fuse new measurements into the GP for online regression.
        In-place function, returns None. Only updates the properties of the ReducedGP.
        Updates the synth_mean and synth_cov, meaning the estimate and corresponding
        covariances at the inducing or synthetic points.

        Inputs
        --------
        x_meas : Nxd np.array
            measurement location
        y_meas : Nx1
            measurement or observation
        y_meas_cov : float
            measurement noise intensity
        '''

        y_meas_cov = y_meas_cov * np.identity(y_meas.shape[0])

        if self.synth_noise > 0:
            self.synth_cov = self.synth_cov + self.synth_noise * np.identity(self.synth_cov.shape[0])

        # get the predicted value at the query location, the full covariance and the cross covariance
        # this is equations 6-7 in the note respectively.
        y_pred, y_cov, y_beta_cov = self.query(x_query=x_meas, cross_cov_flag=True)

        # updating the pseudomeasurements (the estimate held at inducing points) (eqn. 9)
        self.synth_mean = self.synth_mean + y_beta_cov.T @ (scipy.linalg.solve(y_cov + y_meas_cov, y_meas - y_pred, sym_pos=True))

        # updating the covariance of the pseudomeasurements (eqn. 10)
        self.synth_cov = self.synth_cov - y_beta_cov.T @ (scipy.linalg.solve(y_cov + y_meas_cov, y_beta_cov, sym_pos=True))
    
    def query(self, x_query, assume_fitc=True, y_meas_cov=0.01, cross_cov_flag=False):
        ''' Function to query the full GP posterior given the reduced GP belief held only
        at inducing points.
        
        Inputs
        ---------
        x_query : 1xN np.array
            query or test point
        assume_fitc : boolean
            True if using FITC, False if using SoR.
        y_meas_cov : float
            'Measurement' noise for the full belief. 

        Returns
        --------
        self_mean
            The mean of the full GP at the desired query point (eqn 11 in note)
        self_cov
            The covariance of the full GP belief at the desired query point (eqn. 12 or 13)
        cross_cov
            Cross covariance between the query point(s) and the inducing points
        '''
        # prior cross-covariance between x_query, x_synth

        #------- KISS GP --------_#
        # w, nearest_inducing = linear_interpolation(x_query, self.x_synth)
        # rows = np.arange(x_query.shape[0])
        # W_matrix = np.zeros((x_query.shape[0], self.x_synth.shape[0]))
        # W_matrix[rows, nearest_inducing[:, 0]] = (1 - w).squeeze()
        # W_matrix[rows, nearest_inducing[:, 1]] = w.squeeze()
        # prior_cross_cov = self.kernel(x_query, self.x_synth)
        # cross_cov = W_matrix @ self.synth_cov
        # self_cov = cross_cov @ W_matrix.T
        # self_mean = W_matrix @ self.synth_mean

        # current cross-covariance between x_query and x_synth
        # this is eqn. 8 in the note. Needed to update the belief. 
        prior_cross_cov = self.kernel(x_query, self.x_synth)
        cross_cov = prior_cross_cov @ self.synth_prior_cov_inv @ self.synth_cov

        # self-covariance among x_query, sigma^2(x,x'), or the covariance of the full posterior (eqn 12).
        # this is also used in eqn. 7 to get the prior for y_cov that is updated with new measurements
        self_cov = cross_cov @ self.synth_prior_cov_inv @ prior_cross_cov.T

        if y_meas_cov > 0.:
            self_cov = self_cov + np.identity(self_cov.shape[0]) * y_meas_cov

        if assume_fitc:
            self_cov = self_cov + np.diag( np.diag(self.kernel(x_query, x_query) - prior_cross_cov @ self.synth_prior_cov_inv @ prior_cross_cov.T) )

        # self-mean among x_query, or the mean of the full posterior. (mu without tilde in the note) (eqn. 11).
        self_mean = prior_cross_cov @ self.synth_prior_cov_inv @ self.synth_mean

        if cross_cov_flag:
            return self_mean, self_cov, cross_cov
        else:
            return self_mean, self_cov

    @property
    def cov(self):
        return self.synth_cov

    def predict_cov(self, x_meas, init_cov, y_meas_cov=0.01):
        y_meas_cov_matrix = y_meas_cov * np.identity(x_meas.shape[0])

        if self.synth_noise > 0:
            init_cov = init_cov + self.synth_noise * np.identity(init_cov.shape[0])

        # prior cross-covariance between x_meas, x_synth
        prior_cross_cov = self.kernel(x_meas, self.x_synth)

        # current cross-covariance between x_query and x_synth
        y_beta_cross_cov = prior_cross_cov @ self.synth_prior_cov_inv @ init_cov

        # self-covariance among x_query
        y_cov = y_beta_cross_cov @ self.synth_prior_cov_inv @ prior_cross_cov.T

        post_cov = init_cov - y_beta_cross_cov.T @ (scipy.linalg.solve(y_cov + y_meas_cov_matrix, y_beta_cross_cov, sym_pos=True))
        return post_cov