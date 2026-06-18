import numpy as np
from fgp.interpolate import fd_interpolate, linear_interpolation
from fgp.canonical_gaussians import Gaussian
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, eye
# from scipy.linalg import solve, inv

class FactorGP:
    """
    FactorGP class. For regression of scalar functions only.
    
    Attributes:
        var_nodes (list)        : list of VariableNode objects
        factor_nodes (list)     : list of FactorNode objects
        h (float)               : inducing node spacing
        joint_belief (Gaussian) : joint belief of the factor GP in canonical form
    """
    def __init__(self, h):
        self.var_nodes = {}          # var_id -> VariableNode
        self.factor_nodes = {}       # factor_id -> FactorNode
        self.h = h
        self.joint_belief = None     # stores η and Λ
        self.var_index_map = {}      # var_id -> row index in joint Λ/η
        self._next_index = 0  

    def add_var_node(self, id, x_loc, adj_list=None, prior_eta=None, prior_lam=None):
        if prior_eta is None:
            prior_eta = np.zeros((1,1))
        if prior_lam is None:
            prior_lam = np.zeros((1,1))

        # Store variable
        var_node = VariableNode(
            id=id,
            x_loc=x_loc,
            adj_list=adj_list or [],
            prior_eta=prior_eta,
            prior_lam=prior_lam
        )
        self.var_nodes[id] = var_node

        # Assign joint index
        idx = self._next_index
        self.var_index_map[id] = idx
        self._next_index += 1

        # Initialize or expand joint belief
        N = self._next_index
        if self.joint_belief is None:
            # Use LIL for incremental sparse insertion
            lam = lil_matrix((1,1))
            lam[0,0] = prior_lam
            eta = prior_eta.copy()
            self.joint_belief = Gaussian(eta, lam)
        else:
            # Expand sparse matrix
            old_lam = self.joint_belief.lam.tolil()
            lam = lil_matrix((N,N))
            lam[:N-1,:N-1] = old_lam
            lam[N-1,N-1] = prior_lam
            eta = np.zeros((N,1))
            eta[:N-1] = self.joint_belief.eta
            eta[N-1] = prior_eta
            self.joint_belief = Gaussian(eta, lam)

        # Update adj_list of connected factors
        if adj_list:
            for factor_id in adj_list:
                factor = self.get_factor_from_id(factor_id)
                factor.adj_list.append(id)

    def add_data_factor(self, z, x, adj_list, jac, msmt_cov):
        m = jac.shape[0]
        prior_eta = jac * (1 / msmt_cov) * z
        prior_lam = jac @ jac.T * (1 / msmt_cov) + np.eye(m) * 1e-8
        factor_id = len(self.factor_nodes)

        new_factor = FactorNode(
            id=factor_id,
            factor_type="data",
            dim=(m,1),
            adj_idx_list=list(adj_list),
            prior_eta=prior_eta,
            prior_lam=prior_lam,
            z_meas=z,
            x_meas=x
        )
        self.factor_nodes[factor_id] = new_factor

        # Update adjacency lists and messages
        for var_id in adj_list:
            var_node = self.get_variable_from_id(var_id)
            var_node.adj_list.append(factor_id)
            var_node.messages[factor_id] = Gaussian(np.zeros((1,1)), np.zeros((1,1)))

        # Update joint belief sparse matrix
        indices = [self.var_index_map[var_id] for var_id in adj_list]
        lam_block = lil_matrix(prior_lam)
        self.joint_belief.lam[np.ix_(indices, indices)] += lam_block
        self.joint_belief.eta[indices] += prior_eta

    def add_gp_factor(self, id, prior_eta, prior_lam, adj_list):
        gp_factor = FactorNode(
            id=id,
            adj_idx_list=adj_list,
            dim=(len(adj_list),1),
            factor_type="gp",
            prior_eta=prior_eta,
            prior_lam=prior_lam
        )
        self.factor_nodes[id] = gp_factor

        # Update adjacency lists and messages
        for var_id in adj_list:
            var_node = self.get_variable_from_id(var_id)
            var_node.adj_list.append(id)
            var_node.messages[id] = Gaussian(np.zeros((1,1)), np.zeros((1,1)))

        # Update joint belief sparse matrix
        indices = [self.var_index_map[var_id] for var_id in adj_list]
        lam_block = lil_matrix(prior_lam)
        self.joint_belief.lam[np.ix_(indices, indices)] += lam_block
        self.joint_belief.eta[indices] += prior_eta
    
    
    def remove_factor(self, id):
        """
        Removes a factor from the graph
        """
        factor = self.get_factor_from_id(id)
        adj_list = factor.adj_list
        self.factor_nodes.remove(factor)

        for var_idx in adj_list:
            var = self.get_variable_from_id(var_idx)
            if var is not None:
                var.adj_list.remove(id)
                del var.messages[id]
        
    def query(self, x_query, pre_W=None):

        M, d = x_query.shape
        inducing_xs = np.vstack([var.x_loc for var in self.var_nodes.values()])

        if d == 1:
            w, nearest = linear_interpolation(x_query, inducing_xs)

            from scipy.sparse import lil_matrix
            W = lil_matrix((M, len(self.var_nodes)))

            for i in range(M):
                W[i, nearest[i,0]] = 1 - w[i]
                W[i, nearest[i,1]] = w[i]

            W = W.tocsr()
        
        elif pre_W is not None:
            W = pre_W 

        else:
            W = fd_interpolate(
                x_query,
                np.array([
                    np.unique(inducing_xs[:,0]).size,
                    np.unique(inducing_xs[:,1]).size
                ]),
                self.h
            )
            # already CSR — do nothing

        Lambda = self.joint_belief.lam.tocsc()
        eta = self.joint_belief.eta

        try:
            lu = splu(Lambda)
        except RuntimeError as e:
            if "singular" not in str(e).lower():
                raise
            n = Lambda.shape[0]
            Lambda_reg = (Lambda + 1e-6 * eye(n, format="csc")).tocsc()
            lu = splu(Lambda_reg)

        # Mean: $W \Lambda^{-1} \eta$.
        mu = lu.solve(eta)
        predictive_mean = W @ mu

        # covariance: $W \Lambda^{-1} W^T$
        WT = W.transpose()          
        X = lu.solve(WT.toarray()) 
        predictive_cov = W @ X

        return predictive_mean, predictive_cov
    
    def get_posterior_mean(self):
        """Returns the posterior mean at all grid nodes as a 1-D (N,) array.

        Reuses the same regularised LU solve as :meth:`query` so the two are
        always consistent.
        """
        Lambda = self.joint_belief.lam.tocsc()
        eta = self.joint_belief.eta
        try:
            lu = splu(Lambda)
        except RuntimeError as e:
            if "singular" not in str(e).lower():
                raise
            n = Lambda.shape[0]
            Lambda_reg = (Lambda + 1e-6 * eye(n, format="csc")).tocsc()
            lu = splu(Lambda_reg)
        return lu.solve(eta).ravel()

    def get_joint_belief(self):
        """
        Returns the joint posterior distribution over all inducing points in MOMENT form.

        Returns:
            (mean, var)
        """
        return self.joint_belief.get_moment_form()
        
    def get_factor_from_id(self, id):
        """
        Returns factor node corresponding to unique id
        
        Args:
            id (int) : the unique id for the node
        """
        return self.factor_nodes.get(id)
    

    def get_variable_from_id(self, id):
        """
        Returns variable node corresonding to id
        
        Args:
            id (int) : the unique id for the node
        """
        return self.var_nodes.get(id)
    
    def get_connected_variable_ids(self, variable_node):
        """
        Return the ids of all variable nodes connected to the input node by a factor.

        Args:
            variable_node (VariableNode) : the input variable node. 
        """
        connected_var_ids = []
        for factor_id in variable_node.adj_list:
            factor = self.factor_nodes[factor_id]
            connected_var_ids += factor.adj_list
        connected_var_ids = [v for v in list(set(connected_var_ids)) if v != variable_node.id]
        return connected_var_ids

class VariableNode:
    """ VariableNode class definition

     Attributes:
        id (int)             : unique identifier of this variable node
        x_loc (float)        : the location of this variable node in the input domain
        adj_list (list)      : adjacent factor node ids
        prior_eta (np.array) : prior information vector
        prior_lam (np.array) : prior precision matrix
        messages (dict)      : messages to neighbouring factors
     """
    def __init__(self, id, x_loc, prior_eta, prior_lam, adj_list=None):
        if adj_list:
            self.adj_list = adj_list
            self.messages = {adj_id: Gaussian(np.zeros((1,1)), np.ones((1,1))*1e-8) for adj_id in self.adj_list}
        else:
            self.adj_list = []
            self.messages = {}
        self.id = id
        self.x_loc = x_loc
        self.prior_eta = prior_eta
        self.prior_lam = prior_lam
        self.belief = Gaussian(prior_eta,
                               prior_lam)
        
    
    def compute_messages(self, graph):
        """
        Compute the variable to factor messages for this node
        
        Args:
            graph (FactorGraph) : the FactorGraph object this node belongs to.
        """
        for factor_idx in self.adj_list:
            factor_node = graph.get_factor_from_id(factor_idx)
            factor_to_var_msg = factor_node.messages[self.id]
            self.messages[factor_idx] = self.belief.get_quotient(factor_to_var_msg)

    def update_belief(self, graph):
        """ Update the belief stored at this variable node
        
        Args:
            graph (FactorGraph) : the FactorGraph object this node belongs to.
        """
        self.belief = Gaussian(self.prior_eta.copy(),
                               self.prior_lam.copy())
        for factor_idx in self.adj_list:
            factor_node = graph.get_factor_from_id(factor_idx)
            factor_to_var_msg = factor_node.messages[self.id]
            self.belief.product(factor_to_var_msg)

class FactorNode:
    """ FactorNode class definition. For binary factors only.

    Attributes:
        id (int)                : unique identifier of this node
        factor_type (str)       : factor type, i.e. data or gp
        dim (tuple)             : dimension of this factor's potential
        adj_list (list)         : adjacent variable node ids
        z_meas (float)          : measurement of latent function
        x_meas (float)          : location of the measurement
        messages (dict)         : dictionary to store message history to adjacent nodes
        factor (Gaussian)       : the current factor
    """
    def __init__(self, id, factor_type, dim, adj_idx_list, prior_eta, prior_lam, z_meas=None, x_meas=None):
        self.id = id
        self.factor_type = factor_type
        self.dim = dim
        self.adj_list = adj_idx_list
        self.factor = Gaussian(prior_eta, prior_lam)
        self.z_meas = z_meas
        self.x_meas = x_meas
        self.messages = {adj_id: Gaussian(np.zeros((1,1)), np.zeros((1,1))) for adj_id in self.adj_list}

    def compute_messages(self, graph, damping=1):
        """
        Compute the messages from this factor to adjacent variable nodes
        
        Args:
            graph (FactorGraph) : the FactorGraph object this factor node belongs to.
        """

        # o: outgoing, no: not outgoing
        # For each outgoing node
        for o, id in enumerate(self.adj_list):
            product_eta = self.factor.eta.copy()
            product_lam = self.factor.lam.copy()
            
            # Sum all incoming contributions from variables to this factor, except for the outgoing node
            other_idx = [i for i in range(len(self.adj_list)) if i != o]
            for no in other_idx:
                var_node = graph.get_variable_from_id(self.adj_list[no])
                var_to_factor_msg = var_node.messages[self.id]
                product_eta[no] += var_to_factor_msg.eta.flatten()
                product_lam[no,no] += var_to_factor_msg.lam.flatten()

            # Marginalise out the non-outgoing contributions from the message via Schur complement
            lam_oo = product_lam[o,o].reshape(1,1)
            lam_on = product_lam[o, other_idx].reshape(1,-1)
            lam_no = product_lam[other_idx, o].reshape(-1,1)
            lam_nn = product_lam[np.ix_(other_idx, other_idx)]
            eta_o = product_eta[o].reshape(1,1)
            eta_n = product_eta[other_idx].reshape(-1,1)

            lam_nn_inv = np.linalg.solve(lam_nn, np.eye(len(other_idx)))
            marginal_eta = eta_o - lam_on @ lam_nn_inv @ eta_n
            marginal_lam = lam_oo - lam_on @ lam_nn_inv @ lam_no

            out_id = id
            if damping:
                beta = 0.8
                new_msg = Gaussian(beta * marginal_eta.reshape(-1,1), beta * marginal_lam.reshape(-1,1))
                prev_msg = Gaussian((1-beta)*self.messages[out_id].eta, (1-beta)* self.messages[out_id].lam)
                new_msg.product(prev_msg)
                self.messages[out_id] =  new_msg
            else:
                self.messages[out_id] = Gaussian(marginal_eta.reshape(-1,1), marginal_lam.reshape(-1,1))