import numpy as np
import cvxpy as cp
import scipy.linalg
import networkx as nx
from typing import Tuple


class GoemansWilliamson:
    def __init__(self, G: nx.Graph):
        """
        Initialize the Goemans-Williamson Max-Cut solver
        
        Args:
            G: NetworkX graph with edge weights stored in 'weight' attribute
        """
        self.G = G
        self.n = G.number_of_nodes()
        self.edges = list(G.edges())
        self.weights = np.array([G[i][j].get('weight', 1.0) for i, j in self.edges])

    def solve_sdp(self) -> Tuple[np.ndarray, float]:
        """
        Solve the Max-Cut SDP relaxation
        
        Returns:
            vectors: Unit vectors corresponding to the solution
            objective_value: SDP objective value
        """
        X = cp.Variable((self.n, self.n), symmetric=True)
        adj_matrix = np.zeros((self.n, self.n))
        for (i, j), w in zip(self.edges, self.weights):
            adj_matrix[i, j] = adj_matrix[j, i] = w
        
        # Objective: maximize 0.5 * Trace(W * (1 - X))
        # Equivalent to 0.5 * Trace(W) - 0.5 * Trace(W * X)
        objective = cp.Maximize(-cp.trace(adj_matrix @ X))
        
        # Constraints
        constraints = [
            X >> 0,  # Positive semidefinite
            cp.diag(X) == 1  # Unit vectors
        ]
        
        # Solve the SDP
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
        except Exception as e:
            raise Exception(f"Failed to solve SDP: {e}")
        
        if problem.status != cp.OPTIMAL:
            raise Exception(f"SDP solver status: {problem.status}")
        
        X_val = X.value
        
        # Eigendecomposition to extract vectors
        eigvals, eigvecs = scipy.linalg.eigh(X_val)
        eigvals[eigvals < 1e-10] = 0
        vectors = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))
        
        return vectors, problem.value

    def random_hyperplane_rounding(self, vectors: np.ndarray) -> np.ndarray:
        """
        Perform random hyperplane rounding
        
        Args:
            vectors: Unit vectors from SDP solution
            
        Returns:
            x: Array of ±1 indicating cut assignment
        """
        g = np.random.normal(0, 1, size=vectors.shape[1])
        projections = vectors @ g
        x = np.sign(projections)
        return x

    def calculate_cut_value(self, x: np.ndarray) -> float:
        """
        Calculate the value of a cut
        
        Args:
            x: ±1 array representing the cut assignment
            
        Returns:
            cut_value: Value of the cut
        """
        cut_value = 0.0
        for (i, j), w in zip(self.edges, self.weights):
            if x[i] != x[j]:  # Different sides of the cut
                cut_value += w
        return cut_value

    def solve_max_cut(self) -> Tuple[np.ndarray, float]:
        """
        Solve the Max-Cut problem using the Goemans-Williamson algorithm
        
        Returns:
            x: Final cut assignment
            cut_value: Value of the cut
        """
        # Solve the SDP relaxation
        vectors, _ = self.solve_sdp()
        
        # Perform random hyperplane rounding
        x = self.random_hyperplane_rounding(vectors)
        
        # Calculate the cut value
        cut_value = self.calculate_cut_value(x)
        
        return x, cut_value
