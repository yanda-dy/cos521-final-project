import numpy as np
import cvxpy as cp
import scipy.linalg
import networkx as nx
from typing import Tuple, Dict, List, Set
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class NeighborPartition:
    A: Set[int]  # neighbors also in candidate set
    B: Set[int]  # neighbors that could improve cut when vertex is flipped
    C: Set[int]  # neighbors that are good in current cut

class MaxCutSDP:
    def __init__(self, G: nx.Graph):
        """
        Initialize Max-Cut SDP solver
        
        Args:
            G: NetworkX graph with edge weights stored in 'weight' attribute
        """
        self.G = G
        self.n = G.number_of_nodes()
        self.m = G.number_of_edges()
        max_degree = max(dict(self.G.degree()).values())
        # print(f"Max degree: {max_degree}")
        
        self.edges = list(G.edges())
        self.weights = np.array([G[i][j].get('weight', 1.0) for i,j in self.edges])
        
        # All signs are -1 for Max-Cut
        self.signs = -np.ones(len(self.edges))

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Solve the Max-Cut SDP relaxation using matrix of inner products
        
        Returns:
            vectors: Unit vectors for each vertex
            objective_value: Value of the SDP objective
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

        # Find all distinct triplets of vertices with at least two edges between them
        triplets = set()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                for k in range(j + 1, self.n):
                    edges = [(i, j), (i, k), (j, k)]
                    edge_count = sum(1 for u, v in edges if self.G.has_edge(u, v))
                    if edge_count >= 2:
                        triplets.add((i, j, k))
        # print(f"Number of connected triplets: {len(triplets)}")
        for i, j, k in triplets:
            constraints.append(1 + X[i,j] + X[j,k] + X[i,k] >= 0)
        
        # Solve the SDP
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
        except Exception as e:
            raise Exception(f"Failed to solve SDP: {e}")
            
        if problem.status != cp.OPTIMAL:
            raise Exception(f"SDP solver status: {problem.status}")
        
        X_val = X.value
        # Save X_val to a text file
        # np.save('X_val.npy', X_val)
        
        # Eigendecomposition to extract vectors
        eigvals, eigvecs = scipy.linalg.eigh(X_val)
        eigvals[eigvals < 1e-10] = 0
        vectors = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))
        
        return vectors, problem.value

    def goemans_williamson_rounding(self, vectors: np.ndarray, C: float = 10.0) -> Tuple[np.ndarray, List[int], Dict[int, NeighborPartition]]:
        """
        Implement Goemans-Williamson rounding with local improvement identification
        
        Args:
            vectors: Unit vectors from SDP solution
            C: Constant for epsilon calculation
            
        Returns:
            x: Array of ±1 indicating cut assignment
            S: List of candidate vertices for improvement
            partitions: Dictionary mapping vertices to their neighbor partitions
        """
        g = np.random.normal(0, 1, size=vectors.shape[1])
        projections = vectors @ g
        x = np.sign(projections)

        # Calculate epsilon based on maximum degree
        max_degree = max(dict(self.G.degree()).values())
        eps = 1 / (C * max_degree * np.sqrt(np.log(max_degree)))
        
        # Find candidate set S and partitions
        S = [i for i in range(self.n) if abs(projections[i]) < eps]
        partitions = {}
        for i in S:
            neighbors = set(self.G.neighbors(i))
            A_i = set(j for j in S if j in neighbors)
            B_i = set()
            C_i = set()
            
            for j in neighbors - A_i:
                # Note: all signs are -1 for our max-cut problem
                if x[i] * projections[j] >= eps:
                    B_i.add(j)
                else:
                    C_i.add(j)
            
            partitions[i] = NeighborPartition(A_i, B_i, C_i)
        
        return x, S, partitions

    def local_improvement(self, x: np.ndarray, S: List[int], 
                         partitions: Dict[int, NeighborPartition]) -> np.ndarray:
        """
        Apply local improvement step to improve the cut
        
        Args:
            x: Initial cut assignment
            S: Candidate vertices for improvement
            partitions: Neighbor partitions for each vertex
            
        Returns:
            x_prime: Improved cut assignment
        """
        x_prime = x.copy()
        
        for i in S:
            partition = partitions[i]
            weight_B = sum(self.G[i][j].get('weight', 1.0) for j in partition.B)
            weight_AC = sum(self.G[i][j].get('weight', 1.0) for j in (partition.A | partition.C))
            
            # Flip if cut improves
            if weight_B > weight_AC:
                x_prime[i] *= -1
        
        return x_prime

    def solve_max_cut(self, C: float = 10.0) -> Tuple[np.ndarray, float]:
        """
        Solve the Max-Cut problem with local improvement
        
        Args:
            C: Constant for epsilon calculation
            
        Returns:
            x: Final cut assignment
            cut_value: Value of the cut
        """
        vectors, sdp_value = self.solve()
        x, S, partitions = self.goemans_williamson_rounding(vectors, C)
        x_final = self.local_improvement(x, S, partitions)
        cut_value = self.calculate_cut_value(x_final)
        return x_final, cut_value
    
    def calculate_cut_value(self, x: np.ndarray) -> float:
        """Calculate the value of a cut"""
        value = 0.0
        for (i, j), w in zip(self.edges, self.weights):
            if x[i] != x[j]:
                value += w
        return value

    def simulate_sdp_statistics(self, vectors: np.ndarray, num_simulations: int = 1000, C: float = 10.0) -> Tuple[List[int], List[float]]:
        """
        Simulate distributions of |S| and Delta_i over multiple random projections
        
        Args:
            vectors: Unit vectors from SDP solution
            num_simulations: Number of random projections to simulate
            C: Constant for epsilon calculation
            
        Returns:
            S_sizes: List of |S| values from each simulation
            deltas: List of Delta_i values from all simulations
        """
        S_sizes = []
        deltas = []
        
        for _ in tqdm(range(num_simulations)):
            # Apply Goemans-Williamson rounding
            x, S, partitions = self.goemans_williamson_rounding(vectors, C)
            
            # Record |S|
            S_sizes.append(len(S))
            
            # Calculate Delta_i
            # temp = []
            for i in S:
                partition = partitions[i]
                weight_B = sum(self.G[i][j].get('weight', 1.0) for j in partition.B)
                weight_AC = sum(self.G[i][j].get('weight', 1.0) for j in (partition.A | partition.C))
                delta_i = max(0, weight_B - weight_AC)
                degree_i = self.G.degree(i)
                deltas.append((degree_i, delta_i))
            # if len(temp) == 0:
            #     temp = [0]
            # deltas.append(np.mean(temp))
        
        return S_sizes, deltas
    
    def plot_inner_product_distribution(self, vectors: np.ndarray):
        """
        Plot the distribution of <v_i, v_j> for (i, j) in edges of the graph
        
        Args:
            vectors: Unit vectors from SDP solution
        """
        inner_products = []
        
        for (i, j) in self.edges:
            inner_product = np.dot(vectors[i], vectors[j])
            inner_products.append(inner_product)
        
        # Plot the distribution
        import matplotlib.pyplot as plt

        plt.hist(inner_products, bins=200, alpha=0.7)
        plt.title("Distribution of <v_i, v_j> for (i, j) ∈ E")
        plt.xlabel("<v_i, v_j>")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
