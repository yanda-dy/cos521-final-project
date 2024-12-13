import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
from typing import Tuple

class MaxCutExact:
    def __init__(self, G: nx.Graph):
        """
        Initialize exact Max-Cut solver using Gurobi
        
        Args:
            G: NetworkX graph with edge weights stored in 'weight' attribute
        """
        self.G = G
        self.n = G.number_of_nodes()
        self.m = G.number_of_edges()
        
        # Extract edge information
        self.edges = list(G.edges())
        self.weights = {(i,j): G[i][j].get('weight', 1.0) for i,j in self.edges}
        
    def solve(self, time_limit: int = None, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Solve Max-Cut exactly using Integer Linear Programming
        
        Args:
            time_limit: Maximum time (in seconds) to spend solving
            verbose: Whether to print solver output
            
        Returns:
            cut: Array of vertex assignments (0 or 1)
            value: Optimal cut value
        """
        # Create Gurobi model
        model = gp.Model("MaxCut")
        
        # Set verbosity
        model.setParam('OutputFlag', 1 if verbose else 0)
        
        # Set time limit if specified
        if time_limit is not None:
            model.setParam('TimeLimit', time_limit)
        
        # Create binary variables for vertex assignments
        # x[i] = 1 if vertex i is on one side, 0 if on the other
        x = model.addVars(self.n, vtype=GRB.BINARY, name="x")
        
        # Create variables for cut edges
        # y[i,j] = 1 if edge (i,j) is cut, 0 otherwise
        y = model.addVars(self.edges, vtype=GRB.BINARY, name="y")
        
        # Objective: Maximize sum of weights of cut edges
        model.setObjective(
            gp.quicksum(self.weights[i,j] * y[i,j] for i,j in self.edges),
            GRB.MAXIMIZE
        )
        
        # Constraints: Edge is cut if vertices are on different sides
        # y[i,j] = x[i] XOR x[j]
        # This is modeled using two linear constraints:
        # y[i,j] >= x[i] - x[j] and y[i,j] >= x[j] - x[i]
        for i,j in self.edges:
            model.addConstr(y[i,j] >= x[i] - x[j], f"cut1_{i}_{j}")
            model.addConstr(y[i,j] >= x[j] - x[i], f"cut2_{i}_{j}")
            model.addConstr(y[i,j] <= 2 - x[i] - x[j], f"cut3_{i}_{j}")
            model.addConstr(y[i,j] <= x[i] + x[j], f"cut4_{i}_{j}")
        
        # Break symmetry by fixing one vertex to side 0
        model.addConstr(x[0] == 0, "symmetry")
        
        # Solve the model
        model.optimize()
        
        # Check solution status
        if model.status == GRB.OPTIMAL:
            # Extract solution
            cut = np.array([x[i].X for i in range(self.n)])
            value = model.objVal
            
            # Convert to ±1 format to match SDP solver
            cut = 2 * cut - 1
            
            return cut, value
        elif model.status == GRB.TIME_LIMIT:
            # Return best solution found within time limit
            if model.SolCount > 0:
                cut = np.array([x[i].X for i in range(self.n)])
                value = model.objVal
                cut = 2 * cut - 1
                return cut, value
            else:
                raise Exception("No solution found within time limit")
        else:
            raise Exception(f"Optimization failed with status {model.status}")
    
    def verify_cut(self, cut: np.ndarray) -> float:
        """
        Verify a cut solution and return its value
        
        Args:
            cut: Array of ±1 indicating vertex assignments
            
        Returns:
            value: Cut value
        """
        value = 0.0
        for i, j in self.edges:
            if cut[i] != cut[j]:
                value += self.weights[i,j]
        return value
