import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple

def generate_dodecahedral_graph(k, rem) -> nx.Graph:
    adj_matrix = """01110000000000000000
10000000000000100001
10000000000000011000
10000000000000000110
00000001101000000000
00000001000110000000
00000001010001000000
00001110000000000000
00001000010000100000
00000010100000000001
00001000000100010000
00000100001000001000
00000100000001000100
00000010000010000010
01000000100000010000
00100000001000100000
00100000000100000100
00010000000010001000
00010000000001000001
01000000010000000010"""
    adj_matrix = np.array([[int(x) for x in row] for row in adj_matrix.split("\n")])

    expanded_matrix = np.kron(adj_matrix, np.ones((k, k), dtype=int))
    adj_matrix = expanded_matrix

    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    # Remove rem edges at random
    edges = list(G.edges())
    print(f"Removing {rem} edges at random, out of {len(edges)}")
    edges_to_remove = np.random.choice(len(edges), size=rem, replace=False)
    edges_to_remove = [edges[i] for i in edges_to_remove]
    for edge in edges_to_remove:
        G.remove_edge(*edge)

    # Set all edges to have a weight of 1
    for u, v in G.edges():
        G[u][v]['weight'] = 1

    return G

def generate_random_weighted_graph(num_nodes: int, max_degree: int, weight_range: Tuple[float, float] = (1.0, 10.0)) -> nx.Graph:
    """
    Generate a random connected weighted graph with a specified maximum degree.
    
    Args:
        num_nodes: Number of nodes in the graph
        max_degree: Maximum degree of any node
        weight_range: Range of edge weights (min, max)
    
    Returns:
        G: A random connected weighted graph
    """
    # Generate a random graph
    while True:
        # Create a random graph with a maximum degree constraint
        G = nx.random_regular_graph(d=max_degree, n=num_nodes)
        
        # Ensure the graph is connected
        if nx.is_connected(G):
            break

    # Add random weights to the edges
    for (u, v) in G.edges():
        G[u][v]['weight'] = np.random.uniform(*weight_range)
    
    return G

def visualize_graph(G: nx.Graph, title: str = "Graph Visualization") -> None:
    """
    Visualize the graph with edge weights
    
    Args:
        G: NetworkX graph
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Create a layout for the graph
    pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), seed=42)
    
    # Draw vertices
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    
    # Draw edges with different colors and widths based on weights
    edge_colors = [0 for (u, v) in G.edges()]
    edge_widths = [5*G[u][v]['weight'] for (u, v) in G.edges()]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, edge_cmap=plt.cm.gist_gray)
    
    # Add vertex labels
    nx.draw_networkx_labels(G, pos)
    
    # Add edge weight labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Add legend and title
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_cut(G: nx.Graph, cut: np.ndarray, title: str = "Max-Cut Visualization") -> None:
    """
    Visualize the graph cut
    
    Args:
        G: NetworkX graph
        cut: Array of ±1 indicating vertex assignments
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Create a layout for the graph
    pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), seed=42)
    
    # Draw vertices
    side_0 = [i for i in G.nodes() if cut[i] == 1]
    side_1 = [i for i in G.nodes() if cut[i] == -1]
    
    # Draw the two sides in different colors
    nx.draw_networkx_nodes(G, pos, nodelist=side_0, node_color='lightblue', 
                          node_size=500, label='Side 0')
    nx.draw_networkx_nodes(G, pos, nodelist=side_1, node_color='lightgreen', 
                          node_size=500, label='Side 1')
    
    # Draw edges with different colors and widths based on whether they cross the cut
    cut_edges = [(u, v) for (u, v) in G.edges() if cut[u] != cut[v]]
    uncut_edges = [(u, v) for (u, v) in G.edges() if cut[u] == cut[v]]
    
    # Draw cut edges
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='r', 
                          width=[G[u][v]['weight']/2 for (u, v) in cut_edges],
                          label='Cut Edges')
    
    # Draw uncut edges
    nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, edge_color='gray', 
                          width=[G[u][v]['weight']/2 for (u, v) in uncut_edges],
                          alpha=0.5, label='Uncut Edges')
    
    # Add vertex labels
    nx.draw_networkx_labels(G, pos)
    
    # Add edge weight labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Add legend and title
    plt.title(title)
    plt.legend()
    plt.axis('off')
    
    # Calculate and display cut value
    cut_value = sum(G[u][v]['weight'] for (u, v) in cut_edges)
    total_weight = sum(G[u][v]['weight'] for (u, v) in G.edges())
    plt.figtext(0.02, 0.02, f'Cut Value: {cut_value:.2f} ({(cut_value/total_weight*100):.1f}% of total weight)')
    
    plt.show()
