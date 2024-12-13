import networkx as nx
from hsieh_kothari import MaxCutSDP
from max_cut import MaxCutExact
from utils import visualize_cut, visualize_graph
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
from utils import generate_random_weighted_graph

'''
iter = 0
while True:
    iter += 1
    G = generate_random_weighted_graph(10, 3, (1, 1))
    # print(f"Graph has {G.number_of_nodes()} vertices and {G.number_of_edges()} edges")
    sdp_solver = MaxCutSDP(G)
    try:
        vectors, _ = sdp_solver.solve()
    except Exception as e:
        print(f"SDP solver failed: {e}")
    
    cnt = 0
    inner_products = []
    for (i, j) in sdp_solver.edges:
        inner_product = np.dot(vectors[i], vectors[j])
        inner_products.append(inner_product)
        if -0.8 < inner_product < -0.6:
            cnt += 1
    
    if cnt / G.number_of_edges() > 0.8:
        print(f"Iteration {iter}")
        print(G.edges())
        print(inner_products)
        print(nx.adjacency_matrix(G).todense())
        chromatic_number = nx.coloring.greedy_color(G, strategy="largest_first")
        print(f"Chromatic number of G: {max(chromatic_number.values()) + 1}")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        print(f"Average degree: {np.mean([d for n, d in G.degree()])}")
        print(f"Density: {nx.density(G)}")
        print(f"Clustering coefficient: {nx.average_clustering(G)}")
        visualize_graph(G)
        break
# '''

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

k = 5 # Set the value of k
expanded_matrix = np.kron(adj_matrix, np.ones((k, k), dtype=int))
adj_matrix = expanded_matrix

# Create a graph from the adjacency matrix
G = nx.from_numpy_array(adj_matrix)

# Remove rem edges at random
rem = 50
edges = list(G.edges())
print(f"Removing {rem} edges at random, out of {len(edges)}")
edges_to_remove = np.random.choice(len(edges), size=rem, replace=False)
edges_to_remove = [edges[i] for i in edges_to_remove]
for edge in edges_to_remove:
    G.remove_edge(*edge)

# Set all edges to have a weight of 1
for u, v in G.edges():
    G[u][v]['weight'] = 1

print(f"Graph has {G.number_of_nodes()} vertices and {G.number_of_edges()} edges")
sdp_solver = MaxCutSDP(G)
try:
    vectors, _ = sdp_solver.solve()
except Exception as e:
    print(f"SDP solver failed: {e}")

cnt = 0
inner_products = []
for (i, j) in sdp_solver.edges:
    inner_product = np.dot(vectors[i], vectors[j])
    inner_products.append(inner_product)
    if -0.8 < inner_product < -0.6:
        cnt += 1

print(inner_products)
print(nx.adjacency_matrix(G).todense())
chromatic_number = nx.coloring.greedy_color(G, strategy="largest_first")
print(f"Chromatic number of G: {max(chromatic_number.values()) + 1}")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average degree: {np.mean([d for n, d in G.degree()])}")
print(f"Density: {nx.density(G)}")
print(f"Clustering coefficient: {nx.average_clustering(G)}")

plt.hist(inner_products, bins=20, edgecolor='black', density=True)
plt.title('Histogram of Inner Products ($G_3$, 50 deletions)')
plt.xlabel('Inner Product Value')
plt.ylabel('Frequency')
plt.show()
# plt.savefig('inner_products_delmany.png', dpi=300)
# exit()

visualize_graph(G)

cut, value = sdp_solver.solve_max_cut()
print(f"Max-Cut value: {value:.4f}")
visualize_cut(G, cut)

S_sizes, deltas = sdp_solver.simulate_sdp_statistics(vectors, num_simulations=10000, C=0.3)
deltas = [d[1] for d in deltas]

print(f"Expectation of |S|: {np.mean(S_sizes):.4f}")
print(f"Expectation of Delta: {np.mean(deltas):.4f}")

plt.hist(S_sizes, bins=30, alpha=0.7)
plt.title("Distribution of $|S|$")
plt.xlabel("$|S|$")
plt.ylabel("Frequency")
plt.savefig('S_distrib_310.png', dpi=300)
plt.show()

plt.hist(deltas, bins=30, alpha=0.7)
plt.title("Distribution of $\Delta_i$")
plt.xlabel("$\Delta_i$")
plt.ylabel("Frequency")
plt.savefig('delta_distrib_310.png', dpi=300)
plt.show()
