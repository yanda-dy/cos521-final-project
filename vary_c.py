from utils import *
from hsieh_kothari import MaxCutSDP
from tqdm import tqdm

G = generate_dodecahedral_graph(3, 10)
max_degree = max([d for n, d in G.degree()])
print(f"Max degree: {max_degree}")

hk_solver = MaxCutSDP(G)
vectors, hk_sdp_value = hk_solver.solve()

trials = 5000
Cs = [0.01 * i for i in range(1, 10)] + [0.1 * i for i in range(1, 10)] + [i for i in range(1, 10)]
hk_S_ev = []
hk_delta_ev = []
for c in tqdm(Cs):
    S_sizes, deltas = hk_solver.simulate_sdp_statistics(vectors, num_simulations=trials, C=c)
    deltas = [d[1] for d in deltas]
    hk_S_ev.append(np.mean(S_sizes))
    hk_delta_ev.append(np.mean(deltas))

fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.plot(np.log10(Cs), hk_S_ev, label="$|S|$")
ax1.set_xlabel("$\log_{10}(C)$")
ax1.set_ylabel("$\mathbb{E}[|S|]$")
ax1.set_title("$\mathbb{E}[|S|]$ vs. $\log(C)$")
ax1.legend()
plt.tight_layout()
plt.savefig("vary_c_dodec310_S.png", dpi=300)

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.plot(np.log10(Cs), hk_delta_ev, label="$\Delta_i$")
ax2.set_xlabel("$\log_{10}(C)$")
ax2.set_ylabel("$\mathbb{E}[\Delta_i]$")
ax2.set_title("$\mathbb{E}[\Delta_i]$ vs. log(C)")
ax2.legend()
plt.tight_layout()
plt.savefig("vary_c_dodec310_delta.png", dpi=300)

fig3, ax3 = plt.subplots(figsize=(6, 6))
score = [a*b for a,b in zip(hk_S_ev, hk_delta_ev)]
ax3.plot(np.log10(Cs), score, label="$|S| \cdot \Delta_i$")
ax3.set_xlabel("$\log_{10}(C)$")
ax3.set_ylabel("$\mathbb{E}[|S|] \cdot \mathbb{E}[\Delta_i]$")
ax3.set_title("$\mathbb{E}[|S|] \cdot \mathbb{E}[\Delta_i]$ vs. $\log(C)$")
ax3.legend()
plt.tight_layout()
plt.savefig("vary_c_dodec310_score.png", dpi=300)
