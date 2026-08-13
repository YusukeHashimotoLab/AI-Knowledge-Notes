---
title: "Chapter 3: Transpilation — Mapping to Connectivity"
chapter_title: "Chapter 3: Transpilation — Mapping to Connectivity"
subtitle: Coupling Graphs as Consequences of Physics, the Layout Problem, SWAP Insertion, and How to Tell a Correct Router From a Plausible One
reading_time: 45-50 minutes
difficulty: Advanced
code_examples: 7
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-software-stack-introduction/chapter-3.html>) | Last sync: 2026-08-13

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to the Quantum Software Stack](<index.html>) > Chapter 3

Chapter 2 compiled circuits as though any qubit could interact with any other. Exactly one family of hardware behaves that way, and the reason is physical: a trapped-ion chain couples every pair through a shared motional mode, so the coupling graph is complete. A superconducting chip couples qubits capacitively to their neighbours, on a plane, with the additional constraint that neighbouring transition frequencies must not collide — and the graph that results has degree three or four and a diameter that grows with the chip. [Introduction to Quantum Hardware](<../quantum-hardware-introduction/index.html>) derives those graphs from the gate mechanisms; this chapter takes them as given and asks what a compiler has to do about them.

It has to do two things, and only one of them is easy to state. **Layout** is the choice of which physical qubit holds which logical one at the start, and it is a hard combinatorial problem that no compiler solves exactly. **Routing** is the insertion of SWAP gates when the layout turns out to be wrong for a gate that the circuit demands, and each SWAP costs three CX gates on hardware that has no native SWAP. The measurements below dwarf the ones in Chapter 2: the optimizer there removed a quarter of the gates in a circuit, and routing a twelve-qubit QFT onto a heavy-hex graph multiplies its CX count by 4.6. Connectivity is not a secondary consideration.

There is also a new correctness problem, and it is the chapter's methodological centre. A routed circuit is deliberately *not* the unitary that was written down — the SWAPs have permuted which physical line holds which logical qubit. So Chapter 1's equivalence check does not apply as it stands, and a router that reports its permutation carelessly will pass every test that ignores the permutation and fail on hardware. Section 3.3 rebuilds the check, and Code Example 4 shows what the naive version misses: an error of order one, on exactly the circuits where a SWAP was inserted.

## Learning Objectives

After completing this chapter, you will be able to:

  * Represent a coupling graph as data — adjacency plus an all-pairs shortest-path table — and construct all-to-all, linear, square-grid and heavy-hex topologies
  * Predict the routing overhead of a circuit from the mean distance of the coupling graph, and explain why that estimate is an upper bound rather than a prediction
  * State the layout problem, explain why it is NP-hard, and explain why an exact answer would nonetheless be worth very little
  * Implement a nearest-neighbour SWAP router that tracks its own permutation, and state precisely what it does not do that SABRE-style routers do
  * Build the permutation-aware equivalence check $U_{\mathrm{phys}}P(\ell_0) = P(\ell_f)(U_{\mathrm{log}}\otimes I)$, verify routed circuits exactly on small devices and by sampling on large ones, and demonstrate that the permutation-blind version of the check fails
  * Measure SWAP counts for GHZ and QFT circuits on three topologies, and separate the contribution of the graph from the contribution of the circuit's interaction structure
  * Compare a heuristic router against an exact one on cases small enough for both, and attribute the excess to gate order, lookahead, layout and objective

* * *

## 3.1 Connectivity Graphs

### Where the graph comes from

A coupling graph is a statement about a gate mechanism, and the four topologies used in this chapter are four different mechanisms.

| Topology | Physical mechanism | Consequence |
| --- | --- | --- |
| **All-to-all** | every qubit couples to one shared bosonic mode — a motional mode of an ion crystal, or a bus resonator | no routing at all; the gate rate is shared between all pairs and degrades as modes crowd |
| **Line** | nearest-neighbour exchange between qubits in a row, as for gate-defined spins | mean distance grows as $n/3$; the worst case for routing |
| **Square grid** | capacitive coupling on a plane, degree four | mean distance grows as $\sqrt{n}$ |
| **Heavy-hex** | a hexagonal lattice with an extra qubit on every bond, so degree never exceeds three | frequency collisions and crosstalk stay manageable; the graph is very sparse |

The heavy-hex row is the one that shows the trade being made. Its purpose is not connectivity — it has *less* connectivity than a grid with the same number of qubits — but the degree bound. A fixed-frequency superconducting qubit with four neighbours has four chances of a frequency collision and four crosstalk channels; capping the degree at three, and inserting an extra qubit on every bond so that no two computational qubits are directly adjacent, buys yield and calibration stability at the cost of a longer average distance. That is a materials and control decision showing up as a graph, and the compiler pays for it.

### Mean distance is the cost predictor

A two-qubit gate between qubits at distance $d$ in the coupling graph needs $d - 1$ SWAPs to bring them together, and each SWAP costs three CX gates. For a circuit whose interacting pairs are spread uniformly over the device, the expected number of extra CX gates per two-qubit gate is therefore

$$ \text{overhead} \;\approx\; 3\left( \bar{d} - 1 \right), \qquad \bar{d} = \frac{2}{n(n-1)}\sum_{i<j} \mathrm{dist}(i,j) $$

This is the estimate to use when no router is available, and Code Example 1 computes it for each topology. It is an *upper* bound in practice, for a reason worth understanding now: it assumes every gate starts from a fresh random placement, and a router does not undo its own SWAPs, so consecutive gates on nearby qubits are cheaper than independent ones. Code Example 5 measures the gap — a factor of two on the grid.

### Code Example 1: Connectivity Graphs as Data

Everything in this chapter runs on the three modules of Chapter 1, re-listed here so the chapter is self-contained. The first is the state-vector simulator of [Introduction to Quantum Computing](<../quantum-computing-introduction/chapter-2.html>) — the functions this chapter uses, verbatim; `probs` and `sample` from that file are not needed here.

```python
"""Minimal state-vector simulator (big-endian: qubit 0 = leftmost = most significant).

Save this file as qcsim.py; every later example does `from qcsim import *`.
"""
import numpy as np

# ---- single-qubit gates -------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta):
    e = np.exp(-1j * theta / 2)
    return np.array([[e, 0], [0, np.conj(e)]], dtype=complex)


# ---- states -------------------------------------------------------------
def ket(bits: str) -> np.ndarray:
    """'01' -> the 4-dimensional basis state |01> (big-endian)."""
    n = len(bits)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[int(bits, 2)] = 1.0
    return psi


def apply_gate(state, U, targets, n):
    """Apply the 2^k x 2^k unitary U to the listed target qubits of an n-qubit state."""
    k = len(targets)
    psi = state.reshape([2] * n)          # 1. view as an n-index tensor
    psi = np.moveaxis(psi, targets, range(k))   # 2. bring targets to the front
    rest = psi.shape[k:]
    psi = psi.reshape(2 ** k, -1)         # 3. flatten and multiply
    psi = U @ psi
    psi = psi.reshape(list((2,) * k) + list(rest))
    psi = np.moveaxis(psi, range(k), targets)   # 4. put the axes back
    return psi.reshape(-1)


CNOT4 = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=complex)


def cnot(state, control, target, n):
    """CNOT with the given control and target; any pair of qubits, any order."""
    return apply_gate(state, CNOT4, [control, target], n)
```

The second is the circuit IR, verbatim from Chapter 1's Code Example 2.

```python
"""Chapter 1, Example 2: the circuit IR of this course.

A circuit is a list of gate tuples. Gate names are strings; qubits are ints
(big-endian, qubit 0 leftmost). Save this file as qir.py; every later example
does `from qir import *`, and every later chapter re-lists it.

    ("h", q)   ("x", q)   ("z", q)   ("s", q)   ("t", q)
    ("rx", theta, q)      ("ry", theta, q)      ("rz", theta, q)
    ("cx", control, target)                     ("cz", q1, q2)
"""
import numpy as np
from qcsim import *

CZ4 = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)

FIXED_1Q = {"h": H, "x": X, "z": Z, "s": S, "t": T}
ROT_1Q = {"rx": rx, "ry": ry, "rz": rz}
TWO_Q = ("cx", "cz")


def gate_qubits(g):
    """The qubits one gate tuple touches, in the order they are written."""
    if g[0] in ROT_1Q:
        return (g[2],)
    if g[0] in TWO_Q:
        return (g[1], g[2])
    if g[0] in FIXED_1Q:
        return (g[1],)
    raise ValueError(f"unknown gate name {g[0]!r}")


def apply_ir_gate(state, g, n):
    """Apply one gate tuple to an n-qubit state vector."""
    name = g[0]
    if name in FIXED_1Q:
        return apply_gate(state, FIXED_1Q[name], [g[1]], n)
    if name in ROT_1Q:
        return apply_gate(state, ROT_1Q[name](g[1]), [g[2]], n)
    if name == "cx":
        return cnot(state, g[1], g[2], n)
    if name == "cz":
        return apply_gate(state, CZ4, [g[1], g[2]], n)
    raise ValueError(f"unknown gate name {name!r}")


def run_circuit(circ, n, psi0=None):
    """Execute a gate-tuple list on the state-vector simulator; return the final state.

    psi0 defaults to |00...0>. Gates are applied left to right, so the matrix
    of the circuit is the product of the gate matrices in reverse order.
    """
    state = ket("0" * n) if psi0 is None else np.asarray(psi0, dtype=complex)
    for g in circ:
        state = apply_ir_gate(state, g, n)
    return state


def circuit_depth(circ, n):
    """Greedy layering by qubit disjointness: how many layers the circuit needs.

    Every gate is assumed to take one unit of time, which is false on hardware
    and is corrected in Chapter 4.
    """
    ready = [0] * n              # first free layer of each qubit
    for g in circ:
        qs = gate_qubits(g)
        layer = max(ready[q] for q in qs)
        for q in qs:
            ready[q] = layer + 1
    return max(ready) if n else 0


def gate_counts(circ):
    """Gate name -> count, plus the key "2q" holding the total of two-qubit gates."""
    counts = {}
    for g in circ:
        counts[g[0]] = counts.get(g[0], 0) + 1
    counts["2q"] = sum(counts.get(name, 0) for name in TWO_Q)
    return counts
```

The third is the checker, verbatim from Chapter 1's Code Example 4. Section 3.3 extends it rather than replacing it.

```python
"""The unitary-equivalence checker of Chapter 1, re-listed.

Save this file as qcheck.py; every later example does `from qcheck import *`.
"""
import numpy as np
from qir import *


def unitary_of(circ, n):
    """The 2^n x 2^n matrix of a circuit: run it once on each basis state."""
    dim = 2 ** n
    U = np.empty((dim, dim), dtype=complex)
    for j in range(dim):
        e = np.zeros(dim, dtype=complex)
        e[j] = 1.0
        U[:, j] = run_circuit(circ, n, psi0=e)
    return U


def best_global_phase(U, V):
    """The phase that makes e^{i phi} V as close to U as a phase can make it.

    It comes from the Hilbert-Schmidt overlap tr(V^dagger U). By Cauchy-Schwarz
    the modulus of that overlap reaches 2^n exactly when U = e^{i phi} V, so an
    overlap near zero is already a proof that the two are inequivalent.
    """
    tr = np.trace(V.conj().T @ U)
    return 1.0 + 0.0j if abs(tr) < 1e-12 else tr / abs(tr)


def phase_free_error(U, V):
    """max |U - e^{i phi} V| after removing the best global phase."""
    return float(np.max(np.abs(U - best_global_phase(U, V) * V)))


def assert_equivalent(a, b, n, label="", atol=1e-10):
    """The test that guards every rewriting pass in this course."""
    err = phase_free_error(unitary_of(a, n), unitary_of(b, n))
    if err > atol:
        raise AssertionError(f"{label}: circuits differ, max error {err:.3e}")
    return err
```

Now the graphs. A device is a dictionary: adjacency sets, an all-pairs shortest-path table computed by breadth-first search, and a sorted edge list.

```python
"""Chapter 3, Example 1: connectivity graphs as data."""
from itertools import combinations
import numpy as np
from qcheck import *


def device(name, n, edges):
    """A coupling graph: adjacency sets plus the all-pairs shortest-path table."""
    adj = {q: set() for q in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):                              # breadth-first search from s
        dist[s][s] = 0
        frontier = [s]
        while frontier:
            nxt = []
            for p in frontier:
                for r in sorted(adj[p]):
                    if dist[s][r] < 0:
                        dist[s][r] = dist[s][p] + 1
                        nxt.append(r)
            frontier = nxt
    return {"name": name, "n": n, "adj": adj, "dist": dist,
            "edges": sorted(tuple(sorted(e)) for e in edges)}


def all_to_all(n):
    """Every pair coupled: a trapped-ion chain, or any bus-mediated architecture."""
    return device(f"all-to-all {n}", n, list(combinations(range(n), 2)))


def line(n):
    """A one-dimensional chain: gate-defined spin qubits in a row."""
    return device(f"line {n}", n, [(q, q + 1) for q in range(n - 1)])


def grid(rows, cols):
    """A square lattice: the natural layout of a planar superconducting chip."""
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return device(f"grid {rows}x{cols}", rows * cols, edges)


def heavy_hex_7():
    """The smallest heavy-hex fragment: an H shape with two degree-3 qubits."""
    return device("heavy-hex 7", 7,
                  [(0, 1), (1, 2), (1, 3), (3, 5), (4, 5), (5, 6)])


def heavy_hex_16():
    """One heavy hexagon -- a 12-cycle -- with four flag qubits hanging off it.

    A heavy-hex lattice is a hexagonal lattice with an extra qubit on every bond,
    so no qubit has more than three neighbours. That bound is the point: it is
    what keeps frequency collisions and crosstalk manageable on a fixed-frequency
    superconducting chip, at the cost of a very sparse graph.
    """
    return device("heavy-hex 16", 16,
                  [(0, 1), (1, 2), (1, 4), (2, 3), (3, 5), (4, 7), (5, 8),
                   (6, 7), (7, 10), (8, 9), (8, 11), (10, 12), (11, 14),
                   (12, 13), (12, 15), (13, 14)])


DEVICES = [all_to_all(16), grid(4, 4), heavy_hex_16(), line(16),
           all_to_all(7), grid(2, 3), heavy_hex_7()]

head = (f"{'device':<16}{'qubits':>7}{'edges':>7}{'deg (mean)':>12}"
        f"{'deg (max)':>11}{'diameter':>10}{'mean dist':>11}")
print(head)
print("-" * len(head))
for d in DEVICES:
    n = d["n"]
    degs = [len(d["adj"][q]) for q in range(n)]
    pairs = [d["dist"][a][b] for a, b in combinations(range(n), 2)]
    print(f"{d['name']:<16}{n:>7}{len(d['edges']):>7}{np.mean(degs):>12.2f}"
          f"{max(degs):>11}{max(pairs):>10}{np.mean(pairs):>11.2f}")

print(f"{'device':<16}{'mean dist':>11}{'extra CX per 2q gate':>23}")
print("-" * 50)
for d in DEVICES[:4]:
    md = np.mean([d["dist"][a][b] for a, b in combinations(range(d["n"]), 2)])
    print(f"{d['name']:<16}{md:>11.2f}{3 * (md - 1):>23.2f}")
```

```text
device           qubits  edges  deg (mean)  deg (max)  diameter  mean dist
--------------------------------------------------------------------------
all-to-all 16        16    120       15.00         15         1       1.00
grid 4x4             16     24        3.00          4         6       2.67
heavy-hex 16         16     16        2.00          3         8       3.68
line 16              16     15        1.88          2        15       5.67
all-to-all 7          7     21        6.00          6         1       1.00
grid 2x3              6      7        2.33          3         3       1.67
heavy-hex 7           7      6        1.71          3         4       2.29
device            mean dist   extra CX per 2q gate
--------------------------------------------------
all-to-all 16          1.00                   0.00
grid 4x4               2.67                   5.00
heavy-hex 16           3.68                   8.05
line 16                5.67                  14.00
```

**What to look for.** Sixteen qubits, four topologies, and the edge count falls from 120 to 15 while the mean distance rises from 1.00 to 5.67. The heavy-hex fragment has the same number of qubits as the grid and two-thirds as many edges, and its mean distance is 38% larger — that is the price of the degree-3 bound, quantified. The second table converts mean distance into the estimate above: all-to-all pays nothing, the $4\times4$ grid pays five extra CX gates per two-qubit gate, heavy-hex eight, and a 16-qubit line fourteen.

Compare that with Chapter 2, which worked hard to remove a quarter of the gates in a circuit. Connectivity can multiply them by five. The two effects are not the same size, and a compiler that spends its effort on peephole rules while ignoring layout has misallocated it.

### The demand side: what a circuit asks for

A coupling graph is only half the problem. The other half is the **interaction graph** of the circuit: which pairs of logical qubits the circuit needs to couple, and how often. Two circuits bracket the range.

The **GHZ chain** — one Hadamard and $n-1$ CX gates in a line — has a path as its interaction graph. A path exists inside every connected device, so *some* layout makes the GHZ chain entirely native, and finding it is the layout problem in its easiest form.

The **QFT** needs every pair exactly once, so its interaction graph is complete. No layout helps: on anything but all-to-all hardware every pair that is not an edge must be routed. That makes it the standard stress test, and it is also a real circuit — Chapter 2 of [Intermediate Quantum Algorithms](<../quantum-algorithms-intermediate/index.html>) builds it and phase estimation on top of it.

### Code Example 2: The Circuits to Be Routed

Both circuits are written in the IR, and both are verified before being used: the controlled phase against its exact matrix, and the QFT circuit against the dense DFT matrix with the bit-reversal permutation applied by hand.

```python
"""Chapter 3, Example 2: the two circuits to be routed, and their interaction graphs.
Continues from Example 1 (same session)."""
from itertools import combinations
import numpy as np
from qcheck import *


def ghz_chain(n):
    """GHZ by a chain of CNOTs: the interaction graph is a path."""
    return [("h", 0)] + [("cx", q, q + 1) for q in range(n - 1)]


def cphase(theta, a, b):
    """A controlled phase in the IR: two CX gates and three Rz rotations."""
    return [("rz", theta / 2, a), ("rz", theta / 2, b),
            ("cx", a, b), ("rz", -theta / 2, b), ("cx", a, b)]


def qft(n):
    """The textbook QFT without the final reversal: every pair interacts once."""
    circ = []
    for q in range(n):
        circ.append(("h", q))
        for r in range(q + 1, n):
            circ += cphase(np.pi / 2 ** (r - q), q, r)
    return circ


def interaction_graph(circ, n):
    """The set of qubit pairs the circuit needs to couple, and how often."""
    weight = {}
    for g in circ:
        qs = gate_qubits(g)
        if len(qs) == 2:
            key = tuple(sorted(qs))
            weight[key] = weight.get(key, 0) + 1
    return weight


CP = np.diag([1.0, 1.0, 1.0, np.exp(1j * 0.7)])
print("The controlled phase, checked against its matrix:")
print(f"  phase-free error at theta = 0.7 : "
      f"{phase_free_error(CP, unitary_of(cphase(0.7, 0, 1), 2)):.2e}")


def dft_matrix(n):
    """The dense DFT on 2^n amplitudes, for checking the QFT circuit."""
    dim = 2 ** n
    j = np.arange(dim)
    return np.exp(2j * np.pi * np.outer(j, j) / dim) / np.sqrt(dim)


def reversal(n):
    """The bit-reversal permutation the textbook QFT leaves for the caller."""
    dim = 2 ** n
    P = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        bits = format(x, f"0{n}b")
        P[int(bits[::-1], 2), x] = 1.0
    return P


print("\nThe QFT circuit against the dense DFT matrix (reversal applied by hand):")
for n in (2, 3, 4, 5):
    err = phase_free_error(dft_matrix(n), reversal(n) @ unitary_of(qft(n), n))
    print(f"  n = {n}: phase-free error {err:.2e}")

print("\nThe two circuits as compilation problems")
head = (f"{'circuit':<14}{'qubits':>7}{'gates':>7}{'CX':>5}{'depth':>7}"
        f"{'pairs used':>12}{'pairs possible':>16}")
print(head)
print("-" * len(head))
for label, circ, n in [("GHZ chain", ghz_chain(6), 6), ("QFT", qft(6), 6),
                       ("GHZ chain", ghz_chain(10), 10), ("QFT", qft(10), 10)]:
    w = interaction_graph(circ, n)
    print(f"{label:<14}{n:>7}{len(circ):>7}{gate_counts(circ)['2q']:>5}"
          f"{circuit_depth(circ, n):>7}{len(w):>12}{n * (n - 1) // 2:>16}")

print("\nWhether a circuit needs routing at all is a question about two graphs")
head = f"{'circuit':<20}" + "".join(f"{d['name']:>16}" for d in
                                    (all_to_all(6), grid(2, 3), heavy_hex_7()))
print(head)
print("-" * len(head))
for label, circ, n in [("GHZ chain, n = 6", ghz_chain(6), 6),
                       ("QFT, n = 6", qft(6), 6),
                       ("GHZ chain, n = 5", ghz_chain(5), 5),
                       ("QFT, n = 5", qft(5), 5)]:
    pairs = interaction_graph(circ, n)
    row = ""
    for d in (all_to_all(6), grid(2, 3), heavy_hex_7()):
        if d["n"] < n:
            row += f"{'too small':>16}"
            continue
        native = sum(1 for (a, b) in pairs if b in d["adj"][a])
        row += f"{f'{native}/{len(pairs)} native':>16}"
    print(f"{label:<20}{row}")
```

```text
The controlled phase, checked against its matrix:
  phase-free error at theta = 0.7 : 1.12e-16

The QFT circuit against the dense DFT matrix (reversal applied by hand):
  n = 2: phase-free error 1.92e-16
  n = 3: phase-free error 1.31e-15
  n = 4: phase-free error 2.09e-15
  n = 5: phase-free error 3.19e-15

The two circuits as compilation problems
circuit        qubits  gates   CX  depth  pairs used  pairs possible
--------------------------------------------------------------------
GHZ chain           6      6    5      6           5              15
QFT                 6     81   30     38          15              15
GHZ chain          10     10    9     10           9              45
QFT                10    235   90     70          45              45

Whether a circuit needs routing at all is a question about two graphs
circuit                 all-to-all 6        grid 2x3     heavy-hex 7
--------------------------------------------------------------------
GHZ chain, n = 6          5/5 native      4/5 native      3/5 native
QFT, n = 6              15/15 native     7/15 native     5/15 native
GHZ chain, n = 5          4/4 native      3/4 native      2/4 native
QFT, n = 5              10/10 native     5/10 native     3/10 native
```

**What to look for.** The QFT circuit reproduces the dense DFT to $3\times10^{-15}$ at $n = 5$, which is the licence to use it as a benchmark. The compilation table then shows the two extremes: at $n = 10$ the GHZ chain uses 9 of the 45 available pairs and the QFT uses all 45, and the QFT's 90 CX gates come from 45 controlled phases at two CX each.

The last table is the lead-in to the next section. Read with the *trivial* layout — logical $q$ on physical $q$ — the GHZ chain is only 4 of 5 pairs native on the $2\times3$ grid, because physical qubits 2 and 3 are not coupled. A different layout makes it 5 of 5. The QFT is 7 of 15 and no layout improves it, because no six-qubit device except the complete graph has fifteen edges.

* * *

## 3.2 The Layout Problem

### Statement

A **layout** is an injection from the circuit's logical qubits into the device's physical qubits. Given a coupling graph $G$ and a circuit, the layout problem is to choose the injection that minimizes the routing cost that follows.

Two special cases show why this is hard. If the circuit's interaction graph is a subgraph of $G$ under some layout, then that layout costs nothing and finding it is exactly **subgraph isomorphism**, which is NP-complete. If no such layout exists — the usual case — then the objective is the SWAP count produced by whatever router runs afterwards, which makes the problem's difficulty depend on the router as well as on the graphs. Either way there is no exact algorithm to hope for, and the search space is
$$ \frac{N!}{(N-n)!} $$
injections of $n$ logical qubits into $N$ physical ones — $2\times10^{33}$ for sixteen logical qubits on a 127-qubit device.

### Why an exact answer would be worth little

The honest response to NP-hardness here is not resignation but a measurement, and it is the one Code Example 6 makes. Exhaustive search is possible on a six- or seven-qubit device, so the optimum is computable there, and a cheap heuristic can be compared against it. What the measurement shows is that a heuristic which lands within a gate or two of the optimum is worth as much as an oracle would be, because the remaining gap is smaller than the variation between reasonable circuits.

The heuristic worth knowing is the one SABRE introduced, and it is six lines. Route the circuit forwards from the trivial layout and keep the map the router ends with; route the *reversed* gate list from there and keep that map; repeat. Each pass moves the starting map towards one that suits the gates the circuit actually contains, and no search over layouts is involved. The intuition is that the map a router ends with is, by construction, adapted to the end of the circuit — so running the circuit backwards from it produces a map adapted to the beginning.

Measuring layout quality requires a router, so the code for this section is Code Example 6, after the router exists.

* * *

## 3.3 Routing, and How to Check It

### SWAP insertion

Given a layout, a router walks the circuit and, whenever a two-qubit gate acts on a pair that is not an edge of the coupling graph, moves one or both of the qubits until it is. The move is a SWAP, and on hardware with no native SWAP a SWAP is three CX gates:

$$ \mathrm{SWAP}_{a,b} = \mathrm{CX}_{a,b}\,\mathrm{CX}_{b,a}\,\mathrm{CX}_{a,b} $$

which Chapter 2 verified and showed to be optimal — SWAP's canonical coordinates are $(\pi/4,\pi/4,\pi/4)$ and no two-CX circuit reaches them. The cheapest way to satisfy a gate at distance $d$ is $d-1$ SWAPs along a shortest path, and that count is the same whether one endpoint walks the whole way or both walk towards each other; only the depth differs.

The router of Code Example 3 does the simplest thing that works and is named honestly for it: for the next two-qubit gate, walk one of its qubits along a shortest path towards the other, one SWAP per hop, in the order the gates were written, with no lookahead. Two properties of that choice matter.

**It keeps its permutation.** After a SWAP, the physical line holding a given logical qubit has changed, and the router records the new map rather than undoing the SWAP. Undoing it would keep the map trivial and make the equivalence check easy — Chapter 1's Code Example 7 does exactly that, and says so — at the cost of doubling the SWAP count. Keeping the permutation is what a real transpiler does, and it is why the check has to be rebuilt.

**It is a baseline, not a router.** Section 3.4 measures what it gives up.

### What SABRE does instead

The routers in production tools descend from SABRE, and its two ideas are worth stating at the level of principle since neither is implemented here.

**Work on the front layer, not the gate order.** A circuit is a partial order, not a sequence: gates on disjoint qubits may be executed in either order. SABRE keeps the set of gates whose predecessors are all done — the **front layer** — and is free to execute any executable member of it. Often a gate further along in the written order is already on an edge, and executing it first removes the need for a SWAP entirely.

**Score candidate SWAPs by a lookahead cost.** Rather than committing to a shortest path for the gate in front, SABRE considers every SWAP on an edge adjacent to the front layer and scores it by the total distance it leaves the front layer *plus* a decayed contribution from an extended set of upcoming gates. A decay factor on recently used qubits discourages the router from moving the same qubit repeatedly. The result is a greedy search over a much better-shaped cost function than "distance for the current gate".

Both ideas cost implementation complexity and neither changes the asymptotics. What they buy is measured in Code Example 7.

### The equivalence check, rebuilt

Let $\ell$ be the map from virtual lines to physical lines, so $\ell[v]$ is the physical line holding virtual qubit $v$; let $\ell_0$ be its initial value and $\ell_f$ its value after routing. Let $P(\ell)$ be the unitary that relabels lines according to $\ell$. The claim "routing preserves the meaning of the circuit" is then exactly

$$ U_{\mathrm{phys}}\, P(\ell_0) \;=\; P(\ell_f)\, \left( U_{\mathrm{log}} \otimes I \right) $$

up to a global phase, where $U_{\mathrm{log}}$ acts on the $n$ logical qubits and $I$ on the $N - n$ idle ones. Read it left to right: place the logical qubits on their initial physical lines, run the physical circuit, and you get the same state as running the logical circuit and placing the result on the *final* physical lines. Everything the router must report honestly is in $\ell_f$.

Two implementations of the test are needed. On a device small enough to build a $2^N \times 2^N$ matrix — $N \le 7$ here — the identity can be checked exactly. On a 16-qubit device a matrix is $4\times10^9$ complex numbers while a state vector is 65536, so the test is run on random input states instead. That is a randomized check rather than a proof, and it is worth being clear about what it buys: it catches a mishandled permutation immediately, because a permutation error is an $O(1)$ error on almost every input.

### Code Example 3: A Nearest-Neighbour Router

```python
"""Chapter 3, Example 3: a nearest-neighbour router.
Continues from Example 2 (same session)."""
import numpy as np
from qcheck import *


def remap(g, loc):
    """Rewrite a gate tuple from virtual qubits to physical lines."""
    if g[0] in ROT_1Q:
        return (g[0], g[1], loc[g[2]])
    if g[0] in TWO_Q:
        return (g[0], loc[g[1]], loc[g[2]])
    return (g[0], loc[g[1]])


def swap_gates(p, q):
    """A SWAP on two coupled physical lines: three CX gates, no native SWAP."""
    return [("cx", p, q), ("cx", q, p), ("cx", p, q)]


def next_hop(dev, p, target):
    """A neighbour of p that is one step closer to target; the lowest index wins."""
    for r in sorted(dev["adj"][p]):
        if dev["dist"][r][target] == dev["dist"][p][target] - 1:
            return r
    raise ValueError("disconnected coupling graph")


def route(circ, dev, layout=None):
    """Insert SWAPs until every two-qubit gate acts on a coupled pair.

    The strategy is the simplest one that works, and it is named honestly: for
    the next two-qubit gate, walk one of its two qubits along a shortest path
    towards the other, one SWAP per hop. Gates are kept in the order written and
    nothing is looked ahead to. Section 3.3 says what SABRE does instead, and
    Example 7 measures the difference.

    loc[v] is the physical line currently holding virtual qubit v. Returns the
    physical circuit, the initial and final maps, and the SWAP count.
    """
    loc = list(range(dev["n"])) if layout is None else list(layout)
    loc0, out, swaps = list(loc), [], 0
    for g in circ:
        qs = gate_qubits(g)
        if len(qs) == 1:
            out.append(remap(g, loc))
            continue
        a, b = qs
        while dev["dist"][loc[a]][loc[b]] > 1:
            p, r = loc[a], next_hop(dev, loc[a], loc[b])
            out += swap_gates(p, r)
            swaps += 1
            u, v = loc.index(p), loc.index(r)        # the two virtual qubits move
            loc[u], loc[v] = r, p
        out.append(remap(g, loc))
    return out, loc0, loc, swaps


dev = grid(2, 3)
circ = ghz_chain(5)
phys, loc0, locf, swaps = route(circ, dev)
print("GHZ chain on 5 qubits, routed on the 2x3 grid with the trivial layout")
print(f"  coupling edges     : {dev['edges']}")
print(f"  logical circuit    : {circ}")
print(f"  physical circuit   : {phys}")
print(f"  SWAPs inserted     : {swaps}   (CX gates "
      f"{gate_counts(circ)['2q']} -> {gate_counts(phys)['2q']})")
print(f"  initial map v -> p : {loc0}")
print(f"  final map   v -> p : {locf}")
print(f"  depth              : {circuit_depth(circ, 5)} -> "
      f"{circuit_depth(phys, dev['n'])}")

print("\nEvery two-qubit gate of the routed circuit is on a coupled pair:")
bad = [g for g in phys if len(gate_qubits(g)) == 2
       and gate_qubits(g)[1] not in dev["adj"][gate_qubits(g)[0]]]
print(f"  gates on uncoupled pairs: {len(bad)}")

print("\nThe same circuit on three topologies")
head = (f"{'device':<16}{'SWAPs':>7}{'CX in':>7}{'CX out':>8}{'blow-up':>9}"
        f"{'depth in':>10}{'depth out':>11}")
print(head)
print("-" * len(head))
for d in (all_to_all(6), grid(2, 3), heavy_hex_7()):
    p, l0, lf, s = route(circ, d)
    cin, cout = gate_counts(circ)["2q"], gate_counts(p)["2q"]
    print(f"{d['name']:<16}{s:>7}{cin:>7}{cout:>8}{cout / cin:>9.2f}"
          f"{circuit_depth(circ, 5):>10}{circuit_depth(p, d['n']):>11}")

print("\nAnd the QFT, which needs every pair")
print(head)
print("-" * len(head))
for d in (all_to_all(6), grid(2, 3), heavy_hex_7()):
    q = qft(5)
    p, l0, lf, s = route(q, d)
    cin, cout = gate_counts(q)["2q"], gate_counts(p)["2q"]
    print(f"{d['name']:<16}{s:>7}{cin:>7}{cout:>8}{cout / cin:>9.2f}"
          f"{circuit_depth(q, 5):>10}{circuit_depth(p, d['n']):>11}")
```

```text
GHZ chain on 5 qubits, routed on the 2x3 grid with the trivial layout
  coupling edges     : [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)]
  logical circuit    : [('h', 0), ('cx', 0, 1), ('cx', 1, 2), ('cx', 2, 3), ('cx', 3, 4)]
  physical circuit   : [('h', 0), ('cx', 0, 1), ('cx', 1, 2), ('cx', 2, 1), ('cx', 1, 2), ('cx', 2, 1), ('cx', 1, 0), ('cx', 0, 1), ('cx', 1, 0), ('cx', 0, 3), ('cx', 3, 4)]
  SWAPs inserted     : 2   (CX gates 4 -> 10)
  initial map v -> p : [0, 1, 2, 3, 4, 5]
  final map   v -> p : [1, 2, 0, 3, 4, 5]
  depth              : 5 -> 11

Every two-qubit gate of the routed circuit is on a coupled pair:
  gates on uncoupled pairs: 0

The same circuit on three topologies
device            SWAPs  CX in  CX out  blow-up  depth in  depth out
--------------------------------------------------------------------
all-to-all 6          0      4       4     1.00         5          5
grid 2x3              2      4      10     2.50         5         11
heavy-hex 7           2      4      10     2.50         5         11

And the QFT, which needs every pair
device            SWAPs  CX in  CX out  blow-up  depth in  depth out
--------------------------------------------------------------------
all-to-all 6          0     20      20     1.00        30         30
grid 2x3              9     20      47     2.35        30         67
heavy-hex 7          12     20      56     2.80        30         57
```

**What to look for.** The GHZ chain on five qubits needs two SWAPs on the $2\times3$ grid, and the CX count goes from 4 to 10 — a factor of 2.5 for a circuit whose interaction graph is a path, entirely because the trivial layout put logical 2 and 3 on physical lines that are not coupled. The final map `[1, 2, 0, 3, 4, 5]` records where the logical qubits ended up, and the check on uncoupled pairs confirms that every two-qubit gate in the output is legal on the device.

The QFT rows are the ones to remember. On the $2\times3$ grid, 20 CX gates become 47 and the depth goes from 30 to 67; on the seven-qubit heavy-hex fragment, 20 become 56. The heavy-hex device has an extra qubit and *worse* routing cost, which is the degree bound showing up again.

### Code Example 4: The Check, Up to the Tracked Permutation

```python
"""Chapter 3, Example 4: the equivalence check, up to the tracked permutation.
Continues from Example 3 (same session)."""
import numpy as np
from qcheck import *


def place(psi, loc, n):
    """Move the qubit on virtual line v to physical line loc[v]."""
    return np.transpose(psi.reshape([2] * n), np.argsort(loc)).reshape(-1)


def permutation_unitary(loc, n):
    """The 2^n x 2^n matrix of that relabelling."""
    dim = 2 ** n
    P = np.empty((dim, dim), dtype=complex)
    for j in range(dim):
        e = np.zeros(dim, dtype=complex)
        e[j] = 1.0
        P[:, j] = place(e, loc, n)
    return P


def routed_error(circ, n_log, dev, phys, loc0, locf):
    """max error of U_phys P(loc0) against P(locf) (U_log tensor I), up to a phase.

    This is the whole content of "routing preserves the meaning of a circuit".
    The routed circuit is a different unitary from the one written down; what is
    preserved is the unitary composed with the relabelling the SWAPs performed,
    and the router is only correct if it reports that relabelling honestly.
    """
    n = dev["n"]
    U_log = unitary_of(circ, n_log)
    U_emb = np.kron(U_log, np.eye(2 ** (n - n_log))) if n > n_log else U_log
    left = unitary_of(phys, n) @ permutation_unitary(loc0, n)
    return phase_free_error(left, permutation_unitary(locf, n) @ U_emb)


def routed_error_sampled(circ, n_log, dev, phys, loc0, locf, trials=8, seed=0):
    """The same test on random input states, for devices too big for a matrix."""
    n, rng, worst = dev["n"], np.random.default_rng(seed), 0.0
    idle = ket("0" * (n - n_log)) if n > n_log else None
    for _ in range(trials):
        v = rng.normal(size=2 ** n_log) + 1j * rng.normal(size=2 ** n_log)
        psi_l = v / np.linalg.norm(v)
        out_l = run_circuit(circ, n_log, psi0=psi_l)
        psi_v = np.kron(psi_l, idle) if idle is not None else psi_l
        out_v = np.kron(out_l, idle) if idle is not None else out_l
        got = run_circuit(phys, n, psi0=place(psi_v, loc0, n))
        want = place(out_v, locf, n)
        ph = np.vdot(want, got)
        ph = ph / abs(ph) if abs(ph) > 1e-12 else 1.0
        worst = max(worst, float(np.max(np.abs(got - ph * want))))
    return worst


print("First, the relabelling itself: P([1, 0]) must be the SWAP matrix.")
print(f"  error: "
      f"{np.max(np.abs(permutation_unitary([1, 0], 2) - unitary_of([('cx', 0, 1), ('cx', 1, 0), ('cx', 0, 1)], 2))):.2e}")

print("\nRouted circuits, checked exactly and by sampling")
head = (f"{'circuit':<12}{'device':<16}{'SWAPs':>7}{'matrix check':>14}"
        f"{'sampled check':>15}{'ignoring the perm':>19}")
print(head)
print("-" * len(head))
CASES = [("GHZ n=5", ghz_chain(5), 5), ("QFT n=4", qft(4), 4),
         ("QFT n=5", qft(5), 5)]
for label, circ, n_log in CASES:
    for d in (all_to_all(6), grid(2, 3), heavy_hex_7()):
        phys, loc0, locf, swaps = route(circ, d)
        exact = routed_error(circ, n_log, d, phys, loc0, locf)
        sampled = routed_error_sampled(circ, n_log, d, phys, loc0, locf, seed=1)
        naive = routed_error(circ, n_log, d, phys, loc0, loc0)
        assert exact < 1e-10, (label, d["name"])
        print(f"{label:<12}{d['name']:<16}{swaps:>7}{exact:>14.1e}"
              f"{sampled:>15.1e}{naive:>19.1e}")

print("\nA non-trivial initial layout is handled by the same identity")
head = f"{'layout v -> p':<22}{'SWAPs':>6}  {'final map':<22}{'matrix check':>14}"
print(head)
print("-" * len(head))
for layout in ([0, 1, 2, 3, 4, 5], [3, 4, 5, 0, 1, 2], [1, 0, 3, 2, 5, 4],
               [0, 1, 2, 5, 4, 3], [3, 0, 1, 2, 5, 4]):
    phys, loc0, locf, swaps = route(ghz_chain(5), grid(2, 3), layout)
    err = routed_error(ghz_chain(5), 5, grid(2, 3), phys, loc0, locf)
    print(f"{str(layout):<22}{swaps:>6}  {str(locf):<22}{err:>14.1e}")
```

```text
First, the relabelling itself: P([1, 0]) must be the SWAP matrix.
  error: 0.00e+00

Routed circuits, checked exactly and by sampling
circuit     device            SWAPs  matrix check  sampled check  ignoring the perm
-----------------------------------------------------------------------------------
GHZ n=5     all-to-all 6          0       0.0e+00        0.0e+00            0.0e+00
GHZ n=5     grid 2x3              2       0.0e+00        0.0e+00            7.1e-01
GHZ n=5     heavy-hex 7           2       0.0e+00        0.0e+00            7.1e-01
QFT n=4     all-to-all 6          0       0.0e+00        0.0e+00            0.0e+00
QFT n=4     grid 2x3              5       0.0e+00        0.0e+00            5.0e-01
QFT n=4     heavy-hex 7           3       0.0e+00        0.0e+00            5.0e-01
QFT n=5     all-to-all 6          0       0.0e+00        0.0e+00            0.0e+00
QFT n=5     grid 2x3              9       0.0e+00        0.0e+00            3.5e-01
QFT n=5     heavy-hex 7          12       0.0e+00        0.0e+00            3.5e-01

A non-trivial initial layout is handled by the same identity
layout v -> p          SWAPs  final map               matrix check
------------------------------------------------------------------
[0, 1, 2, 3, 4, 5]         2  [1, 2, 0, 3, 4, 5]           0.0e+00
[3, 4, 5, 0, 1, 2]         3  [3, 4, 0, 1, 2, 5]           0.0e+00
[1, 0, 3, 2, 5, 4]         2  [0, 3, 1, 2, 5, 4]           0.0e+00
[0, 1, 2, 5, 4, 3]         0  [0, 1, 2, 5, 4, 3]           0.0e+00
[3, 0, 1, 2, 5, 4]         0  [3, 0, 1, 2, 5, 4]           0.0e+00
```

**What to look for.** The relabelling unitary is verified first, against the SWAP circuit of Chapter 2 — if $P([1,0])$ is not the SWAP matrix, nothing downstream means anything. Then every routed circuit passes both the exact check and the sampled check, and the errors are exactly zero rather than $10^{-16}$. That is not luck: routing only changes which axis of the amplitude tensor each factor multiplies, and the extra CX gates are exact permutations of amplitudes, so the same arithmetic is performed in the same order and no rounding difference arises.

The last column is the point of the example. Comparing the routed circuit against the original while *forgetting* the permutation gives an error of $0.35$ to $0.71$ whenever a SWAP was inserted, and exactly zero when none was. That is the failure mode that makes a broken router look correct on the easy cases: the all-to-all rows pass, the sparse rows fail, and a test suite that only contains the former reports success.

The layout table closes the loop with Section 3.2. Four layouts, four SWAP counts, and two of them are zero — the layouts that map the logical chain $0{-}1{-}2{-}3{-}4$ onto the physical paths $0{-}1{-}2{-}5{-}4$ and $3{-}0{-}1{-}2{-}5$, both of which the grid provides. The router did not find those layouts; it was given them. Finding them is Code Example 6.

* * *

## 3.4 Measuring the Cost

### What the number depends on

The routing overhead of a circuit is a product of three things, and reporting one of them alone is uninformative.

  * **The coupling graph**, through its mean distance — the factor computed in Code Example 1.
  * **The circuit's interaction structure.** A path costs almost nothing on any device; a complete graph costs the maximum. Everything real is in between, and the position matters more than the gate count.
  * **The layout and the router**, which together are worth a factor of order two, as Code Examples 6 and 7 measure.

This is why a benchmark that quotes a single number for "the overhead of topology X" is not meaningful, and why synthetic benchmarks are constructed the way they are: a benchmark like quantum volume fixes the circuit *structure* — square circuits of random two-qubit gates on random pairs — so that the number it produces is a property of the machine and the compiler together rather than of a particular algorithm. This course quotes no vendor numbers, and the tables below are measurements of the code above on the graphs defined above, nothing more.

### Code Example 5: SWAP Counts on Three Topologies

```python
"""Chapter 3, Example 5: the cost of connectivity, measured.
Continues from Example 4 (same session)."""
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from qcheck import *

BIG = [all_to_all(16), grid(4, 4), heavy_hex_16()]

print("GHZ chain: SWAPs inserted, and the CX count before -> after")
head = f"{'n':>3}" + "".join(f"{d['name']:>22}" for d in BIG)
print(head)
print("-" * len(head))
for n in range(4, 13):
    row = ""
    for d in BIG:
        phys, l0, lf, s = route(ghz_chain(n), d)
        cin, cout = gate_counts(ghz_chain(n))["2q"], gate_counts(phys)["2q"]
        row += f"{f'{s} swap, {cin}->{cout}':>22}"
    print(f"{n:>3}{row}")

print("\nQFT: the same table for a circuit that needs every pair")
print(head)
print("-" * len(head))
qft_swaps = {d["name"]: [] for d in BIG}
ns = list(range(4, 13))
for n in ns:
    row = ""
    for d in BIG:
        phys, l0, lf, s = route(qft(n), d)
        cin, cout = gate_counts(qft(n))["2q"], gate_counts(phys)["2q"]
        qft_swaps[d["name"]].append(s)
        row += f"{f'{s} swap, {cin}->{cout}':>22}"
    print(f"{n:>3}{row}")

print("\nThe blow-up factor, and what the mean-distance estimate predicted")
head = (f"{'device':<16}{'QFT n=12 CX in':>16}{'CX out':>9}{'measured':>10}"
        f"{'predicted':>11}")
print(head)
print("-" * len(head))
for d in BIG:
    circ = qft(12)
    phys, l0, lf, s = route(circ, d)
    cin, cout = gate_counts(circ)["2q"], gate_counts(phys)["2q"]
    md = np.mean([d["dist"][a][b] for a, b in combinations(range(d["n"]), 2)])
    print(f"{d['name']:<16}{cin:>16}{cout:>9}{cout / cin:>10.2f}"
          f"{1 + 3 * (md - 1):>11.2f}")
print("\nThe same rows, verified by sampling on 16 physical qubits")
head = f"{'circuit':<12}{'device':<16}{'SWAPs':>7}{'gates out':>11}{'sampled check':>15}"
print(head)
print("-" * len(head))
for label, circ, n_log in [("GHZ n=6", ghz_chain(6), 6),
                           ("GHZ n=12", ghz_chain(12), 12),
                           ("QFT n=6", qft(6), 6), ("QFT n=10", qft(10), 10)]:
    for d in BIG[1:]:
        phys, loc0, locf, s = route(circ, d)
        err = routed_error_sampled(circ, n_log, d, phys, loc0, locf,
                                   trials=3, seed=5)
        assert err < 1e-9, (label, d["name"])
        print(f"{label:<12}{d['name']:<16}{s:>7}{len(phys):>11}{err:>15.1e}")
fig, ax = plt.subplots(figsize=(6.2, 4))
for d, style in zip(BIG, ("o-", "s-", "^-")):
    ax.plot(ns, qft_swaps[d["name"]], style, label=d["name"])
ax.set_xlabel("logical qubits in the QFT")
ax.set_ylabel("SWAPs inserted")
ax.set_title("Routing cost of the QFT on 16-qubit devices")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
```

```text
GHZ chain: SWAPs inserted, and the CX count before -> after
  n         all-to-all 16              grid 4x4          heavy-hex 16
---------------------------------------------------------------------
  4          0 swap, 3->3          0 swap, 3->3          0 swap, 3->3
  5          0 swap, 4->4         3 swap, 4->13         2 swap, 4->10
  6          0 swap, 5->5         3 swap, 5->14         5 swap, 5->20
  7          0 swap, 6->6         3 swap, 6->15        10 swap, 6->36
  8          0 swap, 7->7         3 swap, 7->16        11 swap, 7->40
  9          0 swap, 8->8         6 swap, 8->26        15 swap, 8->53
 10          0 swap, 9->9         6 swap, 9->27        15 swap, 9->54
 11        0 swap, 10->10        6 swap, 10->28       20 swap, 10->70
 12        0 swap, 11->11        6 swap, 11->29       24 swap, 11->83

QFT: the same table for a circuit that needs every pair
  n         all-to-all 16              grid 4x4          heavy-hex 16
---------------------------------------------------------------------
  4        0 swap, 12->12        6 swap, 12->30        6 swap, 12->30
  5        0 swap, 20->20       12 swap, 20->56        9 swap, 20->47
  6        0 swap, 30->30       17 swap, 30->81       17 swap, 30->81
  7        0 swap, 42->42      25 swap, 42->117      45 swap, 42->177
  8        0 swap, 56->56      24 swap, 56->128      63 swap, 56->245
  9        0 swap, 72->72      50 swap, 72->222      68 swap, 72->276
 10        0 swap, 90->90      59 swap, 90->267      83 swap, 90->339
 11      0 swap, 110->110     73 swap, 110->329    124 swap, 110->482
 12      0 swap, 132->132     74 swap, 132->354    158 swap, 132->606

The blow-up factor, and what the mean-distance estimate predicted
device            QFT n=12 CX in   CX out  measured  predicted
--------------------------------------------------------------
all-to-all 16                132      132      1.00       1.00
grid 4x4                     132      354      2.68       6.00
heavy-hex 16                 132      606      4.59       9.05

The same rows, verified by sampling on 16 physical qubits
circuit     device            SWAPs  gates out  sampled check
-------------------------------------------------------------
GHZ n=6     grid 4x4              3         15        0.0e+00
GHZ n=6     heavy-hex 16          5         21        0.0e+00
GHZ n=12    grid 4x4              6         30        0.0e+00
GHZ n=12    heavy-hex 16         24         84        0.0e+00
QFT n=6     grid 4x4             17        132        0.0e+00
QFT n=6     heavy-hex 16         17        132        0.0e+00
QFT n=10    grid 4x4             59        412        0.0e+00
QFT n=10    heavy-hex 16         83        484        0.0e+00
```

**What to look for.** The GHZ table is the easy case and it is still not free: on the $4\times4$ grid the chain costs 3 SWAPs at $n = 5$ and 6 at $n = 12$, entirely because the trivial layout walks logical qubits across a row boundary. On heavy-hex it costs 24 SWAPs at $n = 12$ — for a circuit whose interaction graph is a path, on a connected device that contains long paths. That is a layout failure, not a routing failure, and Code Example 6 recovers all of it.

The QFT table is the honest headline of the chapter. At $n = 12$ the CX count goes from 132 to 354 on the grid and to 606 on heavy-hex: factors of 2.7 and 4.6. The mean-distance estimate predicted 6.0 and 9.05, so it overestimates by about a factor of two — and the reason is the one given in §3.1. The QFT interacts nearby qubits repeatedly and the router leaves them where it moved them, so consecutive gates are cheaper than independent ones. The estimate is the right order of magnitude and it is the one to use when no router is available; it is not a substitute for running one.

The sampling block is the verification. Four circuits, two 16-qubit devices, up to 484 gates in the routed output, and the randomized permutation-aware check passes on every row. A $4^{16}$ matrix would be $4\times10^9$ complex numbers; three random $2^{16}$ state vectors cost nothing and catch the error that matters.

### Code Example 6: What a Layout Is Worth

```python
"""Chapter 3, Example 6: the layout problem, and a cheap heuristic for it.
Continues from Example 5 (same session)."""
from itertools import permutations
from math import factorial
import numpy as np
from qcheck import *


def layout_search(circ, dev):
    """Every layout, scored by the SWAPs the router then needs. Small devices only."""
    scores = [route(circ, dev, p)[3] for p in permutations(range(dev["n"]))]
    return min(scores), max(scores), float(np.mean(scores)), scores


def reverse_traversal(circ, dev, rounds=2):
    """SABRE's initial-layout trick, in six lines.

    Route the circuit forwards from the trivial layout and keep the map it ends
    with; route the reversed gate list from there and keep that map; repeat. Each
    pass moves the starting map towards one that suits the gates the circuit
    actually contains, and no search over layouts is involved.
    """
    loc = list(range(dev["n"]))
    for r in range(2 * rounds):
        gates = circ if r % 2 == 0 else list(reversed(circ))
        loc = route(gates, dev, loc)[2]
    return loc


CASES = [("GHZ n=5", ghz_chain(5), 5), ("QFT n=4", qft(4), 4),
         ("QFT n=5", qft(5), 5)]
print("Exhaustive layout search on two six- and seven-qubit devices")
head = (f"{'circuit':<10}{'device':<14}{'layouts':>9}{'best':>6}{'worst':>7}"
        f"{'mean':>7}{'trivial':>9}{'reverse trav.':>15}")
print(head)
print("-" * len(head))
for label, circ, n_log in CASES:
    for d in (grid(2, 3), heavy_hex_7()):
        best, worst, mean, _ = layout_search(circ, d)
        triv = route(circ, d)[3]
        phys, loc0, locf, rev = route(circ, d, reverse_traversal(circ, d))
        err = routed_error(circ, n_log, d, phys, loc0, locf)
        assert err < 1e-10, label
        print(f"{label:<10}{d['name']:<14}{factorial(d['n']):>9}{best:>6}"
              f"{worst:>7}{mean:>7.2f}{triv:>9}{rev:>15}")

print("\nWhy the search cannot continue: layouts of n logical qubits on N physical")
head = f"{'N':>4}" + "".join(f"{f'n = {n}':>16}" for n in (4, 8, 12, 16))
print(head)
print("-" * len(head))
for N in (7, 16, 27, 65, 127):
    row = ""
    for n in (4, 8, 12, 16):
        row += (f"{factorial(N) // factorial(N - n):>16.3g}" if n <= N
                else f"{'-':>16}")
    print(f"{N:>4}{row}")
print("\nThe heuristic on the 16-qubit devices, where no exhaustive answer exists")
head = (f"{'circuit':<10}{'device':<14}{'trivial':>9}{'reverse trav.':>15}"
        f"{'best of 200 random':>20}{'check':>9}")
print(head)
print("-" * len(head))
for label, circ, n_log in [("QFT n=6", qft(6), 6), ("QFT n=8", qft(8), 8),
                           ("QFT n=10", qft(10), 10),
                           ("GHZ n=12", ghz_chain(12), 12)]:
    for d in (grid(4, 4), heavy_hex_16()):
        triv = route(circ, d)[3]
        lay = reverse_traversal(circ, d)
        rev = route(circ, d, lay)[3]
        rng = np.random.default_rng(17)
        rand = min(route(circ, d, list(rng.permutation(d["n"])))[3]
                   for _ in range(200))
        phys, loc0, locf, _ = route(circ, d, lay)
        err = routed_error_sampled(circ, n_log, d, phys, loc0, locf,
                                   trials=2, seed=9)
        assert err < 1e-9, label
        print(f"{label:<10}{d['name']:<14}{triv:>9}{rev:>15}{rand:>20}"
              f"{err:>9.0e}")
```

```text
Exhaustive layout search on two six- and seven-qubit devices
circuit   device          layouts  best  worst   mean  trivial  reverse trav.
-----------------------------------------------------------------------------
GHZ n=5   grid 2x3            720     0      7   3.04        2              0
GHZ n=5   heavy-hex 7        5040     0     12   6.03        2              0
QFT n=4   grid 2x3            720     2      9   4.97        5              4
QFT n=4   heavy-hex 7        5040     2     15   8.34        3              2
QFT n=5   grid 2x3            720     4     15   8.41        9              8
QFT n=5   heavy-hex 7        5040     5     24  13.98       12             10

Why the search cannot continue: layouts of n logical qubits on N physical
   N           n = 4           n = 8          n = 12          n = 16
--------------------------------------------------------------------
   7             840               -               -               -
  16        4.37e+04        5.19e+08        8.72e+11        2.09e+13
  27        4.21e+05        8.95e+10        8.33e+15        2.73e+20
  65        1.62e+07        2.04e+14        1.93e+21        1.36e+28
 127        2.48e+08         5.4e+16        1.03e+25        1.71e+33

The heuristic on the 16-qubit devices, where no exhaustive answer exists
circuit   device          trivial  reverse trav.  best of 200 random    check
-----------------------------------------------------------------------------
QFT n=6   grid 4x4             17             18                  11    0e+00
QFT n=6   heavy-hex 16         17             15                  20    0e+00
QFT n=8   grid 4x4             24             32                  27    0e+00
QFT n=8   heavy-hex 16         63             55                  35    0e+00
QFT n=10  grid 4x4             59             61                  41    0e+00
QFT n=10  heavy-hex 16         83             99                  77    0e+00
GHZ n=12  grid 4x4              6              0                   9    0e+00
GHZ n=12  heavy-hex 16         24             18                  17    0e+00
```

**What to look for.** On the small devices the whole layout landscape is visible. For the GHZ rows the best layout removes every SWAP and the worst inserts seven or twelve; the mean over all layouts is 3.04 and 6.03, so the trivial layout's 2 is better than average and still not good. For QFT at $n = 5$ on heavy-hex the spread is 5 to 24. The reverse-traversal heuristic beats the trivial layout on all six rows and reaches the true optimum on three of them, at the cost of four routing passes and no search.

The factorial table is why the search stops there. At 127 physical qubits and 16 logical ones there are $2\times10^{33}$ layouts, and the problem is NP-hard, so nothing exact is coming.

The last table is where the honesty is. On the 16-qubit devices the heuristic is sometimes *worse* than the trivial layout — 32 against 24 for QFT at $n = 8$ on the grid — because our version keeps only SABRE's reverse-traversal trick and none of its lookahead cost function, so it optimizes the wrong thing on a long circuit. And two hundred random layouts, which is a real search, beats the heuristic on most rows at the cost of two hundred routing passes instead of four. That trade — search time against SWAP count — is the whole design space of a transpiler, and it is why production tools expose an optimization level rather than a single algorithm.

### Code Example 7: What the Simple Router Loses

The only way to know what a heuristic costs is to compute the optimum where the optimum is computable. Breadth-first search over (device map, number of two-qubit gates already executed) does exactly that: every SWAP costs one, so plain BFS returns the minimum, and seeding the search with every possible starting map solves the layout problem exactly at the same time. It is factorial in the number of physical qubits and therefore a measuring instrument rather than an alternative.

```python
"""Chapter 3, Example 7: what the simple router loses, measured against optimal.
Continues from Example 6 (same session)."""
from collections import deque
from itertools import permutations
import numpy as np
from qcheck import *


def optimal_swaps(circ, dev, layout=None):
    """The minimum SWAP count, by breadth-first search over device mappings.

    The state is (map, number of two-qubit gates already executed). Single-qubit
    gates never need routing, so only the two-qubit gates enter, and the written
    order is preserved -- exactly the model route() uses, so the comparison is
    fair. Every SWAP costs one, so plain breadth-first search returns the
    optimum. With layout=None every starting map is free, which additionally
    solves the layout problem exactly.
    """
    pairs = [gate_qubits(g) for g in circ if len(gate_qubits(g)) == 2]
    n = dev["n"]

    def advance(loc, k):
        while k < len(pairs):
            a, b = pairs[k]
            if loc[b] in dev["adj"][loc[a]]:
                k += 1
            else:
                return k
        return k

    starts = ([tuple(layout)] if layout is not None
              else [p for p in permutations(range(n))])
    queue, seen = deque(), set()
    for st in starts:
        s = (st, advance(st, 0))
        if s not in seen:
            seen.add(s)
            queue.append((s, 0))
    while queue:
        (loc, k), cost = queue.popleft()
        if k == len(pairs):
            return cost, loc
        for p, q in dev["edges"]:
            nl = list(loc)
            u, v = nl.index(p), nl.index(q)
            nl[u], nl[v] = q, p
            nxt = (tuple(nl), advance(tuple(nl), k))
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, cost + 1))
    raise RuntimeError("unreachable on a connected device")


print("The simple router against an exact one, on cases small enough for both")
head = (f"{'circuit':<10}{'device':<14}{'ours, trivial':>14}"
        f"{'optimal, trivial':>18}{'ours, rev trav.':>17}"
        f"{'optimal, free layout':>22}")
print(head)
print("-" * len(head))
rows = []
for label, circ, n_log in [("GHZ n=4", ghz_chain(4), 4),
                           ("GHZ n=5", ghz_chain(5), 5),
                           ("QFT n=3", qft(3), 3),
                           ("QFT n=4", qft(4), 4),
                           ("QFT n=5", qft(5), 5)]:
    for d in (grid(2, 3), heavy_hex_7()):
        ours = route(circ, d)[3]
        opt_fixed = optimal_swaps(circ, d, list(range(d["n"])))[0]
        rev = route(circ, d, reverse_traversal(circ, d))[3]
        opt_free, best_loc = optimal_swaps(circ, d)
        phys, loc0, locf, _ = route(circ, d, best_loc)
        assert routed_error(circ, n_log, d, phys, loc0, locf) < 1e-10
        rows.append((ours, opt_fixed, rev, opt_free))
        print(f"{label:<10}{d['name']:<14}{ours:>14}{opt_fixed:>18}"
              f"{rev:>17}{opt_free:>22}")

ours = np.array([r[0] for r in rows], float)
opt_fixed = np.array([r[1] for r in rows], float)
rev = np.array([r[2] for r in rows], float)
opt_free = np.array([r[3] for r in rows], float)
print(f"\nTotals over the ten rows: ours {ours.sum():.0f}, "
      f"optimal at the same layout {opt_fixed.sum():.0f}, "
      f"ours with reverse traversal {rev.sum():.0f}, "
      f"optimal over layouts {opt_free.sum():.0f}")
print(f"  excess over optimal, same layout      : "
      f"{100 * (ours.sum() / opt_fixed.sum() - 1):.0f}%")
print(f"  excess over optimal, layout included  : "
      f"{100 * (ours.sum() / opt_free.sum() - 1):.0f}%")
print(f"  with the layout heuristic in front    : "
      f"{100 * (rev.sum() / opt_free.sum() - 1):.0f}%")

print("\nThe exact router is not an alternative, it is a measuring instrument:")
from math import factorial
head = f"{'device':<14}{'maps':>10}{'states for QFT n=5':>20}"
print(head)
print("-" * len(head))
pairs = sum(1 for g in qft(5) if len(gate_qubits(g)) == 2)
for N, name in ((6, "grid 2x3"), (7, "heavy-hex 7"), (16, "grid 4x4"),
                (27, "heavy-hex 27")):
    print(f"{name:<14}{factorial(N):>10.3g}{factorial(N) * (pairs + 1):>20.3g}")
```

```text
The simple router against an exact one, on cases small enough for both
circuit   device         ours, trivial  optimal, trivial  ours, rev trav.  optimal, free layout
-----------------------------------------------------------------------------------------------
GHZ n=4   grid 2x3                   2                 2                0                     0
GHZ n=4   heavy-hex 7                1                 1                2                     0
GHZ n=5   grid 2x3                   2                 2                0                     0
GHZ n=5   heavy-hex 7                2                 2                0                     0
QFT n=3   grid 2x3                   2                 1                1                     1
QFT n=3   heavy-hex 7                2                 1                1                     1
QFT n=4   grid 2x3                   5                 3                4                     2
QFT n=4   heavy-hex 7                3                 3                2                     2
QFT n=5   grid 2x3                   9                 5                8                     4
QFT n=5   heavy-hex 7               12                 7               10                     4

Totals over the ten rows: ours 40, optimal at the same layout 27, ours with reverse traversal 28, optimal over layouts 14
  excess over optimal, same layout      : 48%
  excess over optimal, layout included  : 186%
  with the layout heuristic in front    : 100%

The exact router is not an alternative, it is a measuring instrument:
device              maps  states for QFT n=5
--------------------------------------------
grid 2x3             720            1.51e+04
heavy-hex 7     5.04e+03            1.06e+05
grid 4x4        2.09e+13            4.39e+14
heavy-hex 27    1.09e+28            2.29e+29
```

**What to look for.** Over the ten rows our router inserts 40 SWAPs against 27 for an exact router at the same layout, and 14 for an exact router that also chooses the layout: an excess of 48% at fixed layout and 186% when layout is included. Putting the reverse-traversal heuristic in front reduces the excess to 100%. Those numbers are what the chapter's router costs, stated plainly.

Where does the excess come from? On the GHZ rows, nowhere: the interaction graph is a path, walking one qubit along it is already optimal, and our router matches the exact one at 2, 1, 2, 2. The entire excess is in the QFT rows, and four causes are separable.

  * **Gate order.** We execute the gates as written. A real router works on the dependency graph and may execute any gate in the front layer, which often removes a SWAP outright. This is the largest of the four.
  * **Lookahead.** We move a qubit for the gate in front of us and never ask what the next gate wants.
  * **Both endpoints.** We walk one qubit the whole way. Walking both halves the depth, though not the SWAP count.
  * **Layout.** Ours is trivial unless told otherwise, and the table shows that is worth a factor of order two on its own.

There is a fifth, and it is a difference in objective rather than in quality. We minimize SWAP count. What matters on hardware is the total error, which depends on depth as well as on gate count, and on which physical qubits were used — a router that avoids a bad qubit at the cost of two SWAPs may well be right. Also worth noting: the exact router shares our restriction to the written gate order, so its 27 is the optimum *of our model*, not the true optimum. A front-layer router can beat it.

Finally, the state-count table. The search visits one state per (map, progress) pair, which is $N!$ times the number of two-qubit gates: 15 thousand for the $2\times3$ grid, $4\times10^{14}$ for a $4\times4$ grid, $2\times10^{29}$ for 27 qubits. Everything beyond a toy device is heuristic, and the only honest way to report a router is the way this table does — against the optimum where the optimum is computable, and against another heuristic where it is not.

* * *

## Exercises

#### Exercise 1: Mean Distance by Hand

Consider a ring of six qubits, coupled $0{-}1{-}2{-}3{-}4{-}5{-}0$.

  1. Write the all-pairs distance table and compute the mean distance $\bar{d}$.
  2. Estimate the extra CX gates per two-qubit gate from $3(\bar{d}-1)$, and compare with the $2\times3$ grid of Code Example 1.
  3. The ring has six edges and the grid has seven. Which has the smaller mean distance, and does the edge count alone predict it?
  4. Now break one edge of the ring to make a line of six. By what factor does $\bar{d}$ change?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> On a 6-ring the distance depends only on the separation \(k = |i-j| \bmod 6\): \(d = 1\) for \(k \in \lbrace 1, 5\rbrace\), \(d = 2\) for \(k \in \lbrace 2,4 \rbrace\), \(d = 3\) for \(k = 3\). Among the 15 unordered pairs there are 6 at distance 1, 6 at distance 2 and 3 at distance 3, so \(\bar{d} = (6 + 12 + 9)/15 = 1.80\).</p>

<p><strong>2.</strong> \(3(1.80 - 1) = 2.40\) extra CX gates per two-qubit gate. The \(2\times3\) grid has \(\bar{d} = 1.67\) and therefore \(2.00\). The grid is cheaper.</p>

<p><strong>3.</strong> The grid, with \(\bar{d} = 1.67\) against \(1.80\), and it also has one more edge — so here the edge count does predict the ordering. It does not in general: the 16-qubit heavy-hex fragment has 16 edges and \(\bar{d} = 3.68\), while a 16-qubit line has 15 edges and \(\bar{d} = 5.67\), a much larger gap than one edge explains. What matters is how the edges are arranged, and the diameter is often the better single summary.</p>

<p><strong>4.</strong> On a line of six the 15 pairs have distances \(1,1,1,1,1\) (5 pairs), \(2\) (4), \(3\) (3), \(4\) (2), \(5\) (1), giving \(\bar{d} = (5 + 8 + 9 + 8 + 5)/15 = 2.33\). Removing one edge from the ring raises \(\bar{d}\) by a factor \(2.33/1.80 = 1.30\), and raises the estimated overhead from 2.40 to 4.00 — a 67% increase in routing cost from deleting a single coupler. Connectivity degrades non-linearly, which is why a dead coupler on a real device is reported and routed around rather than ignored.</p>

</details>

#### Exercise 2: A Zero-SWAP Layout

Code Example 5 found that the GHZ chain on 12 qubits costs 6 SWAPs on the $4\times4$ grid with the trivial layout, and 0 with the reverse-traversal layout.

  1. Explain in one sentence why the trivial layout fails.
  2. Write down a layout of 12 logical qubits on the $4\times4$ grid that makes the GHZ chain entirely native, and verify by hand that each consecutive pair is an edge.
  3. Does such a layout exist for the GHZ chain on 16 logical qubits on the same grid? On the 16-qubit heavy-hex fragment?
  4. Generalize: what property of the coupling graph guarantees that an $n$-qubit GHZ chain can be laid out with zero SWAPs?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> The trivial layout numbers the grid row by row, so logical qubits 3 and 4 land on physical 3 and 4, which are at opposite ends of adjacent rows and not coupled — and the same happens at every row boundary.</p>

<p><strong>2.</strong> A boustrophedon (snake) path: physical \(0{,}1{,}2{,}3{,}7{,}6{,}5{,}4{,}8{,}9{,}10{,}11\) for logical \(0\) through \(11\). Consecutive pairs are \((0,1),(1,2),(2,3)\) along the first row, \((3,7)\) down a column, \((7,6),(6,5),(5,4)\) back along the second row, \((4,8)\) down, then \((8,9),(9,10),(10,11)\) — every one an edge of the grid.</p>

<p><strong>3.</strong> Yes for the grid: extend the snake through the fourth row, \(\ldots,11,15,14,13,12\). No for the heavy-hex fragment as defined, because it has 16 nodes and 16 edges with a single cycle of length 12 and four pendant qubits; a Hamiltonian path would have to enter and leave each pendant qubit, which its degree of 1 forbids. The best possible there is a path of 13 nodes (the 12-cycle plus one pendant), so three logical qubits must be routed.</p>

<p><strong>4.</strong> The graph must contain a path on \(n\) vertices — a Hamiltonian path when \(n = N\). That is again an NP-complete question in general, which is the layout problem's difficulty appearing in its purest form even for the easiest possible circuit.</p>

</details>

#### Exercise 3: The Permutation Bookkeeping

A router starts with the trivial layout on four physical qubits and inserts, in order, $\mathrm{SWAP}_{0,1}$, then $\mathrm{SWAP}_{1,2}$, then $\mathrm{SWAP}_{0,1}$.

  1. Track $\ell$ after each SWAP, where $\ell[v]$ is the physical line of virtual qubit $v$.
  2. What is $\ell_f$, and what permutation of the *output* wires does it correspond to?
  3. A colleague's router appends the inverse of every SWAP at the end of the circuit so that $\ell_f = \ell_0$ always. What does that cost, and what does it buy?
  4. Under what circumstance is your colleague right?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Start \(\ell = [0,1,2,3]\). \(\mathrm{SWAP}_{0,1}\) exchanges the virtual qubits on physical lines 0 and 1, giving \(\ell = [1,0,2,3]\). \(\mathrm{SWAP}_{1,2}\) exchanges those on lines 1 and 2: virtual 0 is on line 1 and virtual 2 is on line 2, so \(\ell = [2,0,1,3]\). \(\mathrm{SWAP}_{0,1}\) exchanges lines 0 and 1, where virtual 1 and virtual 0 sit, giving \(\ell = [0,2,1,3]\).</p>

<p><strong>2.</strong> \(\ell_f = [0,2,1,3]\): logical 0 on physical 0, logical 1 on physical 2, logical 2 on physical 1, logical 3 on physical 3. Three SWAPs produced a single transposition, which is a sign that a lookahead router would have found a shorter sequence. Reading out logical qubit 1 means measuring physical line 2 — the permutation is a relabelling of the classical measurement record, and it costs nothing at run time.</p>

<p><strong>3.</strong> It costs up to a doubling of the SWAP count, and since each SWAP is three CX gates on the noisiest gate type available, that is the most expensive possible way to buy convenience. What it buys is that Chapter 1's equivalence check applies unmodified, which is exactly why Chapter 1's Code Example 7 did it and said so.</p>

<p><strong>4.</strong> When the permutation cannot be absorbed into classical post-processing — for instance when the circuit is a subroutine whose output wires are fixed by a caller that will not be told about the relabelling, or when a mid-circuit measurement is fed back to a specific physical line, or when the same physical qubits must be reused with a known assignment in a later block. In those cases the permutation must be realized physically. Everywhere else, tracking it is free and undoing it is waste.</p>

</details>

#### Exercise 4: Front-Layer Freedom

Consider the three-gate circuit $\mathrm{CX}_{0,2}$, $\mathrm{CX}_{1,3}$, $\mathrm{CX}_{0,1}$ on the line $0{-}1{-}2{-}3$ with the trivial layout.

  1. Route it with the strategy of Code Example 3 — gates in the written order — and count the SWAPs.
  2. Now allow the gates to be reordered subject only to their dependencies. Which orders are legal, and what does the router give for the other one?
  3. Find the minimum SWAP count at this layout, and the minimum if the layout is also free.
  4. What does this say about the 48% excess measured in Code Example 7?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Four. \(\mathrm{CX}_{0,2}\) is at distance 2, so one SWAP on lines 0 and 1 leaves \(\ell = [1,0,2,3]\) and the gate executes. \(\mathrm{CX}_{1,3}\) then needs logical 1, now on line 0, next to logical 3 on line 3: two more SWAPs, ending at \(\ell = [0,2,1,3]\). \(\mathrm{CX}_{0,1}\) is then at distance 2 and needs one more. Running <code>route</code> returns 4.</p>

<p><strong>2.</strong> The third gate shares qubit 0 with the first and qubit 1 with the second, so it must come last; the first two are independent. The two legal orders are \((0{,}2)(1{,}3)(0{,}1)\) and \((1{,}3)(0{,}2)(0{,}1)\), and the router gives 4 on the first and <strong>2</strong> on the second. Reordering alone halves the cost, with no change to the routing algorithm at all.</p>

<p><strong>3.</strong> Two at the trivial layout, which <code>optimal_swaps</code> confirms: swap lines 1 and 2 first, giving \(\ell = [0,2,1,3]\), which puts logical 0 next to logical 2 <em>and</em> logical 1 next to logical 3 simultaneously, so both gates execute; one more SWAP then brings logical 0 and 1 together. One SWAP serving two gates is exactly what a lookahead cost function is designed to find and what a per-gate shortest path cannot. With the layout free the answer is <strong>zero</strong>: the interaction graph is the path \(2{-}0{-}1{-}3\), which is a path on four vertices and therefore fits the line exactly, under the layout \(\ell_0 = [1,2,0,3]\).</p>

<p><strong>4.</strong> That the excess is structural rather than a matter of tuning, and that it splits the same way Code Example 7 found. Gate order alone took 4 to 2 here; layout alone took 2 to 0. Our router commits to a shortest path for the gate in front of it, so a SWAP that would serve two gates at once is invisible to it, and reordering and lookahead attack that blind spot from two directions — which is why SABRE has both, and why the measured excess concentrated entirely in the QFT rows, the QFT being dense in exactly the pairs where one SWAP can serve several gates.</p>

</details>

#### Exercise 5: Reading a Transpiler Report

A transpiler reports, for the same 20-qubit circuit with 100 two-qubit gates:

| Target | CX after | Depth after | SWAPs |
| --- | --- | --- | --- |
| all-to-all | 100 | 41 | 0 |
| grid $5\times4$ | 361 | 158 | 87 |
| heavy-hex, 20 qubits | 634 | 265 | 178 |

  1. Check the SWAP counts against the CX counts. Are they consistent?
  2. Estimate the mean distance of each graph implied by the overhead, using $3(\bar{d}-1)$, and say whether the estimate is plausible.
  3. The heavy-hex row has 1.8 times the CX count of the grid row. How much of that is the graph and how much could be the layout?
  4. Which single number would you ask for next, and why?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Yes. Each SWAP is three CX gates, so \(100 + 3\times87 = 361\) and \(100 + 3\times178 = 634\). If the arithmetic had not closed, the transpiler would be emitting native SWAPs or fusing gates after routing, and either would have to be known before the numbers could be read.</p>

<p><strong>2.</strong> Grid: overhead \(261/100 = 2.61\) extra CX per gate, so \(\bar{d} \approx 1 + 2.61/3 = 1.87\). Heavy-hex: \(534/100 = 5.34\), so \(\bar{d} \approx 2.78\). Both are <em>smaller</em> than the true mean distances: a \(5\times4\) grid has \(\bar{d} = 3.00\), and a 20-qubit heavy-hex graph is sparser still, since the 16-qubit fragment of Code Example 1 already has \(3.68\). Smaller is the expected direction, because \(3(\bar{d}-1)\) is an upper bound: a router leaves qubits where it moved them, and Code Example 5 measured the same factor-of-two gap. An implied \(\bar{d}\) <em>larger</em> than the true one would indicate a router doing something wrong.</p>

<p><strong>3.</strong> The graph accounts for most of it — heavy-hex's mean distance is roughly 1.4 times the grid's at this size, and routing cost grows slightly faster than mean distance because the harder graph also gives the router fewer choices. The remainder, perhaps a factor of 1.2 to 1.3, could be layout: Code Example 6 found the heuristic performing worse on heavy-hex than on the grid on several rows. The way to separate them is to re-run with several layouts and report the spread.</p>

<p><strong>4.</strong> The spread over layouts, or equivalently the SWAP count for the best of \(k\) random layouts. A single routing pass reports one point from a distribution whose width Code Example 6 showed to be a factor of two or more, and without the spread there is no way to tell a good graph from a lucky layout. A close second would be the two-qubit gate error rate per edge, since a route through a bad coupler can cost more than three extra SWAPs would.</p>

</details>

* * *

## Summary

### Key Takeaways

**1\. A coupling graph is a gate mechanism written as data**

  * All-to-all comes from a shared bosonic mode; a grid from planar capacitive coupling; heavy-hex from capping the degree at three to control frequency collisions and crosstalk, at the cost of a sparser graph.
  * On sixteen qubits the edge count runs from 120 to 15 and the mean distance from 1.00 to 5.67; heavy-hex has two-thirds of the grid's edges and a 38% larger mean distance.

**2\. Mean distance predicts the overhead, as an upper bound**

  * $3(\bar{d}-1)$ extra CX gates per two-qubit gate: 0 for all-to-all, 5.00 for a $4\times4$ grid, 8.05 for heavy-hex, 14.00 for a 16-qubit line.
  * Measured overheads are about half of that, because a router leaves qubits where it moved them and consecutive gates are cheaper than independent ones. Use the estimate when no router is available and never instead of one.

**3\. The demand side is the circuit's interaction graph**

  * The GHZ chain is a path and is native under some layout on any connected device; the QFT is the complete graph and no layout helps.
  * Routing the 12-qubit QFT multiplies its CX count by 2.7 on the grid and 4.6 on heavy-hex. Chapter 2's optimizer removed 26% of the gates in a circuit; connectivity multiplies them by up to five.

**4\. Layout is NP-hard and a good heuristic is nearly as valuable as an oracle**

  * The exact zero-cost case is subgraph isomorphism, and there are $N!/(N-n)!$ layouts — $2\times10^{33}$ for 16 logical qubits on 127 physical ones.
  * On devices small enough for exhaustive search, the spread between the best and worst layout is a factor of two to four; SABRE's reverse-traversal trick beat the trivial layout on all six rows tested and hit the optimum on three, with four routing passes and no search.
  * Our stripped-down version of it is sometimes worse than the trivial layout on 16-qubit devices, because it keeps the reverse traversal and drops the lookahead cost — an honest measurement of what half an algorithm is worth.

**5\. A routed circuit is not the circuit that was written, and the check must say so**

  * The correct statement is $U_{\mathrm{phys}}P(\ell_0) = P(\ell_f)(U_{\mathrm{log}}\otimes I)$ up to a phase, and it passed at exactly zero error on every routed circuit tested.
  * Forgetting the permutation gives an error of 0.35 to 0.71 whenever a SWAP was inserted and exactly zero when none was — so a permutation-blind test suite passes on all-to-all and fails silently on real hardware.
  * On 16-qubit devices a matrix is $4\times10^9$ numbers and a state vector is 65536, so the check runs on random inputs; that is not a proof, but a mishandled permutation is an $O(1)$ error on almost every input.

**6\. A heuristic router should be reported against the optimum where the optimum exists**

  * Over ten small cases: 40 SWAPs for our router against 27 for an exact router at the same layout and 14 for an exact router that also chooses the layout — 48% and 186% excess, reduced to 100% by the layout heuristic.
  * All of the excess is in the QFT rows; on the GHZ rows our router is already optimal. The causes are gate order, lookahead, single-endpoint walking, and layout, in roughly that order of importance.
  * Exact routing visits $N!$ times the gate count states — $1.5\times10^4$ at six qubits, $2\times10^{29}$ at twenty-seven. It is a measuring instrument, not an alternative.

**Practical implications**

  * Never quote a routing overhead without naming the graph, the circuit's interaction structure, and the layout method; the three are multiplicative and the third is worth a factor of two.
  * Build the permutation-aware equivalence check before the router, and make sure the test suite contains a sparse device — a permutation bug is invisible on all-to-all.
  * When a transpiled circuit is unexpectedly large, check whether $\mathrm{CX}_{\text{after}} = \mathrm{CX}_{\text{before}} + 3\times\mathrm{SWAPs}$ closes. If it does, the growth is routing and the fix is layout; if it does not, the growth is synthesis and the fix is in Chapter 2.

### Where This Leads

Chapters 2 and 3 have compiled a circuit down to a gate list that a specific machine can execute, and treated every gate in it as an abstract unitary that takes one unit of time. Chapter 4 opens that abstraction. A rotation gate is a resonant drive, its duration and shape are the compiler's output at the lowest layer, and the reason a pulse has to be shaped at all is that a superconducting qubit is not a two-level system — a pulse that is too fast populates a third level, and leakage is not describable as a qubit error. That chapter builds a three-level pulse simulator, measures the leakage, suppresses it with DRAG, and then implements the calibration loops that a control stack runs against its own machine: Rabi amplitude, Ramsey frequency, and DRAG coefficient, each recovering a parameter that was deliberately mis-set. It is the layer where software stops being software and becomes an experiment.

[← Chapter 2: Circuit Optimization and Gate Synthesis](<chapter-2.html>) [Chapter 4: Pulses and Calibration →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The coupling graphs, SWAP counts and overhead factors in this chapter are measurements of the specific graphs, circuits and layouts defined in the code, not benchmarks of any device or transpiler, and the transpiler report in Exercise 5 is a constructed teaching example rather than data from any machine. Verify against primary sources before using them in a proposal or a paper.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
