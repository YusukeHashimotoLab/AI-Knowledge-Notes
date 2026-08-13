---
title: "Chapter 5: Error Mitigation as Software, and Resource Estimation"
chapter_title: "Chapter 5: Error Mitigation as Software, and Resource Estimation"
subtitle: Readout Correction, Zero-Noise Extrapolation by Gate Folding, Probabilistic Error Cancellation, the Exponential Wall They All Run Into, and the Pipeline That Prices the Alternative
reading_time: 45-50 minutes
difficulty: Advanced
code_examples: 8
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-software-stack-introduction/chapter-5.html>) | Last sync: 2026-08-13

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to the Quantum Software Stack](<index.html>) > Chapter 5

The four chapters before this one built a stack that assumed the hardware works. An optimizer that removes gates, a router that adds SWAPs, a calibration loop that pushes a single-qubit gate error to $5 \times 10^{-6}$ — all of it is an attempt to make a circuit *short* and *accurate*, and all of it stops helping once the circuit is longer than the error rate allows. At $10^{-3}$ per two-qubit gate, a thousand-gate circuit has already lost. This chapter is about the layer that sits on top of that fact and does something about it anyway.

There are exactly two things software can do. It can **mitigate**: leave the noise alone, run modified circuits, and post-process the results so that the *expectation value* comes out closer to the noiseless one. It can also **estimate**: work out what the noise-free alternative would cost, in physical qubits and wall-clock time, so that the mitigation question can be answered with numbers rather than adjectives. Sections 5.1 to 5.3 implement three mitigation methods and measure both what each one buys and what each one costs. Section 5.4 states the boundary as plainly as it can be stated: mitigation costs grow exponentially in the size of the circuit and error correction costs grow polynomially, so there is a circuit size above which no amount of post-processing helps, and Section 5.5 computes where it is. Both halves of that sentence get numbers, because error mitigation is genuinely load-bearing on the hardware that exists *and* exponentially expensive, and dropping either half of that produces a wrong picture of the field.

## Learning Objectives

After completing this chapter, you will be able to:

  * Construct a readout confusion matrix from calibration circuits, correct a measured distribution by inverting it, and explain why the inverse produces negative probabilities and what a constrained least-squares fit does instead
  * State the cost of exact readout mitigation ($2^n$ calibration circuits), implement the tensored approximation that reduces it to $2n$, and demonstrate a case where the tensored model is *worse than no correction at all*
  * Implement gate folding, prove with the Chapter 1 equivalence checker that it leaves the unitary unchanged, and use it to amplify noise by a known factor
  * Extrapolate a folded family of noisy expectation values to zero noise by Richardson interpolation and by a least-squares line, and measure the bias each one leaves
  * Measure the variance cost of extrapolation, verify it against $\lVert w \rVert^2$, and identify the shot budget at which the higher-order estimator starts winning on total error
  * Derive the quasiprobability inverse of a depolarizing channel, implement probabilistic error cancellation as a sampled estimator, and show numerically that its sampling overhead is exponential in the circuit size
  * Implement a resource-estimation pipeline from algorithm inputs to physical qubits and wall-clock time, including the relative-tolerance guard the power-of-ten threshold comparison requires, and locate the circuit size at which correction becomes cheaper than mitigation

### What Carries Over

Everything here runs on the circuit IR of [Chapter 1](<chapter-1.html>) and the state-vector simulator underneath it, both re-listed in Example 1 together with the equivalence checker — because a noise-amplification pass is a rewriting pass, and every rewriting pass in this course has to prove it preserved the meaning. Two functions from the introductory course's simulator that Chapter 1 did not need appear here as well, `PAULI` and `expval`, re-listed verbatim from the same source.

Three numerical anchors come from the sister courses and this chapter stays consistent with all of them. [Introduction to Quantum Computing, Chapter 5](<../quantum-computing-introduction/chapter-5.html>) gives the surface-code scaling $p_L \approx A(p/p_{\text{th}})^{(d+1)/2}$ with $A = 0.1$ and $p_{\text{th}} = 10^{-2}$, and a table of code distances and qubit overheads that Example 8 reproduces line for line. It also gives the Richardson weight norms $\lVert w \rVert_2 = 2.24, 4.36, 8.31$ for two, three and four noise scales, which Example 5 recomputes. [Intermediate Quantum Algorithms, Chapter 4](<../quantum-algorithms-intermediate/chapter-4.html>) gives the qubitized phase-estimation Toffoli counts for a FeMoco-scale calculation and the conversion to days and physical qubits; Example 8 reproduces those too. One convention differs between the two sisters by exactly one qubit — the rotated surface code is quoted as $2d^2 - 1$ physical qubits per logical qubit in the first and $2d^2$ in the second — and the code here uses $2d^2 - 1$ and says so.

One difference from the sister course's treatment of zero-noise extrapolation is deliberate. There, the noise strength was scaled by turning up the error probability directly, which is what a simulator can do and hardware cannot. Here it is scaled by **folding gates**, which is what hardware actually does, and the difference turns out to matter: folding reaches only odd integer scales unless you work for fractional ones, and it is blind to one whole class of error.

* * *

## 5.1 Readout-Error Mitigation

### The one error that is classical

Of all the error channels in a quantum computer, exactly one is a classical, stationary, measurable stochastic process: the assignment error of the final measurement. A qubit in $\lvert 0 \rangle$ is reported as 1 with some probability $\epsilon_{01}$, a qubit in $\lvert 1 \rangle$ is reported as 0 with probability $\epsilon_{10}$, and on current superconducting hardware those numbers are between one and five percent — one to two orders of magnitude larger than the gate errors of Chapter 4. Because it is classical, it can be inverted exactly, and because it is the largest single error in many experiments, inverting it is the highest-value thing in this chapter.

Write the measured distribution over the $2^n$ bitstrings as a vector $\mathbf{p}_{\text{meas}}$ and the distribution the circuit actually produced as $\mathbf{p}_{\text{true}}$. Assignment error is a stochastic matrix acting between them:

$$ \mathbf{p}_{\text{meas}} = M\,\mathbf{p}_{\text{true}}, \qquad M_{mt} = P(\text{read } m \mid \text{prepared } t) $$

$M$ is the **confusion matrix**, its columns sum to 1, and it is measurable: prepare each of the $2^n$ basis states, measure, and each experiment gives you one column. Then $\mathbf{p}_{\text{true}} = M^{-1}\mathbf{p}_{\text{meas}}$, and the whole method is one line of linear algebra. Two things go wrong with that line, and both matter.

### Code Example 1: The Simulator, the IR and the Checker, Re-listed

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


def probs(state):
    """Born-rule probabilities of all 2^n outcomes."""
    return np.abs(state) ** 2


PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}


def expval(state, pauli, coeff_map=None):
    """Expectation value of a Pauli string such as 'ZZ', 'XI' (one character per qubit).

    If coeff_map is given, the result is multiplied by coeff_map[pauli], so that a
    whole Hamiltonian is one line:  sum(expval(psi, p, terms) for p in terms).
    """
    n = len(pauli)
    phi = state.copy()
    for q, ch in enumerate(pauli):
        if ch != 'I':
            phi = apply_gate(phi, PAULI[ch], [q], n)
    val = np.vdot(state, phi).real
    if coeff_map is not None:
        val *= coeff_map.get(pauli, 1.0)
    return val


# ---- the circuit IR of Chapter 1, re-listed (qir.py) --------------------
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


# ---- the equivalence checker of Chapter 1, re-listed --------------------
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


print("The simulator, the IR and the checker, re-listed and validated")
print("=" * 74)
bell = [("h", 0), ("cx", 0, 1)]
ghz = [("h", 0)] + [("cx", i, i + 1) for i in range(3)]
print(f"  Bell amplitudes      : {np.round(run_circuit(bell, 2).real, 6)}")
print(f"  GHZ(4) probabilities : "
      f"{np.round(probs(run_circuit(ghz, 4))[[0, 15]], 6)} at |0000>, |1111>")
print(f"  GHZ(4) <ZZZZ>        : {expval(run_circuit(ghz, 4), 'ZZZZ'):.6f}")
print(f"  GHZ(4) gate counts   : {gate_counts(ghz)}")
print(f"  GHZ(4) depth         : {circuit_depth(ghz, 4)}")
print(f"  checker on itself    : {assert_equivalent(bell, bell, 2):.2e}")
print(f"  checker on a wrong circuit: "
      f"{phase_free_error(unitary_of(bell, 2), unitary_of([('h', 0), ('h', 1)], 2)):.3f}")
```

```text
The simulator, the IR and the checker, re-listed and validated
==========================================================================
  Bell amplitudes      : [0.707107 0.       0.       0.707107]
  GHZ(4) probabilities : [0.5 0.5] at |0000>, |1111>
  GHZ(4) <ZZZZ>        : 1.000000
  GHZ(4) gate counts   : {'h': 1, 'cx': 3, '2q': 3}
  GHZ(4) depth         : 4
  checker on itself    : 0.00e+00
  checker on a wrong circuit: 1.207
```

**What to notice.** Nothing here is new: the simulator functions are the introductory course's, verbatim; `run_circuit`, `circuit_depth` and `gate_counts` are Chapter 1's IR, verbatim; `unitary_of` through `assert_equivalent` are Chapter 1's equivalence checker, verbatim. `assert_equivalent` returns the phase-free error rather than merely raising, which is what lets it be printed in a table — and it is the function that makes the noise amplification of Section 5.2 auditable rather than hopeful.

### Code Example 2: The Confusion Matrix, and Why Its Inverse Is Not Enough

```python
"""Chapter 5, Example 2: the confusion matrix, its inverse, and why the inverse
is not enough. Continues from Example 1 (same session)."""
from functools import reduce

# per-qubit (eps01, eps10) = (P(read 1 | 0), P(read 0 | 1)); the correction
# routines below never read this tuple, they only see calibration counts
RATES = [(0.030, 0.060), (0.020, 0.050), (0.050, 0.090)]


def confusion_matrix(rates):
    """M[m, t] = P(read m | prepared t) for independent per-qubit readout error.

    Big-endian, so qubit 0 is the leftmost factor of the Kronecker product.
    """
    return reduce(np.kron, [np.array([[1 - e01, e10], [e01, 1 - e10]])
                            for e01, e10 in rates])


def calibrate(rates, shots, rng):
    """The 2^n calibration experiments: prepare each basis state, count outcomes.

    This is the whole cost of readout mitigation and the whole reason it does
    not scale: one circuit per basis state.
    """
    M_true = confusion_matrix(rates)
    return np.column_stack([rng.multinomial(shots, M_true[:, t]) / shots
                            for t in range(M_true.shape[1])])


def project_simplex(v):
    """Euclidean projection of v onto the probability simplex."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.nonzero(u * np.arange(1, v.size + 1) > css - 1.0)[0][-1]
    return np.maximum(v - (css[k] - 1.0) / (k + 1.0), 0.0)


def ls_simplex(M, p, iters=4000):
    """argmin ||M x - p||^2 over the probability simplex, by projected gradient."""
    x = np.full(p.size, 1.0 / p.size)
    step = 1.0 / np.linalg.norm(M, 2) ** 2
    for _ in range(iters):
        x = project_simplex(x - step * (M.T @ (M @ x - p)))
    return x


def tvd(p, q):
    """Total variation distance."""
    return 0.5 * float(np.sum(np.abs(p - q)))


def z_expval(p, mask, n):
    """Expectation of a product of Z operators, read off a measured distribution.

    mask selects the qubits; big-endian, so qubit q is bit n - 1 - q.
    """
    return float(sum(p[m] * (-1) ** bin(m & mask).count("1")
                     for m in range(2 ** n)))


N = 3
rng = np.random.default_rng(20260813)
circ = [("h", 0), ("cx", 0, 1), ("cx", 1, 2)]
p_true = probs(run_circuit(circ, N))
M_true = confusion_matrix(RATES)
M_cal = calibrate(RATES, 20000, rng)

print("A. The confusion matrix")
print("=" * 74)
print(f"  qubits = {N}, so the matrix is {2**N} x {2**N} and calibration costs"
      f" {2**N} circuits")
print(f"  per-qubit rates (hidden): {RATES}")
print(f"  M[0,0] = {M_true[0, 0]:.6f}   (all three qubits read correctly"
      " from |000>)")
print(f"  M[7,7] = {M_true[7, 7]:.6f}   (all three read correctly from |111>)")
print(f"  column sums = {np.round(M_true.sum(axis=0), 12)}")
print(f"  condition number of M     : {np.linalg.cond(M_true):.4f}")
print(f"  calibrated at 20000 shots : max |M_cal - M_true| ="
      f" {np.max(np.abs(M_cal - M_true)):.5f}")

print("\nB. Correcting a GHZ measurement")
print("=" * 74)
SHOTS = 8000
p_meas = rng.multinomial(SHOTS, M_true @ p_true) / SHOTS
p_inv = np.linalg.solve(M_cal, p_meas)
p_ls = ls_simplex(M_cal, p_meas)
print(f"  {'state':>7} {'true':>10} {'measured':>10} {'M^-1':>11} {'LS':>10}")
for m in range(2 ** N):
    print(f"  {format(m, '03b'):>7} {p_true[m]:10.4f} {p_meas[m]:10.4f}"
          f" {p_inv[m]:+11.4f} {p_ls[m]:10.4f}")
MASK01 = 0b110                        # Z on qubits 0 and 1; true value +1
print(f"\n  {'':>18} {'TVD to truth':>13} {'<Z0 Z1>':>9} {'error':>10}")
for name, p in (("measured", p_meas), ("M^-1 corrected", p_inv),
                ("LS corrected", p_ls)):
    print(f"  {name:>18} {tvd(p, p_true):13.5f}"
          f" {z_expval(p, MASK01, N):9.4f}"
          f" {z_expval(p, MASK01, N) - z_expval(p_true, MASK01, N):+10.5f}")
print(f"\n  negative entries after M^-1 : "
      f"{int(np.sum(p_inv < 0))} of {2**N}, most negative"
      f" {p_inv.min():+.5f}")
print(f"  sum of |negative part|      : {float(-p_inv[p_inv < 0].sum()):.5f}")
```

```text
A. The confusion matrix
==========================================================================
  qubits = 3, so the matrix is 8 x 8 and calibration costs 8 circuits
  per-qubit rates (hidden): [(0.03, 0.06), (0.02, 0.05), (0.05, 0.09)]
  M[0,0] = 0.903070   (all three qubits read correctly from |000>)
  M[7,7] = 0.812630   (all three read correctly from |111>)
  column sums = [1. 1. 1. 1. 1. 1. 1. 1.]
  condition number of M     : 1.3982
  calibrated at 20000 shots : max |M_cal - M_true| = 0.00538

B. Correcting a GHZ measurement
==========================================================================
    state       true   measured        M^-1         LS
      000     0.5000     0.4461     +0.4906     0.4895
      001     0.0000     0.0262     +0.0045     0.0027
      010     0.0000     0.0130     +0.0016     0.0000
      011     0.0000     0.0250     -0.0039     0.0000
      100     0.0000     0.0146     -0.0004     0.0000
      101     0.0000     0.0206     -0.0030     0.0000
      110     0.0000     0.0421     +0.0037     0.0026
      111     0.5000     0.4123     +0.5071     0.5051

                      TVD to truth   <Z0 Z1>      error
            measured       0.14162    0.8535   -0.14650
      M^-1 corrected       0.01686    1.0117   +0.01171
        LS corrected       0.01046    1.0000   +0.00000

  negative entries after M^-1 : 3 of 8, most negative -0.00393
  sum of |negative part|      : 0.00741
```

**What to notice.** The measurement is badly wrong before correction. A GHZ state should give 0.5 and 0.5 on $\lvert 000 \rangle$ and $\lvert 111 \rangle$ and nothing anywhere else; the measured distribution puts 14% of its weight on the six strings that should be empty, and the parity observable $\langle Z_0 Z_1 \rangle$ comes back as 0.854 instead of 1. That $-0.147$ error is a hundred times a good gate error. Readout mitigation is not a refinement.

Inverting the calibrated matrix removes almost all of it — total variation distance from 0.142 to 0.017, the observable error from $-0.147$ to $+0.012$ — and produces three negative probabilities on the way, the largest $-0.0039$. That is not a bug and better calibration will not fix it: $M^{-1}$ is not a stochastic matrix, it maps the simplex outside itself, and shot noise gets pushed out through the corners. The negative entries are the honest signal that the estimate is unphysical. The standard repair is to minimise $\lVert M x - \mathbf{p}_{\text{meas}} \rVert^2$ over the simplex instead of solving the equation — a ten-line projected-gradient loop, which here beats the raw inverse on both measures (TVD 0.010 against 0.017) and never leaves the simplex. The exact agreement of $\langle Z_0 Z_1 \rangle$ with 1.0000 is luck on this instance; staying physical is not.

### The cost, and the assumption everyone makes

The honest method needs one calibration circuit per basis state. That is fine at three qubits and impossible at fifty. The universal workaround is to assume the readout errors are **independent**, which makes the confusion matrix a tensor product of $2 \times 2$ blocks. Then $2n$ calibration experiments determine it, the inverse is the tensor product of the inverses, and applying it costs $O(n 2^n)$ operations on the distribution without ever forming a $2^n \times 2^n$ matrix. This is what production software does. It is also an assumption about the hardware, and Example 3 breaks it on purpose.

### Code Example 3: What Mitigation Costs, and the Assumption That Makes It Affordable

```python
"""Chapter 5, Example 3: what readout mitigation costs, and the assumption that
makes it affordable. Continues from Example 2 (same session)."""
print("A. The cost of the honest version")
print("=" * 74)
print(f"  {'qubits':>7} {'calib. circuits':>16} {'matrix entries':>15}"
      f" {'cond(M)':>12} {'noise blow-up':>14}")
MQ = confusion_matrix([(0.030, 0.060)])
for n in (1, 3, 10, 20, 50):
    print(f"  {n:7d} {2 ** n:16,d} {4 ** n:15.3e}"
          f" {np.linalg.cond(MQ) ** n:12.3e}"
          f" {np.linalg.norm(np.linalg.inv(MQ), 2) ** n:14.3e}")

print("\nB. The tensored shortcut, and its check")
print("=" * 74)


def tensored_inverse_apply(rates, p):
    """Apply the inverse of a tensored confusion matrix without forming it.

    M = M_0 (x) M_1 (x) ... so M^-1 = M_0^-1 (x) M_1^-1 (x) ..., and each factor
    acts on one tensor axis: O(n 2^n) work and O(2^n) memory instead of O(4^n).
    """
    n = len(rates)
    v = p.reshape([2] * n)
    for q, (e01, e10) in enumerate(rates):
        Minv = np.linalg.inv(np.array([[1 - e01, e10], [e01, 1 - e10]]))
        v = np.moveaxis(np.tensordot(Minv, np.moveaxis(v, q, 0), axes=1), 0, q)
    return v.reshape(-1)


p_tens = tensored_inverse_apply(RATES, p_meas)
print(f"  max |tensored inverse - full inverse| = "
      f"{np.max(np.abs(p_tens - np.linalg.solve(M_true, p_meas))):.2e}")
print(f"  calibration circuits: {2 * N} instead of {2 ** N}"
      f"   (2n instead of 2^n)")

print("\nC. What the tensored assumption misses")
print("=" * 74)
C_CORR = 0.040               # probability that qubits 0 and 1 flip together
PERM = np.zeros((2 ** N, 2 ** N))
for t in range(2 ** N):
    PERM[t ^ 0b110, t] = 1.0
M_real = (1 - C_CORR) * M_true + C_CORR * (M_true @ PERM)
marg = []
for q in range(N):
    bit = 2 ** (N - 1 - q)
    e01 = float(sum(M_real[m, 0] for m in range(2 ** N) if m & bit))
    e10 = float(sum(M_real[m, 2 ** N - 1] for m in range(2 ** N)
                    if not m & bit))
    marg.append((e01, e10))
print(f"  correlated flip probability : {C_CORR}")
print("  per-qubit marginals inferred:",
      [(round(a, 4), round(b, 4)) for a, b in marg])
p_meas_r = M_real @ p_true
print(f"\n  {'':>22} {'TVD to truth':>13} {'<Z0 Z1>':>9} {'error':>10}")
for name, p in (("measured", p_meas_r),
                ("tensored correction", tensored_inverse_apply(marg, p_meas_r)),
                ("full-matrix correction", np.linalg.solve(M_real, p_meas_r))):
    print(f"  {name:>22} {tvd(p, p_true):13.5f}"
          f" {z_expval(p, MASK01, N):9.4f}"
          f" {z_expval(p, MASK01, N) - z_expval(p_true, MASK01, N):+10.5f}")
```

```text
A. The cost of the honest version
==========================================================================
   qubits  calib. circuits  matrix entries      cond(M)  noise blow-up
        1                2       4.000e+00    1.105e+00      1.102e+00
        3                8       6.400e+01    1.347e+00      1.337e+00
       10            1,024       1.049e+06    2.702e+00      2.634e+00
       20        1,048,576       1.100e+12    7.302e+00      6.939e+00
       50 1,125,899,906,842,624       1.268e+30    1.441e+02      1.268e+02

B. The tensored shortcut, and its check
==========================================================================
  max |tensored inverse - full inverse| = 1.11e-16
  calibration circuits: 6 instead of 8   (2n instead of 2^n)

C. What the tensored assumption misses
==========================================================================
  correlated flip probability : 0.04
  per-qubit marginals inferred: [(0.0664, 0.0964), (0.0572, 0.0872), (0.05, 0.09)]

                          TVD to truth   <Z0 Z1>      error
                measured       0.17365    0.8472   -0.15280
     tensored correction       0.09074    1.1815   +0.18147
  full-matrix correction       0.00000    1.0000   +0.00000
```

**What to notice.** Part A prices the exact method: 1024 calibration circuits at ten qubits, a million at twenty, $1.1 \times 10^{15}$ at fifty. The condition number grows too, as $\mathrm{cond}(M_q)^n$ — 2.7 at ten qubits and 144 at fifty — so even if you could run the circuits the correction would amplify shot noise by two orders of magnitude. Part B checks the tensored shortcut against the full inverse and finds agreement at $10^{-16}$, as it must when the assumption holds exactly.

Part C is the part worth remembering. Add a 4% probability that qubits 0 and 1 flip *together* — a completely realistic crosstalk channel — and calibrate the tensored model from the single-qubit marginals, which is all $2n$ experiments can see. The raw measurement has $\langle Z_0 Z_1 \rangle$ low by $0.153$. The tensored correction returns $1.18$: it is now high by $0.181$, **a larger error than doing nothing**, and it has the same sign of confidence attached to it. The full calibrated matrix removes the error exactly. A correction built on a wrong noise model does not fail gracefully; it fails in the opposite direction, and nothing in the output says so. The practical consequence is that correlated readout error has to be *measured* — by preparing correlated calibration states and looking for the residual — and not assumed away.

* * *

## 5.2 Zero-Noise Extrapolation, Implemented by Folding

### Amplifying noise without changing the circuit

The idea of zero-noise extrapolation is to measure an observable at several *known multiples* of the device's noise level and extrapolate the sequence back to zero. The sister course scaled the noise by turning up the error probability, which no user of real hardware can do. What a user can do is make the circuit longer in a way that does not change what it computes:

$$ G \;\longrightarrow\; G\,(G^{-1}G)^{k}, \qquad \lambda = 2k+1 $$

Every gate is replaced by $2k+1$ copies of itself, alternating with its inverse. The product is $G$ — exactly, since the inverses cancel in pairs — so the circuit's unitary is untouched, while the number of places where noise can enter is multiplied by $\lambda$. Under a noise model that is independent of the gate, the effective error rate is therefore $\lambda$ times the device's. This is **gate folding**, and it comes in two flavours: **local**, folding each gate in place, and **global**, folding the entire circuit as one block, $C \to C(C^{-1}C)^k$.

Two consequences of the definition are worth stating before the code. First, integer local folding reaches only $\lambda = 1, 3, 5, \ldots$; fractional scales require folding a *subset* of the gates, which Exercise 2 does. Second, the inverse of every gate must itself be expressible in the IR's gate set, or the folded circuit cannot be emitted. For the rotations that is trivial — negate the angle — and for $S$ and $T$ there are exact identities, $S^{-1} = ZS$ and $T^{-1} = ZST$, with no leftover phase.

### Code Example 4: Noise Amplification by Folding, and the Proof It Changes Nothing

```python
"""Chapter 5, Example 4: noise amplification by gate folding, and the proof that
it changes nothing. Continues from Example 3 (same session)."""


def invert_gate(g):
    """Inverse of one gate, as gate tuples inside the IR's own gate set.

    s^-1 = z s and t^-1 = z s t are exact matrix identities with no leftover
    phase, so a folded circuit stays inside the gate set the compiler emits.
    """
    if g[0] in ("h", "x", "z", "cx", "cz"):
        return [g]
    if g[0] == "s":
        return [("z", g[1]), ("s", g[1])]
    if g[0] == "t":
        return [("z", g[1]), ("s", g[1]), ("t", g[1])]
    if g[0] in ROT_1Q:
        return [(g[0], -g[1], g[2])]
    raise ValueError(f"no inverse rule for {g[0]}")


def fold_local(circ, k):
    """Replace every gate G by G (G^-1 G)^k. Noise scale lambda = 2k + 1."""
    out = []
    for g in circ:
        out.append(g)
        for _ in range(k):
            out.extend(invert_gate(g))
            out.append(g)
    return out


def fold_global(circ, k):
    """Replace the whole circuit C by C (C^-1 C)^k. Same lambda, different noise."""
    inv = [h for g in reversed(circ) for h in invert_gate(g)]
    out = list(circ)
    for _ in range(k):
        out.extend(inv)
        out.extend(circ)
    return out


def ansatz(theta, n, layers):
    """Hardware-efficient ansatz: an ry-rz pair per qubit, then a CX ladder."""
    circ, k = [], 0
    for _ in range(layers):
        for q in range(n):
            circ.append(("ry", theta[k], q))
            circ.append(("rz", theta[k + 1], q))
            k += 2
        for q in range(n - 1):
            circ.append(("cx", q, q + 1))
    return circ


NQ, LAYERS = 4, 2
THETA = 0.25 + 0.31 * np.arange(2 * NQ * LAYERS)
CIRC = ansatz(THETA, NQ, LAYERS)
# 4-site transverse-field Ising chain, h = 1
TFIM = {"ZZII": 1.0, "IZZI": 1.0, "IIZZ": 1.0,
        "XIII": 1.0, "IXII": 1.0, "IIXI": 1.0, "IIIX": 1.0}

print("Gate folding: more noise, same unitary")
print("=" * 74)
print(f"  test circuit: {NQ} qubits, {LAYERS} layers,"
      f" {len(CIRC)} gates, depth {circuit_depth(CIRC, NQ)}")
print(f"  gate counts : {gate_counts(CIRC)}")
psi0 = run_circuit(CIRC, NQ)
E0 = sum(expval(psi0, p, TFIM) for p in TFIM)
print(f"  noiseless energy <H> = {E0:.6f}")

print(f"\n  {'lambda':>7} {'gates':>7} {'depth':>7} {'2q gates':>9}"
      f" {'local == global':>16} {'phase-free error':>18}"
      f" {'<H> noiseless':>14}")
for k in (0, 1, 2, 3):
    fl, fg = fold_local(CIRC, k), fold_global(CIRC, k)
    psi = run_circuit(fl, NQ)
    print(f"  {2 * k + 1:7d} {len(fl):7d} {circuit_depth(fl, NQ):7d}"
          f" {gate_counts(fl)['2q']:9d} {str(fl == fg):>16}"
          f" {max(assert_equivalent(CIRC, fl, NQ), assert_equivalent(CIRC, fg, NQ)):18.2e}"
          f" {sum(expval(psi, p, TFIM) for p in TFIM):14.6f}")
```

```text
Gate folding: more noise, same unitary
==========================================================================
  test circuit: 4 qubits, 2 layers, 22 gates, depth 9
  gate counts : {'ry': 8, 'rz': 8, 'cx': 6, '2q': 6}
  noiseless energy <H> = -1.507650

   lambda   gates   depth  2q gates  local == global   phase-free error  <H> noiseless
        1      22       9         6             True           0.00e+00      -1.507650
        3      66      27        18            False           5.66e-16      -1.507650
        5     110      45        30            False           6.18e-16      -1.507650
        7     154      63        42            False           1.18e-15      -1.507650
```

**What to notice.** The gate count and the depth both grow exactly as $\lambda$: 22 gates at $\lambda = 1$, 154 at $\lambda = 7$, with the two-qubit count going 6 to 42. The phase-free error against the original circuit stays at $10^{-16}$ for every scale and both folding strategies, and the noiseless energy is unchanged to six decimals. That is the whole point: **a noiseless simulator cannot detect that folding happened.** Only a noisy device can, which is what makes it a noise knob and not a compiler bug.

Local and global folding give the same gate count and the same depth here, because this circuit's inverse has the same length as the circuit; the `local == global` column shows that the two gate *lists* nevertheless differ from $\lambda = 3$ on, and Example 5 measures whether that matters.

### Code Example 5: Extrapolating to Zero Noise

```python
"""Chapter 5, Example 5: zero-noise extrapolation on the folded circuits.
Continues from Example 4 (same session)."""


def gate_matrix(g):
    """(unitary, target list) of one gate tuple."""
    if g[0] in FIXED_1Q:
        return FIXED_1Q[g[0]], [g[1]]
    if g[0] in ROT_1Q:
        return ROT_1Q[g[0]](g[1]), [g[2]]
    if g[0] == "cx":
        return CNOT4, [g[1], g[2]]
    if g[0] == "cz":
        return CZ4, [g[1], g[2]]
    raise ValueError(g[0])


def rho_apply(rho, U, targets, n):
    """U rho U^dagger, built out of apply_gate acting on the columns of rho."""
    A = np.column_stack([apply_gate(rho[:, k], U, targets, n)
                         for k in range(rho.shape[1])])
    Ad = A.conj().T
    return np.column_stack([apply_gate(Ad[:, k], U, targets, n)
                            for k in range(Ad.shape[1])])


def depolarize(rho, q, p, n):
    """Single-qubit depolarizing channel: (1-p) rho + (p/3) sum_P P rho P."""
    out = (1.0 - p) * rho
    for P in (X, Y, Z):
        out = out + (p / 3.0) * rho_apply(rho, P, [q], n)
    return out


def noisy_rho(circ, n, p):
    """Density matrix after a circuit with a depolarizing kick on every qubit
    every gate touches. Exact -- no trajectory sampling."""
    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rho[0, 0] = 1.0
    for g in circ:
        U, tg = gate_matrix(g)
        rho = rho_apply(rho, U, tg, n)
        for q in gate_qubits(g):
            rho = depolarize(rho, q, p, n)
    return rho


def rho_expval(rho, pauli):
    """tr(P rho) for a Pauli string."""
    n, A = len(pauli), rho
    for q, ch in enumerate(pauli):
        if ch != "I":
            A = np.column_stack([apply_gate(A[:, k], PAULI[ch], [q], n)
                                 for k in range(A.shape[1])])
    return float(np.trace(A).real)


def energy(rho):
    return sum(TFIM[p] * rho_expval(rho, p) for p in TFIM)


def richardson_weights(lams):
    """Exact polynomial interpolation of the noise scales, evaluated at zero."""
    lams = np.asarray(lams, dtype=float)
    return np.array([np.prod([-lj / (li - lj) for lj in lams if lj != li])
                     for li in lams])


def ls_line_weights(lams):
    """Intercept weights of a least-squares straight line through the scales."""
    A = np.column_stack([np.ones_like(lams, dtype=float), np.asarray(lams,
                                                                    float)])
    return np.linalg.pinv(A)[0]


LAMS = np.array([1, 3, 5, 7])
print("A. The extrapolation weights, and what they cost in variance")
print("=" * 74)
print(f"  {'scales':>16} {'estimator':>12} {'weights':>34} {'||w||^2':>9}")
for lams in ([1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 3, 5, 7]):
    for name, wf in (("Richardson", richardson_weights),
                     ("LS line", ls_line_weights)):
        if name == "LS line" and len(lams) == 2:
            continue
        w = wf(lams)
        print(f"  {str(lams):>16} {name:>12}"
              f" {np.array2string(np.round(w, 4), separator=', '):>34}"
              f" {float(w @ w):9.3f}")

print("\nB. Noisy energies at the four folded noise scales")
print("=" * 74)
print(f"  noiseless <H> = {E0:.6f}  (a fixed circuit, not a variational"
      " minimum)")
print(f"\n  {'p':>8} {'folding':>8}" + "".join(f" {'lam=' + str(l):>10}"
                                              for l in LAMS))
VALS = {}
for p in (0.002, 0.005, 0.010):
    for name, fold in (("local", fold_local), ("global", fold_global)):
        VALS[(p, name)] = np.array(
            [energy(noisy_rho(fold(CIRC, (l - 1) // 2), NQ, p)) for l in LAMS])
        print(f"  {p:8.3f} {name:>8}"
              + "".join(f" {v:10.5f}" for v in VALS[(p, name)]))

print("\nC. Extrapolating to zero noise (local folding)")
print("=" * 74)
print(f"  {'p':>8} {'raw bias':>10} {'Richardson':>11} {'bias':>10}"
      f" {'reduction':>10} {'LS line':>10} {'bias':>10} {'reduction':>10}")
for p in (0.002, 0.005, 0.010):
    v = VALS[(p, "local")]
    rich = float(richardson_weights(LAMS) @ v)
    line = float(ls_line_weights(LAMS) @ v)
    raw = v[0] - E0
    print(f"  {p:8.3f} {raw:+10.4f} {rich:11.5f} {rich - E0:+10.5f}"
          f" {abs(raw / (rich - E0)):9.0f}x {line:10.5f} {line - E0:+10.5f}"
          f" {abs(raw / (line - E0)):9.1f}x")
print("\n  local vs global folding, largest disagreement over the whole grid:"
      f" {max(abs(VALS[(p, 'local')] - VALS[(p, 'global')]).max() for p in (0.002, 0.005, 0.010)):.2e}")
```

```text
A. The extrapolation weights, and what they cost in variance
==========================================================================
            scales    estimator                            weights   ||w||^2
            [1, 2]   Richardson                         [ 2., -1.]     5.000
         [1, 2, 3]   Richardson                    [ 3., -3.,  1.]    19.000
         [1, 2, 3]      LS line        [ 1.3333,  0.3333, -0.6667]     2.333
      [1, 2, 3, 4]   Richardson               [ 4., -6.,  4., -1.]    69.000
      [1, 2, 3, 4]      LS line           [ 1. ,  0.5, -0. , -0.5]     1.500
      [1, 3, 5, 7]   Richardson [ 2.1875, -2.1875,  1.3125, -0.3125]    11.391
      [1, 3, 5, 7]      LS line       [ 0.85,  0.45,  0.05, -0.35]     1.050

B. Noisy energies at the four folded noise scales
==========================================================================
  noiseless <H> = -1.507650  (a fixed circuit, not a variational minimum)

         p  folding      lam=1      lam=3      lam=5      lam=7
     0.002    local   -1.46442   -1.38162   -1.30419   -1.23175
     0.002   global   -1.46442   -1.38162   -1.30419   -1.23175
     0.005    local   -1.40193   -1.21398   -1.05464   -0.91910
     0.005   global   -1.40193   -1.21398   -1.05466   -0.91915
     0.010    local   -1.30380   -0.98264   -0.74986   -0.57884
     0.010   global   -1.30380   -0.98267   -0.75001   -0.57917

C. Extrapolating to zero noise (local folding)
==========================================================================
         p   raw bias  Richardson       bias  reduction    LS line       bias  reduction
     0.002    +0.0432    -1.50795   -0.00030       143x   -1.50058   +0.00707       6.1x
     0.005    +0.1057    -1.50815   -0.00050       212x   -1.46898   +0.03867       2.7x
     0.010    +0.2039    -1.50585   +0.00180       113x   -1.38531   +0.12233       1.7x

  local vs global folding, largest disagreement over the whole grid: 3.24e-04
```

**What to notice.** Part A is the arithmetic that decides everything else. Richardson extrapolation is exact polynomial interpolation through the noise scales, evaluated at zero; for $\lambda = 1,2$ it gives $(2,-1)$, for $1,2,3$ it gives $(3,-3,1)$, for $1,2,3,4$ it gives $(4,-6,4,-1)$, with $\lVert w \rVert^2 = 5, 19, 69$ and therefore $\lVert w \rVert_2 = 2.24, 4.36, 8.31$ — the sister course's numbers, recovered from the definition. The folding scales $1,3,5,7$ give gentler weights, $\lVert w \rVert^2 = 11.4$ rather than 69, simply because the points are further apart. And the least-squares line through the same four points has $\lVert w \rVert^2 = 1.05$, which is *less than one per point*: it is barely more expensive than not extrapolating at all.

Part C is the payoff and the trade. At $p = 0.002$ the raw noisy energy is biased by $+0.043$ and the cubic Richardson estimate brings it to $-0.0003$, a **143-fold** reduction; at $p = 0.005$ it is 212-fold and at $p = 0.01$ it is 113-fold. The numbers are that large because the noise is nearly linear in $\lambda$ here, so a cubic through four points models it very well. The least-squares line on the same data manages only 6.1, 2.7 and 1.7 fold. The two estimators differ by a factor of twenty in bias and eleven in variance, in opposite directions — the choice a practitioner has to make, and cannot make without the shot budget. Local and global folding disagree by at most $3.2 \times 10^{-4}$ anywhere on this grid, because depolarizing noise commutes with everything and only the count of insertions matters; for coherent noise they are not equivalent at all, which is Exercise 3.

### Code Example 6: The Variance Mitigation Buys the Bias With

```python
"""Chapter 5, Example 6: the variance mitigation buys the bias with.
Continues from Example 5 (same session)."""
P_NOISE = 0.005
TERM_EV = [np.array([rho_expval(noisy_rho(fold_local(CIRC, (l - 1) // 2),
                                          NQ, P_NOISE), p) for p in TFIM])
           for l in LAMS]
COEF = np.array([TFIM[p] for p in TFIM])


def sampled_energies(shots, rng):
    """One shot-noise realization of <H> at each of the four noise scales.

    Each Pauli term is measured in its own circuit with `shots` shots; the
    estimator of <P> is 2 k / shots - 1 from a binomial count.
    """
    return np.array([float(COEF @ (2.0 * rng.binomial(shots, 0.5 * (1.0 + ev))
                                   / shots - 1.0)) for ev in TERM_EV])


W_RICH = richardson_weights(LAMS)
W_LINE = ls_line_weights(LAMS)
print("The variance cost of extrapolation, measured")
print("=" * 74)
print(f"  p = {P_NOISE}, four noise scales {LAMS.tolist()},"
      f" {len(TFIM)} Pauli terms")
print(f"  ||w||^2 : Richardson {float(W_RICH @ W_RICH):.3f},"
      f" LS line {float(W_LINE @ W_LINE):.3f}, raw 1.000")
print(f"\n  {'shots/term':>11} {'total shots':>12} {'estimator':>11}"
      f" {'mean bias':>10} {'std':>9} {'predicted std':>14} {'RMSE':>9}")
for shots in (1000, 10000, 100000):
    rng = np.random.default_rng(7)
    draws = np.array([sampled_energies(shots, rng) for _ in range(600)])
    raw = draws[:, 0]
    rich = draws @ W_RICH
    line = draws @ W_LINE
    s_raw = float(np.std(raw))
    for name, est, w2 in (("raw", raw, 1.0),
                          ("LS line", line, float(W_LINE @ W_LINE)),
                          ("Richardson", rich, float(W_RICH @ W_RICH))):
        bias = float(np.mean(est)) - E0
        sd = float(np.std(est))
        print(f"  {shots:11d} {shots * len(TFIM) * len(LAMS):12,d} {name:>11}"
              f" {bias:+10.5f} {sd:9.5f} {s_raw * np.sqrt(w2):14.5f}"
              f" {np.sqrt(bias ** 2 + sd ** 2):9.5f}")
print("\n  Shot count needed for the extrapolated value to match the raw")
print("  estimator's statistical error, at fixed total budget:")
for name, w2 in (("LS line", float(W_LINE @ W_LINE)),
                 ("Richardson", float(W_RICH @ W_RICH))):
    print(f"    {name:>11}: {len(LAMS)} scales x ||w||^2 = {w2:6.3f}"
          f"  ->  {len(LAMS) * w2:6.2f}x the shots")
```

```text
The variance cost of extrapolation, measured
==========================================================================
  p = 0.005, four noise scales [1, 3, 5, 7], 7 Pauli terms
  ||w||^2 : Richardson 11.391, LS line 1.050, raw 1.000

   shots/term  total shots   estimator  mean bias       std  predicted std      RMSE
         1000       28,000         raw   +0.10671   0.07563        0.07563   0.13079
         1000       28,000     LS line   +0.03844   0.07950        0.07749   0.08831
         1000       28,000  Richardson   -0.00194   0.25368        0.25524   0.25369
        10000      280,000         raw   +0.10612   0.02412        0.02412   0.10883
        10000      280,000     LS line   +0.03942   0.02479        0.02472   0.04657
        10000      280,000  Richardson   -0.00122   0.08473        0.08142   0.08474
       100000    2,800,000         raw   +0.10592   0.00804        0.00804   0.10622
       100000    2,800,000     LS line   +0.03899   0.00837        0.00824   0.03988
       100000    2,800,000  Richardson   -0.00086   0.02753        0.02713   0.02755

  Shot count needed for the extrapolated value to match the raw
  estimator's statistical error, at fixed total budget:
        LS line: 4 scales x ||w||^2 =  1.050  ->    4.20x the shots
     Richardson: 4 scales x ||w||^2 = 11.391  ->   45.56x the shots
```

**What to notice.** The measured standard deviations track $\sigma_{\text{raw}}\lVert w \rVert_2$ to within a few percent at every budget — 0.2537 against a predicted 0.2552 for Richardson at a thousand shots per term, 0.0080 against 0.0080 for the raw estimator at a hundred thousand. The variance penalty of extrapolation is not a heuristic; it is the norm of the weight vector, and it is computable before any data is taken.

The RMSE column is the honest summary and it has a crossover in it. At a thousand shots per term the ranking is LS line (0.088), raw (0.131), Richardson (0.254) — the high-order extrapolation is the *worst* of the three, because its variance swamps a bias it removed almost perfectly. At a hundred thousand shots per term the ranking is reversed: Richardson (0.0276), LS line (0.0399), raw (0.106). **Which extrapolation is best is a property of the shot budget, not of the method.** Any paper reporting a ZNE result without its shot count has withheld the information needed to interpret it.

The total budgets are worth reading too: 2.8 million shots in the last block, on a four-qubit circuit with twenty-two gates, to get one expectation value to $\pm 0.03$. Mitigation converts a bias problem into a sampling problem, and the sampling problem was already the binding one.

* * *

## 5.3 Probabilistic Error Cancellation

### Inverting a channel you cannot invert

Zero-noise extrapolation makes an assumption it cannot check: that the observable is a smooth, low-order function of the noise strength. Probabilistic error cancellation makes no such assumption, and in exchange it demands a full characterisation of the noise. The idea is to apply the *inverse* of the noise channel. The obstacle is that the inverse of a noisy channel is not a channel — it is not completely positive, and no hardware can run it. The resolution is that it can always be written as a **quasiprobability** combination of operations that *can* be run:

$$ \mathcal{N}^{-1} = \sum_i c_i\,\mathcal{O}_i, \qquad \sum_i c_i = 1, \qquad \text{some } c_i < 0 $$

Sample $\mathcal{O}_i$ with probability $\lvert c_i \rvert / \gamma$ where $\gamma = \sum_i \lvert c_i \rvert$, insert it into the circuit, and multiply the measured value by $\gamma\,\mathrm{sign}(c_i)$. The average of that estimator is the noiseless expectation value — it is **unbiased**, which extrapolation is not. The price is in the variance: the estimator's spread grows by a factor $\gamma$ per noise location, so the shot count grows as $\gamma^{2N}$ for $N$ locations. That $\gamma^{2N}$ is the exponential wall of this chapter, and Example 7 puts numbers on it.

For a single-qubit depolarizing channel the algebra is short enough to do by hand. Its Pauli transfer matrix is $\mathrm{diag}(1, f, f, f)$ with $f = 1 - 4p/3$; writing the inverse as $\alpha\,\mathrm{id} + \beta(X \cdot X + Y \cdot Y + Z \cdot Z)$ gives $\alpha + 3\beta = 1$ and $\alpha - \beta = 1/f$, hence

$$ \beta = \frac{1}{4}\left(1 - \frac{1}{f}\right) < 0, \qquad \alpha = 1 - 3\beta, \qquad \gamma = \lvert \alpha \rvert + 3\lvert \beta \rvert = 1 + \frac{3}{2}\left(\frac{1}{f} - 1\right) $$

and to leading order $\gamma \approx 1 + 2p$.

### Code Example 7: Probabilistic Error Cancellation, and Its Sampling Bill

```python
"""Chapter 5, Example 7: probabilistic error cancellation, and its sampling bill.
Continues from Example 6 (same session)."""
PAULI_LIST = (I2, X, Y, Z)


def pec_coeffs(p):
    """Quasiprobability decomposition of the inverse depolarizing channel.

    The channel's Pauli transfer matrix is diag(1, f, f, f) with f = 1 - 4p/3.
    Writing the inverse as alpha * identity + beta * (X . X + Y . Y + Z . Z)
    gives diag(alpha + 3 beta, alpha - beta, ...), so alpha + 3 beta = 1 and
    alpha - beta = 1/f. The one-norm gamma = |alpha| + 3|beta| is what the
    sampling cost is exponential in.
    """
    f = 1.0 - 4.0 * p / 3.0
    beta = (1.0 - 1.0 / f) / 4.0
    alpha = 1.0 - 3.0 * beta
    return alpha, beta, abs(alpha) + 3.0 * abs(beta)


def rho1_channel(rho, p):
    """One-qubit depolarizing channel, on a 2 x 2 density matrix."""
    return (1 - p) * rho + (p / 3.0) * sum(P @ rho @ P for P in (X, Y, Z))


def rho1_inverse(rho, p):
    """The quasiprobability inverse, applied exactly (not sampled)."""
    alpha, beta, _ = pec_coeffs(p)
    return alpha * rho + beta * sum(P @ rho @ P for P in (X, Y, Z))


print("A. The inverse of a depolarizing channel is not a channel")
print("=" * 74)
rng = np.random.default_rng(5150)
v = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
rho_test = v @ v.conj().T
rho_test = rho_test / np.trace(rho_test)
print(f"  {'p':>8} {'f':>10} {'alpha':>10} {'beta':>11} {'gamma':>10}"
      f" {'1 + 2p':>9} {'inversion error':>16}")
for p in (0.001, 0.002, 0.005, 0.010, 0.050):
    a, b, g = pec_coeffs(p)
    err = np.max(np.abs(rho1_inverse(rho1_channel(rho_test, p), p) - rho_test))
    print(f"  {p:8.3f} {1 - 4 * p / 3:10.6f} {a:10.6f} {b:+11.6f} {g:10.6f}"
          f" {1 + 2 * p:9.6f} {err:16.2e}")
print("  beta is negative, so the inverse is a quasiprobability and not a")
print("  channel -- it cannot be run on hardware, only sampled from.")

print("\nB. Sampling overhead, which is exponential in the circuit size")
print("=" * 74)
LOCS = sum(len(gate_qubits(g)) for g in CIRC)
print(f"  the Example 4 circuit has {len(CIRC)} gates and {LOCS} noise"
      " locations")
print("  Entries are log10 of the factor by which the shot budget must grow to")
print("  hold the statistical error fixed: overhead = gamma^(2N).")
print(f"\n  {'p':>8} {'gamma':>10}" + "".join(
    f" {'N=' + f'{n:.0e}':>10}" for n in (LOCS, 1e2, 1e3, 1e4, 1e6)))
for p in (0.001, 0.002, 0.005, 0.010):
    _, _, g = pec_coeffs(p)
    print(f"  {p:8.3f} {g:10.6f}"
          + "".join(f" {2 * n * np.log10(g):10.1f}"
                    for n in (LOCS, 1e2, 1e3, 1e4, 1e6)))

print("\nC. PEC implemented, on a circuit small enough to sample")
print("=" * 74)
SMALL = [("h", 0), ("cx", 0, 1), ("ry", 0.7, 1)]
NS, P_PEC = 2, 0.02
psi_ideal = run_circuit(SMALL, NS)
ideal = expval(psi_ideal, "ZZ")
noisy = rho_expval(noisy_rho(SMALL, NS, P_PEC), "ZZ")
alpha, beta, gamma = pec_coeffs(P_PEC)
weights = np.array([abs(alpha), abs(beta), abs(beta), abs(beta)]) / gamma
signs = np.sign([alpha, beta, beta, beta])
n_loc = sum(len(gate_qubits(g)) for g in SMALL)


def pec_sample(rng):
    """One PEC sample: insert Paulis drawn from the quasiprobability, keep the sign."""
    rho = np.zeros((2 ** NS, 2 ** NS), dtype=complex)
    rho[0, 0] = 1.0
    sign = 1.0
    for g in SMALL:
        U, tg = gate_matrix(g)
        rho = rho_apply(rho, U, tg, NS)
        for q in gate_qubits(g):
            rho = depolarize(rho, q, P_PEC, NS)
            j = rng.choice(4, p=weights)
            sign *= signs[j]
            rho = rho_apply(rho, PAULI_LIST[j], [q], NS)
    return sign * gamma ** n_loc * rho_expval(rho, "ZZ"), sign


rng = np.random.default_rng(24680)
draws, sgn = map(np.array, zip(*[pec_sample(rng) for _ in range(4000)]))
print(f"  circuit: {len(SMALL)} gates, {n_loc} noise locations, p = {P_PEC}")
print(f"  gamma = {gamma:.6f}, gamma^{n_loc} = {gamma ** n_loc:.4f},"
      f" overhead gamma^{2 * n_loc} = {gamma ** (2 * n_loc):.4f}")
print(f"\n  {'estimator':>14} {'value':>10} {'bias':>10} {'std of one sample':>18}")
print(f"  {'noiseless':>14} {ideal:10.5f} {0.0:+10.5f} {'--':>18}")
print(f"  {'noisy':>14} {noisy:10.5f} {noisy - ideal:+10.5f} {'--':>18}")
print(f"  {'PEC, 4000':>14} {draws.mean():10.5f} {draws.mean() - ideal:+10.5f}"
      f" {draws.std():18.5f}")
print(f"  standard error of the PEC mean: {draws.std() / np.sqrt(len(draws)):.5f}")
print(f"  fraction of samples drawn with a negative sign:"
      f" {float(np.mean(sgn < 0)):.4f}")
```

```text
A. The inverse of a depolarizing channel is not a channel
==========================================================================
         p          f      alpha        beta      gamma    1 + 2p  inversion error
     0.001   0.998667   1.001001   -0.000334   1.002003  1.002000         1.11e-16
     0.002   0.997333   1.002005   -0.000668   1.004011  1.004000         0.00e+00
     0.005   0.993333   1.005034   -0.001678   1.010067  1.010000         1.11e-16
     0.010   0.986667   1.010135   -0.003378   1.020270  1.020000         1.11e-16
     0.050   0.933333   1.053571   -0.017857   1.107143  1.100000         5.55e-17
  beta is negative, so the inverse is a quasiprobability and not a
  channel -- it cannot be run on hardware, only sampled from.

B. Sampling overhead, which is exponential in the circuit size
==========================================================================
  the Example 4 circuit has 22 gates and 28 noise locations
  Entries are log10 of the factor by which the shot budget must grow to
  hold the statistical error fixed: overhead = gamma^(2N).

         p      gamma    N=3e+01    N=1e+02    N=1e+03    N=1e+04    N=1e+06
     0.001   1.002003        0.0        0.2        1.7       17.4     1737.8
     0.002   1.004011        0.1        0.3        3.5       34.8     3476.7
     0.005   1.010067        0.2        0.9        8.7       87.0     8700.5
     0.010   1.020270        0.5        1.7       17.4      174.3    17430.5

C. PEC implemented, on a circuit small enough to sample
==========================================================================
  circuit: 3 gates, 4 noise locations, p = 0.02
  gamma = 1.041096, gamma^4 = 1.1748, overhead gamma^8 = 1.3801

       estimator      value       bias  std of one sample
       noiseless    0.76484   +0.00000                 --
           noisy    0.70527   -0.05957                 --
       PEC, 4000    0.75812   -0.00672            0.33428
  standard error of the PEC mean: 0.00529
  fraction of samples drawn with a negative sign: 0.0757
```

**What to notice.** Part A confirms the algebra: $\beta$ is negative at every noise level, the quasiprobability inverse composed with the channel returns the input density matrix to $10^{-16}$, and $\gamma$ agrees with the leading-order $1 + 2p$ to four digits at $p = 10^{-3}$ and to two at $p = 0.05$.

Part B is the wall. Read the row at $p = 0.005$. On the twenty-two-gate circuit of Example 4, with twenty-eight noise locations, the sampling overhead is $10^{0.2}$, a factor of 1.6 — completely affordable, and this is why PEC is used in practice on small circuits. At a thousand noise locations it is $10^{8.7}$. At ten thousand it is $10^{87}$, and at a million it is $10^{8700}$. There is no hardware improvement, no algorithmic cleverness and no funding level that touches a number like that. **Mitigation is not a path to large circuits; it is a way to get more out of small ones.**

Part C is PEC implemented, on a circuit small enough that four thousand samples resolve the answer. The noisy expectation is biased by $-0.060$; PEC returns $0.758$ against a noiseless $0.765$, a residual of $-0.007$ against a standard error of $0.005$ — unbiased within one and a half standard errors, as advertised. The standard deviation of a *single* sample is 0.334 for an observable bounded by 1, and 7.6% of samples carry a negative sign. Those two numbers are the exponential cost made visible: as the circuit grows the sign approaches a coin flip, positive and negative contributions cancel ever more completely, and the samples needed to see the difference grow as $\gamma^{2N}$.

* * *

## 5.4 Where Software Cleverness Ends

Put the three methods side by side, honestly.

| | What it fixes | Extra qubits | Assumption | Sampling cost |
| --- | --- | --- | --- | --- |
| Readout mitigation | Assignment error only | none | Noise model of the readout, and $2^n$ or $2n$ calibrations | $\sim 1$, plus calibration |
| Zero-noise extrapolation | Bias from any gate noise that scales with $\lambda$ | none | The observable is low-order in $\lambda$; folding scales the noise | $\lVert w \rVert^2$ per scale, so 4 to 46$\times$ here |
| Probabilistic error cancellation | Bias, with no smoothness assumption | none | Full tomographic knowledge of the noise | $\gamma^{2N}$, exponential in circuit size |
| Error correction | Everything, including leakage if the code is designed for it | $2d^2 - 1$ per logical qubit | Physical error below threshold | $\sim 1$ |

The pattern in the last column is the whole story of this chapter. The three mitigation methods buy accuracy with *samples*, and the cost in samples grows with the size of the circuit — polynomially for extrapolation if the extrapolation model holds, and exponentially for PEC always. Error correction buys accuracy with *qubits*, and the cost in qubits grows with the *logarithm* of the required logical error rate, because the code distance enters the exponent. A quantity that grows exponentially and a quantity that grows logarithmically cross, and they cross early.

Three further asymmetries are worth naming, because they are the parts that a table of overheads hides.

  * **Mitigation corrects expectation values, not states.** Nothing above produces a corrected state a later gate could act on. It is post-processing on measurement statistics, so it cannot run inside a computation, cannot support a deep coherent subroutine, and cannot help an algorithm whose output is a sampled bitstring rather than an average.
  * **Mitigation cannot handle what it cannot model.** Example 3's tensored readout correction made things worse when the model was wrong, and Exercise 3 exhibits a coherent error that folding cancels rather than amplifies, so extrapolation returns the biased answer and reports no difficulty. A mitigation method's failure mode is a confidently wrong number.
  * **And mitigation is genuinely load-bearing today.** Example 2 removed a readout error a hundred times larger than a well-calibrated gate error in Chapter 4; Example 5 removed 99.3% of a noise bias. On the circuits current hardware can run at all, these methods are the difference between a result and a plot of noise. Both statements are true and neither is a hedge.

* * *

## 5.5 The Resource-Estimation Pipeline

The other way to answer "should I mitigate?" is to price the alternative. A fault-tolerant resource estimate is a chain of four functions, each of which is arithmetic, and the discipline lies entirely in stating the inputs.

$$ \text{algorithm} \xrightarrow{\ \lambda,\ \varepsilon,\ C_{\text{walk}}\ } N_{\text{Toffoli}} \xrightarrow{\ \text{failure budget}\ } p_L \xrightarrow{\ p_{\text{phys}},\ p_{\text{th}}\ } d \xrightarrow{\ \text{code}\ } n_{\text{phys}},\ t_{\text{wall}} $$

The first link is the algorithm-level statement of [Intermediate Quantum Algorithms, Chapter 4](<../quantum-algorithms-intermediate/chapter-4.html>): qubitized phase estimation needs about $(\pi/2)(\lambda/\varepsilon)$ walk steps, each costing $C_{\text{walk}}$ Toffolis. The second is a budget choice — hold the whole run's failure probability at 0.1, so $p_L = 0.1/N_{\text{Toffoli}}$. The third is the threshold formula of [Introduction to Quantum Computing, Chapter 5](<../quantum-computing-introduction/chapter-5.html>), $p_L \approx A(p/p_{\text{th}})^{(d+1)/2}$, solved for the smallest odd $d$. The fourth is the code's footprint, $2d^2 - 1$ physical qubits per logical qubit for the rotated surface code, and a placeholder cycle time of 10 $\mu$s per logical Toffoli.

One implementation detail is not optional. The third step compares a ratio raised to a large power against a power of ten, and in binary floating point those are not the numbers they look like: `0.1 * 0.1 ** 5` evaluates to `1.0000000000000004e-06`, which is *not* $\le 10^{-6}$. A bare comparison therefore rejects the distance that meets the target exactly and returns the next one up — two whole steps of code distance, since $d$ is odd, and hundreds of physical qubits per logical qubit. Two of the sister courses hit this trap; the guard is a relative tolerance, and Example 8 demonstrates the failure before applying it.

### Code Example 8: Algorithm Inputs to Physical Qubits and Days

```python
"""Chapter 5, Example 8: the resource-estimation pipeline, logical to physical.
Continues from Example 7 (same session); arithmetic only, no simulator needed."""
CHEM_ACC = 1.6e-3        # Hartree, the conventional chemical-accuracy target
P_THRESHOLD = 1e-2       # representative surface-code threshold, order of magnitude
A_PREFACTOR = 0.1        # dimensionless prefactor, order of magnitude
T_TOFFOLI = 1e-5         # seconds per logical Toffoli, a standard placeholder


def qubitized_toffolis(lam, eps, toffolis_per_walk):
    """Toffoli count of qubitized phase estimation: (pi/2)(lam/eps) walk steps."""
    steps = np.pi * lam / (2 * eps)
    return steps, steps * toffolis_per_walk


def logical_error_target(n_toffoli, total_failure=0.1):
    """Per-gate logical error rate that keeps the whole run's failure at 0.1."""
    return total_failure / n_toffoli


def logical_error(p_phys, d, p_th=P_THRESHOLD, A=A_PREFACTOR):
    """Surface-code scaling p_L ~ A (p/p_th)^((d+1)/2)."""
    return A * (p_phys / p_th) ** ((d + 1) / 2)


def required_distance(p_phys, target, p_th=P_THRESHOLD, A=A_PREFACTOR):
    """Smallest odd code distance reaching a target logical error rate.

    The comparison carries a RELATIVE tolerance because p_L is a ratio raised to
    a large power: a distance that meets the target exactly lands a few ulps
    above it in binary floating point, and a bare `<= target` would reject it
    and return the next distance up. Two sister courses hit this trap.
    """
    if p_phys >= p_th:
        return None                  # at or above threshold, more qubits do not help
    for d in range(3, 201, 2):
        if logical_error(p_phys, d, p_th, A) <= target * (1 + 1e-9):
            return d
    return None


def physical_per_logical(d):
    """Rotated surface code: 2 d^2 - 1 physical qubits per logical qubit."""
    return 2 * d * d - 1


def estimate(lam, eps, toffolis_per_walk, p_phys, n_logical):
    """The whole pipeline as one function: algorithm inputs -> hardware."""
    steps, tof = qubitized_toffolis(lam, eps, toffolis_per_walk)
    pL = logical_error_target(tof)
    d = required_distance(p_phys, pL)
    per = physical_per_logical(d)
    return {"walk_steps": steps, "toffolis": tof, "p_L": pL, "distance": d,
            "per_logical": per, "physical": per * n_logical,
            "seconds": tof * T_TOFFOLI, "days": tof * T_TOFFOLI / 86400}


print("A. The floating-point trap in the threshold comparison")
print("=" * 74)
print(f"  0.1 * 0.1 ** 5  = {0.1 * 0.1 ** 5!r}")
print(f"  is it <= 1e-06? {0.1 * 0.1 ** 5 <= 1e-6}")
print(f"  with a relative tolerance? {0.1 * 0.1 ** 5 <= 1e-6 * (1 + 1e-9)}")
for target in (1e-6, 1e-10, 1e-12):
    d_ok = required_distance(1e-3, target)
    d_bad = next((d for d in range(3, 201, 2)
                  if logical_error(1e-3, d) <= target), None)
    print(f"  p = 1e-3, target {target:.0e}: guarded d = {d_ok:3d},"
          f" unguarded d = {d_bad:3d}"
          + ("   <- off by one distance step" if d_ok != d_bad else ""))

print("\nB. Reproducing the sister courses' tables")
print("=" * 74)
print(f"  {'p_phys':>9} {'target p_L':>12} {'distance d':>11}"
      f" {'phys/logical':>13} {'100 logical qubits':>19}")
for p_phys in (1e-3, 3e-4, 1e-4):
    for target in (1e-6, 1e-10, 1e-15):
        d = required_distance(p_phys, target)
        per = physical_per_logical(d)
        print(f"  {p_phys:9.0e} {target:12.0e} {d:11d} {per:13,d} {100 * per:19,d}")

print("\nC. Algorithm to hardware, in one call")
print("=" * 74)
print(f"  eps = {CHEM_ACC:.1e} Ha, p_phys = 1e-3, {2 * 76 + 1000} logical qubits")
print(f"\n  {'system':>26} {'lambda':>8} {'C_walk':>8} {'Toffolis':>10}"
      f" {'p_L':>9} {'d':>4} {'physical':>10} {'days':>8}")
for name, lam, cw in (("N2, moderate space", 40.0, 1e4),
                      ("FeMoco, low estimate", 300.0, 1e4),
                      ("FeMoco, mid estimate", 1000.0, 3e4),
                      ("FeMoco, high estimate", 4000.0, 1e5)):
    r = estimate(lam, CHEM_ACC, cw, 1e-3, 2 * 76 + 1000)
    print(f"  {name:>26} {lam:8.0f} {cw:8.0e} {r['toffolis']:10.2e}"
          f" {r['p_L']:9.1e} {r['distance']:4d} {r['physical']:10.2e}"
          f" {r['days']:8.2f}")

print("\n  cross-check against Intermediate Quantum Algorithms Chapter 4:")
for tof in (1e9, 1e10, 1e11):
    pL = logical_error_target(tof)
    d = required_distance(1e-3, pL)
    per = physical_per_logical(d)
    print(f"    {tof:.0e} Toffolis -> p_L {pL:.1e}, d = {d}, {per} per logical,"
          f" {per * 1152:.2e} physical, {tof * T_TOFFOLI / 86400:.2f} days")

print("\nD. Where mitigation stops and correction starts")
print("=" * 74)
print("  mitigation: PEC sampling overhead gamma^(2N) at p = 1e-3, N noise")
print("  locations. correction: physical qubits per logical qubit at p = 1e-3")
print("  for a logical error rate of 0.1/N.")
_, _, g3 = pec_coeffs(1e-3)
print(f"\n  {'N (gates)':>11} {'log10 PEC overhead':>19} {'p_L needed':>11}"
      f" {'d':>4} {'phys/logical':>13}")
for n in (1e2, 1e3, 1e4, 1e6, 1e9, 1e11):
    d = required_distance(1e-3, logical_error_target(n))
    print(f"  {n:11.0e} {2 * n * np.log10(g3):19.1f}"
          f" {logical_error_target(n):11.1e} {d:4d}"
          f" {physical_per_logical(d):13,d}")
```

```text
A. The floating-point trap in the threshold comparison
==========================================================================
  0.1 * 0.1 ** 5  = 1.0000000000000004e-06
  is it <= 1e-06? False
  with a relative tolerance? True
  p = 1e-3, target 1e-06: guarded d =   9, unguarded d =  11   <- off by one distance step
  p = 1e-3, target 1e-10: guarded d =  17, unguarded d =  19   <- off by one distance step
  p = 1e-3, target 1e-12: guarded d =  21, unguarded d =  23   <- off by one distance step

B. Reproducing the sister courses' tables
==========================================================================
     p_phys   target p_L  distance d  phys/logical  100 logical qubits
      1e-03        1e-06           9           161              16,100
      1e-03        1e-10          17           577              57,700
      1e-03        1e-15          27         1,457             145,700
      3e-04        1e-06           7            97               9,700
      3e-04        1e-10          11           241              24,100
      3e-04        1e-15          19           721              72,100
      1e-04        1e-06           5            49               4,900
      1e-04        1e-10           9           161              16,100
      1e-04        1e-15          13           337              33,700

C. Algorithm to hardware, in one call
==========================================================================
  eps = 1.6e-03 Ha, p_phys = 1e-3, 1152 logical qubits

                      system   lambda   C_walk   Toffolis       p_L    d   physical     days
          N2, moderate space       40    1e+04   3.93e+08   2.5e-10   17   6.65e+05     0.05
        FeMoco, low estimate      300    1e+04   2.95e+09   3.4e-11   19   8.31e+05     0.34
        FeMoco, mid estimate     1000    3e+04   2.95e+10   3.4e-12   21   1.01e+06     3.41
       FeMoco, high estimate     4000    1e+05   3.93e+11   2.5e-13   23   1.22e+06    45.45

  cross-check against Intermediate Quantum Algorithms Chapter 4:
    1e+09 Toffolis -> p_L 1.0e-10, d = 17, 577 per logical, 6.65e+05 physical, 0.12 days
    1e+10 Toffolis -> p_L 1.0e-11, d = 19, 721 per logical, 8.31e+05 physical, 1.16 days
    1e+11 Toffolis -> p_L 1.0e-12, d = 21, 881 per logical, 1.01e+06 physical, 11.57 days

D. Where mitigation stops and correction starts
==========================================================================
  mitigation: PEC sampling overhead gamma^(2N) at p = 1e-3, N noise
  locations. correction: physical qubits per logical qubit at p = 1e-3
  for a logical error rate of 0.1/N.

    N (gates)  log10 PEC overhead  p_L needed    d  phys/logical
        1e+02                 0.2     1.0e-03    3            17
        1e+03                 1.7     1.0e-04    5            49
        1e+04                17.4     1.0e-05    7            97
        1e+06              1737.8     1.0e-07   11           241
        1e+09           1737757.8     1.0e-10   17           577
        1e+11         173775776.0     1.0e-12   21           881
```

**What to notice.** Part A is the trap, exhibited. At $p = 10^{-3}$ and a target of $10^{-6}$ the guarded search returns $d = 9$ and the unguarded one returns $d = 11$; at $10^{-12}$ it is 21 against 23. In physical qubits that is 161 against 241, and 881 against 1057 — a 20% error in the headline number of a resource estimate, from a floating-point comparison. The fix costs one multiplication.

Part B reproduces the table of [Introduction to Quantum Computing, Chapter 5](<../quantum-computing-introduction/chapter-5.html>) exactly, which it can because that course uses the same $2d^2 - 1$ convention: at $p = 10^{-3}$, a logical error rate of $10^{-10}$ needs distance 17 and 577 physical qubits per logical qubit, so a hundred logical qubits cost 57,700 physical ones; improving the physical error rate to $3 \times 10^{-4}$ brings the same target down to distance 11 and 24,100. Every row matches.

Part C runs the whole pipeline in one call and lands where the intermediate algorithms course landed. A FeMoco-scale calculation at $\lambda = 1000$ Hartree and $C_{\text{walk}} = 3 \times 10^4$ needs $2.95 \times 10^{10}$ Toffolis, a logical error rate of $3.4 \times 10^{-12}$, code distance 21, about $10^6$ physical qubits and 3.4 days — and at the high end of the plausible input range, $3.9 \times 10^{11}$ Toffolis, $1.2 \times 10^6$ physical qubits and 45 days. The cross-check block reproduces the three headline rows of [Intermediate Quantum Algorithms, Chapter 4](<../quantum-algorithms-intermediate/chapter-4.html>) — the code distance and the wall-clock day count at $10^9$, $10^{10}$ and $10^{11}$ Toffolis — digit for digit. The physical-qubit column is the one place the two disagree, and only by the single qubit per logical qubit that separates the conventions: that course counts $2d^2$, this one counts $2d^2 - 1$ as §5.5 disclosed, following [Introduction to Quantum Computing, Chapter 5](<../quantum-computing-introduction/chapter-5.html>).

Part D is the boundary of Section 5.4, computed. At a hundred noise locations the PEC overhead is a factor of 1.6 and error correction would need distance 3, seventeen physical qubits per logical qubit — mitigation wins easily, and it needs no extra qubits at all. At ten thousand locations mitigation costs $10^{17}$ in samples and correction costs 97 physical qubits per logical qubit. At $10^{11}$ — the size of the FeMoco circuit — mitigation costs $10^{1.7 \times 10^{8}}$ and correction costs 881. **The crossover is somewhere around a few thousand gates**, and it is not close on either side of that. Everything above the crossover is a fault-tolerance problem, and everything below it is where all of today's quantum computing happens.

* * *

## 5.6 The Map of the Series

This is the sixth and last of the quantum courses in this collection, and the six were written to be read against each other. The table is the map.

| Course | The question it answers | What it hands to the others |
| --- | --- | --- |
| [Introduction to Quantum Computing](<../quantum-computing-introduction/index.html>) | What is a qubit, a gate, an algorithm; how do you simulate one | The mini-simulator every other course runs on, and the noise, mitigation and correction budgets of its Chapter 5 |
| [Introduction to Quantum Hardware](<../quantum-hardware-introduction/index.html>) | What are $T_1$, $T_2$ and a gate error made of, physically | The transmon spectrum and the DRAG physics that Chapter 4 of this course turns into a control stack |
| [Intermediate Quantum Algorithms](<../quantum-algorithms-intermediate/index.html>) | Which speedups are theorems, and what do they assume | Block encodings, qubitization and the Toffoli-count language of Section 5.5 |
| Introduction to the Quantum Software Stack | What happens between an algorithm and a pulse | A circuit IR, an optimizer, a router, a pulse simulator and a mitigation layer, all built rather than imported |
| [Quantum Machine Learning](<../../MI/quantum-machine-learning-introduction/index.html>) | Does a quantum model learn better than a classical one | The evaluation discipline — matched budgets, paired intervals, honest baselines |
| [Quantum Sensing](<../../MS/quantum-sensing-introduction/index.html>) | Where is quantum technology already delivering an advantage | The one affirmative answer in the collection, and the metrology that calibration in Chapter 4 rests on |

Two cross-cutting themes run through all six and are worth stating once, at the end.

**The layers leak, and knowing which way is the skill.** A compiler that pushes $Z$ rotations to the end is exploiting a pulse-layer fact. A resource estimate that quotes physical qubits is exploiting a threshold formula whose prefactor is a decoder property. A mitigation method that fails silently fails because its noise model came from a different layer than the noise did. Every serious decision here requires reaching one layer down, which is why a course teaching one layer's API would not be enough.

**And the numbers are the argument.** Every claim in these six courses that could be measured was measured, in code that runs, with the failures left in: the ansatz that lost to a fifteen-line local search, the readout correction that was worse than no correction, the calibration loop that stalled on a systematic, the extrapolation that was worst of three at a small budget. The habit worth taking away is the one that catches over-general claims: ask what the budget was, what the baseline was, and which layer the assumption came from.

* * *

## Exercises

#### Exercise 1: Readout Mitigation on a Local Observable

Most observables of interest are $k$-local — a parity on two or three qubits — and the full $2^n$ machinery is unnecessary for them.

  1. For a single qubit $q$ with rates $(\epsilon_{01}, \epsilon_{10})$, show that the measured marginal satisfies $P_{\text{meas}}(1) = (1 - \epsilon_{01} - \epsilon_{10})P_{\text{true}}(1) + \epsilon_{01}$, and invert it.
  2. Verify the inversion numerically for all three qubits of the GHZ state of Example 2.
  3. What does the result imply about the calibration cost of correcting a $k$-local observable on an $n$-qubit device?
  4. Under the correlated readout model of Example 3, does the single-qubit inversion still work for a single-qubit observable?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Marginalising the tensored confusion matrix over every other qubit leaves the \(2 \times 2\) block for qubit \(q\), so \(P_{\text{meas}}(1) = (1-\epsilon_{10})P_{\text{true}}(1) + \epsilon_{01}(1 - P_{\text{true}}(1))\), which rearranges to the stated form. Inverting gives \(P_{\text{true}}(1) = (P_{\text{meas}}(1) - \epsilon_{01})/(1 - \epsilon_{01} - \epsilon_{10})\). The correction is one affine map per qubit, and the denominator is the readout <em>contrast</em>.</p>

<p><strong>2.</strong> All three qubits return 0.500000 against a true 0.5, from measured marginals of 0.4850, 0.4850 and 0.4800.</p>

<p><strong>3.</strong> Only the \(k\) qubits in the observable's support matter, so \(2^k\) calibration circuits suffice — 4 for a two-qubit parity, 8 for a three-qubit one, independent of \(n\). This is why readout mitigation is used routinely on hundred-qubit devices even though the exact method needs \(2^{100}\) circuits: nobody corrects the full distribution, they correct the marginals the observables need. The cost of the exact method is a cost of correcting the <em>distribution</em>, which is a much stronger thing to ask for.</p>

<p><strong>4.</strong> Yes, for a genuinely single-qubit observable. The correlated flip of Example 3 changes each qubit's marginal error rates — which is exactly what the marginal calibration measures — so a single-qubit correction calibrated from marginals is consistent. It is the <em>joint</em> observable \(\langle Z_0 Z_1 \rangle\) that the tensored model gets wrong, because a correlated flip leaves the marginals intact in form while changing the correlations. Correlated noise breaks correlated observables first.</p>

```python
for q in range(N):
    e01, e10 = RATES[q]
    bit = 2 ** (N - 1 - q)
    m_true = float(sum(p_true[m] for m in range(2 ** N) if m & bit))
    m_meas = float(sum(M_true[m, t] * p_true[t] for m in range(2 ** N)
                       for t in range(2 ** N) if m & bit))
    corr = (m_meas - e01) / (1.0 - e01 - e10)
    print(f"  qubit {q}: P(1) true {m_true:.4f}  measured {m_meas:.4f}"
          f"  corrected {corr:.6f}")
#   qubit 0: P(1) true 0.5000  measured 0.4850  corrected 0.500000
#   qubit 1: P(1) true 0.5000  measured 0.4850  corrected 0.500000
#   qubit 2: P(1) true 0.5000  measured 0.4800  corrected 0.500000
```

</details>

#### Exercise 2: Fractional Noise Scales

Integer local folding reaches only $\lambda = 1, 3, 5, \ldots$. Fractional scales are obtained by folding only the first $m$ gates.

  1. Implement partial folding and confirm that the achieved $\lambda = (\lvert C \rvert + 2m)/\lvert C \rvert$ and that the unitary is preserved.
  2. Run the Example 5 extrapolation on $\lambda = \lbrace 1, 1.5, 2, 3 \rbrace$ instead of $\lbrace 1,3,5,7 \rbrace$. What happens to the Richardson weights?
  3. Which set would you use, and on what grounds?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Requesting \(\lambda = 1.5\) on a 22-gate circuit needs \(m = 5.5\), which rounds to 6 and delivers \(\lambda = 34/22 = 1.545\) — the achievable scales are quantised in steps of \(2/\lvert C \rvert\). The equivalence error is \(10^{-16}\) at every scale, so the transformation is still exact.</p>

<p><strong>2.</strong> They blow up: \(\lVert w \rVert^2 = 419\) for \(\lbrace 1, 1.5, 2, 3\rbrace\) against 11.4 for \(\lbrace 1,3,5,7\rbrace\), a factor of 37 in variance and 6 in standard deviation. Exact polynomial interpolation through closely spaced points is an ill-conditioned operation, and clustering the noise scales is exactly how to make it worse. The bias also degrades, to \(+0.072\) against \(-0.0005\), because the lever arm is short: the fit has to extrapolate from \(\lambda \in [1,3]\) rather than \([1,7]\).</p>

<p><strong>3.</strong> The wide set, for a high-order extrapolation. The narrow set is defensible only with a low-order estimator — the least-squares line on \(\lbrace 1,1.5,2,3\rbrace\) has \(\lVert w \rVert^2 = 1.86\) and a bias of \(+0.018\), which is a reasonable operating point. The general rule is that the extrapolation order and the spread of the scales must be chosen together; fractional folding is useful for keeping the largest \(\lambda\) small when the deepest circuit would otherwise depolarize completely.</p>

```python
def fold_partial(circ, k, m):
    """Fold every gate k times and the first m gates once more: lambda = 2k+1+2m/|C|."""
    out = []
    for i, g in enumerate(circ):
        reps = k + (1 if i < m else 0)
        out.append(g)
        for _ in range(reps):
            out.extend(invert_gate(g))
            out.append(g)
    return out


for lam_t in (1.5, 2.0, 2.5):
    m = int(round((lam_t - 1.0) * len(CIRC) / 2.0))
    fc = fold_partial(CIRC, 0, m)
    print(f"  target lambda {lam_t:4.2f}: m = {m:2d}, gates {len(fc):3d},"
          f" achieved lambda {len(fc) / len(CIRC):5.3f},"
          f" equivalence error {assert_equivalent(CIRC, fc, NQ):.1e}")
LAMS_F = np.array([1.0, 1.5, 2.0, 3.0])
vals_f = np.array([energy(noisy_rho(fold_partial(CIRC, 0,
                   int(round((l - 1) * len(CIRC) / 2))), NQ, 0.005))
                   for l in LAMS_F])
wr, wl = richardson_weights(LAMS_F), ls_line_weights(LAMS_F)
print(f"  Richardson bias {float(wr @ vals_f) - E0:+.5f}, ||w||^2 {float(wr @ wr):.3f}")
print(f"  LS line    bias {float(wl @ vals_f) - E0:+.5f}, ||w||^2 {float(wl @ wl):.3f}")
#   target lambda 1.50: m =  6, gates  34, achieved lambda 1.545, equivalence error 3.4e-16
#   target lambda 2.00: m = 11, gates  44, achieved lambda 2.000, equivalence error 7.0e-16
#   target lambda 2.50: m = 16, gates  54, achieved lambda 2.455, equivalence error 5.7e-16
#   Richardson bias +0.07236, ||w||^2 419.000
#   LS line    bias +0.01843, ||w||^2 1.857
```

</details>

#### Exercise 3: A Coherent Error That Folding Cannot See

Replace the depolarizing noise by a purely unitary, systematic error: every $R_z(\theta)$ in the circuit is implemented as $R_z(\theta + \delta)$ for some small $\delta$. Consider two versions: $\delta$ carrying the sign of $\theta$ (a scale-factor error), and $\delta$ with a fixed sign (a zero-offset error).

  1. What does local folding do to each of the two errors? Answer analytically first.
  2. Run the Example 5 extrapolation on both and report the raw and extrapolated bias.
  3. What does this imply about reporting a ZNE result?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Local folding of \(R_z(\theta)\) emits \(R_z(\theta)R_z(-\theta)R_z(\theta)\). With a scale-factor error the three become \(R_z(\theta+\delta)\), \(R_z(-\theta-\delta)\), \(R_z(\theta+\delta)\), whose product is \(R_z(\theta+\delta)\) — <em>identical to the unfolded gate</em>. Folding cancels the error exactly and amplifies nothing. With a zero-offset error the three become \(R_z(\theta+\delta)\), \(R_z(-\theta+\delta)\), \(R_z(\theta+\delta)\), whose product is \(R_z(\theta+3\delta)\): the error is amplified by exactly \(\lambda\), which is what ZNE wants.</p>

<p><strong>2.</strong> Both models give the same raw bias, \(-0.00153\), because every angle in this circuit is positive so the two coincide at \(\lambda = 1\). Under folding they diverge completely. The scale-factor error gives the same energy at all four scales to five decimals, so both extrapolators return \(-0.00153\): <strong>ZNE reports the biased value and there is nothing in the output that says so.</strong> The offset error does vary with \(\lambda\), and Richardson brings the bias to \(+0.00052\) while the least-squares line overshoots to \(-0.034\), 22 times worse than doing nothing — because the \(\lambda\) dependence of a coherent error is trigonometric rather than polynomial, and a straight line is the wrong model for it.</p>

<p><strong>3.</strong> That the flatness of the \(\lambda\) sweep must be reported, and that a flat sweep is not evidence of a small error. It is equally consistent with a large error that folding is blind to. The honest protocol is to report the noisy values at every scale, not just the extrapolated one, and to cross-check with an independent method — PEC with a characterised noise model, or a Clifford circuit whose exact answer is known.</p>

```python
DELTA = 0.05


def coherent_rho(circ, n, delta, proportional):
    """Every rz gets an extra delta: a purely unitary, systematic error."""
    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rho[0, 0] = 1.0
    for g in circ:
        U, tg = gate_matrix(g)
        if g[0] == "rz":
            U = rz(g[1] + (np.sign(g[1]) if proportional else 1.0) * delta)
        rho = rho_apply(rho, U, tg, n)
    return rho


for prop in (True, False):
    vals_c = np.array([energy(coherent_rho(fold_local(CIRC, (l - 1) // 2), NQ,
                                           DELTA, prop)) for l in LAMS])
    rich = float(richardson_weights(LAMS) @ vals_c)
    line = float(ls_line_weights(LAMS) @ vals_c)
    print(f"  proportional = {prop}: energies {np.round(vals_c, 5).tolist()}")
    print(f"    raw bias {vals_c[0] - E0:+.5f}   Richardson bias {rich - E0:+.5f}"
          f"   LS line bias {line - E0:+.5f}")
#   proportional = True: energies [-1.50918, -1.50918, -1.50918, -1.50918]
#     raw bias -0.00153   Richardson bias -0.00153   LS line bias -0.00153
#   proportional = False: energies [-1.50918, -1.4911, -1.44773, -1.38424]
#     raw bias -0.00153   Richardson bias +0.00052   LS line bias -0.03405
```

</details>

#### Exercise 4: The PEC Wall-Clock Budget

Assume a base budget of $10^6$ shots for the unmitigated estimate and a throughput of $10^4$ shots per second.

  1. At $p = 10^{-3}$ and $p = 3\times10^{-4}$, compute the wall-clock time for PEC on circuits with $10^2$, $10^3$ and $10^4$ noise locations.
  2. What is the largest circuit that fits in one day at each error rate?
  3. A tenfold improvement in the physical error rate buys what factor in circuit size?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> At \(p = 10^{-3}\) the times are \(10^{2.17}\) s (2.5 minutes), \(10^{3.74}\) s (15 hours) and \(10^{19.4}\) s — \(10^{11.9}\) years. At \(p = 3\times10^{-4}\) the same three are \(10^{2.05}\) s, \(10^{2.52}\) s (6 minutes) and \(10^{7.2}\) s, about six months. The middle column is the interesting one: a thousand-gate circuit is a day's work at one error rate and a coffee break at the other.</p>

<p><strong>2.</strong> Solving \(\gamma^{2N} = 86400 \times 10^4 / 10^6\) gives \(N = 1690\) noise locations at \(p = 10^{-3}\) and \(N = 5634\) at \(p = 3\times10^{-4}\).</p>

<p><strong>3.</strong> A factor of about 3.3 in circuit size, not a factor of 10. Since \(\log \gamma \approx 2p\), the affordable \(N\) scales as \(1/p\) at fixed budget — so the 3.3-fold improvement in \(p\) here (from \(10^{-3}\) to \(3\times10^{-4}\)) buys the same 3.3-fold in \(N\), and a full decade in \(p\) would buy a decade in \(N\). That is <em>linear</em> in the error rate against a target that needs many orders of magnitude, which is the quantitative form of the statement that mitigation does not scale. Compare with error correction, where the same decade in \(p\) moves the required distance by a couple of steps and the circuit size by an unbounded factor.</p>

```python
RATE, base = 1e4, 1e6
for p in (1e-3, 3e-4):
    _, _, g = pec_coeffs(p)
    print(f"  p = {p:.0e}, gamma = {g:.6f}")
    for nloc in (1e2, 1e3, 1e4):
        log_secs = np.log10(base / RATE) + 2 * nloc * np.log10(g)
        print(f"    N = {nloc:6.0e}: log10 seconds {log_secs:8.2f},"
              f" log10 years {log_secs - np.log10(3.156e7):8.2f}")
    print(f"    largest N in one day:"
          f" {np.log10(86400 * RATE / base) / (2 * np.log10(g)):.0f}")
#   p = 1e-03, gamma = 1.002003
#     N =  1e+02: log10 seconds     2.17, log10 years    -5.33
#     N =  1e+03: log10 seconds     3.74, log10 years    -3.76
#     N =  1e+04: log10 seconds    19.38, log10 years    11.88
#     largest N in one day: 1690
#   p = 3e-04, gamma = 1.000600
#     N =  1e+02: log10 seconds     2.05, log10 years    -5.45
#     N =  1e+03: log10 seconds     2.52, log10 years    -4.98
#     N =  1e+04: log10 seconds     7.21, log10 years    -0.29
#     largest N in one day: 5634
```

</details>

#### Exercise 5: Which Input Does a Resource Estimate Actually Depend On

Using the pipeline of Example 8 at $\lambda = 1000$ Hartree, $C_{\text{walk}} = 3\times10^4$, $p_{\text{phys}} = 10^{-3}$ and 1152 logical qubits:

  1. Relax the target precision from chemical accuracy to ten times coarser. What changes?
  2. Vary $\lambda$ by a factor of two either way. What changes?
  3. Improve $p_{\text{phys}}$ from $10^{-3}$ to $10^{-4}$. What changes?
  4. Which of the two surface-code conventions in the sister courses would change a headline number, and by how much?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Everything scales down by ten: \(2.95\times10^{9}\) Toffolis instead of \(2.95\times10^{10}\), 0.34 days instead of 3.41, and the code distance drops from 21 to 19 because the logical error budget per gate relaxes by a decade. Relaxing \(\varepsilon\) is the single cheapest saving available in a resource estimate, and it is physically legitimate whenever the quantity of interest is an energy <em>difference</em> whose systematic errors cancel.</p>

<p><strong>2.</strong> Runtime scales linearly — 1.70, 3.41, 6.82 days for \(\lambda = 500, 1000, 2000\) — while the code distance stays at 21 throughout, because \(d\) depends on the logarithm of the Toffoli count and a factor of four is less than one distance step. So \(\lambda\) sets the wall clock and does almost nothing to the qubit count.</p>

<p><strong>3.</strong> The distance falls from 21 to 11 and the physical qubit count from \(1.01\times10^{6}\) to \(2.78\times10^{5}\), a factor of 3.6, while the runtime does not change at all. Physical error rate is a <em>qubit-count</em> lever, not a runtime lever; precision and 1-norm are runtime levers. Confusing the two is how resource-estimate discussions go wrong.</p>

<p><strong>4.</strong> At \(d = 21\) the two conventions give 882 and 881 physical qubits per logical qubit, and on 1152 logical qubits \(1.016\times10^{6}\) against \(1.015\times10^{6}\) — a difference of 0.1%, invisible at the two significant figures any such estimate deserves. It is worth checking precisely because it is invisible: a discrepancy of this size between two published estimates is a convention difference and not a disagreement, and mistaking it for one wastes time.</p>

```python
for eps, lab in ((1.6e-3, "chemical accuracy"), (1.6e-2, "10x coarser")):
    r = estimate(1000.0, eps, 3e4, 1e-3, 1152)
    print(f"  {lab:>18}: Toffolis {r['toffolis']:.2e}, d = {r['distance']},"
          f" physical {r['physical']:.2e}, days {r['days']:.2f}")
for lam in (500.0, 1000.0, 2000.0):
    r = estimate(lam, 1.6e-3, 3e4, 1e-3, 1152)
    print(f"  lambda = {lam:6.0f}: Toffolis {r['toffolis']:.2e},"
          f" d = {r['distance']}, days {r['days']:.2f}")
for p in (1e-3, 1e-4):
    r = estimate(1000.0, 1.6e-3, 3e4, p, 1152)
    print(f"  p_phys = {p:.0e}: d = {r['distance']},"
          f" per logical {r['per_logical']}, physical {r['physical']:.2e}")
print(f"  2d^2 vs 2d^2-1 at d = 21: {2*21*21} vs {2*21*21-1},"
      f" on 1152 logical qubits {2*21*21*1152:.3e} vs {(2*21*21-1)*1152:.3e}")
#    chemical accuracy: Toffolis 2.95e+10, d = 21, physical 1.01e+06, days 3.41
#          10x coarser: Toffolis 2.95e+09, d = 19, physical 8.31e+05, days 0.34
#   lambda =    500: Toffolis 1.47e+10, d = 21, days 1.70
#   lambda =   1000: Toffolis 2.95e+10, d = 21, days 3.41
#   lambda =   2000: Toffolis 5.89e+10, d = 21, days 6.82
#   p_phys = 1e-03: d = 21, per logical 881, physical 1.01e+06
#   p_phys = 1e-04: d = 11, per logical 241, physical 2.78e+05
#   2d^2 vs 2d^2-1 at d = 21: 882 vs 881, on 1152 logical qubits 1.016e+06 vs 1.015e+06
```

</details>

* * *

## Summary

### Key Takeaways

**1\. Readout error is classical, large, and correctable — up to a point**

  * $\mathbf{p}_{\text{meas}} = M\mathbf{p}_{\text{true}}$ with $M$ measurable from $2^n$ calibration circuits; the raw error in $\langle Z_0 Z_1 \rangle$ on a GHZ state was $-0.147$, a hundred times a good gate error. $M^{-1}$ cuts the TVD from 0.142 to 0.017 and returns negative probabilities; constrained least squares over the simplex reaches 0.010 and stays physical.
  * The exact method costs $2^n$ circuits — $1.1\times10^{15}$ at fifty qubits — and $\mathrm{cond}(M)$ grows as $\mathrm{cond}(M_q)^n$. Only $k$-local observables are corrected in practice, at $2^k$ circuits.
  * **A correction built on a wrong noise model fails in the wrong direction.** With 4% correlated flips, the tensored correction moved $\langle Z_0 Z_1 \rangle$ from $-0.153$ low to $+0.181$ high: worse than doing nothing.

**2\. Gate folding amplifies noise and provably changes nothing else**

  * $G \to G(G^{-1}G)^k$ multiplies gates and depth by $\lambda = 2k+1$; the phase-free error against the original circuit stayed at $10^{-16}$ for every scale, and $S^{-1} = ZS$, $T^{-1} = ZST$ keep the folded circuit inside the gate set.
  * Integer local folding reaches only odd $\lambda$; fractional scales come from folding a subset, quantised in steps of $2/\lvert C \rvert$. Local and global folding differ as gate lists but agreed to $3.2\times10^{-4}$ under depolarizing noise.

**3\. Extrapolation removes bias by a large factor and pays in variance by a computable one**

  * Cubic Richardson on $\lambda = 1,3,5,7$ cut the noise bias 143-fold at $p = 0.002$, 212-fold at $0.005$ and 113-fold at $0.01$; the least-squares line on the same data managed 6.1, 2.7 and 1.7.
  * Richardson weights are exact interpolation: $(2,-1)$, $(3,-3,1)$, $(4,-6,4,-1)$ with $\lVert w \rVert_2 = 2.24, 4.36, 8.31$ — the sister course's numbers. The $1,3,5,7$ family gives $\lVert w \rVert^2 = 11.4$, the least-squares line $1.05$, and measured standard deviations matched $\sigma_{\text{raw}}\lVert w \rVert_2$ to a few percent at every budget.
  * **The best estimator depends on the budget.** At 1000 shots per term the RMSE ranking was LS line 0.088, raw 0.131, Richardson 0.254; at 100 000 it was Richardson 0.028, LS line 0.040, raw 0.106.

**4\. PEC is unbiased and exponentially expensive, and both are exactly quantifiable**

  * The depolarizing inverse is $\alpha\,\mathrm{id} + \beta\sum_P P \cdot P$ with $\beta < 0$ and $\gamma = 1 + \tfrac{3}{2}(1/f - 1) \approx 1 + 2p$; composed with the channel it returns the input to $10^{-16}$. Implemented on a 4-location circuit at $p = 0.02$ it removed a $-0.060$ bias to $-0.007 \pm 0.005$, with a single-sample standard deviation of 0.334 and 7.6% of samples carrying a negative sign.
  * The shot overhead is $\gamma^{2N}$: a factor of 1.6 at 28 noise locations and $p = 0.005$, $10^{8.7}$ at a thousand, $10^{87}$ at ten thousand, $10^{8700}$ at a million.

**5\. Mitigation is exponential, correction is not, and the crossover is early**

  * At $p = 10^{-3}$: 100 gates cost $10^{0.2}$ in PEC samples against distance 3 and 17 physical qubits per logical qubit; $10^{4}$ gates cost $10^{17.4}$ against 97; $10^{11}$ gates cost $10^{1.7\times10^{8}}$ against 881. The crossover is a few thousand gates.
  * Mitigation corrects expectation values and not states, so it cannot support a coherent subroutine, and it cannot handle noise its model omits — Exercise 3's scale-factor error is invisible to folding, and extrapolation returns the biased value with no warning.
  * **And mitigation is load-bearing today.** It removed a $-0.147$ readout bias and 99.3% of a gate-noise bias on circuits current hardware can run. Both facts belong in the same paragraph.

**6\. A resource estimate is four functions and a floating-point guard**

  * $(\lambda, \varepsilon, C_{\text{walk}}) \to N_{\text{Toffoli}} \to p_L = 0.1/N \to d \to (2d^2-1)n_{\text{logical}}$, plus 10 $\mu$s per logical Toffoli.
  * The threshold comparison needs a relative tolerance: `0.1 * 0.1 ** 5` is `1.0000000000000004e-06`, and an unguarded `<=` returned $d = 11$ instead of 9 and $d = 23$ instead of 21 — hundreds of physical qubits per logical qubit.
  * FeMoco-scale at $\lambda = 1000$, $C_{\text{walk}} = 3\times10^4$: $2.95\times10^{10}$ Toffolis, $p_L = 3.4\times10^{-12}$, $d = 21$, $10^{6}$ physical qubits, 3.4 days — matching the intermediate algorithms course to the digit.
  * $\varepsilon$ and $\lambda$ set the runtime; $p_{\text{phys}}$ sets the qubit count. A decade in $p_{\text{phys}}$ is a factor of 3.6 in qubits and nothing in time.

**Practical implications**

  * Correct readout error first. It is the largest, the cheapest and the only one that can be inverted exactly, and correcting the marginals your observables need costs $2^k$ circuits rather than $2^n$.
  * Never report a mitigated expectation value without the unmitigated values at every noise scale and the shot budget. Without those, a flat sweep and a genuinely low-noise device look identical.
  * Choose the extrapolation order and the shot budget together, and compute $\lVert w \rVert^2$ before running anything. It is the whole variance penalty and it is available in advance.
  * Before proposing mitigation for a circuit, compute $\gamma^{2N}$. If the exponent has more than a couple of digits, the answer is a fault-tolerance question, not a post-processing question.
  * In any resource estimate, put a relative tolerance on every comparison of a power to a power, state the surface-code convention, and quote $\lambda$, $\varepsilon$ and $p_{\text{phys}}$ alongside the answer. The deliverable is the exponent and the assumptions, never the mantissa.

### Where This Leads

This is the end of the course, and the end of the descent it set out to make: an algorithm becomes a circuit, a circuit becomes a shorter circuit, a shorter circuit becomes one that fits the hardware graph, that circuit becomes a schedule of shaped pulses, and the measurements that come back become an estimate that has had its noise argued with. Nothing in any of those five layers was an SDK feature. Every one of them was a few dozen lines of NumPy with a test attached, and that is the claim the course was built to support: the layers are simple enough to build, and building one is the way to understand the documentation of the ones you will actually use.

Where to go next is now a question about your own problem rather than about the stack. With a Hamiltonian, the route runs through [Intermediate Quantum Algorithms](<../quantum-algorithms-intermediate/index.html>) to a Toffoli count and then through Section 5.5 to a number of physical qubits. With a device, it runs through [Introduction to Quantum Hardware](<../quantum-hardware-introduction/index.html>) to the materials question underneath its error rates. With a dataset, [Quantum Machine Learning](<../../MI/quantum-machine-learning-introduction/index.html>) has the evaluation discipline needed to avoid fooling yourself. And if what you want is a quantum technology that works today, [Quantum Sensing](<../../MS/quantum-sensing-introduction/index.html>) is the one course here whose answer to "is there an advantage" is yes.

[← Chapter 4: Pulses and Calibration](<chapter-4.html>) [Back to Series Index →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The readout error rates, gate error rates, code distances, Toffoli counts, cycle times and wall-clock figures in this chapter are illustrative order-of-magnitude values chosen to demonstrate the arithmetic of mitigation and of a resource estimate; they are not device specifications or published estimates and must be checked against primary sources before use in a proposal or a paper.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
