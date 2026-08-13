---
title: "Chapter 2: QFT and Phase Estimation"
chapter_title: "Chapter 2: QFT and Phase Estimation"
subtitle: An O(n²) Circuit Whose Output Nobody Can Read, and the Eigenvalue Algorithm Built Out of It
reading_time: 45-50 minutes
difficulty: Intermediate
code_examples: 7
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-algorithms-intermediate/chapter-2.html>) | Last sync: 2026-08-13

[Fundamental Mathematics Dojo](<../index.html>) > [Intermediate Quantum Algorithms](<index.html>) > Chapter 2

Chapter 1 was about amplitude: Grover's algorithm moves probability towards a marked state and its speedup is quadratic, with all the qualifications that word deserves. This chapter is about phase, and the qualifications are different in kind. The **quantum Fourier transform** costs $O(n^2)$ gates where the classical fast Fourier transform on the same number of amplitudes costs $O(n 2^n)$, which sounds like an exponential speedup and is not one. The **phase estimation** algorithm built from it extracts an eigenvalue of a unitary to $t$ bits of precision using $2^t - 1$ controlled applications of that unitary, which sounds modest and is the single most important primitive in fault-tolerant quantum computing.

Getting both of those statements right is the work of this chapter. The QFT is not a faster FFT because its input must already be a quantum state and its output amplitudes cannot be read; what it can do is convert a *period* hidden in a state's structure into a *measurable* peak, and that is a much narrower and much more useful capability. Phase estimation is the machine that exploits it. Section 2.4 collects the consequence that matters for a materials researcher: an eigenvalue problem is exactly what electronic structure is, and phase estimation is the fault-tolerant successor to the variational algorithms of the introductory course — the thing that VQE is a near-term substitute for.

Everything here runs on the same NumPy state-vector simulator built in [Introduction to Quantum Computing, Chapter 2](<../quantum-computing-introduction/chapter-2.html>), re-listed below so that this chapter is self-contained. Chapter 3 then takes the QFT and the phase-estimation circuit constructed here and factors an integer with them, so the implementations below are load-bearing rather than illustrative.

## Learning Objectives

After completing this chapter, you will be able to:

  * Write the quantum Fourier transform as a product of Hadamards and controlled phase rotations, count its gates as $n(n+1)/2 + \lfloor n/2 \rfloor$, and verify the circuit against the dense DFT matrix numerically
  * State precisely why the QFT is not an exponentially faster FFT — the input-loading problem, the unreadability of output amplitudes, and the $1/\delta^2$ cost of estimating even one of them
  * Show that the QFT converts a period into a set of measurable peaks while leaving the offset entirely in the phases, and explain what happens when the period does not divide $2^n$
  * Derive the phase-estimation circuit from controlled powers of $U$ plus an inverse QFT, and use the standard bound $t = n + \lceil \log_2(2 + 1/(2\varepsilon)) \rceil$ to choose a register size for a target accuracy $2^{-n}$
  * Recover the eigenphases of two-, three- and four-qubit unitaries to arbitrary precision, and demonstrate that the phase error halves with each added counting qubit while the depth doubles
  * Implement iterative phase estimation with a single ancilla and classical feedback, and state what it trades against the textbook circuit
  * Explain why phase estimation is the fault-tolerant route to electronic structure, and why the overlap of the trial state with the target eigenvector is the quantity that decides whether it works at all

* * *

## 2.1 The Quantum Fourier Transform

### The transform

The discrete Fourier transform of a vector of $2^n$ complex numbers is the unitary map

$$ \tilde{x}_k = \frac{1}{\sqrt{2^n}} \sum_{j=0}^{2^n-1} e^{2\pi i jk/2^n}\, x_j $$

The quantum Fourier transform is that same matrix, applied to the amplitudes of an $n$-qubit register. On a basis state it reads

$$ \mathrm{QFT}\, \lvert j \rangle = \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n-1} e^{2\pi i jk/2^n}\, \lvert k \rangle $$

and on a general state it acts by linearity. Nothing about the definition is quantum: it is the ordinary DFT, and the sign convention here — $e^{+2\pi i jk/2^n}$ in the forward direction — is the one this course uses throughout, so that the *inverse* QFT is the transform with the conjugated phases.

### Why the circuit is short

Write $k$ in binary as $k = \sum_{l=1}^{n} k_l 2^{n-l}$, so that $k/2^n = \sum_l k_l 2^{-l} = 0.k_1k_2\ldots k_n$. Then the exponential factorizes over the bits of $k$:

$$ \frac{1}{\sqrt{2^n}}\sum_k e^{2\pi i jk/2^n}\lvert k \rangle = \bigotimes_{l=1}^{n} \frac{\lvert 0 \rangle + e^{2\pi i j/2^{l}}\lvert 1 \rangle}{\sqrt{2}} $$

This is the whole content of the algorithm. The Fourier transform of a basis state is a **product state**, with no entanglement at all, and each factor needs only one phase that depends on the bits of $j$. Reading $j = 0.j_1 j_2 \ldots j_n$ in the same way, the $l$-th output factor carries the phase $2\pi \times 0.j_{n-l+1}\ldots j_n$, which is a Hadamard on qubit $n-l+1$ followed by a controlled phase rotation from each less significant qubit. Writing

$$ R_m = \begin{pmatrix} 1 & 0 \cr 0 & e^{2\pi i/2^m} \end{pmatrix} $$

the circuit is: Hadamard on qubit 1, controlled-$R_2$ from qubit 2, controlled-$R_3$ from qubit 3, and so on; then Hadamard on qubit 2 and its controlled rotations; and finally a reversal of the qubit order, because the construction produces the output bits backwards. The count is

$$ \underbrace{n}_{\text{Hadamards}} + \underbrace{\frac{n(n-1)}{2}}_{\text{controlled phases}} + \underbrace{\left\lfloor n/2 \right\rfloor}_{\text{swaps}} = \frac{n(n+1)}{2} + \left\lfloor \frac{n}{2} \right\rfloor $$

gates, and the depth can be reduced to $O(n)$ by running rotations on disjoint pairs in parallel. Two practical remarks. The phase $2\pi/2^m$ becomes unresolvably small for large $m$, and dropping all rotations with $m > O(\log n)$ changes the result by an amount that is provably harmless — this **approximate QFT** has $O(n\log n)$ gates and is what any real implementation uses. And the final swaps can usually be deleted entirely by relabelling the qubits in whatever consumes the output.

### What the QFT is not

The comparison that invites an overstatement is with the classical FFT, which needs $\Theta(n 2^n)$ arithmetic operations on $2^n$ numbers against the QFT's $\Theta(n^2)$ gates. Three separate facts block the conclusion.

**The input has to be there already.** The FFT takes $2^n$ numbers in memory. The QFT takes a quantum state whose amplitudes *are* those numbers. Preparing an arbitrary $2^n$-amplitude state from a classical list requires $\Theta(2^n)$ gates in general — the state-preparation problem, and the reason the QRAM question discussed in Chapter 1 is not a detail. If the state comes out of a previous quantum computation, as it does in Shor's algorithm, this cost does not arise; if the data is classical, it does, and it wipes out the advantage by itself.

**The output cannot be read.** After the QFT the transformed coefficients are amplitudes. A measurement returns one index $k$ with probability $\lvert \tilde{x}_k \rvert^2$, not the number $\tilde{x}_k$. Estimating a single probability to additive precision $\delta$ costs $O(1/\delta^2)$ repetitions by sampling — not a lower bound, since amplitude estimation reaches $O(1/\delta)$ by spending coherent depth instead; recovering all $2^n$ of them costs at least $2^n$ numbers of output, so no exponential saving can survive. Recovering the *phases* needs further interference experiments on top.

**Nothing useful is being asked for.** Even granting free input and free output, the classical FFT is rarely anyone's bottleneck: $n2^n$ with a tiny constant is one of the fastest algorithms in scientific computing. The QFT earns its place not by being a faster transform but by sitting inside a circuit that asks a *single* question of the transformed state — "where is the peak?" — which is exactly one measurement's worth of information.

| | Classical FFT | Quantum FFT (QFT) |
| --- | --- | --- |
| Input | $2^n$ numbers in memory | amplitudes of an $n$-qubit state |
| Cost | $\Theta(n 2^n)$ arithmetic ops | $\Theta(n^2)$ gates, or $\Theta(n \log n)$ approximate |
| Output | all $2^n$ transformed numbers | one sampled index per run |
| Reading one coefficient | free | $O(1/\delta^2)$ repetitions by sampling, magnitude only |
| Useful for | filtering, convolution, spectra | extracting a period or a phase, then measuring once |

The honest one-line summary: the QFT is not a transform you use to transform data. It is a change of basis in which a hidden periodicity becomes the answer to a single measurement.

### Code Example 1: The Simulator, Re-listed

The whole of this course runs on the state-vector simulator of the introductory series. It is reproduced here verbatim — all ninety-nine lines, unchanged — so that nothing in this chapter depends on a file you have to go and find. Save it as `qcsim.py`; every subsequent example begins with `from qcsim import *`.

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


def sample(state, shots, seed=None):
    """Simulated measurement: {bitstring: count}."""
    n = int(np.log2(state.size))
    rng = np.random.default_rng(seed)
    idx = rng.choice(state.size, size=shots, p=probs(state))
    out = {}
    for i in idx:
        b = format(i, f'0{n}b')
        out[b] = out.get(b, 0) + 1
    return dict(sorted(out.items()))


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
```

Two conventions from that file are used constantly below and are worth restating. Qubit ordering is **big-endian**: qubit 0 is the leftmost symbol in a ket and the most significant bit of the amplitude index, so on a $t$-qubit counting register the measured integer is $k = \sum_j q_j 2^{t-1-j}$. And `apply_gate` accepts a $2^k \times 2^k$ matrix acting on any $k$ named qubits, which is what makes controlled multi-qubit operations below a one-liner rather than an exercise in index arithmetic.

### Code Example 2: The QFT Circuit Against the DFT Matrix

A circuit claim deserves a matrix check. The following builds the QFT from Hadamards and controlled phases exactly as derived above, then extracts the matrix it implements by running it on every basis state and compares that, column by column, with the dense DFT matrix.

```python
import numpy as np
from qcsim import *

SWAP4 = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=complex)


def cphase(theta):
    """Controlled phase gate diag(1, 1, 1, exp(i theta)) on a qubit pair."""
    return np.diag([1.0, 1.0, 1.0, np.exp(1j * theta)]).astype(complex)


def qft(state, qubits, n):
    """QFT on the listed qubits; qubits[0] is the most significant bit.

    Hadamard on each qubit in turn, then a controlled phase from every less
    significant qubit, then a reversal of the qubit order. Gate count is
    m(m+1)/2 rotations plus floor(m/2) swaps.
    """
    m = len(qubits)
    for j in range(m):
        state = apply_gate(state, H, [qubits[j]], n)
        for k in range(j + 1, m):
            state = apply_gate(state, cphase(np.pi / 2 ** (k - j)),
                               [qubits[k], qubits[j]], n)
    for j in range(m // 2):
        state = apply_gate(state, SWAP4, [qubits[j], qubits[m - 1 - j]], n)
    return state


def iqft(state, qubits, n):
    """Inverse QFT: the same gates in reverse order with conjugated phases."""
    m = len(qubits)
    for j in range(m // 2):
        state = apply_gate(state, SWAP4, [qubits[j], qubits[m - 1 - j]], n)
    for j in reversed(range(m)):
        for k in reversed(range(j + 1, m)):
            state = apply_gate(state, cphase(-np.pi / 2 ** (k - j)),
                               [qubits[k], qubits[j]], n)
        state = apply_gate(state, H, [qubits[j]], n)
    return state


def qft_matrix_from_circuit(m):
    """Column j is the circuit's output on the basis state |j>."""
    cols = []
    for j in range(2 ** m):
        psi = np.zeros(2 ** m, dtype=complex)
        psi[j] = 1.0
        cols.append(qft(psi, list(range(m)), m))
    return np.column_stack(cols)


def dft_matrix(m):
    """The unitary DFT matrix F[k, j] = exp(2 pi i j k / 2^m) / sqrt(2^m)."""
    d = 2 ** m
    j, k = np.meshgrid(np.arange(d), np.arange(d))
    return np.exp(2j * np.pi * j * k / d) / np.sqrt(d)


print("QFT circuit against the DFT matrix")
print("-" * 68)
print(f"  {'m':>3}{'dim':>7}{'H+phase gates':>14}{'swaps':>7}"
      f"{'max |U_circ - F|':>20}")
for m in range(1, 8):
    U = qft_matrix_from_circuit(m)
    F = dft_matrix(m)
    n_hp = m * (m + 1) // 2          # m Hadamards + m(m-1)/2 phase gates
    print(f"  {m:>3}{2**m:>7}{n_hp:>14}{m//2:>7}"
          f"{np.max(np.abs(U - F)):>20.2e}")

print("\nUnitarity and inversion, m = 5")
print("-" * 68)
m = 5
U = qft_matrix_from_circuit(m)
print(f"  max |U^dag U - I|            = "
      f"{np.max(np.abs(U.conj().T @ U - np.eye(2**m))):.2e}")
rng = np.random.default_rng(7)
psi = rng.normal(size=2 ** m) + 1j * rng.normal(size=2 ** m)
psi /= np.linalg.norm(psi)
back = iqft(qft(psi, list(range(m)), m), list(range(m)), m)
print(f"  max |iqft(qft(psi)) - psi|   = {np.max(np.abs(back - psi)):.2e}")
fwd = qft(psi, list(range(m)), m)
print(f"  max |qft(psi) - F psi|       = "
      f"{np.max(np.abs(fwd - dft_matrix(m) @ psi)):.2e}")

print("\nQFT of a uniform state and of a single basis state, m = 3")
print("-" * 68)
m = 3
unif = np.ones(2 ** m, dtype=complex) / np.sqrt(2 ** m)
out = qft(unif, list(range(m)), m)
print("  QFT|+++>  amplitudes:",
      "  ".join(f"{v if abs(v) > 5e-4 else 0.0:+.3f}" for v in out.real))
out = qft(ket('001'), list(range(m)), m)
print("  QFT|001>  |amp|     :",
      "  ".join(f"{abs(v):.3f}" for v in out))
print("  QFT|001>  phase/2pi :",
      "  ".join(f"{np.angle(v)/(2*np.pi) % 1.0:.3f}" for v in out))

print("\nGate count: QFT versus a classical FFT on 2^m numbers")
print("-" * 68)
print(f"  {'m':>4}{'2^m':>22}{'QFT gates':>12}{'FFT ops ~ m 2^m':>22}")
for m in [3, 10, 20, 30, 50]:
    print(f"  {m:>4}{2**m:>22d}{m*(m+1)//2 + m//2:>12d}{m*2**m:>22d}")
```

```text
QFT circuit against the DFT matrix
--------------------------------------------------------------------
    m    dim H+phase gates  swaps    max |U_circ - F|
    1      2             1      0            8.66e-17
    2      4             3      1            2.69e-16
    3      8             6      1            1.42e-15
    4     16            10      2            3.78e-15
    5     32            15      2            5.62e-15
    6     64            21      3            7.98e-15
    7    128            28      3            1.21e-14

Unitarity and inversion, m = 5
--------------------------------------------------------------------
  max |U^dag U - I|            = 9.99e-16
  max |iqft(qft(psi)) - psi|   = 3.86e-16
  max |qft(psi) - F psi|       = 2.73e-15

QFT of a uniform state and of a single basis state, m = 3
--------------------------------------------------------------------
  QFT|+++>  amplitudes: +1.000  +0.000  +0.000  +0.000  +0.000  +0.000  +0.000  +0.000
  QFT|001>  |amp|     : 0.354  0.354  0.354  0.354  0.354  0.354  0.354  0.354
  QFT|001>  phase/2pi : 0.000  0.125  0.250  0.375  0.500  0.625  0.750  0.875

Gate count: QFT versus a classical FFT on 2^m numbers
--------------------------------------------------------------------
     m                   2^m   QFT gates       FFT ops ~ m 2^m
     3                     8           7                    24
    10                  1024          60                 10240
    20               1048576         220              20971520
    30            1073741824         480           32212254720
    50      1125899906842624        1300     56294995342131200
```

**What to look for.** The circuit reproduces the DFT matrix to $10^{-14}$ at $m = 7$, with the deviation growing only as accumulated rounding — this is a verification, not an approximation. The gate count column is the $m(m+1)/2$ predicted above, and `iqft(qft(psi))` returns the input to $4 \times 10^{-16}$, confirming that the reversed-order conjugated-phase construction really is the inverse.

The last two blocks are the ones to remember. $\mathrm{QFT}\lvert +++ \rangle = \lvert 000 \rangle$: the uniform superposition is the zero-frequency state, exactly as in the classical transform. $\mathrm{QFT}\lvert 001 \rangle$ has *uniform magnitude* over all eight outcomes, with the information entirely in the phases, which advance by $1/8$ of a turn per index. A basis state and its transform are the two extremes of localization, and the second one is unmeasurable in the computational basis — a preview of the section above, in three lines of output. Finally, at $m = 50$ the QFT needs 1300 gates where the FFT needs $5.6 \times 10^{16}$ operations; the comparison is real arithmetic and still does not amount to a useful speedup, for the three reasons already given.

### Code Example 3: A Period You Can Read and a Phase You Cannot

The capability the QFT actually provides is period detection. This example prepares a state supported on an arithmetic progression of stride $r$, transforms it, and looks at what a measurement returns — first when $r$ divides the register size and then when it does not, which is the situation Chapter 3 has to live with.

```python
"""Chapter 2, Example 3: what the QFT gives you, and what it does not.
Continues from Example 2 (same session)."""

import numpy as np
import matplotlib.pyplot as plt


def periodic_state(m, r, offset):
    """Uniform superposition over {j : j = offset (mod r)} in a 2^m register."""
    psi = np.zeros(2 ** m, dtype=complex)
    psi[np.arange(offset, 2 ** m, r)] = 1.0
    return psi / np.linalg.norm(psi)


m = 6
print(f"Case 1: period r = 8 divides 2^m = {2**m}")
print("-" * 70)
p0 = probs(qft(periodic_state(m, 8, 0), list(range(m)), m))
print("  nonzero probabilities of the QFT output (offset s = 0):")
print("   " + "   ".join(f"k={k}: {p0[k]:.4f}"
                         for k in np.flatnonzero(p0 > 1e-9)))
print("\n  the same distribution for every offset s:")
for s in range(1, 8):
    ps = probs(qft(periodic_state(m, 8, s), list(range(m)), m))
    print(f"    s = {s}: max |p_s - p_0| = {np.max(np.abs(ps - p0)):.2e}")

print("\n  the offset lives in the phases, which no measurement returns:")
a0 = qft(periodic_state(m, 8, 0), list(range(m)), m)
a3 = qft(periodic_state(m, 8, 3), list(range(m)), m)
print(f"  {'k':>4}{'|amp| s=0':>12}{'|amp| s=3':>12}"
      f"{'phase diff/2pi':>17}{'k s / 2^m mod 1':>18}")
for k in np.flatnonzero(p0 > 1e-9):
    d = (np.angle(a3[k]) - np.angle(a0[k])) / (2 * np.pi) % 1.0
    print(f"  {k:>4}{abs(a0[k]):>12.4f}{abs(a3[k]):>12.4f}{d:>17.4f}"
          f"{(k*3/2**m) % 1.0:>18.4f}")

print(f"\nCase 2: period r = 5 does not divide 2^m = {2**m}")
print("-" * 70)
out5 = qft(periodic_state(m, 5, 0), list(range(m)), m)
p5 = probs(out5)
top = np.sort(np.argsort(p5)[::-1][:5])
print(f"  2^m / r = {2**m/5:.2f}; the five largest peaks:")
print("   " + "   ".join(f"k={k}: {p5[k]:.4f}" for k in top))
print(f"  total probability in those five bins: {p5[top].sum():.4f}")
print("  peaks near, not at, multiples of 12.80 -- the leakage that "
      "Chapter 3 has to postprocess")

print("\nCost of reading the output amplitudes")
print("-" * 70)
rng = np.random.default_rng(11)
psi = rng.normal(size=2 ** m) + 1j * rng.normal(size=2 ** m)
psi /= np.linalg.norm(psi)
p_exact = probs(qft(psi, list(range(m)), m))
print(f"  {'shots':>10}{'max |p_hat - p|':>18}{'1/sqrt(shots)':>16}")
for shots in [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]:
    counts = rng.multinomial(shots, p_exact) / shots
    print(f"  {shots:>10d}{np.max(np.abs(counts - p_exact)):>18.5f}"
          f"{1/np.sqrt(shots):>16.5f}")
print("\n  shots to pin all 2^m probabilities to 1 per cent:"
      f" ~ m ln2/delta^2 = {m * np.log(2) * 10**4:.0e}")
print("  (logarithmic in the number of bins, not linear: the table above"
      " crosses")
print("  1 per cent nearer 3e+03 shots)")
print("  and the phases would need separate interference experiments.")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for s, style in [(0, "-o"), (3, "--s")]:
    ax[0].plot(probs(qft(periodic_state(m, 8, s), list(range(m)), m)),
               style, ms=4, lw=1, label=f"offset s = {s}")
ax[0].set_title("r = 8 divides 64: exact peaks")
ax[0].legend(fontsize=8)
ax[1].plot(p5, "-o", ms=4, lw=1, color="tab:red")
for j in range(1, 5):
    ax[1].axvline(j * 2 ** m / 5, color="k", ls=":", lw=0.8)
ax[1].set_title("r = 5 does not: peaks leak")
ax[2].bar(np.arange(2 ** m), p_exact, color="tab:purple", width=0.8)
ax[2].set_title("random state: nothing to read")
for a in ax:
    a.set_xlabel("measured index k"); a.set_ylabel("probability")
plt.tight_layout()
plt.show()
```

```text
Case 1: period r = 8 divides 2^m = 64
----------------------------------------------------------------------
  nonzero probabilities of the QFT output (offset s = 0):
   k=0: 0.1250   k=8: 0.1250   k=16: 0.1250   k=24: 0.1250   k=32: 0.1250   k=40: 0.1250   k=48: 0.1250   k=56: 0.1250

  the same distribution for every offset s:
    s = 1: max |p_s - p_0| = 1.60e-50
    s = 2: max |p_s - p_0| = 4.28e-50
    s = 3: max |p_s - p_0| = 2.14e-50
    s = 4: max |p_s - p_0| = 2.14e-50
    s = 5: max |p_s - p_0| = 2.14e-50
    s = 6: max |p_s - p_0| = 2.14e-50
    s = 7: max |p_s - p_0| = 4.81e-50

  the offset lives in the phases, which no measurement returns:
     k   |amp| s=0   |amp| s=3   phase diff/2pi   k s / 2^m mod 1
     0      0.3536      0.3536           0.0000            0.0000
     8      0.3536      0.3536           0.3750            0.3750
    16      0.3536      0.3536           0.7500            0.7500
    24      0.3536      0.3536           0.1250            0.1250
    32      0.3536      0.3536           0.5000            0.5000
    40      0.3536      0.3536           0.8750            0.8750
    48      0.3536      0.3536           0.2500            0.2500
    56      0.3536      0.3536           0.6250            0.6250

Case 2: period r = 5 does not divide 2^m = 64
----------------------------------------------------------------------
  2^m / r = 12.80; the five largest peaks:
   k=0: 0.2031   k=13: 0.1771   k=26: 0.1146   k=38: 0.1146   k=51: 0.1771
  total probability in those five bins: 0.7865
  peaks near, not at, multiples of 12.80 -- the leakage that Chapter 3 has to postprocess

Cost of reading the output amplitudes
----------------------------------------------------------------------
       shots   max |p_hat - p|   1/sqrt(shots)
         100           0.04393         0.10000
        1000           0.01644         0.03162
       10000           0.00339         0.01000
      100000           0.00210         0.00316
     1000000           0.00047         0.00100

  shots to pin all 2^m probabilities to 1 per cent: ~ m ln2/delta^2 = 4e+04
  (logarithmic in the number of bins, not linear: the table above crosses
  1 per cent nearer 3e+03 shots)
  and the phases would need separate interference experiments.
```

**What to look for.** For $r = 8$ in a 64-dimensional register the output is supported on exactly eight indices, the multiples of $2^m/r = 8$, each with probability $1/8$. Shifting the offset $s$ changes the distribution by $10^{-50}$ — that is, not at all. The offset is not lost, it is stored in the phases, and the table confirms that the phase difference between $s = 0$ and $s = 3$ is exactly $ks/2^m$ at every peak. **The QFT trades the offset for the period**, and a measurement collects the half of the information that is measurable.

The $r = 5$ block is the honest version. Five does not divide 64, the peaks land near but not on multiples of $12.8$, and only $79\%$ of the probability sits in the five best bins; the rest is spread over the other 59. Nothing is broken — the peaks are still where the period says they should be, to within one bin — but a classical postprocessing step is now needed to turn a measured index into a period, and that step is continued fractions in Chapter 3.

The last block prices the alternative, and the price is not the one the shape of the problem suggests. Estimating the transformed probabilities by sampling converges as $1/\sqrt{\text{shots}}$, and the number of shots that holds the *worst* of the $2^m$ bins to an absolute $\delta$ grows only logarithmically in the number of bins — $O(m/\delta^2)$, about $4 \times 10^4$ for $m = 6$ and $\delta = 0.01$, with the measured table crossing one per cent nearer $3 \times 10^3$. What makes the Fourier-engine reading hopeless is not that count but what an absolute $\delta$ buys: the probabilities are themselves of order $2^{-m}$, so a *relative* precision — what a Fourier transform is expected to deliver — needs $\delta \sim \varepsilon 2^{-m}$ and therefore $O(2^m m/\varepsilon^2)$ shots, and even then the $2^m$ numbers come out one measured index at a time.

* * *

## 2.2 Phase Estimation

### The problem

Let $U$ be a unitary and $\lvert u \rangle$ an eigenvector,

$$ U \lvert u \rangle = e^{2\pi i \varphi} \lvert u \rangle, \qquad \varphi \in [0, 1) $$

Given the ability to apply $U$, controlled on another qubit, and given a copy of $\lvert u \rangle$, estimate $\varphi$. That is the whole specification, and its reach comes from how many problems fit it. Take $U = e^{-iH\tau}$ for a Hamiltonian $H$: the eigenphases are the energies, rescaled. Take $U$ to be multiplication by $a$ modulo $N$: the eigenphases are $s/r$ with $r$ the multiplicative order, which is Chapter 3. Take $U$ to be a reflection in Grover's algorithm: estimating its phase counts the solutions, which is amplitude estimation.

### Where the interference happens

Prepare $t$ counting qubits in the uniform superposition and the system in $\lvert u \rangle$. Apply $U^{2^{t-1-j}}$ controlled on counting qubit $j$. Because $\lvert u \rangle$ is an eigenvector, each controlled operation multiplies the branch in which qubit $j$ is $\lvert 1 \rangle$ by the number $e^{2\pi i \varphi 2^{t-1-j}}$ and leaves the system register untouched. Collecting the factors,

$$ \frac{1}{\sqrt{2^t}}\sum_{k=0}^{2^t-1} e^{2\pi i \varphi k}\, \lvert k \rangle \otimes \lvert u \rangle $$

with $k = \sum_j q_j 2^{t-1-j}$ as usual. Compare that with the QFT of a basis state: it is exactly $\mathrm{QFT}\lvert 2^t \varphi \rangle$ whenever $2^t\varphi$ is an integer. So applying the **inverse** QFT to the counting register produces $\lvert 2^t \varphi \rangle$ and measuring it returns the first $t$ binary digits of $\varphi$ with certainty. Phase estimation is the QFT run backwards, and the controlled powers are what write the phase into the register in the first place.

Two consequences follow at once. The total cost in applications of $U$ is $1 + 2 + \cdots + 2^{t-1} = 2^t - 1$, so precision $\varepsilon \sim 2^{-t}$ costs $\Theta(1/\varepsilon)$ applications: the depth is inversely proportional to the accuracy. And the eigenvector was never disturbed, which means an input that is a superposition $\sum_l c_l \lvert u_l \rangle$ produces a superposition of *estimates*, and a measurement of the counting register returns eigenphase $\varphi_l$ with probability $\lvert c_l \rvert^2$ — exactly so when every $\varphi_l$ has an exact $t$-bit expansion, and otherwise with each $\lvert c_l \rvert^2$ spread over the bins around $\varphi_l$ in the pattern of $P_{\text{best}}$ below. Phase estimation is a spectrometer, and the input state is the choice of what to look at.

### Bits, precision and the standard bound

When $2^t \varphi$ is not an integer the peak is no longer a delta function. Writing $\delta$ for the distance from $\varphi$ to the nearest multiple of $2^{-t}$, the probability of measuring that nearest index is

$$ P_{\text{best}} = \frac{1}{2^{2t}} \frac{\sin^2\left(2^t \pi \delta\right)}{\sin^2\left(\pi \delta\right)} \; \ge \; \frac{4}{\pi^2} \approx 0.405 $$

with the worst case at $\delta = 2^{-t-1}$, exactly halfway between bins. The standard statement that follows is worth memorizing because it is how register sizes are chosen in practice: to obtain an estimate $\tilde{\varphi}$ accurate to $\lvert \tilde{\varphi} - \varphi \rvert < 2^{-n}$ with probability at least $1-\varepsilon$, use

$$ t = n + \left\lceil \log_2\left(2 + \frac{1}{2\varepsilon}\right) \right\rceil $$

counting qubits. The extra bits are insurance, not resolution: they buy the probability of landing in the right bin, and the cost is a factor $2^{\text{extra}}$ in circuit depth. The guarantee is on the *number* $\tilde{\varphi}$ and not on the leading bit string, and the difference is real: for $\varphi = 1/2 - 2^{-12}$ the nearest four-bit grid point is $0.1000_2$, whose bits are the complement of $\varphi$'s own leading bits, yet it is within $2^{-4}$ and is the right answer. Example 4's third block therefore measures $\Pr[\lvert \tilde{\varphi} - \varphi \rvert < 2^{-n}]$, which is what the theorem bounds. Note also what the theorem does *not* promise — it says nothing about whether the eigenvector you supplied is the one you wanted, which is Section 2.4's problem.

### Code Example 4: Phase Estimation on a Known Phase

The cleanest possible test case: a one-qubit unitary $\mathrm{diag}(1, e^{2\pi i \varphi})$, whose eigenvector $\lvert 1 \rangle$ is a basis state and whose eigenphase we chose ourselves.

```python
"""Chapter 2, Example 4: phase estimation on a known phase.
Continues from Example 3 (same session)."""

import numpy as np


def controlled(U):
    """Block-diagonal controlled version of U; the control is the first qubit."""
    d = U.shape[0]
    C = np.eye(2 * d, dtype=complex)
    C[d:, d:] = U
    return C


def qpe_state(U, sys_state, t):
    """Run textbook QPE: t counting qubits (qubit 0 = MSB) then the system.

    Returns the full state after the inverse QFT. The controlled powers cost
    2^t - 1 applications of U in total.
    """
    n_sys = int(np.log2(sys_state.size))
    n = t + n_sys
    state = np.kron(ket('0' * t), sys_state)
    for j in range(t):
        state = apply_gate(state, H, [j], n)
    for j in range(t):
        Up = np.linalg.matrix_power(U, 2 ** (t - 1 - j))
        state = apply_gate(state, controlled(Up), [j] + list(range(t, n)), n)
    return iqft(state, list(range(t)), n)


def counting_probs(state, t):
    """Marginal distribution of the t counting qubits."""
    return probs(state).reshape(2 ** t, -1).sum(axis=1)


def phase_gate(phi):
    """One-qubit U = diag(1, exp(2 pi i phi)); |1> is an eigenvector."""
    return np.array([[1.0, 0.0], [0.0, np.exp(2j * np.pi * phi)]],
                    dtype=complex)


print("A phase that is exactly representable: phi = 0.375 = 0.011 (binary)")
print("-" * 74)
phi = 0.375
print(f"  {'t':>3}{'best k':>8}{'k/2^t':>12}{'|error|':>12}"
      f"{'P(best)':>10}{'ctrl-U calls':>14}")
for t in [3, 4, 6]:
    p = counting_probs(qpe_state(phase_gate(phi), ket('1'), t), t)
    k = int(np.argmax(p))
    print(f"  {t:>3}{k:>8}{k/2**t:>12.6f}{abs(k/2**t - phi):>12.2e}"
          f"{p[k]:>10.6f}{2**t - 1:>14d}")

print("\nA phase that is not: phi = 1/3 = 0.0101010101... (binary)")
print("-" * 74)
phi = 1.0 / 3.0
print(f"  {'t':>3}{'best k':>8}{'k/2^t':>12}{'|error|':>12}{'2^-t':>10}"
      f"{'P(best)':>10}{'P(best 2)':>11}")
for t in range(3, 13):
    p = counting_probs(qpe_state(phase_gate(phi), ket('1'), t), t)
    order = np.argsort(p)[::-1]
    k = int(order[0])
    print(f"  {t:>3}{k:>8}{k/2**t:>12.6f}{abs(k/2**t - phi):>12.2e}"
          f"{2.0**-t:>10.2e}{p[k]:>10.6f}{p[order[0]]+p[order[1]]:>11.6f}")

print("\nThe standard guarantee: t = n + ceil(log2(2 + 1/(2 eps)))")
print("-" * 74)
print(f"  {'n bits':>7}{'eps':>9}{'extra':>7}{'t':>5}"
      f"{'measured P(|err| < 2^-n)':>27}")
for n_bits, eps in [(2, 0.25), (2, 0.05), (4, 0.25), (4, 0.05), (6, 0.05)]:
    extra = int(np.ceil(np.log2(2.0 + 1.0 / (2 * eps))))
    t = n_bits + extra
    p = counting_probs(qpe_state(phase_gate(phi), ket('1'), t), t)
    ks = np.arange(2 ** t)
    err = np.minimum(np.abs(ks / 2 ** t - phi), 1.0 - np.abs(ks / 2 ** t - phi))
    good = p[err < 2.0 ** -n_bits].sum()
    print(f"  {n_bits:>7}{eps:>9.2f}{extra:>7}{t:>5}{good:>27.6f}")

print("\nSuperposition input: the register measures which eigenvalue it found")
print("-" * 74)
U = np.diag([np.exp(2j * np.pi * 0.25), np.exp(2j * np.pi * 0.75)])
mix = np.array([np.sqrt(0.8), np.sqrt(0.2)], dtype=complex)
t = 4
p = counting_probs(qpe_state(U, mix, t), t)
for k in np.flatnonzero(p > 1e-9):
    print(f"  k = {k:>2}  ->  phi = {k/2**t:.4f}   probability {p[k]:.4f}")
print("  the two eigenphases are returned with the input's own weights, "
      "0.8 and 0.2")
```

```text
A phase that is exactly representable: phi = 0.375 = 0.011 (binary)
--------------------------------------------------------------------------
    t  best k       k/2^t     |error|   P(best)  ctrl-U calls
    3       3    0.375000    0.00e+00  1.000000             7
    4       6    0.375000    0.00e+00  1.000000            15
    6      24    0.375000    0.00e+00  1.000000            63

A phase that is not: phi = 1/3 = 0.0101010101... (binary)
--------------------------------------------------------------------------
    t  best k       k/2^t     |error|      2^-t   P(best)  P(best 2)
    3       3    0.375000    4.17e-02  1.25e-01  0.687838   0.862778
    4       5    0.312500    2.08e-02  6.25e-02  0.684895   0.856855
    5      11    0.343750    1.04e-02  3.12e-02  0.684162   0.855386
    6      21    0.328125    5.21e-03  1.56e-02  0.683979   0.855020
    7      43    0.335938    2.60e-03  7.81e-03  0.683933   0.854928
    8      85    0.332031    1.30e-03  3.91e-03  0.683922   0.854905
    9     171    0.333984    6.51e-04  1.95e-03  0.683919   0.854899
   10     341    0.333008    3.26e-04  9.77e-04  0.683918   0.854898
   11     683    0.333496    1.63e-04  4.88e-04  0.683918   0.854898
   12    1365    0.333252    8.14e-05  2.44e-04  0.683918   0.854898

The standard guarantee: t = n + ceil(log2(2 + 1/(2 eps)))
--------------------------------------------------------------------------
   n bits      eps  extra    t   measured P(|err| < 2^-n)
        2     0.25      2    4                   0.970284
        2     0.05      4    6                   0.992542
        4     0.25      2    6                   0.962624
        4     0.05      4    8                   0.990626
        6     0.05      4   10                   0.990511

Superposition input: the register measures which eigenvalue it found
--------------------------------------------------------------------------
  k =  4  ->  phi = 0.2500   probability 0.8000
  k = 12  ->  phi = 0.7500   probability 0.2000
  the two eigenphases are returned with the input's own weights, 0.8 and 0.2
```

**What to look for.** For $\varphi = 0.375 = 0.011_2$ the algorithm is exact: three counting qubits return $k = 3$ with probability 1, and adding more qubits changes nothing except the depth. This is the deterministic case, and it is the case Chapter 3's $N = 15$ turns out to be.

For $\varphi = 1/3$ the picture is generic and the table is the one to internalize. The error tracks $2^{-t}$ down to $8 \times 10^{-5}$ at $t = 12$, halving with each added qubit as promised, while $P_{\text{best}}$ settles at $0.6839$ and the best two bins hold $0.8549$ — both comfortably above the $4/\pi^2$ floor, and both *independent of $t$*. That is the important structural fact: extra counting qubits buy resolution, not confidence. Confidence comes from the $\lceil \log_2(2 + 1/(2\varepsilon))\rceil$ padding bits, and the third block confirms the bound empirically — the measured probability of landing within $2^{-n}$ of $\varphi$, which is what the theorem promises, exceeds $1 - \varepsilon$ in every row, by a comfortable margin because the bound is conservative.

The last block is phase estimation used as a spectrometer. An input that is a superposition of two eigenvectors with weights $0.8$ and $0.2$ returns the two eigenphases with probabilities $0.8$ and $0.2$. Nothing averages; the register reports which eigenvalue this particular run found.

### Code Example 5: Eigenphases of a Many-Qubit Unitary

Now the case of interest: $U = e^{-iH\tau}$ for a Hamiltonian on two, three and four qubits. Two practical points appear here for the first time. A constant shift of $H$ costs nothing and is needed to place the whole spectrum inside the phase window $[0,1)$; and $\tau$ should be as large as that window allows, because it sets the resolution per bit.

```python
"""Chapter 2, Example 5: eigenphases of a many-qubit unitary, and how the
error falls with the number of counting qubits.
Continues from Example 4 (same session)."""

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


def kron_all(mats):
    return reduce(np.kron, mats)


def tfim_matrix(n, h):
    """Transverse-field Ising chain: H = -sum Z_i Z_{i+1} - h sum X_i."""
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n - 1):
        M -= kron_all([Z if q in (i, i + 1) else I2 for q in range(n)])
    for i in range(n):
        M -= h * kron_all([X if q == i else I2 for q in range(n)])
    return M


def unitary_from_h(M, tau):
    """exp(-i M tau) by eigendecomposition; M must be Hermitian."""
    w, v = np.linalg.eigh(M)
    return (v * np.exp(-1j * w * tau)) @ v.conj().T


def phase_window(w, frac=0.87, pad=0.05):
    """Constant shift c and evolution time tau putting every eigenphase in (0, 1).

    The ground state is placed at phi = frac and the top of the spectrum near
    phi = frac*pad/(1+pad). Shifting the Hamiltonian by a constant is free: it
    moves no eigenvector and no energy difference. Choosing tau as large as the
    spectral span allows is what makes the window worth having.
    """
    span = w[-1] - w[0]
    c = w[-1] + pad * span
    tau = 2 * np.pi * frac / (c - w[0])
    return c, tau


h_field = 0.5
print(f"Transverse-field Ising chain, h = {h_field}: eigenphases from QPE")
print("-" * 76)
for n_sys, t in [(2, 8), (3, 8), (4, 8)]:
    M = tfim_matrix(n_sys, h_field)
    w, v = np.linalg.eigh(M)
    c, tau = phase_window(w)
    U = unitary_from_h(M - c * np.eye(2 ** n_sys), tau)
    print(f"\n  n = {n_sys} system qubits, t = {t} counting qubits, "
          f"tau = {tau:.6f}, shift c = {c:.6f}")
    print(f"  {'level':>6}{'E exact':>12}{'phi exact':>12}{'best k':>8}"
          f"{'phi QPE':>11}{'E from QPE':>13}{'|dE|':>10}{'P(best)':>10}")
    for k_lev in range(min(4, 2 ** n_sys)):
        phi_ex = (-(w[k_lev] - c) * tau / (2 * np.pi)) % 1.0
        p = counting_probs(qpe_state(U, v[:, k_lev].astype(complex), t), t)
        k = int(np.argmax(p))
        E_qpe = c - (k / 2 ** t) * 2 * np.pi / tau
        print(f"  {k_lev:>6}{w[k_lev].real:>12.6f}{phi_ex:>12.6f}{k:>8}"
              f"{k/2**t:>11.6f}{E_qpe:>13.6f}"
              f"{abs(E_qpe - w[k_lev].real):>10.2e}{p[k]:>10.4f}")

print("\nPrecision scaling: ground state of the n = 3 chain")
print("-" * 76)
n_sys = 3
M = tfim_matrix(n_sys, h_field)
w, v = np.linalg.eigh(M)
c, tau = phase_window(w)
U = unitary_from_h(M - c * np.eye(2 ** n_sys), tau)
psi0 = v[:, 0].astype(complex)
print(f"  exact E_0 = {w[0].real:.9f},  exact phi_0 = "
      f"{(-(w[0]-c)*tau/(2*np.pi)) % 1.0:.9f}")
print(f"  {'t':>3}{'ctrl-U calls':>14}{'phi QPE':>12}{'|d phi|':>11}"
      f"{'2^-(t+1)':>11}{'|dE| (energy)':>15}{'P(best)':>10}")
errs = []
ts = list(range(4, 15))
for t in ts:
    p = counting_probs(qpe_state(U, psi0, t), t)
    k = int(np.argmax(p))
    dphi = abs(k / 2 ** t - (-(w[0] - c) * tau / (2 * np.pi)) % 1.0)
    dE = abs((c - (k / 2 ** t) * 2 * np.pi / tau) - w[0].real)
    errs.append(dE)
    print(f"  {t:>3}{2**t - 1:>14d}{k/2**t:>12.6f}{dphi:>11.2e}"
          f"{2.0**-(t+1):>11.2e}{dE:>15.2e}{p[k]:>10.4f}")

sl = np.polyfit(np.array(ts, float), np.log2(np.array(errs)), 1)[0]
print(f"\n  log2(energy error) versus t: slope = {sl:.3f} "
      f"(exact halving would be -1)")
print("  Cost is 2^t - 1 controlled applications of U, so precision eps "
      "costs O(1/eps) depth.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for t, style in [(5, "-o"), (8, "--s")]:
    p = counting_probs(qpe_state(U, psi0, t), t)
    ax[0].plot(np.arange(2 ** t) / 2 ** t, p, style, ms=3, lw=1,
               label=f"t = {t}")
ax[0].axvline((-(w[0] - c) * tau / (2 * np.pi)) % 1.0, color="k", ls=":",
              lw=1, label="exact $\\phi_0$")
ax[0].set_xlabel("$k/2^t$"); ax[0].set_ylabel("probability")
ax[0].set_title("QPE output, ground state of the 3-site chain")
ax[0].legend(fontsize=8)

ax[1].semilogy(ts, errs, "o-", color="tab:red", label="measured")
ax[1].semilogy(ts, [2.0 ** -(t + 1) * 2 * np.pi / tau for t in ts], "k--",
               label="$2^{-(t+1)} \\cdot 2\\pi/\\tau$")
ax[1].set_xlabel("counting qubits $t$"); ax[1].set_ylabel("energy error")
ax[1].set_title("One extra qubit, one more bit")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

```text
Transverse-field Ising chain, h = 0.5: eigenphases from QPE
----------------------------------------------------------------------------

  n = 2 system qubits, t = 8 counting qubits, tau = 1.840623, shift c = 1.555635
   level     E exact   phi exact  best k    phi QPE   E from QPE      |dE|   P(best)
       0   -1.414214    0.870000     223   0.871094    -1.417947  3.73e-03    0.7673
       1   -1.000000    0.748659     192   0.750000    -1.004579  4.58e-03    0.6675
       2    1.000000    0.162770      42   0.164062     0.995588  4.41e-03    0.6879
       3    1.414214    0.041429      11   0.042969     1.408956  5.26e-03    0.5825

  n = 3 system qubits, t = 8 counting qubits, tau = 1.083148, shift c = 2.643533
   level     E exact   phi exact  best k    phi QPE   E from QPE      |dE|   P(best)
       0   -2.403212    0.870000     223   0.871094    -2.409557  6.34e-03    0.7673
       1   -2.209275    0.836568     214   0.835938    -2.205620  3.66e-03    0.9173
       2   -0.500000    0.541908     139   0.542969    -0.506151  6.15e-03    0.7799
       3   -0.306063    0.508476     130   0.507812    -0.302214  3.85e-03    0.9086

  n = 4 system qubits, t = 8 counting qubits, tau = 0.759559, shift c = 3.769737
   level     E exact   phi exact  best k    phi QPE   E from QPE      |dE|   P(best)
       0   -3.427034    0.870000     223   0.871094    -3.436082  9.05e-03    0.7673
       1   -3.332247    0.858541     220   0.859375    -3.339142  6.90e-03    0.8589
       2   -1.826838    0.676556     173   0.675781    -1.820427  6.41e-03    0.8770
       3   -1.732051    0.665098     170   0.664062    -1.723488  8.56e-03    0.7893

Precision scaling: ground state of the n = 3 chain
----------------------------------------------------------------------------
  exact E_0 = -2.403211926,  exact phi_0 = 0.870000000
    t  ctrl-U calls     phi QPE    |d phi|   2^-(t+1)  |dE| (energy)   P(best)
    4            15    0.875000   5.00e-03   3.12e-02       2.90e-02    0.9792
    5            31    0.875000   5.00e-03   1.56e-02       2.90e-02    0.9186
    6            63    0.875000   5.00e-03   7.81e-03       2.90e-02    0.7054
    7           127    0.867188   2.81e-03   3.91e-03       1.63e-02    0.6401
    8           255    0.871094   1.09e-03   1.95e-03       6.34e-03    0.7673
    9           511    0.869141   8.59e-04   9.77e-04       4.99e-03    0.5050
   10          1023    0.870117   1.17e-04   4.88e-04       6.80e-04    0.9535
   11          2047    0.870117   1.17e-04   2.44e-04       6.80e-04    0.8243
   12          4095    0.870117   1.17e-04   1.22e-04       6.80e-04    0.4380
   13          8191    0.869995   4.88e-06   6.10e-05       2.83e-05    0.9947
   14         16383    0.869995   4.88e-06   3.05e-05       2.83e-05    0.9791

  log2(energy error) versus t: slope = -1.079 (exact halving would be -1)
  Cost is 2^t - 1 controlled applications of U, so precision eps costs O(1/eps) depth.
```

**What to look for.** Every eigenvalue of every chain is recovered to a few times $10^{-3}$ in energy at $t = 8$, which is $2^{-9}$ of the spectral span — the resolution the register size allows and no better. The recovered energies are not fitted or averaged; they are one arithmetic step from one measured integer.

The precision table is the chapter's central scaling result. The phase error stays below $2^{-(t+1)}$ at every $t$, the energy error falls by a factor of two per added counting qubit, and the fitted slope of $\log_2(\text{error})$ against $t$ is $-1.079$ — the excess over $-1$ being the discreteness of the bins rather than a systematic effect. The number in the second column is what this costs: $2^{14}-1 = 16383$ controlled applications of $U$ to reach an energy error of $2.8 \times 10^{-5}$ (the $6.10 \times 10^{-5}$ on the $t = 13$ row is that row's $2^{-(t+1)}$ bound, not an achieved error). Precision is bought linearly in circuit depth, and this is the trade that Chapter 4 has to make concrete, because in a real problem each application of $U$ is itself an expensive simulation of $e^{-iH\tau}$.

The success probability column wanders between $0.44$ and $0.99$ with no trend, which is the same lesson as Example 4 seen from a different angle: the peak height depends on where $\varphi$ happens to sit relative to the bin boundaries, and never on how many bins there are.

* * *

## 2.3 Iterative Phase Estimation

### One ancilla instead of $t$

The textbook circuit needs $t$ counting qubits held coherently while $2^t - 1$ applications of $U$ run. On hardware, qubits are the scarcer resource and mid-circuit measurement is available, and there is an exchange rate between them. Notice that the counting register is measured at the end and never used again, and that the inverse QFT's only non-Clifford content is controlled phase rotations by angles determined by *other* measurement outcomes. Both observations point the same way: replace the quantum controls by classical ones.

**Iterative phase estimation** does this. Write $\varphi = 0.b_1b_2\ldots b_t$ in binary and extract the bits from the least significant upwards. At round $k$, with $b_{k+1},\ldots,b_t$ already known:

  1. Hadamard the ancilla.
  2. Apply $U^{2^{k-1}}$ controlled on it. This writes the phase $2\pi \left(2^{k-1}\varphi\right) = 2\pi\left(\text{integer} + 0.b_k b_{k+1}\ldots b_t\right)$ onto the $\lvert 1 \rangle$ branch.
  3. Rotate the ancilla by $-2\pi \times 0.0b_{k+1}\ldots b_t$, cancelling every bit already known.
  4. Hadamard and measure. The residual relative phase is $\pi b_k$, so the outcome is $b_k$ with certainty.

The feedback in step 3 is the inverse QFT's controlled rotations, executed classically. If $\varphi$ is exactly $t$ bits long every round is deterministic; otherwise the tail $0.b_{k+1}\ldots$ that step 3 cannot know biases each round, and a wrong bit corrupts every bit above it. The remedy is to repeat each round and take a majority vote, which costs applications of $U$ but no qubits.

The exchange rate, stated plainly: iterative QPE uses $1 + n_{\text{sys}}$ qubits instead of $t + n_{\text{sys}}$, needs the same $2^t - 1$ applications of $U$ in the best case and a small multiple of that in practice, and requires $t$ rounds of mid-circuit measurement with classical feedback inside the coherence time. It also requires the system register to survive all $t$ rounds — which for an exact eigenvector is automatic, since $U$ does not move it, and is the reason this variant is the practical one for eigenvalue problems.

### Code Example 6: Iterative Phase Estimation

```python
"""Chapter 2, Example 6: iterative phase estimation with a single ancilla.
Continues from Example 5 (same session)."""

import numpy as np


def iqpe_round(U, sys_state, power, feedback):
    """One round: ancilla + system, controlled U^power, Rz feedback, readout.

    Returns P(ancilla = 1). The ancilla is qubit 0; the system occupies the
    rest. Because sys_state is an eigenvector of U it is returned unchanged,
    which is why the rounds can be run one after another on one ancilla.
    """
    n_sys = int(np.log2(sys_state.size))
    n = 1 + n_sys
    state = np.kron(ket('0'), sys_state)
    state = apply_gate(state, H, [0], n)
    Up = np.linalg.matrix_power(U, power)
    state = apply_gate(state, controlled(Up), list(range(n)), n)
    state = apply_gate(state, rz(-2 * np.pi * feedback), [0], n)
    state = apply_gate(state, H, [0], n)
    p = probs(state)
    return float(p[p.size // 2:].sum())


def iqpe(U, sys_state, t, rng, reps=1):
    """Iterative QPE: t rounds from the least significant bit upwards.

    bits[j] is the (j+1)-th binary digit of phi, so phi = sum bits[j] 2^-(j+1).
    Round k measures bits[k-1] after subtracting the digits already known.
    Each round is repeated `reps` times and decided by majority vote.
    """
    bits = [0] * t
    for k in range(t, 0, -1):
        feedback = sum(bits[j] / 2.0 ** (j - k + 2) for j in range(k, t))
        p1 = iqpe_round(U, sys_state, 2 ** (k - 1), feedback)
        votes = sum(1 for _ in range(reps) if rng.random() < p1)
        bits[k - 1] = 1 if 2 * votes > reps else 0
    return sum(b / 2.0 ** (j + 1) for j, b in enumerate(bits)), bits


print("A dyadic phase is returned bit by bit, deterministically")
print("-" * 74)
rng = np.random.default_rng(2024)
for phi in [0.375, 0.8125, 0.65625]:
    t = 5
    est, bits = iqpe(phase_gate(phi), ket('1'), t, rng)
    print(f"  phi = {phi:.5f} = 0.{''.join(str(int(b)) for b in bits)} "
          f"(binary)   estimate {est:.5f}   error {abs(est-phi):.1e}")
print("  every round had P(1) equal to 0 or 1, so no sampling was involved:")
for phi in [0.375, 0.8125]:
    exact = [int(phi * 2 ** (j + 1)) % 2 for j in range(5)]
    ps = []
    for k in range(5, 0, -1):
        fb = sum(exact[j] / 2.0 ** (j - k + 2) for j in range(k, 5))
        ps.append(iqpe_round(phase_gate(phi), ket('1'), 2 ** (k - 1), fb))
    print(f"    phi = {phi:.5f}: P(1) per round = "
          + "  ".join(f"{p:.3f}" for p in ps))

print("\nA non-dyadic phase needs repetition: phi = 1/3, t = 6")
print("-" * 74)
phi = 1.0 / 3.0
t = 6
n_trials = 2000
print(f"  {'reps/round':>11}{'U calls/trial':>15}{'mean |error|':>14}"
      f"{'P(|error| < 2^-t)':>20}")
for reps in [1, 3, 5, 9, 25]:
    rng = np.random.default_rng(7)
    errs = np.array([abs(iqpe(phase_gate(phi), ket('1'), t, rng, reps)[0] - phi)
                     for _ in range(n_trials)])
    errs = np.minimum(errs, 1.0 - errs)
    calls = reps * (2 ** t - 1)
    print(f"  {reps:>11}{calls:>15}{errs.mean():>14.5f}"
          f"{float((errs < 2.0**-t).mean()):>20.4f}")
print(f"  textbook QPE with t = {t}: {2**t - 1} calls, "
      f"{t} counting qubits, P(best bin) = "
      f"{counting_probs(qpe_state(phase_gate(phi), ket('1'), t), t).max():.4f}")

print("\nSame comparison on the 3-site Ising ground state, t = 8")
print("-" * 74)
M = tfim_matrix(3, h_field)
w, v = np.linalg.eigh(M)
c, tau = phase_window(w)
U = unitary_from_h(M - c * np.eye(8), tau)
psi0 = v[:, 0].astype(complex)
phi_ex = (-(w[0] - c) * tau / (2 * np.pi)) % 1.0
t = 8
print(f"  exact phi_0 = {phi_ex:.6f},  exact E_0 = {w[0].real:.6f}")
print(f"  {'method':<28}{'qubits':>8}{'U calls':>10}{'phi':>12}"
      f"{'E':>13}{'|dE|':>11}")
p = counting_probs(qpe_state(U, psi0, t), t)
k = int(np.argmax(p))
E = c - (k / 2 ** t) * 2 * np.pi / tau
print(f"  {'textbook QPE':<28}{t+3:>8}{2**t-1:>10}{k/2**t:>12.6f}"
      f"{E:>13.6f}{abs(E-w[0].real):>11.2e}")
for reps in [1, 9]:
    rng = np.random.default_rng(31)
    est, _ = iqpe(U, psi0, t, rng, reps)
    E = c - est * 2 * np.pi / tau
    print(f"  {f'iterative QPE, reps = {reps}':<28}{1+3:>8}"
          f"{reps*(2**t-1):>10}{est:>12.6f}{E:>13.6f}"
          f"{abs(E-w[0].real):>11.2e}")
print("  The counting register is gone: t qubits become one, at the price of "
      "t rounds\n  of measurement and classical feedback.")
```

```text
A dyadic phase is returned bit by bit, deterministically
--------------------------------------------------------------------------
  phi = 0.37500 = 0.01100 (binary)   estimate 0.37500   error 0.0e+00
  phi = 0.81250 = 0.11010 (binary)   estimate 0.81250   error 0.0e+00
  phi = 0.65625 = 0.10101 (binary)   estimate 0.65625   error 0.0e+00
  every round had P(1) equal to 0 or 1, so no sampling was involved:
    phi = 0.37500: P(1) per round = 0.000  0.000  1.000  1.000  0.000
    phi = 0.81250: P(1) per round = 0.000  1.000  0.000  1.000  1.000

A non-dyadic phase needs repetition: phi = 1/3, t = 6
--------------------------------------------------------------------------
   reps/round  U calls/trial  mean |error|   P(|error| < 2^-t)
            1             63       0.01367              0.8510
            3            189       0.00689              0.9665
            5            315       0.00593              0.9900
            9            567       0.00551              0.9970
           25           1575       0.00522              1.0000
  textbook QPE with t = 6: 63 calls, 6 counting qubits, P(best bin) = 0.6840

Same comparison on the 3-site Ising ground state, t = 8
--------------------------------------------------------------------------
  exact phi_0 = 0.870000,  exact E_0 = -2.403212
  method                        qubits   U calls         phi            E       |dE|
  textbook QPE                      11       255    0.871094    -2.409557   6.34e-03
  iterative QPE, reps = 1            4       255    0.867188    -2.386897   1.63e-02
  iterative QPE, reps = 9            4      2295    0.871094    -2.409557   6.34e-03
  The counting register is gone: t qubits become one, at the price of t rounds
  of measurement and classical feedback.
```

**What to look for.** For dyadic phases the round-by-round probabilities are exactly 0 or 1: the algorithm reads off the binary expansion, no sampling involved, and the estimate is exact. That is the case the construction was designed for. One reading detail: the per-round list is printed in the order the rounds run, which is least-significant bit first — the reverse of the binary expansion printed on the line above it, so $0.01100$ appears as $0, 0, 1, 1, 0$.

For $\varphi = 1/3$ the middle table shows the trade explicitly. One repetition per round gives the correct $t$-bit answer $85\%$ of the time; nine repetitions give $99.7\%$ at nine times the number of applications of $U$; twenty-five give certainty in 2000 trials, with the mean error converging to $0.00522$, which is the truncation error $\lvert 1/3 - 21/64 \rvert$ and not an algorithmic failure. Textbook QPE with the same $t$ costs 63 applications and needs six counting qubits, so the comparison is: same depth, six qubits saved, at $9\times$ the total number of applications for comparable reliability.

The last block runs both variants on a real eigenvector, and the arithmetic is the point. Eleven qubits become four. A single-shot iterative run lands one bin away from the textbook answer; with nine repetitions per round it lands on exactly the same bin, meaning exactly the same energy. Whether that trade is worth making is a hardware question — it depends on the cost of a mid-circuit measurement relative to the cost of a qubit — and both answers are in use.

* * *

## 2.4 What Phase Estimation Is For

### Eigenvalues are the point

Almost everything a materials researcher wants from a quantum computer is an eigenvalue. A ground-state energy is the lowest eigenvalue of an electronic Hamiltonian; a band structure is a family of eigenvalues; a reaction barrier is a difference of two of them; a vibrational spectrum, a magnetic exchange constant, an optical gap. [Introduction to Quantum Computing, Chapter 4](<../quantum-computing-introduction/chapter-4.html>) set this up in detail, mapped an electronic Hamiltonian onto qubits through second quantization and the Jordan-Wigner transformation, and then compared two ways of extracting the eigenvalue. That comparison is the reason this chapter exists, and it can now be completed.

The **variational quantum eigensolver** prepares a parameterized trial state, measures $\langle H \rangle$ by sampling Pauli terms, and lets a classical optimizer minimize. Its circuits are shallow, which is why it runs on noisy hardware. It pays in two places: the answer is a variational upper bound whose error is unknown unless the ansatz is exact, and the measurement cost to reach precision $\varepsilon$ scales as $1/\varepsilon^2$ because it is estimating an expectation value by averaging.

**Phase estimation** does something categorically different. It does not average anything. It writes the eigenvalue into a register, digit by digit, and reads it. The precision cost is $1/\varepsilon$ rather than $1/\varepsilon^2$ — the Heisenberg scaling — and the answer is an eigenvalue of the Hamiltonian you gave it, with no variational gap. The price is the depth: $\Theta(1/\varepsilon)$ coherent applications of $e^{-iH\tau}$, which is far beyond any uncorrected device and is the single strongest argument for building a fault-tolerant one.

| | VQE | Phase estimation |
| --- | --- | --- |
| What it returns | variational upper bound $\ge E_0$ | an eigenvalue of $H$ |
| Error if the ansatz is wrong | unknown, one-sided | none — but possibly the wrong eigenvalue |
| Precision cost | $O(1/\varepsilon^2)$ measurements | $O(1/\varepsilon)$ applications of $U$ |
| Circuit depth | shallow, fixed | $\Theta(1/\varepsilon)$, coherent |
| Needs error correction | no | yes |
| Needs a good trial state | yes, to be accurate | yes, to succeed at all |
| Excited states | hard; needs constraints or folding | free — they are the other peaks |

The last two rows are the ones usually skipped. Phase estimation gets excited states for nothing, because they are simply the other peaks in the same distribution, and that is a genuine structural advantage over any variational method. And both algorithms need a good trial state, but for different reasons: VQE needs it to be *accurate*, and phase estimation needs it to *overlap*.

### The overlap problem, stated honestly

Feed phase estimation a trial state $\lvert \psi \rangle = \sum_l c_l \lvert u_l \rangle$ and it returns $E_l$ with probability $\lvert c_l \rvert^2$ — as Example 4 already showed. To learn the ground-state energy you therefore need $p_0 = \lvert \langle \psi_0 \vert \psi \rangle \rvert^2$ to be not too small, and the number of repetitions scales as $1/p_0$. For small molecules a Hartree-Fock determinant has $p_0$ close to one and this is a non-issue. For the strongly correlated systems that motivate the whole enterprise it is exactly the hard case: the ground state is a superposition of exponentially many determinants, and $p_0$ can fall off exponentially with system size. There is no known general solution. Adiabatic state preparation, quantum-selected configuration interaction, and using a converged VQE state as the input for phase estimation are the standard responses, and each of them is a research problem rather than a subroutine.

This is the honest boundary of the algorithm, and it is worth being precise about where it lies. Phase estimation does not "solve" electronic structure. It converts the problem of *computing* an eigenvalue into the problem of *preparing a state with decent overlap* — a real and substantial reduction, since the second problem is at least sometimes easy and the first never is, but not the same thing as a solution.

### Code Example 7: Phase Estimation as an Electronic-Structure Method

The concrete version, on the two-qubit H$_2$ Hamiltonian that the introductory course's Chapter 3 solved variationally. The same matrix, the same reference energy, a completely different algorithm.

```python
"""Chapter 2, Example 7: phase estimation as an electronic-structure method.
Continues from Example 6 (same session)."""

import numpy as np
import matplotlib.pyplot as plt

# The same two-qubit H2 Hamiltonian used in the introductory course, Chapter 3:
# STO-3G, R = 0.735 A, frozen-core two-qubit reduction.
H2_TERMS = {'II': 0.252992, 'ZI': 0.344368, 'IZ': -0.451507,
            'ZZ': 0.574116, 'YY': 0.090466, 'XX': 0.090466}


def pauli_matrix(pauli):
    M = np.array([[1.0 + 0j]])
    for ch in pauli:
        M = np.kron(M, PAULI[ch])
    return M


Hm = sum(c * pauli_matrix(p) for p, c in H2_TERMS.items())
w, v = np.linalg.eigh(Hm)
c_shift, tau = phase_window(w)
U = unitary_from_h(Hm - c_shift * np.eye(4), tau)
phi_exact = [(-(e - c_shift) * tau / (2 * np.pi)) % 1.0 for e in w.real]

print("H2 at R = 0.735 A, STO-3G, two-qubit reduction")
print("-" * 72)
print(f"  exact spectrum (Ha): " + "  ".join(f"{e:+.6f}" for e in w.real))
print(f"  tau = {tau:.6f}, constant shift c = {c_shift:.6f} Ha")
print(f"  eigenphases:         " + "  ".join(f"{p:.6f}" for p in phi_exact))
print(f"  ground state = " + "  ".join(
    f"{v[i,0].real:+.4f}|{b}>" for i, b in enumerate(['00', '01', '10', '11'])
    if abs(v[i, 0]) > 1e-8))

hf = ket('10')          # Hartree-Fock: the sigma_g orbital doubly occupied
p0 = abs(np.vdot(v[:, 0], hf)) ** 2
print(f"  Hartree-Fock overlap |<HF|psi_0>|^2 = {p0:.6f}")

print("\nQPE from the Hartree-Fock state: energy versus counting qubits")
print("-" * 72)
print(f"  {'t':>3}{'ctrl-U calls':>14}{'best k':>8}{'E (Ha)':>13}"
      f"{'error (Ha)':>13}{'P(peak)':>10}{'chem. acc.':>12}")
for t in range(4, 13):
    p = counting_probs(qpe_state(U, hf, t), t)
    k = int(np.argmax(p))
    E = c_shift - (k / 2 ** t) * 2 * np.pi / tau
    err = E - w[0].real
    print(f"  {t:>3}{2**t-1:>14d}{k:>8}{E:>13.6f}{err:>13.2e}{p[k]:>10.4f}"
          f"{'yes' if abs(err) < 1.6e-3 else 'no':>12}")
print("  chemical accuracy is 1 kcal/mol = 1.6 mHa")

print("\nThe overlap is the whole story: four trial states, t = 9")
print("-" * 72)
t = 9
trials = [("Hartree-Fock |10>", ket('10')),
          ("doubly excited |01>", ket('01')),
          ("equal mixture", (ket('10') + ket('01')) / np.sqrt(2)),
          ("exact first excited", v[:, 1].astype(complex))]
k_ground = int(round(phi_exact[0] * 2 ** t))
for label, psi in trials:
    p = counting_probs(qpe_state(U, psi, t), t)
    overlaps = [abs(np.vdot(v[:, j], psi)) ** 2 for j in range(4)]
    near = p[max(0, k_ground - 2):k_ground + 3].sum()
    k = int(np.argmax(p))
    E = c_shift - (k / 2 ** t) * 2 * np.pi / tau
    print(f"  {label:<22} |<psi_0|trial>|^2 = {overlaps[0]:.4f}"
          f"   peak E = {E:+.6f} Ha   P(within 2 bins of E_0) = {near:.4f}")
print("  A trial state with no ground-state amplitude returns a different "
      "eigenvalue,\n  correctly and confidently. QPE does not know which "
      "eigenvalue you wanted.")

print("\nWhat the depth costs, extrapolated")
print("-" * 72)
print(f"  {'target error (Ha)':>19}{'t needed':>10}{'ctrl-U calls':>15}"
      f"{'x100 Trotter steps':>21}")
for eps in [1e-2, 1.6e-3, 1e-4, 1e-6]:
    t_need = int(np.ceil(np.log2(2 * np.pi / tau / eps)))
    calls = 2 ** t_need - 1
    print(f"  {eps:>19.1e}{t_need:>10}{calls:>15d}"
          f"{calls * 100:>21d}")
print("  The last column assumes a merely nominal 100 Trotter steps per "
      "controlled U;\n  Chapter 4 replaces that guess with a real "
      "block-encoding cost.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for t_, style in [(6, "-o"), (9, "-")]:
    p = counting_probs(qpe_state(U, hf, t_), t_)
    E_axis = c_shift - (np.arange(2 ** t_) / 2 ** t_) * 2 * np.pi / tau
    ax[0].plot(E_axis, p, style, ms=3, lw=1, label=f"t = {t_}")
for e in w.real:
    ax[0].axvline(e, color="k", ls=":", lw=0.8)
ax[0].set_xlabel("energy (Ha)"); ax[0].set_ylabel("probability")
ax[0].set_title("QPE spectrum of H$_2$ from the HF state")
ax[0].legend(fontsize=8)

t_ = 9
for label, psi in [("HF |10>", ket('10')), ("|01>", ket('01'))]:
    p = counting_probs(qpe_state(U, psi, t_), t_)
    E_axis = c_shift - (np.arange(2 ** t_) / 2 ** t_) * 2 * np.pi / tau
    ax[1].plot(E_axis, p, lw=1, label=label)
for e in w.real:
    ax[1].axvline(e, color="k", ls=":", lw=0.8)
ax[1].set_xlabel("energy (Ha)"); ax[1].set_ylabel("probability")
ax[1].set_title("The trial state selects the eigenvalue")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

```text
H2 at R = 0.735 A, STO-3G, two-qubit reduction
------------------------------------------------------------------------
  exact spectrum (Ha): -1.137306  +0.495058  +0.719969  +0.934247
  tau = 2.513123, constant shift c = 1.037825 Ha
  eigenphases:         0.870000  0.217094  0.127135  0.041429
  ground state = -0.1115|01>  +0.9938|10>
  Hartree-Fock overlap |<HF|psi_0>|^2 = 0.987560

QPE from the Hartree-Fock state: energy versus counting qubits
------------------------------------------------------------------------
    t  ctrl-U calls  best k       E (Ha)   error (Ha)   P(peak)  chem. acc.
    4            15      14    -1.149807    -1.25e-02    0.9671          no
    5            31      28    -1.149807    -1.25e-02    0.9072          no
    6            63      56    -1.149807    -1.25e-02    0.6967          no
    7           127     111    -1.130275     7.03e-03    0.6321          no
    8           255     223    -1.140041    -2.73e-03    0.7577          no
    9           511     445    -1.135158     2.15e-03    0.4987          no
   10          1023     891    -1.137599    -2.93e-04    0.9417         yes
   11          2047    1782    -1.137599    -2.93e-04    0.8140         yes
   12          4095    3564    -1.137599    -2.93e-04    0.4326         yes
  chemical accuracy is 1 kcal/mol = 1.6 mHa

The overlap is the whole story: four trial states, t = 9
------------------------------------------------------------------------
  Hartree-Fock |10>      |<psi_0|trial>|^2 = 0.9876   peak E = -1.135158 Ha   P(within 2 bins of E_0) = 0.9090
  doubly excited |01>    |<psi_0|trial>|^2 = 0.0124   peak E = +0.495800 Ha   P(within 2 bins of E_0) = 0.0115
  equal mixture          |<psi_0|trial>|^2 = 0.3892   peak E = +0.495800 Ha   P(within 2 bins of E_0) = 0.3582
  exact first excited    |<psi_0|trial>|^2 = 0.0000   peak E = +0.495800 Ha   P(within 2 bins of E_0) = 0.0000
  A trial state with no ground-state amplitude returns a different eigenvalue,
  correctly and confidently. QPE does not know which eigenvalue you wanted.

What the depth costs, extrapolated
------------------------------------------------------------------------
    target error (Ha)  t needed   ctrl-U calls   x100 Trotter steps
              1.0e-02         8            255                25500
              1.6e-03        11           2047               204700
              1.0e-04        15          32767              3276700
              1.0e-06        22        4194303            419430300
  The last column assumes a merely nominal 100 Trotter steps per controlled U;
  Chapter 4 replaces that guess with a real block-encoding cost.
```

**What to look for.** The exact ground-state energy of this Hamiltonian is $-1.137306$ Ha and phase estimation reaches chemical accuracy — $1.6$ mHa, the threshold at which a computed reaction energy becomes chemically meaningful — at $t = 10$, using 1023 controlled applications of $e^{-iH\tau}$. Not by optimizing anything: the answer is $c - (k/2^t)\cdot 2\pi/\tau$ for one measured integer $k$. Compare the introductory course's VQE on the same Hamiltonian, which reached $10^{-15}$ Ha because the ansatz happened to be exact and the simulation noiseless, and which would have needed of order $10^6$ shots per energy evaluation at $1.6$ mHa on real hardware.

The trial-state table is the overlap problem in four lines. The Hartree-Fock determinant has $98.8\%$ overlap with the ground state, and $90.9\%$ of the QPE outcomes land within two bins of $E_0$. The doubly excited determinant has $1.2\%$ overlap and the peak moves to the *first excited* energy, $+0.4958$ Ha — reported confidently and correctly, because that is genuinely the eigenvalue this input mostly contains. The equal mixture has $38.9\%$ ground-state weight and still peaks on the excited state, since $61\%$ beats $39\%$. Phase estimation answers the question it was asked, and the question is set by the input state.

The last block extrapolates the depth. Chemical accuracy costs $2 \times 10^3$ controlled applications of $U$ for this toy; $10^{-6}$ Ha costs $4 \times 10^6$. Multiply by whatever it costs to implement one $e^{-iH\tau}$ — the nominal factor of 100 in the last column is a placeholder, and Chapter 4 replaces it with a real block-encoding cost — and the reason phase estimation is a fault-tolerance argument rather than a near-term proposal is arithmetic, not opinion. [Chapter 5 of the introductory course](<../quantum-computing-introduction/chapter-5.html>) puts the matching error-rate requirement at $p \lesssim 10^{-9}$ for a two-site Hubbard model, six orders of magnitude below uncorrected hardware.

* * *

## Exercises

#### Exercise 1: The Product Form by Hand

Take $n = 3$ and $j = 5 = 101_2$.

  1. Write out $\mathrm{QFT}\lvert 101 \rangle$ as a sum over the eight basis states, giving each amplitude as $\frac{1}{\sqrt{8}}e^{i\theta_k}$ with $\theta_k$ in units of $2\pi$.
  2. Write the same state in the product form $\bigotimes_l (\lvert 0 \rangle + e^{2\pi i j/2^l}\lvert 1 \rangle)/\sqrt{2}$ and verify that expanding it reproduces part 1.
  3. Which single-qubit phase in the product form is $-1$, and which qubit of the *circuit* output does it correspond to, before and after the final swaps?
  4. How many of the eight amplitudes have different magnitudes? What does that say about measuring the output in the computational basis?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\theta_k/2\pi = 5k/8 \bmod 1\), i.e. \(0, \tfrac{5}{8}, \tfrac{2}{8}, \tfrac{7}{8}, \tfrac{4}{8}, \tfrac{1}{8}, \tfrac{6}{8}, \tfrac{3}{8}\) for \(k = 0,\ldots,7\). All eight magnitudes are \(1/\sqrt{8}\).</p>

<p><strong>2.</strong> The three factors carry phases \(e^{2\pi i \cdot 5/2} = e^{i\pi} = -1\), \(e^{2\pi i \cdot 5/4} = e^{i\pi/2} = i\), and \(e^{2\pi i \cdot 5/8}\). Expanding the product gives \(2^{-3/2}\sum_{k_1k_2k_3} (-1)^{k_1} i^{k_2} e^{2\pi i \cdot 5k_3/8}\lvert k_1k_2k_3\rangle\); with \(k = 4k_1 + 2k_2 + k_3\) the exponent is \(2\pi i \cdot 5k/8\) modulo 1, matching part 1.</p>

<p><strong>3.</strong> The factor with \(l = 1\), phase \(-1\). In the product form it is the <em>first</em> tensor factor, which the derivation shows carries the <em>least</em> significant bit \(j_n\)'s worth of phase; the circuit produces it on the last qubit and the final swaps move it to the front. Deleting the swaps and relabelling the output is exactly equivalent, which is why most implementations do that.</p>

<p><strong>4.</strong> One: all eight magnitudes are equal. A computational-basis measurement of \(\mathrm{QFT}\lvert j \rangle\) is therefore uniformly random and carries no information about \(j\) at all. The transform of a <em>single</em> basis state is maximally uninformative, and only a state with structure — a period, as in Example 3 — produces a peaked output.</p>

</details>

#### Exercise 2: Choosing the Register Size

You need the eigenphase of a unitary to 6 correct binary digits with probability at least $0.99$.

  1. Use $t = n + \lceil \log_2(2 + 1/(2\varepsilon)) \rceil$ to find $t$, and state how many applications of $U$ that implies.
  2. How does the number of applications change if the required confidence goes from $0.99$ to $0.999$? And if the required precision goes from 6 bits to 7?
  3. A colleague proposes to reach $0.999$ by running the $t$ from part 1 three times and taking the majority. Estimate the total number of applications of $U$ and compare with part 2.
  4. Which of the two strategies fails first on hardware with a fixed coherence time, and why?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\varepsilon = 0.01\), so \(1/(2\varepsilon) = 50\) and \(\lceil \log_2 52 \rceil = 6\); \(t = 12\), and \(2^{12}-1 = 4095\) applications of \(U\).</p>

<p><strong>2.</strong> For \(\varepsilon = 0.001\), \(\lceil \log_2 502 \rceil = 9\), so \(t = 15\) and 32767 applications — a factor of 8 for one extra decimal of confidence. For 7 bits at \(\varepsilon = 0.01\), \(t = 13\) and 8191 applications — a factor of 2 per bit of precision. Confidence is expensive and precision is cheap, per unit of what you get.</p>

<p><strong>3.</strong> Three runs at \(t = 12\) is \(3 \times 4095 = 12285\) applications, against 32767 for the single deeper run: repetition is cheaper by a factor of 2.7 in total work. The majority vote over three independent samples of a distribution whose best bin has probability \(\ge 0.99\) fails with probability \(\approx 3 \times 10^{-4}\), so it does reach the target.</p>

<p><strong>4.</strong> The deeper single run fails first. Its maximum coherent depth is 8 times larger, and coherent depth is the resource a decoherence time limits; the repeated shallower runs need only the same depth as part 1 and can be separated in time arbitrarily. This is the general principle behind every "trade depth for repetitions" scheme, including the \(\alpha\)-QPE family — and its limit is the bound \(\varepsilon \gtrsim 1/(D\sqrt{N})\) at fixed maximum depth \(D\), which is why full Heisenberg scaling genuinely requires depth \(\propto 1/\varepsilon\).</p>

</details>

#### Exercise 3: The Peak Height Formula

Verify the claim $P_{\text{best}} \ge 4/\pi^2$ and its context.

  1. Starting from the pre-QFT state $2^{-t/2}\sum_k e^{2\pi i \varphi k}\lvert k \rangle$, show that the amplitude of outcome $m$ after the inverse QFT is $2^{-t}\sum_k e^{2\pi i (\varphi - m/2^t) k}$, and sum the geometric series.
  2. Write $\delta = \varphi - m/2^t$ and show $P(m) = \sin^2(2^t\pi\delta) / \left(2^{2t}\sin^2(\pi\delta)\right)$.
  3. Evaluate this at the worst case $\delta = 2^{-t-1}$ and take $t \to \infty$.
  4. Evaluate it for $\varphi = 1/3$, $t = 8$, $m = 85$ and compare with the $0.683922$ printed by Code Example 4.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> The inverse QFT sends \(\lvert k \rangle \mapsto 2^{-t/2}\sum_m e^{-2\pi i km/2^t}\lvert m \rangle\), so the amplitude of \(\lvert m \rangle\) is \(2^{-t}\sum_k e^{2\pi i(\varphi - m/2^t)k}\). The geometric sum with ratio \(z = e^{2\pi i \delta}\) is \((z^{2^t}-1)/(z-1)\).</p>

<p><strong>2.</strong> \(\lvert (z^{2^t}-1)/(z-1) \rvert = \lvert \sin(2^t \pi \delta)/\sin(\pi\delta)\rvert\), and squaring with the \(2^{-t}\) prefactor gives the stated \(P(m)\).</p>

<p><strong>3.</strong> At \(\delta = 2^{-t-1}\): \(\sin^2(2^t \pi \delta) = \sin^2(\pi/2) = 1\) and \(2^{2t}\sin^2(\pi 2^{-t-1}) \to 2^{2t}(\pi 2^{-t-1})^2 = \pi^2/4\). So \(P \to 4/\pi^2 = 0.4053\).</p>

<p><strong>4.</strong> \(\delta = 1/3 - 85/256 = 1/768\). Then \(\sin^2(256\pi/768) = \sin^2(\pi/3) = 3/4\) and \(2^{16}\sin^2(\pi/768) = 65536 \times 1.673304\times10^{-5} = 1.096617\), giving \(P = 0.75/1.096617 = 0.683922\) — exactly the printed value, because the formula is exact and nothing was approximated. (The small-angle limit \(2^{16}(\pi/768)^2 = 1.096623\) would give the same five digits, which is why the difference is easy to miss.)</p>

</details>

#### Exercise 4: Iterative Phase Estimation Fails Gracefully

Consider iterative QPE on $\varphi$ with $t$ rounds and one repetition per round.

  1. Argue that if $\varphi$ has an exact $t$-bit expansion, every round measures its bit with probability 1. Where in the argument is exactness used?
  2. Suppose the tail beyond bit $t$ is $0.0\ldots0b_{t+1}b_{t+2}\ldots$. Show that round $k$ measures the correct $b_k$ with probability $\cos^2(\pi \eta_k)$ for some residual $\eta_k$, and identify $\eta_k$.
  3. Why does an error on a *low* bit corrupt the bits above it, whereas in textbook QPE a mis-measurement affects only that outcome?
  4. Code Example 6 reports mean errors of $0.01367$ at one repetition per round and $0.00522$ at twenty-five, for $t = 6$, $\varphi = 1/3$. Explain both numbers.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> If \(\varphi = 0.b_1\ldots b_t\) then \(2^{k-1}\varphi = \text{integer} + 0.b_kb_{k+1}\ldots b_t\), and the feedback subtracts \(0.0b_{k+1}\ldots b_t\) exactly, leaving a relative phase of \(\pi b_k\). The Hadamard then maps \((\lvert 0\rangle + e^{i\pi b_k}\lvert 1 \rangle)/\sqrt{2}\) to \(\lvert b_k \rangle\) with certainty. Exactness is used twice: the tail must terminate at bit \(t\), and the previously measured bits must be correct.</p>

<p><strong>2.</strong> The residual relative phase is \(2\pi(0.b_k + \text{tail})\) minus \(\pi b_k\), i.e. \(2\pi \eta_k\) with \(\eta_k = 2^{k-1}\varphi - (\text{known bits}) - b_k/2 \bmod 1\), the un-cancelled tail \(0.00b_{t+1}b_{t+2}\ldots\) scaled by \(2^{k-1}\). The Hadamard-then-measure step returns \(b_k\) with probability \(\cos^2(\pi\eta_k)\).</p>

<p><strong>3.</strong> Because the feedback angle at round \(k\) is computed from the bits measured in rounds \(k+1,\ldots,t\). A wrong bit there makes the cancellation wrong by a known-to-be-wrong amount, so the residual phase in every later round is displaced and the errors compound. Textbook QPE has no feedback: each shot is an independent sample of a fixed distribution.</p>

<p><strong>4.</strong> At twenty-five repetitions the per-round votes are essentially always correct, so the estimate is the best 6-bit truncation \(21/64 = 0.328125\), and \(\lvert 1/3 - 21/64\rvert = 0.005208\) — the printed \(0.00522\) is the residual failure rate on top of that floor. At one repetition the extra \(0.0085\) is the contribution of runs in which a bit flipped; because low-bit errors propagate upwards, those runs are wrong by much more than one least significant bit, which is why the mean error is 2.6 times the floor rather than slightly above it.</p>

</details>

#### Exercise 5: Reading a Resource Claim

A seminar speaker states: "Our method computes the ground-state energy of a 100-orbital active space to chemical accuracy using phase estimation with 200 logical qubits."

  1. Which three quantities does that sentence leave out, each of which can change the cost by orders of magnitude?
  2. Estimate the number of applications of $e^{-iH\tau}$ implied, assuming the spectral span of a 100-orbital Hamiltonian is of order 100 Ha.
  3. The speaker's trial state is a single Hartree-Fock determinant and the system is strongly correlated. What does that do to the number of repetitions, and how would you ask about it politely?
  4. What would make the claim checkable rather than merely plausible?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> (i) The circuit depth, or equivalently the total non-Clifford gate count — 200 logical qubits is the <em>cheapest</em> of the three resources, as the introductory course's Chapter 4 emphasized. (ii) The cost of implementing one \(e^{-iH\tau}\), which depends entirely on the simulation method (Trotter order, block encoding, qubitization) and on the Hamiltonian's structure. (iii) The overlap \(p_0\) of the trial state with the target eigenvector, which multiplies the whole runtime by \(1/p_0\).</p>

<p><strong>2.</strong> Chemical accuracy is \(1.6\times10^{-3}\) Ha. With a spectral span \(\Lambda \sim 100\) Ha the phase resolution needed is \(\varepsilon_\varphi \sim 1.6\times10^{-3}/100 = 1.6\times10^{-5}\), so \(t \approx \lceil \log_2(1/1.6\times10^{-5})\rceil = 16\) plus padding bits, and \(2^t - 1 \sim 10^5\) applications of \(e^{-iH\tau}\). Each of those is itself \(10^3\)–\(10^6\) gates depending on the method, giving \(10^8\)–\(10^{11}\) — the range that the published fault-tolerant estimates for chemistry actually occupy.</p>

<p><strong>3.</strong> A strongly correlated ground state has a Hartree-Fock overlap that can be small and that generally shrinks with system size, so the expected number of repetitions is \(1/p_0\) and may be large. The polite question is simply "what is \(p_0\) for your trial state, and how does it scale with the active-space size?" — a question with a definite numerical answer that any complete analysis has already computed.</p>

<p><strong>4.</strong> A T-count or Toffoli count for the full circuit, the assumed physical error rate and code distance behind any physical-qubit number, the value of \(p_0\), and a smaller instance of the same pipeline benchmarked against exact diagonalization. The last of these is the one this course insists on: every claim in this chapter is checked against <code>numpy.linalg.eigh</code> on the same Hamiltonian.</p>

</details>

* * *

## Summary

### Key Takeaways

**1\. The QFT is a short circuit for a familiar matrix**

  * $\mathrm{QFT}\lvert j \rangle$ factorizes into a product state, one phase per qubit, which is why the circuit is $n$ Hadamards, $n(n-1)/2$ controlled phase rotations and $\lfloor n/2 \rfloor$ swaps.
  * Verified against the dense DFT matrix to $10^{-14}$ at $n = 7$; dropping the smallest rotations gives the $O(n\log n)$ approximate QFT used in practice.
  * The comparison with the classical FFT is not an exponential speedup: the input must already be a quantum state, one run returns one sampled index, and estimating a single output probability costs $O(1/\delta^2)$ shots by sampling.

**2\. What the QFT actually delivers is a period**

  * A state supported on a stride-$r$ progression transforms into peaks at multiples of $2^n/r$, with the offset stored entirely in phases — the distributions for different offsets agree to $10^{-50}$.
  * When $r$ divides $2^n$ the peaks are exact and $100\%$ of the probability is on them; when it does not, $79\%$ sits in the best bins and classical postprocessing is required.
  * That postprocessing is continued fractions, and it is the subject of Chapter 3.

**3\. Phase estimation is the QFT run backwards**

  * Controlled powers $U^{2^j}$ write $2^{-t/2}\sum_k e^{2\pi i \varphi k}\lvert k \rangle$ into the counting register; the inverse QFT reads it as $\lvert 2^t\varphi \rangle$.
  * Cost $2^t - 1$ applications of $U$, so precision $\varepsilon$ costs $\Theta(1/\varepsilon)$ depth; the phase error stayed below $2^{-(t+1)}$ at every $t$ tested, and the energy error halved per added qubit with fitted slope $-1.079$.
  * Extra qubits buy resolution, not confidence: $P_{\text{best}}$ for $\varphi = 1/3$ is $0.6839$ independent of $t$, floored at $4/\pi^2$. Confidence comes from the $\lceil\log_2(2+1/(2\varepsilon))\rceil$ padding bits, which buy $\Pr[\lvert\tilde{\varphi}-\varphi\rvert < 2^{-n}] \ge 1-\varepsilon$ — an accuracy guarantee, not a promise about the leading $n$ bits.

**4\. Iterative QPE trades qubits for rounds**

  * One ancilla and $t$ rounds of measurement with classical feedback replace $t$ counting qubits and the inverse QFT.
  * Exact for dyadic phases; for generic phases a low-bit error propagates upwards, so rounds are repeated and majority-voted — $85\%$ success at one repetition, $99.7\%$ at nine, at nine times the applications of $U$.
  * On the three-site chain it turned 11 qubits into 4 and returned the identical energy.

**5\. The application is eigenvalues, and the obstacle is overlap**

  * Electronic structure is an eigenvalue problem, so phase estimation is the fault-tolerant successor to VQE: an eigenvalue instead of a variational bound, $1/\varepsilon$ instead of $1/\varepsilon^2$, excited states for free, at $\Theta(1/\varepsilon)$ coherent depth.
  * On the two-qubit H$_2$ Hamiltonian it reached chemical accuracy at $t = 10$ with 1023 controlled applications of $e^{-iH\tau}$ — no optimizer, no averaging.
  * Feed it a state with $1.2\%$ ground-state overlap and it confidently returns the first excited energy. The number of repetitions scales as $1/p_0$, and for strongly correlated systems $p_0$ is the open problem.

**Practical implications**

  * Choose $t$ from the required precision *and* the required confidence; they are separate purchases with different prices.
  * Always report the overlap of your trial state alongside any phase-estimation cost. A resource estimate without $p_0$ is incomplete.
  * When comparing against a variational method, compare $1/\varepsilon$ against $1/\varepsilon^2$ and coherent depth against shot count — those are the axes on which the two differ.

### Where This Leads

Chapter 3 takes the two circuits built here — the QFT of Example 2 and the phase-estimation machinery of Examples 4 and 5 — and points them at a unitary whose eigenphases are $s/r$ for the multiplicative order $r$ of an integer modulo $N$. The result is Shor's algorithm, and it is the one case in this course where the speedup is superpolynomial and not in dispute. The chapter factors 15 and 21 end to end on the simulator, including the continued-fraction postprocessing that Example 3 showed to be unavoidable, and then states plainly what the same circuit costs at cryptographic sizes and what the standard response to that cost already is.

[← Chapter 1: Amplitude Amplification and Grover's Algorithm](<chapter-1.html>) [Chapter 3: Shor's Algorithm →](<chapter-3.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The resource figures in this chapter — gate counts, Trotter-step placeholders and error-rate requirements — are order-of-magnitude teaching estimates derived from the stated assumptions, not measurements or predictions. Verify against primary sources before using them in a proposal or a paper.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
