---
title: "Chapter 4: Modern Hamiltonian Simulation"
chapter_title: "Chapter 4: Modern Hamiltonian Simulation"
subtitle: Block Encoding, Qubitization, Randomised Compilation, and How to Speak in Toffolis
reading_time: 45-50 minutes
difficulty: Advanced
code_examples: 7
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-algorithms-intermediate/chapter-4.html>) | Last sync: 2026-08-13

[Fundamental Mathematics Dojo](<../index.html>) > [Intermediate Quantum Algorithms](<index.html>) > Chapter 4

Phase estimation, built in [Chapter 2](<chapter-2.html>), turns a unitary into an eigenvalue. For chemistry and materials the unitary we want is $e^{-iHt}$, and the sister course built it the obvious way: chop the exponential into a product of Pauli rotations and accept an error. [Introduction to Quantum Computing, Chapter 4](<../quantum-computing-introduction/chapter-4.html>) measured that error rather than quoting its asymptotic form, and the measurement was discouraging — about $3 \times 10^{7}$ Pauli rotations to hold a single time unit of a four-qubit Hubbard dimer to $10^{-6}$, for a model whose exact answer comes out of `numpy.linalg.eigvalsh` in microseconds. This chapter is what the field did about it.

There are three ideas here, in increasing order of departure from Trotterization. **Block encoding** puts $H$ itself — not its exponential — inside a larger unitary, at the cost of one ancilla register and a normalisation factor. **Qubitization** turns that unitary into a quantum walk whose eigenphases are $\arccos$ of the eigenvalues of the normalised Hamiltonian, which makes phase estimation on the walk optimal in the query model. **qDRIFT** goes the other way and abandons determinism altogether, sampling Hamiltonian terms at random. Section 4.4 is about the language in which all of this is reported. A fault-tolerant algorithm paper does not deliver a runtime; it delivers a Toffoli count, a logical-qubit count, and a set of assumptions, and reading one of those correctly is a skill worth acquiring on purpose.

## Learning Objectives

After completing this chapter, you will be able to:

  * Recall the first- and second-order product formulas, and state their error scaling in the form that matters for eigenvalue problems — a $1/\varepsilon^{1/2k}$ dependence that no finite order removes
  * Define an $(\alpha, m, 0)$-block-encoding of a Hamiltonian, and explain why the normalisation $\alpha$ is a 1-norm rather than a spectral norm
  * Construct the linear-combination-of-unitaries block encoding explicitly from PREPARE and SELECT, take both down to gate level on a two-qubit example, and verify that the top-left block of the resulting $16 \times 16$ unitary *is* $H/\alpha$
  * Compute the success probability of one LCU round, relate it to $\lVert H \lvert \psi \rangle \rVert / \alpha$, and count the amplitude-amplification rounds needed to make it near-certain
  * Build the qubitization walk $W$ and verify numerically that its eigenphases satisfy $\cos\theta_k = E_k/\alpha$, then run phase estimation on $W$ to recover a ground-state energy
  * Evaluate the qDRIFT channel exactly as a superoperator, confirm its $O(\lambda^2 t^2/N)$ error and its independence of the term count $L$, and locate the crossover against second-order Trotter
  * Read and write a fault-tolerant resource estimate in Toffoli counts, logical qubits and surface-code distance, and identify which of its inputs the answer is actually sensitive to

### What Carries Over

Everything below runs on the mini-simulator from [Introduction to Quantum Computing, Chapter 2](<../quantum-computing-introduction/chapter-2.html>), re-listed in Example 1 so that this chapter is self-contained. The convention is **big-endian**: qubit 0 is the leftmost symbol in the ket and the most significant bit of the amplitude index. That convention earns its keep immediately in Section 4.2, because it makes the ancilla register occupy the *leading* indices of the state vector, so "the block of the unitary in which the Hamiltonian lives" is literally the top-left corner of a printed matrix.

Two symbols are used throughout and are worth separating now. $\alpha$ is the normalisation of a block encoding, and for the LCU construction it equals $\lambda = \sum_l \lvert c_l \rvert$, the **1-norm** of the Hamiltonian's Pauli decomposition. $\lVert H \rVert$ is the spectral norm. Always $\alpha \ge \lVert H \rVert$, usually by a factor that grows with system size, and every cost formula in this chapter contains $\alpha$ rather than $\lVert H \rVert$. Confusing the two is the fastest way to underestimate a resource requirement by an order of magnitude.

* * *

## 4.1 Trotter, and Where It Stops Being Enough

### The product formula, recalled

Digital simulation of $e^{-iHt}$ starts from a decomposition $H = \sum_{j=1}^{L} H_j$ into pieces we can exponentiate exactly — for us, Pauli strings with real coefficients. The pieces do not commute, so the naive product is wrong, and the first-order Lie-Trotter formula quantifies how wrong:

$$ e^{-iHt} = \left(\prod_{j=1}^{L} e^{-iH_j t/r}\right)^{r} + O\left(\frac{t^2}{r}\right) $$

The symmetric second-order formula halves each step and sweeps forward then backward, which cancels the leading error term:

$$ e^{-iHt} \approx \left(\prod_{j=1}^{L} e^{-iH_j t/2r} \prod_{j=L}^{1} e^{-iH_j t/2r}\right)^{r} + O\left(\frac{t^3}{r^2}\right) $$

Each factor $e^{-i\theta P}$ for a Pauli string $P$ compiles into a CNOT ladder, one $R_z$ and the reverse ladder, so the gate count is (number of non-identity strings) $\times$ (steps) $\times$ (a weight-dependent CNOT cost). The identity string is a global phase and needs no gate at all.

### What the error actually scales as

The $O$ notation hides the constant, and the constant is where all the practical difficulty lives. For a $2k$-th order product formula the standard bound is on the *cost*: the number of gates needed to hold the error to $\varepsilon$ is

$$ \text{gates} \; = \; O\left(\frac{L\,(\alpha t)^{1 + 1/2k}}{\varepsilon^{1/2k}}\right) \quad \text{for target error } \varepsilon $$

with tighter, commutator-dependent versions available. Two features of this expression matter more than its exact form.

**The dependence on $1/\varepsilon$ is polynomial and never becomes logarithmic.** Raising the order $k$ improves the exponent to $1/2k$ but multiplies the per-step cost by roughly $5^{k-1}$ stages in the Suzuki recursion, so there is an optimum $k$ for every $(t, \varepsilon)$ and it does not remove the polynomial. For a *dynamics* calculation, where $\varepsilon = 10^{-3}$ is often plenty, this is tolerable. For an *eigenvalue* calculation it is not, because phase estimation to precision $\varepsilon$ needs a coherent evolution to time $t \sim 1/\varepsilon$, and the $t^{1+1/2k}$ factor then compounds with the $\varepsilon^{-1/2k}$ factor.

**The dependence on $L$ is linear, and $L$ is large.** An electronic-structure Hamiltonian on $M$ spatial orbitals has $O(M^4)$ Pauli strings after the Jordan-Wigner transform. At $M = 76$ — the FeMoco active space of the sister course — that is of order $10^{7}$ terms before any truncation, and every one of them appears in every Trotter step.

Let us measure both constants for the Hamiltonian this chapter will use throughout, exactly as the sister course did for the Hubbard dimer, so that the qubitization numbers later have something concrete to be compared against.

### Code Example 1: The Mini-Simulator, Re-listed

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

Ninety-nine lines and no dependency beyond NumPy. Save it as `qcsim.py`; every later example in this chapter begins with `from qcsim import *`.

### Code Example 2: The Running Hamiltonian and the Trotter Baseline

```python
"""Chapter 4, Example 2: the running Hamiltonian and the Trotter baseline.
Runs on qcsim.py from Example 1."""
import numpy as np
from functools import reduce
from qcsim import *

def pauli_matrix(s):
    """Dense matrix of a Pauli string such as 'ZZ' (qubit 0 leftmost)."""
    return reduce(np.kron, [PAULI[c] for c in s])


def to_matrix(terms):
    """Dense matrix of {'ZZ': 0.8, ...}."""
    n = len(next(iter(terms)))
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for s, c in terms.items():
        M += c * pauli_matrix(s)
    return M


def expm_hermitian(M, scalar):
    """exp(scalar * M) for Hermitian M, via its eigendecomposition."""
    w, v = np.linalg.eigh(M)
    return (v * np.exp(scalar * w)) @ v.conj().T


# The chapter's running example: a two-qubit Hamiltonian with four terms,
# chosen so that the LCU coefficient register is exactly two qubits.
HAM = {'IX': 0.3, 'XI': 0.5, 'YY': 0.2, 'ZZ': 0.8}
STRINGS = sorted(HAM)
COEF = np.array([HAM[s] for s in STRINGS])
ALPHA = COEF.sum()                      # the 1-norm  alpha = sum_l |c_l|
Hmat = to_matrix(HAM)

print("The running example Hamiltonian")
print("=" * 68)
for s, c in zip(STRINGS, COEF):
    print(f"    {c:+.4f} * {s}")
print(f"  1-norm            alpha = sum_l |c_l|  = {ALPHA:.4f}")
print(f"  spectral norm     ||H||               = "
      f"{np.linalg.norm(Hmat, ord=2):.6f}")
print(f"  eigenvalues                           = "
      f"{np.round(np.linalg.eigvalsh(Hmat), 6)}")
print(f"  alpha / ||H||                         = "
      f"{ALPHA/np.linalg.norm(Hmat, ord=2):.4f}"
      "   (the price of the LCU representation)")

tau = 1.0
U_exact = expm_hermitian(Hmat, -1j * tau)
terms = [(s, HAM[s]) for s in STRINGS]

print(f"\nFirst-order Lie-Trotter, r steps of dt = tau/r,  tau = {tau}")
print(f"{'steps r':>8} {'dt':>9} {'spectral error':>16} {'error x r':>11}"
      f" {'rotations':>10}")
for r in (1, 2, 4, 8, 16, 32, 64, 128):
    dt = tau / r
    step = np.eye(4, dtype=complex)
    for s, c in terms:
        step = expm_hermitian(to_matrix({s: c}), -1j * dt) @ step
    err = np.linalg.norm(np.linalg.matrix_power(step, r) - U_exact, ord=2)
    print(f"{r:8d} {dt:9.5f} {err:16.6e} {err*r:11.5f} {r*len(terms):10d}")

print("\nSecond-order (symmetric) Suzuki formula")
print(f"{'steps r':>8} {'dt':>9} {'spectral error':>16} {'error x r^2':>13}"
      f" {'rotations':>10}")
for r in (1, 2, 4, 8, 16, 32):
    dt = tau / r
    step = np.eye(4, dtype=complex)
    for s, c in terms:
        step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
    for s, c in reversed(terms):
        step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
    err = np.linalg.norm(np.linalg.matrix_power(step, r) - U_exact, ord=2)
    print(f"{r:8d} {dt:9.5f} {err:16.6e} {err*r*r:13.5f} "
          f"{2*r*len(terms):10d}")

# the two asymptotic constants, measured
dt = tau / 128
step = np.eye(4, dtype=complex)
for s, c in terms:
    step = expm_hermitian(to_matrix({s: c}), -1j * dt) @ step
C1 = np.linalg.norm(np.linalg.matrix_power(step, 128) - U_exact, ord=2) * 128
dt = tau / 32
step = np.eye(4, dtype=complex)
for s, c in terms:
    step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
for s, c in reversed(terms):
    step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
C2 = np.linalg.norm(np.linalg.matrix_power(step, 32) - U_exact, ord=2) * 32 ** 2

print("\nMeasured constants and the extrapolated cost at tau = 1")
print("-" * 68)
print(f"  first order:   error ~ C1 / r    with C1 = {C1:.4f}")
print(f"  second order:  error ~ C2 / r^2  with C2 = {C2:.4f}")
print(f"{'target error':>13} {'r (1st)':>12} {'rotations':>12}"
      f" {'r (2nd)':>10} {'rotations':>12}")
for eps in (1e-3, 1e-6, 1e-9):
    r1 = C1 / eps
    r2 = np.sqrt(C2 / eps)
    print(f"{eps:13.0e} {r1:12.3e} {r1*4:12.3e} {r2:10.1f} {r2*8:12.3e}")
```

```text
The running example Hamiltonian
====================================================================
    +0.3000 * IX
    +0.5000 * XI
    +0.2000 * YY
    +0.8000 * ZZ
  1-norm            alpha = sum_l |c_l|  = 1.8000
  spectral norm     ||H||               = 1.019804
  eigenvalues                           = [-1.019804 -1.        1.        1.019804]
  alpha / ||H||                         = 1.7650   (the price of the LCU representation)

First-order Lie-Trotter, r steps of dt = tau/r,  tau = 1.0
 steps r        dt   spectral error   error x r  rotations
       1   1.00000     4.292796e-01     0.42928          4
       2   0.50000     2.051686e-01     0.41034          8
       4   0.25000     1.013788e-01     0.40552         16
       8   0.12500     5.053855e-02     0.40431         32
      16   0.06250     2.525042e-02     0.40401         64
      32   0.03125     1.262285e-02     0.40393        128
      64   0.01562     6.311131e-03     0.40391        256
     128   0.00781     3.155528e-03     0.40391        512

Second-order (symmetric) Suzuki formula
 steps r        dt   spectral error   error x r^2  rotations
       1   1.00000     1.065662e-01       0.10657          8
       2   0.50000     2.528742e-02       0.10115         16
       4   0.25000     6.234274e-03       0.09975         32
       8   0.12500     1.553070e-03       0.09940         64
      16   0.06250     3.879236e-04       0.09931        128
      32   0.03125     9.695940e-05       0.09929        256

Measured constants and the extrapolated cost at tau = 1
--------------------------------------------------------------------
  first order:   error ~ C1 / r    with C1 = 0.4039
  second order:  error ~ C2 / r^2  with C2 = 0.0993
 target error      r (1st)    rotations    r (2nd)    rotations
        1e-03    4.039e+02    1.616e+03       10.0    7.971e+01
        1e-06    4.039e+05    1.616e+06      315.1    2.521e+03
        1e-09    4.039e+08    1.616e+09     9964.3    7.971e+04
```

**What to notice.** The two constants are $C_1 = 0.4039$ and $C_2 = 0.0993$, and they are the entire content of the $O(1/r)$ and $O(1/r^2)$ statements for this Hamiltonian. Everything downstream in this chapter is measured against them.

The last block is the honest cost of an eigenvalue calculation done with product formulas. Reaching $10^{-9}$ — which is not an unreasonable target when the quantity is an energy difference between two nearly degenerate spin states — needs $1.6 \times 10^{9}$ Pauli rotations at first order and $8.0 \times 10^{4}$ at second order for a *four-dimensional* Hilbert space. The second-order improvement is a factor of twenty thousand, and it is still a polynomial in $1/\varepsilon$; that is the gap qubitization closes.

Note also $\alpha/\lVert H \rVert = 1.765$ in the header block. The 1-norm exceeds the spectral norm by 76% for a Hamiltonian with four terms. Example 4 shows what that ratio does as the system grows.

* * *

## 4.2 Block Encoding, LCU and Qubitization

### The move: encode $H$, not $e^{-iHt}$

A Hamiltonian is not a unitary, so a quantum computer cannot apply it directly. Trotterization sidesteps this by only ever applying exponentials. Block encoding takes the opposite route: embed the non-unitary $H$ as one block of a unitary acting on a larger space, and then manipulate that unitary.

Formally, a unitary $U_A$ acting on $m$ ancilla qubits plus the system is an **$(\alpha, m, 0)$-block-encoding** of $H$ if

$$ \left(\langle 0^{m} \rvert \otimes I\right) U_A \left(\lvert 0^{m} \rangle \otimes I\right) = \frac{H}{\alpha} $$

In big-endian ordering with the ancillas first, this says exactly that the top-left $2^n \times 2^n$ corner of the matrix $U_A$ equals $H/\alpha$:

$$ U_A = \begin{pmatrix} H/\alpha & \ast \cr \ast & \ast \end{pmatrix} $$

The starred blocks are not free parameters — unitarity fixes their norms — but they are also not our concern, because every algorithm built on a block encoding either post-selects the ancillas on $\lvert 0^m \rangle$ or reflects about that subspace. The normalisation $\alpha$ is unavoidable: the top-left block of a unitary has spectral norm at most 1, so $\alpha \ge \lVert H \rVert$ always.

### LCU: PREPARE and SELECT

The standard construction writes $H$ as a linear combination of unitaries,

$$ H = \sum_{l=0}^{L-1} c_l\, U_l, \qquad c_l > 0, \qquad \alpha = \sum_l c_l $$

which for a Pauli decomposition is free: absorb any sign into $U_l = \pm P_l$ so that all $c_l$ are positive. Two circuits then suffice.

**PREPARE** acts on the $m = \lceil \log_2 L \rceil$ ancilla qubits and loads the square roots of the normalised coefficients:

$$ \mathrm{PREP} \lvert 0^{m} \rangle = \sum_{l} \sqrt{\frac{c_l}{\alpha}} \lvert l \rangle $$

**SELECT** applies $U_l$ to the system, controlled on the ancilla register reading $l$:

$$ \mathrm{SELECT} = \sum_{l} \lvert l \rangle\langle l \rvert \otimes U_l $$

Sandwiching one between the other gives the block encoding:

$$ U_A = \left(\mathrm{PREP}^{\dagger} \otimes I\right) \mathrm{SELECT} \left(\mathrm{PREP} \otimes I\right) $$

and the defining property follows in one line, because $\langle 0^m \rvert \mathrm{PREP}^\dagger \lvert l \rangle = \sqrt{c_l/\alpha}$ and the same factor appears on the right:

$$ \left(\langle 0^{m} \rvert \otimes I\right) U_A \left(\lvert 0^{m} \rangle \otimes I\right) = \sum_l \sqrt{\frac{c_l}{\alpha}} \sqrt{\frac{c_l}{\alpha}}\, U_l = \frac{1}{\alpha}\sum_l c_l U_l = \frac{H}{\alpha} $$

### Both circuits at gate level

The formulas above are not yet circuits, and the whole cost of the method sits in turning them into circuits.

**PREPARE for real non-negative amplitudes** is a binary tree of $R_y$ rotations. On two ancilla qubits with target amplitudes $v = (v_{00}, v_{01}, v_{10}, v_{11})$, one $R_y(\theta_a)$ on the first qubit splits the probability between the two halves, and a *multiplexed* $R_y$ on the second qubit — angle $\theta_{b0}$ if the first qubit is $\lvert 0 \rangle$, $\theta_{b1}$ if it is $\lvert 1 \rangle$ — splits within each half. The multiplexor is not a new primitive: with $\theta_{\pm} = (\theta_{b0} \pm \theta_{b1})/2$,

$$ \lvert 0 \rangle\langle 0 \rvert \otimes R_y(\theta_{b0}) + \lvert 1 \rangle\langle 1 \rvert \otimes R_y(\theta_{b1}) = \mathrm{CNOT}\, \left[I \otimes R_y(\theta_{-})\right]\, \mathrm{CNOT}\, \left[I \otimes R_y(\theta_{+})\right] $$

because $X R_y(\theta) X = R_y(-\theta)$. Five gates in total — three $R_y$ and two CNOTs — for an exact two-qubit PREPARE.

**SELECT** is a product of $L$ multiply-controlled operations, one per term, and this is where the Toffolis are. Written naively, each $\lvert l \rangle\langle l \rvert \otimes U_l$ costs a Toffoli tree of depth $m$. The construction actually used in resource estimates is **unary iteration**: compute the $L$ control conditions incrementally down a ladder of ancillas, reusing partial products, which brings the whole SELECT to $L - 1$ Toffolis rather than $O(L \log L)$. It is a small combinatorial trick with a large effect on published counts, and it is the reason SELECT costs *one* Toffoli per Hamiltonian term.

### Code Example 3: Explicit Block Encoding of a Two-Qubit Hamiltonian

```python
"""Chapter 4, Example 3: explicit block encoding of a two-qubit Hamiltonian.
Continues from Example 2 (same session)."""
I4 = np.eye(4, dtype=complex)
E4 = np.eye(4, dtype=complex)


def prepare_angles(coef):
    """Ry angles of a two-qubit PREPARE for real non-negative amplitudes."""
    v = np.sqrt(coef / coef.sum())
    theta_a = 2 * np.arctan2(np.linalg.norm(v[2:]), np.linalg.norm(v[:2]))
    theta_b0 = 2 * np.arctan2(v[1], v[0])
    theta_b1 = 2 * np.arctan2(v[3], v[2])
    return v, theta_a, theta_b0, theta_b1


AMP, TH_A, TH_B0, TH_B1 = prepare_angles(COEF)
TH_P, TH_M = (TH_B0 + TH_B1) / 2, (TH_B0 - TH_B1) / 2

# PREPARE as a five-gate circuit: Ry(0), Ry(1), CNOT, Ry(1), CNOT
PREP = CNOT4 @ np.kron(I2, ry(TH_M)) @ CNOT4 @ np.kron(I2, ry(TH_P)) \
       @ np.kron(ry(TH_A), I2)

print("PREPARE: the four Ry angles and the resulting amplitudes")
print("=" * 68)
print(f"  target amplitudes sqrt(c_l/alpha) = {np.round(AMP, 6)}")
print(f"  theta_a  = {TH_A:.6f} rad   (Ry on the first ancilla)")
print(f"  theta_b0 = {TH_B0:.6f} rad,  theta_b1 = {TH_B1:.6f} rad"
      "   (multiplexed Ry)")
print(f"  compiled as Ry({TH_P:.6f}) . CNOT . Ry({TH_M:.6f}) . CNOT"
      "  on the second ancilla")
print(f"  PREP|00> = {np.round(PREP[:, 0].real, 6)}")
print(f"  max deviation from the target = "
      f"{np.max(np.abs(PREP[:, 0] - AMP)):.2e}")
print(f"  PREP unitary? {np.allclose(PREP.conj().T @ PREP, I4)}"
      f"    real? {np.allclose(PREP.imag, 0)}")

# SELECT as a product of four doubly-controlled Paulis
CC = []
for l, s in enumerate(STRINGS):
    proj = np.outer(E4[l], E4[l])
    CC.append(np.eye(16, dtype=complex)
              - np.kron(proj, I4) + np.kron(proj, pauli_matrix(s)))
SELECT = reduce(lambda a, b: a @ b, CC)
SELECT_direct = sum(np.kron(np.outer(E4[l], E4[l]), pauli_matrix(STRINGS[l]))
                    for l in range(4))

print("\nSELECT: a product of four doubly-controlled Paulis")
print("-" * 68)
for l, s in enumerate(STRINGS):
    print(f"  ancilla |{l//2}{l%2}>  ->  apply {s} to the system")
print(f"  product of the four CC-Paulis == sum_l |l><l| (x) P_l ? "
      f"{np.allclose(SELECT, SELECT_direct)}")
print(f"  SELECT Hermitian? {np.allclose(SELECT, SELECT.conj().T)}"
      f"   SELECT^2 = I? {np.allclose(SELECT @ SELECT, np.eye(16))}")

U_A = np.kron(PREP.conj().T, I4) @ SELECT @ np.kron(PREP, I4)

print("\nThe block encoding U_A = (PREP^dag (x) I) SELECT (PREP (x) I)")
print("-" * 68)
print(f"  U_A is 16 x 16, unitary? "
      f"{np.allclose(U_A.conj().T @ U_A, np.eye(16))}")
print(f"  U_A Hermitian (hence a reflection)? "
      f"{np.allclose(U_A, U_A.conj().T)}")
print("\n  top-left 4 x 4 block of U_A (real part):")
for row in np.round(U_A[:4, :4].real, 6):
    print("     ", "  ".join(f"{x:+.6f}" for x in row))
print("\n  H / alpha (real part):")
for row in np.round((Hmat / ALPHA).real, 6):
    print("     ", "  ".join(f"{x:+.6f}" for x in row))
print(f"\n  max |U_A[:4,:4] - H/alpha| = "
      f"{np.max(np.abs(U_A[:4, :4] - Hmat/ALPHA)):.3e}"
      "   <- the defining property")
print(f"  spectral norm of the top-left block = "
      f"{np.linalg.norm(U_A[:4, :4], ord=2):.6f}"
      f"  (= ||H||/alpha = {np.linalg.norm(Hmat, ord=2)/ALPHA:.6f})")
print("  the other blocks are not small: they carry the rest of the"
      " unitarity budget,")
print(f"  e.g. ||U_A[:4,4:]|| = {np.linalg.norm(U_A[:4, 4:], ord=2):.6f}")

# post-selection on the simulator
rng = np.random.default_rng(11)
psi_sys = rng.normal(size=4) + 1j * rng.normal(size=4)
psi_sys /= np.linalg.norm(psi_sys)
full = np.kron(ket('00'), psi_sys)
out = apply_gate(full, U_A, [0, 1, 2, 3], 4)
branch = out[:4]
p00 = float(np.vdot(branch, branch).real)
target = Hmat @ psi_sys / ALPHA

print("\nRunning it on the simulator: post-select the ancillas on |00>")
print("-" * 68)
print(f"  P(ancilla = 00)              = {p00:.8f}")
print(f"  ||H|psi>||^2 / alpha^2       = "
      f"{np.linalg.norm(Hmat @ psi_sys)**2 / ALPHA**2:.8f}")
print(f"  max |branch - H|psi>/alpha|  = "
      f"{np.max(np.abs(branch - target)):.3e}")
print("  so the surviving branch is exactly H|psi>/alpha, unnormalised.")

print("\nToffoli bookkeeping for this SELECT")
print("-" * 68)
L = len(STRINGS)
print(f"  terms L = {L},  coefficient register = "
      f"{int(np.ceil(np.log2(L)))} qubits")
print(f"  naive: one doubly-controlled Pauli per term, 2 Toffolis each"
      f"  -> {2*L} Toffolis")
print(f"  unary iteration (the standard construction): L - 1 = {L-1}"
      " Toffolis for the whole SELECT")
print("  PREPARE for L amplitudes costs O(L) Toffolis with QROM,"
      " O(sqrt(L)) with QROAM.")
```

```text
PREPARE: the four Ry angles and the resulting amplitudes
====================================================================
  target amplitudes sqrt(c_l/alpha) = [0.408248 0.527046 0.333333 0.666667]
  theta_a  = 1.682137 rad   (Ry on the first ancilla)
  theta_b0 = 1.823477 rad,  theta_b1 = 2.214297 rad   (multiplexed Ry)
  compiled as Ry(2.018887) . CNOT . Ry(-0.195410) . CNOT  on the second ancilla
  PREP|00> = [0.408248 0.527046 0.333333 0.666667]
  max deviation from the target = 1.11e-16
  PREP unitary? True    real? True

SELECT: a product of four doubly-controlled Paulis
--------------------------------------------------------------------
  ancilla |00>  ->  apply IX to the system
  ancilla |01>  ->  apply XI to the system
  ancilla |10>  ->  apply YY to the system
  ancilla |11>  ->  apply ZZ to the system
  product of the four CC-Paulis == sum_l |l><l| (x) P_l ? True
  SELECT Hermitian? True   SELECT^2 = I? True

The block encoding U_A = (PREP^dag (x) I) SELECT (PREP (x) I)
--------------------------------------------------------------------
  U_A is 16 x 16, unitary? True
  U_A Hermitian (hence a reflection)? True

  top-left 4 x 4 block of U_A (real part):
      +0.444444  +0.166667  +0.277778  -0.111111
      +0.166667  -0.444444  +0.111111  +0.277778
      +0.277778  +0.111111  -0.444444  +0.166667
      -0.111111  +0.277778  +0.166667  +0.444444

  H / alpha (real part):
      +0.444444  +0.166667  +0.277778  -0.111111
      +0.166667  -0.444444  +0.111111  +0.277778
      +0.277778  +0.111111  -0.444444  +0.166667
      -0.111111  +0.277778  +0.166667  +0.444444

  max |U_A[:4,:4] - H/alpha| = 8.327e-17   <- the defining property
  spectral norm of the top-left block = 0.566558  (= ||H||/alpha = 0.566558)
  the other blocks are not small: they carry the rest of the unitarity budget,
  e.g. ||U_A[:4,4:]|| = 0.831479

Running it on the simulator: post-select the ancillas on |00>
--------------------------------------------------------------------
  P(ancilla = 00)              = 0.31090298
  ||H|psi>||^2 / alpha^2       = 0.31090298
  max |branch - H|psi>/alpha|  = 1.130e-16
  so the surviving branch is exactly H|psi>/alpha, unnormalised.

Toffoli bookkeeping for this SELECT
--------------------------------------------------------------------
  terms L = 4,  coefficient register = 2 qubits
  naive: one doubly-controlled Pauli per term, 2 Toffolis each  -> 8 Toffolis
  unary iteration (the standard construction): L - 1 = 3 Toffolis for the whole SELECT
  PREPARE for L amplitudes costs O(L) Toffolis with QROM, O(sqrt(L)) with QROAM.
```

**What to notice.** The top-left $4 \times 4$ block of $U_A$ and the matrix $H/\alpha$ are printed one above the other and they agree entry by entry to $8 \times 10^{-17}$. There is no approximation anywhere in the construction: the block encoding is exact, and what it costs is one extra register and the factor $\alpha$.

Two details are worth dwelling on because they are easy to get backwards. First, the off-diagonal block has spectral norm $0.831$, not something small — a block encoding does not make the unwanted parts of $U_A$ negligible, it makes them *addressable* by post-selection or reflection. Second, $U_A$ came out Hermitian, and therefore $U_A^2 = I$: it is a reflection. That is not an accident of this example. It happens whenever SELECT is self-inverse (a product of Paulis is) and PREPARE is real, and Section 4.2's walk operator depends on it.

The final block runs the encoding on the simulator rather than as matrix algebra: prepare $\lvert 00 \rangle \otimes \lvert \psi \rangle$, apply the $16 \times 16$ gate to all four qubits with `apply_gate`, and look at the first four amplitudes. They are exactly $H \lvert \psi \rangle / \alpha$, unnormalised, and their squared norm is the probability that a measurement of the ancillas returns $00$. That single sentence is the entire operational meaning of a block encoding.

### The success probability, and why $\alpha$ is the number to fear

Post-selection succeeds with probability

$$ P(0^{m}) = \frac{\lVert H \lvert \psi \rangle \rVert^{2}}{\alpha^{2}} $$

which for a normalised state is at most $\lVert H \rVert^2/\alpha^2$. This is where the 1-norm bites. Amplitude amplification (Chapter 1) fixes the probability at the cost of $O(1/\sqrt{P}) = O(\alpha / \lVert H \lvert \psi \rangle \rVert)$ rounds, so $\alpha$ enters every cost formula linearly, and $\alpha$ is a sum over terms.

### Code Example 4: The LCU Success Probability, and What $\alpha$ Costs

```python
"""Chapter 4, Example 4: the LCU success probability, and what alpha costs.
Continues from Example 3 (same session)."""
print("Success probability of one LCU round: P = ||H|psi>||^2 / alpha^2")
print("=" * 68)


def lcu_success(psi):
    full = np.kron(ket('00'), psi)
    out = apply_gate(full, U_A, [0, 1, 2, 3], 4)
    return float(np.vdot(out[:4], out[:4]).real)


w_H, v_H = np.linalg.eigh(Hmat)
print(f"{'state':>26} {'P(00) simulated':>17} {'||H psi||^2/a^2':>17}")
cases = [('|00>', ket('00')), ('|01>', ket('01')),
         ('|++>', apply_gate(apply_gate(ket('00'), H, [0], 2), H, [1], 2))]
for k in range(4):
    cases.append((f'eigenstate E = {w_H[k]:+.4f}', v_H[:, k].astype(complex)))
for name, st in cases:
    print(f"{name:>26} {lcu_success(st):17.8f} "
          f"{np.linalg.norm(Hmat@st)**2/ALPHA**2:17.8f}")

rng = np.random.default_rng(4)
ps = []
for _ in range(20000):
    z = rng.normal(size=4) + 1j * rng.normal(size=4)
    z /= np.linalg.norm(z)
    ps.append(np.linalg.norm(Hmat @ z) ** 2 / ALPHA ** 2)
ps = np.array(ps)
print(f"\n  Haar-random states, 20000 draws: mean P = {ps.mean():.6f},"
      f" min {ps.min():.6f}, max {ps.max():.6f}")
print(f"  analytic mean Tr(H^2)/(d alpha^2) = "
      f"{np.trace(Hmat@Hmat).real/(4*ALPHA**2):.6f}")

print("\nAmplitude amplification: rounds to reach near-certain success")
print("-" * 68)
print(f"{'p (one round)':>14} {'rounds ~ pi/(4 arcsin sqrt p)':>31}"
      f" {'P after':>9}")
for p in (0.5, 0.31, 0.1, 0.03, 0.01, 1e-3):
    th = np.arcsin(np.sqrt(p))
    k = int(np.round((np.pi / 2 - th) / (2 * th)))
    print(f"{p:14.4f} {k:31d} {np.sin((2*k+1)*th)**2:9.5f}")
print("  the round count grows as 1/sqrt(p) = alpha/||H|psi>||:"
      " this is the alpha in")
print("  every qubitization cost formula, and it is a 1-norm,"
      " not a spectral norm.")

print("\nWhy alpha is the quantity to worry about: Heisenberg chains")
print("-" * 68)
print(f"{'sites n':>8} {'terms L':>8} {'alpha (1-norm)':>15}"
      f" {'||H||':>10} {'alpha/||H||':>12}")
for n in range(2, 11):
    ch = {}
    for i in range(n - 1):
        for P in 'XYZ':
            s = ''.join(P if q in (i, i + 1) else 'I' for q in range(n))
            ch[s] = 1.0
    a = sum(abs(c) for c in ch.values())
    Hc = to_matrix(ch)
    nrm = np.linalg.norm(Hc, ord=2)
    print(f"{n:8d} {len(ch):8d} {a:15.2f} {nrm:10.4f} {a/nrm:12.4f}")
print("  alpha grows linearly with the number of terms while ||H||"
      " grows more slowly,")
print("  so the ratio drifts upward: the LCU overhead is real and it is"
      " extensive.")
```

```text
Success probability of one LCU round: P = ||H|psi>||^2 / alpha^2
====================================================================
                     state   P(00) simulated   ||H psi||^2/a^2
                      |00>        0.31481481        0.31481481
                      |01>        0.31481481        0.31481481
                      |++>        0.30864198        0.30864198
    eigenstate E = -1.0198        0.32098765        0.32098765
    eigenstate E = -1.0000        0.30864198        0.30864198
    eigenstate E = +1.0000        0.30864198        0.30864198
    eigenstate E = +1.0198        0.32098765        0.32098765

  Haar-random states, 20000 draws: mean P = 0.314835, min 0.308670, max 0.320936
  analytic mean Tr(H^2)/(d alpha^2) = 0.314815

Amplitude amplification: rounds to reach near-certain success
--------------------------------------------------------------------
 p (one round)   rounds ~ pi/(4 arcsin sqrt p)   P after
        0.5000                               0   0.50000
        0.3100                               1   0.96026
        0.1000                               2   0.99856
        0.0300                               4   0.99998
        0.0100                               7   0.99534
        0.0010                              24   0.99956
  the round count grows as 1/sqrt(p) = alpha/||H|psi>||: this is the alpha in
  every qubitization cost formula, and it is a 1-norm, not a spectral norm.

Why alpha is the quantity to worry about: Heisenberg chains
--------------------------------------------------------------------
 sites n  terms L  alpha (1-norm)      ||H||  alpha/||H||
       2        3            3.00     3.0000       1.0000
       3        6            6.00     4.0000       1.5000
       4        9            9.00     6.4641       1.3923
       5       12           12.00     7.7115       1.5561
       6       15           15.00     9.9743       1.5039
       7       18           18.00    11.3450       1.5866
       8       21           21.00    13.4997       1.5556
       9       24           24.00    14.9453       1.6059
      10       27           27.00    17.0321       1.5852
  alpha grows linearly with the number of terms while ||H|| grows more slowly,
  so the ratio drifts upward: the LCU overhead is real and it is extensive.
```

**What to notice.** The simulated probability and the closed form $\lVert H \lvert \psi \rangle \rVert^2/\alpha^2$ agree to all printed digits for every state tried, including the eigenstates, where the probability reduces to $(E_k/\alpha)^2$. The Haar average over 20 000 random states lands on $\mathrm{Tr}(H^2)/(d\,\alpha^2)$, as it must.

The Heisenberg-chain table is the one to remember. The 1-norm grows exactly linearly with the number of terms — three per bond, so $3(n-1)$ — while the spectral norm grows more slowly, and the ratio $\alpha/\lVert H \rVert$ drifts from 1.0 at two sites to about 1.6 at ten. For molecular Hamiltonians the ratio is far worse: $\alpha$ is a sum of $O(M^4)$ integrals with no cancellation, while $\lVert H \rVert$ enjoys plenty. Reducing $\alpha$ by re-factorising the two-electron tensor is therefore not cosmetic work — it is *the* lever on fault-tolerant chemistry costs, and Section 4.4 shows the arithmetic.

### Qubitization: from a reflection to a walk

Post-selection and amplitude amplification are enough to apply $H$ once. Simulating $e^{-iHt}$ needs something better, and the observation that provides it is the following. Let $\Pi = \lvert 0^m \rangle\langle 0^m \rvert \otimes I$ be the projector onto the good ancilla subspace, and define the **quantum walk operator**

$$ W = \left(2\Pi - I\right) U_A $$

a product of two reflections when $U_A$ is itself a reflection. A product of two reflections is a rotation, and the space decomposes into two-dimensional invariant subspaces — one for each eigenvector $\lvert E_k \rangle$ of $H$ — inside which $W$ rotates by an angle set by the overlap. The result is the central identity of qubitization:

$$ \text{eigenvalues of } W \text{ on the relevant subspaces} = e^{\pm i \theta_k}, \qquad \cos\theta_k = \frac{E_k}{\alpha} $$

So a single call to $U_A$ — one PREPARE, one SELECT, one PREPARE$^\dagger$, plus a reflection on $m$ ancillas — advances a walk whose spectrum is a known invertible function of the spectrum of $H$. Phase estimation on $W$ therefore returns $\theta_k$, and $E_k = \alpha \cos\theta_k$ recovers the energy. The $\arccos$ is a nuisance for error propagation near the band edges and nothing worse.

This is stronger than it may look. Estimating an eigenvalue to precision $\varepsilon$ needs $O(\alpha/\varepsilon)$ calls to $U_A$, which is *optimal* — no algorithm can do better in the model where $U_A$ is a black box. Time evolution built from the same walk by quantum signal processing costs $O(\alpha t + \log(1/\varepsilon))$ queries, which is also optimal, and the logarithm is the whole point: doubling the number of accurate digits adds a constant, not a factor.

### Code Example 5: The Qubitization Walk, and Phase Estimation on It

```python
"""Chapter 4, Example 5: the qubitization walk, and phase estimation on it.
Continues from Example 4 (same session)."""
PI0 = np.kron(np.outer(E4[0], E4[0]), I4)
REFL = 2 * PI0 - np.eye(16)
WALK = REFL @ U_A

print("The qubitization walk W = (2|00><00| (x) I - I) . U_A")
print("=" * 68)
print(f"  W unitary? {np.allclose(WALK.conj().T @ WALK, np.eye(16))}")
ev = np.linalg.eigvals(WALK)
phases = np.angle(ev)
order = np.argsort(phases)
print(f"\n{'eigenphase theta':>18} {'cos(theta)':>12} {'alpha cos(theta)':>18}")
for k in order:
    print(f"{phases[k]:18.9f} {np.cos(phases[k]):12.6f} "
          f"{ALPHA*np.cos(phases[k]):18.6f}")
inner = np.sort(np.unique(np.round(np.cos(phases[np.abs(np.cos(phases)) < 0.999]), 9)))
print(f"\n  distinct cos(theta) away from +-1 : {inner}")
print(f"  eigenvalues of H/alpha             : "
      f"{np.round(np.linalg.eigvalsh(Hmat/ALPHA), 9)}")
print(f"  max mismatch                       : "
      f"{np.max(np.abs(inner - np.linalg.eigvalsh(Hmat/ALPHA))):.3e}")
print("  the +-1 eigenvalues belong to the orthogonal 'junk' subspaces,"
      " where W acts")
print("  as a reflection and carries no spectral information.")

# --- phase estimation on the walk operator -----------------------------
def qft_matrix(m, inverse=False):
    """Dense 2^m x 2^m QFT (big-endian, most significant qubit first)."""
    N = 2 ** m
    j = np.arange(N)
    sign = 1.0 if not inverse else -1.0
    F = np.exp(sign * 2j * np.pi * np.outer(j, j) / N) / np.sqrt(N)
    return F


def qpe_on_walk(m, sys_state):
    """Standard QPE with m ancillas on the 4-qubit walk operator W."""
    n = m + 4
    psi = np.kron(ket('0' * m), np.kron(ket('00'), sys_state))
    for j in range(m):
        psi = apply_gate(psi, H, [j], n)
    for j in range(m):
        power = 2 ** (m - 1 - j)
        Wp = np.linalg.matrix_power(WALK, power)
        cW = np.eye(32, dtype=complex)
        cW[16:, 16:] = Wp
        psi = apply_gate(psi, cW, [j, m, m + 1, m + 2, m + 3], n)
    psi = apply_gate(psi, qft_matrix(m, inverse=True), list(range(m)), n)
    pr = probs(psi).reshape(2 ** m, 16).sum(axis=1)
    return pr


gs = v_H[:, 0].astype(complex)
print("\nPhase estimation on W, system register in the ground state")
print("-" * 68)
print(f"{'m (ancillas)':>13} {'peak readout':>14} {'prob':>8}"
      f" {'theta':>10} {'alpha cos theta':>16} {'error':>10}")
for m in (6, 8, 10):
    pr = qpe_on_walk(m, gs)
    k = int(np.argmax(pr))
    th = 2 * np.pi * k / 2 ** m
    est = ALPHA * np.cos(th)
    print(f"{m:13d} {format(k, f'0{m}b'):>14} {pr[k]:8.4f} {th:10.6f}"
          f" {est:16.6f} {abs(est - w_H[0]):10.2e}")
print(f"  exact ground-state energy = {w_H[0]:.6f}")
pr = qpe_on_walk(8, gs)
top = np.argsort(pr)[::-1][:4]
print("\n  the four largest peaks at m = 8, showing the two-peak structure:")
for k in sorted(top):
    th = 2 * np.pi * k / 2 ** 8
    print(f"    {format(k, '08b')}  p = {pr[k]:.6f}"
          f"  theta = {th:.6f}  alpha cos theta = {ALPHA*np.cos(th):+.6f}")
print("  |00>|E_k> is an equal mixture of the two eigenvectors with phases")
print("  +theta_k and -theta_k, and cos is even, so both peaks give the same")
print("  energy. The residual at small m is set by which side of the true")
print("  phase the nearest grid point falls on, not by the walk operator.")

print("\nQuery complexity: what the two methods buy per query")
print("-" * 68)
print(f"{'target error':>13} {'Trotter-1 rot.':>15} {'Trotter-2 rot.':>15}"
      f" {'qubitization queries':>21}")
for eps in (1e-3, 1e-6, 1e-9):
    r1, r2 = C1 / eps, np.sqrt(C2 / eps)
    q = np.pi * ALPHA / eps / 2
    print(f"{eps:13.0e} {r1*4:15.3e} {r2*8:15.3e} {q:21.3e}")
print("  Trotter costs poly(1/eps) with an exponent set by the order;")
print("  qubitization costs O(alpha t + log(1/eps)) for time evolution and")
print("  O(alpha/eps) for the eigenvalue itself -- optimal in the query"
      " model.")
```

```text
The qubitization walk W = (2|00><00| (x) I - I) . U_A
====================================================================
  W unitary? True

  eigenphase theta   cos(theta)   alpha cos(theta)
      -2.173118745    -0.566558          -1.019804
      -2.159827297    -0.555556          -1.000000
      -0.981765357     0.555556           1.000000
      -0.968473908     0.566558           1.019804
      -0.000000000     1.000000           1.800000
      -0.000000000     1.000000           1.800000
       0.000000000     1.000000           1.800000
       0.000000000     1.000000           1.800000
       0.968473908     0.566558           1.019804
       0.981765357     0.555556           1.000000
       2.159827297    -0.555556          -1.000000
       2.173118745    -0.566558          -1.019804
       3.141592654    -1.000000          -1.800000
       3.141592654    -1.000000          -1.800000
       3.141592654    -1.000000          -1.800000
       3.141592654    -1.000000          -1.800000

  distinct cos(theta) away from +-1 : [-0.56655772 -0.55555556  0.55555556  0.56655772]
  eigenvalues of H/alpha             : [-0.56655772 -0.55555556  0.55555556  0.56655772]
  max mismatch                       : 4.444e-10
  the +-1 eigenvalues belong to the orthogonal 'junk' subspaces, where W acts
  as a reflection and carries no spectral information.

Phase estimation on W, system register in the ground state
--------------------------------------------------------------------
 m (ancillas)   peak readout     prob      theta  alpha cos theta      error
            6         101010   0.4707   4.123340        -1.000026   1.98e-02
            8       01011001   0.2364   2.184389        -1.036455   1.67e-02
           10     1010011110   0.4576   4.111069        -1.018317   1.49e-03
  exact ground-state energy = -1.019804

  the four largest peaks at m = 8, showing the two-peak structure:
    01011000  p = 0.170385  theta = 2.159845  alpha cos theta = -1.000026
    01011001  p = 0.236359  theta = 2.184389  alpha cos theta = -1.036455
    10100111  p = 0.236359  theta = 4.098797  alpha cos theta = -1.036455
    10101000  p = 0.170385  theta = 4.123340  alpha cos theta = -1.000026
  |00>|E_k> is an equal mixture of the two eigenvectors with phases
  +theta_k and -theta_k, and cos is even, so both peaks give the same
  energy. The residual at small m is set by which side of the true
  phase the nearest grid point falls on, not by the walk operator.

Query complexity: what the two methods buy per query
--------------------------------------------------------------------
 target error  Trotter-1 rot.  Trotter-2 rot.  qubitization queries
        1e-03       1.616e+03       7.971e+01             2.827e+03
        1e-06       1.616e+06       2.521e+03             2.827e+06
        1e-09       1.616e+09       7.971e+04             2.827e+09
  Trotter costs poly(1/eps) with an exponent set by the order;
  qubitization costs O(alpha t + log(1/eps)) for time evolution and
  O(alpha/eps) for the eigenvalue itself -- optimal in the query model.
```

**What to notice.** The eigenphase table is the claim, verified. Eight of the sixteen eigenvalues of $W$ come in four $\pm\theta$ pairs whose cosines reproduce the four eigenvalues of $H/\alpha$ to $4 \times 10^{-10}$ — the residual is the non-Hermitian eigensolver's, not the construction's. The remaining eight sit at $\theta = 0$ and $\theta = \pi$, four at each; those are the junk subspaces, where $W$ acts as a plain reflection and carries no information about $H$.

The phase-estimation block is the same circuit as Chapter 2, with $W$ in place of a controlled time evolution, and it shows the characteristic two-peak signature: $\lvert 0^m \rangle \lvert E_k \rangle$ is an equal superposition of the two eigenvectors of $W$ with phases $\pm\theta_k$, so the readout has peaks at $k$ and $2^m - k$. Because $\cos$ is even, both give the same energy — there is no sign ambiguity to resolve, unlike in textbook QPE on $e^{-iHt}$ where the phase wraps.

The query-complexity table at the end is the comparison the whole section exists to make. At $\varepsilon = 10^{-9}$ the second-order product formula needs $8.0 \times 10^{4}$ rotations and qubitization needs $2.8 \times 10^{9}$ queries — qubitization *loses*, badly, on this four-dimensional toy. That is not a mistake in either method. The $O(\alpha/\varepsilon)$ figure is the cost of the eigenvalue to fixed precision with no state-preparation shortcut, while the Trotter figures are for a single unit of evolution time; and $\alpha = 1.8$ here while a real Hamiltonian has $\alpha$ in the hundreds. The asymptotic advantage of qubitization is in $t$ and in $\log(1/\varepsilon)$ for *evolution*, and in the $L$-independence of the walk cost. On four dimensions there is nothing for it to be asymptotic in.

### Quantum signal processing, in one paragraph

Everything above generalises. Given a block encoding of $H/\alpha$, interleaving $U_A$ with single-qubit rotations on one extra ancilla implements a block encoding of $p(H/\alpha)$ for essentially any degree-$d$ polynomial $p$ bounded by 1 on $[-1,1]$, using $d$ queries. Choosing $p$ to approximate $\cos(\alpha t x)$ and $\sin(\alpha t x)$ gives Hamiltonian simulation; choosing $p \approx 1/x$ on a spectral gap gives matrix inversion; choosing a step function gives ground-state projection. Qubitization is the special case where the polynomial is a Chebyshev polynomial, since $\cos(d \arccos x) = T_d(x)$ — which is precisely why the walk's eigenphases came out as $\arccos$. The unification is worth knowing about even if the details are beyond this chapter: it means that "we have a block encoding with normalisation $\alpha$" is the only interface an FTQC algorithm designer needs.

| | Trotter / Suzuki | qDRIFT | Qubitization / QSP |
| --- | --- | --- | --- |
| What is applied | $e^{-iH_j \delta}$, deterministically | $e^{-i\lambda\tau P_l}$, randomly sampled | $U_A$, a block encoding of $H/\alpha$ |
| Ancillas | none | none | $\lceil \log_2 L \rceil$ plus routing |
| Error in $\varepsilon$ | $\varepsilon^{-1/2k}$ | $\varepsilon^{-1}$ | $\log(1/\varepsilon)$ |
| Dependence on $L$ | linear per step | none | one Toffoli per term, once |
| Norm that appears | commutator sums | $\lambda = \sum \lvert c_l \rvert$ | $\alpha = \sum \lvert c_l \rvert$ |
| Output | a unitary | a channel (mixed state) | a unitary |
| Practical niche | dynamics, few terms | very many small terms | eigenvalues, FTQC estimates |

* * *

## 4.3 qDRIFT: Randomised Compilation

### The channel

qDRIFT abandons the idea that the circuit should approximate $e^{-iHt}$ as a unitary. Instead it defines a random circuit whose *average* is close to the right evolution. Sample a term index $l$ with probability $p_l = \lvert c_l \rvert / \lambda$, where $\lambda = \sum_l \lvert c_l \rvert$; apply the single rotation

$$ V_l = \exp\left(-i\,\mathrm{sgn}(c_l)\,\frac{\lambda t}{N}\,P_l\right) $$

and repeat $N$ times. Every rotation has the *same* angle $\lambda t / N$, regardless of how large or small $c_l$ is; the coefficient enters only through how often the term is chosen. Averaging over the randomness gives a quantum channel, and one step of it is

$$ \mathcal{E}_1(\rho) = \sum_l \frac{\lvert c_l \rvert}{\lambda}\, V_l \rho V_l^{\dagger}, \qquad \mathcal{E} = \mathcal{E}_1^{N} $$

The error bound is

$$ \left\lVert \mathcal{E} - e^{-iHt}(\cdot)e^{iHt} \right\rVert_{\diamond} \le O\left(\frac{\lambda^{2} t^{2}}{N}\right) $$

and the striking feature is what is *absent*: no $L$, and no commutator. The number of gates needed for a given accuracy does not depend on how many terms the Hamiltonian has. A Hamiltonian with a million tiny terms and one with four terms of the same total 1-norm cost qDRIFT the same amount.

### Where that helps and where it does not

The trade is a $1/N$ error against Trotter's $1/r^{2}$ or better, paid for by dropping the factor of $L$ per step. Second-order Trotter with $r$ steps costs $2Lr$ rotations and has error $\sim C_2/r^2$; qDRIFT with $N$ rotations has error $\sim 2\lambda^2 t^2/N$. Setting the gate counts equal, qDRIFT wins when

$$ \frac{2\lambda^{2}t^{2}}{G} < \frac{4L^{2}C_2'\,t^{3}}{G^{2}} \quad \Longleftrightarrow \quad G < \frac{2L^{2}C_2' t}{\lambda^{2}} $$

so qDRIFT is the better choice at *coarse* accuracy and *large* term count, and second-order Trotter overtakes it once the budget is large enough. Both statements are testable, and unusually for a randomised algorithm we can test them without any sampling noise at all: the qDRIFT channel is a superoperator, one step is a $d^2 \times d^2$ matrix, and $N$ steps is that matrix to the $N$-th power. Everything below is exact.

The distance we use is $\tfrac{1}{2}\lVert J(\mathcal{E}_1) - J(\mathcal{E}_2) \rVert_{1}$, the trace distance between normalised Choi matrices. It is not the diamond norm, but it is basis-independent, exactly computable, and it treats the unitary Trotter circuit and the mixed-state qDRIFT channel on the same footing — which is the point, because comparing a unitary error to a channel error by any other route invites an apples-to-oranges mistake.

### Code Example 6: Trotter Against qDRIFT at Matched Gate Counts

```python
"""Chapter 4, Example 6: Trotter against qDRIFT at matched gate counts.
Continues from Example 5 (same session)."""
def superop(kraus):
    """Column-stacking superoperator sum_k conj(K) (x) K."""
    return sum(np.kron(K.conj(), K) for K in kraus)


def choi(S, d):
    """Normalised Choi matrix of a superoperator on d x d matrices."""
    J = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d), dtype=complex)
            E[i, j] = 1.0
            out = (S @ E.reshape(-1, order='F')).reshape(d, d, order='F')
            J[i*d:(i+1)*d, j*d:(j+1)*d] = out
    return J / d


def choi_distance(S1, S2, d):
    """(1/2)||J1 - J2||_1 -- a channel distance, exact and basis independent."""
    D = choi(S1, d) - choi(S2, d)
    return 0.5 * float(np.abs(np.linalg.eigvalsh((D + D.conj().T) / 2)).sum())


def trotter_unitary(terms, t, r, order=1):
    step = np.eye(2 ** len(terms[0][0]), dtype=complex)
    dt = t / r
    if order == 1:
        for s, c in terms:
            step = expm_hermitian(to_matrix({s: c}), -1j * dt) @ step
    else:
        for s, c in terms:
            step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
        for s, c in reversed(terms):
            step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
    return np.linalg.matrix_power(step, r)


def qdrift_superop(terms, t, N):
    """Exact qDRIFT channel: N i.i.d. draws, evaluated as a superoperator."""
    lam = sum(abs(c) for _, c in terms)
    d = 2 ** len(terms[0][0])
    kraus = []
    for s, c in terms:
        p = abs(c) / lam
        V = expm_hermitian(to_matrix({s: np.sign(c)}), -1j * lam * t / N)
        kraus.append(np.sqrt(p) * V)
    S1 = superop(kraus)
    return np.linalg.matrix_power(S1, N)


print("Trotter and qDRIFT compared as channels (exact, no sampling)")
print("=" * 68)
print("The qDRIFT channel is E_1^N with E_1(rho) = sum_l (|c_l|/lambda)"
      " V_l rho V_l^dag,")
print("V_l = exp(-i sign(c_l) lambda t P_l / N).  Composing superoperators"
      " evaluates it")
print("exactly, so no Monte-Carlo error enters any number below.")

lam4 = sum(abs(c) for _, c in terms)
S_exact4 = superop([U_exact])
print(f"\nHamiltonian A: L = {len(terms)} terms, lambda = {lam4:.3f},"
      f" t = {tau}")
print(f"{'N (samples)':>12} {'channel error':>15} {'error x N':>11}")
for N in (4, 8, 16, 32, 64, 128, 256, 512):
    e = choi_distance(qdrift_superop(terms, tau, N), S_exact4, 4)
    print(f"{N:12d} {e:15.6e} {e*N:11.5f}")
print("  error x N converges: qDRIFT is O(lambda^2 t^2 / N), independent"
      " of L.")

print("\nMatched gate count, Hamiltonian A (L = 4)")
print(f"{'rotations G':>12} {'Trotter-1 (r=G/L)':>19} {'Trotter-2':>13}"
      f" {'qDRIFT (N=G)':>14} {'winner':>12}")
for G in (16, 32, 64, 128, 256, 512):
    r1 = G // len(terms)
    e1 = choi_distance(superop([trotter_unitary(terms, tau, r1, 1)]),
                       S_exact4, 4)
    r2 = max(1, G // (2 * len(terms)))
    e2 = choi_distance(superop([trotter_unitary(terms, tau, r2, 2)]),
                       S_exact4, 4)
    eq = choi_distance(qdrift_superop(terms, tau, G), S_exact4, 4)
    best = min((e1, 'Trotter-1'), (e2, 'Trotter-2'), (eq, 'qDRIFT'))[1]
    print(f"{G:12d} {e1:19.6e} {e2:13.6e} {eq:14.6e} {best:>12}")

# Part C: the same lambda, the same budget, more and more terms
import itertools

rngB = np.random.default_rng(20260813)
ALL_P4 = [''.join(p) for p in itertools.product('IXYZ', repeat=4)][1:]
PERM = rngB.permutation(len(ALL_P4))
RAW = 1e-4 * (0.05 / 1e-4) ** rngB.random(len(ALL_P4))   # log-uniform, as
LAM0, G0 = 2.0, 1024                                     # molecular integrals


def make_H4(L):
    """A 4-qubit Hamiltonian with L Pauli terms, rescaled to a fixed 1-norm."""
    idx = PERM[:L]
    c = RAW[idx]
    c = c / c.sum() * LAM0
    return {ALL_P4[i]: float(x) for i, x in zip(idx, c)}


print(f"\nPart C: fixed 1-norm lambda = {LAM0}, fixed budget"
      f" G = {G0} rotations, t = {tau}")
print("  only the number of terms L changes -- log-uniform coefficients,"
      " the heavy-tailed")
print("  distribution that molecular integrals actually have")
print(f"{'L':>6} {'||H||':>9} {'Trotter-1':>13} {'Trotter-2':>13}"
      f" {'qDRIFT':>13} {'T2/qDRIFT':>11}")
ratios = []
for L in (4, 15, 63, 255):
    HB = make_H4(L)
    tB = sorted(HB.items())
    HmB = to_matrix(HB)
    SeB = superop([expm_hermitian(HmB, -1j * tau)])
    r1 = max(1, G0 // L)
    r2 = max(1, G0 // (2 * L))
    e1 = choi_distance(superop([trotter_unitary(tB, tau, r1, 1)]), SeB, 16)
    e2 = choi_distance(superop([trotter_unitary(tB, tau, r2, 2)]), SeB, 16)
    eq = choi_distance(qdrift_superop(tB, tau, G0), SeB, 16)
    ratios.append((L, e2 / eq))
    print(f"{L:6d} {np.linalg.norm(HmB, ord=2):9.4f} {e1:13.4e} {e2:13.4e}"
          f" {eq:13.4e} {e2/eq:11.4f}")

Ls = np.array([r[0] for r in ratios], dtype=float)
Rs = np.array([r[1] for r in ratios])
slope, icpt = np.polyfit(np.log(Ls), np.log(Rs), 1)
L_cross = np.exp(-icpt / slope)
print(f"\n  qDRIFT's error is flat in L (it depends only on lambda);"
      " Trotter's grows.")
print(f"  fitted  error(Trotter-2)/error(qDRIFT) ~ L^{slope:.3f}"
      f"  ->  crossover at L ~ {L_cross:.3g}")
print("  Second-order Trotter wins every row above, and would keep winning"
      " until the")
print(f"  Hamiltonian has of order {L_cross:.0e} terms. Molecular and"
      " materials Hamiltonians")
print("  in a few hundred spin orbitals have 1e5 to 1e8 terms, which is"
      " the regime where")
print("  randomised compilation was proposed -- and it is also the regime"
      " where qubitization")
print("  replaces both, because only qubitization turns 1/eps into"
      " log(1/eps).")
```

```text
Trotter and qDRIFT compared as channels (exact, no sampling)
====================================================================
The qDRIFT channel is E_1^N with E_1(rho) = sum_l (|c_l|/lambda) V_l rho V_l^dag,
V_l = exp(-i sign(c_l) lambda t P_l / N).  Composing superoperators evaluates it
exactly, so no Monte-Carlo error enters any number below.

Hamiltonian A: L = 4 terms, lambda = 1.800, t = 1.0
 N (samples)   channel error   error x N
           4    4.103292e-01     1.64132
           8    2.359050e-01     1.88724
          16    1.275293e-01     2.04047
          32    6.645621e-02     2.12660
          64    3.394306e-02     2.17236
         128    1.715587e-02     2.19595
         256    8.624748e-03     2.20794
         512    4.324169e-03     2.21397
  error x N converges: qDRIFT is O(lambda^2 t^2 / N), independent of L.

Matched gate count, Hamiltonian A (L = 4)
 rotations G   Trotter-1 (r=G/L)     Trotter-2   qDRIFT (N=G)       winner
          16        7.751628e-02  2.053134e-02   1.275293e-01    Trotter-2
          32        3.866575e-02  5.062782e-03   6.645621e-02    Trotter-2
          64        1.932131e-02  1.261309e-03   3.394306e-02    Trotter-2
         128        9.659206e-03  3.150531e-04   1.715587e-02    Trotter-2
         256        4.829422e-03  7.874614e-05   8.624748e-03    Trotter-2
         512        2.414688e-03  1.968546e-05   4.324169e-03    Trotter-2

Part C: fixed 1-norm lambda = 2.0, fixed budget G = 1024 rotations, t = 1.0
  only the number of terms L changes -- log-uniform coefficients, the heavy-tailed
  distribution that molecular integrals actually have
     L     ||H||     Trotter-1     Trotter-2        qDRIFT   T2/qDRIFT
     4    1.9341    1.7688e-04    1.7414e-06    4.2964e-04      0.0041
    15    1.1935    4.7436e-03    6.8973e-05    3.1512e-03      0.0219
    63    0.7943    7.7544e-03    2.6280e-04    3.6527e-03      0.0719
   255    0.5280    8.1246e-03    6.1149e-04    3.8532e-03      0.1587

  qDRIFT's error is flat in L (it depends only on lambda); Trotter's grows.
  fitted  error(Trotter-2)/error(qDRIFT) ~ L^0.875  ->  crossover at L ~ 1.62e+03
  Second-order Trotter wins every row above, and would keep winning until the
  Hamiltonian has of order 2e+03 terms. Molecular and materials Hamiltonians
  in a few hundred spin orbitals have 1e5 to 1e8 terms, which is the regime where
  randomised compilation was proposed -- and it is also the regime where qubitization
  replaces both, because only qubitization turns 1/eps into log(1/eps).
```

**What to notice.** Part A confirms the scaling: error $\times N$ converges to about 2.21 for the four-term Hamiltonian, so qDRIFT's error really is $O(1/N)$ with a constant of order $\lambda^2 t^2 = 3.24$. The bound quoted in Section 4.3 is $2\lambda^2t^2/N = 6.48/N$, so on this Hamiltonian the bound is loose by about a factor of three — which is what a diamond-norm bound evaluated against a Choi-trace-distance measurement should be expected to do.

Part B is the matched-budget verdict for a small Hamiltonian and it is unambiguous: second-order Trotter wins at every gate count, by two orders of magnitude at $G = 512$. With $L = 4$ there is nothing for qDRIFT's $L$-independence to save.

Part C is the interesting one, and it was designed to isolate the mechanism. The 1-norm is pinned at $\lambda = 2.0$ and the gate budget at 1024 rotations; only the number of terms changes, with log-uniform coefficients chosen because that is the heavy-tailed distribution real molecular integrals have. qDRIFT's error is flat — $4.3 \times 10^{-4}$ at $L = 4$ and $3.9 \times 10^{-4}$ at $L = 255$ — while second-order Trotter's rises by a factor of 350, because a fixed budget buys fewer and fewer steps. Fitting the ratio gives $\propto L^{0.875}$ and a crossover near $L \approx 1.6 \times 10^{3}$. The fitted exponent is not the $L^{2}$ that the matched-budget algebra above predicts, and the reason is that the algebra holds $C_2'$ fixed while these Hamiltonians do not: at a pinned 1-norm, adding log-uniform terms makes the commutator sum grow far more slowly than the term count. The crossover is therefore an extrapolation of a fit over $4 \le L \le 255$, not a measured point, and the exponent between 0.875 and 2 is where the uncertainty in it lives.

That number is the honest conclusion of the section. **Second-order Trotter wins on everything we can simulate, and would keep winning until the Hamiltonian had a few thousand terms.** Molecular and materials Hamiltonians in a few hundred spin orbitals have $10^5$ to $10^8$ terms, which is exactly the regime randomised compilation was proposed for — and it is also the regime where qubitization displaces both of them, because only qubitization converts $1/\varepsilon$ into $\log(1/\varepsilon)$. qDRIFT's real place in the toolkit is as the cheapest way to get *moderate* accuracy on an enormous Hamiltonian, and as an unusually clean example of randomisation buying a better scaling.

* * *

## 4.4 Resource-Estimation Literacy

### The currency is non-Clifford gates

A fault-tolerant quantum computer does not charge for gates uniformly. Clifford gates — $H$, $S$, CNOT — are transversal or nearly so in the surface code, and by the Gottesman-Knill theorem a circuit made only of them is classically simulable, so they cannot be doing the interesting work. The interesting work is done by non-Clifford gates, and in the surface code those are supplied by **magic-state distillation**: a factory consumes many noisy physical states and produces one high-fidelity logical $T$ or Toffoli state, at a cost in area and time that dominates the whole computation.

Hence the two numbers a fault-tolerant algorithm paper reports:

  * **Toffoli count** (or equivalently T count, with 1 Toffoli $\approx$ 4 T gates in the standard construction). This is the runtime.
  * **Logical qubit count.** This is the width, and it includes system qubits, ancillas for arithmetic and QROM, and routing space — usually several times the system register.

Physical qubit counts and wall-clock times are then derived from those two by choosing a code distance and a cycle time, which is why they move whenever anyone re-optimises a factory layout. The Toffoli count is the stable, algorithm-level statement.

### The formula for qubitized phase estimation

Everything in Section 4.2 collapses into one line. Phase estimation on the walk operator to energy resolution $\varepsilon$ requires

$$ \text{walk steps} \approx \frac{\pi}{2}\,\frac{\alpha}{\varepsilon}, \qquad \text{Toffolis} \approx \frac{\pi}{2}\,\frac{\alpha}{\varepsilon}\, \times C_{\text{walk}} $$

where $C_{\text{walk}}$ is the Toffoli cost of one PREPARE-SELECT-PREPARE$^\dagger$ pair plus the reflection. Three inputs, and only three: the 1-norm $\alpha$ in Hartree, the target resolution $\varepsilon$, and the per-step cost. Chemical accuracy conventionally means $\varepsilon = 1.6 \times 10^{-3}$ Hartree (1 kcal/mol), and that convention is doing a lot of work — a factor of ten in $\varepsilon$ is a factor of ten in runtime.

### Code Example 7: Resource Estimation in Toffolis and Logical Qubits

```python
"""Chapter 4, Example 7: resource estimation in Toffolis and logical qubits.
Continues from Example 6 (same session)."""
print("Resource estimation, in the units the field actually uses")
print("=" * 68)
CHEM_ACC = 1.6e-3        # Hartree, the conventional chemical-accuracy target


def qubitized_qpe_toffolis(lam, eps, toffolis_per_walk):
    """Toffoli count of qubitized QPE: (pi/2)(lam/eps) walk steps."""
    steps = np.pi * lam / (2 * eps)
    return steps, steps * toffolis_per_walk


print("  Toffoli count = (pi/2) (lambda/eps) x (Toffolis per walk step)")
print(f"  eps = {CHEM_ACC:.1e} Hartree (chemical accuracy) throughout")
print("  lambda is the 1-norm of the Hamiltonian in the chosen"
      " factorization, in Hartree\n")
print(f"{'system':>34} {'orbitals':>9} {'qubits':>7} {'lambda(Ha)':>11}"
      f" {'walk steps':>12}")
rows = [("H2, minimal basis", 2, 0.7),
        ("N2, moderate active space", 20, 40.0),
        ("FeMoco active space, low estimate", 76, 300.0),
        ("FeMoco active space, high estimate", 76, 4000.0)]
for name, norb, lam in rows:
    steps, _ = qubitized_qpe_toffolis(lam, CHEM_ACC, 1)
    print(f"{name:>34} {norb:9d} {2*norb:7d} {lam:11.1f} {steps:12.3e}")

print("\nSensitivity of the FeMoco Toffoli count to the two uncertain"
      " inputs")
print("-" * 68)
print(f"{'lambda (Ha)':>12}", end='')
for cw in (1e4, 3e4, 1e5):
    print(f" {'C_walk=' + f'{cw:.0e}':>16}", end='')
print()
for lam in (300.0, 1000.0, 4000.0):
    line = f"{lam:12.0f}"
    for cw in (1e4, 3e4, 1e5):
        steps, tof = qubitized_qpe_toffolis(lam, CHEM_ACC, cw)
        line += f" {tof:14.2e}" + (" *" if 1e10 <= tof <= 1e11 else "  ")
    print(line.rstrip())
print("  * marks the cells inside the 1e10 to 1e11 Toffoli band that"
      " published")
print("  fault-tolerant estimates for this system occupy. Neither input"
      " is known to")
print("  better than a factor of a few, so the deliverable is the"
      " exponent, not the")
print("  mantissa -- and the exponent has moved down by several units over"
      " a decade of")
print("  work on factorizations and on magic-state distillation.")

print("\nFrom Toffolis to wall-clock time")
print("-" * 68)
T_TOF = 1e-5             # seconds per logical Toffoli, a standard placeholder
print(f"{'Toffolis':>10} {'seconds':>12} {'days':>9}"
      f"   at {T_TOF*1e6:.0f} us per logical Toffoli")
for tof in (1e9, 1e10, 1e11):
    print(f"{tof:10.0e} {tof*T_TOF:12.3e} {tof*T_TOF/86400:9.2f}")

print("\nRequired logical error rate per Toffoli, and the surface-code"
      " distance")
print("-" * 68)
print(f"{'Toffolis':>10} {'p_L needed':>12} {'d (p=1e-3)':>11}"
      f" {'phys/logical':>13} {'phys. qubits':>13}")
for tof in (1e9, 1e10, 1e11):
    pL = 0.1 / tof                     # total failure probability 0.1
    d = 1
    # The comparison carries a relative tolerance because p_L is a ratio
    # raised to a large power: a distance that meets the target *exactly*
    # lands a few ulps above it in binary floating point (0.1 * 0.1**9
    # evaluates to 1.0000000000000006e-10), and a bare > would reject that
    # distance and return the next one up.
    while 0.1 * (1e-3 / 1e-2) ** ((d + 1) / 2) > pL * (1 + 1e-9):
        d += 2
    per = 2 * d * d
    n_log = 2 * 76 + 1000              # system + routing and ancillas
    print(f"{tof:10.0e} {pL:12.1e} {d:11d} {per:13d} {per*n_log:13.2e}")
print("  Plus magic-state factories, which in published layouts occupy"
      " a comparable")
print("  or larger footprint. Millions of physical qubits and days of"
      " runtime is the")
print("  standing conclusion, and no algorithmic advance so far has"
      " removed it.")

print("\nWhat the same estimate looks like with Trotter instead")
print("-" * 68)
print(f"{'method':>26} {'scaling in t and eps':>34} {'exponent of 1/eps':>18}")
for name, sc, ex in [("first-order Trotter", "O(L (alpha t)^2 / eps)", "1"),
                     ("2k-th order Trotter",
                      "O(L (alpha t)^{1+1/2k} / eps^{1/2k})", "1/2k"),
                     ("qDRIFT", "O(lambda^2 t^2 / eps)", "1"),
                     ("qubitization / QSP",
                      "O(alpha t + log(1/eps))", "log")]:
    print(f"{name:>26} {sc:>34} {ex:>18}")
print("  The log(1/eps) in the last row is the whole reason FTQC chemistry")
print("  estimates are quoted in qubitization language and not in Trotter"
      " steps.")
```

```text
Resource estimation, in the units the field actually uses
====================================================================
  Toffoli count = (pi/2) (lambda/eps) x (Toffolis per walk step)
  eps = 1.6e-03 Hartree (chemical accuracy) throughout
  lambda is the 1-norm of the Hamiltonian in the chosen factorization, in Hartree

                            system  orbitals  qubits  lambda(Ha)   walk steps
                 H2, minimal basis         2       4         0.7    6.872e+02
         N2, moderate active space        20      40        40.0    3.927e+04
 FeMoco active space, low estimate        76     152       300.0    2.945e+05
FeMoco active space, high estimate        76     152      4000.0    3.927e+06

Sensitivity of the FeMoco Toffoli count to the two uncertain inputs
--------------------------------------------------------------------
 lambda (Ha)     C_walk=1e+04     C_walk=3e+04     C_walk=1e+05
         300       2.95e+09         8.84e+09         2.95e+10 *
        1000       9.82e+09         2.95e+10 *       9.82e+10 *
        4000       3.93e+10 *       1.18e+11         3.93e+11
  * marks the cells inside the 1e10 to 1e11 Toffoli band that published
  fault-tolerant estimates for this system occupy. Neither input is known to
  better than a factor of a few, so the deliverable is the exponent, not the
  mantissa -- and the exponent has moved down by several units over a decade of
  work on factorizations and on magic-state distillation.

From Toffolis to wall-clock time
--------------------------------------------------------------------
  Toffolis      seconds      days   at 10 us per logical Toffoli
     1e+09    1.000e+04      0.12
     1e+10    1.000e+05      1.16
     1e+11    1.000e+06     11.57

Required logical error rate per Toffoli, and the surface-code distance
--------------------------------------------------------------------
  Toffolis   p_L needed  d (p=1e-3)  phys/logical  phys. qubits
     1e+09      1.0e-10          17           578      6.66e+05
     1e+10      1.0e-11          19           722      8.32e+05
     1e+11      1.0e-12          21           882      1.02e+06
  Plus magic-state factories, which in published layouts occupy a comparable
  or larger footprint. Millions of physical qubits and days of runtime is the
  standing conclusion, and no algorithmic advance so far has removed it.

What the same estimate looks like with Trotter instead
--------------------------------------------------------------------
                    method               scaling in t and eps  exponent of 1/eps
       first-order Trotter             O(L (alpha t)^2 / eps)                  1
       2k-th order Trotter O(L (alpha t)^{1+1/2k} / eps^{1/2k})               1/2k
                    qDRIFT              O(lambda^2 t^2 / eps)                  1
        qubitization / QSP            O(alpha t + log(1/eps))                log
  The log(1/eps) in the last row is the whole reason FTQC chemistry
  estimates are quoted in qubitization language and not in Trotter steps.
```

**What to notice.** The sensitivity grid is the substance of this example. The Toffoli count for a FeMoco-scale calculation is the product of two numbers neither of which is known to better than a factor of a few, and the product sweeps from $3 \times 10^{9}$ to $4 \times 10^{11}$ across plausible inputs. The cells inside the $10^{10}$ to $10^{11}$ band — where published fault-tolerant estimates for this system sit, and where the sister course's Chapter 4 places them — are a diagonal stripe through the middle. **The deliverable of a resource estimate is the exponent.** Anyone quoting a resource estimate to two significant figures without also quoting $\alpha$, $\varepsilon$ and the factorisation is quoting a number they cannot defend.

The wall-clock and surface-code blocks turn the exponent into hardware. At ten microseconds per logical Toffoli — a standard placeholder, set by the code cycle time times the distance times a factory latency — $10^{10}$ Toffolis is 1.2 days and $10^{11}$ is 11.6 days. Holding the total failure probability at 0.1 across $10^{11}$ Toffolis needs a logical error rate of $10^{-12}$ per gate, which at a physical error rate of $10^{-3}$ needs surface-code distance 21, hence about 880 physical qubits per logical qubit, hence of order $10^{6}$ physical qubits for a register of about 1150 logical qubits — *before* the magic-state factories, which in published layouts occupy a comparable or larger footprint. Millions of physical qubits and days of runtime: this is the same conclusion the sister course reached, and it has not been softened.

### Why electronic structure is the flagship, and what that means

It is worth being explicit about why this particular application dominates the FTQC literature, because the reasoning is not "chemistry is important" but something sharper.

  * **The problem is an eigenvalue problem.** Phase estimation is the one primitive with an exponential separation that does not depend on an oracle assumption or a data-loading assumption. Ground-state energies are exactly what it computes.
  * **The classical competition is combinatorial, and provably so for exact methods.** Full configuration interaction on $M$ orbitals needs $\binom{M}{N_\alpha}\binom{M}{N_\beta}$ determinants. Approximate methods — DMRG, quantum Monte Carlo, coupled cluster — are strong, but each has a known failure mode, and strong static correlation triggers most of them simultaneously.
  * **The answer is small.** A quantum computer would produce one number per calculation, not a wavefunction. Output bandwidth, which sinks many other proposed quantum applications, is not an obstacle here.
  * **The targets are materials targets.** Transition-metal oxides, multi-centre catalytic clusters, Mott insulators, and the strongly correlated active spaces embedded in an otherwise DFT-treatable solid. This is the same list the sister course identified, and it is a materials-research list.

And it is worth being equally explicit about the two things this does *not* establish. First, there is no proof that these ground-state problems are classically hard; the evidence is the failure of known classical methods, which is a much weaker statement than a complexity-theoretic separation. Second, and more concretely, **phase estimation needs an initial state with non-negligible overlap on the target eigenvector.** Success probability is $\lvert \langle \phi_0 \lvert \psi_{\text{init}} \rangle \rvert^{2}$, and for strongly correlated systems the overlap of any easily-preparable reference — a Hartree-Fock determinant, a small CI expansion — can decay exponentially with system size. Nothing in the algorithm prepares the state for you, and the same static correlation that defeats the classical method is what shrinks the overlap. This is the single most load-bearing assumption in the whole FTQC chemistry programme, and Section 5.5 returns to it when the speedup map is drawn.

### A checklist for reading a resource estimate

  1. **What is $\alpha$, and in what units?** If the paper does not report the 1-norm, it has not reported the cost.
  2. **What factorisation of the two-electron tensor?** Sparse, single-factorised, double-factorised and tensor-hypercontraction give different $\alpha$ and different $C_{\text{walk}}$, and comparing two papers that made different choices is meaningless.
  3. **What $\varepsilon$, and to what quantity?** Total energy to chemical accuracy is a much harder target than an energy *difference* along a reaction coordinate, where errors partly cancel.
  4. **Is the initial-state overlap discussed, or assumed?** If the overlap is quietly taken as $O(1)$, the estimate is a lower bound of unknown tightness.
  5. **Toffolis or T gates?** The factor of four is often left implicit.
  6. **Logical or physical qubits, and at what code distance and physical error rate?** Physical counts are meaningless without both.
  7. **Are the magic-state factories counted in the qubit budget?** Often they are counted in area but reported separately.

* * *

## Exercises

#### Exercise 1: Block-Encoding Arithmetic

A Hamiltonian on three qubits is given as $H = 1.2\,ZZI - 0.8\,IXX + 0.5\,XIY + 0.3\,YYZ - 0.2\,IIZ$.

  1. What is $\alpha$ for the LCU block encoding, and how many ancilla qubits does PREPARE need?
  2. The PREPARE register has more basis states than there are terms. What must the extra amplitudes be, and what happens if they are not?
  3. Give the success probability of one LCU round for a state with $\lVert H \lvert \psi \rangle \rVert = 1.5$.
  4. How many amplitude-amplification rounds would make that success near-certain?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\alpha = 1.2 + 0.8 + 0.5 + 0.3 + 0.2 = 3.0\) (signs are absorbed into the unitaries, so the 1-norm uses absolute values). With \(L = 5\) terms, PREPARE needs \(\lceil \log_2 5 \rceil = 3\) ancilla qubits, giving 8 basis states for 5 terms.</p>

<p><strong>2.</strong> The three unused amplitudes must be exactly zero. If they are not, the block encoding implements \(\sum_l c_l U_l / \alpha\) with those extra \(l\) contributing whatever SELECT does on undefined control values — in practice the identity, which silently adds a spurious term to the encoded operator. Padding SELECT with the identity on unused indices and forcing the amplitudes to zero are two statements of the same requirement, and the second is the one to check numerically.</p>

<p><strong>3.</strong> \(P = \lVert H \lvert \psi \rangle \rVert^2 / \alpha^2 = 1.5^2/3.0^2 = 0.25\).</p>

<p><strong>4.</strong> With \(\theta = \arcsin\sqrt{0.25} = \arcsin(0.5) = \pi/6\), the optimal round count is \(k = \lfloor (\pi/2 - \theta)/(2\theta) \rfloor = \lfloor (\pi/3)/(\pi/3) \rfloor = 1\), and one round gives \(\sin^2(3\theta) = \sin^2(\pi/2) = 1\) exactly. A success probability of exactly 1/4 is the lucky case where a single round is perfect.</p>

```python
import numpy as np
c = np.array([1.2, 0.8, 0.5, 0.3, 0.2])
print(c.sum(), int(np.ceil(np.log2(len(c)))))        # 3.0 3
p = 1.5**2 / 3.0**2
th = np.arcsin(np.sqrt(p))
k = int((np.pi/2 - th) / (2*th))
print(round(p, 4), k, round(np.sin((2*k+1)*th)**2, 6))   # 0.25 1 1.0
```

</details>

#### Exercise 2: Reading the Walk Spectrum Backwards

You are handed a block encoding with $\alpha = 12.0$ Hartree, and phase estimation on its walk operator with $m = 12$ ancilla qubits returns a peak at readout $k = 891$.

  1. What energy does this correspond to?
  2. What is the energy resolution implied by $m = 12$ at this point in the spectrum, and why does it depend on where in the spectrum you are?
  3. Which part of the spectrum is resolved best by a fixed $m$, and which worst?
  4. How many ancilla qubits would you need for chemical accuracy, $\varepsilon = 1.6 \times 10^{-3}$ Hartree, at the same point?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\theta = 2\pi k/2^{m} = 2\pi \times 891/4096 = 1.36678\) rad, so \(E = \alpha\cos\theta = 12.0 \times 0.20261 = 2.4313\) Hartree.</p>

<p><strong>2.</strong> One least-significant bit is \(\delta\theta = 2\pi/4096 = 1.534\times10^{-3}\) rad, and \(\lvert dE/d\theta \rvert = \alpha \lvert \sin\theta \rvert = 12.0 \times 0.97926 = 11.75\), so \(\delta E = 1.8026\times10^{-2}\) Hartree. The Jacobian \(\alpha \sin\theta\) is what makes the resolution position-dependent: the map \(E = \alpha\cos\theta\) is not uniform.</p>

<p><strong>3.</strong> Since \(\delta E = \alpha\lvert\sin\theta\rvert\,\delta\theta\), the resolution is best — \(\delta E\) smallest — where \(\lvert\sin\theta\rvert\) is smallest, that is at the band edges \(E \to \pm\alpha\), and worst at the band centre \(E \approx 0\) where \(\lvert\sin\theta\rvert = 1\). Ground states sit near an edge, which is a rare piece of good luck: the extreme eigenvalues are exactly the ones qubitized QPE resolves most sharply, and the dense middle of the spectrum, which nobody wants, is the part it resolves worst.</p>

<p><strong>4.</strong> Need \(\alpha \lvert\sin\theta\rvert \, 2\pi/2^{m} \le 1.6\times10^{-3}\), so \(2^{m} \ge 2\pi \times 11.75/1.6\times10^{-3} = 4.615\times10^{4}\), giving \(m = 16\) and \(2^{m} = 65\,536\). That is the readout-register width; the deepest controlled power is then \(W^{32768}\) and the total walk-step count is \(2^{m} - 1 \approx 6.6\times10^{4}\), the same order as the \(\pi\alpha/2\varepsilon = 1.18\times10^{4}\) rule of thumb.</p>

```python
import numpy as np
alpha, m, k = 12.0, 12, 891
th = 2*np.pi*k/2**m
print(round(th, 5), round(alpha*np.cos(th), 4))            # 1.36678 2.4313
dE = alpha*abs(np.sin(th))*2*np.pi/2**m
print(f"{dE:.4e}")                                          # 1.8026e-02
need = 2*np.pi*alpha*abs(np.sin(th))/1.6e-3
print(int(np.ceil(np.log2(need))))                          # 16
```

</details>

#### Exercise 3: Choosing Between qDRIFT and Trotter

A Hamiltonian has $L = 40\,000$ Pauli terms and 1-norm $\lambda = 400$ Hartree. You want $e^{-iHt}$ for $t = 1$ Hartree$^{-1}$ to channel error $10^{-3}$.

  1. Estimate the qDRIFT rotation count from $\text{error} \approx 2\lambda^2 t^2/N$.
  2. Second-order Trotter needs $r$ steps at $2Lr$ rotations, with error $\approx C_2/r^2$. Taking $C_2 \approx L^{1.5}$ as a crude stand-in for the commutator sum, estimate the rotation count.
  3. Which wins, and by how much?
  4. What would change the answer?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(N = 2\lambda^2 t^2/\varepsilon = 2 \times 1.6\times10^{5}/10^{-3} = 3.2\times10^{8}\) rotations.</p>

<p><strong>2.</strong> \(C_2 \approx 40000^{1.5} = 8.0\times10^{6}\), so \(r = \sqrt{C_2/\varepsilon} = \sqrt{8.0\times10^{9}} = 8.944\times10^{4}\) steps, and the rotation count is \(2Lr = 2 \times 4\times10^{4} \times 8.944\times10^{4} = 7.16\times10^{9}\).</p>

<p><strong>3.</strong> qDRIFT wins by a factor of about 22. The mechanism is the \(2L\) multiplying every Trotter step: at 40 000 terms, one second-order Trotter step already costs 80 000 rotations, which is a quarter of qDRIFT's entire budget.</p>

<p><strong>4.</strong> Three things. A tighter accuracy target: qDRIFT scales as \(1/\varepsilon\) and second-order Trotter as \(1/\sqrt{\varepsilon}\), so at \(\varepsilon = 10^{-6}\) the counts become \(3.2\times10^{11}\) and \(2.3\times10^{11}\) and Trotter takes the lead. A smaller \(\lambda\) at fixed \(L\) — which is what better factorisations achieve — helps qDRIFT quadratically. And term grouping: commuting terms can be exponentiated simultaneously, which reduces the effective \(L\) in the Trotter count without touching \(\lambda\). None of these changes the fact that qubitization converts the \(1/\varepsilon\) into \(\log(1/\varepsilon)\) and therefore wins at any serious precision.</p>

```python
import numpy as np
L, lam, t = 40000, 400.0, 1.0
for eps in (1e-3, 1e-6):
    N = 2*lam**2*t**2/eps
    r = np.sqrt(L**1.5/eps)
    print(f"eps={eps:.0e}  qDRIFT {N:.2e}   Trotter-2 {2*L*r:.2e}")
# eps=1e-03  qDRIFT 3.20e+08   Trotter-2 7.16e+09
# eps=1e-06  qDRIFT 3.20e+11   Trotter-2 2.26e+11
```

</details>

#### Exercise 4: A Resource Estimate You Have to Defend

A colleague proposes a fault-tolerant calculation of a transition-metal oxide cluster: 60 spatial orbitals, $\alpha = 800$ Hartree with the factorisation they intend to use, $C_{\text{walk}} = 2 \times 10^{4}$ Toffolis, target chemical accuracy.

  1. Give the Toffoli count and the wall-clock time at 10 $\mu$s per logical Toffoli.
  2. They then say they can halve $\alpha$ by switching factorisation, at the price of tripling $C_{\text{walk}}$. Should they?
  3. They want the energy *difference* between two spin states rather than a total energy, and argue that $\varepsilon$ can be relaxed to $5 \times 10^{-3}$ Hartree. What does that buy?
  4. Name the one input in this estimate that could make it wrong by more than all the others combined.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Walk steps \(= \pi\alpha/2\varepsilon = 1.571 \times 800/1.6\times10^{-3} = 7.85\times10^{5}\); Toffolis \(= 7.85\times10^{5} \times 2\times10^{4} = 1.57\times10^{10}\); time \(= 1.57\times10^{5}\) s \(= 1.82\) days. A defensible number, in the published band.</p>

<p><strong>2.</strong> No. The product changes by \(0.5 \times 3 = 1.5\), so the cost rises by 50%. The rule of thumb is that \(\alpha\) and \(C_{\text{walk}}\) enter the product symmetrically, so any trade must be judged on the product alone — and factorisations that cut \(\alpha\) usually do cost more per step, which is why the comparison has to be made explicitly rather than assumed in favour of the smaller norm.</p>

<p><strong>3.</strong> A factor of \(5\times10^{-3}/1.6\times10^{-3} = 3.125\), bringing the estimate to \(5.0\times10^{9}\) Toffolis and 0.58 days. This is the cheapest available saving in the whole business and it is physically justified whenever the quantity of interest is a difference and the two calculations share systematic error — which is why resource estimates should always state which quantity they target.</p>

<p><strong>4.</strong> The initial-state overlap. Every number above assumes phase estimation succeeds, i.e. that a preparable reference state has \(O(1)\) overlap with the target eigenvector. If the overlap is \(10^{-2}\), the whole calculation must be repeated \(\sim 10^{4}\) times or an amplitude-amplification wrapper added, and the estimate moves by four orders of magnitude — more than \(\alpha\), \(\varepsilon\) and \(C_{\text{walk}}\) combined can. It is also the input least often quantified.</p>

```python
import numpy as np
def toffolis(alpha, eps, cw): return np.pi*alpha/(2*eps)*cw
base = toffolis(800, 1.6e-3, 2e4)
print(f"{base:.3e}  {base*1e-5/86400:.2f} days")     # 1.571e+10  1.82 days
print(f"{toffolis(400, 1.6e-3, 6e4)/base:.2f}x")     # 1.50x
print(f"{toffolis(800, 5e-3, 2e4):.3e}")             # 5.027e+09
```

</details>

#### Exercise 5: Verifying Someone Else's Block Encoding

You are given code that claims to block-encode a Hamiltonian, returning a $2^{m+n} \times 2^{m+n}$ matrix `UA` and a scalar `alpha`.

  1. Write down the three checks that establish the claim, in the order you would run them.
  2. One check passes and another fails in a way that shows PREPARE and SELECT disagree about the term ordering. Which check catches that, and what does the failure look like?
  3. What would you see if `alpha` were the spectral norm rather than the 1-norm?
  4. Why is checking the top-left block *not* enough on its own?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> (i) Unitarity: \(\max \lvert U_A^\dagger U_A - I \rvert\) at machine precision, because everything else is meaningless if this fails. (ii) The defining property: \(\max \lvert U_A[:2^n,:2^n] - H/\alpha \rvert\) against an independently constructed \(H\). (iii) An operational check on the simulator: apply \(U_A\) to \(\lvert 0^m \rangle \lvert \psi \rangle\) for a random \(\lvert \psi \rangle\), confirm the leading block of the output equals \(H\lvert\psi\rangle/\alpha\) and that its squared norm equals \(\lVert H \lvert\psi\rangle\rVert^2/\alpha^2\). Step (iii) catches indexing and endianness errors that (ii) can miss when the Hamiltonian happens to be symmetric.</p>

<p><strong>2.</strong> Check (ii) catches it. Unitarity still passes — permuting which unitary goes with which amplitude leaves \(U_A\) unitary — but the top-left block becomes \(\sum_l \sqrt{c_l c_{\sigma(l)}}\,U_{\sigma(l)}/\alpha\) for the mismatched permutation \(\sigma\), so the block is a *different* Hermitian matrix with the right norm structure and the wrong entries. The signature is a mismatch of order the coefficient spread rather than of order machine precision, and the block usually remains Hermitian, which is why "it looks like a Hamiltonian" is not evidence.</p>

<p><strong>3.</strong> The top-left block would have spectral norm 1 rather than \(\lVert H \rVert/\alpha < 1\), and unitarity would fail: a unitary cannot have a norm-1 block unless the rest of that row and column vanish, which they do not for a generic \(H\). So check (i) would catch it immediately. Equivalently, the LCU construction simply cannot produce \(\alpha < \sum_l \lvert c_l \rvert\).</p>

<p><strong>4.</strong> Because the top-left block of a *non*-unitary matrix can be anything at all. Without check (i), "the top-left block is \(H/\alpha\)" is a statement about a matrix that no quantum circuit implements. The pair of checks is the claim; either alone is not.</p>

</details>

* * *

## Summary

### Key Takeaways

**1\. Product formulas have a polynomial dependence on $1/\varepsilon$ that no order removes**

  * Measured constants for the chapter's four-term Hamiltonian: first order $C_1 = 0.4039$, second order $C_2 = 0.0993$, both confirmed by error $\times r$ and error $\times r^2$ converging.
  * Reaching $10^{-9}$ needs $1.6\times10^{9}$ rotations at first order and $8.0\times10^{4}$ at second — a factor of 20 000, still polynomial.
  * The cost is linear in the term count $L$, and $L = O(M^4)$ for electronic structure.

**2\. A block encoding puts $H$ in the top-left corner of a unitary, exactly**

  * $(\langle 0^m \rvert \otimes I) U_A (\lvert 0^m \rangle \otimes I) = H/\alpha$, verified to $8 \times 10^{-17}$ on a $16 \times 16$ explicit matrix.
  * The LCU construction is PREPARE, SELECT, PREPARE$^\dagger$; PREPARE for four real amplitudes is three $R_y$ and two CNOTs, and SELECT costs $L-1$ Toffolis with unary iteration.
  * The off-diagonal blocks are *not* small — they carry the rest of the unitarity budget — and the whole method is about addressing the good block rather than shrinking the bad ones.

**3\. $\alpha$ is a 1-norm, and it is what every cost formula contains**

  * Success probability of one round is $\lVert H\lvert\psi\rangle\rVert^2/\alpha^2$, confirmed to eight digits against the simulator, with Haar average $\mathrm{Tr}(H^2)/(d\alpha^2)$.
  * Amplitude amplification costs $O(\alpha/\lVert H\lvert\psi\rangle\rVert)$ rounds.
  * For Heisenberg chains $\alpha$ grows exactly linearly in the term count while $\lVert H \rVert$ grows more slowly; the ratio drifts from 1.0 to 1.6 between 2 and 10 sites, and is far worse for molecules. Reducing $\alpha$ is the main lever on FTQC chemistry cost.

**4\. Qubitization makes the walk's eigenphases $\arccos$ of the energies**

  * $W = (2\Pi - I)U_A$ has eigenvalues $e^{\pm i\theta_k}$ with $\cos\theta_k = E_k/\alpha$: verified to $4\times10^{-10}$, with the junk subspaces sitting at $\theta = 0, \pi$.
  * Phase estimation on $W$ recovers the ground-state energy, with the characteristic two-peak structure at $\pm\theta_k$ that costs nothing because $\cos$ is even.
  * Eigenvalues cost $O(\alpha/\varepsilon)$ queries and evolution costs $O(\alpha t + \log(1/\varepsilon))$ — both optimal in the query model, and the logarithm is the qualitative break with Trotter.

**5\. qDRIFT trades $1/N$ error for total independence of the term count**

  * Evaluated exactly as a superoperator: error $\times N \to 2.21$ for the four-term Hamiltonian, i.e. $O(\lambda^2t^2/N)$.
  * At matched gate count, second-order Trotter wins every comparison we can simulate, by two orders of magnitude at $L = 4$.
  * With $\lambda$ and the budget held fixed and only $L$ varied, qDRIFT's error is flat while Trotter's grows as $L^{0.875}$; the fitted crossover is $L \approx 1.6\times10^{3}$, which is why randomised compilation was proposed for Hamiltonians with $10^5$ terms and above.

**6\. A resource estimate is a Toffoli count plus a list of assumptions**

  * Qubitized QPE needs $\approx (\pi/2)(\alpha/\varepsilon)$ walk steps, each costing $C_{\text{walk}}$ Toffolis; three inputs, and the answer is the product.
  * Across plausible $(\alpha, C_{\text{walk}})$ for a FeMoco-scale active space the count sweeps $3\times10^{9}$ to $4\times10^{11}$, with the published $10^{10}$–$10^{11}$ band a diagonal stripe through the middle.
  * $10^{11}$ Toffolis is 11.6 days at 10 $\mu$s each, needs logical error $10^{-12}$, surface-code distance 21, about $10^{6}$ physical qubits before magic-state factories. Millions of physical qubits and days of runtime.

**Practical implications**

  * Report $\alpha$ whenever you report a block-encoding cost; a cost without a 1-norm is not a cost.
  * Verify a block encoding with two independent checks — unitarity and the defining block — and one operational check on a simulator. Any one alone can pass on a wrong construction.
  * Choose the simulation method by regime, not by fashion: Trotter for few terms and tight accuracy, qDRIFT for very many small terms at moderate accuracy, qubitization whenever the target is an eigenvalue.
  * When you read "a quantum computer could compute FeMoco", ask for $\alpha$, for $\varepsilon$, for the factorisation, and for the initial-state overlap. The first three set the exponent; the fourth can invalidate it.

### Where This Leads

We now have the FTQC end of the toolkit: an exact block encoding, a walk whose spectrum is the Hamiltonian's, and a vocabulary for saying what any of it would cost. All of it presumes a fault-tolerant machine. Chapter 5 turns to the other end — the variational, near-term end — with the Quantum Approximate Optimization Algorithm, and applies the same discipline to it. We will formulate MaxCut as an Ising problem, build QAOA on the same simulator, watch the approximation ratio climb with depth and with a fixed adiabatic schedule, and then put it against greedy search, local search, simulated annealing and Goemans-Williamson rounding at a matched budget, with paired intervals on every difference. The chapter closes the series with a map of where provable speedups actually live and what each of them assumes — including the two chapters just completed.

[← Chapter 3: Shor's Algorithm](<chapter-3.html>) [Chapter 5: QAOA and Optimization →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The 1-norms, per-step Toffoli costs, code distances and wall-clock figures in this chapter are illustrative order-of-magnitude values chosen to demonstrate the arithmetic of a resource estimate; they are not a substitute for a published estimate and must be checked against primary sources before use in a proposal or a paper.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
