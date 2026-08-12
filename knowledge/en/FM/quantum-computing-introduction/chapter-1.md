---
title: "Chapter 1: Qubits and Superposition"
chapter_title: "Chapter 1: Qubits and Superposition"
subtitle: State Vectors, the Bloch Sphere, the Born Rule, and the Exponential Wall
reading_time: 30-35 minutes
difficulty: Beginner
code_examples: 8
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-computing-introduction/chapter-1.html>) | Last sync: 2026-08-12

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to Quantum Computing](<index.html>) > Chapter 1

This chapter builds the object that everything else in the series acts on: the state of a register of qubits. If you have solved a Schrödinger equation you already know most of the mathematics, and the first job of this chapter is to make that correspondence explicit rather than to teach you a new formalism. The second job is quantitative. The reason a materials researcher should care about quantum computing at all is a scaling argument about the many-electron wave function, and that argument is worth making with numbers before any qubit appears. Eight worked examples build up, by the end of the chapter, the first three functions of a state-vector simulator that will be used unchanged in Chapters 2 through 5.

## Learning Objectives

After completing this chapter, you will be able to:

  * Quantify why the exact solution of the electronic structure problem is exponentially hard, and state the dimension of the full configuration interaction space for a given active space
  * Write a qubit state as a normalized complex vector, compute its norm and inner products, and explain why a global phase carries no physical information while a relative phase does
  * Parameterize any pure single-qubit state by two angles, convert between the state vector and its Bloch vector, and explain why the Bloch picture does not generalize to many qubits
  * Apply the Born rule to obtain measurement probabilities, describe the collapse of the state after a projective measurement, and compute expectation values of Pauli observables
  * Construct multi-qubit states as tensor products, use the big-endian index convention of this series correctly, and distinguish a product state from an entangled one by its Schmidt rank
  * Implement `ket`, `probs` and `sample` in NumPy, and estimate how many measurement shots an expectation value of a given precision requires

* * *

## 1.1 Why a Materials Researcher Should Learn Quantum Computing

### The Problem You Already Have

The non-relativistic electronic Hamiltonian of a molecule or a solid, in the Born-Oppenheimer approximation, is completely known:

\\[ \hat{H} = -\sum_i \frac{\hbar^2}{2m_e}\nabla_i^2 - \sum_{i,A} \frac{Z_A e^2}{4\pi\epsilon_0 r_{iA}} + \sum_{i<j} \frac{e^2}{4\pi\epsilon_0 r_{ij}} \\]

Nothing in it is unknown or approximate at the level of accuracy that chemistry and solid-state physics require. The difficulty is entirely in the solution. The ground state \\(\Psi(r_1, \ldots, r_N)\\) is a function of \\(3N\\) coordinates that must be antisymmetric under exchange, and expanding it in a finite basis of \\(M\\) spatial orbitals gives a linear space whose dimension is a product of two binomial coefficients:

\\[ \dim H_\text{FCI} = \binom{M}{N_\alpha}\binom{M}{N_\beta} \\]

This is the **full configuration interaction** (FCI) dimension: the number of Slater determinants you can build by distributing \\(N_\alpha\\) up-spin and \\(N_\beta\\) down-spin electrons over \\(M\\) orbitals. It grows faster than any polynomial in \\(M\\). For water in a modest 6-31G basis it is already about \\(1.7 \times 10^6\\); for the iron-molybdenum cofactor of nitrogenase in the active space usually quoted for it, it is about \\(10^{35}\\).

### What the Standard Methods Do About It

The entire apparatus of computational chemistry and electronic structure theory is a set of strategies for not solving the FCI problem:

| Method | Formal scaling | What it assumes |
| --- | --- | --- |
| Density functional theory (KS-DFT) | \\(O(M^3)\\) | An unknown exchange-correlation functional is approximated; single-reference character |
| Hartree-Fock | \\(O(M^4)\\) | One determinant; correlation neglected entirely |
| MP2 | \\(O(M^5)\\) | Perturbative correlation about a good single reference |
| CCSD | \\(O(M^6)\\) | Exponential ansatz truncated at double excitations |
| CCSD(T) | \\(O(M^7)\\) | The "gold standard" — for weakly correlated systems |
| DMRG | polynomial in bond dimension | Low entanglement, effectively one-dimensional connectivity |
| Full CI | exponential | Nothing. It is exact in the given basis |

Every polynomial-scaling entry in that table is buying its speed with an assumption about the structure of the wave function, and each assumption has a class of materials that breaks it. The systems where they break are, inconveniently, the interesting ones:

  * **Transition-metal oxides and complexes** : partially filled \\(d\\) shells give several nearly degenerate determinants, so a single reference is qualitatively wrong.
  * **Iron-sulfur clusters and metalloenzyme active sites** : dozens of strongly coupled open-shell metal centres — the FeMoco problem.
  * **Cuprates and other correlated superconductors** : the physics of interest is precisely the part that mean-field theory discards.
  * **Molecular magnets, actinide chemistry, bond dissociation, excited states with multi-reference character.**

The label for all of them is **strong correlation** , and it means the same thing in every case: no single determinant, and no low-order correction to one, is a good starting point.

### Feynman's Observation

In 1981 Feynman pointed out that the difficulty is a mismatch of representations rather than a fundamental impossibility. A classical computer stores the amplitudes of a quantum state one number at a time, so it needs exponentially many numbers. A quantum system with the right number of degrees of freedom stores that state as *its own state*, at no representational cost at all. Simulating a quantum system, he argued, is natural work for a quantum machine.

Concretely: under the Jordan-Wigner mapping of Chapter 4, one spin orbital becomes one qubit. The resource requirement changes character completely:

| Quantity | Classical exact solution | Qubit register |
| --- | --- | --- |
| Storage of the state | \\(\binom{M}{N_\alpha}\binom{M}{N_\beta}\\) amplitudes | \\(2M\\) qubits |
| Growth with orbital count | exponential | linear |

That single row is the whole physical case for quantum computing in materials science. It is not a claim about speed, and it is not a claim about all computation — it is a claim about the representation of many-body quantum states.

### Code Example 1: The Exponential Wall, in Numbers

The abstract statement "exponential" is much less persuasive than the actual byte counts. This example evaluates the FCI dimension for several representative active spaces and compares it with the number of qubits the same problem needs.

```python
import numpy as np
from math import comb


def fci_dimension(n_orb: int, n_elec: int) -> int:
    """Dimension of the full CI space: n_elec electrons in n_orb spatial orbitals."""
    n_alpha = (n_elec + 1) // 2
    n_beta = n_elec // 2
    return comb(n_orb, n_alpha) * comb(n_orb, n_beta)


def human_bytes(nbytes: float) -> str:
    """Format a byte count with binary prefixes."""
    for unit in ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if nbytes < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2e} YB"


# (name, spatial orbitals, electrons) - representative active spaces.
# FeMoco: the (113 electrons, 76 spatial orbitals) active space of the full
# FeMo cofactor; this series uses that convention wherever FeMoco appears.
systems = [
    ("H2 / STO-3G",          2,   2),
    ("H2O / 6-31G",         13,  10),
    ("N2 / cc-pVDZ",        28,  14),
    ("Fe2S2 active space",  20,  30),
    ("FeMoco active space", 76, 113),
]

header = (f"{'system':<22}{'orbitals':>9}{'electrons':>10}"
          f"{'qubits':>8}{'FCI dimension':>16}{'CI vector':>12}")
print(header)
print("-" * len(header))
for name, n_orb, n_elec in systems:
    dim = fci_dimension(n_orb, n_elec)
    print(f"{name:<22}{n_orb:>9}{n_elec:>10}{2 * n_orb:>8}"
          f"{dim:>16.3e}{human_bytes(dim * 8):>12}")

print()
print("Classical state-vector simulation of n qubits (complex128):")
for n in [10, 20, 30, 40, 50]:
    print(f"  n = {n:>3} qubits -> {2**n:>16,d} amplitudes"
          f" -> {human_bytes(2**n * 16):>10}")

# How many qubits does a hard problem need, and how big is that vector?
n_qubits = 2 * 76
print()
print(f"FeMoco active space needs {n_qubits} qubits under Jordan-Wigner.")
print(f"A classical state vector of that size would hold 2^{n_qubits} = "
      f"{2.0**n_qubits:.2e} amplitudes.")
print(f"Atoms in the observable universe: ~1e80 for comparison.")
```

```
system                 orbitals electrons  qubits   FCI dimension   CI vector
-----------------------------------------------------------------------------
H2 / STO-3G                   2         2       4       4.000e+00      32.0 B
H2O / 6-31G                  13        10      26       1.656e+06     12.6 MB
N2 / cc-pVDZ                 28        14      56       1.402e+12     10.2 TB
Fe2S2 active space           20        30      40       2.404e+08      1.8 GB
FeMoco active space          76       113     152       4.169e+35 2.76e+12 YB

Classical state-vector simulation of n qubits (complex128):
  n =  10 qubits ->            1,024 amplitudes ->    16.0 kB
  n =  20 qubits ->        1,048,576 amplitudes ->    16.0 MB
  n =  30 qubits ->    1,073,741,824 amplitudes ->    16.0 GB
  n =  40 qubits -> 1,099,511,627,776 amplitudes ->    16.0 TB
  n =  50 qubits -> 1,125,899,906,842,624 amplitudes ->    16.0 PB

FeMoco active space needs 152 qubits under Jordan-Wigner.
A classical state vector of that size would hold 2^152 = 5.71e+45 amplitudes.
Atoms in the observable universe: ~1e80 for comparison.
```

**What to look for.** Nitrogen in a double-zeta basis — a system a graduate student would call small — already needs ten terabytes for its CI vector, which is why FCI is essentially never run at that size. The FeMoco row is not a large number, it is a meaningless one: no conceivable classical machine stores \\(10^{35}\\) amplitudes. And yet the qubit column stays modest, growing by twos. The second table is the same wall seen from the other side, and it is also the reason every simulation in this series stays below about 20 qubits: 16 MB is comfortable, 16 TB is not.

### Where the Honesty Belongs

Two qualifications should be attached to the argument above before it becomes an overstatement, and they will be developed in Chapters 3 and 5:

  1. **Having the register is not having the state.** A qubit register can *hold* the ground state, but *preparing* it is a separate problem, and in the general case finding the ground state of a local Hamiltonian is QMA-hard — hard even for a quantum computer. Every practical algorithm, VQE included, is a heuristic for this preparation step, with no proof of advantage.
  2. **Reading the answer costs shots.** The state contains \\(2^n\\) amplitudes but each measurement returns a single bit string. Extracting an energy to chemical accuracy requires an enormous number of repetitions — Example 8 measures this, and Chapter 3 treats it as a design constraint.

Neither point cancels the scaling argument. Both explain why the field is 30 years old and still, for materials science, pre-utility.

* * *

## 1.2 State Vectors and Dirac Notation

### The Qubit as a Two-Level System

A **qubit** is any quantum system with two accessible, addressable levels. Nothing about it is specific to computing; it is the two-level system already familiar from magnetic resonance and from optical spectroscopy.

| Physical realization | The two levels | Where materials researchers meet it |
| --- | --- | --- |
| Spin-1/2 in a magnetic field | \\(m_s = +1/2, -1/2\\) | NMR, ESR, magnetization dynamics |
| Superconducting transmon | lowest two levels of an anharmonic oscillator | The dominant hardware platform today |
| Trapped-ion hyperfine levels | two long-lived atomic states | The highest-fidelity gates measured |
| Photon polarization | horizontal, vertical | Quantum optics, linear-optical schemes |
| NV centre in diamond | \\(m_s = 0, \pm 1\\) of the S=1 ground state | Quantum sensing, magnetometry |
| Semiconductor quantum-dot spin | electron spin up, down | Silicon-compatible qubit proposals |

The point of the abstraction is that the mathematics below is identical for all of them, and Chapter 5 will only return to the physical realization where the noise depends on it.

### Kets, Amplitudes, Normalization

The two levels are labelled \\(|0\rangle\\) and \\(|1\rangle\\) and taken as an orthonormal basis of a two-dimensional complex Hilbert space:

\\[ |0\rangle = \begin{pmatrix} 1 \\\\ 0 \end{pmatrix}, \qquad |1\rangle = \begin{pmatrix} 0 \\\\ 1 \end{pmatrix} \\]

A general pure state is a complex linear combination — a **superposition** :

\\[ |\psi\rangle = \alpha|0\rangle + \beta|1\rangle = \begin{pmatrix} \alpha \\\\ \beta \end{pmatrix}, \qquad \alpha, \beta \in \mathbb{C} \\]

The numbers \\(\alpha\\) and \\(\beta\\) are **amplitudes**. They are not probabilities, and the difference is the whole subject. The only constraint is normalization:

\\[ \langle\psi|\psi\rangle = |\alpha|^2 + |\beta|^2 = 1 \\]

The **bra** \\(\langle\psi|\\) is the conjugate transpose of the ket, so the inner product of two states is

\\[ \langle\phi|\psi\rangle = \phi_0^{\ast}\psi_0 + \phi_1^{\ast}\psi_1 \\]

In NumPy this is `np.vdot(phi, psi)`, which conjugates its *first* argument — the same convention as the bra-ket. Using `np.dot` instead is a classic and silent error, because it agrees with `vdot` for real vectors and disagrees for complex ones.

Two states are **orthogonal** when \\(\langle\phi|\psi\rangle = 0\\); orthogonal states are perfectly distinguishable by a suitable measurement, and non-orthogonal states are not.

### Four Superpositions Worth Naming

\\[ |+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}, \qquad |-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}, \qquad |{\pm}i\rangle = \frac{|0\rangle \pm i|1\rangle}{\sqrt{2}} \\]

All four give probability \\(1/2\\) for each outcome when measured in the \\(\\{|0\rangle, |1\rangle\\}\\) basis. They are nevertheless four different physical states: \\(|+\rangle\\) and \\(|-\rangle\\) are orthogonal to each other, so a suitable measurement distinguishes them with certainty.

This is the first place where a qubit departs from a probabilistic classical bit. A coin that comes up heads half the time has one description; a qubit with \\(P(0) = 1/2\\) has a continuum of them, distinguished by the **relative phase** between the amplitudes. That phase is what makes interference possible, and interference is what quantum algorithms are made of.

### Global Phase Versus Relative Phase

Write the general state in polar form:

\\[ |\psi\rangle = e^{i\gamma}\left(\cos\frac{\theta}{2}|0\rangle + e^{i\varphi}\sin\frac{\theta}{2}|1\rangle\right) \\]

The overall factor \\(e^{i\gamma}\\) is a **global phase**. It cancels out of every measurement probability and out of every expectation value, so it is not physical: \\(|\psi\rangle\\) and \\(e^{i\gamma}|\psi\rangle\\) are *the same state*, and the physical state space of one qubit is therefore two real parameters, not three.

The inner factor \\(e^{i\varphi}\\) is a **relative phase** between the two amplitudes, and it is entirely physical. Changing \\(\varphi\\) rotates the state to a different, distinguishable one. Keeping this distinction straight is the difference between a working algorithm and a debugging session.

### Code Example 2: Amplitudes, Norms, and Phases

```python
import numpy as np

# The computational basis
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)

# Four states that all give 50/50 measurement statistics in the Z basis
plus = (ket0 + ket1) / np.sqrt(2)          # |+>
minus = (ket0 - ket1) / np.sqrt(2)         # |->
plus_i = (ket0 + 1j * ket1) / np.sqrt(2)   # |+i>
minus_i = (ket0 - 1j * ket1) / np.sqrt(2)  # |-i>

states = {"|0>": ket0, "|1>": ket1, "|+>": plus,
          "|->": minus, "|+i>": plus_i, "|-i>": minus_i}


def norm(psi):
    """Norm sqrt(<psi|psi>). np.vdot conjugates its first argument."""
    return np.sqrt(np.real(np.vdot(psi, psi)))


print("state    amplitudes                          norm   P(0)    P(1)")
print("-" * 68)
for name, psi in states.items():
    p = np.abs(psi) ** 2
    amp = f"[{psi[0]:+.3f}, {psi[1]:+.3f}]"
    print(f"{name:<8} {amp:<36} {norm(psi):.3f}  {p[0]:.3f}   {p[1]:.3f}")

# Normalising an arbitrary vector
raw = np.array([3, 4j], dtype=complex)
psi = raw / norm(raw)
print(f"\nraw vector      : {raw}, norm = {norm(raw):.4f}")
print(f"normalised      : {np.round(psi, 4)}, norm = {norm(psi):.4f}")
print(f"probabilities   : P(0) = {abs(psi[0])**2:.4f}, P(1) = {abs(psi[1])**2:.4f}")

# Global phase is unobservable, relative phase is not
gamma = 0.7
print("\nGlobal vs relative phase")
print(f"  |+>              probabilities: {np.round(np.abs(plus)**2, 6)}")
print(f"  e^(i*0.7)|+>     probabilities: "
      f"{np.round(np.abs(np.exp(1j*gamma)*plus)**2, 6)}   <- identical")
print(f"  |->              probabilities: {np.round(np.abs(minus)**2, 6)}"
      "   <- also identical in this basis")
print(f"  |<+|->|          = {abs(np.vdot(plus, minus)):.3f}"
      "   <- orthogonal: perfectly distinguishable in some other basis")
print(f"  |<+|e^(i*0.7)|+>|= {abs(np.vdot(plus, np.exp(1j*gamma)*plus)):.3f}"
      "   <- modulus 1: the very same physical state")

# Orthonormality of the computational basis
gram = np.array([[np.vdot(a, b) for b in (ket0, ket1)] for a in (ket0, ket1)])
print("\nGram matrix of {|0>, |1>}:")
print(np.real_if_close(gram))
```

```
state    amplitudes                          norm   P(0)    P(1)
--------------------------------------------------------------------
|0>      [+1.000+0.000j, +0.000+0.000j]       1.000  1.000   0.000
|1>      [+0.000+0.000j, +1.000+0.000j]       1.000  0.000   1.000
|+>      [+0.707+0.000j, +0.707+0.000j]       1.000  0.500   0.500
|->      [+0.707+0.000j, -0.707+0.000j]       1.000  0.500   0.500
|+i>     [+0.707+0.000j, +0.000+0.707j]       1.000  0.500   0.500
|-i>     [+0.707+0.000j, +0.000-0.707j]       1.000  0.500   0.500

raw vector      : [3.+0.j 0.+4.j], norm = 5.0000
normalised      : [0.6+0.j  0. +0.8j], norm = 1.0000
probabilities   : P(0) = 0.3600, P(1) = 0.6400

Global vs relative phase
  |+>              probabilities: [0.5 0.5]
  e^(i*0.7)|+>     probabilities: [0.5 0.5]   <- identical
  |->              probabilities: [0.5 0.5]   <- also identical in this basis
  |<+|->|          = 0.000   <- orthogonal: perfectly distinguishable in some other basis
  |<+|e^(i*0.7)|+>|= 1.000   <- modulus 1: the very same physical state

Gram matrix of {|0>, |1>}:
[[1. 0.]
 [0. 1.]]
```

**What to look for.** The four superpositions have identical `P(0)` and `P(1)` columns and different amplitude columns. That is the entire content of the phrase "a qubit is more than a random bit". The last two printed lines separate the two kinds of phase: \\(|+\rangle\\) and \\(|-\rangle\\) have overlap 0 (different states with identical \\(Z\\)-basis statistics), while \\(|+\rangle\\) and \\(e^{i\gamma}|+\rangle\\) have overlap of modulus 1 (the same state described by different numbers). Note also `dtype=complex` on every array: a real-valued array will silently discard the imaginary part on assignment, which produces wrong physics with no error message.

* * *

## 1.3 The Bloch Sphere

### Two Angles Are Enough

Discarding the global phase and imposing normalization leaves exactly two real parameters, so the pure states of one qubit form a two-dimensional surface. The standard parameterization is

\\[ |\psi(\theta, \varphi)\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\varphi}\sin\frac{\theta}{2}|1\rangle, \qquad \theta \in [0, \pi], \quad \varphi \in [0, 2\pi) \\]

and the surface is a sphere — the **Bloch sphere**. The correspondence is made concrete by the three Pauli expectation values, which define the **Bloch vector**

\\[ \mathbf{r} = (\langle X\rangle, \langle Y\rangle, \langle Z\rangle) = (\sin\theta\cos\varphi,\; \sin\theta\sin\varphi,\; \cos\theta) \\]

with the Pauli matrices

\\[ X = \begin{pmatrix} 0 & 1 \\\\ 1 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & -i \\\\ i & 0 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\\\ 0 & -1 \end{pmatrix} \\]

For every pure state \\(|\mathbf{r}| = 1\\) exactly. The landmarks are worth memorizing:

| State | \\(\theta\\) | \\(\varphi\\) | Bloch vector | Position |
| --- | --- | --- | --- | --- |
| \\(\lvert 0\rangle\\) | 0 | — | \\((0,0,1)\\) | North pole |
| \\(\lvert 1\rangle\\) | \\(\pi\\) | — | \\((0,0,-1)\\) | South pole |
| \\(\lvert +\rangle\\) | \\(\pi/2\\) | 0 | \\((1,0,0)\\) | \\(+x\\) |
| \\(\lvert -\rangle\\) | \\(\pi/2\\) | \\(\pi\\) | \\((-1,0,0)\\) | \\(-x\\) |
| \\(\lvert {+}i\rangle\\) | \\(\pi/2\\) | \\(\pi/2\\) | \\((0,1,0)\\) | \\(+y\\) |
| \\(\lvert {-}i\rangle\\) | \\(\pi/2\\) | \\(3\pi/2\\) | \\((0,-1,0)\\) | \\(-y\\) |

Two features of the geometry are worth stating explicitly because they surprise people. **Orthogonal states are antipodal** , not perpendicular: \\(\langle 0|1\rangle = 0\\) and their Bloch vectors point in opposite directions. And the half-angle in \\(\cos(\theta/2)\\) means the Hilbert-space angle is half the Bloch angle — a \\(2\pi\\) rotation on the Bloch sphere corresponds to a factor \\(-1\\) on the state vector, the familiar spinor double cover.

### Why This Picture Is Physically Real

For a spin-1/2, the Bloch vector is not a bookkeeping device — it *is* the direction of the magnetic moment, up to the gyromagnetic factor. Larmor precession about an applied field is rigid rotation of the Bloch vector about the field axis; a Rabi pulse is a rotation about an axis in the equatorial plane; \\(T_1\\) relaxation shrinks the \\(z\\) component and \\(T_2\\) dephasing shrinks the transverse components. Anyone who has interpreted an ESR or NMR experiment has been doing quantum-gate geometry already. Chapter 2 gives the same rotations their gate names, and Chapter 5 lets the vector shrink inside the sphere, which is where mixed states and the density matrix enter.

### Where the Picture Stops Working

The Bloch sphere is a single-qubit tool and does not generalize. A pure state of \\(n\\) qubits has \\(2 \cdot 2^n - 2\\) real parameters after normalization and global phase, whereas \\(n\\) Bloch spheres carry only \\(3n\\). For \\(n = 2\\) that is 6 versus 6 — and yet the picture still fails, because the two-qubit parameters are not distributed as two independent Bloch vectors: entanglement lives in the correlations, and each qubit of a Bell state has Bloch vector \\(\mathbf{0}\\), the centre of the sphere. Any intuition of the form "each qubit is an arrow" breaks exactly where the interesting physics starts.

### Code Example 3: Bloch Vectors and a Bloch-Sphere Plot

This example converts between the two representations in both directions, verifies numerically that all pure states lie on the unit sphere, and draws the sphere with the named states marked. The figure is produced by the code — run it locally to see it; nothing in the text depends on the image.

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def state_from_angles(theta: float, phi: float) -> np.ndarray:
    """|psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>."""
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def bloch_vector(psi: np.ndarray) -> np.ndarray:
    """Bloch vector (<X>, <Y>, <Z>) of a single-qubit state."""
    return np.array([np.real(np.vdot(psi, P @ psi)) for P in (X, Y, Z)])


def angles_from_state(psi: np.ndarray) -> tuple:
    """Recover (theta, phi) from a state vector, fixing the global phase."""
    a, b = psi
    theta = 2 * np.arccos(np.clip(np.abs(a), 0.0, 1.0))
    phi = 0.0 if np.isclose(np.abs(b), 0.0) else np.angle(b) - np.angle(a)
    return theta, np.mod(phi, 2 * np.pi)


named = {
    "|0>":  (0.0, 0.0),
    "|1>":  (np.pi, 0.0),
    "|+>":  (np.pi / 2, 0.0),
    "|->":  (np.pi / 2, np.pi),
    "|+i>": (np.pi / 2, np.pi / 2),
    "|-i>": (np.pi / 2, 3 * np.pi / 2),
    "(pi/2,pi/4)": (np.pi / 2, np.pi / 4),
}

print("state        theta/pi  phi/pi     <X>      <Y>      <Z>   |r|")
print("-" * 66)
for name, (theta, phi) in named.items():
    psi = state_from_angles(theta, phi)
    r = np.round(bloch_vector(psi), 12) + 0.0
    print(f"{name:<12} {theta/np.pi:>7.3f} {phi/np.pi:>7.3f}  "
          f"{r[0]:>+7.3f} {r[1]:>+7.3f} {r[2]:>+7.3f}  {np.linalg.norm(r):.3f}")

# Round trip: state -> Bloch angles -> state
theta0, phi0 = 1.1, 2.3
psi = state_from_angles(theta0, phi0)
theta1, phi1 = angles_from_state(psi)
print(f"\nround trip: (theta, phi) = ({theta0:.4f}, {phi0:.4f}) -> "
      f"({theta1:.4f}, {phi1:.4f})")

# Every pure single-qubit state sits exactly on the unit sphere
rng = np.random.default_rng(7)
radii = []
for _ in range(2000):
    v = rng.normal(size=4)
    psi = (v[:2] + 1j * v[2:]) / np.linalg.norm(v)
    radii.append(np.linalg.norm(bloch_vector(psi)))
print(f"2000 random pure states: |r| in "
      f"[{min(radii):.12f}, {max(radii):.12f}]  (should be exactly 1)")

# --- Visualisation ---------------------------------------------------------
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 30)
ax.plot_wireframe(np.outer(np.cos(u), np.sin(v)),
                  np.outer(np.sin(u), np.sin(v)),
                  np.outer(np.ones_like(u), np.cos(v)),
                  color="lightgray", linewidth=0.4)

for axis, label in zip(np.eye(3), ["x  (<X>)", "y  (<Y>)", "z  (<Z>)"]):
    ax.quiver(0, 0, 0, *axis, color="k", arrow_length_ratio=0.08, linewidth=1)
    ax.text(*(1.15 * axis), label, fontsize=9)

for name, (theta, phi) in named.items():
    r = bloch_vector(state_from_angles(theta, phi))
    ax.quiver(0, 0, 0, *r, color="tab:purple", arrow_length_ratio=0.12, linewidth=2)
    ax.text(*(1.08 * r), name, fontsize=10, color="tab:purple")

ax.set_box_aspect((1, 1, 1))
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
ax.set_axis_off()
ax.set_title("Bloch sphere: single-qubit pure states", fontsize=13)
plt.tight_layout()
plt.show()
```

```
state        theta/pi  phi/pi     <X>      <Y>      <Z>   |r|
------------------------------------------------------------------
|0>            0.000   0.000   +0.000  +0.000  +1.000  1.000
|1>            1.000   0.000   +0.000  +0.000  -1.000  1.000
|+>            0.500   0.000   +1.000  +0.000  +0.000  1.000
|->            0.500   1.000   -1.000  +0.000  +0.000  1.000
|+i>           0.500   0.500   +0.000  +1.000  +0.000  1.000
|-i>           0.500   1.500   +0.000  -1.000  +0.000  1.000
(pi/2,pi/4)    0.500   0.250   +0.707  +0.707  +0.000  1.000

round trip: (theta, phi) = (1.1000, 2.3000) -> (1.1000, 2.3000)
2000 random pure states: |r| in [1.000000000000, 1.000000000000]  (should be exactly 1)
```

**What to look for.** The tabulated Bloch vectors reproduce the landmark table exactly, and the round trip recovers \\((\theta, \varphi)\\) to machine precision even though the state vector was built with an arbitrary phase convention — `angles_from_state` deliberately subtracts `np.angle(a)` to remove the global phase. The random-state check is the numerical statement that the *surface* of the sphere is the pure-state space: 2000 independent draws all land at radius 1 to twelve digits. Any state with \\(|\mathbf{r}| < 1\\) is mixed, and cannot be written as a single state vector at all.

* * *

## 1.4 Measurement and the Born Rule

### The Rule

Measurement is where the amplitudes finally become observable. For a measurement in the computational basis, the **Born rule** states

\\[ P(k) = |\langle k|\psi\rangle|^2 = |\psi_k|^2 \\]

so for one qubit \\(P(0) = |\alpha|^2\\) and \\(P(1) = |\beta|^2\\). Normalization is exactly the requirement that these sum to one. In the Bloch parameterization,

\\[ P(0) = \cos^2\frac{\theta}{2}, \qquad P(1) = \sin^2\frac{\theta}{2} \\]

so the polar angle alone fixes the \\(Z\\)-basis statistics and the azimuthal angle is invisible to this measurement. Measuring along a different axis is a different experiment, and it is the axis choice that makes \\(\varphi\\) observable.

### Collapse

The second half of the measurement postulate is the **projection postulate** : after a measurement returning outcome \\(k\\), the state is the normalized projection

\\[ |\psi\rangle \;\longrightarrow\; \frac{P_k|\psi\rangle}{\sqrt{\langle\psi|P_k|\psi\rangle}}, \qquad P_k = |k\rangle\langle k| \\]

Three consequences matter in practice:

  * **Repeatability.** Measuring the same qubit twice in the same basis gives the same answer the second time, with probability one. The first measurement has already destroyed the superposition.
  * **Irreversibility.** Collapse is the only non-unitary step in the whole formalism. Every gate in Chapter 2 is reversible; measurement is not.
  * **One bit per shot.** A single measurement of an \\(n\\)-qubit register returns one bit string out of \\(2^n\\). All the other amplitudes are gone. This is why quantum algorithms cannot simply "read out" the exponentially large state, and why the useful ones arrange for interference to concentrate probability on the answer *before* the measurement.

### Expectation Values

Most quantities of interest — energies above all — are not single outcomes but expectation values of observables. For a Hermitian operator \\(A\\),

\\[ \langle A\rangle = \langle\psi|A|\psi\rangle = \sum_k a_k P(a_k) \\]

For the Pauli \\(Z\\), whose eigenvalues are \\(\pm 1\\) with eigenvectors \\(|0\rangle\\) and \\(|1\rangle\\),

\\[ \langle Z\rangle = (+1)P(0) + (-1)P(1) = P(0) - P(1) = \cos\theta \\]

which is the concrete recipe an experiment follows: measure many times, assign \\(+1\\) to outcome 0 and \\(-1\\) to outcome 1, average. The variance follows from \\(Z^2 = I\\):

\\[ \operatorname{Var}(Z) = \langle Z^2\rangle - \langle Z\rangle^2 = 1 - \langle Z\rangle^2 \\]

so the statistical error of an estimate from \\(N\\) shots is \\(\sqrt{(1 - \langle Z\rangle^2)/N}\\). Note the pleasant special case: if the state is an eigenstate, the variance vanishes and one shot suffices. It is the intermediate, genuinely superposed states that are expensive to measure — and those are precisely the states a variational algorithm visits.

A general Hamiltonian is handled by writing it as a linear combination of Pauli strings,

\\[ H = \sum_j c_j\, P_j, \qquad \langle H\rangle = \sum_j c_j \langle P_j\rangle \\]

measuring each term separately and summing with the classical coefficients. This is the machinery of Chapter 3; Example 5 previews it in its smallest possible form.

### Code Example 4: The Born Rule and Collapse

```python
import numpy as np


def state_from_angles(theta, phi):
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def probs(state):
    """Born-rule probabilities of all 2^n outcomes."""
    return np.abs(state) ** 2


def measure_qubit(state: np.ndarray, qubit: int, n: int, rng) -> tuple:
    """Measure one qubit of an n-qubit state in the Z basis (big-endian).

    Returns (outcome, collapsed_state). The collapsed state is renormalised,
    so it can be measured again or evolved further.
    """
    psi = state.reshape([2] * n)
    # Projector onto outcome 0 for this qubit: keep the slice, zero the other
    sl0 = [slice(None)] * n
    sl0[qubit] = 0
    branch0 = np.zeros_like(psi)
    branch0[tuple(sl0)] = psi[tuple(sl0)]
    p0 = float(np.sum(np.abs(branch0) ** 2))

    if rng.random() < p0:
        return 0, (branch0 / np.sqrt(p0)).reshape(-1)
    branch1 = psi - branch0
    return 1, (branch1 / np.sqrt(1.0 - p0)).reshape(-1)


rng = np.random.default_rng(2026)

# A biased single-qubit state: theta = pi/3 gives P(0) = cos^2(pi/6) = 3/4
theta, phi = np.pi / 3, 0.4
psi = state_from_angles(theta, phi)
p = probs(psi)
print(f"state |psi(theta=pi/3, phi=0.4)>  ->  P(0) = {p[0]:.4f}, P(1) = {p[1]:.4f}")
print(f"analytic cos^2(theta/2) = {np.cos(theta/2)**2:.4f}, "
      f"sin^2(theta/2) = {np.sin(theta/2)**2:.4f}")

print("\nFinite-sample frequencies converge on the Born probabilities:")
print(f"{'shots':>9}{'f(0)':>10}{'f(1)':>10}{'|f(0) - P(0)|':>16}")
print("-" * 45)
for shots in [10, 100, 1_000, 10_000, 100_000, 1_000_000]:
    outcomes = rng.choice(2, size=shots, p=p)
    f0 = np.mean(outcomes == 0)
    print(f"{shots:>9}{f0:>10.4f}{1-f0:>10.4f}{abs(f0 - p[0]):>16.5f}")

# Collapse: the second measurement of the same qubit always agrees with the first
print("\nRepeated measurement of the same qubit (collapse):")
agree = 0
for _ in range(1000):
    first, collapsed = measure_qubit(psi, 0, 1, rng)
    second, _ = measure_qubit(collapsed, 0, 1, rng)
    agree += (first == second)
print(f"  first and second outcome agreed in {agree}/1000 trials")

# Collapse in a two-qubit product state leaves the other qubit untouched
two = np.kron(state_from_angles(np.pi / 3, 0.0),
              state_from_angles(np.pi / 2, 0.0))   # |psi> (x) |+>
print("\nTwo-qubit product state |psi(pi/3,0)> (x) |+>:")
print(f"  full probabilities over |q0 q1>: {np.round(probs(two), 4)}")
out, after = measure_qubit(two, 0, 2, rng)
print(f"  measured qubit 0 -> {out}")
print(f"  collapsed state probabilities  : {np.round(probs(after), 4)}")
print("  qubit 1 is still 50/50: measuring one factor of a product state "
      "tells you nothing about the other")
```

```
state |psi(theta=pi/3, phi=0.4)>  ->  P(0) = 0.7500, P(1) = 0.2500
analytic cos^2(theta/2) = 0.7500, sin^2(theta/2) = 0.2500

Finite-sample frequencies converge on the Born probabilities:
    shots      f(0)      f(1)   |f(0) - P(0)|
---------------------------------------------
       10    0.8000    0.2000         0.05000
      100    0.7700    0.2300         0.02000
     1000    0.7270    0.2730         0.02300
    10000    0.7511    0.2489         0.00110
   100000    0.7529    0.2471         0.00294
  1000000    0.7502    0.2498         0.00018

Repeated measurement of the same qubit (collapse):
  first and second outcome agreed in 1000/1000 trials

Two-qubit product state |psi(pi/3,0)> (x) |+>:
  full probabilities over |q0 q1>: [0.375 0.375 0.125 0.125]
  measured qubit 0 -> 0
  collapsed state probabilities  : [0.5 0.5 0.  0. ]
  qubit 1 is still 50/50: measuring one factor of a product state tells you nothing about the other
```

**What to look for.** The convergence column falls roughly as \\(1/\sqrt{N}\\) but not monotonically — the 1000-shot run happens to be worse than the 100-shot run. That is not a bug; it is what a random variable with a shrinking standard deviation looks like, and it is worth internalizing before you try to judge whether an optimizer has converged. The collapse test is exact: 1000 out of 1000 agreements, because after the first measurement the state is an eigenstate. In the last block, note that measuring qubit 0 of a *product* state leaves qubit 1 exactly as it was; Chapter 2 repeats this experiment on an entangled state, where it comes out very differently.

### Code Example 5: Expectation Values and a Two-Term Hamiltonian

```python
import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def state_from_angles(theta, phi):
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def expectation(psi: np.ndarray, A: np.ndarray) -> float:
    """<A> = <psi|A|psi> for a Hermitian A."""
    return float(np.real(np.vdot(psi, A @ psi)))


print("theta/pi  phi/pi     <Z>   cos(theta)     <X>  sin(t)cos(p)     <Y>  sin(t)sin(p)")
print("-" * 84)
for theta in [0.0, np.pi / 4, np.pi / 2, 2 * np.pi / 3, np.pi]:
    for phi in [0.0, np.pi / 3]:
        psi = state_from_angles(theta, phi)
        ez, ex, ey = (expectation(psi, Z), expectation(psi, X), expectation(psi, Y))
        print(f"{theta/np.pi:>8.3f}{phi/np.pi:>8.3f}"
              f"{ez:>8.3f}{np.cos(theta):>13.3f}"
              f"{ex:>8.3f}{np.sin(theta)*np.cos(phi):>14.3f}"
              f"{ey:>8.3f}{np.sin(theta)*np.sin(phi):>14.3f}")

# An expectation value is a weighted sum over measurement outcomes:
# <Z> = (+1) P(0) + (-1) P(1)
theta, phi = 1.0, 0.5
psi = state_from_angles(theta, phi)
p = np.abs(psi) ** 2
print(f"\n<Z> from the operator : {expectation(psi, Z):.6f}")
print(f"<Z> from (+1)P(0) + (-1)P(1) : {p[0] - p[1]:.6f}")

# Estimating <Z> from a finite number of shots
rng = np.random.default_rng(11)
exact = expectation(psi, Z)
print(f"\nShot-based estimation of <Z> (exact value {exact:.6f}):")
print(f"{'shots':>9}{'estimate':>12}{'error':>10}{'1/sqrt(N)':>12}")
print("-" * 43)
for shots in [10, 100, 1_000, 10_000, 100_000]:
    outcomes = rng.choice([+1.0, -1.0], size=shots, p=p)
    est = outcomes.mean()
    print(f"{shots:>9}{est:>12.4f}{abs(est-exact):>10.4f}{1/np.sqrt(shots):>12.4f}")

# A Hamiltonian as a linear combination of Pauli terms (the pattern used in ch.3)
coeffs = {"Z": -0.8, "X": 0.3}
paulis = {"X": X, "Y": Y, "Z": Z}
energy = sum(c * expectation(psi, paulis[k]) for k, c in coeffs.items())
H = sum(c * paulis[k] for k, c in coeffs.items())
print(f"\nH = -0.8 Z + 0.3 X")
print(f"  <H> term by term      : {energy:.6f}")
print(f"  <H> from the matrix   : {expectation(psi, H):.6f}")
print(f"  exact ground energy   : {np.linalg.eigvalsh(H)[0]:.6f}")
```

```
theta/pi  phi/pi     <Z>   cos(theta)     <X>  sin(t)cos(p)     <Y>  sin(t)sin(p)
------------------------------------------------------------------------------------
   0.000   0.000   1.000        1.000   0.000         0.000   0.000         0.000
   0.000   0.333   1.000        1.000   0.000         0.000   0.000         0.000
   0.250   0.000   0.707        0.707   0.707         0.707   0.000         0.000
   0.250   0.333   0.707        0.707   0.354         0.354   0.612         0.612
   0.500   0.000   0.000        0.000   1.000         1.000   0.000         0.000
   0.500   0.333   0.000        0.000   0.500         0.500   0.866         0.866
   0.667   0.000  -0.500       -0.500   0.866         0.866   0.000         0.000
   0.667   0.333  -0.500       -0.500   0.433         0.433   0.750         0.750
   1.000   0.000  -1.000       -1.000   0.000         0.000   0.000         0.000
   1.000   0.333  -1.000       -1.000   0.000         0.000   0.000         0.000

<Z> from the operator : 0.540302
<Z> from (+1)P(0) + (-1)P(1) : 0.540302

Shot-based estimation of <Z> (exact value 0.540302):
    shots    estimate     error   1/sqrt(N)
-------------------------------------------
       10      0.6000    0.0597      0.3162
      100      0.5600    0.0197      0.1000
     1000      0.5920    0.0517      0.0316
    10000      0.5422    0.0019      0.0100
   100000      0.5390    0.0013      0.0032

H = -0.8 Z + 0.3 X
  <H> term by term      : -0.210704
  <H> from the matrix   : -0.210704
  exact ground energy   : -0.854400
```

**What to look for.** The first table verifies the Bloch identities term by term: \\(\langle Z\rangle\\) depends only on \\(\theta\\), while \\(\langle X\rangle\\) and \\(\langle Y\rangle\\) carry the azimuthal angle. Notice the first two rows, \\(\theta = 0\\) with two different \\(\varphi\\): the azimuth of a pole is meaningless, and all three expectation values are unchanged. The last block is a variational calculation in miniature. The state at \\((\theta, \varphi) = (1.0, 0.5)\\) gives \\(\langle H\rangle = -0.2107\\), which is above the exact ground energy \\(-0.8544\\) — as the variational principle guarantees it must be. Chapter 3 does nothing more than replace the two hand-chosen angles with a numerical optimizer.

* * *

## 1.5 Multiple Qubits and the Tensor Product

### The State Space Multiplies

Composite quantum systems combine by the **tensor product** , not by concatenation. For two qubits the basis has four elements,

\\[ |00\rangle, \quad |01\rangle, \quad |10\rangle, \quad |11\rangle \\]

and a general state has four complex amplitudes:

\\[ |\psi\rangle = \alpha_{00}|00\rangle + \alpha_{01}|01\rangle + \alpha_{10}|10\rangle + \alpha_{11}|11\rangle \\]

For \\(n\\) qubits the dimension is \\(2^n\\), and this is the same exponential that appeared in Section 1.1. It is worth being precise about the contrast, because the phrase "a qubit can be 0 and 1 at the same time" is a poor summary of it: \\(n\\) classical bits are described by \\(n\\) numbers, and \\(n\\) qubits by \\(2^n - 1\\) complex ones (after normalization and global phase). The resource is the *joint* amplitude structure, not any individual qubit's indecision.

### The Index Convention of This Series

A state vector is a flat array, so we must fix how bit strings map onto array indices. **This series uses big-endian ordering throughout:**

  * qubit 0 is the **leftmost** bit of the string and the **most significant** bit,
  * the basis state \\(|q_0 q_1 \ldots q_{n-1}\rangle\\) has index \\(\sum_{i=0}^{n-1} q_i\, 2^{\,n-1-i}\\),
  * equivalently, \\(|q_0 q_1 \ldots q_{n-1}\rangle = |q_0\rangle \otimes |q_1\rangle \otimes \cdots \otimes |q_{n-1}\rangle\\) built by `np.kron` in left-to-right order.

| Bit string | Index (\\(n=3\\)) |
| --- | --- |
| \\(\lvert 000\rangle\\) | 0 |
| \\(\lvert 001\rangle\\) | 1 |
| \\(\lvert 010\rangle\\) | 2 |
| \\(\lvert 100\rangle\\) | 4 |
| \\(\lvert 101\rangle\\) | 5 |
| \\(\lvert 111\rangle\\) | 7 |

**A warning that will save you an afternoon.** Qiskit uses the opposite convention — little-endian, with qubit 0 as the *rightmost* and least significant bit — so the same physical state prints with its bit strings reversed. Cirq and PennyLane use big-endian, like this series. The two conventions differ only by a relabelling, but mixing them produces results that are wrong without being obviously wrong: a two-qubit CNOT appears to act on the wrong qubit, and a Hamiltonian built from Pauli strings comes out with its terms permuted. Whenever you port code between frameworks, the first thing to test is `ket('01')`.

### Product States and Entangled States

If the two qubits were prepared independently, the joint state is a tensor product,

\\[ |\psi\rangle = |a\rangle \otimes |b\rangle, \qquad \alpha_{ij} = a_i b_j \\]

and then the joint probability distribution factorizes: \\(P(ij) = P(i)P(j)\\). States that *cannot* be written this way are **entangled**. The canonical example is the Bell state

\\[ |\Phi^{+}\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}} \\]

whose marginals are both uniform — each qubit alone looks maximally random — while the joint outcomes are perfectly correlated. No assignment of individual states to the two qubits reproduces that.

The clean test is the **Schmidt decomposition**. Reshape the \\(2^n\\)-vector into a matrix across the cut of interest and take its singular value decomposition:

\\[ |\psi\rangle = \sum_{k} \lambda_k\, |u_k\rangle_A \otimes |v_k\rangle_B, \qquad \sum_k \lambda_k^2 = 1 \\]

The number of nonzero \\(\lambda_k\\) is the **Schmidt rank** : rank 1 means a product state, rank greater than 1 means entanglement. The **entanglement entropy**

\\[ S = -\sum_k \lambda_k^2 \log_2 \lambda_k^2 \\]

quantifies it, running from 0 for a product state to 1 bit for a Bell state.

### The Bridge to Many-Body Physics

None of this is new to a condensed-matter reader; only the notation is. In second quantization a fermionic state is written in the occupation-number basis,

\\[ |n_1 n_2 \ldots n_{2M}\rangle, \qquad n_p \in \\{0, 1\\} \\]

and a general correlated state is a superposition of such strings — a configuration interaction expansion. That is *literally* a qubit register: each spin orbital is one qubit, occupied or empty, and the CI coefficients are the amplitudes. The Jordan-Wigner transformation of Chapter 4 is nothing more than making this identification careful enough to preserve fermionic antisymmetry.

The Schmidt picture also explains why classical methods sometimes win. Ground states of gapped local Hamiltonians in one dimension obey an **area law** : the entanglement entropy across a cut saturates instead of growing with system size, so only a few Schmidt values matter and a matrix-product state with modest bond dimension is enough. That is why DMRG is so successful for chains and so much less successful in two dimensions and for systems with long-range interactions. A quantum computer is interesting exactly where this classical shortcut fails — that is, for highly entangled states. "Is this state weakly entangled?" is therefore the sharpest single question to ask of any proposed quantum-advantage target.

### Code Example 6: Tensor Products, Conventions, and Schmidt Rank

```python
import numpy as np

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)


def ket(bits: str) -> np.ndarray:
    """'01' -> the 4-dimensional basis state |01> (big-endian)."""
    n = len(bits)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[int(bits, 2)] = 1.0
    return psi


# The index convention, spelled out
print("Big-endian index convention for n = 3 qubits")
print(f"{'|q0 q1 q2>':>12}{'index':>8}{'= 4 q0 + 2 q1 + q2':>22}")
print("-" * 42)
for i in range(8):
    bits = format(i, "03b")
    q0, q1, q2 = (int(b) for b in bits)
    print(f"{'|' + bits + '>':>12}{i:>8}{4*q0 + 2*q1 + q2:>22}")

# kron builds the same vectors, in the same order
print("\nket('01') equals kron(|0>, |1>):",
      np.allclose(ket("01"), np.kron(ket0, ket1)))
print("ket('10') equals kron(|1>, |0>):",
      np.allclose(ket("10"), np.kron(ket1, ket0)))
print("ket('01') nonzero index:", int(np.argmax(np.abs(ket("01")))))
print("ket('10') nonzero index:", int(np.argmax(np.abs(ket("10")))),
      " <- the order of the factors matters")

# A product state: probabilities factorise
a = np.array([np.cos(0.3), np.sin(0.3)], dtype=complex)          # qubit 0
b = np.array([np.cos(1.1), np.exp(0.7j) * np.sin(1.1)], dtype=complex)  # qubit 1
prod = np.kron(a, b)
print("\nProduct state |a> (x) |b>")
print(f"  joint probabilities        : {np.round(np.abs(prod)**2, 5)}")
print(f"  P(q0) x P(q1) outer product: "
      f"{np.round(np.outer(np.abs(a)**2, np.abs(b)**2).ravel(), 5)}")

# A Bell state: probabilities do not factorise
bell = (ket("00") + ket("11")) / np.sqrt(2)
print("\nBell state (|00> + |11>)/sqrt(2)")
print(f"  joint probabilities        : {np.round(np.abs(bell)**2, 5)}")
print("  marginal of qubit 0        :",
      np.round(np.abs(bell.reshape(2, 2)) ** 2 @ np.ones(2), 5))
print("  marginal of qubit 1        :",
      np.round(np.ones(2) @ np.abs(bell.reshape(2, 2)) ** 2, 5))
print("  outer product of marginals : [0.25 0.25 0.25 0.25]  <- not the joint law")

# Schmidt rank: reshape into a matrix and count singular values
print("\nSchmidt decomposition across the cut qubit 0 | qubit 1")
for name, psi in [("product", prod), ("Bell", bell)]:
    s = np.linalg.svd(psi.reshape(2, 2), compute_uv=False)
    rank = int(np.sum(s > 1e-12))
    entropy = max(0.0, -sum(x * np.log2(x) for x in s ** 2 if x > 1e-12))
    print(f"  {name:<8} singular values {np.round(s, 4)}  "
          f"Schmidt rank {rank}  entropy {entropy:.4f} bit")

# The exponential wall, seen from the state vector side
print("\nHow many complex amplitudes does an n-qubit state carry?")
print(f"{'n':>4}{'2^n':>26}{'complex128 memory':>22}")
print("-" * 52)
for n in [1, 2, 10, 20, 30, 40, 50, 60]:
    nbytes = 2.0 ** n * 16
    unit = "B"
    for u in ["kB", "MB", "GB", "TB", "PB", "EB"]:
        if nbytes >= 1024:
            nbytes /= 1024
            unit = u
    print(f"{n:>4}{2**n:>26,d}{f'{nbytes:.1f} {unit}':>22}")
```

```
Big-endian index convention for n = 3 qubits
  |q0 q1 q2>   index    = 4 q0 + 2 q1 + q2
------------------------------------------
       |000>       0                     0
       |001>       1                     1
       |010>       2                     2
       |011>       3                     3
       |100>       4                     4
       |101>       5                     5
       |110>       6                     6
       |111>       7                     7

ket('01') equals kron(|0>, |1>): True
ket('10') equals kron(|1>, |0>): True
ket('01') nonzero index: 1
ket('10') nonzero index: 2  <- the order of the factors matters

Product state |a> (x) |b>
  joint probabilities        : [0.18778 0.72489 0.01797 0.06936]
  P(q0) x P(q1) outer product: [0.18778 0.72489 0.01797 0.06936]

Bell state (|00> + |11>)/sqrt(2)
  joint probabilities        : [0.5 0.  0.  0.5]
  marginal of qubit 0        : [0.5 0.5]
  marginal of qubit 1        : [0.5 0.5]
  outer product of marginals : [0.25 0.25 0.25 0.25]  <- not the joint law

Schmidt decomposition across the cut qubit 0 | qubit 1
  product  singular values [1. 0.]  Schmidt rank 1  entropy 0.0000 bit
  Bell     singular values [0.7071 0.7071]  Schmidt rank 2  entropy 1.0000 bit

How many complex amplitudes does an n-qubit state carry?
   n                       2^n     complex128 memory
----------------------------------------------------
   1                         2                32.0 B
   2                         4                64.0 B
  10                     1,024               16.0 kB
  20                 1,048,576               16.0 MB
  30             1,073,741,824               16.0 GB
  40         1,099,511,627,776               16.0 TB
  50     1,125,899,906,842,624               16.0 PB
  60 1,152,921,504,606,846,976               16.0 EB
```

**What to look for.** The first table is a test of the convention, not a decoration: the index column and the arithmetic column agree row by row, which is exactly the assertion `index = 4*q0 + 2*q1 + q2`. In the product-state block the joint distribution equals the outer product of the marginals to every printed digit; in the Bell block it emphatically does not, even though the two marginals are individually featureless. The Schmidt line makes the distinction quantitative and basis-independent: rank 1 with zero entropy versus rank 2 with one full bit. Keep the `reshape(2, 2)` plus `svd` idiom — it is three lines, it generalizes to any bipartition of any number of qubits, and it is the standard diagnostic for how entangled a simulated state has become.

* * *

## 1.6 Building the Mini Simulator in NumPy

### Design Decisions

The simulator used throughout this series is deliberately small — 99 lines when complete, Code Example 2 of Chapter 2 — and it makes four choices that will not change:

| Decision | Choice | Reason |
| --- | --- | --- |
| Representation | Dense `numpy` array of `complex128` | Exact, transparent, adequate below ~20 qubits |
| Qubit ordering | Big-endian (qubit 0 leftmost, most significant) | Matches the notation of the text and Section 1.5 |
| Interface | Plain functions, no classes | Every function is independently testable and copy-pasteable |
| Purity | Functions return new states, never mutate inputs | Makes the code safe to reuse inside optimization loops |

The last point deserves emphasis. In Chapter 3 these functions are called thousands of times inside an optimizer; a function that quietly modified its input would produce a bug that only appears after a few iterations, which is the most expensive kind to find.

Three functions are built here. The remaining ones — the gate matrices, `apply_gate`, `cnot` and `expval` — arrive in Chapter 2, and the gate constants are already included below so that the module is complete when you get there.

### Code Example 7: `ket`, `probs`, `sample`

```python
"""Mini state-vector simulator, part 1: ket / probs / sample.

Convention (used unchanged in every chapter of this series):
    qubit 0 = leftmost bit = most significant bit
    index of |q0 q1 ... q_{n-1}> = sum_i q_i 2^(n-1-i)
"""
import numpy as np

# --- 2x2 building blocks (used from chapter 2 onwards) ---------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def ket(bits: str) -> np.ndarray:
    """'01' -> the 4-dimensional basis state |01> (big-endian)."""
    n = len(bits)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[int(bits, 2)] = 1.0
    return psi


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


if __name__ == "__main__":
    # 1. Basis states
    print("ket('0')   =", ket("0"))
    print("ket('01')  =", ket("01"))
    print("ket('101') has its amplitude at index",
          int(np.argmax(np.abs(ket("101")))), "= 0b101")

    # 2. Superpositions are built by adding kets and normalising
    bell = (ket("00") + ket("11")) / np.sqrt(2)
    w3 = (ket("100") + ket("010") + ket("001")) / np.sqrt(3)
    print(f"\nnorm of the Bell state : {np.linalg.norm(bell):.6f}")
    print(f"norm of the W state    : {np.linalg.norm(w3):.6f}")

    # 3. probs
    print("\nprobs((|00> + |11>)/sqrt(2)):")
    for i, p in enumerate(probs(bell)):
        print(f"  |{format(i, '02b')}>  {p:.4f}")
    print(f"sum of probabilities = {probs(bell).sum():.6f}")

    # 4. sample - finite statistics, reproducible with a seed
    counts = sample(bell, shots=4000, seed=42)
    print("\nsample(bell, shots=4000, seed=42):", counts)
    print("relative frequencies:",
          {k: round(v / 4000, 4) for k, v in counts.items()})

    counts_w = sample(w3, shots=6000, seed=7)
    print("\nsample(W state, shots=6000, seed=7):", counts_w)
    print("expected 1/3 each ->",
          {k: round(v / 6000, 4) for k, v in counts_w.items()})

    # 5. An unequal superposition: amplitudes are not probabilities
    psi = (0.6 * ket("00") + 0.8j * ket("11"))
    print(f"\npsi = 0.6|00> + 0.8i|11>,  norm = {np.linalg.norm(psi):.4f}")
    print("probs:", np.round(probs(psi), 4))
    print("sample(psi, 10000, seed=1):", sample(psi, 10000, seed=1))

    # 6. Sanity checks worth keeping in your own code
    assert np.isclose(probs(bell).sum(), 1.0)
    assert set(sample(bell, 100, seed=0)) <= {"00", "01", "10", "11"}
    print("\nall assertions passed")
```

```
ket('0')   = [1.+0.j 0.+0.j]
ket('01')  = [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
ket('101') has its amplitude at index 5 = 0b101

norm of the Bell state : 1.000000
norm of the W state    : 1.000000

probs((|00> + |11>)/sqrt(2)):
  |00>  0.5000
  |01>  0.0000
  |10>  0.0000
  |11>  0.5000
sum of probabilities = 1.000000

sample(bell, shots=4000, seed=42): {'00': 2023, '11': 1977}
relative frequencies: {'00': 0.5058, '11': 0.4943}

sample(W state, shots=6000, seed=7): {'001': 2068, '010': 1933, '100': 1999}
expected 1/3 each -> {'001': 0.3447, '010': 0.3222, '100': 0.3332}

psi = 0.6|00> + 0.8i|11>,  norm = 1.0000
probs: [0.36 0.   0.   0.64]
sample(psi, 10000, seed=1): {'00': 3563, '11': 6437}

all assertions passed
```

**What to look for.** `ket` is four lines, and the whole big-endian convention is contained in the single call `int(bits, 2)` — Python's own most-significant-first reading of a binary string. `sample` accepts an optional `seed` so that every printed number in this series is reproducible; on real hardware there is no seed, and rerunning gives different counts. The last example is the one to remember: amplitudes \\(0.6\\) and \\(0.8i\\) give probabilities \\(0.36\\) and \\(0.64\\), and 10 000 shots return 3563 and 6437. The imaginary unit on the second amplitude is completely invisible in the counts — a relative phase never shows up in a measurement in the same basis, only in interference after further gates.

### Code Example 8: How Many Shots Does an Expectation Value Cost?

The state vector holds exact amplitudes, but a real device gives only samples. This example measures the scaling of the statistical error and then converts it into the number that governs Chapter 3.

```python
import numpy as np
import matplotlib.pyplot as plt

Z = np.array([[1, 0], [0, -1]], dtype=complex)


def state_from_angles(theta, phi=0.0):
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def estimate_Z(psi, shots, rng):
    """Estimate <Z> from `shots` projective measurements."""
    p = np.abs(psi) ** 2
    outcomes = rng.choice([+1.0, -1.0], size=shots, p=p)
    return outcomes.mean()


rng = np.random.default_rng(2026)
theta = 1.0
psi = state_from_angles(theta)
exact = float(np.real(np.vdot(psi, Z @ psi)))
variance = 1.0 - exact ** 2          # Var(Z) = <Z^2> - <Z>^2 = 1 - <Z>^2

print(f"state: theta = 1.0 rad,  <Z> = {exact:.6f},  Var(Z) = {variance:.6f}")
print(f"\n{'shots':>9}{'RMS error':>12}{'sqrt(Var/N)':>14}{'ratio':>8}")
print("-" * 43)

shot_list = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000]
rms = []
for shots in shot_list:
    errs = [estimate_Z(psi, shots, rng) - exact for _ in range(400)]
    r = float(np.sqrt(np.mean(np.square(errs))))
    rms.append(r)
    predicted = np.sqrt(variance / shots)
    print(f"{shots:>9}{r:>12.5f}{predicted:>14.5f}{r/predicted:>8.3f}")

slope, intercept = np.polyfit(np.log(shot_list), np.log(rms), 1)
print(f"\nlog-log slope = {slope:.3f}   (theory: -0.5)")

# How many shots for chemical accuracy on one Pauli term?
target = 1.6e-3        # Hartree, "chemical accuracy" (1 kcal/mol)
needed = variance / target ** 2
print(f"\nShots needed for a standard error of {target:.1e} on ONE Pauli term:"
      f" {needed:,.0f}")
print("A molecular Hamiltonian has hundreds of terms, and a VQE optimisation")
print("needs hundreds of iterations. This single number is the reason chapter 3")
print("treats measurement cost as a first-class design constraint.")

# --- Visualisation ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(shot_list, rms, "o-", color="tab:purple", label="measured RMS error")
ax.loglog(shot_list, np.sqrt(variance / np.array(shot_list)), "k--",
          label=r"$\sqrt{\mathrm{Var}(Z)/N}$")
ax.set_xlabel("number of shots $N$")
ax.set_ylabel(r"RMS error of $\langle Z \rangle$")
ax.set_title("Shot noise scales as $1/\\sqrt{N}$")
ax.grid(alpha=0.3, which="both")
ax.legend()
plt.tight_layout()
plt.show()
```

```
state: theta = 1.0 rad,  <Z> = 0.540302,  Var(Z) = 0.708073

    shots   RMS error   sqrt(Var/N)   ratio
-------------------------------------------
       10     0.26891       0.26610   1.011
       30     0.16542       0.15363   1.077
      100     0.08422       0.08415   1.001
      300     0.04961       0.04858   1.021
     1000     0.02626       0.02661   0.987
     3000     0.01614       0.01536   1.050
    10000     0.00840       0.00841   0.998
    30000     0.00493       0.00486   1.016

log-log slope = -0.503   (theory: -0.5)

Shots needed for a standard error of 1.6e-03 on ONE Pauli term: 276,591
A molecular Hamiltonian has hundreds of terms, and a VQE optimisation
needs hundreds of iterations. This single number is the reason chapter 3
treats measurement cost as a first-class design constraint.
```

**What to look for.** The `ratio` column stays within a few percent of 1 across three and a half decades, and the fitted slope is \\(-0.503\\) against the predicted \\(-1/2\\): the error is pure shot noise with no hidden bias. The consequence is unforgiving. Halving the error costs four times the measurements, so the \\(2.8 \times 10^5\\) shots needed for chemical accuracy on a *single* Pauli term become of order \\(10^{10}\\) once a realistic Hamiltonian and a realistic optimization are accounted for. Nothing about the algorithm is wrong; \\(1/\sqrt{N}\\) is simply the price of the Born rule. Chapter 3 discusses grouping commuting terms and other ways of reducing the constant, and Chapter 5 explains why noise makes the constant worse still.

### What Exists Now, and What Comes Next

| Component | Status after this chapter |
| --- | --- |
| `ket(bits)` | ✅ implemented |
| `probs(state)` | ✅ implemented |
| `sample(state, shots, seed=None)` | ✅ implemented |
| `I2, X, Y, Z, H, S, T` | ✅ defined, used from Chapter 2 |
| `rx, ry, rz(theta)` | Chapter 2 |
| `apply_gate(state, U, targets, n)` | Chapter 2 |
| `cnot(state, control, target, n)` | Chapter 2 |
| `expval(state, pauli, coeff_map=None)` | Chapter 2 |

Everything so far is *state preparation and readout*. What is entirely missing is **dynamics** : there is no way yet to turn one state into another. That is the subject of the next chapter, and it will take about sixty more lines of NumPy.

* * *

## Exercises

#### Exercise 1: Normalization, Phase, and the Bloch Vector

Consider

\\[ |\psi\rangle = \frac{1-i}{2}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle \\]

  1. Verify that the state is normalized.
  2. Compute \\(P(0)\\) and \\(P(1)\\).
  3. Find the relative phase between the two amplitudes, and the Bloch angles \\((\theta, \varphi)\\).
  4. Give the Bloch vector, and say which named state on the sphere this is closest to.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\left|\frac{1-i}{2}\right|^2 = \frac{1^2+1^2}{4} = \frac{1}{2}\) and \(\left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}\). The sum is 1, so the state is normalized.</p>

<p><strong>2.</strong> \(P(0) = P(1) = 0.5\). The state lies on the equator of the Bloch sphere.</p>

<p><strong>3.</strong> The relative phase is \(\arg(\beta) - \arg(\alpha) = 0 - (-\pi/4) = \pi/4\), so \(\varphi = \pi/4\). Since \(|\alpha| = \cos(\theta/2) = 1/\sqrt{2}\), we get \(\theta = \pi/2\).</p>

<p><strong>4.</strong> \(\mathbf{r} = (\sin\theta\cos\varphi, \sin\theta\sin\varphi, \cos\theta) = (0.7071, 0.7071, 0)\) — on the equator, halfway between \(|+\rangle\) (at \(+x\)) and \(|{+}i\rangle\) (at \(+y\)). Numerically:</p>

```python
import numpy as np
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
psi = np.array([(1 - 1j) / 2, 1 / np.sqrt(2)], dtype=complex)
print(round(np.real(np.vdot(psi, psi)), 12))           # 1.0
print(np.abs(psi) ** 2)                                # [0.5 0.5]
print((np.angle(psi[1]) - np.angle(psi[0])) / np.pi)   # 0.25  -> phi = pi/4
print(np.round([np.real(np.vdot(psi, P @ psi)) for P in (X, Y, Z)], 6))
# [0.707107 0.707107 0.      ]
```

<p>Note that the global phase of the given amplitudes is not zero, yet the Bloch vector is unaffected — only the <em>difference</em> of the phases entered.</p>

</details>

#### Exercise 2: From Angles to Expectation Values

Take the state \\(|\psi(\theta,\varphi)\rangle\\) with \\(\theta = 2\pi/3\\) and \\(\varphi = \pi/6\\).

  1. Write down the two amplitudes numerically.
  2. Compute \\(P(0)\\) and \\(P(1)\\).
  3. Compute \\(\langle X\rangle\\), \\(\langle Y\rangle\\) and \\(\langle Z\rangle\\) and check them against the Bloch formulas.
  4. Verify \\(|\mathbf{r}| = 1\\), and state the variance of a \\(Z\\) measurement on this state.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\cos(\theta/2) = \cos(\pi/3) = 0.5\) and \(e^{i\pi/6}\sin(\pi/3) = 0.8660\,e^{i\pi/6} = 0.75 + 0.4330i\), so \(|\psi\rangle = 0.5|0\rangle + (0.75 + 0.4330i)|1\rangle\).</p>

<p><strong>2.</strong> \(P(0) = 0.25\), \(P(1) = 0.75\).</p>

<p><strong>3.</strong> \(\langle Z\rangle = \cos\theta = -0.5\); \(\langle X\rangle = \sin\theta\cos\varphi = 0.8660 \times 0.8660 = 0.75\); \(\langle Y\rangle = \sin\theta\sin\varphi = 0.8660 \times 0.5 = 0.4330\).</p>

<p><strong>4.</strong> \(|\mathbf{r}|^2 = 0.5625 + 0.1875 + 0.25 = 1\). The variance is \(1 - \langle Z\rangle^2 = 1 - 0.25 = 0.75\), so estimating \(\langle Z\rangle\) to \(\pm 0.01\) needs about \(0.75/10^{-4} = 7500\) shots.</p>

```python
import numpy as np
theta, phi = 2 * np.pi / 3, np.pi / 6
psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
print(np.round(psi, 4))              # [0.5 +0.j    0.75+0.433j]
print(np.round(np.abs(psi)**2, 6))   # [0.25 0.75]
# <X>, <Y>, <Z> = 0.75, 0.433013, -0.5 ; |r| = 1.0
```

</details>

#### Exercise 3: Index Bookkeeping

Using the big-endian convention of this series:

  1. Which array index does \\(|1011\rangle\\) occupy, for \\(n = 4\\)?
  2. Which bit string corresponds to index 6, for \\(n = 4\\)?
  3. Which indices are nonzero for \\(\frac{1}{\sqrt{2}}(|0011\rangle + |1100\rangle)\\), and what does this state describe if the four qubits are four spin orbitals?
  4. If you loaded the same bit string \\(1011\\) into a little-endian framework such as Qiskit, which index would it land on, and what kind of bug does this produce?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(1011_2 = 8 + 0 + 2 + 1 = 11\).</p>

<p><strong>2.</strong> \(6 = 4 + 2\), so the bit string is \(0110\).</p>

<p><strong>3.</strong> \(0011_2 = 3\) and \(1100_2 = 12\). Reading the qubits as spin-orbital occupations, this is an equal superposition of two configurations: one with the last two orbitals occupied and one with the first two occupied — the structure of a two-configuration (multi-reference) wave function, exactly the situation where a single Slater determinant fails.</p>

<p><strong>4.</strong> Little-endian reads the same string in the opposite order, so \(1011\) becomes \(1101_2 = 13\) instead of 11. The failure mode is the nasty one: nothing raises an error, the state is still normalized, and probabilities still sum to one. What changes is <em>which physical qubit</em> each amplitude refers to, so a CNOT appears to act on the wrong target and a Pauli-string Hamiltonian is silently permuted. Test <code>ket('01')</code> first whenever code crosses a framework boundary.</p>

```python
print(int("1011", 2))          # 11   big-endian
print(format(6, "04b"))        # 0110
print(int("1011"[::-1], 2))    # 13   little-endian reading
```

</details>

#### Exercise 4: Marginals, Collapse, and Entanglement

Consider the two-qubit state

\\[ |\psi\rangle = \frac{1}{\sqrt{6}}\left(|00\rangle + |01\rangle + 2|11\rangle\right) \\]

  1. Confirm normalization and list the four joint probabilities.
  2. Compute the marginal distribution of each qubit.
  3. What is the state after measuring qubit 0 and obtaining 1? With what probability does that happen?
  4. Is the state entangled? Compute the Schmidt coefficients and the entanglement entropy across the cut.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> The amplitudes are \((1, 1, 0, 2)/\sqrt{6}\), and \((1 + 1 + 0 + 4)/6 = 1\). The joint probabilities are \(P(00) = P(01) = 1/6\), \(P(10) = 0\), \(P(11) = 4/6 = 2/3\).</p>

<p><strong>2.</strong> Qubit 0: \(P(q_0 = 0) = 1/6 + 1/6 = 1/3\), \(P(q_0 = 1) = 0 + 2/3 = 2/3\). Qubit 1: \(P(q_1 = 0) = 1/6 + 0 = 1/6\), \(P(q_1 = 1) = 1/6 + 2/3 = 5/6\).</p>

<p><strong>3.</strong> Only \(|11\rangle\) survives the projection, so the collapsed state is \(|11\rangle\) exactly, and the outcome occurs with probability \(2/3\). Note that this measurement has determined qubit 1 as well — that is a correlation, and it is the signature of entanglement.</p>

<p><strong>4.</strong> Reshaping into the matrix \(\frac{1}{\sqrt{6}}\begin{pmatrix} 1 & 1 \\ 0 & 2\end{pmatrix}\) gives singular values \(0.9342\) and \(0.3568\). Both are nonzero, so the Schmidt rank is 2 and the state is entangled — but only partially: the squared coefficients are \(0.8727\) and \(0.1273\), giving \(S = 0.5500\) bit, well below the 1 bit of a Bell state.</p>

```python
import numpy as np
psi = np.array([1, 1, 0, 2], dtype=complex) / np.sqrt(6)
M = psi.reshape(2, 2)
print(np.round(np.sum(np.abs(M)**2, axis=1), 6))   # [0.333333 0.666667] marginal q0
print(np.round(np.sum(np.abs(M)**2, axis=0), 6))   # [0.166667 0.833333] marginal q1
s = np.linalg.svd(M, compute_uv=False)
print(np.round(s, 6))                              # [0.934172 0.356822]
print(round(-sum(x * np.log2(x) for x in s**2), 6))  # 0.550048 bit
```

</details>

#### Exercise 5: A Resource Estimate

You are asked whether a quantum computer could help with a calculation on water in a 6-31G basis (13 spatial orbitals, 10 electrons).

  1. How many qubits does the problem need under a one-qubit-per-spin-orbital mapping, and what is the FCI dimension?
  2. Would the FCI vector fit in the memory of a laptop? Would a 40-qubit state vector?
  3. Suppose the qubit Hamiltonian has 200 Pauli terms, each requiring a standard error of \\(10^{-3}\\) Hartree, and the optimizer needs 300 iterations. Estimate the total number of shots, taking \\(\operatorname{Var}(P_j) \le 1\\).
  4. At a repetition rate of \\(10^4\\) shots per second, how long does that take? What does this tell you about where the research effort has to go?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> 13 spatial orbitals is 26 spin orbitals, hence <strong>26 qubits</strong>. The FCI dimension is \(\binom{13}{5}^2 = 1287^2 = 1\,656\,369 \approx 1.7 \times 10^6\).</p>

<p><strong>2.</strong> The CI vector needs \(1.66 \times 10^6 \times 8\) bytes \(\approx 12.6\) MB — trivially small, which is why this calculation is routinely done classically and would be a poor demonstration target. A 40-qubit state vector needs \(2^{40} \times 16\) bytes = <strong>16 TB</strong>, which is a supercomputer, not a laptop.</p>

<p><strong>3.</strong> With \(\operatorname{Var} \le 1\), one term to \(\sigma = 10^{-3}\) costs \(1/\sigma^2 = 10^6\) shots. Then \(10^6 \times 200 \times 300 = 6 \times 10^{10}\) shots.</p>

<p><strong>4.</strong> \(6\times10^{10}/10^4 = 6 \times 10^6\) s \(\approx\) <strong>69 days</strong> of continuous measurement — for a molecule a laptop handles exactly in seconds. Raising the rate to \(10^6\) shots per second brings it to about 0.7 days, which is why repetition rate, term grouping and shot-allocation strategies are active research areas rather than engineering details. The honest reading is that near-term utility is limited far more by measurement cost and noise than by qubit count.</p>

```python
from math import comb
dim = comb(13, 5) ** 2
print(dim, round(dim * 8 / 1024**2, 2))      # 1656369 12.64   (MB)
print(2**40 * 16 / 1024**4)                  # 16.0           (TB)
total = (1 / 1e-3**2) * 200 * 300
print(f"{total:.2e}", round(total / 1e4 / 86400, 1))   # 6.00e+10 69.4  (days)
```

</details>

* * *

## Summary

### Key Takeaways

**1\. The motivation is a scaling argument**

  * The exact many-electron wave function lives in a space of dimension \\(\binom{M}{N_\alpha}\binom{M}{N_\beta}\\), which is beyond classical storage for the strongly correlated systems that matter most.
  * Every polynomial-scaling classical method buys its speed with a structural assumption; strong correlation is where those assumptions break.
  * A qubit register stores the same state with a number of qubits *linear* in the orbital count. That, and not speed in general, is the physical case for quantum computing in materials science.

**2\. A qubit is a normalized complex vector**

  * \\(|\psi\rangle = \alpha|0\rangle + \beta|1\rangle\\) with \\(|\alpha|^2 + |\beta|^2 = 1\\); amplitudes are not probabilities.
  * A global phase is unphysical; a relative phase is physical and is what makes interference possible.
  * `np.vdot`, not `np.dot`, implements the inner product, and every array should be `dtype=complex`.

**3\. The Bloch sphere is the complete single-qubit picture**

  * Two angles suffice: \\(|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\varphi}\sin(\theta/2)|1\rangle\\), and \\(\mathbf{r} = (\langle X\rangle, \langle Y\rangle, \langle Z\rangle)\\) has unit length for every pure state.
  * Orthogonal states are antipodal, and the Bloch vector of a spin-1/2 is its physical magnetic moment.
  * The picture does not extend to many qubits: each qubit of a Bell state sits at the centre of its own sphere.

**4\. Measurement is where amplitudes become numbers**

  * Born rule \\(P(k) = |\psi_k|^2\\); projection postulate for the collapsed state; one bit string per shot.
  * \\(\langle Z\rangle = P(0) - P(1)\\), and a general Hamiltonian is measured term by term after Pauli decomposition.
  * \\(\operatorname{Var}(Z) = 1 - \langle Z\rangle^2\\), so the statistical error falls only as \\(1/\sqrt{N}\\).

**5\. Many qubits means tensor products**

  * \\(\dim = 2^n\\), with the big-endian index convention \\(\sum_i q_i 2^{\,n-1-i}\\) used throughout this series — the opposite of Qiskit's.
  * Product states factorize; entangled states do not, and the Schmidt rank across a cut is the test.
  * The occupation-number basis of second quantization *is* a qubit register, which is what makes the mapping in Chapter 4 possible; area-law entanglement is what makes DMRG work classically.

**6\. The simulator is small and it is yours**

  * `ket`, `probs`, `sample` are about twenty lines and are enough to prepare and read out any state.
  * Reproducibility comes from an explicit seed; exactness comes from `complex128`; safety comes from never mutating inputs.
  * Nothing yet evolves a state — dynamics is the missing ingredient.

**Practical implications**

  * Test the index convention first whenever code crosses a framework boundary.
  * Budget shots before believing a variational result; a converged optimizer with too few shots is converged to noise.
  * Ask of any quantum-advantage claim: is the target state highly entangled, and how many measurements does the answer require?

In the next chapter we add dynamics. Unitary evolution turns the Schrödinger equation into a finite set of reversible gates; the single-qubit gates become rotations of the Bloch vector, CNOT creates the entanglement we could only write down by hand here, and the mini simulator gains `apply_gate`, `cnot` and `expval` — at which point it can run any circuit in the rest of the series.

[← Series Top](<index.html>) [Chapter 2: Quantum Gates and Circuits →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
