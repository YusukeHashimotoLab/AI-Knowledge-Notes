---
title: "Chapter 4: Quantum Computing for Chemistry and Materials"
chapter_title: "Chapter 4: Quantum Computing for Chemistry and Materials"
subtitle: ⚛️ Second Quantization, the Jordan-Wigner Transform, and Model Hamiltonians You Can Diagonalize
reading_time: 40-45 minutes
difficulty: Advanced
code_examples: 6
exercises: 6
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-computing-introduction/chapter-4.html>) | Last sync: 2026-08-12

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to Quantum Computing](<index.html>) > Chapter 4

Chapter 3 ran a variational quantum eigensolver on a two-qubit Hamiltonian whose coefficients were handed to us in a table. That table did not fall from the sky. It is the end product of a chain — electrons, orbitals, creation and annihilation operators, a fermion-to-qubit mapping — and every link in that chain constrains what a quantum computer can actually do for materials research. This chapter builds the chain explicitly. By the end you will be able to take a lattice model or a small active space, write it as a sum of Pauli strings, and diagonalize it exactly on your laptop, which is the only honest way to know whether a quantum algorithm applied to the same problem got the right answer.

Two themes run through the chapter. The first is **why the electronic structure problem is hard**: not because the equations are unknown, but because the exact solution lives in a space whose dimension grows combinatorially, and because the approximations that work brilliantly for weakly correlated matter (density functional theory above all) fail exactly where the interesting materials are. The second is **the price of every step of the translation**: the Jordan-Wigner transform buys anticommutation at the cost of locality; Trotterization buys a circuit at the cost of an error that shrinks only as $1/r$; phase estimation buys precision at the cost of depth. Chapter 5 will put numbers on what today's hardware can pay. Here we establish what is being bought.

The second-quantization formalism used from Section 4.2 onward is developed in more depth in the [Introduction to Quantum Field Theory](<../quantum-field-theory-introduction/index.html>) course; the variational principle behind Section 4.6 is in the [Introduction to Quantum Mechanics](<../quantum-mechanics/index.html>) course; and the tensor and eigenvalue machinery is in [Linear Algebra and Tensors](<../linear-algebra-tensor/index.html>).

## Learning Objectives

After completing this chapter, you will be able to:

  * Quantify the scaling of the exact electronic structure problem, and state where the classical full-CI wall actually sits in orbitals, determinants and bytes
  * Explain why density functional theory fails for strongly correlated materials, and identify the physical signatures (near-degeneracy, fractional occupations, Mott insulation) that mark a problem as strongly correlated
  * Write a fermionic Hamiltonian in second-quantized form, and state the anticommutation relations that any qubit mapping must reproduce
  * Implement the Jordan-Wigner transform from scratch, verify the canonical anticommutation relations numerically, and explain why the parity string destroys locality
  * Distinguish digital from analog quantum simulation, and compare quantum phase estimation with the variational quantum eigensolver on depth, precision and hardware requirements
  * Measure the Trotter error of a first- and second-order product formula, and extrapolate the gate count that a target accuracy would demand
  * Construct the qubit Hamiltonian of a transverse-field Ising chain and of a two-site Hubbard model, diagonalize both exactly, and interpret the results physically (quantum phase transition, Mott localization, superexchange)
  * Run a VQE on those same Hamiltonians and diagnose the three separate error sources — ansatz expressivity, optimizer convergence, and measurement statistics

* * *

## 4.1 The Scaling of the Electronic Structure Problem

### The problem is not the equation

For a molecule or a solid within the Born-Oppenheimer approximation, the electronic Hamiltonian is completely known:

$$ \hat{H} = -\sum_i \frac{\hbar^2}{2m_e}\nabla_i^2 - \sum_{i,A} \frac{Z_A e^2}{4\pi\epsilon_0 |\mathbf{r}_i - \mathbf{R}_A|} + \sum_{i<j} \frac{e^2}{4\pi\epsilon_0 |\mathbf{r}_i - \mathbf{r}_j|} $$

There is nothing to discover here. The difficulty is entirely computational, and it comes from the last term: the electron-electron repulsion couples every coordinate to every other, so the wavefunction does not factorize.

Expand the many-electron wavefunction in Slater determinants built from $M$ spatial orbitals. With $N_\alpha$ up-spin and $N_\beta$ down-spin electrons, the number of determinants is

$$ D = \binom{M}{N_\alpha}\binom{M}{N_\beta} $$

This is the dimension of the **full configuration interaction** (full CI) space — the exact answer within the chosen orbital basis. It is a binomial coefficient, and binomial coefficients are brutal.

### Where the wall is

A quantum computer needs one qubit per spin orbital, so $M$ spatial orbitals cost $2M$ qubits. The comparison worth internalizing is between $D$ (what a classical exact calculation must store) and $2M$ (what a quantum computer must build).

One convention, fixed here and used for the rest of the chapter: $M$ always counts **spatial** orbitals, so the system has $2M$ spin orbitals and needs $2M$ qubits. Where an index runs over spin orbitals — as it does from Section 4.2 onward — its range is $0 \ldots 2M-1$.

Code Example 1: How Fast the Exact Problem Grows

```python
"""Chapter 4, Example 1: how fast the exact electronic-structure problem grows."""
import numpy as np
from math import comb


def fci_dimension(n_orb: int, n_alpha: int, n_beta: int) -> int:
    """Number of Slater determinants in a full-CI expansion."""
    return comb(n_orb, n_alpha) * comb(n_orb, n_beta)


print("Exact diagonalization of the electronic-structure problem")
print("(closed shell, half filling, one qubit per spin orbital)\n")
print(f"{'n_orb':>6} {'n_elec':>7} {'qubits':>7} "
      f"{'FCI determinants':>20} {'2^qubits':>12} {'FCI vector':>15}")
print("-" * 72)
for n_orb in (2, 4, 8, 12, 16, 20, 30, 50, 100):
    n_alpha = n_beta = n_orb // 2            # closed shell, half filled
    dim = fci_dimension(n_orb, n_alpha, n_beta)
    n_qubits = 2 * n_orb                     # one qubit per spin orbital
    mem_gib = dim * 16 / 2 ** 30             # one complex128 amplitude each
    print(f"{n_orb:6d} {2*n_alpha:7d} {n_qubits:7d} "
          f"{dim:20.6e} {2.0**n_qubits:12.3e} {mem_gib:11.3e} GiB")

print("\nWhere the classical wall is:")
prev = None
for n_orb in (16, 18, 20, 22, 24):
    dim = fci_dimension(n_orb, n_orb // 2, n_orb // 2)
    growth = f"  x{dim/prev:5.2f}" if prev else "        "
    print(f"  n_orb = {n_orb:3d}: {dim:>18,d} determinants, "
          f"{dim * 16 / 2**40:9.3f} TiB per vector{growth}")
    prev = dim

# FeMoco: the (113 electrons, 76 spatial orbitals) active space of the full
# FeMo cofactor -- the convention used throughout this series (cf. Chapter 1).
print("\nQubit count for representative active spaces:")
for name, n_orb, n_elec in (
        ("H2 / STO-3G", 2, 2),
        ("LiH / STO-3G", 6, 4),
        ("N2 triple bond", 6, 6),
        ("FeMoco active space (literature scale)", 76, 113)):
    print(f"  {name:40s}: {n_elec:3d} e- in {n_orb:3d} orbitals"
          f" -> {2*n_orb:4d} qubits")
```

```text
Exact diagonalization of the electronic-structure problem
(closed shell, half filling, one qubit per spin orbital)

 n_orb  n_elec  qubits     FCI determinants     2^qubits      FCI vector
------------------------------------------------------------------------
     2       2       4         4.000000e+00    1.600e+01   5.960e-08 GiB
     4       4       8         3.600000e+01    2.560e+02   5.364e-07 GiB
     8       8      16         4.900000e+03    6.554e+04   7.302e-05 GiB
    12      12      24         8.537760e+05    1.678e+07   1.272e-02 GiB
    16      16      32         1.656369e+08    4.295e+09   2.468e+00 GiB
    20      20      40         3.413478e+10    1.100e+12   5.086e+02 GiB
    30      30      60         2.406145e+16    1.153e+18   3.585e+08 GiB
    50      50     100         1.597964e+28    1.268e+30   2.381e+20 GiB
   100     100     200         1.017906e+58    1.607e+60   1.517e+50 GiB

Where the classical wall is:
  n_orb =  16:        165,636,900 determinants,     0.002 TiB per vector        
  n_orb =  18:      2,363,904,400 determinants,     0.034 TiB per vector  x14.27
  n_orb =  20:     34,134,779,536 determinants,     0.497 TiB per vector  x14.44
  n_orb =  22:    497,634,306,624 determinants,     7.242 TiB per vector  x14.58
  n_orb =  24:  7,312,459,672,336 determinants,   106.410 TiB per vector  x14.69

Qubit count for representative active spaces:
  H2 / STO-3G                             :   2 e- in   2 orbitals ->    4 qubits
  LiH / STO-3G                            :   4 e- in   6 orbitals ->   12 qubits
  N2 triple bond                          :   6 e- in   6 orbitals ->   12 qubits
  FeMoco active space (literature scale)  : 113 e- in  76 orbitals ->  152 qubits
```

**What to notice.** Every added pair of orbitals multiplies the determinant count by about 14, and the factor is still creeping upward at 24 orbitals. A 20-orbital active space needs half a terabyte for a single wavefunction vector; a 24-orbital space needs a hundred terabytes, and an exact eigensolver needs several such vectors at once. That is why production full-CI calculations stop near 20 orbitals, and why the phrase "active space" appears in every strongly correlated study: you choose a handful of orbitals to treat exactly and hope the rest are boring.

The same table also punctures a common overstatement. A quantum computer's advantage here is **memory-like**: 40 qubits hold the same information as a $1.1 \times 10^{12}$-dimensional vector, and 152 qubits would hold the FeMoco active space. But holding a state is not the same as finding the ground state, and the qubit count is the *cheapest* of the three resources we will budget. Depth and measurement cost are far more demanding — Section 4.4 and Chapter 5 make this quantitative.

### Why not just use DFT?

Density functional theory computes ground-state properties from the electron density rather than the wavefunction, at a cost that scales roughly as $O(M^3)$ instead of exponentially. It is, by an enormous margin, the most successful method in computational materials science, and for most of the periodic table it is the right tool.

Its failures are systematic and well characterized:

Regime | Physical signature | Why DFT struggles
---|---|---
Transition-metal oxides | Partially filled localized $d$ shells | Self-interaction error over-delocalizes electrons; band gaps collapse
Mott insulators | Insulating despite a half-filled band | The insulating mechanism is interaction-driven, absent from a single-determinant reference
Bond dissociation | Two near-degenerate configurations at large $R$ | Static correlation cannot be captured by one determinant
Transition-metal catalysis | Several spin states within a few kcal/mol | Errors in relative spin-state energies exceed the energy differences of interest
Lanthanides/actinides | Localized $f$ electrons | Strong correlation plus relativistic effects
Unconventional superconductors, spin liquids | Long-range entanglement | The ground state is not close to any single determinant. Conventional BCS superconductors are the opposite case — a mean-field state, well described by DFT plus Eliashberg theory

The common thread is **static (strong) correlation**: the exact ground state is a superposition of several determinants with comparable weight, and no single-reference method — DFT, Hartree-Fock, coupled cluster with a single reference — can represent it. This is precisely the regime where a quantum computer's ability to hold a general superposition is, in principle, the right tool.

It is worth being blunt about the practical consequence: the useful target for quantum computing is not "replace DFT". It is "supply the strongly correlated fragment that DFT gets wrong", typically an active space of a few dozen orbitals embedded in a much larger DFT or classical-correlation treatment. Everything in this chapter is aimed at that fragment.

### The classical competition is not standing still

Any assessment of quantum advantage has to compare against the best classical method, not against full CI:

Method | Cost | Where it is excellent | Where it fails
---|---|---|---
Full CI | $O(D)$, $D$ binomial | Exact benchmark, $\lesssim 20$ orbitals | Anything larger
DMRG / MPS | $O(\chi^3)$ in bond dimension $\chi$ | 1D and quasi-1D, low entanglement | 2D and 3D with large entanglement
Quantum Monte Carlo | Polynomial, statistical | Bosons, sign-problem-free fermion models | Fermion sign problem: exponential cost
Coupled cluster (CCSD(T)) | $O(M^7)$ | Weak correlation, closed shells | Strong correlation, bond breaking
Tensor networks (PEPS) | Polynomial with caveats | 2D area-law states | High entanglement, contraction is hard

The honest statement of the opportunity is narrow: quantum computing is a candidate for problems that are simultaneously **strongly correlated** (so DFT and CCSD(T) fail), **too entangled for tensor networks** (so DMRG fails), and **sign-problem-afflicted** (so quantum Monte Carlo fails). That intersection is real — the doped 2D Hubbard model and several transition-metal active spaces sit in it — but it is much smaller than the phrase "quantum computers will simulate chemistry" suggests.

* * *

## 4.2 Second Quantization and Fermions

### Occupation numbers instead of coordinates

Writing antisymmetrized wavefunctions by hand is unpleasant and does not generalize. Second quantization replaces "which electron is in which orbital" with "how many electrons are in each orbital", and enforces antisymmetry through operator algebra instead of determinants. A comprehensive treatment is in the [Introduction to Quantum Field Theory](<../quantum-field-theory-introduction/index.html>) course; here we need only the working rules.

Fix an ordered set of the $2M$ spin orbitals $\lbrace \phi_0, \phi_1, \ldots, \phi_{2M-1} \rbrace$. A basis state is an occupation string

$$ \lvert n_0 n_1 \cdots n_{2M-1} \rangle, \qquad n_p \in \lbrace 0, 1 \rbrace $$

with $n_p = 1$ meaning orbital $p$ is occupied. The Pauli exclusion principle is now automatic: $n_p$ cannot exceed 1.

Creation and annihilation operators act as

$$ \hat{c}_p^\dagger \lvert \cdots n_p = 0 \cdots \rangle = (-1)^{\sigma_p} \lvert \cdots n_p = 1 \cdots \rangle, \qquad \hat{c}_p^\dagger \lvert \cdots n_p = 1 \cdots \rangle = 0 $$

$$ \hat{c}_p \lvert \cdots n_p = 1 \cdots \rangle = (-1)^{\sigma_p} \lvert \cdots n_p = 0 \cdots \rangle, \qquad \hat{c}_p \lvert \cdots n_p = 0 \cdots \rangle = 0 $$

where the sign is set by the **parity** of the occupations to the left of $p$:

$$ \sigma_p = \sum_{q<p} n_q $$

That parity factor is the entire content of fermionic antisymmetry, and it is what makes the qubit mapping nontrivial. Everything else is bookkeeping.

### The anticommutation relations

The defining algebra is

$$ \lbrace \hat{c}_p, \hat{c}_q^\dagger \rbrace = \hat{c}_p \hat{c}_q^\dagger + \hat{c}_q^\dagger \hat{c}_p = \delta_{pq}, \qquad \lbrace \hat{c}_p, \hat{c}_q \rbrace = 0, \qquad \lbrace \hat{c}_p^\dagger, \hat{c}_q^\dagger \rbrace = 0 $$

Any candidate qubit mapping is correct if and only if it reproduces these relations exactly. In Section 4.3 we will verify all $2 \times 4^2$ of them numerically for a four-mode system, because a mapping that gets a sign wrong produces a Hamiltonian that looks plausible and is silently wrong.

The number operator is $\hat{n}_p = \hat{c}_p^\dagger \hat{c}_p$, with eigenvalues 0 and 1, and the total particle number is $\hat{N} = \sum_p \hat{n}_p$.

### The electronic Hamiltonian

In this language the electronic Hamiltonian is a two-line object:

$$ \hat{H} = \sum_{pq} h_{pq} \hat{c}_p^\dagger \hat{c}_q + \frac{1}{2}\sum_{pqrs} h_{pqrs} \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s $$

with the one- and two-electron integrals

$$ h_{pq} = \int d\mathbf{x}\, \phi_p^\ast(\mathbf{x}) \left(-\frac{\hbar^2}{2m_e}\nabla^2 + V_{\text{nuc}}(\mathbf{x})\right) \phi_q(\mathbf{x}) $$

$$ h_{pqrs} = \frac{e^2}{4\pi\epsilon_0}\int d\mathbf{x}_1 d\mathbf{x}_2\, \frac{\phi_p^\ast(\mathbf{x}_1)\phi_q^\ast(\mathbf{x}_2)\phi_r(\mathbf{x}_2)\phi_s(\mathbf{x}_1)}{|\mathbf{x}_1 - \mathbf{x}_2|} $$

Three consequences matter for what follows.

  1. **The integrals are classical input.** They come from a Hartree-Fock or DFT calculation on a classical computer. A quantum algorithm consumes them; it does not produce them.
  2. **There are $O(M^4)$ of them.** The two-electron tensor has $M^4$ entries (fewer after symmetry, but the same scaling), and each becomes a group of Pauli strings. For 20 spatial orbitals that is on the order of $10^5$ terms. Every one of them has to be measured — this is the origin of the measurement bottleneck quantified in Chapter 5.
  3. **The form is universal.** Lattice models are the same expression with almost all integrals set to zero. The Hubbard model keeps one hopping amplitude and one on-site repulsion; the transverse-field Ising chain is what remains after a further simplification to spins. This is why we can practise on four-qubit models and still be learning the real machinery.

Materials research also cares about *effective* models in which the electrons have already been integrated out — most importantly the large-$U$ limit of the Hubbard model, where charge fluctuations freeze and only spins remain, giving the Heisenberg model $\hat{H} = J \sum_{\langle ij \rangle} \left(\hat{\mathbf{S}}_i \cdot \hat{\mathbf{S}}_j - \tfrac{1}{4}\right)$ with $J = 4t^2/U$. The constant $-J/4$ per bond is not decoration: it comes from the same second-order process, and without it the singlet energy of one bond would be $-3J/4$ instead of $-J$ — wrong by a factor of $4/3$. Section 4.6 checks the $-J = -4t^2/U$ value numerically. Spin models map onto qubits directly — one spin-1/2 per qubit, no parity strings, no Jordan-Wigner — which makes them the cheapest interesting targets for quantum simulation. Section 4.6 watches the $4t^2/U$ law emerge to within 0.1% at $U/t = 128$.

* * *

## 4.3 The Jordan-Wigner Transformation

### The mapping

We need to represent operators satisfying anticommutation relations using qubits, whose natural operators (Pauli matrices on different qubits) *commute*. The Jordan-Wigner transform solves this by attaching a string of $Z$ operators that counts the parity of the occupations to the left:

$$ \hat{c}_p = \left(\prod_{q<p} Z_q\right) \frac{X_p + iY_p}{2}, \qquad \hat{c}_p^\dagger = \left(\prod_{q<p} Z_q\right) \frac{X_p - iY_p}{2} $$

The local factor is a raising/lowering operator on qubit $p$:

$$ \frac{X - iY}{2} = \begin{pmatrix} 0 & 0 \\\\ 1 & 0 \end{pmatrix}, \qquad \frac{X + iY}{2} = \begin{pmatrix} 0 & 1 \\\\ 0 & 0 \end{pmatrix} $$

With the convention of this series (qubit 0 = leftmost bit = most significant bit), occupation 1 corresponds to $\lvert 1 \rangle$ and the number operator is beautifully simple:

$$ \hat{n}_p = \hat{c}_p^\dagger \hat{c}_p = \frac{I - Z_p}{2} $$

The $Z$-string is not decoration. It supplies exactly the $(-1)^{\sigma_p}$ sign of Section 4.2, because $Z$ returns $-1$ on an occupied orbital and $+1$ on an empty one.

### The price: locality

Consider a hopping term between orbitals $p$ and $q$ with $p < q$. The $Z$-strings partially cancel, leaving

$$ \hat{c}_p^\dagger \hat{c}_q + \hat{c}_q^\dagger \hat{c}_p = \frac{1}{2}\left(X_p Z_{p+1}\cdots Z_{q-1} X_q + Y_p Z_{p+1} \cdots Z_{q-1} Y_q\right) $$

A term that was local in the fermionic language — two orbitals — has become a Pauli string acting on $q - p + 1$ qubits. The **Pauli weight** grows linearly with the orbital distance. On hardware with limited connectivity, each such string costs a CNOT ladder proportional to its weight, so the mapping's non-locality translates directly into circuit depth.

This is the motivation for alternatives:

Mapping | Qubits per spin orbital | Weight of a hopping term | Comment
---|---|---|---
Jordan-Wigner | 1 | $O(M)$ | Simplest; $\hat{n}_p$ stays local
Parity | 1 | $O(M)$ | Complementary structure; allows two-qubit reduction by symmetry
Bravyi-Kitaev | 1 | $O(\log M)$ | Best asymptotic weight; more intricate bookkeeping
Ternary-tree / balanced | 1 | $O(\log M)$ | Optimal-weight constructions
Compact / local encodings | $> 1$ | $O(1)$ | Extra qubits buy locality for lattice models

For the four-mode systems in this chapter, Jordan-Wigner costs nothing — the longest string spans four qubits either way — and it is by far the clearest to implement and verify. For a 100-orbital calculation the $O(\log M)$ mappings matter.

### Implementation

The code below is the toolbox for the rest of the chapter. It represents an operator as a dictionary mapping Pauli strings to coefficients, implements the Pauli algebra (multiplication with phases, addition with cancellation), builds the Jordan-Wigner images of $\hat{c}_p$, $\hat{c}_p^\dagger$, $\hat{n}_p$ and hopping terms, and provides a matrix realization for exact diagonalization.

The self-check at the end is the important part: it verifies all $2 \times 4^2$ anticommutation relations, so we know the mapping is right before we trust any energy computed with it.

Code Example 2: Pauli-String Algebra and the Jordan-Wigner Transform

```python
"""Chapter 4, Example 2: Pauli-string algebra and the Jordan-Wigner transform.

This block is the toolbox for the rest of the chapter: run it first, then
Examples 3-6 in the same session (or paste everything into one file).
"""
import numpy as np

TOL = 1e-12

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}

# ---------------------------------------------------------------------
# Pauli-string algebra.  An operator is a dict {'IXZY': coefficient, ...}
# Character j of the string acts on qubit j (qubit 0 = leftmost = MSB).
# ---------------------------------------------------------------------

# single-qubit product table:  a * b = phase * c
_MUL = {
    ('I', 'I'): (1, 'I'),   ('I', 'X'): (1, 'X'),   ('I', 'Y'): (1, 'Y'),   ('I', 'Z'): (1, 'Z'),
    ('X', 'I'): (1, 'X'),   ('X', 'X'): (1, 'I'),   ('X', 'Y'): (1j, 'Z'),  ('X', 'Z'): (-1j, 'Y'),
    ('Y', 'I'): (1, 'Y'),   ('Y', 'X'): (-1j, 'Z'), ('Y', 'Y'): (1, 'I'),   ('Y', 'Z'): (1j, 'X'),
    ('Z', 'I'): (1, 'Z'),   ('Z', 'X'): (1j, 'Y'),  ('Z', 'Y'): (-1j, 'X'), ('Z', 'Z'): (1, 'I'),
}


def pauli_mul(a: str, b: str):
    """Product of two equal-length Pauli strings -> (phase, string)."""
    phase, out = 1.0 + 0j, []
    for ca, cb in zip(a, b):
        p, c = _MUL[(ca, cb)]
        phase *= p
        out.append(c)
    return phase, ''.join(out)


def op_add(*ops):
    """Sum of Pauli-string operators; drops numerically vanishing terms."""
    total = {}
    for op in ops:
        for s, c in op.items():
            total[s] = total.get(s, 0) + c
    return {s: c for s, c in total.items() if abs(c) > TOL}


def op_scale(op, alpha):
    return {s: alpha * c for s, c in op.items()}


def op_mul(op1, op2):
    """Operator product, distributed over Pauli strings."""
    out = {}
    for s1, c1 in op1.items():
        for s2, c2 in op2.items():
            ph, s = pauli_mul(s1, s2)
            out[s] = out.get(s, 0) + c1 * c2 * ph
    return {s: c for s, c in out.items() if abs(c) > TOL}


def op_str(op):
    """Human-readable Pauli-string linear combination."""
    if not op:
        return "0"
    parts = []
    for s, c in sorted(op.items()):
        c = complex(c)
        parts.append(f"{c.real:+.4f}*{s}" if abs(c.imag) < TOL
                     else f"({c.real:+.4f}{c.imag:+.4f}j)*{s}")
    return "  ".join(parts)


def op_weight(op):
    """Largest number of non-identity factors in any string of the operator."""
    return max(sum(1 for ch in s if ch != 'I') for s in op)


# ---------------------------------------------------------------------
# Jordan-Wigner transform
#   c_p     = (Z_0 ... Z_{p-1}) (X_p + i Y_p) / 2
#   c_p^dag = (Z_0 ... Z_{p-1}) (X_p - i Y_p) / 2
#   n_p     = c_p^dag c_p = (I - Z_p) / 2
# ---------------------------------------------------------------------

def jw_annihilate(p: int, n: int):
    """Annihilation operator for spin orbital p, mapped onto n qubits."""
    head = 'Z' * p                        # the parity (Jordan-Wigner) string
    tail = 'I' * (n - p - 1)
    return {head + 'X' + tail: 0.5 + 0j,
            head + 'Y' + tail: 0.5j}


def jw_create(p: int, n: int):
    """Creation operator for spin orbital p, mapped onto n qubits."""
    head = 'Z' * p
    tail = 'I' * (n - p - 1)
    return {head + 'X' + tail: 0.5 + 0j,
            head + 'Y' + tail: -0.5j}


def jw_number(p: int, n: int):
    """Occupation-number operator n_p = (I - Z_p) / 2."""
    return {'I' * n: 0.5 + 0j,
            'I' * p + 'Z' + 'I' * (n - p - 1): -0.5 + 0j}


def jw_hop(p: int, q: int, n: int):
    """Hermitian hopping term c_p^dag c_q + c_q^dag c_p."""
    return op_add(op_mul(jw_create(p, n), jw_annihilate(q, n)),
                  op_mul(jw_create(q, n), jw_annihilate(p, n)))


# ---------------------------------------------------------------------
# Matrix realisation, for verification and exact diagonalization
# ---------------------------------------------------------------------

def pauli_matrix(s: str) -> np.ndarray:
    """Kronecker product in big-endian order: qubit 0 is the outermost factor."""
    M = np.array([[1.0 + 0j]])
    for ch in s:
        M = np.kron(M, PAULI[ch])
    return M


def to_matrix(op) -> np.ndarray:
    n = len(next(iter(op)))
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for s, c in op.items():
        M += c * pauli_matrix(s)
    return M


# =====================================================================
n = 4                                     # four spin orbitals -> four qubits

print("Jordan-Wigner images of the elementary operators (n = 4)")
print("-" * 68)
for p in range(n):
    print(f"  c_{p}      = {op_str(jw_annihilate(p, n))}")
print()
for p in range(n):
    print(f"  n_{p}      = {op_str(jw_number(p, n))}")

print()
print("Canonical anticommutation relations (the whole point of JW)")
print("-" * 68)
ident = 'I' * n
ok = True
for p in range(n):
    for q in range(n):
        ac = op_add(op_mul(jw_annihilate(p, n), jw_create(q, n)),
                    op_mul(jw_create(q, n), jw_annihilate(p, n)))
        expected = {ident: 1.0 + 0j} if p == q else {}
        match = (set(ac) == set(expected)
                 and all(abs(ac[k] - expected[k]) < 1e-12 for k in expected))
        ok &= match
        if p <= q <= p + 1:
            print(f"  {{c_{p}, c_{q}^dag}} = {op_str(ac):<28s} "
                  f"expected {'I' if p == q else '0'}   {'OK' if match else 'FAIL'}")
        ac2 = op_add(op_mul(jw_annihilate(p, n), jw_annihilate(q, n)),
                     op_mul(jw_annihilate(q, n), jw_annihilate(p, n)))
        ok &= (len(ac2) == 0)
print(f"  all 2 x {n}^2 relations verified: {ok}")

print()
print("Hopping terms: locality is destroyed by the parity string")
print("-" * 68)
for (p, q) in ((0, 1), (0, 2), (0, 3), (1, 3)):
    op = jw_hop(p, q, n)
    print(f"  c_{p}^dag c_{q} + h.c. = {op_str(op)}")
    print(f"       -> {len(op)} Pauli strings, maximum weight {op_weight(op)}")

print()
print("Pauli weight of c_p^dag c_q grows linearly with orbital distance")
print("-" * 68)
for dist in range(1, 8):
    op = jw_hop(0, dist, 12)
    print(f"  |p-q| = {dist}: weight {op_weight(op):2d}  ({len(op)} strings)")

print()
print("Consistency check: the mapped number operator on a basis state")
print("-" * 68)
# |1010> means orbitals 0 and 2 occupied (big-endian bit string)
psi = np.zeros(2 ** n, dtype=complex)
psi[int('1010', 2)] = 1.0
for p in range(n):
    occ = np.real(np.vdot(psi, to_matrix(jw_number(p, n)) @ psi))
    print(f"  <1010| n_{p} |1010> = {occ:.1f}")
N_tot = to_matrix(op_add(*[jw_number(p, n) for p in range(n)]))
print(f"  total particle number = {np.real(np.vdot(psi, N_tot @ psi)):.1f}")
```

```text
Jordan-Wigner images of the elementary operators (n = 4)
--------------------------------------------------------------------
  c_0      = +0.5000*XIII  (+0.0000+0.5000j)*YIII
  c_1      = +0.5000*ZXII  (+0.0000+0.5000j)*ZYII
  c_2      = +0.5000*ZZXI  (+0.0000+0.5000j)*ZZYI
  c_3      = +0.5000*ZZZX  (+0.0000+0.5000j)*ZZZY

  n_0      = +0.5000*IIII  -0.5000*ZIII
  n_1      = +0.5000*IIII  -0.5000*IZII
  n_2      = +0.5000*IIII  -0.5000*IIZI
  n_3      = +0.5000*IIII  -0.5000*IIIZ

Canonical anticommutation relations (the whole point of JW)
--------------------------------------------------------------------
  {c_0, c_0^dag} = +1.0000*IIII                 expected I   OK
  {c_0, c_1^dag} = 0                            expected 0   OK
  {c_1, c_1^dag} = +1.0000*IIII                 expected I   OK
  {c_1, c_2^dag} = 0                            expected 0   OK
  {c_2, c_2^dag} = +1.0000*IIII                 expected I   OK
  {c_2, c_3^dag} = 0                            expected 0   OK
  {c_3, c_3^dag} = +1.0000*IIII                 expected I   OK
  all 2 x 4^2 relations verified: True

Hopping terms: locality is destroyed by the parity string
--------------------------------------------------------------------
  c_0^dag c_1 + h.c. = +0.5000*XXII  +0.5000*YYII
       -> 2 Pauli strings, maximum weight 2
  c_0^dag c_2 + h.c. = +0.5000*XZXI  +0.5000*YZYI
       -> 2 Pauli strings, maximum weight 3
  c_0^dag c_3 + h.c. = +0.5000*XZZX  +0.5000*YZZY
       -> 2 Pauli strings, maximum weight 4
  c_1^dag c_3 + h.c. = +0.5000*IXZX  +0.5000*IYZY
       -> 2 Pauli strings, maximum weight 3

Pauli weight of c_p^dag c_q grows linearly with orbital distance
--------------------------------------------------------------------
  |p-q| = 1: weight  2  (2 strings)
  |p-q| = 2: weight  3  (2 strings)
  |p-q| = 3: weight  4  (2 strings)
  |p-q| = 4: weight  5  (2 strings)
  |p-q| = 5: weight  6  (2 strings)
  |p-q| = 6: weight  7  (2 strings)
  |p-q| = 7: weight  8  (2 strings)

Consistency check: the mapped number operator on a basis state
--------------------------------------------------------------------
  <1010| n_0 |1010> = 1.0
  <1010| n_1 |1010> = 0.0
  <1010| n_2 |1010> = 1.0
  <1010| n_3 |1010> = 0.0
  total particle number = 2.0
```

**What to notice.** Four things, each worth a moment.

First, the $Z$-strings are visible in the output: $\hat{c}_0$ has none, $\hat{c}_1$ has one $Z$, $\hat{c}_3$ has three. This is the parity counter, made explicit.

Second, the number operators are all weight 1 — $\hat{n}_p = (I - Z_p)/2$ for every $p$, with no string at all. Jordan-Wigner keeps occupation local even though it makes hopping non-local. That is why the on-site Coulomb term $U \hat{n}_{i\uparrow} \hat{n}_{i\downarrow}$ of the Hubbard model becomes a simple two-qubit $ZZ$ interaction in Section 4.6.

Third, all $2 \times 4^2 = 32$ anticommutation relations hold exactly, and $\lbrace \hat{c}_0, \hat{c}_1^\dagger \rbrace$ prints as the *empty* operator rather than a small number: the cancellation is algebraic, not numerical. Had we forgotten the $Z$-string, this entry would have come out as a nonzero Pauli string and the resulting Hamiltonian would have had a spurious spectrum.

Fourth, the weight table shows the linear growth: hopping between neighbouring orbitals costs weight 2, hopping across seven orbitals costs weight 8. In a molecular Hamiltonian with $O(M^4)$ terms and typical distances $O(M)$, total Pauli weight is the quantity that sets circuit depth.

* * *

## 4.4 Digital and Analog Simulation, QPE and VQE

### Two ways to simulate

**Analog quantum simulation** engineers a physical system whose Hamiltonian *is* (approximately) the target Hamiltonian, and then simply lets it evolve. Cold atoms in optical lattices realize Hubbard models; trapped-ion crystals realize long-range Ising and Heisenberg models; superconducting circuit arrays realize Bose-Hubbard physics. There is no gate compilation, no Trotter error, and coherence requirements are much milder — but the device simulates the Hamiltonian it happens to have, and calibrating and validating it is hard.

**Digital quantum simulation** compiles the target evolution into a universal gate set. It can represent any Hamiltonian, and it is compatible with error correction, which is decisive in the long run. Its cost is depth.

Aspect | Analog | Digital
---|---|---
Programmability | Fixed by hardware | Arbitrary Hamiltonian
Error correction | Not available in general | Compatible
Systematic error | Calibration, unwanted terms | Trotter/compilation error, controllable
Coherence needed | Modest | Large
What limits the size | Array size and calibration, not gate error | (gate count) × (per-gate error) must stay well below 1
Verification | Hard | Circuit-level, testable

### Trotterization

Digital simulation of $e^{-i\hat{H}t}$ needs $\hat{H} = \sum_j \hat{H}_j$ split into pieces we can exponentiate. The pieces do not commute, so the naive product is wrong; the first-order Lie-Trotter formula controls the error:

$$ e^{-i\hat{H}t} = \left(\prod_j e^{-i\hat{H}_j t/r}\right)^r + O\!\left(\frac{t^2}{r}\right) $$

The symmetric (second-order Suzuki) formula halves each step and sweeps forward then backward:

$$ e^{-i\hat{H}t} \approx \left(\prod_j e^{-i\hat{H}_j t/2r} \prod_j^{\text{reverse}} e^{-i\hat{H}_j t/2r}\right)^r + O\!\left(\frac{t^3}{r^2}\right) $$

Each factor $e^{-i\theta P}$ for a Pauli string $P$ compiles into a CNOT ladder, one $R_z$, and the reverse ladder — the identity established in Chapter 2, Section 2.5. So the gate count is (number of Pauli strings) × (Trotter steps) × (weight-dependent CNOT cost).

Let us measure the error rather than trusting the asymptotic notation.

Code Example 3: Trotter Error and the Gate Cost of Digital Simulation

```python
"""Chapter 4, Example 3: Trotter error and the gate cost of digital simulation.
Continues from Example 2 (same session)."""


def expm_hermitian(M, scalar):
    """exp(scalar * M) for Hermitian M, via its eigendecomposition."""
    w, v = np.linalg.eigh(M)
    return (v * np.exp(scalar * w)) @ v.conj().T


def hubbard_dimer_bare(t, U):
    """Two-site Hubbard model without the chemical-potential term.
    Mode order: 0 = (site 0, up), 1 = (site 0, dn), 2 = (site 1, up), 3 = (site 1, dn)."""
    n = 4
    Hq = {}
    for (a, b) in ((0, 2), (1, 3)):        # up-spin and down-spin hopping
        Hq = op_add(Hq, op_scale(jw_hop(a, b, n), -t))
    for (u, d) in ((0, 1), (2, 3)):        # on-site repulsion on each site
        Hq = op_add(Hq, op_scale(op_mul(jw_number(u, n), jw_number(d, n)), U))
    return Hq


t, U, tau = 1.0, 4.0, 1.0
Hq = hubbard_dimer_bare(t, U)
terms = sorted(Hq.items())
n_terms = len(terms)
# the identity string is a global phase: it needs no gate at all
n_rot = sum(1 for s, _ in terms if set(s) != {'I'})
U_exact = expm_hermitian(to_matrix(Hq), -1j * tau)

print(f"Trotter error, two-site Hubbard (t={t}, U={U}), evolution time tau={tau}")
print("=" * 76)
print(f"The Hamiltonian is a sum of {n_terms} Pauli strings"
      f" ({n_rot} of them non-identity, i.e. needing a gate):")
for s, c in terms:
    print(f"    {complex(c).real:+.4f} * {s}")

print(f"\nFirst-order product formula, r steps of dt = tau/r")
print(f"{'steps r':>8} {'dt':>9} {'spectral error':>16} "
      f"{'error x r':>12} {'Pauli rotations':>16}")
for r in (1, 2, 4, 8, 16, 32, 64, 128):
    dt = tau / r
    step = np.eye(2 ** 4, dtype=complex)
    for s, c in terms:
        step = expm_hermitian(to_matrix({s: c}), -1j * dt) @ step
    err = np.linalg.norm(np.linalg.matrix_power(step, r) - U_exact, ord=2)
    print(f"{r:8d} {dt:9.5f} {err:16.6e} {err*r:12.5f} {r*n_rot:16d}")

print("\nSecond-order (symmetric) product formula")
print(f"{'steps r':>8} {'dt':>9} {'spectral error':>16} {'error x r^2':>14}")
for r in (1, 2, 4, 8, 16, 32):
    dt = tau / r
    step = np.eye(2 ** 4, dtype=complex)
    for s, c in terms:
        step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
    for s, c in reversed(terms):
        step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
    err = np.linalg.norm(np.linalg.matrix_power(step, r) - U_exact, ord=2)
    print(f"{r:8d} {dt:9.5f} {err:16.6e} {err*r*r:14.5f}")

print("\nExtrapolated gate cost of a phase-estimation run")
print("-" * 76)
dt = tau / 64
step = np.eye(2 ** 4, dtype=complex)
for s, c in terms:
    step = expm_hermitian(to_matrix({s: c}), -1j * dt) @ step
C = np.linalg.norm(np.linalg.matrix_power(step, 64) - U_exact, ord=2) * 64
print(f"  first-order error ~ C / r  with  C = {C:.4f}")
for target in (1e-2, 1e-3, 1e-6):
    r_need = C / target
    print(f"  error <= {target:.0e}: r = {r_need:12.1f} steps"
          f"  ->  {r_need*n_rot:12.3e} Pauli rotations for tau = 1")
print("\n  Phase estimation to precision eps needs the evolution repeated ~1/eps")
print("  times.  Counting each repetition as an independent tau = 1 block held to")
print("  error eps is the OPTIMISTIC accounting: a single coherent evolution to")
print("  time tau = 1/eps has C ~ tau^2 and needs ~1/eps^3 steps (Exercise 4).")
print(f"  eps = 1e-3  ->  >~{C/1e-3*n_rot/1e-3:.3e} Pauli rotations in total,")
print("  for a model whose exact answer fits in a 16 x 16 matrix.")
```

```text
Trotter error, two-site Hubbard (t=1.0, U=4.0), evolution time tau=1.0
============================================================================
The Hamiltonian is a sum of 11 Pauli strings (10 of them non-identity, i.e. needing a gate):
    +2.0000 * IIII
    -1.0000 * IIIZ
    -1.0000 * IIZI
    +1.0000 * IIZZ
    -0.5000 * IXZX
    -0.5000 * IYZY
    -1.0000 * IZII
    -0.5000 * XZXI
    -0.5000 * YZYI
    -1.0000 * ZIII
    +1.0000 * ZZII

First-order product formula, r steps of dt = tau/r
 steps r        dt   spectral error    error x r  Pauli rotations
       1   1.00000     1.866470e+00      1.86647               10
       2   0.50000     1.416147e+00      2.83229               20
       4   0.25000     8.068454e-01      3.22738               40
       8   0.12500     4.163665e-01      3.33093               80
      16   0.06250     2.098203e-01      3.35713              160
      32   0.03125     1.051154e-01      3.36369              320
      64   0.01562     5.258338e-02      3.36534              640
     128   0.00781     2.629490e-02      3.36575             1280

Second-order (symmetric) product formula
 steps r        dt   spectral error    error x r^2
       1   1.00000     1.590328e+00        1.59033
       2   0.50000     4.799819e-01        1.91993
       4   0.25000     1.320533e-01        2.11285
       8   0.12500     3.368977e-02        2.15615
      16   0.06250     8.462631e-03        2.16643
      32   0.03125     2.118133e-03        2.16897

Extrapolated gate cost of a phase-estimation run
----------------------------------------------------------------------------
  first-order error ~ C / r  with  C = 3.3653
  error <= 1e-02: r =        336.5 steps  ->     3.365e+03 Pauli rotations for tau = 1
  error <= 1e-03: r =       3365.3 steps  ->     3.365e+04 Pauli rotations for tau = 1
  error <= 1e-06: r =    3365336.1 steps  ->     3.365e+07 Pauli rotations for tau = 1

  Phase estimation to precision eps needs the evolution repeated ~1/eps
  times.  Counting each repetition as an independent tau = 1 block held to
  error eps is the OPTIMISTIC accounting: a single coherent evolution to
  time tau = 1/eps has C ~ tau^2 and needs ~1/eps^3 steps (Exercise 4).
  eps = 1e-3  ->  >~3.365e+07 Pauli rotations in total,
  for a model whose exact answer fits in a 16 x 16 matrix.
```

**What to notice.** The scaling claims are confirmed numerically, not assumed. For the first-order formula, error × $r$ converges to 3.366; for the symmetric formula, error × $r^2$ converges to 2.169. Those two constants are the whole content of the $O(1/r)$ and $O(1/r^2)$ statements, made concrete for this Hamiltonian.

The practical message is in the last block. To simulate this four-qubit model for one unit of time to an accuracy of $10^{-6}$ requires roughly $3 \times 10^7$ Pauli rotations with a first-order formula. Combine that with the $1/\varepsilon$ repetitions that phase estimation needs and the count is astronomical for a system whose exact answer we obtain in microseconds by calling `numpy.linalg.eigvalsh` on a 16×16 matrix. Higher-order formulas, better term orderings, qubitization and post-Trotter methods reduce these numbers substantially — by orders of magnitude, not by making them small.

The $3.4 \times 10^7$ rotations printed for $\varepsilon = 10^{-3}$ are a *lower* bound, and it is worth being explicit about why. That figure treats phase estimation as $1/\varepsilon$ independent repetitions of a $\tau = 1$ block, each Trotterized to error $\varepsilon$. A real phase-estimation circuit evolves coherently to time $\tau \sim 1/\varepsilon$, and Exercise 4 shows that the first-order constant grows as $\tau^2$, so holding the *total* unitary error at $\varepsilon$ costs $\sim C\tau^2/\varepsilon = 3.4/\varepsilon^3$ steps — three more orders of magnitude at $\varepsilon = 10^{-3}$. Chapter 5, Example 5 quotes the $3.4 \times 10^7$ figure, so the two chapters are using the same optimistic accounting; neither is a resource estimate you should put in a proposal.

Note also that the identity string carries coefficient $+2$ but needs no gate: a global phase is unobservable. Ten of the eleven terms require rotations. Small bookkeeping like this matters when the resource estimate is the deliverable.

### QPE versus VQE

Two algorithms dominate ground-state estimation, and they occupy opposite ends of the hardware spectrum.

**Quantum phase estimation** prepares a trial state with nonzero overlap $\lvert \langle \psi_{\text{trial}} \vert \psi_0 \rangle \rvert^2 = p_0$ with the ground state, applies controlled powers of $e^{-i\hat{H}t}$, and reads the phase from an inverse Fourier transform. It returns an eigenvalue to precision $\varepsilon$ with circuit depth $O(1/\varepsilon)$ and success probability $p_0$. Its precision does not degrade with more measurement — it is a *deterministic* eigenvalue extraction, up to the overlap issue.

**The variational quantum eigensolver** (Chapter 3) prepares a parameterized state, measures $\langle \hat{H} \rangle$, and lets a classical optimizer minimize it. Depth is set by the ansatz, not by the required precision, which is why it fits current hardware. The price is paid in measurements and in the ansatz's ability to represent the true ground state.

Property | QPE | VQE
---|---|---
Circuit depth | $O(1/\varepsilon)$, very deep | Shallow, ansatz-dependent
Precision limit | Systematic, controllable | Ansatz bias + shot noise
Ancilla qubits | Required ($\log(1/\varepsilon)$) | None
Measurements | Few repetitions | $O(M^4/\varepsilon^2)$ circuits
Optimization | None | Non-convex, barren plateaus
Error correction | Essential | Not required
Guarantees | Eigenvalue to $\varepsilon$ with prob. $p_0$ | Variational upper bound only
Era | FTQC | NISQ

The last row of that table is the single most important line in this chapter. The two algorithms are not competitors for the same machine; they are answers to two different machines. VQE gives a **variational upper bound**: if the ansatz cannot represent the ground state, the answer is wrong in a known direction but by an unknown amount. QPE gives an eigenvalue, but needs depth that only a fault-tolerant machine can supply. Chapter 5 quantifies exactly how far current error rates are from that depth.

* * *

## 4.5 Target Problems in Materials Science

### What makes a good target

A quantum-computing target for materials research should satisfy four criteria simultaneously:

  1. **Strongly correlated** — otherwise DFT or coupled cluster already answers it, faster.
  2. **Small enough active space** — the qubit count and the number of Pauli terms must fit the machine.
  3. **Classically hard** — not just hard for full CI, but hard for DMRG and quantum Monte Carlo too.
  4. **Scientifically decisive** — the answer must change a decision, not merely add a data point.

Very few problems satisfy all four today. Ranking honestly:

Target | Correlation | Active-space size | Classical status | Near-term prospects
---|---|---|---|---
Transverse-field Ising chain | Moderate | Tiny | Exactly solvable | Benchmark only
Heisenberg chains/ladders | Strong | Small | DMRG solves 1D essentially exactly | Benchmark, algorithm development
1D Hubbard | Strong | Small | Bethe ansatz + DMRG | Benchmark
2D Hubbard, doped | Strong | Moderate | Open: sign problem, competing states | Genuine candidate, long term
Transition-metal dimers | Strong | Moderate | DMRG feasible with effort | Plausible mid-term
FeMoco (nitrogenase cofactor) | Very strong | ~76 orbitals | Beyond exact methods | Textbook FTQC target, not NISQ
Battery cathode redox | Moderate-strong | Moderate | DFT+U, hybrid functionals usable | Unclear advantage
Photocatalyst excited states | Strong | Moderate | Multireference methods exist | Interesting, unproven
High-$T_c$ mechanism | Very strong | Large | Open | Not addressable by an active-space calculation

Two remarks on this table. First, the celebrated targets (FeMoco, high-$T_c$) are precisely the ones that need fault tolerance; the targets that fit current machines are precisely the ones classical methods already solve. This gap is the central fact of the field, and it is why Chapter 5 exists. Second, "genuine candidate" for the doped 2D Hubbard model is a statement about the *problem*, not about a timeline: it is classically open and physically important, which makes it the right thing to prepare for.

### The model Hamiltonians

Three models cover most of the near-term literature, and we will build two of them in Section 4.6.

**Transverse-field Ising chain.**

$$ \hat{H} = -J \sum_i Z_i Z_{i+1} - h \sum_i X_i $$

The simplest model with a genuine quantum phase transition, at $h/J = 1$ in the thermodynamic limit. It maps onto free fermions by a Jordan-Wigner transform followed by a Bogoliubov rotation, so it is exactly solvable — which makes it the ideal benchmark: any numerical or quantum result can be checked against the closed-form answer.

**Hubbard model.**

$$ \hat{H} = -t \sum_{\langle ij \rangle, \sigma} \left(\hat{c}_{i\sigma}^\dagger \hat{c}_{j\sigma} + \hat{c}_{j\sigma}^\dagger \hat{c}_{i\sigma}\right) + U \sum_i \hat{n}_{i\uparrow}\hat{n}_{i\downarrow} - \mu \sum_{i\sigma} \hat{n}_{i\sigma} $$

The minimal model of correlated electrons: kinetic energy $t$ competing with on-site repulsion $U$. At half filling and large $U/t$ it is a Mott insulator; away from half filling in two dimensions its phase diagram is genuinely open. Everything interesting in the model is a fight between the two terms.

**Heisenberg model.**

$$ \hat{H} = J\sum_{\langle ij \rangle} \hat{\mathbf{S}}_i \cdot \hat{\mathbf{S}}_j $$

The large-$U$ effective theory of the Hubbard model, with $J = 4t^2/U$ — per bond, including the constant $-J/4$ of Section 4.2. Spin-only, so no Jordan-Wigner overhead.

### The FeMoco example, honestly

The nitrogenase cofactor FeMoco is the field's standard illustration, so it is worth stating what the published resource estimates actually say. The molecule fixes atmospheric nitrogen at ambient conditions where industrial Haber-Bosch needs several hundred degrees and hundreds of atmospheres, so understanding the mechanism would be genuinely valuable, and its ~76-orbital active space (roughly 152 qubits) is far beyond exact classical diagonalization.

Published fault-tolerant resource estimates for this system have fallen by several orders of magnitude over the past decade, as qubitization, better factorizations of the two-electron tensor and improved magic-state distillation replaced naive Trotterization. The *direction* of that progress is encouraging; the *level* is still millions of physical qubits and days of runtime under standard assumptions, because the non-Clifford gate count is of order $10^{10}$ to $10^{11}$ — at roughly ten microseconds per logical Toffoli that alone is days of wall-clock time — and no error rate achievable without error correction can support it (Chapter 5, Example 5, part C). Treat those exponents as orders of magnitude: the published numbers have moved by several of them and will move again. FeMoco is therefore an excellent argument for building a fault-tolerant quantum computer and a poor argument for expecting near-term chemistry results.

* * *

## 4.6 Implementation: Two Model Hamiltonians, End to End

We now build the qubit Hamiltonians of the transverse-field Ising chain and the two-site Hubbard model, diagonalize them exactly, check them against independent analytic results, and then run a VQE on the same objects. The point of doing all three is that only the comparison is informative: an exact diagonalization tells you the answer, an analytic formula tells you the exact diagonalization is right, and the VQE tells you what a quantum algorithm would have reported.

### The transverse-field Ising chain

The Ising Hamiltonian is already a sum of Pauli strings, so no fermionic mapping is needed — this is the cheapest possible quantum simulation target and the reason it appears in nearly every hardware demonstration.

Code Example 4: The Ising Chain, Diagonalized and Checked

```python
"""Chapter 4, Example 4: transverse-field Ising chain -> qubit Hamiltonian.
Continues from Example 2 (same session)."""


def tfim_hamiltonian(N: int, J: float, h: float, periodic: bool = False) -> dict:
    """H = -J sum_i Z_i Z_{i+1} - h sum_i X_i, as a Pauli-string dictionary."""
    terms = {}
    bonds = list(range(N - 1)) + ([N - 1] if periodic and N > 2 else [])
    for i in bonds:
        j = (i + 1) % N
        s = ''.join('Z' if k in (i, j) else 'I' for k in range(N))
        terms[s] = terms.get(s, 0.0) - J
    for i in range(N):
        s = 'I' * i + 'X' + 'I' * (N - i - 1)
        terms[s] = terms.get(s, 0.0) - h
    return terms


N, J = 4, 1.0
print(f"Transverse-field Ising chain, N = {N}, open boundary, J = {J}")
print("=" * 74)
Hq = tfim_hamiltonian(N, J, 1.0)
print(f"qubit Hamiltonian at h = 1.0  ({len(Hq)} Pauli terms):")
print("  H =", op_str(Hq))

M = to_matrix(Hq)
print(f"\nHermitian: {np.allclose(M, M.conj().T)}")
evals, evecs = np.linalg.eigh(M)
print(f"lowest four eigenvalues: {np.round(evals[:4], 8)}")
print(f"ground-state energy  E0 = {evals[0]:.10f}")
print(f"first excited state  E1 = {evals[1]:.10f}")
print(f"spectral gap            = {evals[1] - evals[0]:.10f}")

print("\nField scan: order parameter, correlation and gap")
print("-" * 74)
print(f"{'h':>6} {'E0':>12} {'E0/N':>10} {'<X_0>':>10} "
      f"{'<Z_0 Z_1>':>11} {'gap':>10}")
for h in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
    evals, evecs = np.linalg.eigh(to_matrix(tfim_hamiltonian(N, J, h)))
    g = evecs[:, 0]
    mx = np.real(np.vdot(g, pauli_matrix('X' + 'I' * (N - 1)) @ g))
    zz = np.real(np.vdot(g, pauli_matrix('ZZ' + 'I' * (N - 2)) @ g))
    print(f"{h:6.2f} {evals[0]:12.6f} {evals[0]/N:10.6f} "
          f"{mx:10.6f} {zz:11.6f} {evals[1]-evals[0]:10.6f}")

print("\nSize dependence of the energy density at h = J = 1 (open chain)")
print("-" * 74)
for N_ in (2, 4, 6, 8, 10):
    e0 = np.linalg.eigvalsh(to_matrix(tfim_hamiltonian(N_, 1.0, 1.0)))[0]
    print(f"  N = {N_:3d}: E0 = {e0:11.6f}   E0/N = {e0/N_:10.6f}   "
          f"Hilbert dim = {2**N_:6d}")

print("\nIndependent check: periodic chain against the free-fermion solution")
print("-" * 74)
for N_ in (4, 6, 8, 10):
    e0 = np.linalg.eigvalsh(to_matrix(
        tfim_hamiltonian(N_, 1.0, 1.0, periodic=True)))[0]
    ks = (2 * np.arange(N_) + 1) * np.pi / N_          # antiperiodic sector
    exact = -np.sum(np.sqrt(1 + 1.0 ** 2 - 2 * 1.0 * np.cos(ks)))
    print(f"  N = {N_:3d}: E0 = {e0:11.6f}   free fermions = {exact:11.6f}"
          f"   difference = {abs(e0-exact):.2e}")
```

```text
Transverse-field Ising chain, N = 4, open boundary, J = 1.0
==========================================================================
qubit Hamiltonian at h = 1.0  (7 Pauli terms):
  H = -1.0000*IIIX  -1.0000*IIXI  -1.0000*IIZZ  -1.0000*IXII  -1.0000*IZZI  -1.0000*XIII  -1.0000*ZZII

Hermitian: True
lowest four eigenvalues: [-4.75877048 -4.06417777 -2.75877048 -2.06417777]
ground-state energy  E0 = -4.7587704831
first excited state  E1 = -4.0641777725
spectral gap            = 0.6945927107

Field scan: order parameter, correlation and gap
--------------------------------------------------------------------------
     h           E0       E0/N      <X_0>   <Z_0 Z_1>        gap
  0.00    -3.000000  -0.750000   0.000000    1.000000   0.000000
  0.25    -3.097889  -0.774472   0.261617    0.957928   0.007325
  0.50    -3.427034  -0.856759   0.545775    0.818662   0.094788
  0.75    -4.005816  -1.001454   0.755422    0.636728   0.335511
  1.00    -4.758770  -1.189693   0.862086    0.494818   0.694593
  1.25    -5.605986  -1.401496   0.913793    0.399018   1.115221
  1.50    -6.503892  -1.625973   0.941298    0.333158   1.566416
  2.00    -8.376799  -2.094200   0.967737    0.250022   2.510954
  3.00   -12.250561  -3.062640   0.985913    0.166677   4.461930
  4.00   -16.187740  -4.046935   0.992125    0.125003   6.439777

Size dependence of the energy density at h = J = 1 (open chain)
--------------------------------------------------------------------------
  N =   2: E0 =   -2.236068   E0/N =  -1.118034   Hilbert dim =      4
  N =   4: E0 =   -4.758770   E0/N =  -1.189693   Hilbert dim =     16
  N =   6: E0 =   -7.296230   E0/N =  -1.216038   Hilbert dim =     64
  N =   8: E0 =   -9.837951   E0/N =  -1.229744   Hilbert dim =    256
  N =  10: E0 =  -12.381490   E0/N =  -1.238149   Hilbert dim =   1024

Independent check: periodic chain against the free-fermion solution
--------------------------------------------------------------------------
  N =   4: E0 =   -5.226252   free fermions =   -5.226252   difference = 1.78e-15
  N =   6: E0 =   -7.727407   free fermions =   -7.727407   difference = 8.88e-16
  N =   8: E0 =  -10.251662   free fermions =  -10.251662   difference = 2.13e-14
  N =  10: E0 =  -12.784906   free fermions =  -12.784906   difference = 1.07e-14
```

**What to notice.** The field scan is a quantum phase transition seen through a four-site window. At $h = 0$ the ground state is a classical ferromagnet: $\langle Z_0 Z_1 \rangle = 1$, $\langle X_0 \rangle = 0$, and the gap vanishes because the two ferromagnetic configurations are exactly degenerate. As $h$ grows the transverse field mixes them, the gap opens, $\langle X_0 \rangle$ rises toward 1 and the $ZZ$ correlation falls. By $h = 4$ the state is nearly the product $\lvert +{+}{+}{+} \rangle$, with $\langle X_0 \rangle = 0.992$ and $\langle Z_0 Z_1 \rangle = 0.125$, which is exactly $J/2h$ — the *first*-order perturbative result. Treating $-J\sum Z_iZ_{i+1}$ as a perturbation on $\lvert{+}{+}{+}{+}\rangle$, the bond term admits an amplitude $J/4h$ onto the state with both spins flipped (energy cost $4h$), giving $\langle Z_0Z_1\rangle \simeq J/2h$. The printed values 0.250, 0.167 and 0.125 at $h = 2, 3, 4$ are $J/2h$ to five digits. The infinite-chain transition sits at $h/J = 1$; on four sites it is smeared into a crossover, exactly as finite-size scaling says it must be.

The energy density converges slowly — $E_0/N$ moves from $-1.118$ at $N=2$ to $-1.238$ at $N=10$, approaching the thermodynamic value $-4/\pi = -1.2732$ from above — a useful reminder that a small quantum simulation of a lattice model gives you a small lattice, not the thermodynamic limit.

The last block is the most important. The periodic-chain energies agree with the closed-form free-fermion expression $E_0 = -\sum_k \sqrt{1 + h^2 - 2h\cos k}$ to $10^{-14}$. Since that formula was derived by an entirely different route (Jordan-Wigner to fermions, then Bogoliubov diagonalization in momentum space), the agreement validates the whole Pauli-string-to-matrix pipeline of Example 2. Before trusting a quantum algorithm's answer, make sure your reference is right.

### The two-site Hubbard model

Now the fermionic case, where the Jordan-Wigner transform does real work. We order the four spin orbitals as (site 0 ↑, site 0 ↓, site 1 ↑, site 1 ↓) and set $\mu = U/2$, the particle-hole symmetric point at which the global ground state is guaranteed to be half filled.

Code Example 5: The Hubbard Dimer via Jordan-Wigner

```python
"""Chapter 4, Example 5: two-site Hubbard model, built by Jordan-Wigner.
Continues from Example 2 (same session)."""

# Spin-orbital ordering (= qubit index):
#   0 = site 0 spin up    1 = site 0 spin down
#   2 = site 1 spin up    3 = site 1 spin down
N_MODES = 4
UP = {0: 0, 1: 2}
DN = {0: 1, 1: 3}


def hubbard_dimer(t: float, U: float, mu: float = None) -> dict:
    """H = -t sum_sigma (c^dag_{0 sigma} c_{1 sigma} + h.c.)
           + U sum_i n_{i up} n_{i dn} - mu sum_p n_p
    With mu = U/2 the model is particle-hole symmetric and the global
    ground state lies in the half-filled sector."""
    n = N_MODES
    if mu is None:
        mu = U / 2.0
    Hq = {}
    for spin in (UP, DN):
        Hq = op_add(Hq, op_scale(jw_hop(spin[0], spin[1], n), -t))
    for site in (0, 1):
        Hq = op_add(Hq, op_scale(op_mul(jw_number(UP[site], n),
                                        jw_number(DN[site], n)), U))
    for p in range(n):
        Hq = op_add(Hq, op_scale(jw_number(p, n), -mu))
    return Hq


def sector_indices(n_up: int, n_dn: int):
    """Basis indices with a given number of up and down electrons."""
    out = []
    for i in range(2 ** N_MODES):
        b = format(i, f'0{N_MODES}b')
        if (int(b[UP[0]]) + int(b[UP[1]]) == n_up
                and int(b[DN[0]]) + int(b[DN[1]]) == n_dn):
            out.append(i)
    return out


t = 1.0
print("Two-site Hubbard model via Jordan-Wigner (4 spin orbitals -> 4 qubits)")
print("=" * 76)

Hq = hubbard_dimer(t, 4.0)
print(f"t = 1, U = 4, mu = U/2:  {len(Hq)} Pauli terms")
print("  H =", op_str(Hq))
M = to_matrix(Hq)
print(f"\nHermitian: {np.allclose(M, M.conj().T)}")
evals = np.linalg.eigvalsh(M)
print(f"full 16-level spectrum:\n  {np.round(evals, 6)}")
print(f"global ground-state energy E0 = {evals[0]:.10f}")

print("\nHalf-filled sector (one up + one down electron), mu = 0")
print("-" * 76)
idx = sector_indices(1, 1)
print(f"  sector dimension: {len(idx)} of {2**N_MODES}")
print(f"{'U':>6} {'E0 (numeric)':>15} {'E0 (analytic)':>15} {'|diff|':>10} "
      f"{'<n_up n_dn>':>12} {'<S_0.S_1>':>11}")
D_tot = to_matrix(op_add(op_mul(jw_number(UP[0], 4), jw_number(DN[0], 4)),
                         op_mul(jw_number(UP[1], 4), jw_number(DN[1], 4))))
Sz0 = (to_matrix(jw_number(UP[0], 4)) - to_matrix(jw_number(DN[0], 4))) / 2
Sz1 = (to_matrix(jw_number(UP[1], 4)) - to_matrix(jw_number(DN[1], 4))) / 2
for U in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
    sub = to_matrix(hubbard_dimer(t, U, mu=0.0))[np.ix_(idx, idx)]
    w, v = np.linalg.eigh(sub)
    g = np.zeros(2 ** N_MODES, dtype=complex)
    g[idx] = v[:, 0]
    analytic = U / 2 - np.sqrt((U / 2) ** 2 + 4 * t ** 2)
    docc = np.real(np.vdot(g, D_tot @ g)) / 2
    spin_corr = 3 * np.real(np.vdot(g, (Sz0 @ Sz1) @ g))   # isotropic singlet
    print(f"{U:6.1f} {w[0]:15.8f} {analytic:15.8f} {abs(w[0]-analytic):10.2e} "
          f"{docc:12.6f} {spin_corr:11.6f}")

print("\nLarge-U limit: the Hubbard dimer becomes a Heisenberg dimer")
print("-" * 76)
print(f"{'U':>8} {'E0':>14} {'-4t^2/U':>12} {'ratio':>10}")
for U in (8.0, 16.0, 32.0, 64.0, 128.0):
    sub = to_matrix(hubbard_dimer(t, U, mu=0.0))[np.ix_(idx, idx)]
    e0 = np.linalg.eigvalsh(sub)[0]
    J_super = -4 * t ** 2 / U
    print(f"{U:8.1f} {e0:14.8f} {J_super:12.8f} {e0/J_super:10.6f}")

print("\nSpin gap and charge gap from sector-resolved diagonalization (mu = 0)")
print("-" * 76)


def particle_sector_energies(U, n_particles):
    Mn = to_matrix(hubbard_dimer(t, U, mu=0.0))
    sel = [i for i in range(2 ** N_MODES) if bin(i).count('1') == n_particles]
    return np.linalg.eigvalsh(Mn[np.ix_(sel, sel)])


print(f"{'U':>6} {'singlet (N=2)':>15} {'triplet (N=2)':>15} "
      f"{'spin gap':>10} {'charge gap':>11}")
for U in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0):
    e2 = particle_sector_energies(U, 2)
    e1 = particle_sector_energies(U, 1)
    e3 = particle_sector_energies(U, 3)
    charge_gap = e1[0] + e3[0] - 2 * e2[0]
    print(f"{U:6.1f} {e2[0]:15.8f} {e2[1]:15.8f} "
          f"{e2[1]-e2[0]:10.6f} {charge_gap:11.6f}")
print("\n  large-U asymptotics for the dimer:"
      " spin gap -> 4t^2/U,  charge gap -> U - 2t")
```

```text
Two-site Hubbard model via Jordan-Wigner (4 spin orbitals -> 4 qubits)
============================================================================
t = 1, U = 4, mu = U/2:  7 Pauli terms
  H = -2.0000*IIII  +1.0000*IIZZ  -0.5000*IXZX  -0.5000*IYZY  -0.5000*XZXI  -0.5000*YZYI  +1.0000*ZZII

Hermitian: True
full 16-level spectrum:
  [-4.828427 -4.       -4.       -4.       -3.       -3.       -3.
 -3.       -1.       -1.       -1.       -1.        0.        0.
  0.        0.828427]
global ground-state energy E0 = -4.8284271247

Half-filled sector (one up + one down electron), mu = 0
----------------------------------------------------------------------------
  sector dimension: 4 of 16
     U    E0 (numeric)   E0 (analytic)     |diff|  <n_up n_dn>   <S_0.S_1>
   0.0     -2.00000000     -2.00000000   0.00e+00     0.250000   -0.375000
   1.0     -1.56155281     -1.56155281   4.44e-16     0.189366   -0.465951
   2.0     -1.23606798     -1.23606798   4.44e-16     0.138197   -0.542705
   4.0     -0.82842712     -0.82842712   2.22e-16     0.073223   -0.640165
   8.0     -0.47213595     -0.47213595   4.44e-16     0.026393   -0.710410
  16.0     -0.24621125     -0.24621125   4.44e-16     0.007464   -0.738803
  32.0     -0.12451550     -0.12451550   1.12e-14     0.001931   -0.747104

Large-U limit: the Hubbard dimer becomes a Heisenberg dimer
----------------------------------------------------------------------------
       U             E0      -4t^2/U      ratio
     8.0    -0.47213595  -0.50000000   0.944272
    16.0    -0.24621125  -0.25000000   0.984845
    32.0    -0.12451550  -0.12500000   0.996124
    64.0    -0.06243908  -0.06250000   0.999025
   128.0    -0.03124237  -0.03125000   0.999756

Spin gap and charge gap from sector-resolved diagonalization (mu = 0)
----------------------------------------------------------------------------
     U   singlet (N=2)   triplet (N=2)   spin gap  charge gap
   0.0     -2.00000000     -0.00000000   2.000000    2.000000
   1.0     -1.56155281     -0.00000000   1.561553    2.123106
   2.0     -1.23606798     -0.00000000   1.236068    2.472136
   4.0     -0.82842712     -0.00000000   0.828427    3.656854
   8.0     -0.47213595     -0.00000000   0.472136    6.944272
  16.0     -0.24621125     -0.00000000   0.246211   14.492423

  large-U asymptotics for the dimer: spin gap -> 4t^2/U,  charge gap -> U - 2t
```

**What to notice.** This output contains a surprising amount of condensed-matter physics for a 16-dimensional matrix.

**The qubit Hamiltonian is remarkably compact, and its structure is legible.** `IIZZ` and `ZZII` are the on-site $U$ terms. Expanding, $\hat{n}_{i\uparrow}\hat{n}_{i\downarrow} = (I - Z_u)(I - Z_d)/4 = (I - Z_u - Z_d + Z_uZ_d)/4$, so each site contributes a constant, two single-$Z$ terms *and* a $ZZ$ term; at the particle-hole symmetric point $\mu = U/2$ the single-$Z$ terms cancel exactly against the chemical potential, which is why only $ZZ$ survives. `XZXI`, `YZYI`, `IXZX`, `IYZY` are the two hopping terms, each carrying one $Z$ from the Jordan-Wigner string because the paired spin orbitals are two apart in our ordering. And `IIII` with coefficient $-2$ is what is left of the two constants: $+U \cdot 2/4 = +2$ from the interaction and $-\mu \cdot 4/2 = -4$ from the chemical potential. A different orbital ordering — grouping by spin instead of by site — would change the strings but not the spectrum. It *would* change the circuit depth, which is why orbital ordering is a real optimization target (Exercise 3).

**The analytic check is exact.** The half-filled ground state of the Hubbard dimer is known in closed form,

$$ E_0 = \frac{U}{2} - \sqrt{\left(\frac{U}{2}\right)^2 + 4t^2} $$

and every numerical value agrees to $10^{-14}$ or better. This is a real test of the Jordan-Wigner implementation, not a tautology: a sign error in the parity string would produce a different spectrum.

**Double occupancy shows Mott localization.** At $U = 0$ the electrons are independent and $\langle \hat{n}_\uparrow \hat{n}_\downarrow \rangle = 0.25$, exactly the uncorrelated value $0.5 \times 0.5$. As $U$ grows it collapses: 0.0732 at $U = 4$, 0.0019 at $U = 32$ — the electrons stop visiting each other's sites. This is the two-site caricature of a Mott insulator, and double occupancy is exactly the observable one would measure on a quantum computer to detect it.

**Superexchange emerges quantitatively.** $\langle \hat{\mathbf{S}}_0 \cdot \hat{\mathbf{S}}_1 \rangle$ runs from $-0.375$ at $U = 0$ to $-0.747$ at $U = 32$, converging on the singlet value $-3/4$; and the ratio $E_0 / (-4t^2/U)$ climbs to 0.99976 at $U/t = 128$. Note which Heisenberg energy that denominator is: the effective bond Hamiltonian is $J(\hat{\mathbf{S}}_0\cdot\hat{\mathbf{S}}_1 - 1/4)$ with $J = 4t^2/U$, whose singlet energy is $-J = -4t^2/U$. Dropping the constant would give $-3J/4 = -3t^2/U$ and a ratio stuck at $4/3$. The famous $J = 4t^2/U$ formula is not a hand-wave — it is the leading term of an expansion we can watch converge, and the same mechanism sets the exchange coupling in the cuprates and in essentially every magnetic insulator.

**The two gaps separate.** The spin gap (singlet-triplet) *closes* as $4t^2/U$ while the charge gap *opens* as $U - 2t$. A large-$U$ Hubbard system is a charge insulator with low-lying spin excitations — which is precisely why an effective spin model is the right description, and why magnetic excitations dominate the low-energy spectroscopy of Mott insulators.

### VQE on the same Hamiltonians

Finally, the quantum algorithm. The code re-lists the Chapter 1-2 mini-simulator (unchanged API, unchanged big-endian convention) so this chapter runs on its own, then builds a hardware-efficient ansatz, computes exact parameter-shift gradients, and runs plain gradient descent.

Code Example 6: VQE Against Exact Diagonalization

```python
"""Chapter 4, Example 6: VQE on both qubit Hamiltonians, against exact diagonalization.
Continues from Examples 2, 4 and 5 (same session).
The first part re-lists the Chapter 1-2 mini simulator so the chapter is
self-contained; the API and the big-endian convention are unchanged."""
import numpy as np

# =====================================================================
# Mini state-vector simulator (Chapters 1-2 API, big-endian:
# qubit 0 = leftmost bit = most significant bit, index = sum_i q_i 2^(n-1-i))
# =====================================================================
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


# =====================================================================
# Hardware-efficient ansatz, energy, parameter-shift gradient, VQE loop
# =====================================================================

def ansatz(theta, n, layers):
    """Ry layer, then `layers` blocks of (CNOT ladder + Ry layer)."""
    psi, k = ket('0' * n), 0
    for q in range(n):
        psi = apply_gate(psi, ry(theta[k]), [q], n)
        k += 1
    for _ in range(layers):
        for q in range(n - 1):
            psi = cnot(psi, q, q + 1, n)
        for q in range(n):
            psi = apply_gate(psi, ry(theta[k]), [q], n)
            k += 1
    return psi


def real_terms(Hq):
    """A Hermitian Pauli-string operator has real coefficients."""
    return {s: complex(c).real for s, c in Hq.items()}


def energy(theta, Hq, n, layers):
    terms = real_terms(Hq)
    psi = ansatz(theta, n, layers)
    return sum(expval(psi, s, terms) for s in terms)


def gradient(theta, Hq, n, layers):
    """Exact parameter-shift rule: dE/dtheta_i = [E(+pi/2) - E(-pi/2)] / 2."""
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += np.pi / 2
        tm[i] -= np.pi / 2
        g[i] = 0.5 * (energy(tp, Hq, n, layers) - energy(tm, Hq, n, layers))
    return g


def vqe(Hq, n, layers=3, steps=1200, lr=0.3, seed=1, log_every=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 0.3, size=n * (layers + 1))
    for it in range(steps):
        if log_every and it % log_every == 0:
            print(f"      iter {it:5d}   E = {energy(theta, Hq, n, layers):+.8f}")
        theta -= lr * gradient(theta, Hq, n, layers)
    return energy(theta, Hq, n, layers), theta


# =====================================================================
print("VQE vs exact diagonalization")
print("=" * 72)

print("\n[1] Transverse-field Ising chain, N = 4, J = h = 1")
H_ising = tfim_hamiltonian(4, 1.0, 1.0)
exact_i = np.linalg.eigvalsh(to_matrix(H_ising))[0]
print(f"    exact   E0 = {exact_i:.8f}")
e_i, th_i = vqe(H_ising, 4, layers=3, steps=1200, lr=0.3, seed=1, log_every=300)
print(f"    VQE     E  = {e_i:.8f}    error = {e_i - exact_i:.3e}")

print("\n[2] Two-site Hubbard model, t = 1, U = 4, mu = U/2")
H_hub = hubbard_dimer(1.0, 4.0)
exact_h = np.linalg.eigvalsh(to_matrix(H_hub))[0]
print(f"    exact   E0 = {exact_h:.8f}")
e_h, th_h = vqe(H_hub, 4, layers=3, steps=600, lr=0.3, seed=3, log_every=150)
print(f"    VQE     E  = {e_h:.8f}    error = {e_h - exact_h:.3e}")

print("\n[3] Does the VQE state reproduce the physics, not just the energy?")
psi = ansatz(th_h, 4, 3)
docc = op_mul(jw_number(0, 4), jw_number(1, 4))
g = np.linalg.eigh(to_matrix(H_hub))[1][:, 0]
docc_r = real_terms(docc)
print(f"    <n_0up n_0dn>   VQE = "
      f"{sum(expval(psi, s, docc_r) for s in docc_r):.6f}")
print(f"    <n_0up n_0dn> exact = "
      f"{np.real(np.vdot(g, to_matrix(docc) @ g)):.6f}")
n_op_r = real_terms(op_add(*[jw_number(p, 4) for p in range(4)]))
print(f"    particle number VQE = "
      f"{sum(expval(psi, s, n_op_r) for s in n_op_r):.6f}"
      f"   (2 at half filling)")
print(f"    state overlap |<VQE|exact>|^2 = {abs(np.vdot(g, psi))**2:.6f}")

print("\n[4] Ansatz depth: how many layers are enough?")
print(f"    {'layers':>7} {'params':>7} {'Ising error':>14} {'Hubbard error':>15}")
for L in (1, 2, 3, 4):
    ei, _ = vqe(H_ising, 4, layers=L, steps=500, lr=0.3, seed=1)
    eh, _ = vqe(H_hub, 4, layers=L, steps=500, lr=0.3, seed=3)
    print(f"    {L:7d} {4*(L+1):7d} {ei-exact_i:14.3e} {eh-exact_h:15.3e}")

print("\n[5] Optimizer restarts: the landscape is not convex")
print(f"    {'seed':>5} {'Ising E':>14} {'error':>12}")
for s in range(6):
    ei, _ = vqe(H_ising, 4, layers=3, steps=500, lr=0.3, seed=s)
    print(f"    {s:5d} {ei:14.8f} {ei-exact_i:12.3e}")

print("\n[6] Measurement cost: exact expectation vs finite sampling")
print("    (20 independent shot budgets per row; <P> = 2 Pr[+1] - 1)")
print(f"    {'shots/term':>11} {'mean E':>12} {'std dev':>10} {'1/sqrt(N)':>11}")
psi_i = ansatz(th_i, 4, 3)
true_p = {s: expval(psi_i, s) for s in H_ising}
rng = np.random.default_rng(0)
for shots in (100, 1_000, 10_000, 100_000):
    ests = []
    for _ in range(20):
        est = 0.0
        for s, c in H_ising.items():
            hits = rng.binomial(shots, (1 + true_p[s]) / 2)
            est += complex(c).real * (2 * hits / shots - 1)
        ests.append(est)
    ests = np.array(ests)
    print(f"    {shots:11,d} {ests.mean():12.6f} {ests.std():10.6f} "
          f"{np.sqrt(len(H_ising)/shots):11.6f}")
```

```text
VQE vs exact diagonalization
========================================================================

[1] Transverse-field Ising chain, N = 4, J = h = 1
    exact   E0 = -4.75877048
      iter     0   E = -2.11673906
      iter   300   E = -4.74508701
      iter   600   E = -4.74696210
      iter   900   E = -4.75090684
    VQE     E  = -4.75479183    error = 3.979e-03

[2] Two-site Hubbard model, t = 1, U = 4, mu = U/2
    exact   E0 = -4.82842712
      iter     0   E = -1.41323792
      iter   150   E = -4.82841059
      iter   300   E = -4.82842704
      iter   450   E = -4.82842712
    VQE     E  = -4.82842712    error = 2.150e-12

[3] Does the VQE state reproduce the physics, not just the energy?
    <n_0up n_0dn>   VQE = 0.073223
    <n_0up n_0dn> exact = 0.073223
    particle number VQE = 2.000000   (2 at half filling)
    state overlap |<VQE|exact>|^2 = 1.000000

[4] Ansatz depth: how many layers are enough?
     layers  params    Ising error   Hubbard error
          1       8      2.466e-02       8.284e-01
          2      12      8.098e-03      -1.776e-15
          3      16      1.260e-02       7.193e-11
          4      20      1.139e-02       9.334e-06

[5] Optimizer restarts: the landscape is not convex
     seed        Ising E        error
        0    -4.75647707    2.293e-03
        1    -4.74616690    1.260e-02
        2    -4.75826562    5.049e-04
        3    -4.70883018    4.994e-02
        4    -4.74486976    1.390e-02
        5    -4.75594391    2.827e-03

[6] Measurement cost: exact expectation vs finite sampling
    (20 independent shot budgets per row; <P> = 2 Pr[+1] - 1)
     shots/term       mean E    std dev   1/sqrt(N)
            100    -4.689000   0.174066    0.264575
          1,000    -4.736500   0.050175    0.083666
         10,000    -4.749890   0.023303    0.026458
        100,000    -4.755493   0.005735    0.008367
```

**What to notice.** This single output separates the three error sources that a real VQE experiment sees mixed together.

**The Hubbard result is essentially exact.** VQE reaches $-4.82842712$ against an exact $-4.82842712$, an error of $2 \times 10^{-12}$. Block [3] shows why we should believe it: the double occupancy matches to six digits, the particle number comes out at exactly 2 even though the ansatz does not conserve it, and the state overlap with the exact ground state is 1.000000. The variational principle did its job, and the physics — not merely the energy — was reproduced.

**The Ising result is not exact, and the reason is the optimizer, not the ansatz.** Block [4] shows the two-layer ansatz reaching $8 \times 10^{-3}$ and the three- and four-layer ansätze doing no better within a fixed step budget. Block [5] shows six random restarts of the same three-layer circuit landing between $5 \times 10^{-4}$ and $5 \times 10^{-2}$ — two orders of magnitude of spread from nothing but the initial parameters. Plain gradient descent on a non-convex landscape is not a reliable minimizer. In practice one uses several restarts and better optimizers, and reports the best value; that is a variational upper bound, so "best" is meaningful.

**More layers can be worse.** The Hubbard column of block [4] is instructive: two layers give $-2 \times 10^{-15}$ (machine precision), three give $7 \times 10^{-11}$, four give $9 \times 10^{-6}$. A more expressive ansatz has a harder landscape — but the mechanism here is mundane, not exotic. With a fixed 500-step budget, 20 parameters simply have not travelled as far as 12 have, and the extra directions add local minima and near-flat valleys for plain gradient descent to stall in. This is *not* a barren plateau: barren plateaus are an exponential-in-$n$ collapse of the gradient variance, and $n = 4$ qubits with 20 parameters is nowhere near that regime (Chapter 3, Section 3.6 measures the real effect on wider circuits). The lesson is that under-convergence masquerades as ansatz error, and only a step-budget scan or a set of restarts tells them apart.

**Measurement noise is the dominant practical error.** Block [6] is the one that should worry a would-be experimentalist. The standard deviation of the energy estimate falls as $1/\sqrt{N_{\text{shots}}}$, as it must, but the constants are unforgiving: 10,000 shots per Pauli term — 70,000 circuit executions for this seven-term Hamiltonian — still leaves a scatter of $0.023$. To reach chemical accuracy ($1.6 \times 10^{-3}$ Hartree) on a Hamiltonian with $10^5$ terms, the shot count becomes the binding constraint. Chapter 5 turns this into wall-clock time, and the answer is measured in years.

* * *

## Exercises

Work through these with the code from this chapter in front of you. Solutions follow each question.

#### Exercise 1: Determinant Counting

(a) A calculation uses an active space of 12 electrons in 12 spatial orbitals. How many determinants does full CI need, and how many qubits would a quantum algorithm use? (b) Adding two orbitals multiplies the determinant count by roughly what factor at this size? (c) Why is the qubit count so much smaller than $\log_2 D$ would suggest for the *whole* basis, yet larger than $\log_2 D$ for the active space?

<details><summary>Solution</summary>
<p>(a) With \(M = 12\) and \(N_\alpha = N_\beta = 6\): \(D = \binom{12}{6}^2 = 924^2 = 853{,}776\) determinants, and \(2M = 24\) qubits. Code Example 1 prints exactly these numbers.</p>
<p>(b) From the growth table, \(12 \to 14\) orbitals multiplies \(D\) by 13.80. The factor grows slowly with \(M\), approaching 16 asymptotically for half filling (each new orbital pair roughly quadruples each binomial).</p>
<p>(c) \(\log_2 853{,}776 = 19.7\), so 24 qubits is slightly more. The qubit register encodes <em>all</em> occupation strings, including those with the wrong particle number and wrong spin — \(2^{24} = 1.7 \times 10^7\) states versus \(8.5 \times 10^5\) physical ones. The factor of 20 is the price of not imposing the symmetries. Symmetry-reduction techniques (parity mapping plus two-qubit reduction, particle-number-preserving ansätze) recover part of it: this is why a four-qubit \(\mathrm{H}_2\) problem can be compressed to two qubits, as in Chapter 3.</p>
</details>

#### Exercise 2: The Parity String Is Not Optional

Modify `jw_annihilate` and `jw_create` in Code Example 2 to omit the $Z$-string (i.e. return only the local raising/lowering operator). (a) Which anticommutation relations now fail? (b) Recompute the half-filled Hubbard dimer ground-state energy at $U = 4$ with the broken mapping. (c) What does this tell you about validating a quantum chemistry pipeline?

<details><summary>Solution</summary>
<p>(a) The same-mode relation \(\{c_p, c_p^\dagger\} = I\) still holds, because the local operators satisfy it. What fails is <em>every</em> cross-mode relation: without the string, operators on different qubits commute rather than anticommute, so \(\{c_0, c_1^\dagger\}\) and \(\{c_0, c_1\}\) become nonzero Pauli strings instead of vanishing. The verification loop in Example 2 reports <code>FAIL</code> and the final <code>all 2 x 4^2 relations verified</code> line prints <code>False</code>.</p>
<p>(b) With the broken mapping the hopping term becomes \((XX + YY)/2\) on the two spin orbitals of the same spin with no \(Z\) between them. The Hamiltonian is still Hermitian and the diagonalization still succeeds; it simply returns the spectrum of a system of <em>hard-core bosons</em> rather than fermions. For the two-site dimer the numbers happen to remain close, which is exactly what makes the bug dangerous — the failure is silent.</p>
<p>(c) Never trust an energy without an independent check. Verify the algebra (the anticommutators), verify against a closed-form solution where one exists (the \(U/2 - \sqrt{(U/2)^2 + 4t^2}\) formula), and verify conserved quantities (particle number, total spin). The three checks in Examples 2, 4 and 5 exist for this reason, and they are cheap compared with the cost of publishing a wrong number.</p>
</details>

#### Exercise 3: Orbital Ordering and Circuit Depth

In Code Example 5 the modes are ordered (site 0 ↑, site 0 ↓, site 1 ↑, site 1 ↓), so hopping terms carry one $Z$. (a) Re-derive the qubit Hamiltonian with the ordering (site 0 ↑, site 1 ↑, site 0 ↓, site 1 ↓). (b) Compare the maximum Pauli weight of the hopping and interaction terms in the two orderings. (c) For a linear chain of $L$ sites, which ordering minimizes the total weight?

<details><summary>Solution</summary>
<p>(a) With spin-grouped ordering, up-spin hopping connects modes 0 and 1 — adjacent — so it becomes \((XX + YY)/2\) with weight 2. Down-spin hopping connects modes 2 and 3, also adjacent, again weight 2. The interaction \(n_{0\uparrow} n_{0\downarrow}\) now connects modes 0 and 2, which are two apart; but since \(n_p = (I - Z_p)/2\) contains only \(Z\) and \(I\), no string is needed and the term is still \(ZZ\) with weight 2. So this ordering gives maximum weight 2 everywhere.</p>
<p>(b) Site-grouped: hopping weight 3, interaction weight 2. Spin-grouped: hopping weight 2, interaction weight 2. The spin-grouped ordering is better for the dimer.</p>
<p>(c) For an \(L\)-site chain, spin-grouped ordering makes all nearest-neighbour hoppings adjacent (weight 2) but makes the on-site interaction span \(L\) modes — still weight 2, because number operators need no string. So spin-grouped ordering is optimal for the 1D Hubbard model, giving \(O(1)\) weight for every term and total weight \(O(L)\). Site-grouped ordering gives weight 3 for every hopping, also \(O(L)\) total, so both are acceptable in 1D. The orderings diverge in 2D, where no ordering makes all neighbours adjacent and hopping weights grow as \(O(\sqrt{L})\) at best — one of the reasons the 2D Hubbard model is the hard and interesting case.</p>
</details>

#### Exercise 4: Trotter Budget for a Real Calculation

Using the fitted constant $C = 3.3653$ from Code Example 3: (a) how many first-order Trotter steps are needed for a unitary error of $10^{-4}$ at $\tau = 1$? (b) How many with the second-order formula, whose constant is 2.169? (c) If the target is instead $\tau = 10$, how does each answer change, and what does that imply about long-time dynamics?

<details><summary>Solution</summary>
<p>(a) First order: \(r = C/\varepsilon = 3.3653/10^{-4} = 33{,}653\) steps, i.e. about \(3.4 \times 10^5\) Pauli rotations for the ten non-identity terms.</p>
<p>(b) Second order: \(r = \sqrt{C_2/\varepsilon} = \sqrt{2.169/10^{-4}} = 147\) steps. Each step costs twice as many rotations (forward and backward sweep), so about 2,950 rotations — a factor of 115 cheaper. This is why nobody uses first-order Trotter in practice.</p>
<p>(c) The error constants scale with the evolution time: roughly \(C \propto \tau^2\) for first order and \(\tau^3\) for second order (the \(O(t^2/r)\) and \(O(t^3/r^2)\) forms). For \(\tau = 10\), first order needs \(r \sim 100 \times 33{,}653 = 3.4 \times 10^6\) steps and second order \(r \sim \sqrt{1000} \times 147 = 4{,}650\). Long-time dynamics is expensive in both cases; the standard remedy is to keep \(\tau\) short and repeat, which is exactly what phase estimation does — and it is why phase-estimation depth is the binding constraint of fault-tolerant chemistry.</p>
</details>

#### Exercise 5: Why Four Sites Cannot Show a Phase Transition

Using Code Example 4: (a) at what value of $h/J$ does $\langle X_0 \rangle$ reach 1/2 for $N = 4$? (b) Why can a four-site calculation never show a true phase transition, and what does that imply about hardware demonstrations on a handful of qubits?

<details><summary>Solution</summary>
<p>(a) Interpolating the printed field scan between \(h = 0.25\) (\(\langle X_0 \rangle = 0.2616\)) and \(h = 0.5\) (0.5458) puts the crossing near \(h/J \approx 0.46\) — well below the infinite-chain transition at \(h/J = 1\), which is the expected finite-size shift for an open chain with only three bonds.</p>
<p>(b) A phase transition is a non-analyticity of the free energy, and the free energy of a finite system is analytic in its parameters — the partition function is a finite sum of exponentials, hence entire. Non-analyticity can appear only in the \(N \to \infty\) limit. Finite systems show crossovers whose sharpness grows with \(N\), and extracting the transition point requires finite-size scaling over several \(N\). This is why a demonstration on a handful of qubits is a benchmark of the hardware, not a discovery about the model.</p>
</details>

#### Exercise 6: Which Error Dominates?

A VQE run on the Hubbard dimer reports $E = -4.81$ against the exact $-4.828427$. Using Code Example 6, decide which of the three error sources — ansatz expressivity, optimizer convergence, measurement statistics — is responsible, and describe the diagnostic for each.

<details><summary>Solution</summary>
<p>The discrepancy is \(1.8 \times 10^{-2}\). Diagnostics, in order of cost:</p>
<p><strong>Ansatz.</strong> Compute the energy of the exact ground state projected onto the ansatz manifold — or more practically, run the optimizer from many starts with a very large step budget and see whether the best value plateaus above the exact answer. Block [4] does this: at two layers the Hubbard error is \(10^{-15}\), so the ansatz is <em>not</em> the limitation for this model. If it were, no amount of optimization or sampling would help.</p>
<p><strong>Optimizer.</strong> Restart from different seeds (block [5]). A spread of results with the same ansatz and exact expectation values proves the optimizer is the problem. For the Hubbard dimer the run converges to \(2 \times 10^{-12}\) in 600 steps, so \(1.8 \times 10^{-2}\) would indicate a badly chosen learning rate or too few iterations.</p>
<p><strong>Measurement.</strong> Repeat the <em>same</em> optimized parameters many times and look at the scatter (block [6]). Statistical error is symmetric about the true value and shrinks as \(1/\sqrt{N}\); ansatz error is always positive and does not shrink at all. From block [6], a scatter of \(1.8 \times 10^{-2}\) corresponds to roughly \(10^4\) shots per term.</p>
<p>The decisive signature: measurement error fluctuates run to run, optimizer error changes with the seed, ansatz error is reproducible and one-sided. In this case \(-4.81\) is above the exact value and would need all three tests to attribute; the fastest discriminator is to rerun with a different seed.</p>
</details>

* * *

## Summary

### Key Takeaways

**1. The exact problem grows combinatorially, and the wall is close**

  * Full CI needs $\binom{M}{N_\alpha}\binom{M}{N_\beta}$ determinants; each added orbital pair multiplies this by about 14 near $M = 20$.
  * A 20-orbital active space is half a terabyte per wavefunction vector; 24 orbitals is a hundred terabytes.
  * A quantum computer needs $2M$ qubits — cheap by comparison. Qubit count is the least of the three resource constraints.

**2. DFT fails where the interesting materials are**

  * The failure mode is static correlation: several determinants with comparable weight — transition-metal oxides, Mott insulators, bond dissociation, catalytic spin-state ordering.
  * The realistic target is not replacing DFT but supplying the strongly correlated active space it cannot handle.
  * The competition is DMRG, quantum Monte Carlo and tensor networks, not full CI. Quantum advantage requires beating all of them at once.

**3. Second quantization plus Jordan-Wigner is the bridge to qubits**

  * The fermionic Hamiltonian $\sum h_{pq}\hat{c}_p^\dagger \hat{c}_q + \frac{1}{2}\sum h_{pqrs}\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s$ has $O(M^4)$ terms.
  * Jordan-Wigner: $\hat{c}_p = (\prod_{q<p} Z_q)(X_p + iY_p)/2$, $\hat{n}_p = (I - Z_p)/2$. Occupation stays local; hopping weight grows linearly with orbital distance.
  * Bravyi-Kitaev and tree-based mappings reduce the weight to $O(\log M)$ and matter beyond a few dozen orbitals.
  * Always verify the anticommutation relations numerically. A missing parity string is a silent bug.

**4. Digital simulation costs depth, and the cost is measurable**

  * First-order Trotter error $\times r \to 3.366$ and second-order error $\times r^2 \to 2.169$ for the Hubbard dimer: the asymptotic scalings confirmed, with constants.
  * Reaching $10^{-6}$ unitary error for a single time step of a four-qubit model needs $\sim 3 \times 10^7$ Pauli rotations at first order.
  * QPE gives eigenvalues with depth $O(1/\varepsilon)$ and needs error correction; VQE gives a variational upper bound at shallow depth and pays in measurements. The two belong to two different machines.

**5. Model Hamiltonians are where the physics is legible**

  * Transverse-field Ising: exactly solvable, ideal benchmark; our periodic-chain energies match the free-fermion formula to $10^{-14}$.
  * Hubbard dimer: seven Pauli strings; ground-state energy matches $U/2 - \sqrt{(U/2)^2 + 4t^2}$ to machine precision.
  * Double occupancy falls from 0.25 to 0.0019 as $U/t$ goes from 0 to 32 — Mott localization on two sites.
  * $\langle \mathbf{S}_0 \cdot \mathbf{S}_1 \rangle \to -3/4$ and $E_0 \to -4t^2/U$ with ratio 0.99976 at $U/t = 128$: superexchange emerging quantitatively.
  * Spin gap closes as $4t^2/U$ while the charge gap opens as $U - 2t$: the Mott insulator's two-scale structure.

**6. VQE has three separable error sources, and only one shrinks with effort**

  * Ansatz error is one-sided and reproducible; more layers can make convergence worse, not better.
  * Optimizer error varies with the random seed: six restarts spanned $5\times10^{-4}$ to $5\times10^{-2}$ on the same circuit.
  * Measurement error falls as $1/\sqrt{N_{\text{shots}}}$; $10^4$ shots per term left a scatter of 0.023 on a seven-term Hamiltonian.
  * Checking observables (double occupancy, particle number, state overlap) and not only the energy is what distinguishes a correct calculation from a lucky one.

**Practical implications**

  * Always compute an exact or analytic reference for the system you test on; without it a quantum result is uninterpretable.
  * Choose orbital ordering deliberately: it changes circuit depth without changing the spectrum.
  * Budget depth, width and measurements separately; the binding constraint is almost never the qubit count.
  * Treat a variational energy as an upper bound and report the diagnostics (restarts, observables, shot counts) alongside it.
  * Be precise about which era a claim belongs to. "A quantum computer could solve FeMoco" is a statement about fault-tolerant machines, and saying so is the difference between a research programme and a press release.

### Where This Leads

We now have the full chain from electrons to Pauli strings, two model Hamiltonians we can diagonalize exactly, and a VQE that reproduces them. Every number so far assumed a perfect quantum computer. Chapter 5 removes that assumption. We will build a noise model on the same state-vector simulator — depolarizing channels realized by the trajectory method — and measure how quickly fidelity decays with circuit depth at realistic error rates. Then we will apply zero-noise extrapolation to the very VQE state computed above and see how much of the noise-induced bias can be recovered, and at what cost in samples. Finally we will put the three budgets of this chapter — depth, width, measurements — next to what error correction demands, and state plainly what near-term quantum computing can and cannot do for materials research.

[← Chapter 3: Variational Quantum Eigensolver](<chapter-3.html>) [Chapter 5: NISQ Reality and Outlook →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * Model parameters, active-space sizes and resource estimates quoted here are representative literature-scale values for teaching purposes; verify against primary sources before using them in a proposal or publication.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
