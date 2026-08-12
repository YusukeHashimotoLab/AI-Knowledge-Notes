---
title: "Chapter 3: The Variational Quantum Eigensolver"
chapter_title: "Chapter 3: The Variational Quantum Eigensolver"
subtitle: ⚛️ Ansätze, Pauli Measurement, Classical Optimisation, and the Ground State of H₂
reading_time: 40-45 minutes
difficulty: Advanced
code_examples: 9
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-computing-introduction/chapter-3.html>) | Last sync: 2026-08-12

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to Quantum Computing](<index.html>) > Chapter 3

This is the chapter where the machinery starts computing chemistry. We will take the hydrogen molecule in a minimal basis, write its electronic Hamiltonian as a sum of six Pauli strings on two qubits, build a one-parameter circuit that prepares a trial wave function, measure the energy on that circuit, and hand the number to a classical optimiser. The result is the **variational quantum eigensolver** (VQE), proposed by Peruzzo and co-workers in 2014 and still the dominant algorithm for quantum chemistry on noisy hardware.

Then we check it. The whole computation is small enough that the same Hamiltonian can be diagonalised exactly, so every VQE energy has a reference value to be compared against — and we compare across the entire dissociation curve, not only at equilibrium. The agreement in Section 3.5 is at the \\(10^{-15}\\) Hartree level, which is a statement about the algorithm rather than about chemistry: with an exact ansatz, noiseless simulation and a converged optimiser, VQE returns the exact ground state of the Hamiltonian you gave it. Every deviation you will ever see on real hardware comes from relaxing one of those three conditions, and Section 3.6 examines each.

A word on what this calculation does and does not demonstrate. Two qubits are simulable on a wristwatch; nothing here is beyond classical reach, and the STO-3G energies we reproduce are 40 mHartree away from the true non-relativistic ground state of H₂ because the *basis set* is crude, not because the algorithm is. What the exercise establishes is that the pipeline is correct end to end — orbital integrals to qubit Hamiltonian to circuit to optimiser to energy — because every link is checked against an independent reference. That is the only honest way to approach a method whose eventual targets cannot be checked at all.

## Learning Objectives

After completing this chapter, you will be able to:

  * State the variational principle and explain why it makes a noisy quantum computer useful for finding ground-state energies
  * Distinguish chemistry-inspired ansätze from hardware-efficient ones, and quantify the difference in terms of the state-space region each explores
  * Compile a Pauli exponential \\(\exp(-i\theta X_0 Y_1)\\) into CNOTs and a single \\(R_z\\), and verify that the resulting circuit conserves particle number and spin
  * Express a molecular electronic Hamiltonian as a weighted sum of Pauli strings and evaluate its expectation value with `expval`
  * Group commuting Pauli terms into measurement settings and estimate the shot cost of reaching a given energy precision
  * Run a VQE optimisation with gradient-free and gradient-based methods, and derive the parameter-shift rule that gives exact analytic gradients from two circuit evaluations
  * Reproduce the H₂ dissociation curve and verify it against exact diagonalisation and against published STO-3G reference values
  * Explain, with numbers, why barren plateaus and measurement cost — not qubit count — are the binding constraints on scaling VQE up

* * *

## 3.1 Why Variational Methods

### The constraint that shapes everything

There is a textbook quantum algorithm for finding energies, and it is not VQE. **Quantum phase estimation** (QPE, Chapter 4) extracts an eigenvalue of \\(H\\) to \\(m\\) bits of precision using a circuit of depth \\(O(2^m)\\) applications of a controlled time evolution. It is efficient in the complexity-theory sense, and it is completely out of reach today: for a molecule of chemical interest the circuit needs millions of gates executed coherently, while current devices lose coherence after hundreds.

VQE is the response to that constraint. Instead of one deep circuit, it uses **many shallow circuits** and moves the hard part — the search — onto a classical computer:

  1. Prepare \\(\lvert \psi(\boldsymbol{\theta}) \rangle\\) with a shallow parameterised circuit.
  2. Measure \\(E(\boldsymbol{\theta}) = \langle \psi(\boldsymbol{\theta}) \rvert H \lvert \psi(\boldsymbol{\theta}) \rangle\\).
  3. Let a classical optimiser propose a better \\(\boldsymbol{\theta}\\).
  4. Repeat.

The circuit depth is set by the ansatz, not by the precision demanded, which is why VQE runs on hardware that exists. The price is paid in the number of circuit repetitions, and Section 3.6 shows that this price is severe.

### The variational principle

The mathematical guarantee comes from a fact you have already met in the [Introduction to Quantum Mechanics](<../quantum-mechanics/index.html>) course. For any Hamiltonian \\(H\\) with ground-state energy \\(E_0\\) and any normalised state \\(\lvert \psi \rangle\\),

\\[ \langle \psi \rvert H \lvert \psi \rangle \geq E_0 \\]

with equality if and only if \\(\lvert \psi \rangle\\) is a ground state. The proof is one line: expand \\(\lvert \psi \rangle = \sum_k c_k \lvert \phi_k \rangle\\) in the eigenbasis of \\(H\\), then

\\[ \langle \psi \rvert H \lvert \psi \rangle = \sum_k \lvert c_k \rvert^2 E_k \geq E_0 \sum_k \lvert c_k \rvert^2 = E_0 \\]

Two consequences make this the right foundation for a noisy machine.

**Every answer is an upper bound.** A trial energy can be wrong, but it cannot be *too low*. If two ansätze give \\(-1.1361\\) and \\(-1.1373\\) Hartree, the second is unambiguously closer to the truth without knowing the truth. Compare this with a method that returns a number of unknown sign of error, where improvement is unverifiable.

**The error is second order in the state error.** If \\(\lvert \psi \rangle\\) deviates from the true ground state by a small amplitude \\(\varepsilon\\) in the excited states, the energy error is \\(O(\varepsilon^2)\\). A trial state with 99% overlap gives an energy error of order 1%, not 10%. This quadratic suppression is exactly why variational methods have dominated electronic-structure theory for a century, and it is what makes VQE tolerant of imperfect state preparation.

**One caution that is often glossed over.** The bound \\(E(\boldsymbol{\theta}) \geq E_0\\) holds for the *exact* expectation value of a *pure* state. On hardware, decoherence makes the state mixed and readout errors bias the estimator, so measured VQE energies routinely fall *below* the true ground state. When you see a reported VQE energy under the exact value, that is not an achievement; it is a diagnostic of uncorrected bias.

### What must be true for VQE to work

Requirement | What it means | What goes wrong if it fails
---|---|---
Expressive ansatz | The circuit can reach a state close to the ground state | Systematic error that no optimiser can remove
Shallow ansatz | Depth within the coherence time | Noise swamps the signal
Efficient measurement | Few Pauli groups, tolerable variance | Shot count explodes
Trainable landscape | Gradients large enough to detect | Barren plateau: optimisation stalls
Good initial guess | Start near the solution | Local minima, slow convergence

The tension between the first two rows is the central design problem of the field. Everything in Section 3.2 is an attempt to buy expressiveness cheaply, using physics rather than depth.

* * *

## 3.2 Parameterised Circuits and Ansätze

### Two philosophies

An **ansatz** is a map from parameters to states, \\(\boldsymbol{\theta} \mapsto \lvert \psi(\boldsymbol{\theta}) \rangle\\), realised as a circuit. There are two ways to choose one.

**Hardware-efficient ansätze** use whatever gates the device performs well: layers of single-qubit rotations interleaved with the native entangling gate, repeated \\(L\\) times. They are shallow and hardware-friendly, and they know nothing about the problem. Their parameters explore the whole Hilbert space, including the overwhelming majority of it that has the wrong number of electrons or the wrong spin.

**Chemistry-inspired ansätze** are built from the physics. The canonical example is **unitary coupled cluster** (UCC), which acts on the Hartree-Fock reference with an exponentiated excitation operator:

\\[ \lvert \psi(\boldsymbol{\theta}) \rangle = e^{T(\boldsymbol{\theta}) - T^\dagger(\boldsymbol{\theta})} \lvert \Phi_{\mathrm{HF}} \rangle, \qquad T = \sum_{ia} \theta_{ia} a_a^\dagger a_i + \sum_{ijab} \theta_{ijab} a_a^\dagger a_b^\dagger a_i a_j + \cdots \\]

Truncating at double excitations gives UCCSD, the workhorse of the field. After the Jordan-Wigner transformation of Chapter 4 each excitation term becomes a Pauli string, and each Pauli string becomes a circuit by the compilation identity of Chapter 2. The parameter count grows as \\(O(N^2 M^2)\\) for \\(N\\) occupied and \\(M\\) virtual orbitals — polynomial, but with a large prefactor, and the circuit depth is the real obstacle.

Ansatz family | Parameters | Depth | Respects symmetry | Where it fails
---|---|---|---|---
Hardware-efficient | \\(O(nL)\\) | \\(O(L)\\) | No | Barren plateaus, unphysical states
UCCSD | \\(O(N^2M^2)\\) | Large | Yes | Too deep for current hardware
k-UpCCGSD | \\(O(kn^2)\\) | Moderate | Yes | Accuracy depends on \\(k\\)
Symmetry-preserving | Problem-dependent | Moderate | Yes | Must be designed per system
ADAPT-VQE | Grown adaptively | Minimal for target accuracy | Yes | Many extra measurements

### The H₂ ansatz, and why one parameter is enough

For our two-qubit H₂ Hamiltonian the entire UCCSD hierarchy collapses to a single double excitation: promote both electrons from the bonding orbital \\(\sigma_g\\) to the antibonding orbital \\(\sigma_u\\). The generator, after the fermion-to-qubit mapping, is the Pauli string \\(X_0 Y_1\\), and the ansatz is

\\[ \lvert \psi(\theta) \rangle = \exp\left(-i\theta X_0 Y_1\right) \lvert \Phi_{\mathrm{HF}} \rangle, \qquad \lvert \Phi_{\mathrm{HF}} \rangle = \lvert 10 \rangle \\]

In our big-endian convention, qubit 0 records whether \\(\sigma_g\\) is doubly occupied and qubit 1 whether \\(\sigma_u\\) is, so the Hartree-Fock state is \\(\lvert 10 \rangle\\) — index 2 of the state vector. (Papers that write this state as \\(\lvert 01 \rangle\\) are using the opposite bit order; the physics is identical.)

Acting on \\(\lvert 10 \rangle\\), the exponential produces a two-term superposition. Since \\(X_0 Y_1 \lvert 10 \rangle = i \lvert 01 \rangle\\) and \\(\left(X_0Y_1\right)^2 = I\\),

\\[ \lvert \psi(\theta) \rangle = \cos\theta \, \lvert 10 \rangle + \sin\theta \, \lvert 01 \rangle \\]

a rotation inside the two-dimensional space spanned by the Hartree-Fock configuration and the doubly excited one. That space is exactly where the true ground state lives, for a reason worth stating: the Hamiltonian commutes with particle number and with total spin, so it cannot connect \\(\lvert 10 \rangle\\) to the \\(\lvert 00 \rangle\\) (no electrons) or \\(\lvert 11 \rangle\\) (four electrons) sectors. **The ansatz is exact because it spans the whole symmetry sector the ground state lives in** — and here that sector is only two-dimensional, so smallness is doing real work too. What transfers to larger molecules is the symmetry half of the argument: build the symmetry into the circuit and the optimiser never has to discover it. What does not transfer is the exactness — for a genuine active space the particle-number and spin sectors are still exponentially large, and a one-parameter circuit spans a vanishing fraction of them.

### Compiling the ansatz into gates

The circuit follows the recipe verified in Chapter 2. Conjugate with basis changes that turn \\(X\\) and \\(Y\\) into \\(Z\\), apply the \\(ZZ\\) exponential as a CNOT ladder around one \\(R_z\\), undo the basis changes:

\\[ \exp(-i\theta X_0 Y_1) = W^\dagger \left[\mathrm{CNOT}\_{0\to1} \left(I \otimes R_z(2\theta)\right) \mathrm{CNOT}\_{0\to1}\right] W, \qquad W = H \otimes (H S^\dagger) \\]

Total cost: one \\(X\\) for state preparation, four single-qubit gates for the basis changes, two CNOTs, one parameterised \\(R_z\\). Eight gates, one parameter, depth 6. This is a circuit that runs on any device ever built.

Code Example 1: The Mini-Simulator (identical to Chapter 2)

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

Code Example 2: The Ansatz Circuit and What Symmetry Buys

```python
import numpy as np
from qcsim import *

Sdg = S.conj().T

def hf_state():
    """Hartree-Fock reference |10>: the bonding orbital sigma_g is doubly occupied."""
    return apply_gate(ket('00'), X, [0], 2)

def ansatz(theta):
    """exp(-i theta X0 Y1)|HF>, compiled into H, S, CNOT and Rz."""
    psi = hf_state()
    psi = apply_gate(psi, H, [0], 2)        # rotate X -> Z on qubit 0
    psi = apply_gate(psi, H @ Sdg, [1], 2)  # rotate Y -> Z on qubit 1
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, rz(2 * theta), [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, S @ H, [1], 2)    # undo the basis change
    psi = apply_gate(psi, H, [0], 2)
    return psi

def hardware_efficient(params):
    """A generic two-qubit ansatz: Ry layer, CNOT, Ry layer (4 parameters)."""
    psi = ket('00')
    psi = apply_gate(psi, ry(params[0]), [0], 2)
    psi = apply_gate(psi, ry(params[1]), [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, ry(params[2]), [0], 2)
    psi = apply_gate(psi, ry(params[3]), [1], 2)
    return psi

print("The Hartree-Fock reference state")
print("-" * 70)
print(f"  |HF> amplitudes = {np.round(hf_state().real, 4)}   (index 2 = |10>)")
print(f"  <ZI> = {expval(hf_state(), 'ZI'):+.1f} (qubit 0 occupied), "
      f"<IZ> = {expval(hf_state(), 'IZ'):+.1f} (qubit 1 empty)")

print("\nThe compiled circuit reproduces cos(theta)|10> + sin(theta)|01>")
print("-" * 70)
print(f"  {'theta':>8} {'amp|00>':>9} {'amp|01>':>9} {'amp|10>':>9} {'amp|11>':>9} "
      f"{'sin(th)':>9} {'cos(th)':>9}")
for th in [0.0, 0.2, -0.35, 1.0, np.pi / 2]:
    psi = ansatz(th)
    print(f"  {th:8.4f} {psi[0].real:9.6f} {psi[1].real:9.6f} {psi[2].real:9.6f} "
          f"{psi[3].real:9.6f} {np.sin(th):9.6f} {np.cos(th):9.6f}")
    assert abs(psi[0]) < 1e-12 and abs(psi[3]) < 1e-12
print("\n  |00> and |11> stay exactly empty: the ansatz never leaves the")
print("  two-electron, spin-singlet sector. Symmetry is built into the circuit,")
print("  not imposed on the optimizer.")

print("\nGate count of the chemistry-inspired ansatz")
print("-" * 70)
print("  1 X (state preparation) + 4 single-qubit basis changes + 2 CNOT + 1 Rz")
print("  parameters: 1")

print("\nA generic hardware-efficient ansatz for comparison")
print("-" * 70)
rng = np.random.default_rng(0)
print(f"  {'trial':>5} {'amp|00>':>9} {'amp|01>':>9} {'amp|10>':>9} {'amp|11>':>9} "
      f"{'in singlet sector?':>19}")
for trial in range(4):
    p = rng.uniform(0, 2 * np.pi, 4)
    psi = hardware_efficient(p)
    leak = abs(psi[0]) ** 2 + abs(psi[3]) ** 2
    print(f"  {trial:5d} {psi[0].real:9.6f} {psi[1].real:9.6f} {psi[2].real:9.6f} "
          f"{psi[3].real:9.6f} {'no, leak %.3f' % leak:>19}")
print("  parameters: 4, and generic values put weight on |00> and |11>, which")
print("  correspond to the wrong number of electrons.")

print("\nHow much of the two-qubit state space does each ansatz reach?")
print("-" * 70)
def reachable_rank(sampler, npar, samples=4000, seed=1):
    rng = np.random.default_rng(seed)
    states = np.array([sampler(rng.uniform(-np.pi, np.pi, npar)) for _ in range(samples)])
    sv = np.linalg.svd(states, compute_uv=False)
    return sv / sv[0]

sv_chem = reachable_rank(lambda p: ansatz(p[0]), 1)
sv_hw = reachable_rank(hardware_efficient, 4)
print(f"  chemistry ansatz, normalised singular values : "
      f"{np.round(sv_chem, 6).tolist()}")
print(f"  hardware-efficient ansatz                    : "
      f"{np.round(sv_hw, 6).tolist()}")
print("  the chemistry ansatz spans a 2-dimensional subspace (|10>, |01>) and")
print("  nothing else; the generic ansatz spans all four dimensions and therefore")
print("  wastes its parameters exploring unphysical states.")
```

```text
The Hartree-Fock reference state
----------------------------------------------------------------------
  |HF> amplitudes = [0. 0. 1. 0.]   (index 2 = |10>)
  <ZI> = -1.0 (qubit 0 occupied), <IZ> = +1.0 (qubit 1 empty)

The compiled circuit reproduces cos(theta)|10> + sin(theta)|01>
----------------------------------------------------------------------
     theta   amp|00>   amp|01>   amp|10>   amp|11>   sin(th)   cos(th)
    0.0000  0.000000  0.000000  1.000000  0.000000  0.000000  1.000000
    0.2000 -0.000000  0.198669  0.980067 -0.000000  0.198669  0.980067
   -0.3500  0.000000 -0.342898  0.939373  0.000000 -0.342898  0.939373
    1.0000  0.000000  0.841471  0.540302 -0.000000  0.841471  0.540302
    1.5708 -0.000000  1.000000  0.000000  0.000000  1.000000  0.000000

  |00> and |11> stay exactly empty: the ansatz never leaves the
  two-electron, spin-singlet sector. Symmetry is built into the circuit,
  not imposed on the optimizer.

Gate count of the chemistry-inspired ansatz
----------------------------------------------------------------------
  1 X (state preparation) + 4 single-qubit basis changes + 2 CNOT + 1 Rz
  parameters: 1

A generic hardware-efficient ansatz for comparison
----------------------------------------------------------------------
  trial   amp|00>   amp|01>   amp|10>   amp|11>  in singlet sector?
      0 -0.340646 -0.405554  0.610523  0.588852      no, leak 0.463
      1 -0.166294 -0.685427 -0.438994  0.556615      no, leak 0.337
      2 -0.226494  0.551667 -0.101712  0.796253      no, leak 0.685
      3  0.611648  0.070399 -0.412223 -0.671567      no, leak 0.825
  parameters: 4, and generic values put weight on |00> and |11>, which
  correspond to the wrong number of electrons.

How much of the two-qubit state space does each ansatz reach?
----------------------------------------------------------------------
  chemistry ansatz, normalised singular values : [1.0, 0.991608, 0.0, 0.0]
  hardware-efficient ansatz                    : [1.0, 0.998371, 0.995819, 0.969004]
  the chemistry ansatz spans a 2-dimensional subspace (|10>, |01>) and
  nothing else; the generic ansatz spans all four dimensions and therefore
  wastes its parameters exploring unphysical states.
```

**What to notice.** The compiled eight-gate circuit reproduces \\(\cos\theta \lvert 10 \rangle + \sin\theta \lvert 01 \rangle\\) to six digits with the other two amplitudes at \\(10^{-17}\\): symmetry conservation is exact, not approximate, because it follows from the structure of the generator rather than from tuning.

The last block quantifies the difference between the two philosophies. Sampling 4000 random parameter sets and taking the singular values of the resulting state matrix reveals the dimension of the reachable set: **two** for the chemistry ansatz (two nonzero singular values, two exactly zero) and **four** for the generic one. The hardware-efficient circuit has four times the parameters and explores twice the dimensions — half of which are physically meaningless for this problem. On two qubits that waste is affordable. At \\(n\\) qubits the physical sector is an exponentially small fraction of the whole space, and the waste becomes the barren plateau of Section 3.6.

* * *

## 3.3 The Hamiltonian as Pauli Strings, and How to Measure It

### The two-qubit hydrogen molecule

The electronic Hamiltonian of a molecule in second quantisation is a sum of one- and two-electron terms over spin orbitals. Chapter 4 derives the machinery — second quantisation, the Jordan-Wigner transformation — that converts it into qubit operators. Here we take the result for H₂ in the STO-3G minimal basis, reduced to two qubits by exploiting particle-number and spin symmetry, in the standard form used by O'Malley and co-workers (*Phys. Rev. X* **6**, 031007, 2016):

\\[ H = g_0\, II + g_1\, Z_0 + g_2\, Z_1 + g_3\, Z_0 Z_1 + g_4\, Y_0 Y_1 + g_5\, X_0 X_1 \\]

with \\(g_4 = g_5\\) by symmetry, and all six coefficients depending on the internuclear distance \\(R\\). Written as a matrix in the basis \\(\lbrace \lvert 00 \rangle, \lvert 01 \rangle, \lvert 10 \rangle, \lvert 11 \rangle \rbrace\\) it is block diagonal: \\(\lvert 00 \rangle\\) (no electrons) and \\(\lvert 11 \rangle\\) (four electrons) are isolated, while \\(\lvert 01 \rangle\\) and \\(\lvert 10 \rangle\\) form a \\(2 \times 2\\) block whose lower eigenvalue is the molecular ground-state energy. The nuclear repulsion is already included, so eigenvalues are total energies in Hartree.

### The coefficient table

The coefficients below were computed from the STO-3G integrals by the standard reduction (the calculation is reproduced in full as Code Example 9, so nothing here is a magic number). They reproduce the published STO-3G reference values: equilibrium at \\(R = 0.735\\) Å with \\(E = -1.137306\\) Hartree, Hartree-Fock energy \\(-1.116999\\) Hartree at that geometry, and a dissociation limit approaching \\(2 \times E(\mathrm{H}) = -0.933164\\) Hartree.

\\(R\\) (Å) | \\(g_0\\) | \\(g_1\\) | \\(g_2\\) | \\(g_3\\) | \\(g_4 = g_5\\) | \\(E_{\mathrm{HF}}\\) | \\(E_{\mathrm{exact}}\\)
---|---|---|---|---|---|---|---
0.300 | 1.684963 | 0.517383 | -1.099915 | 0.661493 | 0.080409 | -0.593828 | -0.601804
0.400 | 1.116353 | 0.470577 | -0.907062 | 0.643076 | 0.082258 | -0.904361 | -0.914150
0.500 | 0.745968 | 0.427871 | -0.738289 | 0.622805 | 0.084435 | -1.042996 | -1.055160
0.600 | 0.488827 | 0.389617 | -0.598410 | 0.601928 | 0.086865 | -1.101128 | -1.116286
0.650 | 0.389395 | 0.372033 | -0.538834 | 0.591525 | 0.088159 | -1.112997 | -1.129905
0.700 | 0.304795 | 0.355426 | -0.485486 | 0.581232 | 0.089500 | -1.117349 | -1.136189
**0.735** | **0.252992** | **0.344368** | **-0.451507** | **0.574116** | **0.090466** | **-1.116999** | **-1.137306**
0.750 | 0.232435 | 0.339769 | -0.437726 | 0.571091 | 0.090886 | -1.116151 | -1.137117
0.800 | 0.170196 | 0.325033 | -0.394886 | 0.561128 | 0.092313 | -1.110850 | -1.134148
0.900 | 0.069455 | 0.298150 | -0.321425 | 0.541795 | 0.095286 | -1.091914 | -1.120560
1.000 | -0.007740 | 0.274331 | -0.260726 | 0.523311 | 0.098395 | -1.066109 | -1.101150
1.100 | -0.068023 | 0.253080 | -0.209712 | 0.505724 | 0.101611 | -1.036539 | -1.079193
1.200 | -0.115657 | 0.233973 | -0.166406 | 0.489070 | 0.104896 | -1.005107 | -1.056741
1.300 | -0.153517 | 0.216707 | -0.129509 | 0.473378 | 0.108209 | -0.973111 | -1.035186
1.500 | -0.207755 | 0.186913 | -0.071290 | 0.444916 | 0.114768 | -0.910874 | -0.998149
1.750 | -0.248929 | 0.157229 | -0.020552 | 0.414639 | 0.122538 | -0.841349 | -0.966335
2.000 | -0.272905 | 0.134559 | 0.013303 | 0.389632 | 0.129569 | -0.783793 | -0.948641
2.500 | -0.296664 | 0.105297 | 0.051028 | 0.352010 | 0.141105 | -0.702944 | -0.936055

A useful sanity check hides in this table: \\(g_0 + g_1 + g_2 + g_3 = \langle 00 \rvert H \lvert 00 \rangle\\) is the energy of the state with no electrons, which must be exactly the nuclear repulsion \\(1/R\\) in atomic units. At \\(R = 0.735\\) Å \\(= 1.388946\\) Bohr that is \\(0.719969\\) Hartree, and the four coefficients sum to \\(0.719969\\). Any error in the reduction would break this identity.

Code Example 3: Building the Hamiltonian and Diagonalising It Exactly

```python
import numpy as np
from qcsim import *

H2_COEFFS = {
    #  R (A):   (g0,        g1,        g2,        g3,        g4,        g5)
    0.300: (  1.684963,  0.517383,  -1.099915,  0.661493,  0.080409,  0.080409),
    0.400: (  1.116353,  0.470577,  -0.907062,  0.643076,  0.082258,  0.082258),
    0.500: (  0.745968,  0.427871,  -0.738289,  0.622805,  0.084435,  0.084435),
    0.600: (  0.488827,  0.389617,  -0.598410,  0.601928,  0.086865,  0.086865),
    0.650: (  0.389395,  0.372033,  -0.538834,  0.591525,  0.088159,  0.088159),
    0.700: (  0.304795,  0.355426,  -0.485486,  0.581232,  0.089500,  0.089500),
    0.735: (  0.252992,  0.344368,  -0.451507,  0.574116,  0.090466,  0.090466),
    0.750: (  0.232435,  0.339769,  -0.437726,  0.571091,  0.090886,  0.090886),
    0.800: (  0.170196,  0.325033,  -0.394886,  0.561128,  0.092313,  0.092313),
    0.900: (  0.069455,  0.298150,  -0.321425,  0.541795,  0.095286,  0.095286),
    1.000: ( -0.007740,  0.274331,  -0.260726,  0.523311,  0.098395,  0.098395),
    1.100: ( -0.068023,  0.253080,  -0.209712,  0.505724,  0.101611,  0.101611),
    1.200: ( -0.115657,  0.233973,  -0.166406,  0.489070,  0.104896,  0.104896),
    1.300: ( -0.153517,  0.216707,  -0.129509,  0.473378,  0.108209,  0.108209),
    1.500: ( -0.207755,  0.186913,  -0.071290,  0.444916,  0.114768,  0.114768),
    1.750: ( -0.248929,  0.157229,  -0.020552,  0.414639,  0.122538,  0.122538),
    2.000: ( -0.272905,  0.134559,   0.013303,  0.389632,  0.129569,  0.129569),
    2.500: ( -0.296664,  0.105297,   0.051028,  0.352010,  0.141105,  0.141105),
}

TERMS = ['II', 'ZI', 'IZ', 'ZZ', 'YY', 'XX']

def h2_hamiltonian(R):
    """Pauli decomposition {string: coefficient} of the 2-qubit H2 Hamiltonian."""
    return dict(zip(TERMS, H2_COEFFS[R]))

def pauli_matrix(pauli):
    M = np.array([[1.0 + 0j]])
    for ch in pauli:
        M = np.kron(M, PAULI[ch])
    return M

def hamiltonian_matrix(terms):
    return sum(c * pauli_matrix(p) for p, c in terms.items())

R = 0.735
terms = h2_hamiltonian(R)
Hm = hamiltonian_matrix(terms)

print(f"H2 at R = {R} A, STO-3G, two-qubit reduction")
print("-" * 66)
print("  H = " + " + ".join(f"({c:+.6f}) {p}" for p, c in terms.items()))
print("\n  matrix (real part; the imaginary part vanishes):")
for row in Hm.real:
    print("   ", "  ".join(f"{v:+9.6f}" for v in row))
print(f"\n  Hermitian: {np.allclose(Hm, Hm.conj().T)}")
print(f"  imaginary part: max |Im| = {np.max(np.abs(Hm.imag)):.1e}")

w, v = np.linalg.eigh(Hm)
print("\nExact diagonalization")
print("-" * 66)
basis = ['00', '01', '10', '11']
for k in range(4):
    comp = "  ".join(f"{v[i, k].real:+.4f}|{basis[i]}>" for i in range(4)
                     if abs(v[i, k]) > 1e-8)
    print(f"  E_{k} = {w[k]:+.6f} Ha   {comp}")

E_hf = Hm[2, 2].real          # |10> = sigma_g doubly occupied = Hartree-Fock
print(f"\n  Hartree-Fock energy  <10|H|10> = {E_hf:+.6f} Ha")
print(f"  exact ground state             = {w[0]:+.6f} Ha")
print(f"  correlation energy             = {w[0] - E_hf:+.6f} Ha "
      f"({(w[0]-E_hf)*627.509:+.2f} kcal/mol)")
print(f"  HF overlap |<10|psi_0>|^2      = {abs(v[2, 0])**2:.6f}")

print("\nDissociation curve by exact diagonalization")
print("-" * 66)
print(f"  {'R (A)':>7} {'E_HF':>11} {'E_exact':>11} {'E_corr':>10} {'|<HF|psi>|^2':>13}")
curve = []
for R in sorted(H2_COEFFS):
    M = hamiltonian_matrix(h2_hamiltonian(R))
    w, v = np.linalg.eigh(M)
    curve.append((R, M[2, 2].real, w[0]))
    print(f"  {R:7.3f} {M[2,2].real:11.6f} {w[0]:11.6f} {w[0]-M[2,2].real:10.6f} "
          f"{abs(v[2,0])**2:13.6f}")

Rs = np.array([c[0] for c in curve])
Es = np.array([c[2] for c in curve])
i = int(np.argmin(Es))
p = np.polyfit(Rs[i-1:i+2], Es[i-1:i+2], 2)
R_eq = -p[1] / (2 * p[0])
E_eq = np.polyval(p, R_eq)
print("\nEquilibrium from a parabola through the three lowest points")
print("-" * 66)
print(f"  R_eq   = {R_eq:.4f} A      (experiment 0.741 A, STO-3G/FCI 0.735 A)")
print(f"  E_min  = {E_eq:.6f} Ha    (STO-3G/FCI reference -1.1373 Ha)")
print(f"  D_e    = {(Es[-1] - E_eq) * 27.2114:.3f} eV  from E(2.5 A) - E(R_eq)")
print(f"  E(2.5 A) = {Es[-1]:.6f} Ha vs 2 x E(H atom, STO-3G) = {2*-0.4665818:.6f} Ha")
```

```text
H2 at R = 0.735 A, STO-3G, two-qubit reduction
------------------------------------------------------------------
  H = (+0.252992) II + (+0.344368) ZI + (-0.451507) IZ + (+0.574116) ZZ + (+0.090466) YY + (+0.090466) XX

  matrix (real part; the imaginary part vanishes):
    +0.719969  +0.000000  +0.000000  +0.000000
    +0.000000  +0.474751  +0.180932  +0.000000
    +0.000000  +0.180932  -1.116999  +0.000000
    +0.000000  +0.000000  +0.000000  +0.934247

  Hermitian: True
  imaginary part: max |Im| = 0.0e+00

Exact diagonalization
------------------------------------------------------------------
  E_0 = -1.137306 Ha   -0.1115|01>  +0.9938|10>
  E_1 = +0.495058 Ha   -0.9938|01>  -0.1115|10>
  E_2 = +0.719969 Ha   +1.0000|00>
  E_3 = +0.934247 Ha   +1.0000|11>

  Hartree-Fock energy  <10|H|10> = -1.116999 Ha
  exact ground state             = -1.137306 Ha
  correlation energy             = -0.020307 Ha (-12.74 kcal/mol)
  HF overlap |<10|psi_0>|^2      = 0.987560

Dissociation curve by exact diagonalization
------------------------------------------------------------------
    R (A)        E_HF     E_exact     E_corr  |<HF|psi>|^2
    0.300   -0.593828   -0.601804  -0.007976      0.997546
    0.400   -0.904362   -0.914150  -0.009788      0.996472
    0.500   -1.042997   -1.055160  -0.012163      0.994839
    0.600   -1.101128   -1.116286  -0.015158      0.992445
    0.650   -1.112997   -1.129905  -0.016908      0.990888
    0.700   -1.117349   -1.136189  -0.018840      0.989043
    0.735   -1.116999   -1.137306  -0.020307      0.987560
    0.750   -1.116151   -1.137117  -0.020966      0.986871
    0.800   -1.110851   -1.134148  -0.023297      0.984327
    0.900   -1.091915   -1.120561  -0.028646      0.977904
    1.000   -1.066108   -1.101149  -0.035041      0.969267
    1.100   -1.036539   -1.079193  -0.042654      0.957806
    1.200   -1.005106   -1.056740  -0.051634      0.942884
    1.300   -0.973111   -1.035187  -0.062076      0.923981
    1.500   -0.910874   -0.998150  -0.087276      0.873689
    1.750   -0.841349   -0.966336  -0.124987      0.793593
    2.000   -0.783793   -0.948641  -0.164848      0.711909
    2.500   -0.702943   -0.936055  -0.233112      0.594420

Equilibrium from a parabola through the three lowest points
------------------------------------------------------------------
  R_eq   = 0.7354 A      (experiment 0.741 A, STO-3G/FCI 0.735 A)
  E_min  = -1.137306 Ha    (STO-3G/FCI reference -1.1373 Ha)
  D_e    = 5.476 eV  from E(2.5 A) - E(R_eq)
  E(2.5 A) = -0.936055 Ha vs 2 x E(H atom, STO-3G) = -0.933164 Ha
```

**What to notice.** The matrix makes the block structure visible: two isolated diagonal entries and one \\(2 \times 2\\) block coupling \\(\lvert 01 \rangle\\) and \\(\lvert 10 \rangle\\) with off-diagonal element \\(2g_4 = 0.180932\\), which is the exchange integral \\(K_{gu}\\) between the bonding and antibonding orbitals. The ground state is \\(0.9938\lvert 10 \rangle - 0.1115\lvert 01 \rangle\\): mostly Hartree-Fock, with an 11% amplitude on the doubly excited configuration.

The correlation energy — the part Hartree-Fock misses — is \\(-20.3\\) mHartree at equilibrium, or 12.7 kcal/mol. That is 12.7 times the 1.6 mHartree "chemical accuracy" threshold, which is why correlation cannot be ignored, and it is *small* compared with the total energy, which is why it is hard to compute.

The most instructive column is the last one. The Hartree-Fock overlap falls from 0.998 at short range to 0.594 at 2.5 Å, and the correlation energy grows from \\(-8\\) mHartree to \\(-233\\) mHartree. This is **static correlation**: as the bond stretches, the two configurations become degenerate and no single determinant describes the state. Restricted Hartree-Fock does not merely lose accuracy here, it fails qualitatively, predicting an energy 0.23 Hartree too high at 2.5 Å. Strongly correlated materials — Mott insulators, transition-metal oxides, frustrated magnets — are systems in which this failure occurs at *equilibrium*. They are the reason anyone builds quantum computers for chemistry, and Chapter 4 takes them up directly.

One caveat on the numbers: \\(D_e = 5.48\\) eV from these two points is not the experimental 4.75 eV, and the reason is the basis set, not the finite bond length. Incomplete dissociation works the *other* way: using the true STO-3G limit \\(2E(\mathrm{H}) = -0.933164\\) Ha instead of \\(E(2.5\\,\text{Å}) = -0.936055\\) Ha gives \\(D_e = 5.56\\) eV, i.e. *larger*, so the residual bond at 2.5 Å is hiding part of the error rather than causing it. What remains — about 0.8 eV — is the minimal basis, which overbinds H₂ badly. The algorithm is exact; the model is crude.

### Measuring the energy: grouping and shot cost

A quantum computer does not have access to \\(\langle \psi \rvert H \lvert \psi \rangle\\). It measures qubits. The bridge is linearity: write \\(H = \sum_j c_j P_j\\) as a sum of Pauli strings, and

\\[ E = \sum_j c_j \langle \psi \rvert P_j \lvert \psi \rangle \\]

Each \\(\langle P_j \rangle\\) is estimated by repeating the circuit and averaging the eigenvalue \\(\pm 1\\) over the shots. Since \\(Z\\) is the only Pauli a device can measure directly, each term needs a basis change first — \\(H\\) for an \\(X\\) factor, \\(HS^\dagger\\) for a \\(Y\\) factor — exactly as in Exercise 1 of Chapter 2.

Two facts make this affordable, or not.

**Commuting terms share a circuit.** All the terms built from \\(I\\) and \\(Z\\) can be estimated from the same measurement record, because they are all diagonal: one circuit gives every bit string, from which \\(\langle Z_0 \rangle\\), \\(\langle Z_1 \rangle\\) and \\(\langle Z_0 Z_1 \rangle\\) all follow. Our six-term Hamiltonian needs only **three** distinct measurement settings. For a real molecule with \\(O(N^4)\\) Pauli terms, sophisticated grouping into commuting families is the difference between feasible and hopeless.

**The variance falls only as \\(1/\sqrt{N}\\).** Each \\(\langle P_j \rangle\\) is a sample mean of a \\(\pm 1\\) variable, so its standard error is \\(\sqrt{(1-\langle P_j\rangle^2)/N}\\). Reaching a target precision \\(\epsilon\\) in the energy needs

\\[ N \sim \left(\frac{\sum_j \lvert c_j \rvert}{\epsilon}\right)^2 \\]

shots in total. There is no way around the square: quantum measurement is sampling, and sampling costs.

Code Example 4: Term-by-Term Measurement and the Cost of Precision

```python
import numpy as np
from qcsim import *

TERMS = ['II', 'ZI', 'IZ', 'ZZ', 'YY', 'XX']
COEFFS_735 = (0.252992, 0.344368, -0.451507, 0.574116, 0.090466, 0.090466)
terms = dict(zip(TERMS, COEFFS_735))
Sdg = S.conj().T
R = 0.735

def ansatz(theta):
    psi = apply_gate(ket('00'), X, [0], 2)
    psi = apply_gate(psi, H, [0], 2)
    psi = apply_gate(psi, H @ Sdg, [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, rz(2 * theta), [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, S @ H, [1], 2)
    psi = apply_gate(psi, H, [0], 2)
    return psi

theta = -0.111769
psi = ansatz(theta)

print(f"Term-by-term energy at R = {R} A, theta = {theta}")
print("-" * 62)
print(f"  {'Pauli':>6} {'coefficient':>13} {'<P>':>11} {'contribution':>14}")
total = 0.0
for p, c in terms.items():
    e = expval(psi, p)
    total += c * e
    print(f"  {p:>6} {c:+13.6f} {e:+11.6f} {c*e:+14.6f}")
print(f"  {'':6} {'':13} {'sum':>11} {total:+14.6f}")

M = sum(c * np.kron(PAULI[p[0]], PAULI[p[1]]) for p, c in terms.items())
print(f"\n  one-liner with coeff_map: {sum(expval(psi, p, terms) for p in terms):+.9f}")
print(f"  <psi|H|psi> from the matrix: {np.vdot(psi, M @ psi).real:+.9f}")

print("\nMeasurement settings: commuting terms share one circuit")
print("-" * 62)
groups = {'Z basis  (II, ZI, IZ, ZZ)': ['II', 'ZI', 'IZ', 'ZZ'],
          'X basis  (XX)': ['XX'],
          'Y basis  (YY)': ['YY']}
for name, ps in groups.items():
    print(f"  {name:26s} -> {len(ps)} term(s), 1 circuit")
print(f"  total: {len(terms)} Pauli terms but only {len(groups)} distinct circuits")

def measure_in_basis(psi, basis, shots, rng):
    """Rotate the measurement basis to Z, then sample bit strings."""
    phi = psi.copy()
    for q in range(2):
        if basis[q] == 'X':
            phi = apply_gate(phi, H, [q], 2)
        elif basis[q] == 'Y':
            phi = apply_gate(phi, H @ Sdg, [q], 2)
    p = probs(phi)
    return rng.choice(4, size=shots, p=p / p.sum())

def estimate_energy(psi, shots_per_setting, rng):
    """Shot-based energy estimate using three measurement settings."""
    E = terms['II']
    z = measure_in_basis(psi, 'ZZ', shots_per_setting, rng)
    s0 = 1 - 2 * ((z >> 1) & 1)         # eigenvalue of Z on qubit 0
    s1 = 1 - 2 * (z & 1)                # eigenvalue of Z on qubit 1
    E += terms['ZI'] * s0.mean() + terms['IZ'] * s1.mean() + terms['ZZ'] * (s0 * s1).mean()
    for basis, label in [('XX', 'XX'), ('YY', 'YY')]:
        m = measure_in_basis(psi, basis, shots_per_setting, rng)
        t0 = 1 - 2 * ((m >> 1) & 1)
        t1 = 1 - 2 * (m & 1)
        E += terms[label] * (t0 * t1).mean()
    return E

exact = np.linalg.eigvalsh(M)[0]
print("\nFinite sampling: the price of every expectation value")
print("-" * 62)
print(f"  {'shots/setting':>14} {'mean E':>12} {'std':>10} {'|bias|':>10} {'std*sqrt(N)':>13}")
rng = np.random.default_rng(0)
for shots in [100, 1000, 10000, 100000]:
    runs = np.array([estimate_energy(psi, shots, rng) for _ in range(200)])
    print(f"  {shots:14d} {runs.mean():12.6f} {runs.std():10.6f} "
          f"{abs(runs.mean() - exact):10.6f} {runs.std()*np.sqrt(shots):13.4f}")
print("\n  the statistical error falls only as 1/sqrt(N): reaching 1 mHa ('chemical")
print("  accuracy' is 1.6 mHa) already needs of order 10^5-10^6 shots per setting,")
print("  and this is for the smallest molecule there is.")
```

```text
Term-by-term energy at R = 0.735 A, theta = -0.111769
--------------------------------------------------------------
   Pauli   coefficient         <P>   contribution
      II     +0.252992   +1.000000      +0.252992
      ZI     +0.344368   -0.975119      -0.335800
      IZ     -0.451507   +0.975119      -0.440273
      ZZ     +0.574116   -1.000000      -0.574116
      YY     +0.090466   -0.221681      -0.020055
      XX     +0.090466   -0.221681      -0.020055
                               sum      -1.137306

  one-liner with coeff_map: -1.137306213
  <psi|H|psi> from the matrix: -1.137306213

Measurement settings: commuting terms share one circuit
--------------------------------------------------------------
  Z basis  (II, ZI, IZ, ZZ)  -> 4 term(s), 1 circuit
  X basis  (XX)              -> 1 term(s), 1 circuit
  Y basis  (YY)              -> 1 term(s), 1 circuit
  total: 6 Pauli terms but only 3 distinct circuits

Finite sampling: the price of every expectation value
--------------------------------------------------------------
   shots/setting       mean E        std     |bias|   std*sqrt(N)
             100    -1.137723   0.022003   0.000417        0.2200
            1000    -1.137365   0.006386   0.000059        0.2019
           10000    -1.137390   0.002192   0.000084        0.2192
          100000    -1.137297   0.000660   0.000009        0.2088

  the statistical error falls only as 1/sqrt(N): reaching 1 mHa ('chemical
  accuracy' is 1.6 mHa) already needs of order 10^5-10^6 shots per setting,
  and this is for the smallest molecule there is.
```

**What to notice.** The term-by-term table shows where the energy comes from: the \\(ZZ\\) term contributes \\(-0.574\\), the two single-\\(Z\\) terms nearly cancel, and the two off-diagonal terms \\(XX\\) and \\(YY\\) contribute only \\(-0.040\\) between them. Yet those small terms are the entire correlation energy — remove them and you are back to Hartree-Fock. Precision matters most in the smallest contributions, which is an uncomfortable fact for a sampling-based estimator.

The sampling table confirms the \\(1/\sqrt{N}\\) law precisely: the product \\(\sigma\sqrt{N}\\) is constant at \\(0.21 \pm 0.01\\) Hartree across three decades of shot count. That constant is the number that governs the cost of VQE. To reach \\(\sigma = 1\\) mHartree needs \\(N \approx (0.21/0.001)^2 \approx 4 \times 10^4\\) shots per setting — and that is one energy evaluation, at one geometry, for the smallest molecule in chemistry, with a converged optimiser needing dozens of such evaluations. Section 3.6 extrapolates.

* * *

## 3.4 The Classical Optimisation Loop

### Gradient-free methods

The optimiser sees a black box: it proposes \\(\boldsymbol{\theta}\\), receives a noisy \\(E(\boldsymbol{\theta})\\). Gradient-free methods are the natural first choice, because on hardware the function values themselves carry statistical noise and finite-difference gradients amplify it.

  * **COBYLA** builds a linear model from a simplex of points and is the de facto default in the VQE literature: few evaluations, no gradient, handles moderate noise.
  * **Nelder-Mead** is robust but slow, and its simplex can collapse on noisy landscapes.
  * **Powell** does successive line searches along conjugate directions; accurate but evaluation-hungry.
  * **SPSA** estimates a stochastic gradient from two evaluations regardless of dimension, and is the method of choice for genuinely noisy hardware with many parameters.

### The parameter-shift rule

The structure of a quantum circuit gives something better than a finite difference: an **exact** derivative from two circuit evaluations at shifted parameter values. Suppose the parameter enters through \\(\exp(-i\theta P)\\) with \\(P^2 = I\\). Then \\(\lvert \psi(\theta) \rangle = (\cos\theta - i \sin\theta P)\lvert \psi_0 \rangle\\), and the energy is a single Fourier mode:

\\[ E(\theta) = A\cos(2\theta) + B\sin(2\theta) + C \\]

for constants determined by the Hamiltonian and the reference state. Differentiating, and using \\(\sin\\) and \\(\cos\\) addition formulas, gives an exact finite difference at a *large* shift:

\\[ \frac{dE}{d\theta} = E\left(\theta + \frac{\pi}{4}\right) - E\left(\theta - \frac{\pi}{4}\right) \\]

For the conventional half-angle convention \\(\exp(-i\theta P/2)\\) — which is what \\(R_x\\), \\(R_y\\), \\(R_z\\) use — the same derivation gives the more familiar form

\\[ \frac{dE}{d\theta} = \frac{1}{2}\left[E\left(\theta + \frac{\pi}{2}\right) - E\left(\theta - \frac{\pi}{2}\right)\right] \\]

This is not an approximation. There is no step size to tune, no truncation error, and — crucially on hardware — the two evaluation points are far apart, so shot noise is not amplified by division by a small number. The cost is two circuit evaluations per parameter, so a full gradient of a \\(p\\)-parameter ansatz costs \\(2p\\) evaluations.

The same three coefficients \\(A\\), \\(B\\), \\(C\\) can be reconstructed from three energy evaluations, after which the minimum is available in closed form, \\(\theta^\* = \tfrac{1}{2}\mathrm{atan2}(-B, -A)\\). For a one-parameter ansatz this makes the optimiser redundant — a fact known as **rotosolve**, and a reminder that the classical part of a variational algorithm is not always the hard part.

Code Example 5: Four Optimisers on the Same Landscape

```python
import numpy as np
from scipy.optimize import minimize
from qcsim import *

TERMS = ['II', 'ZI', 'IZ', 'ZZ', 'YY', 'XX']
COEFFS_735 = (0.252992, 0.344368, -0.451507, 0.574116, 0.090466, 0.090466)
Sdg = S.conj().T

def ansatz(theta):
    psi = apply_gate(ket('00'), X, [0], 2)
    psi = apply_gate(psi, H, [0], 2)
    psi = apply_gate(psi, H @ Sdg, [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, rz(2 * theta), [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, S @ H, [1], 2)
    psi = apply_gate(psi, H, [0], 2)
    return psi

def make_energy(terms):
    def energy(params):
        psi = ansatz(float(params[0]))
        return sum(expval(psi, p, terms) for p in terms)
    return energy

R = 0.735
terms = dict(zip(TERMS, COEFFS_735))
energy = make_energy(terms)
M = sum(c * np.kron(PAULI[p[0]], PAULI[p[1]]) for p, c in terms.items())
exact = float(np.linalg.eigvalsh(M)[0])

print(f"VQE for H2 at R = {R} A  (exact ground state {exact:.9f} Ha)")
print("-" * 76)
print(f"  {'optimizer':>13} {'theta*':>11} {'E_VQE (Ha)':>14} "
      f"{'E_VQE - E_exact':>17} {'evals':>7}")
history = {}
for method, opts in [('COBYLA', {'tol': 1e-12, 'maxiter': 2000}),
                     ('Nelder-Mead', {'xatol': 1e-12, 'fatol': 1e-14}),
                     ('Powell', {'xtol': 1e-12, 'ftol': 1e-14}),
                     ('BFGS', {'gtol': 1e-10})]:
    trace = []
    def wrapped(x):
        e = energy(x)
        trace.append(e)
        return e
    res = minimize(wrapped, x0=[0.0], method=method, options=opts)
    history[method] = trace
    print(f"  {method:>13} {float(res.x[0]):11.6f} {res.fun:14.9f} "
          f"{res.fun - exact:17.2e} {len(trace):7d}")

print("\nConvergence of the COBYLA run (energy after each evaluation)")
print("-" * 76)
tr = history['COBYLA']
for i in [0, 1, 2, 3, 4, 5, 8, 12, len(tr) - 1]:
    if i < len(tr):
        print(f"  evaluation {i:3d}: E = {tr[i]:12.9f} Ha, "
              f"error = {tr[i] - exact:+.2e} Ha")

print("\nStarting point matters less than you might fear (COBYLA)")
print("-" * 76)
print(f"  {'theta_0':>9} {'theta*':>11} {'E_VQE':>14} {'error':>12}")
for x0 in [-1.5, -0.5, 0.0, 0.5, 1.5, 3.0]:
    res = minimize(energy, x0=[x0], method='COBYLA', options={'tol': 1e-12, 'maxiter': 2000})
    print(f"  {x0:9.2f} {float(res.x[0]):11.6f} {res.fun:14.9f} {res.fun - exact:12.2e}")
print("  (theta and theta + pi give the same energy: the ansatz is pi-periodic in E)")

print("\nThe variational principle is a strict bound, never an accident")
print("-" * 76)
for th in [-0.111769, -0.05, 0.0, 0.3]:
    E = energy([th])
    print(f"  theta = {th:+.6f}: E = {E:+.9f} Ha, E - E_exact = {E - exact:+.9e} "
          f"{'<-- optimum' if E - exact < 1e-9 else ''}")
print("  every trial energy lies above the exact ground state; the optimizer can")
print("  only push it down towards the true value, never below it.")
```

```text
VQE for H2 at R = 0.735 A  (exact ground state -1.137306213 Ha)
----------------------------------------------------------------------------
      optimizer      theta*     E_VQE (Ha)   E_VQE - E_exact   evals
         COBYLA   -0.111769   -1.137306213          6.66e-16      51
    Nelder-Mead   -0.111769   -1.137306213          6.66e-16      98
         Powell   -0.111769   -1.137306213          4.44e-16      88
           BFGS   -0.111769   -1.137306213          1.11e-15      28

Convergence of the COBYLA run (energy after each evaluation)
----------------------------------------------------------------------------
  evaluation   0: E = -1.116999000 Ha, error = +2.03e-02 Ha
  evaluation   1: E =  0.174597866 Ha, error = +1.31e+00 Ha
  evaluation   2: E = -0.154444138 Ha, error = +9.83e-01 Ha
  evaluation   3: E = -0.598888069 Ha, error = +5.38e-01 Ha
  evaluation   4: E = -1.106313443 Ha, error = +3.10e-02 Ha
  evaluation   5: E = -1.065188848 Ha, error = +7.21e-02 Ha
  evaluation   8: E = -1.131086000 Ha, error = +6.22e-03 Ha
  evaluation  12: E = -1.137118244 Ha, error = +1.88e-04 Ha
  evaluation  50: E = -1.137306213 Ha, error = +6.66e-16 Ha

Starting point matters less than you might fear (COBYLA)
----------------------------------------------------------------------------
    theta_0      theta*          E_VQE        error
      -1.50   -0.111769   -1.137306213     6.66e-16
      -0.50   -0.111769   -1.137306213     6.66e-16
       0.00   -0.111769   -1.137306213     6.66e-16
       0.50   -0.111769   -1.137306213     4.44e-16
       1.50    3.029824   -1.137306213     6.66e-16
       3.00    3.029824   -1.137306213     6.66e-16
  (theta and theta + pi give the same energy: the ansatz is pi-periodic in E)

The variational principle is a strict bound, never an accident
----------------------------------------------------------------------------
  theta = -0.111769: E = -1.137306213 Ha, E - E_exact = +3.774758284e-15 <-- optimum
  theta = -0.050000: E = -1.131086000 Ha, E - E_exact = +6.220212870e-03 
  theta = +0.000000: E = -1.116999000 Ha, E - E_exact = +2.030721265e-02 
  theta = +0.300000: E = -0.875826091 Ha, E - E_exact = +2.614801221e-01 
  every trial energy lies above the exact ground state; the optimizer can
  only push it down towards the true value, never below it.
```

**What to notice.** All four optimisers land on \\(\theta^\* = -0.111769\\) and an energy that matches exact diagonalisation to \\(10^{-15}\\) Hartree — floating-point noise. The evaluation counts differ by a factor of three, which on hardware is a factor of three in wall-clock time and in total shots; this is why optimiser choice is a real engineering decision even though the answer is the same.

The first COBYLA evaluation is the Hartree-Fock energy, because \\(\theta_0 = 0\\) leaves the reference state untouched. That is the standard initialisation and it is a good one: it starts the search from the best single-determinant state and from a point where the gradient is nonzero. The error then falls from \\(2 \times 10^{-2}\\) to \\(2 \times 10^{-4}\\) in twelve evaluations — about four evaluations per digit, which is the practical scaling to expect.

The last block is the variational principle in action: at every non-optimal \\(\theta\\) the energy sits strictly above the exact value, and the deviation grows quadratically near the minimum (\\(\theta\\) off by 0.06 costs 6 mHartree; off by 0.11 costs 20 mHartree). The quadratic flatness is the reason VQE is robust and also the reason it converges slowly at the end.

Code Example 6: Parameter-Shift Gradients

```python
import numpy as np
from qcsim import *

TERMS = ['II', 'ZI', 'IZ', 'ZZ', 'YY', 'XX']
COEFFS_735 = (0.252992, 0.344368, -0.451507, 0.574116, 0.090466, 0.090466)
terms = dict(zip(TERMS, COEFFS_735))
Sdg = S.conj().T
R = 0.735

def ansatz(theta):
    psi = apply_gate(ket('00'), X, [0], 2)
    psi = apply_gate(psi, H, [0], 2)
    psi = apply_gate(psi, H @ Sdg, [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, rz(2 * theta), [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, S @ H, [1], 2)
    psi = apply_gate(psi, H, [0], 2)
    return psi

def energy(theta):
    psi = ansatz(theta)
    return sum(expval(psi, p, terms) for p in terms)

def grad_parameter_shift(theta):
    """Exact derivative from two energy evaluations, shift = pi/4."""
    return energy(theta + np.pi / 4) - energy(theta - np.pi / 4)

M = sum(c * np.kron(PAULI[p[0]], PAULI[p[1]]) for p, c in terms.items())
A = (M[2, 2].real - M[1, 1].real) / 2
B = M[1, 2].real
def grad_analytic(theta):
    return -2 * A * np.sin(2 * theta) + 2 * B * np.cos(2 * theta)

print("Parameter-shift rule against finite differences")
print("-" * 78)
print(f"  {'theta':>8} {'analytic':>12} {'param-shift':>13} {'finite h=1e-2':>15} "
      f"{'finite h=1e-6':>15}")
for th in [-0.5, -0.111769, 0.0, 0.4, 1.2]:
    fd2 = (energy(th + 1e-2) - energy(th - 1e-2)) / 2e-2
    fd6 = (energy(th + 1e-6) - energy(th - 1e-6)) / 2e-6
    print(f"  {th:8.4f} {grad_analytic(th):12.8f} {grad_parameter_shift(th):13.8f} "
          f"{fd2:15.8f} {fd6:15.8f}")
print("\n  the parameter-shift value is exact at every theta, while the finite")
print("  difference carries a truncation error at large h and, on hardware, an")
print("  amplified shot-noise error at small h.")

err_ps = max(abs(grad_parameter_shift(t) - grad_analytic(t))
             for t in np.linspace(-2, 2, 41))
print(f"\n  max |param-shift - analytic| over theta in [-2, 2] : {err_ps:.2e}")

print("\nGradient descent driven by parameter-shift gradients")
print("-" * 78)
exact = float(np.linalg.eigvalsh(M)[0])
theta, lr = 0.6, 0.25
print(f"  {'step':>5} {'theta':>11} {'E (Ha)':>14} {'dE/dtheta':>12} {'error':>11}")
for step in range(21):
    g = grad_parameter_shift(theta)
    E = energy(theta)
    if step % 2 == 0 or step == 20:
        print(f"  {step:5d} {theta:11.6f} {E:14.9f} {g:12.6f} {E - exact:11.2e}")
    theta -= lr * g
print(f"\n  converged theta = {theta:.9f}, exact stationary point "
      f"{0.5*np.arctan2(-B, -A):.9f}")
print(f"  final energy = {energy(theta):.9f} Ha, exact = {exact:.9f} Ha, "
      f"difference = {energy(theta) - exact:.2e} Ha")

print("\nWhy shift = pi/4: the landscape is a single Fourier mode")
print("-" * 78)
print(f"  E(theta) = A cos(2 theta) + B sin(2 theta) + C with")
print(f"  A = {A:+.6f}, B = {B:+.6f}, C = {(M[2,2].real + M[1,1].real)/2:+.6f}")
print("  For a generator P with P^2 = I the exact rule is")
print("  dE/dtheta = E(theta + pi/4) - E(theta - pi/4).")
print("  Three energy evaluations therefore determine the whole one-parameter")
print("  landscape - and its exact minimum - without any optimizer at all:")
E0, Ep, Em = energy(0.0), energy(np.pi / 4), energy(-np.pi / 4)
A_r = E0 - (Ep + Em) / 2
B_r = (Ep - Em) / 2
C_r = (Ep + Em) / 2
th_star = 0.5 * np.arctan2(-B_r, -A_r)
print(f"    reconstructed A = {A_r:+.6f}, B = {B_r:+.6f}, C = {C_r:+.6f}")
print(f"    theta* = {th_star:.9f}, E = {energy(th_star):.9f} Ha, "
      f"error = {energy(th_star) - exact:.2e} Ha")
```

```text
Parameter-shift rule against finite differences
------------------------------------------------------------------------------
     theta     analytic   param-shift   finite h=1e-2   finite h=1e-6
   -0.5000  -1.14389549   -1.14389549     -1.14381923     -1.14389549
   -0.1118  -0.00000014   -0.00000014     -0.00000014     -0.00000014
    0.0000   0.36186400    0.36186400      0.36183988      0.36186400
    0.4000   1.39396463    1.39396463      1.39387171      1.39396463
    1.2000   0.80833228    0.80833228      0.80827839      0.80833228

  the parameter-shift value is exact at every theta, while the finite
  difference carries a truncation error at large h and, on hardware, an
  amplified shot-noise error at small h.

  max |param-shift - analytic| over theta in [-2, 2] : 1.55e-15

Gradient descent driven by parameter-shift gradients
------------------------------------------------------------------------------
   step       theta         E (Ha)    dE/dtheta       error
      0    0.600000   -0.440879782     1.614697    6.96e-01
      2   -0.039522   -1.128800751     0.235046    8.51e-03
      4   -0.109289   -1.137296172     0.008097    1.00e-05
      6   -0.111685   -1.137306201     0.000274    1.15e-08
      8   -0.111766   -1.137306213     0.000009    1.31e-11
     10   -0.111769   -1.137306213     0.000000    1.53e-14
     12   -0.111769   -1.137306213     0.000000    8.88e-16
     14   -0.111769   -1.137306213     0.000000    6.66e-16
     16   -0.111769   -1.137306213     0.000000    1.11e-15
     18   -0.111769   -1.137306213     0.000000    1.11e-15
     20   -0.111769   -1.137306213     0.000000    6.66e-16

  converged theta = -0.111768957, exact stationary point -0.111768957
  final energy = -1.137306213 Ha, exact = -1.137306213 Ha, difference = 4.44e-16 Ha

Why shift = pi/4: the landscape is a single Fourier mode
------------------------------------------------------------------------------
  E(theta) = A cos(2 theta) + B sin(2 theta) + C with
  A = -0.795875, B = +0.180932, C = -0.321124
  For a generator P with P^2 = I the exact rule is
  dE/dtheta = E(theta + pi/4) - E(theta - pi/4).
  Three energy evaluations therefore determine the whole one-parameter
  landscape - and its exact minimum - without any optimizer at all:
    reconstructed A = -0.795875, B = +0.180932, C = -0.321124
    theta* = -0.111768957, E = -1.137306213 Ha, error = 8.88e-16 Ha
```

**What to notice.** The parameter-shift derivative agrees with the analytic one to \\(1.6 \times 10^{-15}\\) across the whole parameter range, while the \\(h = 10^{-2}\\) finite difference is already wrong in the fourth decimal. On a noiseless simulator you could simply use \\(h = 10^{-6}\\) and be done; on hardware, where each energy carries an error of order \\(10^{-3}\\), dividing by \\(2h = 2\times10^{-6}\\) multiplies that error by \\(5 \times 10^5\\). The parameter-shift rule divides by nothing.

Gradient descent then converges geometrically — one order of magnitude in the energy error every two steps — and the reconstruction at the end is the punchline: three energy evaluations recover \\(A\\), \\(B\\), \\(C\\) to six digits and hand back the exact minimiser. For a one-parameter ansatz, VQE does not need an optimiser at all.

* * *

## 3.5 The Dissociation Curve, Checked

Now the full calculation: run VQE independently at each bond length, and compare every point against exact diagonalisation of the identical Hamiltonian. This is the verification the whole chapter is built around.

Code Example 7: H₂ Dissociation Curve, VQE vs Exact Diagonalisation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qcsim import *

H2_COEFFS = {
    0.300: (  1.684963,  0.517383,  -1.099915,  0.661493,  0.080409,  0.080409),
    0.400: (  1.116353,  0.470577,  -0.907062,  0.643076,  0.082258,  0.082258),
    0.500: (  0.745968,  0.427871,  -0.738289,  0.622805,  0.084435,  0.084435),
    0.600: (  0.488827,  0.389617,  -0.598410,  0.601928,  0.086865,  0.086865),
    0.650: (  0.389395,  0.372033,  -0.538834,  0.591525,  0.088159,  0.088159),
    0.700: (  0.304795,  0.355426,  -0.485486,  0.581232,  0.089500,  0.089500),
    0.735: (  0.252992,  0.344368,  -0.451507,  0.574116,  0.090466,  0.090466),
    0.750: (  0.232435,  0.339769,  -0.437726,  0.571091,  0.090886,  0.090886),
    0.800: (  0.170196,  0.325033,  -0.394886,  0.561128,  0.092313,  0.092313),
    0.900: (  0.069455,  0.298150,  -0.321425,  0.541795,  0.095286,  0.095286),
    1.000: ( -0.007740,  0.274331,  -0.260726,  0.523311,  0.098395,  0.098395),
    1.100: ( -0.068023,  0.253080,  -0.209712,  0.505724,  0.101611,  0.101611),
    1.200: ( -0.115657,  0.233973,  -0.166406,  0.489070,  0.104896,  0.104896),
    1.300: ( -0.153517,  0.216707,  -0.129509,  0.473378,  0.108209,  0.108209),
    1.500: ( -0.207755,  0.186913,  -0.071290,  0.444916,  0.114768,  0.114768),
    1.750: ( -0.248929,  0.157229,  -0.020552,  0.414639,  0.122538,  0.122538),
    2.000: ( -0.272905,  0.134559,   0.013303,  0.389632,  0.129569,  0.129569),
    2.500: ( -0.296664,  0.105297,   0.051028,  0.352010,  0.141105,  0.141105),
}
TERMS = ['II', 'ZI', 'IZ', 'ZZ', 'YY', 'XX']
Sdg = S.conj().T

def ansatz(theta):
    psi = apply_gate(ket('00'), X, [0], 2)
    psi = apply_gate(psi, H, [0], 2)
    psi = apply_gate(psi, H @ Sdg, [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, rz(2 * theta), [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, S @ H, [1], 2)
    psi = apply_gate(psi, H, [0], 2)
    return psi

def hamiltonian_matrix(terms):
    return sum(c * np.kron(PAULI[p[0]], PAULI[p[1]]) for p, c in terms.items())

def run_vqe(terms, theta0=0.0):
    def energy(x):
        psi = ansatz(float(x[0]))
        return sum(expval(psi, p, terms) for p in terms)
    res = minimize(energy, x0=[theta0], method='COBYLA',
                   options={'tol': 1e-12, 'maxiter': 2000})
    return float(res.x[0]), float(res.fun), res.nfev

print("H2 dissociation curve: VQE against exact diagonalization")
print("-" * 88)
print(f"  {'R (A)':>6} {'theta*':>10} {'E_VQE':>13} {'E_exact':>13} "
      f"{'E_VQE-E_exact':>15} {'E_HF':>11} {'calls':>6}")
rows = []
for R in sorted(H2_COEFFS):
    terms = dict(zip(TERMS, H2_COEFFS[R]))
    M = hamiltonian_matrix(terms)
    E_exact = float(np.linalg.eigvalsh(M)[0])
    E_hf = float(M[2, 2].real)
    th, E_vqe, nfev = run_vqe(terms)
    rows.append((R, th, E_vqe, E_exact, E_hf))
    print(f"  {R:6.3f} {th:10.6f} {E_vqe:13.9f} {E_exact:13.9f} "
          f"{E_vqe - E_exact:15.2e} {E_hf:11.6f} {nfev:6d}")

rows = np.array(rows)
dev = np.abs(rows[:, 2] - rows[:, 3])
print("-" * 88)
print(f"  maximum |E_VQE - E_exact| over the whole curve : {dev.max():.3e} Ha")
print(f"  mean    |E_VQE - E_exact|                      : {dev.mean():.3e} Ha")
print(f"  chemical accuracy (1 kcal/mol)                 : {1.6e-3:.3e} Ha")
print(f"  the agreement is {1.6e-3/max(dev.max(), 1e-18):.1e} times tighter than "
      "chemical accuracy")

i = int(np.argmin(rows[:, 3]))
p = np.polyfit(rows[i-1:i+2, 0], rows[i-1:i+2, 2], 2)
R_eq = -p[1] / (2 * p[0])
print(f"\n  VQE equilibrium bond length : {R_eq:.4f} A  "
      f"(STO-3G/FCI 0.735 A, experiment 0.741 A)")
print(f"  VQE minimum energy          : {np.polyval(p, R_eq):.6f} Ha  "
      f"(STO-3G/FCI -1.137306 Ha)")
print(f"  binding energy from E(2.5 A) : "
      f"{(rows[-1, 2] - np.polyval(p, R_eq)) * 27.2114:.3f} eV  "
      "(STO-3G limit 5.55 eV, experiment 4.75 eV)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(rows[:, 0], rows[:, 4], 's--', label='Hartree-Fock', color='tab:orange')
ax1.plot(rows[:, 0], rows[:, 3], '-', label='exact diagonalization',
         color='black', linewidth=2)
ax1.plot(rows[:, 0], rows[:, 2], 'o', label='VQE', color='tab:blue',
         markerfacecolor='none', markersize=9)
ax1.axhline(2 * -0.4665818, color='grey', linestyle=':',
            label='2 x E(H), STO-3G')
ax1.set_xlabel('bond length R (Å)', fontsize=12)
ax1.set_ylabel('energy (Hartree)', fontsize=12)
ax1.set_title('H₂ dissociation curve, STO-3G / 2 qubits', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

ax2.semilogy(rows[:, 0], np.maximum(dev, 1e-18), 'o-', color='tab:blue',
             label='|E_VQE - E_exact|')
ax2.axhline(1.6e-3, color='tab:red', linestyle='--', label='chemical accuracy')
ax2.set_xlabel('bond length R (Å)', fontsize=12)
ax2.set_ylabel('absolute error (Hartree)', fontsize=12)
ax2.set_title('VQE error against exact diagonalization', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.show()
```

```text
H2 dissociation curve: VQE against exact diagonalization
----------------------------------------------------------------------------------------
   R (A)     theta*         E_VQE       E_exact   E_VQE-E_exact        E_HF  calls
   0.300  -0.049555  -0.601803900  -0.601803900        1.11e-16   -0.593828     60
   0.400  -0.059428  -0.914150378  -0.914150378        5.55e-16   -0.904362     55
   0.500  -0.071904  -1.055160480  -1.055160480        8.88e-16   -1.042997     54
   0.600  -0.087028  -1.116285662  -1.116285662        8.88e-16   -1.101128     47
   0.650  -0.095603  -1.129905150  -1.129905150        1.11e-15   -1.112997     47
   0.700  -0.104867  -1.136189285  -1.136189285        6.66e-16   -1.117349     53
   0.735  -0.111769  -1.137306213  -1.137306213        6.66e-16   -1.116999     51
   0.750  -0.114833  -1.137116729  -1.137116729        4.44e-16   -1.116151     50
   0.800  -0.125522  -1.134148070  -1.134148070        2.22e-16   -1.110851     53
   0.900  -0.149200  -1.120561311  -1.120561311        8.88e-16   -1.091915     50
   1.000  -0.176218  -1.101149498  -1.101149498        4.44e-16   -1.066108     57
   1.100  -0.206885  -1.079192958  -1.079192958        6.66e-16   -1.036539     48
   1.200  -0.241325  -1.056740304  -1.056740304        6.66e-16   -1.005106     50
   1.300  -0.279334  -1.035186892  -1.035186892        6.66e-16   -0.973111     51
   1.500  -0.363345  -0.998149747  -0.998149747        6.66e-16   -0.910874     49
   1.750  -0.471609  -0.966335782  -0.966335782        8.88e-16   -0.841349     43
   2.000  -0.566570  -0.948641038  -0.948641038        6.66e-16   -0.783793     51
   2.500  -0.690408  -0.936054599  -0.936054599        6.66e-16   -0.702943     56
----------------------------------------------------------------------------------------
  maximum |E_VQE - E_exact| over the whole curve : 1.110e-15 Ha
  mean    |E_VQE - E_exact|                      : 6.538e-16 Ha
  chemical accuracy (1 kcal/mol)                 : 1.600e-03 Ha
  the agreement is 1.4e+12 times tighter than chemical accuracy

  VQE equilibrium bond length : 0.7354 A  (STO-3G/FCI 0.735 A, experiment 0.741 A)
  VQE minimum energy          : -1.137306 Ha  (STO-3G/FCI -1.137306 Ha)
  binding energy from E(2.5 A) : 5.476 eV  (STO-3G limit 5.55 eV, experiment 4.75 eV)
```

**Reading the verification.** The maximum discrepancy between VQE and exact diagonalisation over eighteen geometries is \\(1.1 \times 10^{-15}\\) Hartree — the resolution of double-precision arithmetic on numbers of order 1. VQE has not approximated the ground state; it has found it. That is the expected outcome when the ansatz contains the exact solution, the simulation is noiseless and the optimiser converges, and seeing anything else would indicate a bug.

Three quantities can be checked against the literature rather than against ourselves, and all three agree:

Quantity | This calculation | Reference (STO-3G) | Experiment
---|---|---|---
Equilibrium bond length | 0.7354 Å | 0.735 Å | 0.741 Å
Minimum energy | -1.137306 Ha | -1.1373 Ha | —
Hartree-Fock energy at 0.735 Å | -1.116999 Ha | -1.1170 Ha | —
Dissociation limit | -0.9361 Ha at 2.5 Å | -0.9332 Ha (\\(2\times\\)H) | —
Binding energy | 5.48 eV | 5.55 eV (minimal-basis limit) | 4.75 eV

The left column is what the algorithm computed; the middle is what the model predicts; the right is nature. The gap between middle and right is the minimal basis set — STO-3G overbinds H₂ by about 0.8 eV — and no quantum algorithm can close it. **Choosing the basis is chemistry; solving within it is what the quantum computer does.** Conflating the two is the most common error in reading quantum-chemistry benchmarks.

Notice also the \\(\theta^\*\\) column. At short bond length the optimal parameter is small (\\(-0.05\\)): Hartree-Fock is nearly correct. At 2.5 Å it has grown to \\(-0.69\\), approaching the value \\(-\pi/4\\) at which the two configurations contribute equally. The single VQE parameter is tracking the crossover from a weakly to a strongly correlated regime, and it does so smoothly — which is exactly what a good ansatz parameter should do.

### Where the coefficients come from

The table in Section 3.3 is not quoted from a paper; it is computed, and here is the computation. The STO-3G basis for hydrogen is a fixed contraction of three Gaussians, so all the integrals have closed forms, the two molecular orbitals follow from symmetry, and the six Hamiltonian coefficients follow from four integrals. Chapter 4 explains *why* the reduction takes this form; this example establishes *that* the numbers are right.

Code Example 8: The Coefficients from STO-3G Integrals

```python
import numpy as np
from scipy.special import erf

BOHR = 0.52917721092                       # Angstrom per Bohr
ALPHA = np.array([3.42525091, 0.62391373, 0.16885540])   # STO-3G hydrogen
COEF = np.array([0.15432897, 0.53532814, 0.44463454])
D = COEF * (2 * ALPHA / np.pi) ** 0.75     # contraction coefficients, normalised

def boys0(t):
    """F_0(t) = int_0^1 exp(-t u^2) du."""
    return 1.0 if t < 1e-12 else 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))

def overlap(a, b, RAB2):
    return (np.pi / (a + b)) ** 1.5 * np.exp(-a * b / (a + b) * RAB2)

def kinetic(a, b, RAB2):
    p = a * b / (a + b)
    return p * (3 - 2 * p * RAB2) * (np.pi / (a + b)) ** 1.5 * np.exp(-p * RAB2)

def nuclear(a, b, RAB2, RPC2):
    return -2 * np.pi / (a + b) * np.exp(-a * b / (a + b) * RAB2) * boys0((a + b) * RPC2)

def repulsion(a, b, c, d, RAB2, RCD2, RPQ2):
    return (2 * np.pi ** 2.5 / ((a + b) * (c + d) * np.sqrt(a + b + c + d))
            * np.exp(-a * b / (a + b) * RAB2 - c * d / (c + d) * RCD2)
            * boys0((a + b) * (c + d) / (a + b + c + d) * RPQ2))

def ao_integrals(R):
    """Overlap, core Hamiltonian and two-electron integrals for H2 (R in Bohr)."""
    C = np.array([0.0, R])
    S = np.zeros((2, 2)); T = np.zeros((2, 2)); V = np.zeros((2, 2))
    for A in range(2):
        for B in range(2):
            RAB2 = (C[A] - C[B]) ** 2
            for a, da in zip(ALPHA, D):
                for b, db in zip(ALPHA, D):
                    w = da * db
                    S[A, B] += w * overlap(a, b, RAB2)
                    T[A, B] += w * kinetic(a, b, RAB2)
                    P = (a * C[A] + b * C[B]) / (a + b)
                    for nuc in range(2):
                        V[A, B] += w * nuclear(a, b, RAB2, (P - C[nuc]) ** 2)
    ERI = np.zeros((2, 2, 2, 2))
    for A in range(2):
        for B in range(2):
            for Cc in range(2):
                for Dd in range(2):
                    RAB2 = (C[A] - C[B]) ** 2
                    RCD2 = (C[Cc] - C[Dd]) ** 2
                    s = 0.0
                    for a, da in zip(ALPHA, D):
                        for b, db in zip(ALPHA, D):
                            P = (a * C[A] + b * C[B]) / (a + b)
                            for c, dc in zip(ALPHA, D):
                                for d, dd in zip(ALPHA, D):
                                    Q = (c * C[Cc] + d * C[Dd]) / (c + d)
                                    s += da * db * dc * dd * repulsion(
                                        a, b, c, d, RAB2, RCD2, (P - Q) ** 2)
                    ERI[A, B, Cc, Dd] = s
    return S, T + V, ERI

def two_qubit_coefficients(R_angstrom):
    """g0...g5 of the two-qubit H2 Hamiltonian, from STO-3G integrals."""
    R = R_angstrom / BOHR
    S, Hcore, ERI = ao_integrals(R)
    s = S[0, 1]
    # symmetry-adapted molecular orbitals: bonding (g) and antibonding (u)
    Cmo = np.column_stack([np.array([1, 1]) / np.sqrt(2 + 2 * s),
                           np.array([1, -1]) / np.sqrt(2 - 2 * s)])
    h = Cmo.T @ Hcore @ Cmo
    g = np.einsum('pi,qj,rk,sl,pqrs->ijkl', Cmo, Cmo, Cmo, Cmo, ERI)
    h1, h2 = h[0, 0], h[1, 1]
    J11, J22, J12, K12 = g[0, 0, 0, 0], g[1, 1, 1, 1], g[0, 0, 1, 1], g[0, 1, 0, 1]
    Enuc = 1.0 / R
    # energies of the four occupation patterns |q0 q1>
    E00 = Enuc                                                   # no electrons
    E10 = Enuc + 2 * h1 + J11                                    # sigma_g^2  (HF)
    E01 = Enuc + 2 * h2 + J22                                    # sigma_u^2
    E11 = Enuc + 2 * h1 + 2 * h2 + J11 + J22 + 4 * J12 - 2 * K12  # four electrons
    return ((E00 + E01 + E10 + E11) / 4,      # g0  (identity)
            (E00 + E01 - E10 - E11) / 4,      # g1  (Z0)
            (E00 - E01 + E10 - E11) / 4,      # g2  (Z1)
            (E00 - E01 - E10 + E11) / 4,      # g3  (Z0 Z1)
            K12 / 2,                          # g4  (Y0 Y1)
            K12 / 2)                          # g5  (X0 X1)

R = 0.735
S, Hcore, ERI = ao_integrals(R / BOHR)
print(f"STO-3G integrals for H2 at R = {R} A ({R/BOHR:.5f} Bohr)")
print("-" * 70)
print(f"  overlap      S_AB  = {S[0,1]:+.6f}")
print(f"  core         H_AA  = {Hcore[0,0]:+.6f},  H_AB = {Hcore[0,1]:+.6f}")
print(f"  two-electron (AA|AA) = {ERI[0,0,0,0]:+.6f},  (AA|BB) = {ERI[0,0,1,1]:+.6f}")
print(f"               (AB|AB) = {ERI[0,1,0,1]:+.6f},  (AA|AB) = {ERI[0,0,0,1]:+.6f}")

print("\nCoefficients recomputed from the integrals vs the tabulated values")
print("-" * 70)
TABLE = {0.500: (0.745968, 0.427871, -0.738289, 0.622805, 0.084435, 0.084435),
         0.735: (0.252992, 0.344368, -0.451507, 0.574116, 0.090466, 0.090466),
         1.000: (-0.007740, 0.274331, -0.260726, 0.523311, 0.098395, 0.098395),
         2.000: (-0.272905, 0.134559, 0.013303, 0.389632, 0.129569, 0.129569)}
worst = 0.0
for R, tab in TABLE.items():
    calc = two_qubit_coefficients(R)
    diff = max(abs(a - b) for a, b in zip(calc, tab))
    worst = max(worst, diff)
    print(f"  R = {R:5.3f} A: " + " ".join(f"{v:+9.6f}" for v in calc)
          + f"   max deviation {diff:.1e}")
print(f"\n  largest deviation over all listed bond lengths: {worst:.1e}")
print("  (the table in this chapter is exactly this calculation, rounded to 6 digits)")

print("\nReference values reproduced by the same integrals")
print("-" * 70)
Rs = np.linspace(0.60, 0.90, 61)
Es = []
for R in Rs:
    g0, g1, g2, g3, g4, g5 = two_qubit_coefficients(R)
    E01 = g0 + g1 - g2 - g3
    E10 = g0 - g1 + g2 - g3
    Es.append(0.5 * (E01 + E10) - np.sqrt(0.25 * (E01 - E10) ** 2 + (2 * g4) ** 2))
Es = np.array(Es)
k = int(np.argmin(Es))
print(f"  equilibrium bond length : {Rs[k]:.3f} A   (literature STO-3G/FCI 0.735 A)")
print(f"  minimum energy          : {Es[k]:.6f} Ha  (literature -1.1373 Ha)")
g = two_qubit_coefficients(0.735)
print(f"  Hartree-Fock at 0.735 A : {g[0]-g[1]+g[2]-g[3]:.6f} Ha  "
      "(literature -1.1170 Ha)")
gd = two_qubit_coefficients(3.0)
E01d, E10d = gd[0] + gd[1] - gd[2] - gd[3], gd[0] - gd[1] + gd[2] - gd[3]
Ed = 0.5 * (E01d + E10d) - np.sqrt(0.25 * (E01d - E10d) ** 2 + (2 * gd[4]) ** 2)
print(f"  E(3.0 A)                : {Ed:.6f} Ha  (2 x E(H) = {2*-0.4665818:.6f} Ha)")
```

```text
STO-3G integrals for H2 at R = 0.735 A (1.38895 Bohr)
----------------------------------------------------------------------
  overlap      S_AB  = +0.663146
  core         H_AA  = -1.124218,  H_AB = -0.965257
  two-electron (AA|AA) = +0.774606,  (AA|BB) = +0.571877
               (AB|AB) = +0.300918,  (AA|AB) = +0.447446

Coefficients recomputed from the integrals vs the tabulated values
----------------------------------------------------------------------
  R = 0.500 A: +0.745968 +0.427871 -0.738289 +0.622805 +0.084435 +0.084435   max deviation 4.3e-07
  R = 0.735 A: +0.252992 +0.344368 -0.451507 +0.574116 +0.090466 +0.090466   max deviation 4.0e-07
  R = 1.000 A: -0.007740 +0.274331 -0.260726 +0.523311 +0.098395 +0.098395   max deviation 4.7e-07
  R = 2.000 A: -0.272905 +0.134559 +0.013303 +0.389632 +0.129569 +0.129569   max deviation 4.0e-07

  largest deviation over all listed bond lengths: 4.7e-07
  (the table in this chapter is exactly this calculation, rounded to 6 digits)

Reference values reproduced by the same integrals
----------------------------------------------------------------------
  equilibrium bond length : 0.735 A   (literature STO-3G/FCI 0.735 A)
  minimum energy          : -1.137306 Ha  (literature -1.1373 Ha)
  Hartree-Fock at 0.735 A : -1.116999 Ha  (literature -1.1170 Ha)
  E(3.0 A)                : -0.933632 Ha  (2 x E(H) = -0.933164 Ha)
```

**What to notice.** The recomputed coefficients match the tabulated ones to \\(5 \times 10^{-7}\\), which is exactly the rounding of the table to six decimals. Every number in Section 3.3 is therefore reproducible from three Gaussian exponents and three contraction coefficients — no external quantum-chemistry package, no unexplained constants.

The bottom block closes the loop with the literature. The equilibrium bond length comes out at 0.735 Å, the minimum at \\(-1.137306\\) Hartree, the Hartree-Fock energy at \\(-1.116999\\) Hartree, and at 3.0 Å the energy has fallen to \\(-0.933632\\) Hartree against the exact dissociation limit of \\(2 \times (-0.4665818) = -0.933164\\) Hartree — a residual 0.5 mHartree of spurious binding at that distance. These are the standard STO-3G values for H₂, so the Hamiltonian we handed to VQE is the right one.

* * *

## 3.6 The Limits of VQE

The result of Section 3.5 is exact and, taken alone, misleading. Four things stood in our favour and none of them survives scaling.

### Barren plateaus

For a deep, unstructured ansatz on \\(n\\) qubits with random parameters, the gradient of a typical cost function has zero mean and a variance that **decays exponentially in \\(n\\)**. The state produced by a sufficiently deep random circuit approaches a Haar-random state, and the derivative of a local observable averaged over such states concentrates around zero. Since the number of shots needed to resolve a gradient of size \\(g\\) is \\(O(1/g^2)\\), an exponentially small gradient means an exponentially large measurement cost. Optimisation does not fail loudly; it simply stops making progress.

Code Example 9: Barren Plateaus, Measured

```python
import numpy as np
from qcsim import *

def hardware_efficient(params, n, layers):
    """Layers of Rz-Ry rotations followed by a ring of CNOTs."""
    psi = ket('0' * n)
    k = 0
    for _ in range(layers):
        for q in range(n):
            psi = apply_gate(psi, rz(params[k]), [q], n); k += 1
            psi = apply_gate(psi, ry(params[k]), [q], n); k += 1
        for q in range(n):
            psi = cnot(psi, q, (q + 1) % n, n)
    return psi

def cost(params, n, layers):
    """A single local observable: <Z0 Z1>."""
    psi = hardware_efficient(params, n, layers)
    return expval(psi, 'ZZ' + 'I' * (n - 2))

def grad_component(params, j, n, layers):
    """Parameter-shift derivative with respect to parameter j (shift pi/2 for Rz/Ry)."""
    p = params.copy()
    p[j] += np.pi / 2
    plus = cost(p, n, layers)
    p[j] -= np.pi
    minus = cost(p, n, layers)
    return (plus - minus) / 2

print("Barren plateaus: variance of one gradient component over random parameters")
print("-" * 84)
print(f"  {'qubits':>7} {'layers':>7} {'params':>7} {'mean dE/dtheta':>16} "
      f"{'variance':>12} {'std':>10}")
rng = np.random.default_rng(0)
samples = 120
results = []
for n in range(2, 10):
    layers = 3 * n
    npar = 2 * n * layers
    j = npar // 2
    gs = []
    for _ in range(samples):
        params = rng.uniform(0, 2 * np.pi, npar)
        gs.append(grad_component(params, j, n, layers))
    gs = np.array(gs)
    results.append((n, gs.var()))
    print(f"  {n:7d} {layers:7d} {npar:7d} {gs.mean():16.6f} "
          f"{gs.var():12.3e} {gs.std():10.5f}")

print("\nScaling of the variance")
print("-" * 84)
ns = np.array([r[0] for r in results], dtype=float)
vs = np.array([r[1] for r in results])
slope, intercept = np.polyfit(ns, np.log(vs), 1)
print(f"  fit  log(Var) = {slope:.4f} * n + {intercept:.4f}")
print(f"  i.e. Var ~ {np.exp(slope):.3f}^n : the variance decays by a factor "
      f"{1/np.exp(slope):.2f} per added qubit")
for k in range(len(ns) - 1):
    print(f"    n = {int(ns[k])} -> {int(ns[k+1])}: ratio "
          f"{vs[k+1]/vs[k]:.3f}")
print("\n  With a gradient of typical size sqrt(Var), the number of shots needed to")
print("  resolve its sign grows as 1/Var - exponentially in the number of qubits.")
print(f"  {'qubits':>7} {'typical |grad|':>15} {'shots to resolve':>18}")
for n, v in results:
    print(f"  {n:7d} {np.sqrt(v):15.5f} {1/v:18.0f}")
print("\n  This is why deep, unstructured, 'hardware-efficient' ansaetze do not")
print("  scale, and why chemistry-inspired ansaetze with a good starting point")
print("  (the Hartree-Fock state) are the only route that currently works.")
```

```text
Barren plateaus: variance of one gradient component over random parameters
------------------------------------------------------------------------------------
   qubits  layers  params   mean dE/dtheta     variance        std
        2       6      24         0.011346    1.422e-01    0.37712
        3       9      54         0.023363    4.746e-02    0.21785
        4      12      96         0.009125    3.292e-02    0.18144
        5      15     150         0.006421    1.557e-02    0.12477
        6      18     216         0.002469    7.933e-03    0.08907
        7      21     294        -0.000764    3.124e-03    0.05589
        8      24     384         0.000784    1.600e-03    0.04000
        9      27     486        -0.002680    8.513e-04    0.02918

Scaling of the variance
------------------------------------------------------------------------------------
  fit  log(Var) = -0.7205 * n + -0.6233
  i.e. Var ~ 0.487^n : the variance decays by a factor 2.06 per added qubit
    n = 2 -> 3: ratio 0.334
    n = 3 -> 4: ratio 0.694
    n = 4 -> 5: ratio 0.473
    n = 5 -> 6: ratio 0.510
    n = 6 -> 7: ratio 0.394
    n = 7 -> 8: ratio 0.512
    n = 8 -> 9: ratio 0.532

  With a gradient of typical size sqrt(Var), the number of shots needed to
  resolve its sign grows as 1/Var - exponentially in the number of qubits.
   qubits  typical |grad|   shots to resolve
        2         0.37712                  7
        3         0.21785                 21
        4         0.18144                 30
        5         0.12477                 64
        6         0.08907                126
        7         0.05589                320
        8         0.04000                625
        9         0.02918               1175

  This is why deep, unstructured, 'hardware-efficient' ansaetze do not
  scale, and why chemistry-inspired ansaetze with a good starting point
  (the Hartree-Fock state) are the only route that currently works.
```

**What to notice.** The gradient mean is zero to within sampling error at every size — the landscape has no systematic slope to follow. The variance falls by a factor of about 2 per added qubit, a clean exponential over the eight sizes measured, so the shots needed to determine the *sign* of one gradient component rise from 7 at two qubits to 1175 at nine. Extrapolating the fitted rate, fifty qubits would need roughly \\(10^{15}\\) shots for a single gradient component of a single parameter. At any realistic repetition rate that is longer than a research career.

Three mitigations are under active study, and it is worth being precise about what each buys. **Structured ansätze** (UCC, symmetry-preserving) restrict the circuit to a physically relevant subspace, so the concentration argument does not apply. **Good initialisation** (starting from Hartree-Fock, or layer-by-layer growth as in ADAPT-VQE) keeps the optimiser in a region where gradients are large. **Local cost functions** with shallow circuits have provably milder decay. None of these is a general solution, and no method currently known removes the exponential in the worst case.

### The measurement wall

Section 3.4 measured the constant in the shot-count law: \\(\sigma\sqrt{N} \approx 0.21\\) Hartree for H₂ with six Pauli terms. The number of Pauli terms in a molecular Hamiltonian grows as \\(O(N^4)\\) in the number of spin orbitals, and \\(\sum_j \lvert c_j \rvert\\) grows with it, so the total shots for fixed precision scale roughly as \\(N^4\\) or worse before grouping. Published estimates for industrially relevant molecules land between \\(10^9\\) and \\(10^{13}\\) measurements *per energy evaluation*, with hundreds of evaluations per optimisation. Improved grouping, classical shadows and low-rank factorisations have reduced these numbers substantially, and they remain the dominant cost.

### Noise

Everything above assumed a noiseless circuit. On hardware, each gate contributes error, the state becomes mixed, and the variational bound is no longer a bound. The measured energy acquires a bias that does not shrink with more shots — only with better gates or error mitigation. Chapter 5 quantifies this with an explicit noise model and shows the fidelity decay against circuit depth.

### Accuracy that matters

Chemical accuracy — 1 kcal/mol, or 1.6 mHartree — is the threshold at which computed reaction rates become predictive, since the rate depends exponentially on the barrier. Our VQE reached \\(10^{-15}\\) Hartree against its own Hamiltonian, but that Hamiltonian is 40 mHartree from the true ground state of H₂ because of the basis set. For a real prediction you need a large basis (many more qubits), a converged ansatz (deeper circuits), and error rates far below today's. All three at once is the requirement, and it is why credible roadmaps for quantum advantage in chemistry point at fault-tolerant hardware rather than at NISQ devices.

**What VQE has genuinely established.** It works. The algorithm is correct, the pipeline from integrals to energies is verifiable end to end, and it has been demonstrated on real hardware for small molecules with error mitigation. It is a proof of principle, not yet a tool — and knowing precisely which of the four limits above binds hardest for your problem is the difference between a useful research programme and a press release.

* * *

## Exercises

#### Exercise 1: The Variational Bound, Analytically

Using \\(E(\theta) = A\cos 2\theta + B\sin 2\theta + C\\) with \\(A = (E_{10} - E_{01})/2\\), \\(B = \langle 01 \rvert H \lvert 10 \rangle\\) and \\(C = (E_{10}+E_{01})/2\\): (a) find the stationary points; (b) show that the minimum equals the lower eigenvalue of the \\(2 \times 2\\) block; (c) evaluate at \\(R = 0.735\\) Å and compare with the table.

<details><summary>Solution</summary>
<p>(a) \(dE/d\theta = -2A\sin 2\theta + 2B\cos 2\theta = 0\) gives \(\tan 2\theta = B/A\), so \(\theta^* = \tfrac{1}{2}\mathrm{atan2}(-B, -A)\) selects the minimum (the other root, \(\theta^* + \pi/2\), is the maximum).</p>
<p>(b) At the minimum \(E = C - \sqrt{A^2 + B^2}\). For the block \(\begin{pmatrix} E_{01} & B \\ B & E_{10}\end{pmatrix}\) the eigenvalues are \(\tfrac{1}{2}(E_{01}+E_{10}) \pm \sqrt{\tfrac{1}{4}(E_{01}-E_{10})^2 + B^2}\), which is exactly \(C \pm \sqrt{A^2+B^2}\). The one-parameter ansatz is therefore exact by construction, not by luck.</p>
<p>(c) With \(A = -0.795875\), \(B = +0.180932\), \(C = -0.321124\): \(\theta^* = -0.111769\) and \(E = -0.321124 - \sqrt{0.633417 + 0.032736} = -1.137306\) Ha, matching the table entry and the VQE result to all printed digits.</p>
</details>

#### Exercise 2: Why Hartree-Fock Fails at Dissociation

From the table in Section 3.3, compute the correlation energy at 0.735 Å and at 2.5 Å in kcal/mol, and the Hartree-Fock overlap at both. Explain in terms of the two configurations why restricted Hartree-Fock cannot describe the stretched bond, and name the materials-science analogue.

<details><summary>Solution</summary>
<p>At 0.735 Å: \(E_{\mathrm{corr}} = -1.137306 - (-1.116999) = -0.020307\) Ha \(= -12.74\) kcal/mol, with \(\lvert\langle \mathrm{HF}\rvert\psi_0\rangle\rvert^2 = 0.9876\). At 2.5 Å: \(E_{\mathrm{corr}} = -0.936055 - (-0.702944) = -0.233112\) Ha \(= -146.28\) kcal/mol, with overlap 0.5944.</p>
<p>As \(R\) grows, the bonding and antibonding orbitals become degenerate, so \(E_{01} \to E_{10}\) and the two configurations \(\sigma_g^2\) and \(\sigma_u^2\) contribute almost equally: the exact state approaches \((\lvert 10 \rangle - \lvert 01 \rangle)/\sqrt{2}\). No single determinant can represent an equal-weight two-configuration state, so restricted Hartree-Fock is qualitatively wrong, not merely imprecise. This is <strong>static</strong> (or strong) correlation, as opposed to the dynamic correlation of short-range electron avoidance.</p>
<p>The materials analogue is the Mott insulator: at large lattice spacing (small hopping \(t\) relative to on-site repulsion \(U\)) the Hubbard model has exactly this degenerate two-configuration structure per bond, and mean-field theory — including density functional theory with common functionals — predicts a metal where experiment finds an insulator. Chapter 4 constructs and solves that model.</p>
</details>

#### Exercise 3: Parameter-Shift Rule for a Rotation Gate

Derive the parameter-shift rule for the half-angle convention \\(R_y(\theta) = \exp(-i\theta Y/2)\\) and verify it numerically on the single-qubit cost function \\(E(\theta) = \langle 0 \rvert R_y^\dagger(\theta) Z R_y(\theta) \lvert 0 \rangle\\).

<details><summary>Solution</summary>
<p>With the half-angle generator the energy is \(E(\theta) = A\cos\theta + B\sin\theta + C\) — one Fourier mode at frequency 1 rather than 2 — so \(dE/d\theta = \tfrac{1}{2}[E(\theta+\pi/2) - E(\theta-\pi/2)]\).</p>
<p>For this cost function \(E(\theta) = \cos\theta\) exactly, so the analytic derivative is \(-\sin\theta\). Numerically at \(\theta = 0.3\): the shift rule gives \(-0.29552021\), the finite difference with \(h = 10^{-6}\) gives \(-0.29552021\), and \(-\sin(0.3) = -0.29552021\). At \(\theta = 1.1\) all three give \(-0.89120736\).</p>
<p>The general statement: a gate \(\exp(-i\theta P/2)\) with \(P^2 = I\) has two distinct eigenvalue gaps \(\pm 1\), giving the \(\pi/2\) shift with a factor \(1/2\); the convention \(\exp(-i\theta P)\) doubles the frequency, giving the \(\pi/4\) shift with no prefactor. Mixing the two conventions is a common source of gradients that are wrong by exactly a factor of two.</p>
</details>

#### Exercise 4: Budgeting the Shots

Using the measured constant \\(\sigma\sqrt{N} \approx 0.21\\) Hartree: (a) how many shots per setting are needed for \\(\sigma = 1\\) mHartree? (b) If a converged optimisation needs 60 energy evaluations at 3 settings each, what is the total shot count? (c) At 5000 shots per second, how long does one geometry take, and one 18-point dissociation curve?

<details><summary>Solution</summary>
<p>(a) \(N = (0.21/0.001)^2 \approx 4.4 \times 10^4\) shots per setting.</p>
<p>(b) \(60 \times 3 \times 4.4\times10^4 \approx 7.9 \times 10^6\) shots.</p>
<p>(c) At 5000 shots per second, \(1.6 \times 10^3\) s \(\approx\) 26 minutes per geometry, and about 8 hours for the 18-point curve — for the smallest molecule in chemistry, at a precision that is still 25 times coarser than the \(10^{-15}\) Ha our noiseless simulator achieved for free.</p>
<p>Now scale it: a molecule with 50 spin orbitals has of order \(50^4 \approx 6\times10^6\) Pauli terms, and \(\sum_j \lvert c_j \rvert\) grows roughly linearly in that count, so the shot budget rises by many orders of magnitude even after commuting-group reduction. This calculation, not the qubit count, is why "how many qubits do we need" is the wrong question to ask first.</p>
</details>

#### Exercise 5: A Redundant Parameter

Insert a second \\(R_z\\) into the ansatz so that the circuit uses \\(R_z(2\alpha)\\) followed by \\(R_z(2\beta)\\) on qubit 1. Show numerically that the energy depends only on \\(\alpha + \beta\\), and explain what this implies about the optimisation landscape and about counting parameters in general.

<details><summary>Solution</summary>
<p>Because \(R_z(a)R_z(b) = R_z(a+b)\), the two gates are one gate with parameter \(\alpha+\beta\). Numerically, \(E(-0.05, -0.061769) = E(-0.3, 0.188231) = E(-0.111769, 0) = -1.137306213\) Ha: any pair whose sum is \(\theta^*\) gives the identical energy.</p>
<p>The landscape therefore has a <strong>flat direction</strong> — a continuous valley of exactly degenerate minima along \(\alpha + \beta = \theta^*\). Consequences: gradient-based optimisers see a singular Hessian and converge slowly along the valley; the Fisher information matrix is rank deficient; and the "number of parameters" overstates the expressiveness of the ansatz.</p>
<p>This is not a contrived case. Real hardware-efficient ansätze contain many such redundancies, which is why the <em>effective</em> dimension of an ansatz — measured for instance by the rank of the quantum Fisher information, or by the singular values computed in Code Example 2 — is a better predictor of performance than the parameter count.</p>
</details>

* * *

## Summary

### Key Takeaways

**1. VQE trades circuit depth for repetitions**

  * Phase estimation needs \\(O(2^m)\\) coherent depth; VQE needs many shallow circuits plus a classical optimiser.
  * The variational principle guarantees \\(E(\boldsymbol{\theta}) \geq E_0\\) for exact expectation values of pure states, so every answer is an upper bound and improvement is verifiable.
  * The energy error is second order in the state error: 99% overlap gives about 1% energy error.
  * On hardware, decoherence and readout bias break the bound. A measured energy *below* the exact value is a diagnostic, not a result.

**2. Ansatz design is where the physics enters**

  * Hardware-efficient circuits are shallow and problem-blind; chemistry-inspired circuits (UCC and relatives) respect particle number and spin.
  * For two-qubit H₂ a single double excitation \\(\exp(-i\theta X_0 Y_1)\\) applied to \\(\lvert 10 \rangle\\) spans exactly the two-dimensional physical sector and is therefore exact.
  * Measured reachable dimension: 2 for the chemistry ansatz with 1 parameter, 4 for the generic ansatz with 4 parameters.
  * Compilation: one \\(X\\), four basis changes, two CNOTs, one \\(R_z\\) — eight gates.

**3. Measurement is linear, and expensive**

  * \\(E = \sum_j c_j \langle P_j \rangle\\); each Pauli string needs a basis change plus a \\(Z\\) measurement.
  * Commuting terms share a circuit: six terms, three settings for H₂.
  * The statistical error obeys \\(\sigma\sqrt{N} \approx 0.21\\) Hartree here, so 1 mHartree needs about \\(4 \times 10^4\\) shots per setting.

**4. Gradients are available exactly**

  * For \\(\exp(-i\theta P)\\) with \\(P^2 = I\\): \\(dE/d\theta = E(\theta+\pi/4) - E(\theta-\pi/4)\\), exact, verified to \\(1.6\times10^{-15}\\).
  * For the half-angle convention \\(\exp(-i\theta P /2)\\) the shift is \\(\pi/2\\) with a factor \\(1/2\\).
  * Three evaluations reconstruct the whole one-parameter landscape and its exact minimum (rotosolve).

**5. The H₂ verification**

  * VQE matches exact diagonalisation of the same Hamiltonian to a maximum of \\(1.1\times10^{-15}\\) Hartree across 18 bond lengths.
  * Equilibrium 0.7354 Å and \\(-1.137306\\) Hartree, against STO-3G references 0.735 Å and \\(-1.1373\\) Hartree.
  * The coefficients are reproducible from three Gaussian exponents; the recomputation agrees to \\(5\times10^{-7}\\).
  * Correlation energy grows from \\(-8\\) mHartree at 0.3 Å to \\(-233\\) mHartree at 2.5 Å: static correlation is the failure mode of single-determinant methods, and the reason to care about quantum algorithms.

**6. The limits are measurement and trainability, not qubits**

  * Barren plateaus: gradient variance measured to fall by a factor 2.06 per added qubit for a deep unstructured ansatz.
  * Shot budgets for industrially relevant molecules are estimated at \\(10^9\\) to \\(10^{13}\\) per energy evaluation.
  * Basis-set error (40 mHartree for STO-3G H₂) is chemistry, not algorithm: the quantum computer solves the model you give it.

**Practical implications**

  * Always initialise from Hartree-Fock, and always report the exact or best-known reference alongside the VQE number.
  * Prefer symmetry-preserving ansätze; measure the effective dimension rather than counting parameters.
  * Use parameter-shift gradients, never finite differences, on hardware.
  * Budget shots before qubits: the measurement count is usually the binding constraint.

### Where This Leads

We have solved one molecule in a minimal basis, and the answer was exact — because the ansatz happened to span the exact solution. Chapter 4 removes that luxury. It builds the general machinery: second quantisation and fermionic operators, the Jordan-Wigner transformation that turns them into Pauli strings, and the resulting qubit Hamiltonians for the models that matter in materials science — the transverse-field Ising chain and the Hubbard model. We will construct those Hamiltonians in code, diagonalise them exactly, run VQE against them, and see where a truncated ansatz starts to fall short. That is also where the contrast between VQE and quantum phase estimation, and between NISQ and fault-tolerant computing, becomes concrete.

[← Chapter 2: Quantum Gates and Circuits](<chapter-2.html>) [Chapter 4: Quantum Computing for Chemistry and Materials →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
