---
title: "Chapter 3: Quantum Gates and Circuits"
chapter_title: "Chapter 3: Quantum Gates and Circuits"
subtitle: "From Unitary Matrices to Working Circuits"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/s2oPPaF7a7Q"
    title="Quantum Computing Ch.3: Quantum Gates and Circuits"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-computing-introduction/chapter-3.html>) | Last sync: 2026-08-16

[Quantum Computing Dojo](<../index.html>) > [Introduction to Quantum Computing](<index.html>) > Chapter 3

In Chapter 2 we learned what a qubit is and how superposition and entanglement let a register of \\(n\\) qubits hold \\(2^n\\) complex amplitudes at once. A state by itself computes nothing, though. In this chapter we learn how to **change** a quantum state on purpose: the gates that act on qubits, the circuit diagrams that describe sequences of gates, and a small simulator you can run on your own laptop. The mathematics here is linear algebra with complex numbers, and every claim we make can be checked numerically — which is exactly what we will do at the end.

## 3.1 Gates as Unitary Matrices

A quantum state of \\(n\\) qubits is a vector \\(|\psi\rangle\\) of \\(2^n\\) complex amplitudes with unit length:

\\[ \sum_{x} |\alpha_x|^2 = 1 \\]

Any operation we apply must keep that length equal to 1, because the amplitudes squared are probabilities and probabilities must sum to one. The matrices that preserve length in a complex vector space are exactly the **unitary matrices**: matrices \\(U\\) satisfying

\\[ U^\dagger U = U U^\dagger = I \\]

where \\(U^\dagger\\) is the **conjugate transpose** (transpose the matrix, then take the complex conjugate of every entry). A **quantum gate** is simply a unitary matrix applied to the state vector:

\\[ |\psi'\rangle = U |\psi\rangle \\]

Three consequences follow immediately, and all three are worth internalizing.

**Quantum gates are reversible.** Every unitary has an inverse, namely \\(U^\dagger\\). There is no quantum equivalent of the classical AND gate, which destroys information by mapping two input bits onto one output bit. Whenever a quantum algorithm needs an irreversible-looking classical function, that function must first be rewritten in reversible form.

**Quantum gates are linear.** If \\(U|0\rangle = |a\rangle\\) and \\(U|1\rangle = |b\rangle\\), then \\(U(\alpha|0\rangle + \beta|1\rangle) = \alpha|a\rangle + \beta|b\rangle\\). This is the entire mechanism by which a gate acts on a superposition of \\(2^n\\) basis states "at once" — and also the reason that fact alone gives no speedup, since measurement returns only one outcome.

**A gate acting on a subset of qubits still acts on the whole register.** A one-qubit gate \\(G\\) applied to qubit \\(k\\) of an \\(n\\)-qubit register is the \\(2^n \times 2^n\\) matrix formed by a **Kronecker product** (tensor product) of \\(G\\) with identity matrices on all other qubits:

\\[ I \otimes \cdots \otimes G \otimes \cdots \otimes I \\]

We will build exactly this matrix in code in Section 3.6.

## 3.2 Single-Qubit Gates

### 📚 The Pauli Gates X, Y, Z

The three **Pauli matrices** are the most fundamental single-qubit gates:

\\[ X = \begin{pmatrix} 0 & 1 \\\ 1 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & -i \\\ i & 0 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\\ 0 & -1 \end{pmatrix} \\]

**The \\(X\\) gate** is the quantum NOT gate. It exchanges the two basis states:

\\[ X|0\rangle = |1\rangle, \qquad X|1\rangle = |0\rangle \\]

**The \\(Z\\) gate** is the phase flip. It leaves \\(|0\rangle\\) alone and multiplies \\(|1\rangle\\) by \\(-1\\):

\\[ Z|0\rangle = |0\rangle, \qquad Z|1\rangle = -|1\rangle \\]

Notice that \\(Z\\) does nothing observable to a qubit that is definitely \\(|0\rangle\\) or definitely \\(|1\rangle\\) — the measurement probabilities \\(|\alpha|^2\\) are unchanged. Its effect only becomes visible in superposition, when the relative sign between the two branches matters. This distinction between a **relative phase** (physically meaningful) and a **global phase** (multiplying the whole state by \\(e^{i\theta}\\), physically undetectable) will run through the rest of this series.

All three Pauli matrices square to the identity: \\(X^2 = Y^2 = Z^2 = I\\). Applying \\(X\\) twice returns the original state.

### 📚 The Hadamard Gate

The **Hadamard gate** \\(H\\) is the gate that creates superposition:

\\[ H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\\ 1 & -1 \end{pmatrix} \\]

Acting on the basis states:

\\[ H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}} \equiv |+\rangle, \qquad H|1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}} \equiv |-\rangle \\]

Both \\(|+\rangle\\) and \\(|-\rangle\\) give outcome 0 or 1 with probability 1/2 when measured in the computational basis. They differ only in the relative sign, and that sign is what makes them distinguishable: applying \\(H\\) a second time undoes the first, since \\(H^2 = I\\). So \\(H|+\rangle = |0\rangle\\) and \\(H|-\rangle = |1\rangle\\), recovering the original state exactly. Interference — amplitudes cancelling or reinforcing — is the mechanism, and it is the resource every quantum algorithm in Chapter 4 exploits.

Two useful identities, both verified numerically in Section 3.6:

\\[ HXH = Z, \qquad HZH = X \\]

Sandwiching a gate between Hadamards converts bit flips into phase flips and back.

### 📚 Phase Gates and Rotation Gates

The **S gate** and **T gate** apply finer phase shifts to \\(|1\rangle\\):

\\[ S = \begin{pmatrix} 1 & 0 \\\ 0 & i \end{pmatrix}, \qquad T = \begin{pmatrix} 1 & 0 \\\ 0 & e^{i\pi/4} \end{pmatrix} \\]

with \\(T^2 = S\\) and \\(S^2 = Z\\).

For continuous control we use the **rotation gates**, defined as exponentials of the Pauli matrices:

\\[ R_x(\theta) = e^{-i\theta X/2}, \qquad R_y(\theta) = e^{-i\theta Y/2}, \qquad R_z(\theta) = e^{-i\theta Z/2} \\]

Because each Pauli matrix squares to the identity, these exponentials have closed forms. For example:

\\[ R_y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix} \\]

so that \\(R_y(\theta)|0\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle\\). The name "rotation" comes from the Bloch sphere picture of Chapter 2: \\(R_x, R_y, R_z\\) rotate the Bloch vector by angle \\(\theta\\) about the \\(x\\), \\(y\\), and \\(z\\) axes respectively. Note the factor of two — a \\(2\pi\\) rotation of the Bloch vector corresponds to \\(\theta = 2\pi\\), at which point \\(R_y(2\pi) = -I\\), a global phase.

Rotation gates matter in practice because they are what real hardware natively implements: a microwave or laser pulse of a given duration and phase produces a rotation by a continuously tunable angle. They are also the gates whose angles get optimized in the variational algorithms we will meet in Chapter 5.

## 3.3 Two-Qubit Gates and Universality

### 📚 The CNOT Gate

Single-qubit gates alone can never create entanglement — they act on each qubit separately, so a product state stays a product state. We need at least one gate that couples two qubits. The standard choice is the **CNOT** (controlled-NOT) gate.

CNOT has a **control** qubit and a **target** qubit. It applies \\(X\\) to the target if and only if the control is \\(|1\rangle\\):

\\[ |00\rangle \to |00\rangle, \quad |01\rangle \to |01\rangle, \quad |10\rangle \to |11\rangle, \quad |11\rangle \to |10\rangle \\]

In the ordered basis \\(\\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\\}\\), with the first qubit as control:

\\[ \text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \\\ 0 & 0 & 0 & 1 \\\ 0 & 0 & 1 & 0 \end{pmatrix} \\]

This is a permutation matrix — it just swaps the last two basis states — and permutation matrices are unitary, so CNOT is a legitimate quantum gate. On basis states it behaves exactly like a classical reversible XOR: \\(|a, b\rangle \to |a, a \oplus b\rangle\\). On superpositions it does something with no classical counterpart, as the next section shows.

A word of caution about conventions: whether \\(|q_0 q_1\rangle\\) means qubit 0 is the most significant bit or the least significant one differs between textbooks and between software packages. The matrix above assumes qubit 0 is the leftmost, most significant bit. Getting this wrong is one of the most common sources of confusing simulation results, so it is worth stating your convention explicitly in code — as we do below.

### 📚 Universal Gate Sets

A finite set of gates is called **universal** if any unitary on any number of qubits can be approximated to arbitrary accuracy by a circuit built from those gates alone. This is the quantum analogue of NAND being universal for classical logic.

Two standard facts, which we state without proof:

- **CNOT together with all single-qubit gates is universal.** Any \\(n\\)-qubit unitary decomposes into these building blocks.
- **The finite set \\(\\{H, T, \text{CNOT}\\}\\) is universal** in the approximate sense: any unitary can be approximated to within any desired error \\(\epsilon\\) by a finite circuit of these three gates.

The second fact is the more remarkable one, because it means a *discrete* set of gates suffices — we do not need infinitely precise analog control. The Solovay–Kitaev theorem further guarantees that the number of gates needed grows only polylogarithmically in \\(1/\epsilon\\), so the approximation is efficient.

Universality is a statement about what is *possible*, not about what is *cheap*. A generic \\(n\\)-qubit unitary requires a number of gates exponential in \\(n\\), and finding short circuits for useful operations is a large part of what quantum algorithm design actually consists of.

## 3.4 The Circuit Model

A **quantum circuit** diagram is the standard notation for a quantum program. Its elements are few:

- **Wires** are horizontal lines, one per qubit. A wire does not represent a physical wire; it represents a qubit persisting through time.
- **Time flows left to right.** Gates drawn further right are applied later. This is the opposite of matrix notation, where \\(U_2 U_1 |\psi\rangle\\) means \\(U_1\\) acts first. Reading a circuit and writing the corresponding matrix product requires reversing the order.
- **Boxes** on a wire are single-qubit gates, labelled \\(H\\), \\(X\\), \\(R_y(\theta)\\), and so on.
- **A filled dot connected by a vertical line to a \\(\oplus\\) symbol** is a CNOT: the dot marks the control, the \\(\oplus\\) marks the target.
- **A meter symbol** at the right end is a **measurement**, which converts the qubit into a classical bit. Measurement is not unitary and is not reversible; it is normally the last operation on a wire.

A circuit that produces a Bell state, written in text form:

```
q0: |0> ──[ H ]───■────  measure
                  │
q1: |0> ──────────⊕────  measure
```

Read it as: start both qubits in \\(|0\rangle\\); apply \\(H\\) to qubit 0; apply CNOT with qubit 0 controlling qubit 1; measure both. The corresponding matrix expression, in the reversed order that matrix multiplication requires, is

\\[ |\psi\rangle = \text{CNOT} \cdot (H \otimes I) \cdot |00\rangle \\]

The **circuit depth** is the number of layers of gates that must be applied in sequence — gates acting on disjoint qubits can occupy the same layer. Depth matters enormously on real hardware, because qubits decohere with time, so a shallower circuit of the same gate count is usually a better circuit. We will return to this constraint in Chapter 5.

## 3.5 Worked Example: Building a Bell State

Let us carry out the Bell state circuit by hand, step by step. The goal is the entangled state

\\[ |\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}} \\]

**Step 0 — the initial state.** Both qubits start in \\(|0\rangle\\), so the register is

\\[ |\psi_0\rangle = |00\rangle = \begin{pmatrix} 1 \\\ 0 \\\ 0 \\\ 0 \end{pmatrix} \\]

**Step 1 — Hadamard on qubit 0.** The gate acting on the full register is \\(H \otimes I\\). Since \\(H|0\rangle = (|0\rangle + |1\rangle)/\sqrt{2}\\) and qubit 1 is untouched:

\\[ |\psi_1\rangle = (H \otimes I)|00\rangle = \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right) \otimes |0\rangle = \frac{|00\rangle + |10\rangle}{\sqrt{2}} \\]

As a vector, \\(|\psi_1\rangle = (1/\sqrt{2}, 0, 1/\sqrt{2}, 0)^T\\). At this point the two qubits are still **unentangled**: the state factorizes as \\(|+\rangle \otimes |0\rangle\\).

**Step 2 — CNOT with qubit 0 as control.** Apply CNOT term by term, using linearity:

\\[ \text{CNOT}\frac{|00\rangle + |10\rangle}{\sqrt{2}} = \frac{\text{CNOT}|00\rangle + \text{CNOT}|10\rangle}{\sqrt{2}} = \frac{|00\rangle + |11\rangle}{\sqrt{2}} \\]

because CNOT leaves \\(|00\rangle\\) alone (control is 0) and maps \\(|10\rangle \to |11\rangle\\) (control is 1, so the target flips). The result is \\(|\Phi^+\rangle\\), with vector \\((1/\sqrt{2}, 0, 0, 1/\sqrt{2})^T\\).

**Why this state is entangled.** Suppose it could be written as a product \\((a|0\rangle + b|1\rangle) \otimes (c|0\rangle + d|1\rangle)\\). Expanding gives amplitudes \\(ac, ad, bc, bd\\) for \\(|00\rangle, |01\rangle, |10\rangle, |11\rangle\\). We need \\(ad = 0\\) and \\(bc = 0\\), so either \\(a = 0\\) or \\(d = 0\\), and either \\(b = 0\\) or \\(c = 0\\). Every such choice forces at least one of \\(ac\\) and \\(bd\\) to vanish, contradicting the requirement that both equal \\(1/\sqrt{2}\\). No product decomposition exists, so the state is entangled.

**What measurement gives.** The probabilities are \\(|1/\sqrt{2}|^2 = 1/2\\) for \\(|00\rangle\\), \\(1/2\\) for \\(|11\rangle\\), and 0 for the other two. Each qubit individually looks like a fair coin. But the two coins always agree — measuring qubit 0 as 0 guarantees qubit 1 is 0 as well. That perfect correlation, present regardless of how far apart the qubits are taken, is the signature of entanglement. It does not permit faster-than-light signalling, because the local outcome is random and the correlation is only visible once the two results are compared over a classical channel.

## 3.6 Python: A Minimal Statevector Simulator

Everything above is linear algebra, so we can check all of it with NumPy alone. The simulator below stores the full state vector of \\(2^n\\) complex amplitudes and applies gates as matrix multiplications. This is the most direct possible implementation — deliberately not the fastest — and it is honest about its limits: memory grows as \\(2^n\\), so a laptop handles roughly 25–30 qubits at most. That exponential wall is precisely why we want real quantum hardware.

**Requirements**: Python 3.9+ and NumPy only. No quantum SDK is needed.

### Code Example 1: Bell State from Scratch

```python
"""Minimal statevector simulator for a small quantum register (NumPy only)."""

import numpy as np

# --- Single-qubit gate matrices (2x2, complex) ---
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1],
              [1, 0]], dtype=complex)
Z = np.array([[1, 0],
              [0, -1]], dtype=complex)
H = np.array([[1, 1],
              [1, -1]], dtype=complex) / np.sqrt(2)


def apply_1q(state, gate, target, n_qubits):
    """Apply a 2x2 gate to qubit `target` of an n-qubit state vector.

    Qubit 0 is the leftmost (most significant) bit of the basis label,
    so |q0 q1 ... > maps to index q0*2^(n-1) + q1*2^(n-2) + ...
    """
    op = np.array([[1]], dtype=complex)
    for q in range(n_qubits):
        op = np.kron(op, gate if q == target else I2)
    return op @ state


def cnot_matrix(control, target, n_qubits=2):
    """Build the 2^n x 2^n permutation matrix for a CNOT gate."""
    dim = 2 ** n_qubits
    M = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = [(i >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if bits[control] == 1:              # control is |1> -> flip target
            bits[target] ^= 1
        j = sum(b << (n_qubits - 1 - q) for q, b in enumerate(bits))
        M[j, i] = 1.0                        # column i -> row j
    return M


def probabilities(state):
    """Measurement probabilities of every computational basis state."""
    return np.abs(state) ** 2


# --- Build the Bell state: H on qubit 0, then CNOT(0 -> 1) ---
n = 2
psi = np.zeros(2 ** n, dtype=complex)
psi[0] = 1.0                                  # start in |00>
print("start      :", np.round(psi.real, 4))

psi = apply_1q(psi, H, target=0, n_qubits=n)
print("after H    :", np.round(psi.real, 4))

CNOT = cnot_matrix(control=0, target=1, n_qubits=n)
psi = CNOT @ psi
print("after CNOT :", np.round(psi.real, 4))

labels = ["00", "01", "10", "11"]
print("\nprobabilities:")
for label, p in zip(labels, probabilities(psi)):
    print(f"  |{label}> : {p:.4f}")

# --- Sanity checks ---
print("\nnorm            :", round(float(np.sum(probabilities(psi))), 10))
print("unitary (CNOT)  :", np.allclose(CNOT.conj().T @ CNOT, np.eye(4)))
print("unitary (H)     :", np.allclose(H.conj().T @ H, np.eye(2)))

# --- Simulated measurement statistics ---
rng = np.random.default_rng(0)
shots = 10000
outcomes = rng.choice(4, size=shots, p=probabilities(psi).real)
counts = np.bincount(outcomes, minlength=4)
print("\n10000 shots:")
for label, c in zip(labels, counts):
    print(f"  |{label}> : {c}")
```

**Verified output**:

```
start      : [1. 0. 0. 0.]
after H    : [0.7071 0.     0.7071 0.    ]
after CNOT : [0.7071 0.     0.     0.7071]

probabilities:
  |00> : 0.5000
  |01> : 0.0000
  |10> : 0.0000
  |11> : 0.5000

norm            : 1.0
unitary (CNOT)  : True
unitary (H)     : True

10000 shots:
  |00> : 4990
  |01> : 0
  |10> : 0
  |11> : 5010
```

The intermediate vectors reproduce the hand calculation exactly: \\((1,0,0,0)\\) becomes \\((0.7071, 0, 0.7071, 0)\\) after \\(H\\), then \\((0.7071, 0, 0, 0.7071)\\) after CNOT. The 10000 simulated measurements give roughly 5000 each of \\(|00\rangle\\) and \\(|11\rangle\\) and **exactly zero** counts for \\(|01\rangle\\) and \\(|10\rangle\\) — the perfect correlation is not approximate.

### Code Example 2: Checking Gate Identities

```python
"""Gate algebra checks with NumPy."""

import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def rot(axis, theta):
    """R_axis(theta) = exp(-i * theta/2 * sigma_axis)."""
    sigma = {"x": X, "y": Y, "z": Z}[axis]
    return np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * sigma


print("H X H == Z ?", np.allclose(H @ X @ H, Z))
print("H Z H == X ?", np.allclose(H @ Z @ H, X))
print("H H   == I ?", np.allclose(H @ H, np.eye(2)))
print("T^2   == S ?", np.allclose(T @ T, np.diag([1, 1j])))
print("Rx(pi) == -iX ?", np.allclose(rot("x", np.pi), -1j * X))

# Ry(theta) acting on |0> gives cos(theta/2)|0> + sin(theta/2)|1>
theta = np.pi / 3
ket0 = np.array([1, 0], dtype=complex)
out = rot("y", theta) @ ket0
print(f"Ry(pi/3)|0> = {out[0].real:.4f}|0> + {out[1].real:.4f}|1>")
print(f"expected      {np.cos(theta/2):.4f}|0> + {np.sin(theta/2):.4f}|1>")
```

**Verified output**:

```
H X H == Z ? True
H Z H == X ? True
H H   == I ? True
T^2   == S ? True
Rx(pi) == -iX ? True
Ry(pi/3)|0> = 0.8660|0> + 0.5000|1>
expected      0.8660|0> + 0.5000|1>
```

Note the last check on \\(R_x(\pi)\\). It equals \\(-iX\\), not \\(X\\) — the rotation gate differs from the Pauli gate by the global phase \\(-i\\). Since global phases are unobservable, the two gates are physically equivalent when applied to the entire register, but the distinction matters when a gate is used as the controlled part of a larger operation, where the phase becomes relative rather than global.

### Code Example 3: Running a Circuit as a List of Instructions

The same helpers scale to any number of qubits. Here we describe a circuit as a list of tuples and execute it left to right, exactly as one reads a circuit diagram.

```python
"""Continuation of Code Example 1: run a circuit given as a list of instructions.
Append this to Code Example 1, or re-import I2, X, Z, H, apply_1q and cnot_matrix."""

GATES = {"X": X, "Z": Z, "H": H}


def run_circuit(circuit, n_qubits):
    """circuit: list of ('H', 0) or ('CNOT', 0, 1) tuples, applied left to right."""
    psi = np.zeros(2 ** n_qubits, dtype=complex)
    psi[0] = 1.0
    for op in circuit:
        if op[0] == "CNOT":
            psi = cnot_matrix(op[1], op[2], n_qubits) @ psi
        else:
            psi = apply_1q(psi, GATES[op[0]], op[1], n_qubits)
    return psi


# GHZ state: (|000> + |111>)/sqrt(2)
ghz = run_circuit([("H", 0), ("CNOT", 0, 1), ("CNOT", 1, 2)], n_qubits=3)
for i, amp in enumerate(ghz):
    if abs(amp) > 1e-12:
        print(f"|{i:03b}> : {amp.real:+.4f}")
print("norm:", round(float(np.sum(np.abs(ghz) ** 2)), 10))
```

**Verified output**:

```
|000> : +0.7071
|111> : +0.7071
norm: 1.0
```

Adding one more CNOT extends the two-qubit Bell state into the three-qubit **GHZ state** \\((|000\rangle + |111\rangle)/\sqrt{2}\\), in which all three qubits are perfectly correlated. GHZ states are a standard benchmark for real hardware: preparing one of \\(n\\) qubits with high fidelity is a demanding test of both gate quality and coherence time.

## 🎯 Exercise Problems

1. **Unitarity**: Verify by hand that \\(H^\dagger H = I\\), and confirm that \\(H\\) is both Hermitian and unitary.
2. **Phase invisibility**: Compute the measurement probabilities of \\(Z|+\rangle\\) and of \\(|+\rangle\\). Then compute the probabilities of \\(HZ|+\rangle\\) and \\(H|+\rangle\\). Explain why the phase becomes visible only in the second pair.
3. **Reversed CNOT**: Write the \\(4 \times 4\\) matrix for a CNOT with qubit 1 as control and qubit 0 as target, and check it against the `cnot_matrix` function in Code Example 1.
4. **The other Bell states**: Modify Code Example 1 to prepare \\((|01\rangle + |10\rangle)/\sqrt{2}\\) and \\((|00\rangle - |11\rangle)/\sqrt{2}\\). Hint: insert an \\(X\\) or a \\(Z\\) before the CNOT.
5. **Scaling wall**: Estimate the memory needed to store the state vector of 30, 40, and 50 qubits as complex128 numbers. At what qubit count does a laptop stop being enough?

## Summary

In this chapter, we learned how quantum computation is expressed as circuits of gates. A **quantum gate** is a **unitary matrix**, which makes every gate reversible and length-preserving, and rules out any direct quantum analogue of irreversible classical logic. The **Pauli gates** \\(X\\), \\(Y\\), and \\(Z\\) flip bits and phases; the **Hadamard gate** creates the superposition that all interference-based algorithms depend on; and the **rotation gates** \\(R_x\\), \\(R_y\\), and \\(R_z\\) provide the continuously tunable operations that real hardware implements natively. The **CNOT gate** couples two qubits and is what makes entanglement possible at all, since single-qubit gates alone can never entangle. The set \\(\\{H, T, \text{CNOT}\\}\\) is **universal**, meaning any unitary can be approximated arbitrarily well from these three gates — though universality guarantees only possibility, never efficiency. In the **circuit model**, wires carry qubits, time runs left to right, and measurement terminates a wire by converting it into a classical bit. We built the **Bell state** \\((|00\rangle + |11\rangle)/\sqrt{2}\\) with a single Hadamard followed by a single CNOT, worked through the algebra term by term, proved that the result cannot be factorized into a product state, and then reproduced every intermediate vector numerically in a **statevector simulator** written with NumPy alone.

In the next chapter, we will put these gates to work and study the quantum algorithms that made the field famous — Deutsch–Jozsa, Grover, and Shor — paying close attention to which speedups are exponential, which are merely quadratic, and which problems get no speedup at all.

[← Chapter 2: Qubits, Superposition, and Entanglement](<chapter-2.html>) [Chapter 4: Quantum Algorithms →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
