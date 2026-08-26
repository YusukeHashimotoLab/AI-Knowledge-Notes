---
title: "Chapter 2: From Molecules to Qubits"
chapter_title: "Chapter 2: From Molecules to Qubits"
subtitle: "Second Quantization, the Electronic Hamiltonian, and the Jordan-Wigner Transformation"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/b9YKfFkOeuk"
    title="Quantum Chemistry Ch.2: From Molecules to Qubits"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-chemistry-computing/chapter-2.html>) | Last sync: 2026-08-17

[Quantum Computing Dojo](<../index.html>) > [Quantum Chemistry with Quantum Computers](<index.html>) > Chapter 2

Chapter 1 left us with a problem and a promise. The problem is that the exact electronic wavefunction lives in a space whose dimension explodes; the promise is that a quantum register spans such a space using only one qubit per spin-orbital. What sits between the two is a translation job, and it is not a formality.

Electrons are **fermions**. Swap two of them and the wavefunction changes sign — a physical fact that the Slater determinant of Chapter 1 encoded by brute force, one antisymmetrized product at a time. Qubits have no such property. Qubit 3 and qubit 7 are labelled, distinguishable objects, and nothing in the algebra of Pauli matrices knows about the exchange of identical particles.

This chapter builds the bridge in three steps. First we rewrite quantum chemistry in the language of **second quantization**, where antisymmetry stops being bookkeeping and becomes an algebraic identity. Then we write the electronic Hamiltonian in that language. Finally we map it onto qubits with the **Jordan-Wigner transformation**, and verify numerically that the matrices we get really do behave like fermions.

## 2.1 Second Quantization from Zero

Start by throwing away a habit. In the first-quantized picture you write \\(\psi(\mathbf{r}_1, \mathbf{r}_2, \ldots)\\) — a function that says where *electron 1* is, where *electron 2* is, and so on. That framing is already wrong, because electrons have no identities to assign. All the antisymmetrization machinery exists to undo a labelling you should never have introduced.

Second quantization removes the labels at the start. Fix the \\(2K\\) spin-orbitals from Chapter 1 and think of them as **slots**. Then a configuration is described by saying, for each slot, whether it is occupied:

\\[ |n_1\, n_2\, n_3 \ldots n_{2K}\rangle, \qquad n_i \in \{0, 1\} \\]

This is an **occupation-number state**, and \\(n_i \in \{0,1\}\\) is the Pauli exclusion principle written into the notation rather than imposed on top of it. Nothing here says which electron is where, because that question no longer exists.

The space spanned by all such strings is the **Fock space**. Note its size: \\(2^{2K}\\) states, one per bit string, which is exactly the size of the Hilbert space of \\(2K\\) qubits. The correspondence we need is already visible.

### 📚 Creation and Annihilation Operators

We move between occupation-number states with two operators per slot.

  * \\(a_i^\dagger\\) — the **creation** operator: fills slot \\(i\\), and gives zero if the slot is already occupied.
  * \\(a_i\\) — the **annihilation** operator: empties slot \\(i\\), and gives zero if the slot is already empty.

Applied to the **vacuum** \\(|00\ldots0\rangle\\), products of creation operators build any configuration:

\\[ a_2^\dagger a_1^\dagger |00\ldots0\rangle = |1\,1\,0\ldots0\rangle \times (\text{a sign}) \\]

That parenthetical sign is the whole point, and it is fixed by the defining algebra:

\\[ \{a_i, a_j^\dagger\} = \delta_{ij}, \qquad \{a_i, a_j\} = 0, \qquad \{a_i^\dagger, a_j^\dagger\} = 0 \\]

where \\(\{A, B\} = AB + BA\\) is the **anticommutator**. These three relations *are* the definition of a fermion. Everything else follows from them.

### 📚 Why Antisymmetry Becomes Automatic

Look at what the second relation says when \\(i \neq j\\):

\\[ a_i^\dagger a_j^\dagger = -\,a_j^\dagger a_i^\dagger \\]

Creating two electrons in the opposite order flips the sign of the state. That is exactly the antisymmetry a Slater determinant was built to enforce — except that here nobody enforced it. It is a consequence of the operator algebra, and it holds for every state you can build, automatically.

Now set \\(i = j\\) in the same relation:

\\[ a_i^\dagger a_i^\dagger = -\,a_i^\dagger a_i^\dagger \quad \Longrightarrow \quad (a_i^\dagger)^2 = 0 \\]

You cannot create two electrons in the same spin-orbital. The Pauli exclusion principle is not an extra rule bolted on; it is the \\(i = j\\) case of antisymmetry.

One more operator earns a name. The **number operator**

\\[ \hat{n}_i = a_i^\dagger a_i, \qquad \hat{N} = \sum_i \hat{n}_i \\]

counts occupation. Acting on an occupation-number state, \\(\hat{n}_i\\) returns \\(n_i\\) — the slot's own occupation — and \\(\hat{N}\\) returns the total electron count. Section 2.7 confirms this numerically, and it matters more than it looks: a chemistry Hamiltonian conserves \\(\hat{N}\\), so the physically relevant states occupy only a small corner of Fock space.

## 2.2 The Electronic Hamiltonian

In this language the electronic Hamiltonian of Chapter 1 takes a compact and completely general form:

\\[ \hat{H} = \sum_{pq} h_{pq}\, a_p^\dagger a_q \;+\; \frac{1}{2} \sum_{pqrs} h_{pqrs}\, a_p^\dagger a_q^\dagger a_r a_s \\]

Two terms, and every molecule differs only in the numbers \\(h_{pq}\\) and \\(h_{pqrs}\\). Reading the operator structure tells you the physics.

### 📚 The One-Electron Term

\\(a_p^\dagger a_q\\) removes an electron from spin-orbital \\(q\\) and puts it into spin-orbital \\(p\\). It is a **hop**. The coefficient \\(h_{pq}\\) is a **one-electron integral**, and it collects everything that happens to a single electron on its own: its kinetic energy, and its attraction to all the fixed nuclei.

The diagonal case \\(p = q\\) gives \\(h_{pp}\, \hat{n}_p\\) — an energy cost for merely occupying spin-orbital \\(p\\). The off-diagonal cases connect different orbitals and are what make the ground state a superposition rather than a single configuration.

### 📚 The Two-Electron Term

\\(a_p^\dagger a_q^\dagger a_r a_s\\) removes electrons from two spin-orbitals and creates them in two others. It describes **two electrons interacting and scattering into new orbitals**, and \\(h_{pqrs}\\) — a **two-electron integral** — is the strength of that process, arising entirely from Coulomb repulsion between electrons.

This term is the source of all the difficulty in Chapter 1. Electron-electron repulsion is what correlates the electrons, what makes the ground state multi-determinantal, and what defeats every attempt to solve for one electron at a time. It is also the reason the term count grows: there are on the order of \\(K^4\\) two-electron integrals, so the Hamiltonian of a modest molecule already has a great many terms.

> **Where the integrals come from**
>
> \\(h_{pq}\\) and \\(h_{pqrs}\\) are definite integrals over the chosen basis functions, and computing them is a solved classical problem — a standard quantum chemistry package produces them in a fraction of a second for small molecules. This is worth emphasising because it locates the division of labour precisely. **The integrals are classical input; the ground-state search is the quantum job.** Chapter 4 computes a real set of integrals and feeds them into exactly this expression.

Note what the Hamiltonian does *not* contain: any reference to which electron is which. It is written entirely in terms of slots and the operators that fill and empty them. That is the payoff of second quantization, and it is what makes the next step possible at all.

## 2.3 The Mapping Problem

We now have a Hamiltonian built from \\(a_p^\dagger\\) and \\(a_q\\), and a machine built from qubits. The temptation is to declare victory: occupation numbers are bits, Fock space has \\(2^{2K}\\) states, a register of \\(2K\\) qubits has \\(2^{2K}\\) states, so map \\(|n_1 n_2 \ldots\rangle\\) onto \\(|q_1 q_2 \ldots\rangle\\) and go home.

The *states* do match. The **operators** do not.

Here is the mismatch in one line. Operators acting on different qubits **commute**: \\(X_1 Z_3 = Z_3 X_1\\), always, because they act on different tensor factors and the tensor product does not care about order. But fermionic operators on different modes **anticommute**: \\(a_1 a_3 = -a_3 a_1\\).

So the naive assignment \\(a_j \mapsto \sigma_j^-\\), where \\(\sigma^- = |0\rangle\langle 1|\\) lowers a single qubit, gets the occupation bookkeeping right and the *signs* wrong. And the signs are not decoration: they are the antisymmetry of the electronic wavefunction. Lose them and you are simulating distinguishable particles that happen to obey an occupancy limit — a different physical system with a different ground state.

**The mapping problem** is therefore: find qubit operators that reproduce the fermionic anticommutation relations exactly. Since qubit operators on different sites commute, the extra minus signs must be manufactured *somewhere*. They have to be built into the operators themselves.

## 2.4 The Jordan-Wigner Transformation

The oldest solution is also the most transparent. Assign the modes a fixed order \\(1, 2, \ldots, 2K\\), map each mode to its own qubit, and define

\\[ a_j = \left( \prod_{k<j} Z_k \right) \sigma_j^-, \qquad a_j^\dagger = \left( \prod_{k<j} Z_k \right) \sigma_j^+ \\]

where \\(\sigma^- = |0\rangle\langle 1| = \tfrac{1}{2}(X + iY)\\) empties a qubit and \\(\sigma^+ = |1\rangle\langle 0| = \tfrac{1}{2}(X - iY)\\) fills it.

The local part, \\(\sigma_j^\pm\\), does the obvious job of emptying or filling qubit \\(j\\). The new ingredient is the **Jordan-Wigner string**: a product of \\(Z\\) operators on every qubit *below* \\(j\\) in the chosen ordering.

### 📚 How the String Manufactures the Sign

\\(Z\\) is diagonal, with \\(Z|0\rangle = +|0\rangle\\) and \\(Z|1\rangle = -|1\rangle\\). So the string \\(\prod_{k<j} Z_k\\) multiplies a state by \\((-1)^{m}\\), where \\(m\\) is the number of occupied modes below \\(j\\). It counts electrons to the left and contributes a minus sign for each.

That is precisely the parity factor that appears when you expand a Slater determinant by moving an electron past its neighbours — the sign a determinant gets from row exchanges. Jordan-Wigner takes that bookkeeping out of the wavefunction and puts it into the operators.

The consequence is exactly what we needed: two Jordan-Wigner operators on different sites pick up a relative minus sign when commuted past each other, because one of them carries a \\(Z\\) on the site where the other acts with \\(X\\) or \\(Y\\), and \\(Z\\) anticommutes with both. Section 2.7 verifies every anticommutation relation numerically rather than asking you to take this on faith.

### 📚 The Cost: Locality

Nothing is free. A fermionic operator that acts on a single mode becomes a qubit operator acting on up to \\(j\\) qubits. A hopping term between adjacent modes stays compact; a hopping term between distant modes drags a string of \\(Z\\) operators across everything in between.

This is the **locality cost** of Jordan-Wigner, and it is a practical concern, not an aesthetic one. A Pauli term acting on many qubits takes a deeper circuit to implement and more measurements to estimate, and the string length grows with the number of orbitals. Section 2.7 shows the strings explicitly for a four-mode register.

Alternative mappings exist that trade this cost differently — the **Bravyi-Kitaev transformation** is the best known, storing occupation information in a tree-like structure so that operator weight grows logarithmically rather than linearly in the number of modes; we will not need its details in this series.

## 2.5 A Worked Micro-Example: Two Spin-Orbitals

Abstraction has gone far enough. Let us take the smallest non-trivial system — **two spin-orbitals, one electron** — and carry out the transformation by hand, term by term.

With only two modes, the Hamiltonian of Section 2.2 has no two-electron term (one electron cannot repel itself), so it reduces to

\\[ \hat{H} = h_{00}\, a_0^\dagger a_0 + h_{11}\, a_1^\dagger a_1 + h_{01} \left( a_0^\dagger a_1 + a_1^\dagger a_0 \right) \\]

Jordan-Wigner gives \\(a_0 = \sigma_0^-\\) — mode 0 has nothing below it, so its string is empty — and \\(a_1 = Z_0\, \sigma_1^-\\).

**The number operators.** For mode 0, the string is trivial:

\\[ a_0^\dagger a_0 = \sigma_0^+ \sigma_0^- = |1\rangle\langle 1|_0 = \frac{I - Z_0}{2} \\]

For mode 1 the strings appear twice and cancel, since \\(Z^2 = I\\):

\\[ a_1^\dagger a_1 = (Z_0 \sigma_1^+)(Z_0 \sigma_1^-) = Z_0^2\, \sigma_1^+\sigma_1^- = \frac{I - Z_1}{2} \\]

So occupation maps onto a single \\(Z\\) per qubit. An occupied spin-orbital is a qubit in \\(|1\rangle\\), and the energy of occupying it is read off with a \\(Z\\) measurement.

**The hopping term.** Here the string survives. Using \\(\sigma^+ Z = \sigma^+\\) and \\(Z \sigma^- = \sigma^-\\):

\\[ a_0^\dagger a_1 = \sigma_0^+ Z_0 \sigma_1^- = \sigma_0^+ \sigma_1^-, \qquad a_1^\dagger a_0 = Z_0 \sigma_1^+ \sigma_0^- = \sigma_0^- \sigma_1^+ \\]

and substituting \\(\sigma^\pm = \tfrac{1}{2}(X \mp iY)\\) makes the imaginary parts cancel:

\\[ \sigma_0^+\sigma_1^- + \sigma_0^-\sigma_1^+ = \frac{1}{2}\left( X_0 X_1 + Y_0 Y_1 \right) \\]

**Putting it together**, the entire Hamiltonian becomes a weighted sum of Pauli strings:

\\[ \hat{H} = \frac{h_{00} + h_{11}}{2}\, I \;-\; \frac{h_{00}}{2} Z_0 \;-\; \frac{h_{11}}{2} Z_1 \;+\; \frac{h_{01}}{2} \left( X_0 X_1 + Y_0 Y_1 \right) \\]

Look at what that expression is. It is a constant plus a handful of Pauli terms with real coefficients — structurally identical to the toy Hamiltonian \\(0.4Z + 0.3X\\) that the *Introduction to Quantum Computing* series fed to a variational solver. A molecular Hamiltonian and a textbook qubit Hamiltonian are the same kind of object; molecules simply have more terms. The code in Section 2.7 confirms this derivation matrix element by matrix element.

## 2.6 H₂ in the Minimal Basis

The example that runs through the rest of this series is the hydrogen molecule in a minimal basis set, and now we can size it exactly.

A minimal basis puts one \\(s\\)-type function on each hydrogen atom, so \\(K = 2\\) **spatial orbitals**. Each holds spin-up or spin-down, giving \\(2K = 4\\) **spin-orbitals**. Jordan-Wigner assigns one qubit per spin-orbital, so the molecule maps onto a **four-qubit register**.

Chapter 1's counting applies directly: two electrons in four spin-orbitals gives \\(\binom{4}{2} = 6\\) determinants. The four-qubit register spans \\(2^4 = 16\\) states, so most of the register describes configurations with the wrong number of electrons. Physics ignores them — the Hamiltonian conserves \\(\hat{N}\\) — but the qubits still carry them.

That gap is an opportunity. Symmetries let us shrink the register:

  * **Particle number.** Only the two-electron sector is physical, which cuts 16 states to 6.
  * **Spin projection.** Of those 6, the ones with equal numbers of up-spin and down-spin electrons number \\(2 \times 2 = 4\\); the remaining two have both electrons with the same spin and belong to different \\(S_z\\) sectors.
  * **Point-group and parity symmetries.** Further conserved quantities constrain the physical states more tightly still, and when a symmetry is known in advance, the qubit that merely records it can be removed from the circuit entirely.

Exploiting these reductions is standard practice and is how a four-qubit molecular problem is made to fit on a much smaller device. **Chapter 4 carries out the reduction explicitly**, on real integrals; here it is enough to see where the room to manoeuvre comes from.

## 2.7 Hands-On: Building Fermions Out of Pauli Matrices

This section is the centrepiece of the chapter. We construct the Jordan-Wigner operators for four modes as explicit NumPy matrices and then *test* them — because a mapping that gets the signs wrong is not obviously wrong to the eye, and the anticommutation relations are a complete check.

The code needs only NumPy. Four modes gives \\(16 \times 16\\) matrices, small enough to build by brute force and large enough that the \\(Z\\) strings are non-trivial.

```python
import math
import numpy as np
from functools import reduce

# ---------------------------------------------------------------
# 1. Single-qubit building blocks.
#    Convention: |0> = empty spin-orbital, |1> = occupied.
# ---------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)   # |0><1| : removes the electron

def kron_all(ops):
    """Tensor product of a list of 2x2 matrices, mode 0 leftmost."""
    return reduce(np.kron, ops)

# ---------------------------------------------------------------
# 2. The Jordan-Wigner transformation.
#
#        a_j = ( Z_0 Z_1 ... Z_{j-1} ) sigma^-_j
#
#    The string of Z operators on every LOWER-INDEXED mode is what
#    supplies the minus signs that fermionic antisymmetry demands.
# ---------------------------------------------------------------
def jw_annihilation(j, n_modes):
    ops = [Z] * j + [SIGMA_MINUS] + [I2] * (n_modes - j - 1)
    return kron_all(ops)

def pauli_on(pauli, j, n_modes):
    """Embed a single-qubit Pauli on mode j into the full register."""
    return kron_all([I2] * j + [pauli] + [I2] * (n_modes - j - 1))

M = 4                                   # four spin-orbitals -> four qubits
DIM = 2 ** M
a = [jw_annihilation(j, M) for j in range(M)]
adag = [op.conj().T for op in a]
IDENT = np.eye(DIM, dtype=complex)

print(f"Jordan-Wigner operators for M = {M} modes: each matrix is {DIM} x {DIM}")
print()

# ---------------------------------------------------------------
# 3. THE test: do these matrices actually behave like fermions?
#       {a_i, a_j^dag} = delta_ij * I        {a_i, a_j} = 0
# ---------------------------------------------------------------
def anticommutator(A, B):
    return A @ B + B @ A

worst_ad, worst_aa = 0.0, 0.0
print("{a_i, a_j^dag} - delta_ij * I   (max |entry|, should be 0)")
for i in range(M):
    row = []
    for j in range(M):
        expected = IDENT if i == j else np.zeros((DIM, DIM), dtype=complex)
        dev = np.max(np.abs(anticommutator(a[i], adag[j]) - expected))
        worst_ad = max(worst_ad, dev)
        row.append(f"{dev:8.1e}")
    print("   " + " ".join(row))
print()

print("{a_i, a_j}                      (max |entry|, should be 0)")
for i in range(M):
    row = []
    for j in range(M):
        dev = np.max(np.abs(anticommutator(a[i], a[j])))
        worst_aa = max(worst_aa, dev)
        row.append(f"{dev:8.1e}")
    print("   " + " ".join(row))
print()
print(f"worst deviation, {{a_i, a_j^dag}} : {worst_ad:.3e}")
print(f"worst deviation, {{a_i, a_j}}     : {worst_aa:.3e}")
print(f"a_j^2 = 0 for every j            : "
      f"{all(np.allclose(a[j] @ a[j], 0) for j in range(M))}")
print()

# ---------------------------------------------------------------
# 4. The number operator. n_j = a_j^dag a_j counts mode j;
#    N = sum_j n_j counts electrons. Its eigenvalues must be the
#    integers 0..M, with binomial degeneracies C(M, k).
# ---------------------------------------------------------------
N_op = sum(adag[j] @ a[j] for j in range(M))
eigvals = np.linalg.eigvalsh(N_op)
print("Eigenvalues of the total number operator N = sum_j a_j^dag a_j")
values, counts = np.unique(np.round(eigvals.real, 10), return_counts=True)
for v, c in zip(values, counts):
    print(f"  N = {v:.0f}   degeneracy {c:2d}   (C({M}, {v:.0f}) = {math.comb(M, int(v))})")
print(f"  imaginary part of every eigenvalue: {np.max(np.abs(eigvals.imag)):.1e}")
print()

# n_j is diagonal: JW maps occupation directly onto qubit basis states.
n0_from_jw = adag[0] @ a[0]
n0_from_pauli = 0.5 * (IDENT - pauli_on(Z, 0, M))
print("Is  a_0^dag a_0 == (I - Z_0)/2 ?  "
      f"{np.allclose(n0_from_jw, n0_from_pauli)}")
print()

# ---------------------------------------------------------------
# 5. The two-mode micro-example, checked term by term.
#    H = h00 n_0 + h11 n_1 + h01 (a_0^dag a_1 + a_1^dag a_0)
#    should become
#    H = (h00+h11)/2 I - (h00/2) Z_0 - (h11/2) Z_1
#          + (h01/2) (X_0 X_1 + Y_0 Y_1)
# ---------------------------------------------------------------
m = 2
b = [jw_annihilation(j, m) for j in range(m)]
bdag = [op.conj().T for op in b]
I4 = np.eye(4, dtype=complex)
h00, h11, h01 = -1.25, -0.40, -0.30       # illustrative integrals, not a real molecule

H_fermionic = (h00 * bdag[0] @ b[0] + h11 * bdag[1] @ b[1]
               + h01 * (bdag[0] @ b[1] + bdag[1] @ b[0]))

Z0, Z1 = pauli_on(Z, 0, m), pauli_on(Z, 1, m)
X0X1 = pauli_on(X, 0, m) @ pauli_on(X, 1, m)
Y0Y1 = pauli_on(Y, 0, m) @ pauli_on(Y, 1, m)
H_pauli = ((h00 + h11) / 2 * I4 - h00 / 2 * Z0 - h11 / 2 * Z1
           + h01 / 2 * (X0X1 + Y0Y1))

print("Two modes, one electron: fermionic form vs. hand-derived Pauli form")
print(f"  matrices agree: {np.allclose(H_fermionic, H_pauli)}")
print(f"  max |difference|: {np.max(np.abs(H_fermionic - H_pauli)):.3e}")

print()

# The one-electron sector is spanned by |10> and |01> -- basis indices 2 and 1.
# Its spectrum must equal the eigenvalues of the 2x2 integral matrix.
h_matrix = np.array([[h00, h01], [h01, h11]])
one_particle = np.linalg.eigvalsh(h_matrix)
sector = H_pauli[np.ix_([2, 1], [2, 1])]
sector_spectrum = np.linalg.eigvalsh(sector)

print("Spectra")
print(f"  full 4x4 qubit Hamiltonian : {np.round(np.linalg.eigvalsh(H_pauli).real, 6)}")
print(f"  N = 1 block of the qubit H  : {np.round(sector_spectrum.real, 6)}")
print(f"  2x2 one-electron integrals  : {np.round(one_particle, 6)}")
print(f"  they match: {np.allclose(sector_spectrum, one_particle)}")
print()

# ---------------------------------------------------------------
# 6. The price of Jordan-Wigner: Z strings. A hopping term between
#    two DISTANT modes turns into Pauli strings acting on every
#    mode in between. We read the strings off by projecting.
# ---------------------------------------------------------------
PAULIS = {"I": I2, "X": X, "Y": Y, "Z": Z}

def pauli_decompose(H, n_modes, tol=1e-12):
    """Expand a 2^n x 2^n matrix in the Pauli basis (Tr[P H] / 2^n)."""
    terms = []
    from itertools import product
    for labels in product("IXYZ", repeat=n_modes):
        P = kron_all([PAULIS[c] for c in labels])
        coeff = np.trace(P.conj().T @ H) / (2 ** n_modes)
        if abs(coeff) > tol:
            terms.append(("".join(labels), coeff))
    return terms

for p, q in [(0, 1), (0, 3)]:
    hop = adag[p] @ a[q] + adag[q] @ a[p]
    terms = pauli_decompose(hop, M)
    weights = [sum(1 for c in label if c != "I") for label, _ in terms]
    print(f"a_{p}^dag a_{q} + a_{q}^dag a_{p}  ->  "
          f"{len(terms)} Pauli terms, max weight {max(weights)}")
    for label, coeff in terms:
        print(f"    {coeff.real:+.3f}  {label}")
```

**Output:**

```
Jordan-Wigner operators for M = 4 modes: each matrix is 16 x 16

{a_i, a_j^dag} - delta_ij * I   (max |entry|, should be 0)
    0.0e+00  0.0e+00  0.0e+00  0.0e+00
    0.0e+00  0.0e+00  0.0e+00  0.0e+00
    0.0e+00  0.0e+00  0.0e+00  0.0e+00
    0.0e+00  0.0e+00  0.0e+00  0.0e+00

{a_i, a_j}                      (max |entry|, should be 0)
    0.0e+00  0.0e+00  0.0e+00  0.0e+00
    0.0e+00  0.0e+00  0.0e+00  0.0e+00
    0.0e+00  0.0e+00  0.0e+00  0.0e+00
    0.0e+00  0.0e+00  0.0e+00  0.0e+00

worst deviation, {a_i, a_j^dag} : 0.000e+00
worst deviation, {a_i, a_j}     : 0.000e+00
a_j^2 = 0 for every j            : True

Eigenvalues of the total number operator N = sum_j a_j^dag a_j
  N = 0   degeneracy  1   (C(4, 0) = 1)
  N = 1   degeneracy  4   (C(4, 1) = 4)
  N = 2   degeneracy  6   (C(4, 2) = 6)
  N = 3   degeneracy  4   (C(4, 3) = 4)
  N = 4   degeneracy  1   (C(4, 4) = 1)
  imaginary part of every eigenvalue: 0.0e+00

Is  a_0^dag a_0 == (I - Z_0)/2 ?  True

Two modes, one electron: fermionic form vs. hand-derived Pauli form
  matrices agree: True
  max |difference|: 5.551e-17

Spectra
  full 4x4 qubit Hamiltonian : [-1.65     -1.345216 -0.304784  0.      ]
  N = 1 block of the qubit H  : [-1.345216 -0.304784]
  2x2 one-electron integrals  : [-1.345216 -0.304784]
  they match: True

a_0^dag a_1 + a_1^dag a_0  ->  2 Pauli terms, max weight 2
    +0.500  XXII
    +0.500  YYII
a_0^dag a_3 + a_3^dag a_0  ->  2 Pauli terms, max weight 4
    +0.500  XZZX
    +0.500  YZZY
```

**Reading the result.** Five things happened here, and each is worth pausing on.

  * **The anticommutation relations hold exactly.** Every entry of every deviation matrix is \\(0.0\\) — not \\(10^{-16}\\), but exactly zero, because the Jordan-Wigner matrices have entries drawn from \\(\{0, \pm 1\}\\) and the products are computed without any rounding. We did not assume the transformation was correct; we tested all sixteen pairs of the \\(\{a_i, a_j^\dagger\}\\) relation and all sixteen of \\(\{a_i, a_j\}\\), and they passed. This is what a mapping being *right* looks like.
  * **\\(a_j^2 = 0\\) for every mode.** The Pauli exclusion principle emerges from matrices that were never told about it.
  * **The number operator has integer eigenvalues with binomial degeneracies.** Its spectrum is \\(0, 1, 2, 3, 4\\) with degeneracies \\(1, 4, 6, 4, 1\\) — exactly \\(\binom{4}{k}\\). The degeneracy-6 level at \\(N = 2\\) is Chapter 1's count of two electrons in four spin-orbitals, appearing here as an eigenvalue multiplicity. Fock space really has decomposed into particle-number sectors.
  * **The hand derivation of Section 2.5 was correct.** The fermionic and Pauli forms of the two-mode Hamiltonian agree to \\(5.6 \times 10^{-17}\\), which is floating-point noise from the complex arithmetic in \\(Y \otimes Y\\), not a discrepancy. And the \\(N = 1\\) block of the four-dimensional qubit Hamiltonian reproduces the eigenvalues of the \\(2 \times 2\\) integral matrix exactly — the qubit encoding did not distort the physics, it only embedded it in a larger space.
  * **The \\(Z\\) strings are visible and they cost you.** A hop between *adjacent* modes gives weight-2 terms, \\(\tfrac{1}{2}(X_0X_1 + Y_0Y_1)\\) — exactly the expression derived by hand. A hop between modes 0 and 3 gives \\(\tfrac{1}{2}(X_0 Z_1 Z_2 X_3 + Y_0 Z_1 Z_2 Y_3)\\): the same two terms, now dragging a \\(Z\\) across every mode in between, weight 4 instead of 2. This is the locality cost of Section 2.4 made visible, and it is why the ordering of spin-orbitals is a design decision rather than an arbitrary labelling.

Try changing `SIGMA_MINUS` to act without the \\(Z\\) string — that is, define `jw_annihilation` with `[I2] * j` instead of `[Z] * j` — and re-run. The occupation bookkeeping will still look perfectly reasonable, the number operator will still have the right eigenvalues, and the anticommutation test will fail immediately for every \\(i \neq j\\). That failure is the entire reason the string exists.

### 🎯 Exercise Problems

  1. **Exclusion from algebra.** Starting only from \\(\{a_i^\dagger, a_j^\dagger\} = 0\\), derive \\((a_i^\dagger)^2 = 0\\) and explain in one sentence why this is the Pauli exclusion principle rather than a separate postulate.
  2. **The number operator by hand.** Show that \\(\hat{n}_i = a_i^\dagger a_i\\) satisfies \\(\hat{n}_i^2 = \hat{n}_i\\) using only the anticommutation relations, and state what this implies about its eigenvalues. Confirm your answer against the printed spectrum.
  3. **Extending the micro-example.** Repeat the Section 2.5 derivation for **three** modes, working out the Jordan-Wigner form of \\(a_0^\dagger a_2 + a_2^\dagger a_0\\) by hand. Verify it with the `pauli_decompose` function in the code.
  4. **Ordering matters.** The Jordan-Wigner string length depends on the order in which spin-orbitals are numbered. For a Hamiltonian whose large hopping terms connect modes 0↔1 and 2↔3, and whose small terms connect 0↔3, argue which ordering minimises the total Pauli weight. Then construct a case where the opposite ordering wins.
  5. **Counting the sectors.** For \\(M = 6\\) modes, predict the eigenvalue spectrum and degeneracies of \\(\hat{N}\\) before running anything. Modify the code to check, and confirm the degeneracies sum to \\(2^6\\).

## Summary

This chapter built the bridge from a molecule to a qubit register. **Second quantization** replaces labelled electrons with occupation-number states \\(|n_1 n_2 \ldots\rangle\\) over a fixed set of spin-orbital slots, and replaces antisymmetrization with an algebra: the anticommutation relations \\(\{a_i, a_j^\dagger\} = \delta_{ij}\\) and \\(\{a_i, a_j\} = 0\\) make antisymmetry automatic and deliver the Pauli exclusion principle as the special case \\((a_i^\dagger)^2 = 0\\). In this language the **electronic Hamiltonian** is \\(\hat{H} = \sum h_{pq} a_p^\dagger a_q + \tfrac{1}{2}\sum h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s\\), where the one-electron integrals carry kinetic energy and nuclear attraction while the two-electron integrals carry the electron repulsion responsible for all correlation — and both sets of integrals are classical input, computed on an ordinary computer before the quantum part begins. We then confronted the **mapping problem**: qubit operators on different sites commute, fermionic operators anticommute, so a naive bit-for-occupation assignment loses exactly the signs that encode antisymmetry. The **Jordan-Wigner transformation** repairs this with \\(a_j = (\prod_{k<j} Z_k)\,\sigma_j^-\\), where the \\(Z\\) string counts occupied modes below \\(j\\) and manufactures the parity sign, at the cost of operators whose Pauli weight grows with mode separation; the **Bravyi-Kitaev** mapping trades that cost differently. Working two modes out by hand produced \\(\hat{H} = \tfrac{h_{00}+h_{11}}{2} I - \tfrac{h_{00}}{2}Z_0 - \tfrac{h_{11}}{2}Z_1 + \tfrac{h_{01}}{2}(X_0X_1 + Y_0Y_1)\\) — a weighted sum of Pauli strings, structurally identical to the toy Hamiltonians of the introductory series. For **H₂ in a minimal basis**, 2 spatial orbitals become 4 spin-orbitals and therefore 4 qubits, of whose 16 states only 6 carry two electrons; the symmetries behind that gap are what Chapter 4 exploits to shrink the register. Our NumPy construction then verified the whole scheme: all anticommutation relations satisfied exactly, \\(a_j^2 = 0\\), a number operator with spectrum \\(0..4\\) and degeneracies \\(1, 4, 6, 4, 1\\), the hand derivation confirmed to \\(10^{-17}\\), and the \\(Z\\) strings visible as weight-4 Pauli terms for a distant hop.

We now have a molecular Hamiltonian expressed as a sum of Pauli strings acting on qubits — a form a quantum computer can actually measure. In the next chapter we take the algorithm that turns that object into a number: the Variational Quantum Eigensolver, its ansatz design, the measurement problem hiding inside every energy estimate, and the reasons it is harder in practice than the four-step loop suggests.

[← Chapter 1: Why Chemistry Is the Killer App](<chapter-1.html>) [Chapter 3: VQE: The Algorithm →](<chapter-3.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
