---
title: "Chapter 4: Hands-On: H2 from Scratch"
chapter_title: "Chapter 4: Hands-On: H2 from Scratch"
subtitle: "From Gaussian Basis Functions to a VQE Energy, with Every Number Produced by the Code"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/FkSyO-F2TPI"
    title="Quantum Chemistry Ch.4: Hands-On: H2 from Scratch"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-chemistry-computing/chapter-4.html>) | Last sync: 2026-08-17

[Quantum Computing Dojo](<../index.html>) > [Quantum Chemistry with Quantum Computers](<index.html>) > Chapter 4

Every VQE tutorial you will meet begins with a molecular Hamiltonian that arrived from somewhere else — a library call, a printed table of Pauli coefficients, a file. That is a reasonable way to teach the algorithm and a poor way to understand the calculation, because the interesting engineering is upstream of the qubits. Where do those coefficients come from? What decisions were made before the quantum part started, and which of them cap the accuracy no matter how well the optimizer performs?

This chapter takes the hydrogen molecule apart down to the integrals. Starting from the published STO-3G basis parameters and nothing else, we compute overlap, kinetic, nuclear-attraction and two-electron integrals over Gaussians in closed form, run a Hartree-Fock self-consistent-field loop to convergence, transform to molecular orbitals, build the second-quantized Hamiltonian, map it onto four qubits with the Jordan-Wigner transformation, diagonalize it exactly, and finally run VQE against that exact answer. The tools are `numpy` and `math` — no chemistry package, no SciPy, and specifically no `scipy.special.erf`, because Python's `math.erf` is all the Boys function needs. **Every number printed here comes out of the run**: the reference energies that VQE is checked against are produced by the same code a few sections earlier, which is what makes the check meaningful rather than decorative.

## 4.1 The Plan

Five stages, each producing an artifact the next one consumes.

| Stage | Input | Output |
|---|---|---|
| 1. Basis and integrals | STO-3G parameters, nuclear geometry | \\(S\\), \\(T\\), \\(V\\), \\((\mu\nu\|\lambda\sigma)\\), \\(E_{\text{nuc}}\\) |
| 2. Hartree-Fock SCF | those integrals | MO coefficients \\(C\\), RHF energy |
| 3. MO transformation | \\(C\\) and the AO integrals | \\(h_{pq}\\), \\((pq\|rs)\\) |
| 4. Jordan-Wigner and FCI | the MO integrals | a \\(16 \times 16\\) matrix, the FCI energy |
| 5. VQE | that matrix | a variational energy, checked against stage 4 |

The geometry is fixed throughout: two protons separated by **\\(R = 1.4\\) bohr**, along the \\(z\\) axis. That is a choice, stated here so that every subsequent number is reproducible; it is close to, but not derived from, the equilibrium bond length, and we make no claim about the minimum of the potential energy curve.

## 4.2 Stage 1: The Basis Set and the Integrals

### 📚 The STO-3G Parameters Are a Definition, Not a Result

A **basis set** is a fixed list of functions in which molecular orbitals are expanded. STO-3G is the smallest useful one: each atomic orbital is a **contraction** of three Gaussian primitives, chosen once and published, fitted to approximate a Slater-type orbital. For hydrogen there is exactly one basis function per atom,

\\[ \phi_\mu(\mathbf{r}) = \sum_{k=1}^{3} d_k \left( \frac{2\alpha_k}{\pi} \right)^{3/4} e^{-\alpha_k |\mathbf{r} - \mathbf{R}_\mu|^2} \\]

with the published parameters:

| \\(k\\) | exponent \\(\alpha_k\\) | coefficient \\(d_k\\) |
|---|---|---|
| 1 | 3.42525091 | 0.15432897 |
| 2 | 0.62391373 | 0.53532814 |
| 3 | 0.16885540 | 0.44463454 |

These six numbers are the only quantities in this chapter taken from outside the code: a **definition** of what "STO-3G for hydrogen" means, not a result. Two consequences follow, and both matter more than they look. With one function per atom the molecule has **two spatial orbitals** and therefore **four spin orbitals**, which is why four qubits will suffice. And a minimal basis is a severe approximation: the ceiling on accuracy is set here, before any quantum algorithm is chosen.

### 📚 The Four Integrals, in Closed Form

Gaussians are used instead of the physically better Slater functions for one reason: every integral we need has a closed form. The engine is the **Gaussian product theorem** — the product of two s-type Gaussians centred at \\(\mathbf{A}\\) and \\(\mathbf{B}\\) is a single Gaussian centred between them:

\\[ p = \alpha + \beta, \qquad \mathbf{P} = \frac{\alpha \mathbf{A} + \beta \mathbf{B}}{p}, \qquad K_{AB} = \exp\left( -\frac{\alpha\beta}{p} |\mathbf{A} - \mathbf{B}|^2 \right) \\]

With \\(N_a = (2\alpha/\pi)^{3/4}\\) the normalization of a primitive, the four integrals over normalized s primitives are:

\\[ S_{ab} = N_a N_b \, K_{AB} \left( \frac{\pi}{p} \right)^{3/2} \\]

\\[ T_{ab} = N_a N_b \, \frac{\alpha\beta}{p} \left[ 3 - \frac{2\alpha\beta}{p} |\mathbf{A} - \mathbf{B}|^2 \right] K_{AB} \left( \frac{\pi}{p} \right)^{3/2} \\]

\\[ V_{ab}^{(C)} = -N_a N_b \, Z_C \, \frac{2\pi}{p} \, K_{AB} \, F_0\!\left( p \, |\mathbf{P} - \mathbf{C}|^2 \right) \\]

\\[ (ab|cd) = N_a N_b N_c N_d \, \frac{2\pi^{5/2}}{p\,q\sqrt{p+q}} \, K_{AB} K_{CD} \, F_0\!\left( \frac{pq}{p+q} |\mathbf{P} - \mathbf{Q}|^2 \right) \\]

where \\(q\\) and \\(\mathbf{Q}\\) are the product parameters of the second pair, and two-electron integrals are written in the **chemist convention** \\((ab|cd) = \iint a(1) b(1) \frac{1}{r_{12}} c(2) d(2)\\).

Only one special function appears anywhere. The **Boys function** of order zero,

\\[ F_0(x) = \int_0^1 e^{-x t^2} \, dt = \frac{1}{2}\sqrt{\frac{\pi}{x}}\,\mathrm{erf}\!\left(\sqrt{x}\right), \qquad F_0(0) = 1 \\]

is what the Coulomb singularity \\(1/r\\) turns into once the Gaussian integrals are done. Note the removable singularity at \\(x = 0\\): the closed form divides by \\(\sqrt{x}\\), so the code must switch to the limiting series \\(F_0(x) \approx 1 - x/3\\) for small argument.

Each **contracted** integral is the corresponding primitive integral summed over all combinations of primitives, weighted by the contraction coefficients — three primitives per function means \\(3^2 = 9\\) primitive terms per overlap element and \\(3^4 = 81\\) per two-electron integral.

```python
import math
import numpy as np
from itertools import product

np.set_printoptions(precision=6, suppress=True)

# The STO-3G parameters for hydrogen: the published basis-set DEFINITION.
EXP_H = np.array([3.42525091, 0.62391373, 0.16885540])
COEF_H = np.array([0.15432897, 0.53532814, 0.44463454])

R_BOHR = 1.4                                    # chosen H-H separation, in bohr
NUC = [(1.0, np.array([0.0, 0.0, 0.0])), (1.0, np.array([0.0, 0.0, R_BOHR]))]
BASIS = [(EXP_H, COEF_H, NUC[0][1]), (EXP_H, COEF_H, NUC[1][1])]
NBF = len(BASIS)

def F0(x):                                      # Boys function, math.erf only
    return 1.0 - x / 3.0 if x < 1e-12 else 0.5 * math.sqrt(math.pi / x) * math.erf(math.sqrt(x))

def Ns(a):                                      # normalization of an s primitive
    return (2.0 * a / math.pi) ** 0.75

def gp(a, A, b, B):                             # Gaussian product theorem
    p = a + b
    return p, (a * A + b * B) / p, math.exp(-a * b / p * float(np.dot(A - B, A - B)))

def s_prim(a, A, b, B):
    p, _, K = gp(a, A, b, B)
    return Ns(a) * Ns(b) * K * (math.pi / p) ** 1.5

def t_prim(a, A, b, B):
    p, _, K = gp(a, A, b, B)
    mu, AB2 = a * b / p, float(np.dot(A - B, A - B))
    return Ns(a) * Ns(b) * mu * (3.0 - 2.0 * mu * AB2) * K * (math.pi / p) ** 1.5

def v_prim(a, A, b, B, Zc, C):
    p, P, K = gp(a, A, b, B)
    return -Ns(a) * Ns(b) * Zc * (2.0 * math.pi / p) * K * F0(p * float(np.dot(P - C, P - C)))

def eri_prim(a, A, b, B, c, C, d, D):
    p, P, Kab = gp(a, A, b, B)
    q, Q, Kcd = gp(c, C, d, D)
    pref = 2.0 * math.pi ** 2.5 / (p * q * math.sqrt(p + q))
    return (Ns(a) * Ns(b) * Ns(c) * Ns(d) * pref * Kab * Kcd
            * F0(p * q / (p + q) * float(np.dot(P - Q, P - Q))))

def contract2(fn):                              # contract a 2-index integral
    M = np.zeros((NBF, NBF))
    for i, (ea, ca, A) in enumerate(BASIS):
        for j, (eb, cb, B) in enumerate(BASIS):
            M[i, j] = sum(wa * wb * fn(a, A, b, B)
                          for a, wa in zip(ea, ca) for b, wb in zip(eb, cb))
    return M

S, T = contract2(s_prim), contract2(t_prim)
V = sum(contract2(lambda a, A, b, B, Z=Z, C=C: v_prim(a, A, b, B, Z, C)) for Z, C in NUC)
H_core = T + V

ERI = np.zeros((NBF,) * 4)
for i, j, k, l in product(range(NBF), repeat=4):
    (ea, ca, A), (eb, cb, B) = BASIS[i], BASIS[j]
    (ec, cc, C), (ed, cd, D) = BASIS[k], BASIS[l]
    ERI[i, j, k, l] = sum(wa * wb * wc * wd * eri_prim(a, A, b, B, c, C, d, D)
                          for a, wa in zip(ea, ca) for b, wb in zip(eb, cb)
                          for c, wc in zip(ec, cc) for d, wd in zip(ed, cd))

E_nuc = NUC[0][0] * NUC[1][0] / float(np.linalg.norm(NUC[0][1] - NUC[1][1]))

for name, M in [("S", S), ("T", T), ("V", V), ("H_core = T + V", H_core)]:
    print(f"{name:16s} = [{M[0,0]:+.6f} {M[0,1]:+.6f} ; {M[1,0]:+.6f} {M[1,1]:+.6f}]")
print(f"\n(00|00) = {ERI[0,0,0,0]:.9f}   (00|11) = {ERI[0,0,1,1]:.9f}")
print(f"(01|01) = {ERI[0,1,0,1]:.9f}   (00|01) = {ERI[0,0,0,1]:.9f}")
print(f"\nS symmetric: {np.allclose(S, S.T)}   "
      f"max |S_ii - 1| = {np.max(np.abs(np.diag(S) - 1.0)):.2e}   "
      f"(ij|kl) = (kl|ij): {np.allclose(ERI, ERI.transpose(2, 3, 0, 1))}")
print(f"nuclear repulsion Z_A Z_B / R = {E_nuc:.9f} hartree")
```

**Output:**

```
S                = [+1.000000 +0.659318 ; +0.659318 +1.000000]
T                = [+0.760032 +0.236455 ; +0.236455 +0.760032]
V                = [-1.880441 -1.194835 ; -1.194835 -1.880441]
H_core = T + V   = [-1.120409 -0.958380 ; -0.958380 -1.120409]

(00|00) = 0.774605930   (00|11) = 0.569675915
(01|01) = 0.297028535   (00|01) = 0.444107650

S symmetric: True   max |S_ii - 1| = 9.11e-09   (ij|kl) = (kl|ij): True
nuclear repulsion Z_A Z_B / R = 0.714285714 hartree
```

**Reading the result.** The overlap between the two hydrogen 1s functions is \\(0.659318\\) — large, because at \\(1.4\\) bohr the atoms are close enough to bond. The diagonal of \\(S\\) equals 1 to within \\(9 \times 10^{-9}\\), a check rather than an input: the published contraction coefficients are supposed to produce normalized functions, and they do, to the precision at which they were tabulated. Nuclear attraction is negative everywhere and dominates the positive kinetic energy, so \\(H^{\text{core}}\\) is negative; the eight-fold permutational symmetry of the two-electron integrals holds exactly.

## 4.3 Stage 2: Restricted Hartree-Fock

Hartree-Fock replaces electron-electron repulsion by an average field: each electron moves in the mean field of the others. For a closed-shell molecule with \\(2n\\) electrons in \\(n\\) doubly occupied spatial orbitals, this is **restricted** Hartree-Fock (RHF). H₂ has \\(n = 1\\).

The orbitals are expanded in the basis, \\(\psi_i = \sum_\mu C_{\mu i}\phi_\mu\\), and the variational condition becomes the **Roothaan equation** \\(\mathbf{F}\mathbf{C} = \mathbf{S}\mathbf{C}\boldsymbol{\varepsilon}\\) — a *generalized* eigenvalue problem, because the basis is not orthogonal. It is reduced to an ordinary one by the symmetric orthogonalizer \\(\mathbf{X} = \mathbf{S}^{-1/2}\\), computed from the eigendecomposition of \\(\mathbf{S}\\), after which \\(\mathbf{F}' = \mathbf{X}^{T}\mathbf{F}\mathbf{X}\\) is diagonalized and \\(\mathbf{C} = \mathbf{X}\mathbf{C}'\\). The equation is **nonlinear** — the Fock matrix depends on the orbitals it determines — hence a self-consistent-field loop over the density matrix \\(P_{\mu\nu} = 2\sum_{i}^{\text{occ}} C_{\mu i} C_{\nu i}\\) and

\\[ F_{\mu\nu} = H^{\text{core}}_{\mu\nu} + \sum_{\lambda\sigma} P_{\lambda\sigma} \left[ (\mu\nu|\lambda\sigma) - \frac{1}{2}(\mu\lambda|\nu\sigma) \right], \qquad E_{\text{elec}} = \frac{1}{2}\sum_{\mu\nu} P_{\mu\nu}\left( H^{\text{core}}_{\mu\nu} + F_{\mu\nu} \right) \\]

with \\(E_{\text{RHF}} = E_{\text{elec}} + E_{\text{nuc}}\\). The two terms in the Fock matrix are the **Coulomb** repulsion and the **exchange** interaction, the latter a consequence of antisymmetry with no classical counterpart, carrying the factor \\(\frac{1}{2}\\) because exchange acts only between electrons of the same spin.

```python
sv, sc = np.linalg.eigh(S)
X = sc @ np.diag(1.0 / np.sqrt(sv)) @ sc.T      # symmetric orthogonalizer S^(-1/2)
N_OCC = 1                                       # 2 electrons, 2 per spatial orbital

def fock(P):
    F = H_core.copy()
    for m, n in product(range(NBF), repeat=2):
        F[m, n] += float(np.sum(P * (ERI[m, n] - 0.5 * ERI[m, :, n, :])))
    return F

# A deliberately lopsided starting guess: both electrons parked on atom A.
P, E_elec = np.array([[2.0, 0.0], [0.0, 0.0]]), 0.0
print(f"X^T S X = I: {np.allclose(X.T @ S @ X, np.eye(NBF))}   "
      f"overlap eigenvalues = {sv}\n")
print("  iter          E_elec (hartree)        change     max |dP|")
for it in range(1, 51):
    eps, Co = np.linalg.eigh(X.T @ fock(P) @ X)
    C = X @ Co
    P_new = 2.0 * (C[:, :N_OCC] @ C[:, :N_OCC].T)
    E_new = 0.5 * float(np.sum(P_new * (H_core + fock(P_new))))
    dE, dP = E_new - E_elec, float(np.max(np.abs(P_new - P)))
    print(f"  {it:4d}   {E_new:+22.12f}   {dE:+11.3e}   {dP:.3e}")
    P, E_elec = P_new, E_new
    if abs(dE) < 1e-12 and dP < 1e-12:
        break

E_rhf = E_elec + E_nuc
print(f"\nFock hermitian: {np.allclose(fock(P), fock(P).T)}   "
      f"C^T S C = I: {np.allclose(C.T @ S @ C, np.eye(NBF))}   "
      f"orbital energies = {eps}")
print(f"MO coefficients C (columns are the molecular orbitals) =")
print(C)
print(f"\nelectronic energy = {E_elec:+.9f}   nuclear repulsion = {E_nuc:+.9f}")
print(f"TOTAL RHF ENERGY  = {E_rhf:+.9f} hartree")
```

**Output:**

```
X^T S X = I: True   overlap eigenvalues = [0.340682 1.659318]

  iter          E_elec (hartree)        change     max |dP|
     1          -1.827181194524    -1.827e+00   1.284e+00
     2          -1.830964837516    -3.784e-03   1.028e-01
     3          -1.830999715265    -3.488e-05   9.533e-03
     4          -1.831000036365    -3.211e-07   9.115e-04
     5          -1.831000039321    -2.956e-09   8.743e-05
     6          -1.831000039348    -2.722e-11   8.389e-06
     7          -1.831000039348    -2.507e-13   8.049e-07
     8          -1.831000039348    -1.998e-15   7.723e-08
     9          -1.831000039348    -2.220e-16   7.410e-09
    10          -1.831000039348    +0.000e+00   7.110e-10
    11          -1.831000039348    -2.220e-16   6.822e-11
    12          -1.831000039348    +2.220e-16   6.546e-12
    13          -1.831000039348    +0.000e+00   6.279e-13

Fock hermitian: True   C^T S C = I: True   orbital energies = [-0.578203  0.670268]
MO coefficients C (columns are the molecular orbitals) =
[[-0.548934 -1.211464]
 [-0.548934  1.211464]]

electronic energy = -1.831000039   nuclear repulsion = +0.714285714
TOTAL RHF ENERGY  = -1.116714325 hartree
```

**Reading the result.** Three points.

  * **The convergence is monotone and roughly linear.** Starting from a deliberately lopsided guess — both electrons parked on one atom — the energy change falls by about two orders of magnitude per iteration until it reaches machine precision at iteration 8. The density settles more slowly than the energy, which is the usual pattern: the energy is stationary at the solution, so first-order errors in the density cost only second order in the energy. That is the variational principle of Chapter 3 appearing in a purely classical calculation.
  * **Symmetry is recovered, not imposed.** The occupied orbital converges to \\(\pm 0.548934\\) on both atoms — an equal, symmetric, bonding combination — while the empty one has opposite signs on the two centres, with orbital energies \\(-0.578203\\) and \\(+0.670268\\) hartree. Nothing in the code enforced this; the lopsided guess was simply wrong and the iteration corrected it.
  * **The total RHF energy is \\(-1.116714325\\) hartree**, made of \\(-1.831000039\\) electronic and \\(+0.714285714\\) nuclear repulsion. This is the number the quantum calculation must improve on.

## 4.4 Stage 3: Into the Molecular-Orbital Basis

The Hamiltonian we want to put on qubits is written in second quantization, in terms of orbitals rather than of basis functions. That requires transforming the integrals with the coefficients we just obtained:

\\[ h_{pq} = \sum_{\mu\nu} C_{\mu p} H^{\text{core}}_{\mu\nu} C_{\nu q}, \qquad (pq|rs) = \sum_{\mu\nu\lambda\sigma} C_{\mu p} C_{\nu q} C_{\lambda r} C_{\sigma s} \, (\mu\nu|\lambda\sigma) \\]

The second is the notorious **four-index transformation**. Done naively it costs \\(O(N^8)\\); done as four successive one-index contractions it costs \\(O(N^5)\\), which is what `np.einsum` with `optimize=True` arranges here. With \\(N = 2\\) the distinction is academic, but it is one of the reasons a "small" molecule is not small.

The electronic Hamiltonian is then

\\[ \hat{H} = \sum_{pq} \sum_{\sigma} h_{pq}\, a^{\dagger}_{p\sigma} a_{q\sigma} \;+\; \frac{1}{2}\sum_{pqrs} \sum_{\sigma\tau} (pq|rs)\, a^{\dagger}_{p\sigma} a^{\dagger}_{r\tau} a_{s\tau} a_{q\sigma} \;+\; E_{\text{nuc}} \\]

where \\(p, q, r, s\\) run over the two **spatial** orbitals and \\(\sigma, \tau\\) over spin. Chapter 2 wrote the same operator with physicist-notation integrals \\(h_{pqrs}\\); the chemist notation used here is the same object with the four indices paired differently, which is why the creation and annihilation operators do not appear in the same order as the labels.

```python
h_mo = C.T @ H_core @ C
eri_mo = np.einsum('ip,jq,kr,ls,ijkl->pqrs', C, C, C, C, ERI, optimize=True)

print("h_pq =")
print(h_mo)
print(f"(00|00) = {eri_mo[0,0,0,0]:.9f}   (00|11) = {eri_mo[0,0,1,1]:.9f}")
print(f"(01|01) = {eri_mo[0,1,0,1]:.9f}   (11|11) = {eri_mo[1,1,1,1]:.9f}")
print(f"(00|01) = {eri_mo[0,0,0,1]:.9f}   (vanishes by symmetry)")

E_check = 2.0 * h_mo[0, 0] + eri_mo[0, 0, 0, 0] + E_nuc
print(f"\nRHF energy rebuilt from the MO integrals = {E_check:+.9f} hartree")
print(f"agrees with the SCF total: {abs(E_check - E_rhf) < 1e-10}   "
      f"(difference {abs(E_check - E_rhf):.2e})")
```

**Output:**

```
h_pq =
[[-1.252797  0.      ]
 [ 0.       -0.475602]]
(00|00) = 0.674594084   (00|11) = 0.663563991
(01|01) = 0.181257915   (11|11) = 0.697495347
(00|01) = 0.000000000   (vanishes by symmetry)

RHF energy rebuilt from the MO integrals = -1.116714325 hartree
agrees with the SCF total: True   (difference 4.44e-16)
```

**Reading the result.** The one-electron matrix is diagonal, \\(h_{00} = -1.252797\\) and \\(h_{11} = -0.475602\\) hartree, because the molecular orbitals diagonalize the Fock matrix and, for H₂, the symmetry that makes them bonding and antibonding also makes them eigenfunctions of the core Hamiltonian; \\((00|01)\\) vanishes for the same reason, since it would couple a gerade and an ungerade orbital. The last two lines are a **cross-check worth its cost**: rebuilding the RHF energy from the transformed integrals alone, \\(E_{\text{RHF}} = 2h_{00} + (00|00) + E_{\text{nuc}}\\), reproduces the SCF total to \\(4 \times 10^{-16}\\) hartree. An error in the four-index transformation is easy to make and silent — this catches it.

## 4.5 Stage 4: Jordan-Wigner and Exact Diagonalization

Four spin orbitals become four qubits. Ordering them as \\(0\alpha, 0\beta, 1\alpha, 1\beta\\) on qubits \\(0,1,2,3\\), the **Jordan-Wigner transformation** represents each fermionic annihilation operator as

\\[ a_p = \left( \bigotimes_{q < p} Z_q \right) \sigma^-_p, \qquad \sigma^- = |0\rangle\langle 1| \\]

The lowering operator does the physical work — it empties an occupied spin orbital — and the string of \\(Z\\) operators on all lower-indexed qubits supplies the fermionic minus sign. Without it, the operators would commute like bosons; with it, they anticommute, and the code verifies \\(\{a_p, a_q^\dagger\} = \delta_{pq}\\) numerically before building anything.

Substituting these matrices into the second-quantized Hamiltonian gives an explicit \\(16 \times 16\\) matrix. We can then do what a real calculation cannot: diagonalize it. Its lowest eigenvalue in the neutral-singlet sector is the **full configuration interaction (FCI)** energy — exact within this basis set, and the target VQE will be measured against.

```python
N_SO, DIM = 4, 16
Zp = np.array([[1.0, 0.0], [0.0, -1.0]])
LOW = np.array([[0.0, 1.0], [0.0, 0.0]])            # sigma^- = |0><1|

def jw(p):                                          # a_p = Z_0 ... Z_{p-1} sigma^-_p
    op = np.array([[1.0]])
    for q in range(N_SO):
        op = np.kron(op, Zp if q < p else (LOW if q == p else np.eye(2)))
    return op

def so(p, sigma):                                   # 0a, 0b, 1a, 1b -> qubits 0..3
    return 2 * p + sigma

a = [jw(p) for p in range(N_SO)]
ad = [op.T for op in a]
err = max(float(np.max(np.abs(a[p] @ ad[q] + ad[q] @ a[p]
                              - (np.eye(DIM) if p == q else 0.0))))
          for p, q in product(range(N_SO), repeat=2))
print(f"max deviation in the anticommutator (a_p, a_q^dag) = delta_pq : {err:.2e}")

H_q = E_nuc * np.eye(DIM)
for (p, q), sg in product(product(range(2), repeat=2), (0, 1)):
    H_q += h_mo[p, q] * (ad[so(p, sg)] @ a[so(q, sg)])
for (p, q, r, s), (sg, tau) in product(product(range(2), repeat=4),
                                       product((0, 1), repeat=2)):
    H_q += 0.5 * eri_mo[p, q, r, s] * (ad[so(p, sg)] @ ad[so(r, tau)]
                                       @ a[so(s, tau)] @ a[so(q, sg)])

HF, DOUBLE = 0b1100, 0b0011
print(f"H is {H_q.shape[0]}x{H_q.shape[1]}, hermitian: {np.allclose(H_q, H_q.T)}")
print(f"<1100|H|1100> = {H_q[HF, HF]:+.9f}   equals the SCF total: "
      f"{abs(H_q[HF, HF] - E_rhf) < 1e-10}")

evals = np.linalg.eigvalsh(H_q)
print("\nfull 16x16 spectrum (hartree)")
for k in range(0, DIM, 4):
    print("  " + "  ".join(f"{v:+14.9f}" for v in evals[k:k + 4]))

sectors = {}
for i in range(DIM):
    o = [(i >> (N_SO - 1 - q)) & 1 for q in range(N_SO)]
    sectors.setdefault((sum(o), o[0] + o[2] - o[1] - o[3]), []).append(i)

print("\n   N   2*Sz   dim      lowest energy (hartree)")
low, allv = {}, []
for key in sorted(sectors):
    vals = np.linalg.eigvalsh(H_q[np.ix_(sectors[key], sectors[key])])
    low[key] = float(vals[0])
    allv.extend(vals.tolist())
    print(f"  {key[0]:2d}   {key[1]:+4d}   {len(sectors[key]):3d}   {vals[0]:+22.9f}")
print(f"\nsector eigenvalues reproduce the full spectrum: "
      f"{np.allclose(np.sort(np.array(allv)), evals)}")

E_fci = low[(2, 0)]
print(f"\nTOTAL RHF ENERGY   = {E_rhf:+.9f} hartree")
print(f"FCI GROUND ENERGY  = {E_fci:+.9f} hartree")
print(f"CORRELATION ENERGY = {E_fci - E_rhf:+.9f} hartree")
print(f"FCI <= RHF: {E_fci <= E_rhf}   "
      f"FCI is the global minimum of the 16x16: {abs(E_fci - evals[0]) < 1e-12}")

dets = [0b1100, 0b1001, 0b0110, 0b0011]
print("\nthe N=2, Sz=0 block in the basis |1100>, |1001>, |0110>, |0011>")
print(H_q[np.ix_(dets, dets)])
print(f"<1100|H|1001> = {H_q[HF, 0b1001]:+.2e}      "
      f"<1100|H|0011> = {H_q[HF, DOUBLE]:+.9f}")

# How many Pauli strings does this Hamiltonian actually contain?
PAULI = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]], dtype=complex),
         'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.array([[1, 0], [0, -1]], dtype=complex)}
n_terms = 0
for st in product('IXYZ', repeat=N_SO):
    M = np.array([[1.0]], dtype=complex)
    for ch in st:
        M = np.kron(M, PAULI[ch])
    if abs(complex(np.trace(M.conj().T @ H_q)) / DIM) > 1e-12:
        n_terms += 1
print(f"\nnonzero Pauli strings out of the 256 four-qubit strings: {n_terms}")
```

**Output:**

```
max deviation in the anticommutator (a_p, a_q^dag) = delta_pq : 0.00e+00
H is 16x16, hermitian: True
<1100|H|1100> = -1.116714325   equals the SCF total: True

full 16x16 spectrum (hartree)
    -1.137275944    -0.538511348    -0.538511348    -0.531807570
    -0.531807570    -0.531807570    -0.446446557    -0.446446557
    -0.169291741    +0.238683415    +0.238683415    +0.353649468
    +0.353649468    +0.481138081    +0.714285714    +0.921316558

   N   2*Sz   dim      lowest energy (hartree)
   0     +0     1             +0.714285714
   1     -1     2             -0.538511348
   1     +1     2             -0.538511348
   2     -2     1             -0.531807570
   2     +0     4             -1.137275944
   2     +2     1             -0.531807570
   3     -1     2             -0.446446557
   3     +1     2             -0.446446557
   4     +0     1             +0.921316558

sector eigenvalues reproduce the full spectrum: True

TOTAL RHF ENERGY   = -1.116714325 hartree
FCI GROUND ENERGY  = -1.137275944 hartree
CORRELATION ENERGY = -0.020561619 hartree
FCI <= RHF: True   FCI is the global minimum of the 16x16: True

the N=2, Sz=0 block in the basis |1100>, |1001>, |0110>, |0011>
[[-1.116714  0.       -0.        0.181258]
 [ 0.       -0.35055  -0.181258  0.      ]
 [-0.       -0.181258 -0.35055  -0.      ]
 [ 0.181258  0.       -0.        0.460576]]
<1100|H|1001> = +2.80e-14      <1100|H|0011> = +0.181257915

nonzero Pauli strings out of the 256 four-qubit strings: 15
```

**Reading the result.** Five points, and the second is the one to remember.

  * **The Hartree-Fock determinant is on the diagonal.** \\(\langle 1100|\hat{H}|1100\rangle = -1.116714325\\) hartree, identical to the SCF total of stage 2. Two independent routes — an SCF loop over density matrices, and a matrix element of a qubit operator — produce the same number, which validates the whole chain from integrals to Jordan-Wigner. The Hamiltonian is also **block diagonal in particle number and spin**: each sector's eigenvalues, computed separately, reassemble into the full spectrum exactly, a consequence of \\(\hat{H}\\) commuting with the number and \\(S_z\\) operators, and the structural fact stage 5 exploits.
  * **FCI lies below RHF, as the variational principle requires.** \\(E_{\text{FCI}} = -1.137275944\\) against \\(E_{\text{RHF}} = -1.116714325\\) hartree, so the **correlation energy** computed by this code is \\(-0.020561619\\) hartree. That is the entire prize on offer here: what the mean-field picture misses. Be precise about what it is — the difference between exact and Hartree-Fock *within the STO-3G basis*. It is not the correlation energy of the hydrogen molecule, because the basis is minimal, and no algorithm run inside this basis can recover what the basis cannot represent.
  * **The neutral singlet is also the global minimum** of the \\(16 \times 16\\) matrix, though nothing guaranteed that in advance. Ionized sectors sit higher: removing an electron costs more than the nuclear repulsion saves.
  * **The Hamiltonian contains fifteen Pauli strings.** Decomposing the \\(16 \times 16\\) matrix in the 256-element four-qubit Pauli basis leaves exactly 15 strings with a nonzero coefficient, identity included. Chapter 5 takes that count as its starting point for the measurement problem, and argues there that it follows from the symmetry rather than from hydrogen's particular integrals.
  * **Only one off-diagonal element matters.** In the four-determinant \\(N=2, S_z=0\\) block, the Hartree-Fock determinant couples to the doubly excited \\(|0011\rangle\\) with strength \\(+0.181257915\\), and to the two singly excited determinants with matrix elements of order \\(10^{-14}\\) — numerically zero. This is Brillouin's theorem and orbital symmetry appearing as a printed number rather than as an assertion.

## 4.6 Stage 5: VQE on the Reduced Problem

The last observation is a gift. Since the ground state has no weight on the singly excited determinants, the entire problem lives in a **two-dimensional** subspace spanned by \\(|1100\rangle\\) and \\(|0011\rangle\\), and a single parameter is enough:

\\[ |\psi(\theta)\rangle = \cos\frac{\theta}{2}\,|1100\rangle + \sin\frac{\theta}{2}\,|0011\rangle \\]

At \\(\theta = 0\\) this is exactly the Hartree-Fock determinant, which is the standard VQE starting point. Because the energy has the closed form

\\[ E(\theta) = \frac{H_{00} + H_{11}}{2} + \frac{H_{00} - H_{11}}{2}\cos\theta + H_{01}\sin\theta \\]

the parameter-shift rule of Chapter 3 is **exact** for this ansatz, not merely a good estimator.

Our code manipulates the state vector directly. On hardware the same state is prepared by \\(X\\) gates on qubits 0 and 1 to make the Hartree-Fock determinant, followed by an entangling block containing a single \\(R_y(\theta)\\) that transfers amplitude coherently to \\(|0011\rangle\\) — a double-excitation, or Givens, rotation. The parameter-shift rule applies because \\(\theta\\) enters through exactly one such rotation. This is a symmetry-preserving ansatz in the sense of Chapter 3: it cannot leave the \\(N=2, S_z=0\\) sector, so the optimizer cannot wander off into a different chemical species.

```python
def ansatz(t):                                  # cos(t/2)|1100> + sin(t/2)|0011>
    psi = np.zeros(DIM)
    psi[HF], psi[DOUBLE] = math.cos(t / 2.0), math.sin(t / 2.0)
    return psi

def energy(t):
    psi = ansatz(t)
    return float(psi @ H_q @ psi)

def grad(t):                                    # parameter-shift rule, exact here
    return 0.5 * (energy(t + math.pi / 2) - energy(t - math.pi / 2))

print("   theta      E(theta) (hartree)")
for t in np.linspace(-1.0, 0.5, 7):
    print(f"  {t:+.4f}   {energy(t):+18.9f}")

theta = 0.0                                     # start at the Hartree-Fock determinant
print("\n step        theta        E(theta) (hartree)     E - E_FCI     dE/dtheta")
for step in range(13):
    if step % 2 == 0:
        print(f"  {step:3d}   {theta:+11.8f}   {energy(theta):+18.9f}   "
              f"{energy(theta) - E_fci:+11.3e}   {grad(theta):+11.3e}")
    theta = theta - grad(theta)

E_vqe, psi = energy(theta), ansatz(theta)
print(f"\noptimal theta = {theta:+.9f} rad")
print(f"VQE ENERGY    = {E_vqe:+.9f} hartree      FCI ENERGY = {E_fci:+.9f} hartree")
print(f"|VQE - FCI|   = {abs(E_vqe - E_fci):.3e} hartree   "
      f"VQE >= FCI: {E_vqe - E_fci > -1e-12}")
print(f"amplitudes: |1100> = {psi[HF]:+.9f}   |0011> = {psi[DOUBLE]:+.9f}   "
      f"HF weight = {psi[HF] ** 2:.9f}")
print(f"\nRHF {E_rhf:+.9f}    FCI {E_fci:+.9f}    VQE {E_vqe:+.9f}    "
      f"correlation {E_fci - E_rhf:+.9f}")
```

**Output:**

```
   theta      E(theta) (hartree)
  -1.0000         -0.906699132
  -0.7500         -1.028664408
  -0.5000         -1.107070050
  -0.2500         -1.137041175
  +0.0000         -1.116714325
  +0.2500         -1.047353324
  +0.5000         -0.933270703

 step        theta        E(theta) (hartree)     E - E_FCI     dE/dtheta
    0   +0.00000000         -1.116714325    +2.056e-02    +1.813e-01
    2   -0.21737965         -1.137246494    +2.945e-05    +6.904e-03
    4   -0.22560061         -1.137275905    +3.903e-08    +2.513e-04
    6   -0.22589989         -1.137275944    +5.172e-11    +9.149e-06
    8   -0.22591078         -1.137275944    +6.883e-14    +3.330e-07
   10   -0.22591118         -1.137275944    +2.220e-16    +1.212e-08
   12   -0.22591119         -1.137275944    +0.000e+00    +4.413e-10

optimal theta = -0.225911191 rad
VQE ENERGY    = -1.137275944 hartree      FCI ENERGY = -1.137275944 hartree
|VQE - FCI|   = 2.220e-16 hartree   VQE >= FCI: True
amplitudes: |1100> = +0.993627297   |0011> = -0.112715549   HF weight = 0.987295205

RHF -1.116714325    FCI -1.137275944    VQE -1.137275944    correlation -0.020561619
```

**Reading the result.** Four points.

  * **The scan shows the Hartree-Fock state is not the minimum.** \\(E(0) = -1.116714325\\) hartree is the RHF energy by construction, and the curve dips below it on the negative-\\(\theta\\) side. Correlation energy is visible as the depth of that dip.
  * **VQE reaches the FCI energy to machine precision.** \\(-1.137275944\\) hartree from both, differing by \\(2 \times 10^{-16}\\) — floating-point roundoff. The gradient descent needs about ten steps to exhaust double precision. The variational check \\(E_{\text{VQE}} \geq E_{\text{FCI}}\\) holds, as it must, since the ansatz is a subset of the full space.
  * **The correlated state is still 98.7% Hartree-Fock.** The optimum sits at \\(\theta = -0.225911\\) rad, giving amplitudes \\(+0.993627\\) on \\(|1100\rangle\\) and \\(-0.112716\\) on \\(|0011\rangle\\). H₂ near equilibrium is a weakly correlated system, and this is why a single reference determinant works so well for it. Stretch the bond and that number falls — which is precisely why bond dissociation is the standard hard case.
  * **This matching proves the implementation, not an advantage.** The whole calculation ran on a laptop in a fraction of a second, and the "quantum" step optimized one angle over a two-dimensional space. What has been demonstrated is that the pipeline is correct end to end.

## 4.7 What Changed, and What Comes Next

Compare this chapter to the toy models that preceded it.

  * **The coefficients are no longer invented.** In Chapter 3 the numbers \\(0.5, 0.5, 0.25, 0.3\\) were chosen to make a readable example. Here every coefficient in the qubit Hamiltonian traces back through the four-index transformation, the SCF loop, and the Gaussian integrals to six published basis-set parameters and one chosen bond length.
  * **There is a real approximation hierarchy.** Minimal basis, then mean field, then exact-within-basis. Each level has a number attached, and each number came out of the run. The VQE energy improves on Hartree-Fock by \\(0.0206\\) hartree and improves on FCI by nothing, because FCI is the ceiling inside this basis.
  * **Symmetry did the heavy lifting, and exact diagonalization was affordable.** Four qubits became a two-dimensional subspace and one parameter, not by approximation but by exploiting conservation laws the Hamiltonian genuinely obeys — every serious VQE implementation does some version of this. And the FCI space we diagonalized had 16 dimensions, which is the whole point, and is temporary.

That last item is where Chapter 5 begins. The Hilbert space grows exponentially with the number of spin orbitals, so the FCI matrix that fitted in a \\(16 \times 16\\) array here becomes astronomically large for a molecule of chemical interest — which is the motivation for the whole enterprise, and also the reason none of it is easy. Chapter 5 examines what actually breaks as the system grows: the number of Pauli terms and the shots they demand, the depth of a chemically adequate ansatz, barren plateaus in the optimization landscape, and the gap between the demonstrations that exist and the calculations that would be useful. It is the honest accounting chapter, and it belongs at the end.

### 🎯 Exercise Problems

  1. **Move the nuclei.** Re-run the code at \\(R = 1.0\\), \\(1.4\\), \\(2.0\\), and \\(3.0\\) bohr and tabulate RHF, FCI, and the correlation energy. At which separation does the weight of the Hartree-Fock determinant in the FCI ground state fall furthest, and why is that the case that mean-field methods handle worst?
  2. **The Boys function limit.** Show that \\(F_0(x) \to 1\\) as \\(x \to 0\\), and derive the first correction \\(F_0(x) \approx 1 - x/3\\). Then remove the small-\\(x\\) branch from the code and find the separation at which the result becomes visibly wrong.
  3. **Break the check.** Deliberately transpose two index labels in the four-index transformation. Which of the printed cross-checks catches it, and which stay silent? What does that tell you about where to put assertions in scientific code?
  4. **Count the terms.** For \\(M\\) spatial orbitals the second-quantized Hamiltonian has \\(O(M^4)\\) two-electron coefficients. Evaluate that count for \\(M = 2\\), \\(10\\), and \\(50\\), and combine it with the shot-scaling argument of Chapter 3 to estimate how the measurement cost grows.
  5. **The variational bound, deliberately violated.** Restrict \\(\theta\\) to \\([0, \pi]\\) and re-optimize. Show numerically that the energy stays above the FCI value, and explain which property of the variational principle guarantees you cannot accidentally report a number below it.

## Summary

This chapter built a complete quantum-chemistry-to-qubits pipeline in NumPy and ran it on H₂ at \\(R = 1.4\\) bohr. Starting from the **STO-3G parameters** for hydrogen — six published numbers that constitute the definition of the basis, and the only external input in the chapter — we evaluated overlap, kinetic, nuclear-attraction and two-electron integrals in closed form over contracted s-type Gaussians, using the **Gaussian product theorem** and the **Boys function** \\(F_0(x) = \frac{1}{2}\sqrt{\pi/x}\,\mathrm{erf}(\sqrt{x})\\) computed with `math.erf` alone. A **restricted Hartree-Fock** SCF loop, started from a deliberately lopsided density, converged monotonically to a **total RHF energy of \\(-1.116714325\\) hartree** (\\(-1.831000039\\) electronic plus \\(+0.714285714\\) nuclear repulsion), recovering the symmetric bonding orbital that the initial guess had broken. Transforming to molecular orbitals gave \\(h_{pq}\\) and \\((pq|rs)\\), cross-checked by rebuilding the RHF energy from them to \\(4 \times 10^{-16}\\) hartree. The **Jordan-Wigner transformation** turned four spin orbitals into four qubits and the second-quantized Hamiltonian into an explicit \\(16 \times 16\\) matrix, whose fermionic anticommutation relations the code verified before use; exact diagonalization gave an **FCI energy of \\(-1.137275944\\) hartree** in the \\(N=2, S_z=0\\) sector and hence a **correlation energy of \\(-0.020561619\\) hartree** — the exact-minus-mean-field difference *within this basis*, with FCI below RHF as the variational principle requires. Finally, exploiting the block structure of the Hamiltonian, a **one-parameter VQE** over the subspace spanned by \\(|1100\rangle\\) and \\(|0011\rangle\\) converged with exact parameter-shift gradients to \\(-1.137275944\\) hartree, matching the code's own FCI value to \\(2 \times 10^{-16}\\), with the optimized state still 98.7% Hartree-Fock in character.

The next chapter asks the question this one has carefully set up. Everything here ran on a laptop, faster and more accurately than any quantum device could manage — so what changes when the molecule gets bigger, and what would have to become true before a quantum computer earned its place in this calculation?

[← Chapter 3: VQE: The Algorithm](<chapter-3.html>) [Chapter 5: Beyond H2: The Honest Frontier →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
