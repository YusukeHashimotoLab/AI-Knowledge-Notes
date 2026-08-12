---
title: "Chapter 5: Photons, Spins, Topology — and the Scorecard"
chapter_title: "Chapter 5: Photons, Spins, Topology — and the Scorecard"
subtitle: ⚛️ Three More Modalities, One Comparison Table, and Where Materials Science Enters
reading_time: 45-50 minutes
difficulty: Advanced
code_examples: 6
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-hardware-introduction/chapter-5.html>) | Last sync: 2026-08-13

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to Quantum Hardware](<index.html>) > Chapter 5

Chapters 2, 3 and 4 covered the three platforms that absorb most of the world's experimental effort. This chapter covers three more, and then does the thing the whole course has been building towards: it puts all six on one table.

The three modalities here are not also-rans. Each one is structurally different from the first three in a way that matters. **Photons** have no idle decoherence whatsoever — a photon in flight does not dephase, and there is no $T_1$ to quote because a photon does not relax; what it does instead is *disappear*, and loss is a different kind of error channel that no Pauli-error model covers — and they pay for that with the most awkward problem in the book, namely that you cannot make two photons interact. **Semiconductor spins** are the only qubit in this course that can be built in a commercial CMOS foundry, on the same wafer as its own control electronics, and they pay for that with the noisiest possible neighbourhood: an amorphous gate oxide a few tens of nanometres away. **Topological qubits** are the only ones whose protection is a mathematical theorem rather than an engineering achievement, and they pay for that by requiring a material that does not yet exist reliably.

Then §5.4 assembles the scorecard, and §5.5 states what this course has been arguing all along. The reason no modality has won is not that anyone lacks good ideas. It is that each modality is stopped by a *different* materials problem — and that is the point at which a materials researcher stops being a spectator.

**Units and conventions.** As throughout, $\hbar = 1$ in Hamiltonians. Energies are quoted in the unit customary to each field: µeV for quantum dots (with $1\ \mu\mathrm{eV} = h \times 242$ MHz), dimensionless units of the hopping $w$ for the Kitaev chain, and dimensionless ratios for the photonic calculations. The definitions of $T_1$, $T_2$ and $T_2^\ast$ from Chapter 1 are used unchanged. Three symbols do double duty in this chapter because each is standard in its own field, so they are fixed here once: $V$ is the HOM interference *visibility* in §5.1 and the Rydberg *interaction* $C_6/R^6$ wherever Chapter 4's error floor $(\gamma/V)^{2/3}$ is quoted; $\gamma$ is the photon mode *overlap* $\langle\phi_1|\phi_2\rangle$ in §5.1, a Rydberg *decay rate* in that same error floor, and a *Majorana operator* in §5.3; and $\eta$ is a photonic *transmission* here, not the Lamb-Dicke parameter of Chapter 3. Each usage is local to its section. Qubit ordering is big-endian throughout, and the fermionic mode ordering of §5.2 follows the Jordan-Wigner convention of the [algorithms companion](<../quantum-computing-introduction/index.html>), Chapter 4.

## Learning Objectives

After completing this chapter, you will be able to:

  * Compute the Hong-Ou-Mandel coincidence amplitude for bosons, fermions and distinguishable particles, derive $P_{cc} = (1 - |\langle\phi_1|\phi_2\rangle|^2)/2$, and explain why interference visibility is a materials specification for single-photon sources
  * Quantify why postselected linear-optics gates do not scale, state what heralding and multiplexing buy, and distinguish photon loss from gate error as failure modes
  * Explain the role of matrix permanents in linear optics and state precisely what a boson-sampling advantage claim does and does not establish
  * Diagonalize a double-quantum-dot Hubbard model, extract the singlet-triplet exchange splitting $J$, recover the $4t^2/U$ limit, and locate the symmetric-exchange sweet spot
  * Derive the $\sqrt{f}$ scaling of the Overhauser field with $^{29}$Si concentration, compute the resulting $T_2^\ast$ improvement from isotopic purification, and identify where purification stops helping
  * Diagonalize the Kitaev chain, measure the exponential splitting of the Majorana pair with system length, and construct a trivial Andreev bound state that mimics a zero-bias peak
  * State the nonlocality test that distinguishes a topological zero mode from a look-alike, and explain why the principle-to-demonstration gap in this modality is real without dismissing the idea
  * Write the Majorana decomposition of a fermion, show that braiding acts as a non-commuting unitary on the degenerate ground space, name the $4\pi$-periodic Josephson effect as its signature, and explain why quasiparticle poisoning rather than $e^{-L/\xi}$ sets the error rate
  * Compare all six modalities on physics-constraint grounds, derive order-of-magnitude coherent-operation counts from materials parameters, and defend the claim that no modality is currently ahead
  * Identify, for each modality, the specific materials problem that binds — and hence where a materials researcher can contribute

* * *

## 5.1 Photonic Qubits

### The good news, stated first

A photon travelling down a low-loss waveguide does not decohere. There is no $T_1$, no $T_2$, no thermal bath it couples to at room temperature, and no fabrication defect that shifts its frequency once it has been emitted. Single-qubit gates are beamsplitters and phase shifters, which are passive and essentially perfect. Detection is a projective measurement with no ambiguity about what was measured. Compared with everything in Chapters 2 through 4, this is an extraordinary starting position.

There is exactly one problem, and it is fundamental: **photons do not interact.** Maxwell's equations are linear, and while nonlinear optical media exist, the nonlinearity available at the single-photon level is many orders of magnitude too weak to build a deterministic two-photon gate. Every architecture for photonic quantum computing is a strategy for getting an effective interaction out of linear optics plus measurement.

### The interaction you can get for free

There *is* one two-photon effect in linear optics, and it is entirely a consequence of statistics. Send one photon into each input port of a 50:50 beamsplitter. Two paths lead to a coincidence — one photon out of each output port — and the beamsplitter gives them amplitudes of opposite sign. For identical bosons they cancel exactly. This is **Hong-Ou-Mandel interference**, and it is the closest thing to a photon-photon interaction that a linear network provides.

It is worth being precise about the bookkeeping, because HOM is often described as "two photons bunch" when the statement that matters is quantitative. Write the beamsplitter as $a^\dagger \to (c^\dagger + d^\dagger)/\sqrt2$, $b^\dagger \to (c^\dagger - d^\dagger)/\sqrt2$. Then

$$ a^\dagger b^\dagger|0\rangle \to \tfrac12\left(c^\dagger c^\dagger - c^\dagger d^\dagger + d^\dagger c^\dagger - d^\dagger d^\dagger\right)|0\rangle $$

For bosons $c^\dagger d^\dagger = d^\dagger c^\dagger$ and the two middle terms cancel; for fermions they add while $c^\dagger c^\dagger = 0$ kills the others, so fermions *always* come out separately. The two statistics give coincidence probabilities of 0 and 1, and distinguishable classical particles give 1/2. Nothing in this is approximate.

The physically important generalization is to photons that are not perfectly identical. If their internal (temporal, spectral, polarization) modes have overlap $\gamma = \langle\phi_1|\phi_2\rangle$, then

$$ P_{cc} = \frac{1 - |\gamma|^2}{2} $$

so the depth of the HOM dip *is* a direct measurement of how identical two photons are. That is the connection to materials science, and it is not a loose one: in a fusion-based photonic architecture every entangling operation is an HOM interference, so $1 - |\gamma|^2$ is an error per gate, and it is set by how reproducible the emitters are.

### Code Example 1: Hong-Ou-Mandel Interference

```python
"""Hong-Ou-Mandel interference from the two-photon amplitude.

Single-photon Hilbert space = 2 spatial modes x M temporal modes. A two-particle
state is the amplitude Psi(alpha, beta) on that space, normalized as
sum |Psi|^2 = 1, symmetric for bosons and antisymmetric for fermions. The
probability of one particle in mode set S and one in a disjoint set T is
2 * sum_{alpha in S, beta in T} |Psi(alpha,beta)|^2.
"""
import numpy as np

M = 401                       # temporal grid points
T = np.linspace(-8.0, 8.0, M)
dt = T[1] - T[0]
SIGMA = 1.0                   # wavepacket duration


def wavepacket(t0):
    """Normalized Gaussian temporal mode centred at t0 (unit l2 norm)."""
    f = np.exp(-(T - t0) ** 2 / (4 * SIGMA ** 2))
    return f / np.linalg.norm(f)


def single_photon(spatial, temporal):
    """Single-photon state vector on the 2M-dimensional space (spatial x time)."""
    v = np.zeros((2, M))
    v[spatial] = temporal
    return v.reshape(-1)


BS = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)   # 50:50 beamsplitter
BS_FULL = np.kron(BS, np.eye(M))


def two_particle(u, v, statistics):
    """Two-particle amplitude built from two orthogonal single-particle states."""
    if statistics == "boson":
        Psi = (np.outer(u, v) + np.outer(v, u)) / np.sqrt(2.0)
    elif statistics == "fermion":
        Psi = (np.outer(u, v) - np.outer(v, u)) / np.sqrt(2.0)
    else:                                   # distinguishable: no exchange symmetry
        Psi = np.outer(u, v)
    return Psi


def coincidence(Psi, statistics):
    """Probability of finding one particle in output c and one in output d."""
    P = np.abs(Psi.reshape(2, M, 2, M)) ** 2
    if statistics == "classical":
        return P[0, :, 1, :].sum() + P[1, :, 0, :].sum()
    return 2.0 * P[0, :, 1, :].sum()


def hom(delay, statistics):
    u = single_photon(0, wavepacket(-0.5 * delay))     # port a
    v = single_photon(1, wavepacket(+0.5 * delay))     # port b
    Psi = two_particle(u, v, statistics)
    Psi_out = BS_FULL @ Psi @ BS_FULL.T                # one BS acting on each particle
    return coincidence(Psi_out, statistics)


print("Two particles, one in each input port of a 50:50 beamsplitter.")
print("Identical wavepackets (zero delay):")
for st in ["boson", "fermion", "classical"]:
    print(f"  {st:<15} P(coincidence) = {hom(0.0, st):.10f}")
print("  Bosons never come out together in different ports: the two paths that give")
print("  a coincidence, (a->c, b->d) and (a->d, b->c), have amplitudes 1/2 and -1/2.")

# --- amplitude bookkeeping, written out -----------------------------------
print()
print("The cancellation, term by term (a^dag -> (c^dag + d^dag)/sqrt2,")
print("                                b^dag -> (c^dag - d^dag)/sqrt2):")
print("  a^dag b^dag |0> = (c^dag c^dag - c^dag d^dag + d^dag c^dag - d^dag d^dag)/2 |0>")
print("  bosons:   c^dag d^dag = d^dag c^dag, the two cross terms cancel exactly")
print("  fermions: c^dag d^dag = -d^dag c^dag, they add, and c^dag c^dag = 0")
print("            so only the coincidence term survives")

# --- the dip -------------------------------------------------------------
print()
print("The HOM dip: coincidence probability vs relative delay.")
hdr = (f"{'delay/sigma':>13}{'overlap |g|':>13}{'boson':>11}{'(1-|g|^2)/2':>13}"
       f"{'fermion':>10}{'classical':>11}")
print(hdr)
print("-" * len(hdr))
for d in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]:
    g = float(wavepacket(-0.5 * d) @ wavepacket(0.5 * d))
    print(f"{d:>13.1f}{abs(g):>13.6f}{hom(d, 'boson'):>11.6f}"
          f"{(1 - g ** 2) / 2:>13.6f}{hom(d, 'fermion'):>10.6f}"
          f"{hom(d, 'classical'):>11.6f}")
print("  P_cc = (1 - |<phi_1|phi_2>|^2)/2 for bosons: the dip *is* the mode overlap.")
print(f"  Gaussian prediction |g| = exp(-delay^2/(8 sigma^2)); at delay = 2 sigma,")
print(f"  exp(-0.5) = {np.exp(-0.5):.6f}")

# --- visibility and what spoils it ---------------------------------------
print()
print("Visibility V = 1 - 2 P_cc(0) as a function of mode mismatch.")
print(f"{'mismatch':>26}{'|g|':>9}{'P_cc(0)':>10}{'visibility':>12}")
cases = [("identical", 0.0, SIGMA), ("delay 0.5 sigma", 0.5, SIGMA),
         ("delay 1.0 sigma", 1.0, SIGMA), ("duration ratio 1.5", 0.0, 1.5 * SIGMA),
         ("duration ratio 3.0", 0.0, 3.0 * SIGMA)]
for name, d, s2 in cases:
    f2 = np.exp(-(T - 0.5 * d) ** 2 / (4 * s2 ** 2))
    f2 /= np.linalg.norm(f2)
    u = single_photon(0, wavepacket(-0.5 * d))
    v = single_photon(1, f2)
    Psi = (np.outer(u, v) + np.outer(v, u)) / np.sqrt(2.0)
    P = coincidence(BS_FULL @ Psi @ BS_FULL.T, "boson")
    g = float(wavepacket(-0.5 * d) @ f2)
    print(f"{name:>26}{abs(g):>9.5f}{P:>10.6f}{1 - 2 * P:>12.6f}")
print()
print("A visibility below one is not a small imperfection: in a fusion-based photonic")
print("architecture every entangling operation is an HOM interference, so 1 - V is an")
print("error per gate, and it is set by how identical two solid-state emitters are -")
print("a materials question about strain, charge environment and spectral diffusion.")
```

```text
Two particles, one in each input port of a 50:50 beamsplitter.
Identical wavepackets (zero delay):
  boson           P(coincidence) = 0.0000000000
  fermion         P(coincidence) = 1.0000000000
  classical       P(coincidence) = 0.5000000000
  Bosons never come out together in different ports: the two paths that give
  a coincidence, (a->c, b->d) and (a->d, b->c), have amplitudes 1/2 and -1/2.

The cancellation, term by term (a^dag -> (c^dag + d^dag)/sqrt2,
                                b^dag -> (c^dag - d^dag)/sqrt2):
  a^dag b^dag |0> = (c^dag c^dag - c^dag d^dag + d^dag c^dag - d^dag d^dag)/2 |0>
  bosons:   c^dag d^dag = d^dag c^dag, the two cross terms cancel exactly
  fermions: c^dag d^dag = -d^dag c^dag, they add, and c^dag c^dag = 0
            so only the coincidence term survives

The HOM dip: coincidence probability vs relative delay.
  delay/sigma  overlap |g|      boson  (1-|g|^2)/2   fermion  classical
-----------------------------------------------------------------------
          0.0     1.000000   0.000000     0.000000  1.000000   0.500000
          0.5     0.969233   0.030293     0.030293  0.969707   0.500000
          1.0     0.882497   0.110600     0.110600  0.889400   0.500000
          2.0     0.606531   0.316060     0.316060  0.683940   0.500000
          3.0     0.324652   0.447300     0.447300  0.552700   0.500000
          4.0     0.135335   0.490842     0.490842  0.509158   0.500000
          6.0     0.011109   0.499938     0.499938  0.500062   0.500000
  P_cc = (1 - |<phi_1|phi_2>|^2)/2 for bosons: the dip *is* the mode overlap.
  Gaussian prediction |g| = exp(-delay^2/(8 sigma^2)); at delay = 2 sigma,
  exp(-0.5) = 0.606531

Visibility V = 1 - 2 P_cc(0) as a function of mode mismatch.
                  mismatch      |g|   P_cc(0)  visibility
                 identical  1.00000  0.000000    1.000000
           delay 0.5 sigma  0.96923  0.030293    0.939413
           delay 1.0 sigma  0.88250  0.110600    0.778801
        duration ratio 1.5  0.96077  0.038461    0.923077
        duration ratio 3.0  0.77752  0.197730    0.604540

A visibility below one is not a small imperfection: in a fusion-based photonic
architecture every entangling operation is an HOM interference, so 1 - V is an
error per gate, and it is set by how identical two solid-state emitters are -
a materials question about strain, charge environment and spectral diffusion.
```

**What to look for.** The calculation is set up as a two-particle amplitude on a $2M$-dimensional single-particle space, which is deliberately more general than the usual textbook treatment: it lets bosons, fermions, distinguishable particles and partial distinguishability all be computed by the same three lines.

**Zero, one half, one.** The three statistics give exactly 0.0000000000, 0.5000000000 and 1.0000000000. The bosonic zero is the interference; the fermionic one is Pauli exclusion; the classical one half is what you would get by flipping coins. Any experiment reporting a coincidence rate between 0 and 1/2 is reporting a *mixture* of the bosonic and classical cases, and the mixing parameter is the mode overlap.

**The dip is the overlap, exactly.** The measured $P_{cc}$ agrees with $(1 - |\gamma|^2)/2$ to every printed digit at every delay. This is worth internalizing because it inverts the experiment: the HOM dip is not a qualitative demonstration of quantum weirdness, it is a **metrology tool for photon indistinguishability**, and it is the standard one.

**Visibility is destroyed by mismatches you would not think to check.** A delay of half a wavepacket duration costs 6% of visibility. But so does a *duration* mismatch: two photons that arrive at exactly the same time with 1.5 times different pulse lengths have $|\gamma| = 0.961$ and a visibility of 0.923. Two solid-state emitters in different local strain and charge environments differ in linewidth, centre frequency and lifetime, and every one of those differences shows up here. This is why "indistinguishable single-photon source" is a materials-growth problem rather than an optics problem.

### Why nondeterminism is the architecture problem

Measurement provides the effective nonlinearity. The standard postselected linear-optics CZ gate succeeds with probability $1/9$: eight times out of nine you find out afterwards that the gate did not happen. For a single gate this is merely inefficient. For a circuit it is fatal, because the probabilities multiply.

The escape route, due to Knill, Laflamme and Milburn and developed since into measurement-based and fusion-based architectures, has three ingredients. **Heralding**: arrange the gate so that ancilla detections tell you whether it worked, *without* destroying the data qubits. **Multiplexing**: run many copies of each gate in parallel and switch in one that succeeded. **Offline resource states**: build large entangled cluster states in advance by many small heralded operations, then run the computation as a sequence of single-qubit measurements on that state, where the only nondeterminism left is in the state preparation.

The third ingredient is the important conceptual move, and it is why photonics is the natural home of measurement-based quantum computation. It converts a depth problem into a *resource-count* problem — which is a much better problem to have if photon sources and detectors are cheap and parallel, and a much worse one if they are not.

### Code Example 2: The Arithmetic of Nondeterminism

```python
"""Why linear optics needs measurement: the arithmetic of nondeterminism and loss."""
import numpy as np
from itertools import permutations

P_CZ = 1.0 / 9.0        # success probability of the postselected linear-optics CZ

print("A postselected linear-optics two-qubit gate succeeds with p = 1/9.")
print("Chaining G of them without repair multiplies the probabilities.")
hdr = f"{'G gates':>9}{'p^G':>13}{'trials for 50%':>16}{'wall time at 1 GHz':>20}"
print(hdr)
print("-" * len(hdr))
for G in [1, 2, 5, 10, 20, 50]:
    p = P_CZ ** G
    trials = np.log(2.0) / p          # p << 1: expected trials for a 50% chance
    secs = trials / 1e9
    unit = (f"{secs:.2e} s" if secs < 3.15e7 else f"{secs / 3.156e7:.2e} yr")
    print(f"{G:>9}{p:>13.3e}{trials:>16.3e}{unit:>20}")
print("  Fifty gates is a trivial circuit and the exponent has already ended the")
print("  conversation. Postselection is not an architecture.")

# --- what heralding and multiplexing buy ----------------------------------
print()
print("Heralded gates can be retried. With n_mux parallel copies per gate slot,")
print("the chance that at least one copy succeeds is 1 - (1-p)^n_mux:")
print(f"{'n_mux':>7}{'P(slot ok)':>13}{'P(all 100 slots ok)':>22}")
for n_mux in [1, 10, 30, 60, 100, 200]:
    q = 1 - (1 - P_CZ) ** n_mux
    print(f"{n_mux:>7}{q:>13.6f}{q ** 100:>22.3e}")
print("  Reaching 99% on a 100-gate circuit needs P(slot ok) > 0.9999, i.e.")
n_needed = int(np.ceil(np.log(1 - 0.9999) / np.log(1 - P_CZ)))
print(f"  n_mux >= {n_needed} copies per gate slot, so {n_needed * 100} heralded resources")
print("  for a circuit a superconducting chip would run with 100 pulses.")

# --- loss is the real enemy ------------------------------------------------
print()
print("Photon loss: transmission eta per component, N components in the path.")
hdr = f"{'eta':>7}" + "".join(f"{'N=' + str(N):>12}" for N in [10, 50, 100, 500])
print(hdr)
print("-" * len(hdr))
for eta in [0.999, 0.99, 0.95, 0.9]:
    print(f"{eta:>7.3f}" + "".join(f"{eta ** N:>12.3e}" for N in [10, 50, 100, 500]))
print("  Loss is not correctable by postselection - a lost photon is a lost qubit -")
print("  so photonic fault tolerance needs loss-tolerant codes, and their thresholds")
print("  are stated in terms of eta per component, not gate error.")

# --- and the reason to keep going: sampling hardness -----------------------
print()
print("The flip side: the output amplitudes of a linear-optics network are matrix")
print("permanents, and the permanent has no known subexponential algorithm.")


def permanent_bruteforce(A):
    n = A.shape[0]
    return sum(np.prod([A[i, p[i]] for i in range(n)]) for p in permutations(range(n)))


def permanent_ryser(A):
    """Ryser's formula: 2^n subsets instead of n! permutations."""
    n = A.shape[0]
    total = 0.0
    for mask in range(1, 1 << n):
        cols = [j for j in range(n) if mask >> j & 1]
        rowsum = A[:, cols].sum(axis=1)
        total += (-1) ** len(cols) * np.prod(rowsum)
    return total * (-1) ** n


rng = np.random.default_rng(7)
A = rng.normal(size=(6, 6))
print(f"  6x6 test matrix: brute force {permanent_bruteforce(A):+.10f}, "
      f"Ryser {permanent_ryser(A):+.10f}")
print(f"{'n':>5}{'n! terms':>14}{'2^n n terms (Ryser)':>22}{'ratio':>12}")
for n in [6, 10, 20, 30, 40, 50]:
    fact = np.exp(sum(np.log(i) for i in range(1, n + 1)))
    ryser = 2.0 ** n * n
    print(f"{n:>5}{fact:>14.3e}{ryser:>22.3e}{fact / ryser:>12.3e}")
print("  Ryser turns n! into 2^n n, which is an enormous win and still exponential.")
print("  That gap is the entire content of a boson-sampling advantage claim: it says")
print("  something about the classical cost of a sampling problem, and nothing about")
print("  ground-state energies, which is what a materials researcher wanted.")
```

```text
A postselected linear-optics two-qubit gate succeeds with p = 1/9.
Chaining G of them without repair multiplies the probabilities.
  G gates          p^G  trials for 50%  wall time at 1 GHz
----------------------------------------------------------
        1    1.111e-01       6.238e+00          6.24e-09 s
        2    1.235e-02       5.614e+01          5.61e-08 s
        5    1.694e-05       4.093e+04          4.09e-05 s
       10    2.868e-10       2.417e+09          2.42e+00 s
       20    8.225e-20       8.427e+18         2.67e+02 yr
       50    1.940e-48       3.572e+47         1.13e+31 yr
  Fifty gates is a trivial circuit and the exponent has already ended the
  conversation. Postselection is not an architecture.

Heralded gates can be retried. With n_mux parallel copies per gate slot,
the chance that at least one copy succeeds is 1 - (1-p)^n_mux:
  n_mux   P(slot ok)   P(all 100 slots ok)
      1     0.111111             3.765e-96
     10     0.692054             1.033e-16
     30     0.970797             5.162e-02
     60     0.999147             9.182e-01
    100     0.999992             9.992e-01
    200     1.000000             1.000e+00
  Reaching 99% on a 100-gate circuit needs P(slot ok) > 0.9999, i.e.
  n_mux >= 79 copies per gate slot, so 7900 heralded resources
  for a circuit a superconducting chip would run with 100 pulses.

Photon loss: transmission eta per component, N components in the path.
    eta        N=10        N=50       N=100       N=500
-------------------------------------------------------
  0.999   9.900e-01   9.512e-01   9.048e-01   6.064e-01
  0.990   9.044e-01   6.050e-01   3.660e-01   6.570e-03
  0.950   5.987e-01   7.694e-02   5.921e-03   7.274e-12
  0.900   3.487e-01   5.154e-03   2.656e-05   1.322e-23
  Loss is not correctable by postselection - a lost photon is a lost qubit -
  so photonic fault tolerance needs loss-tolerant codes, and their thresholds
  are stated in terms of eta per component, not gate error.

The flip side: the output amplitudes of a linear-optics network are matrix
permanents, and the permanent has no known subexponential algorithm.
  6x6 test matrix: brute force -13.9851672727, Ryser -13.9851672727
    n      n! terms   2^n n terms (Ryser)       ratio
    6     7.200e+02             3.840e+02   1.875e+00
   10     3.629e+06             1.024e+04   3.544e+02
   20     2.433e+18             2.097e+07   1.160e+11
   30     2.653e+32             3.221e+10   8.235e+21
   40     8.159e+47             4.398e+13   1.855e+34
   50     3.041e+64             5.629e+16   5.403e+47
  Ryser turns n! into 2^n n, which is an enormous win and still exponential.
  That gap is the entire content of a boson-sampling advantage claim: it says
  something about the classical cost of a sampling problem, and nothing about
  ground-state energies, which is what a materials researcher wanted.
```

**What to look for.** Three separate arithmetic facts, each of which independently shapes the architecture.

**Postselection dies at twenty gates.** $9^{-20} = 8\times10^{-20}$, so even at a gigahertz repetition rate the expected time to see one successful 20-gate circuit is 267 years. Twenty gates is nothing — Chapter 4's blockade gate does better than that in a microsecond. This is not a statement about photonics being bad; it is a proof that postselection cannot be the architecture, which is why nobody proposes it as one.

**Multiplexing works, and it is expensive.** Reaching 99% success on a 100-gate circuit requires 79 parallel copies per gate slot, i.e. about 7900 heralded resources for a circuit a superconducting chip runs with 100 microwave pulses. That ratio, roughly two orders of magnitude, is the price of nondeterminism, and it is why photonic proposals are stated in terms of *numbers of sources and detectors* rather than numbers of qubits.

**Loss is the failure mode that has no analogue elsewhere.** At 99% transmission per component, a 100-component path delivers 37% of the time; at 95%, it delivers 0.6% of the time. And unlike a Pauli error, a lost photon cannot be corrected by a standard stabilizer code — it is leakage, exactly as atom loss was in Chapter 4. Photonic fault tolerance therefore needs loss-tolerant codes whose thresholds are quoted as a transmission per component. Every interface, every waveguide bend, every detector inefficiency spends part of that budget, which makes the entire photonic-integrated-circuit materials stack — low-loss silicon nitride, lithium niobate, coupling to fibre, superconducting nanowire detectors — the binding constraint.

**And the reason the field is nonetheless full of good physics.** The output amplitudes of a linear network are matrix permanents, and Ryser's algorithm reduces $n!$ to $2^n n$ — an improvement of $5\times10^{47}$ at $n = 50$ that is still exponential. That gap is what a boson-sampling or Gaussian-boson-sampling advantage claim is about. It is a real statement about the classical cost of a sampling problem. It is also, as the [algorithms course](<../quantum-computing-introduction/index.html>) argued at length, not a statement about ground-state energies, and treating the two as interchangeable is the single most common error in reading these results.

* * *

## 5.2 Semiconductor Spin Qubits

### One electron, one dot, one CMOS process

Confine a single electron electrostatically in a semiconductor, apply a magnetic field, and its spin is a two-level system with a Zeeman splitting you can address. The dot is defined by metal gates on top of a heterostructure, typically Si/SiGe or Si/SiO$_2$, at a scale of tens of nanometres — which is to say, at a scale that the semiconductor industry has been manufacturing for decades.

That last sentence is the entire strategic argument for this modality, and it deserves to be taken seriously. A silicon spin qubit is the only qubit in this course that can be fabricated in a commercial foundry, on a 300 mm wafer, with the same lithography, the same metrology and the same yield statistics as a transistor — potentially with its own control electronics on the same die, which would sidestep the wiring problem that Chapters 2 and 4 both identified as architectural. The qubits are also small: a dot is $\sim$100 nm across against $\sim$100 µm for a transmon, a factor of $10^6$ in area.

The magnetic and spin-transport physics underneath — Zeeman splitting, $g$-factors, spin-orbit coupling, spin relaxation channels — is the subject of the [Introduction to Spintronics](<../../MS/spintronics-introduction/index.html>) course, and the two subjects are the same physics viewed with different goals: spintronics asks how to move and detect spin information in a device, and spin qubits ask how to keep one spin coherent while rotating it precisely.

### The two-qubit gate is the exchange interaction

Two electrons in adjacent dots with a tunnel barrier between them are described by a two-site Hubbard model — *the same* two-site Hubbard model that the algorithms course diagonalized in its Chapter 4 as a toy problem in quantum chemistry. Here it is not a toy: it is the device Hamiltonian.

$$ H = -t\sum_\sigma \left(c^\dagger_{0\sigma}c_{1\sigma} + \text{h.c.}\right) + U\sum_i n_{i\uparrow}n_{i\downarrow} + \frac{\varepsilon}{2}\left(n_1 - n_0\right) $$

with $t$ the tunnel coupling set by the barrier gate, $U$ the on-site charging energy, and $\varepsilon$ the detuning between the two dots set by the plunger gates. In the two-electron sector the spin singlet can virtually hop into the doubly-occupied $(2,0)$ and $(0,2)$ configurations while the triplet cannot, by Pauli exclusion. The resulting energy difference is the **exchange splitting**

$$ J = E_T - E_S \simeq \frac{2t^2}{U-\varepsilon} + \frac{2t^2}{U+\varepsilon} = \frac{4t^2 U}{U^2 - \varepsilon^2} $$

which reduces to the familiar superexchange $4t^2/U$ at zero detuning. Turning $J$ on for a time $t = \pi\hbar/J$ implements a **SWAP**, which is not entangling at all — it merely exchanges the two spins. The entangling gate is *half* of it: at $t = \pi\hbar/2J$ the evolution is $\sqrt{\mathrm{SWAP}}$, which maps $|{\uparrow\downarrow}\rangle$ onto an equal superposition of $|{\uparrow\downarrow}\rangle$ and $|{\downarrow\uparrow}\rangle$ and, with single-qubit rotations, composes into CNOT. The tables below quote the SWAP time $h/2J$; the entangler takes half of it. Everything about spin-qubit operation is a consequence of the fact that $J$ is a *gate-voltage-controlled* quantity: it is fast, it is local, and it is exposed to every voltage fluctuation in the device.

### Code Example 3: The Exchange-Coupled Double Quantum Dot

```python
"""The exchange-coupled double quantum dot, diagonalized exactly.

Two dots, one orbital each, two spins: four fermionic modes ordered
(dot 0 up, dot 0 down, dot 1 up, dot 1 down). We build the Hubbard dimer with
Jordan-Wigner exactly as in the algorithms course, restrict to two electrons,
and read off the singlet-triplet splitting J. All energies in ueV.
"""
import numpy as np

NM = 4                              # fermionic modes
DIM = 2 ** NM
I2 = np.eye(2, dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
LOWER = np.array([[0, 1], [0, 0]], dtype=complex)      # |0><1| : annihilates


def kron_list(ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def annihilate(p):
    """Jordan-Wigner annihilation operator for mode p (big-endian mode order)."""
    return kron_list([Z] * p + [LOWER] + [I2] * (NM - p - 1))


C = [annihilate(p) for p in range(NM)]
CD = [c.conj().T for c in C]
NOP = [CD[p] @ C[p] for p in range(NM)]
N_TOT = sum(NOP)
# S_z = (n_0u - n_0d + n_1u - n_1d)/2 with modes (0u, 0d, 1u, 1d)
SZ = 0.5 * (NOP[0] - NOP[1] + NOP[2] - NOP[3])
# total S^2 for two spin-1/2 in the singly-occupied sector, built from S_+ S_-
SP = CD[0] @ C[1] + CD[2] @ C[3]
SM = SP.conj().T
S2 = SM @ SP + SZ @ SZ + SZ


def hamiltonian(t, U, eps):
    """-t hopping, U on-site, eps = detuning (energy of dot 1 minus dot 0)."""
    H = np.zeros((DIM, DIM), dtype=complex)
    for s in (0, 1):
        H += -t * (CD[s] @ C[2 + s] + CD[2 + s] @ C[s])
    H += U * (NOP[0] @ NOP[1] + NOP[2] @ NOP[3])
    H += 0.5 * eps * (NOP[2] + NOP[3] - NOP[0] - NOP[1])
    return H


def two_electron_spectrum(t, U, eps):
    """Eigenvalues, <S^2> and <n_dot0> in the two-electron sector."""
    H = hamiltonian(t, U, eps)
    keep = np.isclose(np.diag(N_TOT).real, 2.0)
    idx = np.where(keep)[0]
    Hs = H[np.ix_(idx, idx)]
    w, v = np.linalg.eigh(Hs)
    s2 = np.array([(v[:, i].conj() @ S2[np.ix_(idx, idx)] @ v[:, i]).real
                   for i in range(len(w))])
    DOCC = NOP[0] @ NOP[1] + NOP[2] @ NOP[3]
    d = np.array([(v[:, i].conj() @ DOCC[np.ix_(idx, idx)] @ v[:, i]).real
                  for i in range(len(w))])
    return w, s2, d


def exchange(t, U, eps):
    """J = E(triplet) - E(singlet), both taken as the lowest state of their sector."""
    w, s2, _ = two_electron_spectrum(t, U, eps)
    E_s = w[s2 < 0.5].min()          # S = 0  -> <S^2> = 0
    E_t = w[s2 > 1.5].min()          # S = 1  -> <S^2> = 2
    return E_t - E_s


U, t = 3000.0, 100.0                 # ueV: charging energy and tunnel coupling
print(f"Double quantum dot: U = {U:.0f} ueV, t = {t:.0f} ueV, six two-electron states.")
w, s2, docc = two_electron_spectrum(t, U, 0.0)
print(f"{'E (ueV)':>12}{'<S^2>':>9}{'<n_up n_dn>':>13}{'label':>28}")
for E, s, d in zip(w, s2, docc):
    if s > 1.5:
        lab = "triplet (1,1)"
    elif d < 0.5:
        lab = "singlet, mostly (1,1)"
    else:
        lab = "singlet, mostly (2,0)+(0,2)"
    print(f"{E:>12.4f}{s:>9.4f}{d:>13.4f}{lab:>28}")
print(f"  J = E_T - E_S = {exchange(t, U, 0.0):.4f} ueV")
print(f"  perturbative 4t^2/U = {4 * t ** 2 / U:.4f} ueV")
print(f"  exact two-site formula U/2 - sqrt((U/2)^2 + 4t^2) gives J = "
      f"{-(U / 2 - np.sqrt((U / 2) ** 2 + 4 * t ** 2)):.4f} ueV")

# --- J as a knob: tunnel coupling and detuning ----------------------------
print()
print("J versus tunnel coupling at zero detuning (the 'symmetric operation' knob):")
print(f"{'t (ueV)':>9}{'J (ueV)':>12}{'4t^2/U':>11}{'J/h (MHz)':>12}{'SWAP h/2J':>11}")
for ti in [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
    J = exchange(ti, U, 0.0)
    f_MHz = J * 1e-6 * 1.602176634e-19 / 6.62607015e-34 / 1e6
    print(f"{ti:>9.0f}{J:>12.4f}{4 * ti ** 2 / U:>11.4f}{f_MHz:>12.2f}"
          f"{500.0 / f_MHz:>11.3f}")

print()
print("J versus detuning at fixed t (the 'tilted operation' knob):")
hdr = (f"{'eps/U':>8}{'J (ueV)':>11}{'4t^2 U/(U^2-eps^2)':>20}"
       f"{'dJ/d eps':>11}{'|dJ/deps|/J':>13}")
print(hdr)
print("-" * len(hdr))
h_eps = 1.0
for r in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]:
    eps = r * U
    J = exchange(t, U, eps)
    dJ = (exchange(t, U, eps + h_eps) - exchange(t, U, eps - h_eps)) / (2 * h_eps)
    pert = 2 * t ** 2 / (U - eps) + 2 * t ** 2 / (U + eps)
    print(f"{r:>8.1f}{J:>11.4f}{pert:>20.4f}{dJ:>11.5f}{abs(dJ) / J:>13.5f}")

# --- the sweet spot and what charge noise costs ---------------------------
print()
print("Charge noise: a static detuning offset d_eps changes J, so the exchange phase")
print("is wrong by pi |J(eps+d_eps) - J(eps)| / J per gate. Gates before one radian:")
print(f"{'eps/U':>8}{'J (ueV)':>11}{'SWAP h/2J':>12}"
      f"{'N at d_eps=1 ueV':>18}{'N at d_eps=10 ueV':>19}")
for r in [0.0, 0.2, 0.5, 0.8]:
    eps = r * U
    J = exchange(t, U, eps)
    f_MHz = J * 1e-6 * 1.602176634e-19 / 6.62607015e-34 / 1e6
    ns = [J / (np.pi * abs(exchange(t, U, eps + d) - J)) for d in (1.0, 10.0)]
    print(f"{r:>8.1f}{J:>11.4f}{500.0 / f_MHz:>12.3f}"
          f"{ns[0]:>18.1f}{ns[1]:>19.1f}")
print("  At eps = 0 the derivative vanishes by symmetry: J is flat to first order.")
print("  That is the symmetric-exchange sweet spot, and it is the whole reason")
print("  spin qubits are operated there rather than at the charge-transition slope.")
```

```text
Double quantum dot: U = 3000 ueV, t = 100 ueV, six two-electron states.
     E (ueV)    <S^2>  <n_up n_dn>                       label
    -13.2746   0.0000       0.0044       singlet, mostly (1,1)
     -0.0000   2.0000       0.0000               triplet (1,1)
      0.0000   2.0000       0.0000               triplet (1,1)
      0.0000   2.0000       0.0000               triplet (1,1)
   3000.0000   0.0000       1.0000 singlet, mostly (2,0)+(0,2)
   3013.2746   0.0000       0.9956 singlet, mostly (2,0)+(0,2)
  J = E_T - E_S = 13.2746 ueV
  perturbative 4t^2/U = 13.3333 ueV
  exact two-site formula U/2 - sqrt((U/2)^2 + 4t^2) gives J = 13.2746 ueV

J versus tunnel coupling at zero detuning (the 'symmetric operation' knob):
  t (ueV)     J (ueV)     4t^2/U   J/h (MHz)  SWAP h/2J
        5      0.0333     0.0333        8.06     62.036
       10      0.1333     0.1333       32.24     15.509
       20      0.5332     0.5333      128.94      3.878
       50      3.3296     3.3333      805.10      0.621
      100     13.2746    13.3333     3209.78      0.156
      200     52.4175    53.3333    12674.49      0.039

J versus detuning at fixed t (the 'tilted operation' knob):
   eps/U    J (ueV)  4t^2 U/(U^2-eps^2)   dJ/d eps  |dJ/deps|/J
---------------------------------------------------------------
     0.0    13.2746             13.3333   -0.00000      0.00000
     0.2    13.8199             13.8889    0.00189      0.00014
     0.4    15.7588             15.8730    0.00491      0.00031
     0.6    20.5352             20.8333    0.01239      0.00060
     0.8    35.1675             37.0370    0.04656      0.00132
     0.9    59.1585             70.1754    0.13364      0.00226

Charge noise: a static detuning offset d_eps changes J, so the exchange phase
is wrong by pi |J(eps+d_eps) - J(eps)| / J per gate. Gates before one radian:
   eps/U    J (ueV)   SWAP h/2J  N at d_eps=1 ueV  N at d_eps=10 ueV
     0.0    13.2746       0.156         2902931.1            29029.0
     0.2    13.8199       0.150            2323.1              230.3
     0.5    17.6058       0.117             733.9               72.9
     0.8    35.1675       0.059             240.1               23.7
  At eps = 0 the derivative vanishes by symmetry: J is flat to first order.
  That is the symmetric-exchange sweet spot, and it is the whole reason
  spin qubits are operated there rather than at the charge-transition slope.
```

**What to look for.** This is the same Jordan-Wigner machinery as the algorithms course, applied to a device rather than a molecule, and it produces the central design principle of the modality.

**Six states, and the physics is visible in the labels.** The two-electron sector has one $(1,1)$ singlet at $-13.27$ µeV, a threefold degenerate $(1,1)$ triplet at exactly zero, and two mostly-doubly-occupied singlets near $U$. The triplet is exactly at zero because Pauli exclusion forbids it from hopping — that is the whole mechanism, and here it is as a degeneracy that the tunnel coupling cannot lift. $J = 13.2746$ µeV agrees with the closed-form $\sqrt{(U/2)^2+4t^2} - U/2$ to all printed digits and with the perturbative $4t^2/U = 13.3333$ to 0.4%.

**$J$ spans three orders of magnitude over a plausible range of $t$.** From $t = 5$ µeV to $t = 200$ µeV, $J$ goes from 0.033 to 52 µeV — a factor of 1570, or 3.2 decades — and the SWAP time from 62 ns to 0.04 ns (the entangling $\sqrt{\mathrm{SWAP}}$ taking half of each). A tunnel coupling is an exponential function of a barrier gate voltage, so this three-decade range is available on one electrode — which is simultaneously the modality's greatest convenience and its greatest exposure.

**The sweet spot is real and it is worth a factor of a thousand.** At $\varepsilon = 0$ the derivative $\mathrm{d}J/\mathrm{d}\varepsilon$ vanishes by symmetry: the singlet couples equally to $(2,0)$ and $(0,2)$, and the two linear shifts cancel. A 1 µeV detuning offset then costs one radian of exchange phase only after $2.9\times10^6$ gates, against 2323 gates at $\varepsilon = 0.2U$ and 240 at $\varepsilon = 0.8U$. **Operating at the symmetric point is worth three orders of magnitude in gate count, and it costs nothing but a choice of bias.** This is the spin-qubit analogue of the transmon's charge-noise insensitivity in Chapter 2, and the structure of the argument is identical: find the point where the first derivative of the qubit frequency with respect to the noisy parameter vanishes, and sit there.

**And the honest caveat is in the code's own comment.** The sweet-spot number assumes a *static* offset. Real charge noise in these devices is $1/f$, generated by an ensemble of two-level fluctuators in the gate oxide and at the interface, so a gate lasting tens of nanoseconds samples many offsets and the second-order curvature does the damage. The mechanism — two-level defects in an amorphous oxide — is *the same physical mechanism* that limits superconducting qubits in Chapter 2. Two modalities that could hardly look more different are stopped by the same class of defect.

### Isotopic purification, and where it stops helping

Natural silicon contains 4.7% $^{29}$Si, the only stable silicon isotope with a nuclear spin. Each such nucleus couples to the electron by the contact hyperfine interaction, and the sum over the $\sim10^5$ nuclei inside the electron's envelope is a random effective magnetic field — the **Overhauser field** — that shifts the qubit frequency by a different amount in every shot. That is an inhomogeneous dephasing, so it limits $T_2^\ast$ rather than $T_2$, exactly per the Chapter 1 definitions.

The remedy is to remove the $^{29}$Si. Because the nuclear spins add incoherently, the field spread scales as the square root of their number:

$$ \sigma_{\text{Overhauser}} \propto \sqrt{f\,N_{\text{sites}}}, \qquad T_2^\ast \propto \frac{1}{\sqrt{f}} $$

This is a genuinely unusual situation in quantum hardware: a decoherence channel that can be removed by *chemistry*, at a price set by isotope-separation plants rather than by physics. It is also, as the numbers below show, a channel with sharply diminishing returns.

### Code Example 4: Isotopic Purification and the Overhauser Field

```python
"""Isotopic purification: why 28-Si is a quantum-hardware material.

The electron in a quantum dot overlaps N_sites lattice sites. A fraction f of
them carry a spin-1/2 29-Si nucleus, and each contributes a random hyperfine
field. The sum is the Overhauser field; its spread limits T2*.

Everything is in units of the single-nucleus hyperfine coupling a_hf, so no
device-specific number enters. Ratios are the physics.
"""
import numpy as np

rng = np.random.default_rng(2026)
N_SITES = 400_000          # lattice sites inside the electron envelope
N_TRIALS = 40_000          # Ramsey shots per configuration

ABUNDANCE = {"natural Si (4.7%)": 4.7e-2, "1000 ppm": 1.0e-3,
             "200 ppm": 2.0e-4, "50 ppm": 5.0e-5}


def overhauser_samples(f, n_trials=N_TRIALS):
    """Sample the Overhauser detuning, in units of a_hf, for random nuclear spins.

    A site carries a 29-Si nucleus with probability f; each such nucleus is
    +1/2 or -1/2 at random. In the large-N limit the sum is Gaussian with
    variance N_sites * f / 4, so we sample that directly and check it once.
    """
    return rng.normal(0.0, np.sqrt(N_SITES * f / 4.0), size=n_trials)


# --- verify the Gaussian limit against an explicit spin-by-spin draw -------
f0 = 4.7e-2
explicit = np.array([
    (rng.random(N_SITES) < f0).astype(float).dot(rng.choice([-0.5, 0.5], N_SITES))
    for _ in range(200)])
print("Central limit check, natural Si, 200 explicit configurations:")
print(f"  explicit std  {explicit.std():.3f} a_hf")
print(f"  sqrt(N f / 4) {np.sqrt(N_SITES * f0 / 4.0):.3f} a_hf")
print(f"  explicit mean {explicit.mean():+.3f} a_hf (should be 0)")

# --- Ramsey decay from a static-but-random detuning ------------------------
def ramsey(f, times):
    """Shot-averaged Ramsey signal <cos(delta t)> over Overhauser realizations."""
    delta = overhauser_samples(f)
    return np.array([np.cos(delta * t).mean() for t in times])


print()
print("Ramsey free-induction decay, time in units of 1/a_hf:")
sigmas, t2s = {}, {}
for name, f in ABUNDANCE.items():
    sigma = np.sqrt(N_SITES * f / 4.0)
    sigmas[name] = sigma
    t2s[name] = np.sqrt(2.0) / sigma          # exp(-(t/T2*)^2) with T2* = sqrt2/sigma
print(f"{'material':>20}{'f':>10}{'N_29':>9}{'sigma (a_hf)':>14}"
      f"{'T2* (1/a_hf)':>14}{'gain':>8}")
for name, f in ABUNDANCE.items():
    print(f"{name:>20}{f:>10.1e}{int(N_SITES * f):>9d}{sigmas[name]:>14.4f}"
          f"{t2s[name]:>14.4e}{t2s[name] / t2s['natural Si (4.7%)']:>8.2f}")
print(f"  predicted gain from 4.7% to 50 ppm: sqrt(4.7e-2/5e-5) = "
      f"{np.sqrt(4.7e-2 / 5e-5):.2f}")

print()
print("Measured decay envelope (Monte Carlo) against the Gaussian prediction:")
for name in ["natural Si (4.7%)", "200 ppm"]:
    f = ABUNDANCE[name]
    T2 = t2s[name]
    ts = np.array([0.25, 0.5, 1.0, 1.5, 2.0]) * T2
    sig = ramsey(f, ts)
    print(f"  {name}: T2* = {T2:.4e} / a_hf")
    print(f"    {'t/T2*':>8}{'<cos>':>12}{'exp(-(t/T2*)^2)':>18}")
    for t, s in zip(ts, sig):
        print(f"    {t / T2:>8.2f}{s:>12.6f}{np.exp(-(t / T2) ** 2):>18.6f}")

# --- what purification does NOT fix ---------------------------------------
print()
print("The scaling is sqrt(f), so purification has diminishing returns, and it")
print("stops helping once another mechanism takes over:")
print(f"{'f':>10}{'T2* (1/a_hf)':>15}{'with a charge-noise cap at 2/a_hf':>36}")
for f in [4.7e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    T2 = np.sqrt(2.0) / np.sqrt(N_SITES * f / 4.0)
    print(f"{f:>10.1e}{T2:>15.4e}{1.0 / (1.0 / T2 + 1.0 / 2.0):>36.4e}")
print("  The crossover, where the nuclear and the competing contribution to 1/T2*")
print("  are equal, is at T2* = 2/a_hf, i.e. N_sites f / 4 = 1/2, i.e. f = 5 ppm")
print("  for this dot: below that the nuclear bath is no longer the limit, charge")
print("  noise from oxide and interface defects is, and no isotope separation")
print("  touches it.")
print()
print("Three numbers a materials scientist should take away:")
print(f"  1. sigma_Overhauser ~ sqrt(f N_sites): removing 99.9% of the 29-Si buys")
print(f"     only a factor {np.sqrt(1e3):.1f} in coherence.")
print("  2. The dot must also be small enough for orbital and valley splittings to")
print("     exceed the operating temperature - a strain and interface problem.")
print("  3. Charge noise sets the floor, and it lives in the oxide, not the silicon.")
```

```text
Central limit check, natural Si, 200 explicit configurations:
  explicit std  69.000 a_hf
  sqrt(N f / 4) 68.557 a_hf
  explicit mean -0.427 a_hf (should be 0)

Ramsey free-induction decay, time in units of 1/a_hf:
            material         f     N_29  sigma (a_hf)  T2* (1/a_hf)    gain
   natural Si (4.7%)   4.7e-02    18800       68.5565    2.0628e-02    1.00
            1000 ppm   1.0e-03      400       10.0000    1.4142e-01    6.86
             200 ppm   2.0e-04       80        4.4721    3.1623e-01   15.33
              50 ppm   5.0e-05       20        2.2361    6.3246e-01   30.66
  predicted gain from 4.7% to 50 ppm: sqrt(4.7e-2/5e-5) = 30.66

Measured decay envelope (Monte Carlo) against the Gaussian prediction:
  natural Si (4.7%): T2* = 2.0628e-02 / a_hf
       t/T2*       <cos>   exp(-(t/T2*)^2)
        0.25    0.939976          0.939413
        0.50    0.780715          0.778801
        1.00    0.371234          0.367879
        1.50    0.105538          0.105399
        2.00    0.016772          0.018316
  200 ppm: T2* = 3.1623e-01 / a_hf
       t/T2*       <cos>   exp(-(t/T2*)^2)
        0.25    0.939462          0.939413
        0.50    0.779112          0.778801
        1.00    0.369357          0.367879
        1.50    0.107046          0.105399
        2.00    0.020746          0.018316

The scaling is sqrt(f), so purification has diminishing returns, and it
stops helping once another mechanism takes over:
         f   T2* (1/a_hf)   with a charge-noise cap at 2/a_hf
   4.7e-02     2.0628e-02                          2.0418e-02
   1.0e-03     1.4142e-01                          1.3208e-01
   1.0e-04     4.4721e-01                          3.6549e-01
   1.0e-05     1.4142e+00                          8.2843e-01
   1.0e-06     4.4721e+00                          1.3820e+00
  The crossover, where the nuclear and the competing contribution to 1/T2*
  are equal, is at T2* = 2/a_hf, i.e. N_sites f / 4 = 1/2, i.e. f = 5 ppm
  for this dot: below that the nuclear bath is no longer the limit, charge
  noise from oxide and interface defects is, and no isotope separation
  touches it.

Three numbers a materials scientist should take away:
  1. sigma_Overhauser ~ sqrt(f N_sites): removing 99.9% of the 29-Si buys
     only a factor 31.6 in coherence.
  2. The dot must also be small enough for orbital and valley splittings to
     exceed the operating temperature - a strain and interface problem.
  3. Charge noise sets the floor, and it lives in the oxide, not the silicon.
```

**What to look for.** The whole calculation is in units of the single-nucleus hyperfine coupling, so no device number enters and every conclusion is a ratio.

**The central limit theorem does the work.** An explicit draw of 400,000 sites, each carrying a $^{29}$Si nucleus with probability 4.7%, gives a standard deviation of 69.0 against the predicted $\sqrt{Nf/4} = 68.56$. That is why the Overhauser field is Gaussian, and why the Ramsey envelope is $\exp[-(t/T_2^\ast)^2]$ rather than exponential — a shape difference that distinguishes inhomogeneous dephasing from true $T_2$ decay in the laboratory.

**Purification buys $\sqrt{f}$, and that is a hard ceiling.** Going from natural silicon to 50 ppm removes 99.9% of the nuclear spins and improves $T_2^\ast$ by a factor of 30.7 — exactly $\sqrt{4.7\times10^{-2}/5\times10^{-5}}$. There is no way to do better within this mechanism; the incoherent sum is the incoherent sum. The measured Monte Carlo envelope matches $\exp[-(t/T_2^\ast)^2]$ to three decimals at both concentrations, confirming that the shape as well as the scale is understood.

**And then it stops mattering.** The last table adds a competing mechanism with a fixed rate. Below the crossover, further purification buys almost nothing, because the total dephasing rate is a sum. In real devices that competing mechanism is charge noise, coupled to the spin through the spin-orbit interaction and through $g$-factor variation, and it lives in the oxide and at the interface — not in the silicon crystal. Beyond a certain purity, buying better isotopes is buying the wrong thing.

**Three materials problems, not one.** The code's closing lines list them, and they are worth separating because they belong to different specialists. Isotopic purity is a chemistry and crystal-growth problem, solved in principle. Charge noise is an amorphous-oxide defect problem, shared with superconducting qubits and unsolved. And **valley splitting** — the near-degeneracy of the two lowest conduction-band valleys in strained silicon, which must be lifted by more than the operating temperature and more than $J$, and which depends on atomic-scale details of the Si/SiGe interface — is a heterostructure-growth problem that is arguably the most specifically materials-science-shaped obstacle in this entire course.

* * *

## 5.3 Topological Quantum Computation

### The idea, stated fairly

Every modality so far fights decoherence by isolating the qubit and then correcting the errors that get through. Topological quantum computation proposes something structurally different: **store the information in a way that no local operator can read or disturb.**

The concrete mechanism is a pair of Majorana zero modes — self-conjugate fermionic excitations, $\gamma^\dagger = \gamma$ — bound at the two ends of a one-dimensional topological superconductor. A single fermionic degree of freedom is shared nonlocally between them, so its state is encoded in a *joint* property of two objects separated by the length of the wire. Any perturbation that acts on one end alone cannot change it. The protection is not a large energy gap in the usual sense; it is the statement that the operator you would need in order to cause an error does not exist locally.

The minimal model, due to Kitaev, is a chain of spinless fermions with p-wave pairing:

$$ H = \sum_j \left[-w\left(c^\dagger_j c_{j+1} + \text{h.c.}\right) - \mu\left(n_j - \tfrac12\right) + \Delta\left(c_j c_{j+1} + \text{h.c.}\right)\right] $$

For $|\mu| < 2w$ the chain is topological and hosts a Majorana at each end; for $|\mu| > 2w$ it is trivial and gapped. Because the two Majoranas overlap exponentially little, the energy splitting of the degenerate pair is

$$ \varepsilon_0 \sim e^{-L/\xi} $$

and this is the quantitative content of "topological protection": **error suppression that is exponential in a length, with no error correction whatsoever.** If it worked, it would change the resource arithmetic of Chapter 1 completely.

Gates come from braiding: exchanging two Majoranas implements a discrete unitary that depends only on the topology of the exchange, not on how fast or how precisely it is done. Braiding alone is not universal — it generates only a Clifford-like subgroup for Majoranas — so a topological machine still needs one non-topological gate, supplied by magic-state distillation, exactly as in the surface-code accounting of the algorithms course.

### The Majorana decomposition, and what "non-Abelian" means

The statements above are easier to trust after seeing where the Majoranas come from, because nothing exotic is involved: it is a change of variables. Any fermionic mode can be split into two Hermitian halves,

$$ c_j = \frac{\gamma_{2j-1} + i\gamma_{2j}}{2}, \qquad \gamma_{2j-1} = c_j + c_j^\dagger, \quad \gamma_{2j} = i\left(c_j^\dagger - c_j\right) $$

and these satisfy $\gamma_a^\dagger = \gamma_a$, $\gamma_a^2 = 1$ and $\lbrace \gamma_a, \gamma_b\rbrace = 2\delta_{ab}$. For an ordinary fermion the two halves sit on top of each other and the split is bookkeeping. What the Kitaev chain does is *separate* them: one half localizes at each end of the wire, and the fermionic mode they jointly define — occupied or empty, which is the qubit — has no local carrier at all. Its occupation is read by the operator $i\gamma_L\gamma_R$, a product of one operator at each end, and that is the precise sense in which the information is nonlocal.

With $2N$ Majoranas and the total fermion parity fixed, the ground space is $2^{N-1}$-dimensional: four Majoranas store one qubit, six store two, and so on. **Non-Abelian statistics** is a statement about how that degenerate space transforms when two Majoranas are exchanged. Braiding $\gamma_a$ past $\gamma_b$ acts on the ground space as

$$ U_{ab} = \exp\left(\frac{\pi}{4}\gamma_a\gamma_b\right) = \frac{1}{\sqrt{2}}\left(1 + \gamma_a\gamma_b\right) $$

which is a *unitary operator on the degenerate space*, not a phase. That is the whole difference from ordinary bosons and fermions, where exchange multiplies the state by $+1$ or $-1$ and successive exchanges therefore commute. Here they do not. Take the four-Majorana case and a concrete representation, $\gamma_1 = X_1$, $\gamma_2 = Y_1$, $\gamma_3 = Z_1X_2$, $\gamma_4 = Z_1Y_2$, which satisfies all the algebra above. Restricted to the even-parity ground space, the braids come out as

$$ U_{12} = \exp\left(i\frac{\pi}{4}Z\right), \qquad U_{23} = \exp\left(i\frac{\pi}{4}X\right), \qquad U_{34} = \exp\left(i\frac{\pi}{4}Z\right) $$

so $U_{12}$ and $U_{23}$ are $\pi/2$ rotations about orthogonal axes and do not commute — the commutator is not small, it is maximal — while $U_{12}$ and $U_{34}$, which exchange disjoint pairs, commute exactly. Every braid commutes with the total parity, so no braid ever leaves the ground space. This also shows why braiding is not enough for computation: $\pi/2$ Pauli rotations are Clifford gates, and Clifford gates are classically simulable, so the magic-state cost mentioned above is not an oversight in the proposal but a consequence of the algebra.

There is one more signature worth naming, because it is the observable most directly tied to the encoding. Couple two topological superconductors through a weak link. A single *fermion* tunnelling across carries charge $e$, not $2e$, so its energy depends on the superconducting phase difference as $\pm\varepsilon_M\cos(\varphi/2)$ rather than $\cos\varphi$: advancing $\varphi$ by $2\pi$ does not return the junction to its initial state but *swaps the two fermion-parity branches*. The current-phase relation of a fixed-parity branch is therefore $4\pi$-periodic — the **fractional Josephson effect** — and it shows up in AC measurements as a halved Josephson frequency, or as missing odd Shapiro steps. It is a signature of exactly the right kind, because it tests the parity encoding rather than the local density of states. It is also fragile in exactly the way the next subsection describes: anything that flips the parity during the measurement restores the ordinary $2\pi$ periodicity.

### Code Example 5: The Kitaev Chain, and a Look-Alike

```python
"""The Kitaev chain: real topological protection, and a look-alike that has none.

BdG Hamiltonian of a p-wave chain,
H = sum_j [-w(c_j^dag c_j+1 + h.c.) - mu_j(n_j - 1/2) + D(c_j c_j+1 + h.c.)],
diagonalized in the Nambu basis (c_1..c_L, c_1^dag..c_L^dag). All energies in
units of the hopping w.

Convention: the many-body Hamiltonian is H = (1/2) Psi^dag H_BdG Psi with
Psi = (c, c^dag), so the POSITIVE eigenvalues of the matrix built below are
single-quasiparticle excitation energies, with no further factor of two. That
was checked against exact many-body diagonalization of a short chain: at
mu = 0, w = D = 1 the lowest many-body excitation is 2.0 w, and the lowest
positive BdG eigenvalue below is 2.0 w as well.
"""
import numpy as np


def bdg(mu, L, w=1.0, D=1.0):
    """2L x 2L BdG matrix; mu may be a scalar or a length-L array."""
    mu = np.full(L, mu, dtype=float) if np.isscalar(mu) else np.asarray(mu, float)
    h = np.diag(-mu)
    dl = np.zeros((L, L))
    for j in range(L - 1):
        h[j, j + 1] = h[j + 1, j] = -w
        dl[j, j + 1] = D
        dl[j + 1, j] = -D
    return np.block([[h, dl], [-dl.conj(), -h.conj()]])


def spectrum(mu, L, **kw):
    return np.linalg.eigh(bdg(mu, L, **kw))


def majorana_weight(vec, L):
    """Site-resolved weight |u_j|^2 + |v_j|^2 of a BdG eigenvector."""
    u, v = vec[:L], vec[L:]
    return np.abs(u) ** 2 + np.abs(v) ** 2


L = 40
print(f"Kitaev chain, L = {L}, w = D = 1. Topological phase requires |mu| < 2w.")
hdr = f"{'mu/w':>7}{'|E_0|/w':>12}{'|E_1|/w':>11}{'bulk gap/w':>13}{'phase':>14}"
print(hdr)
print("-" * len(hdr))
for mu in [0.0, 0.5, 1.0, 1.9, 2.0, 2.1, 3.0, 5.0]:
    w_, v_ = spectrum(mu, L)
    a = np.sort(np.abs(w_))
    phase = ("topological" if abs(mu) < 2 else
             "critical" if abs(mu) == 2 else "trivial")
    print(f"{mu:>7.1f}{a[0]:>12.3e}{a[1]:>11.3e}{a[2]:>13.4f}{phase:>14}")
print("  |E_0| and |E_1| are the pair +-eps of the zero mode; the bulk gap is next.")

print()
print("The near-zero mode is a pair of Majoranas, one at each end.")
w_, v_ = spectrum(1.0, L)
i0 = int(np.argmin(np.abs(w_)))
wt = majorana_weight(v_[:, i0], L)
print(f"  mu/w = 1.0, E = {w_[i0]:.3e}")
print(f"  weight on sites  1-4:  " + " ".join(f"{x:.4f}" for x in wt[:4]))
print(f"  weight on sites {L-3}-{L}: " + " ".join(f"{x:.4f}" for x in wt[-4:]))
print(f"  weight in the middle half: {wt[L // 4:3 * L // 4].sum():.3e}")

print()
print("Splitting of the zero mode vs chain length. At mu = 0 Kitaev's zero-mode")
print("equation gives a decay factor |x| = sqrt[(w-D)/(w+D)] per site, so a small")
print("pairing D gives a long, measurable localization length; take D = 0.1 w.")
print(f"{'L':>5}{'|E_0|/w':>14}{'ratio to L-10':>16}")
prev, Ls, es = None, [], []
for Lx in [20, 30, 40, 50, 60, 70]:
    wx, _ = spectrum(0.0, Lx, D=0.1)
    e0 = np.sort(np.abs(wx))[0]
    print(f"{Lx:>5}{e0:>14.4e}{(f'{e0 / prev:.5f}' if prev else '-'):>16}")
    prev = e0
    Ls.append(Lx)
    es.append(e0)
slope = np.polyfit(Ls, np.log(es), 1)[0]
print(f"  fitted decay length xi = {-1.0 / slope:.3f} sites")
print(f"  analytic xi = 2/ln[(w+D)/(w-D)] = {2.0 / np.log(1.1 / 0.9):.3f} sites")
wx, vx = spectrum(0.0, 70, D=0.1)
wt = majorana_weight(vx[:, int(np.argmin(np.abs(wx)))], 70)
print("  x is imaginary at mu = 0, so the envelope alternates; compare over 2 sites:")
print(f"  weight(j+2)/weight(j) at j = 4, 6, 8: "
      + ", ".join(f"{wt[j + 2] / wt[j]:.5f}" for j in (4, 6, 8)))
print(f"  predicted [(w-D)/(w+D)]^2 = {(0.9 / 1.1) ** 2:.5f}")
print("  That exponential is the protection: the two Majoranas cannot talk without")
print("  crossing the gapped bulk, so no local perturbation can split them.")

# --- the honest part ------------------------------------------------------
print()
print("Now the uncomfortable experiment. Put the chain firmly in the TRIVIAL phase")
print("(mu = 3w > 2w) but let mu rise smoothly from 0 over the first few sites, as")
print("any real electrostatic gate would.")
print(f"{'ramp length':>13}{'E_0/w':>12}{'left weight':>13}{'right weight':>14}"
      f"{'looks like':>16}")


def smooth_mu(L, mu_bulk, ramp):
    mu = np.full(L, mu_bulk, dtype=float)
    if ramp > 0:
        mu[:ramp] = mu_bulk * np.linspace(0.0, 1.0, ramp + 1)[1:]
    return mu


for ramp in [0, 4, 8, 12, 20]:
    mu = smooth_mu(L, 3.0, ramp)
    wx, vx = spectrum(mu, L)
    pos = np.argsort(np.abs(wx))
    i = pos[0]
    wt = majorana_weight(vx[:, i], L)
    left, right = wt[:L // 4].sum(), wt[3 * L // 4:].sum()
    e0 = np.sort(wx[wx > -1e-14])[0]
    verdict = "gapped" if e0 > 1e-2 else "a zero mode"
    print(f"{ramp:>13}{e0:>12.4e}{left:>13.4f}{right:>14.4f}{verdict:>16}")

print()
print("The nonlocality test that separates them. For each candidate zero mode,")
print("compare the weight at the two ends of the wire:")
for label, mu, D in [("topological, mu = 1.0 w", np.full(L, 1.0), 1.0),
                     ("trivial + 20-site ramp ", smooth_mu(L, 3.0, 20), 1.0)]:
    wx, vx = spectrum(mu, L, D=D)
    i = int(np.argmin(np.abs(wx)))
    wt = majorana_weight(vx[:, i], L)
    left, right = wt[:L // 4].sum(), wt[3 * L // 4:].sum()
    print(f"  {label}: E = {wx[i]:+.3e}, left {left:.4f}, right {right:.4f},"
          f" ratio {min(left, right) / max(left, right):.4f}")
print()
print("A tunnelling experiment on one end of the wire measures the local density of")
print("states and sees a zero-bias peak in both cases. Only the *nonlocal*")
print("measurement distinguishes them, and that is exactly the hard experiment.")
print("This is why 'we saw a zero-bias peak' and 'we have a Majorana qubit' are")
print("separated by a decade of work rather than a press release.")
```

```text
Kitaev chain, L = 40, w = D = 1. Topological phase requires |mu| < 2w.
   mu/w     |E_0|/w    |E_1|/w   bulk gap/w         phase
---------------------------------------------------------
    0.0   3.111e-17  6.661e-16       2.0000   topological
    0.5   3.034e-16  6.035e-16       1.5021   topological
    1.0   1.363e-12  1.365e-12       1.0065   topological
    1.9   2.651e-02  2.651e-02       0.2280   topological
    2.0   7.757e-02  7.757e-02       0.2326      critical
    2.1   1.528e-01  1.528e-01       0.2758       trivial
    3.0   1.016e+00  1.016e+00       1.0623       trivial
    5.0   3.009e+00  3.009e+00       3.0376       trivial
  |E_0| and |E_1| are the pair +-eps of the zero mode; the bulk gap is next.

The near-zero mode is a pair of Majoranas, one at each end.
  mu/w = 1.0, E = -1.363e-12
  weight on sites  1-4:  0.3751 0.0938 0.0234 0.0059
  weight on sites 37-40: 0.0059 0.0234 0.0937 0.3749
  weight in the middle half: 9.537e-07

Splitting of the zero mode vs chain length. At mu = 0 Kitaev's zero-mode
equation gives a decay factor |x| = sqrt[(w-D)/(w+D)] per site, so a small
pairing D gives a long, measurable localization length; take D = 0.1 w.
    L       |E_0|/w   ratio to L-10
   20    5.1588e-02               -
   30    1.8115e-02         0.35115
   40    6.5843e-03         0.36347
   50    2.4102e-03         0.36606
   60    8.8346e-04         0.36655
   70    3.2390e-04         0.36663
  fitted decay length xi = 9.881 sites
  analytic xi = 2/ln[(w+D)/(w-D)] = 9.967 sites
  x is imaginary at mu = 0, so the envelope alternates; compare over 2 sites:
  weight(j+2)/weight(j) at j = 4, 6, 8: 0.66942, 0.66942, 0.66942
  predicted [(w-D)/(w+D)]^2 = 0.66942
  That exponential is the protection: the two Majoranas cannot talk without
  crossing the gapped bulk, so no local perturbation can split them.

Now the uncomfortable experiment. Put the chain firmly in the TRIVIAL phase
(mu = 3w > 2w) but let mu rise smoothly from 0 over the first few sites, as
any real electrostatic gate would.
  ramp length       E_0/w  left weight  right weight      looks like
            0  1.0159e+00       0.1101        0.1101          gapped
            4  3.0731e-01       0.9994        0.0000          gapped
            8  2.6703e-02       0.9961        0.0000          gapped
           12  2.0923e-03       0.9536        0.0000     a zero mode
           20  1.1587e-05       0.5785        0.0000     a zero mode

The nonlocality test that separates them. For each candidate zero mode,
compare the weight at the two ends of the wire:
  topological, mu = 1.0 w: E = -1.363e-12, left 0.5002, right 0.4998, ratio 0.9993
  trivial + 20-site ramp : E = -1.159e-05, left 0.5785, right 0.0000, ratio 0.0000

A tunnelling experiment on one end of the wire measures the local density of
states and sees a zero-bias peak in both cases. Only the *nonlocal*
measurement distinguishes them, and that is exactly the hard experiment.
This is why 'we saw a zero-bias peak' and 'we have a Majorana qubit' are
separated by a decade of work rather than a press release.
```

**What to look for.** The first half of this output is the strongest quantitative argument in the chapter. The second half is the reason the modality is not further along.

**The zero mode is there, and it is nonlocal.** At $\mu = w$, $L = 40$ the splitting is $1.4\times10^{-12}$ in units of $w$ while the bulk gap is $1.01\,w$ — twelve orders of magnitude of separation, with no error correction, from a Hamiltonian with three parameters. (That gap is worth checking against the closed form: the bulk dispersion is $E(k) = \sqrt{(2w\cos k + \mu)^2 + 4\Delta^2\sin^2 k}$, whose minimum at $\mu = w = \Delta$ is exactly $1.0\,w$, and at $\mu = 0$ is $2\Delta = 2.0\,w$ — which is the first row of the table. The convention matters here and the code states it: the BdG matrix is defined by $H = \tfrac12\Psi^\dagger H_\mathrm{BdG}\Psi$, so its positive eigenvalues are single-quasiparticle energies. With that convention there is no residual factor of two between the BdG spectrum and the many-body one — exact diagonalization of a short chain at $\mu = 0$ puts its lowest excitation at $2.0\,w$ too. An earlier version of this code carried a stray $\tfrac12$ on the matrix and halved every energy in the table.) The wavefunction weight is 0.375 on the first site, 0.375 on the last, and $10^{-6}$ in the middle half of the chain. That is the encoding: the information is in the pair, and the middle of the wire does not know about it.

**The protection is exponential and the exponent is measurable.** With $\Delta = 0.1w$ the splitting falls by a factor 0.3666 for every ten sites, giving a fitted decay length of 9.88 sites against the analytic $2/\ln[(w+\Delta)/(w-\Delta)] = 9.97$. The wavefunction's own envelope ratio matches $[(w-\Delta)/(w+\Delta)]^2 = 0.66942$ to five digits. Two independent quantities — a spectral splitting and a spatial profile — agree with one closed form.

**Note also where protection fails.** At $\mu = 1.9w$, just inside the topological phase, the splitting at $L = 40$ is already $2.7\times10^{-2}$: the coherence length diverges at the phase boundary, so a wire that is nominally topological but close to the transition has no useful protection at all. Protection is exponential in $L/\xi$, and $\xi$ is a bulk property you must control.

**And now the uncomfortable half.** Take the chain firmly into the *trivial* phase, $\mu = 3w > 2w$ (note that $\mu = 2w$ exactly is the critical point, where the bulk gap closes, not a trivial gapped state — the table labels it as such), and let $\mu$ rise smoothly from zero over the first twenty sites, as any real electrostatic gate must. The result is a state at $E = 1.2\times10^{-5}$ — a zero mode by any local measurement — localized entirely at one end, with zero weight at the other. A tunnelling experiment on that end measures the local density of states and sees a zero-bias peak. **It looks the same.** These are the "quasi-Majorana" or trivial Andreev bound states that a smooth confining potential produces generically, and they carry no topological protection whatsoever: their energy moves continuously with any parameter, and both partners sit within reach of the same local perturbation.

**The test that separates them is nonlocality, and that is the hard experiment.** The printed comparison is decisive, and note that it is unaffected by the energy convention above — eigen*vectors* do not care about an overall scale. The topological mode has left/right weight $0.5002/0.4998$, ratio $0.9993$; the look-alike has $0.5785/0.0000$, ratio $0.0000$. A measurement at one end cannot tell the difference; a measurement correlating both ends can. This is why the honest status of this modality is that the *principle* is beautifully established — the code above establishes it, in fifty lines — while the *demonstration* requires ruling out a class of mundane alternatives that reproduce every local signature. That is not a reason to dismiss the idea. It is a reason to hold it to the standard of a nonlocal measurement, and to read any claim in that light.

### What would have to be true

To be concrete about the gap, a working topological qubit needs, simultaneously: a hard induced superconducting gap in a semiconductor with strong spin-orbit coupling, an epitaxial semiconductor-superconductor interface clean enough that the induced gap has no sub-gap states, disorder below the scale that closes the topological gap, and a wire long enough compared with $\xi$ that the splitting is negligible. Every one of those is a materials-growth specification, and they conflict: making the interface more transparent (for a hard gap) also makes the wire more susceptible to disorder from the superconductor. This is the reason the field's progress is measured in molecular-beam-epitaxy improvements rather than in qubit counts, and it is why a materials researcher reading this course should notice that this modality is *entirely* a materials problem.

### Quasiparticle poisoning, and why $e^{-L/\xi}$ is not the error rate

There is one more item on that list, and it is the one that most often goes missing when topological protection is described as a theorem. The protected quantity is a **fermion parity**. An error is therefore anything that changes the parity — and a single unpaired electron arriving from anywhere in the device does exactly that, in one hop, with no reference whatsoever to $L/\xi$. This is **quasiparticle poisoning**, and it is not a small correction to the exponential; it is a different and much larger error channel sitting alongside it.

The population responsible is one we have already counted. Chapter 2 measured the non-equilibrium quasiparticle fraction in an aluminium film as $x_{qp} \sim 10^{-8}$ to $10^{-6}$ — hundreds of unpaired electrons in a 1000 $\mu$m$^3$ film, generated by stray infrared photons above $2\Delta/h$, by cosmic rays and by phonon bursts, and *not* by temperature. A Majorana wire is proximitized by exactly that kind of film. The relevant device figure of merit is therefore the **parity lifetime** $\tau_p$, the mean time before one of those quasiparticles lands on the wire, and measurements of it in superconducting devices land in the microsecond-to-millisecond range depending almost entirely on shielding and filtering — the same engineering that Section 2.6 described, for the same reason.

Put that against the gate time and the arithmetic is sobering, but it has to be done in the same units, which means turning the splitting into a time rather than comparing a dimensionless energy with an error rate. Braiding must be adiabatic with respect to the bulk gap: with an induced gap of $100\ \mu$eV, $\hbar/\Delta = 6.6$ ps, so a braid of a nanosecond is comfortably adiabatic and the braid itself is not the bottleneck. The splitting of the code's $L = 40$ wire, $\varepsilon_0 \approx 10^{-12}\Delta = 1.4\times10^{-16}$ eV, accumulates a $\pi$ phase in $h/2\varepsilon_0 = 15$ s — so the exponential protection alone would allow about $1.5\times10^{10}$ nanosecond braids. The parity lifetime allows $\tau_p/t_\mathrm{braid} \sim 10^{-3}\ \mathrm{s}/10^{-9}\ \mathrm{s} = 10^6$, a respectable number, comparable with the best rows of the scorecard in §5.4 — and about *four orders of magnitude* short of what the splitting would permit. The exponential protection is real, and it is not the limit. Two consequences follow, and both are worth stating plainly:

  * **A topological qubit still needs error correction.** Parity errors are ordinary errors: uncorrelated, detectable by parity measurement, and correctable by a code. What topology buys is a favourable *starting* error rate on one axis, not the removal of the error-correction layer — which is why the scorecard's error-correction cell for this column reads "$L/\xi$ + magic" rather than "intrinsic".
  * **The materials problem is the same one twice.** The interface that has to produce a hard gap without sub-gap states is also the interface that must not host quasiparticle traps, and the shielding that protects a transmon from pair-breaking photons is the shielding that protects a parity. A materials researcher who improves either of those has improved both platforms at once, which is the argument of §5.5 arriving early.

* * *

## 5.4 The Scorecard

We now have six modalities and the six axes of Chapter 1: coherence, gate fidelity and speed, connectivity, reproducibility and yield, operating temperature, and scalability of control. The grid below has **eight** rows rather than six, because two of those axes split when you try to fill cells in: scalability separates into control wiring per qubit and error-correction overhead, which are limited by different things, and this course adds one row of its own — the materials limit. Gate *fidelity* is not a grid row at all; Part A handles it quantitatively, as coherent-operation counts derived from materials parameters. The purpose of this section is to put all of that on one table — and to do so in a way that will not be obsolete next year.

That constraint rules out the obvious approach. A table of qubit counts and record fidelities is a snapshot of a moving target, and it is also the wrong instrument: a record is a statement about one device in one laboratory, whereas what a researcher choosing a direction needs is a statement about what *limits* each approach. So the scorecard below contains no records. Every number in it is derived from a formula established in this course, evaluated at a materials parameter that the row names, and reported as an order of magnitude.

### Code Example 6: The Scorecard

```python
"""The scorecard: comparing modalities by the physics that limits each one.

No performance records appear here. Every number is derived from a formula in
this course, evaluated at a materials parameter that the row names, and is
reported as an order of magnitude only.
"""
import numpy as np

print("=" * 78)
print("PART A - coherent operations, derived from materials parameters")
print("=" * 78)

# --- superconducting: dielectric loss sets Q, anharmonicity sets gate time --
print()
print("Superconducting transmon: N = T2 / t_gate with 1/Q = p * tan(delta),")
print("T2 = Q/omega, and t_gate = k / alpha (k ~ 10 pulses' worth of anharmonicity).")
f_q, alpha, k = 5e9, 200e6, 10.0
print(f"  f_q = {f_q / 1e9:.0f} GHz, alpha/2pi = {alpha / 1e6:.0f} MHz, "
      f"t_gate = {k / alpha * 1e9:.0f} ns")
print(f"  {'participation p':>17}{'tan(delta)':>12}{'Q':>10}{'T2 (us)':>10}{'N_ops':>10}")
for p in [1e-2, 3e-3, 1e-3, 3e-4]:
    for tand in [1e-3]:
        Q = 1.0 / (p * tand)
        T2 = Q / (2 * np.pi * f_q)
        print(f"  {p:>17.1e}{tand:>12.1e}{Q:>10.2e}{T2 * 1e6:>10.2f}"
              f"{T2 / (k / alpha):>10.0f}")
print("  The whole column is an amorphous-oxide loss tangent times a geometric")
print("  participation ratio. That is a materials number, not an engineering one.")

# --- trapped ions: anomalous heating vs the sideband speed limit -----------
print()
print("Trapped ion: N_heat = 1/(nbar_dot * t_gate), the number of gates before the")
print("shared mode absorbs one motional quantum. The gate cannot beat the trap")
print("frequency, so t_gate >= 2pi/omega_trap, and the heating is a surface effect.")
print(f"  {'omega_trap/2pi (MHz)':>22}{'t_gate (us)':>13}"
      + "".join(f"{'ndot=' + str(n):>12}" for n in [1, 10, 100]))
for f_trap in [0.5, 1.0, 3.0]:
    t_gate = 1.0 / (f_trap * 1e6)
    row = "".join(f"{1.0 / (n * t_gate):>12.0f}" for n in [1, 10, 100])
    print(f"  {f_trap:>22.1f}{t_gate * 1e6:>13.2f}" + row)
print("  ndot is quanta per second of anomalous heating from the trap electrode")
print("  surface - again a materials number, and one nobody fully explains yet.")
print("  These are upper bounds from heating alone: in practice laser phase noise")
print("  and spectator motional modes bind first, so the achieved figure is lower.")

# --- neutral atoms: the Rydberg error floor from Chapter 4 -----------------
print()
print("Neutral atom: N = 1 / error_floor, error_floor ~ (gamma/V)^(2/3) from")
print("Chapter 4, Example 5, evaluated at Rb |70S> and |100S>.")
for label, err in [("n = 70, R = 4 um", 1.348e-3), ("n = 70, R = 6 um", 6.718e-3),
                   ("n = 100, R = 8 um", 7.255e-4)]:
    print(f"  {label:<20} error {err:.3e} -> N_ops ~ {1 / err:>7.0f}")
print("  Bounded above by the Rydberg lifetime, which is atomic physics: no")
print("  fabrication improvement changes it. The knob is n and the spacing.")

# --- silicon spins: charge noise at and off the sweet spot -----------------
print()
print("Silicon spin: N from Chapter 5, Example 3 - the exchange gate's phase error")
print("under a detuning offset, at and away from the symmetric sweet spot.")
for label, N in [("sweet spot, 1 ueV", 2.9e6), ("eps = 0.2U, 1 ueV", 2.3e3),
                 ("eps = 0.2U, 10 ueV", 2.3e2)]:
    print(f"  {label:<22} N_ops ~ {N:>9.1e}")
print("  The sweet-spot figure is optimistic: it assumes a static offset. Real 1/f")
print("  charge noise from oxide two-level defects samples many offsets per gate.")

# --- photons: loss per component ------------------------------------------
print()
print("Photon: N ~ 1/(1-eta) per component and 1/(1-V) per interference.")
print(f"  {'eta':>8}{'1/(1-eta)':>12}   {'visibility V':>14}{'1/(1-V)':>10}")
for eta, V in [(0.9, 0.90), (0.99, 0.99), (0.999, 0.999)]:
    print(f"  {eta:>8.3f}{1 / (1 - eta):>12.0f}   {V:>14.3f}{1 / (1 - V):>10.0f}")
print("  Loss lives in the waveguide, the detector and every interface; visibility")
print("  lives in how identical two emitters are. Both are materials problems.")

print()
print("=" * 78)
print("PART B - the scorecard grid: what limits each modality on each axis")
print("=" * 78)

MODS = ["SC", "ION", "ATOM", "PHOT", "SPIN", "TOPO"]
NAMES = {"SC": "superconducting", "ION": "trapped ion", "ATOM": "neutral atom",
         "PHOT": "photonic", "SPIN": "silicon spin", "TOPO": "topological"}

# Each cell names the *physical* limit on that axis - not a performance number.
GRID = {
    "gate speed":       ["10-100ns", "10-100us", "0.1-1us", "ps + herald",
                         "1-100ns", "ns-us adiab"],
    "coherence limit":  ["diel. loss", "heat/B-field", "Rydberg tau",
                         "loss only", "charge noise", "gap, L/xi"],
    "connectivity":     ["planar NN", "all-to-all", "programmable",
                         "routed,lossy", "planar NN", "braiding"],
    "temperature":      ["10 mK", "300 K", "300 K", "mixed", "0.1-1 K", "10 mK"],
    "qubit spread":     ["fabricated", "identical", "identical",
                         "emitter var.", "fabricated", "interface"],
    "wiring/qubit":     ["1+ coax", "optics/zone", "global laser",
                         "src+detector", "3+ gates", "many gates"],
    "error correction": ["surface code", "surface code", "surface code",
                         "loss-toler.", "surface code", "L/xi + magic"],
    "materials limit":  ["oxide TLS", "electrode", "none in qubit",
                         "photon src", "oxide+valley", "SC-SM iface"],
}

w = 14
print(f"{'axis':>16}" + "".join(f"{m:>{w}}" for m in MODS))
print("-" * (16 + w * len(MODS)))
for axis, row in GRID.items():
    print(f"{axis:>16}" + "".join(f"{c:>{w}}" for c in row))
print()
for k, v in NAMES.items():
    print(f"  {k:<6}= {v}")

print()
print("Read the grid honestly and it says two things at once.")
print("  (1) ATOM leads or ties on four of the eight rows - temperature, qubit")
print("      spread, wiring per qubit, materials limit - so on today's")
print("      operational axes it is the least constrained column.")
print("  (2) The rows that dominate the cost of a fault-tolerant machine are the")
print("      ones no column wins. Four of six inherit the same surface-code")
print("      overhead; the two that do not (photonic loss-tolerant codes,")
print("      topological protection plus magic states) swap it for a different")
print("      unsolved cost.")
print("So the rows do NOT all order the six differently: three rows contain ties")
print("and the error-correction row is nearly uniform. The result is sharper than")
print("'no winner'. Leading on the operational axes does not yet buy a lower")
print("fault-tolerance bill, and each column's remaining bill is a different")
print("materials problem.")
```

```text
==============================================================================
PART A - coherent operations, derived from materials parameters
==============================================================================

Superconducting transmon: N = T2 / t_gate with 1/Q = p * tan(delta),
T2 = Q/omega, and t_gate = k / alpha (k ~ 10 pulses' worth of anharmonicity).
  f_q = 5 GHz, alpha/2pi = 200 MHz, t_gate = 50 ns
    participation p  tan(delta)         Q   T2 (us)     N_ops
            1.0e-02     1.0e-03  1.00e+05      3.18        64
            3.0e-03     1.0e-03  3.33e+05     10.61       212
            1.0e-03     1.0e-03  1.00e+06     31.83       637
            3.0e-04     1.0e-03  3.33e+06    106.10      2122
  The whole column is an amorphous-oxide loss tangent times a geometric
  participation ratio. That is a materials number, not an engineering one.

Trapped ion: N_heat = 1/(nbar_dot * t_gate), the number of gates before the
shared mode absorbs one motional quantum. The gate cannot beat the trap
frequency, so t_gate >= 2pi/omega_trap, and the heating is a surface effect.
    omega_trap/2pi (MHz)  t_gate (us)      ndot=1     ndot=10    ndot=100
                     0.5         2.00      500000       50000        5000
                     1.0         1.00     1000000      100000       10000
                     3.0         0.33     3000000      300000       30000
  ndot is quanta per second of anomalous heating from the trap electrode
  surface - again a materials number, and one nobody fully explains yet.
  These are upper bounds from heating alone: in practice laser phase noise
  and spectator motional modes bind first, so the achieved figure is lower.

Neutral atom: N = 1 / error_floor, error_floor ~ (gamma/V)^(2/3) from
Chapter 4, Example 5, evaluated at Rb |70S> and |100S>.
  n = 70, R = 4 um     error 1.348e-03 -> N_ops ~     742
  n = 70, R = 6 um     error 6.718e-03 -> N_ops ~     149
  n = 100, R = 8 um    error 7.255e-04 -> N_ops ~    1378
  Bounded above by the Rydberg lifetime, which is atomic physics: no
  fabrication improvement changes it. The knob is n and the spacing.

Silicon spin: N from Chapter 5, Example 3 - the exchange gate's phase error
under a detuning offset, at and away from the symmetric sweet spot.
  sweet spot, 1 ueV      N_ops ~   2.9e+06
  eps = 0.2U, 1 ueV      N_ops ~   2.3e+03
  eps = 0.2U, 10 ueV     N_ops ~   2.3e+02
  The sweet-spot figure is optimistic: it assumes a static offset. Real 1/f
  charge noise from oxide two-level defects samples many offsets per gate.

Photon: N ~ 1/(1-eta) per component and 1/(1-V) per interference.
       eta   1/(1-eta)     visibility V   1/(1-V)
     0.900          10            0.900        10
     0.990         100            0.990       100
     0.999        1000            0.999      1000
  Loss lives in the waveguide, the detector and every interface; visibility
  lives in how identical two emitters are. Both are materials problems.

==============================================================================
PART B - the scorecard grid: what limits each modality on each axis
==============================================================================
            axis            SC           ION          ATOM          PHOT          SPIN          TOPO
----------------------------------------------------------------------------------------------------
      gate speed      10-100ns      10-100us       0.1-1us   ps + herald       1-100ns   ns-us adiab
 coherence limit    diel. loss  heat/B-field   Rydberg tau     loss only  charge noise     gap, L/xi
    connectivity     planar NN    all-to-all  programmable  routed,lossy     planar NN      braiding
     temperature         10 mK         300 K         300 K         mixed       0.1-1 K         10 mK
    qubit spread    fabricated     identical     identical  emitter var.    fabricated     interface
    wiring/qubit       1+ coax   optics/zone  global laser  src+detector      3+ gates    many gates
error correction  surface code  surface code  surface code   loss-toler.  surface code  L/xi + magic
 materials limit     oxide TLS     electrode none in qubit    photon src  oxide+valley   SC-SM iface

  SC    = superconducting
  ION   = trapped ion
  ATOM  = neutral atom
  PHOT  = photonic
  SPIN  = silicon spin
  TOPO  = topological

Read the grid honestly and it says two things at once.
  (1) ATOM leads or ties on four of the eight rows - temperature, qubit
      spread, wiring per qubit, materials limit - so on today's
      operational axes it is the least constrained column.
  (2) The rows that dominate the cost of a fault-tolerant machine are the
      ones no column wins. Four of six inherit the same surface-code
      overhead; the two that do not (photonic loss-tolerant codes,
      topological protection plus magic states) swap it for a different
      unsolved cost.
So the rows do NOT all order the six differently: three rows contain ties
and the error-correction row is nearly uniform. The result is sharper than
'no winner'. Leading on the operational axes does not yet buy a lower
fault-tolerance bill, and each column's remaining bill is a different
materials problem.
```

**What to look for.** Read Part A row by row, because the point is not the numbers but where they come from.

**Superconducting: the coherent-operation count is a loss tangent.** $N = T_2/t_{\text{gate}}$ with $T_2 = Q/\omega$ and $1/Q = p\tan\delta$ gives 64 operations at a participation ratio of $10^{-2}$ and 2122 at $3\times10^{-4}$. There is no engineering quantity in that formula: $\tan\delta$ is a property of an amorphous oxide, and $p$ is a geometric participation ratio. Chapter 2's central claim — that superconducting qubit coherence is a dielectric-loss problem — is this arithmetic.

**Trapped ions: the bound is a surface property.** $N_{\text{heat}} = 1/(\dot{\bar{n}}\,t_{\text{gate}})$ with the gate no faster than the trap frequency gives $10^4$ to $10^6$ before the shared mode absorbs one quantum. As the code says, this is an upper bound from heating alone — laser phase noise and spectator modes bind first — but the *structural* point stands: the quantity in the denominator is anomalous heating from an electrode surface, which is a materials phenomenon nobody fully explains.

**Neutral atoms: the bound is a radiative lifetime.** From Chapter 4's own optimization, $1/\varepsilon$ is 742 at $n=70$, $R = 4$ µm and 1378 at $n = 100$, $R = 8$ µm. No fabrication step appears, and no fabrication improvement helps.

**Silicon spins: the bound is a sweet spot.** $2.9\times10^6$ at the symmetric point against $2.3\times10^2$ off it — the largest spread of any row, and the one most under the experimenter's control. It is also the most optimistic figure in Part A, for the reason the code states.

**Photons: the bound is a transmission.** $1/(1-\eta)$ per component and $1/(1-V)$ per interference, and both are interface and emitter properties. The grid's "loss only" cell for photonic coherence means exactly that: there is no dephasing term to improve, and the entire budget is a loss-and-indistinguishability budget.

Then read Part B as a whole rather than cell by cell, and read it more carefully than a slogan. It is *not* true that every row orders the six differently: three rows contain outright ties (ions and atoms are both at 300 K and both perfectly uniform) and the error-correction row is nearly a constant, with four of the six modalities inheriting the same surface-code overhead. What the grid actually shows is two things at once.

First, **neutral atoms lead or tie on four of the eight rows** — operating temperature, qubit uniformity, wiring per qubit and materials limit — which makes that column the least constrained on the axes that describe operating a device *today*. Superconducting circuits lead on gate speed and lose on temperature and uniformity; ions tie atoms on uniformity, lead on connectivity, and lose on speed and wiring; photons lead on idle coherence and lose on determinism and loss; spins lead on manufacturability and lose on charge noise; topological qubits would lead the coherence column if the material existed.

Second, **no column leads the rows that dominate the cost of a fault-tolerant machine.** Error-correction overhead is the same surface-code bill for four modalities; the two that escape it do not escape cheaply — photonics substitutes loss-tolerant codes whose thresholds are quoted as a transmission per component, and topology substitutes exponential memory protection that still needs magic states for universality, as §5.3 said explicitly. So an operational lead does not convert into a lower fault-tolerance bill, and each column's remaining bill is a different materials problem.

There is no winner column, and this is a result rather than a hedge. A single figure of merit would require a weighting of the axes, and the correct weighting depends on the application: an analog quantum simulation of a frustrated magnet wants programmable connectivity and tolerates a duty cycle, while a fault-tolerant chemistry calculation wants depth above everything and will pay any amount of hardware for it. The honest summary is that the field has six approaches, each of which has cleared the physics and is now waiting on a *different* materials problem.

* * *

## 5.5 What This Means for a Materials Researcher

This course has had one argument, made six times, and this is the place to state it plainly.

**Every modality is bottlenecked by a materials problem, and they are different problems.** Not "engineering problems" and not "scaling problems" in the vague sense — specific, identifiable defects, interfaces and impurities:

| Modality | The bottleneck | What kind of problem it is |
| --- | --- | --- |
| Superconducting | two-level defects in surface oxides and at the metal-substrate interface; participation ratio | amorphous-materials physics, surface chemistry, deposition |
| Trapped ion | anomalous heating from electrode surfaces, scaling as $d^{-4}$ and unexplained | surface science of patch potentials and adsorbates |
| Neutral atom | none in the qubit — the limit is a radiative lifetime | *no knob in the qubit; the materials work is in the vacuum surfaces and the optics* |
| Photonic | deterministic, indistinguishable single-photon sources; loss per component | epitaxial quantum dots, low-loss photonic integration, detectors |
| Silicon spin | $1/f$ charge noise from the gate oxide; valley splitting at the Si/SiGe interface; residual $^{29}$Si | oxide defect physics, heterostructure growth, isotope separation |
| Topological | hard induced gap without sub-gap states; disorder; epitaxial semiconductor-superconductor interface | the whole modality is a materials problem |

Read that table twice. The first reading says the field is stuck. The second reading is the useful one: **this is a list of open problems in materials science whose solution would move quantum hardware, and most of them are not being worked on by quantum information scientists**, because they are problems in dielectric loss, surface adsorbates, epitaxy and isotope chemistry.

Three specific observations follow.

**First, the same defect appears twice.** Two-level fluctuators in amorphous oxides limit superconducting qubits through dielectric loss (Chapter 2) and silicon spin qubits through charge noise (§5.2). These are the two most industrially invested modalities, and they are limited by the same class of object — a bistable atomic configuration in a disordered oxide. A materials-level understanding of that object, of the kind that would let you predict $\tan\delta$ from a deposition recipe, would be a contribution to both at once. It is also, notably, a hard problem in the physics of amorphous solids that predates quantum computing by decades.

**Second, the neutral-atom row is the exception that proves the rule.** Chapter 4 showed that removing fabrication from the qubit does not remove the limit; it relocates it into the atom, where it becomes a radiative lifetime that no process can change. This is worth sitting with, because it says the materials problems in the other five rows are not a sign of immaturity — they are the *price of tunability*. A fabricated qubit can be designed; a natural qubit cannot. Every modality that offers design freedom pays for it with a material.

**Third, the measurement and control problems are also materials problems.** The wiring bottleneck of superconducting processors is a thermal and microwave-materials problem in the cabling and packaging. The duty-cycle bottleneck of neutral atoms is a vacuum-surface problem. Photonic loss is an interface problem. None of these is the qubit, and all of them are in the critical path.

### What to do with this

If you work on materials and want to contribute here, the useful questions are unglamorous and specific: what is the microscopic structure of a two-level fluctuator in aluminium oxide, and how does it depend on the deposition process? What adsorbate or patch-potential mechanism produces anomalous heating above a gold surface, and why does it scale as it does? What sets valley splitting at a Si/SiGe interface at the atomic scale, and can it be engineered rather than measured? How do you grow a semiconductor-superconductor interface that induces a hard gap without importing disorder? Can a solid-state emitter be made spectrally identical to another one on the same chip?

None of those questions requires a quantum computer to be built in order to be worth answering, and all of them are on the critical path if one is going to be. That asymmetry — the work is valuable whether or not the machine arrives — is the most defensible reason for a materials researcher to be in this field at all.

And the corresponding piece of literacy: when you next encounter a quantum hardware claim, ask which of the axes of Chapter 1 it moves, and which materials limit it addresses. If the answer to the second question is "none", you are looking at a demonstration rather than a step. Both exist, both get published, and telling them apart is what this course was for.

* * *

## Exercises

Work through these with the code from this chapter in front of you. Solutions follow each question.

#### Exercise 1: Reading a HOM Dip

An experiment reports a HOM coincidence dip with visibility $V = 0.94$ using two photons from separate quantum dots. (a) What mode overlap $|\gamma|$ does that correspond to? (b) If the only imperfection were a timing jitter between the two photons, what jitter in units of the wavepacket duration $\sigma$ would explain it? (c) The group later reports $V = 0.99$ after adding a narrowband spectral filter. Explain the mechanism, and state what was necessarily lost. (d) In a fusion-based architecture requiring $10^6$ successful entangling operations, what visibility is needed, and what does that imply about filtering?

<details><summary>Solution</summary>
<p><strong>(a)</strong> \(V = 1 - 2P_{cc}\) and \(P_{cc} = (1-|\gamma|^2)/2\), so \(V = |\gamma|^2\) and \(|\gamma| = \sqrt{0.94} = 0.9695\).</p>
<p><strong>(b)</strong> For Gaussian wavepackets, \(|\gamma| = \exp[-\tau^2/(8\sigma^2)]\), so \(\tau = \sigma\sqrt{-8\ln 0.9695} = \sigma\sqrt{0.2478} = 0.498\,\sigma\). Code Example 1's table confirms it: a delay of \(0.5\,\sigma\) gives \(|\gamma| = 0.96923\) and \(V = 0.9394\).</p>
<p><strong>(c)</strong> Two emitters in different local environments differ in centre frequency and linewidth. A narrowband filter transmits only the common part of the two spectra, so the <em>transmitted</em> photons are more nearly identical and \(|\gamma|\) rises. What was necessarily lost is photons: the filter discards the non-overlapping spectral weight, so the source brightness falls. Filtering trades \(\eta\) against \(V\), and Code Example 2 showed that \(\eta\) enters as \(\eta^N\) over a path — so the trade is not obviously favourable and must be evaluated for the specific architecture.</p>
<p><strong>(d)</strong> With \(10^6\) operations each failing with probability \(1-V\), a total error near unity requires \(1 - V \lesssim 10^{-6}\), i.e. \(V \gtrsim 0.999999\) — and with a loss-tolerant code, the requirement softens to the code's threshold but stays far beyond \(0.99\). Filtering alone cannot get there, because the required transmission would be prohibitive; the emitters must actually be identical. This is why "indistinguishability" is stated as a growth and strain-engineering objective rather than an optics one.</p>
</details>

#### Exercise 2: Designing an Exchange Gate

Using Code Example 3 with $U = 3000$ µeV: (a) What tunnel coupling gives an exchange gate time of 10 ns at zero detuning? (b) Your device has 1 µeV rms detuning noise. Estimate the number of gates available at $\varepsilon = 0$ and at $\varepsilon = 0.5U$, and comment on the ratio. (c) Why can you not simply increase $t$ until the gate is fast enough to beat the noise? Give two independent reasons. (d) Both the superconducting transmon of Chapter 2 and this device are operated at a point where a first derivative vanishes. State the general principle and one difference between the two cases.

<details><summary>Solution</summary>
<p><strong>(a)</strong> A gate time \(h/2J = 10\) ns means \(J/h = 50\) MHz, i.e. \(J = 50/242 = 0.207\) µeV. From \(J \simeq 4t^2/U\), \(t = \sqrt{JU}/2 = \sqrt{0.207\times3000}/2 = 12.5\) µeV. Interpolating Code Example 3's table between \(t = 10\) (15.5 ns) and \(t = 20\) (3.9 ns) is consistent.</p>
<p><strong>(b)</strong> Code Example 3 gives \(2.9\times10^{6}\) gates at \(\varepsilon = 0\) and 734 at \(\varepsilon = 0.5U\), a ratio of about 4000. The sweet spot is worth roughly three and a half orders of magnitude, which is larger than any plausible improvement in the noise itself — biasing correctly is a better investment than reducing \(1/f\) noise by a factor of a few.</p>
<p><strong>(c)</strong> First, \(J\) must stay well below the singlet-triplet splitting scale set by other terms — in particular, once \(J\) approaches the Zeeman splitting or the valley splitting, the two-level description fails and leakage into other states appears. Second, \(J \simeq 4t^2/U\) is a perturbative result valid for \(t \ll U\); Code Example 3's \(t = 200\) row already shows \(J = 52.4\) against the perturbative 53.3, and at larger \(t\) the singlet acquires real double occupancy, which reintroduces exactly the charge character that made it noise-sensitive. Faster gates are more charge-like gates.</p>
<p><strong>(d)</strong> The principle: operate where \(\partial(\text{qubit frequency})/\partial(\text{noisy parameter}) = 0\), so that noise enters only at second order. The transmon does this with respect to charge (by making \(E_J/E_C\) large enough that the charge dispersion is exponentially flat); the double dot does it with respect to detuning (by symmetry between the two virtual charge states). The difference: the transmon's insensitivity is a <em>fixed design choice</em> that cannot be tuned away, whereas the spin qubit's sweet spot is a <em>bias point</em> that must be found and maintained against drift — and the same gate voltages that set it are the ones the noise acts on.</p>
</details>

#### Exercise 3: Is Purification Worth It?

Using Code Example 4's model: (a) Show that improving from 800 ppm to 50 ppm improves $T_2^\ast$ by exactly a factor of 4, and state the general rule. (b) Suppose a competing mechanism independently limits $T_2^\ast$ to twice the value that 800 ppm silicon gives. What total improvement does purification to 50 ppm now deliver? (c) A colleague proposes reducing the dot size to shrink $N_{\text{sites}}$ instead. What does that do to $\sigma_{\text{Overhauser}}$, and what does it do to the two other constraints named at the end of §5.2? (d) State the general lesson for spending effort on a decoherence channel.

<details><summary>Solution</summary>
<p><strong>(a)</strong> \(T_2^\ast \propto f^{-1/2}\), and \(800/50 = 16\), so the improvement is \(\sqrt{16} = 4\). The rule: because the nuclear spins add incoherently, every factor of 100 in isotopic purity buys a factor of 10 in coherence. That is a punishing exchange rate for a process as expensive as isotope separation.</p>
<p><strong>(b)</strong> Let \(T_{800}\) be the 800 ppm value and the competing mechanism give \(2T_{800}\). At 800 ppm the total is \(1/(1/T_{800} + 1/2T_{800}) = 0.667\,T_{800}\). At 50 ppm the nuclear contribution is \(4T_{800}\), so the total is \(1/(1/4T_{800} + 1/2T_{800}) = 1.333\,T_{800}\). The improvement is \(2.0\times\), not \(4\times\) — half the benefit has been eaten. This is the crossover the last table of Code Example 4 exhibits.</p>
<p><strong>(c)</strong> \(\sigma \propto \sqrt{f N_{\rm sites}}\), so halving the dot volume improves \(T_2^\ast\) by \(\sqrt{2}\) — the same weak square root. Meanwhile a smaller dot raises the orbital and valley splittings, which is <em>good</em> for both of the other constraints, and it raises the charging energy, which changes \(J\) and the operating point. So shrinking the dot is a real and multi-purpose lever, which is one reason silicon dots are made as small as lithography allows. It does not, however, escape the square root.</p>
<p><strong>(d)</strong> Compute the crossover before investing. A channel that scales as a square root and competes with a channel that does not will stop being the limit at a predictable point, and effort spent past that point buys nothing. The same logic applied to zero-noise extrapolation in the algorithms course and to Rydberg \(n\) in Chapter 4: know which mechanism binds, and at what value of the knob it stops binding.</p>
</details>

#### Exercise 4: Distinguishing a Majorana From a Look-Alike

Using Code Example 5: (a) At $\mu = 1.9w$ and $L = 40$ the splitting is $2.7\times10^{-2}\,w$. Estimate the length needed to reach $10^{-6}\,w$ if the decay length there is 8 sites, and comment. (b) The 20-site smooth ramp in the trivial phase gave $E_0 = 1.2\times10^{-5}\,w$. Name two experimental signatures it shares with the true topological mode, and the one it does not. (c) Design a measurement, in terms of the quantities the code prints, that would distinguish them. (d) A paper reports a zero-bias conductance peak quantized at $2e^2/h$ and claims Majorana observation. State two mundane explanations that must be excluded.

<details><summary>Solution</summary>
<p><strong>(a)</strong> \(\varepsilon \sim e^{-L/\xi}\), so \(L = 40 + 8\ln(2.7\times10^{-2}/10^{-6}) = 40 + 8 \times 10.2 = 122\) sites, three times longer. That is the point: near the transition \(\xi\) grows, and the required wire length grows with it. Since \(\xi\) depends on \(\mu\), \(w\) and \(\Delta\) — all of which vary along a real wire because of disorder and gate non-uniformity — a wire that is topological on average may be marginal in places. Protection is exponential in \(L/\xi\), and \(\xi\) is not under direct control.</p>
<p><strong>(b)</strong> Shared: (i) an energy essentially at zero on the scale of the gap, and (ii) a wavefunction localized at the wire end, hence a zero-bias peak in the local density of states measured by end tunnelling. Not shared: the partner. The topological mode has half its weight at the <em>far</em> end (0.5002 / 0.4998); the look-alike has 0.5785 / 0.0000. The distinguishing property is nonlocality, not the peak.</p>
<p><strong>(c)</strong> Measure at both ends and correlate. Concretely: a two-terminal experiment that measures nonlocal conductance, or a Coulomb-blockade island whose \(1e\)-periodic charging signature requires a nonlocally shared fermion parity, or an interferometer sensitive to the joint parity of the two ends. In the code's language, the observable is the ratio \(\min(\text{left},\text{right})/\max(\text{left},\text{right})\): 0.9993 for the topological mode, 0.0000 for the look-alike. Any measurement confined to one end cannot produce that number.</p>
<p><strong>(d)</strong> (i) A trivial Andreev bound state from a smooth confining potential, exactly as constructed in the code — these can produce peaks at or near zero bias, and with fine tuning can approach quantized height. (ii) Disorder-induced sub-gap states or a soft induced gap, which produce zero-bias features from an entirely different mechanism; and relatedly, instrumental effects (finite temperature, tunnel-barrier transmission, dissipation) that broaden and flatten a non-quantized peak into something resembling a plateau. The decisive response to both is a nonlocal measurement and a demonstration that the feature is <em>not</em> removable by continuous parameter changes.</p>
</details>

#### Exercise 5: Choosing a Direction

You lead a materials group with expertise in thin-film deposition and interface characterization, and you want to contribute to quantum hardware. (a) Using the table in §5.5, name the two modalities where your expertise is most directly on the critical path, and the specific problem in each. (b) For one of them, state the measurable quantity your group would report and the formula in this course that converts it into a qubit figure of merit. (c) Name one modality where your expertise is essentially irrelevant, and explain why in one sentence. (d) A funding call asks you to justify the work "even if quantum computing does not succeed". Give a defensible answer.

<details><summary>Solution</summary>
<p><strong>(a)</strong> Superconducting qubits — two-level defects in surface and interface oxides, quantified by a dielectric loss tangent and a participation ratio — and silicon spin qubits, where the same class of oxide defect produces \(1/f\) charge noise and where the Si/SiGe interface sets valley splitting. Both are deposition and interface problems, and both are in the critical path for the two most industrially invested modalities. (Topological qubits are also purely a materials problem and would qualify if the group's expertise extends to epitaxial semiconductor-superconductor growth.)</p>
<p><strong>(b)</strong> For the superconducting case: report \(\tan\delta\) of the deposited dielectric and the interface participation ratio \(p\) of a test geometry. Code Example 6 converts them: \(Q = 1/(p\tan\delta)\), \(T_2 = Q/\omega\), and \(N = T_2/t_{\rm gate}\) with \(t_{\rm gate} \approx 10/\alpha\). A factor-of-three reduction in \(p\tan\delta\) is a factor of three in coherent operations, and that sentence is the bridge between a materials measurement and a quantum-information figure of merit.</p>
<p><strong>(c)</strong> Neutral atoms. The qubit contains no fabricated material at all, so its error floor \(\sim(\gamma/V)^{2/3}\) is set by a radiative lifetime that deposition cannot touch; the materials work in that platform is in the vacuum chamber and the optics, not in the qubit. (Note this is a statement about the qubit: optical coatings and vacuum surfaces are real materials work.)</p>
<p><strong>(d)</strong> Every problem named in (a) is an open question in the physics of disordered solids and of buried interfaces, independent of quantum computing. What is a two-level fluctuator in aluminium oxide, microscopically? What controls valley splitting at an atomic-scale-rough Si/SiGe interface? These questions predate the field, bear on dielectric reliability, low-temperature electronics, precision measurement and low-noise sensing, and would be worth answering if no quantum computer were ever built. Quantum hardware supplies something valuable in return: it is an exquisitely sensitive metrology platform for exactly these defects, capable of detecting individual fluctuators. The correct framing is a two-way exchange, not a bet on an outcome.</p>
</details>

* * *

## Summary

### Key Takeaways

**1\. Photons have no idle decoherence and cannot interact**

  * HOM interference gives coincidence probabilities of exactly 0 (bosons), 1 (fermions) and 1/2 (distinguishable particles) — the only two-photon effect linear optics provides.
  * $P_{cc} = (1 - |\langle\phi_1|\phi_2\rangle|^2)/2$, so the dip depth *measures* indistinguishability; a 1.5× pulse-duration mismatch alone costs 8% of visibility.
  * In a fusion-based architecture every entangling operation is an HOM interference, so $1 - V$ is an error per gate — set by how identical two emitters are.

**2\. Nondeterminism, not decoherence, is the photonic architecture problem**

  * Postselected gates succeed with $p = 1/9$; twenty of them need $8\times10^{18}$ trials, or 267 years at 1 GHz. Postselection is not an architecture.
  * Multiplexing to 99% on 100 gates costs 79 copies per slot, about 7900 heralded resources for a 100-pulse circuit.
  * Loss is leakage, not a Pauli error: $\eta = 0.99$ over 100 components delivers 37% of the time, and photonic fault tolerance needs loss-tolerant codes quoted in transmission per component.
  * Linear-network amplitudes are permanents; Ryser turns $n!$ into $2^n n$, still exponential. That gap is what a sampling-advantage claim is about, and it is not about ground-state energies.

**3\. Spin qubits trade CMOS compatibility for charge noise**

  * The device Hamiltonian is the two-site Hubbard model from the algorithms course; the singlet-triplet exchange is $J = 4t^2U/(U^2-\varepsilon^2)$, reducing to $4t^2/U$ at zero detuning.
  * The triplet sits exactly at zero energy because Pauli exclusion forbids it from hopping — the mechanism appears as an unliftable degeneracy.
  * $J$ spans three decades (0.033 to 52 µeV, a factor of 1570) over a plausible tunnel-coupling range, which is both the convenience and the exposure of the modality. $Jt/\hbar = \pi$ is a SWAP; the entangler is $\sqrt{\mathrm{SWAP}}$ at half that time.
  * The symmetric-exchange sweet spot ($\mathrm{d}J/\mathrm{d}\varepsilon = 0$) is worth roughly $10^3$–$10^4$ in gate count and costs only a choice of bias — the same design principle as the transmon's charge insensitivity in Chapter 2.

**4\. Isotopic purification is a decoherence channel removable by chemistry, with a hard $\sqrt{f}$ ceiling**

  * $\sigma_{\text{Overhauser}} \propto \sqrt{f N_{\text{sites}}}$, verified against explicit nuclear-spin draws; the Gaussian bath gives $\exp[-(t/T_2^\ast)^2]$, not exponential decay.
  * Natural silicon to 50 ppm improves $T_2^\ast$ by 30.7 — exactly $\sqrt{940}$ — and every factor of 100 in purity buys only 10 in coherence.
  * Once a competing mechanism is comparable, further purification buys almost nothing; in real devices that mechanism is oxide charge noise.
  * Three distinct materials problems, belonging to three different specialists: isotope chemistry (solved), oxide defects (shared with superconducting qubits, unsolved), and valley splitting at the Si/SiGe interface (a heterostructure-growth problem).

**5\. Topological protection is real in the model and hard to demonstrate in a wire**

  * The Kitaev chain at $\mu = w$, $L = 40$ gives a splitting of $1.4\times10^{-12}\,w$ against a bulk gap of $1.01\,w$ (analytically $1.0\,w$; the gap is $2\Delta = 2.0\,w$ at $\mu = 0$) — twelve orders of magnitude with no error correction.
  * The splitting falls as $e^{-L/\xi}$ with a measured $\xi = 9.88$ sites against the analytic 9.97; but $\xi$ diverges at the phase boundary, so a marginally topological wire has no useful protection.
  * A smooth potential ramp in the *trivial* phase produces a state at $1.2\times10^{-5}\,w$ localized at one end — every local signature of a Majorana, and none of the protection.
  * The distinguishing observable is nonlocality: left/right weight ratio 0.9993 for the topological mode against 0.0000 for the look-alike. One-ended tunnelling cannot tell them apart, and that is why the demonstration is hard rather than why the idea is wrong.
  * $c_j = (\gamma_{2j-1} + i\gamma_{2j})/2$ separates one fermion into two Hermitian halves, and braiding acts on the degenerate ground space as $U_{ab} = (1+\gamma_a\gamma_b)/\sqrt2$ — a unitary, not a phase. In the four-Majorana case $U_{12}$ and $U_{23}$ are $\pi/2$ rotations about orthogonal axes and do not commute; being Clifford, they also cannot be universal without magic states. The $4\pi$-periodic Josephson effect is the signature that tests the parity encoding rather than a local density of states.
  * The real error channel is quasiparticle poisoning: one unpaired electron flips the protected parity regardless of $L/\xi$. With parity lifetimes of microseconds to milliseconds against nanosecond adiabatic braids, the available operation count is $\sim10^6$, while the $10^{-12}$ splitting of a 40-site wire ($\varepsilon_0 = 1.4\times10^{-16}$ eV, a $\pi$ phase in 15 s) would permit $10^{10}$ — four orders of magnitude more. The population responsible is the same non-equilibrium quasiparticle population that Chapter 2 counted.

**6\. The scorecard has no winner, and that is the result**

  * Coherent-operation counts derived from materials parameters: superconducting $10^2$–$10^3$ (from $p\tan\delta$), ions $10^4$–$10^6$ (heating bound), atoms $\sim10^3$ (Rydberg lifetime), spins $10^2$–$10^6$ (bias-point dependent), photons $10^1$–$10^3$ (loss and visibility).
  * Neutral atoms lead or tie on four of the eight rows (temperature, uniformity, wiring, materials limit), yet no column leads the fault-tolerance rows: four of six carry the same surface-code overhead and the other two swap it for an unsolved alternative.
  * A single figure of merit would require weighting the axes, and the right weighting depends on whether you want an analog simulation or a fault-tolerant calculation.
  * The field has six approaches that have cleared the physics and are each waiting on a different materials problem.

**Practical implications**

  * Ask of any hardware claim: which axis of Chapter 1 does it move, and which materials limit does it address? If the answer to the second is "none", it is a demonstration rather than a step.
  * The same class of defect — two-level fluctuators in amorphous oxides — limits the two most industrially invested modalities. Understanding it microscopically would move both.
  * The neutral-atom row shows that removing the materials problem relocates the limit rather than eliminating it: materials problems are the price of design freedom.
  * The materials questions on the critical path are worth answering independently of whether a quantum computer is built, and quantum devices are in return the most sensitive known probes of those same defects.

### Where This Leads

You have reached the end of the series. You began with the question of what makes a good qubit and a common language for decoherence, then took three platforms apart in detail — a fabricated circuit limited by its own oxides, a natural atom limited by the surface that holds it, an atom held by light and limited by its own radiative lifetime — and in this chapter added three more that fail in structurally different ways. Along the way you diagonalized a cosine potential, computed a Mathieu stability diagram, evolved a blockaded atom pair, found an exchange splitting and watched a Majorana pair split exponentially, each time with code you can rerun and modify.

What remains is to use it. The next hardware announcement you read is one you can now place: which physical system, which axis, which materials limit, and whether the number quoted is a record or a scaling law. For the algorithms that this hardware must eventually run, and for the honest accounting of what they will cost, continue with [Introduction to Quantum Computing](<../quantum-computing-introduction/index.html>); for the superconductivity underneath Chapter 2, with [Introduction to Superconductivity](<../../MS/superconductivity-introduction/index.html>); and for the spin physics underneath §5.2, with [Introduction to Spintronics](<../../MS/spintronics-introduction/index.html>).

And if you work on materials: the list in §5.5 is not a list of reasons for pessimism. It is a list of open problems, most of which are not currently being worked on by the people who need them solved.

[← Chapter 4: Neutral Atoms](<chapter-4.html>) [Back to Series Index →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * Model parameters, loss tangents, heating rates, coupling strengths and coherent-operation estimates quoted here are illustrative order-of-magnitude values chosen for teaching; they are not device specifications and must be verified against primary sources before use in any assessment or proposal.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
