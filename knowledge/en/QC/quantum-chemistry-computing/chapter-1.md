---
title: "Chapter 1: Why Chemistry Is the Killer App"
chapter_title: "Chapter 1: Why Chemistry Is the Killer App"
subtitle: "The Electronic Structure Problem, the Exponential Wall, and the Accuracy Chemistry Actually Demands"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/-_ieRB65e_0"
    title="Quantum Chemistry Ch.1: Why Chemistry Is the Killer App"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-chemistry-computing/chapter-1.html>) | Last sync: 2026-08-17

[Quantum Computing Dojo](<../index.html>) > [Quantum Chemistry with Quantum Computers](<index.html>) > Chapter 1

Every list of quantum computing applications puts chemistry near the top, and for once the ordering is not marketing. Chemistry is the application where the match between problem and machine is structural rather than incidental: the thing we cannot afford to store classically is a quantum state, and a quantum computer stores quantum states for a living.

That argument is easy to state and easy to over-state. This chapter does the harder work of making it precise. We will define exactly what problem quantum chemistry is trying to solve, count exactly how fast it becomes impossible, define exactly how accurate an answer has to be before a chemist can use it — and then look honestly at how good the classical methods already are, because that is the bar any quantum approach must clear. By the end you will be able to say which chemical problems are candidates for a quantum computer and, just as importantly, which are not.

## 1.1 The Electronic Structure Problem

A molecule is a collection of nuclei and electrons interacting through the Coulomb force. Writing down its Schrödinger equation is not the difficulty; solving it is.

The first simplification is so universal that it is often left unstated. Nuclei are thousands of times heavier than electrons, so they move far more slowly. To an electron, the nuclei look effectively frozen. This licenses the **Born-Oppenheimer approximation**: fix the nuclei at chosen positions, solve for the electrons in the resulting electrostatic field, and treat the resulting energy as a function of the nuclear coordinates.

### 📚 What Born-Oppenheimer Buys You

Two things, and both are structural rather than numerical.

**A well-posed electronic problem.** With the nuclei fixed, the unknown is a wavefunction of the electrons alone. That is the **electronic structure problem**, and it is the problem this entire series is about.

**A potential energy surface.** Repeat the calculation at many nuclear geometries and you obtain the electronic energy as a *function of molecular shape*. Chemists live on that surface. Its minima are stable structures, the paths between minima are reaction mechanisms, and the passes along those paths are transition states. Almost every chemical question is a question about the shape of this surface.

So the target is the **ground-state energy** \\(E_0\\): the lowest eigenvalue of the electronic Hamiltonian at a given geometry.

Why the *energy*, when the wavefunction contains far more information? Because chemistry is driven by energy *differences*, and differences are what experiments measure.

  * **Bond strength** is the energy difference between a molecule and its separated fragments.
  * **Reaction thermodynamics** — whether a reaction is favourable at all — is the energy difference between products and reactants.
  * **Reaction rates** are governed by the barrier height: the energy difference between the transition state and the reactants.
  * **Spectra** are differences between the ground state and excited states.

Every one of these is a subtraction of two large numbers whose difference is small. That single fact is why quantum chemistry is obsessed with accuracy, and it is the subject of Section 1.3.

## 1.2 The Exponential Wall

Now the difficulty. Electrons are **fermions**: identical, indistinguishable, and required by the Pauli principle to have an antisymmetric wavefunction. They also repel each other, so their motion is **correlated** — where one electron is affects where the others can be. You cannot solve for them one at a time.

The standard way to build a many-electron wavefunction begins by choosing a finite set of one-electron functions. A **basis set** provides \\(K\\) **spatial orbitals**; each of these can hold an electron of either spin, giving \\(2K\\) **spin-orbitals** — the slots an electron may occupy. A single assignment of \\(n\\) electrons to \\(2K\\) slots is called a **Slater determinant** (the determinant structure is what enforces antisymmetry; Chapter 2 revisits this properly).

The exact wavefunction within that basis is a superposition of *every* such assignment. This is **full configuration interaction (FCI)**, and its dimension is a binomial coefficient:

\\[ \dim_{\text{FCI}} = \binom{2K}{n} \\]

There is nothing subtle here — it is the number of ways to choose which \\(n\\) of the \\(2K\\) slots are occupied. But the innocent-looking binomial coefficient is the whole problem. For a half-filled system, \\(\binom{2K}{K}\\) grows roughly like \\(4^K\\): each spatial orbital you add to the basis multiplies the size of the problem by about four.

### 📚 Feynman's Argument, Made Concrete

In the early 1980s Richard Feynman observed that classical computers are structurally bad at simulating quantum systems, because the amount of classical data needed to describe a quantum state grows exponentially with the number of particles. The binomial coefficient above *is* that argument, written for chemistry.

Notice what is and is not exponential. The **description** of the problem is small: a molecule is specified by a handful of nuclear positions and a basis set. The **solution** is enormous: a vector with \\(\binom{2K}{n}\\) complex entries. Classical methods must therefore compress that vector — approximate it, truncate it, or parametrize it — and every classical method in Section 1.4 is a particular strategy for doing so.

A quantum computer proposes a different bargain. A register of \\(2K\\) qubits holds a superposition over all \\(2^{2K}\\) occupation patterns natively, because it *is* a quantum system with that many configurations. The number of qubits grows **linearly** in the number of spin-orbitals while the space they span grows exponentially. That is the entire structural argument for quantum chemistry on quantum computers, and Section 1.6 makes it a table you can read.

## 1.3 What "Chemical Accuracy" Means

Suppose you could compute molecular energies to any accuracy you were willing to pay for. How accurate is accurate enough?

The community's answer is a target called **chemical accuracy**, and it is worth treating as a *definition*:

\\[ \text{chemical accuracy} \;\equiv\; 1\ \text{kcal}\,\text{mol}^{-1} \\]

Expressed in the atomic units that electronic structure calculations actually use, that is close to **1.6 millihartree**. The conversion is pure unit arithmetic, so let us not quote it — let us compute it, and at the same time compute the reason the target sits there.

The reason is kinetics. A reaction rate depends on the barrier height through an Arrhenius factor, \\(k \propto e^{-E_a / RT}\\). An *error* \\(\delta\\) in a computed barrier therefore does not shift the predicted rate by a little; it multiplies it by \\(e^{\delta / RT}\\). The code below evaluates that multiplier at room temperature.

```python
import math

# ---------------------------------------------------------------
# 1. "Chemical accuracy" as a DEFINITION, converted from SI.
#    Every constant below is an exact or CODATA-defined value,
#    not a value quoted from a chemistry paper.
# ---------------------------------------------------------------
HARTREE_J = 4.3597447222060e-18   # 1 hartree in joule (CODATA)
AVOGADRO = 6.02214076e23          # 1/mol, exact by SI definition
KCAL_J = 4184.0                   # 1 thermochemical kcal in joule, exact by definition
R_GAS = 8.314462618               # J/(mol K), exact by SI definition

kcal_per_mol_in_J = KCAL_J / AVOGADRO
chemical_accuracy_hartree = kcal_per_mol_in_J / HARTREE_J

print("Chemical accuracy, derived rather than quoted")
print(f"  1 kcal/mol            = {kcal_per_mol_in_J:.6e} J per molecule")
print(f"  1 kcal/mol            = {chemical_accuracy_hartree:.6e} hartree")
print(f"                        = {chemical_accuracy_hartree * 1e3:.4f} millihartree")
print(f"  1 millihartree        = {1e-3 / chemical_accuracy_hartree:.4f} kcal/mol")
print()

# ---------------------------------------------------------------
# 2. WHY that target: Arrhenius sensitivity of a rate constant.
#    k ~ exp(-Ea / RT), so an error dEa in the barrier multiplies
#    the predicted rate by exp(-dEa / RT).
# ---------------------------------------------------------------
T = 298.15   # K, a round room-temperature reference
RT = R_GAS * T
print(f"Rate-constant error factor at T = {T} K   (RT = {RT:.2f} J/mol)")
for kcal in [0.5, 1.0, 2.0, 5.0, 10.0]:
    dEa = kcal * KCAL_J                      # barrier error in J/mol
    factor = math.exp(dEa / RT)              # k_true / k_predicted
    mhartree = kcal * chemical_accuracy_hartree * 1e3
    print(f"  barrier error {kcal:5.1f} kcal/mol ({mhartree:7.2f} mhartree)"
          f" -> rate off by x {factor:12.2f}")
```

**Output:**

```
Chemical accuracy, derived rather than quoted
  1 kcal/mol            = 6.947695e-21 J per molecule
  1 kcal/mol            = 1.593601e-03 hartree
                        = 1.5936 millihartree
  1 millihartree        = 0.6275 kcal/mol

Rate-constant error factor at T = 298.15 K   (RT = 2478.96 J/mol)
  barrier error   0.5 kcal/mol (   0.80 mhartree) -> rate off by x         2.33
  barrier error   1.0 kcal/mol (   1.59 mhartree) -> rate off by x         5.41
  barrier error   2.0 kcal/mol (   3.19 mhartree) -> rate off by x        29.24
  barrier error   5.0 kcal/mol (   7.97 mhartree) -> rate off by x      4624.08
  barrier error  10.0 kcal/mol (  15.94 mhartree) -> rate off by x  21382125.12
```

**Reading the result.** An error of 1 kcal/mol in a barrier throws the predicted rate off by a factor of about five. That is tolerable: you would still identify the fast pathway and get the qualitative chemistry right. An error of 5 kcal/mol multiplies the rate by several thousand, and at 10 kcal/mol you are wrong by seven orders of magnitude — a reaction predicted to take a second is predicted to take months, or the reverse.

So chemical accuracy is not a round number chosen for elegance. It is roughly the largest error that still leaves a kinetic prediction usable, and the exponential in the Arrhenius expression is what makes the threshold so sharp.

> **The cruelty of the subtraction**
>
> Total electronic energies of molecules are large — many hartree — while the differences chemists care about are of order millihartree. Demanding 1.6 mhartree in a *difference* of two quantities that are individually thousands of times larger means demanding relative precision far beyond what the raw magnitude suggests. This is why methods that look "99% accurate" on a total energy can be useless for chemistry, and why error cancellation between similar calculations is one of the most important practical tools in the field.

## 1.4 The Classical Toolbox, Honestly

Quantum chemistry did not wait for quantum computers. It is a mature discipline with a layered toolbox, and any honest case for quantum computing has to start by respecting it.

### 📚 Hartree-Fock: the Mean Field

**Hartree-Fock (HF)** replaces the electron-electron repulsion with an average: each electron moves in the mean field of all the others. The wavefunction is a *single* Slater determinant, chosen self-consistently to minimize the energy.

HF is cheap and it is the foundation everything else is built on — it supplies the orbitals that later methods correlate. But by construction it misses **electron correlation**, the part of the energy that comes from electrons actively avoiding one another. That missing piece is small as a fraction of the total energy and large compared with chemical accuracy, which is exactly the wrong combination.

### 📚 Density Functional Theory: the Workhorse

**DFT** takes a different route. Rather than the wavefunction, it treats the electron *density* as the fundamental variable — a function of three coordinates instead of \\(3n\\). In principle the ground-state energy is an exact functional of that density. In practice nobody knows the exact functional, so the field uses **approximate exchange-correlation functionals**, of which there are many families, each with its own strengths and failure modes.

DFT is the workhorse of computational chemistry and materials science because its cost scales modestly with system size, letting it treat systems far beyond the reach of wavefunction methods. Its weakness is the flip side of its strength: because the functional is approximate and not systematically improvable, there is no internal knob you can turn to converge toward the exact answer. When DFT is wrong, it does not tell you.

### 📚 Coupled Cluster: the Gold Standard

**Coupled cluster (CC)** builds correlation on top of a Hartree-Fock reference using an exponential operator, \\(|\Psi\rangle = e^{\hat{T}}|\Phi_{\text{HF}}\rangle\\), where \\(\hat{T}\\) generates excitations of electrons out of occupied orbitals into empty ones. Truncating \\(\hat{T}\\) at single and double excitations, with a perturbative treatment of triples, gives the method commonly called the **gold standard** of quantum chemistry.

That reputation is earned — but it comes with a condition attached. Coupled cluster assumes the true wavefunction is *dominated by one determinant*, with everything else a correction. When that assumption holds, the method reaches chemical accuracy routinely. When it fails, coupled cluster can fail dramatically and without warning.

### 📚 Full CI: Exact, and Therefore Tiny

**FCI** is the exact answer within the chosen basis, obtained by diagonalizing the Hamiltonian in the full space of \\(\binom{2K}{n}\\) determinants. It is the reference against which every approximate method is calibrated, and Section 1.6 shows precisely why it is restricted to small systems.

### 📚 Where the Toolbox Struggles

The methods above fail in a correlated way — they tend to struggle on the *same* systems, and for the same reason.

The distinction that matters is between two kinds of correlation. **Dynamic correlation** is the fine-grained, short-range avoidance of electrons that a single-determinant reference describes poorly but that perturbative and coupled-cluster corrections capture well. **Static (or strong) correlation** is different: it occurs when several determinants have comparable weight in the true wavefunction, so no single reference is a good starting point.

Three families of problems are notorious for static correlation:

  * **Bond breaking.** As a bond is stretched toward dissociation, the picture of a doubly occupied bonding orbital stops being appropriate, and configurations that were negligible at equilibrium become equally important. Any method anchored to one determinant degrades along the way — which is a serious problem, since the whole point of a potential energy surface is to follow reactions from reactants to products.
  * **Transition metals.** Partially filled d shells offer many near-degenerate arrangements of electrons, and near-degeneracy is precisely the condition that makes several determinants matter simultaneously. Much of catalysis and much of magnetism lives here.
  * **Excited states.** Excited states are often intrinsically multi-configurational, and the machinery that works so well for a well-separated ground state has to be rebuilt.

There are dedicated multi-reference methods for these cases. They work, and they are also expensive, require expert judgement in choosing an active space, and do not scale to large systems.

**This is the gap.** Not chemistry in general — chemistry in general is served well by the classical toolbox. The gap is strong correlation.

## 1.5 Why Quantum Computers Fit — and What That Does Not Prove

The structural argument is the one from Section 1.2. A classical computer must *encode* a correlated many-electron state in a data structure whose size explodes; a quantum computer *holds* it, using one qubit per spin-orbital. Nothing needs to be compressed because nothing needs to be written down in the classical sense. Strong correlation, which is the hard case classically, carries no special penalty: a superposition with many important configurations is no harder for a quantum register to hold than one with a single dominant configuration.

That is a real and well-founded structural advantage. Now the counterweight, and it deserves equal weight.

### 📚 Three Honest Qualifications

**Classical methods are extremely good, and they keep improving.** The comparison is never against exact diagonalization of a large molecule — nobody does that. It is against the best classical method someone is actively trying to win with, and those methods advance every year: better functionals, better multi-reference approaches, tensor-network and quantum Monte Carlo methods that push exact-in-practice treatment to steadily larger active spaces. Every such advance moves the bar that a quantum computer has to clear.

**Small demonstrations demonstrate the method, not an advantage.** Molecules such as H₂ have been run on quantum hardware since the earliest days of the field, and doing so is a genuine achievement of experimental control. But every molecule small enough to run on today's devices is also small enough to solve faster and more accurately on a laptop. These experiments validate the method; they do not establish a computational advantage.

**The overlap region is still empty.** A convincing demonstration needs a problem that is simultaneously *classically hard* and *quantum-tractable on available hardware*. As of today, no such problem has been convincingly exhibited. Finding one is an active and legitimate research programme — and it is also where most over-claiming in this field happens.

Hold both halves at once. The structural argument is sound and is not going away. The demonstration has not happened yet. A reader who believes only the first half becomes a press release; a reader who believes only the second half misses why serious people work on this.

## 1.6 Hands-On: Counting the Exponential Wall

Let us stop describing the wall and count it. The code below needs only the standard library — the whole argument is combinatorics, and `math.comb` computes it exactly with no floating-point rounding to argue about.

For each system we report the FCI dimension \\(\binom{2K}{n}\\) and the memory needed to store one complex amplitude per determinant, at 16 bytes each for `complex128`. This is a *lower bound* on the cost, and a very generous one: it assumes the vector is the only thing you store and ignores the cost of actually diagonalizing anything.

```python
import math

# ---------------------------------------------------------------
# The exponential wall, counted exactly.
#
# K   = number of SPATIAL orbitals in the basis set
# 2K  = number of SPIN-ORBITALS (each spatial orbital holds up and down)
# n   = number of electrons
#
# The full configuration interaction (FCI) space is spanned by every
# way of placing n indistinguishable electrons into 2K slots:
#
#     dim = C(2K, n)
#
# Storing one complex amplitude per determinant in complex128
# costs 16 bytes each.
# ---------------------------------------------------------------
BYTES_PER_AMPLITUDE = 16     # complex128 = two float64

cases = [
    (2, 2, "4 spin-orbitals: the smallest textbook case"),
    (4, 4, "8 spin-orbitals"),
    (8, 8, "16 spin-orbitals"),
    (16, 16, "32 spin-orbitals"),
    (26, 26, "52 spin-orbitals"),
    (40, 40, "80 spin-orbitals"),
    (50, 50, "100 spin-orbitals"),
]

def human_bytes(nbytes):
    """Format a byte count with binary prefixes, no external libraries."""
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]
    value = float(nbytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.3g} {unit}"
        value /= 1024.0

print(f"{'K':>3} {'2K':>4} {'n':>3} {'FCI determinants':>22} {'state-vector memory':>22}   note")
print("-" * 100)
for K, n, note in cases:
    dim = math.comb(2 * K, n)
    nbytes = dim * BYTES_PER_AMPLITUDE
    print(f"{K:3d} {2*K:4d} {n:3d} {dim:22d} {human_bytes(nbytes):>22}   {note}")
print()

# ---------------------------------------------------------------
# How fast does the wall arrive? Compare against a generous
# classical memory budget.
# ---------------------------------------------------------------
BUDGET_BYTES = 1024**5      # 1 PiB: far more than any single machine
print(f"Largest half-filled case that fits in {human_bytes(BUDGET_BYTES)} of amplitudes:")
K = 1
while math.comb(2 * (K + 1), K + 1) * BYTES_PER_AMPLITUDE <= BUDGET_BYTES:
    K += 1
print(f"  K = {K} spatial orbitals ({2*K} spin-orbitals, {K} electrons)")
print(f"  dim = {math.comb(2*K, K)}  -> {human_bytes(math.comb(2*K, K) * BYTES_PER_AMPLITUDE)}")
print(f"  add ONE more spatial orbital and one more electron:")
print(f"  dim = {math.comb(2*(K+1), K+1)}"
      f"  -> {human_bytes(math.comb(2*(K+1), K+1) * BYTES_PER_AMPLITUDE)}")
print()

# ---------------------------------------------------------------
# The qubit side of the ledger: a quantum register needs one qubit
# per spin-orbital, and the growth is LINEAR.
# ---------------------------------------------------------------
print("Classical amplitudes vs. qubits, for the same half-filled systems")
print(f"{'2K spin-orbitals':>18} {'FCI determinants':>22} {'qubits needed':>15}")
for K in [2, 4, 8, 16, 26, 40, 50]:
    print(f"{2*K:18d} {math.comb(2*K, K):22d} {2*K:15d}")
```

**Output:**

```
  K   2K   n       FCI determinants    state-vector memory   note
----------------------------------------------------------------------------------------------------
  2    4   2                      6                   96 B   4 spin-orbitals: the smallest textbook case
  4    8   4                     70               1.09 KiB   8 spin-orbitals
  8   16   8                  12870                201 KiB   16 spin-orbitals
 16   32  16              601080390               8.96 GiB   32 spin-orbitals
 26   52  26        495918532948104               7.05 PiB   52 spin-orbitals
 40   80  40 107507208733336176461620               1.42 YiB   80 spin-orbitals
 50  100  50 100891344545564193334812497256           1.34e+06 YiB   100 spin-orbitals

Largest half-filled case that fits in 1 PiB of amplitudes:
  K = 24 spatial orbitals (48 spin-orbitals, 24 electrons)
  dim = 32247603683100  -> 469 TiB
  add ONE more spatial orbital and one more electron:
  dim = 126410606437752  -> 1.8 PiB

Classical amplitudes vs. qubits, for the same half-filled systems
  2K spin-orbitals       FCI determinants   qubits needed
                 4                      6               4
                 8                     70               8
                16                  12870              16
                32              601080390              32
                52        495918532948104              52
                80 107507208733336176461620              80
               100 100891344545564193334812497256             100
```

**Reading the result.** Four observations, in increasing order of importance.

  * **The smallest case is trivially small.** Four spin-orbitals with two electrons gives six determinants — 96 bytes. This is the case Chapter 2 works out by hand and Chapter 4 solves numerically, and its smallness is exactly what makes it a good teaching example.
  * **The wall arrives suddenly.** At 32 spin-orbitals the state vector is under nine gibibytes: uncomfortable but possible. At 52 it is petabytes. Nothing dramatic happened in between; the exponential simply did what exponentials do.
  * **One more orbital costs a factor of four.** The budget experiment is the sharpest way to see it. With a generous 1 PiB of memory, the largest half-filled system you can store has 24 spatial orbitals. Adding a *single* spatial orbital and a single electron nearly quadruples the requirement and puts you over budget. You do not run out of headroom gradually — you run out on the next step, every time.
  * **The qubit column barely moves.** The last table is the whole argument of this series on one screen. To go from six determinants to \\(10^{29}\\) of them, the classical column crosses twenty-eight orders of magnitude while the qubit column goes from 4 to 100. Exponential on one side, linear on the other.

One caution, so the table is not read as more than it is. Storing a state is not the same as *finding the ground state* within it, and no serious classical method has ever stored a full FCI vector at these sizes — the entire art of classical quantum chemistry is avoiding that. The table shows why the avoidance is necessary, not that classical chemistry is helpless.

Try adding a row with an unbalanced case — say \\(K = 30\\) spatial orbitals with only \\(n = 4\\) electrons — and you will see the dimension collapse. Few electrons in many orbitals is a much smaller problem than a half-filled shell, which is why the number of *correlated* electrons, not the number of atoms, sets the difficulty.

### 🎯 Exercise Problems

  1. **The factor of four.** Show that for a half-filled system, \\(\binom{2K+2}{K+1} / \binom{2K}{K}\\) tends toward 4 as \\(K\\) grows. Evaluate the ratio at \\(K = 5\\) and \\(K = 25\\) and comment on how quickly the limit is approached.
  2. **Electrons versus orbitals.** Using `math.comb`, compare \\(\binom{60}{4}\\), \\(\binom{60}{10}\\), and \\(\binom{60}{30}\\). Which quantity governs the difficulty of a calculation — the size of the basis or the number of correlated electrons? State the practical consequence for choosing an active space.
  3. **Reading the accuracy target.** Using the Arrhenius code, find the barrier error (in kcal/mol) that changes a predicted rate constant by exactly one order of magnitude at 298.15 K. Repeat at 500 K and explain the direction of the change.
  4. **Diagnosing correlation.** For each of the following, say whether you would expect static correlation to be important and why: (a) methane near its equilibrium geometry, (b) N₂ stretched to twice its equilibrium bond length, (c) a first-row transition-metal complex with a partially filled d shell, (d) the first excited state of a conjugated dye.
  5. **Auditing a claim.** You read that a quantum device computed the energy of a molecule "to chemical accuracy". List four questions you must answer before that sentence means anything, and say what a satisfactory answer to each would look like.

## Summary

This chapter set out the problem the rest of the series attacks. Under the **Born-Oppenheimer approximation** the nuclei are treated as fixed, leaving the **electronic structure problem**: find the ground-state energy of the electrons in their field, and repeat over geometries to trace a **potential energy surface** whose minima are structures and whose passes are transition states. We want energies because chemistry is made of energy *differences* — bond strengths, reaction thermodynamics, barriers, spectra — each a small difference between large numbers. The **exponential wall** follows from counting: with \\(2K\\) spin-orbitals and \\(n\\) electrons, the exact (FCI) space has \\(\binom{2K}{n}\\) dimensions, roughly \\(4^K\\) at half filling. That is Feynman's argument in chemical form. We defined **chemical accuracy** as 1 kcal/mol, derived it as 1.5936 millihartree from SI constants, and justified it with the Arrhenius sensitivity our code computed: a 1 kcal/mol barrier error changes a room-temperature rate by a factor of about 5, and a 10 kcal/mol error by seven orders of magnitude. We surveyed the classical toolbox honestly — **Hartree-Fock** as mean field, **DFT** as the broadly applicable workhorse with non-systematic errors, **coupled cluster** as the gold standard *for single-reference systems*, and **FCI** as exact but tiny — and located the real gap at **static correlation**: bond breaking, transition metals, excited states. Finally, our combinatorial table showed the structural case for quantum hardware: to span \\(10^{29}\\) determinants, the classical memory column crosses twenty-eight orders of magnitude while the qubit column grows from 4 to 100. That argument is sound, and it is not yet a demonstration — no problem has been convincingly shown to be both classically hard and quantum-tractable today.

In the next chapter we build the bridge between the chemistry and the hardware. We introduce second quantization, write the electronic Hamiltonian in terms of creation and annihilation operators, and then confront the mismatch at the centre of the whole enterprise: qubits are distinguishable and electrons are not. The Jordan-Wigner transformation is the repair, and we will verify numerically that it works.

[← Series Top](<index.html>) [Chapter 2: From Molecules to Qubits →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
