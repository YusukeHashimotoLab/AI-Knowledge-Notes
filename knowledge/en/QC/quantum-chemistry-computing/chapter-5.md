---
title: "Chapter 5: Beyond H2: The Honest Frontier"
chapter_title: "Chapter 5: Beyond H2: The Honest Frontier"
subtitle: "What Grows When Molecules Grow, the Measurement Wall, and Where Advantage Might Actually Appear"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/uW35ygoT2jQ"
    title="Quantum Chemistry Ch.5: Beyond H2: The Honest Frontier"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-chemistry-computing/chapter-5.html>) | Last sync: 2026-08-17

[Quantum Computing Dojo](<../index.html>) > [Quantum Chemistry with Quantum Computers](<index.html>) > Chapter 5

## 5.1 Why H₂ Was Easy

In Chapter 4 you built the hydrogen molecule end to end: second-quantized integrals, Jordan–Wigner mapping, a qubit Hamiltonian, an ansatz, a classical optimizer, a potential energy curve. Nothing was hidden. That is an achievement worth pausing on — most people who talk about quantum chemistry on quantum computers have never assembled one.

It is also worth being precise about what you assembled. H₂ in a minimal basis has **two spatial orbitals**, therefore **four spin-orbitals**, therefore **four qubits** under Jordan–Wigner. Its qubit Hamiltonian is a sum of **fifteen Pauli strings**, identity included. Every one of those numbers is small enough to print on one screen, and the entire 16-dimensional Hilbert space fits in a NumPy array you could inspect element by element.

> **Where "fifteen" comes from**
>
> That count is not quoted from anywhere. It follows from the structure of the problem: take a generic real, spin-conserving, symmetry-allowed one- and two-body Hamiltonian on two spatial orbitals, Jordan–Wigner it to four qubits, and decompose the resulting \\(16 \times 16\\) matrix in the 256-element four-qubit Pauli basis. Fifteen strings carry a nonzero coefficient — fourteen plus the identity — regardless of the numerical values of the integrals. The count is a property of the symmetry, not of hydrogen's particular numbers.

So the honest reading of Chapter 4 is the one the [Introduction to Quantum Computing](<../quantum-computing-introduction/index.html>) series already gave for hardware demonstrations of H₂: **it validates the method, not an advantage**. Your laptop solved that Hamiltonian by direct diagonalization in microseconds. A quantum device running the same problem is not competing with your laptop; it is demonstrating that the pipeline works.

This chapter is about what happens to each piece of that pipeline when the molecule stops being hydrogen — and about the honest criterion by which anyone, including you, should judge a claim of quantum advantage in chemistry.

## 5.2 Four Things Grow, and They Grow Differently

When the molecule grows, four quantities grow with it, and they do not grow at the same rate. Keeping them separate in your head is most of what it takes to read this literature critically.

### 📚 The Four Budgets

  * **Qubits** grow linearly with the number of spin-orbitals. Under Jordan–Wigner, one qubit per spin-orbital, and the number of spin-orbitals is twice the number of spatial orbitals in the basis set. Double the basis, double the qubits. This is the *cheapest* growth of the four, which is exactly why qubit count is the least informative number to quote.
  * **Hamiltonian terms** grow like the **fourth power** of the basis size, because the two-electron integrals \\((pq|rs)\\) carry four orbital indices. Permutational symmetry divides the count by a constant; it does not change the exponent.
  * **Circuit depth** grows with the ansatz. A chemically motivated ansatz that includes an excitation operator per orbital pairing inherits polynomial growth in the number of gates, and each of those gates must fit inside the depth budget of order \\(1/\epsilon\\) that the intro series derived. Depth is where the NISQ constraint bites.
  * **Shots** grow like the inverse square of the target accuracy, and are multiplied by the size of the Hamiltonian's coefficients. This is the budget nobody puts in the headline, and Section 5.3 argues it is the one that hurts most.

### 📚 Counting the Hamiltonian

The second budget is pure combinatorics, so we can compute it exactly rather than assert it. Over \\(n\\) real spatial orbitals the two-electron integrals have eight-fold permutational symmetry, so the number of *distinct* integrals is \\(M(M+1)/2\\) with \\(M = n(n+1)/2\\).

```python
import numpy as np

# --- How the Hamiltonian grows with the size of the orbital basis ---
# n     : number of real spatial orbitals
# 2n    : spin-orbitals, and therefore qubits under Jordan-Wigner
# The two-electron integrals (pq|rs) run over four orbital indices, so the
# raw index count is n^4. Real orbitals give 8-fold permutational symmetry:
#   (pq|rs) = (qp|rs) = (pq|sr) = (qp|sr) = (rs|pq) = ... ,
# so the number of DISTINCT integrals is M(M+1)/2 with M = n(n+1)/2.
# Both counts are pure combinatorics - no chemistry, no benchmark numbers.

def unique_two_electron_integrals(n):
    """Distinct (pq|rs) over n real spatial orbitals under 8-fold symmetry."""
    m = n * (n + 1) // 2          # distinct index pairs (pq) with p >= q
    return m * (m + 1) // 2       # distinct pairs of pairs


def unique_one_electron_integrals(n):
    """Distinct h_pq over n real spatial orbitals (h is symmetric)."""
    return n * (n + 1) // 2


sizes = np.array([2, 4, 8, 16, 32, 64, 100])

print(f"{'spatial n':>9} {'qubits 2n':>9} {'h_pq':>8} {'(pq|rs) unique':>15} {'n^4':>12} {'unique/n^4':>11}")
for n in sizes:
    n = int(n)
    u2 = unique_two_electron_integrals(n)
    print(f"{n:>9d} {2*n:>9d} {unique_one_electron_integrals(n):>8d} "
          f"{u2:>15d} {n**4:>12d} {u2 / n**4:>11.3f}")
print()

# --- The growth is fourth-power: doubling the basis multiplies the
# --- integral count by roughly 2^4 = 16.
print("Doubling the basis multiplies the distinct-integral count by:")
for n in [4, 8, 16, 32]:
    ratio = unique_two_electron_integrals(2 * n) / unique_two_electron_integrals(n)
    print(f"  n = {n:>3d} -> {2*n:>3d}:  x {ratio:.2f}")
print()

# --- Same arithmetic read as a bill: how many distinct numbers does the
# --- Hamiltonian consist of? The qubit Hamiltonian's Pauli-term count is a
# --- CONSTANT MULTIPLE of this (each integral spawns several Pauli strings),
# --- so it inherits the same fourth-power growth.
print("Distinct integrals the Hamiltonian is built from:")
for n in [2, 10, 50]:
    n = int(n)
    total = unique_one_electron_integrals(n) + unique_two_electron_integrals(n)
    print(f"  n = {n:>2d} spatial orbitals ({2*n:>3d} qubits): {total:>10d} distinct integrals")
```

**Output:**

```
spatial n qubits 2n     h_pq  (pq|rs) unique          n^4  unique/n^4
        2         4        3               6           16       0.375
        4         8       10              55          256       0.215
        8        16       36             666         4096       0.163
       16        32      136            9316        65536       0.142
       32        64      528          139656      1048576       0.133
       64       128     2080         2164240     16777216       0.129
      100       200     5050        12753775    100000000       0.128

Doubling the basis multiplies the distinct-integral count by:
  n =   4 ->   8:  x 12.11
  n =   8 ->  16:  x 13.99
  n =  16 ->  32:  x 14.99
  n =  32 ->  64:  x 15.50

Distinct integrals the Hamiltonian is built from:
  n =  2 spatial orbitals (  4 qubits):          9 distinct integrals
  n = 10 spatial orbitals ( 20 qubits):       1595 distinct integrals
  n = 50 spatial orbitals (100 qubits):     814725 distinct integrals
```

**Reading the result.** Three observations.

  * **The ratio column settles.** `unique/n^4` drifts toward roughly \\(1/8\\), which is the eight-fold symmetry doing its work. Symmetry buys a constant factor and nothing more — the exponent is untouched.
  * **The doubling column converges to sixteen.** \\(2^4 = 16\\), and the small-\\(n\\) values are below it only because the \\(+1\\) terms in \\(M(M+1)/2\\) still matter when \\(n\\) is tiny. This is the fourth-power law becoming visible.
  * **The last block is the one to remember.** Going from four qubits to two hundred multiplies the count of distinct integrals by roughly a hundred thousand. At two spatial orbitals the nine distinct integrals — three one-electron \\(h_{pq}\\) plus six two-electron \\((pq|rs)\\) — produced the fifteen-term Pauli sum of Chapter 4; each integral spawns a few Pauli strings, so the Pauli count is a constant multiple of this column and grows the same way.

A hundred qubits is a modest-sounding number. The Hamiltonian sitting on those hundred qubits is not modest at all, and that mismatch is the first thing a qubit count fails to tell you.

## 5.3 The Measurement Problem: The Quiet Killer

Here is the part of VQE that rarely appears in an announcement. A quantum computer does not hand you \\(\langle \hat{H} \rangle\\). It hands you samples. Every energy evaluation in the variational loop of Chapter 3 is a statistical estimate built from a finite number of circuit repetitions — **shots** — and statistics has an unforgiving exchange rate.

A Pauli string has eigenvalues \\(+1\\) and \\(-1\\) only, so a single measurement of it returns \\(\pm 1\\) with mean \\(\langle P \rangle\\) and variance \\(1 - \langle P \rangle^2 \le 1\\). Averaging \\(N\\) shots gives a standard error of at most \\(1/\sqrt{N}\\), so reaching an error \\(\epsilon\\) needs

\\[ N \gtrsim \frac{1}{\epsilon^2} \\]

That inverse-square law is a property of sampling, not of any device, and no hardware improvement repeals it. Now recall the accuracy chemistry asks for. **Chemical accuracy** is conventionally 1 kcal/mol — the scale at which computed reaction energetics start to be predictive rather than suggestive. Converting units puts that just under \\(1.6 \times 10^{-3}\\) hartree.

```python
import numpy as np

# --- Shot-noise arithmetic for a variational energy estimate ---
#
# A Pauli string P has eigenvalues +1 and -1 only. Measuring it returns a
# random +-1 whose mean is <P>. For such a variable,
#     Var(P) = <P^2> - <P>^2 = 1 - <P>^2  <=  1 ,
# so the standard error of the mean over N shots obeys
#     SE = sqrt(Var / N) <= 1 / sqrt(N)   ->   N >= 1 / eps^2 .
# That inverse-square law is the whole story: it is a property of sampling,
# not of any particular device.
#
# Chemical accuracy is conventionally quoted as 1 kcal/mol. The number below
# is a UNIT CONVERSION, not a benchmark result.

HARTREE_PER_KCAL_PER_MOL = 1.0 / 627.5094740631  # standard unit conversion

chem_acc = HARTREE_PER_KCAL_PER_MOL
print(f"1 kcal/mol = {chem_acc:.4e} hartree   (unit conversion)")
print()


def shots_for_single_pauli(eps, variance=1.0):
    """Shots needed so the standard error of one Pauli expectation is <= eps.
    variance = 1.0 is the worst case for a +-1 observable."""
    return int(np.ceil(variance / eps ** 2))


targets = [(1e-1, ""), (1e-2, ""), (chem_acc, "  <- 1 kcal/mol"),
           (1e-3, ""), (1e-4, ""), (1e-5, "")]

print("One Pauli expectation, worst-case variance = 1")
print(f"{'target error eps [Ha]':>22} {'shots >= 1/eps^2':>18}")
for eps, note in targets:
    print(f"{eps:>22.4e} {shots_for_single_pauli(eps):>18d}{note}")
print()

# --- The eps^-2 law made explicit: each factor of 10 in accuracy costs 100x
print("Tightening the target by 10x multiplies the shot count by:")
for eps in [1e-1, 1e-2, 1e-3]:
    ratio = shots_for_single_pauli(eps / 10) / shots_for_single_pauli(eps)
    print(f"  eps {eps:.0e} -> {eps/10:.0e}:  x {ratio:.1f}")
print()

# --- A whole Hamiltonian, not one term ---
# H = sum_i c_i P_i. Splitting the shot budget optimally across terms and
# adding the variances gives, with Var(P_i) <= 1,
#     N_total <= (sum_i |c_i|)^2 / eps^2 = lambda^2 / eps^2 ,
# where lambda is the 1-norm of the coefficients. This is a MODEL: it assumes
# independent measurement of every term and no grouping of commuting terms,
# both of which real implementations work hard to improve on.
print("Whole-Hamiltonian model: N_total <= (coefficient 1-norm)^2 / eps^2")
print(f"{'1-norm lambda [Ha]':>19} {'shots at 1 kcal/mol':>22}")
for lam in [1.0, 10.0, 100.0, 1000.0]:
    n_total = lam ** 2 / chem_acc ** 2
    print(f"{lam:>19.1f} {n_total:>22.3e}")
```

**Output:**

```
1 kcal/mol = 1.5936e-03 hartree   (unit conversion)

One Pauli expectation, worst-case variance = 1
 target error eps [Ha]   shots >= 1/eps^2
            1.0000e-01                100
            1.0000e-02              10000
            1.5936e-03             393769  <- 1 kcal/mol
            1.0000e-03            1000000
            1.0000e-04          100000000
            1.0000e-05        10000000000

Tightening the target by 10x multiplies the shot count by:
  eps 1e-01 -> 1e-02:  x 100.0
  eps 1e-02 -> 1e-03:  x 100.0
  eps 1e-03 -> 1e-04:  x 100.0

Whole-Hamiltonian model: N_total <= (coefficient 1-norm)^2 / eps^2
 1-norm lambda [Ha]    shots at 1 kcal/mol
                1.0              3.938e+05
               10.0              3.938e+07
              100.0              3.938e+09
             1000.0              3.938e+11
```

> **What is and is not a real number here.** The \\(\epsilon^{-2}\\) law and the kcal/mol conversion are exact. The 1-norm values \\(\lambda\\) are **chosen by us to make the arithmetic legible** — no device is being described, and no molecule's actual 1-norm is being claimed. Read the last table as "how the cost scales with the size of the Hamiltonian", never as a runtime estimate.

**Reading the result.** The shape of the problem, in three points.

  * **Chemical accuracy alone costs hundreds of thousands of shots for a single Pauli term.** And a chemically interesting Hamiltonian has far more than one term.
  * **Accuracy is the expensive axis.** Every factor of ten in precision multiplies the shot count by one hundred, exactly. Chemistry sits at the far end of that curve because energy *differences* of chemical interest are tiny compared with total electronic energies.
  * **The 1-norm multiplies everything, and it grows with the molecule.** Bigger systems have larger total energies and more terms, so \\(\lambda\\) grows while the target \\(\epsilon\\) stays pinned at chemical accuracy. The two move in opposite directions — a scissor, not a slope.

This is why measurement is the quiet killer. Depth and qubit count are visible constraints that everyone discusses; the shot budget is invisible in a headline and can dominate the total runtime. Serious effort goes into softening it — grouping commuting Pauli terms so they can be measured together, classical shadows and other randomized estimators, smarter shot allocation across terms during the optimization — and these help by constant and sometimes better-than-constant factors. None of them repeals \\(\epsilon^{-2}\\).

## 5.4 Barren Plateaus: When the Optimizer Goes Blind

The measurement problem assumes the optimizer at least knows which way to go. Sometimes it does not.

For parametrized circuits that are deep and expressive enough to behave like random unitaries, the gradient of the energy with respect to any single parameter has **zero mean and a variance that shrinks exponentially with the number of qubits**. The landscape is not rugged in the way a classical optimizer is used to; it is *flat*, almost everywhere, with the minimum hidden in a narrow region you will not stumble into.

The consequence is brutal when combined with Section 5.3. To follow a gradient you must resolve it above the shot noise. If the gradient shrinks exponentially in the qubit count and your resolution improves only as \\(1/\sqrt{N}\\), the shots required to see the gradient at all grow exponentially. The optimizer does not fail loudly — it reports an energy that stops improving, which looks exactly like convergence.

### 📚 Why This Threatens the "Hardware-Efficient" Instinct

There is a natural temptation when circuits are shallow and noisy: build the ansatz out of whatever gates the device happens to do well, layered generically, with no reference to the chemistry. It respects the connectivity, it keeps the depth low, and it is easy to write. This is the **hardware-efficient** instinct, and barren plateaus are precisely the argument against following it blindly. An ansatz designed to be *expressive without structure* is an ansatz designed to look random — which is the condition under which the gradients vanish. Worse, expressiveness and trainability pull against each other: making the family large enough to contain the true ground state tends to make it flat enough that you cannot find it.

Mitigations exist and are active research. **Problem-inspired ansätze** — circuits whose structure comes from the excitation operators of the chemistry rather than from the chip layout — restrict the state family in a way that both encodes physics and avoids randomness. **Initialization strategies** aim to start the optimization inside a region where gradients are still visible, rather than at a random point in parameter space. **Layer-by-layer or adaptive construction** grows the circuit only as far as the problem demands. We name these rather than develop them; the point for this chapter is that the fix is *structure*, and structure comes from the chemistry.

## 5.5 Making the Problem Smaller

If everything grows with the number of orbitals, the most effective lever is to use fewer orbitals — honestly.

### 📚 Active Spaces

Not every electron in a molecule participates in the interesting physics. Core electrons sit far below the frontier orbitals and are essentially inert for bond-breaking, spin states, and reactivity; very high virtual orbitals are essentially unoccupied. An **active space** freezes those and treats correlation exactly only among a chosen set of orbitals and electrons near the frontier, folding the frozen ones into an effective one-body potential.

Look at the fourth-power table in Section 5.2 to see why this matters so much. Cutting the active space from thirty-two spatial orbitals to sixteen does not halve the Hamiltonian — it divides the distinct-integral count by roughly fifteen, and halves the qubit count as well. Active spaces are the highest-leverage reduction available.

They are also the step where a calculation most easily becomes dishonest. Choosing the active space is a *modelling decision*, and results depend on it. A quantum computation on a small active space, compared against a classical computation on a much larger one, is not a comparison at all. If you take one methodological habit from this chapter, take this: **when you read a quantum chemistry result, find the active space before you read the number.**

### 📚 Symmetry Reductions

Chapter 4 already used this lever. The electronic Hamiltonian conserves particle number and spin projection, and a molecule with symmetry conserves more. Each conserved quantity restricts the state to a sector of Hilbert space, and a sector can be encoded in fewer qubits than the full space — the value of the conserved quantity is fixed, so the qubit that stores it can be removed and replaced by a constant. This is why the H₂ Hamiltonian could be reduced below four qubits without approximation.

Symmetry reduction is exact and therefore free of the modelling risk that active spaces carry. Its limitation is that it buys a fixed, modest number of qubits, not a change of exponent.

### 📚 Embedding

A third family treats a small, strongly correlated fragment at high accuracy while the surrounding environment is treated with a cheaper method, coupled through an effective potential. **Density matrix embedding, dynamical mean-field, and QM/MM-style partitions** are the names to look for. The relevance here is architectural: embedding is exactly the shape of problem that suits a small, expensive, high-accuracy solver — which is what a quantum device would be. If quantum chemistry on quantum computers ever becomes routine, embedding is a plausible way it plugs into existing workflows: not replacing the classical stack, but sitting inside it as the fragment solver.

## 5.6 The Classical Competition Is Not Standing Still

Every argument for quantum advantage is implicitly a claim about classical methods. That makes classical progress the most under-discussed risk to the quantum chemistry thesis.

**Density functional theory** remains the workhorse of computational materials science, and functional development is an active field rather than a settled one. **Coupled-cluster** methods keep being extended, with local and reduced-scaling formulations pushing accurate correlation treatment to larger systems than the textbook scaling suggests. **Tensor-network methods** — DMRG and its relatives — attack the strongly correlated regime that is supposed to be the quantum computer's home turf, and have repeatedly handled active spaces once assumed intractable. **Quantum Monte Carlo** and modern selected-configuration-interaction approaches come at the same territory from other directions, and **machine-learned interatomic potentials** amortize expensive calculations across enormous numbers of later evaluations.

Notice the pattern: several of these aim squarely at the "classically hard" region that advantage arguments rely on staying hard. The target is moving, and it is moving toward the quantum device.

### 📚 The Honest Criterion

So what would actually count? The criterion is stringent and worth stating plainly.

> **Quantum advantage in chemistry requires a system where the BEST available classical method fails, AND a quantum device succeeds, AND the accuracy of the quantum result can be verified.**

All three clauses do real work.

  * **"Best available classical method"** — not the most convenient one, and not a method chosen because it does badly. The comparison must be run by someone trying to win with the classical tool.
  * **"A quantum device succeeds"** — on the same system, the same active space, the same basis, the same property. Changing any of these turns a comparison into an analogy.
  * **"Verified accuracy"** — the hardest clause. If the problem is classically intractable, you cannot check the answer classically. Verification has to come from somewhere else: variational bounds, internal consistency across methods, agreement with experiment on related quantities, or error bars that are themselves trustworthy. An unverifiable answer to an intractable problem is not a result.

The intersection of those three conditions is what the intro series called the overlap region, and its status has not changed: **it is still empty.** No convincing demonstration yet exists of a chemistry problem that is simultaneously beyond the best classical methods and within reach of a quantum device at verified accuracy.

That is a statement about today, not a prediction about the decade. It is also not a reason to disengage — it is a reason to know exactly what you are looking at when a claim arrives.

## 5.7 Where It Could Matter for Materials Science

If you came to this series from the materials informatics or materials science side of AI Terakoya, this is the section written for you. What follows are **hypotheses about where the overlap might first appear** — reasoned guesses, not promises, and each one is a research programme rather than a plan.

**Strongly correlated materials.** Transition-metal oxides, competing magnetic orders, systems near a metal-insulator transition — these are where the single-reference picture that DFT and standard coupled-cluster lean on is least reliable, and therefore where a method that represents entanglement natively should have most to offer. The catch is that the interesting physics is often extended and low-energy, while the algorithms in this series are best posed for a finite, well-defined fragment. Embedding is the bridge that would have to hold.

**Catalysis.** Transition-metal active sites, spin-state energetics, and bond-breaking transition states combine multi-reference character with a demand for accuracy at exactly the kcal/mol scale Section 5.3 priced. A catalytic cycle is also naturally *local* — a small active region inside a larger, duller environment — which fits the fragment-solver architecture. Catalysis appears on nearly every candidate list for real reasons; that does not make it a solved case.

**Batteries and energy storage.** Redox potentials, electrolyte decomposition, and interfacial chemistry involve open-shell intermediates and charge-transfer states that classical methods handle unevenly, and the property wanted is again an energy difference at chemical accuracy.

**And the connection that runs the other way.** Materials informatics is bounded by the quality of its training data, and a model trained on DFT energies inherits DFT's errors — including in exactly the strongly correlated cases above. If quantum simulation ever supplies reliable energies where classical methods are unreliable, the effect is not one better calculation but a better *training set*, propagating through every model built on it. That leverage is why this series belongs on a materials informatics site at all.

The MI discipline also transfers directly in the other direction. Hold out a test set; compare against a strong baseline; report the honest error. A quantum chemistry result without a classical baseline is the same category of claim as a machine-learning model reported without one, and deserves the same reception.

## 5.8 The Fault-Tolerant Outlook

Everything above concerns the variational approach — shallow circuits, many repetitions, a classical optimizer, and no error correction. There is a second algorithm in the room, and it has a different shape entirely.

**Quantum phase estimation** is the "eventual" algorithm for electronic structure. Rather than variationally minimizing a sampled expectation value, it extracts an eigenvalue of the Hamiltonian directly from the phase accumulated by a controlled time-evolution, writing the answer into an ancilla register bit by bit. The contrast with VQE is systematic:

| | VQE (this series) | Quantum phase estimation |
|---|---|---|
| Circuit depth | Shallow by design | Deep — long controlled time-evolution |
| Hardware requirement | Runs on noisy devices | Requires error correction to be meaningful |
| Accuracy source | Quality of the ansatz | Number of ancilla bits, in principle systematic |
| Optimization | Classical loop, with barren-plateau risk | No variational loop to get stuck in |
| Shot scaling | The \\(\epsilon^{-2}\\) wall of Section 5.3 | Better precision scaling per run, at the cost of depth |
| Input still needed | Initial parameters | A trial state with decent overlap with the ground state |

The trade is depth for statistics. Phase estimation escapes the measurement wall that dominates VQE, and pays for it in circuit depth that today's devices cannot supply — which means it pays in **error correction**, and therefore in logical qubits, and therefore in the entire engineering programme laid out in Chapter 5 of the [Quantum Computing Hardware](<../quantum-computing-hardware/index.html>) series. If you want to know how far away the fault-tolerant era is, that chapter is where the question actually lives: overhead, thresholds, control wiring, and modularity, not chemistry.

One caveat worth keeping: phase estimation is not magic either. It needs an initial state with meaningful overlap with the true ground state, and preparing such a state for a strongly correlated system is itself an unsolved problem — arguably the same problem the ansatz was trying to solve. The difficulty is relocated, not deleted.

## 5.9 What You Can Do Now

Step back and take stock of what this series actually gave you.

You can take a molecular Hamiltonian in second-quantized form, map it to qubits, build a variational circuit, run the hybrid loop, and produce a potential energy curve — with every step visible in NumPy and no black-box library standing between you and the mathematics. You know why each step is constructed the way it is, which means you can read the corresponding step in a paper or a framework and recognize what it is doing.

You can also do something rarer, and more useful. You can look at a claim about quantum chemistry on quantum computers and ask the questions that decide whether it means anything: How many qubits, and physical or logical? What active space, and what basis? How many shots, and to what target accuracy? What was the classical baseline, and was it run by someone trying to win? Was the answer verified, and how?

That is the mindset this series has been arguing for throughout, and it is the same one the sibling series end on: **calibration over allegiance.** The field does not need more enthusiasm and it does not need more dismissal. It needs people who can tell the difference between a demonstration and an advantage — and who find the actual state of affairs interesting enough to keep watching.

The overlap region is empty today. Someone will eventually put something in it. You now have what you need to check.

### 🎯 Exercise Problems

  1. **Fourth-power arithmetic** : using the code in Section 5.2, compute how many distinct two-electron integrals an active space of twenty spatial orbitals requires, and compare it with ten. State the qubit count in both cases and explain why the qubit count is the misleading number to quote.
  2. **Shot budget** : modify the shot-noise code to compute the total shots needed for a target of \\(10^{-4}\\) hartree at a coefficient 1-norm of \\(\lambda = 50\\). Then explain in one sentence why measuring commuting Pauli terms in groups changes the constant but not the exponent.
  3. **Variance is not always one** : the code uses the worst case \\(\mathrm{Var}(P) = 1\\). Show that \\(\mathrm{Var}(P) = 1 - \langle P \rangle^2\\), and describe the physical situation in which the required shot count is far smaller than the table suggests.
  4. **Barren plateau reasoning** : explain why an ansatz that is *more* expressive can be *harder* to optimize, and why a problem-inspired ansatz sidesteps the tension. Relate your answer to the variational principle from Chapter 3.
  5. **Active space honesty** : you are given two calculations of the same reaction energy — a quantum result on an eight-orbital active space and a classical result on a forty-orbital one. List the reasons this is not a valid comparison, and state what you would need in order to make one.
  6. **Applying the criterion** : find a public claim of a quantum chemistry calculation on hardware and test it against the three clauses of Section 5.6. Note explicitly which clauses the claim does not let you evaluate.
  7. **Depth versus statistics** : using the table in Section 5.8, argue which of VQE and phase estimation you would prefer if gate error rates improved by two orders of magnitude but sampling rates did not improve at all.

## Summary

This chapter took the H₂ calculation of Chapter 4 apart to see what breaks when the molecule grows. **H₂ was easy for structural reasons** : four spin-orbitals, four qubits, and a fifteen-term Pauli Hamiltonian whose count we derived from symmetry rather than quoted — small enough that a laptop solves it exactly, which is why hardware demonstrations of it validate the method and not an advantage. **Four budgets grow at different rates** : qubits linearly in the basis, Hamiltonian terms as the **fourth power** of it, circuit depth with the ansatz against a budget of order \\(1/\epsilon\\), and shots as \\(\epsilon^{-2}\\) multiplied by the coefficient 1-norm. Our first NumPy computation showed the fourth-power law directly, with permutational symmetry buying a constant factor of roughly eight and never touching the exponent. **Measurement is the quiet killer** : chemical accuracy of about \\(1.6 \times 10^{-3}\\) hartree costs hundreds of thousands of shots for a *single* Pauli term, every factor of ten in accuracy costs a factor of one hundred in shots, and the 1-norm grows with the molecule while the accuracy target does not move. **Barren plateaus** threaten the hardware-efficient instinct, because expressive unstructured ansätze have gradients that vanish exponentially in the qubit count; the mitigations named — problem-inspired ansätze, initialization strategies, adaptive construction — all amount to putting structure back in. **Problems can be made smaller** by active spaces (the highest-leverage reduction, and the easiest place for a comparison to become dishonest), exact symmetry reductions, and embedding schemes that cast a quantum device as a fragment solver inside a classical workflow. **The classical competition is not standing still** — DFT, coupled-cluster, tensor networks, quantum Monte Carlo, and machine-learned potentials are all advancing into the same territory — so the honest criterion has three clauses: the best classical method must fail, a quantum device must succeed on the *same* problem, and the result must be verifiable. That overlap region **is still empty**. For materials science, strongly correlated materials, catalysis, and batteries are the reasoned **hypotheses** for where it might first fill, with the leveraged benefit that better energies improve MI training sets rather than single calculations. **Quantum phase estimation** is the eventual algorithm, trading the measurement wall for circuit depth and therefore for error correction — which hands the question over to the Hardware series.

This completes the *Quantum Chemistry with Quantum Computers* series. You can now build a molecular ground-state calculation on a quantum circuit from scratch, and — just as importantly — you can read someone else's claim about one and know which questions decide whether it means anything. Calibration over allegiance: that is the durable part, and it will outlast every specification in the field.

[← Chapter 4: Hands-On: H2 from Scratch](<chapter-4.html>) [Series Top →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
