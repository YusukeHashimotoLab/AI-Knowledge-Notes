---
title: "Chapter 5: NISQ Era and Applications to Chemistry and Materials"
chapter_title: "Chapter 5: NISQ Era and Applications to Chemistry and Materials"
subtitle: "Noisy Hardware, Variational Algorithms, and the Road to Useful Quantum Simulation"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/WpgdCsBz6WQ"
    title="Quantum Computing Ch.5: NISQ Era and Applications to Chemistry and Materials"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-computing-introduction/chapter-5.html>) | Last sync: 2026-08-16

[Quantum Computing Dojo](<../index.html>) > [Introduction to Quantum Computing](<index.html>) > Chapter 5

## 5.1 What "NISQ" Means

In Chapter 4 we met algorithms such as Grover's search and Shor's factoring that promise dramatic speedups. Those algorithms share an uncomfortable assumption: the qubits stay perfectly coherent for the entire computation. Real devices do not behave that way, and the gap between the idealized circuits of Chapter 3 and today's hardware is exactly what this chapter is about.

John Preskill named this situation in 2018 with the term **NISQ** — *Noisy Intermediate-Scale Quantum*.

### 📚 The Two Halves of the Acronym

**Intermediate-Scale** : the device holds enough qubits that simulating it on a classical computer becomes expensive, but not so many that we can spend most of them on error correction. Preskill placed this regime at roughly tens to hundreds of qubits.

**Noisy** : every gate, every measurement, and simply waiting all introduce errors, and there is no full error-correction layer hiding them. The number of gates we can apply before the result dissolves into noise is therefore limited.

The practical consequence is a hard budget. If a single two-qubit gate succeeds with probability \\(1 - \epsilon\\), then a circuit of \\(m\\) such gates succeeds with roughly

\\[ P_{\text{success}} \approx (1 - \epsilon)^m \approx e^{-m\epsilon} \\]

so the useful **circuit depth** is on the order of \\(1/\epsilon\\). This single inequality explains most of NISQ-era algorithm design: we look for algorithms with shallow circuits, because deep ones simply do not survive.

## 5.2 How Qubits Are Actually Built

Several physical platforms are being developed in parallel, and none of them is a clear winner yet. Each makes a different trade between speed, coherence, and connectivity. The table below is deliberately **qualitative** — published device specifications change from month to month, and any number written here would be stale before you read it.

| Platform | Physical qubit | Strengths | Main challenges |
|---|---|---|---|
| **Superconducting circuits** | Microwave circuit on a chip (e.g. transmon) | Very fast gates; chip fabrication borrows from the semiconductor industry; easy to scale the layout | Requires dilution-refrigerator temperatures (millikelvin); relatively short coherence times; typically only neighbouring qubits couple directly |
| **Trapped ions** | Individual ions held in an electromagnetic trap, addressed by lasers | Long coherence times; high gate and measurement fidelity; every ion in a chain can interact with every other (all-to-all connectivity) | Gates are much slower than superconducting gates; scaling one long chain is hard, so architectures shuttle ions between zones |
| **Photonics** | Single photons in waveguides or free space | Photons barely interact with their environment, so they decohere very little; naturally suited to networking and communication | Photons also barely interact with *each other* , making two-qubit gates hard; photon loss is the dominant error; high-efficiency detectors still need cryogenics |
| **Neutral atoms** | Neutral atoms held in optical tweezers, coupled through Rydberg states | Array geometry is reconfigurable; large arrays of identical atoms are relatively easy to assemble; flexible connectivity | Atom loss and reloading; gate fidelities and readout are still maturing relative to trapped ions |

**How to read this table.** Connectivity matters more than beginners expect. An algorithm written for all-to-all connectivity must be compiled onto a nearest-neighbour chip by inserting SWAP gates, and each SWAP costs depth we cannot afford. Gate speed matters for the same reason: what really counts is not the coherence time in seconds, but *how many gates fit inside it*.

## 5.3 Noise and the Idea of Error Correction

### 📚 Where the Errors Come From

**Decoherence** is the loss of quantum information to the environment. Two timescales are conventionally used to describe it:

  * \\(T_1\\) — **energy relaxation** : the qubit decays from \\(|1\rangle\\) toward \\(|0\rangle\\), like an excited atom emitting a photon.
  * \\(T_2\\) — **dephasing** : the *relative phase* in \\(\alpha|0\rangle + \beta|1\rangle\\) drifts randomly, destroying the interference that Chapter 2 showed to be the source of quantum advantage.

**Gate errors** are imperfections in the control pulses themselves: a rotation meant to be \\(\pi/2\\) comes out slightly larger, or a two-qubit interaction leaks a little population into a state outside the computational subspace. **Measurement errors** add a final layer, misreporting a \\(|0\rangle\\) as a \\(|1\rangle\\).

Notice the asymmetry with classical computing. A classical bit is either 0 or 1, so a small voltage drift is simply rounded away. A qubit lives on a continuum of superpositions, so a small error stays small but never disappears — it accumulates.

### 📚 Logical Qubits and Physical Qubits

Quantum error correction solves this by refusing to store information in a single qubit. Instead, one **logical qubit** is encoded across many **physical qubits** , and we repeatedly measure carefully chosen combinations of them.

The trick that makes this possible is the **syndrome measurement** : we measure operators that reveal *whether an error occurred and where* , without ever revealing the encoded state itself. Measuring the state would collapse it; measuring only the error pattern leaves the superposition intact.

The **surface code** is the most-studied scheme for near-term hardware. Physical qubits sit on a two-dimensional lattice, each syndrome check involves only neighbouring qubits — which matches the nearest-neighbour connectivity of a superconducting chip — and the logical information is stored non-locally in the lattice as a whole. Its appeal is a comparatively forgiving **error threshold** : if physical error rates are pushed below roughly the one-percent level, adding more physical qubits per logical qubit makes the logical error rate fall.

The price is steep. Estimates of the encoding **overhead** depend strongly on the physical error rate and on the logical error rate you demand, but figures on the order of a thousand physical qubits per logical qubit are commonly quoted for running large algorithms. Treat that as an order of magnitude and a moving target, not a specification: it is precisely the number that hardware and code improvements are trying to reduce.

> **Why this framing matters**
>
> "How many qubits does the machine have?" is the wrong question on its own. A thousand noisy physical qubits and a thousand error-corrected logical qubits are separated by several generations of engineering. When you read hardware announcements, always check which kind is being counted.

## 5.4 Variational Algorithms: Doing Useful Work Anyway

If deep circuits are unaffordable, can shallow ones still do something valuable? The **variational** family of algorithms is the leading answer.

### 📚 The Variational Quantum Eigensolver (VQE)

VQE targets a problem that is central to chemistry and materials science: find the **ground-state energy** of a Hamiltonian \\(\hat{H}\\), the lowest energy the system can have.

It rests on the **variational principle** , which you may recognize from quantum mechanics. For *any* normalized trial state \\(|\psi\rangle\\),

\\[ E(\psi) = \langle \psi | \hat{H} | \psi \rangle \geq E_0 \\]

where \\(E_0\\) is the true ground-state energy. Every trial state gives an *upper bound* — so if we search over many trial states and keep the smallest value, we can only get closer to the truth, never overshoot below it.

VQE turns this into a **hybrid quantum-classical loop** :

  1. **Prepare** a trial state \\(|\psi(\boldsymbol{\theta})\rangle\\) on the quantum computer using a shallow parametrized circuit, called the **ansatz**. The parameters \\(\boldsymbol{\theta}\\) are just rotation angles.
  2. **Measure** the energy \\(E(\boldsymbol{\theta}) = \langle \psi(\boldsymbol{\theta}) | \hat{H} | \psi(\boldsymbol{\theta}) \rangle\\). In practice \\(\hat{H}\\) is written as a sum of Pauli terms and each term is measured separately, then combined classically.
  3. **Update** \\(\boldsymbol{\theta}\\) with an ordinary classical optimizer running on a laptop.
  4. **Repeat** until the energy stops decreasing.

The division of labour is the point. The quantum computer does only what it is uniquely good at — holding and sampling a state in an exponentially large space — and it does so with a *short* circuit, run many times. All the bookkeeping and optimization happens classically, where noise is not an issue.

**Gradients without finite differences.** For circuits built from the rotation gates of Chapter 3, the derivative of the energy with respect to an angle can be obtained exactly by evaluating the *same circuit* at two shifted angles:

\\[ \frac{\partial E}{\partial \theta} = \frac{1}{2}\left[ E\left(\theta + \frac{\pi}{2}\right) - E\left(\theta - \frac{\pi}{2}\right) \right] \\]

This is the **parameter-shift rule**. Unlike a finite-difference approximation, it is exact and does not require a small step size — a real advantage when every evaluation is a noisy measurement.

**Honest limitations.** VQE is not a free lunch. The accuracy is capped by how expressive the ansatz is: if the true ground state lies outside the family \\(|\psi(\boldsymbol{\theta})\rangle\\) can reach, no optimizer will find it. Measuring many Pauli terms to chemical accuracy requires a very large number of circuit repetitions. And the classical optimization landscape can suffer from **barren plateaus** , regions where gradients vanish exponentially with system size, making the search stall.

### 📚 QAOA in One Paragraph

The **Quantum Approximate Optimization Algorithm (QAOA)** applies the same hybrid recipe to combinatorial optimization — problems like Max-Cut, scheduling, or portfolio selection, where the answer is a choice of bits rather than a molecular energy. The cost function is encoded as a Hamiltonian whose ground state is the optimal bit string, and a shallow alternating circuit of "cost" and "mixer" layers is tuned by a classical optimizer. Whether QAOA beats good classical heuristics on practical instances is still an open research question; treat claims of optimization advantage with the same care as any other NISQ-era claim.

## 5.5 Hands-On: A Toy VQE in NumPy

Let us make the loop concrete with the smallest possible example: a **single qubit**. We take the Hamiltonian

\\[ \hat{H} = 0.4\,Z + 0.3\,X \\]

which is exactly the form a molecular Hamiltonian takes after being mapped onto qubits — a weighted sum of Pauli operators. Our ansatz rotates \\(|0\rangle\\) about the \\(y\\)-axis:

\\[ |\psi(\theta)\rangle = R_y(\theta)|0\rangle = \cos\frac{\theta}{2}|0\rangle + \sin\frac{\theta}{2}|1\rangle \\]

Because this ansatz can reach every real superposition of \\(|0\rangle\\) and \\(|1\rangle\\), and our Hamiltonian is real, the exact ground state *is* inside the family — so a successful optimization should land on the exact answer. That makes it a perfect test case: we can check the VQE result against `numpy.linalg.eigh`.

```python
import numpy as np

# --- 1. The problem: a one-qubit Hamiltonian in the Pauli basis ---
X = np.array([[0, 1], [1, 0]], dtype=float)
Z = np.array([[1, 0], [0, -1]], dtype=float)

# H = 0.4 Z + 0.3 X  (a stand-in for a molecular Hamiltonian mapped to qubits)
H = 0.4 * Z + 0.3 * X

# --- 2. The ansatz: |psi(theta)> = Ry(theta)|0> ---
def ry_state(theta):
    """State obtained by rotating |0> about the y-axis by an angle theta."""
    return np.array([np.cos(theta / 2.0), np.sin(theta / 2.0)])

def energy(theta):
    """<psi(theta)|H|psi(theta)> - the quantity a quantum computer would measure."""
    psi = ry_state(theta)
    return float(psi @ H @ psi)

# --- 3. Hybrid loop, step 1: coarse grid scan ---
grid = np.linspace(0.0, 2.0 * np.pi, 25)
values = np.array([energy(t) for t in grid])
i_best = int(np.argmin(values))
theta = grid[i_best]
print("Grid scan over 25 angles")
print(f"  best theta  = {theta:.4f} rad")
print(f"  best energy = {values[i_best]:.6f}")

# --- 4. Hybrid loop, step 2: gradient descent with the parameter-shift rule ---
# For this ansatz the exact gradient is [E(theta+pi/2) - E(theta-pi/2)] / 2,
# a formula evaluated with the SAME circuit at two shifted angles.
def parameter_shift_gradient(theta):
    return 0.5 * (energy(theta + np.pi / 2) - energy(theta - np.pi / 2))

learning_rate = 0.4
for step in range(200):
    theta -= learning_rate * parameter_shift_gradient(theta)
vqe_energy = energy(theta)
print("Gradient descent (200 steps, parameter-shift rule)")
print(f"  optimal theta = {theta:.6f} rad")
print(f"  VQE energy    = {vqe_energy:.9f}")

# --- 5. The reference: exact diagonalization ---
exact_values, exact_vectors = np.linalg.eigh(H)
print("Exact diagonalization")
print(f"  eigenvalues   = {exact_values[0]:.9f}, {exact_values[1]:.9f}")
print(f"  ground energy = {exact_values[0]:.9f}")
print(f"  |VQE - exact| = {abs(vqe_energy - exact_values[0]):.2e}")

# --- 6. Did we find the ground STATE, not just the ground ENERGY? ---
overlap = abs(ry_state(theta) @ exact_vectors[:, 0])
print(f"Overlap with true ground state = {overlap:.9f}")
```

**Output:**

```
Grid scan over 25 angles
  best theta  = 3.6652 rad
  best energy = -0.496410
Gradient descent (200 steps, parameter-shift rule)
  optimal theta = 3.785094 rad
  VQE energy    = -0.500000000
Exact diagonalization
  eigenvalues   = -0.500000000, 0.500000000
  ground energy = -0.500000000
  |VQE - exact| = 0.00e+00
Overlap with true ground state = 1.000000000
```

**Reading the result.** Three things are worth noticing.

  * The grid scan alone reached \\(-0.4964\\) — close, but not the answer. The classical optimizer did the fine work, exactly as it does in a real VQE run.
  * The converged energy \\(-0.5\\) matches the analytic ground state \\(-\sqrt{0.4^2 + 0.3^2} = -0.5\\) and the `eigh` result to machine precision.
  * The overlap of \\(1.0\\) confirms we found the ground *state* , not merely a state that happens to have the right energy.

Try modifying the Hamiltonian coefficients, or replacing the \\(R_y\\) ansatz with one that *cannot* reach the ground state — you will see the energy plateau above \\(E_0\\), which is the variational principle protecting you from a wrong answer that looks too good.

**What this toy hides.** Here we computed \\(\langle \hat{H} \rangle\\) exactly from the state vector. A real device estimates it from a finite number of measurement shots, so every energy value carries statistical noise, and the optimizer must cope with it. And a single qubit is trivially simulable classically — the interesting regime starts when the state vector no longer fits in memory.

## 5.6 Chemistry and Materials: The Natural Killer Application

### 📚 Feynman's Argument

In the early 1980s Richard Feynman pointed out something that still frames the entire field. Simulating a quantum system on a classical computer is hard for a structural reason: the state of \\(n\\) interacting quantum particles requires an amount of classical data that grows exponentially with \\(n\\). His proposed remedy was direct: *build the simulator out of quantum mechanics too.*

This is why electronic structure is the most-cited application. We are not asking a quantum computer to imitate something foreign to it — we are asking it to represent one quantum system with another. The correspondence is natural, and the exponential wall that stops exact classical methods is not a wall for a quantum device.

Classical quantum chemistry is, of course, enormously successful. Density functional theory (DFT) routinely handles hundreds of atoms; coupled-cluster methods reach high accuracy for well-behaved molecules. The difficulty concentrates in **strongly correlated** systems — transition-metal complexes, bond-breaking, some magnetic and catalytic materials — where the wavefunction cannot be well approximated by a single dominant configuration, and the exact methods scale exponentially.

### 📚 H₂: The Canonical Demonstration

The hydrogen molecule is to quantum chemistry on quantum computers what "Hello, World" is to programming. In a minimal basis set, the electronic Hamiltonian of H₂ maps onto a small number of qubits and becomes a weighted sum of Pauli terms — structurally the same object as the \\(0.4Z + 0.3X\\) in our code above, just with more terms. Running VQE while varying the internuclear distance traces out the **potential energy curve** , with its minimum at the equilibrium bond length.

Since VQE was proposed in 2014, H₂ and other small molecules such as LiH and BeH₂ have been used as benchmarks across several hardware platforms. These are genuine achievements of experimental control. But be clear about what they demonstrate: every one of these molecules can be solved to higher accuracy, faster, on a laptop. They validate the method, not a computational advantage.

### 📚 An Honest Status Report

Where does that leave us? The honest summary has three parts.

  * **The theory is sound.** Quantum algorithms for simulating electronic structure exist, and their scaling advantage over exact classical methods for strongly correlated systems is well founded.
  * **The hardware is not there yet.** Chemically useful accuracy on classically intractable molecules is generally expected to require error correction, which means logical qubits and therefore a large multiple of today's physical qubit counts.
  * **The overlap region is still empty.** Problems that are simultaneously *classically hard* and *quantum-tractable today* have not yet been convincingly demonstrated. Finding them is an active and legitimate research programme — and it is also where a good deal of over-claiming happens.

This is not a reason for pessimism, but it is a reason for calibration. Approach any claim of quantum advantage in chemistry by asking: what is the best classical method on the same problem, run by someone trying to win?

### 📚 Connection to Materials Informatics

If you came to this series from the materials informatics or machine learning side of AI Terakoya, the connection is worth spelling out, because it runs in both directions.

**Quantum computing as a data source.** Machine-learning models for materials are trained on data — usually DFT calculations, occasionally experiments. Their accuracy is bounded by the accuracy of that data. If quantum simulation eventually supplies reliable energies for strongly correlated systems that DFT handles poorly, it improves the *training set* , not just one calculation. That is a leveraged effect.

**Classical methods as the benchmark.** Meanwhile, the discipline of the MI workflow applies directly here: hold out a test set, compare against a strong baseline, and report the honest error. A quantum result that is not compared against the best available classical method tells you very little — the same lesson as a machine-learning model reported without a baseline.

**Hybrid thinking is the shared skill.** VQE is a quantum subroutine wrapped in a classical optimization loop. That architecture — expensive evaluator inside a cheap optimizer — is exactly the pattern behind Bayesian optimization and active learning in materials discovery. The intuition you have built in one field transfers to the other.

### 🎯 Exercise Problems

  1. **Circuit depth budget** : if a two-qubit gate has error rate \\(\epsilon = 10^{-3}\\), estimate how many such gates can be applied before the success probability falls below 1/2. Repeat for \\(\epsilon = 10^{-2}\\) and comment on what an order of magnitude in gate fidelity buys you.
  2. **Variational bound** : replace the ansatz in the code with \\(|\psi(\theta)\rangle\\) restricted to \\(\theta \in [0, \pi/2]\\). Show numerically that the optimized energy stays *above* the exact ground energy, and explain why the variational principle guarantees this.
  3. **Parameter-shift rule** : for \\(\hat{H} = aZ + bX\\) and the \\(R_y\\) ansatz, show analytically that \\(E(\theta) = a\cos\theta + b\sin\theta\\), and verify that the parameter-shift formula reproduces \\(dE/d\theta\\) exactly.
  4. **Platform choice** : an algorithm requires many gates between distant qubits. Using the table in Section 5.2, argue which hardware modality is favoured and what the cost would be on the others.
  5. **Reading claims critically** : find a public announcement of a quantum chemistry calculation and identify (a) physical or logical qubits, (b) the molecule and basis set, and (c) whether a classical baseline is reported.

## Summary

In this chapter we came down from idealized circuits to the machines that actually exist. **NISQ** describes today's regime — tens to hundreds of noisy qubits with no full error correction — and its central constraint is a limited circuit depth of order \\(1/\epsilon\\). We compared the main **hardware modalities** , finding that superconducting circuits, trapped ions, photonics, and neutral atoms each trade gate speed against coherence and connectivity, with no winner yet. **Quantum error correction** offers the way out through logical qubits encoded in many physical ones, with the surface code as the leading near-term scheme, but the encoding overhead is large enough that it defines the timeline of the whole field. **Variational algorithms** such as VQE and QAOA are the practical response: a shallow parametrized circuit on the quantum device wrapped in a classical optimization loop, protected by the variational principle. Our NumPy toy VQE reproduced the exact ground-state energy of \\(0.4Z + 0.3X\\) to machine precision, illustrating the full hybrid loop in under fifty lines. Finally, **quantum simulation of chemistry and materials** is the most natural application — Feynman's argument that quantum systems should be simulated by quantum machines — with H₂ as the canonical demonstration, and with genuinely useful, classically intractable calculations still ahead of us rather than behind.

This completes the *Introduction to Quantum Computing* series. You now have the vocabulary to read the literature, the linear algebra to follow the mathematics, and — just as importantly — the calibration to tell a real result from a press release. If you want to go deeper into the physics underneath the qubit, the *Introduction to Quantum Mechanics* series is the natural next step; if you want to go deeper into the applications, the materials informatics series show what the data-driven side of materials discovery looks like today.

[← Chapter 4: Quantum Algorithms](<chapter-4.html>) [Series Top →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
