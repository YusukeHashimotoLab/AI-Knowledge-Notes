---
title: "Chapter 3: VQE: The Algorithm"
chapter_title: "Chapter 3: VQE: The Algorithm"
subtitle: "The Variational Principle, the Hybrid Loop, and What Each Half of the Machine Is For"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/x2DG05i2n_s"
    title="Quantum Chemistry Ch.3: VQE: The Algorithm"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-chemistry-computing/chapter-3.html>) | Last sync: 2026-08-17

[Quantum Computing Dojo](<../index.html>) > [Quantum Chemistry with Quantum Computers](<index.html>) > Chapter 3

Chapter 2 ended with a Hamiltonian: the electronic structure problem, rewritten in second quantization and mapped onto qubits, arriving as a weighted sum of Pauli strings. That object is a matrix in a space of dimension \\(2^n\\), and we want its lowest eigenvalue. The obvious plan — build the matrix and diagonalize it — is exactly the plan that fails, because the matrix is the thing we cannot afford to write down.

The **Variational Quantum Eigensolver (VQE)** is the leading near-term answer. This chapter takes it apart: why the variational principle turns an eigenvalue problem into a minimization, what each half of the machine contributes, and where each of the four moving parts — ansatz, measurement, optimizer, and the interface between them — can quietly go wrong. The *Introduction to Quantum Computing* series met VQE in one section and made a point that governs this chapter too: demonstrations on small molecules validate the method, not a computational advantage. Nothing here changes that; only the level of detail changes.

## 3.1 The Variational Principle

Everything rests on one inequality. For any normalized trial state \\(|\psi(\boldsymbol{\theta})\rangle\\) and any Hamiltonian \\(\hat{H}\\) with ground-state energy \\(E_0\\),

\\[ E(\boldsymbol{\theta}) = \langle \psi(\boldsymbol{\theta}) | \hat{H} | \psi(\boldsymbol{\theta}) \rangle \geq E_0 \\]

with equality if and only if \\(|\psi(\boldsymbol{\theta})\rangle\\) lies entirely in the ground-state eigenspace.

### 📚 The Proof, in Two Lines

Expand the trial state in the (unknown) eigenbasis of \\(\hat{H}\\), where \\(\hat{H}|k\rangle = E_k|k\rangle\\) with \\(E_0 \leq E_1 \leq \cdots\\). The expectation value is then a weighted average of eigenvalues, and replacing every eigenvalue by the smallest one can only decrease it:

\\[ |\psi\rangle = \sum_k c_k |k\rangle \quad \Longrightarrow \quad \langle \psi | \hat{H} | \psi \rangle = \sum_k |c_k|^2 E_k \;\geq\; E_0 \sum_k |c_k|^2 = E_0 \\]

The inequality is saturated only when \\(|c_k|^2 = 0\\) for every \\(k\\) with \\(E_k > E_0\\) — that is, when the trial state is a ground state. Note what the proof does *not* need: it never asks us to know the eigenbasis, only that one exists. That is why the bound is usable when the spectrum is entirely unknown to us.

An eigenvalue problem has become a **minimization** problem. We do not have to find \\(E_0\\); we push \\(E(\boldsymbol{\theta})\\) down and report the smallest value reached, knowing it is an upper bound. A wrong answer errs in a known direction.

### 📚 Why Energy Errors Are Second Order in State Errors

This is the property most often quoted as VQE's robustness, and it deserves precision rather than enthusiasm. Suppose the circuit prepares not \\(|\psi_0\rangle\\) but a slightly wrong *pure* state, written as a small admixture of an orthogonal normalized \\(|\delta\rangle\\):

\\[ |\psi\rangle = \frac{|\psi_0\rangle + \epsilon |\delta\rangle}{\sqrt{1 + \epsilon^2}} \quad \Longrightarrow \quad E(\psi) = \frac{E_0 + \epsilon^2 \langle \delta | \hat{H} | \delta \rangle}{1 + \epsilon^2} = E_0 + \epsilon^2 \left( \langle \delta | \hat{H} | \delta \rangle - E_0 \right) + O(\epsilon^4) \\]

The state is wrong at order \\(\epsilon\\); the energy is wrong at order \\(\epsilon^2\\). A one-percent error in the prepared state costs about one part in ten thousand in the energy. That is genuine and useful — and narrower than it is usually made to sound.

> **Three things this does not protect you from**
>
> **Incoherent noise.** The derivation assumes a *pure* state near the ground state. Decoherence produces a mixed state, and a mixture carrying weight \\(p\\) on excited states raises the energy linearly in \\(p\\). Depolarizing noise is a first-order error.
>
> **Measurement bias.** The quadratic suppression covers state preparation only. A systematic readout error that biases an estimated Pauli expectation shifts the reported energy at first order, and no variational argument removes it.
>
> **Shot noise and the bound itself.** The inequality \\(E \geq E_0\\) holds for the exact expectation value. A finite-shot *estimate* is a random variable and can land below \\(E_0\\) by chance. Reporting the lowest sampled energy rather than the converged mean is a way to appear to beat the variational bound while doing nothing of the kind.

## 3.2 The Anatomy of the Loop

VQE splits the work across two machines, each doing what it is good at.

    
    
    ```mermaid
    flowchart TD
        A[Classical: choose parameters theta]
        B[Quantum: prepare ansatz state<br/>shallow parametrized circuit]
        C[Quantum: measure each Pauli string<br/>repeated shots per term]
        D[Classical: combine terms into the energy]
        E[Classical: optimizer proposes new theta]
        A --> B --> C --> D --> E
        E -.repeat until converged.-> B
        style A fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
        style B fill:#00bcd4,stroke:#764ba2,stroke-width:2px,color:#fff
        style C fill:#00bcd4,stroke:#764ba2,stroke-width:2px,color:#fff
        style D fill:#7c4dff,stroke:#764ba2,stroke-width:2px,color:#fff
        style E fill:#f57c00,stroke:#764ba2,stroke-width:2px,color:#fff
    ```

**Prepare.** A parametrized circuit \\(U(\boldsymbol{\theta})\\) acts on a fixed reference, usually the Hartree-Fock determinant written as a bit string. The result \\(|\psi(\boldsymbol{\theta})\rangle = U(\boldsymbol{\theta})|\text{ref}\rangle\\) is never written down as a vector; it exists only inside the device.

**Measure.** With \\(\hat{H} = \sum_i c_i \hat{P}_i\\) a sum of Pauli strings, each \\(\langle \hat{P}_i \rangle\\) is estimated by running the circuit many times, applying the basis rotation that turns \\(\hat{P}_i\\) into a product of \\(Z\\) operators, and averaging the \\(\pm 1\\) outcomes.

**Combine and update.** The classical computer forms \\(E(\boldsymbol{\theta}) = \sum_i c_i \langle \hat{P}_i \rangle\\) and hands it to an optimizer, which proposes new parameters.

### 📚 Why This Suits NISQ Hardware

The circuit that runs on the device is **shallow and repeated**, not deep and run once. With a two-qubit gate error \\(\epsilon\\), a circuit of \\(m\\) gates survives with probability roughly \\(e^{-m\epsilon}\\), so affordable depth is of order \\(1/\epsilon\\). An algorithm needing one long coherent computation is disqualified; one needing many short ones is not.

The classical half absorbs everything a noisy device does badly — bookkeeping, accumulating many small numbers, convergence tests — all where arithmetic is exact and free. The quantum device supplies one service classical hardware cannot: sampling from a state in a space too large to enumerate.

That framing also locates the honest question. VQE fits NISQ hardware because it *asks little*, and asking little is not the same as delivering something classically unavailable. Chapter 5 returns to the gap between those two statements.

## 3.3 Ansatz Design: The Choice That Caps Your Accuracy

The optimizer can only search inside the family of states the circuit can produce. If the ground state is not in that family, no optimization finds it — the energy plateaus above \\(E_0\\), and the variational principle guarantees only that you will not be misled downward. Two design philosophies dominate, and they fail in opposite ways.

### 📚 Hardware-Efficient Ansätze

Build the circuit out of whatever the device does natively: layers of single-qubit rotations, each with its own angle, alternating with a fixed pattern of entangling gates between physically coupled qubits.

\\[ U(\boldsymbol{\theta}) = \prod_{\ell=1}^{L} \left[ W_{\text{ent}} \cdot \bigotimes_{q=1}^{n} R(\theta_{\ell,q}) \right] \\]

**The appeal is depth.** Nothing in the circuit is dictated by chemistry, so nothing forces it to be long, and the entangling layer uses only couplings that physically exist — no routing gates are inserted.

**The cost is structure.** The reachable family bears no relation to the physics of the molecule. It generally contains states with the wrong particle number and the wrong spin, so the optimizer can wander out of the physically meaningful sector and return an energy belonging to a different chemical species. It also encodes no prior knowledge, so nothing guides the search.

Worse, expressiveness turns against you. As the circuit is widened and deepened toward a family general enough to contain the ground state, the landscape flattens: gradients become exponentially small in the number of qubits over most of parameter space. These are **barren plateaus**, and Chapter 5 treats them as the central obstacle they are. Note the shape of the trap — the ansatz expressive enough to hold the answer can be the one you cannot train.

### 📚 Chemistry-Inspired Ansätze: Unitary Coupled Cluster

The alternative imports an ansatz from classical quantum chemistry. **Coupled cluster** theory writes the correlated wavefunction as an exponential of excitation operators acting on the Hartree-Fock reference,

\\[ |\psi(\boldsymbol{\theta})\rangle = e^{\hat{T}(\boldsymbol{\theta}) - \hat{T}^\dagger(\boldsymbol{\theta})} |\text{HF}\rangle \\]

where \\(\hat{T}\\) promotes electrons from occupied orbitals into empty ones — singles and doubles in the common **UCCSD** truncation, one parameter per excitation. The antihermitian combination \\(\hat{T} - \hat{T}^\dagger\\) is what makes the exponential unitary, and therefore implementable as a circuit. We stay qualitative, and three points matter.

  * **It respects the symmetries.** Every term moves electrons between orbitals without creating or destroying them, so particle number and spin are conserved by construction and the optimizer cannot leave the physical sector.
  * **The parameter count is polynomial**, set by the number of occupied-virtual orbital pairs rather than by the size of the Hilbert space.
  * **The circuits are deep.** Turning the exponential into gates requires a Trotter decomposition, and the resulting depth is what makes UCCSD demanding on hardware with a depth budget. That is the trade: chemical structure bought with circuit depth.

Between the poles sit adaptive constructions that grow the ansatz one operator at a time, adding whichever excitation has the largest energy gradient. They buy compactness with many extra measurements — a real cost, not a free lunch.

## 3.4 Measurement: Where the Shots Go

A device does not evaluate \\(\langle \hat{H} \rangle\\); it samples. Since \\(\hat{P}_i^2 = I\\), each measurement of a Pauli string returns \\(+1\\) or \\(-1\\), so the single-shot variance is \\(\mathrm{Var}(\hat{P}_i) = 1 - \langle \hat{P}_i \rangle^2\\) and independent terms add in variance:

\\[ \mathrm{Var}\big(\hat{E}\big) = \sum_i \frac{c_i^2 \left( 1 - \langle \hat{P}_i \rangle^2 \right)}{N_i} \\]

**The scaling is the problem.** Halving the statistical error requires quadrupling the shots, so reaching a target precision \\(\varepsilon\\) costs shots growing as \\(1/\varepsilon^2\\) — and molecular Hamiltonians carry many terms whose coefficients do not conveniently shrink. This, rather than gate count, is frequently the dominant cost of a VQE run, and it is why chemical accuracy is expensive to claim rather than merely difficult.

### 📚 Grouping Commuting Terms

Two Pauli strings that commute can in principle be measured simultaneously. The easiest case to exploit is **qubit-wise commutation**: strings that agree, qubit by qubit, on which Pauli they apply, with identity compatible with anything. All members of such a group share one basis rotation, so a single set of shots yields all of them.

Partitioning \\(M\\) terms into as few groups as possible is a graph-colouring problem, solved heuristically in practice. The saving is real and can be large, but it is a constant-factor improvement on a \\(1/\varepsilon^2\\) law. It changes the coefficient, not the exponent.

## 3.5 The Optimizer

The classical half sees a function that is expensive to evaluate and noisy. That combination shapes every sensible choice.

### 📚 The Parameter-Shift Rule

For rotation gates \\(R_G(\theta) = e^{-i\theta \hat{G}/2}\\) whose generator satisfies \\(\hat{G}^2 = I\\) — true for every single-qubit Pauli rotation — the derivative is *exact* and obtained from the same circuit at two shifted angles:

\\[ \frac{\partial E}{\partial \theta_k} = \frac{1}{2}\left[ E\left(\theta_k + \frac{\pi}{2}\right) - E\left(\theta_k - \frac{\pi}{2}\right) \right] \\]

Two features make this the standard tool. It is **not an approximation**: there is no step size to tune and no truncation error to trade against noise, unlike a finite difference where shrinking the step amplifies statistical error. And the shifted circuits are the *same* circuit at different angles, so no new hardware capability is needed. The cost is honest bookkeeping — two energy evaluations per parameter, each a full set of Pauli measurements, so a gradient step for \\(p\\) parameters costs \\(2p\\) evaluations.

### 📚 Gradient-Free Alternatives

When gradients are too expensive or too noisy, optimizers that never form one are used instead. Nelder-Mead and COBYLA search by direct comparison of function values; SPSA estimates a descent direction from a couple of randomly perturbed evaluations regardless of parameter count, which is why it appears often in noisy hardware experiments. We name them and move on — the optimizer matters, but it is downstream of the two dominant costs: shots per evaluation, and whether the landscape has a gradient worth following at all.

## 3.6 Hands-On: A Complete Two-Qubit VQE

The code below runs the entire loop on a two-qubit Hamiltonian of the form Chapter 2 produced,

\\[ \hat{H} = c_1 Z_0 + c_2 Z_1 + c_3 Z_0 Z_1 + c_4 X_0 X_1 \\]

**The coefficients are a teaching choice.** They are round numbers, carry no units, and belong to no molecule: \\(c_1 = c_2 = 0.5\\), \\(c_3 = 0.25\\), \\(c_4 = 0.3\\). What is realistic is the *structure* — a weighted sum of Pauli strings with a diagonal part and one off-diagonal term supplying the correlation. Chapter 4 replaces these invented numbers with coefficients computed from actual molecular integrals. The ansatz is one layer of the hardware-efficient kind, an \\(R_y\\) rotation on each qubit followed by a CNOT:

\\[ |\psi(\theta_0, \theta_1)\rangle = \mathrm{CNOT} \cdot \left( R_y(\theta_0) \otimes R_y(\theta_1) \right) |00\rangle \\]

```python
import numpy as np

# The TOY Hamiltonian. Round teaching coefficients, no units, no molecule.
# What is realistic is the STRUCTURE: a weighted sum of Pauli strings.
I2, X, Z = np.eye(2), np.array([[0., 1.], [1., 0.]]), np.array([[1., 0.], [0., -1.]])
c1, c2, c3, c4 = 0.5, 0.5, 0.25, 0.30
terms = [(c1, "Z0", np.kron(Z, I2)), (c2, "Z1", np.kron(I2, Z)),
         (c3, "Z0Z1", np.kron(Z, Z)), (c4, "X0X1", np.kron(X, X))]
H = sum(coeff * op for coeff, _, op in terms)

# The classical reference, affordable only because this is a 4x4.
eigvals = np.linalg.eigvalsh(H)
E_exact = float(eigvals[0])
print("Exact diagonalization (numpy.linalg.eigvalsh)")
print("  spectrum      = " + ", ".join(f"{v:+.9f}" for v in eigvals))
print(f"  ground energy = {E_exact:.9f}\n")

# Ansatz:  |psi(theta)> = CNOT . (Ry(theta0) (x) Ry(theta1)) |00>
CNOT = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                 [0., 0., 0., 1.], [0., 0., 1., 0.]])


def ry(t):
    c, s = np.cos(t / 2.0), np.sin(t / 2.0)
    return np.array([[c, -s], [s, c]])


def ansatz_state(theta):
    psi = np.array([1.0, 0.0, 0.0, 0.0])                    # |00>
    return CNOT @ (np.kron(ry(theta[0]), ry(theta[1])) @ psi)


def energy(theta):
    psi = ansatz_state(theta)
    return float(psi @ H @ psi)


# Parameter-shift gradient. Each angle sits in exactly one Ry gate whose
# generator has eigenvalues +-1/2, so the rule is EXACT, not approximate.
def parameter_shift_gradient(theta):
    grad = np.zeros_like(theta)
    for k in range(len(theta)):
        shift = np.zeros_like(theta)
        shift[k] = np.pi / 2.0
        grad[k] = 0.5 * (energy(theta + shift) - energy(theta - shift))
    return grad


probe, h = np.array([0.7, -1.3]), 1e-6
fd = np.array([(energy(probe + h * e) - energy(probe - h * e)) / (2 * h) for e in np.eye(2)])
ps = parameter_shift_gradient(probe)
print("Parameter shift vs central finite difference at theta = (0.70, -1.30)")
print(f"  parameter shift   = [{ps[0]:+.9f}, {ps[1]:+.9f}]")
print(f"  finite difference = [{fd[0]:+.9f}, {fd[1]:+.9f}]")
print(f"  max abs deviation = {np.max(np.abs(ps - fd)):.2e}")
```

**Output:**

```
Exact diagonalization (numpy.linalg.eigvalsh)
  spectrum      = -0.794030651, -0.550000000, +0.050000000, +1.294030651
  ground energy = -0.794030651

Parameter shift vs central finite difference at theta = (0.70, -1.30)
  parameter shift   = [-0.178819926, +0.609374521]
  finite difference = [-0.178819926, +0.609374521]
  max abs deviation = 1.22e-10
```

The parameter-shift gradient agrees with a central finite difference to ten digits — the check worth doing before trusting any gradient. The residual deviation is the finite difference's error, not the shift rule's.

Now the loop itself, run twice from two different starting points.

```python
# The hybrid loop: quantum energy evaluations inside classical descent.
def descend(theta0, label, n_steps=400, learning_rate=0.25):
    theta = np.array(theta0, dtype=float)
    print(label + "\n  step       theta0       theta1         energy    |gradient|")
    for step in range(n_steps + 1):
        g = parameter_shift_gradient(theta)
        if step % 200 == 0:
            print(f"  {step:4d}  {theta[0]:+10.6f}  {theta[1]:+10.6f}  "
                  f"{energy(theta):+12.9f}  {np.linalg.norm(g):.3e}")
        theta = theta - learning_rate * g
    print(f"  converged energy = {energy(theta):.9f}   (exact = {E_exact:.9f})\n")
    return theta


# Run A: an arbitrary starting point. The optimizer behaves perfectly.
descend([0.30, 0.90], "Run A: descent from an arbitrary start")

# Run B: seed the same descent with a coarse scan first.
grid = np.linspace(0.0, 2.0 * np.pi, 13)
best = min(((energy(np.array([a, b])), a, b) for a in grid for b in grid))
print(f"Grid scan over {len(grid)}x{len(grid)} angles: "
      f"theta = ({best[1]:.6f}, {best[2]:.6f}), energy = {best[0]:.9f}\n")
theta = descend([best[1], best[2]], "Run B: descent seeded by the grid scan")
print(f"  E_vqe - E_exact = {energy(theta) - E_exact:+.3e}   (machine epsilon)\n")

# What the device would actually have to measure, term by term.
psi_opt = ansatz_state(theta)
print("Pauli-term expectation values at the optimum")
for coeff, label, op in terms:
    ev = float(psi_opt @ op @ psi_opt)
    print(f"  <{label:5s}> = {ev:+.9f}   contribution = {coeff * ev:+.9f}")
print()

# Shot noise: the device samples the +-1 eigenvalue and averages.
rng = np.random.default_rng(2026)
probs = psi_opt ** 2                                # Z-basis outcome probabilities
z0 = np.array([+1.0, +1.0, -1.0, -1.0])             # Z0 on |00>,|01>,|10>,|11>
exact_z0 = float(probs @ z0)
spread = np.sqrt(1.0 - exact_z0 ** 2)               # single-shot standard deviation
print(f"Finite-shot estimation of <Z0>   (exact = {exact_z0:+.9f}, "
      f"single-shot sd = {spread:.6f})")
print("     shots    mean of 400 runs    sd of 400 runs    sd/sqrt(N) predicted    ratio")
previous = None
for n_shots in [100, 400, 1600, 6400]:
    est = np.array([float(np.mean(z0[rng.choice(4, size=n_shots, p=probs)]))
                    for _ in range(400)])
    sd = est.std(ddof=1)
    ratio = "    -" if previous is None else f"{sd / previous:.3f}"
    print(f"  {n_shots:8d}  {est.mean():+17.6f}  {sd:16.6f}  {spread / np.sqrt(n_shots):22.6f}  {ratio:>7s}")
    previous = sd
```

**Output:**

```
Run A: descent from an arbitrary start
  step       theta0       theta1         energy    |gradient|
     0   +0.300000   +0.900000  +1.018650141  5.719e-01
   200   -1.570796   +3.141592  -0.550000000  1.905e-07
   400   -1.570796   +3.141593  -0.550000000  1.560e-13
  converged energy = -0.550000000   (exact = -0.794030651)

Grid scan over 13x13 angles: theta = (3.665191, 0.000000), energy = -0.766025404

Run B: descent seeded by the grid scan
  step       theta0       theta1         energy    |gradient|
     0   +3.665191   +0.000000  -0.766025404  2.402e-01
   200   +3.433049   +0.000000  -0.794030651  4.441e-16
   400   +3.433049   +0.000000  -0.794030651  4.441e-16
  converged energy = -0.794030651   (exact = -0.794030651)

  E_vqe - E_exact = -1.110e-16   (machine epsilon)

Pauli-term expectation values at the optimum
  <Z0   > = -0.957826285   contribution = -0.478913143
  <Z1   > = -0.957826285   contribution = -0.478913143
  <Z0Z1 > = +1.000000000   contribution = +0.250000000
  <X0X1 > = -0.287347886   contribution = -0.086204366

Finite-shot estimation of <Z0>   (exact = -0.957826285, single-shot sd = 0.287348)
     shots    mean of 400 runs    sd of 400 runs    sd/sqrt(N) predicted    ratio
       100          -0.957700          0.028155                0.028735        -
       400          -0.957138          0.014685                0.014367    0.522
      1600          -0.958025          0.007217                0.007184    0.491
      6400          -0.957978          0.003811                0.003592    0.528
```

**Reading the result.** Five observations, in order of importance.

  * **Run A converges cleanly to the wrong answer.** The gradient norm falls to \\(10^{-13}\\) and every convergence test one might apply is satisfied, yet the energy is \\(-0.550000\\) against a true ground energy of \\(-0.794031\\). Nothing is broken: the optimizer found a local minimum, which is all a local optimizer promises. This failure mode does not announce itself. The variational principle tells you the answer is an upper bound and says nothing about how loose the bound is.
  * **Run B, seeded by a coarse scan, reaches the exact energy.** The converged value matches `numpy.linalg.eigvalsh` to \\(10^{-16}\\). The printed \\(-1.1 \times 10^{-16}\\) is floating-point roundoff at machine epsilon; the variational principle permits only a non-negative gap, and this is that gap being zero.
  * **The two runs land in different symmetry sectors.** This Hamiltonian commutes with \\(Z_0 Z_1\\), so its eigenstates split into a block spanned by \\(|00\rangle, |11\rangle\\) and one spanned by \\(|01\rangle, |10\rangle\\). Run A is trapped in the second, whose lowest energy is exactly \\(-0.55\\). Run B sits at \\(\theta_1 = 0\\), precisely where the ansatz stays inside the first block, so the gradient along \\(\theta_1\\) vanishes and the descent proceeds along \\(\theta_0\\) alone. Chapter 4 turns this observation into a deliberate design tool.
  * **The energy is assembled from four separately measured numbers.** The table of Pauli expectation values is what the device would report; the total is a classical sum of four products. Note \\(\langle Z_0 Z_1 \rangle = +1\\) exactly — a symmetry of the converged state, and a term that on hardware would still consume shots to confirm.
  * **Shot noise falls as one over the square root of the shot count.** The measured standard deviation tracks the prediction \\(\sqrt{1 - \langle Z_0 \rangle^2}/\sqrt{N}\\), and each fourfold increase in shots roughly halves it — the ratio column hovers near \\(0.5\\), with its own sampling fluctuation. Reaching \\(10^{-3}\\) precision on this *single* term already needs of order \\(10^5\\) shots, and a molecular Hamiltonian has many terms.

Try replacing the ansatz with a single \\(R_y\\) on qubit 0 and no entangling gate. The optimized energy stops well above \\(E_0\\) — the family cannot produce an entangled state, and the variational principle turns that inability into a visible, honest error rather than a plausible wrong answer.

### 🎯 Exercise Problems

  1. **Equality in the variational principle.** Using the two-line proof of Section 3.1, show that \\(\langle\psi|\hat{H}|\psi\rangle = E_0\\) forces \\(|\psi\rangle\\) into the ground-state eigenspace. Where does the argument use the fact that \\(E_0\\) is the *smallest* eigenvalue?
  2. **Second order, quantitatively.** Take \\(\epsilon = 0.1\\) and \\(\langle\delta|\hat{H}|\delta\rangle - E_0 = 1\\) in the expansion of Section 3.1. How large is the energy error? Now suppose instead that depolarizing noise puts probability \\(p = 0.1\\) on a state with that same excess energy. Compare the two, and explain why one is quadratic and the other linear.
  3. **Shot budget.** A Hamiltonian has 15 Pauli terms with coefficients of magnitude about \\(0.5\\). Assuming the worst case \\(\langle \hat{P}_i \rangle = 0\\) and an equal split of shots, estimate the total shots needed for a standard error of \\(10^{-3}\\) in the energy. Repeat for \\(10^{-4}\\) and comment.
  4. **Parameter-shift by hand.** For \\(\hat{H} = aZ + bX\\) with \\(|\psi(\theta)\rangle = R_y(\theta)|0\rangle\\), show that \\(E(\theta) = a\cos\theta + b\sin\theta\\), and verify algebraically that the parameter-shift formula returns \\(dE/d\theta\\) exactly rather than approximately.
  5. **Diagnosing a plateau.** Your VQE energy stops decreasing. Give three distinct explanations consistent with that observation — one about the ansatz, one about the optimizer, one about the measurements — and a numerical experiment that would distinguish them.

## Summary

This chapter dismantled VQE into its parts. The **variational principle** \\(\langle\psi(\boldsymbol{\theta})|\hat{H}|\psi(\boldsymbol{\theta})\rangle \geq E_0\\) — proved in two lines by expanding in the eigenbasis, with equality only for a ground state — converts an eigenvalue problem into a minimization whose errors point in a known direction. Energy errors are **second order** in state errors, but only for coherent errors on a pure state: incoherent noise, measurement bias, and finite-shot fluctuation each evade that protection, and the last can even push a reported estimate below \\(E_0\\). The **hybrid loop** prepares a state on the device, estimates each Pauli term from repeated shots, and hands the assembled energy to a classical optimizer, so the quantum circuit stays shallow and is run many times — the shape that fits a depth budget of order \\(1/\epsilon\\). **Ansatz design** trades depth against structure: hardware-efficient circuits are shallow but unstructured and slide toward barren plateaus as they grow, while unitary coupled cluster conserves particle number and spin at the price of deep Trotterized circuits. **Measurement** costs shots as \\(1/\varepsilon^2\\), with commuting-group strategies improving the constant and not the exponent. The **parameter-shift rule** gives exact gradients from two evaluations per parameter, with gradient-free methods as the alternative when that is unaffordable. Our two-qubit NumPy VQE showed all of it at once: one run converging confidently to a local minimum at \\(-0.550000\\), a second run seeded by a grid scan reaching the exact \\(-0.794031\\) to machine precision, the energy assembled from four separately measured Pauli terms, and shot noise on one of those terms halving with every quadrupling of the shot count.

In the next chapter we throw away the invented coefficients. Starting from the published STO-3G basis parameters and nothing else, we compute the Gaussian integrals for H₂ in NumPy, run a Hartree-Fock SCF to convergence, transform to molecular orbitals, map to four qubits with the Jordan-Wigner transformation, and run VQE against a full configuration interaction energy that the same code produces — every number in the chapter coming out of the run.

[← Chapter 2: From Molecules to Qubits](<chapter-2.html>) [Chapter 4: Hands-On: H2 from Scratch →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
