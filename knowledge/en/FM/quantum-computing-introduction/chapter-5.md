---
title: "Chapter 5: NISQ Reality and Outlook"
chapter_title: "Chapter 5: NISQ Reality and Outlook"
subtitle: ⚛️ Noise You Can Simulate, Mitigation You Can Measure, and an Assessment You Can Defend
reading_time: 40-45 minutes
difficulty: Advanced
code_examples: 5
exercises: 6
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-computing-introduction/chapter-5.html>) | Last sync: 2026-08-12

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to Quantum Computing](<index.html>) > Chapter 5

Every number in Chapters 1 through 4 came from a perfect quantum computer. The state vector stayed normalized, gates were exact, and expectation values were computed to machine precision. Real devices are none of these things, and the gap between the two is not a detail to be patched later — it is the single fact that determines what quantum computing can do for materials research today, next year, and in the decade after that.

This chapter closes the gap in three steps. First we build noise into the same state-vector simulator we have used all along, using the trajectory method, and validate it against exact density-matrix evolution. Then we measure how quickly a circuit's fidelity decays with depth at realistic error rates, and apply zero-noise extrapolation to the very VQE state computed in Chapter 4 to see how much of the noise-induced bias is recoverable and at what cost. Finally we put the resulting budgets — depth, width, measurements — next to what quantum error correction demands, and state as plainly as we can what near-term hardware can and cannot deliver.

That last section is the point of the chapter. It is deliberately unexciting. A researcher who reads it should come away able to look at a quantum computing claim, a vendor roadmap or a press release, and decide within a few minutes whether it bears on their own work. That skill is more useful than any algorithm in this series.

## Learning Objectives

After completing this chapter, you will be able to:

  * Describe decoherence quantitatively in terms of $T_1$, $T_2$ and gate error, and state the relation $1/T_2 = 1/(2T_1) + 1/T_\phi$
  * Write a quantum channel in Kraus form, and implement the same channel by the trajectory method on a pure-state simulator
  * Validate a trajectory noise model against exact density-matrix evolution, and predict the $1/\sqrt{N}$ convergence of the Monte Carlo average
  * Measure the fidelity-versus-depth decay of a noisy circuit, extract the decay rate, and relate it to the number of noise sites per layer
  * Apply zero-noise extrapolation to a noisy expectation value, quantify the bias reduction achieved, and explain the variance cost of the extrapolation weights
  * Explain the error-correction threshold, compute the surface-code distance needed for a target logical error rate, and state the physical-to-logical qubit overhead
  * Estimate the three independent budgets of a quantum calculation — circuit depth, qubit count, and measurement shots — and identify which one binds
  * Assess a quantum computing claim for materials science on principled grounds, without relying on device-specific announcements

* * *

## 5.1 The Physics of Noise

### What actually goes wrong

A qubit is a two-level subspace of a much larger physical system — a superconducting circuit, a trapped ion, a spin in silicon — and the environment does not respect the abstraction. Four failure modes account for nearly everything:

Mechanism | Physical origin | Timescale symbol | Effect on the state
---|---|---|---
Energy relaxation | Spontaneous emission into the environment | $T_1$ | $\lvert 1 \rangle \to \lvert 0 \rangle$; population decays
Dephasing | Fluctuating energy splitting (flux, charge, magnetic noise) | $T_\phi$ | Relative phase randomizes; coherence decays
Gate error | Imperfect calibration, pulse distortion, crosstalk | per-gate $p$ | A slightly wrong unitary is applied
Readout error | Finite measurement fidelity, discrimination overlap | per-shot | Measured bit differs from the true one

The two coherence times combine as

$$ \frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi} $$

so $T_2 \le 2T_1$ always. Relaxation destroys phase as a side effect; pure dephasing destroys phase without moving population. A device datasheet quoting $T_1$ and $T_2$ has told you $T_\phi$ too.

What matters for algorithms is not the times themselves but their ratio to gate duration. A two-qubit gate lasting $\tau_g$ on a device with coherence time $T$ has an error floor of order $\tau_g / T$, and a circuit of $N_g$ sequential gates accumulates roughly $N_g \tau_g / T$ worth of error. This is why "coherence time" and "gate fidelity" are two views of one number.

### Density matrices, minimally

A pure state $\lvert \psi \rangle$ cannot represent a system that has become correlated with its environment. The minimal extension is the **density matrix**

$$ \rho = \sum_k p_k \lvert \psi_k \rangle \langle \psi_k \rvert $$

with $\mathrm{Tr}\,\rho = 1$, $\rho = \rho^\dagger$, $\rho \succeq 0$. Expectation values become $\langle A \rangle = \mathrm{Tr}(\rho A)$, and the **purity** $\mathrm{Tr}(\rho^2)$ equals 1 for a pure state and $1/2^n$ for the maximally mixed state.

Noise processes are **quantum channels**, written in Kraus form:

$$ \rho \mapsto \mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger, \qquad \sum_k K_k^\dagger K_k = I $$

Three channels cover most of what we need.

**Depolarizing channel** — the workhorse model, in which an unspecified error occurs with probability $p$ and is equally likely to be $X$, $Y$ or $Z$:

$$ \mathcal{E}_{\text{dep}}(\rho) = (1-p)\rho + \frac{p}{3}\left(X\rho X + Y\rho Y + Z\rho Z\right) $$

**Amplitude damping** — $T_1$ relaxation, with $\gamma = 1 - e^{-t/T_1}$:

$$ K_0 = \begin{pmatrix} 1 & 0 \\\\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \qquad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\\\ 0 & 0 \end{pmatrix} $$

**Phase damping** — pure dephasing, with $\lambda$ set by $T_\phi$:

$$ K_0 = \begin{pmatrix} 1 & 0 \\\\ 0 & \sqrt{1-\lambda} \end{pmatrix}, \qquad K_1 = \begin{pmatrix} 0 & 0 \\\\ 0 & \sqrt{\lambda} \end{pmatrix} $$

### The trajectory method

Simulating a density matrix costs $4^n$ numbers instead of $2^n$, which halves the reachable qubit count. The **trajectory** (or quantum-jump, or Monte Carlo wavefunction) method avoids that: keep a *pure* state, apply a randomly chosen error at each noisy location, and average observables over many independent runs. In the limit of many trajectories the average reproduces the density matrix exactly, because

$$ \rho = \mathbb{E}\left[\lvert \psi_{\text{traj}} \rangle \langle \psi_{\text{traj}} \rvert\right] $$

is precisely the statement that the channel is a probabilistic mixture of pure-state maps. For the depolarizing channel the recipe is immediate: with probability $p$ apply a uniformly random Pauli. The cost is statistical error falling as $1/\sqrt{N_{\text{traj}}}$ — and, conveniently, it mirrors what a real device does, since a real device also gives you one sample at a time.

The subtlety is that not every channel's trajectory form is obvious. Amplitude damping requires a state-dependent jump probability and a renormalization of the no-jump branch. Phase damping needs care with the kick probability: a random $Z$ applied with probability $q$ multiplies the off-diagonal element by $(1-2q)$, while the Kraus channel multiplies it by $\sqrt{1-\lambda}$, so $q = (1 - \sqrt{1-\lambda})/2$ — not $\lambda/2$. Getting this wrong produces a plausible-looking decay with the wrong rate, so the first thing to do with any noise model is check it against the exact channel.

Code Example 1: Noise Channels, Trajectory Method vs Exact Density Matrix

```python
"""Chapter 5, Example 1: noise channels by the trajectory method, checked
against the exact density-matrix evolution."""
import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def bloch(rho):
    """Bloch vector (x, y, z) of a single-qubit density matrix."""
    return np.array([np.real(np.trace(rho @ P)) for P in (X, Y, Z)])


# ---------------------------------------------------------------------
# Exact channels, written with Kraus operators
# ---------------------------------------------------------------------

def kraus_apply(rho, kraus):
    return sum(K @ rho @ K.conj().T for K in kraus)


def depolarizing_kraus(p):
    """rho -> (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)."""
    return [np.sqrt(1 - p) * I2,
            np.sqrt(p / 3) * X, np.sqrt(p / 3) * Y, np.sqrt(p / 3) * Z]


def amplitude_damping_kraus(gamma):
    """Energy relaxation (T1): |1> decays to |0> with probability gamma."""
    return [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
            np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)]


def phase_damping_kraus(lam):
    """Pure dephasing: destroys coherence without moving population."""
    return [np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex),
            np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)]


# ---------------------------------------------------------------------
# Trajectory ("quantum jump") realisation: one pure state per shot
# ---------------------------------------------------------------------

def depolarizing_trajectory(psi, p, rng):
    """With probability p, apply one uniformly chosen Pauli error."""
    if rng.random() < p:
        return (X, Y, Z)[rng.integers(3)] @ psi
    return psi


def amplitude_damping_trajectory(psi, gamma, rng):
    """Jump |1> -> |0> with probability gamma |<1|psi>|^2; otherwise
    apply the no-jump operator and renormalize."""
    if rng.random() < gamma * abs(psi[1]) ** 2:
        return np.array([1.0 + 0j, 0.0 + 0j])
    out = np.array([psi[0], np.sqrt(1 - gamma) * psi[1]])
    return out / np.linalg.norm(out)


def phase_damping_trajectory(psi, lam, rng):
    """A random Z kick reproduces pure dephasing. A kick with probability q
    multiplies the off-diagonal element by (1 - 2q), while the Kraus channel
    multiplies it by sqrt(1 - lam), so q = (1 - sqrt(1 - lam)) / 2."""
    q = (1 - np.sqrt(1 - lam)) / 2
    if rng.random() < q:
        return Z @ psi
    return psi


def trajectory_average(psi0, step, arg, trials, seed):
    """Monte Carlo average of |psi><psi| over independent trajectories."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((2, 2), dtype=complex)
    for _ in range(trials):
        psi = step(psi0.copy(), arg, rng)
        acc += np.outer(psi, psi.conj())
    return acc / trials


# =====================================================================
np.set_printoptions(precision=5, suppress=True)
plus = H @ np.array([1.0 + 0j, 0.0 + 0j])          # |+>
trials = 200_000

print("Trajectory method vs exact density matrix, initial state |+>")
print("=" * 74)

for name, exact_kraus, traj_step, arg in (
        ("depolarizing, p = 0.15", depolarizing_kraus(0.15),
         depolarizing_trajectory, 0.15),
        ("amplitude damping, gamma = 0.30", amplitude_damping_kraus(0.30),
         amplitude_damping_trajectory, 0.30),
        ("phase damping, lambda = 0.40", phase_damping_kraus(0.40),
         phase_damping_trajectory, 0.40)):
    rho_exact = kraus_apply(np.outer(plus, plus.conj()), exact_kraus)
    rho_traj = trajectory_average(plus, traj_step, arg, trials, seed=7)
    print(f"\n  {name}")
    print(f"    exact Bloch vector      = {bloch(rho_exact)}")
    print(f"    trajectory Bloch vector = {bloch(rho_traj)}")
    print(f"    max |rho_exact - rho_traj| = "
          f"{np.abs(rho_exact - rho_traj).max():.5f}")
    print(f"    purity Tr(rho^2): exact "
          f"{np.real(np.trace(rho_exact @ rho_exact)):.5f}"
          f"   trajectory {np.real(np.trace(rho_traj @ rho_traj)):.5f}")

print()
print("Convergence of the trajectory average (depolarizing, p = 0.15)")
print("-" * 74)
print("  (mean over 8 independent runs; Monte Carlo error falls as 1/sqrt(N))")
rho_exact = kraus_apply(np.outer(plus, plus.conj()), depolarizing_kraus(0.15))
print(f"  {'trials':>10} {'mean max error':>16} {'1/sqrt(N)':>12}")
for n_tr in (100, 1_000, 10_000, 100_000):
    errs = [np.abs(trajectory_average(plus, depolarizing_trajectory,
                                      0.15, n_tr, seed=s) - rho_exact).max()
            for s in range(8)]
    print(f"  {n_tr:10,d} {np.mean(errs):16.6f} {1/np.sqrt(n_tr):12.6f}")

print()
print("Free decay: T1 and T2 as repeated weak channels")
print("-" * 74)
T1, T2 = 100.0, 60.0                      # microseconds, illustrative values
dt = 1.0
gamma = 1 - np.exp(-dt / T1)              # per-step relaxation probability
rate_phi = 1 / T2 - 1 / (2 * T1)          # 1/T2 = 1/(2 T1) + 1/T_phi
lam = 1 - np.exp(-2 * dt * rate_phi)
print(f"  T1 = {T1} us, T2 = {T2} us -> per-step gamma = {gamma:.5f},"
      f" lambda = {lam:.5f}")
print(f"  pure-dephasing time T_phi = {1/rate_phi:.2f} us")
print(f"  {'t (us)':>8} {'population':>13} {'exp(-t/T1)':>12} "
      f"{'coherence':>11} {'exp(-t/T2)':>12}")
rho_e = np.array([[0, 0], [0, 1]], dtype=complex)      # excited state |1>
rho_p = np.outer(plus, plus.conj())                    # superposition |+>
for step in range(0, 201):
    if step % 40 == 0:
        t = step * dt
        print(f"  {t:8.0f} {np.real(rho_e[1, 1]):13.6f} {np.exp(-t/T1):12.6f} "
              f"{2*abs(rho_p[0, 1]):11.6f} {np.exp(-t/T2):12.6f}")
    rho_e = kraus_apply(kraus_apply(rho_e, amplitude_damping_kraus(gamma)),
                        phase_damping_kraus(lam))
    rho_p = kraus_apply(kraus_apply(rho_p, amplitude_damping_kraus(gamma)),
                        phase_damping_kraus(lam))
```

```text
Trajectory method vs exact density matrix, initial state |+>
==========================================================================

  depolarizing, p = 0.15
    exact Bloch vector      = [0.8 0.  0. ]
    trajectory Bloch vector = [0.7999 0.     0.    ]
    max |rho_exact - rho_traj| = 0.00005
    purity Tr(rho^2): exact 0.82000   trajectory 0.81992

  amplitude damping, gamma = 0.30
    exact Bloch vector      = [0.83666 0.      0.3    ]
    trajectory Bloch vector = [0.83758 0.      0.29923]
    max |rho_exact - rho_traj| = 0.00046
    purity Tr(rho^2): exact 0.89500   trajectory 0.89554

  phase damping, lambda = 0.40
    exact Bloch vector      = [0.7746 0.     0.    ]
    trajectory Bloch vector = [0.77739 0.      0.     ]
    max |rho_exact - rho_traj| = 0.00140
    purity Tr(rho^2): exact 0.80000   trajectory 0.80217

Convergence of the trajectory average (depolarizing, p = 0.15)
--------------------------------------------------------------------------
  (mean over 8 independent runs; Monte Carlo error falls as 1/sqrt(N))
      trials   mean max error    1/sqrt(N)
         100         0.016250     0.100000
       1,000         0.006500     0.031623
      10,000         0.001962     0.010000
     100,000         0.000623     0.003162

Free decay: T1 and T2 as repeated weak channels
--------------------------------------------------------------------------
  T1 = 100.0 us, T2 = 60.0 us -> per-step gamma = 0.00995, lambda = 0.02306
  pure-dephasing time T_phi = 85.71 us
    t (us)    population   exp(-t/T1)   coherence   exp(-t/T2)
         0      1.000000     1.000000    1.000000     1.000000
        40      0.670320     0.670320    0.513417     0.513417
        80      0.449329     0.449329    0.263597     0.263597
       120      0.301194     0.301194    0.135335     0.135335
       160      0.201897     0.201897    0.069483     0.069483
       200      0.135335     0.135335    0.035674     0.035674
```

**What to notice.** The first block is the validation that licenses everything after it. For all three channels the trajectory average reproduces the exact density matrix to within the Monte Carlo error, and — importantly — it reproduces the **purity** as well. That is the nontrivial check: any wrong trajectory rule would still give a trace-1 matrix, but the purity is sensitive to how much genuine mixing occurred. The depolarizing channel on $\lvert + \rangle$ shrinks the Bloch vector from 1 to $1 - 4p/3 = 0.8$, which is exactly what the exact and trajectory columns both show.

The convergence table shows the price. The error falls from 0.016 at 100 trajectories to 0.0006 at 100,000 — a factor of 26 for a factor of 1000 more work, i.e. $1/\sqrt{N}$ as expected. There is no way around this: statistical sampling is what a real quantum computer does too, and Section 5.4 shows that the shot budget, not the qubit count, is what usually kills a proposed calculation.

The last block is the $T_1$/$T_2$ picture assembled from repeated weak channels. Because the per-step parameters were chosen as $\gamma = 1 - e^{-\Delta t/T_1}$ and $\lambda = 1 - e^{-2\Delta t/T_\phi}$, the discrete evolution reproduces $e^{-t/T_1}$ and $e^{-t/T_2}$ exactly at every printed time. With $T_1 = 100\\ \mu\text{s}$ and $T_2 = 60\\ \mu\text{s}$, the pure-dephasing time is $T_\phi = 85.7\\ \mu\text{s}$, and coherence is gone (down to 3.6%) after 200 $\mu\text{s}$. Compare that with the duration of a circuit: at a few hundred nanoseconds per two-qubit gate, 200 $\mu\text{s}$ buys a few hundred sequential gates. That single comparison is the whole NISQ constraint.

* * *

## 5.2 Simulating a Noisy Circuit

### Where the noise goes

We model a circuit as layers of gates and place one depolarizing kick after every gate, on every qubit that gate touched. A layer of the hardware-efficient ansatz from Chapters 3 and 4 has $n$ single-qubit rotations and $n-1$ CNOTs, so the number of **noise sites** per layer is

$$ N_{\text{sites}} = n + 2(n-1) $$

For $n = 4$ that is 10. Note the factor 2 on the CNOTs: the model places an independent kick on *each* qubit a two-qubit gate touches, so the effective two-qubit gate error in this model is $2p$, and a device whose quoted two-qubit error is $p_{2Q}$ corresponds to $p = p_{2Q}/2$ here. Throughout this chapter $p$ is a per-qubit-per-gate rate, and the "gate budget" it buys is counted in noisy gate *locations*, not in gates. This is a deliberately simple model — real devices have different single- and two-qubit error rates, correlated errors, crosstalk and leakage — but it captures the one feature that matters: error accumulates with the number of gate applications, and a circuit's usable depth is set by that accumulation.

The natural figure of merit is the **state fidelity** between the ideal and noisy states,

$$ F(d) = \mathbb{E}_{\text{traj}}\left[\left\lvert \langle \psi_{\text{ideal}}(d) \mid \psi_{\text{noisy}}(d) \rangle \right\rvert^2\right] $$

which starts at 1 and decays toward the fully depolarized value $1/2^n$.

Code Example 2: Fidelity Versus Circuit Depth

```python
"""Chapter 5, Example 2: fidelity vs circuit depth on a noisy state-vector simulator.

This block is the toolbox for the rest of the chapter: run it first, then
Examples 3 and 4 in the same session (or paste everything into one file).
"""
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
# Trajectory noise: one random Pauli kick per noisy gate location
# =====================================================================

def depol_kick(state, q, n, p, rng):
    """Depolarizing channel on qubit q, trajectory realisation."""
    if p and rng.random() < p:
        return apply_gate(state, (X, Y, Z)[rng.integers(3)], [q], n)
    return state


def noisy_layer(state, n, thetas, p, rng):
    """One hardware-efficient layer: Ry on every qubit, then a CNOT ladder.
    Each gate is followed by a depolarizing kick on every qubit it touched."""
    for q in range(n):
        state = apply_gate(state, ry(thetas[q]), [q], n)
        state = depol_kick(state, q, n, p, rng)
    for q in range(n - 1):
        state = cnot(state, q, q + 1, n)
        state = depol_kick(state, q, n, p, rng)
        state = depol_kick(state, q + 1, n, p, rng)
    return state


def noise_sites_per_layer(n):
    """n single-qubit gates + (n-1) two-qubit gates, each two-qubit gate
    contributing a kick on both of its qubits."""
    return n + 2 * (n - 1)


def fidelity_curve(n, max_depth, angles, p, trajectories, seed):
    """F(d) = E_traj |<psi_ideal(d) | psi_noisy(d)>|^2 for every depth d.

    One pass per trajectory records all depths, so the whole curve costs
    about as much as a single run of the deepest circuit.
    """
    ideal, st = [], ket('0' * n)
    for d in range(max_depth):
        st = noisy_layer(st, n, angles[d], 0.0, None)
        ideal.append(st.copy())

    rng = np.random.default_rng(seed)
    acc = np.zeros(max_depth)
    for _ in range(trajectories):
        st = ket('0' * n)
        for d in range(max_depth):
            st = noisy_layer(st, n, angles[d], p, rng)
            acc[d] += abs(np.vdot(ideal[d], st)) ** 2
    return acc / trajectories


# =====================================================================
n, max_depth, trajectories = 4, 24, 2000
angles = np.random.default_rng(2).uniform(0, 2 * np.pi, size=(max_depth, n))
sites = noise_sites_per_layer(n)
depths = np.arange(1, max_depth + 1)

print(f"n = {n} qubits, {sites} noise sites per layer, "
      f"{trajectories} trajectories per point")
print(f"fully depolarized floor 1/2^n = {1/2**n:.4f}")

for p in (0.001, 0.005, 0.01):
    F = fidelity_curve(n, max_depth, angles, p, trajectories, seed=17)
    gamma = -np.polyfit(depths, np.log(F), 1)[0]
    print(f"\nper-gate depolarizing probability p = {p}")
    print(f"  {'depth':>6} {'gates':>6} {'F (measured)':>13} "
          f"{'(1-p)^gates':>13} {'ratio':>7}")
    for i in range(0, max_depth, 2):
        n_sites = sites * depths[i]
        survive = (1 - p) ** n_sites
        print(f"  {depths[i]:6d} {n_sites:6d} {F[i]:13.4f} {survive:13.4f} "
              f"{F[i]/survive:7.3f}")
    print(f"  exponential fit F ~ exp(-gamma d): gamma = {gamma:.5f}")
    print(f"  per-layer survival exp(-gamma) = {np.exp(-gamma):.5f}")
    print(f"  depth where F = 0.5: {np.log(2)/gamma:.1f} layers"
          f"  ({np.log(2)/gamma*sites:.0f} noisy gates)")

print("\nHow deep can we go before the state is meaningless?")
print("-" * 70)
print(f"  {'p':>8} {'F=0.9 depth':>12} {'F=0.5 depth':>12} {'gate budget':>12}")
for p in (0.02, 0.01, 0.005, 0.002, 0.001, 0.0005):
    F = fidelity_curve(n, max_depth, angles, p, trajectories, seed=17)
    gamma = -np.polyfit(depths, np.log(F), 1)[0]
    d90, d50 = np.log(1 / 0.9) / gamma, np.log(2) / gamma
    print(f"  {p:8.4f} {d90:12.1f} {d50:12.1f} {d50*sites:12.0f}")
```

```text
n = 4 qubits, 10 noise sites per layer, 2000 trajectories per point
fully depolarized floor 1/2^n = 0.0625

per-gate depolarizing probability p = 0.001
   depth  gates  F (measured)   (1-p)^gates   ratio
       1     10        0.9927        0.9900   1.003
       3     30        0.9698        0.9704   0.999
       5     50        0.9513        0.9512   1.000
       7     70        0.9360        0.9324   1.004
       9     90        0.9219        0.9139   1.009
      11    110        0.9064        0.8958   1.012
      13    130        0.8856        0.8780   1.009
      15    150        0.8660        0.8606   1.006
      17    170        0.8495        0.8436   1.007
      19    190        0.8308        0.8269   1.005
      21    210        0.8115        0.8105   1.001
      23    230        0.7985        0.7944   1.005
  exponential fit F ~ exp(-gamma d): gamma = 0.00983
  per-layer survival exp(-gamma) = 0.99022
  depth where F = 0.5: 70.5 layers  (705 noisy gates)

per-gate depolarizing probability p = 0.005
   depth  gates  F (measured)   (1-p)^gates   ratio
       1     10        0.9581        0.9511   1.007
       3     30        0.8718        0.8604   1.013
       5     50        0.7897        0.7783   1.015
       7     70        0.7169        0.7041   1.018
       9     90        0.6598        0.6369   1.036
      11    110        0.5961        0.5762   1.035
      13    130        0.5461        0.5212   1.048
      15    150        0.4918        0.4715   1.043
      17    170        0.4460        0.4265   1.046
      19    190        0.4028        0.3858   1.044
      21    210        0.3720        0.3490   1.066
      23    230        0.3431        0.3157   1.087
  exponential fit F ~ exp(-gamma d): gamma = 0.04698
  per-layer survival exp(-gamma) = 0.95410
  depth where F = 0.5: 14.8 layers  (148 noisy gates)

per-gate depolarizing probability p = 0.01
   depth  gates  F (measured)   (1-p)^gates   ratio
       1     10        0.9253        0.9044   1.023
       3     30        0.7842        0.7397   1.060
       5     50        0.6433        0.6050   1.063
       7     70        0.5235        0.4948   1.058
       9     90        0.4401        0.4047   1.087
      11    110        0.3714        0.3310   1.122
      13    130        0.3124        0.2708   1.154
      15    150        0.2649        0.2215   1.196
      17    170        0.2305        0.1811   1.273
      19    190        0.2018        0.1481   1.362
      21    210        0.1738        0.1212   1.434
      23    230        0.1520        0.0991   1.534
  exponential fit F ~ exp(-gamma d): gamma = 0.08229
  per-layer survival exp(-gamma) = 0.92100
  depth where F = 0.5: 8.4 layers  (84 noisy gates)

How deep can we go before the state is meaningless?
----------------------------------------------------------------------
         p  F=0.9 depth  F=0.5 depth  gate budget
    0.0200          0.9          6.1           61
    0.0100          1.3          8.4           84
    0.0050          2.2         14.8          148
    0.0020          5.4         35.6          356
    0.0010         10.7         70.5          705
    0.0005         20.3        133.3         1333
```

**What to notice.** The decay is exponential in depth, and the rate is what a back-of-the-envelope argument predicts. The naive model — "the circuit works only if no error occurs anywhere" — gives survival probability $(1-p)^{N_{\text{gates}}}$, and the measured fidelity tracks it with a ratio between 1.00 and 1.09 at $p = 0.001$ and $p = 0.005$. The ratio exceeds 1 because some Pauli errors are harmless on the particular state that happens to be present, so the true fidelity is slightly better than "no error at all". At $p = 0.01$ the ratio climbs to 1.53 by depth 23, because the fidelity is approaching the $1/2^n = 0.0625$ floor and the naive model keeps falling past it.

The final table is the one to remember. It converts a per-gate error rate into a **gate budget** — the number of noisy gate applications you can afford before the state is half wrong:

Per-gate error | Usable gates at $F = 0.5$ | Usable gates at $F = 0.9$
---|---|---
$2 \times 10^{-2}$ | 61 | 9
$1 \times 10^{-2}$ | 84 | 13
$5 \times 10^{-3}$ | 148 | 22
$1 \times 10^{-3}$ | 705 | 107
$5 \times 10^{-4}$ | 1333 | 203

The budget scales as $1/p$, as it must. Note the second column: a *useful* calculation needs high fidelity, not 50%, and the $F = 0.9$ budget is roughly seven times smaller. Remember what $p$ means in this model: it is a per-qubit-per-gate rate, so $p = 10^{-3}$ describes a device whose *two-qubit* gate error is $2\times10^{-3}$, and such a device supports of order a hundred noisy gate locations at 90% fidelity. Chapter 4's Trotter analysis needed $3.4 \times 10^4$ Pauli rotations for $10^{-3}$ accuracy on a *four-qubit toy model*. Against a budget of 107, that is a factor of about 300 — two and a half orders of magnitude — and no amount of software cleverness closes two and a half orders of magnitude.

### Plotting the decay

The same data plotted linearly and logarithmically makes the two regimes visible at once: a pure exponential over most of the range, and the saturation floor at $1/2^n$.

Code Example 3: The Fidelity-Decay Curve

```python
"""Chapter 5, Example 3: the fidelity-decay curve, plotted.
Continues from Example 2 (same session)."""
import matplotlib.pyplot as plt

n, max_depth, trajectories = 4, 30, 1500
angles = np.random.default_rng(2).uniform(0, 2 * np.pi, size=(max_depth, n))
depths = np.arange(1, max_depth + 1)
sites = noise_sites_per_layer(n)
rates = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
summary = []
for p in rates:
    F = fidelity_curve(n, max_depth, angles, p, trajectories, seed=17)
    mask = F > 0.3            # fit only where the decay is still exponential
    gamma = -np.polyfit(depths[mask], np.log(F[mask]), 1)[0]
    summary.append((p, gamma, np.log(2) / gamma))
    ax1.plot(depths, F, 'o-', ms=3.5, lw=1.4, label=f'p = {p}')
    ax2.semilogy(depths, F, 'o-', ms=3.5, lw=1.4, label=f'p = {p}')

for ax in (ax1, ax2):
    ax.axhline(1 / 2 ** n, color='k', ls=':', lw=1.2)
    ax.axhline(0.5, color='gray', ls='--', lw=1.0)
    ax.set_xlabel('circuit depth (layers)')
    ax.set_ylabel('state fidelity $F(d)$')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
ax1.set_title(f'Fidelity vs depth, {n} qubits, {sites} noise sites per layer')
ax2.set_title('Same data, log scale: the decay is a pure exponential')
ax2.text(0.5, 1 / 2 ** n * 1.15, '$1/2^n$ floor', fontsize=8)
plt.tight_layout()
plt.show()

print(f"{'p':>8} {'gamma (per layer)':>19} {'F=0.5 depth':>13} "
      f"{'gamma/(sites*p)':>17}")
for p, gamma, d50 in summary:
    print(f"{p:8.4f} {gamma:19.5f} {d50:13.1f} {gamma/(sites*p):17.4f}")
print("\nThe last column is close to 1: the decay rate per layer is")
print("(noise sites per layer) x (error probability), with a prefactor")
print("slightly below 1 because some Pauli errors leave the state unchanged.")
```

```text
       p   gamma (per layer)   F=0.5 depth   gamma/(sites*p)
  0.0005             0.00524         132.4            1.0471
  0.0010             0.00990          70.0            0.9901
  0.0020             0.02000          34.7            1.0000
  0.0050             0.04599          15.1            0.9198
  0.0100             0.09297           7.5            0.9297
  0.0200             0.17561           3.9            0.8781

The last column is close to 1: the decay rate per layer is
(noise sites per layer) x (error probability), with a prefactor
slightly below 1 because some Pauli errors leave the state unchanged.
```

**What to notice.** The last column collapses six curves onto one number. Across a factor of 40 in error rate, $\gamma / (N_{\text{sites}} p)$ stays between 0.88 and 1.05, which says the decay rate per layer is simply

$$ \gamma \approx N_{\text{sites}}\, p \qquad \Longrightarrow \qquad F(d) \approx e^{-N_{\text{sites}} p \, d} $$

This is worth committing to memory, because it lets you estimate a circuit's fidelity without any simulation at all: count the gates, multiply by the error rate, exponentiate. A 40-qubit circuit of depth 100 has roughly $40 + 2\times39 = 118$ noise sites per layer, so $1.2 \times 10^4$ gate applications; at $p = 10^{-3}$ the fidelity is $e^{-12} \approx 6 \times 10^{-6}$. The circuit produces noise.

The log-scale panel shows why this simple rule works and where it fails: the decay is a straight line — a genuine exponential — until the fidelity approaches $1/2^n$, at which point the state is essentially the maximally mixed state and cannot get any worse.

* * *

## 5.3 Error Mitigation and Error Correction

Two entirely different responses to noise exist, and conflating them is a common source of confusion.

**Error mitigation** accepts the noise and corrects the *statistics*. It requires no extra qubits, works today, and reduces bias in expectation values — but it does not restore the quantum state, and its sampling cost grows rapidly with circuit size. It is a NISQ-era technique.

**Error correction** removes the noise from the *computation* by encoding one logical qubit in many physical qubits and continuously measuring syndromes. It restores arbitrary-depth computation, but demands physical error rates below a threshold and an overhead of hundreds to thousands of physical qubits per logical qubit. It is the fault-tolerant era.

### Zero-noise extrapolation

The most widely used mitigation technique is **zero-noise extrapolation** (ZNE). Deliberately increase the noise by a known factor $\lambda$, measure the observable at several $\lambda$, fit a curve, and extrapolate back to $\lambda = 0$:

$$ \langle A \rangle_{\lambda} \approx \langle A \rangle_0 + c_1 \lambda + c_2\lambda^2 + \cdots \quad\Longrightarrow\quad \langle A \rangle_0 \approx \sum_i w_i \langle A \rangle_{\lambda_i} $$

In practice $\lambda$ is scaled by stretching gate pulses or by inserting pairs of gates that cancel (unitary folding, $U \to U U^\dagger U$). Here we scale the error probability directly, which is the idealized version of the same idea.

The other common techniques, in one line each:

Technique | Idea | Extra qubits | Sampling overhead | What it fixes
---|---|---|---|---
Zero-noise extrapolation | Measure at amplified noise, extrapolate to zero | None | $\sim 10$ | Bias in expectation values
Probabilistic error cancellation | Sample from a quasi-probability inverse of the noise | None | $\sim 10^2\text{-}10^4$ | Bias, more rigorously
Readout-error correction | Invert the measured confusion matrix | None | $\sim 1$ | Measurement errors only
Symmetry verification | Discard shots violating particle number or spin | None | $\sim 1\text{-}10$ | Errors that break a symmetry
Dynamical decoupling | Pulse sequences that refocus dephasing during idles | None | $\sim 1$ | Idle-time dephasing
Purification / virtual distillation | Use $M$ copies to suppress incoherent error | $\times M$ | $\sim 10^2$ | Incoherent error, not coherent

Every entry in the "extra qubits" column is None or a small multiple, and every entry in the sampling column is a multiplier on an already-large shot budget. That is the essential trade: mitigation buys accuracy with samples.

Code Example 4: Zero-Noise Extrapolation on a Noisy VQE Energy

```python
"""Chapter 5, Example 4: zero-noise extrapolation of a noisy VQE energy.
Continues from Example 2 (same session)."""


def noisy_ansatz(theta, n, layers, p=0.0, rng=None):
    """The Chapter 3-4 hardware-efficient ansatz with depolarizing kicks
    after every gate. p = 0 reproduces the noiseless circuit exactly."""
    psi, k = ket('0' * n), 0
    for q in range(n):
        psi = apply_gate(psi, ry(theta[k]), [q], n)
        k += 1
        psi = depol_kick(psi, q, n, p, rng)
    for _ in range(layers):
        for q in range(n - 1):
            psi = cnot(psi, q, q + 1, n)
            psi = depol_kick(psi, q, n, p, rng)
            psi = depol_kick(psi, q + 1, n, p, rng)
        for q in range(n):
            psi = apply_gate(psi, ry(theta[k]), [q], n)
            k += 1
            psi = depol_kick(psi, q, n, p, rng)
    return psi


def tfim_hamiltonian(N, J, h):
    terms = {}
    for i in range(N - 1):
        s = 'I' * i + 'ZZ' + 'I' * (N - i - 2)
        terms[s] = terms.get(s, 0.0) - J
    for i in range(N):
        s = 'I' * i + 'X' + 'I' * (N - i - 1)
        terms[s] = terms.get(s, 0.0) - h
    return terms


def exact_ground_energy(terms):
    n = len(next(iter(terms)))
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for s, c in terms.items():
        A = np.array([[1.0 + 0j]])
        for ch in s:
            A = np.kron(A, PAULI[ch])
        M += c * A
    return float(np.linalg.eigvalsh(M)[0])


def energy(theta, terms, n, layers):
    psi = noisy_ansatz(theta, n, layers)
    return sum(expval(psi, s, terms) for s in terms)


def gradient(theta, terms, n, layers):
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += np.pi / 2
        tm[i] -= np.pi / 2
        g[i] = 0.5 * (energy(tp, terms, n, layers)
                      - energy(tm, terms, n, layers))
    return g


def noisy_energy(theta, terms, n, layers, p, trajectories, rng):
    """Trajectory-averaged <H> as a noisy device would report it."""
    tot = 0.0
    for _ in range(trajectories):
        psi = noisy_ansatz(theta, n, layers, p=p, rng=rng)
        tot += sum(expval(psi, s, terms) for s in terms)
    return tot / trajectories


n, layers = 4, 3
terms = tfim_hamiltonian(n, 1.0, 1.0)
E_exact = exact_ground_energy(terms)

# noiseless VQE first, so we know the target the noisy device should reproduce
theta = np.random.default_rng(1).normal(0.0, 0.3, size=n * (layers + 1))
for _ in range(800):
    theta -= 0.3 * gradient(theta, terms, n, layers)
E_clean = energy(theta, terms, n, layers)

print("Zero-noise extrapolation, 4-qubit transverse-field Ising chain")
print("=" * 74)
print(f"  exact ground state       E0 = {E_exact:+.6f}")
print(f"  noiseless VQE (3 layers) E  = {E_clean:+.6f}"
      f"   (ansatz error {E_clean - E_exact:+.2e})")
print(f"  circuit: {n*(layers+1)} Ry gates, {layers*(n-1)} CNOTs,"
      f" {n*(layers+1) + 2*layers*(n-1)} noise sites")

trajectories = 6000
for p0 in (0.002, 0.005):
    print(f"\n  base error rate p0 = {p0}"
          f"   ({trajectories} trajectories per noise scale)")
    lams = np.array([1.0, 2.0, 3.0])
    rng = np.random.default_rng(101)
    Es = []
    for lam in lams:
        E = noisy_energy(theta, terms, n, layers, p0 * lam, trajectories, rng)
        Es.append(E)
        print(f"    lambda = {lam:.0f}  (p = {p0*lam:.3f}):"
              f"  <H> = {E:+.6f}   bias = {E - E_clean:+.6f}")
    Es = np.array(Es)
    lin = np.polyval(np.polyfit(lams, Es, 1), 0.0)
    quad = np.polyval(np.polyfit(lams, Es, 2), 0.0)
    print(f"    unmitigated (lambda = 1) : {Es[0]:+.6f}"
          f"   residual bias {Es[0]-E_clean:+.6f}")
    print(f"    linear extrapolation     : {lin:+.6f}"
          f"   residual bias {lin-E_clean:+.6f}")
    print(f"    quadratic extrapolation  : {quad:+.6f}"
          f"   residual bias {quad-E_clean:+.6f}")
    print(f"    bias reduction (linear)  : "
          f"{abs(Es[0]-E_clean)/abs(lin-E_clean):.1f}x")
    # One run is one seed.  Replicate to separate genuine bias from
    # sampling fluctuation: the two fits differ in variance, not in bias.
    lin_b, quad_b = [], []
    for seed in range(201, 207):
        rng_s = np.random.default_rng(seed)
        Es_s = np.array([noisy_energy(theta, terms, n, layers, p0 * lam,
                                      trajectories, rng_s) for lam in lams])
        lin_b.append(np.polyval(np.polyfit(lams, Es_s, 1), 0.0) - E_clean)
        quad_b.append(np.polyval(np.polyfit(lams, Es_s, 2), 0.0) - E_clean)
    lin_b, quad_b = np.array(lin_b), np.array(quad_b)
    print("    the printed run is ONE seed; over 6 independent seeds:")
    print(f"      linear    residual bias = {lin_b.mean():+.4f}"
          f"  +/- {lin_b.std(ddof=1):.4f}")
    print(f"      quadratic residual bias = {quad_b.mean():+.4f}"
          f"  +/- {quad_b.std(ddof=1):.4f}")

print("\n  Why the bias has this sign: depolarizing noise pulls the state")
print("  towards the maximally mixed state, whose energy is Tr(H)/2^n = 0,")
print("  so a negative ground-state energy is systematically raised.")

print("\n  The cost of the mitigation")
print("  " + "-" * 66)
print("  Richardson extrapolation is E(0) = sum_i w_i E(lambda_i), and the")
print("  weights alternate in sign and grow with the number of noise scales:")
print(f"    {'noise scales':<24} {'weights w_i':<24} {'||w||_2':>8} {'sum|w_i|':>9}")
for lams_try in ([1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]):
    V = np.vander(np.array(lams_try), len(lams_try), increasing=True)
    w = np.linalg.inv(V)[0]
    print(f"    {str(lams_try):<24} {str(np.round(w, 2)):<24} "
          f"{np.linalg.norm(w):8.2f} {np.abs(w).sum():9.2f}")
print("  Those are the weights of an EXACT interpolation through m points, which\n"
      "  is what the quadratic fit above is for m = 3.  The linear fit above is a\n"
      "  least-squares line through the same three points, and its weights are\n"
      "  much gentler:")
w_lin = np.linalg.pinv(np.vander(lams, 2, increasing=True))[0]
print(f"    {str([float(x) for x in lams]):<24} {str(np.round(w_lin, 3)):<24} "
      f"{np.linalg.norm(w_lin):8.2f} {np.abs(w_lin).sum():9.2f}")
print("  With m noise scales the shot budget must grow by roughly ||w||^2 to")
print("  hold the statistical error fixed: mitigation buys bias with variance.")
```

```text
Zero-noise extrapolation, 4-qubit transverse-field Ising chain
==========================================================================
  exact ground state       E0 = -4.758770
  noiseless VQE (3 layers) E  = -4.749403   (ansatz error +9.37e-03)
  circuit: 16 Ry gates, 9 CNOTs, 34 noise sites

  base error rate p0 = 0.002   (6000 trajectories per noise scale)
    lambda = 1  (p = 0.002):  <H> = -4.569866   bias = +0.179537
    lambda = 2  (p = 0.004):  <H> = -4.406701   bias = +0.342703
    lambda = 3  (p = 0.006):  <H> = -4.237564   bias = +0.511840
    unmitigated (lambda = 1) : -4.569866   residual bias +0.179537
    linear extrapolation     : -4.737013   residual bias +0.012390
    quadratic extrapolation  : -4.727061   residual bias +0.022343
    bias reduction (linear)  : 14.5x
    the printed run is ONE seed; over 6 independent seeds:
      linear    residual bias = +0.0127  +/- 0.0098
      quadratic residual bias = +0.0124  +/- 0.0341

  base error rate p0 = 0.005   (6000 trajectories per noise scale)
    lambda = 1  (p = 0.005):  <H> = -4.301924   bias = +0.447479
    lambda = 2  (p = 0.010):  <H> = -3.950008   bias = +0.799395
    lambda = 3  (p = 0.015):  <H> = -3.569222   bias = +1.180181
    unmitigated (lambda = 1) : -4.301924   residual bias +0.447479
    linear extrapolation     : -4.673087   residual bias +0.076316
    quadratic extrapolation  : -4.624970   residual bias +0.124434
    bias reduction (linear)  : 5.9x
    the printed run is ONE seed; over 6 independent seeds:
      linear    residual bias = +0.0585  +/- 0.0124
      quadratic residual bias = +0.0282  +/- 0.0770

  Why the bias has this sign: depolarizing noise pulls the state
  towards the maximally mixed state, whose energy is Tr(H)/2^n = 0,
  so a negative ground-state energy is systematically raised.

  The cost of the mitigation
  ------------------------------------------------------------------
  Richardson extrapolation is E(0) = sum_i w_i E(lambda_i), and the
  weights alternate in sign and grow with the number of noise scales:
    noise scales             weights w_i               ||w||_2  sum|w_i|
    [1.0, 2.0]               [ 2. -1.]                    2.24      3.00
    [1.0, 2.0, 3.0]          [ 3. -3.  1.]                4.36      7.00
    [1.0, 2.0, 3.0, 4.0]     [ 4. -6.  4. -1.]            8.31     15.00
  Those are the weights of an EXACT interpolation through m points, which
  is what the quadratic fit above is for m = 3.  The linear fit above is a
  least-squares line through the same three points, and its weights are
  much gentler:
    [1.0, 2.0, 3.0]          [ 1.333  0.333 -0.667]       1.53      2.33
  With m noise scales the shot budget must grow by roughly ||w||^2 to
  hold the statistical error fixed: mitigation buys bias with variance.
```

**What to notice.** Four observations, in increasing order of importance.

**ZNE works, and by a useful factor.** At $p = 0.002$ the raw noisy energy is biased by $+0.180$ and linear extrapolation brings it to $+0.012$ — a 14.5-fold reduction. At $p = 0.005$ the reduction is 5.9-fold. The technique is real, not cosmetic.

**The bias is much larger than the ansatz error, and has the opposite significance.** The three-layer ansatz misses the exact ground state by $+9.4 \times 10^{-3}$. Noise at $p = 0.002$ adds a bias of $+0.18$, nineteen times larger. Every effort spent on ansatz design is wasted until the noise bias is brought below the ansatz error. This ordering — noise first, then algorithm — is the correct priority for near-term work, and it is frequently reversed in the literature.

**Higher-order extrapolation is not worse in bias — it is noisier, and one run cannot tell you which.** In the printed run the quadratic fit is worse than the linear one in both cases ($+0.022$ vs $+0.012$, and $+0.124$ vs $+0.076$), and it would be easy to draw the wrong conclusion from that. The wrong conclusion is that a quadratic "has a spare degree of freedom and spends it fitting noise": a quadratic through *three* points has zero spare degrees of freedom, it interpolates them exactly, so there is nothing to overfit. What it does instead is *amplify* the noise. The six-seed replication printed underneath makes the real behaviour visible: at $p_0 = 0.002$ the two are indistinguishable in mean bias ($+0.013$ linear, $+0.012$ quadratic), and at $p_0 = 0.005$ the quadratic is the *less* biased of the two ($+0.059$ against $+0.028$) — as it should be, since it cancels the $\lambda$ and $\lambda^2$ terms of the expansion rather than just the $\lambda$ term. But its scatter is three to six times larger ($\pm 0.034$ and $\pm 0.077$ against $\pm 0.010$ and $\pm 0.012$), which is exactly the variance penalty of the weights printed at the end of the block: exact interpolation through three points has $\lVert w \rVert^2 = 19$ against $2.33$ for the least-squares line, a factor of 8 in variance, $2.9$ in standard deviation. The single printed run is one seed on which the quadratic's larger scatter went the wrong way. The lesson is about variance, not about degrees of freedom — and about never drawing a methodological conclusion from one Monte Carlo run.

**The variance cost is explicit.** The Richardson weights are $(2, -1)$, $(3, -3, 1)$, $(4, -6, 4, -1)$ for two, three and four noise scales, with $\lVert w \rVert_2 = 2.24, 4.36, 8.31$. Since independent estimates with variance $\sigma^2$ combine to variance $\sigma^2 \lVert w \rVert^2$, going from two to four noise scales multiplies the required shot count by $(8.31/2.24)^2 \approx 14$. And this sits on top of the fact that a noisier circuit has a smaller signal, so each $\langle A \rangle_{\lambda_i}$ is itself harder to estimate. Mitigation converts a bias problem into a sampling problem, and Section 5.4 shows that the sampling problem was already the binding one.

### Error correction, and the threshold

Error correction encodes a logical qubit redundantly and measures **syndromes** — observables that reveal whether an error occurred without revealing the encoded state. The surface code is the leading candidate for superconducting hardware: physical qubits on a 2D lattice with nearest-neighbour parity checks, a code distance $d$ that can be increased by making the patch larger, and a decoder that infers the most likely error from the syndrome history.

The central fact is the **threshold theorem**. Below a critical physical error rate $p_{\text{th}}$, the logical error rate falls exponentially in the code distance:

$$ p_L \approx A\left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2} $$

Above threshold, adding qubits makes things *worse*, because each added qubit contributes more errors than the code can correct. The threshold for the surface code under standard circuit-level noise models is of order $10^{-2}$; the prefactor $A$ and the exact exponent depend on the code, the decoder and the noise model, so everything that follows is an order-of-magnitude statement and should be treated as one.

The rotated surface code uses $2d^2 - 1$ physical qubits per logical qubit.

Code Example 5: Correction, Depth and Measurement Budgets

```python
"""Chapter 5, Example 5: order-of-magnitude budgets for correction, depth
and measurement. Self-contained: only arithmetic, no simulator needed."""
import numpy as np

P_THRESHOLD = 1e-2      # representative surface-code threshold, order of magnitude
A_PREFACTOR = 0.1       # dimensionless prefactor, order of magnitude


def logical_error(p_phys, d, p_th=P_THRESHOLD, A=A_PREFACTOR):
    """Surface-code scaling p_L ~ A (p/p_th)^((d+1)/2).
    Order of magnitude only: the prefactor and the threshold are
    code-, decoder- and noise-model dependent."""
    return A * (p_phys / p_th) ** ((d + 1) / 2)


def required_distance(p_phys, target, p_th=P_THRESHOLD, A=A_PREFACTOR):
    """Smallest odd code distance reaching a target logical error rate.

    The comparison carries a relative tolerance because p_L is a ratio raised
    to a large power: a distance that meets the target *exactly* lands a few
    ulps above it in binary floating point (0.1 * 0.1**5 evaluates to
    1.0000000000000004e-06), and a bare `<= target` would reject it and return
    the next distance up."""
    if p_phys >= p_th:
        return None                # at or above threshold, more qubits do not help
    for d in range(3, 201, 2):
        if logical_error(p_phys, d, p_th, A) <= target * (1 + 1e-9):
            return d
    return None


def physical_per_logical(d):
    """Rotated surface code: 2 d^2 - 1 physical qubits per logical qubit."""
    return 2 * d * d - 1


print("A. Where the error-correction threshold bites")
print("=" * 74)
print(f"  assumed threshold p_th = {P_THRESHOLD:.0e}, prefactor A = {A_PREFACTOR}")
print(f"\n  {'p_phys':>9} {'d = 3':>10} {'d = 7':>10} {'d = 11':>10} "
      f"{'d = 21':>10} {'d = 31':>10}")
for p_phys in (2e-2, 1e-2, 5e-3, 1e-3, 3e-4, 1e-4):
    row = "  ".join(f"{logical_error(p_phys, d):10.2e}" for d in (3, 7, 11, 21, 31))
    print(f"  {p_phys:9.0e} {row}")
print("\n  Above threshold, increasing d makes the logical error WORSE.")
print("  Below threshold it falls exponentially in d. That is the whole game.")

print("\nB. Qubit overhead for a target logical error rate")
print("=" * 74)
print(f"  {'p_phys':>9} {'target p_L':>12} {'distance d':>11} "
      f"{'physical/logical':>17} {'100 logical qubits':>19}")
for p_phys in (1e-3, 3e-4, 1e-4):
    for target in (1e-6, 1e-10, 1e-15):
        d = required_distance(p_phys, target)
        if d is None:
            print(f"  {p_phys:9.0e} {target:12.0e} {'unreachable':>11}")
            continue
        per = physical_per_logical(d)
        print(f"  {p_phys:9.0e} {target:12.0e} {d:11d} {per:17,d} {100*per:19,d}")

print("\nC. Gate budget without error correction")
print("=" * 74)
print("  A circuit carries information only while (gates) x (error rate) << 1.")
print(f"  {'per-gate error':>15} {'gates at error 1':>18} "
      f"{'gates at error 0.1':>20}")
for p in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-10, 1e-12):
    print(f"  {p:15.0e} {1/p:18,.0f} {0.1/p:20,.0f}")

print("\n  Circuit sizes that materials problems actually ask for:")
for label, gates in (("2-site Hubbard, one Trotter step", 1e1),
                     ("2-site Hubbard, phase estimation to 1e-3", 3.4e7),
                     ("20-orbital active space, VQE ansatz", 1e4),
                     ("50-orbital active space, phase estimation", 1e11),
                     ("FeMoco-scale phase estimation, order of mag", 1e11)):
    print(f"    {label:45s}: ~{gates:8.0e} gates  -> needs p < {0.1/gates:.0e}")

print("\nD. Measurement cost of chemical accuracy")
print("=" * 74)
target = 1.6e-3          # Hartree; 1 kcal/mol, the usual 'chemical accuracy'
print(f"  target precision = {target:.1e} Ha (1 kcal/mol)")
print("  shots ~ (sum of term variances) / epsilon^2, variance ~ 1 per term.")
print("  This is the BEST case: it assumes the terms are perfectly grouped into")
print("  one commuting family.  Measuring each Pauli term in its own circuit")
print("  costs (sum_j |c_j| sigma_j)^2 / epsilon^2 instead, which is larger.")
print(f"\n  {'orbitals':>9} {'Pauli terms ~n^4':>17} {'shots (best case)':>18} "
      f"{'time at 1e4/s':>16}")
for n_orb in (4, 10, 20, 50, 100):
    n_terms = n_orb ** 4
    shots = n_terms / target ** 2
    seconds = shots / 1e4
    years = seconds / 3.156e7
    t = f"{years:.2e} yr" if years > 1 else f"{seconds:.2e} s"
    print(f"  {n_orb:9d} {n_terms:17,d} {shots:18.3e} {t:>16}")

print("\n  Precision is quadratically expensive:")
for eps in (1e-1, 1e-2, 1.6e-3, 1e-4):
    print(f"    epsilon = {eps:8.1e} Ha  ->  shots x {(1/eps)**2:12.3e} per term")

print("\nE. The three budgets side by side")
print("=" * 74)
print("  A NISQ calculation must satisfy all three at once:")
print("    (1) depth    : gates x error rate << 1")
print("    (2) width    : qubits <= device size, with no correction overhead")
print("    (3) sampling : shots x circuit time <= available wall-clock time")
print("\n  Worked case: 20-orbital active space (40 qubits), VQE")
n_orb, gates = 20, 1e4
n_terms = n_orb ** 4
shots = n_terms / (1.6e-3) ** 2
print(f"    qubits               : {2*n_orb}")
print(f"    Pauli terms          : {n_terms:,}")
print(f"    circuit gates        : {gates:.0e}  -> needs p < {0.1/gates:.0e}")
print(f"    shots for 1 kcal/mol : {shots:.2e}  (best case, perfect grouping)")
print(f"    at 1e4 circuits/s    : {shots/1e4/3.156e7:.2e} years"
      f" for ONE energy evaluation")
print(f"    a geometry optimization needs ~1e2 evaluations:"
      f" {1e2*shots/1e4/3.156e7:.2e} years")
```

```text
A. Where the error-correction threshold bites
==========================================================================
  assumed threshold p_th = 1e-02, prefactor A = 0.1

     p_phys      d = 3      d = 7     d = 11     d = 21     d = 31
      2e-02   4.00e-01    1.60e+00    6.40e+00    2.05e+02    6.55e+03
      1e-02   1.00e-01    1.00e-01    1.00e-01    1.00e-01    1.00e-01
      5e-03   2.50e-02    6.25e-03    1.56e-03    4.88e-05    1.53e-06
      1e-03   1.00e-03    1.00e-05    1.00e-07    1.00e-12    1.00e-17
      3e-04   9.00e-05    8.10e-08    7.29e-11    1.77e-18    4.30e-26
      1e-04   1.00e-05    1.00e-09    1.00e-13    1.00e-23    1.00e-33

  Above threshold, increasing d makes the logical error WORSE.
  Below threshold it falls exponentially in d. That is the whole game.

B. Qubit overhead for a target logical error rate
==========================================================================
     p_phys   target p_L  distance d  physical/logical  100 logical qubits
      1e-03        1e-06           9               161              16,100
      1e-03        1e-10          17               577              57,700
      1e-03        1e-15          27             1,457             145,700
      3e-04        1e-06           7                97               9,700
      3e-04        1e-10          11               241              24,100
      3e-04        1e-15          19               721              72,100
      1e-04        1e-06           5                49               4,900
      1e-04        1e-10           9               161              16,100
      1e-04        1e-15          13               337              33,700

C. Gate budget without error correction
==========================================================================
  A circuit carries information only while (gates) x (error rate) << 1.
   per-gate error   gates at error 1   gates at error 0.1
            1e-02                100                   10
            1e-03              1,000                  100
            1e-04             10,000                1,000
            1e-05            100,000               10,000
            1e-06          1,000,000              100,000
            1e-10     10,000,000,000        1,000,000,000
            1e-12  1,000,000,000,000      100,000,000,000

  Circuit sizes that materials problems actually ask for:
    2-site Hubbard, one Trotter step             : ~   1e+01 gates  -> needs p < 1e-02
    2-site Hubbard, phase estimation to 1e-3     : ~   3e+07 gates  -> needs p < 3e-09
    20-orbital active space, VQE ansatz          : ~   1e+04 gates  -> needs p < 1e-05
    50-orbital active space, phase estimation    : ~   1e+11 gates  -> needs p < 1e-12
    FeMoco-scale phase estimation, order of mag  : ~   1e+11 gates  -> needs p < 1e-12

D. Measurement cost of chemical accuracy
==========================================================================
  target precision = 1.6e-03 Ha (1 kcal/mol)
  shots ~ (sum of term variances) / epsilon^2, variance ~ 1 per term.
  This is the BEST case: it assumes the terms are perfectly grouped into
  one commuting family.  Measuring each Pauli term in its own circuit
  costs (sum_j |c_j| sigma_j)^2 / epsilon^2 instead, which is larger.

   orbitals  Pauli terms ~n^4  shots (best case)    time at 1e4/s
          4               256          1.000e+08       1.00e+04 s
         10            10,000          3.906e+09       3.91e+05 s
         20           160,000          6.250e+10       6.25e+06 s
         50         6,250,000          2.441e+12      7.74e+00 yr
        100       100,000,000          3.906e+13      1.24e+02 yr

  Precision is quadratically expensive:
    epsilon =  1.0e-01 Ha  ->  shots x    1.000e+02 per term
    epsilon =  1.0e-02 Ha  ->  shots x    1.000e+04 per term
    epsilon =  1.6e-03 Ha  ->  shots x    3.906e+05 per term
    epsilon =  1.0e-04 Ha  ->  shots x    1.000e+08 per term

E. The three budgets side by side
==========================================================================
  A NISQ calculation must satisfy all three at once:
    (1) depth    : gates x error rate << 1
    (2) width    : qubits <= device size, with no correction overhead
    (3) sampling : shots x circuit time <= available wall-clock time

  Worked case: 20-orbital active space (40 qubits), VQE
    qubits               : 40
    Pauli terms          : 160,000
    circuit gates        : 1e+04  -> needs p < 1e-05
    shots for 1 kcal/mol : 6.25e+10  (best case, perfect grouping)
    at 1e4 circuits/s    : 1.98e-01 years for ONE energy evaluation
    a geometry optimization needs ~1e2 evaluations: 1.98e+01 years
```

**What to notice.** Part A contains the whole logic of fault tolerance in one table. The $p = 10^{-2}$ row is flat at $0.1$: exactly at threshold, code distance does nothing. The $p = 2\times10^{-2}$ row *rises* with $d$, reaching 6550 at $d = 31$ — above threshold, a bigger code is a worse code. The $p = 10^{-4}$ row falls to $10^{-33}$ at $d = 31$. Being below threshold is not a quantitative improvement; it is a qualitative change of regime.

Part B prices it. At $p = 10^{-3}$, a logical error rate of $10^{-10}$ needs distance 17, which is 577 physical qubits per logical qubit — so a modest 100-logical-qubit machine needs 57,700 physical qubits. Improving the physical error rate to $3\times10^{-4}$ cuts that to 24,100. This is why hardware groups chase gate fidelity so hard: every factor of 3 in physical error rate saves roughly a factor of 3 in qubit count, compounding.

Part C is the sentence to quote when someone claims a near-term application. A circuit is meaningful while (gates) × (error rate) is well below 1. Phase estimation on the *two-site* Hubbard model needs $3.4 \times 10^7$ gates — Chapter 4, Example 3, on that chapter's own optimistic accounting — hence $p < 3\times10^{-9}$, six orders of magnitude beyond uncorrected hardware. A 50-orbital active space needs $p < 10^{-12}$, and FeMoco-scale estimates land in the same place: gate counts of order $10^{10}$ to $10^{11}$ need $p \lesssim 10^{-12}$ as well. Those numbers are reachable only with error correction, which is exactly why FeMoco is a fault-tolerance argument. Treat every exponent in that list as an order of magnitude; the published FeMoco estimates have already moved by several.

Part D is the constraint people forget. Even with a *perfect* noiseless quantum computer, and even granting perfect grouping of the Pauli terms into one commuting family — the best case, which the printed table assumes — a VQE on a 20-orbital active space needs $6\times10^{10}$ circuit executions to reach chemical accuracy, which at $10^4$ circuits per second is 2.4 months of continuous running for one energy. A geometry optimization needing a hundred energies takes twenty years. The measurement cost is a property of the *algorithm*, not of the hardware: it follows from $\varepsilon \propto 1/\sqrt{N}$ and the $O(M^4)$ term count. Better measurement strategies (grouping commuting terms, classical shadows, low-rank factorizations) reduce the prefactor substantially, but the $1/\varepsilon^2$ scaling is a law.

Part E puts the three together, and the conclusion is uncomfortable: even setting depth aside, VQE on a chemically interesting active space is not merely hard on today's hardware — it is hard on any hardware that estimates expectation values by sampling. This is one of the strongest arguments for phase estimation, whose precision cost is $1/\varepsilon$ rather than $1/\varepsilon^2$ — at a circuit depth that also grows as $1/\varepsilon$ — and therefore one of the strongest arguments for pursuing fault tolerance rather than optimizing NISQ algorithms.

* * *

## 5.4 A Sober Assessment

This section is the centre of the chapter. Everything above was measurement; this is judgement, and it is stated as plainly as possible.

### What NISQ devices can do for materials research today

  * **Benchmark quantum algorithms on models with known answers.** Running VQE on a four-qubit Ising or Hubbard model and comparing against exact diagonalization is genuinely useful: it validates software stacks, characterizes hardware, and trains people. It produces no new physics, and it should not be described as if it did.
  * **Characterize hardware physics.** Randomized benchmarking, cycle benchmarking, cross-entropy benchmarking and tomography are real measurements of real quantum systems, and they are how error rates improve. The device is the experiment.
  * **Develop and test error mitigation.** Techniques like ZNE and probabilistic error cancellation can only be validated where the true answer is independently known — that is, on small systems.
  * **Explore the classical-quantum interface.** Embedding schemes (DMET, DFT-embedding), active-space selection, measurement-reduction strategies and ansatz design are all classical research questions whose answers will be needed later, and which can be studied now.
  * **Analog simulation of lattice models.** This is the one area with a credible claim to results beyond classical reach: cold-atom and trapped-ion platforms have measured dynamics of quantum spin and Hubbard models at sizes where exact classical simulation is infeasible. It is not gate-based quantum computing, and the results are physics rather than chemistry, but it is real.

### What NISQ devices cannot do

  * **Beat DFT on any real material.** DFT handles hundreds to thousands of atoms with useful accuracy. A NISQ device handles a handful of orbitals with noise-limited accuracy. There is no overlap in which the quantum device wins.
  * **Reach chemical accuracy on a chemically interesting system.** Part D above: the shot count alone forbids it, before noise is even considered.
  * **Run phase estimation on anything.** Depth requirements exceed uncorrected hardware by five to ten orders of magnitude.
  * **Handle realistic system sizes.** A catalytic active space of 50 orbitals is 100 qubits with $6\times10^6$ Pauli terms. Neither the depth, the shot budget, nor the classical optimization is within reach.
  * **Deliver a verified quantum advantage in chemistry or materials.** As of this writing there is no calculation of a molecular or materials property, performed on quantum hardware, that is simultaneously (a) more accurate than the best classical method, (b) independently verified, and (c) scientifically useful. Claims to the contrary have repeatedly been matched or exceeded by improved classical methods within months.

### How to read a quantum advantage claim

The pattern of the last several years is consistent: a quantum experiment claims to have performed a task beyond classical reach, and within months a classical algorithm — often a tensor-network method exploiting the specific structure of the sampled circuit — reproduces the result. This is not scandal; it is how the field establishes where the boundary actually lies. But it means claims must be read carefully.

Questions to ask, in order:

  1. **Is the task useful, or constructed?** Random-circuit sampling and boson sampling are designed to be hard for classical computers and are not useful for anything else. Demonstrating them is a legitimate physics milestone and tells you nothing about chemistry.
  2. **What is the classical baseline, and who computed it?** A comparison against a naive classical algorithm is not a comparison. Ask whether the best known classical method was used, whether it was given comparable engineering effort, and whether the authors of the classical baseline agree with the framing.
  3. **Was the quantum result verified?** If the answer cannot be checked classically, how is correctness established? Extrapolation from smaller verifiable instances is the usual approach, and it is an assumption, not a proof.
  4. **What accuracy was achieved?** A quantum energy accurate to 0.1 Hartree is not a chemistry result; chemical accuracy is $1.6\times10^{-3}$.
  5. **What was mitigated, and at what cost?** Heavy post-processing can produce a number close to the right answer while the underlying quantum state has negligible fidelity. Ask for the raw result and the shot count.
  6. **Does the method scale?** Many demonstrations rely on symmetry, small size or problem-specific tricks that vanish at larger scale. Ask what the resource count is at twice the size.

### Principled criteria, not announcements

Device announcements age badly, and any assessment tied to a qubit count is obsolete when it is published. A better approach is to ask three quantitative questions about *any* proposed quantum calculation, all of which can be answered from the physics of the problem:

Question | Quantity to compute | Threshold for plausibility
---|---|---
Is the circuit shallow enough? | (gate count) × (per-gate error) | Well below 1, ideally below 0.1
Is the sampling affordable? | (Pauli terms) / $\varepsilon^2$ × (circuit time), best case | Below available wall-clock time
Is it classically hard? | Best classical method's cost and accuracy | Classical method must fail, not merely be slow

If a proposal fails any of the three, no hardware improvement of the kind announced in a press release will rescue it; what is needed is a change of algorithm or a change of era. Conversely, a proposal that passes all three deserves serious attention regardless of who is making it.

### What would change the picture

To be clear about what progress looks like, here are the developments that would genuinely alter the assessment above:

  * **Physical two-qubit error rates below $10^{-4}$ at scale**, which brings distance-9 surface codes and $10^{-10}$ logical error into reach at 161 physical qubits per logical qubit.
  * **A demonstration of a logical qubit with error rate below its constituent physical qubits, sustained over many rounds** — the transition from "error correction as an experiment" to "error correction as infrastructure".
  * **Algorithms with $1/\varepsilon$ rather than $1/\varepsilon^2$ precision scaling that run at NISQ depth**, which would remove the measurement bottleneck. This entry is different from the others: it is constrained by a theorem, not merely unsolved. Interpolating schemes ($\alpha$-QPE, robust and maximum-likelihood amplitude estimation) do exist and do buy part of the speedup, achieving total query cost $\sim \varepsilon^{-(1+\alpha)}$ at maximum coherent depth $\sim \varepsilon^{-(1-\alpha)}$ for $\alpha \in [0,1]$. But the tradeoff is the content of the result: at a *fixed* maximum depth $D$ the precision obtainable from $N$ total shots is bounded by $\varepsilon \gtrsim 1/(D\sqrt{N})$, so the full Heisenberg scaling $1/\varepsilon$ requires depth $\propto 1/\varepsilon$. Precision is bought with coherence, and coherence is exactly what NISQ lacks — which makes this a restatement of the case for error correction rather than an independent hope.
  * **A verified quantum calculation of a materials property that a classical method cannot reproduce**, on a problem someone outside the quantum computing community cares about.
  * **Reduction of fault-tolerant chemistry resource estimates by another three to four orders of magnitude**, continuing the trend of the past decade.

The first two are engineering problems with clear paths. The third may be impossible. The fourth is the actual goal. The fifth is where theorists can contribute most.

* * *

## 5.5 The Ecosystem

You will not write your own simulator for production work. Three open-source frameworks dominate, and they differ more in philosophy than in capability. We describe their positioning rather than their APIs, which change between versions.

Framework | Origin | Philosophy | Strongest for
---|---|---|---
Qiskit | IBM | Circuit-centric, hardware-oriented, large ecosystem | Running on IBM hardware, transpilation, error mitigation modules
Cirq | Google | Explicit control of gate scheduling and device topology | Hardware-aware circuit construction, NISQ experiments
PennyLane | Xanadu | Differentiable programming, autodiff integration | Variational algorithms, quantum machine learning, hybrid gradients

Around them sit specialized tools worth knowing about by category: quantum chemistry interfaces that produce fermionic Hamiltonians and apply qubit mappings (the role played by Example 2 of Chapter 4); high-performance state-vector and tensor-network simulators, which are what you should compare against before claiming hardware is needed; and error-mitigation libraries implementing ZNE and probabilistic error cancellation.

Two practical recommendations. First, **learn one framework properly rather than three superficially**; the concepts transfer, the APIs do not. Second, **always run the classical simulator first**. If a 30-qubit state-vector simulation answers your question, hardware adds noise and nothing else.

### References and further reading

Category | Suggested entry points
---|---
Textbooks | Nielsen & Chuang, *Quantum Computation and Quantum Information* (the standard reference); Preskill's lecture notes on quantum computation (freely available)
NISQ framing | Preskill, "Quantum Computing in the NISQ era and beyond" (2018) — the paper that named the era and stated its limits
Quantum chemistry on quantum computers | Cao et al., *Chemical Reviews* review of quantum chemistry in the age of quantum computing; McArdle et al., *Reviews of Modern Physics* review of quantum computational chemistry
Variational algorithms | Cerezo et al., *Nature Reviews Physics* review of variational quantum algorithms; the barren-plateau literature starting from McClean et al. (2018)
Error correction | Fowler et al., "Surface codes: towards practical large-scale quantum computation"; Terhal's review of quantum error correction for memories
Error mitigation | Cai et al., review of quantum error mitigation; the Mitiq software paper for implementations
Classical competition | Schollwöck's DMRG review; the literature on classical simulation of quantum supremacy experiments, which is where the boundary is actually being drawn
Resource estimation | The successive FeMoco resource-estimate papers, read in chronological order — the best available education in what dominates fault-tolerant cost

* * *

## 5.6 Series Wrap-Up and Learning Roadmap

### What this series covered

Chapter | Content | What you can now do
---|---|---
1 | Qubits, superposition, measurement, tensor products | Represent and sample from a multi-qubit state; explain why $2^n$ is both resource and curse
2 | Gates, circuits, entanglement, universality, the simulator | Apply arbitrary unitaries to arbitrary qubits; quantify entanglement; compile Pauli exponentials
3 | Variational quantum eigensolver | Build an ansatz, measure a Pauli-decomposed observable, run a full VQE with parameter-shift gradients
4 | Second quantization, Jordan-Wigner, model Hamiltonians | Map a fermionic problem onto qubits and verify it; diagonalize Ising and Hubbard models exactly; compare VQE against exact answers
5 | Noise, mitigation, correction, assessment | Simulate noisy circuits; measure fidelity decay; apply ZNE; budget depth, width and shots; evaluate a claim

The mini-simulator you built in Chapters 1-2 carried every subsequent calculation. That is the point of building it: ninety-nine lines of NumPy are enough to reproduce, verify and understand every quantum algorithm in this series, and anything you cannot reproduce that way you probably do not understand yet.

### Where to go next

Three routes, depending on what you want.

**If you are a materials researcher who wants to keep an eye on the field.** You are done with the essentials. The highest-value follow-up is *classical* methods for strongly correlated systems, because that is what any quantum result must beat: DMRG and matrix product states, quantum Monte Carlo and the sign problem, dynamical mean-field theory, and embedding schemes. Read the annual reviews rather than the preprints, and apply the three criteria of Section 5.4 to anything that looks exciting.

**If you want to do quantum algorithm research.** Deepen the theory: quantum phase estimation and its modern descendants (qubitization, quantum signal processing); Hamiltonian simulation beyond Trotter; the barren-plateau literature and what it says about trainability; measurement-reduction strategies including classical shadows; and quantum error correction proper. The prerequisites in this Dojo — [Linear Algebra and Tensors](<../linear-algebra-tensor/index.html>), [Introduction to Quantum Mechanics](<../quantum-mechanics/index.html>), [Introduction to Quantum Field Theory](<../quantum-field-theory-introduction/index.html>) — are where the mathematics lives.

**If you want to build things.** Pick one framework, implement a VQE on a molecule of your choosing from integrals you generate yourself, run it on real hardware, and compare against your own exact diagonalization. Then implement ZNE and measure how much it helps. The gap between the simulator result and the hardware result, measured by you on a problem you chose, teaches more than any review article.

### A recommended sequence

Stage | Focus | Rough effort
---|---|---
1 | Reproduce every code example in this series from scratch, without looking | 2-3 weeks
2 | Extend the simulator: density matrices, a second noise model, a better optimizer | 2-3 weeks
3 | One classical strong-correlation method (DMRG on a spin chain is ideal) | 1-2 months
4 | One framework, one real molecule, one hardware run | 1-2 months
5 | Read the fault-tolerant resource-estimation literature chronologically | ongoing

Stage 3 is the one people skip and should not. Understanding why DMRG solves 1D problems essentially exactly, and why it fails in 2D, is the single best preparation for judging where quantum computing can contribute.

### A closing note

Quantum computing for materials science is, right now, a field with excellent physics, real engineering progress, a clear long-term target, and no near-term applications. All four of those statements are true simultaneously, and holding them together is the mark of someone who understands the field rather than either its marketing or its dismissal.

The useful posture is neither enthusiasm nor scepticism but *literacy*: the ability to compute the three budgets, identify the classical baseline, and reach your own conclusion. If you can do that — and after this chapter you can — you will be able to evaluate this field's claims for the rest of your career, including the ones that turn out to be true.

* * *

## Exercises

Work through these with the code from this chapter in front of you. Solutions follow each question.

#### Exercise 1: Coherence Times

A device reports $T_1 = 80\\ \mu\text{s}$ and $T_2 = 120\\ \mu\text{s}$. (a) Is this report internally consistent? State the bound that applies, and say why it is easy to misremember. (b) For $T_1 = 80\\ \mu\text{s}$ and $T_2 = 40\\ \mu\text{s}$, find $T_\phi$. (c) If a two-qubit gate takes 300 ns, roughly how many sequential gates fit within $T_2$, and how does that compare with the gate budget of Section 5.2?

<details><summary>Solution</summary>
<p>(a) Yes, it is consistent — and that is the point of the question. The bound is \(T_2 \le 2T_1\), which follows from \(1/T_2 = 1/(2T_1) + 1/T_\phi\) with \(T_\phi &gt; 0\). Here \(2T_1 = 160\ \mu\mathrm{s}\) and \(T_2 = 120\ \mu\mathrm{s} &lt; 160\ \mu\mathrm{s}\), so nothing is wrong. The trap is misremembering the bound as \(T_2 \le T_1\): \(T_2 &gt; T_1\) is perfectly physical and is what a relaxation-limited device with very little pure dephasing looks like. Here \(1/T_\phi = 1/120 - 1/160\), i.e. \(T_\phi = 480\ \mu\mathrm{s}\), four times \(T_1\) — dephasing is nearly absent.</p>
<p>(b) \(1/T_\phi = 1/T_2 - 1/(2T_1) = 1/40 - 1/160 = 0.025 - 0.00625 = 0.01875\ \mu\mathrm{s}^{-1}\), so \(T_\phi = 53.3\ \mu\mathrm{s}\). Dephasing dominates.</p>
<p>(c) \(40\ \mu\mathrm{s} / 300\ \mathrm{ns} \approx 133\) gates fit inside \(T_2\). Fitting inside \(T_2\) is not the same as being usable, and the comparison has to be made against the right row of Section 5.2. This device's coherence-limited per-gate error is \(\tau_g/T_2 = 300\ \mathrm{ns}/40\ \mu\mathrm{s} = 7.5\times10^{-3}\), which sits between the \(p = 5\times10^{-3}\) row (148 locations at \(F = 0.5\), 22 at \(F = 0.9\)) and the \(p = 10^{-2}\) row (84 and 13) — interpolating, about 92 at \(F = 0.5\) and 14 at \(F = 0.9\). So the \(T_2\) count and the fidelity budget agree at the 50% level (133 against ~92), while the <em>usable</em> depth is ten times smaller than either. Quoting the \(p = 10^{-3}\) row (705 and 107) for this device would be wrong by an order of magnitude: that row describes a device with ten times better gates.</p>
</details>

#### Exercise 2: A Wrong Trajectory Rule

In Code Example 1, replace `phase_damping_trajectory` with a version that applies $Z$ with probability $\lambda/2$ instead of $(1-\sqrt{1-\lambda})/2$. (a) What does the trajectory Bloch vector become for $\lambda = 0.4$? (b) Which quantity in the printed output reveals the error most clearly? (c) Why is this bug particularly dangerous?

<details><summary>Solution</summary>
<p>(a) A \(Z\) kick with probability \(q\) multiplies the off-diagonal element by \((1-2q)\). With \(q = \lambda/2 = 0.2\) the coherence becomes \(1 - 0.4 = 0.6\), so the Bloch vector is \((0.60, 0, 0)\) instead of the correct \(\sqrt{1-\lambda} = 0.7746\).</p>
<p>(b) The purity. The correct channel gives \(\mathrm{Tr}(\rho^2) = 0.800\); the wrong one gives \(0.68\). The Bloch vector also differs, but purity is the sharper diagnostic because it is quadratic in the state and therefore doubly sensitive to over-mixing.</p>
<p>(c) Because the wrong model still produces exponential decay with a plausible rate. Every qualitative feature survives — coherence decays, populations are untouched, the channel is trace-preserving — and only the numerical rate is wrong — but wrong by a factor of exactly 2, for every \(\lambda\). The correct rule multiplies the coherence by \(\sqrt{1-\lambda}\) per application, the wrong one by \((1-\lambda)\), and \(\ln(1-\lambda) = 2\ln\sqrt{1-\lambda}\) identically, so the extracted dephasing rate is exactly twice too large regardless of \(\lambda\). (The \(\sim\)23% figure is the error in the single-application coherence, 0.60 against 0.7746; the <em>rate</em> is off by 100%.) A factor of two propagates straight into any \(T_2\) read off a decay curve, or any fidelity estimate. This is why Code Example 1 exists: a noise model must be validated against the exact channel before it is used for anything.</p>
</details>

#### Exercise 3: Predicting a Fidelity Without Simulating

Using the rule $F \approx \exp(-N_{\text{sites}} p\, d)$ from Code Example 3: (a) estimate the fidelity of a 10-qubit, 20-layer hardware-efficient circuit at $p = 2\times10^{-3}$; (b) how deep a circuit can 10 qubits support at $F = 0.9$? (c) At what $n$ does the $1/2^n$ floor stop mattering for the estimate?

<details><summary>Solution</summary>
<p>(a) \(N_{\mathrm{sites}} = n + 2(n-1) = 10 + 18 = 28\) per layer. Total noise sites \(= 28 \times 20 = 560\). \(F \approx e^{-560 \times 2\times10^{-3}} = e^{-1.12} = 0.33\). Roughly a third of the amplitude survives — already marginal.</p>
<p>(b) \(F = 0.9\) needs \(N_{\mathrm{sites}} p\, d = \ln(1/0.9) = 0.105\), so \(d = 0.105/(28 \times 2\times10^{-3}) = 1.9\) layers. Two layers. This is the practical meaning of "NISQ": a ten-qubit device at \(2\times10^{-3}\) error supports a two-layer circuit at useful fidelity.</p>
<p>(c) The floor matters when \(e^{-N_{\mathrm{sites}} p d}\) approaches \(2^{-n}\), i.e. when \(N_{\mathrm{sites}} p d \gtrsim n \ln 2\). For \(n = 4\) that is \(2.8\), and Code Example 2 indeed shows the ratio departing from 1 near that point at \(p = 0.01\). For larger \(n\) the floor is exponentially lower, so the simple exponential rule holds over a much wider range — the floor becomes irrelevant, and the honest reading is that large noisy circuits are not saturating at a floor, they are simply useless.</p>
</details>

#### Exercise 4: When Is Mitigation Worth It?

From Code Example 4: at $p_0 = 0.002$ linear ZNE reduced the bias 14.5-fold using three noise scales. (a) By what factor did the shot budget have to grow? (b) Suppose you had instead spent the same total shots on the unmitigated circuit. What would the statistical error have been, and would that have been a better trade? (c) Under what circumstances is ZNE clearly not worth it?

<details><summary>Solution</summary>
<p>(a) Three noise scales means three separate expectation-value estimates, so 3× the shots even before accounting for the extrapolation weights. Then use the weights of the estimator actually used: the linear extrapolation is a <em>least-squares line</em> through \(\lambda = 1, 2, 3\), whose intercept weights are \((4/3, 1/3, -2/3)\) with \(\lVert w \rVert^2 = 2.33\) — the last row printed by Code Example 4. Holding the statistical error of the extrapolated value fixed therefore costs another factor of 2.33 per point: about 7× in total. (The Richardson weights \((3, -3, 1)\) with \(\lVert w \rVert^2 = 19\) apply to the <em>quadratic</em> extrapolation, which is an exact interpolation through the three points; that one would cost about 57×.)</p>
<p>(b) Spending 7× the shots on the unmitigated circuit reduces its <em>statistical</em> error by \(\sqrt{7} = 2.6\), but does nothing to its <em>bias</em> of \(+0.180\). Bias does not average away. So the trade is worth it precisely when the bias exceeds the statistical error, which is the usual situation for a shallow circuit with many shots. Mitigation attacks the error that sampling cannot.</p>
<p>(c) Three cases. (i) When the statistical error already dominates the bias — then more shots on the raw circuit are better. (ii) When the noise is so strong that the extrapolation is unreliable: at \(p_0 = 0.005\) the residual bias was \(+0.076\), still 8× the ansatz error, and the fit quality degrades as the \(\lambda = 3\) point approaches the depolarized floor. (iii) When the observable's bias is not smooth in \(\lambda\), which happens with coherent (non-depolarizing) errors — ZNE assumes an analytic dependence on noise strength that coherent errors need not satisfy.</p>
</details>

#### Exercise 5: Error-Correction Arithmetic

Using Code Example 5: (a) at $p = 5\times10^{-3}$, what code distance reaches $p_L = 10^{-9}$, and what is the qubit overhead? (b) A useful algorithm needs $10^{12}$ logical gates. What logical error rate does it require, and what physical error rate and distance would supply it? (c) Why does the assumed prefactor $A$ matter less than the ratio $p/p_{\text{th}}$?

<details><summary>Solution</summary>
<p>(a) From part A of the output, \(p = 5\times10^{-3}\) gives \(p_L = 1.53\times10^{-6}\) at \(d = 31\). Reaching \(10^{-9}\) needs \((p/p_\mathrm{th})^{(d+1)/2} = 10^{-8}\) with \(p/p_\mathrm{th} = 0.5\), i.e. \((d+1)/2 = 8/\log_{10}2 = 26.6\), so \(d = 53\) and \(2d^2 - 1 = 5{,}617\) physical qubits per logical qubit. At half the threshold, error correction technically works and is ruinously expensive. Operating within a factor of 2 of threshold is not a viable engineering point.</p>
<p>(b) \(10^{12}\) logical gates need \(p_L \lesssim 10^{-13}\) for the whole computation to have a reasonable chance of being correct (more conservatively \(10^{-15}\)). From part B, \(p_L = 10^{-15}\) needs \(d = 27\) at \(p = 10^{-3}\) (1,457 physical per logical), \(d = 19\) at \(3\times10^{-4}\) (721), or \(d = 13\) at \(10^{-4}\) (337).</p>
<p>(c) Because \(A\) enters linearly while \(p/p_\mathrm{th}\) enters to the power \((d+1)/2\). Changing \(A\) from 0.1 to 0.01 shifts the required distance by about 2; changing \(p/p_\mathrm{th}\) from 0.1 to 0.03 changes it by a factor. This is why all realistic estimates emphasize the physical error rate relative to threshold and treat prefactors as noise — and why every result in Example 5 is presented as an order of magnitude.</p>
</details>

#### Exercise 6: Assess a Claim

A preprint reports: "Using a 127-qubit superconducting processor with zero-noise extrapolation, we compute the ground-state energy of a 20-site Heisenberg chain to within 2% of the exact value, a calculation intractable for classical computers." Apply the six questions of Section 5.4.

<details><summary>Solution</summary>
<p><strong>1. Useful or constructed?</strong> A 20-site Heisenberg chain is a well-studied model with no unknown physics. The calculation is a benchmark, not a discovery. Legitimate as such.</p>
<p><strong>2. Classical baseline?</strong> The claim of classical intractability is false. A 20-site spin-1/2 chain is \(2^{20} = 10^6\) dimensional — exact diagonalization runs on a laptop in seconds, and DMRG handles hundreds of sites to near machine precision. The word "intractable" appears to refer to brute-force full-state simulation of the <em>circuit</em>, which is a different claim and not a relevant one.</p>
<p><strong>3. Verified?</strong> Yes, implicitly — the authors compare against the exact value, which is why they can quote 2%. Good practice, and it simultaneously refutes the intractability claim.</p>
<p><strong>4. Accuracy?</strong> 2% of a ground-state energy is far from chemical accuracy and far from what DMRG delivers (\(10^{-10}\) relative or better for a 1D chain). For extracting physics — critical exponents, correlation functions — 2% on the energy is not usable.</p>
<p><strong>5. What was mitigated?</strong> ZNE, so the raw fidelity should be requested. If the unmitigated result was 30% off and the post-processing brought it to 2%, the quantum state had little to do with the final number. Also ask for the shot count and the noise scales used.</p>
<p><strong>6. Does it scale?</strong> A Heisenberg chain maps to qubits without Jordan-Wigner strings and has \(O(N)\) local terms — the easiest possible case. It says nothing about a fermionic Hamiltonian with \(O(M^4)\) terms and non-local strings.</p>
<p><strong>Verdict.</strong> A respectable hardware benchmark with an unsupportable framing. The correct summary would be: "a 127-qubit processor with error mitigation reproduces a classically exact result to 2%, demonstrating improved device performance." That is genuine progress and worth publishing; the intractability claim is not.</p>
</details>

* * *

## Summary

### Key Takeaways

**1. Noise has a small number of causes and a simple model**

  * $T_1$ (relaxation), $T_\phi$ (dephasing), gate error and readout error account for nearly everything, with $1/T_2 = 1/(2T_1) + 1/T_\phi$ and $T_2 \le 2T_1$.
  * Channels are Kraus maps; the depolarizing channel is the standard workhorse.
  * The trajectory method reproduces a channel exactly on a pure-state simulator, at $1/\sqrt{N}$ statistical cost — and it mirrors what real hardware gives you.
  * Validate any noise model against the exact channel, and check purity, not just the Bloch vector. Our phase-damping rule needed $q = (1-\sqrt{1-\lambda})/2$, not $\lambda/2$.

**2. Fidelity decays exponentially, and the rate is predictable without simulation**

  * $F(d) \approx \exp(-N_{\text{sites}}\, p\, d)$, with $N_{\text{sites}} = n + 2(n-1)$ per layer; measured $\gamma/(N_{\text{sites}}p) = 0.88\text{-}1.05$ over a factor of 40 in $p$.
  * Gate budget at $F = 0.5$: 84 noisy gate locations at $p = 10^{-2}$, 705 at $10^{-3}$, 1333 at $5\times10^{-4}$. At $F = 0.9$, roughly seven times fewer. Each CNOT costs two of those locations, so the corresponding two-qubit gate error is $2p$.
  * The decay saturates at $1/2^n$, which for large $n$ means the exponential rule holds all the way to uselessness.

**3. Mitigation reduces bias and pays in variance**

  * Linear ZNE cut the noise bias 14.5-fold at $p_0 = 0.002$ and 5.9-fold at $p_0 = 0.005$.
  * The noise bias ($+0.18$) was 19 times the ansatz error ($+0.0094$): fix noise before refining ansätze.
  * Quadratic extrapolation was worse than linear *in the printed run only*: over six seeds it is equally or less biased but three to six times noisier, because exact interpolation through three points has $\lVert w \rVert^2 = 19$ against $2.33$ for the least-squares line.
  * Richardson weights $(2,-1)$, $(3,-3,1)$, $(4,-6,4,-1)$ give $\lVert w \rVert = 2.24, 4.36, 8.31$; the shot budget scales as $\lVert w \rVert^2$.

**4. Error correction is a change of regime, not an improvement**

  * $p_L \approx A(p/p_{\text{th}})^{(d+1)/2}$: above threshold, larger codes are worse; below it, logical error falls exponentially in $d$.
  * At $p = 10^{-3}$, reaching $p_L = 10^{-10}$ needs $d = 17$, i.e. 577 physical qubits per logical qubit — 57,700 for a 100-logical-qubit machine.
  * Operating at half the threshold "works" and costs 5,617 physical qubits per logical qubit. Margin below threshold is everything.

**5. Three budgets must be satisfied at once, and depth is not always the binding one**

  * Depth: phase estimation on the *two-site* Hubbard model needs $p < 3\times10^{-9}$; FeMoco-scale estimates ($10^{10}$–$10^{11}$ gates) need $p \lesssim 10^{-12}$.
  * Width: qubit count is the cheapest resource, and error correction multiplies it by $10^2\text{-}10^3$.
  * Sampling: $6\times10^{10}$ shots (best case, perfect grouping) for chemical accuracy on a 20-orbital active space — 2.4 months per energy at $10^4$ circuits/s, twenty years for a geometry optimization, *on perfect hardware*.
  * The $1/\varepsilon^2$ measurement scaling is a property of the algorithm, not the device. Escaping it costs coherent depth — the tradeoff is a theorem, not an open problem — which is why it is the strongest argument for phase estimation and therefore for fault tolerance.

**6. The honest assessment**

  * NISQ devices can benchmark algorithms, characterize hardware, develop mitigation, and support analog lattice simulation.
  * They cannot beat DFT on any real material, reach chemical accuracy on a chemically interesting system, run phase estimation, or deliver a verified advantage in chemistry or materials.
  * Read claims with six questions: is the task useful, what is the classical baseline, was it verified, what accuracy, what was mitigated at what cost, does it scale.
  * Judge on principle — compute the three budgets and identify the classical baseline — not on device announcements, which age within months.

**Practical implications**

  * Simulate classically first. If a 30-qubit state-vector run answers your question, hardware adds only noise.
  * Estimate the three budgets before writing any circuit; the answer is usually "not yet", and knowing why is the valuable part.
  * When reporting hardware results, give the raw and mitigated numbers, the shot count, and the classical baseline computed with real effort.
  * Learn one framework properly, and learn one classical strong-correlation method (DMRG is the best choice) — that is what any quantum result must beat.
  * Hold two ideas at once: this field has excellent physics and a clear long-term target, and it has no near-term applications for materials research. Both are true, and saying so is literacy rather than pessimism.

### Where This Leads

You have reached the end of the series. You built a quantum simulator from an empty file, used it to run a variational eigensolver, mapped a fermionic Hamiltonian onto qubits and verified the mapping, diagonalized model Hamiltonians whose physics you can now read off the numbers, added noise and watched fidelity decay, mitigated that noise and measured the cost, and priced the resources that a real calculation would need.

What remains is to apply it. The next quantum computing claim you encounter — in a seminar, a proposal, a funding call or a press release — is one you can now evaluate quantitatively in a few minutes. Do that, and keep doing it, and you will be among the people who can tell the difference when something genuinely changes. For the classical methods that any quantum result must beat, continue with [Computational Statistical Mechanics](<../computational-statistical-mechanics/index.html>) and the strong-correlation literature; for the theory beneath the algorithms, continue with [Introduction to Quantum Field Theory](<../quantum-field-theory-introduction/index.html>).

[← Chapter 4: Quantum Computing for Chemistry and Materials](<chapter-4.html>) [Back to Series Index →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * Error rates, thresholds, coherence times and resource estimates quoted here are illustrative order-of-magnitude values chosen for teaching; they are not device specifications and must be verified against primary sources before use in any assessment or proposal.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
