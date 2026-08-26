---
title: "Chapter 2: Qubits, Superposition, and Entanglement"
chapter_title: "Chapter 2: Qubits, Superposition, and Entanglement"
subtitle: "The State Vector, the Born Rule, and the Resource That Has No Classical Analogue"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/xsxMSFJY4OA"
    title="Quantum Computing Ch.2: Qubits, Superposition, and Entanglement"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-computing-introduction/chapter-2.html>) | Last sync: 2026-08-16

[Quantum Computing Dojo](<../index.html>) > [Introduction to Quantum Computing](<index.html>) > Chapter 2

In Chapter 1 we argued that quantum computing rests on interference rather than on brute-force parallelism. To make that concrete we need the mathematical objects themselves. This chapter introduces the qubit, the rule that connects its description to measurement outcomes, and entanglement — the correlation that has no classical counterpart. Every claim here is backed by a short NumPy program you can run yourself.

## 2.1 The Qubit as a Two-Level System

A classical **bit** takes one of two values, 0 or 1. A **qubit** (quantum bit) is the quantum version of that object: any physical system with two distinguishable states that we can prepare, manipulate, and measure. The spin of an electron, the polarization of a photon, and two energy levels of a superconducting circuit all serve equally well. What matters is the mathematics, not the hardware.

### 📚 The Computational Basis

We label the two distinguishable states using **ket notation**, written \\(|\cdot\rangle\\), which is simply a compact way of writing a column vector:

\\[ |0\rangle = \begin{pmatrix} 1 \\\ 0 \end{pmatrix}, \qquad |1\rangle = \begin{pmatrix} 0 \\\ 1 \end{pmatrix} \\]

Together they form the **computational basis**. These two states are what a measurement can distinguish perfectly.

### Superposition

The essential departure from classical bits is that a qubit may occupy any **superposition** of the two basis states:

\\[ |\psi\rangle = \alpha |0\rangle + \beta |1\rangle = \begin{pmatrix} \alpha \\\ \beta \end{pmatrix} \\]

Here \\(\alpha\\) and \\(\beta\\) are complex numbers called **probability amplitudes**. They are not probabilities. Probabilities are non-negative real numbers; amplitudes carry a sign and a phase, and that is precisely what allows them to cancel.

The amplitudes must satisfy the **normalization condition**:

\\[ |\alpha|^2 + |\beta|^2 = 1 \\]

This is not an arbitrary convention. It is the statement that the qubit yields *some* outcome when measured, with total probability one.

Two superpositions appear so often that they have their own names:

\\[ |+\rangle = \frac{1}{\sqrt{2}}\left(|0\rangle + |1\rangle\right), \qquad |-\rangle = \frac{1}{\sqrt{2}}\left(|0\rangle - |1\rangle\right) \\]

Both give outcome 0 and outcome 1 with equal probability \\(1/2\\), yet they are physically different states, distinguishable by an appropriate measurement. The difference lies entirely in the **relative phase**, the minus sign in front of \\(|1\rangle\\). Keep this example in mind: it is the smallest possible demonstration that a superposition is not the same thing as ignorance about a hidden classical value.

### Global Phase Carries No Information

If we multiply an entire state by a phase factor, \\(|\psi\rangle \to e^{i\gamma}|\psi\rangle\\), every measurement probability is unchanged, for any measurement whatsoever. The two vectors describe the **same physical state**. Only *relative* phases between components have physical meaning. This is why the parametrization in the next section can describe every qubit state with just two real angles instead of four real numbers.

## 2.2 The Bloch Sphere

Normalization and the irrelevance of global phase together reduce the four real parameters in \\((\alpha, \beta)\\) to two. Every qubit state can therefore be written as

\\[ |\psi\rangle = \cos\frac{\theta}{2} |0\rangle + e^{i\phi} \sin\frac{\theta}{2} |1\rangle \\]

with the **polar angle** \\(0 \le \theta \le \pi\\) and the **azimuthal angle** \\(0 \le \phi < 2\pi\\). These are exactly the spherical coordinates of a point on a unit sphere, called the **Bloch sphere**.

| Point on the sphere | \\(\theta\\) | \\(\phi\\) | State |
|---|---|---|---|
| North pole | \\(0\\) | undefined | \\(\lvert 0 \rangle\\) |
| South pole | \\(\pi\\) | undefined | \\(\lvert 1 \rangle\\) |
| Equator, \\(+x\\) | \\(\pi/2\\) | \\(0\\) | \\(\lvert + \rangle\\) |
| Equator, \\(-x\\) | \\(\pi/2\\) | \\(\pi\\) | \\(\lvert - \rangle\\) |
| Equator, \\(+y\\) | \\(\pi/2\\) | \\(\pi/2\\) | \\((\lvert 0 \rangle + i \lvert 1 \rangle)/\sqrt{2}\\) |

Notice the half-angle \\(\theta/2\\). It is there because \\(|0\rangle\\) and \\(|1\rangle\\) are orthogonal as vectors yet sit at opposite poles, \\(180\\) degrees apart on the sphere. The Bloch sphere is a picture of the *state space*, not of ordinary physical space.

The Bloch sphere is a wonderfully useful picture, and it comes with one warning: it works for a **single** qubit only. There is no comparable picture for two or more qubits, and the reason for that failure is entanglement, which we reach in Section 2.5.

## 2.3 Measurement and the Born Rule

Everything above concerns the description of a qubit. Measurement is the bridge from that description to what an experiment actually records.

### 📚 The Born Rule

Measuring the state \\(|\psi\rangle = \alpha|0\rangle + \beta|1\rangle\\) in the computational basis yields outcome 0 or outcome 1, with probabilities

\\[ P(0) = |\alpha|^2, \qquad P(1) = |\beta|^2 \\]

This is the **Born rule**, named after Max Born, who proposed the probabilistic interpretation of the wave function in 1926. It is a postulate of quantum mechanics, not a theorem derived from the rest of the formalism.

Three consequences deserve to be stated plainly.

  1. **The outcome is random.** Even with complete knowledge of \\(\alpha\\) and \\(\beta\\), you cannot predict which outcome occurs on a single run.
  2. **The state collapses.** After a measurement returning 0, the qubit is left in \\(|0\rangle\\). Measuring again returns 0 with certainty. The original superposition is gone.
  3. **You cannot learn \\(\alpha\\) and \\(\beta\\) from one copy.** Recovering the amplitudes requires many identically prepared copies, measured in several different bases. This procedure is called **quantum state tomography**, and its cost grows quickly with the number of qubits.

Point 2 is the reason quantum algorithms end with a carefully designed final step. You get one bit string per run, so the interference must already have concentrated the probability on the answer you want before you look.

### 💻 Code Example 1: A Qubit State and Its Measurement Statistics

Let us build a state on the Bloch sphere, verify normalization, and sample measurements. We use only NumPy, and we use `np.random.default_rng` with a fixed seed so the output is reproducible.

```python
import numpy as np

# Computational basis states |0> and |1>
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)

# Build |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
theta = np.pi / 3      # polar angle on the Bloch sphere
phi = np.pi / 4        # azimuthal angle
psi = np.cos(theta / 2) * ket0 + np.exp(1j * phi) * np.sin(theta / 2) * ket1

print("state vector =", np.round(psi, 4))
print("norm squared =", round(float(np.vdot(psi, psi).real), 10))

# Born rule: probability of each outcome
p0 = float(abs(psi[0]) ** 2)
p1 = float(abs(psi[1]) ** 2)
print(f"P(0) = {p0:.4f},  P(1) = {p1:.4f},  sum = {p0 + p1:.4f}")

# Simulate 10,000 measurements of identically prepared qubits
rng = np.random.default_rng(seed=42)
shots = 10000
outcomes = rng.choice([0, 1], size=shots, p=[p0, p1])
counts = np.bincount(outcomes, minlength=2)
print(f"measured 0: {counts[0]:5d}  (frequency {counts[0]/shots:.4f})")
print(f"measured 1: {counts[1]:5d}  (frequency {counts[1]/shots:.4f})")
```

Verified output:

```
state vector = [0.866 +0.j     0.3536+0.3536j]
norm squared = 1.0
P(0) = 0.7500,  P(1) = 0.2500,  sum = 1.0000
measured 0:  7558  (frequency 0.7558)
measured 1:  2442  (frequency 0.2442)
```

With \\(\theta = \pi/3\\) we predict \\(P(0) = \cos^2(\pi/6) = 3/4\\) and \\(P(1) = 1/4\\), and the sampled frequencies 0.7558 and 0.2442 sit right where statistical fluctuation of 10,000 shots would put them. Note also that `np.vdot` conjugates its first argument, which is exactly the inner product convention quantum mechanics uses.

## 2.4 Multi-Qubit States and Tensor Products

A single qubit is not yet a computer. To combine qubits we use the **tensor product**, written \\(\otimes\\).

If qubit A is in state \\(|\psi_A\rangle\\) and qubit B is in state \\(|\psi_B\rangle\\), the joint state of the pair is

\\[ |\psi_A\rangle \otimes |\psi_B\rangle \\]

For basis states we abbreviate \\(|0\rangle \otimes |1\rangle\\) as \\(|01\rangle\\). Two qubits therefore have four basis states, \\(|00\rangle, |01\rangle, |10\rangle, |11\rangle\\), and a general two-qubit state is

\\[ |\psi\rangle = c_{00}|00\rangle + c_{01}|01\rangle + c_{10}|10\rangle + c_{11}|11\rangle, \qquad \sum_{ij} |c_{ij}|^2 = 1 \\]

In general \\(n\\) qubits require \\(2^n\\) complex amplitudes. This is the exponential growth we met in Chapter 1, now written explicitly. In NumPy the tensor product of vectors is exactly the Kronecker product, `np.kron`.

## 2.5 Entanglement and the Bell States

Here is the question that separates quantum from classical information: **can every two-qubit state be written as a product of two single-qubit states?**

The answer is no, and the states that cannot are called **entangled**.

### 📚 The Bell States

The four **Bell states**, named after John Stewart Bell, are the standard maximally entangled two-qubit states:

\\[ |\Phi^{\pm}\rangle = \frac{1}{\sqrt{2}}\left(|00\rangle \pm |11\rangle\right), \qquad |\Psi^{\pm}\rangle = \frac{1}{\sqrt{2}}\left(|01\rangle \pm |10\rangle\right) \\]

Take \\(|\Phi^{+}\rangle\\). Measuring both qubits gives 00 half the time and 11 half the time, and **never** 01 or 10. The two results are perfectly correlated. Yet each qubit examined on its own behaves like a fair coin. The correlation lives in the pair, not in either member.

### 💻 Code Example 2: Constructing a Bell State with `np.kron`

```python
import numpy as np

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)

# Two-qubit basis states are tensor (Kronecker) products
ket00 = np.kron(ket0, ket0)
ket11 = np.kron(ket1, ket1)
print("|00> =", ket00.real.astype(int))
print("|11> =", ket11.real.astype(int))

# The Bell state |Phi+> = (|00> + |11>)/sqrt(2)
bell = (ket00 + ket11) / np.sqrt(2)
labels = ["00", "01", "10", "11"]
print("Bell state amplitudes:")
for label, amp in zip(labels, bell):
    print(f"  |{label}> : {amp.real:+.4f}")

# Born rule over four outcomes
probs = np.abs(bell) ** 2
print("probabilities:", {l: round(float(p), 4) for l, p in zip(labels, probs)})

# Sample 10,000 measurements of both qubits
rng = np.random.default_rng(seed=7)
shots = 10000
draws = rng.choice(4, size=shots, p=probs)
counts = np.bincount(draws, minlength=4)
for label, c in zip(labels, counts):
    print(f"  outcome {label}: {c:5d}  (frequency {c/shots:.4f})")

# Marginal statistics of qubit A alone
p_a0 = (counts[0] + counts[1]) / shots
print(f"qubit A alone: P(0) = {p_a0:.4f} -> looks like a fair coin")
```

Verified output:

```
|00> = [1 0 0 0]
|11> = [0 0 0 1]
Bell state amplitudes:
  |00> : +0.7071
  |01> : +0.0000
  |10> : +0.0000
  |11> : +0.7071
probabilities: {'00': 0.5, '01': 0.0, '10': 0.0, '11': 0.5}
  outcome 00:  4983  (frequency 0.4983)
  outcome 01:     0  (frequency 0.0000)
  outcome 10:     0  (frequency 0.0000)
  outcome 11:  5017  (frequency 0.5017)
qubit A alone: P(0) = 0.4983 -> looks like a fair coin
```

The mixed outcomes 01 and 10 never appear, while each qubit on its own is unbiased. That combination is the signature of entanglement.

### Why an Entangled State Cannot Be Factored

Let us prove it for \\(|\Phi^{+}\rangle\\). Suppose it could be written as a product,

\\[ (a_0|0\rangle + a_1|1\rangle) \otimes (b_0|0\rangle + b_1|1\rangle) = a_0 b_0 |00\rangle + a_0 b_1 |01\rangle + a_1 b_0 |10\rangle + a_1 b_1 |11\rangle \\]

Matching coefficients with \\(|\Phi^{+}\rangle\\) requires

\\[ a_0 b_0 = \tfrac{1}{\sqrt{2}}, \quad a_1 b_1 = \tfrac{1}{\sqrt{2}}, \quad a_0 b_1 = 0, \quad a_1 b_0 = 0 \\]

The first equation forces \\(a_0 \neq 0\\) and \\(b_0 \neq 0\\); the second forces \\(a_1 \neq 0\\) and \\(b_1 \neq 0\\). But then \\(a_0 b_1 \neq 0\\), contradicting the third equation. No such \\(a\\) and \\(b\\) exist, so \\(|\Phi^{+}\rangle\\) is entangled.

This argument generalizes neatly. Arrange the amplitudes into a matrix \\(C\\) with \\(C_{ij} = c_{ij}\\). A product state gives \\(C_{ij} = a_i b_j\\), which is a rank-one matrix, so \\(\det C = 0\\). Therefore **the two-qubit state is a product state if and only if \\(a_0 b_1 - a_1 b_0\\) vanishes**, that is, if and only if

\\[ c_{00} c_{11} - c_{01} c_{10} = 0 \\]

### 💻 Code Example 3: Testing Separability Numerically

We can also detect entanglement by ignoring one qubit. The **reduced density matrix** \\(\rho_A\\) describes what is left when qubit B is discarded, and its **purity** \\(\mathrm{Tr}(\rho_A^2)\\) equals 1 for a product state and drops to \\(1/2\\) for a maximally entangled pair of qubits.

```python
import numpy as np


def is_product_state(state):
    """A 2-qubit state is a product state iff its 2x2 amplitude matrix
    C[i, j] = amplitude of |ij> has rank 1, i.e. det(C) = 0."""
    C = state.reshape(2, 2)
    return abs(np.linalg.det(C)) < 1e-12


def reduced_density_matrix_A(state):
    """Trace out qubit B: rho_A = Tr_B |psi><psi|."""
    C = state.reshape(2, 2)
    return C @ C.conj().T


ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
plus = (ket0 + ket1) / np.sqrt(2)

product = np.kron(plus, plus)                 # |+>|+>, separable
bell = (np.kron(ket0, ket0) + np.kron(ket1, ket1)) / np.sqrt(2)

for name, state in [("|+>|+>", product), ("Bell |Phi+>", bell)]:
    C = state.reshape(2, 2)
    det = np.linalg.det(C)
    rho_a = reduced_density_matrix_A(state)
    purity = float(np.trace(rho_a @ rho_a).real)
    eigs = np.linalg.eigvalsh(rho_a)
    print(f"{name}")
    print(f"  det(C)          = {det.real:+.4f}")
    print(f"  product state?  = {is_product_state(state)}")
    print(f"  eigenvalues of rho_A = {np.round(eigs.real, 4)}")
    print(f"  purity Tr(rho_A^2)   = {purity:.4f}")
```

Verified output:

```
|+>|+>
  det(C)          = +0.0000
  product state?  = True
  eigenvalues of rho_A = [0. 1.]
  purity Tr(rho_A^2)   = 1.0000
Bell |Phi+>
  det(C)          = +0.5000
  product state?  = False
  eigenvalues of rho_A = [0.5 0.5]
  purity Tr(rho_A^2)   = 0.5000
```

Both tests agree. The separable state \\(|+\rangle|+\rangle\\) has a vanishing determinant and a pure single-qubit description; the Bell state has \\(\det C = 1/2\\) and leaves qubit A in a state that is completely undetermined on its own, with both eigenvalues equal to \\(1/2\\). The full pair is known exactly while neither half is known at all — a situation with no classical analogue.

### ⚠️ Entanglement Does Not Send Signals

Entangled qubits are often described as "communicating instantly." They do not. If you hold one half of a Bell pair, your measurement statistics are those of a fair coin no matter what your partner does to the other half, as Code Example 2 shows numerically. The correlations only become visible when the two parties **compare their records over an ordinary classical channel**, which travels no faster than light. This result is known as the **no-signalling theorem**. Entanglement is a genuine resource — it powers quantum teleportation and superdense coding — but it is not a faster-than-light telephone.

## 2.6 The No-Cloning Theorem

One more property shapes everything about quantum computing.

> **No-cloning theorem** (Wootters and Zurek, and independently Dieks, 1982): there is no physical process that copies an arbitrary unknown quantum state.

### Proof Sketch

Suppose a universal copier existed. It would be a single operation \\(U\\), the same for every input, satisfying

\\[ U\left(|\psi\rangle \otimes |0\rangle\right) = |\psi\rangle \otimes |\psi\rangle \\]

for **every** state \\(|\psi\rangle\\), where the second slot is a blank register.

Quantum time evolution is **unitary**, and unitary operations preserve inner products. Apply the copier to two arbitrary states \\(|\psi\rangle\\) and \\(|\varphi\rangle\\) and compare inner products before and after:

\\[ \langle \psi | \varphi \rangle = \left(\langle \psi | \varphi \rangle\right)^2 \\]

Writing \\(x = \langle \psi | \varphi \rangle\\), we need \\(x = x^2\\), so \\(x = 0\\) or \\(x = 1\\). That is, a copier can only work for states that are either **orthogonal** (perfectly distinguishable, like \\(|0\rangle\\) and \\(|1\rangle\\)) or **identical**. An arbitrary unknown state, for instance \\(|+\rangle\\) alongside \\(|0\rangle\\), cannot be copied. This completes the argument.

### Why It Matters

  * **Classical error correction is unavailable.** The classical trick of storing three copies and taking a majority vote is forbidden outright. Quantum error correction had to be invented differently, spreading one logical qubit across many physical qubits and measuring only the *errors*, never the data.
  * **Quantum key distribution works.** An eavesdropper cannot silently copy the transmitted qubits, so interception leaves statistical traces.
  * **State tomography is expensive.** Since you cannot mass-produce copies of an unknown state, characterizing it requires re-preparing it many times.

## Summary

In this chapter, we built the working vocabulary of quantum computation. A **qubit** is any two-level quantum system, described by a **state vector** \\(|\psi\rangle = \alpha|0\rangle + \beta|1\rangle\\) whose complex **amplitudes** obey the **normalization condition** \\(|\alpha|^2 + |\beta|^2 = 1\\). Because a **global phase** is unobservable, every qubit state is a point on the **Bloch sphere**, parametrized by the angles \\(\theta\\) and \\(\phi\\). The **Born rule** connects amplitudes to experiment: outcome probabilities are \\(|\alpha|^2\\) and \\(|\beta|^2\\), the outcome is genuinely random, and measurement **collapses** the state, which is why a single run yields only one bit string. Multiple qubits combine through the **tensor product**, giving \\(2^n\\) amplitudes for \\(n\\) qubits, and NumPy's `np.kron` implements this directly. States that cannot be written as such a product are **entangled**; the **Bell states** are the canonical examples, and we verified both algebraically and numerically that \\(|\Phi^{+}\rangle\\) has perfectly correlated joint outcomes while each qubit alone is completely undetermined. We stressed that entanglement produces correlation without **signalling**. Finally, the **no-cloning theorem** follows in three lines from unitarity, and it explains why quantum error correction, quantum key distribution, and state tomography all take the shapes they do.

In the next chapter, we will put these states in motion. Quantum gates are the unitary operations that rotate states on the Bloch sphere and create entanglement, and quantum circuits are how we assemble them into algorithms.

[← Chapter 1: Why Quantum Computing?](<chapter-1.html>) [Chapter 3: Quantum Gates and Circuits →](<chapter-3.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
