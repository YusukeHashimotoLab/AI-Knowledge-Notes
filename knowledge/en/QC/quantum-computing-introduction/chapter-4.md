---
title: "Chapter 4: Quantum Algorithms"
chapter_title: "Chapter 4: Quantum Algorithms"
subtitle: "Deutsch-Jozsa, Grover, and Shor - and What They Really Promise"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/B7ch2oWavJ4"
    title="Quantum Computing Ch.4: Quantum Algorithms"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-computing-introduction/chapter-4.html>) | Last sync: 2026-08-16

[Quantum Computing Dojo](<../index.html>) > [Introduction to Quantum Computing](<index.html>) > Chapter 4

Chapter 3 gave us gates and circuits. Now we ask the question that actually matters: what can we compute with them that a classical computer cannot compute as quickly? This chapter covers the three algorithms that defined the field — Deutsch–Jozsa, Grover's search, and Shor's factoring — and then does something equally important: it separates the speedups that are genuinely exponential from those that are merely quadratic, and both of those from the very large class of problems where quantum computers offer no advantage at all. The honest version of this story is more interesting than the hype, and it is the version you need if you plan to evaluate quantum computing for real work.

## 4.1 Oracles and Query Complexity

Most early quantum algorithms are stated in the **query model**, sometimes called the **black-box model**. In this model we are given a function \\(f\\) that we cannot inspect; we can only evaluate it on inputs of our choosing. The device that performs the evaluation is called an **oracle**, and the cost of an algorithm is measured by the number of oracle calls it makes — its **query complexity**. Everything else (ordinary gates, classical post-processing) is treated as free.

This is a deliberate simplification, and it is worth being clear about why it is useful and where it misleads. It is useful because query complexity can be analyzed rigorously: we can often *prove* that no classical algorithm can do better than some number of queries, which turns "quantum is faster" from a hope into a theorem. It misleads if you forget that a real oracle must be built out of real gates. An oracle whose circuit is enormous can wipe out a query-count advantage entirely.

A quantum oracle must be a unitary, so it cannot simply overwrite its input. Two standard constructions solve this.

**The XOR (bit-flip) oracle** uses an extra output qubit and writes the answer reversibly:

\\[ U_f |x\rangle |y\rangle = |x\rangle |y \oplus f(x)\rangle \\]

Applying it twice returns the original state, so it is its own inverse and therefore unitary.

**The phase oracle** encodes the answer in a sign instead:

\\[ U_f |x\rangle = (-1)^{f(x)} |x\rangle \\]

The two are equivalent: if you prepare the output qubit of the XOR oracle in the state \\(|-\rangle = (|0\rangle - |1\rangle)/\sqrt{2}\\), then \\(|y \oplus f(x)\rangle\\) picks up a factor \\((-1)^{f(x)}\\) and the output qubit factors back out, unchanged. This trick is called **phase kickback**, and it is one of the most reused ideas in quantum algorithm design. The phase oracle is a diagonal matrix, which makes it very easy to write down in code — we will use exactly that in Section 4.6.

The key point is that a quantum algorithm can query the oracle on a **superposition** of all inputs at once. By itself this does nothing useful, because measurement returns only one outcome chosen at random. The art of quantum algorithm design lies in what comes *after* the query: arranging interference so that wrong answers cancel and the right answer's amplitude grows.

## 4.2 The Deutsch–Jozsa Algorithm

### 📚 The Problem

We are given a function \\(f: \\{0,1\\}^n \to \\{0,1\\}\\) with a **promise**: \\(f\\) is either **constant** (the same value on all \\(2^n\\) inputs) or **balanced** (returns 0 on exactly half the inputs and 1 on the other half). We must decide which, with certainty.

**Classically, with certainty**, the worst case requires \\(2^{n-1} + 1\\) queries. After \\(2^{n-1}\\) queries all returning the same value, the function could still be either constant or a balanced function that happens to agree on everything we sampled; one more query settles it.

**Quantumly, one query suffices**, for any \\(n\\).

### 📚 The Circuit and the Intuition

The algorithm is three steps:

1. Prepare \\(n\\) qubits in \\(|0\rangle^{\otimes n}\\) and apply \\(H\\) to each, giving the uniform superposition \\(\frac{1}{\sqrt{2^n}}\sum_x |x\rangle\\).
2. Apply the phase oracle once: \\(\frac{1}{\sqrt{2^n}}\sum_x (-1)^{f(x)}|x\rangle\\).
3. Apply \\(H\\) to each qubit again and measure all of them.

The answer: if every qubit measures 0, \\(f\\) is constant; otherwise \\(f\\) is balanced.

Why does it work? Look at the amplitude of the all-zeros outcome after the final Hadamards. The second layer of Hadamards maps each \\(|x\rangle\\) to a superposition in which \\(|0\cdots 0\rangle\\) always appears with the same coefficient \\(1/\sqrt{2^n}\\). So the final amplitude of \\(|0\cdots 0\rangle\\) is

\\[ \alpha_{0\cdots 0} = \frac{1}{2^n}\sum_{x}(-1)^{f(x)} \\]

If \\(f\\) is constant, every term has the same sign and they add up to \\(\pm 1\\) — the outcome \\(|0\cdots 0\rangle\\) occurs with probability 1. If \\(f\\) is balanced, exactly half the terms are \\(+1\\) and half are \\(-1\\), so they cancel *exactly* and the probability of \\(|0\cdots 0\rangle\\) is 0. This is destructive interference doing the work, and it is the cleanest example of it in the whole subject.

### 📚 An Honest Assessment

Deutsch–Jozsa is a teaching algorithm, not a useful one, and it is worth saying why plainly.

The gap of \\(1\\) versus \\(2^{n-1}+1\\) queries holds only against **deterministic** classical algorithms. A **randomized** classical algorithm does nearly as well as the quantum one: sample a handful of random inputs, and if you ever see two different outputs the function is balanced; if you see \\(k\\) identical outputs, the probability of a balanced function surviving that test is \\(2^{-(k-1)}\\). Twenty-one random queries push the error probability below one in a million, independent of \\(n\\). So the exponential separation evaporates once you allow a small probability of error.

The problem itself is also artificial — nobody needs to distinguish constant from balanced functions. What Deutsch–Jozsa genuinely established, in the early 1990s, is that a provable separation between quantum and classical query complexity *exists at all*. That was the proof of concept that motivated the search for algorithms solving problems people actually care about. Two of those follow.

## 4.3 Grover's Algorithm: Unstructured Search

### 📚 The Problem and the Result

Suppose we have \\(N\\) items and a way to test whether any given item is the one we want, but no structure to exploit — no sorting, no index, nothing but the test. Classically, finding the marked item requires checking about \\(N/2\\) items on average and \\(N\\) in the worst case: \\(O(N)\\) queries.

Grover's algorithm, published in 1996, finds the marked item using \\(O(\sqrt{N})\\) queries. For \\(N = 10^6\\), that is roughly one thousand queries instead of a million.

### 📚 Amplitude Amplification

The algorithm starts in the uniform superposition \\(|s\rangle = \frac{1}{\sqrt{N}}\sum_x |x\rangle\\), in which the marked state \\(|w\rangle\\) has amplitude \\(1/\sqrt{N}\\) and therefore probability \\(1/N\\) — no better than a random guess. It then repeats a two-step **Grover iteration**:

1. **The oracle** \\(U_w = I - 2|w\rangle\langle w|\\) flips the sign of the marked amplitude, leaving all others alone.
2. **The diffusion operator** \\(U_s = 2|s\rangle\langle s| - I\\) reflects every amplitude about the mean of all amplitudes.

The second step is often described as "inversion about the average," and that description makes the mechanism visible. After the oracle, the marked amplitude sits below the average while all the others sit slightly above it. Reflecting about the average therefore pushes the marked amplitude up by roughly twice the gap, and pulls each unmarked amplitude down slightly. Repeat, and the marked amplitude grows steadily.

**The geometric picture** explains both the \\(\sqrt{N}\\) and its limit. The state never leaves the two-dimensional plane spanned by \\(|w\rangle\\) and the uniform superposition of the unmarked states. Writing the initial state as

\\[ |s\rangle = \sin\theta\,|w\rangle + \cos\theta\,|w^{\perp}\rangle, \qquad \sin\theta = \frac{1}{\sqrt{N}} \\]

each Grover iteration is the product of two reflections, which is a **rotation** by \\(2\theta\\) toward \\(|w\rangle\\). After \\(k\\) iterations the success probability is

\\[ P(k) = \sin^2\big((2k+1)\theta\big) \\]

We want \\((2k+1)\theta \approx \pi/2\\). For large \\(N\\), \\(\theta \approx 1/\sqrt{N}\\), so the optimal number of iterations is

\\[ k_{\text{opt}} \approx \frac{\pi}{4}\sqrt{N} \\]

### 📚 Two Consequences People Often Miss

**Over-rotation is real.** Because the dynamics is a rotation, running *more* iterations than optimal rotates the state past \\(|w\rangle\\) and the success probability comes back *down*, periodically. Grover's algorithm is not a process that monotonically converges; you must stop at the right time. Section 4.6 shows this happening numerically.

**The speedup is quadratic, not exponential.** This point deserves emphasis because it is the single most common misunderstanding about quantum computing. Going from \\(N\\) to \\(\sqrt{N}\\) means that a search over \\(2^{100}\\) items takes \\(2^{50}\\) quantum queries — still utterly infeasible. Grover does not make exponentially hard problems easy. It turns a 128-bit brute-force key search into a 64-bit-equivalent effort, which is the reason the standard recommendation for symmetric cryptography is simply to double key lengths, not to abandon the ciphers.

Moreover, the \\(\sqrt{N}\\) is provably the best possible: any quantum algorithm for unstructured search needs \\(\Omega(\sqrt{N})\\) queries. There is no cleverer quantum search waiting to be discovered. And in practice, the quadratic gain is fragile — running Grover on error-corrected hardware carries large constant-factor overheads per logical operation, so for many realistic problem sizes a classical computer running a highly optimized search still wins. Treat claimed practical speedups from Grover with scepticism until the constant factors are on the table.

## 4.4 Shor's Algorithm: Factoring

### 📚 Why Factoring Matters

The RSA cryptosystem, which underpins a large share of internet security, rests on the belief that factoring a large integer is hard. The best known classical algorithm, the general number field sieve, runs in time that is sub-exponential but super-polynomial in the number of digits — fast enough to factor small numbers, hopeless for 2048-bit keys.

Shor's algorithm, published in 1994, factors an \\(n\\)-bit integer in time polynomial in \\(n\\). This is a genuine **exponential speedup**, and it is the result that made governments and corporations pay attention to quantum computing.

### 📚 The Structure of the Algorithm

Shor's algorithm is mostly classical. Only one step is quantum.

**Step 1 — reduce factoring to period finding (classical).** To factor \\(N\\), pick a random \\(a < N\\) coprime to \\(N\\) and consider the function

\\[ f(x) = a^x \bmod N \\]

This function is periodic: there is a smallest \\(r > 0\\) with \\(a^r \equiv 1 \pmod N\\). That \\(r\\) is called the **order** of \\(a\\). If \\(r\\) turns out to be even and \\(a^{r/2} \not\equiv -1 \pmod N\\), then

\\[ \gcd\left(a^{r/2} - 1,\, N\right) \quad \text{and} \quad \gcd\left(a^{r/2} + 1,\, N\right) \\]

are non-trivial factors of \\(N\\). This reduction is pure number theory and involves no quantum mechanics; a randomly chosen \\(a\\) satisfies the conditions with reasonable probability, so a few attempts suffice.

**Step 2 — find the period (quantum).** Finding \\(r\\) is the hard part classically. Quantumly, we evaluate \\(f\\) on a superposition of all \\(x\\), which produces a state whose amplitudes are periodic with period \\(r\\). We then apply the **Quantum Fourier Transform** (QFT), which is the quantum analogue of the discrete Fourier transform and converts periodicity in the amplitudes into sharply peaked amplitudes at multiples of \\(1/r\\). Measuring gives a value from which \\(r\\) can be extracted by the classical continued-fractions algorithm.

The QFT is the engine here, and its efficiency is the reason the whole thing works: it acts on \\(2^n\\) amplitudes using only \\(O(n^2)\\) gates, whereas even the fast classical FFT needs \\(O(n 2^n)\\) operations on \\(2^n\\) numbers. Note carefully what the QFT does *not* give you: you cannot read out all \\(2^n\\) Fourier coefficients. You get one measurement outcome, sampled from the transformed distribution. Shor's algorithm is elegant precisely because period finding is a question that a single such sample can answer.

### 📚 What This Means for RSA — Stated Carefully

This is where discussion usually goes off the rails in both directions, so let us be precise.

**The threat is real in principle.** A sufficiently large, fault-tolerant quantum computer running Shor's algorithm would break RSA and elliptic-curve cryptography. The mathematics is not in doubt.

**It is not a present-day capability.** Shor's algorithm requires a **fault-tolerant** machine. Today's devices have error rates far too high to run the deep circuits involved, and quantum error correction requires encoding each logical qubit in many physical qubits. Published resource estimates for factoring a 2048-bit RSA key are on the order of **millions of physical qubits** running for **hours**, assuming physical error rates comparable to today's best hardware. Current devices have on the order of hundreds to a few thousand physical qubits. The gap is several orders of magnitude, and closing it is an engineering programme measured in years, not months.

Be sceptical of headlines claiming that small numbers have been "factored by a quantum computer." Demonstrations that factor numbers like 15 or 21 typically use circuits simplified with prior knowledge of the answer, and they do not scale.

**"Harvest now, decrypt later" is nevertheless a legitimate concern.** An adversary can record encrypted traffic today and store it until a capable machine exists. Any data whose confidentiality must survive for a decade or more — medical records, state secrets, long-lived credentials — is exposed to a future capability, not just a present one. This is why migration to **post-quantum cryptography** is being pursued now rather than later. NIST published its first post-quantum cryptographic standards in 2024, based on mathematical problems (such as structured lattices) for which no efficient quantum algorithm is known. The correct response to Shor is a migration schedule, not alarm.

## 4.5 A Taxonomy of Quantum Speedups

The single most important thing to take from this chapter is that "quantum speedup" is not one thing.

| Speedup | Examples | What it means |
|---|---|---|
| **Exponential** | Shor's factoring; simulation of quantum systems | Problem sizes that are permanently out of classical reach become tractable |
| **Polynomial (typically quadratic)** | Grover search; some optimization and Monte Carlo methods | \\(N \to \sqrt{N}\\); real but modest, and easily erased by constant factors |
| **None known** | Most everyday computing: databases, web serving, spreadsheets, general software | A quantum computer is not a faster computer |

Three warnings follow from this table.

**Quantum computers do not speed up everything.** They are not faster processors. For the overwhelming majority of computational tasks, a quantum computer offers no advantage whatsoever — and being probabilistic, error-prone, and cryogenically cooled, it is far worse. The right mental model is a special-purpose accelerator for a narrow class of problems with the right mathematical structure, not a replacement for a CPU.

**Quantum computers are not believed to solve NP-complete problems efficiently.** This is a widespread misconception. There is no known quantum algorithm that solves NP-complete problems (travelling salesman, satisfiability, and thousands of others) in polynomial time, and complexity theorists generally do not expect one to exist. Grover's quadratic speedup applies to brute-force search over solutions, but quadratic is not enough to tame exponential growth. Notably, factoring — the problem Shor solves — is *not* known to be NP-complete; it sits in an unusual middle ground, which is part of why it was tractable.

**Beware "exponential speedup" claims in machine learning.** A number of quantum machine learning algorithms advertised exponential speedups that depended on assumptions about how classical data gets loaded into quantum states. Loading \\(N\\) classical data points into a quantum register can itself cost \\(O(N)\\) operations, which destroys the advantage. Several such claimed speedups were later "dequantized" — classical algorithms were found with comparable scaling once the same assumptions were granted to the classical side. When you meet a quantum speedup claim, ask three questions: what exactly is the classical baseline, how does the data get in, and how does the answer get out?

The most credible near-term application is the one Feynman suggested at the outset: using a quantum system to simulate a quantum system. Chapter 5 takes that up.

## 4.6 Python: Watching the Amplitudes Move

Both algorithms in this section can be written directly on the state vector with NumPy alone. Seeing the amplitudes evolve number by number is the fastest way to make the interference arguments concrete.

**Requirements**: Python 3.9+ and NumPy only.

### Code Example 1: Grover's Algorithm on Four Items

```python
"""Grover's algorithm on N = 4 (2 qubits), tracked amplitude by amplitude."""

import numpy as np

N = 4                     # search space size = 2^2
marked = 2                # the index we are looking for, i.e. |10>
labels = ["00", "01", "10", "11"]

# --- Step 1: uniform superposition, produced by H on both qubits ---
s = np.ones(N, dtype=complex) / np.sqrt(N)
psi = s.copy()

# --- Step 2: the oracle, a diagonal unitary that flips the marked sign ---
oracle = np.eye(N, dtype=complex)
oracle[marked, marked] = -1.0

# --- Step 3: the diffusion operator, 2|s><s| - I ---
diffusion = 2 * np.outer(s, s.conj()) - np.eye(N, dtype=complex)


def show(tag, state):
    amps = "  ".join(f"|{l}>{v.real:+.4f}" for l, v in zip(labels, state))
    print(f"{tag:<22}{amps}   P(marked)={abs(state[marked])**2:.4f}")


show("initial", psi)
for k in range(1, 4):
    psi = oracle @ psi
    show(f"iter {k}: after oracle", psi)
    psi = diffusion @ psi
    show(f"iter {k}: after diff.", psi)

print("\nunitary checks:")
print("  oracle unitary   :", np.allclose(oracle.conj().T @ oracle, np.eye(N)))
print("  diffusion unitary:", np.allclose(diffusion.conj().T @ diffusion, np.eye(N)))

# --- Optimal iteration count for larger N ---
print("\nsuccess probability vs. iterations for N = 16 (1 marked item):")
N2, marked2 = 16, 9
s2 = np.ones(N2, dtype=complex) / np.sqrt(N2)
o2 = np.eye(N2, dtype=complex)
o2[marked2, marked2] = -1.0
d2 = 2 * np.outer(s2, s2.conj()) - np.eye(N2, dtype=complex)
state = s2.copy()
for k in range(9):
    print(f"  after {k} iteration(s): P(marked) = {abs(state[marked2])**2:.4f}")
    state = d2 @ (o2 @ state)
print(f"  pi/4 * sqrt(N) = {np.pi / 4 * np.sqrt(N2):.2f}")
```

**Verified output**:

```
initial               |00>+0.5000  |01>+0.5000  |10>+0.5000  |11>+0.5000   P(marked)=0.2500
iter 1: after oracle  |00>+0.5000  |01>+0.5000  |10>-0.5000  |11>+0.5000   P(marked)=0.2500
iter 1: after diff.   |00>+0.0000  |01>+0.0000  |10>+1.0000  |11>+0.0000   P(marked)=1.0000
iter 2: after oracle  |00>+0.0000  |01>+0.0000  |10>-1.0000  |11>+0.0000   P(marked)=1.0000
iter 2: after diff.   |00>-0.5000  |01>-0.5000  |10>+0.5000  |11>-0.5000   P(marked)=0.2500
iter 3: after oracle  |00>-0.5000  |01>-0.5000  |10>-0.5000  |11>-0.5000   P(marked)=0.2500
iter 3: after diff.   |00>-0.5000  |01>-0.5000  |10>-0.5000  |11>-0.5000   P(marked)=0.2500

unitary checks:
  oracle unitary   : True
  diffusion unitary: True

success probability vs. iterations for N = 16 (1 marked item):
  after 0 iteration(s): P(marked) = 0.0625
  after 1 iteration(s): P(marked) = 0.4727
  after 2 iteration(s): P(marked) = 0.9084
  after 3 iteration(s): P(marked) = 0.9613
  after 4 iteration(s): P(marked) = 0.5817
  after 5 iteration(s): P(marked) = 0.1255
  after 6 iteration(s): P(marked) = 0.0204
  after 7 iteration(s): P(marked) = 0.3649
  after 8 iteration(s): P(marked) = 0.8361
  pi/4 * sqrt(N) = 3.14
```

Read the first block carefully — it shows the whole mechanism in seven lines.

The oracle does nothing to the *probabilities*; it only flips a sign, and \\(|-0.5|^2 = |0.5|^2\\). The amplification happens entirely in the diffusion step. After the oracle the four amplitudes are \\((0.5, 0.5, -0.5, 0.5)\\) with mean \\(0.25\\); reflecting each about that mean gives \\((0, 0, 1, 0)\\). For \\(N = 4\\) a single iteration reaches the answer **exactly**, with probability 1 — a special case, since \\(\theta = \pi/6\\) and \\(3\theta = \pi/2\\) precisely.

Then watch what happens if we keep going. Iteration 2 rotates *past* the target and returns us to a uniform-magnitude state; iteration 3 returns the uniform superposition up to an overall minus sign (a global phase, hence physically identical to the start). The success probability oscillates with period 3 rather than converging. This is the over-rotation warning of Section 4.3, visible as raw numbers.

The \\(N = 16\\) block confirms the counting rule. The predicted optimum \\(\frac{\pi}{4}\sqrt{16} \approx 3.14\\) rounds to 3 iterations, and indeed the peak probability of 0.9613 occurs at \\(k = 3\\). Continue past it and the probability collapses to 0.0204 by iteration 6 before climbing again. Knowing when to stop is part of the algorithm.

### Code Example 2: Deutsch–Jozsa with a Phase Oracle

```python
"""Deutsch-Jozsa with a phase oracle, written directly on the state vector."""

import numpy as np


def hadamard_n(n):
    """H tensored n times, built by repeated Kronecker product."""
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    M = np.array([[1]], dtype=complex)
    for _ in range(n):
        M = np.kron(M, H)
    return M


def phase_oracle(f, n):
    """Diagonal unitary U_f |x> = (-1)^f(x) |x>."""
    return np.diag([(-1.0) ** f(x) for x in range(2 ** n)]).astype(complex)


def deutsch_jozsa(f, n):
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0                       # |0...0>
    Hn = hadamard_n(n)
    psi = Hn @ psi                     # uniform superposition
    psi = phase_oracle(f, n) @ psi     # ONE oracle query
    psi = Hn @ psi                     # interference
    return psi


n = 3
constant = lambda x: 1                                  # f(x) = 1 for all x
balanced = lambda x: x & 1                              # half 0, half 1
also_balanced = lambda x: bin(x).count("1") % 2         # parity

for name, f in [("constant", constant),
                ("balanced (x & 1)", balanced),
                ("balanced (parity)", also_balanced)]:
    psi = deutsch_jozsa(f, n)
    p_all_zero = abs(psi[0]) ** 2
    verdict = "CONSTANT" if p_all_zero > 0.5 else "BALANCED"
    print(f"{name:<20} P(|000>) = {p_all_zero:.4f}  ->  {verdict}")

# The single query is the whole point: a deterministic classical algorithm
# may need 2^(n-1) + 1 evaluations of f in the worst case.
print(f"\nn = {n}: quantum queries = 1, "
      f"classical worst case (deterministic) = {2 ** (n - 1) + 1}")
```

**Verified output**:

```
constant             P(|000>) = 1.0000  ->  CONSTANT
balanced (x & 1)     P(|000>) = 0.0000  ->  BALANCED
balanced (parity)    P(|000>) = 0.0000  ->  BALANCED

n = 3: quantum queries = 1, classical worst case (deterministic) = 5
```

The probabilities are not merely close to 1 and 0 — they are exactly 1 and 0, because the interference is perfect. A constant function contributes \\(2^n\\) terms of identical sign that add coherently; a balanced function contributes equal numbers of \\(+1\\) and \\(-1\\) terms that annihilate completely. Both balanced examples give the same verdict despite having very different structure, which is what "black box" is supposed to mean. Try changing `n` to 5 or 6 and the answers stay exact while the classical worst case grows to 17 and 33.

## 🎯 Exercise Problems

1. **Phase kickback**: Show that applying the XOR oracle \\(U_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle\\) with the second register in \\(|-\rangle\\) produces \\((-1)^{f(x)}|x\rangle|-\rangle\\).
2. **Grover geometry**: Using \\(P(k) = \sin^2((2k+1)\theta)\\) with \\(\sin\theta = 1/\sqrt{N}\\), compute the optimal \\(k\\) and the peak success probability for \\(N = 64\\), then check it against a modified Code Example 1.
3. **Multiple marked items**: Modify Code Example 1 so that two of the four states are marked. How many iterations are needed now, and what happens to the success probability?
4. **Randomized classical Deutsch–Jozsa**: For \\(n = 10\\), how many random queries are needed for a classical algorithm to identify a balanced function with error probability below \\(10^{-6}\\)? Compare with \\(2^{n-1}+1\\).
5. **Speedup accounting**: A symmetric cipher has a 256-bit key. Estimate the number of Grover iterations required for a brute-force key search, and explain why doubling key length is considered an adequate response.

## Summary

In this chapter, we learned what quantum algorithms actually deliver. The **query model** measures cost in oracle calls and lets us prove separations between quantum and classical computation; **phase kickback** converts an XOR oracle into the phase oracle that most algorithms use. The **Deutsch–Jozsa algorithm** decides the constant-versus-balanced question in a single query where a deterministic classical algorithm may need \\(2^{n-1}+1\\), through exact destructive interference — but the separation shrinks dramatically against randomized classical algorithms, and the problem itself is artificial. **Grover's algorithm** searches an unstructured space of \\(N\\) items in \\(O(\sqrt{N})\\) queries via **amplitude amplification**, a rotation in a two-dimensional plane that must be stopped after about \\(\frac{\pi}{4}\sqrt{N}\\) iterations or it over-rotates; its speedup is **quadratic, not exponential**, and it is provably optimal. **Shor's algorithm** achieves a genuine **exponential speedup** for factoring by reducing it to **period finding** and using the **Quantum Fourier Transform**; it would break RSA on a **fault-tolerant** machine requiring on the order of millions of physical qubits, which is far beyond present hardware — yet **"harvest now, decrypt later"** makes migration to post-quantum cryptography a present-day task. Above all, we learned the **speedup taxonomy**: exponential for factoring and quantum simulation, polynomial for search and some optimization, and **nothing at all** for the vast majority of computing. A quantum computer is a special-purpose accelerator, not a faster computer, and it is not expected to solve NP-complete problems efficiently.

In the next chapter, we will look at the machines that actually exist today — noisy, intermediate-scale quantum devices — and at the application that motivated the whole field in the first place: simulating molecules and materials.

[← Chapter 3: Quantum Gates and Circuits](<chapter-3.html>) [Chapter 5: NISQ Era and Applications to Chemistry and Materials →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
