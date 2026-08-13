---
title: "Chapter 3: Shor's Algorithm"
chapter_title: "Chapter 3: Shor's Algorithm"
subtitle: Factoring as Period Finding, Two Integers Factored End to End, and What the Same Circuit Costs at Cryptographic Size
reading_time: 45-50 minutes
difficulty: Intermediate
code_examples: 7
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-algorithms-intermediate/chapter-3.html>) | Last sync: 2026-08-13

[Fundamental Mathematics Dojo](<../index.html>) > [Intermediate Quantum Algorithms](<index.html>) > Chapter 3

This is the algorithm that made the field famous, and it is the one place in this course where the speedup is superpolynomial and nobody disputes it. Factoring an $n$-bit integer takes $\exp\left(O(n^{1/3}(\log n)^{2/3})\right)$ operations with the best known classical method and $O(n^3)$ gates with Shor's, and that gap is not a constant factor, not a conditional advantage, and not contingent on a data-loading assumption. Chapter 1's quadratic speedup came with a page of qualifications; this one does not.

What it does come with is a size. The circuit that factors 15 in this chapter uses thirteen qubits; the circuit that would factor a 2048-bit RSA modulus needs of order $10^7$ physical qubits and $10^9$ Toffoli gates under standard error-correction assumptions, and Section 3.4 works that estimate through from stated premises. The honest position sits between two familiar errors. One is to say that quantum computers break RSA — they do not, because none of them can run this circuit at that size, and the distance is measured in orders of magnitude rather than in engineering refinements. The other is to conclude from that distance that the algorithm does not matter, which is equally wrong: the mathematics is settled, the migration to lattice-based cryptography is already specified and underway for exactly this reason, and a factoring circuit is the cleanest existence proof that quantum computation is not merely a different way of writing the same complexity classes.

The technical content is short, because Chapter 2 did most of the work. Factoring reduces to finding the multiplicative order of a random integer modulo $N$; order finding is phase estimation applied to a particular unitary; and phase estimation is the inverse QFT applied after controlled powers. Everything new here is the number theory on either side of the quantum circuit — the reduction going in, and continued fractions coming out — plus the observation that the expensive part of the circuit is not the Fourier transform at all but the modular arithmetic.

## Learning Objectives

After completing this chapter, you will be able to:

  * Carry out the reduction from factoring to order finding, including the classical shortcuts that dispose of even $N$ and perfect powers, and state the two ways a randomly chosen base fails
  * Explain why the failure probability per base is at most $1/2$ for an $N$ with two distinct odd prime factors, and verify the bound by enumeration for $N = 15$ and $N = 21$
  * Construct the order-finding unitary $U_a\lvert y \rangle = \lvert ay \bmod N \rangle$, identify its eigenvalues as $e^{2\pi i s/r}$, and explain why $\lvert 1 \rangle$ is the right input
  * Account for the circuit's cost correctly: $O(n^3)$ gates in the modular exponentiation against $O(n^2)$ in the QFT, so the Fourier transform is the cheap part
  * Recover the order from a measured counting-register value using continued fractions, and explain why the convergent can return $r/\gcd(s,r)$ rather than $r$
  * Run the complete algorithm on the simulator for $N = 15$ and $N = 21$, compute the exact per-run success probability, and reproduce it by sampling
  * Convert a modulus size into logical qubits, Toffoli count, code distance and physical qubits from stated assumptions, and explain why lattice-based cryptography is the standard response while doubling a symmetric key length is enough against Grover

* * *

## 3.1 From Factoring to Order Finding

### The reduction

Let $N$ be an odd composite that is not a prime power, and pick an integer $a$ with $1 < a < N$ and $\gcd(a, N) = 1$. The **multiplicative order** of $a$ modulo $N$ is the least $r > 0$ with

$$ a^r \equiv 1 \pmod N $$

Suppose $r$ is known and even. Then $x = a^{r/2}$ satisfies $x^2 \equiv 1 \pmod N$, so

$$ (x-1)(x+1) \equiv 0 \pmod N $$

which says $N$ divides the product. If neither factor is divisible by $N$ — that is, if $x \not\equiv \pm 1 \pmod N$ — then $N$ must share a nontrivial factor with each, and

$$ \gcd\left(a^{r/2} - 1,\, N\right), \qquad \gcd\left(a^{r/2} + 1,\, N\right) $$

are proper divisors of $N$, computable in microseconds by Euclid's algorithm. That is the entire reduction. Factoring is not attacked directly at any point; what the quantum computer supplies is one integer, $r$.

The construction fails in exactly two ways, and both are visible in the derivation. If $r$ is **odd**, $a^{r/2}$ is not an integer and there is nothing to compute. If $x \equiv -1 \pmod N$, the second gcd is $N$ and the first is 1, so both divisors are trivial. Note that $x \equiv +1$ cannot happen, because it would make $r/2$ an order smaller than $r$.

### Why a random base usually works

The standard theorem: if $N$ has $k \ge 2$ distinct odd prime factors and $a$ is drawn uniformly from the integers coprime to $N$, then

$$ \Pr\left[r \text{ even and } a^{r/2} \not\equiv -1 \pmod N\right] \; \ge \; 1 - \frac{1}{2^{k-1}} \; \ge \; \frac{1}{2} $$

The proof is Chinese-remainder bookkeeping on the 2-adic valuations of the orders modulo each prime power, and it is in every textbook; what matters here is the shape of the conclusion. A failure is not a wrong answer but a detected failure — $\gcd$ returns 1 or $N$, which is checked classically and immediately — so the algorithm simply draws a new base. Repetition converts a per-base probability of one half into failure probability $2^{-m}$ after $m$ bases, and $m = 30$ already makes it negligible.

Two classes of $N$ never reach the quantum circuit. **Even $N$** is factored by inspection. **Perfect powers** $N = c^b$ are found by testing the $\lfloor \log_2 N \rfloor$ integer roots, which is why the theorem's hypothesis of two *distinct* prime factors is not a restriction: the excluded cases are the easy ones. There is also a free lunch on the way in: if the randomly drawn $a$ happens to satisfy $\gcd(a, N) > 1$, that gcd is already a factor and no circuit runs at all. For the tiny $N$ of this chapter that happens often enough to matter, which Example 6 quantifies and which is a caution about reading small demonstrations.

### Continued fractions

The quantum subroutine will not return $r$. It returns an integer $k$ from a $t$-bit register, and with probability at least $4/\pi^2 \approx 0.405$ that $k$ is the grid point nearest to $s/r$, in which case

$$ \left\lvert \frac{k}{2^t} - \frac{s}{r} \right\rvert \le \frac{1}{2^{t+1}} $$

for some unknown $s \in \lbrace 0, 1, \ldots, r-1 \rbrace$. The bound describes that best outcome and not every measurement: a $k$ further away is perfectly possible, the continued-fraction step then returns a denominator that fails the classical check $a^{q} \equiv 1$, and the circuit is simply re-run. Exercise 2 states the bound in that conditional form. Recovering $s/r$ in lowest terms from a real number known to that accuracy is a classical problem with a classical answer. If $2^t \ge N^2$ then $1/2^{t+1} < 1/(2r^2)$ for every $r < N$, and a rational with denominator below $N$ that is that close to $k/2^t$ is unique and appears among the **continued-fraction convergents** of $k/2^t$. The convergents are generated by the recurrence

$$ \frac{h_i}{q_i}, \qquad h_i = c_i h_{i-1} + h_{i-2}, \qquad q_i = c_i q_{i-1} + q_{i-2} $$

where $c_i$ are the partial quotients of the Euclidean algorithm on $k$ and $2^t$; the whole computation is integer arithmetic and costs $O(n^3)$ bit operations. This is why the counting register is $t = 2n + 1$ qubits and not $n$: the extra factor of two is what makes the rational reconstruction unique.

One subtlety survives, and it is the reason the postprocessing in the code below tries more than one candidate. The convergent returns $s/r$ in *lowest terms*, so what comes out is $r/\gcd(s,r)$. When $\gcd(s,r) > 1$ the denominator is a proper divisor of $r$, and the classical check $a^{q} \equiv 1 \pmod N$ detects this immediately. Multiplying the candidate by 2 or 3 repairs the common cases at the cost of a few modular exponentiations; running the circuit twice and taking the least common multiple of the two denominators is the other standard remedy.

### Code Example 1: The Toolbox, Re-listed

This chapter needs the state-vector simulator of [Introduction to Quantum Computing, Chapter 2](<../quantum-computing-introduction/chapter-2.html>) and the Fourier machinery of Chapter 2 of this course. Both are reproduced here — the functions this chapter needs, verbatim; only the module docstring is adapted, because two sources are being merged into one file — so that nothing below depends on a file you have to reconstruct. Save it as `shorlib.py`. The convention is the one inherited from the introductory course and never changed: **big-endian**, qubit 0 leftmost in the ket and most significant in the amplitude index, so the integer read out of a $t$-qubit counting register is $k = \sum_j q_j 2^{t-1-j}$.

```python
"""Chapter 3 toolbox. Save as shorlib.py and start every example with
`from shorlib import *`.

Part 1 is the state-vector simulator of Introduction to Quantum Computing,
Chapter 2 -- the functions this chapter needs, verbatim. Part 2 is the Fourier
machinery of Chapter 2 of this course, also verbatim.
"""
import numpy as np

# ---- part 1: the mini-simulator, verbatim ------------------------------------
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


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


def probs(state):
    """Born-rule probabilities of all 2^n outcomes."""
    return np.abs(state) ** 2


# ---- part 2: the Fourier machinery of Chapter 2, verbatim --------------------
SWAP4 = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=complex)


def cphase(theta):
    """Controlled phase gate diag(1, 1, 1, exp(i theta)) on a qubit pair."""
    return np.diag([1.0, 1.0, 1.0, np.exp(1j * theta)]).astype(complex)


def iqft(state, qubits, n):
    """Inverse QFT: the same gates in reverse order with conjugated phases."""
    m = len(qubits)
    for j in range(m // 2):
        state = apply_gate(state, SWAP4, [qubits[j], qubits[m - 1 - j]], n)
    for j in reversed(range(m)):
        for k in reversed(range(j + 1, m)):
            state = apply_gate(state, cphase(-np.pi / 2 ** (k - j)),
                               [qubits[k], qubits[j]], n)
        state = apply_gate(state, H, [qubits[j]], n)
    return state


def controlled(U):
    """Block-diagonal controlled version of U; the control is the first qubit."""
    d = U.shape[0]
    C = np.eye(2 * d, dtype=complex)
    C[d:, d:] = U
    return C
```

Only four of the simulator's nine functions are needed here, and only the inverse transform of Chapter 2's pair, because order finding never applies a forward QFT. The `controlled` helper of Chapter 2, Example 4 does the rest of the work: it turns any $2^k \times 2^k$ unitary into a $2^{k+1} \times 2^{k+1}$ one, which `apply_gate` then applies to a control qubit plus $k$ targets anywhere in the register.

### Code Example 2: The Classical Half

Everything in Section 3.1 above, computed. The point of tabulating every coprime base rather than sampling a few is that the failure modes then become countable, and the theorem's $1/2$ bound becomes a number you can check rather than a claim you accept.

```python
"""Chapter 3, Example 2: the classical half of Shor's algorithm.
Continues from Example 1 (same session)."""

import numpy as np
from math import gcd


def order(a, N):
    """Smallest r > 0 with a^r = 1 (mod N), by brute force. Classical, O(N)."""
    x, r = a % N, 1
    while x != 1:
        x = (x * a) % N
        r += 1
    return r


def factors_from_order(a, r, N):
    """The last step of the reduction: gcd(a^(r/2) -+ 1, N)."""
    if r % 2 != 0:
        return None, "r is odd"
    x = pow(a, r // 2, N)
    if x == N - 1:
        return None, "a^(r/2) = -1 mod N"
    f1, f2 = gcd(x - 1, N), gcd(x + 1, N)
    for f in (f1, f2):
        if 1 < f < N:
            return (f, N // f), "ok"
    return None, "only trivial gcds"


for N in (15, 21):
    coprime = [a for a in range(2, N) if gcd(a, N) == 1]
    print(f"N = {N}: every base coprime to N, and what it yields")
    print("-" * 74)
    print(f"  {'a':>4}{'r = ord_N(a)':>14}{'r even':>8}"
          f"{'a^(r/2) mod N':>15}{'factors':>12}{'verdict':>20}")
    good = 0
    for a in coprime:
        r = order(a, N)
        fac, why = factors_from_order(a, r, N)
        if fac:
            good += 1
        half = pow(a, r // 2, N) if r % 2 == 0 else None
        print(f"  {a:>4}{r:>14}{'yes' if r % 2 == 0 else 'no':>8}"
              f"{('-' if half is None else str(half)):>15}"
              f"{('-' if fac is None else f'{fac[0]}x{fac[1]}'):>12}"
              f"{why:>20}")
    print(f"\n  {good} of {len(coprime)} bases succeed "
          f"({good/len(coprime):.3f}); the theorem guarantees at least 1/2 "
          f"for\n  an N with two distinct odd prime factors.\n")

print("The cases the quantum part never sees")
print("-" * 74)


def classical_shortcuts(N):
    """Everything Shor's algorithm dispatches classically before measuring."""
    if N % 2 == 0:
        return f"even: 2 x {N//2}"
    for b in range(2, int(np.log2(N)) + 1):
        root = round(N ** (1.0 / b))
        for cand in (root - 1, root, root + 1):
            if cand > 1 and cand ** b == N:
                return f"perfect power: {cand}^{b}"
    return None


for N in (15, 21, 16, 27, 35, 2 ** 10, 91):
    sc = classical_shortcuts(N)
    print(f"  N = {N:>5}: " + (sc if sc else "needs order finding"))
print("  A random base can also be lucky: if gcd(a, N) > 1 the factor is free.")
print(f"  For N = 15 that happens for a in "
      f"{[a for a in range(2, 15) if 1 < gcd(a, 15) < 15]}, "
      f"{len([a for a in range(2,15) if 1 < gcd(a,15) < 15])}/13 of the bases.")

print("\nContinued fractions: recovering r from a noisy s/r")
print("-" * 74)


def convergents(num, den, max_den):
    """Continued-fraction convergents of num/den with denominator <= max_den."""
    out = []
    h2, h1, k2, k1 = 0, 1, 1, 0
    n, d = num, den
    while d:
        q = n // d
        h, k = q * h1 + h2, q * k1 + k2
        if k > max_den:
            break
        out.append((h, k))
        h2, h1, k2, k1 = h1, h, k1, k
        n, d = d, n - q * d
    return out


for num, den, max_den, label in [(1365, 4096, 21, "k = 1365, t = 12, N = 21"),
                                 (683, 2048, 21, "k = 683, t = 11, N = 21"),
                                 (192, 256, 15, "k = 192, t = 8, N = 15"),
                                 (3, 8, 15, "k = 3, t = 3, N = 15")]:
    cs = convergents(num, den, max_den)
    print(f"  {label:<26} {num}/{den} = {num/den:.6f}")
    print("     convergents: " + "  ".join(f"{h}/{k}" for h, k in cs))
print("  The last convergent with denominator <= N is the candidate for r; "
      "the check\n  a^r = 1 mod N is classical and cheap, so a wrong guess "
      "costs nothing.")
```

```text
N = 15: every base coprime to N, and what it yields
--------------------------------------------------------------------------
     a  r = ord_N(a)  r even  a^(r/2) mod N     factors             verdict
     2             4     yes              4         3x5                  ok
     4             2     yes              4         3x5                  ok
     7             4     yes              4         3x5                  ok
     8             4     yes              4         3x5                  ok
    11             2     yes             11         5x3                  ok
    13             4     yes              4         3x5                  ok
    14             2     yes             14           -  a^(r/2) = -1 mod N

  6 of 7 bases succeed (0.857); the theorem guarantees at least 1/2 for
  an N with two distinct odd prime factors.

N = 21: every base coprime to N, and what it yields
--------------------------------------------------------------------------
     a  r = ord_N(a)  r even  a^(r/2) mod N     factors             verdict
     2             6     yes              8         7x3                  ok
     4             3      no              -           -            r is odd
     5             6     yes             20           -  a^(r/2) = -1 mod N
     8             2     yes              8         7x3                  ok
    10             6     yes             13         3x7                  ok
    11             6     yes              8         7x3                  ok
    13             2     yes             13         3x7                  ok
    16             3      no              -           -            r is odd
    17             6     yes             20           -  a^(r/2) = -1 mod N
    19             6     yes             13         3x7                  ok
    20             2     yes             20           -  a^(r/2) = -1 mod N

  6 of 11 bases succeed (0.545); the theorem guarantees at least 1/2 for
  an N with two distinct odd prime factors.

The cases the quantum part never sees
--------------------------------------------------------------------------
  N =    15: needs order finding
  N =    21: needs order finding
  N =    16: even: 2 x 8
  N =    27: perfect power: 3^3
  N =    35: needs order finding
  N =  1024: even: 2 x 512
  N =    91: needs order finding
  A random base can also be lucky: if gcd(a, N) > 1 the factor is free.
  For N = 15 that happens for a in [3, 5, 6, 9, 10, 12], 6/13 of the bases.

Continued fractions: recovering r from a noisy s/r
--------------------------------------------------------------------------
  k = 1365, t = 12, N = 21   1365/4096 = 0.333252
     convergents: 0/1  1/3
  k = 683, t = 11, N = 21    683/2048 = 0.333496
     convergents: 0/1  1/2  1/3
  k = 192, t = 8, N = 15     192/256 = 0.750000
     convergents: 0/1  1/1  3/4
  k = 3, t = 3, N = 15       3/8 = 0.375000
     convergents: 0/1  1/2  1/3  3/8
  The last convergent with denominator <= N is the candidate for r; the check
  a^r = 1 mod N is classical and cheap, so a wrong guess costs nothing.
```

**What to look for.** For $N = 15$ six of the seven coprime bases succeed and for $N = 21$ six of eleven, both above the guaranteed $1/2$ — though both counts run over $1 < a < N$ and so exclude $a = 1$, which the theorem's sample space includes and which always fails; over $\mathbb{Z}_N^\ast$ the $N = 21$ rate is exactly $6/12$, so the bound is attained rather than beaten, as Exercise 1 works out for $N = 33$. And the single failure mode for $N = 15$ is $a = 14 \equiv -1$, whose order is 2 and whose $a^{r/2}$ is $-1$ by construction. For $N = 21$ both failure modes appear: $a = 4$ and $a = 16$ have order 3, and $a = 5, 17, 20$ hit $a^{r/2} \equiv -1$. Reading the table as a whole, every failure is a property of the *base*, detected by two lines of classical arithmetic, and none of them is a property of the algorithm.

The continued-fraction block previews the postprocessing. The row $k = 1365$, $t = 12$ is the one to study: $1365/4096 = 0.333252$, whose convergents stop at $1/3$, and $3$ is not the order of 2 modulo 21 — the order is 6. The true $s/r$ was $2/6$, which reduces to $1/3$, and $\gcd(s,r) = 2$ has been lost. The remedy is the one described above, and the row $k = 683$ shows the alternative path: there the convergents pass through $1/2$ and reach $1/3$, so the same doubling recovers 6.

* * *

## 3.2 The Order-Finding Circuit

### The unitary and its eigenvalues

Define, for $\gcd(a, N) = 1$, the operator

$$ U_a \lvert y \rangle = \lvert ay \bmod N \rangle $$

on an $n$-qubit register with $n = \lceil \log_2 N \rceil$, extended by the identity on the $2^n - N$ basis states with $y \ge N$. It is unitary because multiplication by $a$ permutes the residues coprime to $N$ — indeed $U_a$ is nothing but a permutation matrix, and $U_a^r = I$.

Its eigenvectors are Fourier modes over the orbit of 1. For $s \in \lbrace 0, \ldots, r-1 \rbrace$,

$$ \lvert u_s \rangle = \frac{1}{\sqrt{r}}\sum_{j=0}^{r-1} e^{-2\pi i sj/r}\, \lvert a^j \bmod N \rangle, \qquad U_a \lvert u_s \rangle = e^{2\pi i s/r}\, \lvert u_s \rangle $$

so every eigenphase of $U_a$ is a multiple of $1/r$. Phase estimation on $U_a$ therefore returns an estimate of $s/r$, and that is the whole idea. The remaining problem is that preparing $\lvert u_s \rangle$ requires knowing $r$. The resolution is the reason the algorithm is elegant: the computational basis state $\lvert 1 \rangle$ is the uniform superposition of all of them,

$$ \lvert 1 \rangle = \frac{1}{\sqrt{r}}\sum_{s=0}^{r-1} \lvert u_s \rangle $$

so feeding $\lvert 1 \rangle$ into phase estimation returns a uniformly random $s \in \lbrace 0, \ldots, r-1 \rbrace$ — as Chapter 2's Example 4 showed, an input that is a superposition of eigenvectors returns each eigenphase with its own weight. The outcome $s = 0$ is useless and occurs with probability $1/r$; every other outcome carries information about $r$.

### Where the cost is

The circuit needs controlled $U_a^{2^j}$ for $j = 0, \ldots, t-1$, and the naive reading — $2^t - 1$ applications of $U_a$ — is not how it is done. Because $a^{2^j} \bmod N$ is a *classical* computation, each controlled power is a single controlled modular multiplication by a precomputed constant. There are therefore $t = 2n+1$ of them, each a reversible modular multiplier costing $O(n^2)$ elementary gates with schoolbook arithmetic, for $O(n^3)$ in total. Against that, the inverse QFT on $t$ qubits costs $t(t+1)/2 = O(n^2)$.

The ratio is the point, and it is worth stating as a slogan: **Shor's algorithm is a modular-exponentiation circuit with a Fourier transform stapled to the end.** The Fourier transform is the part that gets the name and the part that is asymptotically free. Every serious resource estimate for factoring is an estimate of the arithmetic — which adder, which multiplier, how many ancillas, how much of it can be windowed or precomputed — and the QFT does not appear in the leading term. This is also why the *approximate* QFT is universally used: its error is negligible next to everything else. One implementation note that Chapter 2 has already paid for: nothing here requires the $t$-qubit counting register to be held coherently, because the iterative single-ancilla phase estimation of Section 2.3 applies to $U_a$ unchanged, replacing $t = 2n+1$ counting qubits by one ancilla and $t$ rounds of measurement and feedback. That is the standard route to the compact factoring circuits in the literature, which quote of order $2n$ qubits rather than $3n$.

| | Modular exponentiation | Inverse QFT |
| --- | --- | --- |
| What it does | writes $a^k \bmod N$ into the work register | reads the period out of the counting register |
| Count | $2n+1$ controlled modular multiplications | $t(t+1)/2$ rotations, $\lfloor t/2 \rfloor$ swaps |
| Gates | $O(n^3)$ | $O(n^2)$ |
| Non-Clifford content | Toffolis in the adders — the dominant cost | small-angle rotations, mostly droppable |
| Ancillas | $O(n)$ working registers | none |

### Code Example 3: The Order-Finding Circuit

```python
"""Chapter 3, Example 3: the order-finding circuit, and where its cost is.
Continues from Example 2 (same session)."""

import numpy as np


def modmul_unitary(a, N, n_work):
    """Permutation matrix for |y> -> |a y mod N>, the identity on y >= N.

    A permutation because multiplication by a is a bijection of Z_N when
    gcd(a, N) = 1. On real hardware this is a reversible modular multiplier
    built from adders, not a matrix.
    """
    d = 2 ** n_work
    U = np.zeros((d, d), dtype=complex)
    for y in range(d):
        U[(a * y) % N if y < N else y, y] = 1.0
    return U


def order_finding_state(a, N, t, n_work):
    """Full state after the controlled modular exponentiation and inverse QFT.

    Counting qubits 0..t-1 (qubit 0 = most significant), work register
    t..t+n_work-1 initialized to |1>. Only t controlled modular multiplications
    are needed, because a^(2^j) mod N is computed classically beforehand.
    """
    n = t + n_work
    state = np.kron(ket('0' * t), ket(format(1, f'0{n_work}b')))
    for j in range(t):
        state = apply_gate(state, H, [j], n)
    for j in range(t):
        a_pow = pow(a, 2 ** (t - 1 - j), N)
        state = apply_gate(state, controlled(modmul_unitary(a_pow, N, n_work)),
                           [j] + list(range(t, n)), n)
    return iqft(state, list(range(t)), n)


print("The modular multiplication operator is a permutation")
print("-" * 74)
for a, N, n_work in [(7, 15, 4), (2, 21, 5)]:
    U = modmul_unitary(a, N, n_work)
    r = order(a, N)
    Ur = np.linalg.matrix_power(U, r)
    print(f"  a = {a}, N = {N}, {n_work} work qubits, dimension {2**n_work}")
    print(f"    unitary: max |U^dag U - I| = "
          f"{np.max(np.abs(U.conj().T @ U - np.eye(2**n_work))):.1e}")
    print(f"    order r = {r}:  max |U^r - I| = "
          f"{np.max(np.abs(Ur - np.eye(2**n_work))):.1e}")
    orbit = [1]
    while True:
        nxt = (orbit[-1] * a) % N
        if nxt == 1:
            break
        orbit.append(nxt)
    print(f"    orbit of |1>: " + " -> ".join(str(y) for y in orbit)
          + " -> 1")
    print(f"    basis states with y >= N: {2**n_work - N} left untouched, "
          f"which is what keeps U unitary")

print("\nWhere the gates actually go")
print("-" * 74)
print("  Counting register t = 2n+1 bits, work register n = ceil(log2 N) bits.")
print(f"  {'N':>10}{'n':>6}{'t':>7}{'qubits':>9}{'ctrl mod-mults':>16}"
      f"{'~gates in them':>17}{'~QFT gates':>12}")
for N in [15, 21, 2 ** 16 + 1, 2 ** 64 + 1, 2 ** 1024 + 1, 2 ** 2048 + 1]:
    n_bits = N.bit_length()
    t = 2 * n_bits + 1
    label = f"{N}" if N < 1000 else f"~2^{n_bits-1}"
    print(f"  {label:>10}{n_bits:>6}{t:>7}{t + n_bits:>9}{t:>16}"
          f"{t * n_bits ** 2:>17d}{t * (t + 1) // 2:>12d}")
print("  The modular arithmetic outweighs the QFT by a factor ~n at every "
      "size:\n  Shor's algorithm is a modular-exponentiation circuit with a "
      "Fourier transform\n  stapled to the end, not the other way round.")

print("\nThe state just before the inverse QFT, N = 15, a = 7, t = 4")
print("-" * 74)
a, N, n_work, t = 7, 15, 4, 4
n = t + n_work
st = np.kron(ket('0' * t), ket(format(1, f'0{n_work}b')))
for j in range(t):
    st = apply_gate(st, H, [j], n)
for j in range(t):
    st = apply_gate(st, controlled(modmul_unitary(pow(a, 2 ** (t-1-j), N),
                                                 N, n_work)),
                    [j] + list(range(t, n)), n)
work_probs = probs(st).reshape(2 ** t, -1).sum(axis=0)
print("  work-register marginal (only the orbit of 1 is populated):")
print("   " + "  ".join(f"|{y}>: {work_probs[y]:.4f}"
                        for y in np.flatnonzero(work_probs > 1e-9)))
print(f"  each of the {int(round(1/work_probs.max()))} orbit states carries "
      f"probability 1/r, and the counting register\n  conditioned on any one "
      "of them is periodic with period r -- which is what the\n  inverse QFT "
      "then reads.")
```

```text
The modular multiplication operator is a permutation
--------------------------------------------------------------------------
  a = 7, N = 15, 4 work qubits, dimension 16
    unitary: max |U^dag U - I| = 0.0e+00
    order r = 4:  max |U^r - I| = 0.0e+00
    orbit of |1>: 1 -> 7 -> 4 -> 13 -> 1
    basis states with y >= N: 1 left untouched, which is what keeps U unitary
  a = 2, N = 21, 5 work qubits, dimension 32
    unitary: max |U^dag U - I| = 0.0e+00
    order r = 6:  max |U^r - I| = 0.0e+00
    orbit of |1>: 1 -> 2 -> 4 -> 8 -> 16 -> 11 -> 1
    basis states with y >= N: 11 left untouched, which is what keeps U unitary

Where the gates actually go
--------------------------------------------------------------------------
  Counting register t = 2n+1 bits, work register n = ceil(log2 N) bits.
           N     n      t   qubits  ctrl mod-mults   ~gates in them  ~QFT gates
          15     4      9       13               9              144          45
          21     5     11       16              11              275          66
       ~2^16    17     35       52              35            10115         630
       ~2^64    65    131      196             131           553475        8646
     ~2^1024  1025   2051     3076            2051       2154831875     2104326
     ~2^2048  2049   4099     6148            4099      17209245699     8402950
  The modular arithmetic outweighs the QFT by a factor ~n at every size:
  Shor's algorithm is a modular-exponentiation circuit with a Fourier transform
  stapled to the end, not the other way round.

The state just before the inverse QFT, N = 15, a = 7, t = 4
--------------------------------------------------------------------------
  work-register marginal (only the orbit of 1 is populated):
   |1>: 0.2500  |4>: 0.2500  |7>: 0.2500  |13>: 0.2500
  each of the 4 orbit states carries probability 1/r, and the counting register
  conditioned on any one of them is periodic with period r -- which is what the
  inverse QFT then reads.
```

**What to look for.** $U_a$ is a permutation matrix to exact zero error, and $U_a^r = I$ exactly — an integer identity computed in floating point with no residue, because permutation matrices multiply without rounding. The orbits printed are the sequences $1, a, a^2, \ldots$ modulo $N$, of length exactly $r$, and the states outside $\lbrace 0, \ldots, N-1 \rbrace$ are fixed points, which is how the operator stays unitary on a register whose dimension is not $N$.

The accounting table is Section 3.2's claim in numbers. At $n = 2049$ bits the modular arithmetic contributes $1.7\times10^{10}$ gates and the QFT $8.4\times10^6$: a ratio of two thousand. Every discussion of "the quantum Fourier transform breaking RSA" has the emphasis in the wrong place.

The last block shows the state that phase estimation actually acts on. After the controlled modular exponentiation, the work register is supported on exactly the $r$ elements of the orbit, each with probability $1/r$, and conditioned on any one of them the counting register holds a superposition periodic with period $r$ — precisely the state of Chapter 2's Example 3. The inverse QFT then reads the period, and the offset, which is which element of the orbit was measured, is discarded exactly as Example 3 predicted.

* * *

## 3.3 Factoring 15 and 21, End to End

### Code Example 4: $N = 15$, Where the Period Divides the Register

$N = 15$ is the traditional demonstration, and it is unrepresentatively kind: every order is 2 or 4, both divide $2^t$, and the output distribution is therefore exactly peaked with no leakage at all. That makes it the right place to check the postprocessing against a case where the answer is visible by eye.

```python
"""Chapter 3, Example 4: factoring N = 15 end to end.
Continues from Example 3 (same session)."""

import numpy as np
import matplotlib.pyplot as plt
from math import gcd


def postprocess(k, t, a, N, max_multiple=3):
    """Turn a measured counting-register value k into (r, factors) or None.

    Continued fractions give candidate denominators for k/2^t ~ s/r. A
    convergent returns r/gcd(s, r) rather than r, so a few small multiples of
    each candidate are tried as well; that repairs the common gcd = 2 or 3 and
    is a fixed number of classical modular exponentiations, not a search. The
    convergent 0/1 is skipped because s = 0 carries no information about r.
    """
    for _, r_cand in convergents(k, 2 ** t, N):
        if r_cand < 2:
            continue
        for m in range(1, max_multiple + 1):
            r = m * r_cand
            if r <= N and pow(a, r, N) == 1:
                return r, factors_from_order(a, r, N)[0]
    return None, None


a, N, n_work, t = 7, 15, 4, 8
print(f"Order finding for a = {a}, N = {N}: t = {t} counting qubits, "
      f"{t + n_work} qubits total")
print("-" * 76)
state = order_finding_state(a, N, t, n_work)
p = probs(state).reshape(2 ** t, -1).sum(axis=1)
r_true = order(a, N)
print(f"  true order r = {r_true}, so the ideal peaks sit at "
      f"k = j * 2^t/r = j * {2**t // r_true}")
print(f"  {'k':>6}{'p(k)':>10}{'k/2^t':>10}{'CF convergents':>26}"
      f"{'r found':>9}{'factors':>10}")
for k in np.flatnonzero(p > 1e-9):
    r, fac = postprocess(int(k), t, a, N)
    cs = "  ".join(f"{h}/{q}" for h, q in convergents(int(k), 2 ** t, N))
    print(f"  {k:>6}{p[k]:>10.4f}{k/2**t:>10.4f}{cs:>26}"
          f"{('-' if r is None else r):>9}"
          f"{('-' if fac is None else f'{fac[0]}x{fac[1]}'):>10}")
print(f"  total probability on those {int((p > 1e-9).sum())} outcomes: "
      f"{p[p > 1e-9].sum():.6f}")

print("\nExact success probability, summed over the whole distribution")
print("-" * 76)
succ = sum(p[k] for k in range(2 ** t)
           if postprocess(k, t, a, N)[1] is not None)
print(f"  P(a nontrivial factor of {N} on one run with a = {a}) = {succ:.6f}")
print(f"  P(failure)  = {1 - succ:.6f}, and p(k = 0) alone is {p[0]:.6f}")

print("\nSampling the circuit, 2000 shots")
print("-" * 76)
rng = np.random.default_rng(20260813)
counts = rng.multinomial(2000, p)
found = {}
for k in np.flatnonzero(counts):
    r, fac = postprocess(int(k), t, a, N)
    key = "no factor" if fac is None else f"{fac[0]} x {fac[1]}"
    found[key] = found.get(key, 0) + int(counts[k])
    print(f"  k = {k:>3} seen {counts[k]:>5} times  ->  "
          f"r = {r if r else '-':>2}, {key}")
print("  tally: " + ",  ".join(f"{k}: {v}" for k, v in sorted(found.items())))

print("\nEvery usable base for N = 15, at t = 8")
print("-" * 76)
print(f"  {'a':>4}{'r':>4}{'peaks':>8}{'P(success)':>13}{'factors found':>16}")
for a_ in [a_ for a_ in range(2, N) if gcd(a_, N) == 1]:
    st = order_finding_state(a_, N, t, n_work)
    pp = probs(st).reshape(2 ** t, -1).sum(axis=1)
    s = 0.0
    facs = set()
    for k in range(2 ** t):
        r, fac = postprocess(k, t, a_, N)
        if fac:
            s += pp[k]
            facs.add(tuple(sorted(fac)))
    print(f"  {a_:>4}{order(a_, N):>4}{int((pp > 1e-9).sum()):>8}{s:>13.6f}"
          f"{(', '.join(f'{f[0]}x{f[1]}' for f in facs) or '-'):>16}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(np.arange(2 ** t), p, width=1.0, color="tab:blue")
for j in range(r_true):
    ax[0].axvline(j * 2 ** t / r_true, color="k", ls=":", lw=0.8)
ax[0].set_xlabel("measured k"); ax[0].set_ylabel("probability")
ax[0].set_title(f"N = 15, a = 7, t = 8: r = {r_true} divides $2^t$")
p14 = probs(order_finding_state(14, N, t, n_work)).reshape(2 ** t, -1).sum(axis=1)
ax[1].bar(np.arange(2 ** t), p14, width=1.0, color="tab:red")
ax[1].set_xlabel("measured k"); ax[1].set_ylabel("probability")
ax[1].set_title("a = 14: r = 2, clean peaks, no factor")
plt.tight_layout()
plt.show()
```

```text
Order finding for a = 7, N = 15: t = 8 counting qubits, 12 qubits total
----------------------------------------------------------------------------
  true order r = 4, so the ideal peaks sit at k = j * 2^t/r = j * 64
       k      p(k)     k/2^t            CF convergents  r found   factors
       0    0.2500    0.0000                       0/1        -         -
      64    0.2500    0.2500                  0/1  1/4        4       3x5
     128    0.2500    0.5000                  0/1  1/2        4       3x5
     192    0.2500    0.7500             0/1  1/1  3/4        4       3x5
  total probability on those 4 outcomes: 1.000000

Exact success probability, summed over the whole distribution
----------------------------------------------------------------------------
  P(a nontrivial factor of 15 on one run with a = 7) = 0.750000
  P(failure)  = 0.250000, and p(k = 0) alone is 0.250000

Sampling the circuit, 2000 shots
----------------------------------------------------------------------------
  k =   0 seen   483 times  ->  r =  -, no factor
  k =  64 seen   530 times  ->  r =  4, 3 x 5
  k = 128 seen   505 times  ->  r =  4, 3 x 5
  k = 192 seen   482 times  ->  r =  4, 3 x 5
  tally: 3 x 5: 1517,  no factor: 483

Every usable base for N = 15, at t = 8
----------------------------------------------------------------------------
     a   r   peaks   P(success)   factors found
     2   4       4     0.750000             3x5
     4   2       2     0.500000             3x5
     7   4       4     0.750000             3x5
     8   4       4     0.750000             3x5
    11   2       2     0.500000             3x5
    13   4       4     0.750000             3x5
    14   2       2     0.000000               -
```

**What to look for.** Four outcomes, each with probability exactly $1/4$, at $k = 0, 64, 128, 192$ — the multiples of $2^t/r = 64$. Three of them yield the factorization and one, $k = 0$, does not: the useless $s = 0$ outcome, which by the argument in Section 3.2 has probability $1/r$. The exact per-run success probability is therefore $3/4$, and the 2000-shot sample returns $1517/2000 = 0.759$.

The middle rows show the continued fractions doing real work. At $k = 128$ the convergent is $1/2$, not $1/4$: the true $s/r$ was $2/4$ and the fraction reduced. The candidate 2 fails the check $7^2 \equiv 4 \pmod{15}$, the doubled candidate 4 passes, and the factors follow. That is the $\gcd(s,r) > 1$ case from Section 3.1, occurring in a quarter of all runs for this base.

The base table closes the loop with Example 2. Every base with $r = 4$ gives success probability $3/4$; every base with $r = 2$ gives $1/2$, because there are only two peaks and one of them is $k = 0$; and $a = 14$ gives exactly zero, because its order is even and its $a^{r/2}$ is $-1$, so no measured $k$ can help. The circuit is not failing there — it is returning $r = 2$ perfectly, and the *reduction* has no use for it.

### Code Example 5: $N = 21$, Where It Does Not

Twenty-one is the smallest instance that shows the general behaviour. The order of 2 modulo 21 is 6, six does not divide any power of two, and the output distribution leaks.

```python
"""Chapter 3, Example 5: factoring N = 21, where r does not divide 2^t.
Continues from Example 4 (same session)."""

import numpy as np
import matplotlib.pyplot as plt
from math import gcd

a, N, n_work = 2, 21, 5
r_true = order(a, N)
print(f"N = {N}, a = {a}: true order r = {r_true}, and 2^t/r is never an "
      f"integer")
print("-" * 78)
print(f"  {'t':>4}{'qubits':>8}{'peak k':>9}{'k/2^t':>10}{'s/r ideal':>11}"
      f"{'p(peak)':>10}{'P(success)':>12}{'P(k=0)':>9}")
dists = {}
for t in range(6, 14):
    st = order_finding_state(a, N, t, n_work)
    p = probs(st).reshape(2 ** t, -1).sum(axis=1)
    dists[t] = p
    k = int(np.argmax(p[1:]) + 1)
    s_over_r = round(k / 2 ** t * r_true) / r_true
    succ = sum(p[j] for j in range(2 ** t)
               if postprocess(j, t, a, N)[1] is not None)
    print(f"  {t:>4}{t + n_work:>8}{k:>9}{k/2**t:>10.6f}{s_over_r:>11.6f}"
          f"{p[k]:>10.4f}{succ:>12.6f}{p[0]:>9.4f}")

t = 11
p = dists[t]
print(f"\nThe six peaks at t = {t}, and what each one postprocesses to")
print("-" * 78)
top = np.sort(np.argsort(p)[::-1][:6])
print(f"  {'k':>6}{'p(k)':>9}{'k/2^t':>10}{'nearest s/r':>13}"
      f"{'CF convergents':>26}{'r':>4}{'factors':>10}")
for k in top:
    r, fac = postprocess(int(k), t, a, N)
    cs = "  ".join(f"{h}/{q}" for h, q in convergents(int(k), 2 ** t, N))
    s = round(k / 2 ** t * r_true)
    print(f"  {k:>6}{p[k]:>9.4f}{k/2**t:>10.6f}{f'{s}/{r_true}':>13}"
          f"{cs:>26}{('-' if r is None else r):>4}"
          f"{('-' if fac is None else f'{fac[0]}x{fac[1]}'):>10}")
print(f"  the six peaks hold {p[top].sum():.4f} of the probability; the rest "
      f"leaks into\n  the {2**t - 6} remaining bins because r = {r_true} does "
      f"not divide 2^{t} = {2**t}")

print(f"\nSampling the t = {t} circuit, 2000 shots")
print("-" * 78)
rng = np.random.default_rng(20260813)
counts = rng.multinomial(2000, p)
tally = {}
for k in np.flatnonzero(counts):
    _, fac = postprocess(int(k), t, a, N)
    key = "no factor" if fac is None else f"{fac[0]} x {fac[1]}"
    tally[key] = tally.get(key, 0) + int(counts[k])
print(f"  distinct k values observed: {int((counts > 0).sum())}")
print("  " + ",  ".join(f"{k}: {v}" for k, v in sorted(tally.items())))
print(f"  measured success rate {tally.get('7 x 3', 0)/2000:.4f} against the "
      f"exact "
      f"{sum(p[j] for j in range(2**t) if postprocess(j, t, a, N)[1]):.4f}")

print("\nAll usable bases for N = 21 at t = 11")
print("-" * 78)
print(f"  {'a':>4}{'r':>4}{'P(success)':>13}{'factors found':>16}"
      f"{'why it can fail':>26}")
for a_ in [x for x in range(2, N) if gcd(x, N) == 1]:
    st = order_finding_state(a_, N, t, n_work)
    pp = probs(st).reshape(2 ** t, -1).sum(axis=1)
    s = 0.0
    facs = set()
    for k in range(2 ** t):
        rr, fac = postprocess(k, t, a_, N)
        if fac:
            s += pp[k]
            facs.add(tuple(sorted(fac)))
    r_ = order(a_, N)
    why = ("r odd" if r_ % 2 else
           ("a^(r/2) = -1" if pow(a_, r_ // 2, N) == N - 1 else
            "only k = 0 and CF misses"))
    print(f"  {a_:>4}{r_:>4}{s:>13.6f}"
          f"{(', '.join(f'{f[0]}x{f[1]}' for f in facs) or '-'):>16}{why:>26}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for t_, style in [(8, "-"), (11, "-")]:
    ax[0].plot(np.arange(2 ** t_) / 2 ** t_, dists[t_], style, lw=1,
               label=f"t = {t_}")
for s in range(r_true):
    ax[0].axvline(s / r_true, color="k", ls=":", lw=0.8)
ax[0].set_xlabel("$k/2^t$"); ax[0].set_ylabel("probability")
ax[0].set_title("N = 21, a = 2: peaks near $s/6$")
ax[0].legend(fontsize=8)
ax[1].plot(list(dists), [sum(dists[t_][j] for j in range(2 ** t_)
                             if postprocess(j, t_, a, N)[1] is not None)
                         for t_ in dists], "o-", color="tab:green")
ax[1].set_xlabel("counting qubits $t$"); ax[1].set_ylabel("P(success)")
ax[1].set_ylim(0, 1)
ax[1].set_title("Success probability of one run")
plt.tight_layout()
plt.show()
```

```text
N = 21, a = 2: true order r = 6, and 2^t/r is never an integer
------------------------------------------------------------------------------
     t  qubits   peak k     k/2^t  s/r ideal   p(peak)  P(success)   P(k=0)
     6      11       32  0.500000   0.500000    0.1670    0.787013   0.1670
     7      12       64  0.500000   0.500000    0.1667    0.815174   0.1667
     8      13      128  0.500000   0.500000    0.1667    0.823615   0.1667
     9      14      256  0.500000   0.500000    0.1667    0.828564   0.1667
    10      15      512  0.500000   0.500000    0.1667    0.830918   0.1667
    11      16     1024  0.500000   0.500000    0.1667    0.832118   0.1667
    12      17     2048  0.500000   0.500000    0.1667    0.832721   0.1667
    13      18     4096  0.500000   0.500000    0.1667    0.833029   0.1667

The six peaks at t = 11, and what each one postprocesses to
------------------------------------------------------------------------------
       k     p(k)     k/2^t  nearest s/r            CF convergents   r   factors
       0   0.1667  0.000000          0/6                       0/1   -         -
     341   0.1140  0.166504          1/6                  0/1  1/6   6       7x3
     683   0.1140  0.333496          2/6             0/1  1/2  1/3   6       7x3
    1024   0.1667  0.500000          3/6                  0/1  1/2   6       7x3
    1365   0.1140  0.666504          4/6        0/1  1/1  1/2  2/3   6       7x3
    1707   0.1140  0.833496          5/6             0/1  1/1  5/6   6       7x3
  the six peaks hold 0.7893 of the probability; the rest leaks into
  the 2042 remaining bins because r = 6 does not divide 2^11 = 2048

Sampling the t = 11 circuit, 2000 shots
------------------------------------------------------------------------------
  distinct k values observed: 78
  7 x 3: 1677,  no factor: 323
  measured success rate 0.8385 against the exact 0.8321

All usable bases for N = 21 at t = 11
------------------------------------------------------------------------------
     a   r   P(success)   factors found           why it can fail
     2   6     0.832118             3x7  only k = 0 and CF misses
     4   3     0.000000               -                     r odd
     5   6     0.000000               -              a^(r/2) = -1
     8   2     0.500000             3x7  only k = 0 and CF misses
    10   6     0.832118             3x7  only k = 0 and CF misses
    11   6     0.832118             3x7  only k = 0 and CF misses
    13   2     0.500000             3x7  only k = 0 and CF misses
    16   3     0.000000               -                     r odd
    17   6     0.000000               -              a^(r/2) = -1
    19   6     0.832118             3x7  only k = 0 and CF misses
    20   2     0.000000               -              a^(r/2) = -1
```

**What to look for.** The peaks sit at $k/2^t = 0.166504$ and so on: near $s/6$ but never on it, exactly as Chapter 2's Example 3 predicted for a period that does not divide the register size. Only $78.9\%$ of the probability is on the six best bins at $t = 11$; the sampled run saw 78 distinct values of $k$ rather than 6.

And yet the success probability is $0.8321$, higher than the fraction of probability sitting on the peaks. That is not a contradiction and it is the most instructive number in the chapter: outcomes in the *shoulders* of a peak still round to the same convergent, so continued fractions recover $r$ from them too. The postprocessing is not merely tidying up — it is what converts a broad distribution into a sharp answer. The $t$-scan shows the convergence: $0.787$ at $t = 6$ rising to $0.833$ at $t = 13$, approaching the ceiling $1 - 1/r = 5/6 = 0.8333$ set by the useless $s = 0$ outcome.

The per-base table repeats Example 2's classification, now with the circuit in the loop. The three bases with $a^{r/2} \equiv -1$ and the two with odd order give exactly zero, deterministically; the six usable bases give $0.5$ or $0.83$ depending on whether $r$ is 2 or 6. Nothing is probabilistic about *which* bases work — only about whether a given run of a working base lands on a useful $k$.

### Code Example 6: The Complete Algorithm, and How Often It Works

Both halves together, with the classical shortcuts, the random base, the retry loop, and statistics over 600 complete runs.

```python
"""Chapter 3, Example 6: the whole algorithm, and how often it works.
Continues from Example 5 (same session)."""

import numpy as np
from math import gcd

_CACHE = {}


def measure_k(a, N, t, n_work, rng):
    """One shot of the quantum subroutine, from the cached exact distribution."""
    key = (a, N, t, n_work)
    if key not in _CACHE:
        st = order_finding_state(a, N, t, n_work)
        _CACHE[key] = probs(st).reshape(2 ** t, -1).sum(axis=1)
    p = _CACHE[key]
    return int(rng.choice(p.size, p=p / p.sum()))


def shor(N, rng, max_rounds=10):
    """Shor's algorithm as it is actually specified, with all classical steps.

    Returns (factors, n_quantum_calls, route). A round is one random base plus
    one call to the order-finding circuit.
    """
    if N % 2 == 0:
        return tuple(sorted((2, N // 2))), 0, "even"
    for b in range(2, N.bit_length() + 1):
        root = round(N ** (1.0 / b))
        for cand in (root - 1, root, root + 1):
            if cand > 1 and cand ** b == N:
                return tuple(sorted((cand, N // cand))), 0, "perfect power"
    n_work = N.bit_length()
    t = 2 * n_work + 1
    calls = 0
    for _ in range(max_rounds):
        a = int(rng.integers(2, N))
        g = gcd(a, N)
        if g > 1:
            return tuple(sorted((g, N // g))), calls, "lucky gcd"
        calls += 1
        k = measure_k(a, N, t, n_work, rng)
        _, fac = postprocess(k, t, a, N)
        if fac:
            return tuple(sorted(fac)), calls, "order finding"
    return None, calls, "gave up"


for N in (15, 21):
    rng = np.random.default_rng(1234)
    n_runs = 600
    routes, calls_hist, results = {}, [], {}
    for _ in range(n_runs):
        fac, calls, route = shor(N, rng)
        routes[route] = routes.get(route, 0) + 1
        calls_hist.append(calls)
        key = "failed" if fac is None else f"{fac[0]} x {fac[1]}"
        results[key] = results.get(key, 0) + 1
    calls_hist = np.array(calls_hist)
    print(f"N = {N}: {n_runs} complete runs "
          f"(t = {2*N.bit_length()+1} counting qubits)")
    print("-" * 78)
    for key in sorted(results):
        print(f"  result {key:<12} {results[key]:>5} runs "
              f"({results[key]/n_runs:.4f})")
    for key in sorted(routes):
        print(f"  route  {key:<12} {routes[key]:>5} runs "
              f"({routes[key]/n_runs:.4f})")
    print(f"  quantum calls per run: mean {calls_hist.mean():.3f}, "
          f"median {int(np.median(calls_hist))}, max {calls_hist.max()}")
    hist = np.bincount(calls_hist)
    print("  calls histogram: " + "  ".join(
        f"{i}: {c}" for i, c in enumerate(hist) if c))
    print()

print("Where the failures come from, counted exactly rather than sampled")
print("-" * 78)
print(f"  {'N':>4}{'bases a':>9}{'lucky gcd':>11}{'r odd':>8}"
      f"{'a^(r/2)=-1':>12}{'usable':>8}{'mean P(succ | usable)':>23}")
for N in (15, 21):
    n_work = N.bit_length()
    t = 2 * n_work + 1
    lucky = [a for a in range(2, N) if gcd(a, N) > 1]
    cop = [a for a in range(2, N) if gcd(a, N) == 1]
    odd_r = [a for a in cop if order(a, N) % 2]
    minus1 = [a for a in cop if order(a, N) % 2 == 0
              and pow(a, order(a, N) // 2, N) == N - 1]
    usable = [a for a in cop if a not in odd_r and a not in minus1]
    ps = []
    for a in usable:
        st = order_finding_state(a, N, t, n_work)
        p = probs(st).reshape(2 ** t, -1).sum(axis=1)
        ps.append(sum(p[k] for k in range(2 ** t)
                      if postprocess(k, t, a, N)[1] is not None))
    print(f"  {N:>4}{len(cop) + len(lucky):>9}{len(lucky):>11}{len(odd_r):>8}"
          f"{len(minus1):>12}{len(usable):>8}{np.mean(ps):>23.6f}")
print("  Overall probability of success per round = "
      "P(lucky gcd) + P(usable base) * P(circuit succeeds).")
print("  Every failure is detected classically in microseconds, so the "
      "algorithm simply\n  draws a new base. Repetition turns a per-round "
      "probability of about one half\n  into a failure probability of 2^-m "
      "after m rounds, at negligible cost.")
```

```text
N = 15: 600 complete runs (t = 9 counting qubits)
------------------------------------------------------------------------------
  result 3 x 5          600 runs (1.0000)
  route  lucky gcd      352 runs (0.5867)
  route  order finding   248 runs (0.4133)
  quantum calls per run: mean 0.747, median 1, max 5
  calls histogram: 0: 262  1: 254  2: 63  3: 17  4: 3  5: 1

N = 21: 600 complete runs (t = 11 counting qubits)
------------------------------------------------------------------------------
  result 3 x 7          600 runs (1.0000)
  route  lucky gcd      404 runs (0.6733)
  route  order finding   196 runs (0.3267)
  quantum calls per run: mean 0.873, median 1, max 7
  calls histogram: 0: 262  1: 215  2: 85  3: 22  4: 10  5: 4  6: 1  7: 1

Where the failures come from, counted exactly rather than sampled
------------------------------------------------------------------------------
     N  bases a  lucky gcd   r odd  a^(r/2)=-1  usable  mean P(succ | usable)
    15       13          6       0           1       6               0.666667
    21       19          8       2           3       6               0.721412
  Overall probability of success per round = P(lucky gcd) + P(usable base) * P(circuit succeeds).
  Every failure is detected classically in microseconds, so the algorithm simply
  draws a new base. Repetition turns a per-round probability of about one half
  into a failure probability of 2^-m after m rounds, at negligible cost.
```

**What to look for.** Every one of the 600 runs succeeds for both $N$, with a mean of $0.75$ and $0.87$ calls to the quantum circuit respectively and a worst case of 5 and 7. That is the retry loop working exactly as the theorem says it should.

Now the caution, and it is the reason this example exists. For $N = 15$, $59\%$ of the *runs* never touched the quantum circuit at all, and for $N = 21$ it is $67\%$: a randomly drawn $a$ had $\gcd(a, N) > 1$ and the factor came out of the gcd. Those two figures are route fractions over completed runs, and they are larger than the per-draw probability of a lucky gcd, which is $1 - (\phi(N)-1)/(N-2) = 6/13 = 0.4615$ for $N = 15$ and $8/19 = 0.4211$ for $N = 21$ — the base is drawn uniformly from $2 \le a < N$, so $a = 1$ is excluded from both counts. The route fractions are the larger numbers because a lucky draw always ends the run immediately, whereas a coprime draw may fail its two classical tests and force another draw. Either way the fraction goes to zero as $N$ grows, but for $N$ this small the *classical* shortcut is doing most of the work. Any demonstration of Shor's algorithm on a two-digit number is therefore a demonstration of the postprocessing and the circuit mechanics, and not evidence about factoring. The exact-count table at the end separates the contributions cleanly, so nothing has to be taken on trust.

* * *

## 3.4 What This Means, and What It Does Not

### The speedup is real, and it is not exponential

The best known classical factoring algorithm, the general number field sieve, runs in

$$ \exp\left(\left(\tfrac{64}{9}\right)^{1/3} (\ln N)^{1/3} (\ln \ln N)^{2/3}\right) $$

operations — subexponential in $\ln N$, superpolynomial. Shor's is $O(n^3)$ with $n = \log_2 N$, i.e. polynomial. The gap is therefore *superpolynomial* rather than exponential, and the distinction is not pedantry: it means the crossover point is much larger than a naive $2^n$-versus-$n^3$ comparison suggests, and it means that no classical algorithm has to be exponentially bad for the quantum one to win.

It also means there is no proof that factoring is classically hard. Shor's algorithm shows factoring is in the quantum polynomial class; nobody has shown it is outside the classical one. A classical polynomial factoring algorithm would be a shock, but it would not contradict any theorem. The correct statement of what Shor proved is about the quantum side only.

### The size, from stated assumptions

Turning $n^3$ into a machine requires three inputs, and quoting a resource estimate without them is meaningless.

  * **A logical circuit.** Modular exponentiation on $n$ bits with $\sim 3n$ logical qubits and a Toffoli count of order $n^3$ — the coefficient depends on the arithmetic, and a decade of optimization has moved it substantially without changing the exponent.
  * **An error-correction assumption.** A physical error rate, a code, and a threshold. The code distance $d$ must satisfy $0.1(p/p_{\text{th}})^{(d+1)/2} < 1/(\text{number of operations})$, and the surface code needs $\approx 2d^2$ physical qubits per logical one, plus magic-state factories for the Toffolis.
  * **A clock.** A syndrome-extraction cycle time, and an assumption about how much of the circuit can run in parallel.

Example 7 carries three explicit choices through — $p = 10^{-3}$, threshold $10^{-2}$, cycle time $1\ \mu$s — and reaches $10^7$ physical qubits and hours of runtime for $n = 2048$. Those figures are in the same range as the published estimates, which is reassuring about the arithmetic and not about the premises: every one of the three inputs is an assumption, and the published numbers have moved by orders of magnitude as the assumptions improved. Read the exponents, not the mantissas.

### The two errors, side by side

| Overstatement | Understatement |
| --- | --- |
| "Quantum computers break RSA" | "Shor's algorithm is a curiosity" |
| No device can run this circuit; the gap is $10^6$ or more in qubit count | The mathematics is settled and the exponent is polynomial |
| Treats a 2-digit demonstration as evidence about 617-digit moduli | Ignores that stored ciphertext can be decrypted later |
| Ignores that most of a small demonstration is classical arithmetic | Ignores that the standards migration is a direct consequence |

The right response to both is the same: state the circuit size, state the assumptions, and separate what is proved from what is engineered.

### What survives, and why

Shor's algorithm is not a general-purpose attack. It solves the **hidden subgroup problem in a finite abelian group**, and factoring, discrete logarithms modulo a prime, and elliptic-curve discrete logarithms are all instances. What all three have in common is a group structure with a period to find; the QFT is the character transform of that group, and it is what makes the period measurable.

Where the hardness of a cryptosystem does *not* come from an abelian group, Shor has nothing to say. Symmetric ciphers and hash functions rest on unstructured search, where only Grover applies: a quadratic speedup, answered by doubling the key or digest length, which costs a few per cent of performance and no redesign. Lattice problems — the shortest and closest vector problems underlying Learning With Errors — are in neither box. There is no abelian hidden subgroup to exploit, and the best known quantum attacks improve only the *constant in the exponent* of the best classical ones: heuristic lattice sieving runs in about $2^{0.292n}$ classically and about $2^{0.265n}$ quantumly, an improvement a modest increase in dimension absorbs. "No known speedup" is the wrong way to say it; "no known speedup that changes the shape of the cost" is right. That, and not any timeline, is why lattice-based key exchange and signatures are the standard post-quantum replacements, and why the migration is being carried out as ordinary standards engineering rather than as a response to a demonstration.

One asymmetry makes the migration urgent independently of when a machine appears: **encrypted traffic can be recorded now and decrypted later**, so the confidentiality of anything with a long secrecy requirement depends on the algorithm in use today, not on the algorithm in use when a quantum computer exists. Signatures do not have this problem, since a signature only has to resist attack while it is being trusted. That distinction, rather than any prediction, is what sets the priority order of a real migration plan.

### Code Example 7: The Distance to Cryptographic Size

```python
"""Chapter 3, Example 7: the distance from N = 21 to a cryptographic modulus.
Continues from Example 6 (same session)."""

import numpy as np

# Every constant below is a stated modelling assumption, not a measurement.
P_PHYS = 1e-3          # physical two-qubit error rate assumed
P_THRESH = 1e-2        # surface-code threshold assumed
CYCLE = 1e-6           # surface-code measurement cycle assumed, seconds
TOFFOLI_COEF = 0.3     # Toffolis in modular exponentiation ~ 0.3 n^3


def gnfs_ops(n_bits):
    """Heuristic asymptotic cost of the general number field sieve."""
    lnN = n_bits * np.log(2.0)
    return np.exp((64.0 / 9.0) ** (1 / 3) * lnN ** (1 / 3)
                  * np.log(lnN) ** (2 / 3))


def surface_code_distance(n_logical_ops):
    """Smallest odd d with 0.1 (p/p_th)^((d+1)/2) < 1/(10 n_logical_ops)."""
    for d in range(3, 60, 2):
        if 0.1 * (P_PHYS / P_THRESH) ** ((d + 1) / 2) < 0.1 / n_logical_ops:
            return d
    return None


print("Circuit size against modulus size (this chapter's own accounting)")
print("-" * 78)
print(f"  {'n bits':>8}{'logical qubits':>16}{'Toffolis ~0.3n^3':>19}"
      f"{'GNFS ops':>13}{'quantum/GNFS':>15}")
for n_bits in [5, 32, 256, 1024, 2048, 4096]:
    logical = 3 * n_bits
    toff = TOFFOLI_COEF * n_bits ** 3
    g = gnfs_ops(n_bits)
    print(f"  {n_bits:>8}{logical:>16d}{toff:>19.2e}{g:>13.2e}"
          f"{toff/g:>15.2e}")
print("  The crossover is not in doubt -- polynomial beats subexponential -- "
      "but it\n  arrives at a circuit size, not at a date.")

print("\nWhat error correction does to those numbers")
print("-" * 78)
print(f"  assumptions: physical error rate {P_PHYS:.0e}, threshold "
      f"{P_THRESH:.0e}, cycle time {CYCLE:.0e} s")
print(f"  {'n bits':>8}{'Toffolis':>12}{'code distance':>15}"
      f"{'physical qubits':>18}   wall clock")
for n_bits in [(21).bit_length(), 256, 1024, 2048]:
    logical = 3 * n_bits
    toff = TOFFOLI_COEF * n_bits ** 3
    d = surface_code_distance(max(toff, 10))
    phys = 2 * logical * d ** 2
    phys_total = 2.5 * phys          # +150% for magic-state factories
    seconds = toff * d * CYCLE       # one Toffoli ~ d cycles, serialized
    unit = ("s" if seconds < 3600 else
            "hours" if seconds < 3 * 86400 else "days")
    val = (seconds if unit == "s" else
           seconds / 3600 if unit == "hours" else seconds / 86400)
    print(f"  {n_bits:>8}{toff:>12.2e}{d:>15}{phys_total:>18.2e}"
          f"{val:>10.1f} {unit}")
print("  Read only the exponents. The number of physical qubits is millions "
      "and the\n  runtime is hours to days -- both several orders of magnitude "
      "beyond anything\n  that has been built, and both sensitive to every "
      "assumption listed above.")

print("\nThe gap, stated as a ratio")
print("-" * 78)
n_demo, n_rsa = 5, 2048
print(f"  N = 21 needed {2*n_demo+1 + n_demo} simulated qubits and "
      f"{2*n_demo+1} controlled modular multiplications.")
print(f"  ratio of Toffoli counts, {n_rsa} bits to {n_demo} bits: "
      f"{(n_rsa/n_demo)**3:.2e}")
print(f"  ratio of logical qubit counts: {n_rsa/n_demo:.0f}")
print(f"  ratio of state-vector memory if you tried to simulate it: "
      f"2^{3*n_rsa} amplitudes")
print("  A demonstration on 21 is a demonstration of the postprocessing, not "
      "of the\n  cryptanalysis. Nothing about it is evidence that the "
      "cryptanalysis is near --\n  and nothing about it is evidence that the "
      "algorithm is wrong, either.")

print("\nWhat a quantum computer does and does not break")
print("-" * 78)
rows = [
    ("RSA", "factoring", "Shor, polynomial", "broken in principle"),
    ("Diffie-Hellman, DSA", "discrete log mod p", "Shor, polynomial",
     "broken in principle"),
    ("Elliptic-curve DH/DSA", "discrete log on a curve", "Shor, polynomial",
     "broken in principle; smaller circuits than RSA"),
    ("AES-128 / AES-256", "unstructured key search", "Grover, quadratic",
     "key length doubles, then safe"),
    ("SHA-2 / SHA-3 preimage", "unstructured search", "Grover, quadratic",
     "output length doubles, then safe"),
    ("Lattice problems (LWE)", "shortest/closest vector", "sieving: 2^0.265n",
     "the basis of the standard replacements"),
    ("Hash-based signatures", "preimage resistance", "Grover, quadratic",
     "parameter bump, then safe"),
]
print(f"  {'primitive':<24}{'hard problem':<26}{'quantum attack':<22}"
      f"status")
for a, b, c, d_ in rows:
    print(f"  {a:<24}{b:<26}{c:<22}{d_}")
print("\n  Two rows need their fine print. Collision resistance is NOT "
      "quadratically\n  sped up: the classical birthday attack already costs "
      "2^(n/2), and the best\n  known quantum collision finder (BHT) reaches "
      "2^(n/3) while needing 2^(n/3)\n  quantum memory, so hash-based "
      "signature parameters are set by preimage and\n  second-preimage "
      "resistance, where Grover does apply. And lattices are not "
      "'no\n  known speedup': heuristic sieving improves from 2^(0.292n) "
      "classically to about\n  2^(0.265n) quantumly. That is a change in the "
      "exponent's constant, absorbed by a\n  modest increase in dimension, "
      "not the polynomial-versus-exponential change Shor\n  gives for "
      "factoring.")
print("\n  The pattern: Shor needs an abelian group with a hidden period. "
      "Where the\n  hardness is unstructured search instead, only Grover "
      "applies, and a quadratic\n  speedup is answered by doubling a key "
      "length. Lattice problems fall in neither\n  box, which is why "
      "lattice-based key exchange and signatures are the standard\n  "
      "post-quantum replacements -- a migration that is engineering, "
      "already specified,\n  and independent of when or whether a "
      "cryptographically useful quantum computer\n  is built.")
```

```text
Circuit size against modulus size (this chapter's own accounting)
------------------------------------------------------------------------------
    n bits  logical qubits   Toffolis ~0.3n^3     GNFS ops   quantum/GNFS
         5              15           3.75e+01     2.89e+01       1.30e+00
        32              96           9.83e+03     9.73e+04       1.01e-01
       256             768           5.03e+06     1.12e+14       4.51e-08
      1024            3072           3.22e+08     1.32e+26       2.45e-18
      2048            6144           2.58e+09     1.53e+35       1.68e-26
      4096           12288           2.06e+10     1.29e+47       1.60e-37
  The crossover is not in doubt -- polynomial beats subexponential -- but it
  arrives at a circuit size, not at a date.

What error correction does to those numbers
------------------------------------------------------------------------------
  assumptions: physical error rate 1e-03, threshold 1e-02, cycle time 1e-06 s
    n bits    Toffolis  code distance   physical qubits   wall clock
         5    3.75e+01              3          6.75e+02       0.0 s
       256    5.03e+06             13          6.49e+05      65.4 s
      1024    3.22e+08             17          4.44e+06       1.5 hours
      2048    2.58e+09             19          1.11e+07      13.6 hours
  Read only the exponents. The number of physical qubits is millions and the
  runtime is hours to days -- both several orders of magnitude beyond anything
  that has been built, and both sensitive to every assumption listed above.

The gap, stated as a ratio
------------------------------------------------------------------------------
  N = 21 needed 16 simulated qubits and 11 controlled modular multiplications.
  ratio of Toffoli counts, 2048 bits to 5 bits: 6.87e+07
  ratio of logical qubit counts: 410
  ratio of state-vector memory if you tried to simulate it: 2^6144 amplitudes
  A demonstration on 21 is a demonstration of the postprocessing, not of the
  cryptanalysis. Nothing about it is evidence that the cryptanalysis is near --
  and nothing about it is evidence that the algorithm is wrong, either.

What a quantum computer does and does not break
------------------------------------------------------------------------------
  primitive               hard problem              quantum attack        status
  RSA                     factoring                 Shor, polynomial      broken in principle
  Diffie-Hellman, DSA     discrete log mod p        Shor, polynomial      broken in principle
  Elliptic-curve DH/DSA   discrete log on a curve   Shor, polynomial      broken in principle; smaller circuits than RSA
  AES-128 / AES-256       unstructured key search   Grover, quadratic     key length doubles, then safe
  SHA-2 / SHA-3 preimage  unstructured search       Grover, quadratic     output length doubles, then safe
  Lattice problems (LWE)  shortest/closest vector   sieving: 2^0.265n     the basis of the standard replacements
  Hash-based signatures   preimage resistance       Grover, quadratic     parameter bump, then safe

  Two rows need their fine print. Collision resistance is NOT quadratically
  sped up: the classical birthday attack already costs 2^(n/2), and the best
  known quantum collision finder (BHT) reaches 2^(n/3) while needing 2^(n/3)
  quantum memory, so hash-based signature parameters are set by preimage and
  second-preimage resistance, where Grover does apply. And lattices are not 'no
  known speedup': heuristic sieving improves from 2^(0.292n) classically to about
  2^(0.265n) quantumly. That is a change in the exponent's constant, absorbed by a
  modest increase in dimension, not the polynomial-versus-exponential change Shor
  gives for factoring.

  The pattern: Shor needs an abelian group with a hidden period. Where the
  hardness is unstructured search instead, only Grover applies, and a quadratic
  speedup is answered by doubling a key length. Lattice problems fall in neither
  box, which is why lattice-based key exchange and signatures are the standard
  post-quantum replacements -- a migration that is engineering, already specified,
  and independent of when or whether a cryptographically useful quantum computer
  is built.
```

**What to look for.** The first table shows the *shape* of the two costs, and its last column should not be read as locating a crossover. The quantum side carries an optimised constant — $0.3n^3$ Toffolis, the product of a decade of arithmetic-circuit work — while the classical side is the bare asymptotic form of the sieve with its constant set to 1, and the sieve's true constant is not 1. What the table does establish is that the ratio falls by twenty-five orders of magnitude between 32 and 2048 bits: one function is polynomial and the other is subexponential, and no choice of constants changes that. Where the two curves actually cross is a question about constants that neither column pins down, and it says nothing about when either side can be run.

The second table is what error correction does. Reaching $2.6\times10^9$ Toffolis at a physical error rate of $10^{-3}$ needs code distance 19, hence $10^7$ physical qubits and about half a day of runtime on the stated assumptions. Every number in that row is a consequence of the four constants declared at the top of the file, and changing the physical error rate by one decade moves $d$ by roughly a factor of two — 19 at $p = 10^{-3}$, 9 at $p = 10^{-4}$ — and therefore the physical qubit count, which goes as $d^2$, by roughly a factor of 4.5. This is the sense in which "how far away is it" is a materials and engineering question rather than an algorithms question: the algorithm has been finished for decades.

The third block states the gap as ratios, and the fourth is the summary a reader should leave with. Shor breaks the three primitives whose hardness is an abelian hidden subgroup problem. Grover halves the effective key length of everything whose hardness is unstructured search, which is repaired by doubling a parameter. Lattice problems are attacked by neither mechanism at more than an exponent-shaving level — quantum sieving $2^{0.265n}$ against classical $2^{0.292n}$ — which is why they are the replacement. None of those three statements depends on a date.

* * *

## Exercises

#### Exercise 1: The Reduction by Hand

Take $N = 33$ and $a = 5$.

  1. Compute the order $r$ of 5 modulo 33 by hand or by repeated multiplication.
  2. Apply the reduction: is $r$ even, what is $a^{r/2} \bmod N$, and what do the two gcds give?
  3. Repeat for $a = 10$ and for $a = 32$. Classify each outcome by the two failure modes of Section 3.1.
  4. $33 = 3 \times 11$ has $k = 2$ distinct odd prime factors, so the theorem promises a success probability of at least $1 - 2^{-1} = 1/2$. Count the coprime bases and the successful ones, and compare.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(5^1 = 5\), \(5^2 = 25\), \(5^3 = 125 = 26\), \(5^4 = 130 = 31\), \(5^5 = 155 = 23\), \(5^6 = 115 = 16\), \(5^7 = 80 = 14\), \(5^8 = 70 = 4\), \(5^9 = 20\), \(5^{10} = 100 = 1\). So \(r = 10\).</p>

<p><strong>2.</strong> \(r\) is even; \(5^5 = 23 \bmod 33\), which is neither 1 nor 32. \(\gcd(22, 33) = 11\) and \(\gcd(24, 33) = 3\): both nontrivial, and \(33 = 3 \times 11\).</p>

<p><strong>3.</strong> \(a = 10\): \(10^2 = 100 = 1\), so \(r = 2\) and \(10^1 = 10\); \(\gcd(9,33) = 3\), \(\gcd(11,33) = 11\) — success. \(a = 32 \equiv -1\): \(r = 2\) and \(a^{r/2} = 32 \equiv -1\), the second failure mode, both gcds trivial. Neither of these two exhibits the first failure mode, odd \(r\); for \(N = 33\) that mode first appears at \(a = 4\), whose order is 5, and part 4 counts every instance of both.</p>

<p><strong>4.</strong> There are \(\phi(33) = 20\) coprime residues, of which 19 lie in \(1 < a < 33\). Enumerating them, ten succeed and nine fail, and both failure modes are represented. The ten successes are \(a = 5, 7, 10, 13, 14, 19, 20, 23, 26, 28\), of order 10 except for \(a = 10\) and \(a = 23\), which have order 2. Five fail with \(a^{r/2} \equiv -1\): \(a = 2, 8, 17, 29\) at order 10 and \(a = 32\) at order 2. Four fail with odd order: \(a = 4, 16, 25, 31\), all of order \(r = 5\).</p>

<p>So the measured rate over \(1 < a < 33\) is \(10/19 = 0.526\). The theorem's own sample space is \(a\) drawn uniformly from the integers coprime to \(N\), which includes \(a = 1\); \(a = 1\) has order 1, which is odd, so it fails, and the rate over \(\mathbb{Z}_{33}^{\ast}\) is exactly \(10/20 = 0.5000\). <strong>The bound is tight here, not conservative</strong> — and the same is true of \(N = 21\), where the count is \(6/12 = 0.5000\). For \(k = 2\) the theorem promises \(\ge 1 - 2^{-1}\) and these two moduli attain it exactly; the slack people expect from "at least a half" is a property of moduli with more prime factors, not of this bound. The systematic way to check is to reuse Code Example 2 with \(N = 33\), which requires no changes to the code.</p>

</details>

#### Exercise 2: Why $t = 2n + 1$

The counting register is twice as long as the work register plus one bit.

  1. Show that if $2^t \ge N^2$ then $1/2^{t+1} < 1/(2r^2)$ for every $r < N$.
  2. The continued-fraction uniqueness theorem says that if $\lvert x - s/r \rvert < 1/(2r^2)$ with $\gcd(s,r) = 1$ then $s/r$ is a convergent of $x$. Combine this with part 1 to justify $t = 2n+1$.
  3. What goes wrong if $t = n$? Estimate the probability that two different fractions with denominators below $N$ lie within $2^{-n-1}$ of each other.
  4. Code Example 5 succeeds at $t = 6$ for $N = 21$, where the theorem asks for $t = 11$. Why is that not a contradiction?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(2^t \ge N^2 > r^2\) for \(r < N\), so \(1/2^{t+1} \le 1/(2N^2) < 1/(2r^2)\).</p>

<p><strong>2.</strong> Phase estimation guarantees \(\lvert k/2^t - s/r \rvert \le 2^{-(t+1)}\) for the best outcome, which by part 1 is below \(1/(2r^2)\); the theorem then puts \(s/r\) among the convergents of \(k/2^t\), which the Euclidean algorithm enumerates in \(O(n)\) steps. \(t = 2n+1\) is the smallest register that guarantees \(2^t \ge N^2\) for all \(n\)-bit \(N\).</p>

<p><strong>3.</strong> With \(t = n\) the guarantee is only \(\lvert k/2^n - s/r\rvert \le 2^{-(n+1)} \approx 1/(2N)\), and there are \(\Theta(N^2)\) fractions with denominator below \(N\) in \([0,1)\), so the typical spacing is \(\Theta(1/N^2)\) — far smaller than the error bar. Many candidate fractions fit, and the reconstruction is not unique. The algorithm still works sometimes, which is exactly part 4.</p>

<p><strong>4.</strong> The theorem is a worst-case guarantee over all \(r < N\). For a specific small \(r\) — here \(r = 6\) — the fractions \(s/6\) are widely separated, so a much coarser register resolves them. \(t = 2n+1\) is what you must choose when \(r\) is unknown and could be as large as \(N\); it is not what a particular instance needs.</p>

</details>

#### Exercise 3: Reading the $N = 21$ Distribution

Use the $t = 11$ output of Code Example 5.

  1. The six peaks hold $0.7893$ of the probability but the success probability is $0.8321$. Where does the extra $0.043$ come from?
  2. The ceiling as $t \to \infty$ is $5/6$. Derive it.
  3. For $a = 8$ (order 2) the success probability is exactly $0.5$ at every $t$. Explain why it is exactly one half and not approximately.
  4. Suppose you may run the circuit twice and combine the results. Describe a postprocessing rule that raises the per-experiment success probability above $5/6$, and estimate the new value for $a = 2$, $N = 21$.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> From the shoulders. An outcome \(k\) one or two bins away from a peak still has \(k/2^t\) closer to the same \(s/r\) than to any other fraction with denominator below 21, so its continued-fraction expansion returns the same convergent. The postprocessing has a capture radius wider than one bin, and that radius is what the extra probability lives in.</p>

<p><strong>2.</strong> The input \(\lvert 1 \rangle\) is the uniform superposition of the \(r\) eigenvectors \(\lvert u_s \rangle\), so \(s\) is uniform on \(\lbrace 0,\ldots,r-1\rbrace\). The outcome \(s = 0\) puts the peak at \(k = 0\) and yields no information about \(r\), so the ceiling is \(1 - 1/r = 5/6\) for \(r = 6\). Nothing about \(t\) changes this.</p>

<p><strong>3.</strong> With \(r = 2\) the eigenphases are \(0\) and \(1/2\), both exactly representable in any \(t \ge 1\) bits, so the distribution is exactly two delta peaks at \(k = 0\) and \(k = 2^{t-1}\), each with probability exactly \(1/2\). There is no leakage and therefore no \(t\) dependence — and the ceiling \(1 - 1/r\) is attained.</p>

<p><strong>4.</strong> Take the two measured denominators \(q_1, q_2\) and test \(\mathrm{lcm}(q_1, q_2)\) as well as each separately. Failure now requires both runs to be uninformative, i.e. both to give \(s = 0\), or the lcm to still miss \(r\). With \(P(s = 0) = 1/6\) per run the probability of both failing that way is \(1/36\), giving a success probability of about \(0.97\) for two runs — at twice the circuit cost. This is the standard remedy and it is why "the algorithm succeeds with probability at least a half" is not a practical limitation.</p>

</details>

#### Exercise 4: The Cost Is the Arithmetic

Use the scaling model of Code Example 3.

  1. For $n = 1025$ — the $\sim 2^{1024}$ row of Code Example 3, where $t = 2n+1 = 2051$ — compute the ratio of modular-exponentiation gates to inverse-QFT gates under the $n^3$ versus $t^2/2$ model, and state how it scales with $n$.
  2. Suppose someone improves the QFT by a factor of 100. By how much does the total circuit shrink at $n = 1024$?
  3. Suppose instead someone improves the modular multiplier from $n^2$ to $n^{1.6}$ gates (Karatsuba-style). By how much does the total shrink?
  4. What does this imply about which part of a factoring circuit deserves optimization effort, and about how to read a paper title containing "efficient QFT"?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Arithmetic \(\approx t n^2 = 2051 \times 1025^2 = 2.15\times10^9\); QFT \(\approx t(t+1)/2 = 2.1\times10^6\). Ratio \(\approx 1020\), and it grows as \(n^3/n^2 = n\), so the disparity widens with size.</p>

<p><strong>2.</strong> The QFT falls from \(2.1\times10^6\) to \(2.1\times10^4\), changing the total from \(2.152\times10^9\) to \(2.150\times10^9\) — a saving of one part in a thousand.</p>

<p><strong>3.</strong> \(t n^{1.6} = 2051 \times 1025^{1.6} = 2051 \times 6.56\times10^4 = 1.35\times10^8\), a factor of 16 off the total. Two orders of magnitude of QFT improvement is worth less than a modest improvement in the multiplier.</p>

<p><strong>4.</strong> All the effort belongs in the arithmetic — which is where it has in fact gone, through windowed arithmetic, better adders and Toffoli-count reduction. A paper about a more efficient QFT is a paper about the negligible term of a factoring circuit; it may matter for other algorithms, where the QFT is not accompanied by \(n^3\) of arithmetic, but not for this one.</p>

</details>

#### Exercise 5: Auditing a Claim

A press release states: "Our processor has factored a 48-bit number using Shor's algorithm, a record. Extrapolating, RSA-2048 is within reach."

  1. Using Code Example 3's model, what circuit size does a genuine 48-bit factorization require, in qubits and in modular multiplications?
  2. Name three ways a demonstration can produce the right factors without running that circuit, and say what published detail would let you rule each one out.
  3. Take the $n^3$ scaling at face value: what is the ratio of circuit sizes between 48 bits and 2048 bits, and what does "within reach" have to mean for the extrapolation to hold?
  4. Write the two sentences you would want the release to contain instead.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(n = 48\), \(t = 97\), so \(t + n = 145\) logical qubits, 97 controlled modular multiplications, and \(\sim t n^2 = 2.2\times10^5\) gates in the arithmetic. With error correction at the assumptions of Example 7 that is a few thousand physical qubits at minimum, and every gate must succeed coherently.</p>

<p><strong>2.</strong> (i) The base \(a\) was chosen so that \(\gcd(a,N) > 1\) or so that \(r\) is tiny and known, which collapses the circuit — ruled out by publishing the base, the order, and the full measured distribution. (ii) The modular exponentiation was compiled <em>using</em> the known factorization, which is the standard criticism of small Shor demonstrations — ruled out by publishing a circuit that depends only on \(N\) and \(a\). (iii) The postprocessing did the work: with \(N\) small, testing a few candidate \(r\) classically finds the factors without any quantum information — ruled out by reporting the per-run success probability against the theoretical value, as Code Examples 4 and 5 do.</p>

<p><strong>3.</strong> \((2048/48)^3 = 7.8\times10^4\) in gate count and \(2048/48 = 43\) in qubit count, before error correction; after it, the physical-qubit ratio is larger still because the code distance grows with the circuit size. "Within reach" would have to mean that five orders of magnitude in coherent gate count and four in physical qubits are incremental, which no reading of the hardware supports.</p>

<p><strong>4.</strong> Something like: "We factored a 48-bit integer with a circuit that uses only \(N\) and the randomly chosen base as input, and the measured per-run success probability agrees with the theoretical \(1 - 1/r\) to within sampling error." And: "Scaling the same circuit to 2048 bits requires \(\sim 10^9\) Toffoli gates and, under standard surface-code assumptions, \(\sim 10^7\) physical qubits; this demonstration is not evidence about that regime." The first sentence is what makes the result checkable; the second is what makes it honest.</p>

</details>

* * *

## Summary

### Key Takeaways

**1\. Factoring reduces to order finding, classically**

  * With $r$ the order of $a$ modulo $N$ and $r$ even, $\gcd(a^{r/2}\pm1, N)$ are proper divisors unless $a^{r/2}\equiv-1$.
  * Failures are detected in microseconds, so the algorithm redraws the base; the per-base success probability is at least $1/2$ for an $N$ with two distinct odd prime factors, and the enumeration gave $6/7$ for $N = 15$ and $6/11$ for $N = 21$.
  * Even $N$ and perfect powers never reach the circuit, and a lucky $\gcd(a,N) > 1$ short-circuits it entirely — which for two-digit $N$ happens in most runs.

**2\. Order finding is phase estimation on a permutation**

  * $U_a\lvert y\rangle = \lvert ay \bmod N\rangle$ has eigenphases $s/r$, and the computational state $\lvert 1 \rangle$ is the uniform superposition of all $r$ eigenvectors — so one run returns a uniformly random $s$, and $s = 0$ is the useless outcome of probability $1/r$.
  * The ceiling on per-run success of the *order-finding circuit* is therefore $1 - 1/r$, attained exactly when $r$ divides $2^t$. The per-round success of the *factoring algorithm* is a different quantity, built in Example 6 out of this one plus the lucky-gcd and bad-base cases.
  * Continued fractions convert the measured $k/2^t$ into $s/r$ in lowest terms; the register size $t = 2n+1$ is what makes that reconstruction unique.

**3\. The Fourier transform is the cheap part**

  * $2n+1$ controlled modular multiplications at $O(n^2)$ gates each give $O(n^3)$; the inverse QFT is $O(n^2)$. At $n = 2049$ the ratio is two thousand to one.
  * Consequently the approximate QFT is free to use, and all optimization effort belongs in the arithmetic.

**4\. Both integers factor, and the statistics match theory**

  * $N = 15$, $a = 7$: four exact peaks, per-run success $3/4$ against a measured $0.759$ over 2000 shots; $a = 14$ gives exactly zero because $a^{r/2} \equiv -1$.
  * $N = 21$, $a = 2$: peaks near $s/6$, only $78.9\%$ of the probability on them, yet success $0.8321$ — the shoulders postprocess correctly, and the ceiling $5/6$ is approached from below as $t$ grows.
  * 600 complete runs of the full algorithm succeeded every time, at a mean of $0.75$ and $0.87$ quantum calls.

**5\. The cryptographic conclusion, in both directions**

  * The separation is superpolynomial and settled; there is nonetheless no proof that factoring is classically hard.
  * From $p = 10^{-3}$, a $10^{-2}$ threshold and a $1\ \mu$s cycle: $n = 2048$ needs $2.6\times10^9$ Toffolis, code distance 19, $\sim 10^7$ physical qubits and hours of runtime. Read the exponents.
  * Shor solves the abelian hidden subgroup problem, so RSA, finite-field and elliptic-curve discrete logs fall in principle. Grover's quadratic speedup against symmetric primitives is answered by doubling a parameter. Lattice problems are attacked by neither at more than the level of an exponent constant, which is why they are the standard replacement — and recorded ciphertext is why the migration cannot wait for a demonstration.

**Practical implications**

  * When you read a Shor demonstration, ask for the base, the measured distribution, and the per-run success probability against $1 - 1/r$. Those three make it checkable.
  * When you read a resource estimate, ask for the Toffoli count, the assumed physical error rate, and the code distance. A physical-qubit number without those is not an estimate.
  * When you plan for post-quantum migration, separate confidentiality from authenticity: only the first is threatened retroactively.

### Where This Leads

Chapter 3 is the high-water mark of provable quantum advantage in this course, and it is worth noticing what made it possible: a problem with a hidden *group* structure, an exact periodicity, and a classical postprocessing step that turns one measurement into an answer. Chapter 4 turns to the application with the strongest physical motivation and no such structure — simulating a Hamiltonian — where the object of study is $e^{-iHt}$ itself. The techniques there (block encoding, qubitization, randomized compilation) are the modern replacements for the Trotter decomposition of the introductory course, and they are also what supplies the controlled $e^{-iH\tau}$ that Chapter 2's phase estimation assumed for free.

[← Chapter 2: QFT and Phase Estimation](<chapter-2.html>) [Chapter 4: Modern Hamiltonian Simulation →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The resource estimates in this chapter — gate counts, code distances, physical-qubit numbers and runtimes — follow from the modelling assumptions stated in the code and are order-of-magnitude teaching figures, not predictions or measurements. Nothing here is cryptographic advice; consult current standards and primary sources before making a security decision.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
