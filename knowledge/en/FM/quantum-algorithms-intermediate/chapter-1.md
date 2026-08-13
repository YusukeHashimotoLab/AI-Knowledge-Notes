---
title: "Chapter 1: Amplitude Amplification and Grover's Algorithm"
chapter_title: "Chapter 1: Amplitude Amplification and Grover's Algorithm"
subtitle: The Oracle Model, Two Reflections, and What a Quadratic Speedup Is Actually Worth
reading_time: 45-50 minutes
difficulty: Intermediate
code_examples: 7
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-algorithms-intermediate/chapter-1.html>) | Last sync: 2026-08-13

[Fundamental Mathematics Dojo](<../index.html>) > [Intermediate Quantum Algorithms](<index.html>) > Chapter 1

Grover's algorithm is the most widely quoted and the most widely misunderstood result in quantum computing. The statement is short: given a function $f$ that recognizes a solution, you can find a solution among $N$ candidates using $O(\sqrt{N})$ evaluations of $f$, where any classical method needs $\Omega(N)$. The statement is also *true*, provably, and provably optimal — nothing better exists. And yet almost every popular rendering of it is wrong, usually in the same two places: it is described as searching a database, which it does not do, and its quadratic advantage is presented as though a quadratic advantage were automatically worth having, which it is not.

This chapter does three things. It builds the algorithm properly, as a rotation in a two-dimensional plane, which makes the optimal iteration count a piece of trigonometry rather than a formula to memorize. It generalizes to **amplitude amplification**, which is the form the technique actually takes when it appears inside other algorithms — including several in the later chapters of this course. And it does the arithmetic that the popular account skips: what happens to a quadratic speedup once you count the cost of the oracle, the ratio of quantum to classical clock rates, the error-correction overhead, and the fact that classical brute force parallelizes perfectly while Grover does not.

The order is deliberate. Section 1.1 fixes the oracle model, because every claim in the rest of the chapter is a claim *within* a model, and the model is where the assumptions are hidden. Sections 1.2 and 1.3 do the mathematics. Section 1.4 leaves the model and asks what survives. Section 1.5 runs the whole thing on the simulator from the introductory course, for $n = 4$ to $10$ qubits, with the optimal iteration count verified numerically and the awkward cases — several solutions, an unknown number of solutions — handled rather than assumed away.

## Learning Objectives

After completing this chapter, you will be able to:

  * State the query model precisely, distinguish a bit-flip oracle from a phase oracle, and explain the phase-kickback identity that relates them
  * Explain what the model assumes — a unitary, reversible, coherent evaluation of $f$ at unit cost — and demonstrate, by counting gates in an explicit 3-SAT oracle, how much that assumption hides
  * Derive Grover's iteration as a product of two reflections, show that it is a rotation by $2\theta$ with $\sin\theta = \sqrt{M/N}$, and obtain both the closed-form success probability and the optimal iteration count
  * Explain over-rotation, and why running a Grover search "for longer" makes it worse
  * Generalize to amplitude amplification around an arbitrary state preparation $A$, and state the two conditions ($A$ reversible, the good subspace recognizable) that the generalization needs
  * Quantify the four separate mechanisms — oracle cost, clock rate, imperfect parallelism, and the base of the comparison — by which a quadratic speedup is consumed, and explain why unstructured search is not database search
  * Implement the full algorithm on a state-vector simulator, verify $\lfloor \pi/(4\theta) \rfloor$ numerically for $n = 4$ to $10$, and handle several or an unknown number of solutions

### Conventions Carried Over

Three conventions come from [Introduction to Quantum Computing](<../quantum-computing-introduction/index.html>) unchanged, and this course never varies them.

**Big-endian qubit ordering.** Qubit 0 is the leftmost symbol in a ket and the most significant bit of the amplitude index, so on three qubits $\lvert 000 \rangle, \lvert 001 \rangle, \ldots, \lvert 111 \rangle$ occupy indices $0$ through $7$. Much of the Qiskit literature uses the opposite convention, and a mismatch is the most common source of silently wrong results in this subject.

**The simulator.** Every example runs on the ninety-nine-line state-vector simulator of that course's Chapter 2. Code Example 1 re-lists the functions this chapter needs, verbatim.

**Queries versus gates.** A *query* is one application of the oracle. A *gate* is one application of something a machine can do. Complexity statements in the oracle model are counted in queries; statements about machines are counted in gates. The whole of Section 1.4 is about the exchange rate between the two.

* * *

## 1.1 The Oracle Model: What Is Assumed, and What Is Not

### The model, stated precisely

Fix $n$, write $N = 2^n$, and let

$$ f : \lbrace 0, 1 \rbrace^n \to \lbrace 0, 1 \rbrace $$

be a function with $M = \lvert f^{-1}(1) \rvert$ solutions. The **search problem** is: find some $x$ with $f(x) = 1$. The **query model** grants access to $f$ only through a unitary. There are two standard forms. The **bit-flip oracle** acts on the search register and one extra qubit,

$$ O_f \lvert x \rangle \lvert y \rangle = \lvert x \rangle \lvert y \oplus f(x) \rangle $$

and the **phase oracle** acts on the search register alone,

$$ O \lvert x \rangle = (-1)^{f(x)} \lvert x \rangle $$

The two are the same object. Setting the extra qubit to $\lvert - \rangle = (\lvert 0 \rangle - \lvert 1 \rangle)/\sqrt{2}$ gives

$$ O_f \lvert x \rangle \lvert - \rangle = \lvert x \rangle \frac{\lvert f(x) \rangle - \lvert 1 \oplus f(x) \rangle}{\sqrt{2}} = (-1)^{f(x)} \lvert x \rangle \lvert - \rangle $$

so the extra qubit is left untouched and the whole effect is a phase on the search register. This is **phase kickback**, and it is worth recognizing now because the same trick is the mechanism of phase estimation in Chapter 2 and of Shor's algorithm in Chapter 3. From here on "oracle" means the phase form.

The cost measure is the **query complexity**: how many applications of $O$ an algorithm needs, with everything else free. Grover's theorem is a statement in exactly this measure, and it comes with a matching lower bound: any quantum algorithm that finds a marked item with constant probability needs $\Omega(\sqrt{N/M})$ queries. That bound — proved by Bennett, Bernstein, Brassard and Vazirani, and independently by the polynomial and adversary methods — is what makes Grover's algorithm *optimal*. There is no cleverer quantum search waiting to be found.

### What the model assumes

Four assumptions are packed into "grants access to $f$ through a unitary", and each one is a real constraint.

**1. $f$ is computed reversibly and coherently.** A classical circuit for $f$ discards intermediate results; a quantum one cannot. Every intermediate must be computed into an ancilla and then *uncomputed*, so that the ancillas are returned to $\lvert 0 \rangle$ and the oracle acts as a clean diagonal on the search register. If an ancilla is left dirty, it carries which-path information, the interference the algorithm depends on is destroyed, and the algorithm silently degrades. Uncomputation roughly doubles the gate count, and Code Example 2 verifies that the ancillas do come back.

**2. One query costs one unit.** This is the assumption that does the most work and receives the least scrutiny. For a real $f$ — a SAT formula, a hash function, a physical simulation — a query is a circuit of hundreds to millions of gates, and the *same* circuit must be run classically to check a candidate. The quadratic advantage in *queries* is therefore inherited by the *gate* count only if the two sides pay the same per-query cost, which they often do not, because the quantum side pays for reversibility, ancillas and error correction.

**3. $f$ is a rule, not a table.** Nothing in the model says where $f$ comes from. If $f$ is a short predicate — "this assignment satisfies the formula", "this key decrypts to plausible plaintext" — then it is a small circuit and the model is honest. If $f$ is a lookup into $N$ stored records, the circuit is not small, and Section 1.4 shows that this single change reverses the conclusion.

**4. There is no structure to exploit.** The lower bound is proved for an *arbitrary* $f$ presented as a black box. Real problems have structure, and classical algorithms use it. A SAT solver does not enumerate $2^n$ assignments, and the honest comparison for Grover on SAT is against the solver, not against enumeration.

### Three things the model does not give you

| The model says | The model does not say |
| --- | --- |
| $O$ is a unitary acting on superpositions | that $O$ is cheap, or that a circuit for it exists at all |
| a query costs one unit | how many gates that unit is worth |
| $\Theta(\sqrt{N/M})$ queries suffice and are necessary | that $M$ is known — and the iteration count depends on it |
| the search space has size $N$ | that anything of size $N$ was ever stored, or could be |

The third row deserves emphasis, because the optimal number of iterations depends on $M$ and is *wrong* if $M$ is guessed badly — over-rotation, in Section 1.2, is a real failure mode and not a curiosity. Code Example 7 handles the unknown-$M$ case properly.

### Code Example 1: The Simulator, Re-listed

This is the state-vector simulator from Chapter 2 of [Introduction to Quantum Computing](<../quantum-computing-introduction/index.html>) — the functions this chapter needs, verbatim. Save it as `qcsim.py`; every example below begins with `from qcsim import *`. Nothing here is new, and nothing here has been modified; if you have the file from the introductory course already, use that one.

```python
"""Minimal state-vector simulator (big-endian: qubit 0 = leftmost = most significant).

Save this file as qcsim.py; every later example does `from qcsim import *`.
"""
import numpy as np

# ---- single-qubit gates -------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


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
```

Two things about this listing are worth noticing before it is used. `apply_gate` takes a list of target qubits and a $2^k \times 2^k$ matrix, so a gate on any subset of qubits costs $O(2^k \cdot 2^n)$ operations and no extra memory — the reason $n = 10$ is comfortable and $n = 24$ would still be possible. And there is no oracle in it, because an oracle is not a gate: it is whatever circuit computes the problem's predicate, which is the subject of the next example.

### Code Example 2: The Same Oracle, Written Two Ways

The point of this example is to make Assumption 2 above visible. The first version of the oracle is the black box of the query model: it is handed the set of marked indices and flips their signs in one line. The second version is a circuit for a concrete 3-SAT formula, built out of Toffoli gates with ancillas and uncomputation, which is what a machine would have to run. They implement the same diagonal unitary, verified to machine precision — and one of them costs 19 Toffoli gates and 4 ancilla qubits while the other costs "one query, by definition".

```python
"""Chapter 1, Example 2: the same oracle written two ways.

A phase oracle is a diagonal unitary O|x> = (-1)^{f(x)}|x>. The first version
below is the black box of the query model: it is handed the marked set and costs
one line. The second is a circuit that evaluates a 3-SAT formula reversibly, and
it is the only one of the two that a machine could run.
"""
import numpy as np
from qcsim import *

COUNT = dict(toffoli=0, cnot=0, x=0, h=0)     # gates actually applied


# ---- version 1: the black box ------------------------------------------
def phase_oracle(state, marked):
    """Flip the sign of the listed basis-state indices. One query, by fiat."""
    psi = state.copy()
    psi[list(marked)] *= -1.0
    return psi


# ---- version 2: the same map, as a reversible circuit -------------------
TOFF8 = np.eye(8, dtype=complex)
TOFF8[[6, 7]] = TOFF8[[7, 6]]      # |110> <-> |111>: controls q0, q1, target q2


def toffoli(state, c0, c1, t, n):
    COUNT["toffoli"] += 1
    return apply_gate(state, TOFF8, [c0, c1, t], n)


def xgate(state, q, n):
    COUNT["x"] += 1
    return apply_gate(state, X, [q], n)


def hgate(state, q, n):
    COUNT["h"] += 1
    return apply_gate(state, H, [q], n)


def cz(state, c, t, n):
    """CZ = (I x H) CNOT (I x H); symmetric in its two qubits."""
    COUNT["cnot"] += 1
    return hgate(cnot(hgate(state, t, n), c, t, n), t, n)


def ccc_x(state, controls, t, work, n):
    """X on t controlled on three qubits, using one work qubit (left clean)."""
    a, b, c = controls
    state = toffoli(state, a, b, work, n)
    state = toffoli(state, c, work, t, n)
    return toffoli(state, a, b, work, n)


# A 3-SAT instance on four variables. Each clause is a list of (variable, sign),
# sign = +1 for a positive literal and -1 for a negated one.
CLAUSES = [[(0, +1), (1, -1), (2, +1)],
           [(0, -1), (1, +1), (3, +1)],
           [(1, +1), (2, -1), (3, -1)]]
FLAG, WORK, N_TOT = [4, 5, 6], 7, 8      # one flag per clause, plus a work bit


def sat_value(bits):
    """Classical evaluation of the formula on a bit tuple; 1 if satisfied."""
    for clause in CLAUSES:
        if not any((bits[v] == 1) if s > 0 else (bits[v] == 0) for v, s in clause):
            return 0
    return 1


def clause_flags(state, order):
    """Compute — or, run a second time, uncompute — one flag per clause."""
    for clause, flag in order:
        for v, s in clause:
            if s > 0:
                state = xgate(state, v, N_TOT)    # so 'all ones' means UNSAT
        state = ccc_x(state, [v for v, _ in clause], flag, WORK, N_TOT)
        for v, s in clause:
            if s > 0:
                state = xgate(state, v, N_TOT)
        state = xgate(state, flag, N_TOT)         # flag = 1 iff clause satisfied
    return state


def sat_phase_oracle(state):
    """Phase oracle for CLAUSES: |x>|0000> -> (-1)^{f(x)}|x>|0000>."""
    order = list(zip(CLAUSES, FLAG))
    state = clause_flags(state, order)
    state = hgate(state, FLAG[2], N_TOT)                    # CCZ on the flags
    state = toffoli(state, FLAG[0], FLAG[1], FLAG[2], N_TOT)
    state = hgate(state, FLAG[2], N_TOT)
    return clause_flags(state, order[::-1])       # ancillas back to |0000>


ALL_BITS = [tuple(int(b) for b in format(i, "04b")) for i in range(16)]
SOLUTIONS = [i for i, b in enumerate(ALL_BITS) if sat_value(b)]

print("3-SAT instance:  " + " AND ".join(
    "(" + " OR ".join(("" if s > 0 else "NOT ") + f"x{v}" for v, s in c) + ")"
    for c in CLAUSES))
print(f"classical truth table: {len(SOLUTIONS)} of 16 assignments satisfy it")
print("  " + " ".join(format(i, "04b") for i in SOLUTIONS))

psi = ket("0" * N_TOT)
for q in range(4):
    psi = apply_gate(psi, H, [q], N_TOT)
amp = sat_phase_oracle(psi).reshape(16, 16)      # (variables, ancillas)
diag = amp[:, 0] * np.sqrt(16.0)                # divide out the 1/sqrt(16)
bb = phase_oracle(np.ones(16) / 4.0, SOLUTIONS) * 4.0

print(f"\nancillas returned to |0000>?   max leakage = "
      f"{np.max(np.abs(amp[:, 1:])):.2e}")
print(f"circuit diagonal == black box? max error   = "
      f"{np.max(np.abs(diag.real - bb)):.2e}")
print("the 16 phases, in index order:")
print("  circuit  " + " ".join(f"{d.real:+.0f}" for d in diag))
print("  blackbox " + " ".join(f"{v:+.0f}" for v in bb))

print("\nCost of one query, counted rather than assumed to be 1:")
print(f"  Toffoli / CNOT / X / H   : {COUNT['toffoli']} / {COUNT['cnot']} / "
      f"{COUNT['x']} / {COUNT['h']}")
print(f"  ancilla qubits           : {len(FLAG) + 1}")
print(f"  CNOT-equivalent count    : "
      f"{6 * COUNT['toffoli'] + COUNT['cnot']}  (6 CNOTs per Toffoli)")
print(f"  black-box version        : 1 query, by definition")


# ---- the cheap special case: one marked string, already known -----------
def mcz_oracle(state, s):
    """Mark the single bit string s by a multi-controlled Z, as a circuit.

    nv = len(s) variables plus nv - 2 ancillas, which start and end in |0>.
    """
    nv = len(s)
    n = 2 * nv - 2
    anc = list(range(nv, n))
    for q, b in enumerate(s):
        if b == "0":
            state = xgate(state, q, n)
    state = toffoli(state, 0, 1, anc[0], n)                 # AND ladder up
    for i in range(2, nv - 1):
        state = toffoli(state, i, anc[i - 2], anc[i - 1], n)
    state = cz(state, anc[-1], nv - 1, n)                   # the phase itself
    for i in range(nv - 2, 1, -1):                          # ladder back down
        state = toffoli(state, i, anc[i - 2], anc[i - 1], n)
    state = toffoli(state, 0, 1, anc[0], n)
    for q, b in enumerate(s):
        if b == "0":
            state = xgate(state, q, n)
    return state


print("\nThe textbook special case: one marked string, as a multi-controlled Z.")
print(f"{'variables':>11}{'ancillas':>10}{'Toffolis':>10}{'CNOT-equiv':>12}"
      f"{'max error vs black box':>25}")
print("-" * 68)
for nv in [3, 4, 5, 6]:
    s = format(2 ** nv - 3, f"0{nv}b")
    n = 2 * nv - 2
    before = dict(COUNT)
    psi = ket("0" * n)
    for q in range(nv):
        psi = apply_gate(psi, H, [q], n)
    out = mcz_oracle(psi, s).reshape(2 ** nv, -1)
    want = np.ones(2 ** nv)
    want[int(s, 2)] = -1.0
    err = (np.max(np.abs(out[:, 0].real * np.sqrt(2.0 ** nv) - want))
           + np.max(np.abs(out[:, 1:])))
    nt = COUNT["toffoli"] - before["toffoli"]
    nc = COUNT["cnot"] - before["cnot"]
    print(f"{nv:>11}{nv - 2:>10}{nt:>10}{6 * nt + nc:>12}{err:>25.2e}")
print("  The Toffoli count grows as 2(n-2): linear, and cheap. But writing this")
print("  oracle down requires knowing the answer, which is not the problem.")
```

```text
3-SAT instance:  (x0 OR NOT x1 OR x2) AND (NOT x0 OR x1 OR x3) AND (x1 OR NOT x2 OR NOT x3)
classical truth table: 10 of 16 assignments satisfy it
  0000 0001 0010 0110 0111 1001 1100 1101 1110 1111

ancillas returned to |0000>?   max leakage = 5.90e-18
circuit diagonal == black box? max error   = 4.44e-16
the 16 phases, in index order:
  circuit  -1 -1 -1 +1 +1 +1 -1 -1 +1 -1 +1 +1 -1 -1 -1 -1
  blackbox -1 -1 -1 +1 +1 +1 -1 -1 +1 -1 +1 +1 -1 -1 -1 -1

Cost of one query, counted rather than assumed to be 1:
  Toffoli / CNOT / X / H   : 19 / 0 / 26 / 2
  ancilla qubits           : 4
  CNOT-equivalent count    : 114  (6 CNOTs per Toffoli)
  black-box version        : 1 query, by definition

The textbook special case: one marked string, as a multi-controlled Z.
  variables  ancillas  Toffolis  CNOT-equiv   max error vs black box
--------------------------------------------------------------------
          3         1         2          13                 3.33e-16
          4         2         4          25                 5.55e-16
          5         3         6          37                 4.44e-16
          6         4         8          49                 6.66e-16
  The Toffoli count grows as 2(n-2): linear, and cheap. But writing this
  oracle down requires knowing the answer, which is not the problem.
```

**What to look for.** The two oracles agree exactly: the 16 phases match, and the ancillas return to $\lvert 0000 \rangle$ to $6 \times 10^{-18}$, which is the check that matters, because an oracle that leaves its ancillas entangled with the search register is not a phase oracle at all and will not amplify. The cost line is the honest accounting: 19 Toffolis, 26 $X$ gates, 4 ancillas, and a CNOT-equivalent count of 114 for *one* query on a four-variable formula. Roughly half of those Toffolis are pure uncomputation — the price of reversibility, paid on every single query.

The closing table is the special case that textbooks show, and it is worth being clear about why it is misleading. Marking one known bit string $s$ needs a multi-controlled $Z$: $2(n-2)$ Toffolis and $n-2$ ancillas, linear in $n$ and genuinely cheap. But you can only write that circuit down if you already know $s$. The circuit is not an oracle for a search problem; it is an oracle for a problem you have already solved, useful for testing the algorithm and for nothing else. Every Grover demonstration in Section 1.5 uses exactly this kind of oracle, and it is honest to say so.

One further remark on the formula chosen here: it has 10 satisfying assignments out of 16. That is a very *dense* solution set, and Code Example 7 shows that Grover gains essentially nothing on it. Real 3-SAT instances near the hardness threshold have exponentially few solutions, which is the regime where the algorithm is interesting; the small instance is used here because its oracle fits in eight qubits and can be printed in full.

* * *

## 1.2 The Geometry of Grover's Algorithm

### Two states, and a plane

Everything follows from noticing that the algorithm only ever needs two states. Let

$$ \lvert \mathrm{good} \rangle = \frac{1}{\sqrt{M}} \sum_{f(x)=1} \lvert x \rangle, \qquad \lvert \mathrm{bad} \rangle = \frac{1}{\sqrt{N-M}} \sum_{f(x)=0} \lvert x \rangle $$

These are orthonormal. The uniform superposition produced by $H^{\otimes n} \lvert 0 \cdots 0 \rangle$ decomposes in them as

$$ \lvert s \rangle = \sqrt{\frac{M}{N}} \lvert \mathrm{good} \rangle + \sqrt{\frac{N-M}{N}} \lvert \mathrm{bad} \rangle = \sin\theta \lvert \mathrm{good} \rangle + \cos\theta \lvert \mathrm{bad} \rangle $$

which *defines* the angle

$$ \theta = \arcsin\sqrt{\frac{M}{N}} $$

For $M \ll N$ this is a small angle, $\theta \approx \sqrt{M/N}$, and the initial success probability is $\sin^2\theta = M/N$ — the probability of a random guess, as it must be.

### Two reflections make a rotation

Grover's iteration is $G = DO$, built from two operators that are both reflections.

The oracle $O = I - 2P_{\mathrm{good}}$, where $P_{\mathrm{good}}$ projects onto the marked subspace, flips the sign of the $\lvert \mathrm{good} \rangle$ component and leaves $\lvert \mathrm{bad} \rangle$ alone. Inside the plane, that is the reflection whose mirror line is the $\lvert \mathrm{bad} \rangle$ axis.

The **diffusion operator** is the reflection about $\lvert s \rangle$,

$$ D = 2 \lvert s \rangle \langle s \rvert - I = H^{\otimes n} \left( 2 \lvert 0 \cdots 0 \rangle \langle 0 \cdots 0 \rvert - I \right) H^{\otimes n} $$

so it is implemented with $2n$ Hadamards and one $n$-fold controlled phase — no knowledge of $f$ required, which is why $D$ is free in the query model. Acting on amplitudes, $D$ sends $a_x \mapsto 2\langle a \rangle - a_x$ where $\langle a \rangle$ is the mean amplitude; the name "inversion about the mean" comes from that form, and it is correct but less useful than the geometric one.

A product of two reflections in a plane, whose mirror lines meet at angle $\theta$, is a rotation by $2\theta$. Since $\lvert s \rangle$ makes angle $\theta$ with the $\lvert \mathrm{bad} \rangle$ axis, $G = DO$ rotates the state by $2\theta$ towards $\lvert \mathrm{good} \rangle$ on every application. Starting from $\lvert s \rangle$ at angle $\theta$:

$$ G^k \lvert s \rangle = \sin\left((2k+1)\theta\right) \lvert \mathrm{good} \rangle + \cos\left((2k+1)\theta\right) \lvert \mathrm{bad} \rangle $$

and the probability that a measurement returns a marked string is

$$ P_k = \sin^2\left((2k+1)\theta\right) $$

Three consequences are immediate and none of them require any further calculation. The state never leaves the plane, so a search over $N = 2^n$ amplitudes is really a problem in two dimensions — verified numerically in Code Example 3. The dynamics are periodic, not convergent. And the algorithm is *exactly* solvable: there is nothing asymptotic about the formula above.

### The optimal number of iterations

We want $(2k+1)\theta$ as close to $\pi/2$ as possible, so the ideal real-valued $k$ is

$$ k^\ast = \frac{\pi}{4\theta} - \frac{1}{2} $$

and the best integer is the nearest one to $k^\ast$, which — except when $\pi/(4\theta)$ is exactly an integer, where two counts tie — equals

$$ k_{\mathrm{opt}} = \left\lfloor \frac{\pi}{4\theta} \right\rfloor = \left\lfloor \frac{\pi}{4 \arcsin\sqrt{M/N}} \right\rfloor $$

In the small-angle regime $\theta \approx \sqrt{M/N}$ this becomes the familiar

$$ k_{\mathrm{opt}} \approx \left\lfloor \frac{\pi}{4} \sqrt{\frac{N}{M}} \right\rfloor $$

The two agree for $M = 1$ at every $n$ tested in Code Example 6, and they disagree once $M/N$ is not small, as Code Example 7 shows.

How good is the result at $k_{\mathrm{opt}}$? Rounding to an integer costs at most half a step of $2\theta$, so $\lvert (2k_{\mathrm{opt}}+1)\theta - \pi/2 \rvert \le \theta$, and therefore

$$ P_{k_{\mathrm{opt}}} \ge \cos^2\theta = 1 - \frac{M}{N} $$

The failure probability is at most $M/N$, which is the *initial* success probability — a pleasing symmetry, and the reason a single Grover run at $n = 10$ succeeds with probability better than $0.999$. For $M = 1$ this is $1 - 1/N$, and the residual failure can be removed entirely by verifying the measured string classically and repeating, which costs one extra query.

### Over-rotation, and why "more iterations" is not "better"

Because $P_k$ is periodic, running past $k_{\mathrm{opt}}$ rotates the state *away* from $\lvert \mathrm{good} \rangle$. At $k \approx 2k_{\mathrm{opt}}$ the state has turned through nearly $\pi$ and the success probability is back near zero: Code Example 6 measures $0.99995$ at $k = 12$ for $n = 8$ and $0.0059$ at $k = 24$. This is not a subtlety to be filed away. It means the iteration count is *part of the algorithm*, it depends on $M$, and if $M$ is unknown the count cannot be computed in advance. Section 1.5 handles that case; Chapter 2 gives the other answer, quantum counting, which uses phase estimation to measure $M$ first.

It also explains a structural difference from classical search that is easy to miss. A classical scan is *anytime*: stop it whenever you like and you have a valid partial search. Grover is not anytime. Stopping early or late gives a state whose measurement is close to useless, and there is no way to hedge.

### Code Example 3: The Two-Dimensional Rotation, Verified

The claims above are exact, so they should be verifiable to machine precision rather than approximately. This example does that: it tracks every amplitude for $n = 3$, confirms that the component of the state outside the plane spanned by $\lvert \mathrm{good} \rangle$ and $\lvert \mathrm{bad} \rangle$ stays at the $10^{-16}$ level, and then follows the full curve for $n = 6$ past its maximum and round to the far side.

```python
"""Chapter 1, Example 3: the two-dimensional rotation, verified.
Continues from Example 2 (same session)."""


def uniform(n):
    """The uniform superposition H^n|0...0>, the starting state of Grover."""
    psi = ket("0" * n)
    for q in range(n):
        psi = apply_gate(psi, H, [q], n)
    return psi


def diffuser(state, n):
    """D = H^n (2|0><0| - I) H^n = 2|s><s| - I, the reflection about |s>."""
    for q in range(n):
        state = apply_gate(state, H, [q], n)
    state = -state
    state[0] = -state[0]
    for q in range(n):
        state = apply_gate(state, H, [q], n)
    return state


def grover_step(state, marked, n):
    """One Grover iteration G = D O: one oracle query, then one diffusion."""
    return diffuser(phase_oracle(state, marked), n)


def subspace_coords(state, marked, n):
    """Components of the state on |good>, |bad>, and everything else."""
    N = 2 ** n
    M = len(marked)
    good = np.zeros(N)
    good[list(marked)] = 1.0 / np.sqrt(M)
    bad = np.ones(N) / np.sqrt(N - M)
    bad[list(marked)] = 0.0
    cg = float(np.vdot(good, state).real)
    cb = float(np.vdot(bad, state).real)
    residual = state - cg * good - cb * bad
    return cg, cb, float(np.max(np.abs(residual)))


# --- n = 3, one marked string: every amplitude, iteration by iteration -----
n, marked = 3, [int("101", 2)]
N, M = 2 ** n, len(marked)
theta = np.arcsin(np.sqrt(M / N))
print(f"n = {n}, N = {N}, M = {M}:  theta = arcsin(sqrt(M/N)) = {theta:.6f} rad"
      f" = {np.degrees(theta):.3f} deg")
print(f"marked string '101' = index {marked[0]}\n")

hdr = (f"{'k':>3}{'amp(101)':>12}{'amp(other)':>12}{'(2k+1)theta':>13}"
       f"{'sin((2k+1)th)':>15}{'P(marked)':>11}{'off-plane':>11}")
print(hdr)
print("-" * len(hdr))
psi = uniform(n)
for k in range(6):
    cg, cb, res = subspace_coords(psi, marked, n)
    other = [psi[i].real for i in range(N) if i not in marked]
    ang = (2 * k + 1) * theta
    print(f"{k:>3}{psi[marked[0]].real:>12.6f}{other[0]:>12.6f}{ang:>13.6f}"
          f"{np.sin(ang):>15.6f}{probs(psi)[marked[0]]:>11.6f}{res:>11.1e}")
    psi = grover_step(psi, marked, n)

# --- n = 6: the whole curve, including over-rotation ----------------------
n, marked = 6, [int("101101", 2)]
N, M = 2 ** n, len(marked)
theta = np.arcsin(np.sqrt(M / N))
k_opt = int(np.floor(np.pi / (4 * theta)))
print(f"\nn = {n}, N = {N}, M = {M}:  theta = {theta:.6f} rad, "
      f"floor(pi/(4 theta)) = {k_opt}")
print(f"the usual approximation floor(pi sqrt(N/M)/4) = "
      f"{int(np.floor(np.pi * np.sqrt(N / M) / 4))}")
print(f"\n{'k':>4}{'P(marked)':>12}{'sin^2((2k+1)theta)':>21}{'angle (deg)':>14}")
print("-" * 51)
psi = uniform(n)
record = []
for k in range(17):
    p = probs(psi)[marked[0]]
    ang = (2 * k + 1) * theta
    record.append((k, p))
    mark = "  <-- optimal" if k == k_opt else ""
    if k <= 8 or k in (12, 16):
        print(f"{k:>4}{p:>12.6f}{np.sin(ang) ** 2:>21.6f}"
              f"{np.degrees(ang):>14.3f}{mark}")
    psi = grover_step(psi, marked, n)

best_k = max(record, key=lambda r: r[1])[0]
print(f"\nargmax over k of the measured success probability : {best_k}")
print(f"floor(pi/(4 theta))                               : {k_opt}")
print(f"P at k = {k_opt}: {dict(record)[k_opt]:.6f}    "
      f"P at k = {k_opt + 1}: {dict(record)[k_opt + 1]:.6f}    "
      f"P at k = {2 * k_opt + 1}: {dict(record)[2 * k_opt + 1]:.6f}")
```

```text
n = 3, N = 8, M = 1:  theta = arcsin(sqrt(M/N)) = 0.361367 rad = 20.705 deg
marked string '101' = index 5

  k    amp(101)  amp(other)  (2k+1)theta  sin((2k+1)th)  P(marked)  off-plane
-----------------------------------------------------------------------------
  0    0.353553    0.353553     0.361367       0.353553   0.125000    5.6e-17
  1    0.883883    0.176777     1.084101       0.883883   0.781250    2.8e-17
  2    0.972272   -0.088388     1.806836       0.972272   0.945312    8.3e-17
  3    0.574524   -0.309359     2.529570       0.574524   0.330078    2.2e-16
  4   -0.110485   -0.375650     3.252304      -0.110485   0.012207    1.7e-16
  5   -0.740252   -0.254116     3.975038      -0.740252   0.547974    2.2e-16

n = 6, N = 64, M = 1:  theta = 0.125328 rad, floor(pi/(4 theta)) = 6
the usual approximation floor(pi sqrt(N/M)/4) = 6

   k   P(marked)   sin^2((2k+1)theta)   angle (deg)
---------------------------------------------------
   0    0.015625             0.015625         7.181
   1    0.134827             0.134827        21.542
   2    0.343895             0.343895        35.904
   3    0.591380             0.591380        50.265
   4    0.816377             0.816377        64.627
   5    0.963515             0.963515        78.988
   6    0.996586             0.996586        93.350  <-- optimal
   7    0.907449             0.907449       107.711
   8    0.718042             0.718042       122.073
  12    0.000071             0.000071       179.519
  16    0.702809             0.702809       236.965

argmax over k of the measured success probability : 6
floor(pi/(4 theta))                               : 6
P at k = 6: 0.996586    P at k = 7: 0.907449    P at k = 13: 0.057550
```

**What to look for.** For $n = 3$ the amplitude of the marked string equals $\sin((2k+1)\theta)$ to six digits at every step, including at $k = 4$ where both are *negative* — the state has rotated past $\lvert \mathrm{good} \rangle$ and the amplitude has changed sign, which no "amplitude grows towards the answer" description would predict. The off-plane column is the structural claim: it never exceeds $2.2 \times 10^{-16}$, so a problem in $2^n$ dimensions has been exactly reduced to one in two.

For $n = 6$ the closed form and the simulation agree at all seventeen iterations, and the shape of the curve is the whole story of the algorithm. It rises to $0.9966$ at $k = 6$, which is $\lfloor \pi/(4\theta) \rfloor$; it falls to $7.1 \times 10^{-5}$ at $k = 12$, where the accumulated angle is $179.5$ degrees; and it then comes back up. The measured argmax and the closed-form iteration count agree. Note also that $\theta = 0.1253$ rad at $n = 6$, so the small-angle approximation $\theta \approx \sqrt{M/N} = 0.125$ is already good to $0.3\%$ — which is why the two formulas for $k_{\mathrm{opt}}$ rarely differ in practice for a single marked item.

* * *

## 1.3 Amplitude Amplification in General

### Replacing $H^{\otimes n}$ by anything

Nothing in Section 1.2 used the fact that the starting state was uniform. The derivation needed only that the initial state lies in the plane spanned by a good and a bad direction, and that both reflections preserve that plane. So let $A$ be *any* unitary — a "state preparation" — and write $\lvert \psi_A \rangle = A \lvert 0 \cdots 0 \rangle$. Define

$$ a = \lVert P_{\mathrm{good}} \lvert \psi_A \rangle \rVert^2, \qquad \theta = \arcsin\sqrt{a} $$

and replace the diffusion operator by the reflection about $\lvert \psi_A \rangle$,

$$ R_A = 2 \lvert \psi_A \rangle \langle \psi_A \rvert - I = A \left( 2 \lvert 0 \cdots 0 \rangle \langle 0 \cdots 0 \rvert - I \right) A^{\dagger} $$

Then the **amplitude amplification** operator $Q = R_A O$ is again a rotation by $2\theta$ in the plane, and everything from Section 1.2 carries over verbatim:

$$ P_k = \sin^2\left((2k+1)\theta\right), \qquad k_{\mathrm{opt}} = \left\lfloor \frac{\pi}{4\theta} \right\rfloor \approx \frac{\pi}{4\sqrt{a}} $$

Grover's algorithm is the special case $A = H^{\otimes n}$, for which $a = M/N$. Nothing else changes.

| | Grover search | Amplitude amplification |
| --- | --- | --- |
| Preparation | $H^{\otimes n}$ | any unitary $A$ |
| Initial success amplitude | $a = M/N$ | $a = \lVert P_{\mathrm{good}} A \lvert 0 \rangle \rVert^2$ |
| Reflection | $D = 2\lvert s \rangle\langle s \rvert - I$ | $R_A = A(2\lvert 0 \rangle\langle 0 \rvert - I)A^{\dagger}$ |
| Cost per iteration | 1 oracle, $2n$ Hadamards | 1 oracle, one $A$, one $A^{\dagger}$ |
| Iterations | $\approx \frac{\pi}{4}\sqrt{N/M}$ | $\approx \frac{\pi}{4}/\sqrt{a}$ |

### Why the general form is the one that matters

Read the last row again. The iteration count depends on the *success probability of the preparation*, not on the size of any search space. That turns amplitude amplification into a general-purpose wrapper: given any quantum subroutine that produces a right answer with probability $a$ and a way to recognize a right answer, you can boost the probability to near one using $\Theta(1/\sqrt{a})$ repetitions where classical repetition needs $\Theta(1/a)$. This is the honest general statement of "quadratic speedup", and it is why the technique appears far more often as a subroutine than as a search algorithm in its own right. Quantum counting (Chapter 2), quantum mean and amplitude estimation, and the quantum-enhanced Monte Carlo methods all sit on top of it.

Two conditions are needed, and both are substantive.

**$A$ must be runnable backwards.** $R_A$ contains $A^{\dagger}$. A quantum circuit is unitary, so this is free — but it is precisely what a classical randomized algorithm cannot offer, because a classical algorithm consumes random bits and throws away intermediate state. "Wrap your Monte Carlo in amplitude amplification" therefore requires re-expressing that Monte Carlo as a reversible circuit with the randomness held in ancillas, and the cost of doing so is a real part of the accounting.

**The good subspace must be recognizable.** $O$ needs $f$. If you can produce candidate answers but not verify them, there is nothing to amplify. This is why the technique fits decision and search problems with efficient verifiers, and does not fit, say, "find the ground state energy" without an independent check.

There is also a subtlety that Code Example 4 makes concrete. In general $\pi/(4\theta) - 1/2$ is not an integer, so the maximum achievable probability is $1 - O(a)$ rather than $1$. If $A$ can be tuned — for instance by padding the search space, or by adjusting one rotation angle — then $a$ can be chosen so that $(2k+1)\theta = \pi/2$ exactly, and the amplification becomes *exact*: success probability $1$ to machine precision. The alternative, when $a$ is known only within a range, is **fixed-point amplification**, which gives up the exact rotation in exchange for a probability that increases monotonically with the number of iterations and therefore cannot over-rotate.

### Code Example 4: Amplification With an Arbitrary Preparation

```python
"""Chapter 1, Example 4: amplitude amplification with an arbitrary preparation.
Continues from Example 3 (same session)."""


def prepare(angles, n):
    """A|0...0>: a product state with a different Ry angle on every qubit."""
    psi = ket("0" * n)
    for q, a in enumerate(angles):
        psi = apply_gate(psi, ry(a), [q], n)
    return psi


def reflect_about(state, psi_A, n):
    """R = 2|psi_A><psi_A| - I, i.e. A (2|0><0| - I) A^dag without building A."""
    return 2.0 * np.vdot(psi_A, state) * psi_A - state


def aa_step(state, psi_A, marked, n):
    """One amplitude-amplification iteration Q = R_A O."""
    return reflect_about(phase_oracle(state, marked), psi_A, n)


# A deliberately lopsided preparation on 5 qubits, and a marked set of 3
# strings chosen without reference to it.
n = 5
angles = [0.7, 1.9, 2.6, 1.1, 0.4]
marked = [int(b, 2) for b in ["00110", "10101", "11011"]]
psi_A = prepare(angles, n)
a = float(sum(probs(psi_A)[m] for m in marked))
theta = np.arcsin(np.sqrt(a))

print(f"n = {n}, Ry angles = {angles}")
print(f"marked strings: {[format(m, '05b') for m in marked]}")
print(f"initial success amplitude a = |<good|A|0>|^2 = {a:.8f}")
print(f"theta = arcsin(sqrt(a))                     = {theta:.8f} rad")
print(f"uniform preparation would give a = M/N      = {len(marked) / 2 ** n:.8f}")

print(f"\n{'k':>4}{'P(good) measured':>19}{'sin^2((2k+1)theta)':>21}{'off-plane':>12}")
print("-" * 56)
good = np.zeros(2 ** n, dtype=complex)
good[marked] = psi_A[marked]
good /= np.linalg.norm(good)
bad = psi_A.copy()
bad[marked] = 0.0
bad /= np.linalg.norm(bad)
psi = psi_A.copy()
for k in range(7):
    p = float(sum(probs(psi)[m] for m in marked))
    resid = psi - np.vdot(good, psi) * good - np.vdot(bad, psi) * bad
    print(f"{k:>4}{p:>19.8f}{np.sin((2 * k + 1) * theta) ** 2:>21.8f}"
          f"{np.max(np.abs(resid)):>12.1e}")
    psi = aa_step(psi, psi_A, marked, n)

k_opt = int(np.round(np.pi / (4 * theta) - 0.5))
print(f"\nround(pi/(4 theta) - 1/2) = {k_opt}, and the table's maximum is there.")

# --- exact amplification: choose a so that pi/(4 theta) - 1/2 is an integer
print("\nExact amplification. If the preparation can be tuned, pick a with")
print("(2k+1) theta = pi/2 exactly, and the k-th iterate succeeds with")
print("probability 1 rather than 1 - O(a).")
print(f"{'k':>4}{'required a':>14}{'P(good) at k':>16}{'1 - P':>12}")
print("-" * 46)
for k in [1, 2, 3, 5]:
    a_star = np.sin(np.pi / (2 * (2 * k + 1))) ** 2
    # one qubit carries all the structure: put amplitude sqrt(a_star) on '1...1'
    ang = [2 * np.arcsin(a_star ** (1 / (2 * n)))] * n
    psi_A2 = prepare(ang, n)
    marked2 = [2 ** n - 1]
    a2 = float(probs(psi_A2)[marked2[0]])
    th2 = np.arcsin(np.sqrt(a2))
    psi2 = psi_A2.copy()
    for _ in range(k):
        psi2 = aa_step(psi2, psi_A2, marked2, n)
    p = float(probs(psi2)[marked2[0]])
    print(f"{k:>4}{a2:>14.8f}{p:>16.10f}{1 - p:>12.2e}")

# --- what the square root is worth as a subroutine speedup ----------------
print("\nAmplitude amplification as a wrapper around a randomized subroutine")
print("that succeeds with probability a. Classical: repeat ~1/a times.")
print(f"{'a':>12}{'classical 1/a':>16}{'quantum pi/(4 arcsin sqrt a)':>31}"
      f"{'ratio':>12}")
print("-" * 71)
for a_ in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
    cl = 1.0 / a_
    qu = np.pi / (4 * np.arcsin(np.sqrt(a_)))
    print(f"{a_:>12.0e}{cl:>16.4g}{qu:>31.4g}{cl / qu:>12.4g}")
```

```text
n = 5, Ry angles = [0.7, 1.9, 2.6, 1.1, 0.4]
marked strings: ['00110', '10101', '11011']
initial success amplitude a = |<good|A|0>|^2 = 0.07386401
theta = arcsin(sqrt(a))                     = 0.27524148 rad
uniform preparation would give a = M/N      = 0.09375000

   k   P(good) measured   sin^2((2k+1)theta)   off-plane
--------------------------------------------------------
   0         0.07386401           0.07386401     1.7e-16
   1         0.54028258           0.54028258     1.1e-16
   2         0.96261067           0.96261067     4.2e-17
   3         0.87859755           0.87859755     8.3e-17
   4         0.38019811           0.38019811     2.2e-16
   5         0.01292541           0.01292541     2.8e-16
   6         0.17877041           0.17877041     1.1e-16

round(pi/(4 theta) - 1/2) = 2, and the table's maximum is there.

Exact amplification. If the preparation can be tuned, pick a with
(2k+1) theta = pi/2 exactly, and the k-th iterate succeeds with
probability 1 rather than 1 - O(a).
   k    required a    P(good) at k       1 - P
----------------------------------------------
   1    0.25000000    1.0000000000    4.44e-16
   2    0.09549150    1.0000000000    1.78e-15
   3    0.04951557    1.0000000000    1.11e-15
   5    0.02025351    1.0000000000    1.78e-15

Amplitude amplification as a wrapper around a randomized subroutine
that succeeds with probability a. Classical: repeat ~1/a times.
           a   classical 1/a   quantum pi/(4 arcsin sqrt a)       ratio
-----------------------------------------------------------------------
       1e-01              10                          2.441       4.097
       1e-02             100                          7.841       12.75
       1e-03            1000                          24.83       40.27
       1e-04           1e+04                          78.54       127.3
       1e-06           1e+06                          785.4        1273
       1e-08           1e+08                           7854   1.273e+04
```

**What to look for.** The preparation here is a product of five different $R_y$ rotations, so the initial amplitude distribution is thoroughly non-uniform, and the marked strings were chosen without reference to it. The measured success probability nevertheless matches $\sin^2((2k+1)\theta)$ to eight digits at every iteration, with $\theta$ computed from the *actual* overlap $a = 0.0739$ rather than from $M/N = 0.0938$. The off-plane residual stays at $10^{-16}$. The geometry is not a property of the uniform superposition; it is a property of two reflections in a plane.

The exact-amplification block is the practical remark. Choosing $a = \sin^2(\pi/(2(2k+1)))$ makes the $k$-th iterate hit probability $1$ to within $2 \times 10^{-15}$ — for $k = 1$ that requires $a = 1/4$, for $k = 5$ it requires $a = 0.0203$. When you control the preparation, you should use this, because it removes the residual failure probability entirely rather than merely making it small.

The last table is the general claim in numbers, and it is also a warning. At $a = 10^{-8}$, classical repetition needs $10^8$ trials and amplitude amplification needs $7854$ — a factor of $1.3 \times 10^4$, which is genuinely large. But note what the quantum column counts: $7854$ *sequential* iterations, each containing one oracle, one $A$ and one $A^{\dagger}$, all of which must run coherently. The classical column counts $10^8$ *independent* trials, which can be spread across as many machines as you can buy. Section 1.4 is about exactly this asymmetry.

* * *

## 1.4 What a Quadratic Speedup Is Actually Worth

### The claim, stated exactly

Inside the query model, the situation is completely settled and there is nothing to argue about:

$$ \text{query complexity of unstructured search} = \Theta\left(\sqrt{N/M}\right) $$

with Grover achieving the upper bound and the adversary method establishing the lower one. Any claim that a quantum computer can search an unstructured space faster than $\sqrt{N}$ is false, and any claim that it cannot do better than $N$ is also false. This section is about what happens when you leave the model, which is where every practical question lives.

### Four ways the square root gets consumed

**1. The cost of a query is not one.** A query is a reversible circuit for $f$, and Code Example 2 measured 114 CNOT-equivalents for a four-variable 3-SAT formula. The classical side evaluates the same predicate with a handful of machine instructions. If the quantum per-query cost exceeds the classical one by a factor $\gamma$, the quadratic advantage only appears above $N \gtrsim \gamma^2$ — and $\gamma$ includes the reversibility overhead, the ancilla management, and the compilation into a fault-tolerant gate set.

**2. The clock rates differ by many orders of magnitude.** The relevant quantum unit is not the physical gate time but the **logical** gate time: a fault-tolerant logical operation requires many rounds of syndrome extraction on many physical qubits, and for the non-Clifford gates that dominate an oracle it requires magic states, which have to be distilled. Whatever the physical cycle time is, the logical period $t_L$ is larger by a substantial factor. Meanwhile the classical side executes billions of predicate evaluations per second per core. Code Example 5 sweeps $t_L$ and the classical rate over decades and reports where the crossover in wall-clock time sits.

**3. Grover does not parallelize.** This is the mechanism that is least often mentioned and does the most damage. Split the space into $P$ pieces and give one to each machine: classical time falls as $1/P$, but Grover time falls only as $\sqrt{1/P}$, because each machine now searches a space of size $N/P$ and $\sqrt{N/P} = \sqrt{N}/\sqrt{P}$. Every factor of $100$ in parallel hardware therefore costs the quantum side a factor of $10$ of its relative advantage. Since large-scale classical computing is overwhelmingly parallel and a Grover run is intrinsically sequential, the comparison is much less favourable than the query counts suggest.

**4. The sequential depth has to be coherent.** The $\Theta(\sqrt{N})$ queries form one unbroken circuit. There is no way to break a Grover search into short independent shots and combine the results, because the amplification is an interference effect built up over the whole sequence. A single uncorrected error anywhere in a circuit of $10^{10}$ logical gates ruins it. This is a fully fault-tolerant workload by construction — Grover is not a near-term algorithm, and no amount of error mitigation changes that.

### Unstructured search is not database search

The most common misstatement of Grover's result is "a quantum computer can search a database of $N$ records in $\sqrt{N}$ steps". Consider what that would require. The records are data, so the oracle must consult them; a coherent lookup into $N$ stored values is not a small circuit. The standard construction, **QRAM**, comes in two flavours, and neither one rescues the claim:

  * a *sequential* lookup, which is a circuit of $\Theta(N)$ gates per query; or
  * a *bucket-brigade* lookup, whose depth is $O(\log N)$ but which requires $\Theta(N)$ physical qubits arranged as a tree, every one of which must be maintained coherently and error-corrected.

Either way, one query costs $\Theta(N)$ in the resource-time product. Grover then costs $\Theta(N^{3/2})$ overall, against $\Theta(N)$ for a classical scan — worse, by a factor of $\sqrt{N}$, and Code Example 5 tabulates it. And the comparison is even less kind than that: if you are willing to build $\Theta(N)$ pieces of hardware, then classically you can look at all $N$ records in parallel in constant time.

The **QRAM problem** is therefore not an engineering detail waiting to be solved. It is a statement that any quantum algorithm whose speedup depends on coherent access to a dataset of size $N$ has to pay $\Theta(N)$ somewhere, which cancels a $\sqrt{N}$ advantage and more. The same objection applies to many quantum machine learning proposals, which is why the honest accounting there begins with the input model rather than the algorithm.

What Grover *does* apply to is a function given as a **rule** that is cheap to evaluate and expensive to invert: a hash, a cipher, a SAT formula, a physical predicate. Nothing of size $N$ is ever stored, so there is nothing to load.

### The base of the comparison

Even for a rule, the exponent matters less than what it is an exponent *of*. Grover square-roots the cost of **brute force**. If the best classical algorithm for a problem already runs in $2^{cn}$ with $c < 1/2$, then Grover-on-brute-force at $2^{n/2}$ is slower, and the correct conclusion is that the quantum algorithm loses. This is not a hypothetical: for many NP-hard problems the best known classical methods have exponents well below one, and a quadratic speedup over enumeration is simply not competitive with them. Combining Grover with a good classical algorithm — amplifying inside a backtracking search, say — is possible and is an active subject, but it is a different and much more delicate claim than "quadratic speedup on SAT".

### What survives

After all of that, a real and defensible core remains.

| Application | Why the objections do not bite |
| --- | --- |
| Cryptographic key search and preimage finding | $f$ is a cipher or hash: a rule, cheap, nothing stored. The best classical attack genuinely *is* brute force, so the base of the comparison is right. This is why symmetric key lengths are doubled in post-quantum recommendations — a $2^{128}$ search becomes $2^{64}$, which is why $256$-bit keys are specified |
| Amplitude estimation for Monte Carlo | the quantity estimated is a probability; the quadratic reduction in samples applies to any reversible sampler, and the "database" is generated rather than stored |
| Subroutine amplification inside other quantum algorithms | the success probability being amplified is already inside a quantum circuit; no input model is involved |
| Problems whose best classical algorithm is enumeration | the base of the comparison is brute force by construction |

And the negative results are equally worth carrying:

  * Grover does not help with database search, for the reasons above.
  * Grover does not make NP-hard problems tractable. $2^{n/2}$ is still exponential; a quadratic speedup moves the feasible problem size by a factor of two in $n$, not by an order of magnitude in difficulty.
  * A quadratic speedup can be worth having, but only when the constant factors, the clock ratio and the parallelism are all accounted for and the answer still comes out positive. That is an arithmetic question, and Code Example 5 is that arithmetic.

### Code Example 5: The Honest Arithmetic

Every rate below is swept over decades rather than quoted, precisely so that no conclusion depends on a device specification or a record. The four blocks are the four mechanisms above, in order.

```python
"""Chapter 1, Example 5: what a quadratic speedup is worth, in arithmetic.
Continues from Example 4 (same session).

Every rate below is a PARAMETER SWEPT OVER DECADES, not a device specification
and not a record. The conclusions depend only on which decade a parameter sits
in, which is why the sweep is printed instead of a single number.
"""

# --- 1. wall-clock crossover, with the constant factors kept --------------
# Classical: N/2 predicate evaluations at r_c evaluations per second.
# Quantum:   (pi/4) sqrt(N) oracle calls, each G_q logical gates, each t_L long.
G_q = 100                      # logical gates per oracle call (order of magnitude)

print("Crossover bit width n at which Grover's wall clock first beats a "
      "classical scan.")
print("t_L is the LOGICAL gate period: the physical cycle times the "
      "error-correction cost.")
print(f"\n{'t_L':>10}{'r_c = 1e9/s':>14}{'1e10/s':>10}{'1e12/s':>10}"
      f"{'1e15/s':>10}")
print("-" * 54)
for t_L, label in [(1e-9, "1 ns"), (1e-7, "100 ns"), (1e-6, "1 us"),
                   (1e-4, "100 us"), (1e-3, "1 ms")]:
    row = f"{label:>10}"
    for r_c in [1e9, 1e10, 1e12, 1e15]:
        n_cross = None
        for n in range(1, 401):
            N = 2.0 ** n
            t_classical = 0.5 * N / r_c
            t_quantum = (np.pi / 4) * np.sqrt(N) * G_q * t_L
            if t_quantum < t_classical:
                n_cross = n
                break
        row += f"{n_cross:>14}" if r_c == 1e9 else f"{n_cross:>10}"
    print(row)

print("\nWhat the machine has to do near that crossover, at t_L = 1 us and "
      "r_c = 1e12/s:")
t_L, r_c = 1e-6, 1e12
print(f"{'n':>4}{'queries':>12}{'logical gates':>15}{'quantum (s)':>13}"
      f"{'classical (s)':>15}")
print("-" * 59)
for n in [40, 50, 60, 80, 100]:
    N = 2.0 ** n
    q_calls = (np.pi / 4) * np.sqrt(N)
    print(f"{n:>4}{q_calls:>12.3g}{q_calls * G_q:>15.3g}"
          f"{q_calls * G_q * t_L:>13.3g}{0.5 * N / r_c:>15.3g}")
print("  The 'logical gates' column is a SEQUENTIAL depth: Grover cannot be "
      "unrolled.")

# --- 2. the quadratic speedup does not survive parallelism ----------------
print("\nSplitting the search space over P machines. Classical time falls as "
      "1/P;")
print("Grover on 1/P of the space falls only as 1/sqrt(P).")
print(f"{'P':>10}{'classical N/(2P)':>20}{'quantum sqrt(N/P)':>20}"
      f"{'quantum advantage':>20}")
print("-" * 70)
N = 2.0 ** 60
for P in [1, 1e2, 1e4, 1e6, 1e8]:
    cl = 0.5 * N / P
    qu = (np.pi / 4) * np.sqrt(N / P)
    print(f"{P:>10.0e}{cl:>20.4g}{qu:>20.4g}{cl / qu:>20.4g}")
print("  Every factor of 100 in parallel hardware costs the quantum side a "
      "factor of 10")
print("  of its advantage. The speedup is quadratic in queries, not in "
      "resources.")

# --- 3. unstructured search is not database search ------------------------
print("\nSearching a STORED TABLE of N records. A coherent table lookup costs "
      "Theta(N)")
print("gates per query (or Theta(N) qubits of hardware), so Grover pays "
      "N^{3/2} in total.")
print(f"\n{'n':>4}{'N':>12}{'classical scan':>17}{'Grover queries':>16}"
       f"{'x lookup cost':>16}{'ratio':>12}")
print("-" * 77)
for n in [10, 20, 30, 40, 50]:
    N = 2.0 ** n
    scan = N
    q = (np.pi / 4) * np.sqrt(N)
    tot = q * N
    print(f"{n:>4}{N:>12.4g}{scan:>17.4g}{q:>16.4g}{tot:>16.4g}"
          f"{tot / scan:>12.4g}")
print("  The ratio is sqrt(N) the WRONG way. For a genuine database, Grover "
       "loses.")
print("  It wins only when f(x) is a rule that is cheap to evaluate and never "
       "stored.")

# --- 4. the base of the comparison matters more than the exponent ---------
print("\nGrover square-roots the cost of BRUTE FORCE. If the best classical")
print("algorithm already runs in 2^{c n} with c < 1, the honest comparison is")
print("2^{0.5 n} against 2^{c n}.")
print(f"\n{'classical exponent c':>22}{'verdict':>14}"
      f"{'n = 50':>12}{'n = 100':>12}{'n = 200':>12}")
print("-" * 72)
for c in [1.0, 0.75, 0.6, 0.5, 0.4, 0.3]:
    verdict = "Grover wins" if c > 0.5 else ("tie" if c == 0.5 else "Grover loses")
    cells = "".join(f"{2.0 ** (c * n) / 2.0 ** (0.5 * n):>12.3g}"
                    for n in [50, 100, 200])
    print(f"{c:>22.2f}{verdict:>14}{cells}")
print("  The numbers are the classical cost divided by the Grover cost: "
       "> 1 favours")
print("  Grover. A classical exponent below 1/2 is not exotic; SAT and "
       "many other")
print("  NP-hard problems have heuristics with exactly that character.")
```

```text
Crossover bit width n at which Grover's wall clock first beats a classical scan.
t_L is the LOGICAL gate period: the physical cycle times the error-correction cost.

       t_L   r_c = 1e9/s    1e10/s    1e12/s    1e15/s
------------------------------------------------------
      1 ns            15        22        35        55
    100 ns            28        35        48        68
      1 us            35        42        55        75
    100 us            48        55        68        88
      1 ms            55        62        75        95

What the machine has to do near that crossover, at t_L = 1 us and r_c = 1e12/s:
   n     queries  logical gates  quantum (s)  classical (s)
-----------------------------------------------------------
  40    8.24e+05       8.24e+07         82.4           0.55
  50    2.64e+07       2.64e+09     2.64e+03            563
  60    8.43e+08       8.43e+10     8.43e+04       5.76e+05
  80    8.64e+11       8.64e+13     8.64e+07       6.04e+11
 100    8.84e+14       8.84e+16     8.84e+10       6.34e+17
  The 'logical gates' column is a SEQUENTIAL depth: Grover cannot be unrolled.

Splitting the search space over P machines. Classical time falls as 1/P;
Grover on 1/P of the space falls only as 1/sqrt(P).
         P    classical N/(2P)   quantum sqrt(N/P)   quantum advantage
----------------------------------------------------------------------
     1e+00           5.765e+17           8.433e+08           6.836e+08
     1e+02           5.765e+15           8.433e+07           6.836e+07
     1e+04           5.765e+13           8.433e+06           6.836e+06
     1e+06           5.765e+11           8.433e+05           6.836e+05
     1e+08           5.765e+09           8.433e+04           6.836e+04
  Every factor of 100 in parallel hardware costs the quantum side a factor of 10
  of its advantage. The speedup is quadratic in queries, not in resources.

Searching a STORED TABLE of N records. A coherent table lookup costs Theta(N)
gates per query (or Theta(N) qubits of hardware), so Grover pays N^{3/2} in total.

   n           N   classical scan  Grover queries   x lookup cost       ratio
-----------------------------------------------------------------------------
  10        1024             1024           25.13       2.574e+04       25.13
  20   1.049e+06        1.049e+06           804.2       8.433e+08       804.2
  30   1.074e+09        1.074e+09       2.574e+04       2.763e+13   2.574e+04
  40     1.1e+12          1.1e+12       8.235e+05       9.055e+17   8.235e+05
  50   1.126e+15        1.126e+15       2.635e+07       2.967e+22   2.635e+07
  The ratio is sqrt(N) the WRONG way. For a genuine database, Grover loses.
  It wins only when f(x) is a rule that is cheap to evaluate and never stored.

Grover square-roots the cost of BRUTE FORCE. If the best classical
algorithm already runs in 2^{c n} with c < 1, the honest comparison is
2^{0.5 n} against 2^{c n}.

  classical exponent c       verdict      n = 50     n = 100     n = 200
------------------------------------------------------------------------
                  1.00   Grover wins    3.36e+07    1.13e+15    1.27e+30
                  0.75   Grover wins    5.79e+03    3.36e+07    1.13e+15
                  0.60   Grover wins          32    1.02e+03    1.05e+06
                  0.50           tie           1           1           1
                  0.40  Grover loses      0.0312    0.000977    9.54e-07
                  0.30  Grover loses    0.000977    9.54e-07    9.09e-13
  The numbers are the classical cost divided by the Grover cost: > 1 favours
  Grover. A classical exponent below 1/2 is not exotic; SAT and many other
  NP-hard problems have heuristics with exactly that character.
```

**What to look for.** The first table is the headline. With a logical gate period of $1\ \mu$s, an oracle of 100 logical gates, and a classical rate of $10^{12}$ predicate evaluations per second, Grover only wins above $n = 55$, that is, above a search space of $3.6 \times 10^{16}$. Push the logical period to $1$ ms and the crossover moves to $n = 75$. Pull it down to $1$ ns and it falls to $n = 35$. The pattern in the table is simple and worth internalizing: **each decade of clock disadvantage costs about 6.6 bits of crossover**, because the quantum cost scales as $2^{n/2}$ and the classical as $2^n$, so a factor $10$ is repaid by $\log_2 10 \times 2 \approx 6.6$ extra bits.

The second block says what the machine must do near the crossover. At $n = 60$, Grover needs $8.4 \times 10^8$ queries, which is $8.4 \times 10^{10}$ logical gates *in sequence*. That is the fault-tolerance requirement, and it is the reason this is a far-future workload rather than a near-term one.

The third block is the parallelism argument, and it is stark: the quantum advantage falls by a factor of $10$ for every factor of $100$ of classical parallel hardware, from $6.8 \times 10^8$ at $P = 1$ to $6.8 \times 10^4$ at $P = 10^8$. Still an advantage, but four decades of it have evaporated purely from the shape of the scaling.

The fourth block is the database calculation. At $n = 30$ the ratio is $2.6 \times 10^4$ *against* Grover. The fifth block is the base-of-comparison calculation: at a classical exponent of $c = 0.4$ and $n = 100$, the classical algorithm is faster by a factor of $10^3$, and the "quadratic speedup" is a quadratic speedup over the wrong baseline.

* * *

## 1.5 Grover on the Simulator

Everything above is now checked at the largest sizes a state-vector simulator handles comfortably. Two examples: the clean case, one marked string with $n$ from 4 to 10, and the awkward cases, several solutions and an unknown number of them. Both use the black-box oracle of Code Example 2 — which, as established there, is legitimate for testing and dishonest as a demonstration of usefulness, so it is labelled as such.

### Code Example 6: The Full Search, $n = 4$ to $10$

```python
"""Chapter 1, Example 6: the full search, n = 4 to 10.
Continues from Example 5 (same session)."""
import matplotlib.pyplot as plt


def grover_run(n, marked, k):
    """Uniform preparation followed by k Grover iterations."""
    psi = uniform(n)
    for _ in range(k):
        psi = grover_step(psi, marked, n)
    return psi


def success_curve(n, marked, k_max):
    """P(any marked string) after 0, 1, ..., k_max iterations."""
    psi = uniform(n)
    out = []
    for _ in range(k_max + 1):
        out.append(float(sum(probs(psi)[m] for m in marked)))
        psi = grover_step(psi, marked, n)
    return np.array(out)


print("One marked string, n = 4 to 10. k_opt is measured (argmax of the curve)")
print("and compared with the two closed forms.\n")
hdr = (f"{'n':>3}{'N':>7}{'theta (rad)':>13}{'argmax k':>10}"
       f"{'floor(pi/4th)':>15}{'floor(pi sqrtN/4)':>19}{'P(k_opt)':>11}"
       f"{'1-P':>10}")
print(hdr)
print("-" * len(hdr))
curves = {}
for n in range(4, 11):
    N = 2 ** n
    marked = [int(N * 0.7) | 1]              # one arbitrary but reproducible index
    theta = np.arcsin(np.sqrt(1.0 / N))
    k_max = int(np.ceil(3.2 * np.pi / (4 * theta)))
    p = success_curve(n, marked, k_max)
    curves[n] = p
    k_zero = int(np.round(np.pi / (2 * theta) - 0.5))   # first zero of the curve
    k_star = int(np.argmax(p[:k_zero + 1]))
    k_form = int(np.floor(np.pi / (4 * theta)))
    k_apx = int(np.floor(np.pi * np.sqrt(N) / 4))
    print(f"{n:>3}{N:>7}{theta:>13.6f}{k_star:>10}{k_form:>15}{k_apx:>19}"
          f"{p[k_star]:>11.6f}{1 - p[k_star]:>10.2e}")

print("\nThe argmax is taken over the first period only, up to the first zero of")
print("the curve. For M = 1 the exact count and its small-angle form agree at")
print("every n above; they part company once M/N is not small (Example 7).")

print("\nOver-rotation: the curve is periodic, not monotone. n = 8, N = 256.")
p8 = curves[8]
print(f"{'k':>5}{'P(marked)':>12}")
print("-" * 17)
for k in range(0, len(p8), 4):
    bar = "#" * int(round(50 * p8[k]))
    print(f"{k:>5}{p8[k]:>12.6f}   {bar}".rstrip())
k_form8 = int(np.floor(np.pi / (4 * np.arcsin(np.sqrt(1 / 256)))))
print(f"\n  optimal k = {k_form8}, P = {p8[k_form8]:.6f}")
print(f"  running 2x too long, k = {2 * k_form8}: P = {p8[2 * k_form8]:.6f}")
print(f"  running 3x too long, k = {3 * k_form8}: P = {p8[3 * k_form8]:.6f}")
print("  'More iterations' is not 'better'. The iteration count is part of the")
print("  algorithm, and it depends on M, which in a real problem is unknown.")

# --- the protocol as actually run: prepare, iterate, measure, verify ------
print("\nThe algorithm as a protocol: 2000 shots at k_opt, then classical")
print("verification of the measured string.")
print(f"{'n':>4}{'k_opt':>7}{'shots hitting the mark':>25}{'empirical P':>14}"
      f"{'predicted P':>14}")
print("-" * 64)
for n in [4, 6, 8, 10]:
    N = 2 ** n
    marked = [int(N * 0.7) | 1]
    theta = np.arcsin(np.sqrt(1.0 / N))
    k_form = int(np.floor(np.pi / (4 * theta)))
    psi = grover_run(n, marked, k_form)
    counts = sample(psi, 2000, seed=20260813 + n)
    target = format(marked[0], f"0{n}b")
    hit = counts.get(target, 0)
    print(f"{n:>4}{k_form:>7}{hit:>25}{hit / 2000:>14.4f}"
          f"{float(probs(psi)[marked[0]]):>14.4f}")

# --- universality of the curve when the axis is rescaled -----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for n in [4, 6, 8, 10]:
    p = curves[n]
    ax[0].plot(np.arange(len(p)), p, marker="o", ms=2.5, lw=0.9,
               label=f"n = {n}")
    theta = np.arcsin(np.sqrt(1.0 / 2 ** n))
    ax[1].plot(np.arange(len(p)) * theta / (np.pi / 2), p, lw=0.9,
               label=f"n = {n}")
ax[0].set_xlabel("Grover iterations k"); ax[0].set_ylabel("P(marked)")
ax[0].set_title("Success probability"); ax[0].legend(fontsize=8)
ax[1].set_xlabel(r"$k\theta/(\pi/2)$"); ax[1].set_ylabel("P(marked)")
ax[1].set_title("The same curves, rescaled"); ax[1].legend(fontsize=8)
ax[1].axvline(1.0, ls="--", color="k", lw=1)
plt.tight_layout()
plt.show()
```

```text
One marked string, n = 4 to 10. k_opt is measured (argmax of the curve)
and compared with the two closed forms.

  n      N  theta (rad)  argmax k  floor(pi/4th)  floor(pi sqrtN/4)   P(k_opt)       1-P
----------------------------------------------------------------------------------------
  4     16     0.252680         3              3                  3   0.961319  3.87e-02
  5     32     0.177711         4              4                  4   0.999182  8.18e-04
  6     64     0.125328         6              6                  6   0.996586  3.41e-03
  7    128     0.088504         8              8                  8   0.995620  4.38e-03
  8    256     0.062541        12             12                 12   0.999947  5.30e-05
  9    512     0.044209        17             17                 17   0.999448  5.52e-04
 10   1024     0.031255        25             25                 25   0.999461  5.39e-04

The argmax is taken over the first period only, up to the first zero of
the curve. For M = 1 the exact count and its small-angle form agree at
every n above; they part company once M/N is not small (Example 7).

Over-rotation: the curve is periodic, not monotone. n = 8, N = 256.
    k   P(marked)
-----------------
    0    0.003906
    4    0.284743   ##############
    8    0.763722   ######################################
   12    0.999947   ##################################################
   16    0.775974   #######################################
   20    0.297969   ###############
   24    0.005932
   28    0.168681   ########
   32    0.636407   ################################
   36    0.978571   #################################################
   40    0.880214   ############################################

  optimal k = 12, P = 0.999947
  running 2x too long, k = 24: P = 0.005932
  running 3x too long, k = 36: P = 0.978571
  'More iterations' is not 'better'. The iteration count is part of the
  algorithm, and it depends on M, which in a real problem is unknown.

The algorithm as a protocol: 2000 shots at k_opt, then classical
verification of the measured string.
   n  k_opt   shots hitting the mark   empirical P   predicted P
----------------------------------------------------------------
   4      3                     1924        0.9620        0.9613
   6      6                     1995        0.9975        0.9966
   8     12                     2000        1.0000        0.9999
  10     25                     1999        0.9995        0.9995
```

**What to look for.** The measured optimal iteration count agrees with $\lfloor \pi/(4\theta) \rfloor$ at every $n$ from 4 to 10 — 3, 4, 6, 8, 12, 17, 25 — and so does the small-angle form $\lfloor \pi\sqrt{N}/4 \rfloor$ for a single marked item. The success probability at the optimum rises from $0.961$ at $n = 4$ to $0.9995$ at $n = 10$, and the failure probability is bounded by $M/N$ as derived in Section 1.2: at $n = 4$ the bound is $1/16 = 0.0625$ and the measured failure is $0.0387$.

Note carefully that the argmax is taken over the first period only. Without that restriction the search finds a *later* revival that happens to land marginally closer to $\pi/2$ — at $n = 10$ the raw argmax over 80 iterations is $k = 75$, which is a correct maximum of the curve and a useless answer, because it costs three times as many queries for the same probability. Reporting the first maximum is not a convenience; it is part of the algorithm.

The bar chart is the over-rotation picture in one glance: $0.9999$ at $k = 12$, $0.0059$ at $k = 24$, back up to $0.979$ at $k = 36$. And the shot-sampling block is the algorithm as a protocol rather than as a state vector — prepare, iterate $k_{\mathrm{opt}}$ times, measure, verify classically — with the empirical frequency over 2000 shots matching the predicted probability to the third decimal.

The right-hand panel of the figure is worth pausing on. Plotting the same curves against $k\theta/(\pi/2)$ collapses them onto a single sinusoid, because the only thing that ever mattered was the accumulated angle. Grover's algorithm at any $n$ is the same rotation; $n$ only sets how small each step is.

### Code Example 7: Several Solutions, and an Unknown Number of Them

```python
"""Chapter 1, Example 7: several solutions, and an unknown number of them.
Continues from Example 6 (same session)."""


def k_exact(N, M):
    """The optimal iteration count, from the exact angle."""
    return int(np.floor(np.pi / (4 * np.arcsin(np.sqrt(M / N)))))


print("n = 10, N = 1024. M marked strings, chosen as the first M multiples of 7.")
n, N = 10, 1024
hdr = (f"{'M':>6}{'M/N':>9}{'theta':>10}{'k exact':>9}{'k approx':>10}"
       f"{'argmax':>8}{'P(k_exact)':>12}{'P(k=0)':>9}{'gain':>8}")
print(hdr)
print("-" * len(hdr))
for M in [1, 2, 4, 8, 16, 64, 128, 256, 384, 512]:
    marked = [(7 * i) % N for i in range(M)]
    marked = sorted(set(marked))[:M]
    theta = np.arcsin(np.sqrt(len(marked) / N))
    k_e = k_exact(N, len(marked))
    k_a = int(np.floor(np.pi * np.sqrt(N / len(marked)) / 4))
    k_lim = max(2, int(np.round(np.pi / (2 * theta) - 0.5)))
    p = success_curve(n, marked, k_lim)
    k_star = int(np.argmax(p))
    p0 = p[0]
    print(f"{len(marked):>6}{len(marked) / N:>9.4f}{theta:>10.5f}{k_e:>9}"
          f"{k_a:>10}{k_star:>8}{p[k_e]:>12.6f}{p0:>9.4f}"
          f"{p[k_e] / p0:>8.2f}")

print("\nAt M = N/2 the angle is exactly pi/4, so one iteration turns the state")
print("through pi/2 and lands back on 50%: k_exact = 0 and Grover gains nothing.")
print("Dense solution sets do not need searching; that is why the interesting")
print("regime is M << N, and why the approximate count fails at large M/N.")

# --- the 3-SAT oracle of Example 2, run as a search -----------------------
print("\nThe 3-SAT instance of Example 2 has 10 solutions out of 16, "
      "M/N = 0.625.")
theta_sat = np.arcsin(np.sqrt(10 / 16))
print(f"theta = {theta_sat:.6f} rad = {np.degrees(theta_sat):.2f} deg, "
      f"k_exact = {k_exact(16, 10)}")
p_sat = success_curve(4, SOLUTIONS, 4)
print(f"{'k':>4}{'P(satisfying)':>16}{'queries used':>14}"
      f"{'classical, same q':>19}")
print("-" * 53)
for k, p in enumerate(p_sat):
    q = k + 1                       # k oracle calls plus one verification
    print(f"{k:>4}{p:>16.6f}{q:>14}{1 - (1 - 10 / 16) ** q:>19.6f}")
print("  Guessing at random already succeeds with probability 0.625, and the")
print("  first Grover iteration makes it WORSE. There is a revival at k = 2,")
print("  but the last column is the comparison that matters: q independent")
print("  classical draws succeed with 1 - (1 - M/N)^q, and at q = 3 that is")
print("  0.947 against Grover's 0.977. The honest verdict on this instance is")
print("  that it does not need a quantum computer.")

# --- unknown M: the randomized exponential-search strategy ---------------
print("\nWhen M is unknown, the iteration count cannot be computed in advance.")
print("The standard fix (Boyer, Brassard, Hoyer, Tapp) is to draw the number of")
print("iterations at random from a window that grows geometrically, and to")
print("verify each measured string classically.")


def bbht(n, marked, rng, lam=6 / 5):
    """Search with M unknown. Returns (found, oracle queries used)."""
    N = 2 ** n
    m, queries = 1.0, 0
    while queries < 20 * np.sqrt(N):
        j = int(rng.integers(0, max(1, int(np.ceil(m)))))
        psi = uniform(n)
        for _ in range(j):
            psi = grover_step(psi, marked, n)
        queries += j + 1                      # j queries, plus one verification
        idx = int(rng.choice(N, p=probs(psi)))
        if idx in marked:
            return True, queries
        m = min(lam * m, np.sqrt(N))
    return False, queries


rng = np.random.default_rng(20260813)
print(f"\n{'n':>4}{'M':>5}{'trials':>8}{'success rate':>14}{'mean queries':>14}"
      f"{'sqrt(N/M)':>12}{'ratio':>8}")
print("-" * 65)
for n_, M in [(8, 1), (8, 4), (8, 16), (10, 1), (10, 4), (10, 16)]:
    N_ = 2 ** n_
    marked = sorted({(7 * i) % N_ for i in range(M)})[:M]
    trials, tot, ok = 200, 0, 0
    for _ in range(trials):
        found, q = bbht(n_, marked, rng)
        ok += found
        tot += q
    ref = np.sqrt(N_ / len(marked))
    print(f"{n_:>4}{len(marked):>5}{trials:>8}{ok / trials:>14.3f}"
          f"{tot / trials:>14.2f}{ref:>12.2f}{tot / trials / ref:>8.2f}")
print("  The mean query count tracks sqrt(N/M) with a constant of order unity,")
print("  and every trial ends in a verified answer. The price of not knowing M")
print("  is a constant factor, not a change of scaling.")
```

```text
n = 10, N = 1024. M marked strings, chosen as the first M multiples of 7.
     M      M/N     theta  k exact  k approx  argmax  P(k_exact)   P(k=0)    gain
---------------------------------------------------------------------------------
     1   0.0010   0.03126       25        25      25    0.999461   0.0010 1023.45
     2   0.0020   0.04421       17        17      17    0.999448   0.0020  511.72
     4   0.0039   0.06254       12        12      12    0.999947   0.0039  255.99
     8   0.0078   0.08850        8         8       8    0.995620   0.0078  127.44
    16   0.0156   0.12533        6         6       6    0.996586   0.0156   63.78
    64   0.0625   0.25268        3         3       3    0.961319   0.0625   15.38
   128   0.1250   0.36137        2         2       2    0.945312   0.1250    7.56
   256   0.2500   0.52360        1         1       1    1.000000   0.2500    4.00
   384   0.3750   0.65906        1         1       1    0.843750   0.3750    2.25
   512   0.5000   0.78540        0         1       0    0.500000   0.5000    1.00

At M = N/2 the angle is exactly pi/4, so one iteration turns the state
through pi/2 and lands back on 50%: k_exact = 0 and Grover gains nothing.
Dense solution sets do not need searching; that is why the interesting
regime is M << N, and why the approximate count fails at large M/N.

The 3-SAT instance of Example 2 has 10 solutions out of 16, M/N = 0.625.
theta = 0.911738 rad = 52.24 deg, k_exact = 0
   k   P(satisfying)  queries used  classical, same q
-----------------------------------------------------
   0        0.625000             1           0.625000
   1        0.156250             2           0.859375
   2        0.976562             3           0.947266
   3        0.009766             4           0.980225
   4        0.881348             5           0.992584
  Guessing at random already succeeds with probability 0.625, and the
  first Grover iteration makes it WORSE. There is a revival at k = 2,
  but the last column is the comparison that matters: q independent
  classical draws succeed with 1 - (1 - M/N)^q, and at q = 3 that is
  0.947 against Grover's 0.977. The honest verdict on this instance is
  that it does not need a quantum computer.

When M is unknown, the iteration count cannot be computed in advance.
The standard fix (Boyer, Brassard, Hoyer, Tapp) is to draw the number of
iterations at random from a window that grows geometrically, and to
verify each measured string classically.

   n    M  trials  success rate  mean queries   sqrt(N/M)   ratio
-----------------------------------------------------------------
   8    1     200         1.000         27.86       16.00    1.74
   8    4     200         1.000         13.21        8.00    1.65
   8   16     200         1.000          6.15        4.00    1.54
  10    1     200         1.000         47.72       32.00    1.49
  10    4     200         1.000         26.34       16.00    1.65
  10   16     200         1.000         12.52        8.00    1.56
  The mean query count tracks sqrt(N/M) with a constant of order unity,
  and every trial ends in a verified answer. The price of not knowing M
  is a constant factor, not a change of scaling.
```

**What to look for.** The first table sweeps $M$ from 1 to $N/2$ at $n = 10$. The iteration count falls as $\sqrt{N/M}$ — 25, 17, 12, 8, 6, 3, 2, 1 — and the gain over random guessing falls with it, from a factor of $1023$ at $M = 1$ to a factor of $4$ at $M = N/4$. At $M = N/2$ the angle is exactly $\pi/4$, one iteration turns the state through $\pi/2$, and the success probability returns to $1/2$: the exact and approximate counts tie at $k = 0$ and $k = 1$, both giving exactly $1/2$, and Grover gains nothing at all. This is also where the small-angle form parts company from the exact one, returning 1 where the exact expression returns 0.

The 3-SAT instance from Code Example 2 makes the point concretely, and the last column is the comparison that should always be made for a dense solution set. With $M/N = 0.625$, three independent random draws succeed with probability $0.947$; Grover with two iterations plus one verification, also three queries, gives $0.977$. That is a real but negligible gain, obtained by a fault-tolerant quantum computer, on a problem that a coin solves. The honest verdict is that this instance does not need a quantum computer, and being able to say so is the point of the whole section.

The last block is the proper treatment of unknown $M$. The strategy is due to Boyer, Brassard, Høyer and Tapp: draw the number of iterations uniformly from a window, grow the window geometrically after each failure, and verify every measured string classically. Every one of the 1200 trials returned a verified solution, and the mean query count tracked $\sqrt{N/M}$ with a constant between $1.5$ and $1.75$. Not knowing $M$ therefore costs a constant factor and not a change of scaling — which is the result you want, and it is worth having measured rather than cited.

### The Toolkit Built Here

| Function | Introduced in | Used for |
| --- | --- | --- |
| `phase_oracle` | Code Example 2 | the black-box query, for testing |
| `sat_phase_oracle`, `mcz_oracle` | Code Example 2 | oracles as real circuits, with gate counts |
| `uniform`, `diffuser`, `grover_step` | Code Example 3 | Grover's algorithm itself |
| `subspace_coords` | Code Example 3 | checking that the state stays in the plane |
| `prepare`, `reflect_about`, `aa_step` | Code Example 4 | general amplitude amplification |
| `success_curve`, `grover_run` | Code Example 6 | success probability as a function of $k$ |
| `bbht` | Code Example 7 | search with an unknown number of solutions |

What this toolkit cannot do is equally worth stating. It has no noise model, so every probability quoted here is the ideal one; the introductory course's Chapter 5 shows what a depolarizing channel does to a circuit of this depth, and the answer for $10^{10}$ gates is that nothing survives without error correction. It also has no way to *count* solutions, which is the missing piece that Chapter 2 supplies: quantum counting applies phase estimation to the Grover operator itself, reads off $\theta$, and hence $M$ — turning the guesswork of Code Example 7 into a measurement.

* * *

## Exercises

#### Exercise 1: Angles and Iteration Counts

For each of the following, compute $\theta = \arcsin\sqrt{M/N}$, the exact optimal iteration count $\lfloor \pi/(4\theta) \rfloor$, the small-angle count $\lfloor \pi\sqrt{N/M}/4 \rfloor$, and the success probability at the optimum.

  1. $N = 4$, $M = 1$.
  2. $N = 1024$, $M = 1$.
  3. $N = 1024$, $M = 4$.
  4. $N = 1024$, $M = 400$.
  5. For which of the four is the small-angle count wrong, and why?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\sin\theta = 1/2\), so \(\theta = \pi/6\) exactly. Then \(\pi/(4\theta) = 1.5\), so \(k_{\mathrm{opt}} = 1\), and \((2k+1)\theta = 3 \times \pi/6 = \pi/2\) exactly, giving \(P = 1\). This is the one case where Grover is <em>deterministic</em>: two qubits, one marked string, one query, certain success. The small-angle count is \(\lfloor \pi \times 2/4 \rfloor = 1\), also correct.</p>

<p><strong>2.</strong> \(\theta = \arcsin(1/32) = 0.031255\) rad. \(\pi/(4\theta) = 25.13\), so \(k_{\mathrm{opt}} = 25\); the small-angle count is \(\lfloor \pi \times 32/4 \rfloor = \lfloor 25.13 \rfloor = 25\). \(P = \sin^2(51 \times 0.031255) = 0.99946\), consistent with the bound \(1 - M/N = 0.99902\).</p>

<p><strong>3.</strong> \(\theta = \arcsin(1/16) = 0.062541\) rad, \(\pi/(4\theta) = 12.56\), \(k_{\mathrm{opt}} = 12\), small-angle count also 12, \(P = 0.99995\).</p>

<p><strong>4.</strong> \(M/N = 0.390625\), \(\theta = \arcsin(0.625) = 0.67513\) rad. \(\pi/(4\theta) = 1.164\), so \(k_{\mathrm{opt}} = 1\) and \(P = \sin^2(3\theta) = \sin^2(2.0254) = 0.80719\). The small-angle count is \(\lfloor \pi\sqrt{2.56}/4 \rfloor = \lfloor 1.257 \rfloor = 1\), which happens to agree here.</p>

<p><strong>5.</strong> None of these four, as it turns out — but the agreement is luck, not a rule. The small-angle form uses \(\theta \approx \sqrt{M/N}\), which underestimates \(\theta\) and therefore overestimates \(\pi/(4\theta)\); the two counts differ whenever the two real numbers straddle an integer, and that becomes likely once \(M/N\) is not small. The extreme case is \(M/N = 1/2\), the last row of Code Example 7's table: there \(\theta = \pi/4\) and \(\pi/(4\theta) = 1\) exactly, so \(k = 0\) and \(k = 1\) tie with \(P_0 = P_1 = 1/2\) and the algorithm has nothing to offer, while the small-angle form confidently returns \(k = 1\). In floating point the exact expression evaluates to \(1 - 10^{-16}\) and the floor returns 0, which is one of the two tied answers. The lesson: use the exact form, it costs one <code>arcsin</code>.</p>

</details>

#### Exercise 2: Deriving the Rotation

  1. Show that $O = I - 2P_{\mathrm{good}}$ and $D = 2\lvert s \rangle\langle s \rvert - I$ are both Hermitian and unitary, and that each squares to the identity.
  2. Working in the two-dimensional basis $\lbrace \lvert \mathrm{good} \rangle, \lvert \mathrm{bad} \rangle \rbrace$, write down the $2 \times 2$ matrices of $O$ and $D$ and multiply them. Identify the result as a rotation and read off the angle.
  3. Prove $P_{k_{\mathrm{opt}}} \ge 1 - M/N$.
  4. What is the smallest $N$ (with $M = 1$) for which a single Grover iteration gives $P = 1$ exactly, and why is there only one such $N$?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(P_{\mathrm{good}}\) is a projector, so \(P^{\dagger} = P\) and \(P^2 = P\). Then \(O^{\dagger} = O\) and \(O^2 = I - 4P + 4P^2 = I\), so \(O\) is a Hermitian involution and hence unitary. The same argument applies to \(D\) with the projector \(\lvert s \rangle\langle s \rvert\). Any Hermitian involution is a reflection: its eigenvalues are \(\pm 1\).</p>

<p><strong>2.</strong> In the ordered basis \((\lvert \mathrm{good} \rangle, \lvert \mathrm{bad} \rangle)\), and writing \(c = \cos\theta\), \(s = \sin\theta\) so that \(\lvert s \rangle = (s, c)^{T}\):</p>

<p>\[ O = \begin{pmatrix} -1 & 0 \cr 0 & 1 \end{pmatrix}, \qquad D = 2\begin{pmatrix} s^2 & sc \cr sc & c^2 \end{pmatrix} - I = \begin{pmatrix} s^2 - c^2 & 2sc \cr 2sc & c^2 - s^2 \end{pmatrix} = \begin{pmatrix} -\cos 2\theta & \sin 2\theta \cr \sin 2\theta & \cos 2\theta \end{pmatrix} \]</p>

<p>\[ DO = \begin{pmatrix} \cos 2\theta & \sin 2\theta \cr -\sin 2\theta & \cos 2\theta \end{pmatrix} \]</p>

<p>which is a rotation by \(2\theta\) (towards \(\lvert \mathrm{good} \rangle\), given the sign convention that the angle is measured from the \(\lvert \mathrm{bad} \rangle\) axis). Applying it \(k\) times to \(\lvert s \rangle\), whose angle is \(\theta\), gives angle \((2k+1)\theta\).</p>

<p><strong>3.</strong> \(k_{\mathrm{opt}}\) is the integer nearest to \(k^\ast = \pi/(4\theta) - 1/2\), so \(\lvert k_{\mathrm{opt}} - k^\ast \rvert \le 1/2\). The angle after \(k\) iterations is \((2k+1)\theta\), which changes by \(2\theta\) per step, hence \(\lvert (2k_{\mathrm{opt}}+1)\theta - \pi/2 \rvert \le \theta\). Since \(\sin^2\) is symmetric about \(\pi/2\) and decreasing away from it, \(P_{k_{\mathrm{opt}}} \ge \sin^2(\pi/2 - \theta) = \cos^2\theta = 1 - \sin^2\theta = 1 - M/N\).</p>

<p><strong>4.</strong> \(P = 1\) after one iteration requires \(3\theta = \pi/2\), i.e. \(\theta = \pi/6\) and \(\sin^2\theta = 1/4 = M/N\). With \(M = 1\) that is \(N = 4\). To see that it is the only case, work with \(\cos 2\theta\) rather than with \(\sin\theta\). Exact success after \(k\) iterations needs \((2k+1)\theta = \pi/2\), so \(2\theta = \pi/(2k+1)\) is a rational multiple of \(\pi\), while \(\cos 2\theta = 1 - 2\sin^2\theta = 1 - 2M/N\) is rational. Niven's theorem in its cosine form — if \(\theta/\pi\) is rational and \(\cos\theta\) is rational then \(\cos\theta \in \lbrace 0, \pm 1/2, \pm 1 \rbrace\) — leaves only finitely many possibilities, and \(\cos(\pi/(2k+1))\) lies strictly between \(1/2\) and \(1\) for every \(k \ge 2\), so no \(k \ge 2\) is possible. Only \(k = 1\) survives, with \(\cos(\pi/3) = 1/2\) and hence \(M/N = 1/4\). Applying Niven's theorem to \(\sin\theta\) directly would not do: \(\sin^2\theta\) rational does not make \(\sin\theta\) rational, and \(\sin(\pi/4) = \sqrt{2}/2\) is the standard counterexample. Code Example 4 shows that a <em>tunable</em> preparation reaches probability 1 for any \(k\), because it is not restricted to \(a = M/N\).</p>

</details>

#### Exercise 3: What the Oracle Really Costs

A predicate $f$ on $n$ bits is evaluated classically in $c_c = 30$ machine operations at $10^{10}$ operations per second. Its reversible quantum circuit needs $g_q = 500$ logical gates including uncomputation, at a logical gate period $t_L$.

  1. Write the wall-clock time of a classical exhaustive scan and of a Grover search, in terms of $N$, and find the crossover $N$ as a function of $t_L$.
  2. Evaluate the crossover bit width $n$ for $t_L = 10^{-6}$ s and for $t_L = 10^{-9}$ s.
  3. At the $t_L = 10^{-6}$ s crossover, how many logical gates must run in sequence, and how long does the quantum run take?
  4. Now suppose the classical side runs on $10^5$ cores. What happens to the crossover?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Classical: \(T_c = (N/2)(c_c/r) = (N/2)(30/10^{10}) = 1.5\times10^{-9} N\) seconds. Quantum: \(T_q = (\pi/4)\sqrt{N} g_q t_L = 392.7\sqrt{N}\,t_L\). Setting \(T_q = T_c\) gives \(\sqrt{N} = 392.7\,t_L/1.5\times10^{-9} = 2.618\times10^{11} t_L\), so \(N_{\times} = 6.85\times10^{22}\,t_L^2\).</p>

<p><strong>2.</strong> At \(t_L = 10^{-6}\): \(N_{\times} = 6.85\times10^{10}\), i.e. \(n = \log_2 N_{\times} = 36.0\), so the crossover is at about 36 bits. At \(t_L = 10^{-9}\): \(N_{\times} = 6.85\times10^{4}\), i.e. \(n = 16.1\). Three decades of clock improvement are worth about 20 bits, consistent with the \(6.6\) bits per decade rule in Section 1.4.</p>

<p><strong>3.</strong> At \(n = 36\), \(N = 6.9\times10^{10}\), \(\sqrt{N} = 2.62\times10^{5}\), so \((\pi/4)\sqrt{N} = 2.06\times10^{5}\) queries and \(1.03\times10^{8}\) logical gates in sequence, taking \(103\) seconds. The classical scan also takes about \(103\) seconds, by construction. Note that a hundred million sequential logical gates already demands a substantial error-corrected machine to reach a crossover on a problem a laptop solves in under two minutes.</p>

<p><strong>4.</strong> Classical time falls by \(10^5\); quantum time is unchanged (a single coherent run cannot use more machines, and splitting the space over \(P\) quantum machines gains only \(\sqrt{P}\)). So \(\sqrt{N_{\times}}\) grows by \(10^5\) and \(N_{\times}\) by \(10^{10}\), moving the crossover from 36 bits to about 69 bits. If instead you allow \(10^5\) quantum machines too, the quantum side gains \(\sqrt{10^5} = 316\), and the crossover lands near 53 bits. Parallelism helps the classical side quadratically more.</p>

</details>

#### Exercise 4: Amplification as a Wrapper

A quantum subroutine $A$ prepares a state whose probability of lying in a recognizable good subspace is $a = 10^{-4}$.

  1. How many amplitude-amplification iterations are needed, and how does that compare with classical repetition?
  2. Each iteration uses one oracle, one $A$ and one $A^{\dagger}$. If $A$ costs 1000 gates and $O$ costs 200, what is the total gate count, and what is the total for the classical strategy if one classical trial costs 1000 gate-equivalents?
  3. The subroutine turns out to be a Monte Carlo sampler that draws 40 random bits internally. What has to change before it can be used as $A$, and what does that cost?
  4. Why does the answer to part 1 not depend on the size of the state space?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\theta = \arcsin\sqrt{10^{-4}} = \arcsin(0.01) = 0.0100002\) rad, so \(k_{\mathrm{opt}} = \lfloor \pi/(4\theta) \rfloor = \lfloor 78.54 \rfloor = 78\), and \(P = \sin^2(157 \times 0.0100002) = 0.999999\). Classical repetition needs about \(1/a = 10^4\) trials for constant success probability; the ratio is \(127\).</p>

<p><strong>2.</strong> Quantum: \(78 \times (1000 + 1000 + 200) = 1.72\times10^{5}\) gates, plus one initial \(A\), so \(1.73\times10^{5}\). Classical: \(10^4 \times 1000 = 10^7\) gate-equivalents. The advantage is a factor of \(58\), smaller than the factor of \(127\) in queries because each amplification iteration pays for \(A\) twice.</p>

<p><strong>3.</strong> The randomness must be made explicit and reversible: the 40 random bits become 40 ancilla qubits prepared in \(H^{\otimes 40}\lvert 0 \rangle\), and the sampler must be rewritten as a reversible circuit acting on them, with no measurement and no discarded intermediates. The costs are 40 extra qubits, the reversibility overhead on the sampler's arithmetic (typically a factor of two or more in gate count, plus ancillas for uncomputation), and the loss of any early-exit shortcuts, since a reversible circuit must run to completion on every branch. This is why "just wrap it in amplitude amplification" is rarely free.</p>

<p><strong>4.</strong> Because the iteration count depends only on \(a\), the overlap of the prepared state with the good subspace, and \(a\) is a property of \(A\) and of the predicate, not of the dimension. Grover's \(\sqrt{N/M}\) is the special case in which \(A = H^{\otimes n}\) makes \(a = M/N\), and it is only there that the space size enters.</p>

</details>

#### Exercise 5: Reading a Claim

A paper abstract states: "We give a quantum algorithm that searches an unsorted database of $N$ entries in $O(\sqrt{N})$ time, providing a quadratic speedup over the best classical algorithm."

  1. Identify every assumption in that sentence that Section 1.4 would challenge.
  2. The paper's method section says the data is loaded into a bucket-brigade QRAM of $O(N)$ qubits with $O(\log N)$ query depth. Does that repair the claim? Give the resource-time accounting.
  3. Rewrite the sentence so that it is defensible.
  4. Give one example of a search problem for which the original sentence, with "database" replaced appropriately, *would* be defensible, and say why.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Four problems. (i) "Database" implies stored data, which requires the oracle to read \(N\) records; the query model charges nothing for this. (ii) "\(O(\sqrt{N})\) time" conflates queries with time; time requires a clock rate and a per-query gate count. (iii) "Best classical algorithm" for scanning a stored table is a linear scan, which is \(O(N)\) sequential but \(O(1)\) with \(O(N)\) parallel hardware — the same resource the QRAM demands. (iv) The claim omits the sequential-coherence requirement, which puts the algorithm in the fully fault-tolerant regime.</p>

<p><strong>2.</strong> No. Bucket-brigade QRAM has \(O(\log N)\) <em>depth</em> but \(\Theta(N)\) <em>qubits</em>, all of which must be error-corrected and kept coherent for the whole run. The resource-time product per query is therefore \(\Theta(N \log N)\), and over \(\Theta(\sqrt{N})\) queries the total is \(\Theta(N^{3/2}\log N)\) — worse than the classical \(\Theta(N)\) scan, which needs \(O(1)\) qubits of working memory. If one is prepared to build \(\Theta(N)\) hardware, the classical comparison becomes a parallel scan in \(O(\log N)\) time with \(O(N)\) classical processors, and the quantum method loses by a wide margin. Bucket-brigade QRAM changes depth, not the resource-time product, and the resource-time product is what the comparison must use.</p>

<p><strong>3.</strong> For instance: "We give a quantum algorithm that, given a predicate \(f\) on \(n\) bits as a reversible circuit, finds a satisfying input using \(\Theta(\sqrt{2^n/M})\) queries to \(f\), which is optimal in the query model. Whether this yields a wall-clock advantage depends on the gate cost of \(f\), on the ratio of logical to classical clock rates, and on the parallel resources available to the classical baseline; it does not apply to search over stored data, for which the input-loading cost dominates."</p>

<p><strong>4.</strong> Preimage search for a cryptographic hash: given \(y\), find \(x\) with \(h(x) = y\). The predicate is a short circuit; nothing of size \(N\) is stored, since candidates are generated; and the best classical attack on a well-designed hash genuinely is brute force, so the base of the comparison is correct. The consequence — that an \(m\)-bit preimage search costs \(2^{m/2}\) quantum queries — is exactly why post-quantum recommendations double symmetric key and hash output lengths. Note even here the parallelism objection survives, which is why the doubling is considered a conservative margin rather than a tight one.</p>

</details>

* * *

## Summary

### Key Takeaways

**1\. The oracle model is where the assumptions live**

  * A phase oracle $O\lvert x \rangle = (-1)^{f(x)}\lvert x \rangle$ and a bit-flip oracle are the same object, related by phase kickback — the mechanism reused in Chapters 2 and 3.
  * The model assumes $f$ is evaluated reversibly and coherently at unit cost, that $f$ is a rule rather than a table, and that there is no structure to exploit. Each assumption is a real restriction.
  * Code Example 2 built the same oracle twice: one line as a black box, 19 Toffolis and 4 ancillas as a circuit for a four-variable 3-SAT formula. Half the Toffolis are uncomputation.

**2\. Grover's algorithm is a rotation in two dimensions**

  * $O$ reflects about $\lvert \mathrm{bad} \rangle$, $D = 2\lvert s \rangle\langle s \rvert - I$ reflects about $\lvert s \rangle$, and two reflections at angle $\theta$ make a rotation by $2\theta$ with $\sin\theta = \sqrt{M/N}$.
  * $P_k = \sin^2((2k+1)\theta)$ exactly, verified to six digits and with the off-plane component below $2.2\times10^{-16}$.
  * $k_{\mathrm{opt}} = \lfloor \pi/(4\theta) \rfloor$, and $P_{k_{\mathrm{opt}}} \ge 1 - M/N$. Measured optima for $n = 4$ to $10$ were 3, 4, 6, 8, 12, 17, 25, matching the formula every time.

**3\. Running longer makes it worse**

  * The success probability is periodic. At $n = 8$ it is $0.99995$ at $k = 12$ and $0.0059$ at $k = 24$.
  * The iteration count depends on $M$, so it cannot be set without knowing $M$ — handled in Code Example 7 by the randomized geometric strategy, or in Chapter 2 by measuring $M$ with phase estimation.
  * Grover is not an anytime algorithm, unlike a classical scan.

**4\. Amplitude amplification is the general statement**

  * Replace $H^{\otimes n}$ by any $A$: with $a = \lVert P_{\mathrm{good}}A\lvert 0 \rangle \rVert^2$ and $\theta = \arcsin\sqrt{a}$, everything carries over unchanged, and $k_{\mathrm{opt}} \approx \pi/(4\sqrt{a})$ regardless of dimension.
  * It requires $A^{\dagger}$, i.e. a reversible preparation, and a recognizable good subspace. Neither is free for a classical randomized algorithm rewritten as $A$.
  * Tuning $a$ so that $(2k+1)\theta = \pi/2$ gives *exact* amplification: probability 1 to $2\times10^{-15}$ in Code Example 4.

**5\. A quadratic speedup is consumed by four separate mechanisms**

  * Oracle cost, the logical-to-classical clock ratio, imperfect parallelism ($1/\sqrt{P}$ against $1/P$), and the requirement that $\Theta(\sqrt{N})$ queries run coherently in sequence.
  * Each decade of clock disadvantage costs about 6.6 bits of crossover. At a $1\ \mu$s logical period against $10^{12}$ classical evaluations per second, the crossover sits near $n = 55$, and needs $10^{10}$ sequential logical gates at $n = 60$.
  * Grover square-roots the cost of *brute force*. Against a classical algorithm with exponent $c < 1/2$ it loses outright.

**6\. Unstructured search is not database search**

  * A coherent lookup into $N$ stored records costs $\Theta(N)$ in the resource-time product, whether sequentially or as a bucket-brigade tree, so Grover on a database is $\Theta(N^{3/2})$ against a classical $\Theta(N)$.
  * The QRAM problem is not an engineering detail; it is a lower bound on the input model, and it applies to any algorithm whose advantage needs coherent access to a dataset of size $N$.
  * What survives is search over a *rule*: cryptographic preimage and key search, amplitude estimation for Monte Carlo, amplification inside other quantum algorithms, and problems whose best classical method genuinely is enumeration.

**Practical implications**

  * When you meet a claimed oracle-based speedup, ask three questions in order: what is the oracle made of, what is the base of the comparison, and does anything of size $N$ have to be loaded?
  * Use the exact iteration count $\lfloor \pi/(4\arcsin\sqrt{M/N}) \rfloor$, not the small-angle form; it costs one `arcsin` and is right for every $M$.
  * Always verify the measured string classically. It costs one query and removes the residual failure probability.
  * If you control the state preparation, tune it for exact amplification rather than accepting $1 - O(a)$.

### Where This Leads

The gap left open twice in this chapter — that the optimal iteration count needs $M$, and that $M$ is generally unknown — is closed by measuring the rotation angle itself. The Grover operator $G$ has eigenvalues $e^{\pm 2i\theta}$ in the plane it acts on, so estimating that phase estimates $\theta$, and hence $M = N\sin^2\theta$. The tool for estimating a phase is the subject of Chapter 2, and it turns out to be far more important than the counting application: phase estimation is the algorithm that extracts an eigenvalue of a unitary, and once the unitary is $e^{-iHt}$ for a Hamiltonian $H$, it becomes the fault-tolerant successor to the variational methods of the introductory course. Chapter 2 builds the quantum Fourier transform, then phase estimation on top of it, and is honest about the one thing the Fourier transform cannot do: let you read out the amplitudes it has just computed.

[← Series Top](<index.html>) [Chapter 2: QFT and Phase Estimation →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * Every resource estimate in this course is a parametric calculation from the formulas stated in the text: the rates and overheads are swept over decades to expose scaling and constant factors, and are not measurements, device specifications, or predictions about any machine.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
