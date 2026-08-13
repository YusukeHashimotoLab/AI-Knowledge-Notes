---
title: "Chapter 2: Circuit Optimization and Gate Synthesis"
chapter_title: "Chapter 2: Circuit Optimization and Gate Synthesis"
subtitle: Rewrite Rules That Preserve Meaning, the Euler and KAK Decompositions, and Why One Gate Costs More Than All the Others
reading_time: 45-50 minutes
difficulty: Advanced
code_examples: 7
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-software-stack-introduction/chapter-2.html>) | Last sync: 2026-08-13

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to the Quantum Software Stack](<index.html>) > Chapter 2

Chapter 1 built the layer that everything else in this course is a pass over, and it ended with a measurement rather than a result: the peephole rules of its Code Example 5 removed nothing at all from a routed circuit, and the reason was that the gates they should have cancelled were adjacent *on a qubit* but not adjacent *in the list*. This chapter fixes that, and then keeps going. It is about the two things a compiler does to a circuit once it has one: **rewrite** it into a shorter circuit that means the same thing, and **synthesize** an arbitrary unitary into whatever gates the machine actually has.

The two halves need different mathematics. Rewriting is combinatorial — a set of local identities, a notion of when two gates may be exchanged, and a fixpoint loop — and its correctness is a property that must be tested rather than argued, which is why the unitary-equivalence checker of Chapter 1 appears in every single example below. Synthesis is linear algebra: the Euler decomposition for one qubit, the KAK decomposition for two, and for the fault-tolerant gate set a counting argument that puts a hard floor under the cost of a single arbitrary rotation. The last of those is the most consequential number in the chapter, and it is measured rather than quoted: an arbitrary single-qubit rotation to six decimal digits costs about ten Toffolis.

## Learning Objectives

After completing this chapter, you will be able to:

  * State the three families of local rewrite rule — fusion, cancellation, commutation — as identities between circuit fragments, and verify each one against the phase-free matrix comparison rather than trusting it
  * Write a commutation predicate that is *sound* rather than complete, explain why soundness is the property a rewriting pass needs, and measure how much completeness a syntactic rule set gives up
  * Implement a peephole optimizer that reaches a fixpoint, prove termination from the fact that every successful rule shortens the circuit, and measure its yield on seeded random circuits and on circuits produced by a compiler
  * Derive the ZYZ decomposition $U = e^{i\delta} R_z(a) R_y(b) R_z(c)$ for any $U \in U(2)$, handle the two degenerate branches, and explain why a synthesis routine must return the phase $\delta$ as well as the three angles
  * Build controlled-$U$ from two CX gates and SWAP from three, state the KAK decomposition, and use the local invariants of a two-qubit gate to read off the minimum CX count it requires
  * Explain why the fault-tolerant gate set is Clifford$+T$ rather than the gate set of Chapter 1, and why the $T$ count and not the gate count is the resource that matters there
  * Derive the bound $t \gtrsim 3\log_2(1/\varepsilon)$ on the $T$ count of an $\varepsilon$-accurate rotation from a counting argument, and confirm it by exhaustive search

* * *

## 2.1 Rewrite Rules

### What a rule is

A **rewrite rule** is a pair of circuit fragments with the same unitary. That is the whole definition, and the only subtlety in it is the word "same", which in this course means *equal up to a global phase* — the relation Chapter 1 built `phase_free_error` to test. A rule is applied by finding the left-hand side in a circuit and replacing it with the right-hand side, and the replacement is legal precisely when the two sides pass that test.

There are exactly three ways a local rule can shorten a circuit, and every peephole optimizer ever written is some collection of them.

| Family | Shape of the rule | Example | What it needs to know |
| --- | --- | --- | --- |
| **Fusion** | two gates on the same qubits become one | $R_z(a)\,R_z(b) = R_z(a+b)$, $T\,T = S$ | that the two gates lie in the same one-parameter subgroup |
| **Cancellation** | two gates become none | $H\,H = I$, $\mathrm{CX}\,\mathrm{CX} = I$ | that the gate is an involution |
| **Commutation** | two gates are exchanged, so that a third rule can fire | $Z_0\,\mathrm{CX}_{0,1} = \mathrm{CX}_{0,1}\,Z_0$ | which pairs of gates commute |

Commutation is the one that does the real work, and it is also the only one of the three that does not shorten anything by itself. Its role is to make the other two applicable: two CX gates separated by a rotation on the control cannot be cancelled by a rule that looks at adjacent list entries, but sliding the rotation past one of them makes them adjacent, and then they annihilate. That single mechanism is the difference between Chapter 1's optimizer, which found nothing after routing, and this chapter's, which removes every two-qubit gate from the demonstration circuit of Code Example 3.

One more thing has to be said before any rule can be applied. Our circuits are Python lists, and in a list the gate after index $i$ is the gate at index $i+1$; on a circuit, the gate "after" a gate $g$ is the next gate that *touches one of $g$'s qubits*, since everything in between acts on other qubits and commutes with $g$ trivially. Chapter 1's `next_touching` implements exactly that, and every rule below is stated in terms of it. Production compilers make the same point structurally, by holding the circuit as a directed acyclic graph instead of a list.

### Soundness versus completeness

A commutation predicate can be wrong in two directions, and they are not symmetric.

  * If it says two gates commute when they do not, every pass built on it silently changes the meaning of circuits. This is **unsoundness**, and it is fatal.
  * If it says two gates do not commute when they do, the optimizer misses an opportunity. This is **incompleteness**, and it is merely a lost optimization.

So a predicate is written to be conservative, and its incompleteness becomes a number to be measured rather than a bug to be fixed.

### Code Example 1: The Contract, Re-listed

Everything in this chapter runs on the three modules of Chapter 1. They are reproduced here so that the chapter is self-contained. The first is the state-vector simulator of [Introduction to Quantum Computing](<../quantum-computing-introduction/chapter-2.html>) — the functions this chapter uses, verbatim; `probs` and `sample` from that file are not needed here and are omitted.

```python
"""Minimal state-vector simulator (big-endian: qubit 0 = leftmost = most significant).

Save this file as qcsim.py; every later example does `from qcsim import *`.
"""
import numpy as np

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
```

The second is the circuit IR, verbatim from Chapter 1's Code Example 2. The gate-tuple format, the big-endian qubit order, and the three signatures `run_circuit`, `circuit_depth`, `gate_counts` are the contract that every chapter of this course shares.

```python
"""Chapter 1, Example 2: the circuit IR of this course.

A circuit is a list of gate tuples. Gate names are strings; qubits are ints
(big-endian, qubit 0 leftmost). Save this file as qir.py; every later example
does `from qir import *`, and every later chapter re-lists it.

    ("h", q)   ("x", q)   ("z", q)   ("s", q)   ("t", q)
    ("rx", theta, q)      ("ry", theta, q)      ("rz", theta, q)
    ("cx", control, target)                     ("cz", q1, q2)
"""
import numpy as np
from qcsim import *

CZ4 = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)

FIXED_1Q = {"h": H, "x": X, "z": Z, "s": S, "t": T}
ROT_1Q = {"rx": rx, "ry": ry, "rz": rz}
TWO_Q = ("cx", "cz")


def gate_qubits(g):
    """The qubits one gate tuple touches, in the order they are written."""
    if g[0] in ROT_1Q:
        return (g[2],)
    if g[0] in TWO_Q:
        return (g[1], g[2])
    if g[0] in FIXED_1Q:
        return (g[1],)
    raise ValueError(f"unknown gate name {g[0]!r}")


def apply_ir_gate(state, g, n):
    """Apply one gate tuple to an n-qubit state vector."""
    name = g[0]
    if name in FIXED_1Q:
        return apply_gate(state, FIXED_1Q[name], [g[1]], n)
    if name in ROT_1Q:
        return apply_gate(state, ROT_1Q[name](g[1]), [g[2]], n)
    if name == "cx":
        return cnot(state, g[1], g[2], n)
    if name == "cz":
        return apply_gate(state, CZ4, [g[1], g[2]], n)
    raise ValueError(f"unknown gate name {name!r}")


def run_circuit(circ, n, psi0=None):
    """Execute a gate-tuple list on the state-vector simulator; return the final state.

    psi0 defaults to |00...0>. Gates are applied left to right, so the matrix
    of the circuit is the product of the gate matrices in reverse order.
    """
    state = ket("0" * n) if psi0 is None else np.asarray(psi0, dtype=complex)
    for g in circ:
        state = apply_ir_gate(state, g, n)
    return state


def circuit_depth(circ, n):
    """Greedy layering by qubit disjointness: how many layers the circuit needs.

    Every gate is assumed to take one unit of time, which is false on hardware
    and is corrected in Chapter 4.
    """
    ready = [0] * n              # first free layer of each qubit
    for g in circ:
        qs = gate_qubits(g)
        layer = max(ready[q] for q in qs)
        for q in qs:
            ready[q] = layer + 1
    return max(ready) if n else 0


def gate_counts(circ):
    """Gate name -> count, plus the key "2q" holding the total of two-qubit gates."""
    counts = {}
    for g in circ:
        counts[g[0]] = counts.get(g[0], 0) + 1
    counts["2q"] = sum(counts.get(name, 0) for name in TWO_Q)
    return counts
```

The third is the checker, verbatim from Chapter 1's Code Example 4. Save it as `qcheck.py`.

```python
"""The unitary-equivalence checker of Chapter 1, re-listed.

Save this file as qcheck.py; every later example does `from qcheck import *`.
"""
import numpy as np
from qir import *


def unitary_of(circ, n):
    """The 2^n x 2^n matrix of a circuit: run it once on each basis state."""
    dim = 2 ** n
    U = np.empty((dim, dim), dtype=complex)
    for j in range(dim):
        e = np.zeros(dim, dtype=complex)
        e[j] = 1.0
        U[:, j] = run_circuit(circ, n, psi0=e)
    return U


def best_global_phase(U, V):
    """The phase that makes e^{i phi} V as close to U as a phase can make it.

    It comes from the Hilbert-Schmidt overlap tr(V^dagger U). By Cauchy-Schwarz
    the modulus of that overlap reaches 2^n exactly when U = e^{i phi} V, so an
    overlap near zero is already a proof that the two are inequivalent.
    """
    tr = np.trace(V.conj().T @ U)
    return 1.0 + 0.0j if abs(tr) < 1e-12 else tr / abs(tr)


def phase_free_error(U, V):
    """max |U - e^{i phi} V| after removing the best global phase."""
    return float(np.max(np.abs(U - best_global_phase(U, V) * V)))


def assert_equivalent(a, b, n, label="", atol=1e-10):
    """The test that guards every rewriting pass in this course."""
    err = phase_free_error(unitary_of(a, n), unitary_of(b, n))
    if err > atol:
        raise AssertionError(f"{label}: circuits differ, max error {err:.3e}")
    return err
```

A short exercise of all three, and the accept/REJECT decision that guards every pass in this chapter.

```python
"""Chapter 2, Example 1: the contract, exercised before anything is rewritten."""
import numpy as np
from qcheck import *

bell = [("h", 0), ("cx", 0, 1)]
print(f"The IR of Chapter 1: {bell}, depth {circuit_depth(bell, 2)}, "
      f"counts {gate_counts(bell)}")

# Four candidate rewrites. Three are rules of Section 2.1; one is wrong.
candidates = [
    ("H H -> (nothing)", 1, [("h", 0), ("h", 0)], []),
    ("Rz(0.4) Rz(0.9) -> Rz(1.3)", 1,
     [("rz", 0.4, 0), ("rz", 0.9, 0)], [("rz", 1.3, 0)]),
    ("CX CX -> (nothing)", 2, [("cx", 0, 1), ("cx", 0, 1)], []),
    ("H H -> X", 1, [("h", 0), ("h", 0)], [("x", 0)]),
]
header = f"{'candidate rewrite':<30}{'n':>3}{'phase-free error':>18}  verdict"
print(f"\n{header}")
print("-" * len(header))
for label, n, before, after in candidates:
    err = phase_free_error(unitary_of(before, n), unitary_of(after, n))
    print(f"{label:<30}{n:>3}{err:>18.2e}  "
          f"{'accept' if err < 1e-10 else 'REJECT'}")
```

```text
The IR of Chapter 1: [('h', 0), ('cx', 0, 1)], depth 2, counts {'h': 1, 'cx': 1, '2q': 1}

candidate rewrite               n  phase-free error  verdict
------------------------------------------------------------
H H -> (nothing)                1          2.22e-16  accept
Rz(0.4) Rz(0.9) -> Rz(1.3)      1          1.11e-16  accept
CX CX -> (nothing)              2          0.00e+00  accept
H H -> X                        1          1.00e+00  REJECT
```

**What to look for.** The last row is the reason the checker exists. Three of the four candidate rewrites are rules of the table above; the fourth is a plausible-looking claim that is simply false, and the phase-free error separates the two cases by sixteen orders of magnitude. Every pass in this chapter is wrapped in that test, and a pass that cannot pass it is a bug rather than an optimization.

### Code Example 2: The Rules, and a Commutation Predicate

The identities first, then the predicate, then an exhaustive check of the predicate against matrix commutation.

```python
"""Chapter 2, Example 2: the rewrite rules, and a commutation predicate.
Continues from Example 1 (same session)."""
import itertools
import numpy as np
from qcheck import *

PI = np.pi

# ---- the identities the optimizer of Example 3 is built out of -----------
IDENTITIES = [
    ("H H = I", 1, [("h", 0), ("h", 0)], []),
    ("S S = Z", 1, [("s", 0), ("s", 0)], [("z", 0)]),
    ("T T = S", 1, [("t", 0), ("t", 0)], [("s", 0)]),
    ("Rz(a) Rz(b) = Rz(a+b)", 1,
     [("rz", 0.4, 0), ("rz", -1.1, 0)], [("rz", -0.7, 0)]),
    ("H X H = Z", 1, [("h", 0), ("x", 0), ("h", 0)], [("z", 0)]),
    ("CX CX = I", 2, [("cx", 0, 1), ("cx", 0, 1)], []),
    ("CZ = H(1) CX H(1)", 2,
     [("cz", 0, 1)], [("h", 1), ("cx", 0, 1), ("h", 1)]),
    ("Z(0) CX = CX Z(0)", 2,
     [("z", 0), ("cx", 0, 1)], [("cx", 0, 1), ("z", 0)]),
    ("X(1) CX = CX X(1)", 2,
     [("x", 1), ("cx", 0, 1)], [("cx", 0, 1), ("x", 1)]),
    ("CX Z(1) CX = Z(0) Z(1)", 2,
     [("cx", 0, 1), ("z", 1), ("cx", 0, 1)], [("z", 0), ("z", 1)]),
    ("SWAP = 3 CX", 2, [("cx", 0, 1), ("cx", 1, 0), ("cx", 0, 1)],
     [("cx", 1, 0), ("cx", 0, 1), ("cx", 1, 0)]),
    ("CX(0,1) CX(0,2) commute", 3,
     [("cx", 0, 1), ("cx", 0, 2)], [("cx", 0, 2), ("cx", 0, 1)]),
    ("CX(0,1) CX(1,2) do NOT", 3,
     [("cx", 0, 1), ("cx", 1, 2)], [("cx", 1, 2), ("cx", 0, 1)]),
]

header = f"{'identity':<28}{'n':>3}{'phase-free error':>18}  holds"
print(header)
print("-" * len(header))
for label, n, left, right in IDENTITIES:
    err = phase_free_error(unitary_of(left, n), unitary_of(right, n))
    print(f"{label:<28}{n:>3}{err:>18.2e}  {'yes' if err < 1e-10 else 'NO'}")

# ---- the commutation predicate the optimizer will consult ----------------
DIAGONAL = ("z", "s", "t", "rz", "cz")     # diagonal in the computational basis
X_LIKE = ("x", "rx")                       # diagonal in the X basis


def family(g):
    """The commuting family of a gate: 'diag', 'x', 'y', or None."""
    if g[0] in DIAGONAL:
        return "diag"
    if g[0] in X_LIKE:
        return "x"
    if g[0] == "ry":
        return "y"
    return None


def commutes(g, h):
    """A sound, deliberately incomplete test for "g h = h g".

    Sound: it never returns True for a pair that does not commute, so a pass
    built on it cannot change the meaning of a circuit. Incomplete: it knows
    only the rules above, so accidental commutations are missed.
    """
    qg, qh = set(gate_qubits(g)), set(gate_qubits(h))
    if not (qg & qh):
        return True                            # disjoint supports
    if g == h:
        return True                            # a gate commutes with itself
    fg, fh = family(g), family(h)
    if fg == "diag" and fh == "diag":
        return True                            # diagonal gates all commute
    if len(qg) == 1 and len(qh) == 1 and fg is not None and fg == fh:
        return True                            # same rotation axis, same qubit
    if g[0] == "cx" and h[0] == "cx":
        return g[1] == h[1] or g[2] == h[2]    # shared control or shared target
    for a, b in ((g, h), (h, g)):
        if a[0] == "cx" and len(gate_qubits(b)) == 1:
            q = gate_qubits(b)[0]
            return family(b) == "diag" if q == a[1] else family(b) == "x"
        if a[0] == "cx" and b[0] == "cz":
            return a[2] not in gate_qubits(b)
        if a[0] == "cz" and len(gate_qubits(b)) == 1:
            return family(b) == "diag"
    return False


# ---- validate the predicate against brute-force matrix commutation ------
LIBRARY = ([(name, q) for name in ("h", "x", "z", "s", "t") for q in range(3)]
           + [(name, 0.7, q) for name in ("rx", "ry", "rz") for q in range(3)]
           + [("cx", a, b) for a, b in itertools.permutations(range(3), 2)]
           + [("cz", a, b) for a, b in itertools.combinations(range(3), 2)]
           + [("ry", 0.0, 0), ("rz", 2 * PI, 1)])   # two disguised identities

mats = {g: unitary_of([g], 3) for g in LIBRARY}
truth_yes = predicate_yes = unsound = missed = 0
examples = []
for g, h in itertools.product(LIBRARY, repeat=2):
    A, B = mats[g], mats[h]
    truth = np.max(np.abs(A @ B - B @ A)) < 1e-12
    said = commutes(g, h)
    truth_yes += truth
    predicate_yes += said
    if said and not truth:
        unsound += 1
    if truth and not said:
        missed += 1
        if len(examples) < 3 and g[0] != h[0]:
            examples.append((g, h))

print(f"\nExhaustive check of the predicate on {len(LIBRARY)} gates over 3 qubits")
print(f"  ordered pairs tested            : {len(LIBRARY) ** 2}")
print(f"  pairs that really commute       : {truth_yes}")
print(f"  pairs the predicate accepts     : {predicate_yes}")
print(f"  unsound answers (must be zero)  : {unsound}")
print(f"  true commutations missed        : {missed}")
print(f"  examples of misses              : {examples[0]} | {examples[1]}")
```

```text
identity                      n  phase-free error  holds
--------------------------------------------------------
H H = I                       1          2.22e-16  yes
S S = Z                       1          0.00e+00  yes
T T = S                       1          1.11e-16  yes
Rz(a) Rz(b) = Rz(a+b)         1          5.55e-17  yes
H X H = Z                     1          2.22e-16  yes
CX CX = I                     2          0.00e+00  yes
CZ = H(1) CX H(1)             2          2.22e-16  yes
Z(0) CX = CX Z(0)             2          0.00e+00  yes
X(1) CX = CX X(1)             2          0.00e+00  yes
CX Z(1) CX = Z(0) Z(1)        2          0.00e+00  yes
SWAP = 3 CX                   2          0.00e+00  yes
CX(0,1) CX(0,2) commute       3          0.00e+00  yes
CX(0,1) CX(1,2) do NOT        3          1.00e+00  NO

Exhaustive check of the predicate on 35 gates over 3 qubits
  ordered pairs tested            : 1225
  pairs that really commute       : 889
  pairs the predicate accepts     : 851
  unsound answers (must be zero)  : 0
  true commutations missed        : 38
  examples of misses              : (('h', 0), ('ry', 0.0, 0)) | (('h', 1), ('rz', 6.283185307179586, 1))
```

**What to look for.** Two rows of the identity table are not identities. `CX(0,1) CX(1,2) do NOT` commute, and the error of $1.00$ says so; it is in the list because a rule set is defined as much by what it excludes as by what it contains. And `CZ = H(1) CX H(1)` holds exactly, which is the basis translation every superconducting compiler performs, in one line.

The predicate report is the honest part. Zero unsound answers over 1225 ordered pairs is the property that matters: no pass built on this predicate can change the meaning of a circuit. The 38 misses all involve the two gates deliberately planted in the library — $R_y(0)$ and $R_z(2\pi)$, the identity written in a form a syntactic rule cannot recognize. The fix is not a better predicate but a **canonicalization** pass that folds angles into $(-\pi, \pi]$ and deletes zero rotations before any rule is consulted, and the optimizer of Code Example 3 has one.

### Code Example 3: A Peephole Optimizer

The optimizer is two passes run to a fixpoint. `local_pass` applies the one-gate and adjacent-pair rules; `commute_pass` slides a single gate forward through gates it commutes with, and fires only when the slide creates a merge. Termination is immediate: every successful rule strictly shortens the circuit, so the loop cannot run more than `len(circ)` times.

```python
"""Chapter 2, Example 3: a peephole optimizer built from those rules.
Continues from Example 2 (same session)."""
import numpy as np
from qcheck import *

TWO_PI = 2.0 * np.pi
DIAG_ANGLE = {"z": np.pi, "s": np.pi / 2, "t": np.pi / 4}
NAMED = (("diag", np.pi, "z"), ("diag", np.pi / 2, "s"),
         ("diag", np.pi / 4, "t"), ("x", np.pi, "x"))
AXIS_GATE = {"diag": "rz", "x": "rx", "y": "ry"}


def as_angle(g):
    """(axis, angle, qubit) if g is a rotation about x, y or z; else None."""
    if g[0] in DIAG_ANGLE:
        return "diag", DIAG_ANGLE[g[0]], g[1]
    if g[0] == "x":
        return "x", np.pi, g[1]
    if g[0] in ("rx", "ry", "rz"):
        return {"rz": "diag", "rx": "x", "ry": "y"}[g[0]], g[1], g[2]
    return None


def wrap(theta):
    """Fold an angle into (-pi, pi]; the difference is a global phase only."""
    t = (theta + np.pi) % TWO_PI - np.pi
    return np.pi if abs(t + np.pi) < 1e-12 else t


def canonical(axis, theta, q):
    """Zero gates, one named gate, or one rotation -- whichever is shortest."""
    theta = wrap(theta)
    if abs(theta) < 1e-12:
        return []
    for a, ref, name in NAMED:
        if a == axis and abs(theta - ref) < 1e-12:
            return [(name, q)]
    return [(AXIS_GATE[axis], theta, q)]


def simplify_one(g):
    """The one-gate rules: fold the angle, delete an identity, name what can be named."""
    a = as_angle(g)
    if a is None:
        return None
    out = canonical(*a)
    if len(out) == 1 and out[0][0] == g[0]:
        folded = g[0] not in ("rx", "ry", "rz") or abs(wrap(g[1]) - g[1]) < 1e-9
        return None if folded else out
    return out


def merge_two(g, h):
    """The replacement for an adjacent pair on the same qubits, or None."""
    ag, ah = as_angle(g), as_angle(h)
    if ag is not None and ah is not None and ag[0] == ah[0] and ag[2] == ah[2]:
        return canonical(ag[0], ag[1] + ah[1], ag[2])      # same axis: add angles
    if g == h and g[0] in ("h", "cx"):
        return []                                          # self-inverse
    if g[0] == "cz" and h[0] == "cz":
        return []                                          # CZ is symmetric
    return None


def next_touching(circ, i, qs):
    """Index of the first gate after i that shares a qubit with qs, or len(circ)."""
    j = i + 1
    while j < len(circ) and not (qs & set(gate_qubits(circ[j]))):
        j += 1
    return j


def local_pass(circ):
    """One sweep of the one-gate and adjacent-pair rules."""
    out, i, fired = list(circ), 0, 0
    while i < len(out):
        rule = simplify_one(out[i])
        if rule is not None:
            out[i:i + 1] = rule
            fired += 1
            i = max(i - 1, 0)
            continue
        qs = set(gate_qubits(out[i]))
        j = next_touching(out, i, qs)
        if j < len(out) and set(gate_qubits(out[j])) == qs:
            merged = merge_two(out[i], out[j])
            if merged is not None:
                del out[j]
                out[i:i + 1] = merged
                fired += 1
                i = max(i - 1, 0)
                continue
        i += 1
    return out, fired


def commute_pass(circ):
    """Slide one gate forward through gates it commutes with, if that lets it merge."""
    for i, g in enumerate(circ):
        qs = set(gate_qubits(g))
        for j in range(i + 1, len(circ)):
            h = circ[j]
            if not (set(gate_qubits(h)) & qs):
                continue                        # disjoint supports: slide past
            merged = merge_two(g, h) if set(gate_qubits(h)) == qs else None
            if merged is not None:
                if j > i + 1:                   # j == i + 1 is the local pass's job
                    return circ[:i] + circ[i + 1:j] + merged + circ[j + 1:], 1
                break
            if not commutes(g, h):
                break                           # blocked: nothing to do for this i
    return list(circ), 0


def peephole(circ, trace=None):
    """Run both passes to a fixpoint. Each success strictly shortens the circuit."""
    cur = list(circ)
    while True:
        cur, a = local_pass(cur)
        cur, b = commute_pass(cur)
        if trace is not None:
            trace.append((len(cur), a, b))
        if a == 0 and b == 0:
            return cur


# ---- a hand-built circuit that exercises every rule ----------------------
demo = [("h", 0), ("h", 0),                            # H H = I
        ("t", 1), ("t", 1),                            # T T = S
        ("rz", 0.3, 2), ("rz", -0.3, 2),               # inverse rotations
        ("cx", 0, 1), ("rz", 0.5, 0), ("cx", 0, 1),    # Rz slides off the control
        ("h", 2), ("x", 2), ("h", 2),                  # H X H = Z -- not our rule
        ("ry", 0.0, 1),                                # an identity in disguise
        ("s", 1), ("s", 1),                            # S S = Z
        ("cz", 0, 2), ("t", 0), ("cz", 2, 0)]          # T slides through CZ

trace = []
opt = peephole(demo, trace)
print("A circuit built to trigger every rule of Section 2.1")
print(f"  before: {len(demo)} gates, depth {circuit_depth(demo, 3)}, "
      f"{gate_counts(demo)['2q']} two-qubit")
print(f"  after : {len(opt)} gates, depth {circuit_depth(opt, 3)}, "
      f"{gate_counts(opt)['2q']} two-qubit")


def show(circ):
    """A circuit as a readable string, angles rounded."""
    return " ".join(
        g[0] + "(" + ",".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                              for v in g[1:]) + ")" for g in circ) or "(empty)"


print(f"  result: {show(opt)}")
print(f"  equivalence error: {assert_equivalent(demo, opt, 3, 'peephole'):.2e}")

print(f"\n{'round':>6}{'gates left':>12}{'local rules':>13}{'commutations':>14}")
print("-" * 45)
for k, (size, a, b) in enumerate(trace, start=1):
    print(f"{k:>6}{size:>12}{a:>13}{b:>14}")
```

```text
A circuit built to trigger every rule of Section 2.1
  before: 18 gates, depth 8, 4 two-qubit
  after : 5 gates, depth 3, 0 two-qubit
  result: rz(-1.5708,1) h(2) x(2) h(2) rz(1.2854,0)
  equivalence error: 4.58e-16

 round  gates left  local rules  commutations
---------------------------------------------
     1           9            5             1
     2           7            1             1
     3           5            0             1
     4           5            0             0
```

**What to look for.** Eighteen gates become five, and all four two-qubit gates disappear. Neither pass alone would have done it: the $R_z(0.5)$ between the two CX gates and the $T$ between the two CZ gates block the cancellations, and it takes a commutation to remove each block. The round table shows the interleaving — five local rules in the first round, then one commutation per round for three rounds as each unblocking exposes the next one.

What survives is instructive too. $R_z(-\pi/2)$ on qubit 1 is the four diagonal gates $T\,T\,S\,S$ folded into one; $R_z(0.5 + \pi/4)$ on qubit 0 is the rotation that was between the CX gates fused with the $T$ that was between the CZ gates. And $H\,X\,H$ on qubit 2 is untouched, because no rule in Code Example 2 rewrites a conjugation by $H$ — the identity $H X H = Z$ is in the table but the *optimizer* has no rule of that shape. Section 2.2 removes it by a different route, which is the honest way round: rather than adding one conjugation rule at a time, resynthesize.

### Code Example 4: What the Optimizer Buys

A yield has to be measured on inputs, and there is no such thing as a neutral input. The generator here is Chapter 1's `random_circuit`, re-listed verbatim so the numbers are comparable with the weaker passes of that chapter, and the first block is a test with a known answer: a circuit followed by its own inverse must optimize to the empty circuit.

```python
"""Chapter 2, Example 4: how much the optimizer buys, measured.
Continues from Example 3 (same session)."""
import numpy as np
from qcheck import *

NAMES = ["h", "x", "z", "s", "t", "rx", "ry", "rz", "cx", "cz"]
ANGLES = [k * np.pi / 4 for k in (-3, -2, -1, 1, 2, 3, 4)]


def random_circuit(n, length, rng):
    """A random circuit over the IR gate set; angles are multiples of pi/4.

    Re-listed verbatim from Chapter 1, Example 5, so that the numbers below are
    comparable with the ones the weaker passes of that chapter produced.
    """
    circ = []
    for _ in range(length):
        name = NAMES[int(rng.integers(len(NAMES)))]
        if name in TWO_Q:
            a, b = (int(v) for v in rng.choice(n, size=2, replace=False))
            circ.append((name, a, b))
        elif name in ROT_1Q:
            circ.append((name, ANGLES[int(rng.integers(len(ANGLES)))],
                         int(rng.integers(n))))
        else:
            circ.append((name, int(rng.integers(n))))
    return circ


def inverse(circ):
    """The inverse circuit: reverse the order, invert every gate."""
    inv = []
    for g in reversed(circ):
        if g[0] in ("h", "x", "z", "cx", "cz"):
            inv.append(g)                                  # self-inverse
        elif g[0] == "s":
            inv.append(("rz", -np.pi / 2, g[1]))
        elif g[0] == "t":
            inv.append(("rz", -np.pi / 4, g[1]))
        else:
            inv.append((g[0], -g[1], g[2]))
    return inv


# ---- a test with a known answer: U followed by U-inverse must vanish -----
print("Sanity test: a circuit followed by its own inverse must optimize to nothing")
print(f"{'seed':>5}{'n':>4}{'gates in':>10}{'gates out':>11}{'error':>11}")
print("-" * 41)
for seed in range(3):
    rng = np.random.default_rng(seed)
    c = random_circuit(4, 30, rng)
    pair = c + inverse(c)
    out = peephole(pair)
    print(f"{seed:>5}{4:>4}{len(pair):>10}{len(out):>11}"
          f"{assert_equivalent(pair, out, 4, 'inverse pair'):>11.1e}")

# ---- seeded random circuits ---------------------------------------------
print("\nSeeded random circuits, 5 qubits, 60 gates, verified one by one")
head = (f"{'seed':>5}{'gates':>13}{'depth':>11}{'two-qubit':>12}"
        f"{'phase-free err':>16}")
print(head)
print("-" * len(head))
tot = {"g0": 0, "g1": 0, "d0": 0, "d1": 0, "q0": 0, "q1": 0}
worst = 0.0
n, m, trials = 5, 60, 200
for seed in range(trials):
    rng = np.random.default_rng(1000 + seed)
    c = random_circuit(n, m, rng)
    o = peephole(c)
    err = assert_equivalent(c, o, n, f"seed {seed}")
    worst = max(worst, err)
    g0, g1 = len(c), len(o)
    d0, d1 = circuit_depth(c, n), circuit_depth(o, n)
    q0, q1 = gate_counts(c)["2q"], gate_counts(o)["2q"]
    for k, v in zip(("g0", "g1", "d0", "d1", "q0", "q1"), (g0, g1, d0, d1, q0, q1)):
        tot[k] += v
    if seed < 4:
        print(f"{seed:>5}{f'{g0} -> {g1}':>13}{f'{d0} -> {d1}':>11}"
              f"{f'{q0} -> {q1}':>12}{err:>16.1e}")
print(f"{'...':>5}")
print(f"\nAveraged over {trials} circuits (n = {n}, {m} gates each):")
print(f"  gates      {tot['g0']/trials:6.2f} -> {tot['g1']/trials:6.2f}"
      f"   ({100*(1-tot['g1']/tot['g0']):5.1f}% removed)")
print(f"  depth      {tot['d0']/trials:6.2f} -> {tot['d1']/trials:6.2f}"
      f"   ({100*(1-tot['d1']/tot['d0']):5.1f}% removed)")
print(f"  two-qubit  {tot['q0']/trials:6.2f} -> {tot['q1']/trials:6.2f}"
      f"   ({100*(1-tot['q1']/tot['q0']):5.1f}% removed)")
print(f"  worst equivalence error over all {trials} circuits: {worst:.2e}")
```

```text
Sanity test: a circuit followed by its own inverse must optimize to nothing
 seed   n  gates in  gates out      error
-----------------------------------------
    0   4        60          0    1.6e-15
    1   4        60          0    7.9e-16
    2   4        60          0    1.1e-15

Seeded random circuits, 5 qubits, 60 gates, verified one by one
 seed        gates      depth   two-qubit  phase-free err
---------------------------------------------------------
    0     60 -> 40   23 -> 18    13 -> 11         3.2e-16
    1     60 -> 44   26 -> 22    15 -> 15         2.7e-16
    2     60 -> 47   32 -> 29    15 -> 15         3.3e-16
    3     60 -> 48   22 -> 19    14 -> 14         2.8e-16
  ...

Averaged over 200 circuits (n = 5, 60 gates each):
  gates       60.00 ->  44.20   ( 26.3% removed)
  depth       22.65 ->  18.39   ( 18.8% removed)
  two-qubit   12.13 ->  11.28   (  7.1% removed)
  worst equivalence error over all 200 circuits: 6.78e-16
```

**What to look for.** The inverse-pair test is the strongest correctness evidence in the chapter: sixty gates in, zero out, three times over, with the equivalence check confirming that the empty circuit really is what the input computed. A cancellation rule that were subtly wrong would leave residue or fail the check, and this test catches both. On uniform random circuits the yield is then a quarter of the gates, a fifth of the depth, and — the number that matters — only seven per cent of the two-qubit gates. That last figure is the honest headline. Peephole optimization is a local method, uniform random circuits have few local redundancies by construction, and their two-qubit gates mostly act on different pairs and never meet. Circuits emitted by a compiler are the opposite case, because each stage adds its own basis changes and each of those meets the inverse of the next stage's; the independently translated CX chain of Chapter 1's Example 5 is the shape this optimizer eats.

* * *

## 2.2 Single-Qubit Synthesis

### Three angles and a phase

Every element of $U(2)$ is

$$ U = e^{i\delta}\, R_z(a)\, R_y(b)\, R_z(c), \qquad R_z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \cr 0 & e^{i\theta/2} \end{pmatrix}, \quad R_y(\theta) = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \cr \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix} $$

and the count is right: $U(2)$ has four real parameters, and this form has four. The construction is elementary. Divide $U$ by $\sqrt{\det U}$ to get an element $V$ of $SU(2)$; that fixes $\delta = \frac{1}{2}\arg\det U$. Then

$$ V = \begin{pmatrix} \cos\frac{b}{2}\, e^{-i(a+c)/2} & -\sin\frac{b}{2}\, e^{-i(a-c)/2} \cr \sin\frac{b}{2}\, e^{i(a-c)/2} & \cos\frac{b}{2}\, e^{i(a+c)/2} \end{pmatrix} $$

so the moduli of the first column give $b = 2\arctan\big(|V_{10}|/|V_{00}|\big)$, and the arguments of $V_{11}$ and $V_{10}$ give $(a+c)/2$ and $(a-c)/2$ respectively. Hence

$$ a = \arg V_{11} + \arg V_{10}, \qquad c = \arg V_{11} - \arg V_{10} $$

There are two degenerate branches, and a routine that ignores them returns `nan`. If $V_{00} = 0$ then $b = \pi$ and only $a - c$ is determined; if $V_{10} = 0$ then $b = 0$ and only $a + c$ is determined. In both cases one of the two $R_z$ angles can be set to zero, which is why the synthesis of $Z$, $S$ and $T$ below comes out as a single gate rather than three.

### Why the phase has to be returned

$\delta$ is a global phase and therefore unobservable, and it must nonetheless be returned, for the reason Chapter 1's Exercise 3 gives: **a global phase stops being global as soon as the fragment is placed under a control**. A routine that discards $\delta$ produces a correct single-qubit circuit and an incorrect controlled-$U$, because the phase then lands on one branch of the controlled block and not the other. Section 2.3 needs precisely that, which is why `zyz_angles` returns four numbers.

### Resynthesis as an optimization pass

Once single-qubit synthesis exists, it can be used as an optimizer rather than as a translator. Take any maximal run of consecutive single-qubit gates on one qubit, multiply the matrices, and re-synthesize: the result is at most three rotations, whatever the run contained. This has a property no finite list of rewrite rules can have — it is *complete* for single-qubit runs. $H X H$ is a run of three gates, its product is $Z$, and resynthesis returns one gate without anyone having written down the identity $H X H = Z$.

The pass costs something in exchange: it emits `rz` and `ry` rotations where the input had named gates, which on a machine with a different native set may be a step backwards. Real compilers therefore run it late, and in a basis chosen to match the hardware.

### Code Example 5: ZYZ Decomposition and the Resynthesis Pass

```python
"""Chapter 2, Example 5: ZYZ synthesis, and what it does to a circuit.
Continues from Example 4 (same session)."""
import numpy as np
from qcheck import *


def zyz_angles(U):
    """(delta, a, b, c) with U = exp(i delta) Rz(a) Ry(b) Rz(c), for any U in U(2)."""
    det = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]
    delta = 0.5 * np.angle(det)                  # U / exp(i delta) is in SU(2)
    V = U * np.exp(-1j * delta)
    b = 2.0 * np.arctan2(abs(V[1, 0]), abs(V[0, 0]))
    if abs(V[0, 0]) < 1e-12:                     # b = pi: only a - c is fixed
        a, c = 2.0 * np.angle(V[1, 0]), 0.0
    elif abs(V[1, 0]) < 1e-12:                   # b = 0: only a + c is fixed
        a, c = 2.0 * np.angle(V[1, 1]), 0.0
    else:
        p, m = np.angle(V[1, 1]), np.angle(V[1, 0])
        a, c = p + m, p - m
    return delta, a, b, c


def zyz_circuit(U, q):
    """A single-qubit gate as at most three rotations on qubit q, in circuit order."""
    _, a, b, c = zyz_angles(U)
    out = []
    for axis, theta in (("diag", c), ("y", b), ("diag", a)):
        out += canonical(axis, theta, q)
    return out


def haar_1q(rng):
    """A Haar-random element of U(2), by QR of a complex Gaussian matrix."""
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / abs(np.diag(R)))


rng = np.random.default_rng(11)
worst_exact = worst_free = 0.0
lengths = {0: 0, 1: 0, 2: 0, 3: 0}
for _ in range(2000):
    U = haar_1q(rng)
    delta, a, b, c = zyz_angles(U)
    rebuilt = np.exp(1j * delta) * (rz(a) @ ry(b) @ rz(c))
    worst_exact = max(worst_exact, np.max(np.abs(U - rebuilt)))
    circ = zyz_circuit(U, 0)
    worst_free = max(worst_free, phase_free_error(U, unitary_of(circ, 1)))
    lengths[len(circ)] += 1

print("ZYZ decomposition on 2000 Haar-random elements of U(2)")
print(f"  worst error, phase included : {worst_exact:.2e}")
print(f"  worst error, phase removed  : {worst_free:.2e}")
print(f"  gate counts of the synthesis: {lengths}")

print("\nThe special cases, where one of the two Rz angles is undetermined")
for label, U in [("identity", np.eye(2, dtype=complex)), ("H", H),
                 ("Z", Z), ("T", T), ("Ry(0.7)", ry(0.7))]:
    d, a, b, c = zyz_angles(U)
    circ = zyz_circuit(U, 0)
    print(f"  {label:<10} delta/pi = {d/np.pi:+.3f}  "
          f"(a,b,c)/pi = ({a/np.pi:+.3f},{b/np.pi:+.3f},{c/np.pi:+.3f})  "
          f"-> {len(circ)} gate(s)")


# ---- the pass: collapse every run of single-qubit gates ------------------
def matrix_1q(g):
    """The 2x2 matrix of a single-qubit gate tuple."""
    if g[0] in FIXED_1Q:
        return FIXED_1Q[g[0]]
    if g[0] in ROT_1Q:
        return ROT_1Q[g[0]](g[1])
    return None


def resynthesize(circ, n):
    """Replace every maximal run of single-qubit gates on one qubit by its ZYZ form.

    Runs on different qubits are flushed independently. That reorders gates on
    disjoint qubits, which is always allowed, and the checker confirms it.
    """
    pending = {q: [] for q in range(n)}
    out = []

    def flush(q):
        run, pending[q] = pending[q], []
        if len(run) == 1:
            out.append(run[0][0])              # a single gate cannot be improved
        elif run:
            U = np.eye(2, dtype=complex)
            for _, M in run:
                U = M @ U
            out.extend(zyz_circuit(U, q))

    for g in circ:
        qs = gate_qubits(g)
        M = matrix_1q(g) if len(qs) == 1 else None
        if M is not None:
            pending[qs[0]].append((g, M))
        else:
            for q in qs:
                flush(q)
            out.append(g)
    for q in range(n):
        flush(q)
    return out


print("\nH X H, which the peephole rules of Example 3 could not touch:")
block = [("h", 0), ("x", 0), ("h", 0)]
print(f"  peephole     : {show(peephole(block))}")
print(f"  resynthesized: {show(resynthesize(block, 1))}")
print(f"  error        : {assert_equivalent(block, resynthesize(block, 1), 1):.2e}")

print("\nThe two passes together, on the seeded random circuits of Example 4")
head = (f"{'pipeline':<34}{'gates':>8}{'1q':>6}{'2q':>6}{'depth':>7}"
        f"{'worst err':>12}")
print(head)
print("-" * len(head))
n, m, trials = 5, 60, 200
stages = [("as written", lambda c: c),
          ("peephole only", lambda c: peephole(c)),
          ("ZYZ resynthesis only", lambda c: resynthesize(c, n)),
          ("peephole, resynthesis, peephole",
           lambda c: peephole(resynthesize(peephole(c), n)))]
for label, pipeline in stages:
    g = q = d = 0
    worst = 0.0
    for seed in range(trials):
        c = random_circuit(n, m, np.random.default_rng(1000 + seed))
        o = pipeline(c)
        worst = max(worst, assert_equivalent(c, o, n, label))
        g += len(o)
        q += gate_counts(o)["2q"]
        d += circuit_depth(o, n)
    print(f"{label:<34}{g/trials:>8.2f}{(g-q)/trials:>6.2f}{q/trials:>6.2f}"
          f"{d/trials:>7.2f}{worst:>12.1e}")
```

```text
ZYZ decomposition on 2000 Haar-random elements of U(2)
  worst error, phase included : 8.47e-16
  worst error, phase removed  : 9.49e-16
  gate counts of the synthesis: {0: 0, 1: 0, 2: 0, 3: 2000}

The special cases, where one of the two Rz angles is undetermined
  identity   delta/pi = +0.000  (a,b,c)/pi = (+0.000,+0.000,+0.000)  -> 0 gate(s)
  H          delta/pi = +0.500  (a,b,c)/pi = (+0.000,+0.500,+1.000)  -> 2 gate(s)
  Z          delta/pi = +0.500  (a,b,c)/pi = (+1.000,+0.000,+0.000)  -> 1 gate(s)
  T          delta/pi = +0.125  (a,b,c)/pi = (+0.250,+0.000,+0.000)  -> 1 gate(s)
  Ry(0.7)    delta/pi = +0.000  (a,b,c)/pi = (+0.000,+0.223,+0.000)  -> 1 gate(s)

H X H, which the peephole rules of Example 3 could not touch:
  peephole     : h(0) x(0) h(0)
  resynthesized: z(0)
  error        : 2.22e-16

The two passes together, on the seeded random circuits of Example 4
pipeline                             gates    1q    2q  depth   worst err
-------------------------------------------------------------------------
as written                           60.00 47.87 12.13  22.65     0.0e+00
peephole only                        44.20 32.93 11.28  18.39     6.8e-16
ZYZ resynthesis only                 43.25 31.11 12.13  18.93     1.1e-15
peephole, resynthesis, peephole      36.40 25.14 11.27  16.54     1.0e-15
```

**What to look for.** The decomposition is exact to $10^{-15}$ on 2000 Haar-random inputs, with the phase included and with it removed, and it always produces three gates for a generic input — as it must, since a generic $U(2)$ element has three non-trivial angles. The special-case table shows the degenerate branches working: $Z$, $T$ and $R_y(0.7)$ come out as one gate each, the identity as none, and $H$ as two.

The pipeline table is the honest comparison, and every one of its 800 rows is checked: 600 rewrites verified across the three optimizing pipelines, plus 200 runs of the `as written` pipeline, which is the identity and therefore a no-op control. Alone, the two passes are close and they fail in different places: resynthesis reaches 43.25 gates against the peephole rules' 44.20, but its two-qubit count is 12.13, exactly its input value, because only the peephole rules can remove a two-qubit gate. Only resynthesis bounds a run of single-qubit gates by three rotations regardless of what the run contained, which is how $H X H$ disappears. Together they reach 36.40 gates and depth 16.54 against 60.00 and 22.65 as written, and the closing peephole pass is not decoration: the rotations resynthesis emits are fresh candidates for commutation, and the pass finds two more two-qubit gates than the peephole rules found alone — 2253 against 2255 across the 200 circuits, which is the whole of the difference between the 11.27 and 11.28 averages.

* * *

## 2.3 Two-Qubit Synthesis

### The structure of a two-qubit gate

The single-qubit story generalizes, and the generalization is the central theorem of two-qubit compilation. Any $U \in SU(4)$ factorizes as

$$ U = (A_1 \otimes A_2)\, \exp\big[ i\big( t_x\, X{\otimes}X + t_y\, Y{\otimes}Y + t_z\, Z{\otimes}Z \big) \big]\, (B_1 \otimes B_2) $$

with $A_i, B_i \in SU(2)$. This is the **KAK decomposition**, or the Cartan decomposition of $SU(4)$ with respect to the subgroup $SU(2)\otimes SU(2)$. The middle factor is the **canonical gate**, and the three numbers $(t_x, t_y, t_z)$ — confined to a tetrahedron once the obvious symmetries are quotiented out — are the entire non-local content of $U$. Everything else is local, and local gates are free in the sense that matters: they cost no entanglement and, on hardware, roughly a hundredth of the error of a two-qubit gate.

Two consequences follow. Two gates need the same number of CX gates whenever their canonical coordinates agree, so "how expensive is this gate" is a question about a point in a tetrahedron rather than about a $4\times4$ matrix. And the well-known landmarks are corners of it: CX and CZ at $(\pi/4, 0, 0)$, iSWAP at $(\pi/4, \pi/4, 0)$, SWAP at $(\pi/4, \pi/4, \pi/4)$, the identity at the origin.

### The explicit constructions

Four constructions cover almost everything a compiler emits, and each is exact.

| Target | CX gates | Construction |
| --- | --- | --- |
| CZ | 1 | conjugate the target of a CX by $H$ |
| $\exp(-i\frac{\theta}{2} Z{\otimes}Z)$ | 2 | $\mathrm{CX}$, $R_z(\theta)$ on the target, $\mathrm{CX}$; conjugating by $H\otimes H$ gives the $XX$ version |
| controlled-$U$, generic $U$ | 2 | $A X B X C$ with $ABC = I$, from the ZYZ angles |
| SWAP | 3 | $\mathrm{CX}_{0,1}\,\mathrm{CX}_{1,0}\,\mathrm{CX}_{0,1}$ |

The controlled-$U$ construction is the one worth writing out, because it is where Section 2.2's insistence on returning $\delta$ pays off. Given $U = e^{i\delta}R_z(a)R_y(b)R_z(c)$, put

$$ A = R_z(a)R_y(b/2), \qquad B = R_y(-b/2)R_z\left(-\tfrac{a+c}{2}\right), \qquad C = R_z\left(\tfrac{c-a}{2}\right) $$

Then $ABC = R_z(a)R_z(-a) = I$, so the $|0\rangle$ branch of the control does nothing. And since $X R_z(\theta) X = R_z(-\theta)$ and $X R_y(\theta) X = R_y(-\theta)$,

$$ A\,X B X\,C = R_z(a)\,R_y(b/2)\cdot R_y(b/2) R_z\left(\tfrac{a+c}{2}\right)\cdot R_z\left(\tfrac{c-a}{2}\right) = R_z(a)R_y(b)R_z(c) $$

so the $|1\rangle$ branch applies $U$ up to the factor $e^{i\delta}$, which an $R_z(\delta)$ on the *control* line restores. The two $X$ gates are the two CX gates. Note what would go wrong with a phase-blind implementation: if the emitted rotations were renamed to $S$ or $T$ where the angle allows — which is what the optimizer's `canonical` does, correctly, for uncontrolled gates — then $ABC$ would equal $e^{i\varphi}I$ rather than $I$, the two branches would pick up different phases, and the result would not be controlled-$U$ at all. Code Example 6 has a separate emitter, `rot`, for exactly this reason.

### The CX count, and how to read it off

How many CX gates does a given two-qubit gate need? Simple parameter counting does not answer it: two CX gates with local blocks on both sides and in between already carry $3 \times 2 \times 3 = 18$ free parameters against the 15 of $PU(4)$, so the obstruction is structural rather than dimensional. What resolves it is a pair of quantities that are invariant under local gates on either side.

Let $M$ be the **magic basis**

$$ M = \frac{1}{\sqrt2}\begin{pmatrix} 1 & 0 & 0 & i \cr 0 & i & 1 & 0 \cr 0 & i & -1 & 0 \cr 1 & 0 & 0 & -i \end{pmatrix} $$

rescale $U$ into $SU(4)$, set $\tilde U = M^\dagger U M$ and

$$ m(U) = \tilde U^{\mathsf T}\, \tilde U $$

The spectrum of $m$ is unchanged by $U \mapsto (A_1\otimes A_2) U (B_1 \otimes B_2)$, because in the magic basis a local gate becomes a *real* orthogonal matrix and $m$ is built so that real orthogonal factors cancel. So $\operatorname{tr} m$ and $\operatorname{tr} m^2$ are functions of $(t_x,t_y,t_z)$ alone, and the classification is:

| Condition on $m(U)$ | Minimum CX gates | Canonical coordinates |
| --- | --- | --- |
| $\lvert\operatorname{tr} m\rvert = 4$ and $\operatorname{tr} m^2 = 4$ | 0 | $(0,0,0)$ |
| $\operatorname{tr} m = 0$ and $\operatorname{tr} m^2 = -4$ | 1 | $(\pi/4, 0, 0)$ |
| $\operatorname{tr} m$ real | 2 | $t_z = 0$ |
| otherwise | 3 | generic |

The rescaling by $(\det U)^{-1/4}$ has a fourfold ambiguity that flips the sign of $m$; every entry in the table is insensitive to that sign, which is why it can be used without choosing a branch. And the third row is the trap: SWAP has $\operatorname{tr} m = -4i$, purely imaginary, so $(\operatorname{tr} m)^2$ is real and negative. A criterion phrased on $(\operatorname{tr} m)^2$ instead of on $\operatorname{tr} m$ therefore classifies SWAP as a two-CX gate, which it is not.

That three CX gates always *suffice* is the constructive half of the KAK theorem and is not re-derived here; the explicit circuit for the hardest class, SWAP, is in the table above.

### Code Example 6: Two-Qubit Synthesis and the CX Count

```python
"""Chapter 2, Example 6: two-qubit synthesis and the CX count.
Continues from Example 5 (same session)."""
import numpy as np
from qcheck import *


def controlled(U):
    """The two-qubit matrix applying U to qubit 1 when qubit 0 is |1> (big-endian)."""
    C = np.eye(4, dtype=complex)
    C[2:, 2:] = U
    return C


def rot(axis, theta, q):
    """One exact rotation gate, or nothing if the angle is zero.

    canonical() must not be used here: renaming Rz(pi/2) to S changes the gate
    by a phase, and a phase on one branch of a controlled block is not global.
    """
    if abs(theta) < 1e-12:
        return []
    return [({"diag": "rz", "x": "rx", "y": "ry"}[axis], theta, q)]


def controlled_circuit(U, control, target):
    """Controlled-U in two CX gates, from the ZYZ angles of U.

    With U = exp(i d) Rz(a) Ry(b) Rz(c), set A = Rz(a) Ry(b/2),
    B = Ry(-b/2) Rz(-(a+c)/2), C = Rz((c-a)/2); then A B C = I and
    A X B X C = U up to exp(i d), which Rz(d) on the control line restores.
    """
    d, a, b, c = zyz_angles(U)
    out = rot("diag", (c - a) / 2, target)                  # C
    out.append(("cx", control, target))
    out += rot("diag", -(a + c) / 2, target)                # B, rightmost factor first
    out += rot("y", -b / 2, target)
    out.append(("cx", control, target))
    out += rot("y", b / 2, target)                          # A, rightmost factor first
    out += rot("diag", a, target)
    out += rot("diag", d, control)                          # the leftover phase
    return out


rng = np.random.default_rng(23)
worst, counts = 0.0, {}
for _ in range(400):
    U = haar_1q(rng)
    circ = controlled_circuit(U, 0, 1)
    worst = max(worst, phase_free_error(controlled(U), unitary_of(circ, 2)))
    k = gate_counts(circ)["2q"]
    counts[k] = counts.get(k, 0) + 1
print("Controlled-U in two CX gates, 400 Haar-random U in U(2)")
print(f"  worst phase-free error : {worst:.2e}")
print(f"  CX counts used         : {counts}")


# ---- the standard explicit constructions, every one verified -------------
def rzz_circuit(theta, a, b):
    """exp(-i theta/2 Z Z) in two CX gates."""
    return [("cx", a, b), ("rz", theta, b), ("cx", a, b)]


def exp_pauli(theta, P):
    """exp(-i theta/2 P) for a Pauli word P with P P = I."""
    return np.cos(theta / 2) * np.eye(4) - 1j * np.sin(theta / 2) * P


ZZ = np.kron(Z, Z)
SWAP4 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                 dtype=complex)
ISWAP4 = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
                  dtype=complex)

CONSTRUCTIONS = [
    ("CZ", CZ4, [("h", 1), ("cx", 0, 1), ("h", 1)]),
    ("controlled-T", controlled(T), controlled_circuit(T, 0, 1)),
    ("exp(-i 0.3 Z Z)", exp_pauli(0.6, ZZ), rzz_circuit(0.6, 0, 1)),
    ("SWAP", SWAP4, [("cx", 0, 1), ("cx", 1, 0), ("cx", 0, 1)]),
]
head = f"{'target':<22}{'CX':>4}{'gates':>7}{'phase-free error':>19}"
print(f"\nExplicit constructions in the CX basis")
print(head)
print("-" * len(head))
for label, target, circ in CONSTRUCTIONS:
    err = phase_free_error(target, unitary_of(circ, 2))
    assert err < 1e-10, label
    print(f"{label:<22}{gate_counts(circ)['2q']:>4}{len(circ):>7}{err:>19.2e}")


# ---- the local invariants, and the CX count they imply -------------------
MAGIC = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
                 dtype=complex) / np.sqrt(2)


def magic_m(U):
    """m = (M^dagger U M)^T (M^dagger U M) for U rescaled into SU(4).

    The spectrum of m is invariant under single-qubit gates on either side, so
    it labels the local equivalence class of U.
    """
    U4 = U / np.linalg.det(U) ** 0.25
    Ut = MAGIC.conj().T @ U4 @ MAGIC
    return Ut.T @ Ut


def cx_count(U, tol=1e-9):
    """The minimum number of CX gates needed for U, from the invariants of m."""
    m = magic_m(U)
    tr, tr2 = np.trace(m), np.trace(m @ m)
    if abs(abs(tr) - 4) < tol and abs(tr2 - 4) < tol:
        return 0                                   # U is a product of local gates
    if abs(tr) < tol and abs(tr2 + 4) < tol:
        return 1                                   # the CX class
    return 2 if abs(tr.imag) < tol else 3          # tr m real: two CX suffice


print(f"\nMagic basis unitarity check: "
      f"{np.max(np.abs(MAGIC.conj().T @ MAGIC - np.eye(4))):.1e}")
sqrt_swap = np.array([[1, 0, 0, 0], [0, (1 + 1j) / 2, (1 - 1j) / 2, 0],
                      [0, (1 - 1j) / 2, (1 + 1j) / 2, 0], [0, 0, 0, 1]],
                     dtype=complex)
GATES = [("identity", np.eye(4, dtype=complex)), ("CX", CNOT4),
         ("controlled-H", controlled(H)), ("controlled-T", controlled(T)),
         ("iSWAP", ISWAP4), ("exp(-i 0.3 Z Z)", exp_pauli(0.6, ZZ)),
         ("SWAP", SWAP4), ("sqrt(SWAP)", sqrt_swap)]
head = f"{'gate':<22}{'Re tr m':>10}{'Im tr m':>10}{'tr m^2':>18}{'CX':>5}"
print(f"\nLocal invariants of the standard two-qubit gates")
print(head)
print("-" * len(head))
for label, U in GATES:
    m = magic_m(U)
    tr, tr2 = np.trace(m), np.trace(m @ m)
    print(f"{label:<22}{tr.real:>10.4f}{tr.imag:>10.4f}"
          f"{f'{tr2.real:+.4f}{tr2.imag:+.4f}j':>18}{cx_count(U):>5}")

# ---- the same statement as an experiment ---------------------------------
def random_2q_circuit(k, rng):
    """k CX gates with Haar-random single-qubit gates in between."""
    circ = []
    for layer in range(k + 1):
        for q in (0, 1):
            circ += zyz_circuit(haar_1q(rng), q)
        if layer < k:
            circ.append(("cx", 0, 1) if layer % 2 == 0 else ("cx", 1, 0))
    return circ


print("\n2000 random circuits at each CX count, classified by the invariants")
head = (f"{'CX in the circuit':>18}{'max |Im tr m|':>16}{'median':>12}"
        f"{'counts returned':>20}")
print(head)
print("-" * len(head))
for k in range(4):
    rng = np.random.default_rng(500 + k)
    ims, verdicts = [], {}
    for _ in range(2000):
        U = unitary_of(random_2q_circuit(k, rng), 2)
        ims.append(abs(np.trace(magic_m(U)).imag))
        v = cx_count(U)
        verdicts[v] = verdicts.get(v, 0) + 1
    print(f"{k:>18}{max(ims):>16.2e}{np.median(ims):>12.2e}"
          f"{str(verdicts):>20}")
```

```text
Controlled-U in two CX gates, 400 Haar-random U in U(2)
  worst phase-free error : 7.11e-16
  CX counts used         : {2: 400}

Explicit constructions in the CX basis
target                  CX  gates   phase-free error
----------------------------------------------------
CZ                       1      3           2.22e-16
controlled-T             2      6           2.28e-16
exp(-i 0.3 Z Z)          2      3           0.00e+00
SWAP                     3      3           0.00e+00

Magic basis unitarity check: 2.2e-16

Local invariants of the standard two-qubit gates
gate                     Re tr m   Im tr m            tr m^2   CX
-----------------------------------------------------------------
identity                  4.0000    0.0000   +4.0000+0.0000j    0
CX                        0.0000    0.0000   -4.0000-0.0000j    1
controlled-H              0.0000    0.0000   -4.0000-0.0000j    1
controlled-T              3.6955    0.0000   +2.8284+0.0000j    2
iSWAP                     0.0000    0.0000   +4.0000+0.0000j    2
exp(-i 0.3 Z Z)           3.3013    0.0000   +1.4494+0.0000j    2
SWAP                      0.0000   -4.0000   -4.0000-0.0000j    3
sqrt(SWAP)                1.4142   -1.4142   -0.0000-4.0000j    3

2000 random circuits at each CX count, classified by the invariants
 CX in the circuit   max |Im tr m|      median     counts returned
------------------------------------------------------------------
                 0        6.50e-16    6.52e-17           {0: 2000}
                 1        1.50e-15    2.22e-16           {1: 2000}
                 2        1.55e-15    2.22e-16           {2: 2000}
                 3        3.76e+00    4.15e-01           {3: 2000}
```

**What to look for.** The controlled-$U$ construction is exact to $7\times10^{-16}$ over 400 Haar-random targets and uses two CX gates every time. The construction table then shows the whole hierarchy in one place, every row verified against the dense matrix of its target.

The invariant table contains three surprises worth naming. Controlled-$H$ needs only **one** CX, not two: $H^2 = I$, so a single basis change on the target turns CX into it, and the invariants put it in the CX class. Controlled-$T$ is generic among controlled gates and needs two. And iSWAP has $\operatorname{tr} m = 0$ exactly like CX, but $\operatorname{tr} m^2 = +4$ rather than $-4$, which is what separates the two-CX class from the one-CX class.

The last table is the lower bound as an experiment. Two thousand random circuits at each CX count, with Haar-random single-qubit gates everywhere, and the classification is right 2000 times out of 2000 at every count. At two CX gates $\lvert\operatorname{Im}\operatorname{tr} m\rvert$ never exceeds $1.6\times10^{-15}$ no matter what the local gates are; at three it has median $0.42$. So no two-CX circuit can be a SWAP — and the three-CX construction for it is optimal rather than merely convenient.

* * *

## 2.4 Clifford$+T$ and the $T$ Count

### The gate set changes at the fault-tolerant boundary

Everything above treats the gate set of Chapter 1 as given, which on near-term hardware is right. Inside an error-correcting code the situation inverts: a logical gate has to be implementable on encoded qubits without spreading errors, and for the standard codes only a discrete set is — the **Clifford group**, generated by $H$, $S$ and CX. Clifford gates are cheap because they can be done transversally, by lattice surgery, or in some cases by relabelling the code's frame at no cost at all.

The Clifford group is also not universal, and its failure is spectacular. By the Gottesman-Knill theorem a circuit of Clifford gates on $n$ qubits with computational-basis input and measurement can be simulated classically in time polynomial in $n$: whatever a Clifford circuit does, a laptop can do. Universality requires one more gate, and the standard choice is

$$ T = \begin{pmatrix} 1 & 0 \cr 0 & e^{i\pi/4} \end{pmatrix} $$

which cannot be performed transversally in the codes that make the Cliffords cheap. It is instead produced by **magic-state distillation**: prepare many noisy copies of a particular state, consume them in a Clifford circuit that outputs fewer, cleaner copies, and repeat. The output is one $T$ gate per distilled state, and the space-time cost of the distillation dwarfs that of a Clifford gate by orders of magnitude. Chapter 5 makes that arithmetic explicit; here the consequence is enough:

> Inside a fault-tolerant computation, the cost of a circuit is its $T$ count. Clifford gates are approximately free, and the total gate count is approximately irrelevant.

### Why one rotation is expensive

The gate set $\lbrace H, S, T\rbrace$ is discrete, and $U(2)$ is not. So an arbitrary rotation cannot be represented at all — it can only be approximated, and the cost of the approximation is the cost of the rotation. The **Solovay-Kitaev theorem**, covered in [Introduction to Quantum Computing](<../quantum-computing-introduction/index.html>), says the approximation is always possible in $\mathrm{polylog}(1/\varepsilon)$ gates from any dense generating set, and gives a constructive algorithm; it is not re-implemented here.

What can be derived here is the floor, and it needs only counting. Let $N(t)$ be the number of distinct single-qubit Clifford$+T$ operators with $T$ count at most $t$. Covering the three-dimensional manifold $U(2)/\text{phase}$ to accuracy $\varepsilon$ needs of order $\varepsilon^{-3}$ operators, so any gate set can reach accuracy $\varepsilon$ at $T$ count $t$ only if

$$ N(t) \gtrsim \varepsilon^{-3} $$

Code Example 7 measures $N(t)$ and finds $N(t+1)/N(t) \to 1.98$, i.e. $N(t) \sim 2^t$. Substituting,

$$ 2^{t} \gtrsim \varepsilon^{-3} \qquad \Longrightarrow \qquad t \gtrsim 3\log_2(1/\varepsilon) $$

Three $T$ gates per bit of accuracy, and no synthesis algorithm can do better. The measured cost of an exhaustive search over short words is $3.30$ per bit, so the bound is nearly saturated already at $t \le 12$.

### Code Example 7: Clifford$+T$, and the Price of One Rotation

The Toffoli comes first, because it is the standard unit of account: three qubits, six CX gates, and seven $T$-type rotations, which is the known optimal $T$ count for it. The IR has no $T^\dagger$; $R_z(-\pi/4)$ is $T^\dagger$ times $e^{i\pi/8}$, and since these are uncontrolled single-qubit gates the leftover scalar is a global phase that the checker ignores.

```python
"""Chapter 2, Example 7: Clifford+T, and the price of one arbitrary rotation.
Continues from Example 6 (same session)."""
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from qcheck import *


def tdg(q):
    """T-dagger. The IR has no such gate; Rz(-pi/4) is T-dagger times exp(i pi/8),
    and an uncontrolled scalar is a global phase, which the checker ignores."""
    return ("rz", -np.pi / 4, q)


def toffoli(a, b, c):
    """CCX with controls a, b and target c: 6 CX gates and 7 T-type rotations."""
    return [("h", c),
            ("cx", b, c), tdg(c), ("cx", a, c), ("t", c),
            ("cx", b, c), tdg(c), ("cx", a, c), ("t", b), ("t", c),
            ("h", c),
            ("cx", a, b), tdg(b), ("cx", a, b), ("t", a)]


def t_count(circ):
    """Gates that consume a magic state: T, and Rz by an odd multiple of pi/4."""
    return sum(1 for g in circ if g[0] == "t"
               or (g[0] == "rz" and abs(abs(wrap(g[1])) - np.pi / 4) < 1e-12))


def permutation_matrix(n, action):
    """The 2^n x 2^n matrix of a classical reversible map on bit lists."""
    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=complex)
    for j in range(dim):
        bits = [(j >> (n - 1 - q)) & 1 for q in range(n)]
        out = action(list(bits))
        U[sum(b << (n - 1 - q) for q, b in enumerate(out)), j] = 1.0
    return U


ccx = toffoli(0, 1, 2)
ccz = ccx[1:10] + ccx[11:]                    # drop both H gates on qubit 2
fredkin = [("cx", 2, 1)] + ccx + [("cx", 2, 1)]
CCX8 = permutation_matrix(3, lambda b: [b[0], b[1], b[2] ^ (b[0] & b[1])])
CSWAP8 = permutation_matrix(
    3, lambda b: [b[0], b[2], b[1]] if b[0] else b)
CCZ8 = np.diag([1.0] * 7 + [-1.0]).astype(complex)

print("Three three-qubit gates in Clifford+T, each verified against its matrix")
head = (f"{'gate':<10}{'gates':>7}{'CX':>5}{'T':>4}{'depth':>7}"
        f"{'phase-free error':>19}")
print(head)
print("-" * len(head))
for label, circ, target in [("Toffoli", ccx, CCX8), ("CCZ", ccz, CCZ8),
                            ("Fredkin", fredkin, CSWAP8)]:
    err = phase_free_error(target, unitary_of(circ, 3))
    assert err < 1e-10, label
    print(f"{label:<10}{len(circ):>7}{gate_counts(circ)['2q']:>5}"
          f"{t_count(circ):>4}{circuit_depth(circ, 3):>7}{err:>19.2e}")

opt = peephole(ccx)
print("\nThe peephole optimizer of Example 3 on the Toffoli:")
print(f"  before: {len(ccx)} gates, {t_count(ccx)} T, "
      f"{gate_counts(ccx)['2q']} CX")
print(f"  after : {len(opt)} gates, {t_count(opt)} T, "
      f"{gate_counts(opt)['2q']} CX, error "
      f"{assert_equivalent(ccx, opt, 3, 'toffoli'):.1e}")
# ---- how many operators are there, and how accurate can they be ----------
def phase_key(U, digits=6):
    """A hashable label for a 2x2 unitary, insensitive to the global phase."""
    flat = U.ravel()
    i = int(np.argmax(np.abs(flat)))
    return tuple(np.round(flat * np.conj(flat[i]) / abs(flat[i]), digits))


def clifford_t_ball(t_max):
    """Every single-qubit Clifford+T operator with T count at most t_max, by a
    0-1 breadth-first search over H, S (free) and T (cost one)."""
    start = np.eye(2, dtype=complex)
    cost, reps = {phase_key(start): 0}, {phase_key(start): start}
    queue = deque([(0, start)])
    while queue:
        t, U = queue.popleft()
        for G, dt in ((H, 0), (S, 0), (T, 1)):
            if t + dt > t_max:
                continue
            V, k = G @ U, phase_key(G @ U)
            if k not in cost or t + dt < cost[k]:
                cost[k], reps[k] = t + dt, V
                (queue.appendleft if dt == 0 else queue.append)((t + dt, V))
    return cost, reps


T_MAX = 12
cost, reps = clifford_t_ball(T_MAX)
keys = list(reps.keys())
ops = np.array([reps[k] for k in keys])
tcost = np.array([cost[k] for k in keys])
sizes = [int(np.sum(tcost <= t)) for t in range(T_MAX + 1)]

rng = np.random.default_rng(3)
targets = [haar_1q(rng) for _ in range(200)]
best = np.zeros((len(targets), T_MAX + 1))
for i, U in enumerate(targets):
    tr = np.einsum("nij,ij->n", ops.conj(), U)      # tr(V^dagger U) for every V
    ph = np.where(np.abs(tr) < 1e-12, 1.0, tr / np.abs(tr))
    err = np.abs(U[None] - ph[:, None, None] * ops).reshape(len(ops), -1).max(1)
    for t in range(T_MAX + 1):
        best[i, t] = err[tcost <= t].min()
median = np.median(best, axis=0)

print(f"\nThe Clifford+T ball around the identity, and how well it covers U(2)")
head = (f"{'T count':>8}{'operators':>11}{'growth':>8}"
        f"{'median error, 200 targets':>27}{'log2(1/err)':>13}")
print(head)
print("-" * len(head))
for t in range(T_MAX + 1):
    ratio = f"{sizes[t] / sizes[t - 1]:.3f}" if t else "-"
    print(f"{t:>8}{sizes[t]:>11}{ratio:>8}{median[t]:>27.4f}"
          f"{np.log2(1 / median[t]):>13.2f}")

bits = np.log2(1.0 / median)
slope = np.polyfit(bits[6:], np.arange(T_MAX + 1)[6:], 1)[0]
print(f"\nT gates per bit of accuracy, fitted over t >= 6: {slope:.2f}")
print(f"Operators per T gate, measured                 : "
      f"{sizes[T_MAX] / sizes[T_MAX - 1]:.2f}")
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.semilogy(range(T_MAX + 1), median, "o-", color="tab:blue",
            label="exhaustive search, median of 200 targets")
ax.semilogy(range(T_MAX + 1), median[0] * 2.0 ** (-np.arange(T_MAX + 1) / 3.0),
            "--", color="k", lw=1, label=r"$\varepsilon \propto 2^{-t/3}$")
ax.set_xlabel("T count"); ax.set_ylabel("best achievable error")
ax.set_title("The price of one arbitrary single-qubit rotation")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
```

```text
Three three-qubit gates in Clifford+T, each verified against its matrix
gate        gates   CX   T  depth   phase-free error
----------------------------------------------------
Toffoli        15    6   7     12           2.48e-16
CCZ            13    6   7     11           2.43e-16
Fredkin        17    8   7     13           2.48e-16

The peephole optimizer of Example 3 on the Toffoli:
  before: 15 gates, 7 T, 6 CX
  after : 15 gates, 7 T, 6 CX, error 0.0e+00

The Clifford+T ball around the identity, and how well it covers U(2)
 T count  operators  growth  median error, 200 targets  log2(1/err)
-------------------------------------------------------------------
       0         34       -                     0.3188         1.65
       1        154   4.529                     0.2013         2.31
       2        420   2.727                     0.1474         2.76
       3        977   2.326                     0.1096         3.19
       4       2056   2.104                     0.0874         3.52
       5       4081   1.985                     0.0733         3.77
       6       7987   1.957                     0.0563         4.15
       7      15632   1.957                     0.0441         4.50
       8      30382   1.944                     0.0380         4.72
       9      59476   1.958                     0.0322         4.96
      10     116887   1.965                     0.0262         5.25
      11     231311   1.979                     0.0207         5.59
      12     458360   1.982                     0.0152         6.04

T gates per bit of accuracy, fitted over t >= 6: 3.30
Operators per T gate, measured                 : 1.98
```

**What to look for.** All three three-qubit gates verify against their exact matrices and all three have $T$ count 7: CCZ is the Toffoli with the two outer Hadamards deleted, and the Fredkin is the Toffoli conjugated by a CX, so neither costs a magic state more. The peephole optimizer of Code Example 3 finds nothing at all in the Toffoli — 15 gates in, 15 out. That is the correct answer, and it is worth stating plainly: reducing a $T$ count is a different problem from reducing a gate count, it is not solved by local rules, and it is an active research subject.

The main table is the counting argument, measured. The ball of operators with $T$ count at most $t$ grows by $1.98$ per $T$ gate, and the median best approximation error over 200 Haar-random targets falls by $2^{1/3.30}$ per $T$ gate; those two numbers are the same statement, related by the covering bound above. Extrapolating the fitted $3.30$, one arbitrary rotation costs about 33 $T$ gates at $10^{-3}$, 66 at $10^{-6}$ and 110 at $10^{-10}$ — about ten Toffolis for six digits and fifteen for ten. Three consequences follow, all of which reappear in Chapter 5. A fault-tolerant compiler avoids emitting arbitrary rotations wherever it can, preferring Cliffords and Toffolis. The approximate QFT of the sister course, which discards the smallest controlled rotations, is a necessity rather than an optimization — those rotations are the expensive ones and most of them are below the accuracy the algorithm needs anyway. And a resource estimate quoted in "gates" is uninterpretable, because the only meaningful currency is the $T$ count.

* * *

## Exercises

#### Exercise 1: A Rule That Is Only Sometimes a Rule

The optimizer's `canonical` rewrites $R_z(\pi)$ to $Z$, and Code Example 1 confirmed the two are equal up to a phase.

  1. Compute $R_z(\pi)$ and $Z$ explicitly and give the phase relating them.
  2. Give a circuit in which replacing $R_z(\pi)$ by $Z$ changes the physical result, not merely the phase.
  3. State the rule that decides when a phase may be dropped.
  4. `controlled_circuit` in Code Example 6 emits rotations through `rot` rather than through `canonical`. Which of the three emitted blocks would break first if it used `canonical`, and why?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(R_z(\pi) = \mathrm{diag}(e^{-i\pi/2}, e^{i\pi/2}) = \mathrm{diag}(-i, i) = -i\,\mathrm{diag}(1,-1) = -i\,Z\). The phase is \(e^{-i\pi/2}\).</p>

<p><strong>2.</strong> Any circuit in which the fragment is placed under a control. Controlled-\(Z\) is \(\mathrm{diag}(1,1,1,-1)\), while controlled-\(R_z(\pi)\) is \(\mathrm{diag}(1,1,-i,i)\). These differ on the \(|1\rangle\) branch by the factor \(-i\) and on the \(|0\rangle\) branch not at all, so they are not equal up to any single phase: the naive error is \(\sqrt2\); the phase-free error is 0.765, not zero. Chapter 1's Code Example 4 prints exactly this comparison.</p>

<p><strong>3.</strong> A global phase may be discarded only when the fragment will never be conditioned on another qubit and never be used to build a controlled version of itself. Equivalently: phases may be dropped at the top level of a circuit, never inside a block that a later pass will control.</p>

<p><strong>4.</strong> The \(C\) block, which is a single \(R_z((c-a)/2)\), and the \(B\) block: the construction relies on \(ABC = I\) <em>exactly</em>, and each renaming of a rotation to \(Z\), \(S\) or \(T\) multiplies its block by a phase. Then \(ABC = e^{i\varphi}I\), the \(|0\rangle\) branch acquires \(e^{i\varphi}\) and the \(|1\rangle\) branch does not, and the result is not controlled-\(U\). Running Code Example 6 with <code>canonical</code> in place of <code>rot</code> gives a worst error of about \(7\times10^{-1}\) instead of \(7\times10^{-16}\).</p>

</details>

#### Exercise 2: When Two CX Gates Commute

Code Example 2's predicate says $\mathrm{CX}_{a,b}$ and $\mathrm{CX}_{c,d}$ commute if and only if $a = c$ or $b = d$.

  1. Prove the "if" direction for the shared-control case using $\mathrm{CX}_{a,b} = |0\rangle\langle 0|_a \otimes I + |1\rangle\langle 1|_a \otimes X_b$.
  2. Prove it for the shared-target case.
  3. Show that $\mathrm{CX}_{0,1}$ and $\mathrm{CX}_{1,2}$ do not commute, by evaluating both orders on $|110\rangle$.
  4. The library of Code Example 2 contains 6 ordered CX pairs on 3 qubits with overlapping support and distinct control and target. How many of them commute, and does the predicate get them all?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> With a shared control \(a\), both gates are block-diagonal in the \(a\) basis: the \(|0\rangle_a\) block is the identity in both, and the \(|1\rangle_a\) blocks are \(X_b\) and \(X_d\) with \(b \ne d\), which act on different qubits and commute. So the products agree block by block.</p>

<p><strong>2.</strong> With a shared target \(b\), the two gates are \(X_b^{n_a}\) and \(X_b^{n_c}\) where \(n_a, n_c\) are the number operators of the two controls. Those controls are distinct from each other and from \(b\), so \(n_a\) and \(n_c\) commute with everything in sight, and the two gates are both powers of the same \(X_b\).</p>

<p><strong>3.</strong> \(\mathrm{CX}_{1,2}\mathrm{CX}_{0,1}|110\rangle\): the first gate flips qubit 1 (control 0 is set), giving \(|100\rangle\); the second leaves it alone (control 1 is now clear), so the result is \(|100\rangle\). The other order: \(\mathrm{CX}_{1,2}\) flips qubit 2 (control 1 is set), giving \(|111\rangle\); then \(\mathrm{CX}_{0,1}\) flips qubit 1, giving \(|101\rangle\). The two differ.</p>

<p><strong>4.</strong> Six: the three shared-control pairs \((0{,}1)(0{,}2)\), \((1{,}0)(1{,}2)\), \((2{,}0)(2{,}1)\) and their reverses, and by the same count six shared-target ones. All are caught by the predicate's single line <code>g[1] == h[1] or g[2] == h[2]</code>: the exhaustive check reports no missed CX pair, since all 38 misses involve the two disguised identities.</p>

</details>

#### Exercise 3: ZYZ by Hand

  1. Apply the formulas of Section 2.2 to $H$ and check against the printed row of Code Example 5.
  2. Apply them to $T$ and explain which degenerate branch is taken.
  3. A colleague's implementation returns three angles and no phase, and reports that it verifies correctly on 10000 random inputs. What is their test, and what is it missing?
  4. Why does the synthesis of a generic $U$ always produce exactly three gates rather than sometimes two?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\det H = -1\), so \(\delta = \frac{1}{2}\arg(-1) = \pi/2\) and \(V = e^{-i\pi/2}H = -iH\). Then \(|V_{00}| = |V_{10}| = 1/\sqrt2\), so \(b = 2\arctan 1 = \pi/2\). \(V_{11} = -i/\sqrt2 \cdot(-1) = i/\sqrt2\) has argument \(\pi/2\) and \(V_{10} = -i/\sqrt2\) has argument \(-\pi/2\), so \(a = 0\) and \(c = \pi\). That is the printed row: \(\delta/\pi = +0.5\), \((a,b,c)/\pi = (0, 0.5, 1)\), and the emitted circuit is \(Z\) then \(R_y(\pi/2)\) — two gates, because \(a = 0\) contributes nothing.</p>

<p><strong>2.</strong> \(\det T = e^{i\pi/4}\), so \(\delta = \pi/8\) and \(V = \mathrm{diag}(e^{-i\pi/8}, e^{i\pi/8}) = R_z(\pi/4)\). Now \(V_{10} = 0\), which is the \(b = 0\) branch: only \(a + c\) is determined, the code sets \(c = 0\), and \(a = 2\arg V_{11} = \pi/4\). One gate.</p>

<p><strong>3.</strong> Their test is the phase-free comparison, which is exactly the test that cannot see the missing number. It will pass on every input. What it misses is every use of the routine inside a controlled construction — Section 2.3's controlled-\(U\), Chapter 3's routed controlled gates, and any coherent comparison of two circuits, as in Chapter 5's error mitigation. The bug appears the first time the output is controlled and not before.</p>

<p><strong>4.</strong> A generic element of \(SU(2)\) has three non-trivial Euler angles, and each emitted gate carries one parameter. Two gates could cover only a two-parameter subset, which has measure zero. The printed histogram over 2000 Haar samples is \(\lbrace 3: 2000\rbrace\) for that reason; the shorter outputs occur only on the measure-zero degenerate set.</p>

</details>

#### Exercise 4: Reading a CX Count Off the Invariants

Consider the controlled-$S$ gate, $\mathrm{diag}(1,1,1,i)$, and $\sqrt{\mathrm{CZ}} = \mathrm{diag}(1,1,1,i)$ — the same matrix under two names.

  1. Using `magic_m` and `cx_count` from Code Example 6, what CX count does it get? Predict before running.
  2. The canonical coordinates of a controlled phase $\mathrm{diag}(1,1,1,e^{i\varphi})$ are $(\varphi/4, 0, 0)$. For which $\varphi$ does the gate need one CX rather than two?
  3. iSWAP and $\mathrm{CX}$ both have $\operatorname{tr} m = 0$. Which invariant separates them, and what are its two values?
  4. A colleague proposes the criterion "$\operatorname{Im} G_1 = 0$ implies at most two CX", where $G_1 = (\operatorname{tr} m)^2/(16\det U)$. Test it on SWAP and explain the failure.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Two. It is a controlled phase with \(\varphi = \pi/2\), so its canonical coordinates are \((\pi/8, 0, 0)\), which is neither the origin nor the CX corner; \(\operatorname{tr} m\) is real, so the classification returns 2. Numerically \(\operatorname{tr} m = 3.4142\), \(\operatorname{tr} m^2 = 2.0000\).</p>

<p><strong>2.</strong> Only \(\varphi = \pi\), which puts the coordinates at \((\pi/4,0,0)\) — the CZ corner. Every other controlled phase, however small \(\varphi\) is, costs two CX gates. This is why a compiler that discards small controlled rotations saves two CX gates each and not a fraction of one, and it is the routing-level counterpart of the approximate QFT argument in §2.4.</p>

<p><strong>3.</strong> \(\operatorname{tr} m^2\): it is \(-4\) for CX and \(+4\) for iSWAP. In canonical coordinates the two are \((\pi/4,0,0)\) and \((\pi/4,\pi/4,0)\), both with \(t_z = 0\), so both are reachable with two CX; only the first is reachable with one.</p>

<p><strong>4.</strong> SWAP has \(\operatorname{tr} m = -4i\), so \((\operatorname{tr} m)^2 = -16\) is real and \(\operatorname{Im} G_1 = 0\), and the criterion wrongly reports that two CX gates suffice. Squaring the trace destroys the distinction between "real" and "purely imaginary", which is precisely the distinction the two-CX condition rests on. The criterion must be stated on \(\operatorname{tr} m\), not on its square.</p>

</details>

#### Exercise 5: A $T$-Count Budget

An algorithm needs $10^4$ Toffoli gates and $3\times10^3$ arbitrary single-qubit rotations, each to accuracy $10^{-8}$.

  1. Using the fitted $3.30$ $T$ gates per bit, estimate the $T$ count of the rotations and of the Toffolis, and the total.
  2. Which of the two dominates, and by how much?
  3. The compiler is improved so that half the rotations can be replaced by exact Clifford gates. What is the new total, and what fractional saving is that?
  4. A second proposal halves the number of Toffolis at the cost of doubling the number of rotations. Is it an improvement?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\log_2(10^{8}) = 26.6\) bits, so \(3.30 \times 26.6 \approx 88\) \(T\) gates per rotation, and \(3\times10^3 \times 88 \approx 2.6\times10^{5}\). The Toffolis cost \(7 \times 10^4 = 7\times10^{4}\). Total \(\approx 3.3\times10^{5}\).</p>

<p><strong>2.</strong> The rotations, by a factor of about 3.8, despite being three times fewer in number. One arbitrary rotation is worth about twelve Toffolis at this accuracy.</p>

<p><strong>3.</strong> \(1.5\times10^3\) rotations at 88 each is \(1.3\times10^{5}\), plus \(7\times10^{4}\), giving \(2.0\times10^{5}\) — a saving of \(40\%\) of the total \(T\) count from removing half of one gate type. That asymmetry is why fault-tolerant compilation is largely the study of how to avoid rotations.</p>

<p><strong>4.</strong> No. It saves \(3.5\times10^{4}\) \(T\) gates on the Toffolis and adds \(2.6\times10^{5}\) on the rotations, for a net increase of a factor of about 1.7 in the total. A trade that looks like "half the gates" in the gate-count currency is a large loss in the only currency that matters. Always convert to \(T\) count before comparing two circuits meant for a fault-tolerant machine.</p>

</details>

* * *

## Summary

### Key Takeaways

**1\. A rewrite rule is a pair of fragments with the same unitary, and "same" has to be tested**

  * Three families do all the work: fusion, cancellation, and commutation — of which only the third shortens nothing by itself, and it is the one that makes the other two applicable.
  * Adjacency on a circuit means the next gate touching the same qubits, not the next list entry; that distinction is why Chapter 1's optimizer found nothing after routing, and every rewrite here is guarded by the phase-free matrix comparison, worst error $10^{-15}$.

**2\. A commutation predicate should be sound, not complete**

  * Unsoundness silently changes the meaning of circuits; incompleteness only loses an optimization, so the predicate is written conservatively and its incompleteness is measured.
  * On 1225 ordered pairs from a 35-gate library: zero unsound answers and 38 missed commutations, every miss involving $R_y(0)$ or $R_z(2\pi)$ — the identity in a form a syntactic rule cannot see. The fix is canonicalization, not a longer rule list.

**3\. The yield of peephole optimization is a property of the input**

  * On 200 uniform random circuits: $26.3\%$ of the gates, $18.8\%$ of the depth, and only $7.1\%$ of the two-qubit gates. On a circuit followed by its own inverse: everything, every time.
  * Termination is free — every successful rule strictly shortens the circuit — and correctness is not, which is why the check comes first.

**4\. Synthesis is complete where rules are not**

  * ZYZ: any $U \in U(2)$ is $e^{i\delta}R_z(a)R_y(b)R_z(c)$, exact to $10^{-15}$ on 2000 Haar samples, with two degenerate branches that must be handled explicitly, and the phase $\delta$ returned because it stops being global the moment the fragment is controlled.
  * Resynthesizing runs of single-qubit gates removes $H X H$, which no rule in the chapter's list could touch; combined with the peephole pass it reaches 36.40 gates against 60.00 as written, over 600 verified rewrites across three optimizing pipelines plus 200 runs of a no-op control.

**5\. Two qubits: three numbers, and a hard CX count**

  * KAK: $U = (A_1\otimes A_2)\exp\big[ i(t_xXX + t_yYY + t_zZZ) \big]\,(B_1\otimes B_2)$, so the cost of a gate is a point in a tetrahedron. CZ costs 1 CX, $\exp(-i\frac{\theta}{2}ZZ)$ and generic controlled-$U$ cost 2, SWAP costs 3.
  * The count is read off $\operatorname{tr} m$ and $\operatorname{tr} m^2$ in the magic basis, not guessed, and verified as an experiment: 2000 random circuits at each CX count classified correctly 2000 times, with $\lvert\operatorname{Im}\operatorname{tr} m\rvert < 1.6\times10^{-15}$ at two CX and median $0.42$ at three.

**6\. In a fault-tolerant computation the only currency is the $T$ count**

  * Clifford gates are cheap and not universal — Gottesman-Knill says a laptop can simulate them; $T$ is universal and comes from magic-state distillation.
  * The number of Clifford$+T$ operators with $T$ count $\le t$ grows by $1.98$ per $T$ gate, and covering $U(2)/$phase to $\varepsilon$ needs $\varepsilon^{-3}$ of them, giving $t \gtrsim 3\log_2(1/\varepsilon)$. Measured by exhaustive search: $3.30$ per bit, so one arbitrary rotation costs 66 $T$ gates at $10^{-6}$ against 7 for a whole Toffoli.

**Practical implications**

  * Write the equivalence check before the pass, and include a test with a known answer — a circuit followed by its own inverse must optimize to nothing.
  * Report an optimizer's yield separately for total gates, depth, and two-qubit gates, and say what generated the inputs; the three numbers move by different amounts and only the last predicts error.
  * Never drop a global phase inside a fragment that a later pass may control, and never quote a gate count for a fault-tolerant circuit — convert to $T$ count first.

### Where This Leads

Every circuit in this chapter was compiled as though any qubit could interact with any other, which is true of exactly one family of hardware. Chapter 3 removes that assumption. It builds coupling graphs for the three topologies that Chapter 1's layer map and [Introduction to Quantum Hardware](<../quantum-hardware-introduction/index.html>) motivate — all-to-all, a square grid, and heavy-hex — and then confronts the two problems connectivity creates: choosing which physical qubit holds which logical one, and inserting SWAP networks when the choice turns out to be wrong. The measurements there dwarf the ones here. This chapter's optimizer removed a quarter of the gates in a circuit; routing the same circuit onto a sparse graph multiplies them by between two and five, and the equivalence check has to be rebuilt, because a routed circuit is deliberately *not* the unitary that was written down.

[← Chapter 1: The Stack from Algorithms to Pulses](<chapter-1.html>) [Chapter 3: Transpilation — Mapping to Connectivity →](<chapter-3.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The gate counts, $T$ counts and reduction percentages in this chapter are measurements of the specific circuits and seeds listed in the code, not benchmarks of any compiler or hardware, and the extrapolated rotation costs of §2.4 are order-of-magnitude teaching estimates derived from the stated fit. Verify against primary sources before using them in a proposal or a paper.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
