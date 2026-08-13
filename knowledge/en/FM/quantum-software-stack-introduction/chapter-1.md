---
title: "Chapter 1: The Stack from Algorithms to Pulses"
chapter_title: "Chapter 1: The Stack from Algorithms to Pulses"
subtitle: Why There Are Layers, What a Circuit IR Is, and How to Prove a Rewrite Did Not Change the Meaning
reading_time: 45-50 minutes
difficulty: Intermediate
code_examples: 7
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-software-stack-introduction/chapter-1.html>) | Last sync: 2026-08-13

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to the Quantum Software Stack](<index.html>) > Chapter 1

An algorithm is a mathematical object. A machine accepts analog waveforms. Nothing in between is obvious, and everything in between is software. This chapter is a map of that software: seven layers, what each one is for, which of them can be written without knowing anything about the hardware, and — the part that makes the whole subject tractable — how you check that a layer did its job.

That last point is what this course is built on, so it is worth stating immediately. A compiler pass here is a function from a circuit to a circuit, and it is correct when the two circuits mean the same thing. For quantum circuits, "mean the same thing" has an exact definition: the same unitary, up to a global phase. That definition is *checkable*. You can build the matrix of a small circuit, build the matrix of the rewritten circuit, remove the global phase, and subtract. Every rewriting rule in Chapters 2 and 3, every pulse in Chapter 4, and every mitigation formula in Chapter 5 is accompanied by a check of this kind. It is the reason this course can be written honestly without access to a machine, and it is the single most useful habit to take away from it.

The chapter builds three things. Section 1.2 fixes the **circuit intermediate representation** — the data structure that every later chapter manipulates — and argues that the choice of gate set is what makes a layer boundary a layer boundary. Section 1.3 defines compilation as meaning-preserving rewriting and builds the equivalence checker. Section 1.4 is the map of what commercial SDKs call these layers, in general terms only: the words differ between frameworks and between versions, and the layers do not.

## Learning Objectives

After completing this chapter, you will be able to:

  * Name the seven layers between an algorithm and a control waveform, state what each layer consumes and produces, and say for each one exactly which piece of hardware information it needs
  * Explain why an intermediate representation exists at all — the $N \times M$ argument — and why the choice of gate set, rather than anything about data structures, is what fixes a layer boundary
  * Implement the circuit IR of this course — gate tuples, `run_circuit`, `circuit_depth`, `gate_counts` — with the big-endian convention of the introductory course
  * State the three correctness relations a compiler pass can be asked to satisfy — exact up to a global phase, exact up to a qubit permutation, approximate within $\varepsilon$ — and say which layer uses which
  * Build a unitary-equivalence checker with global-phase removal, and explain why the Hilbert-Schmidt overlap gives the optimal phase
  * Explain why a global phase may never be discarded inside a controlled block, and demonstrate the resulting bug numerically
  * Distinguish the cost metrics — total gate count, two-qubit gate count, depth, wall-clock time, error budget — and show that they disagree, so that "optimize" is meaningless without a named objective
  * Read the table of contents of any quantum SDK's documentation and say which layer each part of it is about

### Conventions Carried Over

Three conventions come from [Introduction to Quantum Computing](<../quantum-computing-introduction/index.html>) unchanged.

**Big-endian qubit ordering.** Qubit 0 is the leftmost symbol in a ket and the most significant bit of the amplitude index, so on three qubits $\lvert 000 \rangle, \lvert 001 \rangle, \ldots, \lvert 111 \rangle$ occupy indices $0$ through $7$. Much of the SDK literature uses the opposite convention, and a mismatch is the most common source of silently wrong results in this subject. It is also, as it happens, a good illustration of the theme of this chapter: qubit ordering is a property of the *interface between layers*, not of the physics, and it costs real debugging time precisely because it is a convention rather than a fact.

**The simulator.** Every example runs on the ninety-nine-line state-vector simulator of that course's Chapter 2. Code Example 1 re-lists the functions this course needs, verbatim.

**Reduced units.** Where a Hamiltonian appears — which in this chapter is nowhere, and in Chapter 4 is everywhere — $\hbar = 1$.

One convention is new, and it is the technical contract of this course: the **circuit IR** of Section 1.2. Chapters 2 through 5 use it without variation, so that a pass written for one chapter can be applied to a circuit from another.

* * *

## 1.1 Why There Are Layers

### The distance to be covered

Take Grover's algorithm on ten qubits. As mathematics it is a product of two reflections, iterated $\lfloor \pi/(4\theta) \rfloor$ times. As a thing that happens in a laboratory it is some tens of thousands of shaped microwave or laser pulses, each with an amplitude, a duration, a frequency, a phase and an envelope, delivered on a schedule accurate to the nanosecond, followed by an analog measurement whose voltage has to be discriminated into a bit.

Nobody writes the second description by hand, and nobody would want to: it is specific to one machine on one day, since the pulse amplitude that produces a $\pi$ rotation drifts and is re-measured daily. Nobody writes the first description directly onto hardware either, because it says nothing about which qubits can interact. The way out is the same as in classical computing: interpose representations, each of which throws away detail that the layer above should not care about and adds detail that the layer below requires. The seven layers below are the ones this course uses; the names vary between frameworks and the division of labour does not.

| Layer | Consumes | Produces | What it must know about the machine |
| --- | --- | --- | --- |
| 1. Algorithm | a problem | a parameterized construction | nothing |
| 2. Circuit IR | a construction | a gate list on abstract qubits | nothing |
| 3. Optimization | a gate list | a shorter gate list | nothing, or only the cost weights |
| 4. Placement and routing | a gate list, a coupling graph | a gate list respecting connectivity | which pairs of qubits can interact |
| 5. Gate synthesis | a gate list, a native gate set | a gate list in the native set | which gates the machine implements |
| 6. Pulses and scheduling | native gates | waveforms and a timetable | the Hamiltonian and today's calibration |
| 7. Readout and mitigation | raw counts | corrected expectation values | the measurement error model |

The **consumes/produces** columns show that layers 2 to 5 all speak the same language: a gate list in, a gate list out. That is the whole point of an intermediate representation — every pass has the same type, so passes compose, can be reordered, and can be tested individually. Layer 6 is where the type changes, and that is why pulse-level programming feels so different. The **must know** column is the criterion for where a layer boundary belongs. A pass that needs no hardware information can be written once and used forever; a pass that needs the coupling graph must be re-run for each machine; a pass that needs today's calibration data must be re-run each morning. Sorting passes by how perishable their input is gives exactly the layering above, and it explains the otherwise curious fact that the *optimizer* is the most portable part of a quantum compiler while the *pulse compiler* is the least.

### The $N \times M$ argument

The economic argument for an IR is the one that produced classical compiler IRs. With $N$ algorithms and $M$ machines and no intermediate representation, you write $N \times M$ translations; with one, $N$ front ends and $M$ back ends. There is a second, less obvious benefit, and in quantum computing it matters more. An IR is a place to *measure*. Once a circuit is a data structure, you can count its gates, compute its depth, count its two-qubit gates, count its $T$ gates, and estimate its error — before running anything. Since running anything is expensive, and since the interesting question about a quantum algorithm is usually "how big does the machine have to be", the IR is where most of the actual work of the field happens. Chapter 5 ends the course with a resource estimate computed entirely at this level.

### What is different about quantum compilation

Three things, and they are worth naming because they explain why this course is not simply a translation of a classical compilers course.

**Correctness is decidable at small size, and expensive at large size.** Two classical programs are equivalent when they compute the same function, which is undecidable in general. Two quantum circuits are equivalent when they implement the same unitary up to a phase — a statement about two $2^n \times 2^n$ matrices, which is a finite computation. That is a gift, and Section 1.3 spends it. The catch is the $2^n$: exhaustive checking is a small-$n$ tool, and everything above about twenty qubits has to be verified structurally or statistically instead.

**Optimization is not optional.** A classical program that is twice as long runs twice as slowly. A quantum circuit that is twice as long may not run at all, because the error accumulates past the point where the output carries information. Code Example 6 makes this quantitative. Compilation is therefore not a convenience layer in quantum computing; it is part of what determines whether a computation is feasible, and a factor of two in two-qubit gate count is a factor of two in the size of machine you need.

**The bottom layer is a physical experiment.** Layer 6 does not translate a gate into a pulse by looking up a table; it uses parameters that were *measured* on that machine recently by running a calibration experiment, and the calibration experiment is itself software in the stack. Chapter 4 implements the loop. Software that measures its own machine and then reprograms itself is unusual, and it is one of the genuinely distinctive features of this field.

### Where the layers leak

The layering above is an engineering convention, not a theorem, and the interesting research sits at its seams. Three examples, all of which the later chapters return to.

  * **Hardware-aware optimization.** Layer 3 is supposed to know nothing about the machine, but an optimizer that knows the coupling graph can prefer rewrites that reduce routing cost later. Merging layers 3 and 4 gives better circuits and a much harder problem.
  * **Pulse-level compilation.** Two adjacent gates compiled separately produce two pulses; compiled together they may produce one shorter pulse. Collapsing layers 5 and 6 buys real fidelity and gives up portability entirely.
  * **Error correction.** A fault-tolerant stack inserts a whole additional layer between 3 and 4 in which one logical gate becomes thousands of physical operations, and in which the native gate set is Clifford$+T$ regardless of what the hardware natively does. Chapter 5 covers the resource arithmetic.

A reasonable reading of the field is that the layers exist because they let different people work on different problems, and that every few years someone gets a factor of two by ignoring one of them.

* * *

## 1.2 The Design of a Circuit IR

### A circuit is data, not code

The representation used in this course is deliberately minimal:

$$ \text{circuit} = \left[ g_1, g_2, \ldots, g_L \right], \qquad g_i = (\text{name}, \text{args} \ldots) $$

A circuit is a Python list of tuples. A gate is a tuple whose first element is a string naming the gate and whose remaining elements are its angle, if it has one, followed by the qubits it acts on. The full set is:

| Gate tuple | Meaning | Matrix |
| --- | --- | --- |
| `("h", q)` | Hadamard on qubit $q$ | $H$ |
| `("x", q)`, `("z", q)` | Pauli gates | $X$, $Z$ |
| `("s", q)`, `("t", q)` | phase gates | $\mathrm{diag}(1, i)$, $\mathrm{diag}(1, e^{i\pi/4})$ |
| `("rx", theta, q)` | rotation about $x$ | $R_x(\theta) = e^{-i\theta X/2}$ |
| `("ry", theta, q)` | rotation about $y$ | $R_y(\theta) = e^{-i\theta Y/2}$ |
| `("rz", theta, q)` | rotation about $z$ | $R_z(\theta) = e^{-i\theta Z/2}$ |
| `("cx", control, target)` | controlled-$X$ | CNOT |
| `("cz", q1, q2)` | controlled-$Z$, symmetric | $\mathrm{diag}(1,1,1,-1)$ |

Gates are applied left to right, so the matrix of a circuit is the product of the gate matrices in *reverse* order:

$$ U(\left[g_1, \ldots, g_L\right]) = U_{g_L} \cdots U_{g_2} U_{g_1} $$

This is the ordering that every diagram in this course uses and the opposite of the order in which the matrices are written; getting it backwards produces a circuit that is the reverse of the intended one, which for many circuits is *almost* right and therefore hard to notice.

Three properties are worth defending, because each is a design decision a real IR also has to make.

**It is data, not objects.** A gate is a tuple, so it can be compared with `==`, used as a dictionary key, printed, and pattern-matched. A pass is then an ordinary function from a list to a list, with no state, and two passes compose by function composition. Real IRs use richer structures — a directed acyclic graph, so that commuting gates are not artificially ordered — and Chapter 2 discusses what the list representation costs when a pass wants to look for gates that are adjacent *on a qubit* rather than adjacent *in the list*.

**Qubits are integers, and the integers are wires.** They are positions in the register, not names of physical qubits. The map from a program's qubits to a machine's qubits is the *layout*, it is chosen at layer 4, and it is a separate object; Chapter 3 makes it explicit. Conflating the two is the most common source of confusion when reading transpiler output, because after routing, wire $k$ no longer holds the qubit the programmer called $k$.

### What the IR deliberately leaves out

Four things, each of which a production IR includes and each of which is added back later in this course.

| Omitted | Why it can wait | Where it comes back |
| --- | --- | --- |
| Measurement, and classical control on its result | nothing in Chapters 1 to 3 branches on a measurement | Chapter 5, where mitigation is exactly post-processing of measurement results |
| Timing: durations, delays, explicit scheduling | at the circuit level a gate is instantaneous | Chapter 4, where a gate is a pulse with a duration |
| Qubit allocation, ancilla lifetimes, register names | $n$ is small and fixed in every example | Chapter 3, where the layout is the object being optimized |
| Barriers and pass-control annotations | our passes are few and their order is written by hand | Chapter 2, where pass ordering starts to matter |

The general point is that an IR is defined as much by what it refuses to represent as by what it represents. A representation that can express everything is a programming language, and it cannot be optimized.

### Why the gate set is the layer boundary

Here is the claim of this section: the layer boundaries in the table of §1.1 are exactly the points at which the gate set changes. Three gate sets appear in this course, and each boundary between them is a synthesis problem with a cost.

| Gate set | Chosen for | Who chose it |
| --- | --- | --- |
| The *authoring* set — the eight gates above | convenience: the gates that appear in textbook constructions | the person writing the algorithm |
| The *native* set — what a machine implements directly | physics: which Hamiltonian terms the control hardware can turn on | the device |
| The *fault-tolerant* set — Clifford$+T$ | the error-correcting code: which gates have a cheap fault-tolerant implementation | the code |

Each change of set is a rewriting problem, and each has a characteristic cost. Translating the authoring set into a native set of the form $\lbrace R_z, R_y, \mathrm{CX}\rbrace$ costs a constant factor per gate — Code Example 7 measures it as roughly $1.2$ on a small circuit, because most gates expand into one or two natives. Translating into Clifford$+T$ is different in kind: a general rotation $R_z(\theta)$ requires a *sequence* of Clifford$+T$ gates whose length grows as $\log(1/\varepsilon)$ for accuracy $\varepsilon$, so the cost is not a constant factor but a function of the precision you demand. That asymmetry is the single most important fact about fault-tolerant resource estimation, and Chapter 5 uses it.

The corollary is the practical reason to care. When an SDK's documentation says a backend has a "basis gate set", it is telling you where one of these boundaries sits for that machine, and therefore which synthesis problem its compiler had to solve. Nothing else about the interface matters as much.

### Depth, and what it assumes

The last piece of the contract is a cost model. Two functions summarize a circuit:

$$ \mathrm{depth}(C) = \text{number of layers when gates are packed greedily}, \qquad \mathrm{counts}(C) = \lbrace \text{name} \mapsto \text{count} \rbrace $$

Greedy layering is the obvious algorithm: keep, for each qubit, the index of the first layer in which it is free; place each gate in the earliest layer in which *all* the qubits it touches are free; advance those qubits. The result is the length of the longest chain of gates that share qubits, which is the critical path of the circuit.

Depth is the right metric when errors come from waiting — a circuit that finishes sooner has less time to decohere — and it is the wrong metric when errors come from gates, because a layer of ten gates counts the same as a layer of one. It also rests on an assumption that is simply false: that every gate takes the same time. On real hardware a two-qubit gate is typically an order of magnitude slower than a single-qubit gate, so the wall-clock time of a circuit is a *weighted* depth. Code Example 6 computes both and shows how far apart they get; Chapter 4 replaces the weights with pulse durations.

* * *

## 1.3 Compilation Is Meaning-Preserving Rewriting

### The three correctness relations

A compiler pass is a function $P$ that takes a circuit $C$ to a circuit $C' = P(C)$. It is *correct* when $C'$ means what $C$ meant. There are three versions of that statement in use, and every pass in this course declares which one it satisfies.

**Exact, up to a global phase.** The strongest relation and the default:

$$ U(C') = e^{i\varphi}\, U(C) \quad \text{for some } \varphi \in \mathbb{R} $$

The phase is quotiented out because it is unobservable: measurement probabilities depend on $\lvert \langle x \rvert \psi \rangle \rvert^2$, and expectation values on $\langle \psi \rvert A \lvert \psi \rangle$, neither of which sees an overall $e^{i\varphi}$. Optimization passes and basis translation satisfy this relation, and it is what the checker below tests.

**Exact, up to a qubit permutation.** A router that inserts SWAP gates and does not undo them ends with the program's qubits on different wires:

$$ U(C') = e^{i\varphi}\, \Pi\, U(C)\, \Pi'^{\dagger} $$

for permutation matrices $\Pi, \Pi'$ that the compiler must record and hand to the layer that interprets the measurement results. Forgetting to record them is a classic bug whose symptom is a perfectly plausible output distribution with the bits scrambled. Chapter 3 carries the permutation explicitly; the router in Code Example 7 avoids the issue by undoing its SWAPs, which is correct and wasteful.

**Approximate, within $\varepsilon$.** Synthesis into a discrete gate set cannot be exact, so the guarantee becomes a distance:

$$ \min_{\varphi} \left\lVert U(C') - e^{i\varphi} U(C) \right\rVert \le \varepsilon $$

in the operator norm, or the diamond norm if the circuit contains measurement or noise. This is the relation Chapter 2 needs for Clifford$+T$ synthesis and Chapter 4 for a finite-duration pulse. The important structural fact is that $\varepsilon$ adds up: $L$ gates each synthesized to $\varepsilon$ give a circuit accurate to $L\varepsilon$, by subadditivity of the norm, so the per-gate precision demanded by a long circuit grows with its length.

### Removing the global phase properly

Given two matrices $U$ and $V$ of size $d = 2^n$, we want the phase that makes them as close as a phase can make them, and then the residual distance. Write the Hilbert-Schmidt inner product

$$ \langle V, U \rangle = \mathrm{tr}\left( V^{\dagger} U \right) $$

The optimal phase is $e^{i\varphi} = \langle V, U \rangle / \lvert \langle V, U \rangle \rvert$, the standard least-squares answer: minimizing $\lVert U - e^{i\varphi}V\rVert_F^2$ over $\varphi$ means aligning $e^{i\varphi}$ with the overlap. For unitary $U, V$ Cauchy-Schwarz gives

$$ \left\lvert \mathrm{tr}\left(V^{\dagger}U\right) \right\rvert \le \sqrt{\mathrm{tr}\left(U^{\dagger}U\right)\,\mathrm{tr}\left(V^{\dagger}V\right)} = d $$

with equality exactly when $U = e^{i\varphi} V$. So the normalized overlap $\lvert \mathrm{tr}(V^{\dagger}U)\rvert / d$ is itself a complete test: it equals $1$ for equivalent circuits and is strictly less otherwise. That also disposes of the degenerate case — if the overlap is zero the phase is undefined, but two circuits with zero overlap are as inequivalent as it is possible to be, so any phase will do.

### The rule that a global phase is not always global

This is the one place where the convenience of quotienting out phases turns into a bug, so it gets its own rule.

$$ R_z(\pi) = \begin{pmatrix} e^{-i\pi/2} & 0 \cr 0 & e^{i\pi/2} \end{pmatrix} = -i \begin{pmatrix} 1 & 0 \cr 0 & -1 \end{pmatrix} = -i\, Z $$

$R_z(\pi)$ and $Z$ are equivalent: no experiment distinguishes them. But now control both on another qubit. The controlled version applies the factor $-i$ only when the control is $\lvert 1 \rangle$, and a phase applied conditionally is a *relative* phase, which is completely observable:

$$ \mathrm{C}\text{-}R_z(\pi) = \mathrm{diag}\left(1, 1, -i, i\right) \ne \mathrm{diag}\left(1,1,1,-1\right) = \mathrm{CZ} $$

Code Example 4 measures the discrepancy as $0.765$ — not a rounding error, a different gate. Hence the rule, which every quantum compiler enforces somewhere and every implementer gets wrong once:

> A rewrite that is correct only up to a global phase may not be applied to the body of a controlled gate. Either track the phase as data, or synthesize the controlled version from scratch.

Exercise 3 works out the correction that repairs the example above, and finds that the discarded phase reappears as an honest gate on the control qubit.

### The verification methodology of this course

Everything above becomes one reusable routine, and from Code Example 4 onwards every pass is called through it:

  1. Build $U(C)$ by running the circuit on each of the $2^n$ basis states. Cost: $2^n$ circuit simulations.
  2. Build $U(C')$ the same way.
  3. Remove the global phase with the trace formula and take $\max_{jk} \lvert U_{jk} - e^{i\varphi} V_{jk} \rvert$.
  4. Compare against a tolerance of $10^{-10}$, which is eight orders of magnitude above the $10^{-16}$ floating-point floor and eight below any error worth calling an error.

The honest limitation is the $2^n$ in step 1. At $n = 8$ this is 256 simulations and it costs milliseconds; at $n = 20$ it is a million simulations of a million-amplitude state, and it is out of reach. Above the exhaustive range there are two options, both used in this course. **Random input states**: run both circuits on one Haar-random state and compare the overlap. A pass that changes the unitary changes the output of a random state with probability one, so a single random state is a remarkably strong test — Exercise 4 measures it agreeing with the exhaustive check on every circuit it was tried on, at one simulation instead of $2^n$. **Structural arguments**: prove the rewrite rule once on the two or three qubits it touches, then argue that it is applied only where its precondition holds. Real compilers rely on the second and test with the first, which is why compiler bugs in this field are usually not wrong rules but rules applied in the wrong place. Code Example 5 constructs exactly that kind of bug and catches it.

### What "better" means

One more piece of vocabulary before the code. A pass that preserves meaning is not thereby an improvement; improvement needs an objective, and there are at least five in use.

| Metric | Why anyone cares | Which layer it belongs to |
| --- | --- | --- |
| Total gate count | crude proxy for everything | 3 |
| Two-qubit gate count | two-qubit errors dominate on most hardware | 3, 4 |
| Depth | proxy for time, hence for decoherence | 3, 6 |
| Wall-clock time | the real cost of waiting, with per-gate durations | 6 |
| $T$ count | the dominant cost under error correction | 3, and Chapter 5 |

These disagree, and Code Example 6 shows by how much: the same unitary, compiled two ways, can be $60\%$ cheaper in time on a machine whose two-qubit gate is fast and $1.8\%$ cheaper on one where it is slow. An optimizer is therefore always an optimizer *for* something, and a compiler that does not let you say what will optimize the wrong thing.

* * *

## 1.4 The Map of the SDKs

This course does not teach an SDK. It is nonetheless written for people who will use one, so this section maps the layers onto the parts of a framework, in terms general enough to survive the next several releases of all of them.

### The layer correspondence

Every circuit-model framework has the same components, under different names.

| Layer | The component, generically | What it is usually called |
| --- | --- | --- |
| 2. Circuit IR | a mutable circuit object with an append-a-gate method, and a serializable form | circuit, program, tape; the serialized form is an assembly-like text or a graph |
| 3. Optimization | a collection of rewriting passes, plus a manager that runs them in an order | transpiler passes, compiler passes, optimization levels |
| 4. Placement and routing | a description of which qubit pairs interact, plus layout and SWAP-insertion passes | coupling map, device topology, layout and routing passes |
| 5. Gate synthesis | a declared set of directly implementable gates, plus a translator into it | basis gates, native gates, gate set, decomposition passes |
| 6. Pulses and scheduling | waveform objects, channels, a scheduler, and a store of measured parameters | pulse or waveform API, schedules, calibration store |
| 7. Readout and mitigation | routines that turn raw counts into corrected estimates | measurement error mitigation, resilience levels, error-mitigation module |
| — | the machine description that layers 4 to 6 read | backend, target, device |
| — | the entry points that run a circuit and return counts or expectation values | primitives, samplers, estimators, executors |

Three observations about this table are worth more than the table itself.

**The machine description is the interesting object.** Layers 4, 5 and 6 are all parameterized by the same thing: a description of the target containing the coupling graph, the native gate set, and calibration data. When a framework is portable across hardware, it is because those three pieces of information have been factored out into one object, and reading that object's definition is the fastest way to learn what the framework can actually target.

**The differences between frameworks are mostly about which layer they put in front.** A framework built around circuits puts layer 2 in front and treats gradients as a service. One built around differentiable programs puts the algorithm layer in front, treats the circuit as an implementation detail, and is organized around parameter-shift rules and backpropagation — a genuinely different design, not a naming difference. One built around pulses puts layer 6 in front. All three contain all the layers.

**Optimization levels are a menu of pass orders.** The single knob most users touch — an integer labelled "optimization level" — selects a pass sequence, which is why Chapter 2 spends its time on individual rewriting rules rather than on the knob.

### What an SDK gives you that this course does not

Being fair about this is part of the point of the section. A production framework provides hardware access and queueing; passes that are far better than the ones written here, developed against real devices; a maintained machine description for every supported backend, updated as the calibrations change; and a community of people who have already hit the bug you are about to hit. None of that is replaceable, and none of it is what this course is for.

What this course provides is the other half: knowing what the passes did. That knowledge is what lets you read a transpiled circuit and see why it grew, decide whether an optimization level is helping, tell a routing problem from a synthesis problem when a result is wrong, and estimate the cost of an algorithm before writing any code at all. Concretely, when you finish these five chapters you should be able to open the documentation of any quantum framework, read its table of contents, and assign every entry to a row of the table above. That is the deliverable.

* * *

## 1.5 Building the Layer

The rest of the chapter is the implementation. Code Examples 1 and 2 build the IR; 3 exercises it; 4 builds the checker; 5 writes real passes and catches a real bug; 6 attaches a cost model; 7 assembles the pipeline that Chapters 2 to 5 will fill in.

### Code Example 1: The Simulator, Re-listed

This is the state-vector simulator from Chapter 2 of [Introduction to Quantum Computing](<../quantum-computing-introduction/index.html>) — the functions this course needs, verbatim. Save it as `qcsim.py`; the IR module below begins with `from qcsim import *`. Nothing here is new and nothing has been modified; if you have the file from the introductory course already, use that one.

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

The only function that does real work is `apply_gate`, and it is worth one sentence because the whole course leans on it: it reshapes the state into an $n$-index tensor, moves the target axes to the front, multiplies by the $2^k \times 2^k$ matrix, and moves the axes back. A gate on any subset of qubits therefore costs $O(2^k 2^n)$ operations and no extra memory, which is why $n = 8$ is instant and $n = 20$ would still be possible.

### Code Example 2: The Circuit IR

This is the contract of the course. Save it as `qir.py`; every later example, and every later chapter, begins with `from qir import *`.

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

Notice how little there is. Four public functions, one dispatch table, and no classes; the entire remaining content of the course is passes over this data structure. Two details will matter later. `run_circuit` accepts an optional initial state, which is what makes it possible to extract a circuit's matrix by running it on each basis vector in turn — Code Example 4 does exactly that. And `gate_counts` reports both the per-name counts and the aggregate `"2q"`, because the two-qubit count is the number that predicts error, and it should not have to be recomputed by every caller.

### Code Example 3: Bell and GHZ Through the IR

The first thing to do with a new representation is to run something whose answer is known.

```python
"""Chapter 1, Example 3: Bell and GHZ states through the IR.
Continues from Example 2 (same session)."""
import numpy as np
from qir import *

# ---- the two-line circuit that starts every quantum computing course -----
bell = [("h", 0), ("cx", 0, 1)]
psi = run_circuit(bell, 2)

print("Bell circuit :", bell)
print(f"  amplitudes : {np.round(psi.real, 6)}")
print(f"  depth      : {circuit_depth(bell, 2)}")
print(f"  gate counts: {gate_counts(bell)}")
print(f"  norm == 1  : "
      f"{np.isclose(probs(psi).sum(), 1.0, rtol=1e-12, atol=0.0)}")
print(f"  amplitude error vs 1/sqrt(2): "
      f"{abs(psi[0] - 2 ** -0.5):.2e}, {abs(psi[3] - 2 ** -0.5):.2e}")


# ---- the same state on n qubits, built two different ways ----------------
def ghz_chain(n):
    """GHZ by a chain of CNOTs: n - 1 two-qubit gates, depth n."""
    return [("h", 0)] + [("cx", q, q + 1) for q in range(n - 1)]


def ghz_log(n):
    """GHZ by doubling: the same n - 1 two-qubit gates, depth about log2(n)."""
    circ, span = [("h", 0)], 1
    while span < n:
        for q in range(min(span, n - span)):
            circ.append(("cx", q, q + span))
        span *= 2
    return circ


print("\nTwo circuits for the same n-qubit GHZ state")
head = (f"{'n':>3}{'chain depth':>13}{'log depth':>11}{'cx (chain)':>12}"
        f"{'cx (log)':>10}{'max |dpsi|':>13}")
print(head)
print("-" * len(head))
for n in range(2, 9):
    a, b = ghz_chain(n), ghz_log(n)
    pa, pb = run_circuit(a, n), run_circuit(b, n)
    err = np.max(np.abs(pa - pb))
    print(f"{n:>3}{circuit_depth(a, n):>13}{circuit_depth(b, n):>11}"
          f"{gate_counts(a)['2q']:>12}{gate_counts(b)['2q']:>10}{err:>13.2e}")

# ---- what a measurement of the four-qubit GHZ state returns --------------
n = 4
psi4 = run_circuit(ghz_log(n), n)
print(f"\nGHZ on {n} qubits, 2000 shots, seed 7:")
print(f"  {sample(psi4, 2000, seed=7)}")
print(f"  nonzero amplitudes: "
      f"{[format(i, f'0{n}b') for i in np.flatnonzero(np.abs(psi4) > 1e-12)]}")
print(f"  depth chain / log : {circuit_depth(ghz_chain(n), n)} / "
      f"{circuit_depth(ghz_log(n), n)}")

# ---- the depth of a circuit is not its gate count divided by n -----------
print("\nFour single-qubit gates, two arrangements:")
for label, circ in [("all on different qubits", [("h", q) for q in range(4)]),
                    ("all on qubit 0", [("h", 0), ("x", 0), ("z", 0), ("s", 0)])]:
    print(f"  {label:<24} gates = {len(circ)}  depth = "
          f"{circuit_depth(circ, 4)}")
```

```text
Bell circuit : [('h', 0), ('cx', 0, 1)]
  amplitudes : [0.707107 0.       0.       0.707107]
  depth      : 2
  gate counts: {'h': 1, 'cx': 1, '2q': 1}
  norm == 1  : True
  amplitude error vs 1/sqrt(2): 1.11e-16, 1.11e-16

Two circuits for the same n-qubit GHZ state
  n  chain depth  log depth  cx (chain)  cx (log)   max |dpsi|
--------------------------------------------------------------
  2            2          2           1         1     0.00e+00
  3            3          3           2         2     0.00e+00
  4            4          3           3         3     0.00e+00
  5            5          4           4         4     0.00e+00
  6            6          4           5         5     0.00e+00
  7            7          4           6         6     0.00e+00
  8            8          4           7         7     0.00e+00

GHZ on 4 qubits, 2000 shots, seed 7:
  {'0000': 1013, '1111': 987}
  nonzero amplitudes: ['0000', '1111']
  depth chain / log : 4 / 3

Four single-qubit gates, two arrangements:
  all on different qubits  gates = 4  depth = 1
  all on qubit 0           gates = 4  depth = 4
```

**What to look for.** The Bell state comes out exactly right, and the norm check uses a *relative* tolerance rather than testing a float against $1.0$ — a habit worth forming now, because Chapter 5 compares quantities spanning ten decades, where a naive equality test is not merely inelegant but wrong. The GHZ table is the first appearance of the course's central theme. Both constructions use exactly $n-1$ two-qubit gates, and both produce the same state to the last bit. They differ only in depth: the chain is $n$, the doubling construction is $\lceil \log_2 n \rceil + 1$, and at $n = 8$ that is 8 against 4. On a machine limited by decoherence the second is twice as good; on a machine limited by two-qubit gate error the two are identical. That is the same circuit compiled two ways, differing in one metric and not another, before any compiler has been written — and it is why §1.3 insisted that "optimize" needs an objective.

The last block makes the definition of depth concrete: four gates on four qubits have depth 1, four gates on one qubit have depth 4, and the gate count cannot tell them apart.

### Code Example 4: The Equivalence Checker, and the Rules It Validates

Now the tool that the rest of the course is built on.

```python
"""Chapter 1, Example 4: the equivalence checker, and the rules it validates.
Continues from Example 3 (same session)."""
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


def verdict(U, V, atol=1e-10):
    """Which of the three possible answers the comparison gives."""
    if np.max(np.abs(U - V)) <= atol:
        return "identical"
    if phase_free_error(U, V) <= atol:
        return "up to phase"
    return "DIFFERENT"


PI = np.pi
RULES = [
    (1, "H H = I", [("h", 0), ("h", 0)], []),
    (1, "X X = I", [("x", 0), ("x", 0)], []),
    (1, "S S = Z", [("s", 0), ("s", 0)], [("z", 0)]),
    (1, "T T = S", [("t", 0), ("t", 0)], [("s", 0)]),
    (1, "H Z H = X", [("h", 0), ("z", 0), ("h", 0)], [("x", 0)]),
    (1, "H X H = Z", [("h", 0), ("x", 0), ("h", 0)], [("z", 0)]),
    (1, "Rz(a) Rz(b) = Rz(a+b)",
     [("rz", 0.3, 0), ("rz", 0.7, 0)], [("rz", 1.0, 0)]),
    (1, "Rz(pi) = Z", [("rz", PI, 0)], [("z", 0)]),
    (1, "Rz(pi/2) = S", [("rz", PI / 2, 0)], [("s", 0)]),
    (2, "CX CX = I", [("cx", 0, 1), ("cx", 0, 1)], []),
    (2, "CZ = H(1) CX H(1)",
     [("cz", 0, 1)], [("h", 1), ("cx", 0, 1), ("h", 1)]),
    (2, "CZ(0,1) = CZ(1,0)", [("cz", 0, 1)], [("cz", 1, 0)]),
    (2, "CX(0,1) = CX(1,0)", [("cx", 0, 1)], [("cx", 1, 0)]),
    (2, "H,H CX(0,1) H,H = CX(1,0)",
     [("h", 0), ("h", 1), ("cx", 0, 1), ("h", 0), ("h", 1)], [("cx", 1, 0)]),
    (4, "GHZ chain = GHZ doubling", ghz_chain(4), ghz_log(4)),
]

head = (f"{'rule':<28}{'n':>3}{'max |U-V|':>12}{'phase-free':>12}"
        f"{'phase/pi':>10}  verdict")
print(head)
print("-" * len(head))
for n, label, left, right in RULES:
    U, V = unitary_of(left, n), unitary_of(right, n)
    phi = np.angle(best_global_phase(U, V)) / PI
    phi = phi if abs(phi) > 1e-9 else 0.0
    print(f"{label:<28}{n:>3}{np.max(np.abs(U - V)):>12.2e}"
          f"{phase_free_error(U, V):>12.2e}{phi:>10.3f}  {verdict(U, V)}")

print("\nTwo of those lines are the interesting ones.")
print("  Rz(pi) and Z differ by the unobservable factor exp(-i pi/2) = -i.")
print("  GHZ chain and GHZ doubling are different unitaries that agree on the")
print("  one input state the circuit is ever given:")
for n in (4, 6, 8):
    a, b = ghz_chain(n), ghz_log(n)
    on_zero = np.max(np.abs(run_circuit(a, n) - run_circuit(b, n)))
    as_unitary = phase_free_error(unitary_of(a, n), unitary_of(b, n))
    print(f"    n = {n}: error on |0...0> = {on_zero:.2e},  "
          f"as unitaries = {as_unitary:.2e}")


# ---- the trap: a global phase is not global inside a controlled block ----
def controlled(U):
    """The two-qubit gate applying U to qubit 1 when qubit 0 is |1> (big-endian)."""
    C = np.eye(4, dtype=complex)
    C[2:, 2:] = U
    return C


print("\nA global phase stops being global as soon as the gate is controlled:")
print(f"  phase-free error  Rz(pi) vs Z          : "
      f"{phase_free_error(rz(PI), Z):.2e}")
print(f"  phase-free error  C-Rz(pi) vs CZ       : "
      f"{phase_free_error(controlled(rz(PI)), CZ4):.2e}")
print("  diagonal of C-Rz(pi): "
      + " ".join(f"{v:+.3f}" for v in np.diag(controlled(rz(PI)))))
print("  diagonal of CZ      : "
      + " ".join(f"{v:+.3f}" for v in np.diag(CZ4)))
```

```text
rule                          n   max |U-V|  phase-free  phase/pi  verdict
--------------------------------------------------------------------------
H H = I                       1    2.22e-16    2.22e-16     0.000  identical
X X = I                       1    0.00e+00    0.00e+00     0.000  identical
S S = Z                       1    0.00e+00    0.00e+00     0.000  identical
T T = S                       1    2.22e-16    1.11e-16     0.000  identical
H Z H = X                     1    2.22e-16    2.22e-16     0.000  identical
H X H = Z                     1    2.22e-16    2.22e-16     0.000  identical
Rz(a) Rz(b) = Rz(a+b)         1    1.11e-16    1.11e-16     0.000  identical
Rz(pi) = Z                    1    1.41e+00    6.12e-17    -0.500  up to phase
Rz(pi/2) = S                  1    7.65e-01    1.11e-16    -0.250  up to phase
CX CX = I                     2    0.00e+00    0.00e+00     0.000  identical
CZ = H(1) CX H(1)             2    2.22e-16    2.22e-16     0.000  identical
CZ(0,1) = CZ(1,0)             2    0.00e+00    0.00e+00     0.000  identical
CX(0,1) = CX(1,0)             2    1.00e+00    1.00e+00     0.000  DIFFERENT
H,H CX(0,1) H,H = CX(1,0)     2    3.33e-16    3.33e-16     0.000  identical
GHZ chain = GHZ doubling      4    7.07e-01    7.07e-01     0.000  DIFFERENT

Two of those lines are the interesting ones.
  Rz(pi) and Z differ by the unobservable factor exp(-i pi/2) = -i.
  GHZ chain and GHZ doubling are different unitaries that agree on the
  one input state the circuit is ever given:
    n = 4: error on |0...0> = 0.00e+00,  as unitaries = 7.07e-01
    n = 6: error on |0...0> = 0.00e+00,  as unitaries = 7.07e-01
    n = 8: error on |0...0> = 0.00e+00,  as unitaries = 7.07e-01

A global phase stops being global as soon as the gate is controlled:
  phase-free error  Rz(pi) vs Z          : 6.12e-17
  phase-free error  C-Rz(pi) vs CZ       : 7.65e-01
  diagonal of C-Rz(pi): +1.000+0.000j +1.000+0.000j +0.000-1.000j +0.000+1.000j
  diagonal of CZ      : +1.000+0.000j +1.000+0.000j +1.000+0.000j -1.000+0.000j
```

**What to look for.** Eleven of the fifteen rules come out *identical*, at the $10^{-16}$ floor: they are the textbook circuit identities, and this is the last time anyone should take them on trust. The remaining four rows carry the content.

`Rz(pi) = Z` and `Rz(pi/2) = S` are equivalent up to a phase and not identical. The naive comparison reports errors of $1.41$ and $0.765$ — enormous, on the scale of a unitary matrix whose entries have modulus at most 1 — while the phase-free comparison reports $10^{-17}$. A checker without global-phase removal would reject correct rewrites here, and since a compiler contains many such rewrites, it would reject almost everything. The recovered phases, $-0.5\pi$ and $-0.25\pi$, are exactly $-\theta/2$: rotation gates carry the phase convention $R_z(\theta) = e^{-i\theta Z/2}$, and the phase gates do not.

`CX(0,1) = CX(1,0)` is properly rejected, with an error of $1.0$: swapping control and target changes the gate, and the checker says so. The way to *make* it true is on the next line — conjugating by Hadamards on both qubits exchanges the roles, because CNOT is asymmetric only in the computational basis.

The GHZ row is the most instructive failure in the chapter. The two constructions of Code Example 3 agreed to the last bit on $\lvert 0\ldots0\rangle$, and as unitaries they differ by $0.707$. Both statements are true, and they answer different questions. **State equivalence** is what you want when a circuit is a state preparation with a fixed input. **Unitary equivalence** is what you must have when the circuit is a subroutine that will be applied to something else, or controlled, or inverted. A pass justified by state equivalence and then applied inside a larger circuit is a bug, and it is a subtle one, because the circuit still works whenever it happens to be used at the top level. This course uses unitary equivalence throughout for that reason.

The controlled-gate block is the rule of §1.3 made numerical. $\mathrm{C}$-$R_z(\pi)$ has diagonal $(1, 1, -i, i)$ and CZ has diagonal $(1,1,1,-1)$; the discarded factor of $-i$ has become a relative phase between the two branches of the control.

### Code Example 5: Three Passes, and the Bug the Checker Catches

An optimizer, and then a deliberately broken optimizer.

```python
"""Chapter 1, Example 5: three rewriting passes, and the test that guards them.
Continues from Example 4 (same session)."""
import numpy as np
from qir import *

SELF_INVERSE = {"h", "x", "z", "cx", "cz"}
NAMES = ["h", "x", "z", "s", "t", "rx", "ry", "rz", "cx", "cz"]
ANGLES = [k * np.pi / 4 for k in (-3, -2, -1, 1, 2, 3, 4)]


def next_touching(circ, i, qs):
    """Index of the first gate after i that shares a qubit with qs, or len(circ)."""
    j = i + 1
    while j < len(circ) and not (qs & set(gate_qubits(circ[j]))):
        j += 1
    return j


def cancel_inverses(circ):
    """Pass 1: delete a pair of identical self-inverse gates that nothing separates."""
    out = list(circ)
    i = 0
    while i < len(out):
        g = out[i]
        if g[0] in SELF_INVERSE:
            j = next_touching(out, i, set(gate_qubits(g)))
            if j < len(out) and out[j] == g:
                del out[j], out[i]
                i = max(i - 1, 0)
                continue
        i += 1
    return out


def fuse_rotations(circ):
    """Pass 2: merge neighbouring rotations about the same axis on the same qubit."""
    out = list(circ)
    i = 0
    while i < len(out):
        g = out[i]
        if g[0] in ROT_1Q:
            j = next_touching(out, i, {g[2]})
            if j < len(out) and out[j][0] == g[0]:
                out[i] = (g[0], g[1] + out[j][1], g[2])
                del out[j]
                continue
        i += 1
    return out


def drop_null_rotations(circ, atol=1e-12):
    """Pass 3: delete rotations by a multiple of 4 pi, which are exactly the identity."""
    period = 4 * np.pi
    keep = []
    for g in circ:
        if g[0] in ROT_1Q:
            residual = abs((g[1] + period / 2) % period - period / 2)
            if residual < atol:
                continue
        keep.append(g)
    return keep


PASSES = [cancel_inverses, fuse_rotations, drop_null_rotations]


def optimize(circ, n, passes=PASSES, max_rounds=50):
    """Run the passes to a fixed point, checking equivalence after each one."""
    current = list(circ)
    for _ in range(max_rounds):
        before = current
        for p in passes:
            candidate = p(current)
            assert_equivalent(current, candidate, n, label=p.__name__)
            current = candidate
        if current == before:
            break
    return current


def random_circuit(n, length, rng):
    """A random circuit over the IR gate set; angles are multiples of pi/4."""
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


# ---- two hundred random circuits per size, optimized and checked ---------
head = (f"{'n':>3}{'length':>8}{'gates out':>11}{'2q out':>9}{'depth in':>10}"
        f"{'depth out':>11}{'worst error':>13}{'failures':>10}")
print(head)
print("-" * len(head))
for n, length in [(2, 20), (3, 30), (4, 40), (5, 50)]:
    rng = np.random.default_rng(1000 + n)
    tally = np.zeros(6)                  # gates in/out, 2q in/out, depth in/out
    worst, failures = 0.0, 0
    for _ in range(200):
        c = random_circuit(n, length, rng)
        o = optimize(c, n)
        err = phase_free_error(unitary_of(c, n), unitary_of(o, n))
        worst = max(worst, err)
        failures += int(err > 1e-10)
        tally += [len(c), len(o), gate_counts(c)["2q"], gate_counts(o)["2q"],
                  circuit_depth(c, n), circuit_depth(o, n)]
    print(f"{n:>3}{length:>8}{tally[1] / tally[0]:>10.1%}"
          f"{tally[3] / max(tally[2], 1):>9.1%}{tally[4] / 200:>10.2f}"
          f"{tally[5] / 200:>11.2f}{worst:>13.2e}{failures:>10}")


# ---- and now a pass with a bug in it -------------------------------------
def buggy_commute(circ):
    """WRONG: pretends Z commutes through the target of a CX. It commutes
    through the control, and not through the target."""
    out = list(circ)
    for i in range(len(out) - 1):
        g, h = out[i], out[i + 1]
        if g[0] == "cx" and h[0] == "z" and h[1] == g[2]:
            out[i], out[i + 1] = h, g
    return out


print("\nA pass with a plausible-sounding bug, and what the checker does:")
trap = [("h", 0), ("cx", 0, 1), ("z", 1), ("h", 1)]
print(f"  circuit        : {trap}")
print(f"  after the pass : {buggy_commute(trap)}")
print(f"  phase-free error: "
      f"{phase_free_error(unitary_of(trap, 2), unitary_of(buggy_commute(trap), 2)):.3f}")

rng = np.random.default_rng(5)
caught = tried = 0
for _ in range(500):
    c = random_circuit(3, 12, rng)
    b = buggy_commute(c)
    if b != c:
        tried += 1
        if phase_free_error(unitary_of(c, 3), unitary_of(b, 3)) > 1e-10:
            caught += 1
print(f"  random property test: the pass fired on {tried} of 500 circuits and "
      f"was caught on {caught}")
print(f"  Z through the *control* instead: "
      f"{phase_free_error(unitary_of([('cx', 0, 1), ('z', 0)], 2), unitary_of([('z', 0), ('cx', 0, 1)], 2)):.2e}")


# ---- where the wins actually are: structure, not randomness --------------
def cz_star(n):
    """CZ from every other qubit onto qubit n-1, twice over."""
    ctrl = list(range(n - 1))
    return [("cz", q, n - 1) for q in ctrl + ctrl]


def expand_cz(circ):
    """A basis-translation pass: every CZ becomes H CX H on its second qubit."""
    out = []
    for g in circ:
        if g[0] == "cz":
            out += [("h", g[2]), ("cx", g[1], g[2]), ("h", g[2])]
        else:
            out.append(g)
    return out


print("\nStructured circuits: a CZ star, translated to CX and then optimized.")
head = (f"{'n':>3}{'cz':>5}{'gates after expand':>20}{'after optimize':>16}"
        f"{'h out':>7}{'depth in':>10}{'depth out':>11}{'error':>10}")
print(head)
print("-" * len(head))
for n in (4, 5, 6, 7):
    logical = cz_star(n)
    expanded = expand_cz(logical)
    assert_equivalent(logical, expanded, n, label="expand_cz")
    tuned = optimize(expanded, n)
    assert_equivalent(logical, tuned, n, label="expand+optimize")
    err = phase_free_error(unitary_of(logical, n), unitary_of(tuned, n))
    print(f"{n:>3}{gate_counts(logical)['cz']:>5}{len(expanded):>20}"
          f"{len(tuned):>16}{gate_counts(tuned).get('h', 0):>7}"
          f"{circuit_depth(expanded, n):>10}{circuit_depth(tuned, n):>11}"
          f"{err:>10.1e}")
print("Every internal H pair produced by translating one CZ at a time is")
print("cancelled by the next one; only the two at the ends survive.")
```

```text
  n  length  gates out   2q out  depth in  depth out  worst error  failures
---------------------------------------------------------------------------
  2      20     91.1%    93.9%     14.94      13.76     4.71e-16         0
  3      30     91.8%    96.6%     16.84      15.89     5.24e-16         0
  4      40     92.2%    96.9%     18.25      17.27     4.78e-16         0
  5      50     92.7%    98.1%     19.18      18.25     5.90e-16         0

A pass with a plausible-sounding bug, and what the checker does:
  circuit        : [('h', 0), ('cx', 0, 1), ('z', 1), ('h', 1)]
  after the pass : [('h', 0), ('z', 1), ('cx', 0, 1), ('h', 1)]
  phase-free error: 1.000
  random property test: the pass fired on 19 of 500 circuits and was caught on 18
  Z through the *control* instead: 0.00e+00

Structured circuits: a CZ star, translated to CX and then optimized.
  n   cz  gates after expand  after optimize  h out  depth in  depth out     error
----------------------------------------------------------------------------------
  4    6                  18               8      2        18          8   2.2e-16
  5    8                  24              10      2        24         10   2.2e-16
  6   10                  30              12      2        30         12   2.2e-16
  7   12                  36              14      2        36         14   2.2e-16
Every internal H pair produced by translating one CZ at a time is
cancelled by the next one; only the two at the ends survive.
```

**What to look for.** The random-circuit table is the honest measurement, and the honest answer is that a peephole optimizer barely helps on random input: about $8\%$ of the gates and $5\%$ of the depth. That is not a defect in the passes. A random circuit has no structure to exploit and almost no cancellable pairs; real circuits are full of structure, because they were generated by a human or a template, and the last block shows what happens then. Eight hundred circuits were optimized and eight hundred were verified, worst-case discrepancy $5.9 \times 10^{-16}$, zero failures. That last column is the point of the table.

The bug is the most valuable part of the example. "$Z$ commutes through CX" is the kind of half-remembered rule that gets written into a pass: it is true for the *control* qubit — verified at exactly $0.00$ on the last line — and false for the target, where $Z$ anticommutes with the $X$ that CX applies. The checker catches it on a four-gate circuit with an error of $1.0$. The property test then says something more interesting: on 500 random circuits the buggy pass found something to rewrite only 19 times, and of those 19, one produced a circuit that was still equivalent by coincidence. Random testing will find the bug, but you have to sample enough for it to fire, and a single passing test means less than it appears to.

The structured example closes the loop. Translating each CZ independently into $H\,\mathrm{CX}\,H$ is the obvious basis translation and it is wasteful: for $n = 6$ it produces 30 gates where 12 suffice, because every internal pair of Hadamards on the shared qubit is redundant. Cancelling them removes 18 gates and 18 layers. This is why real compilers interleave synthesis and optimization instead of running each once, and it is the subject Chapter 2 opens with.

### Code Example 6: A Time Budget and an Error Budget

Two circuits that implement the same unitary do not cost the same. Here is how much they differ, as a function of the machine.

```python
"""Chapter 1, Example 6: what the metrics buy — a time budget and an error budget.
Continues from Example 5 (same session)."""
import math
import numpy as np
from qir import *


def circuit_layers(circ, n):
    """The greedy layering that circuit_depth counts, returned as lists of gates."""
    ready, layers = [0] * n, []
    for g in circ:
        qs = gate_qubits(g)
        k = max(ready[q] for q in qs)
        while len(layers) <= k:
            layers.append([])
        layers[k].append(g)
        for q in qs:
            ready[q] = k + 1
    return layers


def wall_clock(circ, n, t_2q):
    """Duration in units of one single-qubit gate time, layer by layer.

    t_2q is the two-qubit gate duration in the same units. A layer takes as
    long as its slowest gate, which is why depth alone does not fix the time.
    """
    total = 0.0
    for layer in circuit_layers(circ, n):
        total += max(t_2q if g[0] in TWO_Q else 1.0 for g in layer)
    return total


def gate_error(circ, eps_1q, eps_2q):
    """1 minus the product of the per-gate success probabilities."""
    n2 = gate_counts(circ)["2q"]
    n1 = len(circ) - n2
    return 1.0 - (1.0 - eps_1q) ** n1 * (1.0 - eps_2q) ** n2


def idle_error(circ, n, t_2q, t_coh):
    """Decoherence of n qubits held for the duration of the circuit."""
    return 1.0 - math.exp(-n * wall_clock(circ, n, t_2q) / t_coh)


# ---- the layering is the same object circuit_depth counts ----------------
n = 6
logical = cz_star(n)
naive, tuned = expand_cz(logical), optimize(expand_cz(logical), n)
for label, circ in [("naive", naive), ("optimized", tuned)]:
    print(f"{label:>10}: {len(circuit_layers(circ, n))} layers, circuit_depth = "
          f"{circuit_depth(circ, n)}, gates = {len(circ)}")

# ---- the same unitary, two costs, as the machine gets more lopsided ------
print("\nThe two circuits implement the same unitary. Cost in units of one")
print("single-qubit gate time, as the two-qubit gate gets slower:")
head = (f"{'t_2q':>6}{'naive time':>12}{'tuned time':>12}{'saved':>8}"
        f"{'naive/depth':>13}{'tuned/depth':>13}")
print(head)
print("-" * len(head))
for t_2q in (1.0, 3.0, 10.0, 30.0, 100.0):
    a, b = wall_clock(naive, n, t_2q), wall_clock(tuned, n, t_2q)
    print(f"{t_2q:>6.0f}{a:>12.0f}{b:>12.0f}{1 - b / a:>8.1%}"
          f"{a / circuit_depth(naive, n):>13.2f}"
          f"{b / circuit_depth(tuned, n):>13.2f}")

# ---- which count dominates the error ------------------------------------
print("\nError budget of the optimized circuit, sweeping the two-qubit error")
print("rate over decades at a fixed single-qubit rate of 1e-4:")
eps_1q = 1e-4
head = (f"{'eps_2q':>9}{'ratio':>8}{'from 1q':>11}{'from 2q':>11}"
        f"{'total':>11}{'2q share':>10}")
print(head)
print("-" * len(head))
n2 = gate_counts(tuned)["2q"]
n1 = len(tuned) - n2
for eps_2q in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
    e1 = 1.0 - (1.0 - eps_1q) ** n1
    e2 = 1.0 - (1.0 - eps_2q) ** n2
    tot = gate_error(tuned, eps_1q, eps_2q)
    print(f"{eps_2q:>9.0e}{eps_2q / eps_1q:>8.0f}{e1:>11.2e}{e2:>11.2e}"
          f"{tot:>11.2e}{e2 / (e1 + e2):>10.1%}")
print(f"  the circuit has {n1} single-qubit and {n2} two-qubit gates")

# ---- time and coherence together ----------------------------------------
print("\nAdding decoherence, with the coherence time in the same units:")
head = (f"{'t_coh':>9}{'naive gate':>12}{'naive idle':>12}{'naive tot':>11}"
        f"{'tuned tot':>11}{'improvement':>13}")
print(head)
print("-" * len(head))
for t_coh in (1e3, 1e4, 1e5, 1e6):
    ga = gate_error(naive, eps_1q, 1e-3)
    ia = idle_error(naive, n, 10.0, t_coh)
    gb = gate_error(tuned, eps_1q, 1e-3)
    ib = idle_error(tuned, n, 10.0, t_coh)
    ta, tb = 1 - (1 - ga) * (1 - ia), 1 - (1 - gb) * (1 - ib)
    print(f"{t_coh:>9.0e}{ga:>12.2e}{ia:>12.2e}{ta:>11.2e}{tb:>11.2e}"
          f"{ta / tb:>12.2f}x")


# ---- comparing a float against a power of ten ---------------------------
def relative_verdict(value, reference, rel_tol=1e-9):
    """Compare a float with a power of ten without ever testing equality."""
    if math.isclose(value, reference, rel_tol=rel_tol):
        return "at"
    return "below" if value < reference else "above"


print("\nWhy threshold comparisons carry a relative tolerance:")
budget = 0.0
for _ in range(100):
    budget += 1e-5           # an error budget accumulated one gate at a time
cubed = (1e-1) ** 3          # the same number, reached by a different route
print(f"  the literal 1e-3               : {1e-3!r}")
print(f"  accumulated 100 x 1e-5         : {budget!r}   == 1e-3 ? "
      f"{budget == 1e-3}")
print(f"  (1e-1) ** 3                    : {cubed!r}   == 1e-3 ? "
      f"{cubed == 1e-3}")
print(f"  relative_verdict of each       : "
      f"{relative_verdict(budget, 1e-3)}, {relative_verdict(cubed, 1e-3)}")
tot = gate_error(tuned, eps_1q / 10, 1e-4)
print(f"  this circuit at eps_2q = 1e-4  : {tot:.6e}  -> "
      f"{relative_verdict(tot, 1e-3)} 1e-3")
```

```text
     naive: 30 layers, circuit_depth = 30, gates = 30
 optimized: 12 layers, circuit_depth = 12, gates = 12

The two circuits implement the same unitary. Cost in units of one
single-qubit gate time, as the two-qubit gate gets slower:
  t_2q  naive time  tuned time   saved  naive/depth  tuned/depth
----------------------------------------------------------------
     1          30          12   60.0%         1.00         1.00
     3          50          32   36.0%         1.67         2.67
    10         120         102   15.0%         4.00         8.50
    30         320         302    5.6%        10.67        25.17
   100        1020        1002    1.8%        34.00        83.50

Error budget of the optimized circuit, sweeping the two-qubit error
rate over decades at a fixed single-qubit rate of 1e-4:
   eps_2q   ratio    from 1q    from 2q      total  2q share
------------------------------------------------------------
    1e-04       1   2.00e-04   1.00e-03   1.20e-03     83.3%
    3e-04       3   2.00e-04   3.00e-03   3.20e-03     93.7%
    1e-03      10   2.00e-04   9.96e-03   1.02e-02     98.0%
    3e-03      30   2.00e-04   2.96e-02   2.98e-02     99.3%
    1e-02     100   2.00e-04   9.56e-02   9.58e-02     99.8%
  the circuit has 2 single-qubit and 10 two-qubit gates

Adding decoherence, with the coherence time in the same units:
    t_coh  naive gate  naive idle  naive tot  tuned tot  improvement
--------------------------------------------------------------------
    1e+03    1.19e-02    5.13e-01   5.19e-01   4.63e-01        1.12x
    1e+04    1.19e-02    6.95e-02   8.06e-02   6.89e-02        1.17x
    1e+05    1.19e-02    7.17e-03   1.90e-02   1.62e-02        1.17x
    1e+06    1.19e-02    7.20e-04   1.26e-02   1.08e-02        1.18x

Why threshold comparisons carry a relative tolerance:
  the literal 1e-3               : 0.001
  accumulated 100 x 1e-5         : 0.001000000000000002   == 1e-3 ? False
  (1e-1) ** 3                    : 0.0010000000000000002   == 1e-3 ? False
  relative_verdict of each       : at, at
  this circuit at eps_2q = 1e-4  : 1.019530e-03  -> above 1e-3
```

**What to look for.** The first table answers the question §1.3 posed. The optimized circuit is $60\%$ faster than the naive one on a machine whose two-qubit gate is as fast as a single-qubit gate, and $1.8\%$ faster on a machine where it is a hundred times slower — because everything the optimizer removed was a single-qubit gate, and single-qubit gates stop mattering when two-qubit gates dominate the clock. The same rewrite, the same two circuits, and a factor of thirty difference in how much it was worth. This is what it means to say that an optimizer needs an objective: the objective is a property of the target machine, and the compiler has to be told.

The `naive/depth` and `tuned/depth` columns give the average duration of a layer, which climbs from 1 to 34 and 84 as the two-qubit gate slows down. Depth is a proxy for time only when the layers are alike, and after optimization they are not: the optimized circuit is *denser* in two-qubit gates, so its layers are individually more expensive, and a pass that reduces depth by removing cheap gates has improved the wrong number. The error table is a one-way result: the two-qubit share of the error is already $83\%$ when the two error rates are equal, because the optimized circuit has ten two-qubit gates and two single-qubit gates, and it reaches $99.8\%$ when the two-qubit rate is a hundred times worse. This is why every real compiler's cost function is dominated by the two-qubit count, and why Chapter 2 spends its effort on CNOT counts rather than on total gate counts.

The decoherence table shows the crossover. When the coherence time is only a thousand gate times, idling dominates the error completely — $0.513$ against $0.0119$ from gates — and the optimizer's shortening of the circuit is worth a factor of $1.12$ overall. At a coherence time of $10^6$ the idle term has become negligible and the improvement settles at $1.18$. The two regimes want different objectives: depth when idling dominates, two-qubit count when gates do.

The last block is a floating-point rule that this course applies everywhere. An error budget accumulated as a hundred additions of $10^{-5}$ gives $0.001000000000000002$, and $(10^{-1})^3$ gives $0.0010000000000000002$; neither equals the literal `1e-3`, and a threshold test written as `==`, or as `<` against a boundary that the arithmetic is supposed to land exactly on, will misfire. Every comparison against a power of ten in this course goes through a relative tolerance. Chapter 5, which multiplies decades together to reach a physical qubit count, is where this stops being pedantry.

### Code Example 7: The Stack as a Pipeline

The last example assembles everything into the shape the remaining chapters will fill in.

```python
"""Chapter 1, Example 7: the whole stack as a pipeline of guarded passes.
Continues from Example 6 (same session)."""
import numpy as np
from qir import *


def swap_gates(a, b):
    """SWAP as three CX, the only way to move a qubit with the IR gate set."""
    return [("cx", a, b), ("cx", b, a), ("cx", a, b)]


def route_on_line(circ, n):
    """Layer 3 stub: make every two-qubit gate act on neighbours of a line.

    A distant gate is brought together by SWAPs and the SWAPs are then undone,
    so the qubit-to-wire map is the same at the end as at the start and the
    circuit stays exactly equivalent. Chapter 3 does the cheaper thing, which
    is to leave the permutation in place and carry it in the compiler state.
    """
    out = []
    for g in circ:
        if g[0] in TWO_Q and abs(g[1] - g[2]) > 1:
            c, t = g[1], g[2]
            d = 1 if c > t else -1
            pairs, p = [], t
            while p + d != c:
                pairs.append((p, p + d))
                p += d
            forward = [s for pair in pairs for s in swap_gates(*pair)]
            out += forward + [(g[0], c, p)] + forward[::-1]
        else:
            out.append(g)
    return out


# ---- Layer 5 stub: translate into a native basis {rz, ry, cx} ------------
PI = np.pi
NATIVE_SET = {"rz", "ry", "cx"}
NATIVE_RULES = {
    "z": lambda q: [("rz", PI, q)],
    "s": lambda q: [("rz", PI / 2, q)],
    "t": lambda q: [("rz", PI / 4, q)],
    "x": lambda q: [("rz", PI, q), ("ry", PI, q)],
    "h": lambda q: [("rz", PI, q), ("ry", PI / 2, q)],
}


def translate_to_native(circ):
    """Rewrite every gate into {rz, ry, cx}. Each rule holds up to a phase."""
    out = []
    for g in circ:
        name = g[0]
        if name in NATIVE_SET:
            out.append(g)
        elif name in NATIVE_RULES:
            out += NATIVE_RULES[name](g[1])
        elif name == "rx":
            theta, q = g[1], g[2]
            out += [("rz", PI / 2, q), ("ry", theta, q), ("rz", -PI / 2, q)]
        elif name == "cz":
            out += translate_to_native([("h", g[2])])
            out.append(("cx", g[1], g[2]))
            out += translate_to_native([("h", g[2])])
        else:
            raise ValueError(f"no native rule for {name!r}")
    return out


print("Every translation rule, checked before it is used:")
head = f"{'rule':<34}{'phase-free error':>18}"
print(head)
print("-" * len(head))
for name, circ in [("z  -> rz(pi)", [("z", 0)]),
                   ("s  -> rz(pi/2)", [("s", 0)]),
                   ("t  -> rz(pi/4)", [("t", 0)]),
                   ("x  -> rz(pi) ry(pi)", [("x", 0)]),
                   ("h  -> rz(pi) ry(pi/2)", [("h", 0)]),
                   ("rx(0.7) -> rz ry rz", [("rx", 0.7, 0)])]:
    err = phase_free_error(unitary_of(circ, 1),
                           unitary_of(translate_to_native(circ), 1))
    print(f"{name:<34}{err:>18.2e}")
err = phase_free_error(unitary_of([("cz", 0, 1)], 2),
                       unitary_of(translate_to_native([("cz", 0, 1)]), 2))
print(f"{'cz -> h cx h, then h expanded':<34}{err:>18.2e}")
print("Each rule is exact only up to a global phase, which is why none of them")
print("may be applied to the body of a controlled gate.")

# ---- the pipeline --------------------------------------------------------
STACK = [("optimize", lambda c, n: optimize(c, n)),
         ("route on a line", route_on_line),
         ("translate to native", lambda c, n: translate_to_native(c)),
         ("optimize again", lambda c, n: optimize(c, n))]


def compile_stack(circ, n, verbose=True):
    """Run the stack, checking equivalence against the input after every stage."""
    rows = [("input", circ)]
    current = circ
    for label, stage in STACK:
        current = stage(current, n)
        assert_equivalent(circ, current, n, label=label)
        rows.append((label, current))
    if verbose:
        head = (f"{'stage':<22}{'gates':>7}{'2q':>5}{'depth':>7}"
                f"{'time@10':>9}{'error':>12}{'vs input':>11}")
        print(head)
        print("-" * len(head))
        for label, c in rows:
            print(f"{label:<22}{len(c):>7}{gate_counts(c)['2q']:>5}"
                  f"{circuit_depth(c, n):>7}{wall_clock(c, n, 10.0):>9.0f}"
                  f"{gate_error(c, 1e-4, 1e-3):>12.2e}"
                  f"{phase_free_error(unitary_of(circ, n), unitary_of(c, n)):>11.1e}")
    return current


# A five-qubit circuit whose two-qubit gates are deliberately non-local.
n = 5
algorithm = [("h", 0), ("h", 1), ("t", 2),
             ("cz", 0, 4), ("cx", 0, 3), ("s", 4),
             ("rx", PI / 3, 2), ("cx", 4, 1), ("h", 4), ("h", 4),
             ("cz", 1, 3), ("rz", PI / 8, 0), ("rz", -PI / 8, 0)]
print(f"\nOne five-qubit circuit through the whole stack "
      f"({len(algorithm)} gates in):")
final = compile_stack(algorithm, n)
print(f"\nnative gate names in the output: "
      f"{sorted(set(g[0] for g in final))}")
print(f"two-qubit gates: {gate_counts(algorithm)['2q']} logical -> "
      f"{gate_counts(final)['2q']} physical, "
      f"a factor {gate_counts(final)['2q'] / gate_counts(algorithm)['2q']:.1f}")

# ---- the same circuit on three connectivities ----------------------------
print("\nThe same circuit, if the machine were connected differently:")
head = (f"{'connectivity':<20}{'2q out':>8}{'depth out':>11}{'error':>11}"
        f"{'check':>10}")
print(head)
print("-" * len(head))
for label, routed in [("all-to-all", optimize(algorithm, n)),
                      ("line 0-1-2-3-4", route_on_line(optimize(algorithm, n), n))]:
    tuned = optimize(translate_to_native(routed), n)
    print(f"{label:<20}{gate_counts(tuned)['2q']:>8}{circuit_depth(tuned, n):>11}"
          f"{gate_error(tuned, 1e-4, 1e-3):>11.2e}"
          f"{assert_equivalent(algorithm, tuned, n, label=label):>10.1e}")
print("the circuit asks for the pairs "
      f"{sorted((min(g[1], g[2]), max(g[1], g[2])) for g in algorithm if g[0] in TWO_Q)}")
```

```text
Every translation rule, checked before it is used:
rule                                phase-free error
----------------------------------------------------
z  -> rz(pi)                                6.12e-17
s  -> rz(pi/2)                              1.21e-16
t  -> rz(pi/4)                              2.48e-16
x  -> rz(pi) ry(pi)                         6.12e-17
h  -> rz(pi) ry(pi/2)                       1.19e-16
rx(0.7) -> rz ry rz                         1.24e-16
cz -> h cx h, then h expanded               2.09e-16
Each rule is exact only up to a global phase, which is why none of them
may be applied to the body of a controlled gate.

One five-qubit circuit through the whole stack (13 gates in):
stage                   gates   2q  depth  time@10       error   vs input
-------------------------------------------------------------------------
input                      13    4      6       42    4.89e-03    0.0e+00
optimize                    9    4      5       41    4.49e-03    1.2e-16
route on a line            57   52     53      521    5.12e-02    1.2e-16
translate to native        69   52     63      531    5.23e-02    3.6e-16
optimize again             69   52     63      531    5.23e-02    3.6e-16

native gate names in the output: ['cx', 'ry', 'rz']
two-qubit gates: 4 logical -> 52 physical, a factor 13.0

The same circuit, if the machine were connected differently:
connectivity          2q out  depth out      error     check
------------------------------------------------------------
all-to-all                 4         10   5.59e-03   4.3e-16
line 0-1-2-3-4            52         63   5.23e-02   3.6e-16
the circuit asks for the pairs [(0, 3), (0, 4), (1, 3), (1, 4)]
```

**What to look for.** The translation table first: seven rules, each verified before use, each exact to $10^{-16}$ *after* the global phase is removed and none of them exact before. A native basis of $\lbrace R_z, R_y, \mathrm{CX}\rbrace$ is enough to express every gate in the IR, which is the content of the claim that the authoring set and the native set are different vocabularies for the same thing — but the translation from one to the other is phase-sloppy in every single line, which is exactly why the rule about controlled blocks has to be a rule.

The pipeline table is the number to remember from this chapter. Thirteen logical gates with four two-qubit gates go in. After optimization, nine gates. After routing onto a line, **fifty-seven gates with fifty-two two-qubit gates** — a factor of thirteen on the count that dominates the error, and a factor of ten in wall-clock time, from nothing but connectivity. The error estimate follows: $4.9 \times 10^{-3}$ before routing, $5.1 \times 10^{-2}$ after. And every stage was checked against the input, at $10^{-16}$.

Three honest observations. First, the router is deliberately naive: it undoes its own SWAPs so that the qubit-to-wire map is unchanged and the equivalence check can be exact, which doubles the SWAP count. Chapter 3 keeps the permutation instead and halves it, then does much better still by choosing a good initial layout. Second, the final `optimize` stage finds nothing at all — 69 gates in, 69 gates out. That is not padding; it is the measurement that the peephole rules of Example 5 are too weak to clean up after a router, because the cancellable pairs it produces are separated in the list rather than adjacent in it. Chapter 2 fixes that with commutation-aware cancellation, and it illustrates that pass *ordering* and pass *strength* are separate problems. Third, the `all-to-all` row is the counterfactual: the same circuit on a machine that could execute its four gates directly needs four two-qubit gates and depth 10, against 52 and 63. The requested pairs — $(0,3), (0,4), (1,3), (1,4)$ — are precisely the ones a line does not have.

Every stage of that pipeline is a stub, and the rest of the course replaces them one at a time while the guard rail — `assert_equivalent` — stays exactly where it is.

| Chapter | Stub it replaces | With |
| --- | --- | --- |
| 2 | `optimize` | peephole and commutation rules, ZYZ and KAK synthesis, $T$ counting |
| 3 | `route_on_line` | layout search and SABRE-style routing on real connectivity graphs |
| 4 | the gate durations | pulse shapes, DRAG, and the calibration loops that set them |
| 5 | the error model | readout correction, ZNE, PEC, and resource estimation |

* * *

## Exercises

#### Exercise 1: Extending the IR

Add two gates to the IR: `("y", q)` for the Pauli $Y$, and `("swap", a, b)` for the SWAP gate.

  1. Which parts of `qir.py` have to change for each of the two? Answer for `gate_qubits`, `apply_ir_gate`, `circuit_depth` and `gate_counts` separately.
  2. Implement both, and verify with the checker that `("swap", 0, 1)` equals three CX gates and that `("y", 0)` equals $Z$ followed by $X$ up to a phase.
  3. What does `gate_counts` report for `[("swap", 0, 1), ("y", 0), ("cx", 0, 1)]`, and is that the right answer for a machine that has no native SWAP?
  4. Compare the depth of a circuit written with `swap` against the same circuit with SWAP expanded into three CX. Which number should a scheduler believe?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Adding <code>y</code> requires exactly one change: a new entry in <code>FIXED_1Q</code>. Everything else keys off that dictionary — <code>gate_qubits</code> returns <code>(g[1],)</code> for any name in it, <code>apply_ir_gate</code> looks the matrix up in it, and the two metric functions do not care what a gate is called. Adding <code>swap</code> requires three: a branch in <code>apply_ir_gate</code>, because it needs a matrix that is not in either table; membership in <code>TWO_Q</code>, so that <code>gate_qubits</code> returns both qubits and <code>gate_counts</code> includes it in <code>"2q"</code>; and nothing at all in <code>circuit_depth</code>, which is written entirely in terms of <code>gate_qubits</code>. That asymmetry is the practical content of the exercise: a well-factored IR localizes a new *kind* of gate to the dispatch, and a new *instance* of an existing kind to a table.</p>

<p><strong>2.</strong> Both check out exactly. Note the Python subtlety in the code below: <code>from qir import *</code> copies bindings into your namespace, so rebinding <code>apply_ir_gate</code> there does nothing — <code>run_circuit</code> resolves the name in <em>its own</em> module's globals. Patching <code>qir</code> itself is what works, and editing <code>qir.py</code> is what you would actually do.</p>

<p><strong>3.</strong> <code>{'swap': 1, 'y': 1, 'cx': 1, '2q': 2}</code>. It is the right answer for a machine with a native SWAP and misleading for one without, where the honest count is 4 two-qubit gates because the SWAP will become three CX. This is a real design question for an IR: a gate that the target cannot execute should either be forbidden or be counted at its expanded cost, and quoting a two-qubit count before basis translation is a common way to under-report a circuit's difficulty by a factor of three.</p>

<p><strong>4.</strong> The <code>swap</code> version has depth 2 and the expanded version depth 4. The scheduler should believe the expanded one, because depth is only a proxy for time when its unit of time is a gate the machine can actually run. The general rule: metrics are meaningful only relative to a declared gate set, which is the same point as §1.2 in a different guise.</p>

```python
import qir                                  # patch the module, not a local copy

qir.FIXED_1Q["y"] = Y                       # 1. one more fixed single-qubit gate
qir.TWO_Q = ("cx", "cz", "swap")            # 2. gate_qubits and gate_counts follow
SWAP4 = np.eye(4, dtype=complex)[[0, 2, 1, 3]]
_base_apply = qir.apply_ir_gate


def apply_ir_gate(state, g, n):             # 3. one more branch in the dispatch
    if g[0] == "swap":
        return apply_gate(state, SWAP4, [g[1], g[2]], n)
    return _base_apply(state, g, n)


qir.apply_ir_gate = apply_ir_gate           # run_circuit resolves this at call time

swap3 = [("cx", 0, 1), ("cx", 1, 0), ("cx", 0, 1)]
print(f"{phase_free_error(unitary_of([('swap', 0, 1)], 2), unitary_of(swap3, 2)):.1e}")
print(f"{phase_free_error(unitary_of([('y', 0)], 1), unitary_of([('z', 0), ('x', 0)], 1)):.1e}")
print(gate_counts([("swap", 0, 1), ("y", 0), ("cx", 0, 1)]))
print(circuit_depth([("swap", 0, 1), ("y", 0)], 2),
      circuit_depth(swap3 + [("y", 0)], 2))
# 0.0e+00
# 0.0e+00
# {'swap': 1, 'y': 1, 'cx': 1, '2q': 2}
# 2 4
```

</details>

#### Exercise 2: Depth, Time, and What Greedy Layering Assumes

Take the circuit $C = [(h,0), (cx,0,1), (h,2), (cx,2,3), (z,1), (z,3)]$ on four qubits.

  1. Compute `circuit_depth(C, 4)` by hand, and give the contents of each layer. Then check.
  2. Construct two circuits on four qubits with the same depth but wall-clock times differing by the full factor $t_{2q}$. What does this say about using depth as an objective?
  3. Greedy layering gives a *lower bound* on the real execution time. Name three things it omits.
  4. Under which of the two error models of Code Example 6 is depth the right objective, and under which is the two-qubit gate count? What decides which regime a machine is in?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Depth 3, with two gates in each layer. Layer 0 is \([(h,0), (h,2)]\): both qubits are free at the start. Layer 1 is \([(cx,0,1), (cx,2,3)]\): each needs a qubit that layer 0 used. Layer 2 is \([(z,1), (z,3)]\). The greedy rule places a gate in the earliest layer in which all of its qubits are free, so the circuit's written order — which interleaves the two halves — has no effect on the answer, and neither would writing all of the first half before all of the second. Greedy layering is invariant under reordering gates that act on disjoint qubits, which is exactly the freedom a scheduler has.</p>

<p><strong>2.</strong> Four CZ gates arranged as two layers of two, against eight Hadamards arranged as two layers of four: both have depth 2, and with \(t_{2q} = 10\) the times are 20 and 2. Depth counts layers and says nothing about what is in them, so it is a good objective only when the gates are alike. Since the whole business of compilation is to change what the gates are, that condition is rarely met — and the <code>naive/depth</code> column of Code Example 6 shows the average layer cost changing by a factor of two under optimization alone.</p>

<p><strong>3.</strong> It omits (i) unequal gate durations, the largest effect and the one Chapter 4 supplies; (ii) everything about routing — a gate between distant qubits is not one layer but a SWAP chain, and Code Example 7 turns depth 5 into depth 53 for that reason; (iii) classical latency, including measurement time, the discrimination of an analog signal into a bit, and the round trip to a controller for anything conditioned on a measurement result, which on real hardware can exceed the duration of the entire gate sequence.</p>

<p><strong>4.</strong> Depth is right when the dominant error is idling, i.e. when the circuit duration is a substantial fraction of the coherence time; the two-qubit count is right when the dominant error is the gates themselves. In Code Example 6 the crossover for this circuit sits between \(10^4\) and \(10^5\) single-qubit gate times: below that range the idle term exceeds the gate term, above it the reverse. What decides the regime is the ratio of the circuit's duration to the coherence time, so it depends on the circuit as much as on the machine — one more reason a compiler needs an objective handed to it rather than assumed.</p>

```python
c = [("h", 0), ("cx", 0, 1), ("h", 2), ("cx", 2, 3), ("z", 1), ("z", 3)]
print(circuit_depth(c, 4), [len(v) for v in circuit_layers(c, 4)])
heavy = [("cz", 0, 1), ("cz", 2, 3), ("cz", 0, 1), ("cz", 2, 3)]
light = [("h", q) for q in range(4)] * 2
for label, circ in [("2q only", heavy), ("1q only", light)]:
    print(f"{label}: depth {circuit_depth(circ, 4)}, "
          f"time {wall_clock(circ, 4, 10.0):.0f}")
# 3 [2, 2, 2]
# 2q only: depth 2, time 20
# 1q only: depth 2, time 2
```

</details>

#### Exercise 3: The Global-Phase Rule, and the Repair

Code Example 4 showed that $R_z(\pi)$ and $Z$ are equivalent while $\mathrm{C}$-$R_z(\pi)$ and CZ are not.

  1. Find $\varphi$ with $R_z(\pi) = e^{i\varphi} Z$, both by hand and with `best_global_phase`.
  2. Write down the diagonal of $\mathrm{C}$-$R_z(\pi)$ and of CZ, and identify the extra factor as a gate acting on the *control* qubit alone.
  3. Using only gates in the IR, build a circuit equal to $\mathrm{C}$-$R_z(\pi)$ *exactly* — not up to a phase. You will need $S^{\dagger}$, which is not in the gate set; express it with the gates that are.
  4. State the general rule for when a compiler may drop a global phase, and give one example of a pass that must therefore carry the phase as data.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(R_z(\pi) = \mathrm{diag}(e^{-i\pi/2}, e^{i\pi/2}) = e^{-i\pi/2}\,\mathrm{diag}(1, e^{i\pi}) = -i\,Z\), so \(\varphi = -\pi/2\). <code>best_global_phase</code> returns \(-0.500\pi\), and the general statement is \(R_z(\theta) = e^{-i\theta/2}\,\mathrm{diag}(1, e^{i\theta})\): the rotation convention distributes the phase symmetrically about zero, while the phase-gate convention puts it all on the \(\lvert 1 \rangle\) branch.</p>

<p><strong>2.</strong> \(\mathrm{C}\text{-}R_z(\pi) = \mathrm{diag}(1, 1, -i, i)\) and \(\mathrm{CZ} = \mathrm{diag}(1,1,1,-1)\). Their ratio, entry by entry, is \((1, 1, -i, -i)\) — a factor that depends only on the control qubit, i.e. \(\mathrm{diag}(1, -i) \otimes I = S^{\dagger} \otimes I\). So \(\mathrm{C}\text{-}R_z(\pi) = \mathrm{CZ}\cdot(S^{\dagger}\otimes I)\), and the phase discarded when \(R_z(\pi)\) was identified with \(Z\) has reappeared as a real gate on the control.</p>

<p><strong>3.</strong> \(S^{\dagger} = S^3\), since \(S^4 = I\): three <code>s</code> gates in a row. The circuit <code>[("s", 0), ("s", 0), ("s", 0), ("cz", 0, 1)]</code> matches \(\mathrm{C}\text{-}R_z(\pi)\) at \(6\times 10^{-17}\) with no phase removal at all — the two matrices are equal entry by entry. The order does not matter here because both factors are diagonal.</p>

<p><strong>4.</strong> The rule: a global phase may be dropped only when the rewritten fragment will never be conditioned on another qubit, and never be used to build a controlled version of itself. A pass that must carry the phase is any synthesis pass whose output will be controlled — for instance a routine that decomposes an arbitrary \(U(2)\) into \(R_z R_y R_z\) and is then asked for controlled-\(U\), which is the exact situation Chapter 2's ZYZ decomposition is in, and the reason its implementation returns four numbers rather than three. The same applies to Chapter 5's error mitigation, where circuits are compared coherently.</p>

```python
print(f"{np.angle(best_global_phase(rz(PI), Z)) / PI:+.3f}")
print(f"{phase_free_error(controlled(rz(PI)), CZ4):.3f}")
fix = [("s", 0), ("s", 0), ("s", 0), ("cz", 0, 1)]     # S^3 = S^dagger
print(f"{np.max(np.abs(controlled(rz(PI)) - unitary_of(fix, 2))):.1e}")
# -0.500
# 0.765
# 6.1e-17
```

</details>

#### Exercise 4: What Verification Costs, and What Random Testing Buys

The exhaustive check runs the circuit $2^n$ times. Consider replacing it with a check on random input states.

  1. Write `equivalent_random(a, b, n, trials)` that draws Haar-random states, runs both circuits, and compares the overlap magnitude $\lvert \langle \psi_b \vert \psi_a \rangle \rvert$ against 1. Why the overlap magnitude and not the state difference?
  2. Argue that if $U(a) \ne e^{i\varphi} U(b)$, a single Haar-random state distinguishes them with probability 1. Where does that argument break down in floating-point arithmetic?
  3. Run it against the buggy pass of Code Example 5 and compare its verdict with the exhaustive check on several hundred circuits.
  4. At which $n$ would you switch, and what would you do above $n = 30$ where neither method is available?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> The overlap magnitude, because the equivalence relation quotients out the global phase: two equivalent circuits give output states differing by \(e^{i\varphi}\), so their difference is not small but the modulus of their overlap is exactly 1. Comparing \(\lVert \psi_a - \psi_b \rVert\) would reject every correct phase-sloppy rewrite, which is the same mistake the naive matrix comparison makes in Code Example 4.</p>

<p><strong>2.</strong> If \(W = U(b)^{\dagger}U(a)\) is not a multiple of the identity, then \(\lvert\langle\psi\rvert W\lvert\psi\rangle\rvert = 1\) forces \(\lvert\psi\rangle\) to be an eigenvector of \(W\), and the eigenvectors of a matrix that is not a multiple of the identity form a set of measure zero in state space. A Haar-random state misses it with probability 1. Two things break in practice: a rewrite that is wrong by a tiny amount — a mis-synthesized angle, say — gives an overlap deficit of order \(\varepsilon^2\), which can hide under the tolerance even though the circuits differ; and a state that happens to be nearly an eigenvector gives a small deficit for a large error, so the test is quantitatively unreliable even when it is qualitatively right. Both argue for using more than one state when the answer matters, and for reporting the deficit rather than a boolean.</p>

<p><strong>3.</strong> On 600 random circuits the buggy pass fired 26 times, and a single random state agreed with the exhaustive verdict on all 26, with no false passes, at one simulation instead of eight. The cost ratio grows as \(2^n\), so this is the method to use as soon as \(n\) is beyond a handful.</p>

<p><strong>4.</strong> Exhaustive checking is comfortable to \(n \approx 12\) and painful beyond \(n \approx 16\); random states remain cheap to wherever the simulator itself stops, around \(n = 26\) to \(30\) for a laptop. Above that no state-vector method is available and verification must become structural: prove each rewrite rule once on the two or three qubits it touches, prove that the pass applies it only where its precondition holds, and test the composition on small instances of the same circuit family. This is what production compilers do, and it is why their bugs are almost never wrong rules — they are correct rules fired in the wrong context, exactly like the pass in Code Example 5.</p>

```python
def equivalent_random(a, b, n, trials=1, rng=None, atol=1e-10):
    """Compare two circuits on random input states instead of all 2^n of them."""
    rng = np.random.default_rng() if rng is None else rng
    for _ in range(trials):
        psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        psi /= np.linalg.norm(psi)
        overlap = abs(np.vdot(run_circuit(b, n, psi0=psi),
                             run_circuit(a, n, psi0=psi)))
        if abs(overlap - 1.0) > atol:
            return False
    return True


rng = np.random.default_rng(11)
fired = agree = missed = 0
for _ in range(600):
    c = random_circuit(3, 12, rng)
    b = buggy_commute(c)
    if b == c:
        continue
    fired += 1
    exact = phase_free_error(unitary_of(c, 3), unitary_of(b, 3)) <= 1e-10
    single = equivalent_random(c, b, 3, trials=1, rng=rng)
    agree += int(exact == single)
    missed += int(single and not exact)
print(f"fired {fired}, one random state agreed with the exhaustive test "
      f"{agree} times, missed {missed}")
# fired 26, one random state agreed with the exhaustive test 26 times, missed 0
```

</details>

#### Exercise 5: Reading the Stack

Six descriptions of compiler passes are given below, with no framework names attached. For each one, name the layer of §1.1 it belongs to, state what hardware information it needs, and say whether its output could be cached and reused on a different machine.

  1. "Combines runs of single-qubit gates on the same qubit into a single $U(2)$ and re-synthesizes it."
  2. "Chooses which physical qubits the program's qubits start on, minimizing the estimated number of SWAPs."
  3. "Replaces each two-qubit gate that is not in the target's instruction set with an equivalent sequence that is."
  4. "Adds a delay instruction so that a gate begins at a time that is a multiple of the waveform sample period."
  5. "Multiplies the measured bit-string distribution by the inverse of a matrix obtained by preparing and measuring each basis state."
  6. "Replaces each two-qubit gate with three copies of itself, then extrapolates the result to zero copies."

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Layer 3, optimization. It needs no hardware information at all if it re-synthesizes into the authoring set, and only the native gate set if it synthesizes directly into that — which is what a real implementation does, making it a layer 3/5 hybrid. Fully cacheable in the first form, not in the second. This is Chapter 2's ZYZ pass.</p>

<p><strong>2.</strong> Layer 4, placement. It needs the coupling graph, and in a good implementation also the per-pair error rates so that it can prefer good links. Not reusable across machines; not even reusable across recalibrations if it uses error rates. Chapter 3.</p>

<p><strong>3.</strong> Layer 5, gate synthesis. It needs the native gate set and nothing else — notably not the coupling graph, which is why this pass runs after routing in most stacks. Cacheable per gate set rather than per machine, which is why the same translation tables are shared between devices of the same family. Chapter 2 supplies the general version via KAK decomposition.</p>

<p><strong>4.</strong> Layer 6, scheduling. It needs the sample period of the control electronics, and in general also the gate durations. Not reusable at all. Chapter 4.</p>

<p><strong>5.</strong> Layer 7, readout mitigation. It needs a measured confusion matrix, which is calibration data and goes stale; it also needs \(2^n\) preparations to obtain in full, which is the practical problem Chapter 5 addresses. Not reusable.</p>

<p><strong>6.</strong> Layer 7, and it is the odd one out in an instructive way: zero-noise extrapolation is the only entry here that <em>deliberately breaks</em> the equivalence relation of §1.3. Folding a gate into three copies preserves the unitary exactly — \(G G^{\dagger} G = G\) — but the whole point is that it does not preserve the noise, and the extra copies are inserted precisely so that the noisy circuit is different. It needs no hardware information to construct, only the assumption that noise scales with gate count. Cacheable, and Chapter 5 implements it.</p>

<p>The pattern worth extracting: perishability increases down the list. A layer 3 pass can be written once; a layer 4 pass per machine; a layer 6 pass per calibration. That ordering is the same one that produced the table in §1.1, and it is the most reliable way to guess which layer an unfamiliar pass belongs to.</p>

</details>

* * *

## Summary

### Key Takeaways

**1\. Seven layers, sorted by how perishable their input is**

  * Algorithm, circuit IR, optimization, placement and routing, gate synthesis, pulses and scheduling, readout and mitigation. The criterion that fixes the boundaries is what each pass must know about the machine: nothing, the coupling graph, the native gate set, or today's calibration.
  * Layers 2 to 5 all consume and produce the same object — a gate list — which is why passes compose and can be tested one at a time. Layer 6 is where the type changes, and that is why pulse programming feels different.

**2\. The IR is data, and its gate set is the layer boundary**

  * A circuit is a list of tuples; a gate is a name, an optional angle, and some integer wires, big-endian. Four functions — `run_circuit`, `circuit_depth`, `gate_counts`, and the checker — are the whole interface.
  * An IR is defined by what it refuses to represent. Measurement, timing, allocation and annotations are all absent here, and each is added back by a later chapter.
  * Three gate sets appear: the authoring set, chosen for convenience; the native set, chosen by physics; Clifford$+T$, chosen by the error-correcting code. Translating between the first two costs a constant factor; translating into the third costs a factor growing as $\log(1/\varepsilon)$.

**3\. Compilation is meaning-preserving rewriting, and the meaning is checkable**

  * Three correctness relations: exact up to a global phase (optimization, synthesis), exact up to a qubit permutation (routing), approximate within $\varepsilon$ (discrete synthesis, pulses), with $\varepsilon$ adding up over a circuit.
  * The optimal phase comes from the Hilbert-Schmidt overlap, and $\lvert\mathrm{tr}(V^{\dagger}U)\rvert = 2^n$ is itself a complete test of equivalence.
  * State equivalence is weaker than unitary equivalence: the two GHZ constructions agree exactly on $\lvert 0\ldots 0\rangle$ and differ by $0.707$ as unitaries. A pass justified by the weaker relation and used at the stronger one is a bug that hides.

**4\. A global phase is not global inside a controlled block**

  * $R_z(\pi) = -iZ$ is equivalent, and $\mathrm{C}$-$R_z(\pi) \ne \mathrm{CZ}$ by $0.765$: the unobservable factor becomes an observable relative phase. Every translation rule of Code Example 7 is phase-sloppy, which is why the rule matters in practice rather than in principle.
  * The repair, worked out in Exercise 3, is that the discarded phase reappears as an honest gate on the control qubit: $\mathrm{C}$-$R_z(\pi) = \mathrm{CZ}\cdot(S^{\dagger}\otimes I)$.

**5\. Verification is cheap at small $n$ and must change character above it**

  * Exhaustive checking costs $2^n$ simulations: comfortable to $n \approx 12$, out of reach at $n = 20$. One Haar-random input state distinguishes inequivalent circuits with probability 1, and agreed with the exhaustive verdict on all 26 circuits where the buggy pass fired.
  * Random testing is not proof: the buggy pass fired on only 19 of 500 circuits, and on one of those it produced an equivalent circuit by coincidence. Above the simulable range verification becomes structural, and compiler bugs become correct rules applied in the wrong context.

**6\. "Optimize" is meaningless without an objective**

  * Total gates, two-qubit gates, depth, wall clock and $T$ count disagree. The same rewrite was worth $60\%$ of the runtime on a machine with fast two-qubit gates and $1.8\%$ on one with slow ones.
  * The two-qubit share of the gate error was $83\%$ at equal error rates and $99.8\%$ at a hundredfold ratio, which is why compiler cost functions are dominated by the two-qubit count.
  * Depth is the right objective when idling dominates and the wrong one when gates do; the crossover for the example circuit sat between coherence times of $10^4$ and $10^5$ gate times.

**Practical implications**

  * Write the equivalence check before writing the pass. Every rewriting rule in the rest of this course arrives with one, and it costs a few lines.
  * Never compare a computed float against a power of ten with `==` or a bare inequality: $100 \times 10^{-5}$ accumulated in a loop is not `1e-3`. Use a relative tolerance.
  * Quote a two-qubit gate count only after saying which gate set and which connectivity it assumes; routing a five-qubit circuit onto a line multiplied it by thirteen. When a transpiled circuit is unexpectedly large, ask which layer grew it, because routing and synthesis grow circuits for different reasons and are fixed by different means.

### Where This Leads

The pipeline of Code Example 7 is the course in miniature, and every stage of it is a stub. Chapter 2 replaces the optimizer: the peephole rules here cancel only adjacent identical gates, and the real thing needs commutation rules, single-qubit synthesis by Euler decomposition, two-qubit synthesis with the correct CNOT count, and an honest account of why the $T$ gate is the expensive one. That chapter also explains the most conspicuous non-result in this one — why the final `optimize` stage of the pipeline found nothing to remove after routing, and what a pass has to know in order to find it.

[← Series Top](<index.html>) [Chapter 2: Circuit Optimization and Gate Synthesis →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The gate durations, error rates and coherence times in this course are dimensionless parameters swept over decades to expose scaling and constant factors. They are not device specifications, not measurements, and not predictions about any machine, and the layer correspondence of §1.4 describes framework architecture in general terms rather than any specific product or version.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
