---
title: "Chapter 1: From Physical Qubit to Quantum Computer"
chapter_title: "Chapter 1: From Physical Qubit to Quantum Computer"
subtitle: "What Any Physical System Must Provide, and the Trade-Off Every Platform Must Face"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/OfXiRIHny1U"
    title="QC Hardware Ch.1: From Physical Qubit to Quantum Computer"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-computing-hardware/chapter-1.html>) | Last sync: 2026-08-16

[Quantum Computing Dojo](<../index.html>) > [Quantum Computing Hardware](<index.html>) > Chapter 1

The *Introduction to Quantum Computing* series treated the qubit as a mathematical object: a two-dimensional vector that gates rotate and measurement collapses. This series asks the harder question. What does it take to build one out of matter, keep it alive, talk to it, and read it back — a few million times per second, on thousands of copies at once?

This chapter deliberately contains almost no device specifications. Hardware numbers age badly, and a table of qubit counts written today would mislead you within a year. What does not age is the physics: the requirements any candidate system must satisfy, the tension those requirements create, and the figures of merit that let you compare platforms honestly. Everything in the chapters that follow is a different engineering answer to the same physical problem posed here.

## 1.1 What Makes a Physical System a Qubit

A qubit is not a thing you find; it is a role a physical system can be made to play. Almost anything with two distinguishable quantum states is a candidate — an electron spin pointing up or down, an atom in its ground or excited state, a photon in one path or another, a superconducting circuit oscillating with one energy quantum or none.

The candidates are plentiful. The ones that survive contact with engineering are not.

### 📚 The DiVincenzo Criteria

In 2000, David DiVincenzo published a paper titled "The Physical Implementation of Quantum Computation" that set out what a physical system must provide before it can be called a quantum computer. Twenty-five years later, his list remains the standard checklist, and every platform in this series is best understood as a particular set of answers to it.

**The five computing criteria:**

  1. **A scalable physical system with well-characterized qubits.** You need many copies of the same two-level system, and you need to know its properties precisely — its energy splitting, how it couples to its neighbours, what states lie outside the computational subspace. "Well-characterized" is doing real work here: a system whose parameters drift unpredictably cannot be controlled, however clean its physics looks in isolation.
  2. **The ability to initialize the qubits to a simple fiducial state.** Every computation begins from a known starting point, conventionally \\(|00\ldots0\rangle\\). If initialization is imperfect, every subsequent operation inherits the error, and no amount of good gates will recover it.
  3. **Long relevant decoherence times, much longer than the gate operation time.** Note the phrasing carefully. DiVincenzo did not ask for long coherence times in absolute terms — he asked for them *relative to the gate time*. This ratio is the single most important number in hardware comparison, and Section 1.3 makes it quantitative.
  4. **A universal set of quantum gates.** The hardware must implement a set of operations from which any unitary can be built to arbitrary accuracy. In practice this means arbitrary single-qubit rotations plus one entangling two-qubit gate.
  5. **A qubit-specific measurement capability.** You must be able to read out an individual qubit in the computational basis, reliably, and ideally without destroying its neighbours.

**The two networkability criteria**, which DiVincenzo added for quantum communication rather than for computation alone:

  6. **The ability to interconvert stationary and flying qubits.** Stationary qubits store information; flying qubits — almost always photons — carry it. Linking two processors requires translating between them.
  7. **The ability to transmit flying qubits faithfully between specified locations.** Sending the photon is not enough; it must arrive with its quantum state intact.

> **Why criteria 6 and 7 matter more each year**
>
> They were once treated as the quantum-networking appendix to a computing list. As it becomes clear that a single monolithic chip cannot be scaled indefinitely, modular architectures — several processors linked by photonic channels — have moved from a curiosity toward a serious plan. The last two criteria are the ones that make such an architecture possible.

## 1.2 The Tension at the Heart of All Quantum Hardware

Here is the difficulty that shapes every design decision in this series.

**Criterion 3 asks the qubit to be isolated.** Decoherence is the loss of quantum information into the environment. To keep coherence long, the qubit must be shielded from everything — stray electromagnetic fields, mechanical vibration, thermal photons, fluctuating charges in the surrounding material. The ideal qubit talks to nothing.

**Criteria 4 and 5 ask the qubit to be controllable.** Gates are applied by coupling the qubit deliberately to a control field. Readout requires coupling it to a measuring apparatus. Two-qubit gates require coupling it to another qubit. The ideal qubit talks to everything you choose, promptly and strongly.

These two demands pull in opposite directions, and they pull on the *same* physical coupling. A qubit that couples strongly enough to a control line for fast gates also couples to whatever noise that line carries. A qubit isolated well enough to stay coherent for a long time is, for the same reason, slow and difficult to address.

### 📚 There Is No Way Around This

It is tempting to look for a system that is both perfectly isolated and perfectly controllable. No such system exists, and the reason is not a lack of cleverness. Coupling is a property of the system, not of your intentions: the same matrix element that lets your pulse rotate the state lets an unwanted field rotate it too. What engineering *can* do is make the coupling **selective** — strong at the frequency you drive, weak everywhere else — and this is precisely what the rest of this series describes.

Every platform sits somewhere on the resulting spectrum:

  * **Strongly coupled, fast, noisy.** Engineered circuits interact vigorously with their control electronics. Gates are quick; coherence is comparatively short.
  * **Weakly coupled, slow, quiet.** Isolated atomic systems barely notice their surroundings. Coherence is long; gates take much more time.

Neither end is automatically better, because what matters is not the gate speed and not the coherence time, but their ratio.

## 1.3 The Metrics That Actually Matter

Four quantities describe the health of a qubit. Understand these and you can read a hardware paper; memorize device specifications instead and you will be out of date by the next conference.

### 📚 T₁ — Energy Relaxation

\\(T_1\\) is the timescale on which an excited qubit loses its energy to the environment and decays toward \\(|0\rangle\\), like an excited atom emitting a photon. Starting from \\(|1\rangle\\), the probability of still finding the qubit excited falls exponentially:

\\[ P_1(t) = e^{-t/T_1} \\]

This is an *irreversible* loss. The energy has gone into the environment and is not coming back.

### 📚 T₂ — Dephasing

\\(T_2\\) is the timescale on which the *relative phase* between \\(|0\rangle\\) and \\(|1\rangle\\) becomes randomized. A state \\(\frac{1}{\sqrt{2}}(|0\rangle + e^{i\phi}|1\rangle)\\) still has the right populations after dephasing — a measurement in the computational basis looks completely normal — but \\(\phi\\) has drifted to an unknown value, and the interference that quantum algorithms depend on is gone.

Dephasing is the more insidious of the two, because it destroys the resource without destroying the energy.

**The relationship between the two is a strict inequality:**

\\[ \frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi} \qquad \Longrightarrow \qquad T_2 \leq 2T_1 \\]

Here \\(T_\phi\\) is the **pure dephasing** time, collecting every phase-randomizing mechanism that is not energy loss. The inequality has a clean physical reading: relaxation destroys phase coherence too, since a qubit that has decayed to \\(|0\rangle\\) no longer holds any phase relationship at all. So even a qubit with *zero* pure dephasing cannot have \\(T_2\\) longer than \\(2T_1\\). Relaxation sets a ceiling; pure dephasing pushes you below it. We will confirm this numerically in Section 1.6.

### 📚 Gate Fidelity

**Fidelity** measures how close the operation you performed is to the operation you intended, on a scale where 1 is perfect. An error rate of \\(\epsilon = 1 - F\\) accumulates across a circuit: as the *Introduction* series showed, a circuit of \\(m\\) gates succeeds with probability roughly \\(e^{-m\epsilon}\\), so the affordable circuit depth is of order \\(1/\epsilon\\).

Two cautions when reading fidelity claims. First, single-qubit and two-qubit gate fidelities are very different quantities — two-qubit gates are harder by a wide margin, and a headline figure that does not say which it quotes is not telling you much. Second, fidelities measured on isolated pairs can degrade when the whole device runs at once, because neighbouring operations interfere; this is why *simultaneous* benchmarking is reported separately.

### 📚 Gate Time, and the Figure of Merit That Combines Everything

**Gate time** is how long one operation takes. On its own it means nothing — a fast gate on a qubit that decoheres faster still is useless.

The honest comparison is the ratio:

\\[ N_{\text{gates}} \sim \frac{T_2}{t_{\text{gate}}} \\]

**How many gates fit inside a coherence time?** This is DiVincenzo's third criterion made quantitative, and it is the number to look for. It explains why a platform with microsecond-scale coherence and nanosecond-scale gates can compete with a platform whose coherence is a thousand times longer but whose gates are a thousand times slower — they land in the same place.

> **What "qubit count" does and does not tell you**
>
> A device's qubit count is the most quoted and least informative hardware number. It says nothing about \\(T_2/t_{\text{gate}}\\), nothing about two-qubit fidelity, nothing about which qubits can talk to which, and nothing about whether the qubits are physical or error-corrected. A machine with many poor qubits runs strictly shallower circuits than a machine with fewer good ones. When you encounter a qubit count, treat it as one axis of a multi-dimensional object, not as a score.

## 1.4 The Hardware Stack

A quantum computer is not a chip. It is a stack of four layers, each translating the layer above into the language of the layer below, and every layer is a research problem.

    
    
    ```mermaid
    flowchart TD
        A[Compiler and software<br/>circuits, routing, error correction]
        B[Digital electronics<br/>sequencing, timing, feedback]
        C[Analog control<br/>shaped pulses, amplifiers, filters]
        D[Quantum layer<br/>the physical qubits themselves]
        A --> B --> C --> D
        D -.measurement results.-> C
        C -.digitized signal.-> B
        B -.outcomes.-> A
        style A fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
        style B fill:#00bcd4,stroke:#764ba2,stroke-width:2px,color:#fff
        style C fill:#7c4dff,stroke:#764ba2,stroke-width:2px,color:#fff
        style D fill:#f57c00,stroke:#764ba2,stroke-width:2px,color:#fff
    ```

**The quantum layer** holds the physical qubits and the structures that couple them: the chip, the trap, the optical lattice, whatever the platform provides. Its job is to keep well-characterized two-level systems coherent and addressable. Everything above exists to serve it, and every imperfection here propagates upward.

**The analog control layer** turns an abstract gate into a physical stimulus — a shaped microwave pulse, a laser pulse, a voltage waveform. Its job is precision in amplitude, frequency, phase, and timing, because a gate *is* its pulse: a rotation angle is set by a pulse area, so an amplitude error is a rotation error. This layer also carries the readout signal back out, usually as a tiny voltage that must be amplified before anything digital can see it.

**The digital electronics layer** sequences those pulses, timestamps them, digitizes the returning signals, and — increasingly the demanding part — closes feedback loops fast enough to matter. Error correction requires measuring syndromes and reacting *before* the qubit decoheres, which puts a hard real-time deadline on classical hardware sitting outside the refrigerator.

**The compiler and software layer** takes the circuit a user wrote and makes it executable on this specific device: decomposing gates into the native set, mapping logical qubits onto physical ones, inserting routing operations where the connectivity demands them, calibrating and recalibrating parameters that drift. A well-written compiler can change the effective performance of a device substantially without touching the physics.

The lesson of the stack is that hardware progress is rarely one breakthrough. It is usually a coordinated improvement across four layers that must be co-designed, which is also why platform comparisons are difficult: you are comparing whole stacks, not qubits.

## 1.5 Connectivity, and Why It Costs You

An algorithm designer draws a two-qubit gate between any pair of qubits. A chip designer must decide, in advance and in metal, which pairs are physically coupled.

**Connectivity** is the graph of which qubits can interact directly. Platforms differ enormously here, and the difference is not cosmetic.

  * **Nearest-neighbour connectivity**: qubits are laid out on a line or a two-dimensional grid and couple only to their immediate neighbours. This is typical of solid-state chips, where the coupling is a physical structure that must be fabricated.
  * **All-to-all connectivity**: every qubit can interact with every other, usually because the coupling is mediated by a shared mode — a collective vibration, a common optical field — rather than by dedicated wiring.

When your algorithm needs a gate between two qubits that are not connected, the compiler must move the information. The standard tool is the **SWAP** gate, which exchanges the states of two neighbouring qubits. A SWAP is not free: in the usual decomposition it costs **three CNOT gates**, each with its own error rate and duration.

The overhead scales with distance. On a linear chain, bringing two qubits a distance \\(d\\) apart into contact takes on the order of \\(d\\) SWAPs. On a two-dimensional grid of \\(n\\) qubits, a typical pair sits about \\(\sqrt{n}\\) steps apart, so routing cost grows as the device grows — precisely when you can least afford it.

This connects back to the figure of merit. Every inserted SWAP consumes part of the \\(T_2/t_{\text{gate}}\\) budget, which is why an algorithm's *effective* depth on real hardware can be several times its depth on paper. A platform with all-to-all connectivity and slow gates may beat a fast nearest-neighbour platform on a routing-heavy circuit, and lose on a local one. There is no universal winner, only a match or mismatch between circuit structure and device topology.

There is one consolation. **Error-correcting codes designed for near-term hardware, such as the surface code, use only nearest-neighbour checks by construction** — the layout was chosen to match the constraint rather than to fight it. Connectivity limits hurt algorithms far more than they hurt error correction.

## 1.6 Hands-On: Watching a Qubit Forget

Let us make decoherence concrete. The code below needs only NumPy. It does three things: it shows the exponential energy decay behind \\(T_1\\), it simulates a Ramsey-style dephasing measurement of \\(T_2\\), and it demonstrates the inequality \\(T_2 \leq 2T_1\\) numerically.

The timescales are chosen for teaching. They are round numbers, not measurements of any device.

```python
import numpy as np

# ---------------------------------------------------------------
# 1. Two illustrative timescales (microseconds).
#    These are teaching numbers, not measurements of any device.
# ---------------------------------------------------------------
T1 = 60.0        # energy relaxation time
T_phi = 40.0     # pure dephasing time (everything except relaxation)

# The two channels add as RATES, not as times:
#     1/T2 = 1/(2*T1) + 1/T_phi
T2 = 1.0 / (1.0 / (2.0 * T1) + 1.0 / T_phi)
print("Timescales (microseconds)")
print(f"  T1              = {T1:8.3f}")
print(f"  T_phi           = {T_phi:8.3f}")
print(f"  T2              = {T2:8.3f}")
print(f"  2*T1 (ceiling)  = {2 * T1:8.3f}")
print(f"  T2 <= 2*T1 ?      {T2 <= 2 * T1}")
print()

# The ceiling is reached only when pure dephasing is switched off.
for tphi in [10.0, 40.0, 1.0e3, 1.0e6]:
    t2 = 1.0 / (1.0 / (2.0 * T1) + 1.0 / tphi)
    print(f"  T_phi = {tphi:10.1f} -> T2 = {t2:8.3f}  (ratio T2/2T1 = {t2 / (2 * T1):.4f})")
print()

# ---------------------------------------------------------------
# 2. T1: the excited-state population decays exponentially.
#    Bloch vector component: r_z(t) = -1 + 2*exp(-t/T1) starting from |1>.
# ---------------------------------------------------------------
print("Excited-state population P1(t) = exp(-t/T1)")
for t in [0.0, 10.0, 30.0, 60.0, 120.0]:
    p1 = np.exp(-t / T1)
    print(f"  t = {t:6.1f} us   P1 = {p1:.6f}")
print()

# ---------------------------------------------------------------
# 3. T2: a Ramsey experiment. Put the Bloch vector on the equator,
#    let it precess at a small detuning, and watch the transverse
#    length shrink. Only the LENGTH loss is decoherence; the
#    rotation itself is harmless and can be calibrated away.
# ---------------------------------------------------------------
detuning = 0.03   # cycles per microsecond (deliberate offset in a Ramsey scan)

print("Ramsey signal  r_x(t) = exp(-t/T2) * cos(2*pi*detuning*t)")
for t in [0.0, 10.0, 20.0, 30.0, 50.0, 100.0]:
    envelope = np.exp(-t / T2)
    r_x = envelope * np.cos(2.0 * np.pi * detuning * t)
    r_y = envelope * np.sin(2.0 * np.pi * detuning * t)
    length = np.hypot(r_x, r_y)
    print(f"  t = {t:6.1f} us   r_x = {r_x:+.6f}   |r_xy| = {length:.6f}")
print()

# ---------------------------------------------------------------
# 4. The figure of merit that actually matters:
#    how many gates fit inside a coherence time?
# ---------------------------------------------------------------
print("Gates within one coherence time  N = T2 / t_gate")
for t_gate, label in [(0.02, "fast gate"), (0.20, "medium gate"), (2.00, "slow gate")]:
    n_gates = T2 / t_gate
    print(f"  t_gate = {t_gate:5.2f} us ({label:11s}) -> N = {n_gates:10.1f}")
print()

# A platform with 100x longer coherence but 1000x slower gates is WORSE.
slow_T2, slow_gate = 100 * T2, 1000 * 0.02
print("Comparing two hypothetical platforms:")
print(f"  A: T2 = {T2:8.2f} us, t_gate = {0.02:7.2f} us -> N = {T2 / 0.02:10.1f}")
print(f"  B: T2 = {slow_T2:8.2f} us, t_gate = {slow_gate:7.2f} us -> N = {slow_T2 / slow_gate:10.1f}")
```

**Output:**

```
Timescales (microseconds)
  T1              =   60.000
  T_phi           =   40.000
  T2              =   30.000
  2*T1 (ceiling)  =  120.000
  T2 <= 2*T1 ?      True

  T_phi =       10.0 -> T2 =    9.231  (ratio T2/2T1 = 0.0769)
  T_phi =       40.0 -> T2 =   30.000  (ratio T2/2T1 = 0.2500)
  T_phi =     1000.0 -> T2 =  107.143  (ratio T2/2T1 = 0.8929)
  T_phi =  1000000.0 -> T2 =  119.986  (ratio T2/2T1 = 0.9999)

Excited-state population P1(t) = exp(-t/T1)
  t =    0.0 us   P1 = 1.000000
  t =   10.0 us   P1 = 0.846482
  t =   30.0 us   P1 = 0.606531
  t =   60.0 us   P1 = 0.367879
  t =  120.0 us   P1 = 0.135335

Ramsey signal  r_x(t) = exp(-t/T2) * cos(2*pi*detuning*t)
  t =    0.0 us   r_x = +1.000000   |r_xy| = 1.000000
  t =   10.0 us   r_x = -0.221420   |r_xy| = 0.716531
  t =   20.0 us   r_x = -0.415363   |r_xy| = 0.513417
  t =   30.0 us   r_x = +0.297621   |r_xy| = 0.367879
  t =   50.0 us   r_x = -0.188876   |r_xy| = 0.188876
  t =  100.0 us   r_x = +0.035674   |r_xy| = 0.035674

Gates within one coherence time  N = T2 / t_gate
  t_gate =  0.02 us (fast gate  ) -> N =     1500.0
  t_gate =  0.20 us (medium gate) -> N =      150.0
  t_gate =  2.00 us (slow gate  ) -> N =       15.0

Comparing two hypothetical platforms:
  A: T2 =    30.00 us, t_gate =    0.02 us -> N =     1500.0
  B: T2 =  3000.00 us, t_gate =   20.00 us -> N =      150.0
```

**Reading the result.** Four points deserve attention.

  * **Rates add, times do not.** With \\(T_1 = 60\\) and \\(T_\phi = 40\\), the resulting \\(T_2 = 30\\) is shorter than either. Decoherence channels combine by adding their rates, so the fastest mechanism dominates — which is why hardware work so often consists of hunting down the single worst noise source.
  * **The ceiling is real and unreachable.** As \\(T_\phi\\) grows, \\(T_2\\) climbs toward \\(2T_1 = 120\\) but never passes it. The last row reaches \\(0.9999\\) of the ceiling with pure dephasing suppressed a thousandfold. This is the inequality \\(T_2 \leq 2T_1\\) as a numerical fact rather than an assertion.
  * **The Ramsey signal separates two effects.** \\(r_x\\) wobbles because the qubit precesses at the detuning frequency; the *envelope* \\(|r_{xy}|\\) shrinks because of dephasing. Only the second is decoherence. In a real experiment the oscillation is a feature — its frequency calibrates the qubit, and the decay of its envelope measures \\(T_2\\).
  * **Platform A beats platform B.** Platform B has one hundred times the coherence and one thousand times the gate duration, and it fits *ten times fewer* gates into a coherence time. If you take one number away from this chapter, take \\(T_2/t_{\text{gate}}\\).

Try changing `T_phi` to a very large value and re-running: you will see \\(T_2\\) saturate at exactly twice \\(T_1\\), no matter how quiet you make the environment.

### 🎯 Exercise Problems

  1. **The inequality, analytically.** Starting from \\(1/T_2 = 1/(2T_1) + 1/T_\phi\\) with \\(T_\phi > 0\\), prove that \\(T_2 \leq 2T_1\\), and state the physical condition under which equality holds.
  2. **Budgeting a circuit.** A device has \\(T_2 = 30\\,\mu\text{s}\\) and a two-qubit gate time of \\(200\\,\text{ns}\\). Ignoring gate errors, roughly how many sequential two-qubit gates fit inside one coherence time? Now suppose the compiler must insert two SWAPs for every algorithmic gate — what happens to your answer?
  3. **Reading a criterion.** For each of DiVincenzo's five computing criteria, name one concrete way it could fail in practice, and say which layer of the stack in Section 1.4 would be responsible for fixing it.
  4. **Connectivity cost.** On a \\(7 \times 7\\) grid of qubits with nearest-neighbour coupling, estimate the number of SWAP gates needed to bring two opposite-corner qubits together. Convert your answer into CNOT gates.
  5. **Spotting a bad comparison.** You read that Device X has more qubits than Device Y and is therefore more powerful. List three specific questions you would need answered before accepting that conclusion.

## Summary

This chapter established the physical requirements behind every quantum computer, independent of which platform builds it. The **DiVincenzo criteria (2000)** define the checklist: scalable well-characterized qubits, reliable initialization, decoherence times long *relative to gate times*, a universal gate set, and qubit-specific measurement — plus two further criteria, stationary-to-flying qubit conversion and faithful transmission, that make modular and networked architectures possible. We identified the **central tension**: coherence demands isolation while gates and readout demand coupling, and both act through the same physical channel, so every platform must choose a point on that spectrum rather than escape it. We defined the metrics that matter — \\(T_1\\) for **energy relaxation**, \\(T_2\\) for **dephasing** with the strict ceiling \\(T_2 \leq 2T_1\\), **gate fidelity**, and **gate time** — and argued that the honest figure of merit is the ratio \\(T_2/t_{\text{gate}}\\), *how many gates fit inside a coherence time*, rather than a qubit count. We walked the **four-layer stack** from the quantum layer through analog control and digital electronics to the compiler, noting that progress requires all four to advance together. Finally we examined **connectivity**, where limited coupling forces the compiler to insert SWAP gates at three CNOTs each, with routing cost growing as the device grows. Our NumPy simulation made the decoherence picture concrete, showing rates adding rather than times, the \\(2T_1\\) ceiling approached but never crossed, and a fast short-lived qubit outperforming a slow long-lived one.

In the next chapter, we turn to the first and most industrially developed platform: superconducting circuits, where a Josephson junction turns a harmonic oscillator into a usable qubit, and where the price of very fast gates is a refrigerator running near absolute zero.

[← Series Top](<index.html>) [Chapter 2: Superconducting Qubits →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
