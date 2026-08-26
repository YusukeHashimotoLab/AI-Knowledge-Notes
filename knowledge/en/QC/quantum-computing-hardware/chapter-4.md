---
title: "Chapter 4: Photonic, Spin, and Topological Platforms"
chapter_title: "Chapter 4: Photonic, Spin, and Topological Platforms"
subtitle: "Light That Will Not Interact, Silicon That Might Scale, and an Idea Still Waiting for Evidence"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/XLDmuO1geQc"
    title="QC Hardware Ch.4: Photonic, Spin, and Topological Platforms"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-computing-hardware/chapter-4.html>) | Last sync: 2026-08-16

[Quantum Computing Dojo](<../index.html>) > [Quantum Computing Hardware](<index.html>) > Chapter 4

Chapters 2 and 3 covered the platforms with the longest experimental track records. This chapter covers three that are pursuing very different bets. **Photonics** starts from a carrier that is almost immune to decoherence and struggles to make it interact. **Silicon spin qubits** start from the most powerful manufacturing base humanity has ever built and struggle to make it uniform enough. **Topological qubits** start from a beautiful theoretical idea about protecting information and struggle, so far, to produce conclusive experimental evidence at all. Reading these three honestly — including where the honest answer is "we do not yet know" — is the point of the chapter.

## 4.1 Photons: Almost No Decoherence, Almost No Interaction

A photon is an appealing qubit for a reason that is easy to state. It essentially does not decohere.

Chapter 2's superconducting circuit and Chapter 3's atoms both need heroic isolation — dilution refrigerators, ultra-high vacuum, magnetic shielding — because they sit in an environment eager to disturb them. A photon travelling through a transparent medium has almost nothing to couple to. It ignores thermal vibrations. It ignores stray magnetic fields. Optical photons carry an energy far above room-temperature thermal energy, so the surrounding warmth cannot excite the mode; photonic quantum information can therefore be manipulated at **room temperature**, unlike every other platform in this series. And photons travel at the speed of light through ordinary optical fibre, which makes them the natural carrier for sending quantum information from one place to another.

Where can the qubit live? Several encodings are used: **polarization** (horizontal versus vertical), **path** (which of two waveguides the photon is in), **time bin** (early versus late arrival). All are ordinary two-level systems, and any of them can be rotated at will with waveplates, beam splitters, and phase shifters. Single-qubit gates in photonics are, remarkably, passive optical components with no control electronics at all.

### 📚 And Now the Problem

The property that makes photons excellent carriers makes them terrible computers.

A two-qubit gate requires two qubits to *interact* — the state of one must influence the evolution of the other. Photons in a linear optical medium do not interact with each other at all. Two light beams cross and pass through each other unchanged; this is precisely why optics is linear and why we can see through a room criss-crossed by light. There is no photon-photon force to build a controlled-NOT from.

One might try to induce an interaction using a strongly nonlinear optical material, where the presence of one photon changes the refractive index seen by another. Such materials exist, but the nonlinearity available at the single-photon level in conventional media is far too weak to produce the large conditional phase a gate requires, and pushing it up generally brings absorption and noise along with it. This is the central difficulty of photonic quantum computing, and everything in the next two sections is a way around it.

## 4.2 Measurement as the Missing Nonlinearity

In 2001, Emanuel Knill, Raymond Laflamme, and Gerard Milburn published a result that reframed the whole problem. Their scheme — universally referred to as **KLM** — showed that efficient universal quantum computation is possible using *only* linear optics, single-photon sources, and photon detectors. No nonlinear material is required.

The insight is that **measurement is itself a nonlinear operation**. Beam splitters and phase shifters evolve the photons linearly, but detecting a photon collapses the state, and collapse does not act linearly on amplitudes. So: send your computational photons through a linear optical network together with some extra **ancilla** photons, then measure the ancillas. Conditioned on a particular detection pattern, the surviving photons have undergone an effective nonlinear transformation — a genuine entangling gate.

The catch is in the words "conditioned on." The desired detection pattern occurs only some of the time. A KLM-style gate is therefore **probabilistic**: it announces its own success or failure through the detector outcomes. This is better than it sounds, because a *heralded* failure is far more benign than a silent error — you know it happened. The strategy is to attempt the gate repeatedly, or in parallel, and use the successes. KLM's technical achievement was showing that with enough ancillas and clever teleportation-based tricks, the success probability can be pushed arbitrarily close to one, so the resource cost stays polynomial rather than exponential.

Polynomial is not cheap, however. Building large-scale linear-optical computing this way demands enormous numbers of single photons, interferometers stable to a fraction of a wavelength, and detectors of very high efficiency — and high-efficiency single-photon detectors are typically superconducting, which quietly reintroduces cryogenics into a "room-temperature" platform. Photonics escapes the dilution refrigerator for the computation but usually not for the measurement.

## 4.3 Measurement-Based Computing: Build the Entanglement First

Probabilistic gates suggest a different way to organize a computation, and it turns out to be a deep one.

In the **measurement-based** or **one-way** model, introduced by Robert Raussendorf and Hans Briegel in 2001, you separate the computation into two phases:

  1. **Prepare** a large, highly entangled resource state — a **cluster state** — in which many qubits are entangled in a regular lattice pattern. This state does not depend on the algorithm you intend to run.
  2. **Compute** by measuring the qubits one at a time, each in a basis chosen according to the algorithm and adapted to the outcomes of earlier measurements. Every measurement consumes a qubit and steers the remaining entanglement toward the answer.

It is called "one-way" because the resource is destroyed as it is used: measurement is irreversible, so the computation runs forward only. Remarkably, this model is exactly as powerful as the circuit model of Chapter 3 — any circuit can be compiled into a measurement pattern on a large enough cluster state.

Why does this suit photonics so well? Because it moves all the hard, probabilistic work into the *preparation* phase, which is offline. Entangling operations that only sometimes succeed are acceptable when you are assembling a resource: keep trying, stitch together the pieces that worked, and discard the rest. Once the cluster exists, the computation itself needs only single-photon measurements — which photonics does well and fast. This is why large-scale photonic architectures are generally designed around cluster states rather than around a straightforward gate-by-gate circuit.

## 4.4 Loss Is the Dominant Error

Every platform has a characteristic failure mode. For superconducting circuits it is decoherence; for neutral atoms, atom loss and gate error. For photonics, the dominant error is simply that **the photon does not arrive**.

Photons are absorbed in fibres and waveguides, scattered at every interface, lost at couplers, and missed by imperfect detectors. Each of these is an independent chance of disappearing, so the survival probability of a single photon through a full protocol is a product of many factors — and a product of numbers slightly less than one falls off exponentially in the number of factors.

The consequence is brutal for protocols that need many photons to *all* survive. If a single photon survives with probability \\(\eta\\), then \\(N\\) independent photons all survive with probability

\\[ P_{\text{all}} = \eta^N \\]

Three lines of NumPy make the scale of the problem clear.

```python
import numpy as np

# Probability that ALL N photons survive a channel whose per-photon transmission is eta
N = np.array([1, 10, 100, 1000])
for eta in [0.99, 0.95, 0.90]:
    survival = eta ** N
    print(f"per-photon transmission eta = {eta:.2f}")
    for n, p in zip(N, survival):
        print(f"   N = {n:>4}  ->  P(all survive) = {p:.3e}")
```

**Output:**

```
per-photon transmission eta = 0.99
   N =    1  ->  P(all survive) = 9.900e-01
   N =   10  ->  P(all survive) = 9.044e-01
   N =  100  ->  P(all survive) = 3.660e-01
   N = 1000  ->  P(all survive) = 4.317e-05
per-photon transmission eta = 0.95
   N =    1  ->  P(all survive) = 9.500e-01
   N =   10  ->  P(all survive) = 5.987e-01
   N =  100  ->  P(all survive) = 5.921e-03
   N = 1000  ->  P(all survive) = 5.292e-23
per-photon transmission eta = 0.90
   N =    1  ->  P(all survive) = 9.000e-01
   N =   10  ->  P(all survive) = 3.487e-01
   N =  100  ->  P(all survive) = 2.656e-05
   N = 1000  ->  P(all survive) = 1.748e-46
```

**Reading this.** A per-photon transmission of 0.99 sounds excellent — a one-percent loss. At ten photons it is still fine. At a thousand photons, that same "excellent" component has reduced the all-survive probability to about one in twenty thousand. Drop the transmission to 0.95 and a thousand-photon protocol becomes hopeless by any margin.

Two lessons follow, and they shape photonic architecture. First, **every component matters multiplicatively**, so photonic engineering is an unglamorous, relentless campaign against loss at every interface, in every waveguide, in every detector. Second, no realistic component set will ever make \\(\eta^N\\) acceptable for large \\(N\\) — which is why serious photonic proposals do not rely on all photons surviving. They build in **loss tolerance**: heralding, so failures are detected rather than silently corrupting the result; redundant encodings, so a lost photon can be identified and its information recovered; and cluster states with enough extra connectivity that missing pieces can be routed around. Loss is treated as an error to be corrected, not a specification to be met.

## 4.5 The Continuous-Variable Alternative

There is a second way to encode quantum information in light, and it deserves an honest paragraph because it changes the trade-offs rather than removing them.

Instead of counting individual photons, the **continuous-variable** approach encodes information in the *quadratures* of a light field — the amplitude and phase variables of an optical mode, which take continuous values much like the position and momentum of a harmonic oscillator. Its practical appeal is that the necessary entanglement can be generated **deterministically** rather than probabilistically: squeezed light sources produce entangled modes on demand, and very large cluster states can be built by entangling modes separated in time or frequency within a single beam, using modest hardware. Measurement is done by homodyne detection, which is fast and efficient and needs no photon counting.

The difficulties are equally real. Squeezing is always finite, so every mode carries a residual noise that acts like a built-in error, and that noise accumulates as the computation proceeds. Worse, the naturally available operations in this setting — the so-called Gaussian operations, which include squeezing, displacement, and beam splitters — are **not sufficient for a quantum speedup**: a computation built only from Gaussian states and Gaussian measurements can be simulated efficiently on a classical computer. A non-Gaussian element must be added, and supplying one with high quality is exactly as hard as the photon-photon interaction problem we started with. A leading strategy is to encode a qubit into an oscillator using specially structured non-Gaussian states, an approach that also builds in error correction against small displacements. This is an active and serious research direction, not a solved problem.

## 4.6 Silicon Spin Qubits: Betting on the Fab

Now change substrate entirely. Consider a single electron confined in a **quantum dot** — a small pocket in a semiconductor, defined by voltages on metal gates lithographed above it, small enough that the electron's energy levels are discrete. Apply a magnetic field, and the electron's spin splits into two states, spin-up and spin-down. Call them \\(|0\rangle\\) and \\(|1\rangle\\). That is the qubit.

The argument for this platform is industrial rather than physical, and it is a strong one.

  * **It is made of silicon.** Every technique that the semiconductor industry has refined over more than half a century — lithography, deposition, etching, testing, yield management on wafers with billions of devices — is in principle available. No other quantum platform can be manufactured in an existing commercial foundry.
  * **The footprint is minuscule.** A quantum dot is tens of nanometres across, orders of magnitude smaller than a superconducting qubit, which occupies a substantial fraction of a millimetre. If a future machine needs an enormous number of physical qubits for error correction, the amount of chip area per qubit stops being a footnote and becomes a system-level constraint.
  * **Gates are electrically controlled.** Both single- and two-qubit operations are driven by voltages on nearby gates — the same kind of signal a classical chip already distributes.

### 📚 Isotopic Purification: Making the Substrate Quiet

An electron spin in a solid is surrounded by nuclei, and any nucleus with nonzero spin generates a small magnetic field. Many such nuclei, fluctuating randomly, produce a jittering local field that dephases the qubit — the \\(T_2\\) process of the introductory series.

Silicon offers an unusually clean escape. Its dominant isotope, silicon-28, has **zero nuclear spin**; the magnetic culprit is silicon-29, which makes up only a few percent of natural silicon. Growing the device layer from **isotopically purified silicon-28** removes most of that nuclear-spin bath, and coherence improves substantially as a result. This is a good example of a general principle in qubit engineering: sometimes the most effective control knob is not the qubit at all, but the material it sits in.

### 📚 Exchange Gates

Two-qubit gates use the **exchange interaction**, a purely quantum effect arising from the requirement that the total wavefunction of two identical electrons be antisymmetric. When the wavefunctions of two neighbouring electrons overlap, their spin states become coupled with a strength conventionally written \\(J\\).

The elegant part is the control. \\(J\\) depends on the overlap of the two electron wavefunctions, and that overlap is set by the height of the tunnel barrier between the dots — which is set by a voltage on a gate electrode sitting between them. Lower the barrier and the interaction turns on; raise it and the interaction switches off. The coupling is therefore both **fast** and **switchable by an ordinary voltage pulse**, using local nearest-neighbour physics rather than a shared bus.

### 📚 The Honest Challenge: Uniformity

The introductory series listed the challenge for this platform in three words: *uniformity across many devices*. That understates neither the difficulty nor the reason for optimism.

The difficulty is that a quantum dot's properties depend on the atomic-scale details of its immediate surroundings — charge traps in the oxide, imperfections at the interface, small variations in gate geometry. Two neighbouring dots on the same chip can have noticeably different operating voltages and resonance frequencies. Where an ion's identity is guaranteed by physical law (Chapter 3), a quantum dot's identity is a manufacturing outcome, so every device needs tuning, and tuning a large array by hand does not scale. Charge noise from the same imperfections limits gate quality directly. And because the qubits are small and closely spaced, getting enough control wiring to them — and dissipating the heat that wiring carries into a cold refrigerator — is a serious systems problem, which is one reason there is strong interest in operating these qubits at slightly higher temperatures where more cooling power is available.

The reason for optimism is that these are exactly the categories of problem the semiconductor industry knows how to attack: process control, automated testing, and design for yield. Whether that expertise transfers to devices holding a single electron is the open question the platform is built on.

## 4.7 Topological Qubits: A Beautiful Idea, and Its Present Status

Every platform so far fights noise by isolating the qubit and then correcting the errors that get through anyway. The topological approach proposes something more ambitious: build a qubit that is **intrinsically protected**, so that most local noise cannot corrupt it in the first place.

### 📚 The Idea

The strategy is to store information **non-locally**. Suppose the quantum state is not held at any single point but is encoded in a global, collective property of a system — in the pattern of how certain quasiparticles are arranged, rather than in the state of any one of them. Then a disturbance acting on a small region simply has nothing local to corrupt: to change the encoded information, the environment would have to act coherently across the whole system at once, which random noise essentially never does.

The candidate quasiparticles are **anyons**, excitations that can exist in effectively two-dimensional systems and whose statistics are neither bosonic nor fermionic: exchanging two of them — **braiding** one around another — can change the system's quantum state in a way that depends only on the *topology* of the paths taken, not on their precise shape, speed, or small wobbles. Computation would then be performed by braiding: moving quasiparticles around each other in prescribed patterns, with the answer determined by the pattern's topology. Small errors in the trajectories would not matter, because a slightly deformed path is topologically the same path. That is the source of the protection, and it is genuinely beautiful — error resistance built into the physics rather than layered on top through error correction.

The most-pursued candidate is the **Majorana zero mode**, a state predicted to appear at the ends of certain engineered one-dimensional systems, typically a semiconducting nanowire coupled to a superconductor under a magnetic field. A single qubit would be encoded across a pair of these spatially separated modes — non-local by construction, since neither end alone carries the information.

### 📚 The Status, Stated Carefully

This is the section of the series where accuracy requires restraint, so let us be direct.

**A working topological qubit has not been conclusively demonstrated.** The theory is well developed and taken seriously by the community. The experimental programme is substantial and has run for well over a decade. But the field has not reached the point where an unambiguous topological qubit — one that has been initialized, manipulated, and measured with the protection the theory promises — is an accepted experimental fact.

The reason is that the experimental signatures are difficult to interpret. Measurements consistent with Majorana modes have been reported repeatedly, but several of the signatures that were initially taken as evidence have turned out to be reproducible by mundane alternatives — disorder, ordinary bound states near an interface, and other non-topological effects can imitate them. The history of the field includes claims that were subsequently disputed on reanalysis, and published results that were corrected or retracted. Recent work has moved toward more stringent, multi-signature protocols precisely because the community recognized that earlier criteria were not sufficient to distinguish topological from conventional explanations, and interpretation of these newer measurements remains under active debate.

None of this means the approach is wrong. It means that the standard of evidence in this area is high for good reason, and that the sensible position is to watch it rather than to assume it. The pragmatic way to read any announcement here is to ask what alternative, non-topological explanation was ruled out, and how. And note the payoff if the programme succeeds: hardware-level protection could dramatically reduce the error-correction overhead that, as the introductory series emphasized, currently defines the timeline of the entire field. High potential value combined with unproven feasibility is exactly the profile of a research bet, and it should be described as one.

## 4.8 No Winner Yet

Across two chapters we have now seen the major approaches, and the honest conclusion is the one the introductory series reached: **there is no winner, and it is not obvious that there needs to be one.**

Notice that the platforms do not merely differ in quality — they differ in *character*, and the characters map onto different jobs.

| Role in a future system | What it demands | Who looks suited |
|---|---|---|
| **Computation** | Fast, high-fidelity gates; dense connectivity; a credible path to error correction | Superconducting circuits (speed, fabrication), trapped ions (fidelity, connectivity), neutral atoms (flexible geometry), silicon spins (density, manufacturing) |
| **Communication** | Travel over distance with low loss; interface to fibre | Photonics, essentially uncontested |
| **Memory** | Very long coherence; gate speed matters less | Atomic and nuclear-spin systems, where storage times are longest |

Read that table and the "hybrid" conclusion writes itself. A large future machine need not be built from one kind of qubit. A plausible architecture computes in one modality, stores in another, and links separate processing modules with photons — exactly the photonic interconnect idea we met in Chapter 3, which is already how ion and atom platforms propose to scale beyond a single trap or a single array. Interfaces between modalities — converting a stationary qubit's state into a flying photon and back, faithfully — then become a first-class engineering problem rather than a curiosity.

Two closing cautions, both consistent with the calibration this dojo has tried to build.

**Do not rank platforms by qubit count.** The number is not comparable across modalities, it says nothing about gate quality or connectivity, and it conflates physical with logical qubits. A platform with fewer, better qubits and a clean error-correction path may be far closer to a useful machine than one with a larger register.

**Expect the ranking to change.** Every platform in these two chapters has improved substantially in the last several years, and each has a different bottleneck: fabrication uniformity here, gate speed there, loss elsewhere, evidence itself in one case. Bottlenecks fall at unequal rates, and the fastest-improving platform today may not be the one that arrives first.

### 🎯 Exercise Problems

  1. **Why linear optics is stuck.** Explain in your own words why beam splitters and phase shifters, no matter how many you connect, cannot produce a deterministic entangling gate between two photons — and what KLM adds that changes this.
  2. **Loss budget.** Using \\(P_{\text{all}} = \eta^N\\), find the per-photon transmission \\(\eta\\) needed to keep the all-survive probability above 1/2 for \\(N = 100\\) and for \\(N = 10{,}000\\). Comment on whether "improve the components" is a viable strategy on its own.
  3. **Heralded versus silent.** Explain why a probabilistic gate that *announces* its failures is much more useful than one that fails silently, and relate this to why loss is easier to handle than an unnoticed phase error.
  4. **Materials as a control knob.** Explain how isotopic purification improves spin coherence, and identify one other platform in this series where a material or environmental change — rather than a change to the qubit itself — is the main lever on performance.
  5. **Reading a topological claim.** Suppose you encounter a report of evidence for Majorana modes. List three questions you would want answered before treating it as a demonstration of a topological qubit.

## Summary

This chapter surveyed three platforms with sharply different bets.

**Photonics** begins from an almost decoherence-free carrier that operates at room temperature and travels naturally through fibre, and runs immediately into the fact that photons barely interact with each other, so deterministic two-qubit gates are the core difficulty. The **KLM** scheme of Knill, Laflamme, and Milburn (2001) supplies the missing nonlinearity through **measurement** on ancilla photons, at the price of gates that are probabilistic but **heralded**. The **measurement-based** model of Raussendorf and Briegel (2001) makes that price affordable by pushing all the probabilistic work into offline preparation of a large entangled **cluster state**, after which computing is just a sequence of single-qubit measurements. The dominant error is **loss**, and our three-line calculation showed why: \\(\eta^N\\) collapses so fast that no component improvement rescues a large protocol, forcing architectures to be loss-tolerant by design. The **continuous-variable** alternative buys deterministic entanglement with squeezed light, at the cost of finite-squeezing noise and the need for a hard-to-supply non-Gaussian element.

**Silicon spin qubits** encode information in an electron spin in a gate-defined quantum dot. Their case is industrial: compatibility with semiconductor manufacturing, a nanometre-scale footprint, and purely electrical control, with **isotopic purification** to silicon-28 quieting the nuclear-spin bath and **exchange coupling**, switched by a barrier voltage, providing fast two-qubit gates. The challenge is **uniformity** — a quantum dot's properties are a manufacturing outcome rather than a law of nature — along with charge noise and the wiring and heat load of dense control.

**Topological qubits** propose to store information **non-locally** in the arrangement of anyonic quasiparticles, so that local noise has nothing local to corrupt, with computation performed by topologically robust **braiding** and **Majorana zero modes** as the leading candidate. The status must be stated plainly: despite sustained effort, a working topological qubit **has not been conclusively demonstrated**, several reported signatures have proved reproducible by non-topological effects, and some claims in this area have been disputed, corrected, or retracted. The idea remains scientifically serious and its payoff would be large; it is a research bet, and belongs in that category.

Finally, **there is no winner yet** — and different platforms may end up filling different roles, with computation, communication, and memory plausibly handled by different modalities linked through photonic interfaces. Judge platforms by gate fidelity, connectivity, and progress toward error correction rather than by qubit count, and expect the ranking to shift as different bottlenecks fall at different rates.

In the next chapter we take the question that has appeared at the end of every platform discussion so far and make it the subject: what does it actually take to scale, from control wiring and cryogenic budgets to the error-correction overhead that sets the timeline for the whole field.

[← Chapter 3: Trapped Ions and Neutral Atoms](<chapter-3.html>) [Chapter 5: The Scaling Challenge →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
