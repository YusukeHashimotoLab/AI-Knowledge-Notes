---
title: "Chapter 5: The Scaling Challenge"
chapter_title: "Chapter 5: The Scaling Challenge"
subtitle: "Error Correction Overhead, Control Wiring, Modularity, and How to Read a Benchmark"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/X-uxpyNhmqI"
    title="QC Hardware Ch.5: The Scaling Challenge"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-computing-hardware/chapter-5.html>) | Last sync: 2026-08-16

[Quantum Computing Dojo](<../index.html>) > [Quantum Computing Hardware](<index.html>) > Chapter 5

## 5.1 Why More Qubits Alone Is Not Progress

Chapters 2 to 4 walked through the physical platforms one by one: superconducting circuits, trapped ions and neutral atoms, photons, spins, and the topological proposal. Each chapter ended in the same place — the physics works, and the engineering is unfinished. This final chapter is about the engineering, because that is where the next decade of the field will actually be decided.

Start with the single most misread number in quantum computing: the qubit count.

The *Introduction to Quantum Computing* series set up the constraint in one line. If a two-qubit gate fails with probability \\(\epsilon\\), a circuit of \\(m\\) gates survives with roughly \\(e^{-m\epsilon}\\), so the usable **circuit depth** is of order \\(1/\epsilon\\) — that is the whole NISQ argument. Notice what does *not* appear in that expression: the number of qubits. Adding qubits widens the circuit; it does nothing for the depth budget.

### 📚 Width, Depth, and the Product

A useful mental model is that a device offers a rectangle of computational space: **width** (how many qubits) times **depth** (how many gate layers survive before the state dissolves). A useful algorithm needs a rectangle of a certain minimum area *and* a certain minimum shape.

  * Doubling the qubit count doubles the width. If the algorithm you care about needs a hundred times more depth, the rectangle is still the wrong shape.
  * Halving the error rate doubles the depth — and it does so for *every* qubit at once. That is why fidelity improvements are worth more than they sound.
  * Worse, the two are coupled in the wrong direction. Larger chips are harder to fabricate uniformly, harder to wire, harder to calibrate, and more prone to crosstalk. Naive scaling of width tends to *degrade* depth.

This is the honest reason to be sceptical when a hardware announcement leads with a qubit count and mentions error rates in a footnote, or not at all. The footnote is the headline.

## 5.2 The Arithmetic of Error Correction

The escape from the depth budget is quantum error correction, introduced conceptually in the intro series. Here we look at what it *costs* and, more importantly, why the cost is payable at all.

### 📚 A Logical Qubit Is a Machine, Not a Qubit

One **logical qubit** is a block of many **physical qubits** together with a process that runs continuously:

  1. **Encode.** The logical state lives non-locally across the block; no single physical qubit holds it.
  2. **Measure the syndrome.** Repeatedly measure carefully chosen multi-qubit operators that reveal *whether an error occurred and where*, without revealing the encoded state. Measuring the state would collapse it; measuring only the error pattern leaves the superposition intact.
  3. **Decode.** A classical algorithm ingests the stream of syndrome bits and infers which correction to apply — in real time, faster than errors accumulate.

Step 3 deserves emphasis because it is easy to forget: a fault-tolerant quantum computer contains a substantial *classical* computer running a decoder at the syndrome measurement rate. Error correction is a hardware problem, a control problem, and a classical computing problem simultaneously.

The **surface code** is the most studied scheme for solid-state hardware because its syndrome checks involve only neighbouring qubits on a two-dimensional lattice — exactly the connectivity a planar chip provides. Its **threshold** is comparatively forgiving: if physical error rates are pushed below roughly the one-percent level, adding more physical qubits per logical qubit makes the logical error rate fall.

The price is steep. Estimates of the encoding **overhead** depend strongly on the physical error rate and on the logical error rate you demand, but figures on the order of a thousand physical qubits per logical qubit are commonly quoted for running large algorithms. Treat that as an order of magnitude and a moving target, not a specification: it is precisely the number that hardware and code improvements are trying to reduce.

### 📚 The Asymmetry That Makes Fault Tolerance Possible

Here is the insight that turns error correction from a curiosity into a plan.

A code is characterized by its **distance** \\(d\\), roughly "how many individual errors must conspire before the logical information is corrupted". A distance-\\(d\\) surface code patch corrects up to \\((d-1)/2\\) errors. The two quantities that matter scale very differently with \\(d\\):

  * **Cost grows polynomially.** A surface-code patch is a \\(d \times d\\) lattice, so the number of physical qubits grows like \\(d^2\\).
  * **Logical error falls exponentially** — *provided* the physical error rate \\(p\\) sits below the threshold \\(p_{\text{th}}\\). Schematically,

\\[ p_L(d) \approx A \left( \frac{p}{p_{\text{th}}} \right)^{(d+1)/2} \\]

When \\(p < p_{\text{th}}\\) the base of that power is less than one, so \\(p_L\\) is suppressed exponentially in \\(d\\) while the qubit bill grows only as \\(d^2\\). Exponential beats polynomial, and that single inequality is the entire reason a fault-tolerant machine is thought to be buildable rather than merely definable.

The flip side is unforgiving. When \\(p > p_{\text{th}}\\) the base exceeds one, and adding qubits makes things *worse*: more components, more errors, no suppression. There is no amount of scale that rescues a device operating above threshold. **Getting below threshold is not an optimization — it is a precondition.**

> **What the threshold is and is not**
>
> The threshold is a property of a code *plus* a noise model *plus* a decoder, not of hardware alone. Quoted values move when any of the three changes, and the "one-percent level" figure for the surface code is a round number for a family of estimates, not a constant of nature. Use it as an order of magnitude for reasoning, never as a specification to be met exactly.

## 5.3 Hands-On: Threshold Behaviour in NumPy

Let us make the asymmetry concrete. The code below evaluates the toy scaling law above for a physical error rate below the threshold and for one above it.

> **This is a pedagogical toy formula, not a real code's performance.** The expression \\(p_L = A(p/p_{\text{th}})^{(d+1)/2}\\) captures the *shape* of threshold behaviour and nothing else. Real logical error rates depend on the noise model, the decoder, the syndrome-extraction circuit, and correlated error mechanisms that this one-line formula ignores entirely. We fix \\(A = 1\\) and \\(p_{\text{th}} = 0.01\\) purely for illustration.

```python
import numpy as np

# --- A pedagogical toy model of the error-correction threshold ---
# p_L(d) = A * (p / p_th)^((d+1)/2)
# p    : physical error rate per operation
# p_th : threshold error rate (we use the surface-code figure of ~1%)
# d    : code distance (odd); the code corrects (d-1)/2 errors
# A    : an O(1) prefactor, set to 1 here
#
# This captures the SHAPE of threshold behaviour, not any real code's
# exact performance. Real numbers depend on the noise model, the decoder,
# and the details of the syndrome circuit.

p_th = 0.01
A = 1.0


def logical_error_rate(p, d, p_th=p_th, A=A):
    """Toy scaling law for the logical error rate of a distance-d code."""
    return A * (p / p_th) ** ((d + 1) / 2.0)


def physical_qubits(d):
    """Surface-code cost grows only polynomially: roughly d^2 data qubits
    (plus a comparable number of syndrome qubits)."""
    return d * d


distances = np.array([3, 5, 7, 9, 11, 13])

print("Threshold p_th = 1.0e-02   (toy model, prefactor A = 1)")
print()

for p, label in [(0.001, "BELOW threshold"), (0.02, "ABOVE threshold")]:
    print(f"Physical error rate p = {p:.0e}   ({label}, p/p_th = {p/p_th:.2f})")
    print(f"  {'d':>3}  {'physical qubits ~ d^2':>21}  {'logical error p_L':>20}")
    for d in distances:
        pL = logical_error_rate(p, d)
        print(f"  {d:>3}  {physical_qubits(d):>21d}  {pL:>20.3e}")
    print()

# --- The asymmetry that makes fault tolerance possible ---
p = 0.001
d_small, d_large = 3, 13
pL_small = logical_error_rate(p, d_small)
pL_large = logical_error_rate(p, d_large)
print(f"Going from d = {d_small} to d = {d_large} at p = {p:.0e}:")
print(f"  cost      x {physical_qubits(d_large) / physical_qubits(d_small):>10.1f}   (polynomial, ~d^2)")
print(f"  error     x {pL_large / pL_small:>10.3e}   (exponential in d)")
print()

# --- Distance needed to reach a target logical error rate ---
target = 1e-12
print(f"Distance needed for p_L < {target:.0e} (odd d only):")
for p in [0.005, 0.002, 0.001]:
    d = 3
    while logical_error_rate(p, d) >= target and d < 199:
        d += 2
    print(f"  p = {p:.0e}  ->  d = {d:>3}   physical qubits ~ {physical_qubits(d):>6d}   p_L = {logical_error_rate(p, d):.2e}")
```

**Output:**

```
Threshold p_th = 1.0e-02   (toy model, prefactor A = 1)

Physical error rate p = 1e-03   (BELOW threshold, p/p_th = 0.10)
    d  physical qubits ~ d^2     logical error p_L
    3                      9             1.000e-02
    5                     25             1.000e-03
    7                     49             1.000e-04
    9                     81             1.000e-05
   11                    121             1.000e-06
   13                    169             1.000e-07

Physical error rate p = 2e-02   (ABOVE threshold, p/p_th = 2.00)
    d  physical qubits ~ d^2     logical error p_L
    3                      9             4.000e+00
    5                     25             8.000e+00
    7                     49             1.600e+01
    9                     81             3.200e+01
   11                    121             6.400e+01
   13                    169             1.280e+02

Going from d = 3 to d = 13 at p = 1e-03:
  cost      x       18.8   (polynomial, ~d^2)
  error     x  1.000e-05   (exponential in d)

Distance needed for p_L < 1e-12 (odd d only):
  p = 5e-03  ->  d =  79   physical qubits ~   6241   p_L = 9.09e-13
  p = 2e-03  ->  d =  35   physical qubits ~   1225   p_L = 2.62e-13
  p = 1e-03  ->  d =  25   physical qubits ~    625   p_L = 1.00e-13
```

**Reading the result.** Four observations, in order of importance.

  * **Below threshold, the two columns diverge.** Going from \\(d = 3\\) to \\(d = 13\\) costs about 19 times more qubits and buys five orders of magnitude in logical error. That gap is the whole game.
  * **Above threshold, more qubits make it worse.** The second table climbs instead of falling. (The values exceeding 1 are not probabilities — the toy formula has left its domain of validity. Read them as "the encoding has failed completely", which is the correct qualitative message.)
  * **The margin below threshold matters enormously.** In the last block, a factor of five in the physical error rate changes the required distance from 79 to 25, and the qubit bill by an order of magnitude. This is why hardware teams chase fidelity long after they are technically "below threshold": the overhead per logical qubit shrinks fast as the margin grows.
  * **The overhead is real even in the best case.** Hundreds to thousands of physical qubits for *one* well-protected logical qubit, and an algorithm needs many logical qubits. Multiply, and you see why the sections that follow — wiring, cooling, calibration, modularity — are not side issues.

Try changing \\(p_{\text{th}}\\) to see how sensitive the required distance is to the code's threshold, or add a realistic \\(A > 1\\) prefactor and watch the required distance shift.

## 5.4 Control Electronics: Every Qubit Needs Wires

Suppose the physics is solved. You still have to talk to each qubit, and this is where scaling quietly becomes a systems-engineering problem.

### 📚 The Wiring Problem

In the standard architecture, each qubit needs at least one control channel and shares or owns a readout channel. Control signals are generated by room-temperature electronics — arbitrary waveform generators, microwave sources, lasers and modulators depending on the platform — and delivered to the qubits by physical lines.

The trouble is that the lines scale linearly with the qubit count while the space, the cooling power, and the money do not.

  * **Physical room.** A refrigerator's cross-section and a vacuum chamber's optical access are fixed. Coaxial lines and laser beams are not free to multiply.
  * **Heat load.** For solid-state platforms, every line running from room temperature down to the coldest stage carries heat, both by conduction along the metal and through the signal power dissipated at the bottom. A dilution refrigerator's cooling power at its coldest stage is minuscule, and it is a *hard* budget: exceed it and the fridge simply warms up, degrading every qubit at once.
  * **Cost and reliability.** Each channel is a chain of connectors, amplifiers, attenuators, and filters, and each element is a candidate failure point.

Note the shape of the constraint. It is not "we cannot build a bigger chip"; it is "we cannot get signals to and from a bigger chip without breaking the thermal budget". Qubit fabrication and qubit *access* scale differently.

### 📚 Multiplexing and Cryo-CMOS

Two directions are being pursued to break the one-line-per-qubit scaling.

**Frequency multiplexing** already helps on the readout side: many qubits with distinct resonator frequencies can share one line, with their signals separated in frequency. It works well, but it trades against fabrication precision — the frequencies must be well spread and well controlled — and it does not remove the need for individual control.

**Cryogenic control electronics (cryo-CMOS)** is the more ambitious idea: move the signal generation itself into the refrigerator, close to the qubits, so that only a few digital lines cross from room temperature. The physics is attractive and prototype circuits exist. The obstacle is thermodynamic rather than conceptual: transistors dissipate power, and power at the coldest stage is exactly the resource in shortage. Realistic designs therefore place electronics at *intermediate* temperature stages where cooling power is far greater, and trade signal quality against heat. Whether that trade closes at scale is one of the genuinely open engineering questions of the field.

Trapped-ion and neutral-atom systems face the same problem in optical form: individually addressing many sites means many beams, or fast beam-steering, or integrated photonics delivering light on-chip. The vocabulary differs; the scaling pressure is identical.

### 📚 The Calibration Burden

There is a subtler cost that rarely makes the headlines. Every qubit and every gate needs calibrated parameters — frequencies, pulse amplitudes, durations, phases, readout thresholds — and these drift over hours or days.

The number of things to calibrate grows at least linearly with the qubit count, and two-qubit gate calibrations grow with the number of *connections*. Crosstalk makes matters worse, because parameters are not fully independent: tuning one qubit can shift its neighbour. Beyond a certain size, calibrating by hand between experiments simply stops being possible, and the machine must calibrate itself — automated routines running continuously, deciding what to re-tune and when, without human supervision.

This is why quantum computing companies employ so many control-software engineers. A machine that cannot keep itself calibrated is not usable at scale, no matter how good its qubits are on a good day.

## 5.5 Modularity: Building Big Out of Small

If a single device cannot be made arbitrarily large, the alternative is to make many small ones and connect them. This is the same answer classical computing reached decades ago, when single processors stopped getting faster and multi-core and networked machines took over.

### 📚 Monolithic Versus Modular

A **monolithic** processor puts every qubit on one chip or in one trap. All interactions are direct and fast, which is the ideal case for algorithm compilation. But yield falls as area grows — one bad qubit can compromise a large device — and the wiring and thermal constraints of Section 5.4 bind hardest here.

A **modular** processor is built from units small enough to fabricate reliably, linked so that qubits in different modules can interact. The cost is that inter-module operations are slower and noisier than intra-module ones, so the machine has a *hierarchy* of connectivity that compilers must respect — again exactly like memory hierarchies and interconnect topologies in classical high-performance computing.

### 📚 How Modules Are Linked

**Photonic interconnects** are the leading candidate for solid-state and atomic modules alike. A stationary qubit emits a photon entangled with its own state; photons from two modules interfere at a beamsplitter and are detected; a successful detection pattern leaves the two distant stationary qubits entangled. The scheme is **heralded** — it announces its own success — which is precisely what makes it usable despite being probabilistic: failures are known and simply retried. Success probability per attempt is low, since photon collection and loss are unforgiving, so the engineering effort concentrates on emission efficiency, photon indistinguishability, and detector performance. For superconducting modules there is an added difficulty: the qubits speak microwave, while low-loss links speak optical, so a coherent microwave-to-optical transducer is required — an active research area in its own right.

Once modules are entangled, that entanglement is a *resource*. Given a shared entangled pair plus classical communication, a gate between distant qubits can be performed, and a quantum state can be teleported between modules. This is the same physics as **quantum networking**: entanglement distribution over distance, entanglement swapping through intermediate nodes, and eventually quantum repeaters that extend the range beyond direct photon loss limits. A large modular quantum computer and a quantum network are, architecturally, the same object at different length scales — which is why progress in one tends to help the other.

**Ion shuttling** is the trapped-ion route to modularity, and it is a nice contrast because it moves the *qubits* rather than entanglement. Chains of ions become slow and spectrally crowded as they grow, so instead of one long chain the trap is divided into zones with dedicated roles — storage, gate interaction, readout — and ions are physically transported between them by time-varying electrode voltages. The demanding part is doing it without heating the ions' motion, since motional modes are the very thing two-qubit gates use. Shuttling keeps everything inside one vacuum system, so it scales further than a single chain but not indefinitely; photonic links between separate traps are the next tier up.

Neutral atoms occupy an interesting middle ground: atoms can be moved between tweezer sites, giving reconfigurable connectivity inside one array without any inter-module link at all.

## 5.6 Benchmarking Without Fooling Yourself

If you take away one habit from this chapter, make it this one: be very careful with single numbers.

### 📚 Why Qubit Counts Do Not Compare

The intro series made this point and it deserves repeating with hardware detail behind it. Two devices with the same qubit count can differ by orders of magnitude in what they can compute, because the count says nothing about:

  * **Error rates** — one-qubit, two-qubit, readout, and idling errors, which set the depth budget.
  * **Connectivity** — all-to-all versus nearest-neighbour. Compiling an algorithm onto limited connectivity inserts SWAP gates, and each SWAP spends depth you cannot afford.
  * **Gate speed relative to coherence** — what matters is not the coherence time in seconds but how many gates fit inside it.
  * **Physical versus logical** — a thousand noisy physical qubits and a thousand error-corrected logical qubits are separated by several generations of engineering.
  * **Usability** — whether all the qubits work simultaneously, at the quoted fidelity, on the day of the experiment.

A qubit count is a *bound* on capability, not a measure of it.

### 📚 Randomized Benchmarking and Holistic Metrics

**Randomized benchmarking** is the standard way to measure gate quality. Apply sequences of random gates of increasing length, chosen so that the ideal result would return the qubits to a known state, then measure how quickly the success probability decays with sequence length. The decay rate gives an average error per gate.

Its strengths are real: the result is largely insensitive to state-preparation and measurement errors, and it scales to routine use. Its limits are equally real. It reports an *average* over random gates, which can be kinder than the structured circuits real algorithms run; it can under-report coherent, systematic errors that partly cancel in random sequences; and it says nothing about crosstalk unless it is deliberately designed to — a qubit benchmarked alone often looks better than the same qubit benchmarked while its neighbours are busy.

Because per-gate numbers can mislead, the field has tried **holistic metrics** that fold qubit number, connectivity, gate fidelity, and compiler quality into a single figure by asking what size of random circuit a machine can run with a statistically meaningful result. **Quantum volume** is the best-known attempt of this kind. Its virtue is honesty about the trade-offs: a machine cannot inflate it by adding poor qubits, since width and depth are demanded together, and it measures the *system* — compiler included — rather than a component.

But holistic metrics carry their own caveats. They compress a multi-dimensional object into one dimension, and any such compression discards information; they use random circuits, which are not the circuits you want to run; they can be optimized for as targets; and being defined around a square circuit shape, they suit some architectures better than others. Application-level benchmarks — run a real algorithm class and report the quality of the answer — address different weaknesses and have their own, notably the difficulty of comparing across problem instances.

> **A practical reading checklist**
>
> When you meet a hardware claim, ask five questions. (1) Physical or logical qubits? (2) What are the two-qubit gate and readout error rates, and were they measured with the whole device active? (3) What is the connectivity, and does the reported circuit account for compilation onto it? (4) Is the metric an average over random instances or a worst case? (5) What does the best classical method achieve on the same task, run by someone trying to win? A claim that survives all five is worth taking seriously.

## 5.7 The Honest Outlook

The gap between today's noisy devices and a fault-tolerant quantum computer is not one breakthrough wide. It is an engineering marathon, and its defining feature is that **every layer must scale together**.

  * **Qubits** must get better, not just more numerous — because the margin below threshold sets the overhead, and the overhead sets everything else.
  * **Control** must scale sublinearly in wires and be self-calibrating, or the machine drowns in cables and tuning.
  * **Cryogenics and vacuum** must supply cooling power, optical access, and stability for systems far larger than today's.
  * **Interconnects** must link modules with high enough rate and fidelity that modularity is a gain rather than a bottleneck.
  * **Classical computing** must decode syndromes in real time, at scale, without becoming the limiting factor itself.
  * **Software** — compilers, calibration systems, error-correction stacks — must turn all of the above into something a user can program.

A weakness in any one layer caps the whole system. That is the honest reason timelines are hard to predict: the schedule is set by whichever layer is currently worst, and which layer that is keeps changing.

No platform has all the answers, and this is the fair summary of Chapters 2 to 4. Superconducting circuits have fast gates and a fabrication industry behind them, and a wiring and thermal problem in front of them. Trapped ions have superb control and connectivity, and slow gates and a hard scaling path. Neutral atoms have large reconfigurable arrays and maturing gate and readout performance. Photonics has room-temperature transport and networking, and the difficulty of making photons interact. Spin qubits have the smallest footprint and semiconductor manufacturing in principle, and uniformity to prove. Topological qubits promise protection at the hardware level, and must first be shown to exist as a usable qubit.

That is not a discouraging picture — it is an accurate one, and it is a far more interesting field than a horse race with a known winner. What it asks of you as a reader is calibration rather than allegiance. When the next announcement appears, you now have the questions to ask of it: which layer did this improve, at what cost to the others, and measured how? Watch the field with informed eyes, and the news becomes readable.

### 🎯 Exercise Problems

  1. **Width versus depth** : a device doubles its qubit count while its two-qubit gate error rate stays fixed. Which algorithms benefit, and which do not? Frame your answer in terms of the width-depth rectangle of Section 5.1.
  2. **Threshold sensitivity** : modify the code to use \\(p_{\text{th}} = 0.005\\) and recompute the distance needed for \\(p_L < 10^{-12}\\) at \\(p = 10^{-3}\\). Explain physically why halving the threshold costs so much.
  3. **Overhead arithmetic** : assuming roughly \\(2d^2\\) physical qubits per logical qubit (data plus syndrome), estimate the physical qubit count for an algorithm needing one hundred logical qubits at \\(d = 25\\). Compare with the "order of a thousand physical per logical" rule of thumb and comment on why the two agree only approximately.
  4. **Heat budget** : explain why moving control electronics into a refrigerator does not automatically solve the wiring problem, and what quantity ultimately limits how much can be moved in.
  5. **Modularity trade-off** : an algorithm requires frequent gates between qubits in different modules, where inter-module operations are far slower and noisier than intra-module ones. What does a compiler have to do, and what does that imply about which algorithms suit modular machines?
  6. **Benchmark critique** : find a public hardware announcement and answer the five questions in the reading checklist of Section 5.6. Note which questions the announcement does not let you answer.

## Summary

This chapter argued that scaling a quantum computer is a systems problem, not a qubit-count problem. **More qubits alone is not progress** : the depth budget of order \\(1/\epsilon\\) does not improve when the width grows, and larger devices tend to degrade fidelity through crosstalk, non-uniformity, and calibration load. **Error correction** buys depth at a steep price — a logical qubit is a block of physical qubits plus continuous syndrome measurement plus a real-time classical decoder, with overhead on the order of a thousand physical qubits per logical qubit, an order of magnitude and a moving target — but it works because of a scaling asymmetry: below the threshold of roughly the one-percent level, the logical error rate falls **exponentially** in the code distance while the cost grows only **polynomially** (\\(\sim d^2\\)). Our NumPy toy model made that asymmetry visible, and showed the flip side: above threshold, adding qubits only makes things worse. **Control electronics** then imposes its own limits — one line per qubit does not scale against fixed space and cooling power, multiplexing and cryo-CMOS are the responses, and the calibration burden grows with every qubit and connection. **Modularity** is the structural answer: smaller units linked by photonic interconnects, using the same heralded entanglement physics as quantum networking, with ion shuttling as the trapped-ion variant. **Benchmarking** requires discipline — qubit counts do not compare across platforms, randomized benchmarking reports averages that can flatter, and holistic metrics such as quantum volume improve on single-component numbers while still compressing a many-dimensional reality into one.

This completes the *Quantum Computing Hardware* series. You have seen what a qubit must physically provide, how six platform families try to provide it, and why the remaining work is an engineering marathon across every layer at once. Specifications will keep moving — that is the point of a qualitative treatment — but the principles here should let you read the next decade of announcements, and the papers behind them, with informed eyes.

[← Chapter 4: Photonic, Spin, and Topological Platforms](<chapter-4.html>) [Series Top →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
