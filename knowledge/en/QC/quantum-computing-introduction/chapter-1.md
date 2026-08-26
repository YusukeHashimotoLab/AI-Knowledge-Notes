---
title: "Chapter 1: Why Quantum Computing?"
chapter_title: "Chapter 1: Why Quantum Computing?"
subtitle: "Where Classical Computers Struggle, and What Quantum Machines Honestly Offer"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/HrI5uz2c9cU"
    title="Quantum Computing Ch.1: Why Quantum Computing?"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-computing-introduction/chapter-1.html>) | Last sync: 2026-08-16

[Quantum Computing Dojo](<../index.html>) > [Introduction to Quantum Computing](<index.html>) > Chapter 1

Quantum computing attracts a great deal of attention, and also a great deal of exaggeration. This chapter builds an honest foundation: which problems genuinely resist classical computers, what a quantum computer really does differently, and where the field stands today. The concepts may feel unfamiliar at first, but we will build them up one step at a time.

## 1.1 Where Classical Computers Run Out of Room

Modern classical computers are extraordinary machines. They route global logistics, train large neural networks, and simulate weather systems. For the overwhelming majority of computational tasks, a classical computer is the right tool, and it will remain so.

But a small number of problem classes scale so badly that no amount of engineering rescues them. The difficulty is not that our processors are too slow. The difficulty is that the amount of work grows faster than any realistic increase in speed.

### 📚 The Cost of Simulating Quantum Systems

Consider a system of \\(n\\) interacting two-level quantum particles, for example \\(n\\) electron spins in a molecule. A **quantum state** of that system is described by a list of complex numbers called **amplitudes**, one for each possible configuration of the \\(n\\) particles. Because each particle has two configurations, the total number of amplitudes is

\\[ N = 2^n \\]

This exponential growth is the heart of the matter. Storing every amplitude as two 8-byte floating point numbers gives the following memory requirement:

| Number of particles \\(n\\) | Amplitudes \\(2^n\\) | Memory to store the state |
|---|---|---|
| 10 | about \\(10^{3}\\) | 16 kilobytes |
| 30 | about \\(10^{9}\\) | 17 gigabytes |
| 50 | about \\(10^{15}\\) | about 18 petabytes |
| 300 | about \\(10^{90}\\) | more entries than there are atoms in the observable universe |

At around 50 particles the exact state no longer fits in the largest supercomputers. At 300 particles the bookkeeping exceeds the physical universe. Note carefully what this table does *not* say: it does not say that useful approximations are impossible. Chemists and physicists have built superb approximate methods, such as density functional theory and quantum Monte Carlo, and these methods solve real problems every day. What the table says is that **exact** classical simulation of strongly correlated quantum matter has a hard ceiling.

### Factoring Large Integers

A second example comes from cryptography. Multiplying two large prime numbers is fast. Recovering those primes from the product is, as far as we know, hard. The best known general-purpose classical factoring algorithm is the **general number field sieve**, whose running time grows sub-exponentially in the number of digits, fast enough that doubling the key length makes the attack enormously more expensive. Public-key cryptosystems such as RSA rest on exactly this asymmetry.

Importantly, nobody has *proven* that factoring is hard for classical computers. It is a well-tested belief, not a theorem.

### A Note on Moore's Law

You will often hear that quantum computing is needed "because Moore's law is ending." This is context, not the main argument, and it is worth separating the two.

**Moore's law** is the observation, first made about integrated circuits in the 1960s, that the number of transistors on a chip roughly doubles every couple of years. Transistor counts have continued to grow, but the accompanying gains in clock speed and power efficiency slowed sharply once transistors approached atomic length scales and heat density became the limiting factor. The industry responded with multi-core processors and specialized accelerators such as GPUs and TPUs.

Here is the key point. Even if classical hardware had continued to double in speed forever, an exponential problem would still defeat it. A doubling of speed buys you exactly **one** more particle in the table above. The argument for quantum computing is about the *shape* of the growth curve, not about the slope of the hardware curve.

## 1.2 Feynman's Question

In 1982, Richard Feynman published a paper titled "Simulating Physics with Computers" in the International Journal of Theoretical Physics. He asked a question that now reads as the founding question of the field: if nature is quantum mechanical, and simulating quantum mechanics on a classical machine costs exponentially much, why not build a simulator that is itself quantum mechanical?

The insight is elegant. A quantum system tracks its own amplitudes for free, simply by existing. If we could build a controllable quantum system and tune its interactions to imitate the molecule we care about, the exponential bookkeeping would be handled by physics rather than by memory chips.

This is why **quantum simulation** — of molecules, catalysts, superconductors, and magnetic materials — is widely regarded as the most natural application of quantum computers, and the one where a clear advantage is most plausible.

## 1.3 What Quantum Computing Actually Offers

Now we come to the most important correction in this chapter.

### ❌ The Misconception: "It Tries All Answers in Parallel"

The popular explanation goes like this: a classical bit is 0 or 1, a **qubit** (quantum bit) can be both at once, so \\(n\\) qubits explore all \\(2^n\\) possibilities simultaneously and the answer pops out.

This picture is wrong, and believing it will make every real quantum algorithm look mysterious.

Here is why it fails. It is true that a quantum computer can be placed in a state that involves all \\(2^n\\) configurations at once. But you cannot read that state out. **Measurement returns exactly one outcome**, a single \\(n\\)-bit string, chosen at random according to probabilities set by the amplitudes. Everything else is destroyed. A quantitative version of this limit is known as the **Holevo bound**, proved by Alexander Holevo in 1973: no matter how cleverly you encode, \\(n\\) qubits can deliver at most \\(n\\) classical bits of information to a receiver.

So if the algorithm did nothing but spread itself over all possibilities, measuring it would be no better than guessing.

### ✅ The Honest Picture: Interference

The real mechanism is **interference**. Amplitudes are complex numbers, so they carry a sign and a phase, and they can cancel. A quantum algorithm is a carefully choreographed sequence of operations that arranges for the amplitudes of *wrong* answers to cancel each other out (destructive interference) while the amplitudes of *right* answers add up (constructive interference). Only then does measurement become useful, because the probability of landing on the right answer has been amplified.

This is why quantum algorithms are rare and hard to invent. Spreading over many possibilities is easy. Engineering the cancellation is the difficult part, and it only works when the problem has a mathematical structure the algorithm can exploit, such as the hidden periodicity that Shor's algorithm finds inside the factoring problem.

### What This Implies About Speedups

| Problem class | Best classical | Quantum | Character of the gain |
|---|---|---|---|
| Factoring, discrete logarithm | Sub-exponential | Polynomial (Shor) | Exponential-scale, structure-dependent |
| Simulating quantum systems | Exponential in general | Polynomial for many cases | The most natural fit |
| Unstructured search over \\(N\\) items | \\(O(N)\\) | \\(O(\sqrt{N})\\) (Grover) | Quadratic only, and provably optimal |
| Sorting, arithmetic, most databases | Already efficient | No meaningful gain | Use a classical computer |
| NP-complete problems in general | Exponential (believed) | Not believed to be efficient | No general exponential speedup expected |

Two rows deserve emphasis. Grover's quadratic speedup is genuine, but a quadratic gain can be eaten by the slower clock speeds and error-correction overhead of quantum hardware. And the last row is a persistent source of hype: quantum computers are **not** believed to solve NP-complete problems such as general travelling-salesman instances efficiently. Anyone promising otherwise is going beyond what the theory supports.

## 1.4 A Short History

The field has a clear lineage, from a physicist's question to a research industry.

    
    
    ```mermaid
    flowchart TD
        A[1982 Feynman<br/>Simulating Physics with Computers]
        B[1985 Deutsch<br/>Universal quantum computer defined]
        C[1994 Shor<br/>Polynomial-time factoring algorithm]
        D[1996 Grover<br/>Quadratic speedup for search]
        E[Late 1990s and 2000s<br/>First small experimental devices]
        F[2019<br/>Superconducting advantage experiment]
        G[2018 onward NISQ era<br/>Noisy Intermediate-Scale Quantum]
        A --> B --> C --> D --> E --> F --> G
        style A fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
        style B fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
        style C fill:#00bcd4,stroke:#764ba2,stroke-width:2px,color:#fff
        style D fill:#00bcd4,stroke:#764ba2,stroke-width:2px,color:#fff
        style E fill:#7c4dff,stroke:#764ba2,stroke-width:2px,color:#fff
        style F fill:#7c4dff,stroke:#764ba2,stroke-width:2px,color:#fff
        style G fill:#f57c00,stroke:#764ba2,stroke-width:2px,color:#fff
    ```

**1982 — Feynman poses the question.** As described above, he argues that simulating quantum physics requires a quantum machine.

**1985 — Deutsch formalizes the machine.** David Deutsch publishes "Quantum theory, the Church-Turing principle and the universal quantum computer" in the Proceedings of the Royal Society A. This paper turns Feynman's idea into a definition: a universal quantum computer, an abstract model as precise as the classical Turing machine. Without this step there would be nothing to write algorithms *for*.

**1994 — Shor's algorithm.** Peter Shor presents an algorithm that factors integers and computes discrete logarithms in time polynomial in the number of digits. This was the moment the field acquired urgency, because the security of widely deployed public-key cryptography rests on those two problems being hard.

**1996 — Grover's algorithm.** Lov Grover publishes a quantum search algorithm that finds a marked item among \\(N\\) unstructured items using about \\(\sqrt{N}\\) queries instead of \\(N\\). Unlike Shor's algorithm it applies very broadly, but the speedup is quadratic rather than exponential, and it has been proven that no quantum algorithm can do better for genuinely unstructured search.

**Late 1990s and 2000s — the first devices.** Early demonstrations used nuclear magnetic resonance on molecules in solution, where nuclear spins act as qubits. In 2001 a seven-qubit NMR experiment ran Shor's algorithm to factor the number 15. This was a landmark for the physics, and simultaneously a reminder of the distance still to travel: the answer, 3 times 5, was known in advance, and NMR of this type does not scale.

Alongside this, quantum information produced a practical spin-off much earlier: **quantum key distribution**, proposed by Charles Bennett and Gilles Brassard in 1984, which uses quantum measurement to detect eavesdropping on a communication channel. It is a different technology from quantum computing and should not be confused with it.

## 1.5 The 2019 Advantage Experiment, Described Carefully

In 2019 a team at Google reported an experiment on a 53-qubit superconducting processor named Sycamore. The processor sampled from the output distribution of a random quantum circuit. The team reported that the task took roughly 200 seconds on the device, and estimated that the leading classical supercomputer of the day would need on the order of 10,000 years to produce the same samples. The result was published in Nature and described as **quantum supremacy**, a technical term meaning only "a quantum device performed some task faster than classical hardware could," with no implication that the task is useful.

The claim was contested promptly and productively, and understanding the dispute matters more than memorizing the headline.

  * **The classical estimate was challenged.** Researchers at IBM argued that with a different classical strategy, using the substantial disk storage of a large supercomputer, the same sampling could be done in a matter of days rather than millennia. Subsequent work by several groups, particularly using tensor-network contraction methods, reduced the estimated classical cost much further.
  * **The task was chosen to be hard for classical computers, not to be useful.** Random circuit sampling has no known application. It was selected precisely because it stresses the classical simulation bottleneck.
  * **The right way to read the result.** The experiment demonstrated real control over a device large enough to be uncomfortable for classical simulation. It did not demonstrate a useful computation, and it did not settle the classical-versus-quantum boundary, which continues to move as classical algorithms improve.

Later experiments on other platforms have made similar sampling-based claims, and the same pattern of classical counter-claims has followed each time. Treat every "quantum advantage" announcement as a claim about a specific task with a specific classical baseline, and check what that baseline was.

## 1.6 The NISQ Era and Today's Landscape

In 2018, John Preskill published "Quantum Computing in the NISQ Era and Beyond" in the journal Quantum, giving the present period its name. **NISQ** stands for **Noisy Intermediate-Scale Quantum**: devices with enough qubits to be beyond easy classical simulation, but without the error correction needed to run long computations reliably.

The central obstacle is **decoherence**. A qubit interacts with its environment — stray electromagnetic fields, vibrations, thermal noise — and this interaction randomizes its delicate phase relationships. Since interference is exactly what quantum algorithms depend on, noise does not merely add small errors; it erodes the mechanism itself. Every physical operation also carries a nonzero error rate, so circuits can only run so deep before the output is indistinguishable from noise.

The accepted long-term answer is **quantum error correction**, which encodes one reliable **logical qubit** across many physical qubits, using repeated measurements to detect and correct errors without disturbing the stored information. The cost is steep: published resource estimates for cryptographically relevant tasks such as breaking RSA-2048 run into the millions of physical qubits. Experimental progress on error correction has been real and encouraging in recent years, but the gap between today's devices and a large fault-tolerant machine remains substantial.

### Hardware Platforms

Several physical implementations are being pursued in parallel, and it is genuinely unclear which will win.

| Platform | Qubit is | Broad strengths | Broad challenges |
|---|---|---|---|
| Superconducting circuits | A microwave-frequency circuit on a chip | Very fast gates, mature chip fabrication | Millikelvin refrigeration, shorter coherence times |
| Trapped ions | An individual ion held in an electromagnetic trap | Long coherence, high-fidelity gates, identical qubits | Slower gates, engineering challenges in scaling up |
| Photonics | A single photon or an optical mode | Operates at room temperature, natural for networking | Photon loss, probabilistic operations |
| Neutral atoms | An atom held by optical tweezers | Large, flexible, reconfigurable arrays | Relatively young platform, atom loss |
| Semiconductor spin qubits | An electron spin in silicon | Compatible with existing semiconductor industry | Uniformity across many devices |

Companies and national laboratories worldwide are active on all of these. Rather than tracking qubit-count records, which change frequently and are not comparable across platforms, look at **gate fidelity**, **coherence time**, **connectivity**, and above all **demonstrated error-corrected performance**.

### Realistic Expectations

Being honest about timelines is part of being useful.

  * **What is real today**: high-quality control of tens to hundreds of physical qubits, steady improvement in error rates, and small-scale demonstrations of error correction.
  * **What is plausible in the medium term**: modest but genuine advantage on quantum simulation problems in chemistry and materials, where the fit between problem and machine is best.
  * **What remains distant**: large-scale fault-tolerant computation, including cryptographically relevant factoring. Credible forecasts span a wide range of years, and anyone offering a confident date is guessing.
  * **What is unlikely regardless**: quantum computers replacing your laptop, or efficiently solving NP-complete optimization problems in general.

One consequence deserves attention today rather than later. Because encrypted data can be recorded now and decrypted after a capable machine exists — sometimes called "harvest now, decrypt later" — standards bodies have been moving to **post-quantum cryptography**, classical algorithms designed to resist quantum attack. That migration is happening now, independent of when quantum hardware matures.

## Summary

In this chapter, we established why quantum computing exists as a field. Certain problems, above all the **exact simulation of quantum systems**, cost classical computers an amount of work that grows exponentially with system size, and no improvement in hardware speed changes that shape. **Feynman's 1982 argument** was that a quantum simulator handles this bookkeeping by physics rather than by memory. We corrected the most common misconception directly: a quantum computer does **not** try all answers in parallel, because **measurement returns only one outcome**. The real mechanism is **interference**, in which the amplitudes of wrong answers cancel while those of right answers reinforce, and this only works for problems with exploitable structure. We traced the milestones from **Deutsch's 1985** universal quantum computer through **Shor's 1994** factoring algorithm and **Grover's 1996** quadratic search speedup to the first small devices. We examined the **2019 random circuit sampling experiment** together with the classical counter-claims it provoked, and we placed today's hardware in the **NISQ era** named by **Preskill in 2018**, where **decoherence** limits circuit depth and **quantum error correction** remains the essential unfinished work.

In the next chapter, we will build the mathematical objects behind all of this: the qubit, superposition, measurement under the Born rule, and the uniquely quantum resource of entanglement, with numerical examples in Python.

[← Series Top](<index.html>) [Chapter 2: Qubits, Superposition, and Entanglement →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
