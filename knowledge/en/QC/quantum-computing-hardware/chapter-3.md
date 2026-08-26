---
title: "Chapter 3: Trapped Ions and Neutral Atoms"
chapter_title: "Chapter 3: Trapped Ions and Neutral Atoms"
subtitle: "Qubits Made of Atoms, Held in Place by Light and Electric Fields"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/WiRSYrq10Oo"
    title="QC Hardware Ch.3: Trapped Ions and Neutral Atoms"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-computing-hardware/chapter-3.html>) | Last sync: 2026-08-16

[Quantum Computing Dojo](<../index.html>) > [Quantum Computing Hardware](<index.html>) > Chapter 3

Chapter 2 built a qubit out of a fabricated circuit: a superconducting element patterned on a chip, engineered atom by atom in the lithography sense but never quite identical to its neighbour. This chapter takes the opposite approach. Instead of *making* a quantum system, we *borrow* one that nature has already made — an atom — and hold it still long enough to compute with it. Two platforms follow this philosophy: **trapped ions**, which hold charged atoms with electric fields, and **neutral atoms**, which hold uncharged atoms with focused laser beams. They share a great deal, and they differ in exactly the places that matter for scaling.

## 3.1 Nature's Identical Qubits

Start with the single most attractive property of atomic qubits, because it explains why physicists have pursued them for decades despite the engineering difficulty.

**Every atom of a given isotope is exactly identical to every other.** Not similar to within manufacturing tolerance — identical, as a matter of physical law. Two calcium ions of the same isotope have precisely the same energy levels, precisely the same transition frequencies, precisely the same lifetimes. There is no fabrication variation to characterize, no device-to-device calibration table that grows with the number of qubits, and no yield problem in the semiconductor sense: you cannot manufacture a defective atom.

Contrast this with a fabricated qubit. Every superconducting circuit comes out of the fab with slightly different parameters, and each one must be measured and its control pulses tuned individually. That calibration burden grows with the size of the processor. Atomic platforms start from a position where the qubits themselves need no such per-device characterization — although, as we will see, the *control apparatus* pointed at each atom certainly does.

The second attraction follows from the same physics. The qubit states we choose in an atom are internal electronic states, well isolated from their surroundings by the atom's own structure. Left alone in a good vacuum, an atomic qubit holds its quantum information for a very long time compared with the duration of a gate. This is why the intro series listed **long coherence** as a defining strength of the trapped-ion platform.

## 3.2 Holding a Charged Atom Still: The Paul Trap

An ion is an atom with an electron removed, so it carries net electric charge — and charge is precisely what lets us grab it with electric fields. But there is a classic obstacle in the way.

### 📚 Why Static Fields Cannot Trap

To trap a charged particle at a point in empty space, you need a potential energy minimum in all three directions at once: push it back toward the centre whether it drifts left, up, or forward. Static electric fields cannot supply that. In a charge-free region the electrostatic potential obeys Laplace's equation, and a consequence of that equation — the result is usually called **Earnshaw's theorem** — is that the potential has no local minimum in free space. Every stationary point is a saddle: confining along some directions, expelling along others. Squeeze the ion inward in two directions and it escapes along the third.

The physical picture is worth holding onto, because it is the reason the trap looks the way it does. You cannot build a bowl out of static electric fields. You can only build a saddle.

### 📚 The Trick: Spin the Saddle

The **Paul trap**, developed by Wolfgang Paul and recognized with a share of the 1989 Nobel Prize in Physics, solves this with time. Apply an oscillating (radio-frequency) voltage instead of a static one. The saddle now flips its orientation rapidly: the direction that was expelling becomes confining, and vice versa, many times per ion oscillation.

The mechanical analogy is a ball on a rotating saddle surface. At any instant the ball is rolling downhill somewhere. But if the saddle rotates fast enough, the ball never has time to escape along the downhill direction before that direction becomes uphill. Averaged over the fast oscillation, the ball behaves as though it sits in a genuine bowl.

That time-averaged bowl is called the **pseudopotential**, and it is what actually confines the ion. The ion's motion therefore has two components: a slow, nearly harmonic oscillation in the pseudopotential (the part we will use for computing) and a small, fast jitter at the drive frequency (called **micromotion**, and something experimentalists work hard to minimize, since it broadens transitions and degrades gates).

Modern traps usually implement this on a chip — electrodes patterned on a planar surface, with the ions floating some tens of micrometres above them — which brings a degree of lithographic control to what was once a hand-assembled apparatus.

## 3.3 Cooling: Slowing Atoms Down with Light

A trap holds an ion in a region, but a hot ion rattles around inside that region with far too much energy for the gates we are about to describe. It must be cooled — and cooled to a regime where its motion is quantum mechanical, not merely cold in the everyday sense.

The workhorse is **Doppler cooling**. Tune a laser slightly *below* the atom's transition frequency. An ion moving toward the laser sees the light Doppler-shifted upward, closer to resonance, so it absorbs photons preferentially — and each absorbed photon delivers a momentum kick opposing its motion. An ion moving away sees the light shifted further off resonance and absorbs less. The re-emitted photons go out in random directions, so on average they contribute no net push. The result is a velocity-dependent drag force that removes kinetic energy. Doppler cooling is robust and it is how essentially every ion experiment begins, but it has a floor set by the random recoil of those re-emitted photons. To reach the regime where the ion occupies the lowest quantum states of its harmonic motion — which is what high-fidelity gates require — experiments follow up with more refined techniques, typically **resolved-sideband cooling**, in which the laser drives a transition that removes exactly one quantum of motion at a time.

It is worth being honest about what this means practically. Every ion processor carries a substantial optical apparatus: stabilized lasers at several wavelengths, ultra-high vacuum, and beam paths that must be aligned to individual atoms. The qubits are free; the machinery around them is not.

## 3.4 Where the Qubit Lives

Once an ion is trapped and cold, which two of its many internal states do we call \\(|0\rangle\\) and \\(|1\rangle\\)? Two families of choice dominate, and the trade-off between them recurs throughout hardware design.

| Encoding | The two states are | Splitting is in the | Character |
|---|---|---|---|
| **Hyperfine (or Zeeman) qubit** | Two sublevels within the atom's electronic ground-state manifold, separated by the interaction between the electron and the nuclear spin | Microwave range | Both states are ground states, so neither can decay by emitting a photon; storage times are extremely long |
| **Optical qubit** | The ground state and a long-lived *metastable* excited state | Optical range | Addressed by a narrow-linewidth laser; storage time is ultimately capped by the finite lifetime of the metastable state |

Neither is universally better. Hyperfine qubits win on raw storage lifetime and can be driven by microwaves as well as lasers; optical qubits allow certain operations to be performed with a single tightly focused beam. Both are read out the same way — by **state-dependent fluorescence**. Shine in light that is resonant only with one of the two qubit states: that state scatters many photons and glows, the other stays dark. A camera or photodetector then distinguishes them. Because a single ion can scatter a very large number of photons before anything goes wrong, this readout is one of the cleanest measurements in all of quantum technology.

## 3.5 Two-Qubit Gates through Shared Motion

Here is the mechanism that makes trapped ions distinctive, and it is genuinely elegant.

Ions in a common trap all carry the same sign of charge, so they repel each other. Confined by the trap and repelled by their neighbours, a group of ions settles into a line — a **Coulomb crystal** — with the ions spaced apart at an equilibrium separation. Now push one ion. Because they are electrically coupled, all of them respond. The chain does not have independent per-ion motion; it has **collective normal modes**, exactly like a row of masses joined by springs. Every ion participates in every mode.

Those modes are quantum harmonic oscillators, and their excitations are **phonons**. And this is the key: the phonon modes are *shared by all the ions in the chain*. They form a bus.

### 📚 The Phonon Bus

A two-qubit gate then works in outline as follows. Use a laser to couple ion A's internal state to a shared motional mode, so that the mode's state becomes conditioned on A's qubit state. Use a second laser pulse to couple that same mode to ion B, whose evolution now depends on what A was doing. Finally disentangle the motion, leaving the two internal states entangled with each other and the motion back where it started.

The founding proposal along these lines is due to **Juan Ignacio Cirac and Peter Zoller in 1995**, who showed that a set of laser pulses acting through a shared motional mode implements a controlled quantum gate between arbitrary ions in a chain — the paper that turned trapped ions from a spectroscopy platform into a computing platform. Modern experiments generally use a descendant of this idea known as the **Mølmer–Sørensen gate**, which is designed to be far less sensitive to the exact motional state of the chain and is the workhorse entangling operation on today's ion processors.

### 📚 All-to-All Connectivity, and Its Limits

The consequence for algorithms is important. Because *every* ion couples to the *same* shared mode, any ion in the chain can be entangled with any other ion directly. This is **all-to-all connectivity**, and it is a real architectural advantage. On a nearest-neighbour chip, entangling two distant qubits requires a chain of SWAP gates to walk one qubit over to the other, and each SWAP costs circuit depth we cannot afford in the NISQ regime. On an ion chain, distant pairs cost the same as neighbouring pairs.

But this advantage does not survive arbitrary scaling, and it is important to see why.

  * **The mode spectrum gets crowded.** A chain of \\(N\\) ions has \\(3N\\) motional modes. As \\(N\\) grows, the modes bunch together in frequency. Gates work by addressing chosen modes without accidentally exciting their neighbours, which requires spectral resolution — and resolving closely spaced frequencies requires longer pulses. **Gates get slower as the chain gets longer.**
  * **Heating gets worse.** Electrical noise from the trap electrodes continuously feeds energy into the motional modes. Longer chains sit closer to more electrode surface and take longer to run a gate, so more heating accumulates during each operation.
  * **Chains become mechanically fragile.** Long ion strings are floppy: the confinement along the chain axis weakens as ions are added, mode frequencies drift, and eventually the linear arrangement becomes unstable and buckles into a zigzag.
  * **Optical addressing gets harder.** Every ion needs its own individually steerable, phase-controlled beam.

This is the concrete meaning of the intro series' summary that ion gates are "much slower than superconducting gates" and that "scaling one long chain is hard." The slowness is not an incidental engineering defect — it is tied to the spectral resolution the phonon-bus mechanism requires.

### 📚 The Scaling Response: Don't Build One Long Chain

The dominant architectural answer is to give up on the single long chain and instead build a trap with **many small zones**. Ions are held in short chains where the physics is clean, and are physically **shuttled** between zones — moved along the trap by adjusting electrode voltages, split apart and recombined — so that any two ions can be brought together in an interaction zone when a gate between them is required. This architecture is usually called **QCCD**, by analogy with the charge-coupled devices in old digital cameras that move charge packets across a chip.

The bargain is explicit. Short chains keep gate quality high and preserve all-to-all connectivity *within* a zone; the cost is that transport takes time, adds motional heating that must be re-cooled away, and requires an intricate trap with many independently controlled electrodes. A longer-range variant of the same idea connects separate trap modules through **photonic links**: an ion emits a photon entangled with its internal state, photons from two modules interfere on a beam splitter, and a successful detection heralds entanglement between ions in different modules. That process is probabilistic and must be repeated until it succeeds, but it is one of the few known routes to connecting physically separate quantum processors.

## 3.6 Neutral Atoms: Tweezers Instead of Electrodes

Now remove the charge. A neutral atom cannot be grabbed by electric fields the way an ion can — but it can be grabbed by *light*.

Focus a laser beam tightly, and its electric field induces a dipole moment in a nearby atom; that induced dipole then interacts with the field itself. When the laser is tuned below the atom's resonance, the resulting force pulls the atom toward the region of highest intensity — the focal spot. A single tightly focused beam therefore becomes a microscopic bowl that can hold one atom. These are **optical tweezers**.

Three features follow, and together they define the platform.

**Arrays are built optically, not lithographically.** Split one laser into hundreds of beams with a hologram or an acousto-optic deflector, and you have hundreds of traps. Their positions are set by a pattern of light, so the *geometry of the processor is programmable*: one-dimensional chains, two-dimensional lattices, arbitrary shapes, even three-dimensional stacks. Nothing has to be re-fabricated to change the layout.

**Loading is stochastic, and then repaired.** When atoms are captured from a cold cloud, each trap ends up occupied or empty essentially at random, so a freshly loaded array is full of holes. The standard fix is beautiful: image the array to see which sites are filled, then use a movable tweezer to pick up atoms one at a time and carry them into the gaps, assembling a defect-free array before the computation starts. This same ability to *move atoms while holding their quantum state* has a second use — bringing distant qubits together for a gate, giving the platform a form of reconfigurable connectivity that a fixed chip cannot offer.

**The atoms are still identical, and still well isolated.** Everything said in Section 3.1 applies here too. Neutral atoms also interact more weakly with stray electric fields than ions do, precisely because they carry no charge.

## 3.7 The Rydberg Blockade

Weak interaction is a blessing for storage and a problem for computing: two ground-state atoms held a few micrometres apart barely notice each other, so there is nothing to build a two-qubit gate from. The solution is to switch the interaction on only when we want it, by promoting an atom to a **Rydberg state** — a state in which the outermost electron is excited to a very high principal quantum number and orbits far from the nucleus.

A Rydberg atom is enormous compared with a ground-state atom, and it carries a correspondingly enormous electric dipole moment. Two Rydberg atoms therefore interact strongly even when separated by several micrometres — an interaction many orders of magnitude larger than between the same two atoms in their ground states.

### 📚 How the Blockade Produces a Gate

Consider two neighbouring atoms and a laser tuned exactly to the transition from the qubit state to the Rydberg state.

  * If only one atom is present, or if the atoms are far apart, the laser drives that transition on resonance and excites the atom efficiently.
  * Now suppose one atom is *already* Rydberg-excited. Its strong interaction with its neighbour **shifts the neighbour's Rydberg level** away from where it used to be. The laser, still at its original frequency, is now *off resonance* for the second atom. The second excitation is therefore suppressed: within a certain distance — the **blockade radius**, where the interaction shift exceeds the drive strength — only one of the two atoms can be excited at a time.

This is the **Rydberg blockade**. It is a conditional mechanism, and conditional is exactly what a two-qubit gate needs: whether one atom responds to the laser depends on the state of the other. A well-known family of gates uses precisely this — a sequence of laser pulses whose effect on the pair differs depending on which qubit states are populated, imprinting a controlled phase and thereby producing entanglement. The interaction never has to be switched on mechanically; it appears and disappears with the Rydberg excitation itself.

Section 3.8 makes the suppression quantitative with a small numerical experiment, because the mechanism is a specific and checkable statement about a driven two-level system.

### 📚 Honest Challenges

Consistent with the platform summary in the introductory series:

  * **Atom loss.** Atoms are held by light in a vacuum chamber, and they get lost: from background-gas collisions, from heating out of the trap, and from decay of the Rydberg state into untrapped states during a gate. A lost atom is not a wrong answer — it is a *missing* qubit, an error mode that has no counterpart on a solid-state chip. Detecting loss and reloading atoms mid-computation is an active engineering problem.
  * **Fidelities are maturing.** Two-qubit gate performance on neutral atoms has improved rapidly, but the platform is younger than trapped ions and its entangling operations are generally not yet as clean. Rydberg states have finite lifetimes, laser phase noise couples directly into the excitation, and the atoms' residual thermal motion blurs the interatomic distance that sets the blockade strength.
  * **Measurement is disruptive.** Reading out an atom typically involves scattering many photons from it, which heats it and can eject it from the trap. Measuring some qubits mid-circuit without disturbing the others — required for error correction — takes extra work here.

## 3.8 Python: Why Detuning Suppresses an Excitation

The blockade rests on one claim: shifting a level away from resonance suppresses the transition the laser is trying to drive. Let us verify that claim and see how strong the suppression is.

In the rotating frame, a two-level atom driven by a laser is described by the \\(2 \times 2\\) Hamiltonian

\\[ H = \frac{\Omega}{2}\sigma_x + \frac{\Delta}{2}\sigma_z \\]

where \\(\Omega\\) is the **Rabi frequency** (how hard the laser drives the transition) and \\(\Delta\\) is the **detuning** (how far the laser sits from resonance). Starting from the ground state, the excited-state population oscillates, and the *peak* it reaches is

\\[ P_{\max} = \frac{\Omega^2}{\Omega^2 + \Delta^2} \\]

On resonance (\\(\Delta = 0\\)) the atom can be driven completely into the excited state. Off resonance it cannot, no matter how long we wait. We integrate the Schrödinger equation directly with a fourth-order Runge–Kutta step and compare against that formula.

```python
import numpy as np

# --- Rotating-frame Hamiltonian of a driven two-level atom (hbar = 1) ---
# H = (Omega/2) * sigma_x + (Delta/2) * sigma_z
#   Omega = Rabi frequency (drive strength), Delta = detuning (laser - transition)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def hamiltonian(omega, delta):
    return 0.5 * omega * SX + 0.5 * delta * SZ


def evolve(omega, delta, times):
    """Integrate i d|psi>/dt = H|psi> from |g> with a small fixed step (RK4)."""
    H = hamiltonian(omega, delta)
    dt = 1e-4
    psi = np.array([1.0, 0.0], dtype=complex)  # start in the ground state |g>
    excited = []
    t_now = 0.0
    for t_target in times:
        n_steps = int(round((t_target - t_now) / dt))
        for _ in range(n_steps):
            k1 = -1j * (H @ psi)
            k2 = -1j * (H @ (psi + 0.5 * dt * k1))
            k3 = -1j * (H @ (psi + 0.5 * dt * k2))
            k4 = -1j * (H @ (psi + dt * k3))
            psi = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t_now += n_steps * dt
        excited.append(abs(psi[1]) ** 2)
    return np.array(excited)


# --- Scan the excitation probability over one drive period for three detunings ---
omega = 1.0                                   # fix the drive strength
times = np.linspace(0.0, 2.0 * np.pi / omega, 401)

print("Peak excitation probability of a driven two-level atom")
print(f"{'detuning D/Omega':>18} | {'numerical max':>13} | {'analytic Omega^2/(Omega^2+D^2)':>30}")
print("-" * 70)
for ratio in [0.0, 1.0, 3.0, 10.0]:
    delta = ratio * omega
    p_max = evolve(omega, delta, times).max()
    analytic = omega ** 2 / (omega ** 2 + delta ** 2)
    print(f"{ratio:>18.1f} | {p_max:>13.6f} | {analytic:>30.6f}")

# --- Suppression factor: how much a level shift protects an atom from the drive ---
print()
for ratio in [3.0, 10.0]:
    print(f"detuning = {ratio:>4.0f} x Omega  ->  excitation suppressed by a factor "
          f"{1.0 + ratio ** 2:.0f}")
```

**Output:**

```
Peak excitation probability of a driven two-level atom
  detuning D/Omega | numerical max | analytic Omega^2/(Omega^2+D^2)
----------------------------------------------------------------------
               0.0 |      1.000000 |                       1.000000
               1.0 |      0.499989 |                       0.500000
               3.0 |      0.099997 |                       0.100000
              10.0 |      0.009901 |                       0.009901

detuning =    3 x Omega  ->  excitation suppressed by a factor 10
detuning =   10 x Omega  ->  excitation suppressed by a factor 101
```

**Reading the result, and connecting it to the blockade.** On resonance the drive achieves complete population transfer — this is the ordinary Rabi flop that a single-qubit gate uses. Push the level off resonance by ten times the drive strength and the atom barely responds: the peak excitation falls to about one percent, a suppression of roughly \\(1 + (\Delta/\Omega)^2\\).

Now reinterpret \\(\Delta\\). In the blockade, the detuning is not something the experimenter dials in — it is *generated by the neighbouring atom*. When one atom is already Rydberg-excited, its interaction shifts the second atom's Rydberg level by an amount that plays exactly the role of \\(\Delta\\) in this calculation. The same laser that fully excites a lone atom leaves the second atom almost untouched. The numbers above are why the blockade is a usable gate mechanism and not merely a small perturbation: the suppression grows quadratically with the interaction strength, so pushing atoms slightly closer together buys a great deal of blockade quality.

The same arithmetic also explains a design constraint. Drive too hard (large \\(\Omega\\)) and the ratio \\(\Delta/\Omega\\) falls, letting the forbidden double excitation leak through; drive too gently and the gate becomes slow enough for decoherence and atom loss to matter. The workable window between those two failure modes is exactly what neutral-atom gate design optimizes.

## 3.9 Side by Side

| | Trapped ions | Neutral atoms |
|---|---|---|
| **What holds the qubit** | Oscillating electric fields (Paul trap) acting on the ion's charge | Focused laser beams (optical tweezers) acting on an induced dipole |
| **Two-qubit mechanism** | Shared motional modes of the Coulomb crystal — a phonon bus | Rydberg blockade: an excited atom shifts its neighbour off resonance |
| **Connectivity** | All-to-all within a chain; degrades as the chain lengthens | Set by the tweezer geometry, and *reconfigurable* by physically moving atoms |
| **Distinctive strength** | Long coherence and the highest gate and measurement fidelities among current platforms | Programmable array geometry; arrays of identical atoms assemble relatively easily |
| **Distinctive challenge** | Slow gates; a single long chain does not scale, motivating shuttling architectures | Atom loss and reloading; gate fidelity and readout still maturing |

What the table cannot show is how much the two platforms have in common, and that is easy to lose behind the differences.

**Shared strengths**: qubits that are identical by physical law, excellent natural isolation from the environment, and no fabrication variation to characterize.

**Shared costs**: ultra-high vacuum, stabilized laser systems, and optics that must be aimed at individual atoms. Both platforms buy their qubit quality with optical complexity. The engineering effort that a superconducting processor spends on cryogenics and fabrication uniformity, an atomic processor spends on lasers, vacuum, and beam control.

### 🎯 Exercise Problems

  1. **Earnshaw in one sentence.** Explain in your own words why a set of static, charged electrodes cannot hold an ion at a stable point in free space, and what the oscillating field in a Paul trap changes about that argument.
  2. **Connectivity cost.** An algorithm needs an entangling gate between the first and the last qubit of a register. Describe what this costs on an ion chain and what it costs on a fixed nearest-neighbour chip. Then explain why the ion advantage shrinks as the chain grows.
  3. **Blockade window.** Using \\(P_{\max} = \Omega^2/(\Omega^2 + \Delta^2)\\), find how large the interaction-induced shift \\(\Delta\\) must be, in units of \\(\Omega\\), to keep the unwanted double excitation below \\(10^{-3}\\). Comment on what happens to this requirement if you double the drive strength to make the gate faster.
  4. **Loss is a different error.** Explain why losing an atom is a qualitatively different failure from a qubit flipping from \\(|0\rangle\\) to \\(|1\rangle\\), and why an error-correction scheme has to be told about it explicitly.
  5. **Choose a platform.** For each of the following, argue which of the two platforms in this chapter is better suited, and why: (a) a circuit with many entangling gates between arbitrary distant pairs; (b) a simulation of a two-dimensional lattice model with a triangular geometry; (c) storing a quantum state for as long as possible.

## Summary

In this chapter we studied the two platforms that build qubits out of real atoms. Their shared advantage is fundamental: **atoms of a given isotope are exactly identical**, so there is no fabrication variation and no per-device parameter spread, and their internal states are naturally well isolated from the environment.

**Trapped ions** are held by a **Paul trap**, which sidesteps the impossibility of static trapping — a consequence of Earnshaw's theorem — by rapidly oscillating the field so that the time-averaged **pseudopotential** becomes a genuine bowl. **Doppler cooling** followed by sideband cooling brings the ion's motion near its quantum ground state. The qubit lives in **hyperfine** or **optical** internal levels and is read out by state-dependent fluorescence, one of the cleanest measurements available. Two-qubit gates run through the **shared motional modes** of the Coulomb crystal — the phonon bus proposed by **Cirac and Zoller in 1995** and refined into the **Mølmer–Sørensen** gate used today. Because every ion couples to the same mode, the platform enjoys **all-to-all connectivity**, but that advantage erodes with chain length as the mode spectrum crowds together, forcing slower gates. The architectural response is not a longer chain but many short ones, with ions **shuttled** between zones, and photonic links between separate modules.

**Neutral atoms** are held by **optical tweezers**, which makes the array geometry a pattern of light rather than a fabricated layout — reconfigurable, and repairable by moving atoms into empty sites. Their two-qubit mechanism is the **Rydberg blockade**: exciting one atom to a large, strongly interacting Rydberg state shifts its neighbour's level off resonance and prevents a second excitation. Our NumPy calculation made that suppression concrete, reproducing \\(P_{\max} = \Omega^2/(\Omega^2 + \Delta^2)\\) and showing that a shift of ten drive strengths cuts the unwanted excitation to about one percent. The honest costs are **atom loss** and gate fidelities that are still maturing relative to trapped ions.

Neither platform has an obvious path to a large fault-tolerant machine yet, and neither is out of the running. What they demonstrate is that qubit quality and system scale pull in different directions, and that every architecture in this series is a different answer to the same tension.

In the next chapter we turn to three more platforms with very different characters: **photonics**, where the qubit flies at the speed of light but refuses to interact; **silicon spin qubits**, which stake everything on the existing semiconductor industry; and **topological qubits**, an idea of remarkable elegance whose experimental status demands careful and neutral reading.

[← Chapter 2: Superconducting Qubits](<chapter-2.html>) [Chapter 4: Photonic, Spin, and Topological Platforms →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
