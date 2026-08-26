---
title: "Chapter 2: Superconducting Qubits"
chapter_title: "Chapter 2: Superconducting Qubits"
subtitle: "A Circuit That Behaves Like an Atom, and the Refrigerator It Demands"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/2Uo4DpZDk3Y"
    title="QC Hardware Ch.2: Superconducting Qubits"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/QC/quantum-computing-hardware/chapter-2.html>) | Last sync: 2026-08-16

[Quantum Computing Dojo](<../index.html>) > [Quantum Computing Hardware](<index.html>) > Chapter 2

Chapter 1 laid out what any physical system must deliver to serve as a qubit, and the tension that makes delivering it hard. We now look at the first concrete answer: an electrical circuit, patterned on a chip, cooled until it behaves as a single quantum object.

Superconducting qubits are the most industrially developed platform, and the reason is not that their physics is superior. It is that they are **designed rather than found**. An atom has the energy levels nature gave it; a circuit has the energy levels you drew in the layout software. That freedom is the platform's great strength, and the refrigerator it requires is its great cost.

## 2.1 Why Superconductivity Is the Starting Point

An ordinary wire is a terrible place to store quantum information. Electrons scatter, resistance dissipates energy as heat, and dissipation is decoherence — the environment learns what the circuit was doing.

**Superconductivity removes the dissipation.** Below a critical temperature, electrons in certain materials bind into **Cooper pairs**, and these pairs condense into a single collective state described by one macroscopic wavefunction with a well-defined phase. Current flows without resistance, because there is no low-energy way for a pair to scatter out of the condensate: an **energy gap** protects it.

Two consequences make this the foundation of the platform.

  * **No dissipation means no unavoidable decoherence channel.** The circuit can, in principle, hold a superposition.
  * **The whole circuit shares one quantum degree of freedom.** Billions of electrons behave as a single object, so a macroscopic piece of metal — visible under an ordinary microscope — can display quantum behaviour. This is the surprise at the heart of the field.

### 📚 The Problem With a Plain LC Circuit

The simplest quantum circuit is a capacitor and an inductor: the **LC oscillator**. Classically, charge sloshes back and forth at the resonant frequency

\\[ \omega_0 = \frac{1}{\sqrt{LC}} \\]

Quantized, it is a harmonic oscillator, with energy levels

\\[ E_n = \hbar\omega_0\left(n + \tfrac{1}{2}\right), \qquad n = 0, 1, 2, \ldots \\]

And here the plan collapses. **The levels are equally spaced.** Every transition — \\(|0\rangle \to |1\rangle\\), \\(|1\rangle \to |2\rangle\\), \\(|2\rangle \to |3\rangle\\) — occurs at exactly the same frequency \\(\omega_0\\).

Why is that fatal? Because a gate is applied by sending in a pulse at the transition frequency. If you drive the circuit to move population from \\(|0\rangle\\) to \\(|1\rangle\\), that same drive is perfectly resonant with \\(|1\rangle \to |2\rangle\\), and with \\(|2\rangle \to |3\rangle\\) after that. The population marches up the ladder and out of your two-dimensional computational subspace. You cannot address two levels selectively in a system where every pair of neighbouring levels looks identical.

**A qubit needs anharmonicity**: the \\(|0\rangle \to |1\rangle\\) transition must sit at a *different* frequency from \\(|1\rangle \to |2\rangle\\), so that a pulse tuned to the first is off-resonant with the second and leaves the higher levels alone. Every circuit element in an ordinary laboratory — resistors, capacitors, inductors — is linear, and linear elements give harmonic oscillators. We need a nonlinear element that does not dissipate.

## 2.2 The Josephson Junction

Exactly one circuit element meets both requirements, and it is the reason this platform exists.

A **Josephson junction** is two superconductors separated by a barrier thin enough — a few nanometres of insulating oxide — that Cooper pairs can **tunnel** through it. Brian Josephson predicted in 1962 that a supercurrent would flow across such a barrier with no voltage drop at all, governed by the difference in the quantum phase of the condensate on either side. The prediction was confirmed experimentally soon afterwards, and junctions have been standard components of superconducting electronics ever since.

Two properties matter for us:

  * **It is nonlinear.** A junction acts as an inductor whose effective inductance depends on the current flowing through it. Feed the resulting nonlinear inductance into an LC circuit, and the potential energy is no longer a parabola — so the energy levels are no longer evenly spaced.
  * **It is non-dissipative.** Unlike a resistor or a semiconductor diode, the junction supplies nonlinearity without converting energy into heat. That combination is rare, and no ordinary component offers it.

The result is an **artificial atom**: a circuit with a discrete, unevenly spaced ladder of levels, of which the lowest two are used as \\(|0\rangle\\) and \\(|1\rangle\\). Unlike a real atom, its transition frequency is set by the capacitance and junction parameters you choose at design time, typically placing it in the **microwave** range of a few gigahertz — convenient, because that is exactly the band where commercial signal-generation and amplification technology is mature.

> **Why "artificial atom" is more than a metaphor**
>
> The physics an atomic physicist uses — Rabi oscillations, resonance fluorescence, cavity quantum electrodynamics — transfers to these circuits essentially unchanged. The circuit community imported decades of atomic-physics technique wholesale, which is one reason the platform matured quickly. The difference is that the "atom" here is engineered, so its frequency, its coupling to the drive line, and its anharmonicity are all design parameters.

## 2.3 The Transmon, and the Trade It Makes

The junction gives us anharmonicity, but early superconducting qubits had a serious problem: they were exquisitely sensitive to **charge noise**. Stray charges moving in the substrate and in surface defects shifted the qubit's transition frequency at random, and a frequency that wanders is a phase that randomizes — precisely the pure dephasing that Chapter 1 showed drives \\(T_2\\) below its \\(2T_1\\) ceiling.

The **transmon** is the design that solved this, and it is the workhorse of the platform today. The idea is a deliberate change of regime, achieved by shunting the junction with a comparatively large capacitor.

  * The circuit has two characteristic energies: the **Josephson energy** \\(E_J\\), which favours a well-defined phase across the junction, and the **charging energy** \\(E_C\\), the cost of putting one more Cooper pair on the island.
  * A transmon operates deep in the regime \\(E_J \gg E_C\\). Adding capacitance lowers \\(E_C\\), and the ratio grows.

In this regime the qubit frequency becomes **exponentially insensitive to charge noise**: the dependence on stray charge is suppressed exponentially as \\(E_J/E_C\\) increases. This is why the transmon works — a whole class of dephasing simply stops mattering.

**But the trade is real.** Increasing \\(E_J/E_C\\) also flattens the anharmonicity, and it does so only **algebraically** — as a weak power of the ratio, not exponentially. The exponential gain in charge insensitivity therefore costs a comparatively slow loss of anharmonicity, and that asymmetry is the whole design argument. Push the ratio far enough and you win enormously on noise while losing modestly on level separation.

Modest is not zero, and the residual anharmonicity sets a hard limit on gate speed:

  * A short pulse is spectrally broad. Its bandwidth scales roughly as the inverse of its duration.
  * If that bandwidth exceeds the anharmonicity, the pulse meant for \\(|0\rangle \to |1\rangle\\) also drives \\(|1\rangle \to |2\rangle\\), causing **leakage** out of the computational subspace.
  * So the anharmonicity sets a floor on gate duration. You cannot simply make pulses shorter to fit more gates inside \\(T_2\\).

Pulse-shaping techniques — smoothly ramped envelopes, and corrections derived from the presence of the third level — push this floor down considerably, and much of the practical art of superconducting control lives here. But the floor exists, and it is a direct consequence of the transmon's design compromise.

## 2.4 Control: Gates Are Pulses

A transmon is controlled by sending **microwave pulses** down a line coupled to it.

### 📚 Rabi Oscillations

Drive the qubit at its transition frequency and its population oscillates between \\(|0\rangle\\) and \\(|1\rangle\\) at the **Rabi frequency** \\(\Omega\\), which is proportional to the drive amplitude. This is the fundamental control mechanism, and the whole single-qubit gate set follows from it:

  * The **rotation angle** is set by the pulse *area* — amplitude multiplied by duration. A pulse with \\(\Omega t = \pi\\) is a \\(\pi\\)**-pulse**, a complete flip from \\(|0\rangle\\) to \\(|1\rangle\\), which is the X gate. Half that area gives a \\(\pi/2\\)-pulse, producing an equal superposition.
  * The **rotation axis** is set by the *phase* of the microwave relative to the qubit's own precession. Shifting the drive phase by \\(90°\\) turns an X rotation into a Y rotation, and since that shift is applied in the control electronics rather than on the chip, it costs no time at all.
  * Rotations about \\(z\\) can be performed by simply redefining the phase reference of all subsequent pulses — a bookkeeping change in software, again essentially free.

The consequence is worth stating plainly: on this platform, a single-qubit gate is a shaped waveform lasting a small number of nanoseconds. Gate speed is the platform's signature advantage, and it comes directly from the strong coupling between the circuit and its control line.

**Detuning is the enemy of a clean gate.** If the drive frequency misses the qubit frequency by \\(\Delta\\), the oscillation both speeds up and fails to reach the top:

\\[ P_1(t) = \frac{\Omega^2}{\Omega^2 + \Delta^2}\,\sin^2\!\left(\frac{\sqrt{\Omega^2 + \Delta^2}}{2}\,t\right) \\]

The maximum reachable population is \\(\Omega^2/(\Omega^2 + \Delta^2)\\), which is less than one for any nonzero detuning. No pulse duration recovers a full flip once the frequency is wrong. This is why superconducting devices are recalibrated continually: the qubit frequency drifts, and a drifted frequency is a broken gate. We will compute this curve in Section 2.7.

### 📚 Two-Qubit Gates

Entangling gates require two transmons to interact, and there are two broad philosophies.

**Fixed coupling.** The qubits are permanently connected, usually through a shared capacitance or a bus resonator, and are detuned from each other so that the interaction does nothing on its own. A gate is then activated by driving one qubit at the *other* qubit's frequency — the **cross-resonance** scheme. The always-on coupling produces an entangling interaction only while that drive is present. The hardware stays simple; the price is that residual coupling never truly switches off, producing small unwanted interactions that must be characterized and cancelled.

**Tunable coupling.** A separate tunable element sits between the qubits, and a control signal changes its properties so that the effective coupling can be turned on for the gate and driven close to zero afterwards. This suppresses residual interactions and typically supports faster, cleaner gates. The price is an extra control line per coupler and an extra channel through which noise can reach the qubits — a direct instance of Chapter 1's isolation-versus-control tension, reappearing one level up.

Because coupling is a fabricated structure, transmons interact only with the neighbours they were wired to. Chip layouts are therefore **nearest-neighbour** graphs, and every non-local gate in an algorithm must be routed with SWAPs at three CNOTs apiece, as Chapter 1 described.

## 2.5 Readout: Asking Without Touching

You cannot simply attach a voltmeter to a transmon. Measurement must extract one bit — is the qubit in \\(|0\rangle\\) or \\(|1\rangle\\)? — without the measuring apparatus destroying its neighbours or dragging the qubit's energy away between operations.

The standard solution is **dispersive readout**, borrowed directly from cavity quantum electrodynamics.

Each qubit is coupled to its own **resonator**, a small microwave cavity on the chip, deliberately detuned so that its frequency is far from the qubit's. In this **dispersive** regime the two systems cannot exchange energy — the detuning forbids it — but they do shift each other. The resonator's frequency moves by a small amount whose **sign depends on the qubit's state**.

The measurement then works like this:

  1. Send a weak microwave probe tone at the resonator.
  2. The tone reflects or transmits, acquiring a phase and amplitude that depend on where the resonator's frequency currently sits.
  3. Because that position depends on the qubit state, the returning signal carries the answer.
  4. Amplify the signal — including with quantum-limited amplifiers at the cold stage, since the outgoing signal is only a handful of photons — and digitize it outside the refrigerator.

Two features make this scheme good. It is **quantum non-demolition** in principle: the measurement asks about the qubit's energy without changing it, so a qubit measured as \\(|1\rangle\\) stays in \\(|1\rangle\\) and can be measured again. And it is **multiplexed**: many resonators, each at a slightly different frequency, can share one output line, with the individual answers separated by frequency afterwards. That matters enormously for wiring, which is the subject of the next section.

## 2.6 The Cold: Why Millikelvin Is Not Negotiable

Superconducting qubits live inside a **dilution refrigerator** at temperatures of a few tens of millikelvin — colder than interstellar space. This is not an engineering preference. It follows from a single comparison of energies.

### 📚 The kT Versus hf Argument

A qubit's two states are separated by an energy \\(hf\\), where \\(f\\) is the transition frequency. The environment, at temperature \\(T\\), carries thermal energy of order \\(k_B T\\). If \\(k_B T\\) is comparable to \\(hf\\), the environment will simply excite the qubit at random, and the device never sits reliably in \\(|0\rangle\\). The requirement is

\\[ k_B T \ll h f \\]

Put numbers to it. For a design frequency of about 5 GHz, the level splitting corresponds to a temperature of

\\[ \frac{hf}{k_B} \approx 0.24\ \text{K} \\]

The thermal excited-state population follows a Boltzmann factor \\(e^{-hf/k_BT}\\):

| Temperature | \\(k_BT / hf\\) | Thermal excitation \\(e^{-hf/k_BT}\\) |
|---|---|---|
| 300 mK | about 1.25 | about \\(0.45\\) — the qubit is nearly randomized |
| 20 mK | about 0.083 | about \\(6 \times 10^{-6}\\) |
| 10 mK | about 0.042 | about \\(4 \times 10^{-11}\\) |

At a few hundred millikelvin — already extremely cold by any everyday standard — the qubit is close to a coin toss. Only in the tens of millikelvin does thermal excitation drop far enough to be a small error rather than the dominant one. **This is why the operating temperature is what it is**, and it is set by the qubit frequency together with two constants of nature. No improvement in materials will change it. The only alternative would be a much higher transition frequency, which brings its own control and fabrication difficulties.

### 📚 The Dilution Refrigerator in Two Sentences

A dilution refrigerator exploits a mixture of the two helium isotopes, helium-3 and helium-4, which below roughly 0.87 K separates into a helium-3-rich phase floating on a helium-3-dilute phase. Forcing helium-3 atoms across that phase boundary costs energy — an effect analogous to evaporation, but one that does not shut off as the temperature falls — so continuously pumping helium-3 through the boundary provides steady cooling down to the millikelvin range.

The machine is a series of nested stages, each colder than the last, with the qubit chip mounted at the coldest one.

### 📚 Wiring: The Scaling Constraint Nobody Can Design Away

Here is the practical limit that shapes the platform's future, and it has nothing to do with qubit physics.

Every qubit needs control lines running from room-temperature electronics down to the coldest stage. Each line is a physical object, and each line carries two burdens:

  * **Heat conduction.** A metal line is a thermal bridge between a room-temperature world and a millikelvin one. It leaks heat downward simply by existing.
  * **Thermal noise.** The signal arriving from room temperature carries room-temperature noise. Suppressing it requires attenuators at each cold stage, and those attenuators dissipate the very heat they are removing, at the stage where cooling power is scarcest.

The cooling power available at the coldest stage is minute — measured in microwatts. The heat load, meanwhile, grows with the number of lines. Multiply lines by qubits and the arithmetic becomes uncomfortable well before the qubit physics does.

This is why frequency multiplexing of readout matters so much, why cryogenic control electronics operating inside the refrigerator is an active research direction, and why "how do we wire a large machine" has become as serious a question as "how do we make a better qubit." We return to it in Chapter 5, where scaling is the whole subject.

## 2.7 Hands-On: Simulating a Rabi Oscillation

Let us watch a gate happen. In the frame rotating with the drive, a driven two-level system has the time-independent Hamiltonian (taking \\(\hbar = 1\\))

\\[ H = \frac{1}{2}\begin{pmatrix} -\Delta & \Omega \\ \Omega & \Delta \end{pmatrix} \\]

where \\(\Omega\\) is the drive strength and \\(\Delta\\) the detuning between the drive and the qubit. The code integrates the Schrödinger equation \\(i\,d\psi/dt = H\psi\\) with a fixed-step fourth-order Runge-Kutta method — no solver library, only NumPy — and compares the result against the analytic formula from Section 2.4.

Times are in nanoseconds and frequencies in radians per nanosecond. The drive strength is chosen so that a full Rabi cycle takes 20 ns, making a \\(\pi\\)-pulse 10 ns long.

```python
import numpy as np

# ---------------------------------------------------------------
# Rotating-frame Hamiltonian of a driven two-level system (hbar = 1):
#
#     H = (1/2) * [ [-Delta,  Omega ],
#                   [ Omega,  Delta ] ]
#
# Omega = drive strength (Rabi frequency), Delta = drive detuning.
# Time in nanoseconds, frequencies in rad/ns.
# ---------------------------------------------------------------
OMEGA = 2.0 * np.pi * 0.05        # 50 MHz drive -> 20 ns Rabi period on resonance


def hamiltonian(omega, delta):
    return 0.5 * np.array([[-delta, omega],
                           [omega, delta]], dtype=complex)


def schrodinger_rk4(omega, delta, t_final, dt):
    """Integrate i dpsi/dt = H psi with fixed-step RK4.

    Returns the excited-state population at every step and the final norm.
    """
    H = hamiltonian(omega, delta)
    deriv = lambda psi: -1j * (H @ psi)

    n_steps = int(round(t_final / dt))
    psi = np.array([1.0 + 0j, 0.0 + 0j])          # start in |0>
    pop = np.empty(n_steps + 1)
    pop[0] = abs(psi[1]) ** 2

    for k in range(n_steps):
        k1 = deriv(psi)
        k2 = deriv(psi + 0.5 * dt * k1)
        k3 = deriv(psi + 0.5 * dt * k2)
        k4 = deriv(psi + dt * k3)
        psi = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        pop[k + 1] = abs(psi[1]) ** 2

    return pop, float(np.vdot(psi, psi).real)


def analytic(omega, delta, t):
    """P1(t) = (Omega^2 / Omega_R^2) * sin^2(Omega_R t / 2), Omega_R = sqrt(Omega^2+Delta^2)."""
    omega_r = np.hypot(omega, delta)
    return (omega ** 2 / omega_r ** 2) * np.sin(0.5 * omega_r * t) ** 2


DT = 0.001          # 1 ps integration step
T_FINAL = 40.0      # two Rabi periods on resonance
SAMPLES = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0]

for label, delta in [("on resonance   (Delta = 0)", 0.0),
                     ("detuned        (Delta = Omega)", OMEGA),
                     ("far detuned    (Delta = 2*Omega)", 2.0 * OMEGA)]:
    pop, _ = schrodinger_rk4(OMEGA, delta, T_FINAL, DT)
    omega_r = np.hypot(OMEGA, delta)
    print(label)
    print(f"  generalized Rabi frequency Omega_R/2pi = {omega_r / (2 * np.pi):.6f} GHz")
    print(f"  predicted maximum population           = {OMEGA ** 2 / omega_r ** 2:.6f}")
    print(f"  observed  maximum population           = {pop.max():.6f}")
    print("   t (ns)   P1 (RK4)   P1 (analytic)   |diff|")
    for t in SAMPLES:
        idx = int(round(t / DT))
        exact = analytic(OMEGA, delta, t)
        print(f"  {t:6.1f}   {pop[idx]:.6f}     {exact:.6f}     {abs(pop[idx] - exact):.2e}")
    print()

# ---------------------------------------------------------------
# The gate: a pulse of the right DURATION performs a rotation.
# On resonance a pi-pulse (Omega * t = pi) is a full 0 -> 1 flip.
# ---------------------------------------------------------------
t_pi = np.pi / OMEGA
t_half = 0.5 * t_pi
pop, norm = schrodinger_rk4(OMEGA, 0.0, T_FINAL, DT)
print("Pulse calibration on resonance")
print(f"  pi/2-pulse: t = {t_half:5.2f} ns -> P1 = {pop[int(round(t_half / DT))]:.6f}")
print(f"  pi-pulse  : t = {t_pi:5.2f} ns -> P1 = {pop[int(round(t_pi / DT))]:.6f}")
print(f"  2pi-pulse : t = {2 * t_pi:5.2f} ns -> P1 = {pop[int(round(2 * t_pi / DT))]:.6f}")

# Norm conservation is the sanity check that the integrator is trustworthy.
print(f"  norm at t = {T_FINAL:.0f} ns (should be 1)     = {norm:.12f}")
```

**Output:**

```
on resonance   (Delta = 0)
  generalized Rabi frequency Omega_R/2pi = 0.050000 GHz
  predicted maximum population           = 1.000000
  observed  maximum population           = 1.000000
   t (ns)   P1 (RK4)   P1 (analytic)   |diff|
     0.0   0.000000     0.000000     0.00e+00
     5.0   0.500000     0.500000     7.77e-16
    10.0   1.000000     1.000000     1.33e-15
    15.0   0.500000     0.500000     1.14e-14
    20.0   0.000000     0.000000     1.26e-28
    30.0   1.000000     1.000000     4.22e-15
    40.0   0.000000     0.000000     4.43e-28

detuned        (Delta = Omega)
  generalized Rabi frequency Omega_R/2pi = 0.070711 GHz
  predicted maximum population           = 0.500000
  observed  maximum population           = 0.500000
   t (ns)   P1 (RK4)   P1 (analytic)   |diff|
     0.0   0.000000     0.000000     0.00e+00
     5.0   0.401425     0.401425     1.28e-15
    10.0   0.316564     0.316564     3.55e-15
    15.0   0.017940     0.017940     4.37e-16
    20.0   0.464554     0.464554     2.55e-15
    30.0   0.069184     0.069184     5.55e-17
    40.0   0.131732     0.131732     2.58e-15

far detuned    (Delta = 2*Omega)
  generalized Rabi frequency Omega_R/2pi = 0.111803 GHz
  predicted maximum population           = 0.200000
  observed  maximum population           = 0.200000
   t (ns)   P1 (RK4)   P1 (analytic)   |diff|
     0.0   0.000000     0.000000     0.00e+00
     5.0   0.193203     0.193203     2.78e-16
    10.0   0.026263     0.026263     1.49e-16
    15.0   0.144247     0.144247     2.83e-15
    20.0   0.091257     0.091257     3.33e-16
    30.0   0.160844     0.160844     8.05e-16
    40.0   0.198471     0.198471     9.44e-16

Pulse calibration on resonance
  pi/2-pulse: t =  5.00 ns -> P1 = 0.500000
  pi-pulse  : t = 10.00 ns -> P1 = 1.000000
  2pi-pulse : t = 20.00 ns -> P1 = 0.000000
  norm at t = 40 ns (should be 1)     = 1.000000000000
```

**Reading the result.** Four observations connect the numbers to the physics.

  * **On resonance the flip is complete.** The population reaches exactly 1 at \\(t = 10\\) ns and returns to 0 at 20 ns. The \\(\pi\\)-pulse is an X gate; the \\(\pi/2\\)-pulse at 5 ns produces the equal superposition used to open a Ramsey sequence.
  * **Detuning caps the amplitude, and no duration recovers it.** At \\(\Delta = \Omega\\) the ceiling is 0.5; at \\(\Delta = 2\Omega\\) it is 0.2. Both match \\(\Omega^2/(\Omega^2 + \Delta^2)\\) exactly. A miscalibrated frequency does not delay the gate — it makes the intended gate unreachable.
  * **Detuning also speeds the oscillation up.** The generalized Rabi frequency \\(\sqrt{\Omega^2 + \Delta^2}\\) rises from 50 to 70.7 to 111.8 MHz. Faster *and* weaker: an off-resonant drive wiggles the state without ever flipping it, which is exactly why a well-detuned neighbouring transition is largely left alone.
  * **The integrator is trustworthy.** RK4 agrees with the closed form to about \\(10^{-15}\\), and the norm is conserved to twelve decimal places. When you write your own dynamics code, always check norm conservation — it catches step-size and sign errors immediately.

Try setting the detuning to the anharmonicity of a transmon and adding a third level to the Hamiltonian: you will see leakage into \\(|2\rangle\\) appear as soon as the pulse gets short, reproducing the gate-speed floor of Section 2.3.

## 2.8 Strengths and Challenges, Honestly

Collecting the chapter into the same qualitative form used in the *Introduction* series:

| | Superconducting circuits |
|---|---|
| **The qubit is** | A microwave-frequency circuit on a chip — an LC oscillator made anharmonic by a Josephson junction, most commonly a transmon |
| **Strengths** | Very fast gates, measured in nanoseconds; fabrication borrows directly from the semiconductor industry, so layouts are lithographically patterned and reproducible; qubit frequency, coupling, and anharmonicity are design parameters rather than gifts of nature; control and readout use mature commercial microwave technology |
| **Challenges** | Requires dilution-refrigerator temperatures in the tens of millikelvin; coherence times are relatively short compared with atomic platforms; coupling is fabricated, so connectivity is typically nearest-neighbour and non-local gates cost SWAPs; qubits are not identical — each is calibrated individually and drifts; wiring heat load into the cold stage is a scaling constraint |

Two of these entries deserve a closing comment, because they are easy to read as verdicts when they are really trade-offs.

**"Short coherence" is not the whole story.** Recall the figure of merit from Chapter 1: what counts is \\(T_2/t_{\text{gate}}\\). Superconducting circuits pay for their fast gates with shorter coherence, and gain back through speed most of what they lose. Comparing the coherence column alone across platforms is exactly the mistake Chapter 1 warned against.

**"Qubits are not identical" is the deep cost of being engineered.** An atom of a given isotope is exactly like every other such atom in the universe; a fabricated circuit is like its neighbour only to within manufacturing tolerance. Every device therefore requires per-qubit calibration, repeated as parameters drift, and this calibration burden grows with device size. It is the flip side of the design freedom that makes the platform attractive in the first place — the same freedom, seen from the cost column.

### 🎯 Exercise Problems

  1. **Why the harmonic ladder fails.** Explain in your own words why equally spaced energy levels make a two-level gate impossible, and estimate how the leakage into \\(|2\rangle\\) should depend on the ratio of pulse bandwidth to anharmonicity.
  2. **The temperature requirement.** Repeat the \\(k_BT\\) versus \\(hf\\) calculation for a qubit frequency of 10 GHz. What operating temperature keeps thermal excitation below \\(10^{-4}\\)? Comment on whether raising the qubit frequency is an attractive route to a warmer machine.
  3. **Detuning tolerance.** Using \\(P_{\max} = \Omega^2/(\Omega^2+\Delta^2)\\), find the detuning, as a fraction of \\(\Omega\\), at which the maximum reachable population falls to 0.99. What does this say about how tightly qubit frequency must be tracked?
  4. **Gate budget with routing.** A circuit requires 40 two-qubit gates between arbitrary pairs on a nearest-neighbour grid, and the compiler inserts on average 3 SWAPs per gate. Using the three-CNOT decomposition, count the total two-qubit gates actually executed, and comment on the effect on circuit fidelity.
  5. **Modify the code.** Extend the simulation to a three-level system by adding a \\(|2\rangle\\) state detuned by the anharmonicity, and measure the leakage population as you shorten the \\(\pi\\)-pulse. Where does the gate-speed floor appear?

## Summary

This chapter examined superconducting circuits, the most industrially developed qubit platform. **Superconductivity** removes resistive dissipation and lets a macroscopic circuit behave as a single quantum object, but a plain **LC oscillator** is useless as a qubit because its energy levels are **equally spaced**, so any pulse driving \\(|0\rangle \to |1\rangle\\) also drives the transitions above it. The **Josephson junction** — two superconductors separated by a thin insulating barrier through which Cooper pairs tunnel, an effect predicted by Josephson in 1962 — supplies nonlinearity without dissipation, turning the circuit into an **artificial atom** with an unevenly spaced ladder whose lowest two levels serve as the qubit. The **transmon** operates deep in the \\(E_J \gg E_C\\) regime, where charge-noise sensitivity falls exponentially while anharmonicity falls only algebraically; the surviving anharmonicity sets a floor on gate duration through **leakage**. Control is by **resonant microwave pulses**: pulse area sets the rotation angle, drive phase sets the axis, and detuning both caps the reachable population at \\(\Omega^2/(\Omega^2+\Delta^2)\\) and speeds the oscillation up — all of which our NumPy Runge-Kutta simulation reproduced to machine precision. Two-qubit gates come from **fixed coupling driven cross-resonantly** or from **tunable couplers**, with the usual trade between hardware simplicity and residual interaction. **Dispersive readout** measures the state through the state-dependent frequency shift of a coupled resonator. Finally, the **millikelvin requirement** follows from \\(k_BT \ll hf\\) and nothing else: at a few gigahertz, thermal excitation is near-total at a few hundred millikelvin and negligible only in the tens of millikelvin, which forces a **dilution refrigerator** and makes **wiring heat load** a genuine scaling constraint.

In the next chapter, we turn to the opposite design philosophy. Trapped ions and neutral atoms use qubits that nature manufactures identically, held in place by electromagnetic fields and light rather than patterned in metal — trading the nanosecond gate for coherence and connectivity that a chip cannot match.

[← Chapter 1: From Physical Qubit to Quantum Computer](<chapter-1.html>) [Chapter 3: Trapped Ions and Neutral Atoms →](<chapter-3.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
