---
title: "Chapter 1: What Makes a Good Qubit"
chapter_title: "Chapter 1: What Makes a Good Qubit"
subtitle: The DiVincenzo Criteria, the Axes of Comparison, and the Common Language of Decoherence
reading_time: 40-45 minutes
difficulty: Intermediate
code_examples: 7
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-hardware-introduction/chapter-1.html>) | Last sync: 2026-08-13

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to Quantum Hardware](<index.html>) > Chapter 1

A quantum computer is a piece of apparatus, and this chapter is about what kind of apparatus it has to be. The sister course [Introduction to Quantum Computing](<../quantum-computing-introduction/index.html>) treated the hardware as a set of parameters — a coherence time, an error rate — and got a long way on that abstraction. The abstraction now has to be opened. Behind every one of those parameters is a physical system with a Hamiltonian, an environment, and a fabrication history, and the last of the three is where a materials researcher has something to contribute that a computer scientist does not.

The chapter has two jobs. The first is to establish what "good" means, which is harder than it sounds: there are at least six independent axes along which a qubit platform can be judged, they are not independent of each other physically, and no platform is best on all of them. The second is to fix a common language — $T_1$, $T_2$, $T_2^\ast$, noise spectral density, Ramsey and echo — that Chapters 2 through 5 will use without redefining. Five of the chapter's seven worked examples — the ones gathered in §1.6 — build a small Bloch-equation laboratory that reproduces the standard coherence measurements from first principles, because these are the measurements every platform reports and they are also, read correctly, materials characterization data.

## Learning Objectives

After completing this chapter, you will be able to:

  * Explain why a good qubit requires two properties that are in physical tension — isolation from the environment and strong coupling to the control field — and how the two families of platforms resolve that tension differently
  * State the five DiVincenzo criteria plus the two communication criteria, and identify for each one which physical property of a material or device it constrains
  * List the six comparison axes used throughout this course and demonstrate, quantitatively, that ranking platforms by any single one of them produces a different winner
  * Define $T_1$, $T_2$ and $T_2^\ast$ precisely, derive the bound $T_2 \le 2T_1$, and explain which part of the environmental noise spectrum each one samples
  * Integrate the Bloch equations numerically to reproduce Rabi oscillations, free-induction decay, Ramsey fringes, the Hahn echo and CPMG dynamical decoupling, and extract the corresponding time constants by fitting
  * Read a coherence measurement as a statement about a material — a static spread of qubit frequencies, a $1/f$ noise amplitude, a defect density — rather than as a device specification

### Conventions and Units

This course follows three conventions without exception, and it is worth reading them once now rather than being surprised later.

**Reduced units.** Hamiltonians are written with $\hbar = 1$, so that energies and angular frequencies are the same object. Where a physical energy is meant, the factor is restored explicitly: a transmon transition at $\omega_q/2\pi = 5$ GHz has energy $\hbar\omega_q = h \times 5\ \mathrm{GHz} = 20.7\ \mu\mathrm{eV}$.

**Angular versus cyclic frequency.** Symbols such as $\Omega$ (Rabi frequency) and $\Delta$ (detuning) are *angular* frequencies in rad/s. Quoted numbers are *cyclic* frequencies in Hz, written as $\Omega/2\pi$ or $\Delta/2\pi$. Every factor of $2\pi$ in the code is explicit for this reason. Mixing the two is the single most common numerical error in this subject; it costs a factor of 6.28 and produces plots that look plausible.

**Field-specific energy units.** Superconducting circuits are quoted in GHz, trapped ions and neutral atoms in MHz, spin qubits in MHz or in tesla of applied field. Code Example 1 gives the conversion table once, and the rest of the course uses whichever unit the relevant literature uses.

**Qubit ordering and gate symbols** follow the sister course exactly: qubit 0 is the leftmost and most significant bit, and $X, Y, Z, H$, CNOT mean what they mean there. Chapter 1 needs only single-qubit language, but Chapters 2 through 5 assume the convention.

* * *

## 1.1 Why There Are So Many Kinds of Qubit

### A Requirement That Contradicts Itself

Write down what a qubit must do, in the order a physicist would:

  1. It must have two levels that are addressable — resolvable from every other level of the system, and reachable by a control field.
  2. The relative phase between those levels must survive long enough for a computation.
  3. A control field must be able to rotate the state through a full $\pi$ in a time short compared with that survival.

Requirements 2 and 3 pull in opposite directions. A system that responds strongly to an applied field also responds strongly to the *fluctuating* fields it did not ask for; the fluctuation-dissipation theorem is the formal statement that the same coupling constant governs both. A perfectly isolated two-level system has infinite coherence time and cannot be operated. This is not an engineering inconvenience to be designed away — it is the central design tension of the field, and every platform is a particular compromise on it.

There is a partial escape, and it is the reason the tension does not simply forbid quantum computing: the coupling can be made *frequency-selective*. A qubit that couples strongly to a field at $\omega_q$ but weakly to fields at other frequencies is both controllable and protected, provided the environment has little spectral weight at $\omega_q$. Every platform in this course exploits this, and Section 1.4 makes it quantitative. It is also why the shape of the environmental noise spectrum — not just its magnitude — is the quantity that matters.

### Two Families of Answers

The proposals divide cleanly into two families, and the division is essentially about who chose the Hamiltonian.

**Natural qubits** use levels that nature already provides: hyperfine or optical transitions in atoms and ions, nuclear or electron spins, photon polarization. Their advantages follow directly from that origin. Every $^{171}\mathrm{Yb}^{+}$ ion in the universe has the same hyperfine splitting to fifteen digits, so there is no fabrication disorder and no device-to-device calibration problem in principle. The levels are weakly coupled to almost everything, which is why coherence times of seconds and longer are achievable. The disadvantages also follow: the Hamiltonian is not adjustable, the qubits must be trapped and held in ultra-high vacuum, and the control apparatus — lasers, optics, magnetic shielding — does not shrink when the qubit count grows.

**Engineered qubits** build an artificial two-level system out of a solid: a superconducting circuit, a gate-defined quantum dot, a defect centre in a crystal. Here the Hamiltonian is a design parameter, set by lithography and film thickness rather than by nature. Coupling strengths can be made large, gates fast, and the whole device fabricated by processes that already exist at wafer scale. The price is that the qubit inherits its host material. It sits a few tens of nanometres from an amorphous oxide full of two-level defects, from an interface with dangling bonds, from a substrate with nuclear spins — and its coherence time is set by those, not by anything in the design. Fabrication also introduces parameter spread: nominally identical devices are not identical, and the distribution has a tail.

| | Natural qubits | Engineered qubits |
| --- | --- | --- |
| Hamiltonian | fixed by nature | set by design |
| Reproducibility | exact, by definition | limited by process control |
| Coherence | long; limited by apparatus | short; limited by host material |
| Gate speed | slow (kHz-MHz) | fast (MHz-GHz) |
| Integration | optics and vacuum do not scale down | wafer-scale processes already exist |
| Dominant research problem | control complexity | materials |

This table is the reason the course exists in this form. For the right-hand column, the bottleneck is a materials problem, and it is a *hard* materials problem: amorphous dielectric loss at millikelvin temperatures and gigahertz frequencies, interface defect densities, isotopic purity, superconductor-semiconductor epitaxy. For the left-hand column the bottleneck is elsewhere, but even there the trap surface, the vacuum, and the optical coatings are materials.

### A Short Genealogy

The proliferation is historical as well as physical. Each of the platforms in this course was proposed as a solution to a specific difficulty in the ones before it.

| Year | Proposal | The physical idea |
| --- | --- | --- |
| 1995 | Cirac and Zoller: trapped-ion gate | Use a shared vibrational mode of an ion crystal as a bus between internal states |
| 1997 | Gershenfeld, Chuang, Cory: NMR ensembles | Nuclear spins in molecules already have long $T_2$; use them, at the cost of working with an ensemble |
| 1997-2003 | Kitaev: topological quantum computation | Store information in non-local degrees of freedom, so that local noise cannot read or corrupt it |
| 1998 | Loss and DiVincenzo: quantum-dot spins | An electron spin in a semiconductor, with gates from the exchange interaction — compatible with existing lithography |
| 1999 | Nakamura, Pashkin, Tsai: superconducting charge qubit | A Josephson junction makes a macroscopic circuit behave as an artificial atom |
| 2000 | Jaksch and co-workers: Rydberg blockade | Excite one atom to a huge-dipole Rydberg state and it forbids its neighbours from being excited |
| 2001 | Knill, Laflamme, Milburn: linear optics | Photons do not interact, but measurement plus post-selection can substitute for an interaction |
| 2007 | Koch and co-workers: the transmon | Shunt a charge qubit with a large capacitance to make it insensitive to charge noise, accepting weaker anharmonicity |

Two of these entries are worth pausing on, because they are the pattern the whole field follows. The transmon exists because charge noise — a materials problem, from moving charges in oxides and on surfaces — killed its predecessor; the fix was to redesign the Hamiltonian so that the qubit frequency no longer depends on charge. Topological qubits exist because *all* local noise is a materials problem; the proposed fix is to store the information where no local operator can reach it. Both are responses to decoherence, and both illustrate that a platform is best understood as an answer to a specific noise channel.

### What the Sister Course Left Out

[Introduction to Quantum Computing](<../quantum-computing-introduction/index.html>) is explicit that it is not a hardware course: superconducting qubits, trapped ions and neutral atoms appear there only where their error characteristics matter. This course fills exactly that gap, and the correspondence between the two is direct.

| In the algorithms course | In this course |
| --- | --- |
| A qubit is a normalized complex vector | A two-level subspace of a real physical spectrum, with leakage out of it |
| A gate is a unitary matrix | A shaped control pulse, with calibration error and finite duration |
| $T_1$ and $T_2$ are input parameters of a noise model | Dependent variables, set by defects, interfaces and isotopes |
| Connectivity is a graph | A capacitor network, a shared phonon mode, or a blockade radius |
| Error correction has an overhead | The overhead is what makes the materials problem urgent |

Reading the two courses together, the useful question changes from "what can a quantum computer do" to "what would have to improve, physically, for it to do that".

* * *

## 1.2 The DiVincenzo Criteria

In 2000 David DiVincenzo wrote down the minimal list of capabilities a physical system must have to implement quantum computation. It remains the standard checklist, because it is stated at the level of physics rather than of technology. Five criteria concern computation; two more concern communication.

### The Five

**1. A scalable physical system with well-characterized qubits.** Two requirements hide in one sentence. *Well-characterized* means the Hamiltonian is known: the level splitting, the couplings to control fields, the couplings to other qubits, and — crucially — the couplings to levels *outside* the computational subspace. A transmon is not a two-level system; it is a weakly anharmonic oscillator whose third level is only a few hundred MHz away, and a pulse that is too fast will populate it. Such **leakage** is not describable as a qubit error at all, which is why anharmonicity is a first-class design parameter in Chapter 2. *Scalable* means that adding qubits does not require a new invention each time — and in practice the constraint is rarely the qubits themselves but the control channels, the wiring heat load, or the laser power.

**2. The ability to initialize the state to a simple fiducial state.** Usually $|00\ldots0\rangle$. There are two physical routes. **Thermal initialization** works if $k_BT \ll \hbar\omega_q$, which for a 5 GHz microwave qubit means a dilution refrigerator: the requirement is not merely "cold" but $T \lesssim 30$ mK, as Code Example 1 shows. **Dissipative initialization** — optical pumping, active reset — works at any temperature, because it drives the system into a dark state instead of waiting for equilibrium. This is why a trapped ion with a 12.6 GHz hyperfine qubit operates in a room-temperature vacuum chamber while a transmon with a lower frequency needs millikelvin. Error correction sharpens the requirement considerably: fresh, high-fidelity ancilla qubits are consumed continuously, so initialization must be fast *and* repeatable, not merely possible.

**3. Long relevant decoherence times, much longer than the gate operation time.** The operative word is *relevant*. What matters is the dimensionless ratio $T_2/t_\mathrm{gate}$, not either number alone. A platform with $T_2 = 1$ s and 100 $\mu$s gates and a platform with $T_2 = 100\ \mu$s and 10 ns gates have comparable budgets, differing by a factor of a few. Section 1.3 works this out, and it is the single most useful number to extract from any hardware claim.

**4. A universal set of quantum gates.** Any set that generates $SU(2^n)$: in practice arbitrary single-qubit rotations plus one entangling two-qubit gate. The two-qubit gate is the hard one, and its physical mechanism differs completely between platforms — direct capacitive coupling in circuits, a shared motional mode in ion traps, the Rydberg blockade for neutral atoms, the exchange interaction for spins. Chapters 2 through 5 each devote a section to it, because the gate mechanism determines the connectivity graph and the gate time simultaneously.

**5. Qubit-specific measurement capability.** Read out one named qubit, with high fidelity, without disturbing the others. Two properties beyond fidelity matter in practice. **Speed**: error correction requires many measurement rounds within a coherence time, so a readout that takes a large fraction of $T_2$ is useless even at perfect fidelity. **Non-destructiveness**: an atom that is lost from the trap on measurement, or an ion that must be re-cooled, imposes a reload cycle that dominates the duty factor.

### The Two More

**6. The ability to interconvert stationary and flying qubits.** A network needs to move quantum information between modules, and the carrier is almost always a photon. The physical problem is impedance matching between a solid-state qubit at GHz and an optical photon at hundreds of THz.

**7. The ability to transmit flying qubits faithfully between locations.** Loss in fibre, loss in free space, and the impossibility of amplifying an unknown quantum state.

These two are what make **modular** architectures possible, and they are the reason photonic platforms remain interesting even where they are poor computers: the same physics that makes photons bad at interacting makes them excellent at travelling.

### What the Criteria Do Not Say

The list is necessary, not sufficient, and reading it as a scorecard is a mistake in three ways.

  * **It contains no thresholds.** "Long" and "high-fidelity" are not numbers. The numbers come from the error-correction threshold theorem, and they depend on the code, the noise model, and the acceptable overhead.
  * **It says nothing about error correlations.** A single cosmic-ray impact that produces quasiparticles across a whole chip, or a laser intensity fluctuation common to every atom, violates the independent-error assumption on which the thresholds are derived. Correlated errors are a materials and engineering question that the criteria do not raise.
  * **It says nothing about cost.** Refrigeration power, laser count, control electronics per qubit, wafer yield. These are what determine whether a demonstration becomes a machine.

Every criterion nonetheless maps onto a physical property that a materials scientist can influence, which is worth tabulating explicitly.

| Criterion | Physical content | Where materials science enters |
| --- | --- | --- |
| 1. Well-characterized, scalable | known Hamiltonian; small parameter spread; controlled leakage | film thickness and junction area uniformity; isotopic and chemical purity |
| 2. Initialization | $k_BT \ll \hbar\omega_q$, or a fast dissipative channel | thermal anchoring; residual heat load; trap-surface cleanliness |
| 3. $T_2 \gg t_\mathrm{gate}$ | noise spectral density at DC and at $\omega_q$ | defect densities, dielectric loss, nuclear spin bath, quasiparticles |
| 4. Universal gates | a controllable interaction, and anharmonicity | junction materials; substrate permittivity; surface electric-field noise |
| 5. Measurement | a strong, fast, qubit-specific coupling to a meter | resonator quality factor; detector efficiency; amplifier noise |
| 6-7. Flying qubits | coherent transduction and low-loss transmission | electro-optic and piezoelectric materials; fibre and coating loss |

* * *

## 1.3 Six Axes, and Why They Cannot Be Collapsed Into One

The DiVincenzo criteria are pass/fail. Comparing platforms that all pass requires continuous axes, and this course uses six.

| Axis | What it measures | Physical origin of the limit |
| --- | --- | --- |
| **Coherence time** | $T_1$, $T_2$, $T_2^\ast$ | noise spectral density of the environment at the relevant frequencies |
| **Gate fidelity and speed** | error per gate; $t_\mathrm{gate}$ | anharmonicity or mode structure sets the speed limit; calibration and decoherence set the error |
| **Connectivity** | which pairs can interact directly | the physical mechanism of the two-qubit gate |
| **Reproducibility and yield** | spread of qubit parameters; fraction of working devices | fabrication disorder, or its absence for natural qubits |
| **Operating temperature** | $T$ required, and the refrigeration budget | the qubit energy scale relative to $k_BT$ |
| **Scalability** | what breaks when $n$ grows | control channels, wiring heat load, laser power, footprint |

### Why the Axes Fight Each Other

They are not independent, and the couplings between them are physical rather than accidental.

  * **Speed against coherence.** Faster gates need stronger coupling to the control field, and by fluctuation-dissipation the same coupling admits more noise. The escape — frequency selectivity — is only partial, because a shorter pulse has a broader spectrum and therefore samples more of the noise.
  * **Connectivity against speed.** All-to-all connectivity is usually obtained by coupling every qubit to a shared bosonic mode. The resulting gate rate is suppressed by the coupling to that mode, and it degrades as more modes crowd into the same bandwidth. Nearest-neighbour connectivity keeps the gate fast, and pays with SWAP networks.
  * **Temperature against energy scale.** A qubit splitting of 5 GHz demands $T \lesssim 30$ mK for thermal initialization; a splitting of 500 THz is effectively at zero temperature in a warm room. But a large splitting means the control field is a laser, with its own noise, alignment and power budget.
  * **Reproducibility against tunability.** A qubit whose frequency is set by design can be tuned into resonance with its neighbour; it can also drift and be disordered. A qubit whose frequency is set by nature cannot be tuned at all — which removes a control knob along with the disorder.
  * **Scalability against everything.** Each of the other five is measured on one or two qubits. Scalability is the axis on which single-qubit excellence can be irrelevant, and it is usually limited by something unglamorous: the number of coaxial lines that fit through a refrigerator, the heat load of those lines, the number of laser beams that can be pointed independently.

### Code Example 1: Energy Scales, Units, and Who Needs a Dilution Refrigerator

Before any comparison, one conversion table. The transition frequency of a qubit fixes its energy, its equivalent temperature, and therefore whether it can be initialized by cooling at all.

```python
import numpy as np

h = 6.62607015e-34      # Planck constant, J s
kB = 1.380649e-23       # Boltzmann constant, J/K
eV = 1.602176634e-19    # elementary charge, C  (1 eV in joules)


def thermal_population(f_hz: float, T_kelvin: float) -> float:
    """Excited-state population of a two-level system in thermal equilibrium."""
    x = h * f_hz / (kB * T_kelvin)
    return np.exp(-x) / (1.0 + np.exp(-x))


# One conversion table, used for the rest of the course.
print("Unit conversions for a transition frequency (h f):")
print(f"{'f':>10}{'energy':>14}{'h f / kB':>14}{'wavenumber':>14}")
print("-" * 52)
for label, f in [("1 MHz", 1e6), ("1 GHz", 1e9), ("1 THz", 1e12),
                 ("500 THz", 5e14)]:
    E_ueV = h * f / eV * 1e6
    T_K = h * f / kB
    nu_cm = f / 2.99792458e10
    print(f"{label:>10}{E_ueV:>11.4g} ueV{T_K:>11.4g} K{nu_cm:>11.4g} 1/cm")

# Representative qubit transitions. The frequencies are the *physics* of each
# level structure (a hyperfine splitting, a Josephson-circuit plasma frequency,
# a Zeeman splitting), not a device specification.
qubits = [
    ("Transmon, microwave",        5.0e9,   "thermal"),
    ("Electron spin, B = 1 T",     28.0e9,  "thermal"),
    ("NV centre, zero field",      2.87e9,  "optical pumping"),
    ("Rb-87 hyperfine",            6.835e9, "optical pumping"),
    ("Yb-171+ hyperfine",          12.64e9, "optical pumping"),
    ("Ca-40+ optical, 729 nm",     4.11e14, "optical pumping"),
]

print("\nThermal excited-state population of a two-level system:")
header = (f"{'qubit transition':<26}{'f (GHz)':>12}{'hf/kB (K)':>11}"
          f"{'300 K':>10}{'4 K':>10}{'100 mK':>10}{'10 mK':>10}")
print(header)
print("-" * len(header))
for name, f, _ in qubits:
    pops = [thermal_population(f, T) for T in (300.0, 4.0, 0.1, 0.010)]
    print(f"{name:<26}{f/1e9:>12.4g}{h*f/kB:>11.4g}"
          + "".join(f"{p:>10.2e}" for p in pops))

print("\nHow each platform actually reaches its fiducial state:")
for name, f, route in qubits:
    if route == "thermal":
        T_needed = h * f / (kB * np.log(1.0 / 1e-3 - 1.0))   # P_exc = 1e-3
        print(f"  {name:<26} thermal        needs T < {T_needed*1e3:6.1f} mK "
              f"for P_exc < 1e-3")
    else:
        print(f"  {name:<26} optical pumping temperature of the apparatus is "
              f"irrelevant")

# The drive side of the same energy scale: what one Rabi period costs.
print("\nDrive strength and pulse duration (Omega = 2 pi f_Rabi):")
print(f"{'f_Rabi':>10}{'Omega (rad/s)':>16}{'pi pulse':>14}")
print("-" * 40)
for label, f_rabi in [("1 kHz", 1e3), ("100 kHz", 1e5),
                      ("1 MHz", 1e6), ("50 MHz", 5e7)]:
    Omega = 2 * np.pi * f_rabi
    print(f"{label:>10}{Omega:>16.4g}{np.pi/Omega*1e6:>11.4g} us")
```

```text
Unit conversions for a transition frequency (h f):
         f        energy      h f / kB    wavenumber
----------------------------------------------------
     1 MHz   0.004136 ueV  4.799e-05 K  3.336e-05 1/cm
     1 GHz      4.136 ueV    0.04799 K    0.03336 1/cm
     1 THz       4136 ueV      47.99 K      33.36 1/cm
   500 THz  2.068e+06 ueV    2.4e+04 K  1.668e+04 1/cm

Thermal excited-state population of a two-level system:
qubit transition               f (GHz)  hf/kB (K)     300 K       4 K    100 mK     10 mK
-----------------------------------------------------------------------------------------
Transmon, microwave                  5       0.24  5.00e-01  4.85e-01  8.32e-02  3.79e-11
Electron spin, B = 1 T              28      1.344  4.99e-01  4.17e-01  1.46e-06  4.37e-59
NV centre, zero field             2.87     0.1377  5.00e-01  4.91e-01  2.01e-01  1.04e-06
Rb-87 hyperfine                  6.835      0.328  5.00e-01  4.80e-01  3.63e-02  5.67e-15
Yb-171+ hyperfine                12.64     0.6066  4.99e-01  4.62e-01  2.31e-03  4.51e-27
Ca-40+ optical, 729 nm        4.11e+05  1.972e+04  2.79e-29  0.00e+00  0.00e+00  0.00e+00

How each platform actually reaches its fiducial state:
  Transmon, microwave        thermal        needs T <   34.7 mK for P_exc < 1e-3
  Electron spin, B = 1 T     thermal        needs T <  194.6 mK for P_exc < 1e-3
  NV centre, zero field      optical pumping temperature of the apparatus is irrelevant
  Rb-87 hyperfine            optical pumping temperature of the apparatus is irrelevant
  Yb-171+ hyperfine          optical pumping temperature of the apparatus is irrelevant
  Ca-40+ optical, 729 nm     optical pumping temperature of the apparatus is irrelevant

Drive strength and pulse duration (Omega = 2 pi f_Rabi):
    f_Rabi   Omega (rad/s)      pi pulse
----------------------------------------
     1 kHz            6283        500 us
   100 kHz       6.283e+05          5 us
     1 MHz       6.283e+06        0.5 us
    50 MHz       3.142e+08       0.01 us
```

**What to look for.** The first table is the conversion to memorize: 1 GHz corresponds to 4.14 $\mu$eV and to 48 mK. That last number is the whole reason superconducting quantum computing happens in dilution refrigerators — a 5 GHz qubit has $\hbar\omega_q/k_B = 0.24$ K, so a 4 K helium bath leaves it 48% excited, and only below about 35 mK is the residual excitation below $10^{-3}$. The optical row is the opposite extreme: $\hbar\omega/k_B \approx 2\times10^4$ K, and the thermal population at room temperature is $10^{-29}$, printed as exact zeros at 4 K and below because $e^{-5000}$ underflows double precision. That is not a rounding artifact worth fixing; it is the statement that an optical qubit is in its ground state, absolutely, at any temperature a laboratory will ever see.

The second block is the important qualification. A $^{171}\mathrm{Yb}^{+}$ hyperfine qubit at 12.6 GHz would be 46% excited at 4 K, and yet ion traps operate at room temperature. Thermal equilibrium is simply not how they are initialized: optical pumping drives the ion into a specific hyperfine state in microseconds, and the temperature of the apparatus never enters. DiVincenzo's criterion 2 is a statement about *a* mechanism, not about cooling — and confusing the two is the origin of the widespread belief that all quantum computers need refrigerators.

### Code Example 2: Three Rankings of the Same Four Platforms

The claim that a single figure of merit does not exist deserves a demonstration rather than an assertion. The numbers below are order-of-magnitude scales fixed by the physics of each platform — a transmon gate is nanoseconds because its anharmonicity is a few hundred MHz; an ion gate is tens to hundreds of microseconds because it is mediated by a motional mode of a few MHz — and they are given to one significant figure precisely so that no reading of the table depends on a record or a device specification.

```python
import numpy as np

# Order-of-magnitude scales set by the physics of each platform, quoted to one
# significant figure. These are NOT device specifications and NOT records: the
# transmon gate time is short because the anharmonicity is a few hundred MHz,
# the ion gate time is long because it is mediated by a motional mode of a few
# MHz, and so on. Only the decades matter for the argument below.
platforms = [
    # name,                T2 (s), t_2q (s), lattice,          lanes
    ("Superconducting",     1e-4,   5e-8,    "2D nearest-nbr",  50),
    ("Trapped ion",         1e0,    1e-4,    "all-to-all",       1),
    ("Neutral atom",        1e-1,   5e-7,    "reconfigurable",  10),
    ("Semiconductor spin",  1e-3,   1e-7,    "1D nearest-nbr",  50),
]

n_qubits = 100        # register size assumed for the routing estimate
n_gates = 1000        # two-qubit gates on arbitrary pairs required by the task


def routing_overhead(lattice: str, n: int) -> float:
    """Extra two-qubit gates per logical gate, from SWAP networks.

    A SWAP costs three CNOTs. On a 2D lattice the mean distance between two
    random sites is of order sqrt(n)/2; on a line it is of order n/3.
    All-to-all and reconfigurable couplings need no SWAPs at all.
    """
    if lattice == "2D nearest-nbr":
        return 1.0 + 3.0 * np.sqrt(n) / 2.0
    if lattice == "1D nearest-nbr":
        return 1.0 + 3.0 * n / 3.0
    return 1.0


rows = []
for name, T2, t2q, lattice, lanes in platforms:
    budget = T2 / t2q                       # gates that fit inside coherence
    ovh = routing_overhead(lattice, n_qubits)
    phys_gates = n_gates * ovh              # gates actually executed
    layers = phys_gates / lanes             # gates run in parallel lanes
    wall = layers * t2q                     # wall-clock time of the circuit
    rows.append(dict(name=name, T2=T2, t2q=t2q, lattice=lattice, lanes=lanes,
                     budget=budget, ovh=ovh, wall=wall, ratio=wall / T2))

header = (f"{'platform':<20}{'T2 (s)':>9}{'t_2q (s)':>10}{'T2/t_2q':>10}"
          f"  {'coupling':<16}{'SWAP ovh':>9}{'lanes':>7}")
print(header)
print("-" * len(header))
for r in rows:
    print(f"{r['name']:<20}{r['T2']:>9.0e}{r['t2q']:>10.0e}{r['budget']:>10.0f}"
          f"  {r['lattice']:<16}{r['ovh']:>9.0f}{r['lanes']:>7d}")


def ranking(key, reverse):
    order = sorted(rows, key=lambda r: r[key], reverse=reverse)
    return " > ".join(r["name"] for r in order)


print("\nThree defensible single-number rankings of the same four platforms:")
print(f"  by coherence time T2      : {ranking('T2', True)}")
print(f"  by gate speed 1/t_2q      : {ranking('t2q', False)}")
print(f"  by gate budget T2/t_2q    : {ranking('budget', True)}")

print(f"\nA workload: {n_gates} two-qubit gates on arbitrary pairs of "
      f"{n_qubits} qubits.")
print(f"{'platform':<20}{'gates run':>12}{'layers':>10}{'wall clock':>14}"
      f"{'wall/T2':>11}")
print("-" * 67)
for r in rows:
    print(f"{r['name']:<20}{n_gates*r['ovh']:>12.0f}"
          f"{n_gates*r['ovh']/r['lanes']:>10.0f}"
          f"{r['wall']*1e3:>11.3g} ms{r['ratio']:>11.4f}")

print(f"\n  by circuit time / T2      : {ranking('ratio', False)}")

# What is one improvement worth, measured against another?
print("\nTrade-off, made explicit, for the superconducting row:")
base = rows[0]
variants = [("baseline", base["T2"], base["t2q"], base["ovh"]),
            ("10x longer T2", 10 * base["T2"], base["t2q"], base["ovh"]),
            ("10x faster gates", base["T2"], base["t2q"] / 10, base["ovh"]),
            ("all-to-all coupling", base["T2"], base["t2q"], 1.0)]
for label, T2, t2q, ovh in variants:
    wall = n_gates * ovh / base["lanes"] * t2q
    print(f"  {label:<22} wall/T2 = {wall/T2:8.4f}")
print(f"  Removing the SWAP network is worth a factor {base['ovh']:.0f} here, "
      f"i.e. more than a decade of T2.")
```

```text
platform               T2 (s)  t_2q (s)   T2/t_2q  coupling         SWAP ovh  lanes
-----------------------------------------------------------------------------------
Superconducting         1e-04     5e-08      2000  2D nearest-nbr         16     50
Trapped ion             1e+00     1e-04     10000  all-to-all              1      1
Neutral atom            1e-01     5e-07    200000  reconfigurable          1     10
Semiconductor spin      1e-03     1e-07     10000  1D nearest-nbr        101     50

Three defensible single-number rankings of the same four platforms:
  by coherence time T2      : Trapped ion > Neutral atom > Semiconductor spin > Superconducting
  by gate speed 1/t_2q      : Superconducting > Semiconductor spin > Neutral atom > Trapped ion
  by gate budget T2/t_2q    : Neutral atom > Trapped ion > Semiconductor spin > Superconducting

A workload: 1000 two-qubit gates on arbitrary pairs of 100 qubits.
platform               gates run    layers    wall clock    wall/T2
-------------------------------------------------------------------
Superconducting            16000       320      0.016 ms     0.1600
Trapped ion                 1000      1000        100 ms     0.1000
Neutral atom                1000       100       0.05 ms     0.0005
Semiconductor spin        101000      2020      0.202 ms     0.2020

  by circuit time / T2      : Neutral atom > Trapped ion > Superconducting > Semiconductor spin

Trade-off, made explicit, for the superconducting row:
  baseline               wall/T2 =   0.1600
  10x longer T2          wall/T2 =   0.0160
  10x faster gates       wall/T2 =   0.0160
  all-to-all coupling    wall/T2 =   0.0100
  Removing the SWAP network is worth a factor 16 here, i.e. more than a decade of T2.
```

**What to look for.** Three defensible single-number rankings of the same four platforms, and all three disagree. By coherence time the trapped ion wins by four decades over the superconducting circuit. By gate speed the order is exactly reversed. By the dimensionless ratio $T_2/t_{2q}$ — which is what DiVincenzo's criterion 3 actually asks for — the neutral atom leads and the trapped ion and the spin qubit tie. Any statement of the form "platform X is ahead" therefore has to name its axis, and most public statements do not.

The workload block adds the axis that the ratio misses. Routing an algorithm that needs arbitrary pairs onto a nearest-neighbour lattice costs SWAP gates: with the mean-distance estimate used here, a factor of 16 on a 2D lattice of 100 qubits and a factor of 101 on a line. That overhead is why the last ranking differs again from the third. The final block quantifies the trade-off directly: for this workload, removing the SWAP network entirely is worth a factor of 16, which is more than a decade of coherence time. Connectivity is not a secondary consideration.

Two honest caveats belong with this example. The parallelism model — the `lanes` column — is crude, and it is doing real work in the result: a superconducting chip can run many gates simultaneously in different regions, a single ion chain fundamentally cannot, and the true numbers depend on the algorithm's structure. And the estimate ignores gate *error*, treating a circuit as feasible if it finishes inside $T_2$. Chapter 5 assembles the honest version of this table, with the physical constraints in place of the numbers.

* * *

## 1.4 The Common Language of Decoherence

Every platform in Chapters 2 through 5 reports the same three time constants. They are defined here once, in a form that does not depend on the physical realization, and used unchanged for the rest of the course.

### The Two Ways to Lose Information

A qubit state is a point on or in the Bloch sphere, and there are exactly two ways for it to degrade. The **longitudinal** component $z = \langle Z\rangle$ can relax towards thermal equilibrium, which requires exchanging energy with the environment. The **transverse** components $x$ and $y$ can shrink, which requires only that the phase becomes uncertain — no energy has to go anywhere.

**$T_1$, the energy relaxation time**, governs the first:

$$ z(t) = z_\mathrm{eq} + \left(z(0) - z_\mathrm{eq}\right) e^{-t/T_1} $$

Because it is an energy exchange, $1/T_1$ is a golden-rule rate. Schematically, if the environment couples to the qubit through an operator $\hat{A}$ with noise spectral density $S_A$,

$$ \frac{1}{T_1} \;\propto\; \left| \langle 0 | \hat{A} | 1 \rangle \right|^2 S_A(\omega_q) $$

The essential feature is the argument: **$T_1$ samples the noise at the qubit frequency**, in the GHz range for microwave qubits. A material that is quiet at kHz can be loud at 5 GHz, and $T_1$ is the measurement that finds out.

**$T_2$, the coherence time**, governs the second. Two mechanisms contribute. Losing the excitation destroys the phase as a side effect, contributing half of the relaxation rate; genuine **pure dephasing**, at rate $1/T_\varphi$, contributes the rest:

$$ \frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\varphi} $$

Since $1/T_\varphi \ge 0$, this immediately gives the bound

$$ T_2 \le 2T_1 $$

which is worth remembering as a sanity check: a reported $T_2$ larger than $2T_1$ is a mistake, in the measurement or in the arithmetic. The limit $T_2 = 2T_1$ describes a qubit whose only noise is energy relaxation, and it is a genuine limit that good superconducting devices approach.

**$T_2^\ast$, the inhomogeneous coherence time**, is what a straightforward experiment actually returns. Suppose the qubit frequency is not exactly the same on every repetition of the experiment — because a nearby nuclear spin bath has reoriented, or a trapped charge has moved, or the magnetic field drifted. Each shot then accumulates a different phase, and averaging over shots erases the coherence even though every individual shot was perfectly coherent. For a Gaussian distribution of static detunings with standard deviation $\sigma$,

$$ \left\langle e^{i\Delta t} \right\rangle = e^{i\bar{\Delta}t}\, e^{-\sigma^2 t^2/2} \quad \Longrightarrow \quad T_{2,\mathrm{inh}}^\ast = \frac{\sqrt{2}}{\sigma} $$

and the total measured decay is the product of this Gaussian and the intrinsic decay. Note the shape: **a static spread of frequencies gives a Gaussian decay, while a bath that is white in the band the sequence samples gives an exponential one**, and the shape of a measured decay curve is therefore diagnostic in itself. The corollary matters for everything below: an exponential envelope is evidence about the *spectrum* of the noise, not about which time constant is being measured. Code Example 7 shows a Hahn echo — a genuine $T_2$ measurement — decaying as a Gaussian, because $1/f$ noise is anything but white.

$T_2^\ast$ is not a property of the qubit alone. It depends on how long the averaging took, because slower drifts contribute only if the experiment lasted long enough to see them. This sounds like a defect of the definition and is in fact a physical statement about $1/f$ noise, which Code Example 7 makes precise.

| Symbol | Name | What decays | Typical decay shape | What it measures about the material |
| --- | --- | --- | --- | --- |
| $T_1$ | energy relaxation | $\langle Z\rangle$ towards equilibrium | exponential | noise power at $\omega_q$: quasiparticles, lossy dielectrics, phonons |
| $T_\varphi$ | pure dephasing | transverse components, no energy loss | exponential for white noise | fluctuating noise power near DC |
| $T_2$ | coherence | transverse, after an echo | exponential for white noise, $\exp[-(t/T_2)^2]$ under $1/f$ | combination of the two above |
| $T_2^\ast$ | inhomogeneous coherence | transverse, without an echo | Gaussian | *static* spread of qubit frequencies |

The "typical decay shape" column carries a caveat that is worth stating once and reusing. Exponential decay of a transverse component follows from noise that is *white* over the frequencies the sequence is sensitive to; it is a statement about the environment, not a definition of $T_2$. For the $1/f$ noise that solid-state qubits actually see, the echo envelope is Gaussian as well: in Code Example 7 the $N = 1$ (Hahn echo) curve is reproduced by $\exp[-(t/T_2)^2]$ with $T_2 = 0.683\ \mu$s to a few per cent over the first decade of decay — 0.593 predicted against 0.586 measured at $t = 0.494\ \mu$s — where a pure exponential would predict 0.485. Only the $T_2^\ast$ row's Gaussian is shape-*defining*; the other rows are shape-*conditional*.

### Noise as a Spectral Density

The three time constants are summary statistics of one underlying object. Let the instantaneous qubit frequency fluctuate as $\omega_q(t) = \bar{\omega}_q + \delta\omega(t)$, and describe $\delta\omega$ by its one-sided power spectral density $S(f)$, normalized so that

$$ \left\langle \delta\omega^2 \right\rangle = \int_0^\infty S(f)\, df $$

A pulse sequence accumulates the phase $\varphi(t) = \int_0^t s(t')\,\delta\omega(t')\,dt'$, where $s(t) = \pm1$ records the sign flips imposed by any $\pi$ pulses. For Gaussian noise the coherence is $C(t) = |\langle e^{i\varphi}\rangle| = e^{-\langle\varphi^2\rangle/2}$, and the mean-square phase is exactly

$$ \left\langle \varphi^2(t) \right\rangle = \int_0^\infty df\; S(f)\, \left| \tilde{s}(f,t) \right|^2 , \qquad \tilde{s}(f,t) = \int_0^t s(t')\, e^{-2\pi i f t'}\, dt' $$

The function $|\tilde{s}(f,t)|^2$ is the **filter function** of the sequence. It says which frequencies of the environment the experiment is sensitive to, and it is entirely under the experimenter's control. Two cases are worth having in closed form:

$$ \left| \tilde{s}_\mathrm{FID}(f,t) \right|^2 = \frac{\sin^2(\pi f t)}{(\pi f)^2}, \qquad \left| \tilde{s}_\mathrm{echo}(f,t) \right|^2 = \frac{4\sin^4(\pi f t/2)}{(\pi f)^2} $$

Free induction, with no $\pi$ pulses, has $|\tilde{s}|^2 \to t^2$ as $f \to 0$: it is maximally sensitive to the slowest drifts. The echo, with one $\pi$ pulse at the centre, has $|\tilde{s}|^2 \to \pi^2 f^2 t^4/4$: it is *blind* at DC, and its first passband sits near $f \approx 1/2t$. That single difference is the entire mechanism of dynamical decoupling.

### $1/f$ Noise and Two-Level Fluctuators

Solid-state qubits almost invariably see

$$ S(f) = \frac{A}{f^{\alpha}}, \qquad \alpha \approx 1 $$

over many decades. The near-universality has a standard microscopic explanation: an ensemble of independent **two-level fluctuators** (TLS), each switching randomly between two configurations with its own rate $\gamma$. A single fluctuator produces a Lorentzian spectrum with a corner at $\gamma$; a distribution of rates that is uniform in $\log\gamma$ — which is what a distribution of thermally activated barrier heights gives — sums to $1/f$ over the corresponding range.

This matters here for one reason: **the fluctuators are defects, so $A$ is a materials parameter**. In superconducting circuits they are believed to be atomic-scale configurational defects in amorphous oxides at surfaces, interfaces and junction barriers. In semiconductor qubits they are charge traps at the oxide interface, and the nuclear spins of $^{29}\mathrm{Si}$. On trap electrodes they are patch potentials from adsorbates. Chapters 2 and 5 go into each case; the common thread is that lowering $A$ means changing what the device is made of, or how it was grown.

$1/f$ noise has two consequences that appear repeatedly in the rest of the course. First, its power diverges logarithmically as $f \to 0$, so the variance $\langle\delta\omega^2\rangle$ depends on the low-frequency cutoff — and the experimental cutoff is set by the duration of the measurement. A "measured $T_2^\ast$" is therefore only meaningful together with the averaging time. Second, because the noise is concentrated at low frequency, it is exactly the kind that an echo removes, which is why $T_2$ in a solid-state qubit routinely exceeds $T_2^\ast$ by a factor of between about 2 and 10. This chapter's own examples bracket that range: Code Example 7 gets $T_2/T_2^\ast = 2.2$ from a single $\pi$ pulse against synthesised $1/f$ noise, and Code Example 6 gets a factor of 9 in the opposite limit, where a static spread dominates and the echo removes nearly all of it. Larger ratios are reached by adding pulses rather than by adding one — the same Example 7 reaches $8\times$ with CPMG-16.

### The Standard Sequences

| Sequence | Pulses | Measures | Filter behaviour |
| --- | --- | --- | --- |
| Inversion recovery | $\pi$, wait, read | $T_1$ | samples $S$ at $\omega_q$ |
| Free-induction decay / Ramsey | $\pi/2$, wait $t$, $\pi/2$ | $T_2^\ast$ | sensitive down to DC |
| Hahn echo | $\pi/2$, $t/2$, $\pi$, $t/2$, $\pi/2$ | $T_2$ | blind at DC, passband near $1/2t$ |
| CPMG-$N$ | $\pi/2$, $N$ $\pi$ pulses, read | $T_2(N)$ | passband near $N/2t$ |
| Noise spectroscopy | CPMG-$N$ for many $N$ | $S(f)$ itself | scan the passband across the spectrum |

The last row is the one a materials researcher should take away. Sweeping $N$ moves the filter passband across the spectrum, so the family of coherence curves inverts to give $S(f)$ over several decades. A qubit is a sensitive, frequency-resolved probe of the noise in its own host material — and that is a characterization technique, not merely a diagnostic for a computer.

* * *

## 1.5 Quantum Hardware as Materials Science

This section states the position the rest of the course argues from.

### Where Each Platform Actually Stops

For every platform, the coherence time is limited by something that a materials scientist would recognize as a materials problem.

| Platform | Dominant coherence limit | The materials question behind it |
| --- | --- | --- |
| Superconducting circuits | dielectric loss from TLS in amorphous surface oxides and interfaces; non-equilibrium quasiparticles | Which oxide, which interface, and what fraction of the electric-field energy sits in it? |
| Trapped ions | anomalous heating of the motional mode from electric-field noise at the trap surface | What is on the electrode surface, and how does the noise scale with distance and temperature? |
| Neutral atoms | photon scattering, atom loss, laser and trap intensity noise; not the atom itself | Mostly an optics problem — but coatings, vacuum surfaces and stable reference cavities are materials |
| Semiconductor spins | hyperfine coupling to nuclear spins; charge noise at the oxide interface | Can the host be made spin-free by isotopic purification, and the interface made trap-free? |
| NV and other defect centres | surface spins and surface charge traps for shallow centres | How do you terminate a diamond surface without creating a spin bath? |
| Topological qubits | disorder in the semiconductor-superconductor hybrid; soft induced gap | Can an epitaxial interface be made clean enough for a hard gap to survive? |

Reading down the third column, the same three items recur: an **interface**, an **amorphous layer**, and an **isotopic or chemical impurity**. These are the oldest subjects in materials science. What is new is the sensitivity of the probe: a qubit responds to a defect density that no conventional characterization technique would notice, and it responds at millikelvin temperatures and gigahertz frequencies where very little dielectric loss data exists.

### The Quantities Are Materials Quantities

The performance figures of a quantum processor translate, without much loss, into quantities from the materials literature.

| Device figure | Materials quantity |
| --- | --- |
| $T_1$ of a superconducting qubit | loss tangent $\tan\delta$ of the participating dielectrics at mK and GHz |
| $T_2^\ast$ of a spin qubit in silicon | residual $^{29}\mathrm{Si}$ concentration and interface trap density |
| Resonator internal quality factor $Q_i$ | surface participation ratio times surface loss tangent |
| Ion motional heating rate | electric-field noise spectral density at the electrode surface |
| Two-qubit gate error floor | parameter drift, i.e. the $1/f$ amplitude $A$ |
| Device yield | statistical spread of junction area and barrier thickness |

Each right-hand entry is something that can be attacked by growth, processing, surface chemistry or purification, and each is measured in units a materials scientist already uses. The bridge is real, and it is narrow enough to cross.

### The Position of This Course

Three commitments follow, and they shape what is and is not in the remaining chapters.

**Physics and materials, not specifications.** No qubit counts, no fidelity records, no roadmaps. Such numbers are obsolete before they are read, and — more importantly — they are not explanatory. A scaling law, a selection rule, or a participation-ratio argument stays true.

**Every platform is presented as an answer to a noise channel.** Chapter 2 explains the transmon as a response to charge noise, Chapter 3 the Mølmer-Sørensen gate as a response to motional-state sensitivity, Chapter 5 the topological proposal as a response to local noise in general. This is the most compact way to understand a design.

**Where the honest answer is "the material is not good enough yet", that is the answer given.** It is also, for the intended reader, the most interesting answer, because it identifies where the work is.

* * *

## 1.6 A Numerical Laboratory: the Bloch Equations

Everything defined in Section 1.4 can be computed from one equation of motion, and the rest of this chapter does so. The tools built here are used in Chapters 2 through 5 whenever a coherence claim needs checking.

### The Equations

In the frame rotating at the drive frequency, with $\hbar = 1$, a driven two-level system has the Hamiltonian

$$ H = \frac{1}{2}\left( \Delta\, \sigma_z + \Omega\, \sigma_x \right) $$

where $\Omega$ is the Rabi frequency and $\Delta = \omega_q - \omega_d$ the detuning, both angular. A pure state evolves as the rigid rotation $\dot{\mathbf{r}} = \boldsymbol{\omega} \times \mathbf{r}$ with $\boldsymbol{\omega} = (\Omega, 0, \Delta)$. Adding relaxation towards the ground state at the north pole gives the **Bloch equations**:

$$ \begin{aligned} \dot{x} &= -\Delta y - x/T_2 \cr \dot{y} &= \Delta x - \Omega z - y/T_2 \cr \dot{z} &= \Omega y - (z - 1)/T_1 \end{aligned} $$

Three remarks before the code. The equations are phenomenological: $T_1$ and $T_2$ are put in by hand, and the price of that convenience is that they cannot describe non-Markovian noise, which is why Code Example 7 abandons them for explicit noise trajectories. They describe a *mixed* state, since $|\mathbf{r}|$ shrinks — the density-matrix language of the sister course's Chapter 5, in the equivalent vector form. And they are exactly the equations of magnetic resonance, so anyone who has interpreted an NMR or ESR experiment has already used them.

The population of the excited state is $P(1) = (1 - z)/2$, with $z = +1$ the ground state $|0\rangle$.

### Code Example 3: Rabi Oscillations, With and Without Relaxation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# All frequencies below are angular (rad/s). Cyclic frequencies carry an
# explicit factor 2 pi, and times are printed in microseconds.
US = 1e-6


def bloch_rhs(t, r, Omega, Delta, T1, T2):
    """Right-hand side of the Bloch equations in the rotating frame.

    r = (x, y, z) with z = +1 the ground state |0>. The drive is along x with
    Rabi frequency Omega, the detuning Delta = omega_qubit - omega_drive is
    along z. Relaxation pulls z back to +1 with rate 1/T1 and shrinks the
    transverse components with rate 1/T2.
    """
    x, y, z = r
    return [-Delta * y - x / T2,
            Delta * x - Omega * z - y / T2,
            Omega * y - (z - 1.0) / T1]


def evolve(r0, t_grid, Omega=0.0, Delta=0.0, T1=np.inf, T2=np.inf):
    """Integrate the Bloch equations on t_grid, starting from r0."""
    sol = solve_ivp(bloch_rhs, (t_grid[0], t_grid[-1]), r0, t_eval=t_grid,
                    args=(Omega, Delta, T1, T2), rtol=1e-10, atol=1e-12,
                    method="DOP853")
    return sol.y                      # shape (3, len(t_grid))


def p_excited(r):
    """Population of |1> from the Bloch vector: P(1) = (1 - z)/2."""
    return (1.0 - r[2]) / 2.0


# --- Coherent Rabi oscillations, checked against the analytic solution -------
f_rabi = 10e6                          # 10 MHz Rabi frequency
Omega = 2 * np.pi * f_rabi
t = np.linspace(0, 0.30 * US, 3001)

print("Generalized Rabi oscillation, no relaxation")
print(f"  Omega/2pi = {f_rabi/1e6:.1f} MHz, pi pulse = "
      f"{np.pi/Omega/US*1e3:.1f} ns")
print(f"\n{'Delta/2pi (MHz)':>16}{'max P(1) num':>15}{'analytic':>11}"
      f"{'eff. rate/2pi':>15}")
print("-" * 57)
for f_det in [0.0, 5.0, 10.0, 20.0]:
    Delta = 2 * np.pi * f_det * 1e6
    r = evolve([0.0, 0.0, 1.0], t, Omega=Omega, Delta=Delta)
    num = p_excited(r).max()
    ana = Omega**2 / (Omega**2 + Delta**2)
    rate = np.sqrt(Omega**2 + Delta**2) / (2 * np.pi) / 1e6
    print(f"{f_det:>16.1f}{num:>15.6f}{ana:>11.6f}{rate:>12.3f} MHz")

# --- The same drive with relaxation: oscillations decay into saturation ------
T1, T2 = 20 * US, 10 * US
t_long = np.linspace(0, 30 * US, 60001)
r = evolve([0.0, 0.0, 1.0], t_long, Omega=Omega, Delta=0.0, T1=T1, T2=T2)
p = p_excited(r)

z_ss = 1.0 / (1.0 + Omega**2 * T1 * T2)
T_rabi = 2.0 / (1.0 / T1 + 1.0 / T2)      # envelope decay of driven Rabi
print(f"\nDriven decay with T1 = {T1/US:.0f} us, T2 = {T2/US:.0f} us")
print(f"  P(1) at the first maximum      : {p[:400].max():.6f}")
print(f"  P(1) at t = 30 us              : {p[-1]:.6f}")
print(f"  analytic steady state (1-z)/2  : {(1 - z_ss)/2:.6f}")
print(f"  Omega^2 T1 T2                  : {Omega**2 * T1 * T2:.4g}")
print(f"  envelope decay 2/(1/T1 + 1/T2) : {T_rabi/US:.4f} us")

# Contrast of the oscillation, window by window, against that prediction
print(f"\n{'window (us)':>13}{'measured envelope':>19}{'0.5 exp(-t/T_rabi)':>21}")
print("-" * 53)
for t0 in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]:
    m = (t_long >= t0 * US) & (t_long <= (t0 + 0.2) * US)
    env = (p[m].max() - p[m].min()) / 2
    t_mid = (t0 + 0.1) * US
    print(f"{f'{t0:.1f}-{t0+0.2:.1f}':>13}{env:>19.6f}"
          f"{0.5*np.exp(-t_mid/T_rabi):>21.6f}")

# --- Visualisation ----------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for f_det in [0.0, 5.0, 10.0, 20.0]:
    r_ = evolve([0.0, 0.0, 1.0], t, Omega=Omega,
                Delta=2 * np.pi * f_det * 1e6)
    ax[0].plot(t / US, p_excited(r_), label=f"$\\Delta/2\\pi$ = {f_det:.0f} MHz")
ax[0].set_xlabel("time (us)"); ax[0].set_ylabel("P(1)")
ax[0].set_title("Rabi oscillations vs detuning"); ax[0].legend(fontsize=8)

ax[1].plot(t_long / US, p, lw=0.4, color="tab:purple")
ax[1].axhline((1 - z_ss) / 2, ls="--", color="k", lw=1,
              label="steady state")
ax[1].set_xlabel("time (us)"); ax[1].set_ylabel("P(1)")
ax[1].set_title("Driven decay into saturation"); ax[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

```text
Generalized Rabi oscillation, no relaxation
  Omega/2pi = 10.0 MHz, pi pulse = 50.0 ns

 Delta/2pi (MHz)   max P(1) num   analytic  eff. rate/2pi
---------------------------------------------------------
             0.0       1.000000   1.000000      10.000 MHz
             5.0       0.800000   0.800000      11.180 MHz
            10.0       0.500000   0.500000      14.142 MHz
            20.0       0.200000   0.200000      22.361 MHz

Driven decay with T1 = 20 us, T2 = 10 us
  P(1) at the first maximum      : 0.998127
  P(1) at t = 30 us              : 0.447300
  analytic steady state (1-z)/2  : 0.499999
  Omega^2 T1 T2                  : 7.896e+05
  envelope decay 2/(1/T1 + 1/T2) : 13.3333 us

  window (us)  measured envelope   0.5 exp(-t/T_rabi)
-----------------------------------------------------
      0.0-0.2           0.499064             0.496264
      1.0-1.2           0.463003             0.460406
      2.0-2.2           0.429548             0.427138
      5.0-5.2           0.343001             0.341077
    10.0-10.2           0.235741             0.234419
    20.0-20.2           0.111356             0.110731
```

**What to look for.** The detuning table is the standard check on any two-level solver. The maximum excited-state population is $\Omega^2/(\Omega^2 + \Delta^2)$ and the oscillation runs at the generalized Rabi frequency $\sqrt{\Omega^2 + \Delta^2}$; the numerics reproduce both to six digits. The practical content is that a detuning equal to the Rabi frequency already caps the population at $1/2$, so a $\pi$ pulse on a mis-set qubit does not merely rotate too far — it cannot reach $|1\rangle$ at all. This is why frequency calibration precedes amplitude calibration in every laboratory.

The second block shows what a drive does over many decay times. The oscillation envelope decays at the rate $\frac{1}{2}(1/T_1 + 1/T_2)$, which the windowed measurement confirms to three digits, and the population settles at $\frac{1}{2}\left(1 - (1 + \Omega^2 T_1 T_2)^{-1}\right)$, which for $\Omega^2 T_1 T_2 \approx 8\times10^5$ is one part in $10^6$ below one half. A strongly driven qubit ends up maximally mixed: continuous driving is not a way to hold a state, and the useful operating regime is always many Rabi periods short of saturation.

### Code Example 4: $T_1$, $T_2$, and the Bound $T_2 \le 2T_1$

The two decay times come from two different experiments on the same qubit. This example runs both, fits them, and verifies the relation between them.

```python
"""Chapter 1, Example 4: T1, T2 and the bound T2 <= 2 T1.
Continues from Example 3 (same session)."""


def fit_exponential(t, y):
    """Least-squares fit of y = A exp(-t/tau); returns (tau, A)."""
    mask = y > 1e-6 * y[0]
    slope, intercept = np.polyfit(t[mask], np.log(y[mask]), 1)
    return -1.0 / slope, np.exp(intercept)


def T2_from(T1, T_phi):
    """1/T2 = 1/(2 T1) + 1/T_phi: relaxation contributes half its rate."""
    return 1.0 / (1.0 / (2.0 * T1) + 1.0 / T_phi)


T1 = 30 * US
print("Energy relaxation and dephasing are separate measurements.\n")
print(f"{'T_phi (us)':>11}{'T2 (us)':>10}{'T2/(2 T1)':>11}"
      f"{'fitted T1 (us)':>16}{'fitted T2 (us)':>16}")
print("-" * 64)
for T_phi in [np.inf, 300 * US, 60 * US, 15 * US, 3 * US]:
    T2 = T2_from(T1, T_phi)

    # T1 experiment: prepare |1> (z = -1), no drive, watch z relax to +1.
    t1_grid = np.linspace(0, 4 * T1, 4001)
    z = evolve([0.0, 0.0, -1.0], t1_grid, T1=T1, T2=T2)[2]
    tau1, _ = fit_exponential(t1_grid, (1.0 - z) / 2.0)

    # T2 experiment: prepare |+> (x = 1), no drive, watch the transverse
    # component decay. This is the free-induction decay of a single qubit.
    t2_grid = np.linspace(0, 4 * T2, 4001)
    x = evolve([1.0, 0.0, 0.0], t2_grid, T1=T1, T2=T2)[0]
    tau2, _ = fit_exponential(t2_grid, x)

    label = "inf" if not np.isfinite(T_phi) else f"{T_phi/US:.0f}"
    print(f"{label:>11}{T2/US:>10.3f}{T2/(2*T1):>11.4f}"
          f"{tau1/US:>16.4f}{tau2/US:>16.4f}")

print("\nT_phi -> inf reproduces T2 = 2 T1 exactly: with no pure dephasing,")
print("the only way to lose phase information is to lose the excitation.")

# Which frequency of the environment does each rate sample?
print("\nThe two rates are sensitive to different parts of the noise spectrum:")
f_q = 5e9
for name, f_probe in [("1/T1  <- S(f) at the qubit frequency", f_q),
                      ("1/T2  <- S(f) near DC, set by the sequence", 1.0 / (30 * US))]:
    print(f"  {name:<45} f = {f_probe:9.4g} Hz")
print("  A single number cannot describe both: T1 is spectroscopy at GHz,")
print("  T2 is spectroscopy at kHz. Example 7 makes that quantitative.")

# --- Visualisation ----------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
t_grid = np.linspace(0, 120 * US, 6001)
for T_phi, c in [(np.inf, "tab:blue"), (60 * US, "tab:orange"),
                 (15 * US, "tab:green"), (3 * US, "tab:red")]:
    T2 = T2_from(T1, T_phi)
    lbl = ("T_phi = inf" if not np.isfinite(T_phi)
           else f"T_phi = {T_phi/US:.0f} us")
    ax[0].plot(t_grid / US,
               (1 - evolve([0, 0, -1.0], t_grid, T1=T1, T2=T2)[2]) / 2,
               color=c, label=lbl)
    ax[1].plot(t_grid / US, evolve([1.0, 0, 0], t_grid, T1=T1, T2=T2)[0],
               color=c, label=lbl)
ax[0].set_title(f"T1 decay of P(1), T1 = {T1/US:.0f} us")
ax[1].set_title("Free-induction decay of the transverse component")
for a in ax:
    a.set_xlabel("time (us)"); a.set_yscale("log"); a.set_ylim(1e-3, 1.2)
    a.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

```text
Energy relaxation and dephasing are separate measurements.

 T_phi (us)   T2 (us)  T2/(2 T1)  fitted T1 (us)  fitted T2 (us)
----------------------------------------------------------------
        inf    60.000     1.0000         30.0000         60.0000
        300    50.000     0.8333         30.0000         50.0000
         60    30.000     0.5000         30.0000         30.0000
         15    12.000     0.2000         30.0000         12.0000
          3     2.857     0.0476         30.0000          2.8571

T_phi -> inf reproduces T2 = 2 T1 exactly: with no pure dephasing,
the only way to lose phase information is to lose the excitation.

The two rates are sensitive to different parts of the noise spectrum:
  1/T1  <- S(f) at the qubit frequency          f =     5e+09 Hz
  1/T2  <- S(f) near DC, set by the sequence    f = 3.333e+04 Hz
  A single number cannot describe both: T1 is spectroscopy at GHz,
  T2 is spectroscopy at kHz. Example 7 makes that quantitative.
```

**What to look for.** The fits recover the input $T_1$ and $T_2$ to four digits, which is the boring and necessary part. The interesting column is $T_2/(2T_1)$: with no pure dephasing at all it equals exactly 1, and the two decay curves in the plot then differ only by that factor of two. Every finite $T_\varphi$ pushes the ratio down, and a strongly dephased qubit has $T_2 \ll T_1$. In real devices the ratio is a useful diagnostic on its own — a qubit with $T_2 \approx 2T_1$ is limited by energy loss and needs better dielectrics, while a qubit with $T_2 \ll T_1$ is limited by frequency fluctuations and needs a quieter charge or flux environment. The two diagnoses point at different fabrication steps.

The closing block previews the reason the two rates cannot be lumped into one number: they sample the environment at frequencies nine orders of magnitude apart.

### Code Example 5: Ramsey Fringes and $T_2^\ast$

A Ramsey experiment is two $\pi/2$ pulses separated by a wait, and it measures the accumulated phase as a population. This example runs it on a single qubit and then on an ensemble whose frequency differs from shot to shot.

```python
"""Chapter 1, Example 5: Ramsey fringes and the inhomogeneous time T2*.
Continues from Example 3 (same session)."""


def rotate(r, axis, angle):
    """Rotate a Bloch vector about `axis` by `angle` (Rodrigues formula).

    An instantaneous pulse. Real pulses have finite duration; the idealization
    is good as long as the pulse is short compared with every decay time.
    """
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    r = np.asarray(r, dtype=float)                  # shape (3,) or (3, N)
    cross = np.stack([n[1] * r[2] - n[2] * r[1],
                      n[2] * r[0] - n[0] * r[2],
                      n[0] * r[1] - n[1] * r[0]])
    ndotr = n[0] * r[0] + n[1] * r[1] + n[2] * r[2]
    return (r * np.cos(angle) + cross * np.sin(angle)
            + np.multiply.outer(n, ndotr) * (1.0 - np.cos(angle)))


def free_precession(r, t, Delta, T1, T2):
    """Analytic free evolution: precession about z plus T1/T2 decay."""
    x, y, z = r
    c, s = np.cos(Delta * t), np.sin(Delta * t)
    d2, d1 = np.exp(-t / T2), np.exp(-t / T1)
    return np.array([(x * c - y * s) * d2,
                     (x * s + y * c) * d2,
                     1.0 + (z - 1.0) * d1])


def ramsey(tau, Delta, T1, T2):
    """pi/2 about x, free evolution for tau, pi/2 about x, then read z."""
    r = rotate([0.0, 0.0, 1.0], [1, 0, 0], np.pi / 2)
    r = free_precession(r, tau, Delta, T1, T2)
    r = rotate(r, [1, 0, 0], np.pi / 2)
    return (1.0 - r[2]) / 2.0                # P(1)


# Check the analytic propagator against the ODE integrator of Example 3.
chk_t = np.array([0.0, 0.37 * US, 1.4 * US])
chk_num = evolve([0.6, -0.3, 0.5], chk_t, Delta=2 * np.pi * 1.7e6,
                 T1=30 * US, T2=8 * US)
chk_ana = np.array([free_precession([0.6, -0.3, 0.5], t, 2 * np.pi * 1.7e6,
                                    30 * US, 8 * US) for t in chk_t]).T
print("free_precession vs the ODE integrator: max |difference| = "
      f"{np.abs(chk_num - chk_ana).max():.2e}")

# --- One qubit with one detuning: clean fringes at the detuning frequency ----
T1, T2 = 30 * US, 10 * US
f_bar = 3.0e6                       # mean detuning, 3 MHz
sigma_f = 0.2e6                     # spread of the static detuning, 0.2 MHz
Delta_bar, sigma = 2 * np.pi * f_bar, 2 * np.pi * sigma_f

tau = np.linspace(0, 6 * US, 1201)
single = ramsey(tau, Delta_bar, T1, T2)

# --- An ensemble of static detunings: the fringes wash out ------------------
rng = np.random.default_rng(20260813)
n_shots = 20000
detunings = Delta_bar + sigma * rng.standard_normal(n_shots)
ensemble = np.zeros_like(tau)
for d in detunings:                   # one shot of the experiment each
    ensemble += ramsey(tau, d, T1, T2)
ensemble /= n_shots

# Quasi-static theory: <exp(i Delta tau)> = exp(i Delta_bar tau - sigma^2 tau^2 / 2)
T2_star_inh = np.sqrt(2.0) / sigma
envelope = np.exp(-tau / T2) * np.exp(-(tau / T2_star_inh) ** 2)

print(f"\nRamsey with mean detuning {f_bar/1e6:.1f} MHz, "
      f"static spread {sigma_f/1e6:.1f} MHz (Gaussian)")
print(f"  intrinsic T2                       : {T2/US:8.3f} us")
print(f"  inhomogeneous sqrt(2)/sigma        : {T2_star_inh/US:8.3f} us")
print(f"  fringe period 1/f_bar              : {1/f_bar/US:8.3f} us")
print(f"  fringes visible before the envelope dies: "
      f"{T2_star_inh*f_bar:.1f}")

print(f"\n{'tau (us)':>9}{'single detuning':>17}{'ensemble':>11}"
      f"{'|contrast|':>12}{'theory envelope':>17}")
print("-" * 66)
for t_mark in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
    i = int(np.argmin(np.abs(tau - t_mark * US)))
    contrast = abs(2 * ensemble[i] - 1.0)
    print(f"{tau[i]/US:>9.3f}{single[i]:>17.6f}{ensemble[i]:>11.6f}"
          f"{contrast:>12.6f}{envelope[i]:>17.6f}")

# Extract T2* from the ensemble curve the way an experiment would
peaks = []
for i in range(1, len(tau) - 1):
    if ensemble[i] >= ensemble[i - 1] and ensemble[i] >= ensemble[i + 1]:
        peaks.append(i)
amp = np.array([abs(2 * ensemble[i] - 1.0) for i in peaks])
tp = tau[peaks]
use = amp > 0.05
coef = np.polyfit(tp[use] ** 2, np.log(amp[use]), 1)
T2_star_fit = 1.0 / np.sqrt(-coef[0])
print(f"\nGaussian fit to the fringe maxima ({use.sum()} points):")
print(f"  T2* from the fit                   : {T2_star_fit/US:8.3f} us")
print(f"  sqrt(2)/sigma for comparison       : {T2_star_inh/US:8.3f} us")
print(f"  T2 / T2*                           : {T2/T2_star_fit:8.2f}")
print("The measured coherence time is set by the *spread* of qubit")
print("frequencies, not by the coherence of any single qubit.")

# --- Visualisation ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(tau / US, single, lw=0.8, color="lightsteelblue",
        label="single detuning")
ax.plot(tau / US, ensemble, lw=1.2, color="tab:purple", label="ensemble")
ax.plot(tau / US, 0.5 + 0.5 * envelope, "k--", lw=1, label="theory envelope")
ax.plot(tau / US, 0.5 - 0.5 * envelope, "k--", lw=1)
ax.set_xlabel("free evolution time tau (us)"); ax.set_ylabel("P(1)")
ax.set_title("Ramsey fringes: T2* from inhomogeneous broadening")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

```text
free_precession vs the ODE integrator: max |difference| = 7.46e-11

Ramsey with mean detuning 3.0 MHz, static spread 0.2 MHz (Gaussian)
  intrinsic T2                       :   10.000 us
  inhomogeneous sqrt(2)/sigma        :    1.125 us
  fringe period 1/f_bar              :    0.333 us
  fringes visible before the envelope dies: 3.4

 tau (us)  single detuning   ensemble  |contrast|  theory envelope
------------------------------------------------------------------
    0.000         1.000000   1.000000    1.000000         1.000000
    0.250         0.500000   0.499079    0.001842         0.928349
    0.500         0.024385   0.110374    0.779253         0.780834
    1.000         0.952419   0.703752    0.407504         0.410833
    1.500         0.069646   0.429123    0.141754         0.145653
    2.000         0.909365   0.515379    0.030757         0.034795
    3.000         0.870409   0.499695    0.000610         0.000608

Gaussian fit to the fringe maxima (5 points):
  T2* from the fit                   :    1.081 us
  sqrt(2)/sigma for comparison       :    1.125 us
  T2 / T2*                           :     9.25
The measured coherence time is set by the *spread* of qubit
frequencies, not by the coherence of any single qubit.
```

**What to look for.** The check at the top matters more than it looks: the analytic free-evolution propagator agrees with the ODE integrator to $10^{-11}$, which licenses the use of the fast closed form for the 20 000-shot ensemble average that follows.

Then the physics. A single detuning gives fringes that persist for the intrinsic $T_2 = 10\ \mu$s. Averaging over a Gaussian spread of only 0.2 MHz — a spread of under 7% of the mean detuning — destroys them in about $1\ \mu$s, a factor of nine sooner. Nothing about any individual qubit changed. The Gaussian fit to the fringe maxima returns $1.081\ \mu$s against the quasi-static prediction $\sqrt{2}/\sigma = 1.125\ \mu$s; the small deficit is the intrinsic exponential decay, which the pure-Gaussian fit absorbs. This is exactly the systematic error present in a laboratory fit, and quoting $T_2^\ast$ without stating the fit model is therefore ambiguous at the ten-percent level.

The one row of the table worth explaining is $\tau = 0.25\ \mu$s, where the contrast reads 0.0018 while the envelope is 0.93. That time falls on a fringe node, where $P(1) = 1/2$ regardless of coherence. Contrast has to be read at the fringe maxima, which is what the fit does.

### Code Example 6: The Hahn Echo Separates $T_2^\ast$ from $T_2$

Inserting one $\pi$ pulse halfway through the wait reverses the sign of the accumulated phase. Any detuning that did not change during the sequence therefore cancels exactly, and the inhomogeneous contribution disappears.

```python
"""Chapter 1, Example 6: the Hahn echo separates T2* from T2.
Continues from Examples 3, 4 and 5 (same session)."""


def ramsey_state(t, Delta, T1, T2):
    """Bloch vector just before the final pi/2 pulse of a Ramsey sequence."""
    r = rotate([0.0, 0.0, 1.0], [1, 0, 0], np.pi / 2)
    return free_precession(r, t, Delta, T1, T2)


def echo_state(t, Delta, T1, T2):
    """Same, for pi/2 - t/2 - pi - t/2.

    The pi pulse reverses the sign of the phase accumulated so far, so any
    detuning that is constant over the sequence cancels exactly.
    """
    r = rotate([0.0, 0.0, 1.0], [1, 0, 0], np.pi / 2)
    r = free_precession(r, t / 2.0, Delta, T1, T2)
    r = rotate(r, [1, 0, 0], np.pi)
    return free_precession(r, t / 2.0, Delta, T1, T2)


def hahn_echo(t, Delta, T1, T2):
    """Full echo sequence with a final -pi/2 read-out pulse; returns P(1)."""
    r = rotate(echo_state(t, Delta, T1, T2), [1, 0, 0], -np.pi / 2)
    return (1.0 - r[2]) / 2.0


def coherence(sequence, t, ensemble, T1, T2):
    """Ensemble average of the off-diagonal element, |<x + i y>|.

    This is the fringe *envelope*: the phase of the average carries the fringe,
    its modulus carries the coherence. Averaging the complex quantity is what
    an experiment does when it repeats the sequence shot after shot.
    """
    acc = np.zeros(np.shape(t), dtype=complex)
    for d in ensemble:
        r = sequence(t, d, T1, T2)
        acc += r[0] + 1j * r[1]
    return np.abs(acc / len(ensemble))


t_tot = np.linspace(0, 40 * US, 2001)
coh_ramsey = coherence(ramsey_state, t_tot, detunings, T1, T2)
coh_echo = coherence(echo_state, t_tot, detunings, T1, T2)

print("The echo is blind to a detuning that is constant during the sequence.")
print(f"{'t (us)':>8}{'Ramsey coherence':>18}{'echo coherence':>16}"
      f"{'exp(-t/T2)':>12}{'echo P(1), one shot':>21}")
print("-" * 75)
for t_mark in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]:
    i = int(np.argmin(np.abs(t_tot - t_mark * US)))
    one = hahn_echo(t_tot[i], Delta_bar + 5 * sigma, T1, T2)
    print(f"{t_tot[i]/US:>8.2f}{coh_ramsey[i]:>18.6f}{coh_echo[i]:>16.6f}"
          f"{np.exp(-t_tot[i]/T2):>12.6f}{one:>21.6f}")


def one_over_e(t, c):
    """First crossing of 1/e, by linear interpolation; nan if never reached."""
    below = np.nonzero(c < 1.0 / np.e)[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    k = below[0]
    return np.interp(1.0 / np.e, [c[k], c[k - 1]], [t[k], t[k - 1]])


t_ramsey = one_over_e(t_tot, coh_ramsey)
t_echo = one_over_e(t_tot, coh_echo)
print(f"\n{'sequence':<28}{'1/e coherence time':>20}")
print("-" * 48)
print(f"{'Ramsey (free induction)':<28}{t_ramsey/US:>17.3f} us")
print(f"{'Hahn echo':<28}{t_echo/US:>17.3f} us")
print(f"{'intrinsic T2 (input)':<28}{T2/US:>17.3f} us")
print(f"{'echo gain':<28}{t_echo/t_ramsey:>17.2f} x")

# The echo removes inhomogeneity, not energy relaxation.
t_wide = np.linspace(0, 200 * US, 2001)
print("\nThe echo cannot go past T2, and T2 cannot go past 2 T1:")
print(f"{'T_phi (us)':>11}{'T2 (us)':>10}{'echo 1/e (us)':>15}{'2 T1 (us)':>11}")
print("-" * 47)
T1_ref = 30 * US
for T_phi in [np.inf, 60 * US, 15 * US]:
    T2_i = T2_from(T1_ref, T_phi)
    e_curve = coherence(echo_state, t_wide, detunings[:2000], T1_ref, T2_i)
    label = "inf" if not np.isfinite(T_phi) else f"{T_phi/US:.0f}"
    print(f"{label:>11}{T2_i/US:>10.3f}"
          f"{one_over_e(t_wide, e_curve)/US:>15.3f}{2*T1_ref/US:>11.3f}")

print("\nMeasuring T2* and T2 on the same sample therefore measures two")
print("different things about the material: the static spread of qubit")
print("frequencies, and the fluctuating part of it.")

# --- Visualisation ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(t_tot / US, np.maximum(coh_ramsey, 1e-6), color="tab:orange",
            label="Ramsey envelope (T2*)")
ax.semilogy(t_tot / US, np.maximum(coh_echo, 1e-6), color="tab:purple",
            label="Hahn echo (T2)")
ax.semilogy(t_tot / US, np.exp(-t_tot / T2), "k--", lw=1, label="exp(-t/T2)")
ax.axhline(1 / np.e, color="gray", lw=0.8, ls=":")
ax.set_xlabel("total sequence time (us)"); ax.set_ylabel("coherence")
ax.set_ylim(1e-4, 1.5)
ax.set_title("Ramsey vs Hahn echo on the same qubit ensemble")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

```text
The echo is blind to a detuning that is constant during the sequence.
  t (us)  Ramsey coherence  echo coherence  exp(-t/T2)  echo P(1), one shot
---------------------------------------------------------------------------
    0.00          1.000000        1.000000    1.000000             1.000000
    0.50          0.779258        0.951229    0.951229             0.975615
    1.00          0.407513        0.904837    0.904837             0.952419
    2.00          0.030780        0.818731    0.818731             0.909365
    5.00          0.006403        0.606531    0.606531             0.803265
   10.00          0.001959        0.367879    0.367879             0.683940
   20.00          0.000506        0.135335    0.135335             0.567668
   40.00          0.000139        0.018316    0.018316             0.509158

sequence                      1/e coherence time
------------------------------------------------
Ramsey (free induction)                 1.059 us
Hahn echo                              10.000 us
intrinsic T2 (input)                   10.000 us
echo gain                                9.45 x

The echo cannot go past T2, and T2 cannot go past 2 T1:
 T_phi (us)   T2 (us)  echo 1/e (us)  2 T1 (us)
-----------------------------------------------
        inf    60.000         60.000     60.000
         60    30.000         30.000     60.000
         15    12.000         12.000     60.000

Measuring T2* and T2 on the same sample therefore measures two
different things about the material: the static spread of qubit
frequencies, and the fluctuating part of it.
```

**What to look for.** The echo coherence column and the $e^{-t/T_2}$ column are identical to six decimal places, and the single-shot echo column — computed for a detuning five standard deviations off the mean — matches the ensemble average exactly. That agreement *is* the echo: the observable no longer depends on the detuning at all, so averaging over detunings does nothing. On the same qubit ensemble, the 1/e coherence time goes from 1.06 $\mu$s to 10.0 $\mu$s, a gain of 9.5, and the recovered value is the intrinsic $T_2$ to four digits.

The second table draws the boundary. The echo removes inhomogeneity; it cannot remove energy relaxation. With no pure dephasing the echo returns $2T_1$ and not one nanosecond more, because at that point the only remaining decay channel is the loss of the excitation itself, and no pulse sequence can undo that. Every dynamical-decoupling result in the rest of the course is bounded by this line.

The physical reading is the one stated in the last three printed lines. $T_2^\ast$ and $T_2$ measured on the same sample are two different materials measurements: the first gives the *static* spread of qubit frequencies across shots and across devices, the second the *fluctuating* part fast enough to survive the echo. A process change that improves one and not the other tells you which defect population it affected.

### Code Example 7: $1/f$ Noise, Dynamical Decoupling, and Noise Spectroscopy

The quasi-static picture of Example 5 and the exact-cancellation picture of Example 6 are both idealizations. Real noise has a spectrum, and the echo works only on the part of it that is slow. This example generates explicit $1/f$ noise trajectories, runs CPMG sequences on them, and recovers the scaling law that connects the coherence time to the noise exponent.

```python
"""Chapter 1, Example 7: 1/f noise, filter functions, and CPMG scaling.
Continues from Example 3 (same session)."""


def noise_trajectories(n_traj, n_t, dt, alpha, A, f_low, rng):
    """Real Gaussian noise delta_omega(t) with one-sided PSD S(f) = A / f^alpha.

    Synthesised by giving each Fourier component a random phase and a variance
    set by S(f), then transforming back. Components below f_low are dropped:
    1/f noise has no finite variance without a low-frequency cutoff, and in an
    experiment that cutoff is set by how long the measurement lasts.
    """
    f = np.fft.rfftfreq(n_t, dt)
    S = np.zeros_like(f)
    band = f >= f_low
    S[band] = A / f[band] ** alpha
    scale = np.sqrt(S * n_t / (4.0 * dt))
    spec = (rng.standard_normal((n_traj, f.size))
            + 1j * rng.standard_normal((n_traj, f.size))) * scale
    return np.fft.irfft(spec, n=n_t, axis=1), f, S


def cpmg_modulation(n_steps, n_pi):
    """The +-1 switching function of a CPMG sequence with n_pi pi pulses.

    n_pi = 0 is free induction (Ramsey); n_pi = 1 is the Hahn echo, with the
    pulse at the centre; general n_pi puts pulses at (j - 1/2)/n_pi of the
    total time, which is the standard CPMG timing.
    """
    s = np.ones(n_steps)
    if n_pi == 0:
        return s
    u = (np.arange(n_steps) + 0.5) / n_steps
    flips = np.searchsorted((np.arange(1, n_pi + 1) - 0.5) / n_pi, u)
    return np.where(flips % 2 == 0, 1.0, -1.0)


dt, n_t, n_traj = 2e-9, 8192, 2000
m_max = n_t // 4                  # sequences use only the first quarter, so
f_low = 1.0 / (n_t * dt)          # that the synthesis period is never probed
alpha, A = 1.0, 6.0e12            # (rad/s)^2 per Hz at f = 1 Hz
rng = np.random.default_rng(11)
traj, f_axis, S_target = noise_trajectories(n_traj, n_t, dt, alpha, A,
                                            f_low, rng)

# Check the synthesis: sample PSD against the target, and the variance
# against the analytic integral A ln(f_max / f_low).
spec = np.fft.rfft(traj[:500], axis=1)
S_sample = 2.0 * np.mean(np.abs(spec) ** 2, axis=0) * dt / n_t
band = (f_axis >= 10 * f_low) & (f_axis <= 0.2 / (2 * dt))
ratio = np.mean(S_sample[band] / S_target[band])
var_analytic = A * np.log((1.0 / (2 * dt)) / f_low)
print(f"1/f noise synthesis, {n_traj} trajectories of {n_t} samples "
      f"at dt = {dt*1e9:.0f} ns")
print(f"  band                          : {f_low/1e3:.1f} kHz to "
      f"{1/(2*dt)/1e6:.0f} MHz")
print(f"  mean sample S(f) / target S(f): {ratio:.4f}")
print(f"  sample variance of delta_omega: {np.var(traj):.4e} (rad/s)^2")
print(f"  A ln(f_max/f_low)             : {var_analytic:.4e} (rad/s)^2")
print(f"  rms detuning                  : "
      f"{np.sqrt(np.var(traj))/(2*np.pi)/1e3:.1f} kHz")
print(f"  quasi-static sqrt(2)/sigma    : "
      f"{np.sqrt(2/np.var(traj))/US:.3f} us")
print(f"  statistical floor 1/sqrt(n)   : {1/np.sqrt(n_traj):.4f}")

# --- Coherence under each sequence ------------------------------------------
n_pis = [0, 1, 2, 4, 8, 16]
t_idx = np.unique(np.round(np.logspace(np.log10(16), np.log10(m_max), 40)
                           ).astype(int))
times = t_idx * dt
curves = {}
for n_pi in n_pis:
    c = np.empty(t_idx.size)
    for k, m in enumerate(t_idx):
        s = cpmg_modulation(m, n_pi)
        phi = dt * (traj[:, :m] * s).sum(axis=1)      # accumulated phase
        c[k] = np.abs(np.mean(np.exp(1j * phi)))
    curves[n_pi] = c

print(f"\nCoherence |<exp(i phi)>| for CPMG-N under the same 1/f noise")
print(f"{'t (us)':>8}" + "".join(f"{('N=' + str(n)):>10}" for n in n_pis))
print("-" * (8 + 10 * len(n_pis)))
for t_mark in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]:
    k = int(np.argmin(np.abs(times - t_mark * US)))
    print(f"{times[k]/US:>8.3f}"
          + "".join(f"{curves[n][k]:>10.4f}" for n in n_pis))


def one_over_e_log(t, c):
    """1/e crossing, interpolated in log-coherence against log-time."""
    below = np.nonzero(c < 1.0 / np.e)[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    k = below[0]
    lo, hi = np.log(c[k]), np.log(c[k - 1])
    return np.exp(np.interp(-1.0, [lo, hi], [np.log(t[k]), np.log(t[k - 1])]))


T2s = {n: one_over_e_log(times, curves[n]) for n in n_pis}
print(f"\n{'N (pi pulses)':>14}{'T2(N) (us)':>13}{'T2(N)/T2(1)':>14}")
print("-" * 41)
for n in n_pis:
    label = "0 (Ramsey)" if n == 0 else str(n)
    print(f"{label:>14}{T2s[n]/US:>13.4f}{T2s[n]/T2s[1]:>14.3f}")

# For S(f) = A/f^alpha the echo family obeys T2(N) ~ N^(alpha/(alpha+1)).
ns = np.array([n for n in n_pis if n >= 1], dtype=float)
ts = np.array([T2s[n] for n in n_pis if n >= 1])
slope, _ = np.polyfit(np.log(ns), np.log(ts), 1)
print(f"\nfitted exponent of T2(N) vs N   : {slope:.4f}")
print(f"predicted alpha/(alpha+1)       : {alpha/(alpha+1):.4f}")
print(f"Ramsey T2* / echo T2            : {T2s[0]/T2s[1]:.4f}")

# --- The filter-function picture --------------------------------------------
print("\nWhy: the sequence acts as a band-pass filter on the noise.")
t_fix = m_max * dt
print(f"  sequence duration t = {t_fix/US:.3f} us")
print(f"{'N':>4}{'first passband f (MHz)':>24}{'S(f) there':>14}")
print("-" * 42)
for n in n_pis:
    f_peak = f_low if n == 0 else n / (2.0 * t_fix)
    print(f"{n:>4}{f_peak/1e6:>24.4f}{A/f_peak**alpha:>14.4g}")
print("  Free induction is sensitive down to DC, where 1/f noise is largest.")
print("  Every pi pulse moves the passband up in frequency, where there is")
print("  less noise - until it reaches the flat, unfilterable part.")

# --- Visualisation ----------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for n in n_pis:
    lbl = "Ramsey" if n == 0 else f"CPMG-{n}"
    ax[0].loglog(times / US, np.maximum(curves[n], 1e-4), label=lbl)
ax[0].axhline(1 / np.e, color="gray", ls=":", lw=0.8)
ax[0].set_xlabel("t (us)"); ax[0].set_ylabel("coherence")
ax[0].set_ylim(1e-3, 1.5); ax[0].legend(fontsize=7)
ax[0].set_title("Dynamical decoupling under 1/f noise")

ax[1].loglog(ns, ts / US, "o-", color="tab:purple", label="simulation")
ax[1].loglog(ns, ts[0] / US * ns ** (alpha / (alpha + 1)), "k--", lw=1,
             label=f"$N^{{{alpha/(alpha+1):.2f}}}$")
ax[1].set_xlabel("number of pi pulses N"); ax[1].set_ylabel("T2(N) (us)")
ax[1].legend(fontsize=8); ax[1].set_title("CPMG scaling")
plt.tight_layout()
plt.show()
```

```text
1/f noise synthesis, 2000 trajectories of 8192 samples at dt = 2 ns
  band                          : 61.0 kHz to 250 MHz
  mean sample S(f) / target S(f): 1.0008
  sample variance of delta_omega: 5.3497e+13 (rad/s)^2
  A ln(f_max/f_low)             : 4.9907e+13 (rad/s)^2
  rms detuning                  : 1164.1 kHz
  quasi-static sqrt(2)/sigma    : 0.193 us
  statistical floor 1/sqrt(n)   : 0.0224

Coherence |<exp(i phi)>| for CPMG-N under the same 1/f noise
  t (us)       N=0       N=1       N=2       N=4       N=8      N=16
--------------------------------------------------------------------
   0.052    0.9557    0.9943    0.9967    0.9980    0.9989    0.9991
   0.098    0.8672    0.9800    0.9885    0.9940    0.9967    0.9982
   0.206    0.6005    0.9155    0.9493    0.9732    0.9866    0.9930
   0.494    0.0990    0.5859    0.7557    0.8631    0.9241    0.9604
   1.042    0.0351    0.1018    0.2699    0.5231    0.7167    0.8372
   1.942    0.0147    0.0434    0.0255    0.0936    0.3374    0.5501
   4.096    0.0121    0.0356    0.0151    0.0240    0.0218    0.0691

 N (pi pulses)   T2(N) (us)   T2(N)/T2(1)
-----------------------------------------
    0 (Ramsey)       0.3041         0.445
             1       0.6829         1.000
             2       0.9032         1.323
             4       1.2780         1.872
             8       1.7891         2.620
            16       2.5266         3.700

fitted exponent of T2(N) vs N   : 0.4761
predicted alpha/(alpha+1)       : 0.5000
Ramsey T2* / echo T2            : 0.4454

Why: the sequence acts as a band-pass filter on the noise.
  sequence duration t = 4.096 us
   N  first passband f (MHz)    S(f) there
------------------------------------------
   0                  0.0610      9.83e+07
   1                  0.1221     4.915e+07
   2                  0.2441     2.458e+07
   4                  0.4883     1.229e+07
   8                  0.9766     6.144e+06
  16                  1.9531     3.072e+06
  Free induction is sensitive down to DC, where 1/f noise is largest.
  Every pi pulse moves the passband up in frequency, where there is
  less noise - until it reaches the flat, unfilterable part.
```

**What to look for.** The synthesis is verified before it is used: the sample spectral density matches the target to within 0.1% across the band, and the sample variance sits 7% above $A\ln(f_\mathrm{max}/f_\mathrm{low})$ because the lowest retained bin contributes at the edge of the band. That check is not optional — a coloured-noise generator with the wrong normalization produces plausible-looking curves with wrong time constants.

The coherence table is the central result. Every $\pi$ pulse buys coherence, and the gain is systematic rather than accidental: fitting $T_2(N)$ against $N$ over $N = 1$ to $16$ gives an exponent of 0.476 against the prediction $\alpha/(\alpha+1) = 0.5$ for $\alpha = 1$. The filter-function block explains it. Free induction is sensitive down to the lowest frequency in the band, where $S(f) = A/f$ is largest; each doubling of $N$ moves the first passband up by a factor of two, halving the noise power there. The scaling law is not a fitting convenience — it is a direct consequence of the shape of $S(f)$, which means that measuring the exponent measures $\alpha$, and $\alpha$ is a statement about the distribution of defect switching rates in the material.

Three details deserve attention. The Ramsey time is $0.304\ \mu$s while the quasi-static estimate $\sqrt{2}/\sigma$ gives $0.193\ \mu$s: the quasi-static approximation *overestimates* the dephasing, because the fast components of the noise average out within the sequence instead of contributing a static phase. The deep tails of the table are noise, not signal — the statistical floor of a 2000-trajectory average is $1/\sqrt{2000} = 0.022$, and the apparent crossings below that value are sampling error. And the low-frequency cutoff is a physical parameter, not a numerical one: it stands for the finite duration of a real measurement, which is why the same qubit yields a shorter $T_2^\ast$ when averaged for an hour than when averaged for a second.

### What This Toolkit Is For

Five functions — `bloch_rhs`, `evolve`, `rotate`, `free_precession` and `coherence` — plus the noise generator reproduce every standard coherence measurement. Chapters 2 through 5 use them to check platform-specific claims:

| Function | Introduced | Used for |
| --- | --- | --- |
| `bloch_rhs`, `evolve` | Example 3 | driven dynamics, pulse calibration, saturation |
| `rotate` | Example 5 | ideal pulses in a sequence |
| `free_precession` | Example 5 | fast exact evolution between pulses |
| `coherence` | Example 6 | ensemble-averaged off-diagonal element |
| `noise_trajectories`, `cpmg_modulation` | Example 7 | non-Markovian noise, filter functions, decoupling |

What the toolkit cannot do is equally worth stating. It has one qubit, so no two-qubit gate and no entanglement; it has two levels, so no leakage; and it takes $T_1$, $T_2$ and $S(f)$ as inputs, so it cannot predict them. Deriving those three from a Hamiltonian and a material is the work of the next four chapters.

* * *

## Exercises

#### Exercise 1: Coherence-Time Bookkeeping

A qubit is measured to have $T_1 = 40\ \mu$s and, with a Hahn echo, $T_2 = 25\ \mu$s. A Ramsey experiment on the same device gives a Gaussian decay with a $1/e$ time of $0.9\ \mu$s.

  1. Find the pure dephasing time $T_\varphi$.
  2. What is the largest $T_2$ this device could have if the dielectric losses setting $T_1$ were unchanged?
  3. Assuming the Ramsey decay is dominated by a static Gaussian spread of qubit frequencies, find the standard deviation $\sigma/2\pi$ in kHz.
  4. A colleague reports $T_1 = 40\ \mu$s and $T_2 = 95\ \mu$s on a similar device. What can you say without repeating the measurement?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> From \(1/T_2 = 1/(2T_1) + 1/T_\varphi\): \(1/T_\varphi = 1/25 - 1/80 = 0.04 - 0.0125 = 0.0275\ \mu\mathrm{s}^{-1}\), so \(T_\varphi = 36.4\ \mu\mathrm{s}\). Pure dephasing and relaxation contribute comparably here.</p>

<p><strong>2.</strong> The bound is \(T_2 \le 2T_1 = 80\ \mu\mathrm{s}\), reached when \(T_\varphi \to \infty\). Eliminating all pure dephasing would therefore buy a factor 3.2, and nothing more; going beyond that requires improving \(T_1\).</p>

<p><strong>3.</strong> For a Gaussian static spread, \(T_{2,\mathrm{inh}}^\ast = \sqrt{2}/\sigma\), so \(\sigma = \sqrt{2}/0.9\ \mu\mathrm{s} = 1.571\times10^{6}\) rad/s and \(\sigma/2\pi = 250\) kHz. Strictly this ignores the intrinsic exponential, which shortens the observed decay slightly and so makes 250 kHz a mild overestimate.</p>

<p><strong>4.</strong> It is impossible: \(T_2 \le 2T_1 = 80\ \mu\mathrm{s}\). Either \(T_1\) was underestimated, or the "\(T_2\)" is not a \(T_2\) — a common cause is fitting a CPMG-\(N\) decay and reporting it as a Hahn echo, since \(T_2(N)\) grows with \(N\) while remaining bounded by \(2T_1\).</p>

```python
T1, T2, T2s = 40.0, 25.0, 0.9
print(round(1 / (1 / T2 - 1 / (2 * T1)), 3))     # 36.364  T_phi in us
print(2 * T1)                                    # 80.0    the bound
import numpy as np
print(round(np.sqrt(2) / (T2s * 1e-6) / (2 * np.pi) / 1e3, 1))   # 250.1  kHz
```

</details>

#### Exercise 2: Reading a Platform From Its Energy Scale

Consider a hypothetical qubit whose transition frequency is 400 MHz — the scale of a low-frequency flux qubit, or of a nuclear spin in a strong field.

  1. What is its energy in $\mu$eV, and its equivalent temperature $\hbar\omega_q/k_B$ in mK?
  2. At a base temperature of 20 mK, what is the thermal excited-state population?
  3. Is thermal initialization viable? What would you do instead?
  4. A colleague proposes lowering the frequency to 40 MHz to gain coherence, arguing that $T_1$ improves because $S(\omega_q)$ falls with frequency. Give one argument for and one against.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(E = h \times 4\times10^{8} = 2.65\times10^{-25}\) J = \(1.654\ \mu\mathrm{eV}\); \(hf/k_B = 19.20\) mK.</p>

<p><strong>2.</strong> With \(x = hf/k_BT = 19.2/20 = 0.960\), the two-level Boltzmann population is \(e^{-x}/(1+e^{-x}) = 0.2769\). More than a quarter of the population is in the wrong state.</p>

<p><strong>3.</strong> No. Thermal initialization needs \(k_BT \ll \hbar\omega_q\), i.e. a base temperature well below 2 mK here, which is not available. The alternative is active reset: measure and conditionally flip, or drive a dissipative transition through an auxiliary level. This is the same reasoning that makes low-frequency flux qubits and nuclear spins hard to initialize, and it is why active reset is standard rather than optional.</p>

<p><strong>4.</strong> <em>For:</em> if the environment has a \(1/f\)-like or otherwise decreasing spectral density, a lower \(\omega_q\) does sit where there is less noise power, and the golden-rule \(T_1\) improves. <em>Against:</em> initialization becomes worse still (at 40 MHz, \(hf/k_B = 1.9\) mK, so the qubit is essentially fully mixed at any achievable temperature), readout contrast degrades because thermal photons swamp the signal, and the gate time must lengthen since \(\Omega \ll \omega_q\) is required for the rotating-frame description to hold. Coherence is not the only axis, which is the lesson of Section 1.3.</p>

```python
import numpy as np
h, kB = 6.62607015e-34, 1.380649e-23
for f in (400e6, 40e6):
    x = h * f / (kB * 0.020)
    print(f"{f/1e6:5.0f} MHz  {h*f/1.602176634e-19*1e6:6.3f} ueV  "
          f"{h*f/kB*1e3:6.2f} mK  P_exc = {np.exp(-x)/(1+np.exp(-x)):.4f}")
# 400 MHz   1.654 ueV   19.20 mK  P_exc = 0.2769
#  40 MHz   0.165 ueV    1.92 mK  P_exc = 0.4760
```

</details>

#### Exercise 3: The Gate Budget

Take two platforms: A with $T_2 = 200\ \mu$s and a two-qubit gate time of 40 ns, and B with $T_2 = 2$ s and a two-qubit gate time of 200 $\mu$s.

  1. Estimate the decoherence-limited error per two-qubit gate as $\epsilon \approx t_\mathrm{gate}/T_2$ for each.
  2. How many gates can each run before the accumulated error reaches 1?
  3. A surface-code threshold requires roughly $\epsilon < 10^{-3}$. Does either platform clear it on decoherence alone? By what factor would $T_2/t_\mathrm{gate}$ have to improve to reach $\epsilon = 10^{-4}$?
  4. Which is physically easier to obtain — a tenfold longer $T_2$, or tenfold faster gates? Answer separately for a transmon and for a trapped ion.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> A: \(\epsilon \approx 40\ \mathrm{ns}/200\ \mu\mathrm{s} = 2\times10^{-4}\). B: \(\epsilon \approx 200\ \mu\mathrm{s}/2\ \mathrm{s} = 1\times10^{-4}\). The two are within a factor of two despite four decades of difference in both inputs — which is the point of the dimensionless ratio.</p>

<p><strong>2.</strong> \(1/\epsilon\): 5000 gates for A, 10 000 for B.</p>

<p><strong>3.</strong> Both clear \(10^{-3}\) on decoherence alone, which is why neither platform's error is decoherence-dominated in practice: calibration error, crosstalk and leakage dominate instead. Reaching \(10^{-4}\) needs a factor 2 for A and no change for B on this budget — the honest conclusion being that \(T_2/t_\mathrm{gate}\) is not the binding constraint for either platform at these numbers, and that error budgets must be measured rather than estimated this way.</p>

<p><strong>4.</strong> <em>Transmon:</em> a tenfold longer \(T_2\) means a tenfold reduction in dielectric loss or \(1/f\) amplitude — a materials programme, hard but with a clear target. Tenfold faster gates run into the anharmonicity: a pulse shorter than roughly \(1/\alpha\) populates the third level, so speed is capped by the circuit's spectrum, not by the amplifier. <em>Trapped ion:</em> \(T_2\) is already long and is limited by magnetic-field stability, so shielding and clock transitions can buy more; faster gates are limited by the motional frequency and by the requirement of staying in the resolved-sideband regime, and pushing past it costs fidelity through off-resonant excitation. In both cases the speed axis is bounded by a spectral scale of the system, which is why the coherence axis receives most of the effort.</p>

</details>

#### Exercise 4: Filter Functions

Using the definition $\tilde{s}(f,t) = \int_0^t s(t')e^{-2\pi i f t'}dt'$ with $s = +1$ throughout for free induction and $s = +1$ then $-1$ for the echo:

  1. Verify numerically, at $t = 1$ and $f = 0.03$, that $|\tilde{s}_\mathrm{FID}|^2 = \sin^2(\pi f t)/(\pi f)^2$ and $|\tilde{s}_\mathrm{echo}|^2 = 4\sin^4(\pi f t/2)/(\pi f)^2$.
  2. Expand both for $f t \ll 1$ and state the leading power of $f$ in each.
  3. Explain, from the results of part 2, why an echo helps enormously against $1/f$ noise and hardly at all against white noise.
  4. For $S(f) = A/f^\alpha$ the CPMG coherence time scales as $T_2(N) \propto N^{\alpha/(\alpha+1)}$. What does this predict for white noise, and does it agree with your answer to part 3?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Both agree to eight digits; see the code below. \(|\tilde{s}_\mathrm{FID}|^2 = 0.99704262\) and \(|\tilde{s}_\mathrm{echo}|^2 = 0.00221738\) at these values — the echo is already 450 times less sensitive at \(ft = 0.03\).</p>

<p><strong>2.</strong> For FID, \(\sin^2(\pi f t) \approx (\pi f t)^2\), so \(|\tilde{s}|^2 \to t^2\): a constant in \(f\), maximal at DC. For the echo, \(\sin^4(\pi f t/2) \approx (\pi f t/2)^4\), so \(|\tilde{s}|^2 \to \pi^2 f^2 t^4/4\): it vanishes as \(f^2\).</p>

<p><strong>3.</strong> The mean-square phase is \(\int_0^\infty S(f)|\tilde{s}(f)|^2 df\). For \(S \propto 1/f\) the integrand for FID goes as \(1/f\) and diverges logarithmically at small \(f\) — all the damage comes from the slowest components, which is exactly where the echo filter has a zero. The echo integrand goes as \(f\), and is finite and small. For white noise, \(S\) is constant and the FID integrand is \(\sin^2(\pi f t)/(\pi f)^2\), which is already integrable with no low-frequency enhancement; there is no low-frequency weight for the echo to remove, so it gains nothing.</p>

<p><strong>4.</strong> White noise is \(\alpha = 0\), giving \(T_2(N) \propto N^0\) — no gain from any number of \(\pi\) pulses. That is precisely the statement in part 3, and it is also the reason dynamical decoupling cannot extend \(T_2\) past \(2T_1\): the relaxation channel is effectively white at the frequencies the sequence can reach.</p>

```python
import numpy as np
t, f, N = 1.0, 0.03, 200000
tt = np.linspace(0, t, N + 1)
for name, s, ana in [
        ("FID ", np.ones_like(tt), np.sin(np.pi*f*t)**2 / (np.pi*f)**2),
        ("echo", np.where(tt < t/2, 1.0, -1.0),
         4*np.sin(np.pi*f*t/2)**4 / (np.pi*f)**2)]:
    num = abs(np.trapezoid(s * np.exp(-2j*np.pi*f*tt), tt))**2
    print(name, f"numeric {num:.8f}   analytic {ana:.8f}")
# FID  numeric 0.99704262   analytic 0.99704262
# echo numeric 0.00221738   analytic 0.00221738
```

</details>

#### Exercise 5: Diagnosing a Material From Coherence Data

Two wafers are processed identically except for the deposition conditions of one dielectric layer. Qubits are measured on both.

| Wafer | $T_1$ | $T_2$ (echo) | $T_2^\ast$ (Ramsey, Gaussian) |
| --- | --- | --- | --- |
| P | 60 $\mu$s | 80 $\mu$s | 1.2 $\mu$s |
| Q | 61 $\mu$s | 22 $\mu$s | 1.1 $\mu$s |

  1. For each wafer, decide whether $T_2$ is relaxation-limited or dephasing-limited, and give $T_\varphi$.
  2. Which wafer's process change affected the *fluctuating* part of the environment, and which the *static* part? Justify from the table.
  3. Estimate $\sigma/2\pi$ for each wafer. Are the two significantly different?
  4. Where would you look next, physically, and what single further measurement would settle it?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Wafer P: \(2T_1 = 120\ \mu\mathrm{s}\) and \(T_2 = 80\ \mu\mathrm{s}\), so \(1/T_\varphi = 1/80 - 1/120 = 0.004167\), \(T_\varphi = 240\ \mu\mathrm{s}\) — pure dephasing is a small correction, and \(T_2\) is largely relaxation-limited. Wafer Q: \(2T_1 = 122\ \mu\mathrm{s}\), \(T_2 = 22\ \mu\mathrm{s}\), so \(1/T_\varphi = 1/22 - 1/122 = 0.03726\), \(T_\varphi = 26.8\ \mu\mathrm{s}\) — strongly dephasing-limited.</p>

<p><strong>2.</strong> \(T_1\) is unchanged, so the density of loss channels at the qubit frequency did not move. \(T_2^\ast\) is unchanged, so the static spread of qubit frequencies did not move either. What changed is \(T_2\), i.e. \(T_\varphi\) — by nearly a factor of 9. The process change therefore added noise power in the intermediate band that the echo does not filter out: fluctuators with switching rates comparable to \(1/T_2\), roughly tens of kHz. It affected the fluctuating part, and neither the static disorder nor the GHz loss.</p>

<p><strong>3.</strong> \(\sigma = \sqrt{2}/T_2^\ast\): wafer P gives \(1.18\times10^{6}\) rad/s, i.e. 188 kHz; wafer Q gives \(1.29\times10^{6}\) rad/s, i.e. 205 kHz. A 9% difference, which for a handful of devices per wafer is almost certainly not significant — the static disorder is the same.</p>

<p><strong>4.</strong> The suspect is a population of two-level fluctuators introduced by the new deposition, active at tens of kHz. The decisive measurement is CPMG noise spectroscopy: sweep \(N\) and invert the family of \(T_2(N)\) curves for \(S(f)\) on both wafers. If wafer Q shows excess \(S(f)\) in the 10-100 kHz band with the \(1/f^\alpha\) exponent of a thermally activated ensemble, the diagnosis is confirmed and the fitted amplitude \(A\) becomes the figure of merit for the next deposition trial. A second, cheaper check is the temperature dependence of \(T_\varphi\), since thermally activated fluctuators freeze out.</p>

```python
for name, T1, T2, T2s in [("P", 60.0, 80.0, 1.2), ("Q", 61.0, 22.0, 1.1)]:
    Tphi = 1 / (1 / T2 - 1 / (2 * T1))
    print(f"{name}: 2T1 = {2*T1:6.1f} us   T_phi = {Tphi:7.1f} us"
          f"   sigma/2pi = {2**0.5 / (T2s*1e-6) / (2*3.141592653589793) / 1e3:5.0f} kHz")
# P: 2T1 =  120.0 us   T_phi =   240.0 us   sigma/2pi =   188 kHz
# Q: 2T1 =  122.0 us   T_phi =    26.8 us   sigma/2pi =   205 kHz
```

</details>

* * *

## Summary

### Key Takeaways

**1\. A good qubit has to be two contradictory things**

  * Isolated enough to keep a phase, and coupled enough to be driven; the same coupling constant governs both, by fluctuation-dissipation.
  * The partial escape is frequency selectivity, which is why the *shape* of the environmental noise spectrum matters more than its overall size.
  * Natural qubits are identical and long-lived but unwiring; engineered qubits are fast and integrable but inherit their host material's defects. Every platform is a specific compromise, not a step on a single ladder.

**2\. The DiVincenzo criteria are a checklist, not a scorecard**

  * Five for computation — characterized qubits, initialization, $T_2 \gg t_\mathrm{gate}$, universal gates, qubit-specific readout — plus two for flying qubits.
  * They contain no thresholds, say nothing about correlated errors, and say nothing about cost.
  * Every criterion maps onto a physical property that growth, processing or purification can change.

**3\. Six axes, and no single winner**

  * Coherence, gate fidelity and speed, connectivity, reproducibility and yield, operating temperature, scalability.
  * The axes are physically coupled: speed against coherence, connectivity against speed, temperature against energy scale, tunability against disorder.
  * Code Example 2 gives four rankings of four platforms and no two agree; connectivity alone was worth more than a decade of coherence for the workload considered.

**4\. Three time constants, one spectral density**

  * $T_1$ samples $S$ at $\omega_q$; $T_\varphi$ samples it near DC; $1/T_2 = 1/(2T_1) + 1/T_\varphi$, so $T_2 \le 2T_1$ always.
  * $T_2^\ast$ measures the *static* spread of qubit frequencies and decays as a Gaussian; $T_2$ measures the fluctuating part, and its shape follows the noise spectrum — exponential only for white noise, and Gaussian, $\exp[-(t/T_2)^2]$, under the $1/f$ noise of Example 7.
  * The coherence of any sequence is $\exp\left(-\frac{1}{2}\int_0^\infty S(f)|\tilde{s}(f,t)|^2 df\right)$, and $|\tilde{s}|^2$ — the filter function — is the experimenter's free choice.

**5\. $1/f$ noise is a defect ensemble**

  * $S(f) = A/f^\alpha$ with $\alpha \approx 1$ arises from two-level fluctuators with a log-uniform distribution of switching rates, i.e. from disorder in an amorphous host.
  * Because the power is concentrated at low frequency, an echo removes most of it: $T_2/T_2^\ast$ between about 2 and 10 is routine in solid-state qubits — 2.2 in Example 7, 9 in Example 6 — and more pulses buy more, $8\times$ at CPMG-16.
  * CPMG-$N$ scaling $T_2(N)\propto N^{\alpha/(\alpha+1)}$ was recovered numerically as $N^{0.476}$ against the predicted $N^{0.5}$ — so the measured exponent measures the defect ensemble.

**6\. The measurements are materials characterization**

  * $T_1$ is a loss tangent at mK and GHz; $T_2^\ast$ is a static disorder distribution; CPMG spectroscopy is a frequency-resolved noise spectrum of the host.
  * For every platform, the limit lies at an interface, in an amorphous layer, or in an isotopic or chemical impurity.
  * A qubit is a sensitive probe of its own material, and that is the entry point this course is written around.

**Practical implications**

  * Never compare platforms on one number; state the axis, and prefer the dimensionless $T_2/t_\mathrm{gate}$ to either factor alone.
  * Check $T_2 \le 2T_1$ on any reported pair, and ask what fit model produced a quoted $T_2^\ast$ and over what averaging time.
  * When a process change moves $T_2$ but not $T_1$ or $T_2^\ast$, you have learned which frequency band, and therefore which defect population, it affected.

The next three chapters take the three leading platforms in turn, and each one is organized as an answer to a noise channel. Chapter 2 starts with superconducting qubits: how a Josephson junction turns an LC circuit into an artificial atom, why the transmon trades anharmonicity for charge-noise immunity, and why the coherence of the best circuits is set by a few nanometres of amorphous oxide — the point at which this subject becomes indistinguishable from materials science.

[← Series Top](<index.html>) [Chapter 2: Superconducting Qubits →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
