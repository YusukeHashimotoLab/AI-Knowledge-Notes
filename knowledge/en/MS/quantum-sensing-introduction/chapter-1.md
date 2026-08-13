---
title: "Chapter 1: Principles of Quantum Sensing"
chapter_title: "Chapter 1: Principles of Quantum Sensing"
subtitle: Ramsey as the Universal Template, the Sensitivity η, and the Stability That Decides Whether Averaging Helps
reading_time: 40-45 minutes
difficulty: Intermediate
code_examples: 6
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/MS/quantum-sensing-introduction/chapter-1.html>) | Last sync: 2026-08-13

[Materials Science Dojo](<../index.html>) > [Introduction to Quantum Sensing](<index.html>) > Chapter 1

A quantum sensor is a two-level system that has been asked a different question. The sister course [Introduction to Quantum Hardware](<../../FM/quantum-hardware-introduction/index.html>) spends five chapters on how to stop a two-level system from noticing its environment, because a computer that notices its environment has made an error. This course inverts the sign of that sentence. Here the environment is the signal, decoherence is the measurement, and the figure of merit is not how long the phase survives but how much information about a field was written into it before it died.

The inversion is not a metaphor. It is the same Hamiltonian, the same $\pi/2$ pulses, the same Ramsey sequence and the same $T_2$; only the intention has changed. That is why this chapter can be short about the physics of two-level systems — the hardware course established $T_1$, $T_2$, $T_2^\ast$, Ramsey and echo, and [its Chapter 1](<../../FM/quantum-hardware-introduction/chapter-1.html>) is the reference for all of them, used here without re-derivation — and long about the three things a measurement scientist needs that a computer scientist does not: a sensitivity with units, a stability statement that says whether averaging helps, and a map of what can be measured at what spatial resolution.

The chapter's job is to establish the template that Chapters 2, 3 and 4 are three variations on. Nitrogen-vacancy magnetometry, the dc SQUID, the optical clock and the atom interferometer look like four different subjects, and in the laboratory they are. Written out as interferometers they are one subject with four sets of hardware, and the quantity they all report is a phase divided by a coupling constant. Section 1.2 makes that claim precisely, and the rest of the course spends four chapters cashing it.

## Learning Objectives

After completing this chapter, you will be able to:

  * Explain the two properties that make a quantum system a good instrument — a discrete transition that acts as a frequency standard, and a phase that integrates a field — and identify both in each platform of this course
  * Write any of the four sensing modalities in this course as a Ramsey interferometer, naming the beamsplitter, the phase, and the readout in each case
  * Derive the projection-noise limit $\delta\phi = 1/(C\sqrt{N})$ from binomial statistics, verify the $1/\sqrt{N}$ scaling numerically, and state what the standard quantum limit does and does not forbid
  * Define the sensitivity $\eta$ in units of signal per root hertz, use it correctly to predict the resolution reached after a stated averaging time, and derive the optimal interrogation time $\tau \approx T_2/2$ including the dead-time correction
  * Compute an Allan deviation, recognize the white, flicker and random-walk regimes from its slope, and explain why a sensitivity quoted without a stability curve is an incomplete specification
  * Use the filter function of a pulse sequence to select a measurement band, and read the same machinery in both directions — as decoupling from noise and as spectroscopy of it
  * Place a measurement problem on the sensitivity-against-resolution map and decide, from the scaling laws alone, whether any quantum sensor can address it

### Conventions and Units

Five conventions, fixed here and used unchanged for the rest of the course. Three of them are inherited from the hardware course; two are new because sensing needs them.

**Inherited: reduced units, angular versus cyclic frequency, and the coherence times.** Hamiltonians are written with $\hbar = 1$. Symbols such as $\Omega$ (Rabi frequency), $\Delta$ (detuning) and $\gamma$ (gyromagnetic ratio) are *angular*, in rad/s or rad s$^{-1}$ T$^{-1}$; quoted numbers are *cyclic*, written $\Omega/2\pi$ or $\gamma/2\pi$. $T_1$, $T_2$, $T_2^\ast$, the bound $T_2 \le 2T_1$, the Ramsey and Hahn-echo sequences, and the filter-function formalism mean exactly what they mean in [quantum-hardware Chapter 1](<../../FM/quantum-hardware-introduction/chapter-1.html>) §1.4, and this course does not redefine them.

**New: the sensitivity $\eta$.** Written throughout as a signal amplitude per root hertz — T/$\sqrt{\mathrm{Hz}}$ for a magnetometer, V m$^{-1}$/$\sqrt{\mathrm{Hz}}$ for an electrometer, K/$\sqrt{\mathrm{Hz}}$ for a thermometer. The definition is fixed once in §1.3: $\eta = \delta X_\mathrm{min}\sqrt{T}$, where $\delta X_\mathrm{min}$ is the uncertainty after total averaging time $T$. It is an amplitude spectral density, not a power spectral density, and the two differ by a square — a distinction that is worth checking every single time, because both conventions appear in the literature and they disagree by orders of magnitude.

**New: contrast $C$.** The fringe amplitude actually achieved, $0 < C \le 1$, absorbing every imperfection between the ideal interferometer and the recorded signal: decoherence, imperfect initialization, imperfect pulses, and readout that does not fully resolve the two outcomes. $C$ enters every sensitivity formula in the denominator, and it is usually the cheapest thing to improve.

* * *

## 1.1 What Makes a Quantum System a Good Instrument

### Two Properties, Not One

Every measuring instrument needs a reference and a transducer. A ruler has both: the engraved marks are the reference, and the act of laying it alongside the object is the transducer. A quantum two-level system happens to supply both at once, which is why it is such an unreasonably good instrument.

**The reference is a discrete transition.** Two levels separated by an energy $\hbar\omega_0$ define a frequency, and because the levels come from a Hamiltonian rather than from a fabrication step, that frequency is reproducible. Every $^{87}\mathrm{Rb}$ atom has the same ground-state hyperfine splitting; every nitrogen-vacancy centre in diamond has the same zero-field splitting; every superconducting loop encloses flux in the same quantum $\Phi_0 = h/2e$. A calibration that relies on such a number does not drift, because there is nothing in it to drift. This is the property that makes atomic clocks possible, and it is also what makes a quantum magnetometer *absolute*: the conversion from measured frequency to field is a fundamental constant, so the instrument needs no calibration against a reference magnet.

**The transducer is the phase.** Prepare a superposition of the two levels and the relative phase evolves as $\phi(t) = \int_0^t \delta\omega(t')\,dt'$, where $\delta\omega$ is whatever shift the external world imposed on the splitting. The phase is therefore a *time integral of the perturbation*, accumulated by the system itself, with no amplifier in the loop and no thermal noise added. Reading it out at the end recovers the integral. The system is a perfect integrator until it decoheres, and how long that is sets everything else.

Neither property alone is enough. A stable reference with no coupling is a clock that cannot be read; a strong coupling with no stable reference is a thermometer without a scale. The tension between them is exactly the one the hardware course opens with — isolation against controllability — and it has the same resolution here, frequency selectivity, developed in §1.4.

### The Perturbations Worth Measuring

What can shift a level splitting? The list is short, and it is the list of quantities quantum sensors measure.

| Perturbation | Coupling term | Shift of the splitting | Sensor of choice in this course |
| --- | --- | --- | --- |
| Magnetic field | $-\boldsymbol{\mu}\cdot\mathbf{B}$ | $\delta\omega = \gamma B$, linear | NV centre (Ch. 2), SQUID (Ch. 3), vapour cell (Ch. 4) |
| Electric field | $-\mathbf{d}\cdot\mathbf{E}$ | linear for a polar defect, quadratic (Stark) otherwise | NV centre (Ch. 2) |
| Temperature | $\partial D/\partial T$ via lattice expansion | linear in $\delta T$ over a working range | NV centre (Ch. 2) |
| Strain and pressure | crystal-field change | linear in the strain tensor component | NV centre (Ch. 2) |
| Rotation and acceleration | path-dependent action | Sagnac and gravimetric phases | atom interferometer (Ch. 4) |
| Magnetic flux | $2\pi\Phi/\Phi_0$ | exactly periodic, not merely linear | dc SQUID (Ch. 3) |

Two entries deserve a note now. The flux row is the odd one out because the response is *periodic* rather than linear — a SQUID reports a phase modulo $2\pi$ and must be operated in a feedback loop to unwrap it, which is Chapter 3's flux-locked loop. And the temperature and strain rows are what turn a magnetometer into a multi-parameter instrument whether the experimenter wants it or not: a measured shift of the NV splitting is a magnetic field only after the thermal and strain contributions have been separated, which is a large fraction of the practical work in Chapter 2.

### Why This Is a Characterization Technique

This course sits in the Materials Science Dojo rather than in the quantum dojos, and the reason is visible already. Every quantity in the table above is a materials quantity when it is measured with spatial resolution. A magnetic field mapped at 50 nm over a thin film is a magnetic-domain image. The same map over a current-carrying device is a current-density reconstruction. A relaxation rate $1/T_1$ measured as a function of position is a map of gigahertz magnetic noise, which is to say of spin fluctuations in the sample. A temperature map at 100 nm inside an operating transistor is a thermal-management measurement that no thermocouple can make.

The instruments that already live in this dojo — [X-ray diffraction](<../xrd-analysis-introduction/index.html>), [electron microscopy](<../electron-microscopy-introduction/index.html>), [spectroscopy](<../spectroscopy-introduction/index.html>), [electrical and magnetic testing](<../electrical-magnetic-testing-introduction/index.html>) — each answer a question about structure, composition or bulk response. Quantum sensors answer a question none of them do: what is the *local field* here, now, without touching the sample and without an ensemble average over the whole specimen. That is the gap this course is about.

* * *

## 1.2 The Ramsey Protocol: One Template for the Whole Course

### The Three Steps, and the Fringe

Every measurement in this course has the same three steps.

  1. **Split.** From an initialized state, prepare an equal superposition of the two levels. In a spin system this is a $\pi/2$ pulse; in an atom interferometer it is a laser pulse that splits the wavepacket in momentum; in a SQUID the "splitting" is spatial and permanent, since the supercurrent traverses both arms of the loop at once.
  2. **Accumulate.** Let the superposition evolve freely for a time $\tau$. The relative phase grows as $\phi = \int_0^\tau \delta\omega\,dt$, which for a static perturbation is simply $\phi = \delta\omega\,\tau$.
  3. **Recombine and read.** Apply a second $\pi/2$ pulse, which converts the invisible phase into a visible population difference, and measure. The probability of the second outcome is

$$ P = \frac{1}{2}\left[1 - C\cos\phi\right] $$

with $C$ the contrast. Sweeping either $\tau$ or the perturbation traces out **Ramsey fringes**, and the entire art of the subject is arranging for the fringe to be steep where the signal sits.

Step 3 is where the quantum measurement problem is disposed of. The phase is not observable; a population is. The final pulse is an interferometric recombination that maps one onto the other, and it is the only reason a phase-accumulating system can be used as an instrument at all.

### The Same Interferometer, Four Times

The claim that the four platforms of this course are one protocol is worth writing out as a table, because it is the organizing device of the whole series.

| | Beamsplitter | Phase accumulated from | Recombination | Readout |
| --- | --- | --- | --- | --- |
| **NV magnetometry** (Ch. 2) | microwave $\pi/2$ pulse | Zeeman shift, $\phi = \gamma B \tau$ | second $\pi/2$ pulse | spin-dependent fluorescence |
| **dc SQUID** (Ch. 3) | the loop itself, permanently | flux, $\phi = 2\pi\Phi/\Phi_0$ | the second junction | critical current, read as a voltage |
| **Atomic clock** (Ch. 4) | $\pi/2$ pulse on the clock transition | detuning of the local oscillator, $\phi = \Delta\tau$ | second $\pi/2$ pulse | fluorescence or electron shelving |
| **Atom interferometer** (Ch. 4) | laser pulse, splitting in momentum | path-dependent action, including $\mathbf{g}$ and $\boldsymbol{\Omega}$ | final laser pulse | populations in the two output ports |

Read across any row and the sequence is the same. Read down any column and the physics is completely different — which is why the four chapters exist. But the *sensitivity formula* depends only on the row structure, and that is why it can be derived once, in §1.3, and used four times.

A fifth row is worth keeping in mind even though this course does not devote a chapter to it: nuclear magnetic resonance is the same interferometer with nuclear spins, and its free-induction decay is a Ramsey measurement in which the recombination is performed by the detection coil rather than by a pulse. Anyone who has run an NMR experiment has already run everything in this section.

### Ramsey Versus Rabi: Why the Free Interval Wins

There is an older way to measure a level splitting: sweep a continuous drive across the transition and find the resonance. This is the Rabi method, and it is what a continuous-wave ODMR spectrum is in Chapter 2. It works, and it has a resolution problem: the linewidth of a driven resonance is set by the drive duration *and* by the drive strength, so making the line narrow requires driving weakly for a long time — during which the drive itself is perturbing the levels it is measuring.

Ramsey's insight was to separate the two functions in time. The pulses are short and strong, so they are efficient and their own perturbation is brief; the interval between them is long and completely free, so the system evolves under the quantity being measured and nothing else. The fringe spacing is then $1/\tau$, set only by the free interval. Every high-precision measurement in this course therefore uses separated pulses, and where a continuous method appears it is for finding the resonance before the Ramsey sequence refines it.

* * *

## 1.3 How Well Can It Be Done? Projection Noise, the SQL, and $\eta$

### The Irreducible Noise of a Quantum Measurement

Reading out the final state of one two-level system gives one bit: $\lvert 0 \rangle$ or $\lvert 1 \rangle$. The information about $\phi$ is in the *probability* $P(\phi)$, and a probability is estimated from counts. With $N$ independent systems measured once each — or one system measured $N$ times — the number of $\lvert 1 \rangle$ outcomes is binomial, so the estimate $\hat{P} = k/N$ has standard error

$$ \delta P = \sqrt{\frac{P(1-P)}{N}} $$

This is **projection noise**, sometimes called quantum projection noise or spin projection noise. It is not technical noise. It is a consequence of the measurement postulate: a superposition does not have a value of the observable before it is measured, and the randomness of the outcome is irreducible.

Converting a probability error into a phase error requires the slope of the transfer function, $|dP/d\phi| = \tfrac{1}{2}C|\sin\phi|$:

$$ \delta\phi = \frac{\delta P}{\lvert dP/d\phi \rvert} = \frac{2\sqrt{P(1-P)}}{C\lvert\sin\phi\rvert\sqrt{N}} $$

Substituting $P = \tfrac{1}{2}(1 - C\cos\phi)$ gives $4P(1-P) = 1 - C^2\cos^2\phi$, and hence the exact result

$$ \delta\phi = \frac{\sqrt{1 - C^2\cos^2\phi}}{C\lvert\sin\phi\rvert\sqrt{N}} $$

Two readings of this formula matter. At **unit contrast** it collapses to $\delta\phi = 1/\sqrt{N}$ *everywhere on the fringe*, because the numerator and the denominator both carry the same factor $|\sin\phi|$ and they cancel. That is a genuine and slightly surprising fact, and Code Example 2 confirms it. At **finite contrast** the cancellation fails, and the minimum sits at the quadrature point $\phi = \pi/2$, where

$$ \delta\phi_\mathrm{SQL} = \frac{1}{C\sqrt{N}} $$

This is the **standard quantum limit** (SQL) for phase estimation with $N$ uncorrelated probes. It is why real experiments bias the interferometer to the steepest point of the fringe: not because projection noise is smaller there in the ideal case, but because every real imperfection makes it smaller there.

What the SQL forbids and permits is worth stating carefully, because both halves are misquoted. It forbids doing better than $1/\sqrt{N}$ *with uncorrelated probes and a single independent measurement each*. It does not forbid doing better than $1/\sqrt{N}$ at all: entangled probes can reach $\delta\phi \sim 1/N$, the Heisenberg limit, which is Chapter 5's subject and — Chapter 5 will argue — is much harder to keep than to reach.

### The Sensitivity $\eta$, Defined Once

A phase uncertainty is not a useful specification. Divide by the transduction coefficient to get a field uncertainty, $\delta B = \delta\phi/(\gamma\tau)$, and then account for the fact that the experiment is repeated. In a total measurement time $T$ the sequence runs $M = T/(\tau + t_d)$ times, where $t_d$ is the **dead time** spent per shot on initialization and readout. Averaging $M$ independent estimates improves the result by $\sqrt{M}$, so

$$ \delta B(T) = \frac{1}{\gamma\,\tau\,C(\tau)\sqrt{N}}\sqrt{\frac{\tau + t_d}{T}} $$

The $T$-dependence is a bare $1/\sqrt{T}$, which is what makes it possible to quote a single number for the instrument. Define

$$ \boxed{\;\eta \;\equiv\; \delta B(T)\,\sqrt{T} \;=\; \frac{\sqrt{\tau + t_d}}{\gamma\,\tau\,C(\tau)\,\sqrt{N}}\;} \qquad \left[\mathrm{T}/\sqrt{\mathrm{Hz}}\right] $$

and the instrument is specified by $\eta$ alone: the field resolution after averaging for a time $T$ is $\eta/\sqrt{T}$. Three warnings come with the definition, and all three are routinely violated in practice.

  * **$\eta$ is an amplitude spectral density.** Its square is a power spectral density. A magnetometer quoted at 1 pT/$\sqrt{\mathrm{Hz}}$ has a noise power of $10^{-24}\ \mathrm{T}^2$/Hz, and confusing the two is a factor of $10^{12}$.
  * **$\eta$ presumes the $1/\sqrt{T}$ law.** It is a valid extrapolation only over the averaging times where the noise is white. §1.4 is entirely about when that fails, and it fails sooner than anyone would like.
  * **$\eta$ says nothing about bandwidth or dynamic range.** A sensor with a superb $\eta$ may be usable only over a few nanotesla, because the Ramsey fringe is $2\pi$-periodic and $\tau$ was made long.

### The Optimal Interrogation Time

The formula for $\eta$ contains a genuine optimum. Longer $\tau$ helps linearly through the factor $\gamma\tau$ and hurts exponentially through $C(\tau) = \exp[-(\tau/T_2)^p]$, where $p = 1$ for exponential and $p = 2$ for Gaussian decay. With $t_d = 0$,

$$ \eta \propto \frac{1}{\sqrt{\tau}\,\exp\left[-(\tau/T_2)^p\right]} \quad \Longrightarrow \quad \tau_\mathrm{opt} = \frac{T_2}{(2p)^{1/p}} $$

which is $T_2/2$ for both $p = 1$ and $p = 2$ — an accident of the arithmetic that is convenient enough to remember. At that point the contrast has fallen to $e^{-1/2} = 0.61$ (exponential) or $e^{-1/4} = 0.78$ (Gaussian), and

$$ \eta_\mathrm{min} = \frac{\sqrt{2e}}{\gamma\sqrt{N\,T_2}} \qquad (p = 1,\; t_d = 0) $$

The structure of that result is the single most useful thing in the chapter. **Only the product $N T_2$ appears.** One spin with a one-second coherence time and $10^{12}$ spins with a picosecond coherence time have, on this formula, identical sensitivity. Every design decision in Chapters 2 to 4 is a trade within that product: a dense ensemble buys $N$ and pays in $T_2$ through dipolar coupling; a dilute one does the reverse; a cryogenic apparatus buys $T_2$ and pays in everything else.

The formula also identifies where it stops being honest. Dead time breaks the symmetry of $N T_2$, because a short $T_2$ means many short shots and therefore a duty cycle dominated by $t_d$. Code Example 3 makes both statements quantitative, and the second one is why "just use more spins" is not a strategy.

* * *

## 1.4 Noise and Stability: When Does Averaging Stop Helping?

### The Question $\eta$ Cannot Answer

Suppose a magnetometer is specified at $\eta = 1\ \mathrm{pT}/\sqrt{\mathrm{Hz}}$ and the measurement needs 1 fT. The naive arithmetic says average for $(10^{-12}/10^{-15})^2 = 10^6$ seconds, i.e. twelve days, and go home. The arithmetic is wrong, and it is wrong in a way that no improvement to $\eta$ can fix: after some averaging time the reading stops improving and starts getting worse, because the instrument's own reference has drifted by more than the noise you were trying to average away.

This is not a defect of a particular sensor. It is the generic behaviour of any physical reference, and the tool for describing it is the Allan deviation, borrowed wholesale from the time-and-frequency community — which is fitting, since Chapter 4's atomic clocks are where it was invented.

### The Allan Deviation

Take a record of readings $y_k$, each an average over a time $\tau_0$. Group them into bins of $m$ consecutive readings, so each bin average $\bar{y}_j$ represents an averaging time $\tau = m\tau_0$. The **Allan variance** is half the mean square difference of *successive* bin averages:

$$ \sigma_y^2(\tau) = \frac{1}{2}\left\langle \left(\bar{y}_{j+1} - \bar{y}_j\right)^2 \right\rangle $$

The Allan deviation $\sigma_y(\tau)$ is its square root. Two features explain why this and not the ordinary standard deviation. It uses only *differences of neighbours*, so a slow drift contributes to it only through how much the signal moved during one averaging interval, and it therefore converges for noise processes whose ordinary variance does not exist — $1/f$ noise among them. And the factor $\tfrac{1}{2}$ is chosen so that for white noise $\sigma_y(\tau)$ equals the ordinary standard deviation of the bin averages, making the two agree in the one case where both are valid.

The diagnostic power is in the slope. Each noise process has its own power law, and they are distinguishable by eye on a log-log plot:

| Regime | Power spectral density | $\sigma_y(\tau)$ | Slope | Physical origin |
| --- | --- | --- | --- | --- |
| White | $S_y \propto f^0$ | $\propto \tau^{-1/2}$ | $-1/2$ | projection noise, photon shot noise, amplifier noise |
| Flicker | $S_y \propto 1/f$ | constant | $0$ | two-level fluctuators, temperature fluctuations, $1/f$ everything |
| Random walk | $S_y \propto 1/f^2$ | $\propto \tau^{+1/2}$ | $+1/2$ | slow thermal drift of the reference, magnetic shield relaxation |
| Linear drift | deterministic | $= D\tau/\sqrt{2}$ | $+1$ | ageing, monotone temperature ramp |

The white-noise regime is the only one in which $\eta$ means anything, because it is the only one where $1/\sqrt{T}$ holds. The flicker plateau is the sensor's **floor**: below it, no amount of averaging helps. The rising branch is worse than useless — averaging longer there actively degrades the answer.

Two consequences run through the rest of the course. First, a sensitivity specification without a stability curve is incomplete, and the useful pair of numbers is $\eta$ *and* the averaging time at which $\sigma_y(\tau)$ turns over. Second, the standard escape is not to average longer but to modulate: chop the signal at a frequency above the flicker corner, and the measurement lives in the white region no matter how long the total experiment runs. Every serious magnetometry measurement in Chapters 2 and 3 is a modulated measurement for exactly this reason, and Chapter 4 shows the same logic applied to a clock.

### Filter Functions: the Same Machinery, Read Both Ways

Modulating a measurement is the same operation as decoupling a qubit, and the formalism is the one already established in the hardware course. A sequence of $\pi$ pulses imposes a switching function $s(t) \in \lbrace +1, -1 \rbrace$ on the accumulating phase, so that

$$ \phi(\tau) = \int_0^\tau s(t)\,\delta\omega(t)\,dt , \qquad \left\langle\phi^2\right\rangle = \int_0^\infty S(f)\,\left|\tilde{s}(f,\tau)\right|^2 df $$

where $\tilde{s}(f,\tau) = \int_0^\tau s(t)e^{-2\pi i f t}dt$ and $|\tilde{s}|^2$ is the **filter function**. [Quantum-hardware Chapter 1](<../../FM/quantum-hardware-introduction/chapter-1.html>) §1.4 derives this and gives the closed forms for free induction and the Hahn echo; this course uses them without repeating the derivation. What is new here is the reading.

**Read as protection**, a CPMG-$N$ sequence places its passband near $f \approx N/2\tau$ and has a zero at DC, so it rejects the slow noise that would otherwise limit $T_2$. That is the hardware course's use.

**Read as a measurement**, the identical filter is a lock-in amplifier. A field oscillating at $f_\mathrm{ac}$ that is *synchronized* with the pulse train no longer averages to zero: the $\pi$ pulses flip the sign of the accumulation exactly when the field changes sign, so the contributions rectify and add. For a square-wave filter matched to a sinusoid the rectification efficiency is $2/\pi$, giving

$$ \phi = \frac{2}{\pi}\,\gamma\,B_\mathrm{ac}\,\tau \qquad \Longrightarrow \qquad \eta_\mathrm{AC} = \frac{\pi}{2}\,\eta_\mathrm{DC}\Big|_{\tau \to T_2} $$

The factor $\pi/2$ is a small loss; the gain is that $\tau$ may now be as long as $T_2$ rather than $T_2^\ast$, which in a solid-state sensor is one to two decades. AC magnetometry is more sensitive than DC magnetometry, and the reason is entirely this.

**Read as spectroscopy**, sweeping $N$ moves the passband across the environment and the family of coherence curves inverts to give $S(f)$. For $S(f) = A/f^{\alpha}$ the coherence time follows $T_2(N) \propto N^{\alpha/(\alpha+1)}$, so the *exponent* of a decoupling curve measures the exponent of the noise spectrum, which is a statement about the defect ensemble in the host material. This is the sense in which the sensor characterizes its own crystal, and it is the direct ancestor of the $T_1$ relaxometry of Chapter 2.

* * *

## 1.5 What Can Be Measured, and at What Resolution

### The Trade-off Is Not Negotiable

Sensitivity and spatial resolution are the same conversation, and the reason is geometry rather than engineering. Consider a sensor of linear size $d$ containing spins at density $n$, so $N = n d^3$. From $\eta \propto 1/\sqrt{N T_2}$,

$$ \eta_B \;\propto\; \frac{1}{\sqrt{n\,d^3\,T_2}} \;\propto\; d^{-3/2} $$

A bigger sensor is a better magnetometer, by three halves of a power. But the spatial resolution is $d$ itself, so improving the resolution by a decade costs a factor of $10^{3/2} \approx 32$ in field sensitivity. There is no arrangement of the same physics that avoids this.

The trade-off does invert if the quantity of interest is a *source* rather than a field. A magnetic dipole $m$ at standoff $z$ produces $B \sim \mu_0 m/4\pi z^3$, and a sensor small enough to sit at $z \approx d$ therefore detects a minimum moment

$$ m_\mathrm{min} \;\sim\; \frac{4\pi d^3}{\mu_0}\,\eta_B \;\propto\; d^{3/2} $$

*Smaller is better* for moment sensitivity, by the same three halves. This is why the two branches of the field look so different in practice: a millimetre-scale vapour cell or SQUID is unbeatable for measuring a weak field over a large region, and a single defect at the apex of a scanning tip is unbeatable for measuring a small object. They are the same formula read in opposite directions.

### Where the Scaling Stops

Two limits deserve to be named before the chapters that hit them.

**Density does not buy sensitivity indefinitely.** Increasing $n$ raises $N$, but it also shortens $T_2$, because the dominant dephasing mechanism in a dense spin ensemble is the dipolar field of the other spins, giving $T_2 \propto 1/n$. The product $N T_2$ is then *independent of $n$*, and the sensitivity stops improving entirely. Code Example 6 shows the cancellation explicitly. The escape routes are all interesting and all difficult: decoupling the ensemble from itself, using a species with a smaller moment, or accepting the inhomogeneity and modulating faster.

**Standoff is brutal.** The $z^{-3}$ falloff of a dipole field means one decade of standoff costs three decades of signal and therefore six decades in the required $N T_2$. This single scaling law explains why nanoscale magnetic imaging is a surface technique, why sensor-to-sample distance is the number reported first in every scanning-probe magnetometry paper, and why so much of the practical difficulty in Chapter 2 is about how close a defect can be placed to a crystal surface without the surface destroying it.

### The Map

Putting the two axes together gives the map this course navigates by. It is deliberately stated as scalings and regimes, not as numbers: the numbers move, and the scalings do not.

| Configuration | Resolution scale | Sensitivity scale | What it is for |
| --- | --- | --- | --- |
| Single defect on a scanning tip | set by standoff, tens of nm | poorest $\eta_B$, best $m_\mathrm{min}$ | domain walls, skyrmions, single molecules, edge currents |
| Shallow defect layer, wide-field imaging | optical, a few hundred nm | intermediate | current maps of devices, magnetic textures over large fields of view |
| Bulk ensemble, millimetre scale | none — it averages | best $\eta_B$ | susceptibility, bulk magnetization, unshielded field measurement |
| Superconducting pickup loop | loop size | best $\eta_\Phi$ per unit area | flux imaging, susceptometry, anything cryogenic already |
| Vapour cell | cell size, mm | excellent, no cryogenics | field measurement where the sample cannot be cooled |
| Atom interferometer | metres — it is the baseline | best inertial | gravity, rotation, equivalence-principle tests |

Reading the last column is the honest way to choose an instrument, and it is the structure of the rest of the course: Chapter 2 takes the first two rows, Chapter 3 the fourth, Chapter 4 the fifth and sixth, and Chapter 5 asks what entanglement adds to any of them.

* * *

## 1.6 A Numerical Laboratory

Six examples build the toolkit the whole course uses. The first three establish the sensitivity formula and verify its statistical content; the fourth and fifth establish stability and band selection; the sixth builds the resolution map. Everything is NumPy and SciPy, and every number below was produced by the code above it.

### Code Example 1: Phase Accumulation, and the Sensitivity $\eta$

The starting point is one line of arithmetic — how much phase a given field writes into a given system in a given time — followed by the definition of $\eta$ in a form the rest of the course calls as a function.

```python
import numpy as np

# Fundamental constants (SI). Every number in this course is either a
# fundamental constant or a property of a particle; no device or vendor
# specification appears anywhere.
h = 6.62607015e-34            # Planck constant, J s
hbar = 1.054571817e-34        # reduced Planck constant, J s
mu_B = 9.2740100783e-24       # Bohr magneton, J/T
Phi0 = 2.067833848e-15        # magnetic flux quantum, Wb
kB = 1.380649e-23             # Boltzmann constant, J/K
mu0 = 1.25663706212e-6        # vacuum permeability, N/A^2

NT = 1e-9                     # one nanotesla
US = 1e-6                     # one microsecond

# Angular gyromagnetic ratios, rad s^-1 T^-1: the phase a spin accumulates
# per second per tesla. These are particle properties, not specifications.
systems = [
    ("electron spin, S = 1/2", 1.76085963e11),
    ("Rb-87 ground state, g_F = 1/2", 0.5 * mu_B / hbar),
    ("proton, 1H", 2.675221874e8),
    ("Xe-129 nucleus", 7.441e7),
]

tau = 1.0 * US
print("A two-level system as a field integrator: phi = gamma B tau")
hdr = (f"{'system':<32}{'gamma/2pi':>16}{'phi per nT':>17}"
       f"{'B for 1 rad':>16}")
print(hdr)
print("-" * len(hdr))
for name, g in systems:
    print(f"{name:<32}{g/(2*np.pi)/1e6:>11.4g} MHz/T"
          f"{g*NT*tau:>13.3e} rad{1.0/(g*tau)/NT:>12.4g} nT")
print(f"(interrogation time tau = {tau/US:.0f} us throughout the table)")

# The same statement for a superconducting loop, where the phase is set by
# the flux through the loop rather than by a Zeeman energy: phi = 2 pi Phi/Phi0.
print("\nThe loop version of the same idea: phi = 2 pi Phi / Phi0")
hdr2 = f"{'loop side':<14}{'area':>14}{'B for one Phi0':>18}{'phi per nT':>16}"
print(hdr2)
print("-" * len(hdr2))
for label, side in [("10 um", 10e-6), ("100 um", 100e-6), ("1 mm", 1e-3)]:
    A = side ** 2
    print(f"{label:<14}{A:>11.3e} m2{Phi0/A/NT:>15.4g} nT"
          f"{2*np.pi*NT*A/Phi0:>12.3e} rad")


def eta_ramsey(gamma, tau, T2, N=1.0, p=1.0, t_dead=0.0):
    """Magnetic-field sensitivity of a Ramsey measurement, in T/sqrt(Hz).

    eta = dB_min * sqrt(T_total): the field uncertainty left after averaging
    for one second. tau is the free-precession time, T2 the coherence time
    that limits the fringe contrast as C = exp(-(tau/T2)**p), N the number of
    independent spins read out per shot, and t_dead the time spent per shot
    on initialization and readout.
    """
    C = np.exp(-(tau / T2) ** p)
    return np.sqrt(tau + t_dead) / (gamma * tau * C * np.sqrt(N))


gamma_e = systems[0][1]
print("\nSensitivity of one electron spin at the optimal tau = T2/2, N = 1")
hdr3 = (f"{'T2':>10}{'tau = T2/2':>13}{'eta numeric':>18}"
        f"{'sqrt(2e)/(gamma sqrt(T2))':>27}")
print(hdr3)
print("-" * len(hdr3))
for label, T2 in [("1 us", 1e-6), ("10 us", 1e-5), ("100 us", 1e-4),
                  ("1 ms", 1e-3), ("1 s", 1.0)]:
    num = eta_ramsey(gamma_e, T2 / 2, T2)
    ana = np.sqrt(2 * np.e) / (gamma_e * np.sqrt(T2))
    print(f"{label:>10}{T2/2/US:>10.4g} us{num/NT:>12.4g} nT/rtHz"
          f"{ana/NT:>21.4g} nT/rtHz")
print("eta scales as 1/sqrt(T2): a hundredfold longer coherence buys one")
print("decade of sensitivity, and nothing else in the formula is free.")
```

```text
A two-level system as a field integrator: phi = gamma B tau
system                                 gamma/2pi       phi per nT     B for 1 rad
---------------------------------------------------------------------------------
electron spin, S = 1/2            2.802e+04 MHz/T    1.761e-04 rad        5679 nT
Rb-87 ground state, g_F = 1/2          6998 MHz/T    4.397e-05 rad   2.274e+04 nT
proton, 1H                            42.58 MHz/T    2.675e-07 rad   3.738e+06 nT
Xe-129 nucleus                        11.84 MHz/T    7.441e-08 rad   1.344e+07 nT
(interrogation time tau = 1 us throughout the table)

The loop version of the same idea: phi = 2 pi Phi / Phi0
loop side               area    B for one Phi0      phi per nT
--------------------------------------------------------------
10 um           1.000e-10 m2      2.068e+04 nT   3.039e-04 rad
100 um          1.000e-08 m2          206.8 nT   3.039e-02 rad
1 mm            1.000e-06 m2          2.068 nT   3.039e+00 rad

Sensitivity of one electron spin at the optimal tau = T2/2, N = 1
        T2   tau = T2/2       eta numeric  sqrt(2e)/(gamma sqrt(T2))
--------------------------------------------------------------------
      1 us       0.5 us       13.24 nT/rtHz                13.24 nT/rtHz
     10 us         5 us       4.187 nT/rtHz                4.187 nT/rtHz
    100 us        50 us       1.324 nT/rtHz                1.324 nT/rtHz
      1 ms       500 us      0.4187 nT/rtHz               0.4187 nT/rtHz
       1 s     5e+05 us     0.01324 nT/rtHz              0.01324 nT/rtHz
eta scales as 1/sqrt(T2): a hundredfold longer coherence buys one
decade of sensitivity, and nothing else in the formula is free.
```

**What to look for.** The first table is the conversion to keep. An electron spin accumulates $1.76\times10^{-4}$ rad per nanotesla per microsecond, so one radian of phase — a fully resolved fringe — needs about 5.7 $\mu$T at $\tau = 1\ \mu$s, or 5.7 nT at $\tau = 1$ ms. The four rows span three and a half decades of $\gamma$, and that single factor is why electron-spin sensors dominate small-volume magnetometry while nuclear-spin sensors are used where their much longer coherence more than compensates.

The loop table makes the same point in the language of Chapter 3. A 1 mm loop encloses one flux quantum at 2.1 nT, so a SQUID's phase response to a nanotesla is 3 rad — five orders of magnitude more phase per nanotesla than an electron spin interrogated for a microsecond. That is the whole reason a SQUID is the most sensitive magnetometer per unit volume ever built, and it is also why it is useless at 50 nm resolution: the sensitivity came from the area.

The last block confirms that the closed form $\eta_\mathrm{min} = \sqrt{2e}/(\gamma\sqrt{N T_2})$ reproduces the numerical minimum exactly, and it fixes the scale that recurs for the rest of the course: a single electron spin with a millisecond of coherence is a 0.42 nT/$\sqrt{\mathrm{Hz}}$ magnetometer. That number is not impressive on its own — until §1.5's map recalls that it is being achieved in a volume of a few cubic nanometres.

### Code Example 2: The Transfer Function, the Working Point, and Projection Noise

The standard quantum limit is a statistical statement, and it deserves a Monte Carlo rather than an assertion. This example first tabulates where on the fringe to sit, then draws binomial samples and measures the scaling of the resulting phase error against $N$.

```python
"""Chapter 1, Example 2: the Ramsey transfer function, the working point and
projection noise. Continues from Example 1 (same session)."""


def ramsey_probability(phi, contrast=1.0):
    """Probability of the |1> outcome after a Ramsey sequence.

    The two pi/2 pulses convert the accumulated phase phi into a population.
    phi includes any deliberate bias phase, which is what selects the
    working point on the fringe.
    """
    return 0.5 * (1.0 - contrast * np.cos(phi))


def phase_uncertainty(phi, N, contrast=1.0):
    """Single-shot phase uncertainty from projection noise alone.

    Error propagation: the measured quantity is the fraction of |1> outcomes,
    whose standard error is sqrt(P(1-P)/N); dividing by the slope |dP/dphi|
    converts that into a phase.
    """
    P = ramsey_probability(phi, contrast)
    slope = 0.5 * contrast * np.abs(np.sin(phi))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(slope > 1e-9, np.sqrt(P * (1 - P) / N) / slope, np.inf)


print("Where to sit on the fringe: N = 10000 spins")
hdr = (f"{'bias phase':<12}{'P at C=1':>10}{'|dP/dphi|':>12}"
       f"{'d phi, C=1':>13}{'d phi, C=0.3':>15}{'C=0.3 penalty':>15}")
print(hdr)
print("-" * len(hdr))
N_demo = 10000
sql = 1.0 / np.sqrt(N_demo)
for label, phi in [("0", 0.0), ("pi/8", np.pi/8), ("pi/4", np.pi/4),
                   ("pi/2", np.pi/2), ("3pi/4", 3*np.pi/4), ("pi", np.pi)]:
    P = ramsey_probability(phi)
    slope = 0.5 * np.abs(np.sin(phi))
    d1 = phase_uncertainty(phi, N_demo)
    d2 = phase_uncertainty(phi, N_demo, contrast=0.3)
    print(f"{label:<12}{P:>10.4f}{slope:>12.4f}{d1:>13.5f}{d2:>15.5f}"
          f"{d2/(sql/0.3):>15.4f}")
print(f"Standard quantum limit 1/sqrt(N) = {sql:.5f}")
print("At unit contrast the projection noise is the same everywhere on the")
print("fringe -- P(1-P) and the slope both carry a factor sin(phi), and they")
print("cancel. Contrast breaks the tie: at C = 0.3 only the quadrature point")
print("phi = pi/2 still reaches 1/(C sqrt(N)).")

# --- Monte Carlo: does the estimator really follow 1/sqrt(N)? ---------------
rng = np.random.default_rng(20260813)
phi_true = 0.05                      # a small phase, read at quadrature
n_repeats = 4000

print(f"\nMonte Carlo at the quadrature point, phi_true = {phi_true}, "
      f"{n_repeats} repeats")
hdr = (f"{'N spins':>10}{'d phi (MC)':>14}{'1/sqrt(N)':>13}{'ratio':>9}"
       f"{'bias':>12}")
print(hdr)
print("-" * len(hdr))
Ns = np.array([10, 100, 1000, 10000, 100000, 1000000])
mc = []
for N in Ns:
    P = ramsey_probability(np.pi / 2 + phi_true)
    k = rng.binomial(N, P, size=n_repeats)
    phi_hat = 2.0 * (k / N - 0.5)            # linear inversion at quadrature
    mc.append(phi_hat.std(ddof=1))
    print(f"{N:>10d}{mc[-1]:>14.6f}{1/np.sqrt(N):>13.6f}"
          f"{mc[-1]*np.sqrt(N):>9.4f}{phi_hat.mean()-phi_true:>12.2e}")
mc = np.array(mc)
slope_fit = np.polyfit(np.log10(Ns), np.log10(mc), 1)[0]
print(f"log-log slope of d phi against N: {slope_fit:.4f}  "
      f"(standard quantum limit: -0.5)")

# --- Reduced contrast simply rescales the whole curve ------------------------
print("\nFinite contrast at the quadrature point, N = 10000")
print(f"{'contrast C':>12}{'d phi':>12}{'1/(C sqrt(N))':>16}")
print("-" * 40)
for C in [1.0, 0.7, 0.3, 0.1]:
    d = phase_uncertainty(np.pi / 2, N_demo, contrast=C)
    print(f"{C:>12.2f}{d:>12.6f}{1/(C*np.sqrt(N_demo)):>16.6f}")

# --- And the price of sensitivity: the fringe is 2 pi periodic --------------
print("\nDynamic range against sensitivity for one electron spin")
hdr = f"{'tau':>10}{'B for 2 pi (unambiguous)':>28}{'eta at N = 1':>20}"
print(hdr)
print("-" * len(hdr))
for label, t in [("1 us", 1e-6), ("10 us", 1e-5), ("100 us", 1e-4),
                 ("1 ms", 1e-3)]:
    B2pi = 2 * np.pi / (gamma_e * t)
    print(f"{label:>10}{B2pi/NT:>23.4g} nT"
          f"{np.sqrt(t)/(gamma_e*t)/NT:>13.4g} nT/rtHz")
print("Longer interrogation buys sensitivity and costs unambiguous range,")
print("in exactly compensating powers of tau: their product is fixed.")
```

```text
Where to sit on the fringe: N = 10000 spins
bias phase    P at C=1   |dP/dphi|   d phi, C=1   d phi, C=0.3  C=0.3 penalty
-----------------------------------------------------------------------------
0               0.0000      0.0000          inf            inf            inf
pi/8            0.0381      0.1913      0.01000        0.08369         2.5108
pi/4            0.1464      0.3536      0.01000        0.04607         1.3820
pi/2            0.5000      0.5000      0.01000        0.03333         1.0000
3pi/4           0.8536      0.3536      0.01000        0.04607         1.3820
pi              1.0000      0.0000          inf            inf            inf
Standard quantum limit 1/sqrt(N) = 0.01000
At unit contrast the projection noise is the same everywhere on the
fringe -- P(1-P) and the slope both carry a factor sin(phi), and they
cancel. Contrast breaks the tie: at C = 0.3 only the quadrature point
phi = pi/2 still reaches 1/(C sqrt(N)).

Monte Carlo at the quadrature point, phi_true = 0.05, 4000 repeats
   N spins    d phi (MC)    1/sqrt(N)    ratio        bias
----------------------------------------------------------
        10      0.311725     0.316228   0.9858   -2.65e-03
       100      0.099353     0.100000   0.9935    4.70e-04
      1000      0.031597     0.031623   0.9992    1.64e-04
     10000      0.009952     0.010000   0.9952   -2.25e-04
    100000      0.003190     0.003162   1.0087    3.87e-05
   1000000      0.001007     0.001000   1.0075   -1.74e-05
log-log slope of d phi against N: -0.4981  (standard quantum limit: -0.5)

Finite contrast at the quadrature point, N = 10000
  contrast C       d phi   1/(C sqrt(N))
----------------------------------------
        1.00    0.010000        0.010000
        0.70    0.014286        0.014286
        0.30    0.033333        0.033333
        0.10    0.100000        0.100000

Dynamic range against sensitivity for one electron spin
       tau    B for 2 pi (unambiguous)        eta at N = 1
----------------------------------------------------------
      1 us              3.568e+04 nT        5.679 nT/rtHz
     10 us                   3568 nT        1.796 nT/rtHz
    100 us                  356.8 nT       0.5679 nT/rtHz
      1 ms                  35.68 nT       0.1796 nT/rtHz
Longer interrogation buys sensitivity and costs unambiguous range,
in exactly compensating powers of tau: their product is fixed.
```

**What to look for.** The first table contains the result that most treatments skip. At unit contrast the phase uncertainty is $1/\sqrt{N}$ at every bias phase except the two extrema, because $\sqrt{P(1-P)} = \tfrac{1}{2}|\sin\phi|$ and $|dP/d\phi| = \tfrac{1}{2}|\sin\phi|$ are the same function. The received wisdom that one must work at quadrature is therefore not a statement about projection noise at all — it is a statement about everything else. Set $C = 0.3$ and the degeneracy lifts immediately: the penalty column shows a factor 2.5 for sitting at $\phi = \pi/8$, and only $\phi = \pi/2$ still attains $1/(C\sqrt{N})$.

The Monte Carlo is the load-bearing check. Six decades of $N$, 4000 repeats each, and the measured standard deviation tracks $1/\sqrt{N}$ to better than 1% throughout, with a fitted log-log slope of $-0.4981$ against the exact $-1/2$. The bias column is there to certify that the linear inversion around quadrature is not quietly systematic: the residual bias is $2.7\times10^{-3}$ at $N = 10$ and falls with $N$, which is the expected $O(\phi^3)$ curvature of the fringe rather than a defect of the estimator. This is what "standard quantum limit" means operationally — not a philosophical bound but a slope of $-1/2$ on a log-log plot, which any experiment can and should verify for itself.

The last block prices the sensitivity. Because the fringe is $2\pi$-periodic, an interrogation time long enough to resolve 36 nT is also short enough to *confuse* 36 nT with 0 nT. The product of the sensitivity and the unambiguous range is fixed, and escaping that requires either a prior estimate of the field or a multi-$\tau$ protocol that resolves the ambiguity coarsely before refining it — the phase-estimation strategy that Chapter 2 needs the moment it tries to image a real sample with an unknown background field.

### Code Example 3: The Optimal Interrogation Time, and What Dead Time Costs

The sensitivity formula has one free parameter, $\tau$, and one non-negotiable overhead, $t_d$. This example locates the optimum, checks it against the closed form, and then shows how much of the ideal is left once initialization and readout are paid for.

```python
"""Chapter 1, Example 3: the optimal interrogation time. Continues from
Examples 1 and 2 (same session)."""

T2_ref = 100.0 * US
tau_grid = np.logspace(-3, 0.7, 40001) * T2_ref

print(f"Optimal interrogation time, T2 = {T2_ref/US:.0f} us, no dead time")
hdr = (f"{'decay exponent p':>18}{'tau_opt numeric':>18}"
       f"{'T2/(2p)^(1/p)':>17}{'contrast there':>17}")
print(hdr)
print("-" * len(hdr))
for p in [1.0, 1.5, 2.0, 3.0]:
    eta = eta_ramsey(gamma_e, tau_grid, T2_ref, p=p)
    t_opt = tau_grid[np.argmin(eta)]
    ana = T2_ref / (2 * p) ** (1.0 / p)
    print(f"{p:>18.1f}{t_opt/US:>15.4f} us{ana/US:>14.4f} us"
          f"{np.exp(-(t_opt/T2_ref)**p):>17.4f}")
print("For p = 1 and p = 2 the optimum sits at exactly T2/2, which is why")
print("'interrogate for about half the coherence time' is the standing rule.")

# --- Dead time moves the optimum, and it is often the dominant cost ----------
print(f"\nDead time per shot, p = 1, T2 = {T2_ref/US:.0f} us")
hdr = (f"{'t_dead':>10}{'tau_opt':>13}{'duty cycle':>13}"
       f"{'eta_min':>18}{'penalty':>10}")
print(hdr)
print("-" * len(hdr))
eta0 = None
for label, td in [("0", 0.0), ("1 us", 1e-6), ("10 us", 1e-5),
                  ("100 us", 1e-4), ("1 ms", 1e-3)]:
    eta = eta_ramsey(gamma_e, tau_grid, T2_ref, p=1.0, t_dead=td)
    i = int(np.argmin(eta))
    t_opt, e_min = tau_grid[i], eta[i]
    if eta0 is None:
        eta0 = e_min
    print(f"{label:>10}{t_opt/US:>10.4g} us{t_opt/(t_opt+td):>13.4f}"
          f"{e_min/NT:>12.4g} nT/rtHz{e_min/eta0:>10.2f}")
print("A readout that takes ten coherence times costs a decade of")
print("sensitivity, and no improvement in T2 recovers it.")

# --- How the same eta is reached by one long-lived spin or by many ----------
print("\nOne spin against an ensemble: only the product N T2 enters")
hdr = (f"{'N spins':>12}{'T2':>10}{'N T2':>10}{'eta, no dead time':>22}"
       f"{'eta, t_dead = 1 us':>23}")
print(hdr)
print("-" * len(hdr))
for N, T2 in [(1, 1.0), (1e4, 1e-4), (1e6, 1e-6), (1e12, 1e-12)]:
    e0 = eta_ramsey(gamma_e, T2 / 2, T2, N=N)
    e1 = eta_ramsey(gamma_e, T2 / 2, T2, N=N, t_dead=1e-6)
    print(f"{N:>12.0e}{T2:>10.0e}{N*T2:>10.0e}{e0/NT:>14.4g} nT/rtHz"
          f"{e1/NT:>15.4g} nT/rtHz")
print("The middle column is the design variable, and the last column is the")
print("catch: a large N bought with a short T2 pays for it in duty cycle.")
```

```text
Optimal interrogation time, T2 = 100 us, no dead time
  decay exponent p   tau_opt numeric    T2/(2p)^(1/p)   contrast there
----------------------------------------------------------------------
               1.0        49.9994 us       50.0000 us           0.6065
               1.5        48.0778 us       48.0750 us           0.7165
               2.0        49.9994 us       50.0000 us           0.7788
               3.0        55.0288 us       55.0321 us           0.8465
For p = 1 and p = 2 the optimum sits at exactly T2/2, which is why
'interrogate for about half the coherence time' is the standing rule.

Dead time per shot, p = 1, T2 = 100 us
    t_dead      tau_opt   duty cycle           eta_min   penalty
----------------------------------------------------------------
         0        50 us       1.0000       1.324 nT/rtHz      1.00
      1 us     50.97 us       0.9808       1.337 nT/rtHz      1.01
     10 us     57.41 us       0.8517       1.442 nT/rtHz      1.09
    100 us     78.08 us       0.4385       2.119 nT/rtHz      1.60
      1 ms     95.64 us       0.0873       5.115 nT/rtHz      3.86
A readout that takes ten coherence times costs a decade of
sensitivity, and no improvement in T2 recovers it.

One spin against an ensemble: only the product N T2 enters
     N spins        T2      N T2     eta, no dead time     eta, t_dead = 1 us
-----------------------------------------------------------------------------
       1e+00     1e+00     1e+00       0.01324 nT/rtHz        0.01324 nT/rtHz
       1e+04     1e-04     1e+00       0.01324 nT/rtHz        0.01337 nT/rtHz
       1e+06     1e-06     1e+00       0.01324 nT/rtHz        0.02293 nT/rtHz
       1e+12     1e-12     1e+00       0.01324 nT/rtHz          18.73 nT/rtHz
The middle column is the design variable, and the last column is the
catch: a large N bought with a short T2 pays for it in duty cycle.
```

**What to look for.** The first table verifies $\tau_\mathrm{opt} = T_2/(2p)^{1/p}$ to four digits for four decay shapes, and it exposes the coincidence: exponential and Gaussian decay put the optimum at exactly $T_2/2$, even though the contrast left there differs (0.61 against 0.78). The rule of thumb is therefore robust in a way that its derivation does not obviously guarantee, and it is safe to use without knowing the decay shape.

The dead-time table is the honest correction, and it does two separate things. It moves the optimum to longer $\tau$ — with $t_d = T_2$ the best interrogation time is $0.96\,T_2$, not $0.5\,T_2$, because a shot is now expensive and must be made worth its overhead. And it degrades $\eta$ by $\sqrt{(\tau+t_d)/\tau}$, which for $t_d = 10\,T_2$ is a factor of 3.9. The prose claim in §1.3 that readout overhead is a first-class design parameter is this column: an experimenter with a 1 ms readout and a 100 $\mu$s $T_2$ is losing four times more sensitivity to the camera than to the crystal.

The last table is the one to memorize, together with its final column. Four configurations spanning twelve decades in $N$ and twelve in $T_2$ give *identical* sensitivity, because only $N T_2$ enters. But the same four with a 1 $\mu$s dead time do not: the $10^{12}$-spin, picosecond-coherence configuration is three decades worse, because it spends all its time being reset. $N T_2$ is the right figure of merit for the physics and the wrong one for the apparatus, and the gap between those two statements is where most of the engineering in this field happens.

### Code Example 4: The Allan Deviation and Its Three Regimes

Now the stability question. This example synthesises white, flicker and random-walk noise from a single generator, computes the Allan deviation of each, recovers the three characteristic slopes, and then combines them to produce the floor that a real instrument has.

```python
"""Chapter 1, Example 4: the Allan deviation and its three regimes.
Continues from Example 1 (same session)."""


def colored_noise(n, alpha, rng):
    """Gaussian noise of unit variance with one-sided PSD S(f) ~ 1/f**alpha.

    Built by shaping white noise in the frequency domain: alpha = 0 gives
    white noise, alpha = 1 flicker noise, alpha = 2 a random walk.
    """
    f = np.fft.rfftfreq(n, d=1.0)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** (-alpha / 2.0)
    spec = amp * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    x = np.fft.irfft(spec, n)
    return x / x.std()


def allan_deviation(y, m_list, tau0=1.0):
    """Non-overlapping Allan deviation of a series of readings.

    sigma_y(tau)**2 = <(ybar_{j+1} - ybar_j)**2> / 2, where ybar_j is the
    mean of m consecutive readings and tau = m tau0. The factor 1/2 makes
    the estimator equal to the standard deviation for white noise.
    """
    out = []
    for m in m_list:
        n_bins = len(y) // m
        ybar = y[:n_bins * m].reshape(n_bins, m).mean(axis=1)
        d = np.diff(ybar)
        out.append(np.sqrt(0.5 * np.mean(d ** 2)))
    return np.array(m_list) * tau0, np.array(out)


rng = np.random.default_rng(20260813)
n = 2 ** 20
m_list = [2 ** k for k in range(0, 12)]          # tau from 1 to 2048 samples

print(f"Allan deviation of {n} readings, tau0 = 1 s")
hdr = (f"{'tau (s)':>9}{'white':>13}{'flicker 1/f':>14}"
       f"{'random walk':>14}{'drift only':>13}")
print(hdr)
print("-" * len(hdr))
series = {
    "white": colored_noise(n, 0.0, rng),
    "flicker": colored_noise(n, 1.0, rng),
    "walk": colored_noise(n, 2.0, rng),
}
drift_rate = 3e-4                                 # per sample, deterministic
series["drift"] = drift_rate * np.arange(n)
taus, dev = {}, {}
for key, y in series.items():
    taus[key], dev[key] = allan_deviation(y, m_list)
for i, t in enumerate(taus["white"]):
    print(f"{t:>9.0f}{dev['white'][i]:>13.5f}{dev['flicker'][i]:>14.5f}"
          f"{dev['walk'][i]:>14.5f}{dev['drift'][i]:>13.5f}")

print("\nFitted log-log slopes (theory in the last column)")
hdr = f"{'process':<16}{'PSD':<14}{'fitted slope':>14}{'expected':>11}"
print(hdr)
print("-" * len(hdr))
expected = {"white": (-0.5, "S ~ f^0"), "flicker": (0.0, "S ~ 1/f"),
            "walk": (0.5, "S ~ 1/f^2"), "drift": (1.0, "deterministic")}
for key in ["white", "flicker", "walk", "drift"]:
    s = np.polyfit(np.log10(taus[key]), np.log10(dev[key]), 1)[0]
    exp_slope, psd = expected[key]
    print(f"{key:<16}{psd:<14}{s:>14.4f}{exp_slope:>11.1f}")

# --- The realistic case: all three at once, so the curve has a floor --------
amp = {"white": 0.1, "flicker": 0.09, "walk": 0.72}
total = sum(amp[k] * series[k] for k in amp)
t_tot, d_tot = allan_deviation(total, m_list)
print("\nA sensor with all three processes present")
hdr = (f"{'tau (s)':>9}{'sigma_y total':>15}{'white part':>13}"
       f"{'flicker part':>15}{'walk part':>12}")
print(hdr)
print("-" * len(hdr))
for i, t in enumerate(t_tot):
    print(f"{t:>9.0f}{d_tot[i]:>15.6f}{amp['white']*dev['white'][i]:>13.6f}"
          f"{amp['flicker']*dev['flicker'][i]:>15.6f}"
          f"{amp['walk']*dev['walk'][i]:>12.6f}")
i_min = int(np.argmin(d_tot))
promise = amp["white"] * dev["white"][0] / np.sqrt(t_tot[-1])
print(f"minimum at tau = {t_tot[i_min]:.0f} s, sigma_y = {d_tot[i_min]:.6f}")
print(f"white-noise extrapolation to tau = {t_tot[-1]:.0f} s would promise "
      f"{promise:.6f},")
print(f"the true value there is {d_tot[-1]:.6f}: a factor "
      f"{d_tot[-1]/promise:.1f} of wishful thinking.")
```

```text
Allan deviation of 1048576 readings, tau0 = 1 s
  tau (s)        white   flicker 1/f   random walk   drift only
---------------------------------------------------------------
        1      0.99986       0.35842       0.00240      0.00021
        2      0.70702       0.34291       0.00323      0.00042
        4      0.50006       0.33377       0.00448      0.00085
        8      0.35397       0.33000       0.00630      0.00170
       16      0.25058       0.32931       0.00891      0.00339
       32      0.17696       0.32932       0.01263      0.00679
       64      0.12427       0.33024       0.01787      0.01358
      128      0.08710       0.32950       0.02511      0.02715
      256      0.06147       0.33100       0.03522      0.05431
      512      0.04382       0.32736       0.04914      0.10861
     1024      0.03024       0.33240       0.06931      0.21722
     2048      0.02201       0.33655       0.09879      0.43445

Fitted log-log slopes (theory in the last column)
process         PSD             fitted slope   expected
-------------------------------------------------------
white           S ~ f^0              -0.5028       -0.5
flicker         S ~ 1/f              -0.0055        0.0
walk            S ~ 1/f^2             0.4909        0.5
drift           deterministic         1.0000        1.0

A sensor with all three processes present
  tau (s)  sigma_y total   white part   flicker part   walk part
----------------------------------------------------------------
        1       0.105086     0.099986       0.032258    0.001726
        2       0.077174     0.070702       0.030862    0.002327
        4       0.058494     0.050006       0.030039    0.003226
        8       0.046479     0.035397       0.029700    0.004539
       16       0.039331     0.025058       0.029638    0.006417
       32       0.035779     0.017696       0.029638    0.009092
       64       0.034676     0.012427       0.029721    0.012865
      128       0.036019     0.008710       0.029655    0.018080
      256       0.039645     0.006147       0.029790    0.025358
      512       0.046361     0.004382       0.029463    0.035378
     1024       0.058558     0.003024       0.029916    0.049903
     2048       0.077805     0.002201       0.030289    0.071131
minimum at tau = 64 s, sigma_y = 0.034676
white-noise extrapolation to tau = 2048 s would promise 0.002209,
the true value there is 0.077805: a factor 35.2 of wishful thinking.
```

**What to look for.** The first table is the three regimes, side by side, from one generator differing only in the exponent $\alpha$. Reading down the columns: white noise falls by $\sqrt{2}$ per doubling of $\tau$, flicker noise does not move at all across eleven doublings, and the random walk rises by $\sqrt{2}$ per doubling. The fitted slopes recover $-0.503$, $-0.006$ and $+0.491$ against the exact $-1/2$, $0$ and $+1/2$; the deterministic drift column returns $+1.0000$, which it must, since $\sigma_y = D\tau/\sqrt{2}$ has no statistical content at all.

The combined table is the one that matters operationally. This sensor has a genuine optimum averaging time of 64 s and a floor of $\sigma_y = 0.0347$, and the last two lines quantify the trap: extrapolating the white-noise branch to $\tau = 2048$ s would promise 0.0022, whereas the instrument actually delivers 0.0778 — a factor of 35 in the wrong direction. Anyone who quotes $\eta$ and averages beyond the turnover is not measuring more precisely; they are measuring the drift of their own reference.

The remedy is visible in the same table. The flicker plateau sits at 0.0297 and the white branch crosses it at about $\tau = 11$ s, so *any* modulation faster than roughly 0.1 Hz moves the measurement permanently into the white region, where averaging works as advertised. That is the entire argument for lock-in detection, and it is why the AC protocols of Chapters 2 and 3 are not refinements but prerequisites.

### Code Example 5: Filter Functions — Band Selection, AC Sensitivity, Noise Spectroscopy

The filter function of a pulse sequence is computed here exactly, from the piecewise-constant switching function, with no time grid and no quadrature error. It is then used three times: to locate the passband, to invert a coherence curve into a noise exponent, and to measure an AC field.

```python
"""Chapter 1, Example 5: filter functions, band selection and AC sensitivity.
Continues from Example 1 (same session)."""


def pulse_edges(tau, n_pi):
    """Sign-reversal times of a CPMG-n_pi sequence of total duration tau.

    The pi pulses sit at tau*(k - 1/2)/n_pi for k = 1..n_pi, which is the
    CPMG placement; n_pi = 0 is free induction and n_pi = 1 the Hahn echo.
    """
    inner = tau * (np.arange(1, n_pi + 1) - 0.5) / n_pi if n_pi else np.array([])
    return np.concatenate(([0.0], inner, [tau]))


def filter_function(f, tau, n_pi):
    """|s_tilde(f, tau)|**2 for a CPMG-n_pi sequence, evaluated exactly.

    s(t) = +-1 records the sign flips imposed by the pi pulses, and
    s_tilde is its finite-time Fourier transform. Because s is piecewise
    constant the integral is a closed-form sum over the intervals, so no
    time grid and no quadrature error enter.
    """
    edges = pulse_edges(tau, n_pi)
    signs = (-1.0) ** np.arange(len(edges) - 1)
    f = np.atleast_1d(np.asarray(f, dtype=float))
    out = np.empty(f.shape, dtype=complex)
    zero = f == 0.0
    out[zero] = np.sum(signs * np.diff(edges))
    fz = f[~zero]
    E = np.exp(-2j * np.pi * np.outer(fz, edges))
    out[~zero] = np.sum(signs * (E[:, 1:] - E[:, :-1]), axis=1) / (
        -2j * np.pi * fz)
    return np.abs(out) ** 2


tau_f, f_test = 1.0, 0.03
print("Filter function against the closed forms of the sister course")
print(f"{'sequence':<10}{'numeric':>14}{'analytic':>14}")
print("-" * 38)
fid_ana = np.sin(np.pi * f_test * tau_f) ** 2 / (np.pi * f_test) ** 2
echo_ana = 4 * np.sin(np.pi * f_test * tau_f / 2) ** 4 / (np.pi * f_test) ** 2
print(f"{'FID':<10}{filter_function(f_test, tau_f, 0)[0]:>14.8f}"
      f"{fid_ana:>14.8f}")
print(f"{'echo':<10}{filter_function(f_test, tau_f, 1)[0]:>14.8f}"
      f"{echo_ana:>14.8f}")

# --- Where each sequence listens -------------------------------------------
f_scan = np.linspace(1e-4, 20.0, 400000)
print("\nPassband of CPMG-N at fixed total time tau = 1 s")
hdr = (f"{'N pulses':>9}{'peak f':>11}{'N/(2 tau)':>12}"
       f"{'|s|^2 at peak':>15}{'|s|^2 at f = 0.01':>19}")
print(hdr)
print("-" * len(hdr))
for n_pi in [1, 2, 4, 8, 16, 32]:
    ff = filter_function(f_scan, tau_f, n_pi)
    i = int(np.argmax(ff))
    print(f"{n_pi:>9d}{f_scan[i]:>11.4f}{n_pi/(2*tau_f):>12.4f}"
          f"{ff[i]:>15.5f}{filter_function(0.01, tau_f, n_pi)[0]:>19.3e}")
print("Free induction, for comparison: |s|^2 at f = 0.01 is "
      f"{filter_function(0.01, tau_f, 0)[0]:.3e}")


# --- Band selection means noise spectroscopy: T2(N) for S = A/f^alpha -------
def coherence(tau, n_pi, A, alpha, decades=6, per_decade=400):
    """C(tau) = exp(-<phi^2>/2) with <phi^2> = int S(f) |s_tilde|^2 df.

    The integral runs on a logarithmic grid centred on 1/tau, which is
    where the filter function has all of its weight.
    """
    f = np.logspace(np.log10(1.0 / tau) - decades / 2,
                    np.log10(1.0 / tau) + decades / 2, decades * per_decade)
    S = A / f ** alpha
    chi = np.trapezoid(S * filter_function(f, tau, n_pi), f)
    return np.exp(-0.5 * chi)


def coherence_time(n_pi, A, alpha):
    """Smallest tau at which C(tau) falls to 1/e, found by bisection."""
    lo, hi = 1e-9, 1e3
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if coherence(mid, n_pi, A, alpha) > np.exp(-1.0):
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


A_noise = 1.0
print("\nCPMG noise spectroscopy: T2(N) for S(f) = A/f^alpha, A = 1")
hdr = (f"{'alpha':>7}{'T2(1)':>11}{'T2(64)':>11}{'exponent, N=1-64':>19}"
       f"{'exponent, N=4-64':>19}{'alpha/(alpha+1)':>18}")
print(hdr)
print("-" * len(hdr))
Ns = np.array([1, 2, 4, 8, 16, 32, 64])
for alpha in [0.5, 1.0, 1.5, 2.0]:
    T2s = np.array([coherence_time(int(nn), A_noise, alpha) for nn in Ns])
    beta_all = np.polyfit(np.log(Ns), np.log(T2s), 1)[0]
    beta_asy = np.polyfit(np.log(Ns[2:]), np.log(T2s[2:]), 1)[0]
    print(f"{alpha:>7.1f}{T2s[0]:>11.5f}{T2s[-1]:>11.5f}{beta_all:>19.4f}"
          f"{beta_asy:>19.4f}{alpha/(alpha+1):>18.4f}")
print("Reading the table backwards is the measurement: the exponent of the")
print("coherence-vs-N curve returns alpha, i.e. the defect ensemble itself.")

# --- The same filter, used to detect a field instead of to reject noise -----
print("\nAC field detection: phase picked up from B(t) = B_ac cos(2 pi f t + q)")
hdr = f"{'N pulses':>9}{'matched f':>12}{'best phase':>13}{'phi/(gamma B tau)':>19}{'2/pi':>8}"
print(hdr)
print("-" * len(hdr))
for n_pi in [1, 2, 4, 8, 16]:
    edges = pulse_edges(tau_f, n_pi)
    signs = (-1.0) ** np.arange(len(edges) - 1)
    f_ac = n_pi / (2 * tau_f)
    best = 0.0
    for q in np.linspace(0.0, 2 * np.pi, 721):
        # exact integral of s(t) cos(2 pi f t + q) over each interval
        w = (np.sin(2 * np.pi * f_ac * edges[1:] + q)
             - np.sin(2 * np.pi * f_ac * edges[:-1] + q)) / (2 * np.pi * f_ac)
        best = max(best, abs(np.sum(signs * w)))
    print(f"{n_pi:>9d}{f_ac:>12.2f}{'optimal':>13}{best/tau_f:>19.6f}"
          f"{2/np.pi:>8.4f}")
print("A synchronized square-wave filter rectifies the AC field with")
print("efficiency 2/pi, so AC sensitivity is (pi/2) times the DC formula.")
```

```text
Filter function against the closed forms of the sister course
sequence         numeric      analytic
--------------------------------------
FID           0.99704262    0.99704262
echo          0.00221738    0.00221738

Passband of CPMG-N at fixed total time tau = 1 s
 N pulses     peak f   N/(2 tau)  |s|^2 at peak  |s|^2 at f = 0.01
------------------------------------------------------------------
        1     0.7420      0.5000        0.52506          2.467e-04
        2     1.1478      1.0000        0.44083          1.522e-08
        4     2.0825      2.0000        0.41496          9.510e-10
        8     4.0428      4.0000        0.40777          5.943e-11
       16     8.0216      8.0000        0.40591          3.715e-12
       32    16.0108     16.0000        0.40544          2.322e-13
Free induction, for comparison: |s|^2 at f = 0.01 is 9.997e-01

CPMG noise spectroscopy: T2(N) for S(f) = A/f^alpha, A = 1
  alpha      T2(1)     T2(64)   exponent, N=1-64   exponent, N=4-64   alpha/(alpha+1)
-------------------------------------------------------------------------------------
    0.5    2.35818    8.63255             0.3150             0.3271            0.3333
    1.0    1.69864   12.26899             0.4794             0.4932            0.5000
    1.5    1.32943   15.00037             0.5859             0.5956            0.6000
    2.0    1.06785   17.07673             0.6666             0.6667            0.6667
Reading the table backwards is the measurement: the exponent of the
coherence-vs-N curve returns alpha, i.e. the defect ensemble itself.

AC field detection: phase picked up from B(t) = B_ac cos(2 pi f t + q)
 N pulses   matched f   best phase  phi/(gamma B tau)    2/pi
-------------------------------------------------------------
        1        0.50      optimal           0.636620  0.6366
        2        1.00      optimal           0.636620  0.6366
        4        2.00      optimal           0.636620  0.6366
        8        4.00      optimal           0.636620  0.6366
       16        8.00      optimal           0.636620  0.6366
A synchronized square-wave filter rectifies the AC field with
efficiency 2/pi, so AC sensitivity is (pi/2) times the DC formula.
```

**What to look for.** The verification block reproduces the sister course's closed forms to eight digits, which licenses everything after it: $|\tilde{s}_\mathrm{FID}|^2 = 0.99704262$ and $|\tilde{s}_\mathrm{echo}|^2 = 0.00221738$ at $f\tau = 0.03$, identical to [quantum-hardware Chapter 1](<../../FM/quantum-hardware-introduction/chapter-1.html>) Exercise 4. The two courses share this machinery, and this is the check that they share it correctly.

The passband table is band selection made quantitative. The rule of thumb $f_\mathrm{peak} \approx N/2\tau$ is poor for the Hahn echo (0.742 against 0.5, a 48% error) and excellent from $N = 4$ upward (2.083 against 2.000, then 4.043, 8.022, 16.011). The last column is why decoupling works: at $f = 0.01/\tau$ the free-induction filter has weight 1.0 while CPMG-32 has $2.3\times10^{-13}$, twelve decades of rejection of the slow noise, achieved with nothing but pulse timing. Note also that the peak height saturates at $0.405$ rather than growing — a decoupling sequence does not become more sensitive at its passband as $N$ increases, it only becomes narrower and better placed.

The spectroscopy table closes the loop back to §1.4. For each noise exponent $\alpha$, the coherence time grows with the number of pulses as $T_2(N) \propto N^{\beta}$, and the fitted $\beta$ approaches the predicted $\alpha/(\alpha+1)$ from below: $0.327$ against $0.333$, $0.493$ against $0.500$, $0.596$ against $0.600$, and $0.6667$ against $0.6667$ exactly for $\alpha = 2$. The residual deficit at small $\alpha$ is a finite-$N$ effect, not an error — the scaling law is asymptotic, and fitting over $N = 4$ to $64$ rather than $1$ to $64$ recovers most of the difference. Turned around, this table is a measurement recipe: run CPMG at several $N$, fit the exponent, and you have the spectral exponent of the fluctuator ensemble in your host crystal without ever having identified a single defect.

The AC block is the same filter used as a lock-in. For every $N$, a field at the matched frequency $N/2\tau$ and optimal relative phase produces a total phase of exactly $(2/\pi)\gamma B_\mathrm{ac}\tau$ — the rectification efficiency of a square wave against a sinusoid, $0.636620$ against $2/\pi = 0.6366$, independent of $N$. The $\pi/2$ penalty is the price of admission; what is bought is the right to use $\tau \sim T_2$ instead of $\tau \sim T_2^\ast$, and in a solid-state sensor that is worth one to two decades of $\eta$. This is the single most important practical fact in Chapter 2.

### Code Example 6: The Sensitivity-Resolution Map

The last example turns the scaling arguments of §1.5 into numbers, and finds the two places where the scaling stops.

```python
"""Chapter 1, Example 6: the sensitivity-resolution trade-off map.
Continues from Examples 1 and 3 (same session)."""

n_spin = 1e24            # spin density, m^-3: an illustrative round number
T2_dilute = 1e-3         # coherence time of an isolated spin, s

print("Sensing volume against sensitivity, at fixed spin density")
hdr = (f"{'sensor side':<14}{'volume':>12}{'N spins':>12}"
       f"{'eta_B':>18}{'min moment':>16}")
print(hdr)
print("-" * len(hdr))
for label, d in [("10 nm", 1e-8), ("100 nm", 1e-7), ("1 um", 1e-6),
                 ("10 um", 1e-5), ("100 um", 1e-4), ("1 mm", 1e-3)]:
    V = d ** 3
    N = n_spin * V
    e = eta_ramsey(gamma_e, T2_dilute / 2, T2_dilute, N=N)
    # A point dipole at standoff d produces B ~ mu0 m / (4 pi d^3).
    m_min = e * 4 * np.pi * d ** 3 / mu0
    print(f"{label:<14}{V:>12.2e}{N:>12.3e}{e/NT:>12.4g} nT/rtHz"
          f"{m_min/9.2740100783e-24:>11.3g} muB")
print("eta_B improves as d^(-3/2) while the resolution degrades as d, so")
print("the detectable *moment* improves as d^(3/2) on shrinking the sensor.")

print("\nWhy raising the spin density stops helping: dipolar T2 ~ 1/n")
hdr = (f"{'density (m^-3)':>16}{'T2':>13}{'N in (100 nm)^3':>17}"
       f"{'N T2':>12}{'eta_B':>18}")
print(hdr)
print("-" * len(hdr))
V = (1e-7) ** 3
n_ref, T2_at_ref = 1e22, 1e-3
for n in [1e22, 1e23, 1e24, 1e25]:
    T2 = T2_at_ref * n_ref / n                   # dipolar-limited scaling
    N = n * V
    e = eta_ramsey(gamma_e, T2 / 2, T2, N=N)
    print(f"{n:>16.0e}{T2*1e3:>10.4g} ms{N:>17.4g}{N*T2:>12.3e}"
          f"{e/NT:>12.4g} nT/rtHz")
print("N T2 is constant along this column, so eta_B is too. Sensitivity")
print("stops improving once the spins dephase each other, and the honest")
print("figure of merit is the product, never the count.")

print("\nStandoff: what a distant dipole demands of the sensor")
hdr = (f"{'moment':>10}{'standoff':>10}{'B at sensor':>17}"
       f"{'N T2 needed':>16}{'N needed at T2 = 1 ms':>23}")
print(hdr)
print("-" * len(hdr))
for label_m, m in [("1 muB", 9.2740100783e-24),
                   ("1000 muB", 9.2740100783e-21)]:
    for label_z, z in [("10 nm", 1e-8), ("100 nm", 1e-7), ("1 um", 1e-6)]:
        B = mu0 * m / (4 * np.pi * z ** 3)
        # eta = sqrt(2e)/(gamma sqrt(N T2)) must reach B in one second
        NT2 = 2 * np.e / (gamma_e * B) ** 2
        print(f"{label_m:>10}{label_z:>10}{B/NT:>14.4g} nT{NT2:>13.3e} s"
              f"{NT2/1e-3:>23.3e}")
print("The z^-3 falloff is why spatial resolution and sensitivity are one")
print("conversation: one decade of standoff costs three decades of field and")
print("therefore six decades of N T2.")
```

```text
Sensing volume against sensitivity, at fixed spin density
sensor side         volume     N spins             eta_B      min moment
------------------------------------------------------------------------
10 nm             1.00e-24   1.000e+00      0.4187 nT/rtHz   0.000452 muB
100 nm            1.00e-21   1.000e+03     0.01324 nT/rtHz     0.0143 muB
1 um              1.00e-18   1.000e+06   0.0004187 nT/rtHz      0.452 muB
10 um             1.00e-15   1.000e+09   1.324e-05 nT/rtHz       14.3 muB
100 um            1.00e-12   1.000e+12   4.187e-07 nT/rtHz        452 muB
1 mm              1.00e-09   1.000e+15   1.324e-08 nT/rtHz   1.43e+04 muB
eta_B improves as d^(-3/2) while the resolution degrades as d, so
the detectable *moment* improves as d^(3/2) on shrinking the sensor.

Why raising the spin density stops helping: dipolar T2 ~ 1/n
  density (m^-3)           T2  N in (100 nm)^3        N T2             eta_B
----------------------------------------------------------------------------
           1e+22         1 ms               10   1.000e-02      0.1324 nT/rtHz
           1e+23       0.1 ms              100   1.000e-02      0.1324 nT/rtHz
           1e+24      0.01 ms             1000   1.000e-02      0.1324 nT/rtHz
           1e+25     0.001 ms            1e+04   1.000e-02      0.1324 nT/rtHz
N T2 is constant along this column, so eta_B is too. Sensitivity
stops improving once the spins dephase each other, and the honest
figure of merit is the product, never the count.

Standoff: what a distant dipole demands of the sensor
    moment  standoff      B at sensor     N T2 needed  N needed at T2 = 1 ms
----------------------------------------------------------------------------
     1 muB     10 nm         927.4 nT    2.039e-10 s              2.039e-07
     1 muB    100 nm        0.9274 nT    2.039e-04 s              2.039e-01
     1 muB      1 um     0.0009274 nT    2.039e+02 s              2.039e+05
  1000 muB     10 nm     9.274e+05 nT    2.039e-16 s              2.039e-13
  1000 muB    100 nm         927.4 nT    2.039e-10 s              2.039e-07
  1000 muB      1 um        0.9274 nT    2.039e-04 s              2.039e-01
The z^-3 falloff is why spatial resolution and sensitivity are one
conversation: one decade of standoff costs three decades of field and
therefore six decades of N T2.
```

**What to look for.** The first table is the map, in two columns that point in opposite directions. Across five decades of sensor size, $\eta_B$ improves by seven and a half decades — the $d^{-3/2}$ law — while the minimum detectable moment degrades by the same seven and a half. A millimetre sensor reaches $1.3\times10^{-8}$ nT/$\sqrt{\mathrm{Hz}}$ and cannot see anything smaller than $10^4$ Bohr magnetons; a 10 nm sensor is 30 million times worse at measuring a field and can see 0.0005 $\mu_B$. Neither is better. They answer different questions, and §1.5's map is the statement of which.

The middle table is the limit that surprises people. Raising the spin density by three decades raises $N$ by three decades, and every one of them is cancelled by the dipolar shortening of $T_2$: the product $N T_2$ is constant at $10^{-2}$ s and $\eta_B$ does not move by a single digit. This is not a numerical coincidence in the model, it is the model's whole content, and it is the reason ensemble magnetometry has spent so much effort on decoupling ensembles from themselves rather than on making them denser. It is also a warning about reading any specification that quotes a spin count without a coherence time.

The last table converts geometry into a requirement. A single Bohr magneton at 100 nm standoff produces 0.93 nT, which needs $N T_2 = 2\times10^{-4}$ s — comfortable. Move to 1 $\mu$m standoff and the same moment needs $N T_2 = 204$ s, six decades more, because the field fell by three decades and $\eta$ enters squared. That factor of $10^6$ for one decade of standoff is the most important number in scanning magnetometry, and it explains why the sensor-to-sample distance is the first thing reported and the last thing improved.

* * *

## Exercises

#### Exercise 1: Reading a Sensitivity Specification

A magnetometer is specified at $\eta = 1.324$ nT/$\sqrt{\mathrm{Hz}}$, the single-electron-spin value from Code Example 1 with $T_2 = 100\ \mu$s.

  1. What field resolution does it reach after 1 s, 100 s and 1 hour of averaging?
  2. How long must it average to resolve 1 pT, assuming white noise all the way?
  3. A colleague proposes a tenfold improvement in $\eta$ instead. How does the required averaging time change, and what would have to improve physically to get that factor of ten?
  4. Under what circumstance is the answer to part 2 a fiction, and what single additional measurement would reveal it?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\delta B = \eta/\sqrt{T}\), so 1324 pT after 1 s, 132 pT after 100 s, and 22.1 pT after 1 hour.</p>

<p><strong>2.</strong> \(T = (\eta/\delta B)^2 = (1.324\times10^{-9}/10^{-12})^2 = 1.75\times10^{6}\) s, i.e. 487 hours or about 20 days.</p>

<p><strong>3.</strong> \(T\) scales as \(\eta^2\), so a tenfold better \(\eta\) cuts the averaging time by a hundred, to \(1.75\times10^{4}\) s or under 5 hours. Physically, \(\eta \propto 1/\sqrt{N T_2}\), so a factor of ten needs a hundredfold increase in \(N T_2\) — a hundred times more spins at the same coherence, or a hundredfold longer \(T_2\), or any combination. Improving the contrast \(C\) or reducing the dead time is usually far cheaper than either, and both enter \(\eta\) linearly.</p>

<p><strong>4.</strong> It is a fiction as soon as the Allan deviation departs from its \(\tau^{-1/2}\) branch, which for any real instrument happens long before 20 days. The additional measurement is the stability curve itself: record the sensor output with no signal for many hours and compute \(\sigma_y(\tau)\). Its minimum is the best resolution the instrument can reach by averaging at all, and if that minimum lies above 1 pT then no averaging schedule will succeed and the measurement must be modulated instead.</p>

```python
import numpy as np
eta = 1.324e-9                      # T/sqrt(Hz), from Code Example 1
for T in (1.0, 100.0, 3600.0, 86400.0):
    print(f"T = {T:8.0f} s   dB = {eta/np.sqrt(T)*1e12:9.3f} pT")
target = 1e-12
print(f"to reach 1 pT: T = {(eta/target)**2:.0f} s "
      f"= {(eta/target)**2/3600:.2f} h")
print(f"tenfold better eta instead: T = {(eta/10/target)**2:.2f} s")
# T =        1 s   dB =  1324.000 pT
# T =      100 s   dB =   132.400 pT
# T =     3600 s   dB =    22.067 pT
# T =    86400 s   dB =     4.504 pT
# to reach 1 pT: T = 1752976 s = 486.94 h
# tenfold better eta instead: T = 17529.76 s
```

</details>

#### Exercise 2: The Optimum With Dead Time

For exponential contrast decay, $\eta(\tau) \propto \sqrt{\tau + t_d}\,e^{\tau/T_2}/\tau$.

  1. Differentiate $\ln\eta$ and show that the optimum satisfies $\dfrac{1}{2(\tau + t_d)} - \dfrac{1}{\tau} + \dfrac{1}{T_2} = 0$.
  2. Verify that $t_d = 0$ recovers $\tau_\mathrm{opt} = T_2/2$.
  3. Solve the equation numerically for $t_d = 10\ \mu$s, $100\ \mu$s and 1 ms at $T_2 = 100\ \mu$s, and compare with a brute-force grid search.
  4. What is the limiting value of $\tau_\mathrm{opt}$ as $t_d \to \infty$, and what does that limit mean for an experiment whose readout is much slower than its coherence?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\ln\eta = \tfrac{1}{2}\ln(\tau + t_d) - \ln\tau + \tau/T_2 + \mathrm{const}\). Differentiating gives \(\tfrac{1}{2(\tau+t_d)} - 1/\tau + 1/T_2 = 0\) directly.</p>

<p><strong>2.</strong> With \(t_d = 0\) the first two terms combine to \(-1/(2\tau)\), so \(1/(2\tau) = 1/T_2\) and \(\tau_\mathrm{opt} = T_2/2\).</p>

<p><strong>3.</strong> Root-finding and grid search agree to six digits: \(57.417\ \mu\mathrm{s}\), \(78.078\ \mu\mathrm{s}\) and \(95.636\ \mu\mathrm{s}\), i.e. \(\tau_\mathrm{opt}/T_2 = 0.574,\ 0.781,\ 0.956\). The optimum moves towards \(T_2\) as the overhead grows.</p>

<p><strong>4.</strong> As \(t_d \to \infty\) the term \(1/(2(\tau+t_d))\) vanishes and the condition becomes \(1/\tau = 1/T_2\), so \(\tau_\mathrm{opt} \to T_2\). The interpretation is that when a shot is dominated by overhead there is no reason to end the interrogation early: the marginal cost of another microsecond of precession is nothing compared with the cost of another readout, so one interrogates until the contrast is genuinely gone. The practical corollary is that slow-readout experiments should use longer sequences than the \(T_2/2\) rule suggests, and that reducing \(t_d\) buys sensitivity twice over — once through the duty cycle and once by moving the optimum back to where the contrast is high.</p>

```python
import numpy as np
from scipy.optimize import brentq
T2, gamma = 100e-6, 1.76085963e11


def eta(tau, td):
    return np.sqrt(tau + td) / (gamma * tau * np.exp(-tau / T2))


for td in (0.0, 10e-6, 100e-6, 1e-3):
    root = brentq(lambda t: 1/(2*(t+td)) - 1/t + 1/T2, 1e-9, 50*T2)
    grid = np.logspace(-9, np.log10(5*T2), 2000001)
    num = grid[np.argmin(eta(grid, td))]
    print(f"t_dead = {td*1e6:6.1f} us   root = {root*1e6:7.3f} us   "
          f"grid = {num*1e6:7.3f} us   tau/T2 = {root/T2:5.3f}")
# t_dead =    0.0 us   root =  50.000 us   grid =  50.000 us   tau/T2 = 0.500
# t_dead =   10.0 us   root =  57.417 us   grid =  57.416 us   tau/T2 = 0.574
# t_dead =  100.0 us   root =  78.078 us   grid =  78.077 us   tau/T2 = 0.781
# t_dead = 1000.0 us   root =  95.636 us   grid =  95.636 us   tau/T2 = 0.956
```

</details>

#### Exercise 3: Diagnosing an Instrument From Its Allan Deviation

A magnetometer is left running with no signal applied, and its Allan deviation is tabulated in tesla:

| $\tau$ (s) | 1 | 10 | 100 | 1000 | 10000 |
| --- | --- | --- | --- | --- | --- |
| $\sigma_y(\tau)$ | $2.0\times10^{-13}$ | $6.3\times10^{-14}$ | $2.0\times10^{-14}$ | $2.2\times10^{-14}$ | $6.5\times10^{-14}$ |

  1. Compute the log-log slope in each interval and name the dominant process in each.
  2. Extract the white-noise amplitude $w$ (defined by $\sigma_y = w/\sqrt{\tau}$) and the random-walk amplitude $a$ (defined by $\sigma_y = a\sqrt{\tau}$). What is $\eta$ for this instrument?
  3. Model the total as $\sigma_y^2 = w^2/\tau + a^2\tau$, and predict the optimal averaging time and the floor. Compare with the table.
  4. The measurement needs $5\times10^{-15}$ T. Give two strategies, and say which physical quantity each one attacks.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> The slopes are \(-0.502\), \(-0.498\), \(+0.041\) and \(+0.470\). The first two decades are white noise (projection or photon shot noise); the third is the flicker plateau, or equivalently the turnover region; the last is random-walk drift of the reference.</p>

<p><strong>2.</strong> \(w = \sigma_y(1\,\mathrm{s})\sqrt{1\,\mathrm{s}} = 2.0\times10^{-13}\ \mathrm{T}\sqrt{\mathrm{s}}\), which <em>is</em> \(\eta = 200\) fT/\(\sqrt{\mathrm{Hz}}\) — the white-noise amplitude and the sensitivity are the same number, and this is the only regime in which quoting \(\eta\) is meaningful. \(a = \sigma_y(10^4\,\mathrm{s})/\sqrt{10^4\,\mathrm{s}} = 6.5\times10^{-16}\ \mathrm{T}/\sqrt{\mathrm{s}}\).</p>

<p><strong>3.</strong> Minimizing \(w^2/\tau + a^2\tau\) gives \(\tau^\ast = w/a = 308\) s and \(\sigma_\mathrm{min} = \sqrt{2wa} = 1.61\times10^{-14}\) T. The tabulated minimum is \(2.0\times10^{-14}\) T at \(\tau = 100\) s, which is consistent given that the two-term model omits the flicker contribution visible in the third interval — the real floor is a little higher and a little earlier than the pure white-plus-walk estimate, as it always is.</p>

<p><strong>4.</strong> The target is below the floor, so averaging cannot reach it and the answer is not "average longer". <em>Strategy one: modulate.</em> Chop the signal at a frequency above the flicker corner — a few tens of millihertz here — so that the measurement lives permanently on the \(\tau^{-1/2}\) branch. This attacks the <em>reference drift</em>, and it does not require improving anything about the sensor. <em>Strategy two: improve \(\eta\).</em> Reducing \(w\) by a factor of 3 moves the crossing to shorter \(\tau\) and the floor to \(\sqrt{2wa}\), i.e. down by \(\sqrt{3} = 1.7\) only — the floor improves as the square root of \(\eta\), so this attacks the <em>projection noise</em> and gets much less for it. The honest conclusion is that when a measurement sits below the Allan floor, modulation is the strategy and sensitivity is a distraction.</p>

```python
import numpy as np
tau_tab = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
sig = np.array([2.0e-13, 6.3e-14, 2.0e-14, 2.2e-14, 6.5e-14])
for i in range(len(tau_tab) - 1):
    s = np.log10(sig[i+1]/sig[i]) / np.log10(tau_tab[i+1]/tau_tab[i])
    print(f"tau {tau_tab[i]:7.0f} -> {tau_tab[i+1]:7.0f} s   slope = {s:+.3f}")
w = sig[0] * np.sqrt(tau_tab[0])          # white amplitude, sigma = w/sqrt(tau)
a_w = sig[-1] / np.sqrt(tau_tab[-1])      # random-walk amplitude
print(f"white amplitude w   = {w:.3e}   walk amplitude a = {a_w:.3e}")
print(f"crossing at tau*    = {w/a_w:.0f} s")
print(f"predicted floor     = {np.sqrt(2*w*a_w):.3e}")
print(f"observed minimum    = {sig.min():.3e} at "
      f"tau = {tau_tab[int(np.argmin(sig))]:.0f} s")
# tau       1 ->      10 s   slope = -0.502
# tau      10 ->     100 s   slope = -0.498
# tau     100 ->    1000 s   slope = +0.041
# tau    1000 ->   10000 s   slope = +0.470
# white amplitude w   = 2.000e-13   walk amplitude a = 6.500e-16
# crossing at tau*    = 308 s
# predicted floor     = 1.612e-14
# observed minimum    = 2.000e-14 at tau = 100 s
```

</details>

#### Exercise 4: Designing a Filter for a Target Frequency

A sample is expected to produce magnetic noise at 100 kHz, and the sensor has $T_2 = 1$ ms.

  1. For CPMG-$N$ with $N = 1, 8, 32, 128$, choose the total sequence duration $\tau$ that places the passband at 100 kHz, and evaluate $|\tilde{s}|^2$ there.
  2. Which choice gives the largest response, and what is the constraint that stops you from simply increasing $N$?
  3. Show, numerically, that for white noise $\langle\phi^2\rangle = S\tau/2$ regardless of the number of $\pi$ pulses. Explain the result analytically.
  4. From part 3, what is the maximum factor by which dynamical decoupling can extend $T_2$ when the environment is white in the accessible band?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Matching \(f_\mathrm{ac} = N/2\tau\) gives \(\tau = N/(2f_\mathrm{ac})\): 5 \(\mu\)s, 40 \(\mu\)s, 160 \(\mu\)s and 640 \(\mu\)s. The corresponding \(|\tilde{s}|^2\) values are \(1.01\times10^{-11}\), \(6.48\times10^{-10}\), \(1.04\times10^{-8}\) and \(1.66\times10^{-7}\) s\(^2\), growing roughly as \(\tau^2 \propto N^2\).</p>

<p><strong>2.</strong> CPMG-128 gives the largest response, and the constraint is \(T_2\): the sequence lasts \(\tau = 640\ \mu\mathrm{s}\), already comparable with the 1 ms coherence time, so the contrast is falling and going further gains nothing. In practice the limit is whichever comes first — \(T_2\) itself, or pulse-error accumulation over 128 pulses, which no filter function describes.</p>

<p><strong>3.</strong> The numerics give \(\langle\phi^2\rangle/S = 4.9995\times10^{-4}\), \(4.9985\times10^{-4}\) and \(4.9833\times10^{-4}\) for \(n_\pi = 0, 1, 16\) at \(\tau = 1\) ms, all equal to \(\tau/2 = 5\times10^{-4}\) within the integration error. Analytically, Parseval's theorem gives \(\int_0^\infty |\tilde{s}(f)|^2 df = \tfrac{1}{2}\int_{-\infty}^{\infty} |\tilde{s}|^2 df = \tfrac{1}{2}\int_0^\tau s(t)^2 dt = \tau/2\), and \(s(t)^2 = 1\) whatever the pulses do. The pulses redistribute the filter weight in frequency but cannot change its integral.</p>

<p><strong>4.</strong> None. If the noise is white over the band the sequence can reach, the accumulated phase variance depends only on \(\tau\), so \(T_2\) is fixed and no number of pulses changes it. This is the same statement as the hardware course's \(T_2(N) \propto N^{\alpha/(\alpha+1)}\) with \(\alpha = 0\), and it is why dynamical decoupling cannot push \(T_2\) past \(2T_1\): the relaxation channel is effectively white at every frequency the pulses can select.</p>

```python
import numpy as np


def s_tilde2(f, tau, n_pi):
    inner = tau*(np.arange(1, n_pi+1)-0.5)/n_pi if n_pi else np.array([])
    edges = np.concatenate(([0.0], inner, [tau]))
    signs = (-1.0)**np.arange(len(edges)-1)
    f = np.atleast_1d(np.asarray(f, float))
    out = np.empty(f.shape, complex)
    z = f == 0.0
    out[z] = np.sum(signs*np.diff(edges))
    fz = f[~z]
    E = np.exp(-2j*np.pi*np.outer(fz, edges))
    out[~z] = np.sum(signs*(E[:, 1:]-E[:, :-1]), axis=1)/(-2j*np.pi*fz)
    return np.abs(out)**2


f_target = 1e5                      # 100 kHz AC field to be detected
for n_pi in (1, 8, 32, 128):
    tau_match = n_pi/(2*f_target)
    print(f"CPMG-{n_pi:<4d} tau = {tau_match*1e6:8.2f} us   "
          f"|s|^2 at 100 kHz = {s_tilde2(f_target, tau_match, n_pi)[0]:.4e}")
f = np.logspace(-3, 6, 900001)
for n_pi in (0, 1, 16):
    chi = np.trapezoid(np.ones_like(f)*s_tilde2(f, 1e-3, n_pi), f)
    print(f"white noise S = 1: n_pi = {n_pi:2d}   <phi^2>/A = {chi:.6e}")
# CPMG-1    tau =     5.00 us   |s|^2 at 100 kHz = 1.0132e-11
# CPMG-8    tau =    40.00 us   |s|^2 at 100 kHz = 6.4846e-10
# CPMG-32   tau =   160.00 us   |s|^2 at 100 kHz = 1.0375e-08
# CPMG-128  tau =   640.00 us   |s|^2 at 100 kHz = 1.6600e-07
# white noise S = 1: n_pi =  0   <phi^2>/A = 4.999483e-04
# white noise S = 1: n_pi =  1   <phi^2>/A = 4.998480e-04
# white noise S = 1: n_pi = 16   <phi^2>/A = 4.983257e-04
```

</details>

#### Exercise 5: Choosing an Instrument for a Materials Problem

Four measurement problems, each stated as a materials question rather than as a sensing specification:

  * **A.** Map the stray field above a magnetic domain wall in a 5 nm thin film, well enough to locate the wall to 50 nm.
  * **B.** Measure the magnetic susceptibility of a 1 mg powder sample as a function of temperature.
  * **C.** Detect gigahertz spin fluctuations in a two-dimensional magnet, as a function of position across a 10 $\mu$m flake.
  * **D.** Measure the temperature inside an operating power transistor with 200 nm resolution.

  1. For each, name the configuration from the §1.5 map that fits, and the chapter of this course that covers it.
  2. For problem A, estimate the required $\eta_B$ if the wall's stray field at 50 nm standoff is of order 1 $\mu$T and a 10:1 signal-to-noise ratio is wanted within 1 ms per pixel.
  3. For problem C, explain why $\eta_B$ is the wrong figure of merit and say what the right one is.
  4. Which of the four is *not* well served by any quantum sensor, and why?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> <em>A:</em> a single defect on a scanning tip, or a shallow defect layer imaged wide-field — Chapter 2. A scanning SQUID (Chapter 3) reaches better field sensitivity but not 50 nm resolution. <em>B:</em> a bulk pickup loop and a SQUID susceptometer — Chapter 3, and the natural companion of the bulk magnetometry in <a href="../electrical-magnetic-testing-introduction/index.html">Introduction to Electrical and Magnetic Testing</a>. <em>C:</em> shallow NV centres used as relaxometers, not as magnetometers — Chapter 2. <em>D:</em> shallow NV thermometry, reading the temperature dependence of the zero-field splitting — Chapter 2.</p>

<p><strong>2.</strong> A 10:1 ratio on 1 \(\mu\)T means \(\delta B = 100\) nT in \(T = 10^{-3}\) s, so \(\eta_B = \delta B\sqrt{T} = 100\ \mathrm{nT}\times0.0316\ \sqrt{\mathrm{s}} = 3.16\) nT/\(\sqrt{\mathrm{Hz}}\). Code Example 1 shows a single electron spin with \(T_2 = 10\ \mu\)s already reaching 4.2 nT/\(\sqrt{\mathrm{Hz}}\), so this measurement is comfortable on sensitivity and the real difficulty is elsewhere: holding the standoff at 50 nm, and the millisecond-per-pixel budget over a large image.</p>

<p><strong>3.</strong> Problem C asks about noise at gigahertz frequencies, and no Ramsey or CPMG filter function reaches there — the passband \(N/2\tau\) would need \(\tau\) of order nanoseconds with an absurd pulse count. Gigahertz noise is detected instead through \(T_1\), because \(1/T_1\) is a golden-rule rate proportional to \(S(\omega_0)\) at the sensor's own transition frequency. The right figure of merit is therefore the smallest fractional change in \(1/T_1\) that can be resolved per pixel per unit time, and the tuning knob is the sensor's transition frequency — swept with an applied field to move the sampling point across the spectrum. This is relaxometry, and §1.4's remark that the same machinery reads both ways is exactly this: a coherence measurement is a spectrometer.</p>

<p><strong>4.</strong> Problem B. It needs a bulk, calibrated, temperature-swept susceptibility on a milligram of powder, and while a SQUID magnetometer is the standard instrument for it, the quantum-ness is incidental: the SQUID is being used as an extremely good flux transducer, not as a quantum interferometer whose phase encodes a local field. Nothing in this chapter's sensitivity-resolution trade-off is being exploited, because the measurement wants an ensemble average over the whole sample — which is the one thing quantum sensors are <em>not</em> distinctively good at. Recognizing this case matters: the honest use of the map in §1.5 includes noticing when a conventional instrument already wins.</p>

```python
import numpy as np
mu0, muB = 1.25663706212e-6, 9.2740100783e-24
for label, z, eta_s in [("scanning single spin", 20e-9, 1e-9),
                        ("wide-field ensemble", 500e-9, 1e-11)]:
    B = eta_s                                    # SNR 1 after 1 s
    m = B * 4*np.pi*z**3/mu0
    print(f"{label:<22} standoff {z*1e9:5.0f} nm   "
          f"m_min = {m/muB:9.3f} muB")
# scanning single spin   standoff    20 nm   m_min =     0.009 muB
# wide-field ensemble    standoff   500 nm   m_min =     1.348 muB
```

</details>

* * *

## Summary

### Key Takeaways

**1\. A quantum system is an instrument because it supplies a reference and a transducer at once**

  * The discrete transition is a frequency standard fixed by a Hamiltonian, which is what makes a quantum sensor *absolute* — the conversion from frequency to field is a fundamental constant, not a calibration.
  * The relative phase of a superposition is a time integral of the perturbation, accumulated without an amplifier and without added thermal noise.
  * The perturbations that shift a level splitting are exactly the quantities quantum sensors measure: magnetic and electric field, temperature, strain, rotation, and flux.

**2\. Every method in this course is a Ramsey interferometer**

  * Split, accumulate, recombine, read. NV magnetometry, the dc SQUID, the optical clock and the atom interferometer differ in every piece of hardware and in no step of the protocol.
  * The separated-pulse geometry is what decouples the resolution from the drive: the fringe spacing is $1/\tau$, set by the free interval alone.
  * Because the structure is shared, the sensitivity formula is derived once and used four times.

**3\. Projection noise sets the standard quantum limit, and contrast decides where to stand**

  * $\delta\phi = \sqrt{1 - C^2\cos^2\phi}\,/\,(C|\sin\phi|\sqrt{N})$ exactly; at $C = 1$ this is $1/\sqrt{N}$ *everywhere* on the fringe, and only finite contrast makes the quadrature point special.
  * The $-1/2$ log-log slope was verified over six decades of $N$ to better than 1%, with a fitted exponent of $-0.4981$.
  * The SQL constrains uncorrelated probes only; entanglement can reach $1/N$, which is Chapter 5's subject and Chapter 5's caution.

**4\. The sensitivity $\eta$, and the product that decides it**

  * $\eta = \sqrt{\tau + t_d}\,/\,(\gamma\tau C\sqrt{N})$ in signal per root hertz, so the resolution after averaging for $T$ is $\eta/\sqrt{T}$ — and $\eta$ is an *amplitude* spectral density, whose square is a power.
  * The optimum sits at $\tau = T_2/(2p)^{1/p}$, which is $T_2/2$ for both exponential and Gaussian decay, giving $\eta_\mathrm{min} = \sqrt{2e}/(\gamma\sqrt{N T_2})$.
  * Only the product $N T_2$ appears — until dead time is included, at which point a large $N$ bought with a short $T_2$ is punished by the duty cycle, by three decades in the worst case tabulated.

**5\. Averaging stops helping, and the Allan deviation says when**

  * White noise gives $\sigma_y \propto \tau^{-1/2}$, flicker a plateau, random walk $\tau^{+1/2}$ and deterministic drift $\tau^{+1}$; all four slopes were recovered numerically to within 0.01.
  * A sensor with all three processes has a floor and an optimal averaging time; extrapolating the white branch past the turnover overstated the achievable resolution by a factor of 35 in Code Example 4.
  * The remedy is modulation, not patience: chopping above the flicker corner keeps the measurement permanently in the regime where $\eta$ means what it says.

**6\. The filter function reads three ways, and the resolution trade-off does not negotiate**

  * As protection, CPMG-$N$ rejects slow noise by twelve decades at $f\tau = 0.01$; as a lock-in, it rectifies a synchronized AC field with efficiency $2/\pi$ exactly; as spectroscopy, $T_2(N)\propto N^{\alpha/(\alpha+1)}$ returns the noise exponent of the host material.
  * $\eta_B \propto d^{-3/2}$ while resolution $\propto d$, so field sensitivity and moment sensitivity improve in opposite directions and no configuration is best at both.
  * Raising the spin density does not help once $T_2 \propto 1/n$: $N T_2$ is constant and $\eta_B$ does not move. And one decade of standoff costs six decades of $N T_2$, which is why nanoscale magnetometry is a surface technique.

**Practical implications**

  * Never accept an $\eta$ without an averaging time and a stability curve; the pair $(\eta, \tau^\ast)$ is the specification, and $\eta$ alone is half of one.
  * Check whether a quoted sensitivity is amplitude or power spectral density before comparing two instruments, and check whether it is DC or AC before comparing two protocols.
  * Optimize the cheap parameters first: contrast and dead time enter $\eta$ linearly, while $N$ and $T_2$ enter as square roots.
  * When a target lies below the Allan floor, the answer is a modulation scheme, not a longer measurement.

The next three chapters take the three platform families in turn, each one an instance of the template built here. Chapter 2 starts with the nitrogen-vacancy centre, where the sensor is a point defect in diamond — a *materials* defect, deliberately created, whose spin-dependent fluorescence provides the readout and whose proximity to a surface sets both the resolution and the coherence. It is the clearest case in the whole subject of a materials problem and a measurement problem being the same problem.

[← Series Top](<index.html>) [Chapter 2: NV-Center Magnetometry →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * Every sensitivity figure in this course is computed from the stated formulas and the stated assumptions: the numbers illustrate scaling laws and are not measurements, specifications, or claims about any instrument.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
