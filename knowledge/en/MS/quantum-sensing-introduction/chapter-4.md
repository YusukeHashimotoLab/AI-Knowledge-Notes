---
title: "Chapter 4: Atomic Clocks and Atom Interferometry"
chapter_title: "Chapter 4: Atomic Clocks and Atom Interferometry"
subtitle: Ramsey as a Time Standard, the Light-Pulse Interferometer, and the Magnetometer That Needs No Cryostat
reading_time: 40-45 minutes
difficulty: Advanced
code_examples: 6
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/MS/quantum-sensing-introduction/chapter-4.html>) | Last sync: 2026-08-13

[Materials Science Dojo](<../index.html>) > [Introduction to Quantum Sensing](<index.html>) > Chapter 4

Chapters 2 and 3 measured magnetic fields with solids: a defect in diamond, a ring of superconductor. This chapter measures with free atoms, and in doing so it closes the argument that Chapter 1 opened. The Ramsey sequence of §1.2 was introduced as a phase estimator; a clock is that estimator wired into a feedback loop, and an atom interferometer is the same estimator with the two arms separated in space rather than only in energy. Nothing new is needed. What changes is which term in the accumulated phase is the signal and which are the systematic errors, and that reassignment is the whole content of metrology.

Three things make this chapter worth a materials scientist's time even though no sample is ever inserted into an atomic clock. The first is that clocks are where the discipline of an error budget was invented, and that discipline transfers directly: the distinction between **stability** — how the uncertainty falls with averaging time — and **accuracy** — the floor that averaging cannot reach — is the single most useful idea in the whole subject, and Chapter 3's SQUID and Chapter 2's NV centre both have it. The second is that the light-pulse atom interferometer is a gravimeter and a gradiometer, which makes it a density probe: it measures mass distribution without touching it. The third is the vapour-cell magnetometer, which reaches femtotesla sensitivity in a glass cell at 150 °C with no cryogenics at all — and pays for it in bandwidth and in physical size, in a trade that is quantitative enough to compute.

**Units and conventions.** $T_1$, $T_2$, $T_2^\ast$, the Ramsey and echo sequences, and the sensitivity notation $\eta$ in unit$/\sqrt{\mathrm{Hz}}$ are exactly as fixed in [Chapter 1](<chapter-1.html>) §1.3-§1.4, which in turn follows the sister course [Introduction to Quantum Hardware](<../../FM/quantum-hardware-introduction/chapter-1.html>). Symbols such as $\omega$, $\delta$ and $\Omega$ are *angular* frequencies in rad/s; quoted numbers are cyclic frequencies in Hz, written $\nu$ or $\omega/2\pi$. Every factor of $2\pi$ in the code is explicit. Fractional frequency is $y = (\nu - \nu_0)/\nu_0$, and the Allan deviation of $y$ at averaging time $\tau$ is written $\sigma_y(\tau)$. Magnetic field sensitivities are quoted in T$/\sqrt{\mathrm{Hz}}$, accelerations in m s$^{-2}/\sqrt{\mathrm{Hz}}$.

## Learning Objectives

After completing this chapter, you will be able to:

  * Derive the projection-noise-limited fractional stability of a Ramsey clock, $\sigma_y(\tau) = 1/(\pi Q \sqrt{N})\sqrt{T_c/\tau}$, and explain why an optical transition wins by five decades of $Q$ rather than by anything about the atom
  * Compute an Allan deviation from a frequency record, identify white, flicker and random-walk frequency noise from its slope, and locate the averaging time beyond which a clock gets worse
  * Distinguish stability from accuracy, assemble a systematic-shift budget with blackbody, Zeeman, gravitational and light-shift terms, and state which single term to attack first
  * Explain why an optical lattice clock needs a magic wavelength and an ion clock needs micromotion control, reusing the dipole-trap and Paul-trap physics of the sister hardware course
  * Compute the Mach-Zehnder phase $k_\mathrm{eff} a T^2$ of a light-pulse atom interferometer, its shot-noise-limited acceleration sensitivity, the apparatus height it implies, and the fringe ambiguity it creates
  * Integrate the spin dynamics of an optically pumped vapour cell, recover the Larmor frequency and the transverse relaxation rate from a free-induction decay, and recognize a field gradient as a $T_2^\ast$ problem
  * Explain the spin-exchange relaxation-free regime as a design trade, and quantify how much bandwidth and spatial resolution are given up for how much field sensitivity
  * Place the vapour cell, the SQUID and the NV centre on one sensitivity-versus-size map and say which physical property decides between them

* * *

## 4.1 Ramsey as a Time Standard

### From a phase estimator to a frequency reference

A clock is an oscillator plus a correction. The oscillator is technological — a quartz crystal, a microwave synthesizer, a laser locked to a cavity — and it drifts. The atoms do not drift, and their role is only to tell the oscillator by how much it has: run the Ramsey sequence of §1.2 with the oscillator as the drive, read the fringe, and steer the oscillator back towards the atomic resonance. Everything that determines how good the clock is lies in how sharply the fringe reports a frequency error.

Take the standard $\pi/2$ - wait $T$ - $\pi/2$ sequence with the drive detuned from resonance by an angular amount $\delta$. The accumulated phase is $\varphi = \delta T$, and with the convention of Chapter 1 the excited-state probability is

$$ P(1) = \frac{1}{2}\left(1 - \cos \delta T\right) $$

This is a *discriminator*: its derivative converts a frequency error into a population change. The derivative is largest where $P = 1/2$, at $\delta T = \pi/2$, and there

$$ \left| \frac{\partial P}{\partial \delta} \right| = \frac{T}{2} $$

so a longer free evolution is a steeper discriminator, in exact proportion. That is the entire reason a clock wants a long $T$, and it is also why $T$ is bounded by $T_2$ — beyond the coherence time the fringe has no contrast left to have a slope.

### The stability formula

Now add noise. Each of the $N$ atoms is projected independently, so at the half-fringe bias point the estimate of $P$ has standard deviation $1/(2\sqrt{N})$, and dividing by the slope gives the frequency uncertainty of one shot:

$$ \sigma_\delta = \frac{1}{T\sqrt{N}} \qquad\Longrightarrow\qquad \sigma_\nu = \frac{1}{2\pi T \sqrt{N}} $$

Fractionally, and averaged over $\tau/T_c$ independent cycles of duration $T_c$,

$$ \sigma_y(\tau) = \frac{1}{2\pi \nu_0 T \sqrt{N}} \sqrt{\frac{T_c}{\tau}} = \frac{1}{\pi Q \sqrt{N}}\sqrt{\frac{T_c}{\tau}}, \qquad Q \equiv \frac{\nu_0}{\Delta\nu} = 2\nu_0 T $$

with $\Delta\nu = 1/2T$ the Ramsey linewidth. The second form is the one to remember, because it separates the three levers cleanly. **Quality factor $Q$**: the transition frequency divided by the linewidth. **Atom number $N$**: only under a square root. **Duty cycle $T_c/T$**: dead time between cycles is pure loss. Note what is absent — nothing about the *kind* of atom appears except through $\nu_0$ and through the $T$ that its coherence permits.

This is the frequency-domain twin of the field-sensitivity result already derived in [Chapter 1](<chapter-1.html>) §1.3. There the optimum interrogation time under a decay envelope $\exp[-(t/T_2)^p]$ was $\tau_\mathrm{opt} = T_2/(2p)^{1/p}$, which for the exponential case $p = 1$ is $T_2/2$, and the resulting best sensitivity was $\eta_\mathrm{min} = \sqrt{2e}/(\gamma\sqrt{N T_2})$. The same structure governs a clock: the free evolution $T$ cannot exceed the coherence time without losing fringe contrast, the optimum sits at a fixed fraction of $T_2$, and the factor $\sqrt{e}$ that appears there is the price of running long enough to be sensitive while short enough to still have contrast. Chapter 5 §5.2 uses exactly this figure of merit to price entanglement, so it is worth having the $T_2/2$ and the $\sqrt{2e}$ in mind now.

### Code Example 1: The Discriminator, and the Stability It Allows

```python
"""Chapter 4, Example 1: the Ramsey fringe as a frequency discriminator,
and the fractional stability that projection noise allows."""
import numpy as np

TWO_PI = 2.0 * np.pi


def ramsey_prob(delta, T):
    """Probability of the |1> outcome after a pi/2 - T - pi/2 Ramsey sequence.

    delta is the ANGULAR detuning omega - omega_0 in rad/s, T the free
    evolution time in s. The convention is the one fixed in Chapter 1:
    P = 0 on resonance, and the steepest slope sits at delta*T = pi/2.
    """
    return 0.5 * (1.0 - np.cos(delta * T))


def discriminator_slope(delta, T):
    """dP/d(delta) of the Ramsey fringe, in seconds per radian."""
    return 0.5 * T * np.sin(delta * T)


T_free = 0.5                      # free evolution time, s
print("Ramsey fringe as a discriminator (T = "
      f"{T_free:.1f} s, so the fringe period is {1.0 / T_free:.1f} Hz)")
print(f"{'detuning (Hz)':>15}{'delta*T (rad)':>15}{'P(1)':>10}"
      f"{'|dP/dnu| (1/Hz)':>18}")
print("-" * 58)
for nu_det in [0.0, 0.125, 0.25, 0.375, 0.5]:
    delta = TWO_PI * nu_det
    print(f"{nu_det:>15.3f}{delta * T_free:>15.4f}"
          f"{ramsey_prob(delta, T_free):>10.4f}"
          f"{abs(discriminator_slope(delta, T_free)) * TWO_PI:>18.4f}")

# --- Projection noise, and the single-shot frequency uncertainty ------------
# At the half-fringe bias point P = 1/2 the per-atom variance is 1/4, so the
# uncertainty of the estimated P from N atoms is 1/(2 sqrt(N)). Dividing by
# the slope T/2 gives sigma_delta = 1/(T sqrt(N)).
rng = np.random.default_rng(20260813)
N_atoms = 10_000
delta_bias = 0.5 * np.pi / T_free            # the half-fringe bias point
trials = 4000
counts = rng.binomial(N_atoms, ramsey_prob(delta_bias, T_free), size=trials)
p_hat = counts / N_atoms
# invert the fringe locally: delta_hat = delta_bias + (p_hat - 0.5)/(T/2)
delta_hat = delta_bias + (p_hat - 0.5) / (0.5 * T_free)
print(f"\nMonte-Carlo check with N = {N_atoms} atoms, {trials} trials")
print(f"  rms of estimated angular detuning : {delta_hat.std(ddof=1):.6f} rad/s")
print(f"  prediction 1/(T sqrt(N))          : "
      f"{1.0 / (T_free * np.sqrt(N_atoms)):.6f} rad/s")
print(f"  ratio                             : "
      f"{delta_hat.std(ddof=1) * T_free * np.sqrt(N_atoms):.4f}")


def stability(nu0, T, N, T_cycle, tau):
    """Projection-noise-limited Allan deviation of a Ramsey clock.

    sigma_y(tau) = 1/(2 pi nu0 T sqrt(N)) * sqrt(T_cycle/tau), which is the
    familiar 1/(pi Q sqrt(N)) * sqrt(T_cycle/tau) with Q = nu0/(1/2T).
    """
    per_shot = 1.0 / (TWO_PI * nu0 * T * np.sqrt(N))
    return per_shot * np.sqrt(T_cycle / tau)


# Transition frequencies are atomic constants; T, N and T_cycle are round
# illustrative operating parameters, not specifications of any apparatus.
clocks = [
    # label,                     nu0 (Hz),   T (s), N,      T_cycle (s)
    ("Cs fountain, microwave",   9.192631770e9, 0.5, 1.0e6, 1.0),
    ("Rb vapour cell, microwave", 6.834682611e9, 0.005, 1.0e10, 0.01),
    ("Sr lattice, optical",      4.292280042e14, 1.0, 1.0e4, 2.0),
    ("Al+ single ion, optical",  1.121015393e15, 1.0, 1.0, 2.0),
]

print(f"\n{'clock':<26}{'nu0 (Hz)':>12}{'Q = 2 nu0 T':>13}{'N':>10}"
      f"{'sigma_y(1 s)':>14}{'tau to 1e-18':>16}")
print("-" * 91)
for label, nu0, T, N, Tc in clocks:
    Q = 2.0 * nu0 * T
    s1 = stability(nu0, T, N, Tc, 1.0)
    # check the Q form against the direct form
    s1_Q = 1.0 / (np.pi * Q * np.sqrt(N)) * np.sqrt(Tc / 1.0)
    assert abs(s1 / s1_Q - 1.0) < 1e-12
    tau_18 = Tc * (1.0 / (TWO_PI * nu0 * T * np.sqrt(N)) / 1e-18) ** 2
    print(f"{label:<26}{nu0:>12.4e}{Q:>13.3e}{N:>10.0e}{s1:>14.3e}"
          f"{tau_18:>13.3e} s")

print("\nThe tau^(-1/2) law, for the Sr lattice row:")
nu0, T, N, Tc = clocks[2][1:]
print(f"{'tau (s)':>12}{'sigma_y(tau)':>16}{'slope on log-log':>20}")
print("-" * 48)
taus = [1.0, 10.0, 100.0, 1000.0, 10000.0, 86400.0]
prev = None
for tau in taus:
    s = stability(nu0, T, N, Tc, tau)
    if prev is None:
        slope = "  (reference)"
    else:
        slope = f"{np.log(s / prev[1]) / np.log(tau / prev[0]):>20.4f}"
    print(f"{tau:>12.0f}{s:>16.4e}{slope:>20}")
    prev = (tau, s)
```

```text
Ramsey fringe as a discriminator (T = 0.5 s, so the fringe period is 2.0 Hz)
  detuning (Hz)  delta*T (rad)      P(1)   |dP/dnu| (1/Hz)
----------------------------------------------------------
          0.000         0.0000    0.0000            0.0000
          0.125         0.3927    0.0381            0.6011
          0.250         0.7854    0.1464            1.1107
          0.375         1.1781    0.3087            1.4512
          0.500         1.5708    0.5000            1.5708

Monte-Carlo check with N = 10000 atoms, 4000 trials
  rms of estimated angular detuning : 0.019956 rad/s
  prediction 1/(T sqrt(N))          : 0.020000 rad/s
  ratio                             : 0.9978

clock                         nu0 (Hz)  Q = 2 nu0 T         N  sigma_y(1 s)    tau to 1e-18
-------------------------------------------------------------------------------------------
Cs fountain, microwave      9.1926e+09    9.193e+09     1e+06     3.463e-14    1.199e+09 s
Rb vapour cell, microwave   6.8347e+09    6.835e+07     1e+10     4.657e-15    2.169e+07 s
Sr lattice, optical         4.2923e+14    8.585e+14     1e+04     5.244e-18    2.750e+01 s
Al+ single ion, optical     1.1210e+15    2.242e+15     1e+00     2.008e-16    4.031e+04 s

The tau^(-1/2) law, for the Sr lattice row:
     tau (s)    sigma_y(tau)    slope on log-log
------------------------------------------------
           1      5.2438e-18         (reference)
          10      1.6582e-18             -0.5000
         100      5.2438e-19             -0.5000
        1000      1.6582e-19             -0.5000
       10000      5.2438e-20             -0.5000
       86400      1.7840e-20             -0.5000
```

**What to look for.** The discriminator table says that half a fringe of detuning is the operating point, and that a clock is deliberately run *off* resonance — at $\delta T = \pi/2$ — because that is where the signal responds to a frequency error at all. On resonance the derivative vanishes and the atoms report nothing. The Monte-Carlo block then confirms the projection-noise formula to two parts in a thousand from $4\times10^7$ simulated atom measurements, which is worth doing once so that $1/(T\sqrt{N})$ stops being a quoted result.

The platform table is where the physics lives. The caesium fountain and the strontium lattice clock differ in $Q$ by five decades and in $N$ by two in the *wrong* direction, and the lattice clock still wins by four decades of $\sigma_y(1\,\mathrm{s})$: $5.244\times10^{-18}$ against $3.463\times10^{-14}$. The reason is entirely the numerator $\nu_0$. This is the argument for optical clocks in one line, and it is why the definition of the second is under review — not because caesium is a bad atom, but because 9.19 GHz is a small number. The single aluminium ion is the interesting row: with $N = 1$ it throws away a factor of 100 in $\sqrt{N}$ relative to the lattice and still reaches $2.008\times10^{-16}$, because its $Q$ is the largest in the table. And the vapour-cell row is a warning label: $4.657\times10^{-15}$ is a *projection-noise bound* for $10^{10}$ atoms, not an achievable stability, because such clocks are limited by light shifts and detection noise long before they see projection noise. A bound computed from one noise source is not a prediction.

The last block verifies the $\tau^{-1/2}$ law to four digits. It has to hold, because averaging independent estimates is all that is happening — which is exactly why the *departures* from it, in the next example, are informative.

### Allan deviation, revisited

Section 1.4 introduced the Allan variance as the two-sample variance of a frequency record,

$$ \sigma_y^2(\tau) = \frac{1}{2}\left\langle \left( \bar{y}_{k+1} - \bar{y}_k \right)^2 \right\rangle $$

where $\bar{y}_k$ is the fractional frequency averaged over the $k$-th interval of length $\tau$. Its virtue is that it converges for noise processes whose ordinary variance does not, and its use here is diagnostic: the slope on a log-log plot names the noise process.

| Noise process | Power spectral density of $y$ | Allan slope | Physical origin in a clock |
| --- | --- | --- | --- |
| White frequency | $S_y \propto f^0$ | $\tau^{-1/2}$ | projection noise, detection shot noise |
| Flicker frequency | $S_y \propto 1/f$ | $\tau^{0}$ | the same $1/f$ defect ensembles as §1.4; cavity and laser flicker |
| Random-walk frequency | $S_y \propto 1/f^2$ | $\tau^{+1/2}$ | slow environmental drift, temperature |
| Linear frequency drift | not a stationary process | $\tau^{+1}$ | ageing of a component |

The flicker row is the one that connects to the rest of this series: $1/f$ noise in a clock has the same microscopic origin — an ensemble of two-level fluctuators with a log-uniform distribution of switching rates — as the $1/f$ noise that limits a transmon in [Chapter 2 of the hardware course](<../../FM/quantum-hardware-introduction/chapter-2.html>) and the $1/f$ flux noise that limits a SQUID in [Chapter 3](<chapter-3.html>) of this one. Three quite different instruments, one materials problem.

**Which estimator.** The definition above is the two-sample, non-overlapping Allan variance, which is what Chapter 1's Example 4 computed. Example 2 below uses the **overlapping** estimator instead, which averages over every starting phase rather than over disjoint blocks. Both estimate the same $\sigma_y(\tau)$ and agree in expectation; the overlapping one simply has more degrees of freedom at long $\tau$, where a non-overlapping estimate has only a handful of independent differences left and scatters badly. That is why clock work uses it and why the two chapters differ. Slopes and regime boundaries are unaffected; only the error bars are.

### Code Example 2: Allan Slopes, and the Averaging Time That Stops Helping

```python
"""Chapter 4, Example 2: Allan deviation of synthesized clock noise.
Continues from Example 1 (same session)."""


def synth_power_law(n, alpha, rng, sigma=1.0):
    """Fractional-frequency series y_k whose one-sided PSD goes as 1/f^alpha.

    alpha = 0 gives white frequency noise, alpha = 1 flicker frequency noise,
    alpha = 2 random-walk frequency noise. The series is scaled to unit
    standard deviation before the caller's sigma is applied, so that the three
    noise types enter a budget with comparable weight.
    """
    m = n // 2 + 1
    f = np.arange(m, dtype=float)
    f[0] = 1.0
    amp = f ** (-0.5 * alpha)
    amp[0] = 0.0                      # no DC term: it is not observable
    phase = rng.uniform(0.0, TWO_PI, m)
    spec = amp * np.exp(1j * phase)
    y = np.fft.irfft(spec, n)
    return sigma * y / y.std(ddof=1)


def overlapping_avar(y, tau0, m_list):
    """Overlapping Allan variance from a fractional-frequency series.

    sigma_y^2(m tau0) = sum_j ( sum_{i=j}^{j+m-1} (y_{i+m} - y_i) )^2
                        / (2 m^2 (M - 2m + 1))
    """
    y = np.asarray(y, dtype=float)
    M = len(y)
    taus, avars = [], []
    for m in m_list:
        if M - 2 * m + 1 < 1:
            continue
        d = y[m:] - y[:-m]                       # length M - m
        c = np.concatenate(([0.0], np.cumsum(d)))
        s = c[m:] - c[:-m]                       # rolling sums of m terms
        s = s[:M - 2 * m + 1]
        avars.append(np.sum(s ** 2) / (2.0 * m ** 2 * len(s)))
        taus.append(m * tau0)
    return np.array(taus), np.array(avars)


def loglog_slope(tau, adev, lo, hi):
    """Least-squares slope of log(adev) against log(tau) over [lo, hi]."""
    k = (tau >= lo) & (tau <= hi)
    return np.polyfit(np.log(tau[k]), np.log(adev[k]), 1)[0]


n_pts, tau0 = 2 ** 18, 1.0
rng2 = np.random.default_rng(4242)
m_list = np.unique(np.round(np.logspace(0, 4.4, 30)).astype(int))

print(f"Overlapping Allan deviation of {n_pts} one-second samples")
print(f"{'noise type':<28}{'PSD':<12}{'slope fit':>12}{'expected':>11}")
print("-" * 63)
series = {}
for label, alpha, expect in [("white frequency", 0.0, -0.5),
                             ("flicker frequency", 1.0, 0.0),
                             ("random-walk frequency", 2.0, +0.5)]:
    y = synth_power_law(n_pts, alpha, rng2, sigma=1e-13)
    series[label] = y
    tau, av = overlapping_avar(y, tau0, m_list)
    s = loglog_slope(tau, np.sqrt(av), 4.0, 3000.0)
    print(f"{label:<28}{'1/f^' + str(int(alpha)):<12}{s:>12.4f}{expect:>11.2f}")

# A pure linear drift is not a noise process at all, and its Allan slope is +1.
t_axis = np.arange(n_pts) * tau0
y_drift = 2e-19 * t_axis
tau_d, av_d = overlapping_avar(y_drift, tau0, m_list)
print(f"{'linear frequency drift':<28}{'-':<12}"
      f"{loglog_slope(tau_d, np.sqrt(av_d), 4.0, 3000.0):>12.4f}{1.0:>11.2f}")

# --- The white-frequency law, checked against the sqrt(tau0/tau) prediction --
y_w = series["white frequency"]
tau_w, av_w = overlapping_avar(y_w, tau0, m_list)
ad_w = np.sqrt(av_w)
print("\nWhite frequency noise: measured against sigma_y(1 s) sqrt(1 s / tau)")
print(f"{'tau (s)':>10}{'measured':>14}{'predicted':>14}{'ratio':>9}")
print("-" * 47)
ref = ad_w[0]
for target in [1, 10, 100, 1000, 10000]:
    k = int(np.argmin(np.abs(tau_w - target)))
    pred = ref * np.sqrt(tau_w[0] / tau_w[k])
    print(f"{tau_w[k]:>10.0f}{ad_w[k]:>14.4e}{pred:>14.4e}"
          f"{ad_w[k] / pred:>9.4f}")

# --- A realistic budget: white noise, a flicker floor, and a drift ----------
parts = {
    "white": series["white frequency"],                     # sigma 1e-13
    "flicker": 0.08 * series["flicker frequency"],          # sigma 8e-15
    "drift": y_drift,                                       # 2e-19 per second
}
y_tot = sum(parts.values())
tau_t, av_t = overlapping_avar(y_tot, tau0, m_list)
ad_t = np.sqrt(av_t)
ad_parts = {}
for label, y_i in parts.items():
    _, av_i = overlapping_avar(y_i, tau0, m_list)
    ad_parts[label] = np.sqrt(av_i)
k_min = int(np.argmin(ad_t))
print("\nCombined budget: which term dominates is computed, not asserted")
head = (f"{'tau (s)':>9}{'white':>12}{'flicker':>12}{'drift':>12}"
        f"{'quadrature':>13}{'combined':>12}{'dominant':>10}")
print(head)
print("-" * len(head))
for k in range(0, len(tau_t), 3):
    vals = {L: ad_parts[L][k] for L in parts}
    quad = np.sqrt(sum(v ** 2 for v in vals.values()))
    dom = max(vals, key=lambda L: vals[L])
    print(f"{tau_t[k]:>9.0f}{vals['white']:>12.3e}{vals['flicker']:>12.3e}"
          f"{vals['drift']:>12.3e}{quad:>13.3e}{ad_t[k]:>12.3e}{dom:>10}")
print(f"\n  best stability {ad_t[k_min]:.4e} at tau = {tau_t[k_min]:.0f} s;")
print("  averaging longer than that makes this clock worse, not better.")
ratio = ad_t / np.sqrt(sum(ad_parts[L] ** 2 for L in parts))
print(f"  combined / quadrature sum, over all tau: min {ratio.min():.4f}, "
      f"max {ratio.max():.4f}")
print("  -- the three terms add in quadrature, as they must.")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for label in series:
    tau_i, av_i = overlapping_avar(series[label], tau0, m_list)
    ax.loglog(tau_i, np.sqrt(av_i), marker="o", ms=3, label=label)
ax.loglog(tau_t, ad_t, "k-", lw=1.6, label="combined budget")
ax.set_xlabel("averaging time tau (s)")
ax.set_ylabel("Allan deviation sigma_y(tau)")
ax.set_title("Three noise types, three slopes")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
```

```text
Overlapping Allan deviation of 262144 one-second samples
noise type                  PSD            slope fit   expected
---------------------------------------------------------------
white frequency             1/f^0            -0.4995      -0.50
flicker frequency           1/f^1            -0.0016       0.00
random-walk frequency       1/f^2             0.4995       0.50
linear frequency drift      -                 1.0000       1.00

White frequency noise: measured against sigma_y(1 s) sqrt(1 s / tau)
   tau (s)      measured     predicted    ratio
-----------------------------------------------
         1    1.0000e-13    1.0000e-13   1.0000
         8    3.5355e-14    3.5355e-14   1.0000
        94    1.0312e-14    1.0314e-14   0.9997
      1083    3.0421e-15    3.0387e-15   1.0011
      8807    1.0472e-15    1.0656e-15   0.9828

Combined budget: which term dominates is computed, not asserted
  tau (s)       white     flicker       drift   quadrature    combined  dominant
--------------------------------------------------------------------------------
        1   1.000e-13   2.921e-15   1.414e-19    1.000e-13   1.000e-13     white
        4   5.000e-14   2.718e-15   5.657e-19    5.007e-14   5.007e-14     white
       12   2.887e-14   2.685e-15   1.697e-18    2.899e-14   2.899e-14     white
       33   1.741e-14   2.680e-15   4.667e-18    1.761e-14   1.760e-14     white
       94   1.031e-14   2.680e-15   1.329e-17    1.065e-14   1.062e-14     white
      268   6.112e-15   2.682e-15   3.790e-17    6.674e-15   6.629e-15     white
      763   3.623e-15   2.667e-15   1.079e-16    4.500e-15   4.549e-15     white
     2177   2.157e-15   2.683e-15   3.079e-16    3.456e-15   3.419e-15   flicker
     6210   1.287e-15   2.730e-15   8.782e-16    3.143e-15   3.253e-15   flicker
    17712   6.969e-16   2.585e-15   2.505e-15    3.666e-15   3.654e-15   flicker

  best stability 3.2428e-15 at tau = 4379 s;
  averaging longer than that makes this clock worse, not better.
  combined / quadrature sum, over all tau: min 0.8711, max 1.0412
  -- the three terms add in quadrature, as they must.
```

**What to look for.** The four slopes come out at $-0.4995$, $-0.0016$, $+0.4995$ and $+1.0000$ against the predicted $-1/2$, $0$, $+1/2$ and $+1$. That is the diagnostic in action: given a measured Allan deviation and nothing else, the slope identifies the dominant noise process, and therefore where to look for its cause. A slope of $-1/2$ says the clock is doing as well as its atoms allow and needs more of them or more time. A slope of $0$ says something is fluctuating on the timescale of the measurement and no amount of averaging will help. A slope of $+1$ says a component is ageing.

The white-noise verification is worth reading with its final row. At $\tau = 8807$ s the measured value falls $1.7\%$ below the $\tau^{-1/2}$ prediction, not because the law fails but because a $2^{18}$-second record contains only about thirty independent differences at that $\tau$. Allan deviations at the right-hand end of any plot are noisy estimates, and error bars there are not decoration.

The combined budget is the shape every real clock has. Below a few hundred seconds the white term dominates and averaging pays; between roughly 2000 and 20 000 seconds the flicker floor takes over and averaging stops paying; beyond that the drift term grows and averaging actively hurts. The best stability of this synthetic clock is $3.2428\times10^{-15}$ at $\tau = 4379$ s, and running it for a day would be worse than running it for an hour. The final check — that the three components add in quadrature to within about 13% across the whole range, the ratio running from 0.8711 to 1.0412 — is what licenses treating the budget as a sum of independent terms in the first place. The 13% shortfall is not a failure of independence but finite-sample scatter in a single synthetic realisation, worst where two terms cross and neither dominates; it would shrink with an ensemble of realisations, and it is the size of disagreement to expect from one record of finite length.

The one physical effect deliberately left out of this example is the **Dick effect**: because a pulsed clock is blind during dead time, the local oscillator's noise is aliased down into the measurement band, and a clock with a $50\%$ duty cycle can be limited by its laser rather than by its atoms. It appears here as a concept only, because modelling it requires the sensitivity function of the pulse sequence, which is the filter-function machinery of §1.4 applied to the interrogation cycle rather than to a single sequence. The practical consequence is memorable, though: for the best optical clocks the local oscillator, not the atom, sets the short-term stability, and the fix is a better cavity — which is a materials problem about coating loss and thermal noise in a mirror.

* * *

## 4.2 Optical Lattice and Ion Clocks

### Holding an atom without moving its energy levels

An optical clock needs its atoms held still for a second and not perturbed while they are held. Those two requirements fight, and the two families of optical clock resolve the fight differently.

An **optical lattice clock** confines thousands of neutral atoms in the intensity maxima of a standing wave. The physics is the dipole trap of [Chapter 4 of the hardware course](<../../FM/quantum-hardware-introduction/chapter-4.html>) §4.2: a far-detuned laser shifts the atomic levels by $-\frac{1}{2}\alpha(\omega_L) \langle E^2\rangle$, and the gradient of that shift is a conservative force. The difficulty is immediate — the trap light shifts the clock levels, so the very thing holding the atoms moves the frequency being measured. The escape is that the two clock levels have different polarizabilities $\alpha_g(\omega_L)$ and $\alpha_e(\omega_L)$, and these curves cross. At the **magic wavelength** where $\alpha_g = \alpha_e$ the trap shifts both levels equally and the transition frequency is untouched at first order. The residual shift is then proportional to the detuning from that crossing and to the trap depth, which is why lattice-laser wavelength stabilization is a clock specification.

An **ion clock** holds a single charged atom in the radio-frequency Paul trap of [Chapter 3 of the hardware course](<../../FM/quantum-hardware-introduction/chapter-3.html>) §3.1. There is no trap light at all, so no light shift; instead the ion sits at a node of the rf field and any stray static field that pushes it off that node produces **excess micromotion**, an rf-driven oscillation. Micromotion does two things to a clock: it modulates the transition, putting sidebands on the line, and it gives the ion a mean-square velocity that produces a second-order Doppler (time-dilation) shift $-\langle v^2\rangle/2c^2$. That is the same Mathieu-equation physics as the hardware course's stability diagram, read for its relativistic consequence instead of its trapping consequence.

The trade is clean. The lattice clock has $N \sim 10^4$ and therefore a factor of 100 in projection noise, and pays with a light shift it must engineer away. The ion clock has $N = 1$ and pays 100 in stability, and buys a much simpler systematic budget. Both are represented in Example 1's table, and both reach comparable accuracy by different routes.

### The shifts that have to be counted

Every term below is a frequency shift with a known functional form, and the budget consists of bounding each one's *uncertainty* rather than its size. A large shift that is known to a part in $10^4$ is harmless; a small shift that is unknown by half of itself is not.

| Shift | Scaling | What has to be controlled |
| --- | --- | --- |
| Blackbody radiation | $\propto \Delta\alpha\, T^4$ | temperature of every surface the atoms can see |
| Second-order Zeeman | $\propto B^2$ | magnitude of the bias field, including its ac components |
| Gravitational redshift | $\propto g h/c^2$ | height of the atoms above the reference geoid |
| Lattice light shift | $\propto (\lambda - \lambda_\mathrm{magic})\times$ depth | lattice wavelength and intensity |
| Cold collisions | $\propto$ atom density | how many atoms, and how they are distributed |
| Second-order Doppler | $\propto -\langle v^2\rangle/2c^2$ | micromotion, and residual thermal motion |

The $T^4$ in the first row is the strongest lever in the table, and it is the reason cryogenic enclosures appear in clock laboratories: the sensitivity to a temperature error scales as $T^3$, so cooling the surroundings buys quartically on the shift and cubically on the uncertainty. That is a materials-engineering answer to a physics problem.

### Code Example 3: A Systematic Budget, and the Floor It Sets

```python
"""Chapter 4, Example 3: a systematic-shift budget, and why accuracy and
stability are different numbers.
Continues from Examples 1-2 (same session)."""

c_light = 2.99792458e8            # m/s
g_earth = 9.80665                 # m/s^2

# Every coefficient below is an order-of-magnitude stand-in chosen to make the
# scaling visible; none of them is a measured value for a particular apparatus.
# What is being demonstrated is the STRUCTURE of a budget, not its contents.

nu_optical = 4.292280042e14       # Sr clock transition, Hz (an atomic constant)


def bbr_shift(T_kelvin, frac_at_300K=-5.5e-15):
    """Blackbody-radiation shift, scaling as T^4 (differential polarizability)."""
    return frac_at_300K * (T_kelvin / 300.0) ** 4


def bbr_uncertainty(T_kelvin, dT, frac_at_300K=-5.5e-15):
    """Uncertainty of the BBR shift from an uncertainty dT of the enclosure."""
    return abs(4.0 * bbr_shift(T_kelvin, frac_at_300K) * dT / T_kelvin)


print("Blackbody radiation: the T^4 lever")
print(f"{'enclosure T (K)':>17}{'shift':>13}{'unc. for dT = 1 K':>21}")
print("-" * 51)
for T_env in [300.0, 200.0, 100.0, 77.0]:
    print(f"{T_env:>17.0f}{bbr_shift(T_env):>13.2e}"
          f"{bbr_uncertainty(T_env, 1.0):>21.2e}")

beta_zeeman = 2.33e7              # Hz/T^2, second-order Zeeman coefficient
print("\nSecond-order Zeeman: the B^2 lever")
print(f"{'bias field (T)':>16}{'shift (Hz)':>14}{'fractional':>13}"
      f"{'unc. at 1% of B':>18}")
print("-" * 61)
for B in [1e-3, 1e-4, 1e-5]:
    shift = beta_zeeman * B ** 2
    frac = shift / nu_optical
    print(f"{B:>16.0e}{shift:>14.4e}{frac:>13.2e}{2.0 * 0.01 * frac:>18.2e}")

print("\nGravitational redshift: g h / c^2 per metre of height")
print(f"  fractional shift per metre : {g_earth / c_light ** 2:.3e}")
for dh in [1.0, 0.01, 0.001]:
    print(f"  uncertainty for dh = {dh * 1e3:7.1f} mm : "
          f"{g_earth * dh / c_light ** 2:.3e}")

# Second-order Doppler from excess micromotion in an ion trap. The Mathieu
# physics behind v_rf is the same one derived in the quantum-hardware course,
# Chapter 3; here only the relativistic consequence is needed.
print("\nSecond-order Doppler from ion micromotion: -<v^2>/(2 c^2)")
print(f"{'rf micromotion amplitude v (m/s)':>34}{'fractional shift':>19}")
print("-" * 53)
for v_rf in [1.0, 0.3, 0.1, 0.03]:
    print(f"{v_rf:>34.2f}{-0.5 * v_rf ** 2 / (2 * c_light ** 2):>19.2e}")

# --- The budget itself ------------------------------------------------------
budget = [
    ("blackbody radiation, 300 K enclosure, dT = 1 K",
     bbr_shift(300.0), bbr_uncertainty(300.0, 1.0)),
    ("second-order Zeeman, B = 0.1 mT known to 1%",
     beta_zeeman * 1e-4 ** 2 / nu_optical,
     2.0 * 0.01 * beta_zeeman * 1e-4 ** 2 / nu_optical),
    ("gravitational redshift, height known to 10 mm",
     0.0, g_earth * 0.010 / c_light ** 2),
    ("lattice light shift, residual after magic-wavelength tuning",
     0.0, 2.0e-17),
    ("cold-collision density shift", -1.0e-17, 5.0e-18),
    ("residual second-order Doppler, v = 0.1 m/s", -5.6e-19, 5.6e-19),
]
print(f"\n{'contribution':<60}{'shift':>12}{'uncertainty':>14}")
print("-" * 86)
tot_sq = 0.0
for name, shift, unc in budget:
    tot_sq += unc ** 2
    print(f"{name:<60}{shift:>12.1e}{unc:>14.1e}")
u_tot = np.sqrt(tot_sq)
print("-" * 86)
print(f"{'total, added in quadrature':<60}{'':>12}{u_tot:>14.1e}")

# When does the statistical uncertainty of Example 1 hit that floor?
print(f"\nHow long until the projection noise of Example 1 reaches the "
      f"{u_tot:.1e} floor?")
print(f"{'clock':<26}{'sigma_y(1 s)':>14}{'tau to reach floor':>21}")
print("-" * 61)
for label, nu0_i, T_i, N_i, Tc_i in clocks:
    per_shot = 1.0 / (TWO_PI * nu0_i * T_i * np.sqrt(N_i))
    tau_floor = Tc_i * (per_shot / u_tot) ** 2
    print(f"{label:<26}{stability(nu0_i, T_i, N_i, Tc_i, 1.0):>14.3e}"
          f"{tau_floor:>18.3e} s")
print("  Beyond that time, averaging buys precision and not accuracy: a")
print("  systematic budget does not average down with tau. The Sr row reaches")
print("  this floor in milliseconds and the single ion in seconds, which is")
print("  why optical clock work is a systematics-reduction programme first and")
print("  a stability programme second.")
print(f"\nWith the enclosure at 100 K instead of 300 K the BBR term falls to "
      f"{bbr_uncertainty(100.0, 1.0):.1e},")
u_cold = np.sqrt(tot_sq - bbr_uncertainty(300.0, 1.0) ** 2
                 + bbr_uncertainty(100.0, 1.0) ** 2)
print(f"  and the total to {u_cold:.1e} -- a factor "
      f"{u_tot / u_cold:.1f} for one cryostat. The lattice light shift is now")
print("  the largest single term, so that is where the next effort goes.")
```

```text
Blackbody radiation: the T^4 lever
  enclosure T (K)        shift    unc. for dT = 1 K
---------------------------------------------------
              300    -5.50e-15             7.33e-17
              200    -1.09e-15             2.17e-17
              100    -6.79e-17             2.72e-18
               77    -2.39e-17             1.24e-18

Second-order Zeeman: the B^2 lever
  bias field (T)    shift (Hz)   fractional   unc. at 1% of B
-------------------------------------------------------------
           1e-03    2.3300e+01     5.43e-14          1.09e-15
           1e-04    2.3300e-01     5.43e-16          1.09e-17
           1e-05    2.3300e-03     5.43e-18          1.09e-19

Gravitational redshift: g h / c^2 per metre of height
  fractional shift per metre : 1.091e-16
  uncertainty for dh =  1000.0 mm : 1.091e-16
  uncertainty for dh =    10.0 mm : 1.091e-18
  uncertainty for dh =     1.0 mm : 1.091e-19

Second-order Doppler from ion micromotion: -<v^2>/(2 c^2)
  rf micromotion amplitude v (m/s)   fractional shift
-----------------------------------------------------
                              1.00          -2.78e-18
                              0.30          -2.50e-19
                              0.10          -2.78e-20
                              0.03          -2.50e-21

contribution                                                       shift   uncertainty
--------------------------------------------------------------------------------------
blackbody radiation, 300 K enclosure, dT = 1 K                  -5.5e-15       7.3e-17
second-order Zeeman, B = 0.1 mT known to 1%                      5.4e-16       1.1e-17
gravitational redshift, height known to 10 mm                    0.0e+00       1.1e-18
lattice light shift, residual after magic-wavelength tuning      0.0e+00       2.0e-17
cold-collision density shift                                    -1.0e-17       5.0e-18
residual second-order Doppler, v = 0.1 m/s                      -5.6e-19       5.6e-19
--------------------------------------------------------------------------------------
total, added in quadrature                                                     7.7e-17

How long until the projection noise of Example 1 reaches the 7.7e-17 floor?
clock                       sigma_y(1 s)   tau to reach floor
-------------------------------------------------------------
Cs fountain, microwave         3.463e-14         2.025e+05 s
Rb vapour cell, microwave      4.657e-15         3.663e+03 s
Sr lattice, optical            5.244e-18         4.643e-03 s
Al+ single ion, optical        2.008e-16         6.807e+00 s
  Beyond that time, averaging buys precision and not accuracy: a
  systematic budget does not average down with tau. The Sr row reaches
  this floor in milliseconds and the single ion in seconds, which is
  why optical clock work is a systematics-reduction programme first and
  a stability programme second.

With the enclosure at 100 K instead of 300 K the BBR term falls to 2.7e-18,
  and the total to 2.3e-17 -- a factor 3.3 for one cryostat. The lattice light shift is now
  the largest single term, so that is where the next effort goes.
```

**What to look for.** Read the four levers first. Blackbody radiation at room temperature is a shift of $-5.5\times10^{-15}$, four decades above the level anyone cares about, and a 1 K uncertainty in the enclosure temperature leaves $7.33\times10^{-17}$ of that unknown. Cooling the enclosure to 100 K does not reduce the shift by a factor of 3; it reduces it by $3^4 = 81$, and the uncertainty by 27, to $2.72\times10^{-18}$. The second-order Zeeman term shows the same structure with a square instead of a fourth power: at $0.1$ mT the shift is $0.233$ Hz, and knowing the field to $1\%$ leaves $1.09\times10^{-17}$. The gravitational redshift is the term with no adjustable physics at all — $1.091\times10^{-16}$ per metre of height, so a clock's *altitude* must be surveyed to a centimetre before its frequency means anything. Two clocks in the same building at different floors do not agree, and general relativity says they should not.

The budget itself totals $7.7\times10^{-17}$ in quadrature, and the table beneath it is the point of the section. The Sr lattice row of Example 1 reaches that statistical uncertainty in $4.6$ milliseconds. Everything after those four milliseconds buys precision that the systematic budget will not honour. That is the operational content of the distinction between stability and accuracy, and it is why the literature of optical clocks is a literature about blackbody enclosures, lattice-wavelength servos and geodetic surveys rather than about atoms. The last block makes the design consequence explicit: cooling the enclosure improves the total by a factor $3.3$, after which the lattice light shift becomes the largest single term and the next effort goes there. An error budget is a to-do list sorted by size.

The same reasoning is what Chapter 3 applied to a SQUID and Chapter 2 to an NV centre, in each case with a different top entry. It is the transferable skill of this chapter.

* * *

## 4.3 Light-Pulse Atom Interferometry

### A laser pulse is a beam splitter

Everything so far has separated the two arms of the Ramsey interferometer only in energy. Now separate them in space. A two-photon Raman or Bragg pulse transfers momentum $\hbar k_\mathrm{eff}$ at the same time as it drives the internal transition, with $k_\mathrm{eff} = k_1 + k_2 \approx 2k$ for counter-propagating beams. An atom in a superposition of the two internal states is then also in a superposition of two momenta, and the two components physically separate as they fly. A $\pi/2$ pulse is a beam splitter, a $\pi$ pulse is a mirror, and the sequence $\pi/2$ - $T$ - $\pi$ - $T$ - $\pi/2$ is a **Mach-Zehnder interferometer** made of atoms.

The phase difference between the two arms, for a uniform acceleration $a$ along $k_\mathrm{eff}$, is

$$ \Delta\varphi = k_\mathrm{eff}\, a\, T^2 $$

The $T^2$ is the reason this instrument exists. It is not a coherence-time factor like the $T$ of a clock; it is the classical statement that displacement grows quadratically in time, and it means that doubling the free-evolution interval is worth a factor of four. Every other lever in the problem is worth less: atom number enters as $\sqrt{N}$, and $k_\mathrm{eff}$ can be multiplied only by adding photon pairs, at a fidelity cost.

Rotation enters the same way, because in a rotating frame an acceleration $2\boldsymbol{\Omega}\times\mathbf{v}$ appears. A Mach-Zehnder interferometer with an atom moving transversely at $v$ therefore measures rotation with the same phase-to-acceleration conversion, and the same instrument is a gyroscope. A pair of interferometers separated vertically measures the *difference* of $g$ and is a gradiometer, which rejects common-mode platform vibration — the noise that otherwise dominates.

### Code Example 4: Mach-Zehnder Sensitivity Against $T$

```python
"""Chapter 4, Example 4: the light-pulse Mach-Zehnder atom interferometer.
Continues from Examples 1-3 (same session)."""

hbar = 1.054571817e-34
u_mass = 1.66053906660e-27
m_Rb = 86.909180527 * u_mass       # Rb-87
lam_D2 = 780.241209e-9             # m
k_eff = 2.0 * (TWO_PI / lam_D2)    # two-photon Raman/Bragg momentum transfer


def mz_phase(T, a=g_earth, keff=k_eff):
    """Leading Mach-Zehnder phase for a uniform acceleration a: keff a T^2."""
    return keff * a * T ** 2


def accel_sensitivity(T, N, keff=k_eff):
    """Projection-noise-limited single-shot acceleration uncertainty, m/s^2.

    At the half-fringe bias point the phase uncertainty is 1/sqrt(N), exactly
    as in the Ramsey case of Example 1, and the phase-to-acceleration
    conversion is keff T^2.
    """
    return 1.0 / (np.sqrt(N) * keff * T ** 2)


print(f"Rb-87 two-photon beam splitter: k_eff = {k_eff:.4e} 1/m,")
print(f"  recoil velocity hbar k_eff / m = "
      f"{hbar * k_eff / m_Rb * 1e3:.3f} mm/s")

N_shot = 1.0e6
print(f"\nMach-Zehnder gravimeter, N = {N_shot:.0e} atoms per shot")
head = (f"{'T (ms)':>8}{'phase (rad)':>14}{'fringes':>11}{'apex h (m)':>12}"
        f"{'sep. (um)':>11}{'d_a (m/s^2)':>14}{'d_g/g':>11}")
print(head)
print("-" * len(head))
for T_ms in [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
    T = T_ms * 1e-3
    phi = mz_phase(T)
    apex = 0.5 * g_earth * T ** 2                 # fountain apex above launch
    sep = hbar * k_eff / m_Rb * T                 # arm separation at the apex
    da = accel_sensitivity(T, N_shot)
    print(f"{T_ms:>8.0f}{phi:>14.4e}{phi / TWO_PI:>11.3e}{apex:>12.4f}"
          f"{sep * 1e6:>11.2f}{da:>14.3e}{da / g_earth:>11.2e}")

print("\nThe T^2 lever, stated as a slope:")
Ts = np.array([1e-3, 1e-2, 1e-1, 1.0])
das = accel_sensitivity(Ts, N_shot)
slope = np.polyfit(np.log(Ts), np.log(das), 1)[0]
print(f"  d log(d_a) / d log(T) = {slope:.4f}   (exactly -2)")
print("  Doubling T is worth a factor 4; doubling N is worth a factor 1.41.")

# --- What the same instrument measures besides g ---------------------------
Omega_earth = 7.2921150e-5         # rad/s
T_ref, N_ref = 0.1, N_shot
da_ref = accel_sensitivity(T_ref, N_ref)
print(f"\nOne instrument (T = {T_ref * 1e3:.0f} ms, N = {N_ref:.0e}), "
      f"single-shot d_a = {da_ref:.3e} m/s^2:")
print(f"{'quantity':<34}{'signal':>16}{'signal / d_a':>16}")
print("-" * 66)
rows = [
    ("g itself", g_earth),
    ("Coriolis, Omega_Earth x v, v = 1 m/s", 2.0 * Omega_earth * 1.0),
    ("gravity gradient over 1 m, 3e-6 /s^2", 3.0e-6 * 1.0),
    ("tidal variation of g, ~1e-6 m/s^2", 1.0e-6),
    ("1 tonne at 5 m, G M / r^2", 6.674e-11 * 1.0e3 / 25.0),
]
for name, sig in rows:
    print(f"{name:<34}{sig:>16.4e}{sig / da_ref:>16.3e}")

# --- Averaging: the eta convention of Chapter 1, section 1.3 ---------------
print("\nSensitivity in the eta = (unit)/sqrt(Hz) convention of section 1.3,")
print("with one shot per cycle time T_c:")
print(f"{'T (ms)':>8}{'T_c (s)':>10}{'eta_a (m/s^2/rtHz)':>21}"
      f"{'time for 1 nano-g (s)':>23}")
print("-" * 62)
for T_ms in [10.0, 100.0, 300.0]:
    T = T_ms * 1e-3
    Tc = 2.0 * T + 0.5                            # dead time dominates
    eta_a = accel_sensitivity(T, N_shot) * np.sqrt(Tc)
    target = 1e-9 * g_earth
    print(f"{T_ms:>8.0f}{Tc:>10.2f}{eta_a:>21.3e}{(eta_a / target) ** 2:>23.2f}")

print("\nDynamic range: the phase must be unwrapped.")
for T_ms in [10.0, 100.0]:
    T = T_ms * 1e-3
    da_wrap = TWO_PI / (k_eff * T ** 2)
    print(f"  T = {T_ms:5.0f} ms: one fringe is d_a = {da_wrap:.3e} m/s^2, "
          f"i.e. {da_wrap / g_earth:.2e} g")

fig, ax = plt.subplots(figsize=(6.2, 4.2))
T_grid = np.logspace(-3, 0, 60)
for N_i, style in [(1e4, "--"), (1e6, "-"), (1e8, ":")]:
    ax.loglog(T_grid * 1e3, accel_sensitivity(T_grid, N_i), style,
              label=f"N = {N_i:.0e}")
ax.set_xlabel("free-evolution time T (ms)")
ax.set_ylabel("single-shot acceleration uncertainty (m/s$^2$)")
ax.set_title("Mach-Zehnder sensitivity: $T^{-2}$ and $N^{-1/2}$")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
```

```text
Rb-87 two-photon beam splitter: k_eff = 1.6106e+07 1/m,
  recoil velocity hbar k_eff / m = 11.769 mm/s

Mach-Zehnder gravimeter, N = 1e+06 atoms per shot
  T (ms)   phase (rad)    fringes  apex h (m)  sep. (um)   d_a (m/s^2)      d_g/g
---------------------------------------------------------------------------------
       1    1.5794e+02  2.514e+01      0.0000      11.77     6.209e-05   6.33e-06
      10    1.5794e+04  2.514e+03      0.0005     117.69     6.209e-07   6.33e-08
      30    1.4215e+05  2.262e+04      0.0044     353.07     6.899e-08   7.03e-09
     100    1.5794e+06  2.514e+05      0.0490    1176.91     6.209e-09   6.33e-10
     300    1.4215e+07  2.262e+06      0.4413    3530.72     6.899e-10   7.03e-11
    1000    1.5794e+08  2.514e+07      4.9033   11769.08     6.209e-11   6.33e-12

The T^2 lever, stated as a slope:
  d log(d_a) / d log(T) = -2.0000   (exactly -2)
  Doubling T is worth a factor 4; doubling N is worth a factor 1.41.

One instrument (T = 100 ms, N = 1e+06), single-shot d_a = 6.209e-09 m/s^2:
quantity                                    signal    signal / d_a
------------------------------------------------------------------
g itself                                9.8066e+00       1.579e+09
Coriolis, Omega_Earth x v, v = 1 m/s      1.4584e-04       2.349e+04
gravity gradient over 1 m, 3e-6 /s^2      3.0000e-06       4.832e+02
tidal variation of g, ~1e-6 m/s^2       1.0000e-06       1.611e+02
1 tonne at 5 m, G M / r^2               2.6696e-09       4.300e-01

Sensitivity in the eta = (unit)/sqrt(Hz) convention of section 1.3,
with one shot per cycle time T_c:
  T (ms)   T_c (s)   eta_a (m/s^2/rtHz)  time for 1 nano-g (s)
--------------------------------------------------------------
      10      0.52            4.477e-07                2084.49
     100      0.70            5.195e-09                   0.28
     300      1.10            7.236e-10                   0.01

Dynamic range: the phase must be unwrapped.
  T =    10 ms: one fringe is d_a = 3.901e-03 m/s^2, i.e. 3.98e-04 g
  T =   100 ms: one fringe is d_a = 3.901e-05 m/s^2, i.e. 3.98e-06 g
```

**What to look for.** The first table contains the whole engineering problem of the field. At $T = 100$ ms the interferometer accumulates $1.579\times10^{6}$ radians of phase, its two arms separate by $1.18$ mm, and its single-shot acceleration uncertainty is $6.209\times10^{-9}$ m s$^{-2}$, or $6.33\times10^{-10}$ of $g$, from $10^{6}$ atoms. Push $T$ to one second and the sensitivity improves by a hundred — and the atoms must now be in free fall for two seconds, which means a fountain apex $4.90$ m above the launch and an apparatus that is a tower. That is why long-baseline atom interferometry is built in disused mine shafts and drop towers: the height in metres is $gT^2/2$ and there is no way around it. The measured slope confirms $\partial \log \delta a/\partial \log T = -2.0000$ exactly.

The signal table says what such an instrument can see. Earth's rotation, entering as a Coriolis acceleration for an atom moving at 1 m/s, is $2.3\times10^{4}$ times the single-shot noise — which is why an atom gyroscope works, and also why rotation is a *systematic* for a gravimeter and has to be nulled by tilting the mirror. The gravity gradient over a metre is 480 times the noise. A tonne of mass at five metres produces $2.7\times10^{-9}$ m s$^{-2}$, which is $0.43$ of the single-shot noise and therefore invisible in one shot and straightforward after a few hundred — which is the working principle behind using such instruments to look for voids and density anomalies underground. This is the sense in which an atom interferometer is a materials instrument: it measures a mass distribution remotely, with no contact and no assumption about composition.

The $\eta$ block converts to the sensitivity convention of §1.3 and shows what the duty cycle costs. At $T = 100$ ms the cycle time is dominated by the $0.5$ s of preparation, not by the interferometer, so $\eta_a = 5.195\times10^{-9}$ m s$^{-2}/\sqrt{\mathrm{Hz}}$ and one nano-$g$ takes a fraction of a second. Dead time is the second reason for long $T$: it amortizes the preparation.

The last block is the honest caveat. At $T = 100$ ms one fringe corresponds to $3.901\times10^{-5}$ m s$^{-2}$, so an acceleration known only to worse than that is ambiguous by an integer number of fringes. Sensitivity and dynamic range trade against each other exactly as they do for the vapour-cell magnetometer in §4.4 and for the SQUID's flux-locked loop in §3.2, and the standard resolution is the same in all three cases: a coarse, unambiguous measurement first, then the interferometric one.

* * *

## 4.4 Vapour-Cell Magnetometers and the SERF Regime

### Polarization from light, not from cold

A SQUID measures femtotesla and needs liquid helium. An NV centre works at room temperature and measures microtesla in a volume of tens of cubic nanometres. The vapour-cell magnetometer occupies the third corner: femtotesla sensitivity, no cryogenics, and a sensing volume of cubic millimetres.

The reason it needs no refrigerator is that its polarization does not come from thermal equilibrium. A Zeeman splitting at a few nanotesla is smaller than $k_BT$ at 420 K by twelve orders of magnitude, so thermal polarization is exactly zero for any practical purpose. **Optical pumping** supplies the polarization instead: circularly polarized light resonant with a D line drives atoms out of one ground-state sublevel and they accumulate in the other, reaching a polarization of order unity in milliseconds regardless of temperature. This is the same argument that DiVincenzo's initialization criterion makes in the hardware course — that a dissipative preparation mechanism decouples state preparation from temperature — used here as a sensor design principle.

The dynamics are then the Bloch equations of §1.6 with two additions: an optical pumping term that drives the spin towards the beam direction, and relaxation rates that come from collisions rather than from a solid-state bath.

$$ \frac{d\mathbf{S}}{dt} = \gamma\, \mathbf{S}\times\mathbf{B} \;-\; \Gamma_{2}\, \mathbf{S}_\perp \;-\; \Gamma_{1}\left(S_z - S_z^{0}\right)\hat{z} \;+\; R_\mathrm{op}\left(\tfrac{1}{2}\hat{s} - \mathbf{S}\right) $$

The gyromagnetic ratio is the electron's, reduced by a nuclear slowing-down factor $q$ of order a few because the electron spin is coupled to the nucleus: $\gamma_\mathrm{eff} = \gamma_e/q$. That is still four orders of magnitude larger than a nuclear gyromagnetic ratio, and it is why an alkali vapour is used rather than a noble gas.

### Code Example 5: Spin Precession in a Cell

```python
"""Chapter 4, Example 5: spin precession in a vapour cell, integrated.
Continues from Examples 1-4 (same session)."""
from scipy.integrate import solve_ivp

gamma_e = 1.760859630e11           # rad/s/T, free-electron gyromagnetic ratio
q_slow = 6.0                       # nuclear slowing-down factor, order unity
gamma_eff = gamma_e / q_slow       # effective alkali gyromagnetic ratio


def spin_rhs(t, S, B_vec, Gamma1, Gamma2, R_op, s_hat):
    """Phenomenological Bloch equations for an optically pumped alkali vapour.

    S is the ensemble spin polarization vector (dimensionless). The three terms
    are Larmor precession about B, anisotropic relaxation towards zero, and
    optical pumping at rate R_op towards the beam direction s_hat.
    """
    Sx, Sy, Sz = S
    Bx, By, Bz = B_vec
    prec = gamma_eff * np.array([Sy * Bz - Sz * By,
                                 Sz * Bx - Sx * Bz,
                                 Sx * By - Sy * Bx])
    relax = np.array([Gamma2 * Sx, Gamma2 * Sy, Gamma1 * Sz])
    pump = R_op * (0.5 * np.array(s_hat) - S)
    return prec - relax + pump


def free_induction(B0, Gamma2, t_end, n_pts=20001, Gamma1=None, S0=None):
    """Integrate the free precession of a transverse spin in a field B0 z."""
    if Gamma1 is None:
        Gamma1 = Gamma2
    if S0 is None:
        S0 = [0.5, 0.0, 0.0]
    t_grid = np.linspace(0.0, t_end, n_pts)
    sol = solve_ivp(spin_rhs, (0.0, t_end), S0, t_eval=t_grid,
                    args=((0.0, 0.0, B0), Gamma1, Gamma2, 0.0, (1.0, 0.0, 0.0)),
                    rtol=1e-10, atol=1e-13, method="DOP853")
    return t_grid, sol.y


def fit_fid(t, sx, sy):
    """Recover the Larmor frequency and the transverse decay rate from an FID."""
    env = np.hypot(sx, sy)
    k = env > 1e-6 * env[0]
    Gamma_fit = -np.polyfit(t[k], np.log(env[k]), 1)[0]
    phase = np.unwrap(np.arctan2(sy, sx))
    omega_fit = abs(np.polyfit(t, phase, 1)[0])
    return omega_fit, Gamma_fit


print("Free induction decay of an alkali vapour "
      f"(gamma_eff/2pi = {gamma_eff / TWO_PI:.4e} Hz/T)")
head = (f"{'B0 (nT)':>10}{'Gamma2 (1/s)':>14}{'nu_L input (Hz)':>17}"
        f"{'nu_L fit':>12}{'Gamma2 fit':>12}")
print(head)
print("-" * len(head))
for B0_nT, G2 in [(1000.0, 100.0), (100.0, 100.0), (100.0, 10.0),
                  (10.0, 10.0)]:
    B0 = B0_nT * 1e-9
    omega0 = gamma_eff * B0
    t_end = min(8.0 / G2, 400.0 / max(omega0, 1e-9))
    t, S = free_induction(B0, G2, t_end)
    om_fit, G_fit = fit_fid(t, S[0], S[1])
    print(f"{B0_nT:>10.0f}{G2:>14.1f}{omega0 / TWO_PI:>17.4f}"
          f"{om_fit / TWO_PI:>12.4f}{G_fit:>12.4f}")

# --- A field gradient across the cell: T2* in the sense of Chapter 1 -------
# Atoms at different places in the cell see different fields, so the ensemble
# average is a sum of FIDs at slightly different Larmor frequencies. That is
# exactly the inhomogeneous dephasing of Chapter 1, and it produces a Gaussian
# envelope with T2* = sqrt(2)/sigma_omega rather than an exponential one.
B_mean, dB_spread, Gamma2_hom = 1000e-9, 20e-9, 5.0
n_sub = 41
offsets = np.linspace(-4.0, 4.0, n_sub) * dB_spread
weights = np.exp(-0.5 * (offsets / dB_spread) ** 2)
weights /= weights.sum()
t_end = 0.06
t_grid = np.linspace(0.0, t_end, 12001)
Sx_avg = np.zeros_like(t_grid)
Sy_avg = np.zeros_like(t_grid)
for w, off in zip(weights, offsets):
    _, S_i = free_induction(B_mean + off, Gamma2_hom, t_end,
                            n_pts=len(t_grid))
    Sx_avg += w * S_i[0]
    Sy_avg += w * S_i[1]
# The transverse magnitude is the coherence envelope, with the carrier removed.
env = np.hypot(Sx_avg, Sy_avg)
sigma_omega = gamma_eff * dB_spread
print(f"\nInhomogeneous dephasing from a {dB_spread * 1e9:.0f} nT spread "
      f"across the cell")
print(f"  homogeneous Gamma_2 = {Gamma2_hom:.1f} 1/s, so T2 = "
      f"{1.0 / Gamma2_hom * 1e3:.1f} ms")
print(f"  sigma_omega = gamma_eff dB = {sigma_omega:.1f} rad/s")
print(f"  predicted T2* = sqrt(2)/sigma_omega = "
      f"{np.sqrt(2.0) / sigma_omega * 1e3:.4f} ms")
model_g = 0.5 * np.exp(-Gamma2_hom * t_grid
                       - 0.5 * (sigma_omega * t_grid) ** 2)
model_e = 0.5 * np.exp(-Gamma2_hom * t_grid
                       - t_grid / (np.sqrt(2.0) / sigma_omega))
print(f"{'t (ms)':>9}{'envelope':>12}{'Gaussian model':>16}"
      f"{'exponential':>14}")
print("-" * 52)
for t_ms in [0.0, 1.0, 2.0, 3.0, 5.0]:
    k = int(np.argmin(np.abs(t_grid - t_ms * 1e-3)))
    print(f"{t_ms:>9.1f}{env[k]:>12.6f}{model_g[k]:>16.6f}"
          f"{model_e[k]:>14.6f}")
print("  The Gaussian column tracks the measured envelope; the exponential")
print("  one does not. A gradient is a T2* problem, and no amount of")
print("  optical power fixes it -- only shimming the field does.")

# --- Steady-state magnetometer response: the dispersive lineshape ----------
# The working geometry of a vapour-cell magnetometer: pump along z, measure
# the field along x, read the spin component along y that the field rotates
# out of the pumped direction.
print("\nSteady state under continuous pumping: the dispersive Sy signal")
R_op, Gamma_rel = 30.0, 60.0
Gamma_tot = Gamma_rel + R_op
print(f"  R_op = {R_op:.0f} 1/s, Gamma_rel = {Gamma_rel:.0f} 1/s, "
      f"total rate = {Gamma_tot:.0f} 1/s")
print(f"{'Bx (nT)':>10}{'x = w0/Gtot':>14}{'Sy':>12}{'Sy analytic':>14}"
      f"{'Sz':>11}")
print("-" * 61)
S_sat = R_op / (2.0 * Gamma_tot)
for Bx_nT in [-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0]:
    Bx = Bx_nT * 1e-9
    sol = solve_ivp(spin_rhs, (0.0, 3.0), [0.0, 0.0, 0.0],
                    args=((Bx, 0.0, 0.0), Gamma_rel, Gamma_rel, R_op,
                          (0.0, 0.0, 1.0)),
                    rtol=1e-11, atol=1e-14, method="DOP853")
    Sx, Sy, Sz = sol.y[:, -1]
    x = gamma_eff * Bx / Gamma_tot
    print(f"{Bx_nT:>10.1f}{x:>14.4f}{Sy:>12.6f}"
          f"{S_sat * x / (1.0 + x ** 2):>14.6f}{Sz:>11.6f}")
slope_num = (gamma_eff / Gamma_tot) * S_sat
print(f"  slope through zero field dSy/dBx = {slope_num * 1e-9:.4e} per nT")
print("  The half-width in field units is Gamma_tot/gamma_eff = "
      f"{Gamma_tot / gamma_eff * 1e9:.3f} nT:")
print("  narrower relaxation means a steeper transfer function AND a "
      "narrower")
print("  dynamic range. That trade is the subject of Example 6.")

fig, ax = plt.subplots(figsize=(6.2, 4.0))
for B0_nT, G2 in [(1000.0, 100.0), (1000.0, 20.0)]:
    t, S = free_induction(B0_nT * 1e-9, G2, 0.25)
    ax.plot(t * 1e3, S[0], lw=0.7,
            label=f"B0 = {B0_nT:.0f} nT, $\\Gamma_2$ = {G2:.0f} s$^{{-1}}$")
ax.set_xlabel("time (ms)")
ax.set_ylabel("$S_x$")
ax.set_title("Vapour-cell free induction decay")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

```text
Free induction decay of an alkali vapour (gamma_eff/2pi = 4.6708e+09 Hz/T)
   B0 (nT)  Gamma2 (1/s)  nu_L input (Hz)    nu_L fit  Gamma2 fit
-----------------------------------------------------------------
      1000         100.0        4670.8252   4670.8252    100.0000
       100         100.0         467.0825    467.0825    100.0000
       100          10.0         467.0825    467.0825     10.0000
        10          10.0          46.7083     46.7083     10.0000

Inhomogeneous dephasing from a 20 nT spread across the cell
  homogeneous Gamma_2 = 5.0 1/s, so T2 = 200.0 ms
  sigma_omega = gamma_eff dB = 587.0 rad/s
  predicted T2* = sqrt(2)/sigma_omega = 2.4094 ms
   t (ms)    envelope  Gaussian model   exponential
----------------------------------------------------
      0.0    0.500000        0.500000      0.500000
      1.0    0.418815        0.418782      0.328511
      2.0    0.248537        0.248534      0.215839
      3.0    0.104511        0.104511      0.141811
      5.0    0.006558        0.006574      0.061216
  The Gaussian column tracks the measured envelope; the exponential
  one does not. A gradient is a T2* problem, and no amount of
  optical power fixes it -- only shimming the field does.

Steady state under continuous pumping: the dispersive Sy signal
  R_op = 30 1/s, Gamma_rel = 60 1/s, total rate = 90 1/s
   Bx (nT)   x = w0/Gtot          Sy   Sy analytic         Sz
-------------------------------------------------------------
     -20.0       -6.5217   -0.024969     -0.024969   0.003829
      -5.0       -1.6304   -0.074280     -0.074280   0.045559
      -1.0       -0.3261   -0.049124     -0.049124   0.150648
       0.0        0.0000    0.000000      0.000000   0.166667
       1.0        0.3261    0.049124      0.049124   0.150648
       5.0        1.6304    0.074280      0.074280   0.045559
      20.0        6.5217    0.024969      0.024969   0.003829
  slope through zero field dSy/dBx = 5.4348e-02 per nT
  The half-width in field units is Gamma_tot/gamma_eff = 3.067 nT:
  narrower relaxation means a steeper transfer function AND a narrower
  dynamic range. That trade is the subject of Example 6.
```

**What to look for.** The first table is a calibration of the integrator against the physics it is supposed to contain: the fitted Larmor frequency and transverse decay rate return the inputs to four decimal places, so the solver can be trusted for the less obvious cases below. The useful number in it is $\gamma_\mathrm{eff}/2\pi = 4.67$ Hz/nT — an alkali spin precesses at kilohertz in the tens-of-nanotesla fields where such a magnetometer works, and megahertz in the Earth's field.

The second block is where Chapter 1's vocabulary earns its keep. A 20 nT spread of field across the cell — a gradient, not a fluctuation — makes different atoms precess at different rates, and the ensemble average decays as a **Gaussian** with $T_2^\ast = \sqrt{2}/\sigma_\omega = 2.409$ ms even though every individual atom stays coherent for the full $T_2 = 200$ ms. The measured envelope, $0.418815$ at 1 ms, matches the Gaussian prediction $0.418782$ to five digits and is nowhere near the exponential $0.328511$. That is the diagnostic of §1.4 used in reverse: the *shape* of the decay says whether the problem is a fluctuation or a static inhomogeneity, and here it says the fix is a better magnetic shield rather than a better cell coating. Two orders of magnitude of sensitivity are lost to a gradient in this example, and no amount of laser power recovers them.

The third block computes the actual transfer function. With the pump along $z$ and the field to be measured along $x$, the steady-state $S_y$ is the dispersive Lorentzian $S_\mathrm{sat}\, x/(1+x^2)$ with $x = \omega_0/\Gamma_\mathrm{tot}$ — verified against the closed form to six digits — and its slope through zero field, $5.43\times10^{-2}$ per nT, is what a magnetometer actually reads. The half-width, $\Gamma_\mathrm{tot}/\gamma_\mathrm{eff} = 3.067$ nT, is the linear range. Note the structure: making the relaxation slower makes the slope steeper *and* the linear range narrower, in exact proportion. That is the same sensitivity-against-dynamic-range trade as the interferometer fringe of §4.3, and it sets up the next example.

### Spin exchange, and the regime where it stops mattering

The dominant relaxation in a dense alkali vapour is **spin-exchange collision**: two alkali atoms collide, their electron spins swap, and total spin is conserved but each atom's hyperfine state is randomized. In the ordinary regime, where the Larmor precession is fast compared with the collision rate $R_\mathrm{se}$, this randomizes the phase and contributes about $R_\mathrm{se}/2$ to the transverse relaxation. Since $R_\mathrm{se}$ grows with density and density is what gives the sensor its atoms, this looks like a hard ceiling.

The escape is that spin exchange conserves total spin. If the collisions are much *faster* than the precession, the atoms exchange spin many times before any relative phase can develop, the ensemble precesses as a single collective spin, and the spin-exchange contribution is suppressed as $(\omega_0/R_\mathrm{se})^2$. This is the **spin-exchange relaxation-free (SERF)** regime, and reaching it requires a small field, because the condition is $\omega_0 \ll R_\mathrm{se}$. The suppression is remarkable and it comes with an immediate cost: a magnetometer that must sit in a nanotesla field needs magnetic shielding, cannot measure the field it is shielded from, and — because the relaxation rate it has worked so hard to reduce also sets its bandwidth — is slow.

### Code Example 6: The SERF Crossover, Priced

```python
"""Chapter 4, Example 6: the SERF crossover, and the sensitivity-bandwidth
trade it forces.
Continues from Examples 1-5 (same session)."""

kB = 1.380649e-23


def gamma_spin_exchange(omega0, R_se):
    """Transverse relaxation from spin exchange, interpolated across the
    crossover.

    The two limits are the physics: for omega0 >> R_se the Zeeman levels are
    resolved and every exchange collision randomizes the phase, giving
    R_se/2; for omega0 << R_se the collisions are too fast for the spins to
    dephase between them and the contribution is suppressed as
    (omega0/R_se)^2. The Lorentzian interpolation below reproduces both ends
    with the correct powers; the crossover region is only qualitative.
    """
    x = (omega0 / R_se) ** 2
    return 0.5 * R_se * x / (1.0 + x)


def cell_rates(B0, R_se, Gamma_sd, Gamma_wall, R_op):
    """Total transverse relaxation rate of an alkali vapour, 1/s."""
    omega0 = gamma_eff * B0
    return (gamma_spin_exchange(omega0, R_se) + Gamma_sd + Gamma_wall + R_op)


def dB_sensitivity(Gamma2, n_density, volume, gamma=gamma_eff):
    """Spin-projection-noise-limited field sensitivity, T/sqrt(Hz).

    delta_B = (1/gamma) sqrt(Gamma2 / (n V)), the standard result for an
    ensemble of n*V uncorrelated spins each coherent for 1/Gamma2.
    """
    return np.sqrt(Gamma2 / (n_density * volume)) / gamma


n_alkali = 1.0e20        # atoms/m^3, a hot cell near 150 C
V_cell = (3e-3) ** 3     # 3 mm cube
R_se = 1.0e5             # 1/s spin-exchange rate at that density
Gamma_sd = 30.0          # 1/s spin destruction
Gamma_wall = 20.0        # 1/s wall and diffusion losses
R_op_bg = 30.0           # 1/s optical pumping

print("Spin-exchange relaxation across the SERF crossover")
print(f"  R_se = {R_se:.0e} 1/s, so the crossover sits at "
      f"B0 = {R_se / gamma_eff * 1e9:.1f} nT")
head = (f"{'B0 (nT)':>10}{'nu_L (Hz)':>12}{'G_SE (1/s)':>12}"
        f"{'G_2 (1/s)':>11}{'BW (Hz)':>9}{'eta_B (fT/rtHz)':>17}")
print(head)
print("-" * len(head))
for B0_nT in [10_000.0, 3000.0, 1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 1.0]:
    B0 = B0_nT * 1e-9
    G_se = gamma_spin_exchange(gamma_eff * B0, R_se)
    G2 = cell_rates(B0, R_se, Gamma_sd, Gamma_wall, R_op_bg)
    eta = dB_sensitivity(G2, n_alkali, V_cell)
    print(f"{B0_nT:>10.0f}{gamma_eff * B0 / TWO_PI:>12.2f}{G_se:>12.2f}"
          f"{G2:>11.2f}{G2 / TWO_PI:>9.2f}{eta * 1e15:>17.3f}")

print("\nThe trade, stated plainly:")
B_hi, B_lo = 3000e-9, 3e-9
G_hi = cell_rates(B_hi, R_se, Gamma_sd, Gamma_wall, R_op_bg)
G_lo = cell_rates(B_lo, R_se, Gamma_sd, Gamma_wall, R_op_bg)
print(f"  at B0 = 3000 nT: Gamma_2 = {G_hi:8.1f} 1/s, "
      f"bandwidth {G_hi / TWO_PI:7.1f} Hz, "
      f"eta = {dB_sensitivity(G_hi, n_alkali, V_cell) * 1e15:7.2f} fT/rtHz")
print(f"  at B0 =    3 nT: Gamma_2 = {G_lo:8.1f} 1/s, "
      f"bandwidth {G_lo / TWO_PI:7.1f} Hz, "
      f"eta = {dB_sensitivity(G_lo, n_alkali, V_cell) * 1e15:7.2f} fT/rtHz")
print(f"  a factor {np.sqrt(G_hi / G_lo):.1f} in sensitivity, paid for with a "
      f"factor {G_hi / G_lo:.1f} in bandwidth.")

# --- Density is not a free parameter either --------------------------------
print("\nRaising the density raises the collision rates too, so the gain "
      "saturates:")
print(f"{'n (1/m^3)':>12}{'R_se (1/s)':>12}{'G_SE at 3 nT':>14}"
      f"{'G_2 (1/s)':>11}{'eta (fT/rtHz)':>15}")
print("-" * 64)
for n_i in [1e19, 3e19, 1e20, 3e20, 1e21, 3e21, 1e22]:
    R_se_i = 1.0e5 * (n_i / 1.0e20)
    G_sd_i = 30.0 * (n_i / 1.0e20)          # spin destruction also scales with n
    G2_i = cell_rates(B_lo, R_se_i, G_sd_i, Gamma_wall, R_op_bg)
    eta_i = dB_sensitivity(G2_i, n_i, V_cell)
    print(f"{n_i:>12.0e}{R_se_i:>12.0e}"
          f"{gamma_spin_exchange(gamma_eff * B_lo, R_se_i):>14.2e}"
          f"{G2_i:>11.1f}{eta_i * 1e15:>15.3f}")
# Once spin destruction dominates, Gamma_2 is proportional to n and eta stops
# improving: the asymptote is set by the per-atom spin-destruction cross
# section alone, not by how many atoms are in the cell.
eta_inf = np.sqrt(30.0 / 1e20 / V_cell) / gamma_eff
print(f"  spin-destruction asymptote sqrt(G_sd/n / V)/gamma = "
      f"{eta_inf * 1e15:.3f} fT/rtHz")
print("  Once Gamma_2 grows in proportion to n, more atoms buy nothing.")

# --- Volume scaling: the trade against spatial resolution ------------------
print("\nSensitivity against sensor size, at fixed density and Gamma_2:")
print(f"{'cell edge a':>13}{'volume (m^3)':>15}{'eta_B (fT/rtHz)':>18}"
      f"{'eta * a^1.5 (a.u.)':>21}")
print("-" * 67)
G2_fixed = cell_rates(B_lo, R_se, Gamma_sd, Gamma_wall, R_op_bg)
for a_mm in [10.0, 3.0, 1.0, 0.3, 0.1]:
    a = a_mm * 1e-3
    eta_a = dB_sensitivity(G2_fixed, n_alkali, a ** 3)
    print(f"{a_mm:>10.1f} mm{a ** 3:>15.2e}{eta_a * 1e15:>18.3f}"
          f"{eta_a * a ** 1.5 * 1e15:>21.4e}")
print("  eta_B scales as a^(-3/2): every decade of spatial resolution costs")
print("  a factor 31.6 in field sensitivity. Chapter 5 returns to this,")
print("  because for a localized source the field grows faster than that.")

print("\nNo cryogenics anywhere in this example:")
h_planck = 6.62607015e-34
eV = 1.602176634e-19
kT = kB * 420.0
hnu = h_planck * gamma_eff * B_lo / TWO_PI
print(f"  cell temperature ~ 420 K, thermal energy kT = {kT / eV * 1e3:.2f} meV")
print(f"  Zeeman splitting at 3 nT: h nu = {hnu / eV * 1e15:.3f} feV")
print(f"  h nu / kT = {hnu / kT:.3e}, so the thermal polarization is "
      f"utterly negligible")
print("  -- and it does not matter, because the polarization comes from")
print("  optical pumping, not from kT. Compare the SQUID of Chapter 3, whose")
print("  operating temperature is set by a superconducting gap.")
```

```text
Spin-exchange relaxation across the SERF crossover
  R_se = 1e+05 1/s, so the crossover sits at B0 = 3407.4 nT
   B0 (nT)   nu_L (Hz)  G_SE (1/s)  G_2 (1/s)  BW (Hz)  eta_B (fT/rtHz)
-----------------------------------------------------------------------
     10000    46708.25    44798.63   44878.63  7142.66            4.393
      3000    14012.48    21833.47   21913.47  3487.64            3.070
      1000     4670.83     3964.93    4044.93   643.77            1.319
       300     1401.25      384.60     464.60    73.94            0.447
       100      467.08       43.03     123.03    19.58            0.230
        30      140.12        3.88      83.88    13.35            0.190
        10       46.71        0.43      80.43    12.80            0.186
         3       14.01        0.04      80.04    12.74            0.186
         1        4.67        0.00      80.00    12.73            0.185

The trade, stated plainly:
  at B0 = 3000 nT: Gamma_2 =  21913.5 1/s, bandwidth  3487.6 Hz, eta =    3.07 fT/rtHz
  at B0 =    3 nT: Gamma_2 =     80.0 1/s, bandwidth    12.7 Hz, eta =    0.19 fT/rtHz
  a factor 16.5 in sensitivity, paid for with a factor 273.8 in bandwidth.

Raising the density raises the collision rates too, so the gain saturates:
   n (1/m^3)  R_se (1/s)  G_SE at 3 nT  G_2 (1/s)  eta (fT/rtHz)
----------------------------------------------------------------
       1e+19       1e+04      3.88e-01       53.4          0.479
       3e+19       3e+04      1.29e-01       59.1          0.291
       1e+20       1e+05      3.88e-02       80.0          0.186
       3e+20       3e+05      1.29e-02      140.0          0.142
       1e+21       1e+06      3.88e-03      350.0          0.123
       3e+21       3e+06      1.29e-03      950.0          0.117
       1e+22       1e+07      3.88e-04     3050.0          0.115
  spin-destruction asymptote sqrt(G_sd/n / V)/gamma = 0.114 fT/rtHz
  Once Gamma_2 grows in proportion to n, more atoms buy nothing.

Sensitivity against sensor size, at fixed density and Gamma_2:
  cell edge a   volume (m^3)   eta_B (fT/rtHz)   eta * a^1.5 (a.u.)
-------------------------------------------------------------------
      10.0 mm       1.00e-06             0.030           3.0484e-05
       3.0 mm       2.70e-08             0.186           3.0484e-05
       1.0 mm       1.00e-09             0.964           3.0484e-05
       0.3 mm       2.70e-11             5.867           3.0484e-05
       0.1 mm       1.00e-12            30.484           3.0484e-05
  eta_B scales as a^(-3/2): every decade of spatial resolution costs
  a factor 31.6 in field sensitivity. Chapter 5 returns to this,
  because for a localized source the field grows faster than that.

No cryogenics anywhere in this example:
  cell temperature ~ 420 K, thermal energy kT = 36.19 meV
  Zeeman splitting at 3 nT: h nu = 57.951 feV
  h nu / kT = 1.601e-12, so the thermal polarization is utterly negligible
  -- and it does not matter, because the polarization comes from
  optical pumping, not from kT. Compare the SQUID of Chapter 3, whose
  operating temperature is set by a superconducting gap.
```

**What to look for.** The crossover sits where $\omega_0 = R_\mathrm{se}$, at $3407$ nT for this cell, and the table walks across it. At $B_0 = 3000$ nT the spin-exchange term is $2.18\times10^{4}$ s$^{-1}$ and dominates everything; by $B_0 = 30$ nT it has fallen to $3.88$ s$^{-1}$ and the residual relaxation is spin destruction, wall collisions and the pump light. The field sensitivity improves from $3.07$ to $0.19$ fT$/\sqrt{\mathrm{Hz}}$ — a factor $16.5$ — and the bandwidth falls from $3488$ Hz to $12.7$ Hz, a factor $274$. That is the SERF trade, and it is now a number rather than an adjective: **a factor of 16 in sensitivity costs a factor of 274 in bandwidth**, because sensitivity goes as $\sqrt{\Gamma_2}$ and bandwidth goes as $\Gamma_2$.

The density scan corrects a natural but wrong intuition. More atoms should mean better sensitivity as $1/\sqrt{n}$, but the collision rates grow with $n$ too. In the SERF regime spin exchange is suppressed, but spin *destruction* is not, and once $\Gamma_2 \propto n$ the sensitivity $\sqrt{\Gamma_2/n}$ stops improving: the scan flattens at $0.115$ fT$/\sqrt{\mathrm{Hz}}$ against the computed spin-destruction asymptote of $0.114$. The lever that remains is the per-atom spin-destruction cross-section, which is chemistry — buffer gas, cell coating, which alkali — not physics.

The volume scan is the number to carry into Chapter 5. Sensitivity scales as $a^{-3/2}$ in the linear size of the cell, so every decade of spatial resolution costs a factor $31.6$ in field sensitivity, and the product $\eta_B a^{3/2}$ is constant to five digits down the column. A 3 mm cell at $0.186$ fT$/\sqrt{\mathrm{Hz}}$ becomes a 100 µm cell at $30.5$ fT$/\sqrt{\mathrm{Hz}}$. Whether that is a bad trade depends entirely on the source: §5.4 shows that for a localized source the available field grows as $a^{-3}$, faster than the sensitivity degrades, so the small probe wins — and for a uniform field it does not.

The closing block states the thermodynamic point plainly. The Zeeman energy at 3 nT is $1.6\times10^{-12}$ of $k_BT$ in the cell, so the thermal polarization is nothing at all, and the sensor works anyway because optical pumping does not care. Compare the SQUID of [Chapter 3](<chapter-3.html>), whose operating temperature is not a design choice but a consequence of needing a superconducting gap. Two femtotesla magnetometers, two entirely different reasons for their operating temperature.

### Where the three magnetometers of this course sit

| | Vapour cell (SERF) | SQUID | NV centre |
| --- | --- | --- | --- |
| Physical quantity coupled to | alkali electron spin | magnetic flux through a loop | electron spin of a lattice defect |
| Operating temperature | ~420 K, no cryogenics | below $T_c$ of the film | 4 K to 600 K |
| Sensing volume | mm$^3$ | loop area, µm$^2$ upwards | nm$^3$ (single) to mm$^3$ (ensemble) |
| Standoff achievable | mm | µm | nm |
| Bandwidth limit | relaxation rate, tens of Hz in SERF | amplifier and loop, MHz | $1/T_2$ or drive, MHz to GHz |
| Dominant noise at best | spin projection, spin destruction | $1/f$ flux noise from surface spins | spin projection, surface spin bath |
| The materials question | cell coating, buffer gas, alkali choice | film surface, junction oxide | diamond surface termination, N-to-NV yield |
| Vector or scalar | either, by geometry | flux, hence one component | vector, from the N-V axis |

Reading the last row but one: all three are limited by a surface. The cell wall, the superconducting film's oxide, the diamond's terminated surface. That is the same conclusion the hardware course reaches for qubits, and it is the reason this course sits in a materials-science dojo.

* * *

## 4.5 What This Chapter Adds

Three transferable items, stated compactly.

**The error budget is the deliverable.** A sensitivity figure is one line of a budget. The rest of the budget decides whether the instrument is useful, and the discipline of writing it — every term with a functional form, an estimated size and a bounded uncertainty — is what Example 3 demonstrates. Applied to a scanning NV microscope or a SQUID susceptometer, the same table has different entries and the same structure.

**Stability and accuracy are different numbers.** Averaging reduces one and not the other, and the crossover time is computable. Reporting a sensitivity without the averaging time at which the systematic floor arrives is reporting half a result.

**Free atoms remove the materials problem and reveal what is under it.** A vapour cell has no lattice, no interface and no defect ensemble in its sensing volume, and its limits are collision cross-sections, wall coatings and optical pumping efficiency. This is the same lesson the hardware course draws from neutral atoms: removing the solid does not remove the limit, it relocates it. Chapter 5 asks whether entanglement can move it again.

* * *

## Exercises

#### Exercise 1: Clock Stability Arithmetic

A Ramsey clock operates on a transition at $\nu_0 = 4.0\times10^{14}$ Hz with a free evolution time $T = 0.4$ s, $N = 2500$ atoms, and a cycle time $T_c = 1.0$ s.

  1. Compute $Q$, the Ramsey linewidth, and $\sigma_y(1\ \mathrm{s})$.
  2. How long must the clock average to reach a fractional uncertainty of $1\times10^{-17}$?
  3. Which is worth more, doubling $T$ or quadrupling $N$? Give both factors and state which is physically easier.
  4. The cycle time is reduced from $1.0$ s to $0.5$ s with $T$ unchanged. By what factor does $\sigma_y(\tau)$ improve, and why is this often the cheapest available gain?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(Q = 2\nu_0 T = 2 \times 4.0\times10^{14} \times 0.4 = 3.2\times10^{14}\). The Ramsey linewidth is \(\Delta\nu = 1/2T = 1.25\) Hz. Then \(\sigma_y(1\ \mathrm{s}) = 1/(\pi Q \sqrt{N}) \times \sqrt{T_c/\tau} = 1/(\pi \times 3.2\times10^{14} \times 50) = 1.989\times10^{-17}\).</p>

<p><strong>2.</strong> Since \(\sigma_y \propto \tau^{-1/2}\), \(\tau = (1.989\times10^{-17}/1\times10^{-17})^2 = 3.96\) s. Reaching \(10^{-17}\) takes four seconds of averaging, which is the point of Example 3: the statistical part of this problem is easy.</p>

<p><strong>3.</strong> Doubling \(T\) doubles \(Q\) and therefore gains a factor 2; quadrupling \(N\) gains \(\sqrt{4} = 2\) as well. They are worth exactly the same. Physically, doubling \(T\) is bounded by \(T_2\) and by the coherence of the interrogation laser, and beyond a second it is usually the laser; quadrupling \(N\) is bounded by the cold-collision density shift, which enters the accuracy budget rather than the stability. Neither is free, and which is easier depends on which budget has room.</p>

<p><strong>4.</strong> \(\sigma_y(\tau) \propto \sqrt{T_c}\), so halving \(T_c\) improves the stability by \(\sqrt{2} = 1.41\). It is often cheapest because dead time is preparation and read-out, i.e. engineering rather than physics, and because it also reduces the Dick-effect aliasing of local-oscillator noise, which shrinks a second term at the same time.</p>

```python
import numpy as np
nu0, T, N, Tc = 4.0e14, 0.4, 2500.0, 1.0
Q = 2 * nu0 * T
s1 = 1.0 / (np.pi * Q * np.sqrt(N)) * np.sqrt(Tc / 1.0)
print(f"Q = {Q:.3e}, linewidth = {1/(2*T):.2f} Hz, sigma_y(1 s) = {s1:.4e}")
print(f"tau to 1e-17 = {(s1 / 1e-17)**2:.2f} s")
print(f"halving Tc gains {np.sqrt(2):.3f}")
# Q = 3.200e+14, linewidth = 1.25 Hz, sigma_y(1 s) = 1.9894e-17
# tau to 1e-17 = 3.96 s
# halving Tc gains 1.414
```

</details>

#### Exercise 2: Reading an Allan Plot

A clock's Allan deviation is measured as $\sigma_y(1\ \mathrm{s}) = 2\times10^{-14}$, falling as $\tau^{-1/2}$ until $\tau = 100$ s, then flat at $2\times10^{-15}$ until $\tau = 10^{4}$ s, then rising as $\tau^{+1}$.

  1. Name the dominant noise process in each of the three regions.
  2. What is the best achievable stability, and at what averaging time?
  3. A colleague reports "$\sigma_y = 4\times10^{-16}$" for this clock. Under what measurement could that be true, and why is the claim incomplete?
  4. The flat region is traced to the interrogation laser. Does that make it a stability problem or an accuracy problem, and what does the answer imply about where to spend effort?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> White frequency noise (slope \(-1/2\)), then flicker frequency noise (slope 0), then linear frequency drift (slope \(+1\)). The middle region is a floor, not a decay.</p>

<p><strong>2.</strong> The best stability is the floor, \(2\times10^{-15}\), first reached at \(\tau = 100\) s and held until \(10^{4}\) s. Averaging beyond \(10^{4}\) s makes the result worse, so the useful operating point is anywhere in that decade-and-a-half plateau.</p>

<p><strong>3.</strong> It cannot be true for this clock at any \(\tau\): the minimum of the curve is \(2\times10^{-15}\). A quoted stability without its averaging time is meaningless, and the most common way such a number appears is by extrapolating the \(\tau^{-1/2}\) region past the point where it stops applying — here, \(2\times10^{-14}/\sqrt{2500} = 4\times10^{-16}\) at \(\tau = 2500\) s, a region where the real curve is already rising.</p>

<p><strong>4.</strong> It is a stability problem: a fluctuating local oscillator moves the clock's output but does not bias its mean, so it does not enter the accuracy budget. The effort therefore goes to the cavity — thermal-noise-limited coating loss, spacer material, temperature control — rather than to the atoms. This is the Dick-effect route by which a laser limits an atomic clock, and it is the reason reference-cavity materials are an active subject.</p>

</details>

#### Exercise 3: A Budget Decision

An optical clock has the following uncertainty contributions: blackbody radiation $8\times10^{-17}$ (enclosure at 300 K, temperature known to 1 K), lattice light shift $2\times10^{-17}$, second-order Zeeman $1\times10^{-17}$, density shift $5\times10^{-18}$, redshift $1\times10^{-18}$.

  1. Compute the total in quadrature.
  2. You may either cool the enclosure to 150 K or improve the enclosure temperature knowledge to $0.1$ K. Compute the new total for each, and choose.
  3. After your chosen fix, which term dominates, and by what factor would it have to improve to matter less than the next one?
  4. A reviewer asks why the redshift term is not simply removed by putting the clock in the basement. Answer.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\sqrt{80^2 + 20^2 + 10^2 + 5^2 + 1^2}\times10^{-18} = \sqrt{6400 + 400 + 100 + 25 + 1}\times10^{-18} = 83.4\times10^{-18} = 8.34\times10^{-17}\).</p>

<p><strong>2.</strong> Cooling to 150 K scales the BBR uncertainty by \((150/300)^3 = 1/8\), giving \(1\times10^{-17}\); the total becomes \(\sqrt{10^2+20^2+10^2+5^2+1^2} = 24.6\times10^{-18} = 2.46\times10^{-17}\). Improving the temperature knowledge tenfold scales the BBR uncertainty by 10, to \(8\times10^{-18}\); the total becomes \(\sqrt{8^2+20^2+10^2+5^2+1^2} = 23.9\times10^{-18} = 2.39\times10^{-17}\). The two are equivalent to within the rounding, so the choice is made on cost and on side effects: a cryogenic enclosure also reduces the ac Stark contribution from stray thermal light and the collision rate with background gas, while a better thermometer does neither. Cooling is the better engineering answer even though the arithmetic is a tie.</p>

<p><strong>3.</strong> The lattice light shift, at \(2\times10^{-17}\), now dominates both options. To fall below the next largest term (\(1\times10^{-17}\), the Zeeman) it must improve by a factor greater than 2, which means stabilizing the lattice wavelength or operating at a lower trap depth — the latter at the cost of atom number, hence of stability. Budgets couple.</p>

<p><strong>4.</strong> Because the redshift is not a nuisance to be removed but a real, well-understood frequency difference: a clock lower down genuinely runs slower, at \(1.09\times10^{-16}\) per metre. The uncertainty in the term is the uncertainty in the clock's <em>height above the reference geoid</em>, and that is a geodetic measurement, not a physics one. Moving to the basement does not reduce it; surveying the basement does. This is also why clock comparisons are proposed as a geodetic tool in their own right.</p>

</details>

#### Exercise 4: Sizing an Atom Interferometer

A Mach-Zehnder gravimeter uses $k_\mathrm{eff} = 1.61\times10^{7}$ m$^{-1}$ and $N = 10^{6}$ atoms per shot.

  1. What $T$ is needed for a single-shot uncertainty of $1\times10^{-9}\,g$?
  2. What fountain apex height does that $T$ imply, and what is the arm separation at the apex if the recoil velocity is $11.8$ mm/s?
  3. With that $T$, what acceleration corresponds to one fringe? If the local $g$ is known beforehand to $1\times10^{-5}\,g$, is the measurement unambiguous?
  4. Vibration of the retro-reflecting mirror at $10^{-6}$ m s$^{-2}$ rms is present. What does it do to the measurement, and what is the standard remedy?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\delta a = 1/(\sqrt{N} k_\mathrm{eff} T^2)\), so \(T = \left[1/(\sqrt{N} k_\mathrm{eff}\, \delta a)\right]^{1/2}\) with \(\delta a = 10^{-9} \times 9.807 = 9.807\times10^{-9}\) m s\(^{-2}\). Then \(T^2 = 1/(10^3 \times 1.61\times10^7 \times 9.807\times10^{-9}) = 6.33\times10^{-3}\) s\(^2\), so \(T = 79.6\) ms.</p>

<p><strong>2.</strong> Apex \(h = gT^2/2 = 9.807 \times 6.33\times10^{-3}/2 = 3.1\) cm above the launch point — modest, which is why nano-\(g\) gravimeters fit on a table. Arm separation at the apex is \(v_\mathrm{rec} T = 11.8\ \mathrm{mm/s} \times 0.0796\ \mathrm{s} = 0.94\) mm.</p>

<p><strong>3.</strong> One fringe is \(\delta a_\mathrm{fringe} = 2\pi/(k_\mathrm{eff}T^2) = 2\pi/(1.61\times10^7 \times 6.33\times10^{-3}) = 6.16\times10^{-5}\) m s\(^{-2}\), i.e. \(6.3\times10^{-6}\,g\). Prior knowledge to \(10^{-5}\,g\) spans more than one fringe, so the measurement is <em>not</em> unambiguous: one must either improve the prior (a mechanical gravimeter, or a short-\(T\) run first) or scan \(T\) and find the phase consistent across values.</p>

<p><strong>4.</strong> Mirror vibration is indistinguishable from acceleration of the atoms, because the interferometer measures their relative acceleration; \(10^{-6}\) m s\(^{-2}\) is a hundred times the single-shot noise, so it dominates completely and randomizes the fringe shot to shot. The remedies are all differential: a seismometer on the mirror whose signal is subtracted from the measured phase, passive isolation, or — most robustly — a gradiometer, two interferometers sharing the same laser and mirror, whose <em>difference</em> is immune to common platform motion. That last option is why gradiometry is easier than absolute gravimetry despite measuring a smaller quantity.</p>

```python
import numpy as np
keff, N, g = 1.61e7, 1e6, 9.80665
da = 1e-9 * g
T = (1.0 / (np.sqrt(N) * keff * da)) ** 0.5
print(f"T = {T*1e3:.1f} ms, apex = {0.5*g*T**2*100:.1f} cm, "
      f"separation = {11.8e-3*T*1e3:.2f} mm")
print(f"one fringe = {2*np.pi/(keff*T**2):.3e} m/s^2 = "
      f"{2*np.pi/(keff*T**2)/g:.2e} g")
# T = 79.6 ms, apex = 3.1 cm, separation = 0.94 mm
# one fringe = 6.162e-05 m/s^2 = 6.28e-06 g
```

</details>

#### Exercise 5: Choosing Between a SERF Cell and a SQUID

A measurement requires detecting a $50$ fT signal at $200$ Hz, from a source $2$ mm below the surface of a sample held at $300$ K.

  1. From Example 6, what bandwidth does a SERF cell have when it is operated deep in the SERF regime, and can it see a 200 Hz signal?
  2. What options exist for raising the bandwidth, and what does each cost?
  3. The SQUID of Chapter 3 has ample bandwidth. What does it cost instead, given that the sample is at 300 K?
  4. Suppose the source is $2$ µm below the surface instead of $2$ mm, and the required spatial resolution is $1$ µm. Which of the three sensors of this course survives, and why is its worse field sensitivity not disqualifying?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Deep in the SERF regime the total transverse rate in Example 6 is about \(80\ \mathrm{s^{-1}}\), i.e. a bandwidth of \(12.7\) Hz. A 200 Hz signal is a factor 16 above that and would be attenuated by roughly \(200/12.7 \approx 16\) in a single-pole response — so no, not as configured.</p>

<p><strong>2.</strong> Three options, all of them trades. (i) Increase the pump rate \(R_\mathrm{op}\): bandwidth grows linearly with \(\Gamma_2\) and sensitivity degrades as \(\sqrt{\Gamma_2}\), so reaching 200 Hz costs a factor \(\sqrt{16} = 4\) in \(\eta_B\) — from \(0.19\) to about \(0.75\) fT\(/\sqrt{\mathrm{Hz}}\), still well under 50 fT. (ii) Leave the SERF regime and operate at a larger field, which also broadens the response but adds the spin-exchange term. (iii) Use a closed-loop configuration, which extends the <em>usable</em> bandwidth without changing \(\Gamma_2\), at the cost of loop complexity. Option (i) is sufficient here and is the honest answer: the sensitivity margin is large enough to spend.</p>

<p><strong>3.</strong> The SQUID must be below the critical temperature of its film while the sample is at 300 K, so the two are separated by a vacuum gap and a window. That standoff is the cost: with a 2 mm source depth plus a cryostat window the total standoff may reach several millimetres, and for a localized source the field falls as \(1/d^3\). A femtotesla sensor at 5 mm can be worse than a picotesla sensor at 0.5 mm. Room-temperature operation is not a convenience, it is a standoff budget.</p>

<p><strong>4.</strong> Only the NV centre. Its sensing volume can be nanometres and it works at 300 K in contact with the sample, so a standoff of order 1 µm or less is achievable. Its field sensitivity is orders of magnitude worse than either alternative, and that does not disqualify it because the field from a localized source grows as \(1/d^3\) while the sensitivity of a shrinking probe degrades only as \(a^{-3/2}\) (Example 6). Reducing the standoff from 5 mm to 1 µm is a factor \(1.25\times10^{11}\) in field; no sensitivity ratio in this course is that large. Chapter 5 §5.4 does this comparison quantitatively.</p>

```python
import numpy as np
G2_serf, eta_serf = 80.0, 0.186e-15
needed_bw = 200.0
G2_needed = 2 * np.pi * needed_bw
print(f"bandwidth as built  {G2_serf/(2*np.pi):.1f} Hz")
print(f"Gamma_2 needed      {G2_needed:.0f} 1/s, factor "
      f"{G2_needed/G2_serf:.1f}")
print(f"eta after broadening {eta_serf*np.sqrt(G2_needed/G2_serf)*1e15:.2f} "
      f"fT/rtHz vs 50 fT signal")
# bandwidth as built  12.7 Hz
# Gamma_2 needed      1257 1/s, factor 15.7
# eta after broadening 0.74 fT/rtHz vs 50 fT signal
```

</details>

* * *

## Summary

### Key Takeaways

**1\. A clock is the Ramsey estimator in a feedback loop**

  * The fringe is a frequency discriminator with slope $T/2$ at the half-fringe bias point, so a clock is deliberately operated off resonance.
  * Projection noise gives $\sigma_y(\tau) = 1/(\pi Q\sqrt{N})\sqrt{T_c/\tau}$ with $Q = 2\nu_0 T$; the Monte-Carlo check in Example 1 reproduces the $1/(T\sqrt{N})$ term to $0.2\%$.
  * Optical clocks win on $Q$ alone: $5.244\times10^{-18}$ against $3.463\times10^{-14}$ at one second, despite a hundredfold disadvantage in $\sqrt{N}$.

**2\. The Allan slope names the noise**

  * Measured slopes $-0.4995$, $-0.0016$, $+0.4995$, $+1.0000$ for white, flicker, random-walk frequency noise and drift.
  * The three terms add in quadrature, and the minimum of the resulting bathtub — $3.2428\times10^{-15}$ at $\tau = 4379$ s in Example 2 — is the longest averaging time that helps.
  * The flicker floor has the same two-level-fluctuator origin as the $1/f$ noise limiting transmons and SQUIDs elsewhere in this series.

**3\. Stability and accuracy are different quantities**

  * A systematic budget bounds the *uncertainty* of each shift, not its size; the illustrative budget of Example 3 totals $7.7\times10^{-17}$.
  * The strontium row reaches that floor in $4.6$ ms of averaging, so essentially all of the difficulty in optical clocks is systematic.
  * Blackbody radiation scales as $T^4$ and its uncertainty as $T^3$, which is why a 100 K enclosure is worth a factor 27 on that term and 3.3 on the total.

**4\. Light-pulse interferometry converts $T^2$ into sensitivity**

  * $\Delta\varphi = k_\mathrm{eff} a T^2$, so $\delta a \propto T^{-2}N^{-1/2}$; the measured slope is $-2.0000$ exactly.
  * The price of $T$ is height: $T = 1$ s means a $4.90$ m fountain, which is why long-baseline instruments are towers and shafts.
  * The same instrument measures rotation through the Coriolis term and mass distribution through the gradient, and the fringe periodicity — $3.901\times10^{-5}$ m s$^{-2}$ at $T = 100$ ms — bounds the dynamic range.

**5\. The vapour cell buys room temperature and pays in bandwidth and size**

  * Optical pumping replaces thermal polarization, and at 3 nT the Zeeman energy is $1.6\times10^{-12}$ of $k_BT$ — irrelevant, because none of the polarization comes from $k_BT$.
  * A field gradient produces a Gaussian $T_2^\ast$ decay ($2.409$ ms against a $200$ ms $T_2$), diagnosable from the decay shape exactly as in §1.4.
  * SERF suppresses spin exchange as $(\omega_0/R_\mathrm{se})^2$, buying a factor $16.5$ in sensitivity for a factor $274$ in bandwidth; raising the density saturates at the spin-destruction asymptote, $0.114$ fT$/\sqrt{\mathrm{Hz}}$.
  * Sensitivity scales as $a^{-3/2}$ in cell size, so a decade of spatial resolution costs $31.6$ in field sensitivity — a trade whose verdict depends on the geometry of the source.

**Practical implications**

  * Never quote a sensitivity without the averaging time and the bandwidth; the three are one statement.
  * Compute where the systematic floor arrives before designing for longer averaging.
  * When a decay is Gaussian rather than exponential, suspect an inhomogeneity and shim the field before improving the sensor.
  * Choose a magnetometer by standoff and source geometry first, and by $\eta_B$ second.

### Where This Leads

Every sensitivity in this chapter and the three before it contains a factor $1/\sqrt{N}$, and every one of them assumed the probes were independent. Chapter 5 removes that assumption. Entanglement can in principle replace $1/\sqrt{N}$ with $1/N$, and the chapter works out — numerically, and without flattery in either direction — how much of that survives contact with decoherence, what it costs to prepare, where in practice it is already load-bearing, and what a sensor's output looks like when it is treated as quantum data rather than as a number.

[← Chapter 3: SQUIDs](<chapter-3.html>) [Chapter 5: Beyond the Standard Quantum Limit →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * Clock transition frequencies quoted here are atomic constants, but the interrogation times, atom numbers, cycle times, shift coefficients, relaxation rates and sensitivities are illustrative order-of-magnitude values chosen for teaching; they are not specifications of any apparatus and must be verified against primary sources before use in any design or proposal.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
