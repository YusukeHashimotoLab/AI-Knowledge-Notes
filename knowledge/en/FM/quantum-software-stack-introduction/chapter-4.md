---
title: "Chapter 4: Pulses and Calibration"
chapter_title: "Chapter 4: Pulses and Calibration"
subtitle: Resonant Drive in the Language of Control, Pulse Shaping Against Leakage, Calibration Loops as Software-Driven Experiments, and Why Randomized Benchmarking Separates Gate Error from SPAM
reading_time: 50-55 minutes
difficulty: Advanced
code_examples: 9
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-software-stack-introduction/chapter-4.html>) | Last sync: 2026-08-13

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to the Quantum Software Stack](<index.html>) > Chapter 4

The three chapters before this one treated a gate as a symbol. `("rx", theta, q)` went into an optimizer, came out shorter, went into a router, came out with SWAPs around it, and at no point did anything in the pipeline care what an $R_x$ physically *is*. This chapter opens the last box. Underneath every entry of a gate list is a shaped microwave envelope a few tens of nanoseconds long, and underneath the envelope are three or four numbers — an amplitude, a frequency, a derivative weight — that no compiler can know and that no data sheet can supply, because they change from device to device and from Tuesday to Wednesday. They are *measured*, by software, on the machine, continuously. That measurement loop is the layer this chapter builds.

Two things are being claimed here and both are load-bearing. The first is that pulse-level control is not exotic physics: the whole of it is a three-level Hamiltonian in a rotating frame, and 40 lines of NumPy reproduce the leakage figures, the DRAG suppression and the calibration curves that a control stack is built around. The second is that a calibration routine is an experiment written as a program — a parametrized sequence, a fit, and an update rule — and that its correctness is testable in exactly the way the rest of this course tests things: put a known error into a simulated device, hide it from the routine, and check that the routine finds it. Every loop in Sections 4.3 gets that treatment, and one of them fails the test in an instructive way. Section 4.4 then asks the question a calibration loop cannot answer on its own: how well is the gate actually working, given that every measurement of it is contaminated by the errors of preparation and readout?

## Learning Objectives

After completing this chapter, you will be able to:

  * Write the driven three-level Hamiltonian in the rotating frame, explain why a gate is a calibrated pulse *area*, and identify which entries of the Hamiltonian a pulse-level API actually exposes
  * Derive leakage as a Fourier coefficient of the envelope evaluated at the anharmonicity, and predict from that why a square pulse's leakage falls as $\tau^{-2}$ while a Gaussian's falls much faster
  * Implement the DRAG correction, measure both the leakage suppression and the gate-error suppression it produces, and explain why the two have different optimal weights
  * Structure a calibration routine as a parametrized sequence plus a fit plus an update, and use error amplification to reach milliradian precision on a finite shot budget
  * Implement Rabi-amplitude, Ramsey-frequency and DRAG-weight calibration, verify that each recovers a deliberately mis-set parameter, and diagnose the case where one of them stalls on a systematic
  * Decompose a residual gate error onto $X$, $Y$ and $Z$ and say which calibration knob addresses which component, and why the loops must therefore be iterated
  * State the randomized-benchmarking model $F(m) = Ap^m + B$, show numerically that state-preparation and measurement error lands in $A$ and $B$ while gate error lands in $p$, and explain what RB does *not* tell you

### What Carries Over

The physics of Sections 4.1 and 4.2 is the physics of [Introduction to Quantum Hardware, Chapter 2](<../quantum-hardware-introduction/chapter-2.html>), restated in the language of a control stack. That chapter derives the transmon spectrum from the Josephson Hamiltonian and arrives at $E_J/h = 13.775$ GHz, $E_C/h = 0.250$ GHz, a transition at $f_{01} = 4.9851$ GHz and an anharmonicity $\alpha/2\pi = -284.87$ MHz. Those four numbers are inputs here, reproduced by Example 1 from the same diagonalization, and every pulse in this chapter runs on that spectrum. This chapter truncates the ladder at three levels rather than five, which moves the quoted leakage of a 20 ns Gaussian $\pi$ pulse from $3.117\times10^{-5}$ to $3.201\times10^{-5}$ — a 3% difference, and the reason for stating it is that a leakage number is only meaningful together with the number of levels kept.

The unit conventions are the sister course's, without exception. Hamiltonians are written with $\hbar = 1$. Symbols such as $\Omega$ and $\Delta$ are *angular* frequencies, quoted numbers are *cyclic*, and every factor of $2\pi$ in the code is explicit for that reason. Times are in nanoseconds and frequencies in GHz throughout, so $\alpha \tau$ is a dimensionless number that can be read straight off a table.

Nothing in this chapter needs the circuit IR of [Chapter 1](<chapter-1.html>): a pulse-level model is a single qubit and a single unitary, and the gate list never appears. [Chapter 5](<chapter-5.html>) picks the IR back up.

* * *

## 4.1 What Lies Beneath a Gate

### A gate is a pulse area

Apply a microwave tone to the qubit's capacitor and the Hamiltonian acquires a term

$$ H_d(t) = \Omega(t)\cos(\omega_d t + \phi)\,\hat{n} $$

where $\Omega(t)$ is the envelope the control electronics produces, $\omega_d$ the carrier, $\phi$ the carrier phase and $\hat{n}$ the charge operator. Move to the frame rotating at $\omega_d$, drop the counter-rotating terms, and for a genuine two-level system what is left is

$$ H_{\mathrm{rf}} = \frac{\Delta}{2}\sigma_z + \frac{\Omega_x(t)}{2}\sigma_x + \frac{\Omega_y(t)}{2}\sigma_y, \qquad \Delta = \omega_{01} - \omega_d $$

with $\Omega_x = \Omega\cos\phi$ and $\Omega_y = \Omega\sin\phi$. On resonance ($\Delta = 0$) this is a rotation about an axis in the equatorial plane set by $\phi$, through an angle

$$ \theta = \int_0^{\tau}\Omega(t)\,dt $$

That is the whole content of "a gate". An $R_x(\theta)$ is an envelope whose *integral* is $\theta$ and whose carrier phase is 0; an $R_y(\theta)$ is the same envelope at $\phi = \pi/2$. The shape of the envelope is free, and the entire subject of pulse shaping is about spending that freedom well. An $R_z(\theta)$ is cheaper still: because the frame is a software construction, a $Z$ rotation can be implemented by *redefining the phase of every subsequent pulse*, which costs no time and no error at all. This is the reason a compiler's decision to push $Z$ rotations to the end of a circuit — Chapter 2's peephole rules — has a physical payoff and not merely a bookkeeping one.

### The level that is not in the gate set

A transmon is not a two-level system, and the correction is not small. Keeping three levels, the rotating-frame Hamiltonian is

$$ H_{\mathrm{rf}}(t) = \sum_{j=0}^{2}(E_j - j\omega_d)\lvert j \rangle\langle j \rvert + \frac{\Omega_x(t)}{2}\left(b + b^{\dagger}\right) + \frac{\Omega_y(t)}{2}\,i\left(b^{\dagger} - b\right) $$

with a ladder operator $b = \sum_j r_j \lvert j \rangle\langle j+1 \rvert$ normalized so that $r_0 = 1$. Two facts about this Hamiltonian decide everything that follows. Level 2 sits at detuning $\alpha$ in the rotating frame, not at zero, so the drive is off-resonant for the $1\to2$ transition by $|\alpha| = 285$ MHz. And $r_1 = |n_{12}/n_{01}| = 1.3726$ is *larger* than one, so the transition the drive is not supposed to reach has a stronger matrix element than the one it is.

The consequence is a speed limit. A pulse of duration $\tau$ has spectral content out to about $1/\tau$; if that overlaps $|\alpha|$, the drive excites $1\to2$ and population leaves the computational subspace. Leakage is not an ordinary qubit error — no two-level error model describes it and no standard error-correcting code corrects it — which is why $|\alpha|\tau$ is the first number a control engineer computes about a device.

### Code Example 1: The Three-Level Control Model

```python
"""Chapter 4, Example 1: the three-level control model.
The transmon spectrum of Introduction to Quantum Hardware Chapter 2, truncated
to the three levels a single-qubit gate can reach, written as the rotating-frame
generators the pulse layer actually programs.  Frequencies in GHz, times in ns;
hbar = 1."""
import numpy as np
from scipy.linalg import expm, logm

TWOPI = 2.0 * np.pi
NLEV = 3


def transmon_eigen(EJ, EC, ncut=60, m=NLEV):
    """Eigenenergies (GHz, relative to the ground state) and the charge
    operator of a transmon, both in the transmon eigenbasis."""
    n = np.arange(-ncut, ncut + 1)
    H = np.diag(4.0 * EC * n ** 2.0)
    off = -0.5 * EJ * np.ones(2 * ncut)
    H += np.diag(off, 1) + np.diag(off, -1)
    E, V = np.linalg.eigh(H)
    nop = V[:, :m].conj().T @ np.diag(n.astype(float)) @ V[:, :m]
    return E[:m] - E[0], nop


EJ, EC = 13.775, 0.250
E, nop = transmon_eigen(EJ, EC)
f01 = E[1]
alpha = (E[2] - E[1]) - E[1]
print("Three-level control model from the transmon spectrum")
print("=" * 70)
print(f"  EJ/h = {EJ:.3f} GHz, EC/h = {EC:.3f} GHz, EJ/EC = {EJ / EC:.1f}")
print(f"  f01 = {f01:.4f} GHz, alpha/2pi = {alpha * 1e3:.2f} MHz")
print(f"  relative anharmonicity |alpha|/f01 = {abs(alpha) / f01 * 100:.2f} %")
print(f"  drive matrix element ratio |n12/n01| = {abs(nop[1, 2] / nop[0, 1]):.4f}"
      f"   (harmonic value sqrt(2) = {np.sqrt(2):.4f})")

# ---- rotating-frame generators -----------------------------------------
LOWER = np.zeros((NLEV, NLEV), dtype=complex)
for j in range(NLEV - 1):
    LOWER[j, j + 1] = abs(nop[j, j + 1] / nop[0, 1])
AX = TWOPI * 0.5 * (LOWER + LOWER.conj().T)
AY = TWOPI * 0.5 * 1j * (LOWER.conj().T - LOWER)


def frame(f_drive):
    """Diagonal part of the rotating-frame Hamiltonian at drive frequency f_drive."""
    return TWOPI * np.diag([E[j] - j * f_drive for j in range(NLEV)])


H0 = frame(f01)
print("\n  rotating-frame level detunings at f_drive = f01 (GHz):",
      np.round(np.diag(H0).real / TWOPI, 6))
print(f"  Ax/2pi has 0.5 on the 0-1 element and"
      f" {AX[1, 2].real / TWOPI:.4f} on the 1-2 element")


def propagate(t, ox, oy, H0):
    """Time-ordered propagator of H0 + Ox(t) Ax + Oy(t) Ay, midpoint rule."""
    U = np.eye(NLEV, dtype=complex)
    dt = t[1] - t[0]
    for k in range(len(t) - 1):
        H = H0 + 0.5 * (ox[k] + ox[k + 1]) * AX + 0.5 * (oy[k] + oy[k + 1]) * AY
        U = expm(-1j * H * dt) @ U
    return U


IDENT = np.eye(NLEV, dtype=complex)
XGATE = np.array([[0, 1], [1, 0]], dtype=complex)


def leakage(U):
    """Worst-case population outside the qubit subspace over three input states."""
    return max(float(abs(U @ psi)[2] ** 2) for psi in
               [IDENT[0], IDENT[1], (IDENT[0] + IDENT[1]) / np.sqrt(2)])


def gate_error(U, target):
    """Average gate error of the qubit block against target, global phase free."""
    Uq = U[:2, :2]
    return 1.0 - (abs(np.trace(target.conj().T @ Uq)) ** 2
                  + np.trace(Uq.conj().T @ Uq).real) / 6.0


# ---- a first pulse: does the model do what a gate is supposed to do? ----
tau, nstep = 20.0, 800
t = np.linspace(0.0, tau, nstep + 1)
env = np.exp(-((t - tau / 2) ** 2) / (2 * (tau / 4) ** 2)) - np.exp(-2.0)
ox = 0.5 * env / np.trapezoid(env, t)          # pulse area = pi
U = propagate(t, ox, 0.0 * ox, H0)
print(f"\n  a Gaussian pi pulse of {tau:.0f} ns, no DRAG:")
print(f"    peak Omega/2pi   = {ox.max() * 1e3:7.1f} MHz")
print(f"    pulse area / pi  = {2 * np.trapezoid(ox, t):7.4f}")
print(f"    leakage          = {leakage(U):7.3e}")
print(f"    gate error vs X  = {gate_error(U, XGATE):7.3e}")
```

```text
Three-level control model from the transmon spectrum
======================================================================
  EJ/h = 13.775 GHz, EC/h = 0.250 GHz, EJ/EC = 55.1
  f01 = 4.9851 GHz, alpha/2pi = -284.87 MHz
  relative anharmonicity |alpha|/f01 = 5.71 %
  drive matrix element ratio |n12/n01| = 1.3726   (harmonic value sqrt(2) = 1.4142)

  rotating-frame level detunings at f_drive = f01 (GHz): [ 0.        0.       -0.284873]
  Ax/2pi has 0.5 on the 0-1 element and 0.6863 on the 1-2 element

  a Gaussian pi pulse of 20 ns, no DRAG:
    peak Omega/2pi   =    46.7 MHz
    pulse area / pi  =  1.0000
    leakage          = 3.201e-05
    gate error vs X  = 2.970e-03
```

**What to notice.** The model reproduces the sister course's spectrum exactly — $f_{01} = 4.9851$ GHz, $\alpha/2\pi = -284.87$ MHz, $r_1 = 1.3726$ against the harmonic $\sqrt{2} = 1.4142$ — because it is the same diagonalization. What is new is the last block. The unit-area calibration `ox = 0.5 * env / trapezoid(env, t)` gives a pulse area of exactly $\pi$, and the resulting gate is wrong by $3.0\times10^{-3}$ even though only $3.2\times10^{-5}$ of the population left the qubit subspace. The gate error is a hundred times the leakage. That ratio is the single most important thing in this section, and Section 4.2 explains it: most of the damage is a *phase*, acquired by the qubit while population visits $\lvert 2 \rangle$ and comes back.

`propagate`, `leakage` and `gate_error` are the three functions every later example calls. The gate-error metric is the average gate infidelity of the qubit block against a target, computed phase-free; because the block of a leaky propagator is not unitary, $\mathrm{tr}(U^{\dagger}U) < 2$ and the leakage is counted in the error as well as separately.

* * *

## 4.2 Pulse Shaping

### Leakage is a Fourier coefficient

Treat the $1 \to 2$ coupling as a perturbation on top of the intended rotation. To first order the amplitude that ends up in $\lvert 2 \rangle$ is

$$ a_{1\to2} \simeq -\frac{i r_1}{2}\int_0^{\tau}\Omega_x(t)\,e^{i\alpha t}\,dt = -\frac{i r_1}{2}\,\tilde{\Omega}_x(\alpha) $$

The leakage is the squared Fourier coefficient of the envelope evaluated at the anharmonicity. That single line is the design rule for the whole pulse layer: **shape the envelope so that its spectrum has as little weight as possible at $\alpha$.** For a square pulse of unit area the spectrum is a sinc,

$$ \left\lvert\frac{\tilde{\Omega}(\alpha)}{\tilde{\Omega}(0)}\right\rvert^{2} = \mathrm{sinc}^{2}(\alpha\tau), \qquad \mathrm{sinc}(x) = \frac{\sin \pi x}{\pi x} $$

whose envelope falls only as $1/(\alpha\tau)^2$, because a discontinuous envelope has a $1/f$ spectral tail. A Gaussian has a Gaussian tail and does far better. The exponent, not the prefactor, is what shaping buys.

### Code Example 2: Square, Gaussian, and Where the Leakage Comes From

```python
"""Chapter 4, Example 2: square, Gaussian, and where the leakage comes from.
Continues from Example 1 (same session)."""


def envelope(shape, tau, nstep):
    """Unit-area envelope on [0, tau]: 'square' or 'gauss' (4-sigma, truncated)."""
    t = np.linspace(0.0, tau, nstep + 1)
    if shape == "square":
        env = np.ones_like(t)
    elif shape == "gauss":
        sig = tau / 4.0
        env = np.exp(-((t - tau / 2) ** 2) / (2 * sig ** 2)) - np.exp(-2.0)
    else:
        raise ValueError(shape)
    return t, env / np.trapezoid(env, t)


def pulse(shape, tau, theta=np.pi, beta=0.0, amp=1.0, axis="x", nstep=400):
    """Control waveform of a rotation by theta about axis, with DRAG weight beta."""
    t, u = envelope(shape, tau, nstep)
    ox = amp * (theta / TWOPI) * u
    oy = -beta * np.gradient(ox, t) / (TWOPI * 2.0 * alpha)
    return (t, ox, oy) if axis == "x" else (t, -oy, ox)


def run(shape, tau, theta=np.pi, beta=0.0, amp=1.0, axis="x",
        f_drive=None, nstep=400):
    """Propagator of one shaped pulse."""
    t, cx, cy = pulse(shape, tau, theta, beta, amp, axis, nstep)
    return propagate(t, cx, cy, frame(f01 if f_drive is None else f_drive))


def spectral_weight(shape, tau, f, nstep=4000):
    """|Omega(f)|^2 / |Omega(0)|^2 of the envelope: its Fourier weight at f."""
    t, u = envelope(shape, tau, nstep)
    return abs(np.trapezoid(u * np.exp(-2j * np.pi * f * t), t)) ** 2 \
        / abs(np.trapezoid(u, t)) ** 2


print("A. The envelope's Fourier weight at the anharmonicity")
print("=" * 74)
print(f"  the 1-2 transition sits {abs(alpha) * 1e3:.1f} MHz from the drive;"
      " an envelope that")
print("  has weight there drives it.  Relative weight |Omega(alpha)|^2 /"
      " |Omega(0)|^2:")
print(f"\n  {'tau (ns)':>9} {'|alpha| tau':>12} {'square':>12}"
      f" {'sinc^2 (exact)':>15} {'gauss':>12}")
for tau in (8.0, 10.0, 12.0, 16.0, 20.0, 30.0, 40.0):
    x = np.pi * alpha * tau
    print(f"  {tau:9.1f} {abs(alpha) * tau:12.2f}"
          f" {spectral_weight('square', tau, alpha):12.3e}"
          f" {(np.sin(x) / x) ** 2:15.3e}"
          f" {spectral_weight('gauss', tau, alpha):12.3e}")

print("\nB. Measured leakage and gate error of a pi pulse")
print("=" * 74)
print(f"  {'tau (ns)':>9} {'shape':>7} {'peak Om/2pi':>12} {'leak |1>->|2>':>14}"
      f" {'worst leak':>12} {'gate error':>12}")
meas = {"square": [], "gauss": []}
taus = (8.0, 10.0, 12.0, 16.0, 20.0, 30.0, 40.0)
for tau in taus:
    for shape in ("square", "gauss"):
        t, ox, oy = pulse(shape, tau)
        U = run(shape, tau)
        l12 = abs(U[2, 1]) ** 2
        meas[shape].append(l12)
        print(f"  {tau:9.1f} {shape:>7} {ox.max() * 1e3:12.1f} {l12:14.3e}"
              f" {leakage(U):12.3e} {gate_error(U, XGATE):12.3e}")

print("\n  power law of the |1>->|2> leakage over tau = 10 to 40 ns:")
for shape in ("square", "gauss"):
    y = np.array(meas[shape][1:])
    x = np.array(taus[1:])
    slope = np.polyfit(np.log(x), np.log(y), 1)[0]
    print(f"    {shape:>7}: leakage ~ tau^{slope:+.2f}"
          f"   -> doubling tau buys {2 ** (-slope):6.1f}x")
```

```text
A. The envelope's Fourier weight at the anharmonicity
==========================================================================
  the 1-2 transition sits 284.9 MHz from the drive; an envelope that
  has weight there drives it.  Relative weight |Omega(alpha)|^2 / |Omega(0)|^2:

   tau (ns)  |alpha| tau       square  sinc^2 (exact)        gauss
        8.0         2.28    1.152e-02       1.152e-02    3.250e-04
       10.0         2.85    2.614e-03       2.614e-03    9.376e-05
       12.0         3.42    8.114e-03       8.114e-03    2.832e-05
       16.0         4.56    4.717e-03       4.717e-03    4.078e-08
       20.0         5.70    2.067e-03       2.067e-03    2.512e-06
       30.0         8.55    1.358e-03       1.358e-03    2.729e-09
       40.0        11.39    6.983e-04       6.983e-04    1.352e-07

B. Measured leakage and gate error of a pi pulse
==========================================================================
   tau (ns)   shape  peak Om/2pi  leak |1>->|2>   worst leak   gate error
        8.0  square         62.5      2.138e-02    3.461e-02    2.583e-02
        8.0   gauss        116.8      9.393e-05    1.054e-04    1.815e-02
       10.0  square         50.0      1.382e-02    1.562e-02    1.973e-02
       10.0   gauss         93.4      3.427e-04    5.821e-04    1.198e-02
       12.0  square         41.7      9.853e-03    1.089e-02    1.284e-02
       12.0   gauss         77.8      9.594e-05    1.093e-04    8.270e-03
       16.0  square         31.2      5.614e-03    5.988e-03    7.683e-03
       16.0   gauss         58.4      4.052e-05    5.310e-05    4.648e-03
       20.0  square         25.0      3.607e-03    3.928e-03    5.004e-03
       20.0   gauss         46.7      1.761e-05    3.198e-05    2.970e-03
       30.0  square         16.7      1.609e-03    1.648e-03    2.146e-03
       30.0   gauss         31.1      3.140e-06    3.989e-06    1.318e-03
       40.0  square         12.5      9.039e-04    1.356e-03    1.172e-03
       40.0   gauss         23.4      9.251e-07    9.652e-07    7.407e-04

  power law of the |1>->|2> leakage over tau = 10 to 40 ns:
     square: leakage ~ tau^-1.97   -> doubling tau buys    3.9x
      gauss: leakage ~ tau^-4.09   -> doubling tau buys   17.1x
```

**What to notice.** Part A validates the spectral picture: the numerically integrated Fourier weight of the square envelope agrees with $\mathrm{sinc}^2(\alpha\tau)$ to every printed digit, and the Gaussian column is three to five orders of magnitude smaller. The Gaussian column also *oscillates* by orders of magnitude — $4\times10^{-8}$ at $\tau = 16$ ns against $2.5\times10^{-6}$ at 20 ns — and that is not numerical noise. The envelope used here is a Gaussian *truncated* at $\pm 2\sigma$ with the pedestal subtracted, so it still has small discontinuities in its derivative at the ends, and the residual sinc-like tail from those endpoints has zeros at particular durations. Real control stacks smooth the ends further for exactly this reason.

Part B is the measurement, and the last two lines are the deliverable: fitted over $\tau = 10$ to 40 ns, square-pulse leakage goes as $\tau^{-1.97}$ and Gaussian leakage as $\tau^{-4.09}$. Doubling the pulse duration buys a factor of 3.9 with a square pulse and a factor of 17 with a Gaussian. At $\tau = 20$ ns the two differ by a factor of 205 in leakage.

Two honest qualifications. First, the Gaussian is *worse* than the square pulse at $\tau = 8$ ns in gate error, and the reason is in the peak-amplitude column: at fixed area and fixed duration the Gaussian must reach 116.8 MHz where the square pulse needs 62.5 MHz, and a peak Rabi rate that approaches $|\alpha|$ breaks the perturbative picture entirely. Shaping helps only in the regime where the pulse is long enough to be shaped. Second, in every row the gate error exceeds the leakage, by two orders of magnitude at the long durations. Shaping alone does not fix a gate.

### DRAG

The excess is a phase. During the pulse, population makes a *virtual* excursion to $\lvert 2 \rangle$ and returns; the excursion shifts the qubit's effective frequency while the drive is on, and the accumulated phase is not part of the intended rotation. The standard fix, DRAG — Derivative Removal by Adiabatic Gate — drives the quadrature with the derivative of the in-phase envelope,

$$ \Omega_y(t) = -\beta\,\frac{\dot{\Omega}_x(t)}{2\alpha} $$

where the first-order theory gives $\beta = 1$. The derivative term is chosen so that it cancels the leading leakage amplitude, and it cancels the leading phase error at the same time. In a real stack $\beta$ is a calibrated number, not a constant, and Example 3 shows why.

### Code Example 3: DRAG, and the Two Optima It Does Not Share

```python
"""Chapter 4, Example 3: DRAG, and the two optima it does not share.
Continues from Example 2 (same session)."""
print("A. Sweeping the DRAG weight at tau = 20 ns")
print("=" * 74)
TAU = 20.0
PAULI3 = {"X": XGATE, "Y": np.array([[0, -1j], [1j, 0]]),
          "Z": np.array([[1, 0], [0, -1]], dtype=complex)}


def error_axis(U, target):
    """Residual rotation of the qubit block after the target is undone,
    resolved onto X, Y and Z: the vector (theta n_x, theta n_y, theta n_z)."""
    R = U[:2, :2] @ target.conj().T
    R = R / np.sqrt(complex(np.linalg.det(R)))       # into SU(2)
    if np.trace(R).real < 0.0:
        R = -R                                       # the lift closest to I
    A = logm(R)
    A = A - 0.5 * np.trace(A) * np.eye(2)
    return {k: float((1j * np.trace(P @ A)).real) for k, P in PAULI3.items()}


print(f"  {'beta':>6} {'leak |1>->|2>':>14} {'gate error':>12}"
      f" {'theta_x err':>12} {'theta_y err':>12} {'theta_z err':>12}")
for beta in np.arange(0.0, 2.26, 0.25):
    U = run("gauss", TAU, np.pi, beta)
    e = error_axis(U, XGATE)
    print(f"  {beta:6.2f} {abs(U[2, 1]) ** 2:14.3e} {gate_error(U, XGATE):12.3e}"
          f" {e['X']:12.2e} {e['Y']:12.2e} {e['Z']:12.2e}")


def golden_min(f, lo, hi, tol=1e-6):
    """Golden-section minimum of a unimodal f on [lo, hi] -- 60 lines less than
    an optimizer import, and the calibration loops below reuse it."""
    g = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


b_err = golden_min(lambda b: gate_error(run("gauss", TAU, np.pi, b), XGATE),
                   0.5, 1.5)
b_leak = golden_min(lambda b: abs(run("gauss", TAU, np.pi, b)[2, 1]) ** 2,
                    1.5, 2.5)
print("\nB. The two optima are different, and that is the whole subtlety")
print("=" * 74)
for name, b in (("no DRAG", 0.0), ("gate-error optimum", b_err),
                ("leakage optimum", b_leak)):
    U = run("gauss", TAU, np.pi, b)
    print(f"  {name:>19}  beta = {b:7.4f}   leakage = {abs(U[2, 1]) ** 2:9.3e}"
          f"   gate error = {gate_error(U, XGATE):9.3e}")
print(f"\n  improvement at the gate-error optimum:"
      f" {gate_error(run('gauss', TAU, np.pi, 0.0), XGATE) / gate_error(run('gauss', TAU, np.pi, b_err), XGATE):.0f}x")
print(f"  penalty at the leakage optimum       :"
      f" {gate_error(run('gauss', TAU, np.pi, b_leak), XGATE) / gate_error(run('gauss', TAU, np.pi, b_err), XGATE):.0f}x worse")

print("\nC. The gate-error optimum drifts with the pulse duration")
print("=" * 74)
print(f"  {'tau (ns)':>9} {'beta*':>7} {'err (beta=0)':>13} {'err (beta*)':>13}"
      f" {'improvement':>12}")
for tau in (8.0, 12.0, 20.0, 30.0, 40.0):
    bs = golden_min(lambda b: gate_error(run("gauss", tau, np.pi, b), XGATE),
                    0.0, 2.0)
    e0 = gate_error(run("gauss", tau, np.pi, 0.0), XGATE)
    e1 = gate_error(run("gauss", tau, np.pi, bs), XGATE)
    print(f"  {tau:9.1f} {bs:7.4f} {e0:13.3e} {e1:13.3e} {e0 / e1:11.0f}x")

```

```text
A. Sweeping the DRAG weight at tau = 20 ns
==========================================================================
    beta  leak |1>->|2>   gate error  theta_x err  theta_y err  theta_z err
    0.00      1.761e-05    2.970e-03    -7.35e-03     1.33e-01    -8.98e-17
    0.25      1.307e-05    1.604e-03    -6.96e-03     9.75e-02    -3.02e-16
    0.50      9.307e-06    6.569e-04    -6.69e-03     6.20e-02     1.18e-16
    0.75      6.262e-06    1.304e-04    -6.52e-03     2.65e-02    -2.05e-16
    1.00      3.883e-06    2.423e-05    -6.46e-03    -8.96e-03     2.97e-16
    1.25      2.116e-06    3.378e-04    -6.52e-03    -4.44e-02     3.18e-16
    1.50      9.130e-07    1.070e-03    -6.68e-03    -7.98e-02     1.16e-16
    1.75      2.232e-07    2.220e-03    -6.96e-03    -1.15e-01    -1.10e-16
    2.00      1.091e-10    3.784e-03    -7.35e-03    -1.51e-01     9.63e-17
    2.25      1.983e-07    5.762e-03    -7.85e-03    -1.86e-01     5.55e-17

B. The two optima are different, and that is the whole subtlety
==========================================================================
              no DRAG  beta =  0.0000   leakage = 1.761e-05   gate error = 2.970e-03
   gate-error optimum  beta =  0.9382   leakage = 4.412e-06   gate error = 1.140e-05
      leakage optimum  beta =  2.0030   leakage = 7.859e-11   gate error = 3.806e-03

  improvement at the gate-error optimum: 261x
  penalty at the leakage optimum       : 334x worse

C. The gate-error optimum drifts with the pulse duration
==========================================================================
   tau (ns)   beta*  err (beta=0)   err (beta*)  improvement
        8.0  0.9122     1.815e-02     3.297e-04          55x
       12.0  0.9311     8.270e-03     8.380e-05          99x
       20.0  0.9382     2.970e-03     1.140e-05         261x
       30.0  0.9403     1.318e-03     2.241e-06         588x
       40.0  0.9411     7.407e-04     7.061e-07        1049x
```

**What to notice.** Part A resolves the residual error of the $\pi$ pulse onto the three Pauli axes, which is the diagnostic that makes the rest of the chapter possible. Read down the columns. The $Z$ component is zero to machine precision — a single $\pi$ pulse about $x$ cannot accumulate a $Z$ error, which is exactly why Section 4.3 needs a Ramsey experiment and not a $\pi$ pulse to calibrate the frequency. The $X$ component sits at $-6.5\times10^{-3}$ rad and barely moves with $\beta$: it is a rotation-angle error, and DRAG has no purchase on it. The $Y$ component is what $\beta$ controls, sweeping from $+0.133$ rad through zero to $-0.186$ rad. One knob, one error component.

Part B is the subtlety. The $\beta$ that minimizes the *gate error* is 0.9382; the $\beta$ that minimizes the *leakage* is 2.0030. They are different, and going to the leakage optimum makes the gate 334 times worse, because at $\beta = 2$ the leakage is $8\times10^{-11}$ and the phase error is completely uncancelled. Anyone who tunes DRAG by minimizing the population in $\lvert 2 \rangle$ has optimized the wrong thing, and the sign of having done so is a beautiful leakage number attached to a bad gate. Part C adds that the gate-error optimum is not the analytic $\beta = 1$ either, and drifts from 0.912 to 0.941 across the durations in the table. A pulse-level API exposes $\beta$ as a parameter because there is no value to hard-code.

* * *

## 4.3 Calibration as a Software-Driven Experiment

### The shape of a loop

Every calibration routine in every control stack has the same three parts.

  1. **A parametrized sequence** whose outcome depends on the parameter to be calibrated, and — this is the hard part — depends on it *more* than on anything else.
  2. **A fit** that maps a set of noisy readout fractions to an estimate of the parameter.
  3. **An update** that writes the estimate back and, usually, decides whether to iterate.

Two design principles do most of the work. The first is **error amplification**: build the sequence so that a small parameter error appears in the outcome multiplied by a repetition count $N$. A single $\pi$ pulse with a 1% amplitude error gives a readout 0.02% away from 1, which no realistic shot budget resolves; eighty-one of them gives an error of 81%, which one shot resolves. The second is **cancellation by construction**: build the observable as a *difference* of two sequences chosen so that the error you want cancels and the error you do not want does not. Readout error is 2 to 5% on good hardware and a gate error you care about is $10^{-4}$; no amount of shots will let you subtract the former from the latter, so the sequence has to do it.

### What the software is allowed to see

The honest way to test a calibration routine is to hide the truth from it. Example 4 defines a simulated device with four undisclosed parameters — an amplifier gain, the true qubit frequency, two readout error rates — plus an initialization error and a quasi-static frequency spread. Every routine from here on calls `device()`, which returns one number between 0 and 1 with binomial shot noise on it, and nothing else. No routine reads `HIDDEN`.

### Code Example 4: The Device, as the Calibration Software Sees It

```python
"""Chapter 4, Example 4: the device the calibration software is allowed to see.
Continues from Example 3 (same session)."""
# The four numbers below stand for everything the control software does not
# know: the gain of the amplifier chain, the true qubit frequency, and the two
# readout error rates. No routine in Examples 5 to 8 reads this dictionary --
# they only call device(), which is the whole point of the exercise.
HIDDEN = {"gain": 1.137, "f_qubit": f01, "eps01": 0.020, "eps10": 0.045,
          "init_err": 0.012, "sigma_f": 2.0e-4}

TAU_G = 20.0                 # the gate duration the pulse layer has settled on
_CACHE = {}


def _prop(theta, axis, amp, beta, f_drive):
    """Cached propagator of one pulse, with the hidden gain applied."""
    key = (theta, axis, amp, beta, f_drive)
    if key not in _CACHE:
        _CACHE[key] = run("gauss", TAU_G, theta, beta, amp * HIDDEN["gain"],
                          axis, f_drive=f_drive)
    return _CACHE[key]


def sequence_unitary(seq, f_drive, amp, beta, df=0.0):
    """Propagator of a sequence: ('p', theta, axis) pulses and ('i', T) idles.

    df is a static offset of the qubit frequency, used to average over the
    quasi-static frequency noise that gives a Ramsey fringe its envelope.
    """
    U = np.eye(NLEV, dtype=complex)
    for item in seq:
        if item[0] == "p":
            U = _prop(item[1], item[2], amp, beta, f_drive - df) @ U
        else:
            ph = [TWOPI * ((E[j] + j * df) - j * f_drive) * item[1]
                  for j in range(NLEV)]
            U = np.diag(np.exp(-1j * np.array(ph))) @ U
    return U


def p_read1(seq, f_drive, amp, beta, nquad=9):
    """Probability of reading 1, including initialization and readout error and
    an average over the quasi-static frequency spread."""
    hz, hw = np.polynomial.hermite_e.hermegauss(nquad)
    hw = hw / hw.sum()
    p1 = p2 = 0.0
    for z, w in zip(hz, hw):
        U = sequence_unitary(seq, f_drive, amp, beta, df=HIDDEN["sigma_f"] * z)
        a0, a1 = U[:, 0], U[:, 1]
        p1 += w * ((1 - HIDDEN["init_err"]) * abs(a0[1]) ** 2
                   + HIDDEN["init_err"] * abs(a1[1]) ** 2)
        p2 += w * ((1 - HIDDEN["init_err"]) * abs(a0[2]) ** 2
                   + HIDDEN["init_err"] * abs(a1[2]) ** 2)
    p0 = 1.0 - p1 - p2
    return p1 * (1 - HIDDEN["eps10"]) + p0 * HIDDEN["eps01"] + p2


def device(seq, f_drive, amp, beta, shots, rng):
    """One experiment: prepare, run the sequence, read out, return a frequency.

    shots=None returns the exact probability -- an infinite shot budget, which
    no experiment has, but which tells a calibration loop's designer whether a
    residual is statistical or systematic.
    """
    p = p_read1(seq, f_drive, amp, beta)
    return p if shots is None else rng.binomial(shots, p) / shots


print("The device, as the calibration software sees it")
print("=" * 74)
rng = np.random.default_rng(20260813)
for label, seq in [("nothing (readout of |0>)", []),
                   ("one nominal pi pulse", [("p", np.pi, "x")]),
                   ("two nominal pi pulses", [("p", np.pi, "x")] * 2),
                   ("pi/2, idle 500 ns, pi/2",
                    [("p", np.pi / 2, "x"), ("i", 500.0),
                     ("p", np.pi / 2, "x")])]:
    v = device(seq, f01, 1.0, 0.0, 8000, rng)
    print(f"  {label:>26}: read-1 fraction = {v:.4f}")
```

```text
The device, as the calibration software sees it
==========================================================================
    nothing (readout of |0>): read-1 fraction = 0.0359
        one nominal pi pulse: read-1 fraction = 0.9019
       two nominal pi pulses: read-1 fraction = 0.1852
     pi/2, idle 500 ns, pi/2: read-1 fraction = 0.8161
```

**What to notice.** The readout of a freshly initialized qubit is 0.036 rather than 0, and one nominal $\pi$ pulse gives 0.90 rather than 1. Both numbers are dominated by errors that have nothing to do with the gate: initialization leaves 1.2% in $\lvert 1 \rangle$, and the readout confuses 2.0% of zeros and 4.5% of ones. The $\pi/2$-idle-$\pi/2$ sequence returns 0.82 because 500 ns of idling at $\sigma_f = 200$ kHz of quasi-static spread has already cost visible contrast; the exact probability is averaged over that spread with a nine-node Gauss-Hermite quadrature, which is what makes the Ramsey fringe in Example 6 decay like a real one. The `shots=None` branch is a deliberate convenience for the routine's *designer*, not for the routine: it returns the exact probability, and comparing a loop's residual with and without shot noise is how you tell a systematic from a statistic.

### Code Example 5: Calibration Loop I — the Rabi Amplitude

```python
"""Chapter 4, Example 5: calibration loop I -- the Rabi amplitude.
Continues from Example 4 (same session)."""


def fit_cosine(x, y, w_grid):
    """Least-squares fit of y = c0 + c1 cos(w x) over a grid of w.

    Linear in (c0, c1) at fixed w, so the whole fit is a one-dimensional scan
    with a 2x2 solve inside it: no optimizer, and no initial guess to get wrong.
    """
    best = (np.inf, None)
    for w in w_grid:
        A = np.column_stack([np.ones_like(x), np.cos(w * x)])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = float(np.sum((A @ c - y) ** 2))
        if r < best[0]:
            best = (r, w)
    return best[1]


def rabi_calibration(stages, start, shots, rng, f_drive, beta):
    """Coarse amplitude sweep, then error-amplified refinements.

    n_rep repetitions of the pi pulse multiply the fringe frequency in the
    amplitude knob by n_rep, so the same sweep with the same shot budget locates
    the pi amplitude n_rep times more sharply. That is error amplification, and
    it is the only reason calibration reaches milliradian precision on a finite
    budget.
    """
    est, trace = start, []
    for n_rep in stages:
        half = 0.45 if n_rep == 1 else 0.40 / n_rep
        npts = 41 if n_rep == 1 else 21
        amps = np.linspace(est * (1 - half), est * (1 + half), npts)
        seq = [("p", np.pi, "x")] * n_rep
        y = np.array([device(seq, f_drive, a, beta, shots, rng) for a in amps])
        grid = np.linspace(est * (1 - half), est * (1 + half), 8001)
        est = n_rep * np.pi / fit_cosine(amps, y, n_rep * np.pi / grid)
        trace.append((n_rep, npts, est))
    return est, trace


def true_pi_amplitude(beta, f_drive):
    """Oracle, used only to report the answer -- never by a calibration loop."""
    return golden_min(lambda a: -abs(run("gauss", TAU_G, np.pi, beta,
                                        a * HIDDEN["gain"],
                                        f_drive=f_drive)[1, 0]) ** 2, 0.7, 1.1)


print("Calibration loop I: the Rabi amplitude")
print("=" * 74)
amp_true = true_pi_amplitude(0.0, f01)
print(f"  hidden amplifier gain      : {HIDDEN['gain']:.4f}, so the nominal"
      f" amp = 1 pulse")
print(f"  over-rotates by {(HIDDEN['gain'] - 1) * np.pi * 1e3:.0f} mrad")
print(f"  true pi amplitude (oracle) : {amp_true:.6f}"
      f"   (1/gain = {1 / HIDDEN['gain']:.6f})")

STAGES = (1, 5, 21, 81)
print("\nA. The loop, with the DRAG weight still zero")
print("  'exact' repeats the same loop with an infinite shot budget, which no")
print("  experiment has, but which separates systematics from statistics.")
print(f"  {'n_rep':>6} {'points':>7} {'amp_pi (2000 shots)':>20}"
      f" {'over-rot':>10} {'amp_pi (exact)':>15} {'over-rot':>10}")
rng = np.random.default_rng(4242)
amp_true = true_pi_amplitude(0.0, f01)
est, exact = 1.0, 1.0
for n_rep in STAGES:
    est, tr = rabi_calibration((n_rep,), est, 2000, rng, f01, 0.0)
    exact, _ = rabi_calibration((n_rep,), exact, None, None, f01, 0.0)
    print(f"  {n_rep:6d} {tr[0][1]:7d} {est:20.6f}"
          f" {abs(est - amp_true) * np.pi * 1e3:7.2f} mrad {exact:15.6f}"
          f" {abs(exact - amp_true) * np.pi * 1e3:7.2f} mrad")
amp_cal = est

print("\nB. The same loop with the DRAG weight of Example 7 already in place")
b_known = 0.9382
amp_true_b = true_pi_amplitude(b_known, f01)
print(f"  {'n_rep':>6} {'amp_pi found':>13} {'residual':>11} {'over-rotation':>15}")
rng = np.random.default_rng(4242)
est = 1.0
for n_rep in STAGES:
    est, _ = rabi_calibration((n_rep,), est, 2000, rng, f01, b_known)
    print(f"  {n_rep:6d} {est:13.6f} {est - amp_true_b:+11.2e}"
          f" {abs(est - amp_true_b) * np.pi * 1e3:12.2f} mrad")
e0 = gate_error(run("gauss", TAU_G, np.pi, 0.0, HIDDEN["gain"]), XGATE)
e1 = gate_error(run("gauss", TAU_G, np.pi, 0.0, amp_cal * HIDDEN["gain"]), XGATE)
print(f"\n  gate error at the nominal amplitude: {e0:.3e}")
print(f"  gate error after loop A            : {e1:.3e}")
```

```text
Calibration loop I: the Rabi amplitude
==========================================================================
  hidden amplifier gain      : 1.1370, so the nominal amp = 1 pulse
  over-rotates by 430 mrad
  true pi amplitude (oracle) : 0.879429   (1/gain = 0.879507)

A. The loop, with the DRAG weight still zero
  'exact' repeats the same loop with an infinite shot budget, which no
  experiment has, but which separates systematics from statistics.
   n_rep  points  amp_pi (2000 shots)   over-rot  amp_pi (exact)   over-rot
       1      41             0.879062    1.15 mrad        0.879962    1.67 mrad
       5      21             0.881735    7.24 mrad        0.881494    6.49 mrad
      21      21             0.881684    7.08 mrad        0.881586    6.78 mrad
      81      21             0.881706    7.15 mrad        0.881688    7.10 mrad

B. The same loop with the DRAG weight of Example 7 already in place
   n_rep  amp_pi found    residual   over-rotation
       1      0.881763   +4.35e-04         1.37 mrad
       5      0.881198   -1.29e-04         0.41 mrad
      21      0.881076   -2.51e-04         0.79 mrad
      81      0.881292   -3.55e-05         0.11 mrad

  gate error at the nominal amplitude: 3.298e-02
  gate error after loop A            : 2.981e-03
```

**What to notice.** The amplifier delivers 13.7% more amplitude than nominal, so the nominal $\pi$ pulse over-rotates by 430 mrad. Part A's coarse sweep finds the $\pi$ amplitude to 1.15 mrad, and then the amplified stages make it *worse* and stop improving: 7.24, 7.08, 7.15 mrad at $n_{\mathrm{rep}} = 5, 21, 81$. The `exact` columns settle the diagnosis. With an infinite shot budget the residual stalls at the same 7 mrad, so this is a systematic and not a sampling problem. Its source is the uncalibrated DRAG weight: repeating the pulse accumulates the $Y$ error of Example 3 as well as the amplitude error, and the fit — which assumes a clean cosine in the amplitude knob — absorbs part of it into the fitted frequency.

Part B is the same loop with $\beta$ set to the value Example 7 will measure, and it behaves as advertised: 1.37, 0.41, 0.79, 0.11 mrad. A 13.7% amplitude error is recovered to about a tenth of a milliradian, and the gate error falls from $3.3\times10^{-2}$ to $3.0\times10^{-3}$, which is the ceiling set by the *still* uncalibrated DRAG weight. Calibration loops are not independent. They are run in a fixed order and then run again, and Example 8 is the demonstration that the fixed point exists.

### Why the frequency needs an idle

The sensitivity analysis of Example 3 said that a single $\pi$ pulse produces no $Z$ error. The converse is what matters here: a *detuning* produces almost no signature in a $\pi$ pulse either, because the drive is strong compared with the detuning and simply rotates the error away. A 1.5 MHz detuning on a 20 ns pulse costs a gate error of $5\times10^{-4}$, which is smaller than the DRAG error already present. To make a detuning visible you have to turn the drive *off* and let the phase accumulate, which is what a Ramsey sequence does: $\pi/2$, wait $T$, $\pi/2$. The accumulated phase is $2\pi\Delta f\,T$ and the sensitivity grows linearly in $T$ until $T$ reaches $T_2^{\ast}$.

One subtlety is unavoidable and is worth building into the routine rather than discovering later: a Ramsey fringe measures $\lvert \Delta f \rvert$, not $\Delta f$. The standard resolution is to detune the drive deliberately by a known $f_{\mathrm{art}}$ in both directions and take the difference of the two fringe frequencies, which recovers the sign and, as a bonus, never asks the fit to resolve a fringe near zero frequency.

### Code Example 6: Calibration Loop II — the Qubit Frequency

```python
"""Chapter 4, Example 6: calibration loop II -- the qubit frequency.
Continues from Example 5 (same session)."""
T2_GRID = np.linspace(300.0, 2500.0, 12)


def fit_ramsey(delays, y, nu_hi, rounds=2, npts=401):
    """Fit y = c0 + exp(-(T/T2)^2)(c1 cos 2 pi nu T + c2 sin 2 pi nu T).

    Linear in (c0, c1, c2) at fixed (nu, T2), so the fit is a scan over the two
    nonlinear parameters with a 3x3 solve inside, refined once around the best
    point. Returns the fringe frequency and the Gaussian envelope time.
    """
    lo, hi = 0.0, nu_hi
    nu = t2 = None
    for _ in range(rounds):
        best = (np.inf, None, None)
        for t2c in T2_GRID:
            dec = np.exp(-(delays / t2c) ** 2)
            for nuc in np.linspace(lo, hi, npts):
                A = np.column_stack([np.ones_like(delays),
                                     dec * np.cos(TWOPI * nuc * delays),
                                     dec * np.sin(TWOPI * nuc * delays)])
                c, *_ = np.linalg.lstsq(A, y, rcond=None)
                r = float(np.sum((A @ c - y) ** 2))
                if r < best[0]:
                    best = (r, nuc, t2c)
        _, nu, t2 = best
        step = (hi - lo) / (npts - 1)
        lo, hi = nu - 2 * step, nu + 2 * step
    return nu, t2


def ramsey(f_drive, amp, beta, delays, shots, rng):
    """One Ramsey scan: pi/2, wait, pi/2, read out."""
    return np.array([device([("p", np.pi / 2, "x"), ("i", T),
                             ("p", np.pi / 2, "x")],
                            f_drive, amp, beta, shots, rng) for T in delays])


F_ART = 2.0e-3          # deliberate 2 MHz offset, applied both ways
DELAYS = np.linspace(0.0, 2000.0, 41)

print("Calibration loop II: the qubit frequency, by Ramsey")
print("=" * 74)
print(f"  hidden qubit frequency     : {HIDDEN['f_qubit']:.7f} GHz")
print(f"  quasi-static spread sigma_f: {HIDDEN['sigma_f'] * 1e6:.0f} kHz ->"
      f" T2* = sqrt(2)/(2 pi sigma_f) ="
      f" {np.sqrt(2) / (TWOPI * HIDDEN['sigma_f']):.0f} ns")
f_est = HIDDEN["f_qubit"] + 1.5e-3          # the software starts 1.5 MHz high
print(f"  software's initial guess   : {f_est:.7f} GHz"
      f"  (off by {(f_est - HIDDEN['f_qubit']) * 1e6:+.0f} kHz)")
print(f"\n  two scans per iteration, detuned by {F_ART * 1e6:.0f} kHz either"
      " way:")
print("  nu+ = |D - f_art|, nu- = |D + f_art|, so D = (nu- - nu+)/2, and")
print("  (nu- + nu+)/2 must return f_art -- a free check on the fit.")

rng = np.random.default_rng(90210)
print(f"\n  {'iter':>5} {'f_est (GHz)':>13} {'nu+ (kHz)':>10} {'nu- (kHz)':>10}"
      f" {'check (kHz)':>12} {'T2* (ns)':>9} {'residual (kHz)':>15}")
for it in range(1, 5):
    yp = ramsey(f_est + F_ART, amp_cal, 0.0, DELAYS, 2000, rng)
    ym = ramsey(f_est - F_ART, amp_cal, 0.0, DELAYS, 2000, rng)
    nup, t2p = fit_ramsey(DELAYS, yp, 5.0e-3)
    num, t2m = fit_ramsey(DELAYS, ym, 5.0e-3)
    delta = 0.5 * (num - nup)
    f_est = f_est + delta
    print(f"  {it:5d} {f_est:13.7f} {nup * 1e6:10.1f} {num * 1e6:10.1f}"
          f" {0.5 * (num + nup) * 1e6:12.1f} {0.5 * (t2p + t2m):9.0f}"
          f" {(f_est - HIDDEN['f_qubit']) * 1e6:+15.2f}")

f_cal = f_est
e_bad = gate_error(run("gauss", TAU_G, np.pi, 0.0, amp_cal * HIDDEN["gain"],
                       f_drive=HIDDEN["f_qubit"] + 1.5e-3), XGATE)
e_ok = gate_error(run("gauss", TAU_G, np.pi, 0.0, amp_cal * HIDDEN["gain"],
                      f_drive=f_cal), XGATE)
print(f"\n  gate error of the pi pulse at the initial guess: {e_bad:.3e}")
print(f"  gate error of the pi pulse at the calibrated f  : {e_ok:.3e}")
```

```text
Calibration loop II: the qubit frequency, by Ramsey
==========================================================================
  hidden qubit frequency     : 4.9850876 GHz
  quasi-static spread sigma_f: 200 kHz -> T2* = sqrt(2)/(2 pi sigma_f) = 1125 ns
  software's initial guess   : 4.9865876 GHz  (off by +1500 kHz)

  two scans per iteration, detuned by 2000 kHz either way:
  nu+ = |D - f_art|, nu- = |D + f_art|, so D = (nu- - nu+)/2, and
  (nu- + nu+)/2 must return f_art -- a free check on the fit.

   iter   f_est (GHz)  nu+ (kHz)  nu- (kHz)  check (kHz)  T2* (ns)  residual (kHz)
      1     4.9850834     3507.8      499.4       2003.6      1100           -4.19
      2     4.9850847     1997.8     2000.2       1999.0      1100           -2.94
      3     4.9850882     1995.5     2002.6       1999.1      1100           +0.63
      4     4.9850899     2001.0     2004.2       2002.6      1100           +2.25

  gate error of the pi pulse at the initial guess: 4.962e-04
  gate error of the pi pulse at the calibrated f  : 2.975e-03
```

**What to notice.** One iteration takes a 1500 kHz error to 4.2 kHz, and the loop then wanders at the few-kHz level set by shot noise and by the finite $T_2^{\ast}$ — there is no gain from iterating further, and a routine that kept iterating would be spending device time to track noise. The consistency check does its job: $(\nu_- + \nu_+)/2$ returns 2004, 1999, 1999 and 2003 kHz against the 2000 kHz that was deliberately applied, which is a free test that the fit has locked onto the right fringe and not an alias of it. The fitted $T_2^{\ast}$ comes back as 1100 ns against the 1125 ns implied by $\sigma_f = 200$ kHz through $T_2^{\ast} = \sqrt{2}/(2\pi\sigma_f)$ — the same relation [Introduction to Quantum Hardware, Chapter 1](<../quantum-hardware-introduction/chapter-1.html>) uses to read a static frequency spread as a statement about disorder in the host material. A frequency-calibration routine is also, for free, a materials measurement.

The last two lines are the chapter's best accident. The gate error is *lower* at the wrong frequency ($5.0\times10^{-4}$) than at the right one ($3.0\times10^{-3}$), because the $Y$ error from the 1.5 MHz detuning happens to partially cancel the $Y$ error from the missing DRAG weight. Two uncalibrated parameters conspiring to look good is the standard way a control stack fools its operator, and it is the reason parameters are calibrated against sequences that isolate them rather than against a single figure of merit.

### Code Example 7: Calibration Loop III — the DRAG Weight

```python
"""Chapter 4, Example 7: calibration loop III -- the DRAG weight.
Continues from Example 6 (same session)."""


def pingpong(beta, f_drive, amp, shots, rng):
    """The two mirrored sequences whose readouts cross where beta is right.

    (X90, Y180) and (Y90, X180) are the same rotation except for the sign with
    which the residual out-of-plane error enters, so their difference is odd in
    that error and even in everything else -- readout error in particular
    cancels from it, which is why the difference and not either readout is the
    calibration observable.
    """
    a = [("p", np.pi / 2, "x"), ("p", np.pi, "y")]
    b = [("p", np.pi / 2, "y"), ("p", np.pi, "x")]
    return (device(a, f_drive, amp, beta, shots, rng)
            - device(b, f_drive, amp, beta, shots, rng))


def drag_calibration(betas, f_drive, amp, shots, rng):
    """Sweep beta, fit the straight line, return where it crosses zero."""
    d = np.array([pingpong(b, f_drive, amp, shots, rng) for b in betas])
    slope, intercept = np.polyfit(betas, d, 1)
    return -intercept / slope, slope, d


print("Calibration loop III: the DRAG weight")
print("=" * 74)
b_opt = golden_min(lambda b: gate_error(
    run("gauss", TAU_G, np.pi, b, amp_cal * HIDDEN["gain"], f_drive=f_cal),
    XGATE), 0.0, 2.0)
print(f"  oracle: the beta minimizing the gate error is {b_opt:.4f}."
      f"  The loop never")
print("  evaluates a gate error; it finds where a difference changes sign.")

BETAS = np.linspace(0.4, 1.6, 7)
rng = np.random.default_rng(1357)
print(f"\nA. The observable, at {8000} shots per point")
print(f"  {'beta':>6} {'(X90,Y180)':>12} {'(Y90,X180)':>12} {'difference':>12}")
for b in BETAS:
    a = device([("p", np.pi / 2, "x"), ("p", np.pi, "y")], f_cal, amp_cal, b,
               8000, rng)
    c = device([("p", np.pi / 2, "y"), ("p", np.pi, "x")], f_cal, amp_cal, b,
               8000, rng)
    print(f"  {b:6.2f} {a:12.4f} {c:12.4f} {a - c:+12.4f}")

print("\nB. Where the line crosses zero, against the shot budget")
print("  Five independent seeds per budget, so that the systematic part of the")
print("  residual can be told apart from the statistical part.")
print(f"  {'shots/pt':>9} {'slope':>8} {'mean beta':>10} {'mean resid':>11}"
      f" {'spread':>9} {'predicted':>10} {'mean gate error':>16}")
for shots in (500, 2000, 8000, 32000):
    bs, es, sl = [], [], []
    for seed in range(5):
        r = np.random.default_rng(1000 + 17 * seed)
        b_hat, slope, _ = drag_calibration(BETAS, f_cal, amp_cal, shots, r)
        bs.append(b_hat)
        sl.append(slope)
        es.append(gate_error(run("gauss", TAU_G, np.pi, b_hat,
                                 amp_cal * HIDDEN["gain"], f_drive=f_cal),
                             XGATE))
    # shot noise on each point is sqrt(2 p(1-p)/N) ~ sqrt(0.5/N); seven points
    # and a straight-line fit divide that by sqrt(7)
    pred = np.sqrt(0.5 / shots) / (np.mean(sl) * np.sqrt(7.0))
    print(f"  {shots:9d} {np.mean(sl):8.4f} {np.mean(bs):10.4f}"
          f" {np.mean(bs) - b_opt:+11.4f} {np.std(bs):9.4f} {pred:10.4f}"
          f" {np.mean(es):16.3e}")
beta_cal = np.mean(bs)
print(f"\n  calibrated beta = {beta_cal:.4f}, gate error"
      f" {gate_error(run('gauss', TAU_G, np.pi, beta_cal, amp_cal * HIDDEN['gain'], f_drive=f_cal), XGATE):.3e}")
```

```text
Calibration loop III: the DRAG weight
==========================================================================
  oracle: the beta minimizing the gate error is 0.9373.  The loop never
  evaluates a gate error; it finds where a difference changes sign.

A. The observable, at 8000 shots per point
    beta   (X90,Y180)   (Y90,X180)   difference
    0.40       0.4541       0.5194      -0.0653
    0.60       0.4680       0.5121      -0.0441
    0.80       0.4808       0.4965      -0.0157
    1.00       0.4981       0.4836      +0.0145
    1.20       0.4966       0.4680      +0.0286
    1.40       0.5069       0.4589      +0.0480
    1.60       0.5324       0.4445      +0.0879

B. Where the line crosses zero, against the shot budget
  Five independent seeds per budget, so that the systematic part of the
  residual can be told apart from the statistical part.
   shots/pt    slope  mean beta  mean resid    spread  predicted  mean gate error
        500   0.1205     0.9521     +0.0149    0.1451     0.0992        7.663e-05
       2000   0.1200     0.9410     +0.0038    0.0282     0.0498        7.484e-06
       8000   0.1282     0.9333     -0.0039    0.0206     0.0233        6.246e-06
      32000   0.1288     0.9333     -0.0040    0.0108     0.0116        5.198e-06

  calibrated beta = 0.9333, gate error 4.806e-06
```

**What to notice.** Part A shows why the difference and not either readout is the observable. Both sequences return values within a few percent of 0.5, and neither is informative on its own; their difference runs from $-0.065$ to $+0.088$ and is linear in $\beta$. The readout errors — 2.0% and 4.5% — are common to both sequences and cancel out of the difference exactly, which is the "cancellation by construction" principle in its simplest form.

Part B separates statistics from systematics with five seeds per budget. The spread falls as $0.145, 0.028, 0.021, 0.011$ against the shot-noise prediction $0.099, 0.050, 0.023, 0.012$ — the right scaling and the right size — and the mean residual stays inside the spread at every budget, so unlike the amplitude loop there is no systematic to stall on here. The calibrated $\beta = 0.9333$ gives a gate error of $4.8\times10^{-6}$ against the oracle's $4.6\times10^{-6}$ at $\beta = 0.9373$: the $\beta$ residual is not the limiting term.

What does *not* work is worth recording. Repeating the mirrored pulse pair does not amplify this observable. The pair returns the state to a fixed point of the rotation, so the sensitivity cancels between repetitions instead of accumulating, and a naive $n_{\mathrm{rep}}$ sweep produces slopes that shrink towards zero and crossings that are meaningless. Error amplification is a property of a specific sequence, not a trick that can be applied to any of them.

### Code Example 8: The Three Loops, End to End

```python
"""Chapter 4, Example 8: the three loops, end to end.
Continues from Example 7 (same session)."""


def calibrate(rounds, shots, rng, f_start, amp_start, beta_start):
    """Frequency, then amplitude, then DRAG weight; repeat. Returns a trace."""
    f, a, b = f_start, amp_start, beta_start
    trace = []
    for r in range(1, rounds + 1):
        yp = ramsey(f + F_ART, a, b, DELAYS, shots, rng)
        ym = ramsey(f - F_ART, a, b, DELAYS, shots, rng)
        f = f + 0.5 * (fit_ramsey(DELAYS, ym, 5.0e-3)[0]
                       - fit_ramsey(DELAYS, yp, 5.0e-3)[0])
        a, _ = rabi_calibration(STAGES, a, shots, rng, f, b)
        b, _, _ = drag_calibration(BETAS, f, a, 4 * shots, rng)
        U = run("gauss", TAU_G, np.pi, b, a * HIDDEN["gain"], f_drive=f)
        trace.append((r, f, a, b, U))
    return (f, a, b), trace


def report(label, f, a, b):
    U = run("gauss", TAU_G, np.pi, b, a * HIDDEN["gain"], f_drive=f)
    e = error_axis(U, XGATE)
    print(f"  {label:>9} {(f - HIDDEN['f_qubit']) * 1e6:+10.1f} {a:9.5f}"
          f" {b:7.4f} {e['X']:+9.1e} {e['Y']:+9.1e} {e['Z']:+9.1e}"
          f" {abs(U[2, 1]) ** 2:10.2e} {gate_error(U, XGATE):11.3e}")


print("The three loops, end to end")
print("=" * 74)
print("  Start: frequency 1.5 MHz high, amplitude 13.7% high, DRAG weight zero.")
print("  Order: frequency, amplitude, DRAG, repeated.")
print(f"\n  {'round':>9} {'df (kHz)':>10} {'amp':>9} {'beta':>7} {'theta_x':>9}"
      f" {'theta_y':>9} {'theta_z':>9} {'leakage':>10} {'gate error':>11}")
F0, A0, B0 = HIDDEN["f_qubit"] + 1.5e-3, 1.0, 0.0
report("start", F0, A0, B0)
rng = np.random.default_rng(31415)
(f_f, a_f, b_f), trace = calibrate(3, 2000, rng, F0, A0, B0)
for r, f, a, b, U in trace:
    report(f"{r}", f, a, b)

b_star = golden_min(lambda bb: gate_error(run("gauss", TAU_G, np.pi, bb,
                                             a_f * HIDDEN["gain"],
                                             f_drive=HIDDEN["f_qubit"]),
                                         XGATE), 0.0, 2.0)
a_star = true_pi_amplitude(b_star, HIDDEN["f_qubit"])
report("oracle", HIDDEN["f_qubit"], a_star, b_star)

n_shots = 3 * (2 * 41 * 2000 + (41 + 3 * 21) * 2000 + 7 * 2 * 8000)
print(f"\n  total shots spent by the three rounds: {n_shots:,}")
```

```text
The three loops, end to end
==========================================================================
  Start: frequency 1.5 MHz high, amplitude 13.7% high, DRAG weight zero.
  Order: frequency, amplitude, DRAG, repeated.

      round   df (kHz)       amp    beta   theta_x   theta_y   theta_z    leakage  gate error
      start    +1500.0   1.00000  0.0000  +4.1e-01  +1.0e-01  +3.4e-16   2.30e-05   2.940e-02
          1       -7.5   0.88164  0.9420  +1.2e-03  -3.4e-04  -2.2e-16   4.42e-06   4.655e-06
          2       -0.2   0.88134  0.9365  +5.5e-05  +5.3e-05  -2.4e-16   4.46e-06   4.457e-06
          3       +0.9   0.88139  0.9464  +2.2e-04  -1.4e-03  -5.1e-18   4.37e-06   4.711e-06
     oracle       +0.0   0.88133  0.9381  -7.4e-06  -1.8e-04  +2.2e-16   4.44e-06   4.448e-06

  total shots spent by the three rounds: 1,452,000
```

**What to notice.** This is the test the chapter was built to run. Three parameters are set wrong on purpose — frequency 1.5 MHz high, amplitude 13.7% high, DRAG weight zero — and three loops that never read the truth recover all three. After one round the frequency is within 7.5 kHz, the amplitude within $3\times10^{-4}$, and the gate error has fallen from $2.94\times10^{-2}$ to $4.66\times10^{-6}$: a factor of 6300, landing within 5% of the $4.45\times10^{-6}$ that this 20 ns Gaussian can achieve at all. Round 2 reaches $4.457\times10^{-6}$, which is the oracle to three digits. Round 3 is slightly worse, and that is shot noise on the $\beta$ fit rather than drift — a real control stack keeps the previous value unless the new one is a statistically significant improvement, precisely to avoid walking around inside the noise.

Read the three error columns across the rows and the structure of the whole section appears: $\theta_x$ falls from $4.1\times10^{-1}$ to $5.5\times10^{-5}$, $\theta_y$ from $1.0\times10^{-1}$ to $5.3\times10^{-5}$, $\theta_z$ was never the problem. And the leakage column does not improve at all — $2.3\times10^{-5}$ to $4.4\times10^{-6}$, and that only because the amplitude changed. **Calibration cannot fix leakage.** Leakage is set by the pulse duration and the anharmonicity, which are design choices, not calibration parameters. The only knobs that touch it are a longer pulse or a better-designed circuit.

The cost is worth stating plainly: 1.45 million shots for three rounds of one qubit's single-qubit gates. On hardware that is minutes, per qubit, and it has to be repeated as the device drifts. Calibration is a scheduled background job consuming a substantial fraction of a machine's duty cycle, and the reason vendors publish "median" error rates rather than per-qubit ones is partly that the per-qubit numbers are only as fresh as the last calibration pass.

* * *

## 4.4 Benchmarking

### The problem with measuring a gate

A calibration loop knows when it has stopped improving, but it does not know how good the gate is. Every measurement of a gate is contaminated: the state was not prepared perfectly, the readout is not faithful, and in Example 4 those two effects together move the readout of an *ideal* $\pi$ pulse from 1 to 0.90. A gate error of $10^{-4}$ cannot be extracted from a measurement with a 10% offset by any amount of averaging, because the offset is a bias and not a variance.

**Randomized benchmarking** solves this by measuring a *decay rate* instead of a value. Draw $m$ random Clifford gates, compose them, append the single Clifford that inverts the product, and measure the probability of returning to the initial state — the *survival probability*. Repeat over many random sequences and average. The model is

$$ F(m) = A\,p^{m} + B $$

and the point of the construction is that state-preparation and measurement errors affect only $A$ and $B$, while the per-gate error affects only $p$. The gate error is then read off as the **error per Clifford**,

$$ \mathrm{EPC} = \left(1 - \frac{1}{d}\right)(1-p) = \frac{1-p}{2} \quad (d = 2) $$

### Why it works

Averaging over the Clifford group **twirls** the error channel. For any channel $\mathcal{E}$,

$$ \bar{\mathcal{E}} = \frac{1}{\lvert \mathbb{C} \rvert}\sum_{C \in \mathbb{C}} \mathcal{C}^{-1} \circ \mathcal{E} \circ \mathcal{C} = \mathcal{D}_{p} $$

is a depolarizing channel with the same average fidelity as $\mathcal{E}$. A sequence of $m$ twirled channels is a depolarizing channel with parameter $p^m$, and a depolarizing channel of that strength produces a survival probability that is *exactly* $A p^m + B$ with $A$ and $B$ fixed by the preparation and the readout alone. SPAM does not enter the exponent because SPAM happens once and the gates happen $m$ times: they are separated by their scaling in $m$, not by any cleverness about calibrating them away.

### Code Example 9: Randomized Benchmarking, and Why It Ignores SPAM

```python
"""Chapter 4, Example 9: randomized benchmarking, and why it ignores SPAM.
Continues from Example 8 (same session)."""
SIG = [np.eye(2, dtype=complex), XGATE, PAULI3["Y"], PAULI3["Z"]]


def ptm(U):
    """Pauli transfer matrix of a single-qubit unitary: real, 4 x 4."""
    return np.array([[0.5 * np.trace(SIG[i] @ U @ SIG[j] @ U.conj().T).real
                      for j in range(4)] for i in range(4)])


def canon(U):
    """Global-phase-free fingerprint of a 2 x 2 unitary, for group lookup."""
    k = int(np.argmax(np.abs(U).ravel() > 1e-9)) if np.any(np.abs(U) > 1e-9) else 0
    z = U.ravel()[k]
    return tuple(np.round((U * np.conj(z) / abs(z)).ravel(), 6))


def clifford_group():
    """The 24 single-qubit Cliffords, closed from H and S up to global phase."""
    S_ = np.array([[1, 0], [0, 1j]], dtype=complex)
    H_ = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    seen, group = {canon(np.eye(2, dtype=complex)): 0}, [np.eye(2, dtype=complex)]
    frontier = [np.eye(2, dtype=complex)]
    while frontier:
        nxt = []
        for U in frontier:
            for G in (S_, H_):
                V = G @ U
                c = canon(V)
                if c not in seen:
                    seen[c] = len(group)
                    group.append(V)
                    nxt.append(V)
        frontier = nxt
    return group, seen


CLIFF, CLIFF_INDEX = clifford_group()
CLIFF_PTM = [ptm(U) for U in CLIFF]
print("Randomized benchmarking")
print("=" * 74)
print(f"  single-qubit Clifford group: {len(CLIFF)} elements")


def depolarizing_ptm(eps):
    """Depolarizing channel whose average gate infidelity is exactly eps."""
    return np.diag([1.0, 1.0 - 2 * eps, 1.0 - 2 * eps, 1.0 - 2 * eps])


def rb_survival(lengths, noise, seqs, rng, init_err, eps01, eps10):
    """Mean survival probability of RB sequences, computed exactly per sequence.

    The noise channel is applied after every Clifford including the inverting
    one, so a sequence of length m carries m + 1 noisy gates.
    """
    out = []
    for m in lengths:
        tot = 0.0
        for _ in range(seqs):
            idx = rng.integers(0, len(CLIFF), size=m)
            M = np.eye(4)
            U = np.eye(2, dtype=complex)
            for i in idx:
                M = noise @ CLIFF_PTM[i] @ M
                U = CLIFF[i] @ U
            inv = CLIFF_INDEX[canon(U.conj().T)]
            M = noise @ CLIFF_PTM[inv] @ M
            z = M[3, 3] * (1.0 - 2 * init_err) + M[3, 0]
            p0 = 0.5 * (1.0 + z)
            tot += p0 * (1 - eps01) + (1 - p0) * eps10
        out.append(tot / seqs)
    return np.array(out)


def fit_rb(lengths, y, rounds=3, npts=2001):
    """Fit y = A p^m + B: linear in (A, B) at fixed p, so scan p and refine.

    Candidates whose A or B falls outside (0, 1) are rejected. Without that
    guard the fit is degenerate as p approaches 1, where A p^m + B tends to a
    constant and any large A with B = -A fits the data equally well: the
    unconstrained scan happily returns A = 2.6e4, B = -2.6e4 and an error per
    Clifford of 1e-10. Both coefficients are probabilities, and saying so is
    what makes the fit well posed.
    """
    lo, hi = 0.5, 1.0
    out = (None, None, None)
    for _ in range(rounds):
        best = (np.inf, None, None, None)
        for p in np.linspace(lo, hi, npts):
            A = np.column_stack([p ** lengths, np.ones_like(y)])
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = float(np.sum((A @ c - y) ** 2))
            if r < best[0] and 0.0 < c[0] < 1.0 and 0.0 < c[1] < 1.0:
                best = (r, p, c[0], c[1])
        _, p, a, b = best
        step = (hi - lo) / (npts - 1)
        lo, hi = p - 2 * step, min(p + 2 * step, 1.0)
        out = (p, a, b)
    return out


LENGTHS = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
print("\nA. The configured error comes back out")
print(f"  {'configured EPC':>15} {'fitted p':>10} {'EPC = (1-p)/2':>15}"
      f" {'ratio':>8} {'A':>8} {'B':>8}")
for eps in (1e-4, 1e-3, 5e-3, 2e-2):
    rng = np.random.default_rng(11)
    y = rb_survival(LENGTHS, depolarizing_ptm(eps), 60, rng, 0.012, 0.020, 0.045)
    p, A, B = fit_rb(LENGTHS, y)
    print(f"  {eps:15.1e} {p:10.6f} {(1 - p) / 2:15.3e}"
          f" {(1 - p) / 2 / eps:8.4f} {A:8.4f} {B:8.4f}")

print("\nB. The same gate, three different readout and initialization errors")
print(f"  {'init err':>9} {'eps01':>7} {'eps10':>7} {'F(m=1)':>9} {'A':>8}"
      f" {'B':>8} {'fitted EPC':>12} {'ratio':>8}")
for ie, e01, e10 in ((0.000, 0.000, 0.000), (0.012, 0.020, 0.045),
                     (0.050, 0.100, 0.150)):
    rng = np.random.default_rng(11)
    y = rb_survival(LENGTHS, depolarizing_ptm(1e-3), 60, rng, ie, e01, e10)
    p, A, B = fit_rb(LENGTHS, y)
    print(f"  {ie:9.3f} {e01:7.3f} {e10:7.3f} {y[0]:9.4f} {A:8.4f} {B:8.4f}"
          f" {(1 - p) / 2:12.3e} {(1 - p) / 2 / 1e-3:8.4f}")

print("\nC. Why the twirl works, and a coherent error instead of a stochastic one")


def clifford_twirl(noise):
    """Average the channel over the Clifford group: (1/24) sum_C C^-1 N C.

    PTMs of unitaries are orthogonal, so the inverse is the transpose. The
    result is always a depolarizing channel, and that is the theorem RB rests
    on: whatever the error was, the sequence average sees only its average
    fidelity.
    """
    return sum(M.T @ noise @ M for M in CLIFF_PTM) / len(CLIFF_PTM)


print(f"  {'over-rotation':>14} {'exact infidelity':>17} {'twirled channel':>16}"
      f" {'off-diagonal':>13} {'RB, 200 seqs':>13} {'ratio':>8}")
for phi in (0.03, 0.05, 0.10, 0.20, 0.30):
    V = expm(-0.5j * phi * XGATE)
    r_exact = (2.0 - abs(np.trace(V)) ** 2 / 2.0) / 3.0
    Nbar = clifford_twirl(ptm(V))
    off = float(np.max(np.abs(Nbar - np.diag(np.diag(Nbar)))))
    rng = np.random.default_rng(11)
    y = rb_survival(LENGTHS, ptm(V), 200, rng, 0.012, 0.020, 0.045)
    p, A, B = fit_rb(LENGTHS, y)
    print(f"  {phi:14.3f} {r_exact:17.3e} {(1 - Nbar[1, 1]) / 2:16.3e}"
          f" {off:13.1e} {(1 - p) / 2:13.3e} {(1 - p) / 2 / r_exact:8.4f}")
```

```text
Randomized benchmarking
==========================================================================
  single-qubit Clifford group: 24 elements

A. The configured error comes back out
   configured EPC   fitted p   EPC = (1-p)/2    ratio        A        B
          1.0e-04   0.999800       1.000e-04   1.0000   0.4562   0.5125
          1.0e-03   0.998000       1.000e-03   1.0000   0.4554   0.5125
          5.0e-03   0.990000       5.000e-03   1.0000   0.4517   0.5125
          2.0e-02   0.960000       2.000e-02   1.0000   0.4380   0.5125

B. The same gate, three different readout and initialization errors
   init err   eps01   eps10    F(m=1)        A        B   fitted EPC    ratio
      0.000   0.000   0.000    0.9980   0.4990   0.5000    1.000e-03   1.0000
      0.012   0.020   0.045    0.9670   0.4554   0.5125    1.000e-03   1.0000
      0.050   0.100   0.150    0.8612   0.3368   0.5250    1.000e-03   1.0000

C. Why the twirl works, and a coherent error instead of a stochastic one
   over-rotation  exact infidelity  twirled channel  off-diagonal  RB, 200 seqs    ratio
           0.030         1.500e-04        1.500e-04       1.1e-17     1.336e-04   0.8907
           0.050         4.166e-04        4.166e-04       1.1e-17     4.442e-04   1.0664
           0.100         1.665e-03        1.665e-03       1.1e-17     1.802e-03   1.0824
           0.200         6.644e-03        6.644e-03       1.2e-17     6.059e-03   0.9119
           0.300         1.489e-02        1.489e-02       1.2e-17     1.451e-02   0.9744
```

**What to notice.** One property of this example frames all three parts: unlike every routine in Section 4.3, `rb_survival` computes each sequence's survival probability *exactly*, as its docstring says, so there is no shot noise anywhere below. Whatever scatter appears is sequence-to-sequence variation in the random Cliffords, not a sampling error, and a real experiment would carry both.

Part A is the correctness test: for configured errors per Clifford of $10^{-4}$, $10^{-3}$, $5\times10^{-3}$ and $2\times10^{-2}$, the fitted EPC returns the configured value with ratio 1.0000 in every row, over four orders of magnitude.

Part B is the claim about SPAM, and it is worth reading as three columns rather than one. Going from a perfect measurement to a 5% initialization error with 10% and 15% readout errors, the raw survival at $m = 1$ moves from 0.9980 to 0.8612 — nearly 14 percentage points, which is 137 times the gate error being measured. Over the same three rows $A$ moves from 0.4990 to 0.3368 and $B$ from 0.5000 to 0.5250. And the fitted error per Clifford is $1.000\times10^{-3}$ in all three rows, to four digits. The SPAM lives entirely in $A$ and $B$; the gate error lives entirely in $p$. That separation is why RB, and not a direct fidelity measurement, is the number every hardware group reports.

Part C makes the twirl explicit and then tests it. Column 3 is the average infidelity of the *exactly* twirled channel, computed by averaging over all 24 Cliffords, and it equals the exact average infidelity of the coherent over-rotation in column 2 to every printed digit, with the off-diagonal entries of the twirled Pauli transfer matrix wiped out to $10^{-17}$. That identity is the theorem RB rests on. Column 5 is what 200 random sequences actually deliver, and it agrees to about 10%.

### What RB does not tell you

Three limitations, each of which has misled someone.

  * **It reports an average over the group, not a worst case, and it cannot tell a coherent error from a stochastic one.** Part C's over-rotations are perfectly unitary and RB reports them as depolarizing errors of the same average fidelity. This matters because coherent errors accumulate in *amplitude* over a deep circuit while stochastic errors accumulate in probability: two gates with the same EPC can behave very differently at depth 1000. The scatter in Part C is also worse for small coherent errors than for stochastic ones of the same size, because the per-sequence survival varies much more from sequence to sequence — the decay is exponential only after the average, not before it.
  * **It does not see leakage.** The twirl is over the Clifford group acting on the qubit subspace; population that leaves that subspace is not described by the model at all. Example 3's $\beta = 2.0030$ pulse has a leakage of $8\times10^{-11}$ and a gate error 334 times worse than the optimum, and a standard RB fit would not distinguish it from a pulse with the same average fidelity and no leakage. Leakage needs its own experiment.
  * **It measures the *Clifford* error, not the error of the gate you care about.** An EPC is an average over the 24 elements, each of which is compiled into one or two physical pulses. Extracting a specific gate's error requires interleaved RB — run the same experiment with the gate of interest inserted between every Clifford and take the ratio of the two decay constants — and that measurement inherits the uncertainty of both fits.

* * *

## 4.5 The Calibration Layer as Software

Everything in this chapter is a program, and it is worth naming the interfaces because they are the ones every SDK exposes, under whatever names.

**Below the gate layer there is a pulse layer, and it is a different kind of object.** A gate list is discrete, hardware-agnostic and exactly composable; a pulse schedule is continuous, device-specific and constrained by sample rates, memory depth and channel counts. The boundary between them is the *calibration table*: a mapping from gate name and qubit index to a waveform plus its parameters. A compiler emits gate names; the control stack looks them up. This is why the same circuit can run before and after a recalibration pass and produce different error rates without a single symbol changing.

**Calibration parameters are state, and state drifts.** A frequency that was right this morning is wrong by tens of kHz this afternoon, and the amplitude follows the temperature of an attenuator. Every production stack therefore runs a hierarchy of checks — a cheap check that decides whether an expensive recalibration is needed, and a scheduler that decides which of a hundred qubits to spend the next minute on. The literature on this is about *decision policy* far more than about physics.

**The pulse layer is where a user can do something a compiler cannot.** A pulse-level API exists so that you can implement a gate the vendor did not provide, run a sequence the gate abstraction cannot express — dynamical decoupling inside an idle, a custom two-qubit gate, a spectroscopy sweep — or measure something about the device that the gate abstraction hides. The corresponding entries in an SDK's documentation are, in vendor-neutral terms, the *pulse schedule builder*, the *backend calibration data*, the *channel map*, and the *experiment library* of Rabi, Ramsey, DRAG and RB routines. Every one of those is a thing this chapter built from scratch, which was the point: the API is a set of names for objects that have already been constructed here.

* * *

## Exercises

#### Exercise 1: The Anharmonicity Sets the Speed Limit

A different device is designed with $E_C/h = 0.180$ GHz and $E_J/h = 19.0$ GHz, giving a smaller $\lvert \alpha \rvert$.

  1. Diagonalize the new transmon and report $f_{01}$, $\alpha/2\pi$ and $r_1$.
  2. Evaluate $\mathrm{sinc}^{2}(\alpha\tau)$ at $\tau = 20$ ns for both devices. Does the comparison tell you which device leaks more with a square pulse? Be careful.
  3. Measure the Gaussian leakage at $\tau = 20$ ns on the new device and compare with the chapter's $1.76\times10^{-5}$.
  4. What duration on the new device restores the old leakage? Interpret the answer as a statement about $\lvert \alpha \rvert \tau$.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(f_{01} = 5.0438\) GHz, \(\alpha/2\pi = -196.64\) MHz, \(r_1 = 1.3861\). The relative anharmonicity falls from 5.71% to 3.90% and \(r_1\) moves closer to \(\sqrt{2}\): the ladder has become more harmonic, which is exactly the direction that makes gates harder.</p>

<p><strong>2.</strong> No, and this is the trap. The old device has \(\alpha\tau = -5.697\) with \(\mathrm{sinc}^2 = 2.07\times10^{-3}\); the new one has \(\alpha\tau = -3.933\) with \(\mathrm{sinc}^2 = 2.88\times10^{-4}\), which is seven times <em>smaller</em> despite the worse anharmonicity. The sinc oscillates, and \(\alpha\tau = -3.93\) happens to sit near one of its zeros. A single-point comparison of an oscillating function is meaningless; only the envelope scaling \((\alpha\tau)^{-2}\) is a physical statement, and comparing two devices at one duration is not a way to test it.</p>

<p><strong>3.</strong> \(7.76\times10^{-5}\) against \(1.76\times10^{-5}\): 4.4 times worse. The Gaussian's tail is exponential in \(\alpha\tau\) rather than power-law, so it is far more sensitive to the anharmonicity than the square-pulse estimate suggests.</p>

<p><strong>4.</strong> 29 ns, at which the measured leakage is \(1.80\times10^{-5}\) — the old device's value to within 2%. The old device had \(\lvert \alpha \rvert \tau = 5.70\) and the new one reaches \(0.19664 \times 29 = 5.70\). The dimensionless product is the figure of merit, so a 31% smaller anharmonicity costs a 45% longer gate, and every decoherence channel gets 45% longer to act. This is the trade that fixes superconducting single-qubit gates in the tens of nanoseconds.</p>

```python
import numpy as np
E2, nop2 = transmon_eigen(19.0, 0.180)
a2 = (E2[2] - E2[1]) - E2[1]
print(f"f01 = {E2[1]:.4f} GHz   alpha/2pi = {a2*1e3:.2f} MHz"
      f"   r1 = {abs(nop2[1, 2]/nop2[0, 1]):.4f}")
for name, aa in (("old", alpha), ("new", a2)):
    x = np.pi * aa * 20.0
    print(f"  {name}: alpha*tau = {aa*20:6.3f}   sinc^2 = {(np.sin(x)/x)**2:.3e}")
LOW2 = np.zeros((NLEV, NLEV), dtype=complex)
for j in range(NLEV - 1):
    LOW2[j, j + 1] = abs(nop2[j, j + 1] / nop2[0, 1])
E_s, AX_s, AY_s, al_s = E, AX, AY, alpha
E, alpha = E2, a2
AX = TWOPI * 0.5 * (LOW2 + LOW2.conj().T)
AY = TWOPI * 0.5 * 1j * (LOW2.conj().T - LOW2)
for tau in (20.0, 29.0):
    U = run("gauss", tau, np.pi, 0.0, 1.0, f_drive=E2[1])
    print(f"  new device, gaussian pi pulse at {tau:4.1f} ns:"
          f" leakage = {abs(U[2, 1])**2:.3e}")
E, AX, AY, alpha = E_s, AX_s, AY_s, al_s
print(f"  equal |alpha|*tau duration: {abs(alpha)*20.0/abs(a2):.1f} ns")
# f01 = 5.0438 GHz   alpha/2pi = -196.64 MHz   r1 = 1.3861
#   old: alpha*tau = -5.697   sinc^2 = 2.067e-03
#   new: alpha*tau = -3.933   sinc^2 = 2.883e-04
#   new device, gaussian pi pulse at 20.0 ns: leakage = 7.757e-05
#   new device, gaussian pi pulse at 29.0 ns: leakage = 1.796e-05
#   equal |alpha|*tau duration: 29.0 ns
```

</details>

#### Exercise 2: Why the Amplitude Loop Stalled

Example 5's amplitude loop stalls at a 7 mrad systematic when $\beta = 0$ and converges when $\beta = 0.9382$.

  1. Run the loop at $\beta = 0, 0.5, 0.9382, 1.4$ with an infinite shot budget and tabulate the stall against the $\theta_y$ of the corresponding pulse. What is the functional relationship?
  2. From that relationship, does the sign of $\theta_y$ matter? What does that tell you about the mechanism?
  3. A colleague proposes to fix the stall by fitting a cosine plus a linear drift instead of a pure cosine. Would that help?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> The stalls are \(+7.10, +1.61, +0.05, +1.68\) mrad at \(\theta_y = +0.1330, +0.0620, -0.0002, -0.0657\) rad. Dividing the stall by \(\theta_y^{2}\) gives 401, 420 and 389 mrad per rad\(^2\) for the three non-zero rows: the stall is quadratic in the out-of-plane error, with a coefficient that is constant to 4%.</p>

<p><strong>2.</strong> No — the stall is positive for both signs of \(\theta_y\). A quadratic, sign-independent contamination is the signature of an <em>axis tilt</em> rather than an added rotation: tilting the rotation axis out of the equatorial plane by an angle \(\epsilon\) shortens the projection of the trajectory onto the measured axis by \(\cos\epsilon \approx 1 - \epsilon^2/2\), which the cosine fit reports as a slightly wrong fringe frequency regardless of which way the tilt went.</p>

<p><strong>3.</strong> No. The contamination is not a drift in the amplitude direction; it is a distortion of the fringe amplitude that is even in \(\theta_y\) and grows with the repetition count. Adding a linear term hands the fit one more degree of freedom with which to absorb noise and leaves the bias in place. The only fix is to remove the physical cause, which is what iterating the loops in Example 8 does.</p>

```python
for b in (0.0, 0.5, 0.9382, 1.4):
    at = true_pi_amplitude(b, f01)
    e = 1.0
    for n in STAGES:
        e, _ = rabi_calibration((n,), e, None, None, f01, b)
    ty = error_axis(run("gauss", TAU_G, np.pi, b), XGATE)["Y"]
    print(f"beta = {b:6.4f}  theta_y = {ty:+8.4f} rad"
          f"  stall = {(e - at)*np.pi*1e3:+6.2f} mrad"
          f"  stall/theta_y^2 = {(e - at)*np.pi*1e3/ty**2:7.1f}")
# beta = 0.0000  theta_y =  +0.1330 rad  stall =  +7.10 mrad  stall/theta_y^2 =   401.1
# beta = 0.5000  theta_y =  +0.0620 rad  stall =  +1.61 mrad  stall/theta_y^2 =   420.0
# beta = 0.9382  theta_y =  -0.0002 rad  stall =  +0.05 mrad  stall/theta_y^2 = 1263089.4
# beta = 1.4000  theta_y =  -0.0657 rad  stall =  +1.68 mrad  stall/theta_y^2 =   388.6
```

</details>

#### Exercise 3: Fitting a Ramsey Fringe With the Wrong Envelope

Example 6 fits a Gaussian envelope $\exp[-(T/T_2^{\ast})^2]$, which is the correct form for quasi-static noise.

  1. Refit the same scan with an exponential envelope $\exp(-T/T_2^{\ast})$. How much does the fringe frequency move? How much does the time constant move?
  2. Why is the frequency robust to the envelope model and the time constant not?
  3. [Introduction to Quantum Hardware, Chapter 1](<../quantum-hardware-introduction/chapter-1.html>) converts $T_2^{\ast}$ into a static frequency spread through $\sigma = \sqrt{2}/T_2^{\ast}$. Do that for both fits and state what the modelling choice costs in inferred material properties.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> The fringe frequency moves from 3507.8 to 3503.1 kHz — 4.7 kHz, or 0.13% — while the time constant moves from 1100 ns to 900 ns, an 18% change in the opposite direction to what a casual reading would expect.</p>

<p><strong>2.</strong> The frequency is carried by the <em>phase</em> of the oscillation, and the envelope is a real, positive, slowly varying multiplier that to first order does not move zero crossings. The time constant <em>is</em> the envelope, so a wrong functional form maps directly onto it. A Gaussian and an exponential differ most in curvature near \(T = 0\), which is precisely where the signal-to-noise is best and therefore where the fit is most strongly weighted.</p>

<p><strong>3.</strong> \(\sigma/2\pi = \sqrt{2}/(2\pi T_2^{\ast})\) gives 205 kHz from the Gaussian fit against the 200 kHz configured in <code>HIDDEN</code>, and 250 kHz from the exponential fit — a 25% error in an inferred property of the host material, caused entirely by a line in a fitting routine. This is the concrete reason the sister course insists that a quoted \(T_2^{\ast}\) is uninterpretable without the fit model beside it.</p>

```python
rng = np.random.default_rng(90210)
yr = ramsey(HIDDEN["f_qubit"] + 1.5e-3 + F_ART, amp_cal, 0.0, DELAYS, 2000, rng)


def fit_exp(delays, y, nu_hi, rounds=2, npts=401):
    lo, hi, out = 0.0, nu_hi, (None, None)
    for _ in range(rounds):
        best = (np.inf, None, None)
        for t2 in np.linspace(300.0, 3000.0, 28):
            dec = np.exp(-delays / t2)
            for nu in np.linspace(lo, hi, npts):
                A = np.column_stack([np.ones_like(delays),
                                     dec * np.cos(TWOPI * nu * delays),
                                     dec * np.sin(TWOPI * nu * delays)])
                c, *_ = np.linalg.lstsq(A, y, rcond=None)
                r = float(np.sum((A @ c - y) ** 2))
                if r < best[0]:
                    best = (r, nu, t2)
        _, nu, t2 = best
        step = (hi - lo) / (npts - 1)
        lo, hi = nu - 2 * step, nu + 2 * step
        out = (nu, t2)
    return out


for name, f in (("gaussian", fit_ramsey), ("exponential", fit_exp)):
    nu, t2 = f(DELAYS, yr, 5.0e-3)
    print(f"{name:>12}: nu = {nu*1e6:7.1f} kHz   T2* = {t2:6.0f} ns"
          f"   sigma/2pi = {np.sqrt(2)/(TWOPI*t2)*1e6:5.0f} kHz")
#     gaussian: nu =  3507.8 kHz   T2* =   1100 ns   sigma/2pi =   205 kHz
#  exponential: nu =  3503.1 kHz   T2* =    900 ns   sigma/2pi =   250 kHz
```

</details>

#### Exercise 4: Two Ways to Misread a Randomized Benchmarking Fit

  1. Benchmark a coherent over-rotation of 0.01 rad and fit it both without and with the $0 < A < 1$, $0 < B < 1$ guard. Compare both against the exact average infidelity $2\sin^{2}(\phi/2)/3$.
  2. Suppose a gate has an error per Clifford of $10^{-3}$ of which half is leakage out of the qubit subspace. What would the RB model of this chapter report, and what would be missing?
  3. An interleaved RB experiment gives $p_{\mathrm{ref}} = 0.9980$ and $p_{\mathrm{int}} = 0.9955$. What is the interleaved gate's error, and what is the leading systematic in that number?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Unconstrained, the scan returns \(A = 2.58\times10^{4}\), \(B = -2.58\times10^{4}\) and an error per Clifford of \(2.8\times10^{-10}\) against an exact \(1.67\times10^{-5}\) — wrong by five orders of magnitude. As \(p \to 1\) the model \(Ap^m + B\) degenerates to a constant and any large \(A\) with \(B \approx -A\) fits a nearly flat data set equally well. With the guard the fit returns \(7.5\times10^{-6}\), still a factor of 2.2 low and sitting on the \(B \to 0\) boundary: the guard makes the problem identifiable but 200 sequences cannot resolve a \(10^{-5}\) coherent error. Both coefficients being probabilities is a physical constraint, and imposing it is not a numerical convenience.</p>

<p><strong>2.</strong> It would report an error per Clifford of \(10^{-3}\), correct as an average infidelity, and would say nothing about the leakage. How the leaked population enters depends on how the readout classifies \(\lvert 2 \rangle\): if \(\lvert 2 \rangle\) reads as 1 the leakage looks like ordinary depolarization and hides inside \(p\); if it reads as 0 the decay acquires a second exponential that a single-exponential fit averages over. Either way the operationally crucial fact — that the error is outside the code space and therefore outside what an error-correcting code can fix — is invisible, and a separate leakage experiment is required.</p>

<p><strong>3.</strong> \(r = (1 - p_{\mathrm{int}}/p_{\mathrm{ref}})(1 - 1/d) = (1 - 0.9955/0.9980)/2 = 1.25\times10^{-3}\). The leading systematic is that the reference and interleaved sequences do not sample the same twirl: inserting the gate of interest changes which Cliffords sit next to which and how they compile, so the two decay constants are not measurements of the same channel. Published interleaved-RB uncertainties therefore carry a systematic comparable with the quantity measured whenever \(p_{\mathrm{int}}\) is close to \(p_{\mathrm{ref}}\) — which is the interesting regime.</p>

```python
V = expm(-0.005j * XGATE)
rq = np.random.default_rng(11)
yq = rb_survival(LENGTHS, ptm(V), 200, rq, 0.012, 0.020, 0.045)
lo, hi = 0.5, 1.0
for _ in range(3):
    best = (np.inf, None, None, None)
    for p in np.linspace(lo, hi, 2001):
        A = np.column_stack([p ** LENGTHS, np.ones_like(yq)])
        c, *_ = np.linalg.lstsq(A, yq, rcond=None)
        r = float(np.sum((A @ c - yq) ** 2))
        if r < best[0]:
            best = (r, p, c[0], c[1])
    _, p, a, b = best
    step = (hi - lo) / 2000
    lo, hi = p - 2 * step, min(p + 2 * step, 1.0)
print(f"  unconstrained: A = {a:.3e}  B = {b:.3e}  EPC = {(1-p)/2:.3e}")
p2, a2f, b2 = fit_rb(LENGTHS, yq)
print(f"  guarded      : A = {a2f:.4f}  B = {b2:.4f}  EPC = {(1-p2)/2:.3e}")
print(f"  exact average infidelity = {(2 - abs(np.trace(V))**2/2)/3:.3e}")
print(f"  interleaved gate error = {(1 - 0.9955/0.9980)/2:.3e}")
#   unconstrained: A = 2.578e+04  B = -2.578e+04  EPC = 2.813e-10
#   guarded      : A = 0.9687  B = 0.0000  EPC = 7.512e-06
#   exact average infidelity = 1.667e-05
#   interleaved gate error = 1.253e-03
```

</details>

#### Exercise 5: Designing a Recalibration Schedule

A device drifts: the qubit frequency by 40 kHz per hour and the amplifier gain by 0.15% per hour. The calibrated gate error is $4.5\times10^{-6}$ and the target is to keep it below $1\times10^{-4}$.

  1. From the sensitivities in this chapter, how long can each parameter be left uncalibrated?
  2. Which loop sets the recalibration cadence?
  3. A round of all three loops costs 480 000 shots at 5000 shots per second including overhead. What fraction of the device's time goes to calibrating one qubit's single-qubit gates at that cadence, and what does that imply for a 1000-qubit machine?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> A gain drift of \(g\) produces a rotation-angle error \(\theta_x \approx \pi g\), and an average gate error \(\approx \theta_x^2/6\) for a small coherent rotation error. Reaching \(10^{-4}\) needs \(\theta_x = 0.0245\) rad, i.e. \(g = 0.78\%\), which at 0.15% per hour is 5.2 hours. For the frequency, the chapter measured a gate error of \(4.96\times10^{-4}\) at a 1.5 MHz detuning; scaling quadratically, \(10^{-4}\) is reached at 674 kHz, which at 40 kHz per hour is 17 hours.</p>

<p><strong>2.</strong> The amplitude, by a factor of three, which is the general case on superconducting hardware: amplitude and phase calibration are run far more often than frequency calibration, and frequency calibration is often triggered rather than scheduled.</p>

<p><strong>3.</strong> 480 000 shots at 5000 per second is 96 seconds, every 5.2 hours, so 0.51% of the device's time per qubit. On 1000 qubits that is 5.1 times the available time if the qubits are calibrated one at a time — which is why calibration is heavily parallelized across qubits, why cheap checks that decide whether a full recalibration is needed matter so much, and why two-qubit gate calibration (not covered here, and considerably more expensive per pair) is the real budget problem. The arithmetic here is the honest reason a large machine's error rates are not all freshly measured.</p>

```python
th = np.sqrt(6 * 1e-4)
print(f"  theta_x budget {th:.4f} rad -> gain {th/np.pi*100:.2f} %"
      f" -> {th/np.pi*100/0.15:.1f} h")
print(f"  frequency budget {1500*np.sqrt(1e-4/4.96e-4):.0f} kHz"
      f" -> {1500*np.sqrt(1e-4/4.96e-4)/40:.0f} h")
duty = 480000 / 5000 / 3600 / (th / np.pi * 100 / 0.15)
print(f"  duty per qubit {duty*100:.2f} %  -> x1000 = {duty*1000:.1f}")
#   theta_x budget 0.0245 rad -> gain 0.78 % -> 5.2 h
#   frequency budget 674 kHz -> 17 h
#   duty per qubit 0.51 %  -> x1000 = 5.1
```

</details>

* * *

## Summary

### Key Takeaways

**1\. A gate is a calibrated pulse area, and $Z$ is free**

  * In the rotating frame a resonant drive rotates the qubit through $\theta = \int \Omega\,dt$ about an axis set by the carrier phase; the envelope shape is free and is the pulse layer's only degree of freedom.
  * A $Z$ rotation is a redefinition of the phase of all later pulses: zero time, zero error. This is the physical payoff of a compiler that pushes $Z$ rotations to the end.
  * The transmon's third level sits $\lvert \alpha \rvert = 285$ MHz away with a *larger* matrix element, $r_1 = 1.3726$ against the harmonic $\sqrt{2}$.

**2\. Leakage is a Fourier coefficient, so shaping changes the exponent**

  * $a_{1\to2} \simeq -(i r_1/2)\tilde{\Omega}_x(\alpha)$; a square pulse's sinc tail gives leakage $\propto \tau^{-1.97}$ as measured, a Gaussian's gives $\tau^{-4.09}$.
  * At $\tau = 20$ ns the two differ by a factor of 205; doubling $\tau$ buys 3.9$\times$ for a square pulse and 17$\times$ for a Gaussian.
  * Shaping helps only where the pulse is long enough to be shaped: at $\tau = 8$ ns the Gaussian is *worse*, because at fixed area its peak amplitude is 117 MHz against the square pulse's 62 MHz.

**3\. DRAG fixes a phase, and its two optima are different**

  * $\Omega_y = -\beta\dot{\Omega}_x/2\alpha$; at $\tau = 20$ ns the gate-error optimum is $\beta = 0.9382$ and improves the error 261$\times$, from $2.97\times10^{-3}$ to $1.14\times10^{-5}$.
  * The leakage optimum is $\beta = 2.0030$, where leakage is $8\times10^{-11}$ and the gate is **334 times worse**. Tuning DRAG by minimizing leakage optimizes the wrong quantity.
  * $\beta^{\ast}$ is not the analytic 1 and drifts from 0.912 to 0.941 over $\tau = 8$ to 40 ns, which is why it is a calibrated parameter.

**4\. A calibration loop is a sequence, a fit and an update — and the loops are coupled**

  * Error amplification: 81 repetitions of a $\pi$ pulse sharpen the amplitude estimate by about 50$\times$ over a single pulse at the same shot budget.
  * Cancellation by construction: the DRAG ping-pong observable is a *difference* of two sequences, from which the 2.0% and 4.5% readout errors cancel exactly.
  * The amplitude loop stalls at a 7 mrad systematic when $\beta$ is wrong, and the stall is identical at an infinite shot budget — the diagnostic that separates a systematic from a statistic. With $\beta$ right it reaches 0.11 mrad.
  * Ramsey needs an idle because a $\pi$ pulse produces no $Z$ error at all; two deliberately detuned scans recover the sign and give $(\nu_-+\nu_+)/2 = f_{\mathrm{art}}$ as a free consistency check.
  * End to end: frequency 1.5 MHz high, amplitude 13.7% high, $\beta = 0$ — recovered in one round to a gate error of $4.66\times10^{-6}$ against the achievable $4.45\times10^{-6}$, a 6300-fold improvement, for 1.45 million shots.

**5\. Calibration cannot fix leakage, and two wrongs can look right**

  * Across all three rounds the leakage stays at $4\times10^{-6}$: it is set by $\tau$ and $\alpha$, which are design parameters, not calibration parameters.
  * The gate error was *lower* at the wrong frequency ($5.0\times10^{-4}$) than at the right one ($3.0\times10^{-3}$), because a detuning error partially cancelled the missing DRAG. Calibrate against sequences that isolate parameters, never against a single figure of merit.

**6\. Randomized benchmarking separates gate error from SPAM by scaling in $m$**

  * $F(m) = Ap^m + B$ with $\mathrm{EPC} = (1-p)/2$; configured errors of $10^{-4}$ to $2\times10^{-2}$ come back with ratio 1.0000.
  * Across SPAM settings the raw survival at $m=1$ moves 14 percentage points and $A$ moves from 0.499 to 0.337, while the fitted EPC is $1.000\times10^{-3}$ in every case.
  * The Clifford twirl of any single-qubit channel is exactly depolarizing — verified to $10^{-17}$ in the off-diagonal Pauli transfer matrix entries — which is the theorem the method rests on.
  * RB is blind to the coherence of the error, blind to leakage, and reports a Clifford average rather than a specific gate. The fit is also degenerate as $p \to 1$ unless $A$ and $B$ are constrained to be probabilities.

**Practical implications**

  * Quote $\lvert \alpha \rvert \tau$ before quoting a leakage number, and quote how many levels the simulation kept: three levels versus five moves this chapter's 20 ns leakage by 3%.
  * Never tune a DRAG weight against leakage. Tune it against an error-sensitive difference of sequences and check the resulting gate error separately.
  * Test every calibration routine by hiding a known error from it, and run the routine once with an infinite shot budget: if the residual does not shrink, more shots will not help.
  * Report a $T_2^{\ast}$ only together with the envelope model used to fit it. A Gaussian-versus-exponential choice moved the inferred static spread from 205 to 250 kHz in Exercise 3, against a true 200 kHz.
  * When you read an error-per-gate figure, ask whether it is a Clifford average or an interleaved measurement, whether leakage was measured separately, and how long ago the device was calibrated.

### Where This Leads

The pulse layer is the bottom of the stack, and reaching it completes the descent this course started from an algorithm. What remains is the other direction: given that gates have the error rates this chapter has been measuring, what can software do about them? [Chapter 5](<chapter-5.html>) implements the answer that near-term hardware actually uses — readout-error mitigation, zero-noise extrapolation by gate folding, probabilistic error cancellation — measures both what each one buys and what each one costs in samples, and then builds the resource-estimation pipeline that says what the fault-tolerant alternative would require instead. Both halves of that comparison get numbers, because the honest statement about error mitigation is that it is genuinely load-bearing today *and* exponentially expensive, and neither half of that sentence should be dropped.

[← Chapter 3: Transpilation — Mapping to Connectivity](<chapter-3.html>) [Chapter 5: Error Mitigation as Software, and Resource Estimation →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The transmon parameters, pulse durations, drift rates, readout error rates and shot budgets in this chapter are representative literature-scale values chosen so that the calibration and benchmarking arithmetic can be followed and checked; they are not device specifications and must be verified against primary sources before use in a proposal or a paper.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
