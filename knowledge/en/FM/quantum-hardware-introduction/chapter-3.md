---
title: "Chapter 3: Trapped Ions"
chapter_title: "Chapter 3: Trapped Ions"
subtitle: ⚛️ Paul Traps, Shared Phonons, the Mølmer-Sørensen Gate, and Anomalous Heating
reading_time: 45-50 minutes
difficulty: Advanced
code_examples: 7
exercises: 6
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-hardware-introduction/chapter-3.html>) | Last sync: 2026-08-13

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to Quantum Hardware](<index.html>) > Chapter 3

Chapter 2 built a qubit. Trapped ions do the opposite: they *find* one. An ion of $^{40}$Ca$^+$ has two long-lived internal states whose splitting is a property of the element, identical in every ion in the universe, with no fabrication tolerance and no device-to-device variation. There is nothing to design and nothing to get wrong about the qubit itself. The engineering is entirely in holding the ion still, addressing it, and coupling two of them together.

That shifts where the difficulty lives, and it is worth stating the shift plainly, because it is the reason both platforms are still being pursued. A superconducting qubit is easy to hold and hard to make identical. A trapped ion is trivially identical and hard to hold: it needs ultra-high vacuum, a radio-frequency trap whose stability is a non-trivial dynamical question, laser cooling to the motional ground state, and a set of laser beams stable enough to define the phase of a quantum gate. The payoff is coherence times of seconds instead of microseconds, and — because the ions share their vibrational modes — a two-qubit gate between *any* pair, not just neighbours.

This chapter follows the physics in the order the experiment does: trap the ion (3.1-3.2), cool it (3.3), gate it (3.4-3.5), and then confront what actually limits it (3.6). The numerical work is heavier than in Chapter 2 because the objects are less familiar: we compute the Mathieu stability diagram from Floquet theory rather than looking it up, we derive the normal modes of an ion crystal and find the zigzag instability numerically, and we verify the Mølmer-Sørensen gate against an exact Magnus expansion — and then break the Lamb-Dicke approximation to see what it was hiding.

**Units and conventions.** Motional frequencies are quoted in MHz, gate detunings in kHz, and laser linewidths in MHz. Energies in Hamiltonians are frequencies ($E/h$); $\hbar = 1$ inside every simulation, with SI units restored when a laboratory quantity is needed. Qubit ordering and gate symbols follow [Introduction to Quantum Computing](<../quantum-computing-introduction/index.html>) (big-endian, $X$, $Y$, $Z$, $H$, CNOT); $T_1$, $T_2$ and $T_2^\ast$ keep the definitions from Chapter 1. The harmonic-oscillator and quantized-motion machinery used from 3.3 onward is developed in [Introduction to Quantum Mechanics](<../quantum-mechanics/index.html>).

## Learning Objectives

After completing this chapter, you will be able to:

  * Reduce the equation of motion in a radio-frequency quadrupole trap to the canonical Mathieu form, compute the stability region by Floquet analysis, and explain why $q < 0.908$
  * Separate secular motion from micromotion, compute the secular frequency from the characteristic exponent $\beta$, and quantify how a stray field turns micromotion into a spectroscopic problem
  * Explain the difference between hyperfine and optical qubits, and what a "clock" transition buys
  * Compute the equilibrium positions and normal modes of an ion crystal, locate the linear-to-zigzag instability, and explain why transverse mode crowding bounds the gate speed of a long chain
  * Compute exact sideband Rabi matrix elements, simulate a sideband spectrum, and extract $\bar{n}$ from a sideband ratio
  * State the Doppler cooling limit, integrate the rate equations of resolved-sideband cooling, and explain why the steady state is $(\Gamma/4\omega)^2$
  * Derive the Mølmer-Sørensen gate from a Magnus expansion that terminates, verify the loop-closure and gate conditions numerically, and quantify what breaks beyond the Lamb-Dicke approximation
  * Convert an electric-field noise spectral density into a heating rate and a gate-error budget, and state the evidence that anomalous heating is a surface-materials problem

* * *

## 3.1 The Physics of a Paul Trap

### Earnshaw's obstacle, and the way around it

You cannot trap a charged particle with static electric fields. Laplace's equation forbids it: $\nabla^2\phi = 0$ means the potential has no local minimum in free space, so any static configuration that confines in two directions expels in the third. This is Earnshaw's theorem, and it rules out the obvious approach entirely.

Wolfgang Paul's solution is to make the saddle rotate. Apply a radio-frequency voltage to a quadrupole electrode geometry, so the potential is

$$ \phi(x, y, t) = \frac{U + V\cos\Omega t}{2 r_0^2}\left(x^2 - y^2\right) $$

At any instant this confines along one axis and expels along the other. But the expulsion direction flips every half RF cycle, and if the flipping is fast compared with the ion's response the net effect is confinement in both. The ion executes a small, fast wiggle at $\Omega$ — the **micromotion** — superposed on a slow, large **secular** oscillation in an effective time-averaged potential. Whether that actually works, and for which parameters, is a genuine dynamical question with a genuine answer.

### The Mathieu equation

Newton's equation for the radial motion, with $\xi = \Omega t/2$, becomes the canonical Mathieu equation:

$$ \frac{d^2u}{d\xi^2} + \left(a - 2q\cos 2\xi\right)u = 0 $$

with the dimensionless trap parameters

$$ q = \frac{2eV}{m r_0^2 \Omega^2}, \qquad a = -\frac{4eU}{m r_0^2 \Omega^2} $$

(sign conventions vary between references; this chapter uses the ones above throughout.) As written, these are the parameters for the $x$ direction of a linear trap, whose RF potential is proportional to $x^2 - y^2$: the $y$ equation is the same one with $q \to -q$ and $a \to -a$. Stability depends on $q$ only through $q^2$, so the diagram below is symmetric in $q$, but it is *not* symmetric in $a$ — a DC voltage that helps one radial direction hurts the other, which is why $|a|$ is always kept small.

This is a linear equation with a periodic coefficient, so Floquet theory applies. Integrate two independent solutions over one period of the coefficient ($\xi$ from 0 to $\pi$, since $\cos 2\xi$ has period $\pi$) to build the $2\times2$ monodromy matrix $M$. Because the equation has no damping, $\det M = 1$ exactly, so the eigenvalues are $\lambda^{\pm1}$ with $\lambda\lambda^{-1} = 1$. Two cases:

  * $|\mathrm{tr}\,M| < 2$: eigenvalues $e^{\pm i\pi\beta}$ on the unit circle, solutions bounded — **the ion is trapped**.
  * $|\mathrm{tr}\,M| > 2$: eigenvalues real and reciprocal, one of them larger than 1, solutions grow exponentially — **the ion is lost**.

The trapped case defines the **characteristic exponent** $\beta$ through $\cos\pi\beta = \mathrm{tr}\,M/2$, and the secular frequency is

$$ \omega_\mathrm{sec} = \frac{\beta\Omega}{2} $$

For small $q$ the pseudopotential approximation gives $\beta \approx \sqrt{a + q^2/2}$, and at $a = 0$ simply $\beta \approx q/\sqrt{2}$.

### Code Example 1: The Stability Diagram, Computed

The stability diagram of a Paul trap is usually presented as a figure to be looked up. It is a twenty-line calculation, and doing it yourself makes the physics — and the meaning of the famous number 0.908 — concrete.

```python
"""Chapter 3, Example 1: the Mathieu stability diagram, computed with Floquet
theory rather than looked up.

Canonical form, with xi = Omega t / 2:
    d^2 u / d xi^2 + (a - 2 q cos 2 xi) u = 0
The coefficient has period pi in xi, so one period of the map is enough."""
import numpy as np
from scipy.integrate import solve_ivp


def monodromy(a, q):
    """Transfer matrix over one period xi: 0 -> pi of the Mathieu equation."""
    def rhs(xi, y):
        u1, v1, u2, v2 = y
        k = a - 2.0 * q * np.cos(2.0 * xi)
        return [v1, -k * u1, v2, -k * u2]
    sol = solve_ivp(rhs, [0.0, np.pi], [1.0, 0.0, 0.0, 1.0],
                    rtol=1e-11, atol=1e-13, dense_output=False)
    u1, v1, u2, v2 = sol.y[:, -1]
    return np.array([[u1, u2], [v1, v2]])


def stable(a, q):
    """Bounded solutions exist iff |trace of the monodromy matrix| < 2."""
    return abs(np.trace(monodromy(a, q))) < 2.0


def beta_exact(a, q):
    """Characteristic exponent from cos(pi beta) = tr(M)/2."""
    t = np.trace(monodromy(a, q)) / 2.0
    return np.arccos(np.clip(t, -1.0, 1.0)) / np.pi


# --- check: the determinant must be 1 (Liouville) -----------------------
M = monodromy(0.0, 0.5)
print(f"determinant of the monodromy matrix at (a, q) = (0, 0.5): "
      f"{np.linalg.det(M):.12f}   (Liouville: exactly 1)")
print()

# --- the first stability boundary at a = 0 ------------------------------
lo, hi = 0.5, 1.5
for _ in range(60):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if stable(0.0, mid) else (lo, mid)
print(f"a = 0: the ion is trapped for q < {0.5 * (lo + hi):.6f}")
print()

# --- the stability region -----------------------------------------------
print(f"{'q':>7}{'a_min':>11}{'a_max':>11}{'beta at a=0':>14}"
      f"{'sqrt(q^2/2)':>14}{'error':>10}")
for q in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    def scan(sign):
        """March away from a = 0 until stability is lost, then bisect.

        A plain bisection on [0, +-1.2] would fail: above the first unstable
        band there is a second stable band, and the bisection would walk into
        it and report a boundary that is not the edge of the region we are in.
        """
        step = 1.0e-2
        lo2 = 0.0
        while stable(lo2 + sign * step, q) and abs(lo2) < 1.5:
            lo2 += sign * step
        hi2 = lo2 + sign * step
        for _ in range(40):
            mid = 0.5 * (lo2 + hi2)
            lo2, hi2 = (mid, hi2) if stable(mid, q) else (lo2, mid)
        return 0.5 * (lo2 + hi2)
    amin, amax = scan(-1.0), scan(+1.0)
    b = beta_exact(0.0, q)
    approx = np.sqrt(q ** 2 / 2.0)
    print(f"{q:>7.2f}{amin:>11.5f}{amax:>11.5f}{b:>14.6f}"
          f"{approx:>14.6f}{abs(b - approx) / b * 100:>9.3f}%")
print()

# --- the same numbers in laboratory units ------------------------------
u = 1.66053906660e-27
e = 1.602176634e-19
mass = 40.078 * u                # Ca-40
r0 = 0.5e-3                      # m, ion-electrode distance
frf = 20.0e6                     # Hz, RF drive
Om = 2.0 * np.pi * frf
print(f"A linear Paul trap for Ca+ (m = {mass / u:.1f} u), r0 = "
      f"{r0 * 1e3:.1f} mm, Omega/2pi = {frf / 1e6:.0f} MHz.")
print(f"{'V_RF (V)':>10}{'q':>8}{'beta (exact)':>14}"
      f"{'f_sec (MHz)':>13}{'q/sqrt(2) est':>15}{'depth (eV)':>12}")
for VRF in [100.0, 245.0, 400.0, 600.0, 742.0]:
    q = 2.0 * e * VRF / (mass * r0 ** 2 * Om ** 2)
    if q >= 0.908:
        print(f"{VRF:>10.0f}{q:>8.4f}{'  UNSTABLE':>14}")
        continue
    b = beta_exact(0.0, q)
    wsec = b * Om / 2.0
    depth = 0.5 * mass * wsec ** 2 * r0 ** 2 / e
    print(f"{VRF:>10.0f}{q:>8.4f}{b:>14.6f}{wsec / 2 / np.pi / 1e6:>13.4f}"
          f"{q / np.sqrt(2.0):>15.6f}{depth:>12.2f}")
print()
q = 0.3
b = beta_exact(0.0, q)
wsec = b * Om / 2.0
depth = 0.5 * mass * wsec ** 2 * r0 ** 2
kB = 1.380649e-23
print(f"At q = {q:.1f}: f_sec = {wsec / 2 / np.pi / 1e6:.4f} MHz, trap depth "
      f"{depth / e:.2f} eV = {depth / kB:.2e} K.")
```

```text
determinant of the monodromy matrix at (a, q) = (0, 0.5): 1.000000000000   (Liouville: exactly 1)

a = 0: the ion is trapped for q < 0.908046

      q      a_min      a_max   beta at a=0   sqrt(q^2/2)     error
   0.05   -0.00125    0.94969      0.035373      0.035355    0.049%
   0.10   -0.00499    0.89877      0.070850      0.070711    0.196%
   0.20   -0.01991    0.79512      0.142551      0.141421    0.793%
   0.30   -0.04457    0.68917      0.216059      0.212132    1.818%
   0.50   -0.12177    0.47065      0.373744      0.353553    5.402%
   0.70   -0.23317    0.24391      0.563066      0.494975   12.093%
   0.90   -0.37456    0.00958      0.915911      0.636396   30.518%

A linear Paul trap for Ca+ (m = 40.1 u), r0 = 0.5 mm, Omega/2pi = 20 MHz.
  V_RF (V)       q  beta (exact)  f_sec (MHz)  q/sqrt(2) est  depth (eV)
       100  0.1220      0.086493       0.8649       0.086240        1.53
       245  0.2988      0.215168       2.1517       0.211289        9.49
       400  0.4878      0.363556       3.6356       0.344961       27.09
       600  0.7318      0.599230       5.9923       0.517442       73.60
       742  0.9050      0.947949       9.4795       0.639903      184.20

At q = 0.3: f_sec = 2.1606 MHz, trap depth 9.57 eV = 1.11e+05 K.
```

**What to notice.** The determinant of the monodromy matrix is 1 to twelve digits, which is not a numerical accident: it is Liouville's theorem for a Hamiltonian system, and it is the cheapest available check that the integration is correct. Any drift there means the tolerances are too loose.

The boundary at $a = 0$ comes out at $q = 0.908046$, against the textbook 0.90800. The number is now yours rather than borrowed. It is the practical ceiling on how hard a Paul trap can be driven: beyond it the ion is not weakly confined, it is *expelled*, exponentially.

The stability region shrinks as $q$ grows: at $q = 0.05$ the DC parameter $a$ may range over $[-0.001, 0.950]$, and at $q = 0.9$ only over $[-0.375, 0.010]$. Traps are therefore run at modest $q$, typically 0.1 to 0.3, which leaves room for the DC electrodes that provide axial confinement and micromotion compensation.

The comparison of $\beta$ with $q/\sqrt{2}$ says how far the pseudopotential picture can be trusted: 0.05% error at $q = 0.05$, 1.8% at $q = 0.3$, 30% at $q = 0.9$. At the working point $q = 0.3$ the approximation is good to two significant figures, which is why the rest of this chapter can forget the RF drive and treat the ion as sitting in a static harmonic well.

The laboratory table converts to volts. A 0.5 mm trap driven at 20 MHz needs 245 V of RF amplitude for $q = 0.30$, giving a 2.16 MHz secular frequency and a trap depth of 9.6 eV — which is $1.1 \times 10^5$ K. A millikelvin ion in a $10^5$ K well is not going anywhere. What limits the storage time is not the depth but chemistry: collisions with background gas, and reactions that turn Ca$^+$ into CaH$^+$. Hence ultra-high vacuum, and hence the fact that a single ion can be held for days.

### Micromotion

The pseudopotential is an approximation, and what it discards is physically important. The exact Floquet solution is $u(\xi) = e^{i\beta\xi}P(\xi)$ with $P$ periodic, so it contains not only the secular frequency $\beta\Omega/2$ but sidebands at $\beta\Omega/2 \pm n\Omega$. The $n = \pm1$ components are the micromotion, with amplitude $q/2$ relative to the secular amplitude.

For an ion sitting exactly at the RF null this is a small effect and mostly harmless. The problem is **excess micromotion**: a stray DC field — from a charged patch of insulator on an electrode, from a stray electron somewhere in the vacuum can — pushes the ion off the null, into a region where the RF field does not vanish. The ion is then driven at $\Omega$ with an amplitude set by its *displacement*, not by its temperature. Cooling does not help.

The consequence is spectroscopic. An ion oscillating at $\Omega$ along the laser direction sees a phase-modulated field, and the modulation index is $k \cdot x_\mathrm{micro}$. The spectrum acquires micromotion sidebands at $\pm\Omega$ with a carrier-to-sideband ratio given by Bessel functions, and the carrier itself is weakened. Nulling those sidebands is how the stray field is measured and compensated — and it has to be redone, because the stray field drifts.

### Code Example 2: Micromotion, Exactly

```python
"""Chapter 3, Example 2: micromotion.  The exact Floquet solution of the Mathieu
equation against the pseudopotential (secular) approximation that the rest of
the chapter uses."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv

TWOPI = 2.0 * np.pi


def monodromy(q, a=0.0):
    def rhs(xi, y):
        k = a - 2.0 * q * np.cos(2.0 * xi)
        return [y[1], -k * y[0], y[3], -k * y[2]]
    s = solve_ivp(rhs, [0.0, np.pi], [1.0, 0.0, 0.0, 1.0],
                  rtol=1e-12, atol=1e-14)
    u1, v1, u2, v2 = s.y[:, -1]
    return np.array([[u1, u2], [v1, v2]])


def floquet_harmonics(q, a=0.0, nmax=4, npts=4001):
    """Fourier content of the Floquet solution u = exp(i beta xi) P(xi).

    P has period pi, so u = sum_n C_n exp(i(beta + 2n) xi): in real time the
    component n has frequency beta*Omega/2 + n*Omega.  n = 0 is the secular
    motion, n = +-1 the micromotion at the drive frequency.
    """
    M = monodromy(q, a)
    w, V = np.linalg.eig(M)
    k = int(np.argmax(np.imag(w)))
    beta = np.angle(w[k]) / np.pi
    y0 = V[:, k]

    def rhs(xi, y):
        kk = a - 2.0 * q * np.cos(2.0 * xi)
        return [y[1], -kk * y[0]]
    xi = np.linspace(0.0, np.pi, npts)
    s = solve_ivp(rhs, [0.0, np.pi], y0, t_eval=xi, rtol=1e-12, atol=1e-14)
    P = s.y[0] * np.exp(-1j * beta * xi)
    C = {n: np.trapezoid(P * np.exp(-2j * n * xi), xi) / np.pi
         for n in range(-nmax, nmax + 1)}
    return beta, C


print(f"{'q':>7}{'beta':>10}{'|C1|/|C0|':>12}{'|C-1|/|C0|':>12}"
      f"{'micro/secular':>15}{'q/2':>8}{'|C2|/|C0|':>12}")
for q in [0.05, 0.1, 0.2, 0.3, 0.5]:
    beta, C = floquet_harmonics(q)
    c0 = abs(C[0])
    ratio = (abs(C[1]) + abs(C[-1])) / c0
    print(f"{q:>7.2f}{beta:>10.6f}{abs(C[1]) / c0:>12.6f}"
          f"{abs(C[-1]) / c0:>12.6f}{ratio:>15.6f}{q / 2.0:>8.4f}"
          f"{abs(C[2]) / c0:>12.3e}")
print()

# --- laboratory numbers -------------------------------------------------
u_amu = 1.66053906660e-27
e = 1.602176634e-19
kB = 1.380649e-23
hbar = 1.054571817e-34
mass = 40.078 * u_amu
r0 = 0.5e-3
frf = 20.0e6
Om = TWOPI * frf
q = 0.3
beta, _ = floquet_harmonics(q)
wsec = beta * Om / 2.0
lam = 729e-9
kvec = TWOPI / lam
print(f"Ca+ at q = {q}: f_sec = {wsec / TWOPI / 1e6:.4f} MHz, "
      f"f_RF = {frf / 1e6:.0f} MHz")
x0 = np.sqrt(hbar / (2.0 * mass * wsec))
print(f"  ground-state extent x0 = sqrt(hbar / 2 m omega) = {x0 * 1e9:.3f} nm")
print(f"  intrinsic micromotion of a ground-state ion: "
      f"{q / 2.0 * x0 * 1e9:.3f} nm, "
      f"modulation index k x = {kvec * q / 2 * x0:.4f}")
print()
print(f"{'E_stray (V/m)':>15}{'offset (nm)':>13}{'micromotion (nm)':>18}"
      f"{'v (m/s)':>10}{'mod. index k x':>16}{'sideband/carrier':>18}")
for Estray in [0.1, 1.0, 10.0, 100.0]:
    d = e * Estray / (mass * wsec ** 2)
    amp = q / 2.0 * d
    v = amp * Om
    mi = kvec * amp
    sb = (jv(1, mi) / jv(0, mi)) ** 2
    print(f"{Estray:>15.1f}{d * 1e9:>13.2f}{amp * 1e9:>18.3f}{v:>10.3f}"
          f"{mi:>16.4f}{sb:>18.3e}")
print()
```

```text
      q      beta   |C1|/|C0|  |C-1|/|C0|  micro/secular     q/2   |C2|/|C0|
   0.05  0.035373    0.012070    0.012955       0.025024  0.0250   3.706e-05
   0.10  0.070850    0.023322    0.026875       0.050197  0.0500   1.407e-04
   0.20  0.142551    0.043590    0.058014       0.101604  0.1000   5.081e-04
   0.30  0.216059    0.061151    0.094454       0.155605  0.1500   1.032e-03
   0.50  0.373744    0.088943    0.190427       0.279369  0.2500   2.325e-03

Ca+ at q = 0.3: f_sec = 2.1606 MHz, f_RF = 20 MHz
  ground-state extent x0 = sqrt(hbar / 2 m omega) = 7.640 nm
  intrinsic micromotion of a ground-state ion: 1.146 nm, modulation index k x = 0.0099

  E_stray (V/m)  offset (nm)  micromotion (nm)   v (m/s)  mod. index k x  sideband/carrier
            0.1         1.31             0.196     0.025          0.0017         7.131e-07
            1.0        13.06             1.959     0.246          0.0169         7.131e-05
           10.0       130.63            19.595     2.462          0.1689         7.182e-03
          100.0      1306.32           195.948    24.624          1.6889         2.036e+00
```

**What to notice.** Decomposing the exact Floquet solution into its harmonics confirms the standard picture with no fitting. The combined weight of the $n = \pm1$ components relative to $n = 0$ is 0.02502 at $q = 0.05$ against the predicted $q/2 = 0.025$, and 0.1556 at $q = 0.3$ against 0.15 — a 4% third-order correction. The second harmonic appears only at order $q^2$, as it should.

The asymmetry between $|C_{+1}|$ and $|C_{-1}|$ (0.061 against 0.094 at $q = 0.3$) is real and is invisible in the first-order solution: the upper and lower micromotion sidebands are not equally strong.

The laboratory numbers give the scale. A ground-state Ca$^+$ ion in a 2.16 MHz well is spread over 7.6 nm, and its intrinsic micromotion is 1.1 nm — a modulation index of 0.01 on a 729 nm transition, which is negligible. Now add a stray field. At 1 V/m the ion is displaced 13 nm and driven with an amplitude of 2.0 nm, giving a modulation index of 0.017 and a sideband-to-carrier ratio of $7\times10^{-5}$: detectable, and small. At 10 V/m the ratio is 0.7%; at 100 V/m the modulation index exceeds 1 and the micromotion sideband is *stronger* than the carrier.

Stray fields of tens of V/m are entirely ordinary in a trap with exposed dielectric surfaces, so micromotion compensation is not an optional refinement. Note what the physical origin is: charge accumulating on insulating patches near the electrodes, which drifts as those patches charge and discharge. That is the same class of problem as the two-level defects of Chapter 2 — an uncontrolled surface state with slow dynamics — and it will return in Section 3.6 wearing a different name.

* * *

## 3.2 What Carries the Qubit

The trap holds the ion; the qubit lives in its internal state. Two families are in use, and the choice propagates through the whole experiment.

**Hyperfine qubits** use two sublevels of the electronic ground state, split by the interaction of the electron with the nuclear spin — 12.6 GHz in $^{171}$Yb$^+$, 1.25 GHz in $^9$Be$^+$. Because both states are in the ground manifold, there is no spontaneous emission at all, and the natural lifetime is essentially infinite. Coherence is limited instead by magnetic-field noise, since the two levels have different magnetic moments. The remedy is a **clock transition**: a pair of states whose energy difference is stationary in the field, $\partial\nu/\partial B = 0$, at zero field or at a "magic" field. First-order magnetic sensitivity vanishes there, and coherence times of many seconds — and in dedicated experiments far longer — follow. The cost is that the splitting is a microwave frequency, so a direct microwave drive gives no spatial resolution and no coupling to the motion; two-qubit gates need a pair of laser beams driving a stimulated Raman transition, and the effective wavevector is the *difference* of the two beams.

**Optical qubits** use a metastable excited state connected to the ground state by a forbidden transition — the $S_{1/2} \to D_{5/2}$ quadrupole line at 729 nm in $^{40}$Ca$^+$, with a natural lifetime around a second. A single narrow-linewidth laser drives it directly, which makes single-ion addressing straightforward and gives direct access to the motional sidebands with one beam. The price is that the excited state *does* decay, capping $T_1$ at the metastable lifetime, and that the laser's own frequency stability now enters the qubit coherence: a laser with a 1 Hz linewidth is a piece of apparatus, not a purchase.

Neither choice dominates. What matters for the rest of this chapter is that both couple to the ion's *motion* through the laser wavevector, and that coupling is the resource from which two-qubit gates are built.

### The ion crystal

Several ions in the same trap repel each other and settle into a linear crystal along the weakly confined axis. Their small oscillations about equilibrium are normal modes shared by all the ions — and that sharing is what makes a two-qubit gate between arbitrary pairs possible.

In the harmonic axial well the equilibrium positions follow from minimizing

$$ V = \sum_i \frac{u_i^2}{2} + \sum_{i<j} \frac{1}{|u_i - u_j|} $$

in units of $\ell = \left(e^2/4\pi\epsilon_0 m\omega_z^2\right)^{1/3}$, and the mode frequencies are the square roots of the eigenvalues of the Hessian. Two modes matter conceptually: the **centre-of-mass** mode at exactly $\omega_z$, in which all ions move together, and the **stretch** mode at $\sqrt{3}\omega_z$ for two ions.

Transverse modes work differently and are the ones actually used for gates. The Coulomb repulsion *softens* transverse motion — an ion pushed sideways is pushed further sideways by its neighbours — so the transverse spectrum lies *below* the single-ion radial frequency $\omega_r$, squeezed into a narrow band. When the softening exceeds the confinement the lowest transverse mode goes imaginary and the chain buckles into a zigzag. That is a purely geometric statement about the Coulomb matrix, and Example 3 finds the boundary numerically.

### Code Example 3: The Ion Crystal and Its Modes

```python
"""Chapter 3, Example 3: the ion crystal.  Equilibrium positions, axial and
transverse normal modes, the zigzag instability, and the mode crowding that
bounds how fast a long chain can be gated."""
import numpy as np

u_amu = 1.66053906660e-27
e = 1.602176634e-19
eps0 = 8.8541878128e-12
TWOPI = 2.0 * np.pi
KE = e ** 2 / (4.0 * np.pi * eps0)


def equilibrium(N, iters=300):
    """Dimensionless equilibrium positions of N ions in a harmonic axial well.

    Length unit l = (e^2 / 4 pi eps0 m omega_z^2)^(1/3); the potential is
    V = sum_i u_i^2/2 + sum_{i<j} 1/|u_i - u_j|.  Newton's method.
    """
    if N == 1:
        return np.zeros(1)
    u = np.linspace(-1.0, 1.0, N) * (0.5 * N ** 0.6)
    for _ in range(iters):
        d = u[:, None] - u[None, :]
        np.fill_diagonal(d, np.inf)
        inv3 = 1.0 / np.abs(d) ** 3
        grad = u - np.sum(np.sign(d) / d ** 2, axis=1)
        H = -2.0 * inv3
        np.fill_diagonal(H, 1.0 + 2.0 * np.sum(inv3, axis=1))
        u = u - np.linalg.solve(H, grad)
    return np.sort(u)


def modes(N, ratio=None):
    """Axial mode frequencies (units of omega_z), and transverse ones if the
    radial/axial frequency ratio is given.  Returns (u, w_ax, V_ax, w_tr)."""
    u = equilibrium(N)
    d = u[:, None] - u[None, :]
    np.fill_diagonal(d, np.inf)
    inv3 = 1.0 / np.abs(d) ** 3
    A = -2.0 * inv3
    np.fill_diagonal(A, 1.0 + 2.0 * np.sum(inv3, axis=1))
    w2, V = np.linalg.eigh(A)
    w_ax = np.sqrt(np.maximum(w2, 0.0))
    w_tr = None
    if ratio is not None:
        B = inv3.copy()
        np.fill_diagonal(B, ratio ** 2 - np.sum(inv3, axis=1))
        t2 = np.linalg.eigvalsh(B)
        w_tr = np.sign(t2) * np.sqrt(np.abs(t2))    # negative = unstable
    return u, w_ax, V, w_tr


print(f"{'N':>4}{'length (l)':>12}{'min spacing (l)':>17}"
      f"   mode frequencies (omega_z)")
for N in [1, 2, 3, 5, 8, 10]:
    u, w, V, _ = modes(N)
    sp = np.diff(u).min() if N > 1 else np.nan
    print(f"{N:>4}{(u[-1] - u[0]):>12.4f}{sp:>17.4f}   "
          + "  ".join(f"{x:.4f}" for x in w[:6]))
print()
u2, w2, V2, _ = modes(2)
print(f"  separation      {u2[1] - u2[0]:.9f} l   "
      f"(2 x 4^(-1/3) = {2.0 * 4.0 ** (-1.0 / 3.0):.9f})")
print(f"  mode 0 (COM)    {w2[0]:.9f} omega_z   (exactly 1)")
print(f"  mode 1 (stretch){w2[1]:.9f} omega_z   "
      f"(sqrt(3) = {np.sqrt(3.0):.9f})")
print(f"  COM mode vector     {np.round(np.abs(V2[:, 0]), 6)}  "
      f"(uniform, 1/sqrt(2) = {1 / np.sqrt(2):.6f})")
print(f"  stretch mode vector {np.round(V2[:, 1], 6)}")
print()

# --- laboratory units ---------------------------------------------------
mass = 40.078 * u_amu
fz = 1.0e6
wz = TWOPI * fz
ell = (KE / (mass * wz ** 2)) ** (1.0 / 3.0)
print(f"Ca+ with f_z = {fz / 1e6:.1f} MHz: length unit l = "
      f"{ell * 1e6:.3f} um, so the ions sit a few microns apart.")
print(f"{'N':>4}{'chain length (um)':>19}{'min spacing (um)':>18}"
      f"{'f_COM (MHz)':>13}{'f_ax,max (MHz)':>16}")
for N in [2, 6, 10, 20, 30]:
    u, w, V, _ = modes(N)
    print(f"{N:>4}{(u[-1] - u[0]) * ell * 1e6:>19.2f}"
          f"{np.diff(u).min() * ell * 1e6:>18.3f}"
          f"{w[0] * fz / 1e6:>13.4f}{w[-1] * fz / 1e6:>16.4f}")
print()

# --- transverse modes and the zigzag instability -----------------------
print(f"{'omega_r/omega_z':>16}{'largest linear N':>18}"
      f"{'0.73 N^0.86 at that N':>24}")
for ratio in [3.0, 5.0, 8.0, 12.0, 20.0]:
    Nmax = 1
    for N in range(2, 200):
        _, _, _, wt = modes(N, ratio)
        if wt.min() <= 0.0:
            break
        Nmax = N
    print(f"{ratio:>16.1f}{Nmax:>18}{0.73 * Nmax ** 0.86:>24.2f}")
print()

print(f"{'N':>4}{'bandwidth (kHz)':>17}{'min spacing (kHz)':>19}"
      f"{'implied gate time (us)':>24}")
fr = 5.0e6
rows = []
for N in [2, 6, 10, 14, 20, 30]:
    _, _, _, wt = modes(N, fr / fz)
    if wt.min() <= 0:
        # restore linear stability by raising the radial confinement
        ratio = 0.9 * N ** 0.86 + 1.0
        _, _, _, wt = modes(N, ratio)
        tag = f"  (needs omega_r/omega_z = {ratio:.1f})"
    else:
        tag = ""
    f = np.sort(wt) * fz
    gap = np.diff(f).min()
    rows.append((N, gap / fz))
    print(f"{N:>4}{(f[-1] - f[0]) / 1e3:>17.2f}{gap / 1e3:>19.3f}"
          f"{1e6 / gap:>24.2f}{tag}")
print()
sp = []
for N in [8, 16, 32, 64]:
    _, _, _, wt = modes(N, 0.9 * N ** 0.86 + 1.0)
    sp.append((N, np.diff(np.sort(wt)).min()))
c = np.polyfit(np.log([s[0] for s in sp]), np.log([s[1] for s in sp]), 1)
print(f"Fitted scaling at the stability margin: min spacing / omega_z ~ "
      f"N^{c[0]:.2f}")
print()
u, w, V, _ = modes(5)
print("axial mode participation for N = 5 (rows = ions, columns = modes)")
print("  frequencies (omega_z): " + "  ".join(f"{x:.4f}" for x in w))
for i in range(5):
    print(f"   ion {i}: " + "  ".join(f"{V[i, m]:+.4f}" for m in range(5)))
print()
```

```text
   N  length (l)  min spacing (l)   mode frequencies (omega_z)
   1      0.0000              nan   1.0000
   2      1.2599           1.2599   1.0000  1.7321
   3      2.1544           1.0772   1.0000  1.7321  2.4083
   5      3.4858           0.8221   1.0000  1.7321  2.4120  3.0549  3.6708
   8      4.9516           0.6360   1.0000  1.7321  2.4153  3.0632  3.6847  4.2859
  10      5.7417           0.5642   1.0000  1.7321  2.4168  3.0672  3.6914  4.2955

  separation      1.259921050 l   (2 x 4^(-1/3) = 1.259921050)
  mode 0 (COM)    1.000000000 omega_z   (exactly 1)
  mode 1 (stretch)1.732050808 omega_z   (sqrt(3) = 1.732050808)
  COM mode vector     [0.707107 0.707107]  (uniform, 1/sqrt(2) = 0.707107)
  stretch mode vector [-0.707107  0.707107]

Ca+ with f_z = 1.0 MHz: length unit l = 4.445 um, so the ions sit a few microns apart.
   N  chain length (um)  min spacing (um)  f_COM (MHz)  f_ax,max (MHz)
   2               5.60             5.600       1.0000          1.7321
   6              17.89             3.288       1.0000          4.2738
  10              25.52             2.508       1.0000          6.5758
  20              38.47             1.708       1.0000         11.9280
  30              47.75             1.355       1.0000         16.9902

 omega_r/omega_z  largest linear N   0.73 N^0.86 at that N
             3.0                 6                    3.41
             5.0                11                    5.74
             8.0                18                    8.77
            12.0                30                   13.60
            20.0                53                   22.19

   N  bandwidth (kHz)  min spacing (kHz)  implied gate time (us)
   2           101.02            101.021                    9.90
   6           954.37            101.021                    9.90
  10          3030.29            101.021                    9.90
  14          2203.44             51.642                   19.36  (needs omega_r/omega_z = 9.7)
  20          3134.88             39.019                   25.63  (needs omega_r/omega_z = 12.8)
  30          4656.95             28.158                   35.51  (needs omega_r/omega_z = 17.8)

Fitted scaling at the stability margin: min spacing / omega_z ~ N^-0.80

axial mode participation for N = 5 (rows = ions, columns = modes)
  frequencies (omega_z): 1.0000  1.7321  2.4120  3.0549  3.6708
   ion 0: +0.4472  +0.6395  +0.5377  -0.3017  +0.1045
   ion 1: +0.4472  +0.3017  -0.2805  +0.6395  -0.4704
   ion 2: +0.4472  -0.0000  -0.5143  +0.0000  +0.7318
   ion 3: +0.4472  -0.3017  -0.2805  -0.6395  -0.4704
   ion 4: +0.4472  -0.6395  +0.5377  +0.3017  +0.1045
```

**What to notice.** The two-ion case is exact and everything checks: separation $2\times4^{-1/3} = 1.259921$, modes at exactly 1 and $\sqrt{3} = 1.7320508$, COM mode vector uniform at $1/\sqrt{2}$, stretch mode antisymmetric. Nine digits of agreement on a closed-form result is the licence to trust the code for $N = 30$, where no closed form exists.

In laboratory units, Ca$^+$ at $f_z = 1$ MHz gives $\ell = 4.4\ \mu$m, so ions sit a few microns apart — comfortably resolvable optically, which is what makes single-ion addressing and single-ion imaging possible. The spacing shrinks as the chain grows: 5.6 $\mu$m for two ions, 1.4 $\mu$m for thirty, and at some point the addressing beam can no longer be focused between neighbours.

The zigzag table is the striking result. Requiring the lowest transverse eigenvalue to stay positive reproduces the empirical criterion $\omega_r/\omega_z \approx 0.73\,N^{0.86}$ across a factor of ten in $N$ (6 ions at ratio 3, 53 at ratio 20, against the criterion's 3.4 and 22.2). The radial confinement must grow almost linearly with the number of ions just to keep the chain straight. Nothing in that derivation involves quantum mechanics; it is electrostatics and a Hessian.

The mode-crowding table is where the price of all-to-all connectivity appears. At a fixed $\omega_r/\omega_z = 5$ the minimum transverse mode spacing is 101 kHz *independent of N* — because the smallest gap is between the top two modes, and that gap is $\approx \omega_z^2/2\omega_r$, which does not know about $N$. But the chain is only linear up to $N = 11$ at that ratio. Force linearity at larger $N$ by raising $\omega_r$ as the stability criterion requires, and the same gap shrinks as $1/\omega_r$: 51 kHz at $N = 14$, 28 kHz at $N = 30$. The fitted scaling at the stability margin is $N^{-0.80}$.

A gate that addresses one mode must be spectrally resolved from its neighbours, so the detuning — and hence the gate duration $\sim 2\pi/\delta$ — is bounded by that spacing. The implied minimum gate time rises from 10 $\mu$s at $N = 2$ to 36 $\mu$s at $N = 30$, growing as $N^{0.8}$. **That is the real cost of a long chain**, and it is not the electrode count or the laser power. It is why the field turned to architectures that connect many short chains — shuttling ions between zones on a segmented trap, or linking separate traps photonically — rather than making one chain longer.

The participation matrix for $N = 5$ makes the connectivity concrete: every ion has non-zero amplitude in almost every mode, so a beam on ion $i$ and a beam on ion $j$ couple to the same oscillator regardless of how far apart they are. That is where all-to-all connectivity comes from. It is also where the crosstalk comes from, because the statement holds for every other pair at the same time.

* * *

## 3.3 Laser Cooling and Sideband Spectroscopy

### Two stages, two limits

Cooling happens in two stages with two entirely different limits.

**Doppler cooling** uses a broad, dipole-allowed transition red-detuned from resonance. An ion moving towards the laser sees it Doppler-shifted into resonance and absorbs preferentially, so photon scattering opposes the motion. The limit is set by the random recoil of the spontaneously emitted photons against the cooling force, and comes out at

$$ T_\mathrm{Doppler} = \frac{\hbar\Gamma}{2k_B} $$

Note what this says: a *broad* transition cools fast and stops early. For the 397 nm line of Ca$^+$ with $\Gamma/2\pi = 21.6$ MHz the limit is 0.5 mK, which in a 2 MHz trap means $\bar{n} \approx 5$. That is cold enough to trap and image and far too warm to gate, because as Example 4 shows the sideband Rabi frequency depends on $n$.

**Resolved-sideband cooling** requires the opposite: a transition *narrow* compared with the trap frequency, so that the red sideband ($|g, n\rangle \to |e, n-1\rangle$) can be driven without touching the carrier or the blue sideband. Each cycle removes one quantum. The residual heating comes from off-resonant excitation of the blue sideband, suppressed by the ratio of linewidth to sideband spacing, and the steady state is thermal with

$$ \bar{n}_\mathrm{ss} \approx \left(\frac{\Gamma}{4\omega}\right)^2 $$

### The Lamb-Dicke parameter

The coupling between internal state and motion is controlled by one dimensionless number,

$$ \eta = k x_0 = \frac{2\pi}{\lambda}\sqrt{\frac{\hbar}{2m\omega}} $$

with $k = 2\pi/\lambda$ the (effective) laser wavevector, so $\eta$ is $2\pi$ times the ratio of the ion's ground-state extent to the laser wavelength, not that ratio itself. The interaction operator is $e^{i\eta(a + a^\dagger)}$, whose matrix elements are

$$ \langle n + s|e^{i\eta(a+a^\dagger)}|n\rangle = e^{-\eta^2/2}\,(i\eta)^s\sqrt{\frac{n!}{(n+s)!}}\,L_n^{s}(\eta^2) $$

with $L_n^s$ a generalized Laguerre polynomial. The **Lamb-Dicke regime** is the condition $\eta^2(2n+1) \ll 1$ — small $\eta$ *and* a cold ion, since it is the spread of the wavepacket rather than its ground-state extent that has to be small compared with the wavelength. There the carrier is strong and the first sidebands go as $\eta\sqrt{n+1}$ (blue) and $\eta\sqrt{n}$ (red), and all of the standard gate theory is derived in that limit. Example 6 measures what it costs when the condition is only nearly satisfied.

### Code Example 4: Sidebands, Exactly and Approximately

```python
"""Chapter 3, Example 4: the Lamb-Dicke parameter and the sideband spectrum.
Exact matrix elements of exp(i eta (a + a^dag)), a simulated sideband spectrum
for a thermal state, and sideband-ratio thermometry."""
import numpy as np
from scipy.linalg import expm
from scipy.special import eval_genlaguerre, factorial

hbar = 1.054571817e-34
u_amu = 1.66053906660e-27
kB = 1.380649e-23
TWOPI = 2.0 * np.pi


def lamb_dicke(mass_u, f_trap, wavelength, n_beams=1):
    """eta = k x0 with x0 = sqrt(hbar / 2 m omega); n_beams = 2 for a Raman
    pair of counter-propagating beams, where the effective k doubles."""
    m = mass_u * u_amu
    w = TWOPI * f_trap
    x0 = np.sqrt(hbar / (2.0 * m * w))
    return n_beams * TWOPI / wavelength * x0, x0


print(f"{'ion':>6}{'m (u)':>9}{'lambda (nm)':>13}{'beams':>7}"
      f"{'f_trap (MHz)':>14}{'x0 (nm)':>10}{'eta':>9}   transition")
cases = [
    ("Ca+", 40.078, 729e-9, 1, 2.0e6, "quadrupole optical qubit"),
    ("Ca+", 40.078, 729e-9, 1, 1.0e6, "the same, weaker trap"),
    ("Sr+", 87.62, 674e-9, 1, 1.0e6, "quadrupole optical qubit"),
    ("Yb+", 171.0, 355e-9, 2, 3.0e6, "Raman pair, hyperfine qubit"),
    ("Be+", 9.012, 313e-9, 2, 3.0e6, "Raman pair, hyperfine qubit"),
]
for name, m_u, lam, nb, ft, trans in cases:
    eta, x0 = lamb_dicke(m_u, ft, lam, nb)
    print(f"{name:>6}{m_u:>9.3f}{lam * 1e9:>13.0f}{nb:>7}{ft / 1e6:>14.1f}"
          f"{x0 * 1e9:>10.3f}{eta:>9.4f}   {trans}")
print()

# --- exact matrix elements ---------------------------------------------
def rabi_matrix(eta, nmax=90):
    """|<m| exp(i eta (a + a^dag)) |n>| by matrix exponential."""
    a = np.diag(np.sqrt(np.arange(1, nmax + 1)), 1)
    return np.abs(expm(1j * eta * (a + a.conj().T)))


def rabi_analytic(eta, n, s):
    """Analytic |<n+s| exp(i eta (a+a^dag)) |n>| for s >= 0."""
    return (np.exp(-eta ** 2 / 2.0) * eta ** s
            * np.sqrt(factorial(n) / factorial(n + s))
            * abs(eval_genlaguerre(n, s, eta ** 2)))


eta = 0.0684
M = rabi_matrix(eta)
print(f"Sideband matrix elements at eta = {eta:.4f} "
      "(Ca+, 729 nm, 2 MHz trap)")
print(f"{'n':>4}{'carrier':>12}{'red (n-1)':>12}{'blue (n+1)':>12}"
      f"{'analytic blue':>15}{'LD: eta sqrt(n+1)':>19}{'LD error':>10}")
for n in [0, 1, 5, 10, 20, 40, 80]:
    car, blue = M[n, n], M[n + 1, n]
    red = M[n - 1, n] if n > 0 else 0.0
    ld = eta * np.sqrt(n + 1)
    print(f"{n:>4}{car:>12.6f}{red:>12.6f}{blue:>12.6f}"
          f"{rabi_analytic(eta, n, 1):>15.6f}{ld:>19.6f}"
          f"{(ld / blue - 1) * 100:>9.2f}%")
print()

# --- a simulated sideband spectrum -------------------------------------
def thermal(nbar, nmax):
    """Thermal Fock distribution, computed in log space to avoid overflow."""
    if nbar == 0.0:
        p = np.zeros(nmax)
        p[0] = 1.0
        return p
    n = np.arange(nmax)
    p = np.exp(n * np.log(nbar) - (n + 1) * np.log1p(nbar))
    return p / p.sum()


def excitation(eta, nbar, om_car, t_pulse, f_trap, detuning, nmax=80):
    """Excitation probability at one detuning.

    Each Fock state undergoes a detuned two-level Rabi oscillation on every
    sideband; the transitions are treated as independent, which is valid when
    the Rabi frequencies are small compared with the trap frequency.  The
    off-resonant carrier is kept, because it is what limits the method.
    """
    Mm = rabi_matrix(eta, nmax + 6)
    P = thermal(nbar, nmax)
    pe = 0.0
    for s in range(-3, 4):
        off = detuning - s * f_trap
        for n in range(nmax):
            if n + s < 0:
                continue
            om = om_car * Mm[n + s, n]
            geff = np.hypot(om, off)
            if geff == 0.0:
                continue
            pe += P[n] * (om / geff) ** 2 * np.sin(np.pi * geff * t_pulse) ** 2
    return pe


f_trap, om_car = 2.0e6, 50.0e3
t_pi = 1.0 / (2.0 * om_car)
print(f"Simulated spectrum: f_trap = {f_trap / 1e6:.1f} MHz, carrier Rabi "
      f"{om_car / 1e3:.0f} kHz, pulse = carrier pi time = {t_pi * 1e6:.1f} us")
print(f"{'line':>10}{'detuning':>12}{'nbar=0':>10}{'nbar=1':>10}"
      f"{'nbar=5':>10}{'nbar=20':>10}")
for s, lab in [(-2, "2nd red"), (-1, "1st red"), (0, "carrier"),
               (1, "1st blue"), (2, "2nd blue")]:
    vals = [excitation(eta, nb, om_car, t_pi, f_trap, s * f_trap)
            for nb in [0.0, 1.0, 5.0, 20.0]]
    print(f"{lab:>10}{s:>+9d} x f" + "".join(f"{v:>10.5f}" for v in vals))
print()
off_bg = excitation(eta, 0.0, om_car, t_pi, f_trap, -f_trap)
print(f"  the residual {off_bg:.2e} at nbar = 0 on the red sideband is the "
      f"off-resonant carrier:")
print(f"  it is bounded by (Omega/2 f_trap)^2 = "
      f"{(om_car / (2 * f_trap)) ** 2:.2e} and oscillates with the pulse "
      f"duration.")
print()

# --- sideband-ratio thermometry ----------------------------------------
print(f"{'nbar (true)':>13}{'red strength':>15}{'blue strength':>15}"
      f"{'ratio r':>11}{'r/(1-r)':>11}{'error':>9}")
Mt = rabi_matrix(eta, 400)
for nb in [0.02, 0.2, 1.0, 5.0, 20.0]:
    nmx = min(380, int(40 + 12 * nb))
    P = thermal(nb, nmx)
    red = sum(P[n] * Mt[n - 1, n] ** 2 for n in range(1, nmx))
    blue = sum(P[n] * Mt[n + 1, n] ** 2 for n in range(nmx))
    r = red / blue
    est = r / (1.0 - r)
    print(f"{nb:>13.2f}{red:>15.4e}{blue:>15.4e}{r:>11.6f}{est:>11.4f}"
          f"{(est / nb - 1) * 100:>8.3f}%")
print()
for nb in [0.02, 5.0]:
    T = TWOPI * f_trap * hbar / (kB * np.log(1.0 + 1.0 / nb))
    print(f"  nbar = {nb:>5.2f} at f_trap = {f_trap / 1e6:.1f} MHz is "
          f"T = {T * 1e6:>6.1f} uK")
```

```text
   ion    m (u)  lambda (nm)  beams  f_trap (MHz)   x0 (nm)      eta   transition
   Ca+   40.078          729      1           2.0     7.940   0.0684   quadrupole optical qubit
   Ca+   40.078          729      1           1.0    11.229   0.0968   the same, weaker trap
   Sr+   87.620          674      1           1.0     7.595   0.0708   quadrupole optical qubit
   Yb+  171.000          355      2           3.0     3.139   0.1111   Raman pair, hyperfine qubit
   Be+    9.012          313      2           3.0    13.672   0.5489   Raman pair, hyperfine qubit

Sideband matrix elements at eta = 0.0684 (Ca+, 729 nm, 2 MHz trap)
   n     carrier   red (n-1)  blue (n+1)  analytic blue  LD: eta sqrt(n+1)  LD error
   0    0.997663    0.000000    0.068240       0.068240           0.068400     0.23%
   1    0.992996    0.068240    0.096280       0.096280           0.096732     0.47%
   5    0.974434    0.151165    0.165205       0.165205           0.167545     1.42%
  10    0.951476    0.211279    0.221070       0.221070           0.226857     2.62%
  20    0.906366    0.291804    0.298300       0.298300           0.313448     5.08%
  40    0.819309    0.393364    0.397289       0.397289           0.437974    10.24%
  80    0.657389    0.504221    0.506095       0.506095           0.615600    21.64%

Simulated spectrum: f_trap = 2.0 MHz, carrier Rabi 50 kHz, pulse = carrier pi time = 10.0 us
      line    detuning    nbar=0    nbar=1    nbar=5   nbar=20
   2nd red       -2 x f   0.00000   0.00003   0.00064   0.00704
   1st red       -1 x f   0.00000   0.01126   0.05279   0.16363
   carrier       +0 x f   0.99999   0.99977   0.99689   0.96993
  1st blue       +1 x f   0.01145   0.02251   0.06334   0.17233
  2nd blue       +2 x f   0.00003   0.00011   0.00092   0.00790

  the residual 2.37e-07 at nbar = 0 on the red sideband is the off-resonant carrier:
  it is bounded by (Omega/2 f_trap)^2 = 1.56e-04 and oscillates with the pulse duration.

  nbar (true)   red strength  blue strength    ratio r    r/(1-r)    error
         0.02     9.3117e-05     4.7490e-03   0.019608     0.0200  -0.000%
         0.20     9.2961e-04     5.5776e-03   0.166667     0.2000  -0.000%
         1.00     4.6135e-03     9.2269e-03   0.500000     1.0000  -0.000%
         5.00     2.2227e-02     2.6672e-02   0.833333     5.0000  -0.000%
        20.00     7.7594e-02     8.1474e-02   0.952381    19.9999  -0.000%

  nbar =  0.02 at f_trap = 2.0 MHz is T =   24.4 uK
  nbar =  5.00 at f_trap = 2.0 MHz is T =  526.5 uK
```

**What to notice.** The $\eta$ table shows the range: 0.068 for an optical qubit in Ca$^+$, 0.11 for a Raman-driven hyperfine qubit in Yb$^+$ (the Raman pair doubles the effective $k$), and 0.55 for Be$^+$, which is light and therefore delocalized. Small $\eta$ means weak sidebands and therefore slow gates; large $\eta$ means fast gates and larger corrections to everything derived in the Lamb-Dicke limit. Be$^+$ is a genuinely different regime.

The matrix-element table checks the Laguerre formula against the matrix exponential to every printed digit, then shows the Lamb-Dicke approximation failing: 0.2% error at $n = 0$, 1.4% at $n = 5$, 10% at $n = 40$, 22% at $n = 80$, controlled by $\eta^2(n+1)$. This is not a noise problem. A hot ion has the *wrong* Rabi frequency, so a calibrated $\pi$ pulse is not a $\pi$ pulse, and the error differs from shot to shot as $n$ fluctuates. That is a coherent error, and averaging does not remove it.

The simulated spectrum shows the structure an experiment sees. At $\bar{n} = 0$ the red sideband vanishes *exactly* — there is no lower phonon state to go to — and that vanishing is the cleanest available signature of the motional ground state. As $\bar{n}$ rises, the red sideband grows towards the blue one and the second-order sidebands appear. The residual $2\times10^{-7}$ on the red sideband at $\bar{n} = 0$ is the off-resonant carrier, bounded by $(\Omega/2f_\mathrm{trap})^2 = 1.6\times10^{-4}$ and oscillating with pulse duration; it is a real background, and it is why ground-state cooling is verified with long, weak sideband pulses rather than short strong ones.

The thermometry table is the pleasant surprise. The inversion $\bar{n} = r/(1-r)$ is exact to every printed digit over three decades — not to first order in $\eta$, but exactly. The reason is detailed balance, not the Lamb-Dicke approximation: a thermal state satisfies $P(n) = P(n+1)e^{\hbar\omega/k_BT}$ and the matrix elements satisfy $|\langle n|\ldots|n+1\rangle| = |\langle n+1|\ldots|n\rangle|$, so the ratio of the two weighted sums is the Boltzmann factor whatever the matrix elements happen to be. Sideband thermometry needs no absolute calibration of anything, which is why every cooling result in this field is quoted as an $\bar{n}$.

### Code Example 5: Cooling to the Ground State

```python
"""Chapter 3, Example 5: from the Doppler limit to the motional ground state.
The Doppler cooling limit in numbers, then a rate-equation integration of
resolved-sideband cooling."""
import numpy as np
from scipy.linalg import expm

hbar = 1.054571817e-34
h = 2.0 * np.pi * hbar
u_amu = 1.66053906660e-27
kB = 1.380649e-23
TWOPI = 2.0 * np.pi

# --- Doppler limit ------------------------------------------------------
print(f"{'ion':>6}{'transition':>14}{'Gamma/2pi (MHz)':>17}{'T_D (uK)':>11}"
      f"{'v_rms (m/s)':>13}{'nbar at 2 MHz':>15}")
f_trap = 2.0e6
for name, m_u, lab, gam in [("Ca+", 40.078, "397 nm S-P", 21.6e6),
                            ("Sr+", 87.62, "422 nm S-P", 21.5e6),
                            ("Yb+", 171.0, "369 nm S-P", 19.6e6),
                            ("Be+", 9.012, "313 nm S-P", 19.4e6)]:
    T = hbar * TWOPI * gam / (2.0 * kB)
    m = m_u * u_amu
    vrms = np.sqrt(kB * T / m)
    nbar = 1.0 / (np.exp(h * f_trap / (kB * T)) - 1.0)
    print(f"{name:>6}{lab:>14}{gam / 1e6:>17.1f}{T * 1e6:>11.1f}"
          f"{vrms:>13.4f}{nbar:>15.2f}")
print()

# --- the narrow line is the QUBIT, not a second cooling stage -----------
# Ca+ 729 nm S1/2-D5/2 is an electric-quadrupole transition; the D5/2 lifetime
# is 1.168 s, so Gamma/2pi = 0.136 Hz.  Two things follow, and neither is
# cooling: (i) the Doppler formula T_D = hbar Gamma / 2 kB stops being the
# floor long before this, because the single-photon recoil energy is larger
# than hbar Gamma; (ii) the scattering rate is nine orders of magnitude below
# a dipole line, so no cooling happens on a laboratory timescale.
tau_D5 = 1.168                              # s, Ca+ D5/2 lifetime
gam_q = 1.0 / tau_D5                        # s^-1
lam_q = 729.147e-9
k_q = TWOPI / lam_q
m_ca = 40.078 * u_amu
E_r = hbar ** 2 * k_q ** 2 / (2.0 * m_ca)   # single-photon recoil energy
print(f"Ca+ 729 nm S1/2-D5/2 quadrupole line (the optical qubit):")
print(f"  lifetime {tau_D5:.3f} s -> Gamma/2pi = {gam_q / TWOPI:.3f} Hz, "
      f"Q = f/(Gamma/2pi) = {(2.998e8 / lam_q) / (gam_q / TWOPI):.2e}")
print(f"  Doppler formula would say T_D = {hbar * gam_q / (2 * kB):.2e} K, but "
      f"the recoil limit is")
print(f"  T_r = 2 E_r/kB = {2 * E_r / kB * 1e9:.0f} nK with E_r/h = "
      f"{E_r / h / 1e3:.2f} kHz: hbar*Gamma is {(E_r / h) / (gam_q / TWOPI):.1e} "
      f"times smaller than E_r,")
print(f"  so the Doppler expression is meaningless here, and the scattering "
      f"rate is {gam_q / (TWOPI * 21.6e6):.1e} of the 397 nm line.")
print()

# --- resolved-sideband cooling as a rate equation ----------------------
def rates(om_car, gamma_eff, f_trap, eta):
    """Cooling and heating rates per quantum, in 1/s.

    A red-sideband drive of carrier Rabi frequency Omega, with an excited state
    of effective linewidth Gamma, drives n -> n-1 at
        R(delta) = eta^2 Omega^2 Gamma / [(Gamma/2)^2 + delta^2]   per quantum,
    resonantly (delta = 0) for cooling and off-resonantly (delta = 2 omega, the
    blue sideband) for heating.  Valid for Omega < Gamma.
    """
    Om, G, w = TWOPI * om_car, TWOPI * gamma_eff, TWOPI * f_trap
    Rc = eta ** 2 * Om ** 2 * G / (G / 2.0) ** 2
    Rh = eta ** 2 * Om ** 2 * G / ((G / 2.0) ** 2 + (2.0 * w) ** 2)
    return Rc, Rh


def cool(nbar0, Rc, Rh, nmax=60, tmax=None, nt=400):
    """Integrate dP_n/dt with cooling rate Rc*n and heating rate Rh*(n+1)."""
    n = np.arange(nmax)
    P = np.exp(n * np.log(nbar0) - (n + 1) * np.log1p(nbar0))
    P /= P.sum()
    A = np.zeros((nmax, nmax))
    for k in range(nmax):
        if k > 0:
            A[k - 1, k] += Rc * k
            A[k, k] -= Rc * k
        if k < nmax - 1:
            A[k + 1, k] += Rh * (k + 1)
            A[k, k] -= Rh * (k + 1)
    tmax = tmax if tmax else 25.0 / Rc
    out = []
    for t in np.linspace(0.0, tmax, nt):
        Pt = expm(A * t) @ P
        out.append((t, float(np.sum(n * Pt)), float(Pt[0])))
    return out


eta = 0.0684
om_car = 50.0e3
T_D = hbar * TWOPI * 21.6e6 / (2.0 * kB)
nbar_D = 1.0 / (np.exp(h * f_trap / (kB * T_D)) - 1.0)
print(f"Resolved-sideband cooling, Ca+ at f_trap = {f_trap / 1e6:.1f} MHz, "
      f"eta = {eta:.4f},")
print(f"carrier Rabi frequency {om_car / 1e3:.0f} kHz, starting from the "
      f"Doppler-limited nbar = {nbar_D:.2f}.")
print(f"{'Gamma_eff/2pi':>16}{'R_c (1/s)':>12}{'r = R_h/R_c':>14}"
      f"{'nbar steady':>14}{'(Gamma/4 omega)^2':>20}{'P(n=0)':>10}")
for gam in [10.0e6, 2.0e6, 200.0e3, 100.0e3]:
    Rc, Rh = rates(om_car, gam, f_trap, eta)
    r = Rh / Rc
    nss = r / (1.0 - r)
    approx = (gam / (4.0 * f_trap)) ** 2
    print(f"{gam / 1e6:>14.3f} M{Rc:>12.3e}{r:>14.3e}{nss:>14.3e}"
          f"{approx:>20.3e}{1.0 / (1.0 + nss):>10.6f}")
print()

Rc, Rh = rates(om_car, 200.0e3, f_trap, eta)
traj = cool(nbar_D, Rc, Rh)
print(f"Cooling trajectory at Gamma_eff/2pi = 200 kHz "
      f"(R_c = {Rc:.1f} 1/s per quantum):")
print(f"{'t (ms)':>10}{'nbar':>12}{'P(n=0)':>12}")
for k in [0, 12, 25, 50, 100, 200, 399]:
    t, nb, p0 = traj[k]
    print(f"{t * 1e3:>10.3f}{nb:>12.6f}{p0:>12.6f}")
print()

# --- what the residual nbar costs a gate --------------------------------
eta = 0.0684
for nb in [0.01, 0.05, 0.5, nbar_D]:
    nmx = min(300, int(40 + 12 * nb))
    n = np.arange(nmx)
    P = np.exp(n * np.log(nb) - (n + 1) * np.log1p(nb))
    P /= P.sum()
    om = eta * np.sqrt(n + 1)                 # Lamb-Dicke sideband strength
    mean = float(np.sum(P * om))
    rms = float(np.sqrt(np.sum(P * om ** 2) - mean ** 2))
    print(f"  nbar = {nb:>5.2f}: mean sideband strength {mean:.5f}, "
          f"spread {rms:.5f} ({rms / mean * 100:>5.2f}% of the mean)")
print()
```

```text
   ion    transition  Gamma/2pi (MHz)   T_D (uK)  v_rms (m/s)  nbar at 2 MHz
   Ca+    397 nm S-P             21.6      518.3       0.3279           4.92
   Sr+    422 nm S-P             21.5      515.9       0.2213           4.89
   Yb+    369 nm S-P             19.6      470.3       0.1512           4.42
   Be+    313 nm S-P             19.4      465.5       0.6554           4.37

Ca+ 729 nm S1/2-D5/2 quadrupole line (the optical qubit):
  lifetime 1.168 s -> Gamma/2pi = 0.136 Hz, Q = f/(Gamma/2pi) = 3.02e+15
  Doppler formula would say T_D = 3.27e-12 K, but the recoil limit is
  T_r = 2 E_r/kB = 899 nK with E_r/h = 9.36 kHz: hbar*Gamma is 6.9e+04 times smaller than E_r,
  so the Doppler expression is meaningless here, and the scattering rate is 6.3e-09 of the 397 nm line.

Resolved-sideband cooling, Ca+ at f_trap = 2.0 MHz, eta = 0.0684,
carrier Rabi frequency 50 kHz, starting from the Doppler-limited nbar = 4.92.
   Gamma_eff/2pi   R_c (1/s)   r = R_h/R_c   nbar steady   (Gamma/4 omega)^2    P(n=0)
        10.000 M   2.940e+01     6.098e-01     1.563e+00           1.562e+00  0.390244
         2.000 M   1.470e+02     5.882e-02     6.250e-02           6.250e-02  0.941176
         0.200 M   1.470e+03     6.246e-04     6.250e-04           6.250e-04  0.999375
         0.100 M   2.940e+03     1.562e-04     1.562e-04           1.563e-04  0.999844

Cooling trajectory at Gamma_eff/2pi = 200 kHz (R_c = 1469.8 1/s per quantum):
    t (ms)        nbar      P(n=0)
     0.000    4.914527    0.169052
     0.512    2.318517    0.301306
     1.066    1.027613    0.493152
     2.131    0.215262    0.822853
     4.263    0.010000    0.990099
     8.526    0.000643    0.999358
    17.009    0.000625    0.999375

  nbar =  0.01: mean sideband strength 0.06868, spread 0.00283 ( 4.13% of the mean)
  nbar =  0.05: mean sideband strength 0.06980, spread 0.00635 ( 9.10% of the mean)
  nbar =  0.50: mean sideband strength 0.08122, spread 0.02050 (25.24% of the mean)
  nbar =  4.92: mean sideband strength 0.15192, spread 0.06778 (44.62% of the mean)
```

**What to notice.** The Doppler table shows how little the choice of ion matters for this stage: every dipole-allowed cooling line in use has $\Gamma/2\pi \approx 20$ MHz, so every species stops at 0.5 mK and $\bar{n} \approx 4.5$ in a 2 MHz trap. Doppler cooling is a commodity, and it is not enough: $\bar{n} = 4.9$ is not the ground state.

The block underneath says why there is no way to fix that by picking a narrower line. The obvious candidate in Ca$^+$ is the 729 nm $S_{1/2}$-$D_{5/2}$ quadrupole transition, whose upper state lives 1.168 s, giving $\Gamma/2\pi = 0.136$ Hz and a line quality factor of $3\times10^{15}$. Substituting that into $T_D = \hbar\Gamma/2k_B$ returns $3\times10^{-12}$ K, which is not a temperature any ion reaches: the expression stops being the floor once $\hbar\Gamma$ drops below the single-photon recoil energy, and here $E_r/h = 9.4$ kHz is $7\times10^4$ times larger than $\Gamma/2\pi$. The real floor for a narrow line is the recoil limit, $T_r = 2E_r/k_B = 0.9\ \mu$K, and in any case the scattering rate on that line is $6\times10^{-9}$ of the 397 nm line — about one photon per second — so nothing cools on it. A 0.136 Hz linewidth is not a cooling resource; it is what makes the transition a *qubit*. Deep cooling therefore needs a different idea rather than a different line, and that idea is to cool on a resolved motional sideband, which is the second half of this example.

(The transition often mistaken for a narrow cooling line in Ca$^+$ is 866 nm. It is the $D_{3/2}$-$P_{1/2}$ repump, and it shares the $P_{1/2}$ level with the 397 nm cooling line, so its natural linewidth is that level's $\approx 22$ MHz. The $\sim$1.7 MHz sometimes quoted for it is a branching *partial* rate out of $P_{1/2}$, not a linewidth, and it does not make the transition narrow.)

The sideband-cooling table confirms $\bar{n}_\mathrm{ss} = (\Gamma/4\omega)^2$ to three digits across four decades. Read it as a design rule: the cooling limit is set by *how well the sidebands are resolved*, not by any temperature. At $\Gamma/2\pi = 10$ MHz — a dipole line — the limit is $\bar{n} = 1.6$, no better than Doppler cooling. At 100 kHz it is $1.6\times10^{-4}$, i.e. the ground state with probability 0.9998. Since $\omega$ appears in the denominator, the trap frequency is itself a cooling parameter: a stiffer trap cools deeper, which is one more reason gates use the high-frequency transverse modes.

The trajectory shows the shape of the approach. The cooling rate is proportional to $n$, so the high Fock states empty first and the last factor of two costs as much time as everything before it: $\bar{n}$ falls from 4.9 to 1.0 in 1 ms and takes another 3 ms to reach $10^{-2}$. Real sequences are hundreds of sideband pulses interleaved with repumping, taking milliseconds, and that preparation time is a large part of the duty cycle of a trapped-ion processor. The clock speed of the machine is set as much by cooling and readout as by gates.

The last table prices the residual. At $\bar{n} = 0.5$ the spread in sideband Rabi frequency is 25% of its mean, and at the Doppler limit it is 45%. A quarter of the Rabi frequency is a quarter of the rotation angle. That is why the first-generation Cirac-Zoller gate, which uses sideband $\pi$ pulses directly, demands $\bar{n} \ll 1$ — and why the gate that replaced it does not.

* * *

## 3.4 Gate Mechanisms

### Cirac-Zoller: the direct route

The original proposal (Cirac and Zoller, 1995) is a three-step sequence: map the internal state of ion 1 onto the shared phonon mode with a red-sideband $\pi$ pulse, apply a conditional operation on ion 2 that depends on whether the mode holds a phonon, then map the phonon back onto ion 1. The bus is the motion, and the gate is exact in principle.

In practice it has two hard requirements. The mode must start in $|n = 0\rangle$, because the sideband Rabi frequency depends on $n$ and the $\pi$ pulses would otherwise be miscalibrated by the amounts tabulated in Example 5. And the phonon must not be lost or heated during the sequence, because the intermediate state genuinely has the quantum information stored in the motion. Both requirements are met by heroic effort, and neither scales.

### Mølmer-Sørensen: the geometric route

The Mølmer-Sørensen gate removes both requirements by never putting a real excitation in the motion. Drive both ions with a *bichromatic* field, tuned symmetrically about the qubit frequency at $\omega_0 \pm (\omega - \delta)$, so that neither tone is resonant with either sideband — each is detuned by $\delta$. Neither ion can absorb a photon alone. But the *pair* can: the two-photon process $|00\rangle \to |11\rangle$, one photon from each tone, conserves energy with no net change in phonon number. The motion is used virtually.

In the Lamb-Dicke regime and after the rotating-wave approximation, the interaction is

$$ H(t) = g\left(a e^{-i\delta t} + a^\dagger e^{i\delta t}\right)\left(\sigma_y^{(1)} + \sigma_y^{(2)}\right), \qquad g = \frac{\eta\Omega}{2} $$

with $\Omega$ the carrier Rabi frequency. This Hamiltonian has a remarkable property: its Magnus expansion **terminates**. Because $H$ is linear in $a$ and $a^\dagger$, the commutator $[H(t_1), H(t_2)]$ is a c-number times $S_y^2$, which commutes with $H$; every higher Magnus term therefore vanishes, and the evolution is exact:

$$ U(\tau) = e^{-i\Phi S_y^2}\,D\!\left(\alpha(\tau)S_y\right), \qquad S_y = \sigma_y^{(1)} + \sigma_y^{(2)} $$

$$ \alpha(\tau) = -\frac{g\left(e^{i\delta\tau} - 1\right)}{\delta}, \qquad \Phi(\tau) = -\frac{g^2}{\delta}\left(\tau - \frac{\sin\delta\tau}{\delta}\right) $$

Two conditions make this a gate. **Loop closure**: $\alpha(\tau) = 0$ requires $\delta\tau = 2\pi K$ for integer $K$ — the trajectory in motional phase space must return to where it started, otherwise the motion stays entangled with the spins and no spin-only unitary describes the outcome. **Gate angle**: with the loop closed, $\Phi = -2\pi K g^2/\delta^2$, and since $S_y^2 = 2 + 2\sigma_y^{(1)}\sigma_y^{(2)}$, the operation is $e^{-2i\Phi\sigma_y\sigma_y}$ up to a global phase. Demanding $2|\Phi| = \pi/4$ gives

$$ g = \frac{\delta}{4\sqrt{K}} \quad\Longleftrightarrow\quad \eta\Omega = \frac{\delta}{2\sqrt{K}} $$

And here is the payoff: **$\Phi$ does not contain $n$**. The Hamiltonian is independent of the phonon number, so the gate works identically on a cold ion and a warm one. Doppler cooling is enough in principle. That single fact is why every trapped-ion processor built since uses this gate.

### Code Example 6: Verifying the Mølmer-Sørensen Gate

```python
"""Chapter 3, Example 6: the Molmer-Sorensen gate, verified numerically.

Lamb-Dicke form, with g = eta Omega / 2 and Omega the carrier Rabi frequency:
    H(t) = g (a exp(-i delta t) + a^dag exp(+i delta t)) (sy_1 + sy_2).
Its Magnus expansion terminates, giving exactly
    U(tau) = exp(-i Phi Sy^2) D(alpha Sy),
    alpha(tau) = -g (exp(i delta tau) - 1) / delta,
    Phi(tau)   = -(g^2/delta) (tau - sin(delta tau)/delta).
We integrate the Schrodinger equation, check every claim, then repeat the
calculation without the Lamb-Dicke approximation."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

TWOPI = 2.0 * np.pi

# --- operators ----------------------------------------------------------
NMAX = 30                                    # Fock cutoff
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sp = np.array([[0, 1], [0, 0]], dtype=complex)     # |0><1| raising in our order
I2 = np.eye(2, dtype=complex)
a = np.diag(np.sqrt(np.arange(1, NMAX + 1)), 1).astype(complex)
Im = np.eye(NMAX + 1, dtype=complex)


def spin(op, j):
    return np.kron(op, I2) if j == 0 else np.kron(I2, op)


SY = np.kron(spin(sy, 0) + spin(sy, 1), Im)
A = np.kron(np.eye(4, dtype=complex), a)
AD = A.conj().T
NOP = AD @ A
DIM = 4 * (NMAX + 1)

# --- parameters ---------------------------------------------------------
eta = 0.0684
f_mode = 2.0e6            # Hz, the shared mode
K = 1                     # number of phase-space loops
tau = 100e-6              # s, gate duration
delta = K / tau           # Hz, so that delta*tau = K cycles
g = TWOPI * delta / 4.0 / np.sqrt(K)      # rad/s; the gate condition, below
om_car = 2.0 * g / (TWOPI * eta)          # Hz, carrier Rabi frequency
wdelta = TWOPI * delta
print(f"eta = {eta:.4f}, mode f = {f_mode / 1e6:.1f} MHz, "
      f"gate time tau = {tau * 1e6:.0f} us, loops K = {K}")
print(f"detuning from the sideband delta/2pi = {delta / 1e3:.3f} kHz")
print(f"coupling g/2pi = {g / TWOPI / 1e3:.4f} kHz  "
      f"-> carrier Rabi Omega/2pi = {om_car / 1e3:.3f} kHz")
Phi = -(g ** 2 / wdelta) * (tau - np.sin(wdelta * tau) / wdelta)
print(f"analytic Phi(tau) = {Phi:.9f} rad, 2 Phi = {2 * Phi:.9f}, "
      f"-pi/4 = {-np.pi / 4:.9f}")
print(f"analytic |alpha(tau)| = "
      f"{abs(-g * (np.exp(1j * wdelta * tau) - 1) / wdelta):.3e} "
      f"(zero when delta tau is an integer number of cycles)")
print()


def evolve_ld(psi0, tau, delta_hz, gcoup):
    """Integrate the Lamb-Dicke Molmer-Sorensen Hamiltonian."""
    wd = TWOPI * delta_hz

    def rhs(t, y):
        H = gcoup * (A * np.exp(-1j * wd * t) + AD * np.exp(1j * wd * t)) @ SY
        return -1j * (H @ y)
    s = solve_ivp(rhs, [0.0, tau], psi0, rtol=1e-10, atol=1e-12,
                  method="DOP853")
    return s.y[:, -1]


def spin_block(nfock, tau, delta_hz, gcoup):
    """4x4 spin propagator: start in Fock state nfock and project back onto it.
    Its unitarity measures how completely the motion has been disentangled."""
    ev = evolve_ld
    U = np.zeros((4, 4), dtype=complex)
    for s0 in range(4):
        psi0 = np.zeros(DIM, dtype=complex)
        psi0[s0 * (NMAX + 1) + nfock] = 1.0
        out = ev(psi0, tau, delta_hz, gcoup)
        for s1 in range(4):
            U[s1, s0] = out[s1 * (NMAX + 1) + nfock]
    return U


# --- the analytic target -----------------------------------------------
SY4 = (np.kron(sy, I2) + np.kron(I2, sy))
U_target = expm(-1j * Phi * (SY4 @ SY4))
U0 = spin_block(0, tau, delta, g)
print("spin propagator, rows/columns ordered 00, 01, 10, 11")
print("  numerically integrated          |  analytic exp(-i Phi Sy^2)")
for r0, r1 in zip(np.round(U0, 4), np.round(U_target, 4)):
    fmt = lambda row: " ".join(f"{z.real:+.3f}{z.imag:+.3f}j" for z in row)
    print(f"  {fmt(r0)}  |  {fmt(r1)}")
print(f"  unitarity check ||U^dag U - I|| = "
      f"{np.linalg.norm(U0.conj().T @ U0 - np.eye(4)):.3e}")
print(f"  agreement with the analytic form ||U - U_target|| = "
      f"{np.linalg.norm(U0 - U_target):.3e}")
print()

# --- the Bell state ----------------------------------------------------
psi0 = np.zeros(DIM, dtype=complex)
psi0[0] = 1.0                                  # |00> |n=0>
out = evolve_ld(psi0, tau, delta, g)
bell = np.zeros(DIM, dtype=complex)
bell[0 * (NMAX + 1)] = 1.0 / np.sqrt(2.0)
bell[3 * (NMAX + 1)] = -1j / np.sqrt(2.0)      # (|00> - i|11>)/sqrt(2)
print("from |00>|n=0>:")
pops = [float(np.sum(np.abs(out[s * (NMAX + 1):(s + 1) * (NMAX + 1)]) ** 2))
        for s in range(4)]
print(f"  spin populations 00, 01, 10, 11 = "
      + ", ".join(f"{p:.6f}" for p in pops))
print(f"  mean phonon number at the end   = "
      f"{float(np.real(out.conj() @ (NOP @ out))):.3e}")
print(f"  fidelity with the target Bell state x |n=0> = "
      f"{abs(bell.conj() @ out) ** 2:.9f}")
print()

# --- insensitivity to the initial phonon number ------------------------
print(f"{'n initial':>11}{'||U(n) - U(0)||':>18}{'Bell fidelity':>16}")
for n0 in [0, 1, 3, 8]:
    Un = spin_block(n0, tau, delta, g)
    psi0 = np.zeros(DIM, dtype=complex)
    psi0[n0] = 1.0
    o = evolve_ld(psi0, tau, delta, g)
    b = np.zeros(DIM, dtype=complex)
    b[n0] = 1.0 / np.sqrt(2.0)
    b[3 * (NMAX + 1) + n0] = -1j / np.sqrt(2.0)
    print(f"{n0:>11}{np.linalg.norm(Un - U0):>18.3e}"
          f"{abs(b.conj() @ o) ** 2:>16.9f}")
print()

# --- what a mistuned gate does -----------------------------------------
print(f"{'delta tau (cycles)':>19}{'|alpha| analytic':>18}"
      f"{'spin-block norm defect':>24}{'Bell fidelity':>16}")
for cycles in [1.0, 1.02, 1.05, 1.1, 1.5]:
    d = cycles / tau
    wd = TWOPI * d
    al = abs(-g * (np.exp(1j * wd * tau) - 1) / wd)
    Um = spin_block(0, tau, d, g)
    psi0 = np.zeros(DIM, dtype=complex)
    psi0[0] = 1.0
    o = evolve_ld(psi0, tau, d, g)
    print(f"{cycles:>19.2f}{al:>18.4f}"
          f"{np.linalg.norm(Um.conj().T @ Um - np.eye(4)):>24.3e}"
          f"{abs(bell.conj() @ o) ** 2:>16.6f}")
print()

# --- beyond Lamb-Dicke -------------------------------------------------
D0 = expm(1j * eta * (a + a.conj().T))
nvec = np.arange(NMAX + 1)
SPS = np.array(spin(sp, 0) + spin(sp, 1))          # 4x4 spin raising sum
SPSd = SPS.conj().T


def evolve_exact(psi0, tau, delta_hz, gcoup):
    """Integrate the full bichromatic Hamiltonian, no Lamb-Dicke expansion.

    Expanding D to first order reproduces the Lamb-Dicke form with coupling
    Omega eta / 2, so the carrier Rabi frequency that matches a given g is
    Omega = 2 g / eta.
    """
    wm = TWOPI * f_mode
    wbi = TWOPI * (f_mode - delta_hz)
    Om = 2.0 * gcoup / eta

    def rhs(t, y):
        ph = np.exp(1j * wm * t * nvec)
        Dt = (ph[:, None] * D0) * np.conj(ph)[None, :]
        Y = y.reshape(4, NMAX + 1)
        out = SPS @ Y @ Dt.T + SPSd @ Y @ Dt.conj()
        return (-1j * Om * np.cos(wbi * t) * out).ravel()
    s = solve_ivp(rhs, [0.0, tau], psi0, rtol=1e-9, atol=1e-11,
                  method="DOP853")
    return s.y[:, -1]


print()
print(f"{'n initial':>11}{'Bell fidelity (LD)':>21}{'Bell fidelity (exact)':>23}"
      f"{'infidelity (exact)':>20}")
for n0 in [0, 1, 3, 8]:
    b = np.zeros(DIM, dtype=complex)
    b[n0] = 1.0 / np.sqrt(2.0)
    b[3 * (NMAX + 1) + n0] = -1j / np.sqrt(2.0)
    p0 = np.zeros(DIM, dtype=complex)
    p0[n0] = 1.0
    f_ld = abs(b.conj() @ evolve_ld(p0, tau, delta, g)) ** 2
    oe = evolve_exact(p0, tau, delta, g)
    # the sign of the geometric phase depends on the drive phase, so accept
    # either of the two Bell states that -i and +i give
    bp = np.zeros(DIM, dtype=complex)
    bp[n0] = 1.0 / np.sqrt(2.0)
    bp[3 * (NMAX + 1) + n0] = +1j / np.sqrt(2.0)
    f_ex = max(abs(b.conj() @ oe) ** 2, abs(bp.conj() @ oe) ** 2)
    print(f"{n0:>11}{f_ld:>21.9f}{f_ex:>23.9f}{1 - f_ex:>20.3e}")
print()
print(f"    Debye-Waller reduction at n = 0: {eta ** 2 / 2 * 100:.3f}% of the "
      f"coupling")
```

```text
eta = 0.0684, mode f = 2.0 MHz, gate time tau = 100 us, loops K = 1
detuning from the sideband delta/2pi = 10.000 kHz
coupling g/2pi = 2.5000 kHz  -> carrier Rabi Omega/2pi = 73.099 kHz
analytic Phi(tau) = -0.392699082 rad, 2 Phi = -0.785398163, -pi/4 = -0.785398163
analytic |alpha(tau)| = 1.608e-16 (zero when delta tau is an integer number of cycles)

spin propagator, rows/columns ordered 00, 01, 10, 11
  numerically integrated          |  analytic exp(-i Phi Sy^2)
  +0.500+0.500j +0.000+0.000j +0.000+0.000j +0.500-0.500j  |  +0.500+0.500j +0.000+0.000j +0.000+0.000j +0.500-0.500j
  +0.000+0.000j +0.500+0.500j -0.500+0.500j +0.000+0.000j  |  +0.000+0.000j +0.500+0.500j -0.500+0.500j +0.000+0.000j
  +0.000+0.000j -0.500+0.500j +0.500+0.500j +0.000+0.000j  |  +0.000+0.000j -0.500+0.500j +0.500+0.500j +0.000+0.000j
  +0.500-0.500j +0.000+0.000j +0.000+0.000j +0.500+0.500j  |  +0.500-0.500j +0.000+0.000j +0.000+0.000j +0.500+0.500j
  unitarity check ||U^dag U - I|| = 3.493e-12
  agreement with the analytic form ||U - U_target|| = 5.771e-12

from |00>|n=0>:
  spin populations 00, 01, 10, 11 = 0.500000, 0.000000, 0.000000, 0.500000
  mean phonon number at the end   = 8.475e-25
  fidelity with the target Bell state x |n=0> = 1.000000000

  n initial   ||U(n) - U(0)||   Bell fidelity
          0         0.000e+00     1.000000000
          1         3.104e-12     1.000000000
          3         8.131e-12     1.000000000
          8         1.832e-11     1.000000000

 delta tau (cycles)  |alpha| analytic  spin-block norm defect   Bell fidelity
               1.00            0.0000               3.493e-12        1.000000
               1.02            0.0308               5.349e-03        0.997182
               1.05            0.0745               3.104e-02        0.983814
               1.10            0.1405               1.073e-01        0.945000
               1.50            0.3333               5.074e-01        0.757025


  n initial   Bell fidelity (LD)  Bell fidelity (exact)  infidelity (exact)
          0          1.000000000            0.999943995           5.600e-05
          1          1.000000000            0.999760759           2.392e-04
          3          1.000000000            0.999090978           9.090e-04
          8          1.000000000            0.995730633           4.269e-03

    Debye-Waller reduction at n = 0: 0.234% of the coupling
```

**What to notice.** The analytic $2\Phi$ comes out at $-0.785398163$, which is $-\pi/4$ to nine digits, and $|\alpha(\tau)| = 1.6\times10^{-16}$ — the loop closes to floating-point precision. The numerically integrated spin propagator agrees with $e^{-i\Phi S_y^2}$ to $6\times10^{-12}$ and is unitary to $3.5\times10^{-12}$, which means the motion has genuinely factored out. From $|00\rangle|n=0\rangle$ the gate produces exactly half population in $|00\rangle$ and half in $|11\rangle$, leaves $8\times10^{-25}$ phonons behind, and reaches the Bell state with fidelity 1.000000000.

The thermal-insensitivity table is the important one, and the numbers are almost too clean: starting from $n = 0$, 1, 3 or 8 gives the *same* spin propagator to $2\times10^{-11}$. That is not an approximate cancellation, it is an exact property of a Hamiltonian that is linear in $a$ and independent of $\hat{n}$, and the numerics confirm it rather than discovering it.

The mistuning table shows what "loop closure" means operationally. At $\delta\tau = 1.02$ cycles the loop misses by $|\alpha| = 0.031$, the spin block loses unitarity by $5\times10^{-3}$, and the Bell fidelity drops to 0.997. At 1.10 cycles the fidelity is 0.945. The failure is not a phase error one could calibrate away: the residual displacement means the motion carries away which-path information about the spin state, so the two-qubit operation is not a unitary at all. A 2% timing error costs $3\times10^{-3}$ of fidelity, which sets the stability requirement on the pulse and the mode frequency.

The final table breaks the Lamb-Dicke approximation and is worth the extra machinery it takes. Writing the displacement operator as $D(t) = R(\omega t)D(i\eta)R^\dagger(\omega t)$ with $R$ diagonal makes the exact bichromatic Hamiltonian cheap to integrate, and the result is that the ideal gate loses $5.6\times10^{-5}$ at $n = 0$ — of order $\eta^4$ — rising to $4.3\times10^{-3}$ at $n = 8$.

Two mechanisms share that loss, and it is worth separating them because they fail differently. The first is the **Debye-Waller factor** $e^{-\eta^2(2n+1)/2}$, which reduces the sideband coupling by 0.23% at $n = 0$ and by 3.9% at $n = 8$. Since the geometric phase goes as the square of the coupling, $\Phi$ falls short of $\pi/4$ — by 7.9% at $n = 8$, against the 7.8% that the Debye-Waller factor predicts, which is as close as an $\eta^4$ estimate gets. Note how that deficit *appears*: not as a wrong relative phase between $|00\rangle$ and $|11\rangle$, which the exact integration gets right to $2.6\times10^{-5}$ rad, but as an incomplete transfer. The populations come out $\cos^2\Phi$ and $\sin^2\Phi$ instead of one half each — an imbalance of 1.1% at $n = 0$ and 12.4% at $n = 8$ — and the resulting infidelity $\cos^2 2\Phi/4$ is $3.0\times10^{-5}$ at $n = 0$ and $3.9\times10^{-3}$ at $n = 8$. So this channel supplies 53% of the loss at $n = 0$ and 90% at $n = 8$, and being an over- or under-rotation it can in principle be calibrated out at a known $\bar{n}$.

The second is **residual spin-motion entanglement**: the loop no longer closes exactly, so $2.6\times10^{-5}$ of the population at $n = 0$ (and $4.0\times10^{-4}$ at $n = 8$) is left in $|01\rangle$ and $|10\rangle$ *with the motion displaced by one phonon*. That part cannot be calibrated away by adjusting a pulse area, because the motion carries off which-path information — the same failure mode as the mistuning table, in miniature. At $n = 0$ the two channels are comparable; as the ion warms, the calibratable one grows faster.

So the thermal insensitivity is a leading-order statement, not an exact one, and the correction grows with both $\bar{n}$ and $\eta$. That is why a Mølmer-Sørensen gate is still preceded by ground-state cooling even though it does not strictly require it, and why the light ions that give strong coupling also pay the largest non-Lamb-Dicke penalty. The exact calculation tells you which; the approximate one cannot.

* * *

## 3.5 All-to-All Connectivity and Its Price

### What the shared bus buys

Because every ion participates in every mode, any pair can be entangled directly. On a superconducting chip a gate between distant qubits needs a chain of SWAPs, and the depth overhead for a general circuit can be large; on an ion chain it is one gate. For algorithms whose natural interaction graph is not planar — quantum chemistry with its all-to-all Coulomb terms is the standard example — this is a genuine architectural advantage, and it shows in the effective circuit depth rather than in any single-gate number.

Two further advantages follow from the same physics. All ions of a species are identical, so there is no frequency-collision problem and no fabrication yield: a chain of ions has no analogue of the crowded, individually calibrated frequency landscape of a multi-qubit chip. And qubit coherence times are seconds, four to five orders of magnitude longer than the 10-100 $\mu$s gate time.

### What it costs

**Gate speed.** The bus is a mechanical oscillator at a few MHz, and the gate detuning $\delta$ must be small compared with the mode spacing. Gate times are tens to hundreds of microseconds, against tens of nanoseconds for superconducting circuits. Against seconds of coherence that ratio is still favourable, but the wall-clock time of a deep circuit is not.

**Mode crowding.** Example 3 quantified it: at the linear-stability margin the minimum transverse mode spacing falls as $N^{-0.8}$, so the minimum gate time grows as $N^{0.8}$. A chain of 30 ions needs gates four times slower than a chain of two, before any other consideration.

**Spectator modes.** The gate addresses one mode, but the drive is not spectrally clean, and residual coupling to neighbouring modes leaves residual spin-motion entanglement — the mistuning failure of Example 6, once per unwanted mode. Multi-tone and amplitude-shaped pulses that close *all* the phase-space loops simultaneously are the standard remedy, and they cost pulse-shaping bandwidth and calibration effort.

**Cooling and readout overhead.** Every experimental cycle re-cools the chain (milliseconds, Example 5) and reads it out by state-dependent fluorescence (hundreds of microseconds). Both scale awkwardly with $N$.

### The architectural response

The response to mode crowding is not to fix it but to avoid it: keep chains short and connect them.

**QCCD** (quantum charge-coupled device) uses a segmented trap with many electrodes, holding several small chains in separate zones and physically *shuttling* ions between zones by ramping the DC voltages. Gates always happen in a short chain, so the mode spectrum stays clean; connectivity comes from transport. In principle the ion is moved adiabatically and the motional state is preserved; in practice transport heats the ion, and the heating rate is set by the same electric-field noise as everything else in Section 3.6. Transport is also slow compared with a gate, so the effective connectivity is bought with time.

**Photonic interconnects** entangle ions in physically separate traps by interfering photons they emit and detecting a coincidence. The success probability per attempt is small — solid angle, fibre coupling, detector efficiency — so the link rate is low, but it is heralded: when the detection pattern is right, the entanglement is there. That converts a hard scaling problem into a rate problem, which is a better kind of problem.

Both are principles rather than solved engineering, and both are limited by the same underlying materials issue.

* * *

## 3.6 Anomalous Heating: A Surface-Materials Problem

### The observation

An ion cooled to $|n = 0\rangle$ and then left alone in the dark heats up. The rate is measured by waiting a variable time and then doing sideband thermometry, and it comes out one to several orders of magnitude *larger* than the Johnson noise of the trap circuit can account for. This is **anomalous heating**, and it has been the central technical obstacle of the platform ever since it was first measured.

The physics is simple to state. Fluctuating electric field at the ion position drives the motion, and for a field-noise spectral density $S_E(\omega)$ in $\mathrm{V}^2\mathrm{m}^{-2}\mathrm{Hz}^{-1}$,

$$ \frac{d\bar{n}}{dt} = \frac{e^2 S_E(\omega)}{4m\hbar\omega} $$

(the charge of a singly ionized atom is written $e$ here, not $q$, because $q$ is already the Mathieu parameter of Section 3.1)

All the difficulty is in $S_E$, and $S_E$ is a materials quantity.

### What the scaling says

The empirical facts, and what each implies:

  * $S_E \propto d^{-4}$, with measured exponents between 3.5 and 4, where $d$ is the ion-electrode distance. That is the signature of **uncorrelated patches** on the surface: a fluctuating patch of area $A$ produces a field falling as $A/d^3$, and $N \sim (d/\sqrt{A})^2$ independent patches within the relevant solid angle add in quadrature, giving $d^{-4}$ overall. A source correlated over a patch much larger than $d$ would fall off far more slowly, so the measured exponent is itself the evidence that the patches are small and independent.
  * $S_E \propto 1/f^\alpha$ with $\alpha$ near 1. A $1/f$ spectrum is the signature of a broad distribution of activation energies — exactly the same signature that two-level defects leave in the dielectric loss of a superconducting qubit.
  * Cooling the electrodes to 4-10 K suppresses $S_E$ by one to two orders of magnitude. Johnson noise cannot explain that: its spectral density is proportional to $T$, so 300 K $\to$ 10 K buys a factor of 30 — one and a half decades — and the observed suppression exceeds it while also depending on how the surface was prepared. The mechanism is thermally activated, with an activation energy rather than a linear temperature dependence.
  * *In situ* argon-ion milling of the electrode surface reduces $S_E$ by one to two orders of magnitude, and the improvement decays over days as the surface re-adsorbs contaminants from the residual gas.
  * Traps of identical geometry made by different processes differ by orders of magnitude.

Put together, these say: the noise comes from thermally activated motion of adsorbates and patch potentials in the last few nanometres of the electrode surface. It is not a property of the Paul trap.

### Code Example 7: The Heating Budget

```python
"""Chapter 3, Example 7: anomalous heating as a surface-materials problem.

Electric-field noise at the ion position drives the motion:
    dn/dt = e^2 S_E(omega) / (4 m hbar omega),   e = ion charge,
so the heating rate is set by one materials quantity, the noise spectral
density S_E of the electrode surface, and by trap geometry through it."""
import numpy as np

hbar = 1.054571817e-34
e = 1.602176634e-19
u_amu = 1.66053906660e-27
kB = 1.380649e-23
TWOPI = 2.0 * np.pi


def heating_rate(S_E, mass_u, f_trap):
    """Quanta per second from a field noise density S_E in V^2 m^-2 Hz^-1."""
    m = mass_u * u_amu
    w = TWOPI * f_trap
    return e ** 2 * S_E / (4.0 * m * hbar * w)


# reference point: a room-temperature surface trap
D_REF, F_REF, S_REF = 50e-6, 1.0e6, 1.0e-11
print(f"reference surface trap: d = {D_REF * 1e6:.0f} um, "
      f"f = {F_REF / 1e6:.1f} MHz, S_E = {S_REF:.0e} V^2 m^-2 Hz^-1")
print(f"For Ca+ that is dn/dt = {heating_rate(S_REF, 40.078, F_REF):.1f} "
      f"quanta/s.")
print()

print(f"{'d (um)':>9}{'S_E (V^2/m^2/Hz)':>19}{'dn/dt (quanta/s)':>19}"
      f"{'quanta in 100 us':>19}")
for d in [20e-6, 50e-6, 100e-6, 500e-6]:
    S = S_REF * (D_REF / d) ** 4
    r = heating_rate(S, 40.078, F_REF)
    print(f"{d * 1e6:>9.0f}{S:>19.3e}{r:>19.3f}{r * 100e-6:>19.3e}")
print()

print(f"{'f (MHz)':>9}{'S_E (1/f model)':>18}{'dn/dt (quanta/s)':>19}"
      f"{'relative':>11}")
base = None
for f in [0.5e6, 1.0e6, 2.0e6, 5.0e6]:
    S = S_REF * (F_REF / f)
    r = heating_rate(S, 40.078, f)
    base = base or r
    print(f"{f / 1e6:>9.1f}{S:>18.3e}{r:>19.3f}{r / base:>11.3f}")
print()

print(f"{'electrode T':>13}{'suppression':>13}{'S_E':>13}"
      f"{'dn/dt (quanta/s)':>19}{'quanta in 100 us':>19}")
for T, supp in [(300.0, 1.0), (77.0, 10.0), (10.0, 100.0), (4.0, 200.0)]:
    S = S_REF / supp
    r = heating_rate(S, 40.078, F_REF)
    print(f"{T:>11.0f} K{supp:>13.0f}x{S:>13.3e}{r:>19.4f}"
          f"{r * 100e-6:>19.3e}")
print()

print(f"{'ion':>7}{'m (u)':>9}{'dn/dt (quanta/s)':>19}{'relative':>11}")
ref = heating_rate(S_REF, 40.078, F_REF)
for name, m_u in [("Be+", 9.012), ("Ca+", 40.078), ("Yb+", 171.0)]:
    r = heating_rate(S_REF, m_u, F_REF)
    print(f"{name:>7}{m_u:>9.3f}{r:>19.3f}{r / ref:>11.3f}")
print()

# --- the gate error budget ---------------------------------------------
print(f"{'trap':>34}{'dn/dt':>12}{'tau = 30 us':>14}{'100 us':>12}{'1 ms':>12}")
configs = [
    ("macroscopic, 300 K, d = 500 um", 500e-6, 1.0, 1.0e6),
    ("surface, 300 K, d = 50 um", 50e-6, 1.0, 1.0e6),
    ("surface, 10 K, d = 50 um", 50e-6, 100.0, 1.0e6),
    ("surface, 10 K, d = 50 um, 3 MHz", 50e-6, 100.0, 3.0e6),
    ("surface, 10 K, d = 30 um, 3 MHz", 30e-6, 100.0, 3.0e6),
]
for lab, d, supp, f in configs:
    S = S_REF * (D_REF / d) ** 4 * (F_REF / f) / supp
    r = heating_rate(S, 40.078, f)
    print(f"{lab:>34}{r:>12.3f}"
          + "".join(f"{r * t:>12.2e}" for t in [30e-6, 100e-6, 1e-3]))
print()

# --- what the surface is doing -----------------------------------------
print()
S_needed = S_REF * 1e-4 / (heating_rate(S_REF, 40.078, 1e6) * 100e-6)
print(f"To reach 1e-4 absorbed quanta in a 100 us gate on Ca+ at 1 MHz needs")
print(f"S_E = {S_needed:.3e} V^2 m^-2 Hz^-1, i.e. "
      f"{S_REF / S_needed:.0f}x below the reference above")
```

```text
reference surface trap: d = 50 um, f = 1.0 MHz, S_E = 1e-11 V^2 m^-2 Hz^-1
For Ca+ that is dn/dt = 1455.3 quanta/s.

   d (um)   S_E (V^2/m^2/Hz)   dn/dt (quanta/s)   quanta in 100 us
       20          3.906e-10          56847.277          5.685e+00
       50          1.000e-11           1455.290          1.455e-01
      100          6.250e-13             90.956          9.096e-03
      500          1.000e-15              0.146          1.455e-05

  f (MHz)   S_E (1/f model)   dn/dt (quanta/s)   relative
      0.5         2.000e-11           5821.161      1.000
      1.0         1.000e-11           1455.290      0.250
      2.0         5.000e-12            363.823      0.062
      5.0         2.000e-12             58.212      0.010

  electrode T  suppression          S_E   dn/dt (quanta/s)   quanta in 100 us
        300 K            1x    1.000e-11          1455.2903          1.455e-01
         77 K           10x    1.000e-12           145.5290          1.455e-02
         10 K          100x    1.000e-13            14.5529          1.455e-03
          4 K          200x    5.000e-14             7.2765          7.276e-04

    ion    m (u)   dn/dt (quanta/s)   relative
    Be+    9.012           6471.940      4.447
    Ca+   40.078           1455.290      1.000
    Yb+  171.000            341.083      0.234

                              trap       dn/dt   tau = 30 us      100 us        1 ms
    macroscopic, 300 K, d = 500 um       0.146    4.37e-06    1.46e-05    1.46e-04
         surface, 300 K, d = 50 um    1455.290    4.37e-02    1.46e-01    1.46e+00
          surface, 10 K, d = 50 um      14.553    4.37e-04    1.46e-03    1.46e-02
   surface, 10 K, d = 50 um, 3 MHz       1.617    4.85e-05    1.62e-04    1.62e-03
   surface, 10 K, d = 30 um, 3 MHz      12.477    3.74e-04    1.25e-03    1.25e-02


To reach 1e-4 absorbed quanta in a 100 us gate on Ca+ at 1 MHz needs
S_E = 6.871e-15 V^2 m^-2 Hz^-1, i.e. 1455x below the reference above
```

**What to notice.** The reference point — a room-temperature surface trap with $d = 50\ \mu$m — gives 1455 quanta per second for Ca$^+$ at 1 MHz. That is a hostile number: the ion gains a quantum every 700 $\mu$s, comparable to a gate time.

The geometry scan shows why miniaturization is not free. Going from $d = 500\ \mu$m to $d = 20\ \mu$m raises the heating rate from 0.15 to 57000 quanta per second — a factor of $4\times10^5$ from a factor of 25 in distance, which is $25^4$. Every argument for shrinking a trap (more electrodes, tighter confinement, integrated optics and electronics, faster shuttling) runs directly into $d^{-4}$. This is the platform's central engineering tension, and it is quantitative.

The frequency scan is a free win: because $S_E \sim 1/f$ and the heating rate carries an explicit $1/\omega$, the rate falls as $1/f^2$. A 5 MHz mode heats 100 times more slowly than a 0.5 MHz one. Combined with the deeper sideband cooling limit of Example 5 and the smaller $\eta$, this is why gates use high-frequency transverse modes.

The temperature scan is the reason cryogenic ion traps exist. Two orders of magnitude at 10 K takes the reference from 1455 to 14.6 quanta per second, and it costs a cryostat rather than a research programme.

The error budget table pulls it together. A gate-error target of $10^{-4}$ requires of order $10^{-4}$ quanta absorbed during the gate, and the table says which configurations can supply it: a room-temperature surface trap at 50 $\mu$m cannot (it absorbs 0.15 quanta in 100 $\mu$s, three orders of magnitude too many), a cryogenic one at 50 $\mu$m and 3 MHz just about can ($1.6\times10^{-4}$), and shrinking that same cryogenic trap to 30 $\mu$m loses a factor of eight and puts it out of reach again. Read that as a conditional rather than a slogan: **once the target is $10^{-4}$ gate error at an electrode distance of tens of micrometres, cryogenic operation is a requirement rather than a convenience** — while a small register in a larger, room-temperature trap remains entirely workable at more modest error targets, which is where much of the field's work has been done. What is unconditional is the tension: trap miniaturization and gate fidelity pull in opposite directions.

The final line states the materials target as a number: reaching $10^{-4}$ absorbed quanta in a 100 $\mu$s gate on Ca$^+$ at 1 MHz needs $S_E = 7\times10^{-15}\ \mathrm{V^2m^{-2}Hz^{-1}}$, a factor of 1455 below the room-temperature surface-trap reference. Some of that comes from cryogenics, some from surface treatment, some from geometry and mode choice. All of it is available in principle, and none of it is available by better circuit design.

### The same pattern, again

Set Chapter 2 and Chapter 3 side by side.

| | superconducting transmon | trapped ion |
| --- | --- | --- |
| The Hamiltonian | known exactly, designed to a percent | a property of the periodic table |
| What limits it | $T_1$ from dielectric loss and quasiparticles | $\bar{n}$ growth from electric-field noise |
| Where the loss lives | 3 nm of amorphous oxide on the surfaces | the last few nanometres of the electrode |
| Spectral signature | $1/f$, broad distribution of activation energies | $1/f$, broad distribution of activation energies |
| Response to cleaning | in-situ clean, encapsulation, different oxide | argon-ion milling, decays over days |
| Response to cooling | 20 mK required, quasiparticles non-thermal | 4-10 K gives $10^2$ suppression |
| Geometric lever | $p \propto t/w$: dilute the field | $S_E \propto d^{-4}$: move away from the surface |
| Nominally identical devices differ by | an order of magnitude in $T_1$ | orders of magnitude in $S_E$ |

Two platforms with nothing physically in common — one a centimetre of aluminium at 20 millikelvin, the other a single atom in vacuum — are limited by the same thing: uncontrolled dynamics in a few nanometres of surface, with a $1/f$ spectrum, responsive to cleaning and to cooling, and irreproducible between nominally identical devices. That is not a coincidence, and it is the argument of this course. The bottleneck of quantum hardware is a materials problem, and it is the same materials problem twice.

* * *

## Exercises

Work through these with the code from this chapter in front of you. Solutions follow each question.

#### Exercise 1: Designing a Trap

A trap for $^{171}$Yb$^+$ ($m = 171$ u) must give a 3 MHz radial secular frequency with $q = 0.25$, using $r_0 = 200\ \mu$m. (a) What RF frequency and amplitude are needed? (b) What is the trap depth in eV? (c) What happens to $q$ if the same trap is loaded with $^{40}$Ca$^+$ at the same voltages?

<details><summary>Solution</summary>
<p>(a) At \(q = 0.25\), \(\beta \approx q/\sqrt{2} = 0.177\) (Example 1 gives 0.178 exactly), so \(\Omega = 2\omega_\mathrm{sec}/\beta = 2\times2\pi\times3\times10^6/0.178\), i.e. \(\Omega/2\pi = 33.7\) MHz. Then \(V = q m r_0^2\Omega^2/2e\): with \(m = 171\times1.66\times10^{-27} = 2.84\times10^{-25}\) kg, \(r_0^2 = 4\times10^{-8}\) m\(^2\) and \(\Omega = 2.12\times10^8\) s\(^{-1}\), \(V = 0.25\times2.84\times10^{-25}\times4\times10^{-8}\times4.49\times10^{16}/(2\times1.602\times10^{-19}) = 398\) V.</p>
<p>(b) \(U = \frac{1}{2}m\omega_\mathrm{sec}^2 r_0^2 = 0.5\times2.84\times10^{-25}\times(1.885\times10^7)^2\times4\times10^{-8} = 2.02\times10^{-18}\) J = 12.6 eV.</p>
<p>(c) \(q \propto 1/m\), so Ca\(^+\) at the same voltages has \(q = 0.25\times171/40 = 1.07\), which is above the stability boundary 0.908 — the lighter ion is not trapped at all. This is exploited: mass-selective trapping is how an unwanted species is expelled, and it is also why a trap loaded with two species needs its parameters checked for both.</p>
</details>

#### Exercise 2: Compensating Micromotion

Using Example 2's model for Ca$^+$ at $q = 0.3$: (a) what stray field gives a sideband-to-carrier ratio of exactly 1%? (b) If the compensation electrode can null the field to 0.2 V/m, what residual ratio remains? (c) Why does the answer depend on the laser wavelength, and which qubit type is more forgiving?

<details><summary>Solution</summary>
<p>(a) The table brackets it: 10 V/m gives \(7.2\times10^{-3}\). Since the ratio is \((J_1/J_0)^2 \approx (\mathrm{mi}/2)^2\) for small modulation index, and mi \(\propto E_\mathrm{stray}\), scaling from 10 V/m needs \(\sqrt{0.01/0.00718} = 1.18\), i.e. about 11.8 V/m.</p>
<p>(b) The ratio scales as \(E^2\): \((0.2/11.8)^2\times0.01 = 2.9\times10^{-6}\). Compensation to a fraction of a V/m makes micromotion sidebands negligible — which is exactly why the procedure is worth the trouble.</p>
<p>(c) The modulation index is \(k x_\mathrm{micro}\), so it is proportional to \(1/\lambda\) — or, for a Raman-driven hyperfine qubit, to the <em>difference</em> wavevector. A microwave-driven hyperfine transition has \(k\) smaller by four orders of magnitude and is essentially immune to micromotion; a 729 nm optical qubit is not. That immunity is one of the underappreciated advantages of hyperfine qubits, and it is lost as soon as a Raman pair is used for gates.</p>
</details>

#### Exercise 3: When Does the Chain Buckle?

(a) Using Example 3, what is the largest linear chain at $\omega_r/\omega_z = 10$? (b) A 20-ion chain is wanted with $f_z = 200$ kHz. What radial frequency does linearity require, and what is the resulting transverse mode spacing? (c) What gate time does that imply, and what does it say about scaling a single chain?

<details><summary>Solution</summary>
<p>(a) Interpolating the printed table between ratios 8 (\(N = 18\)) and 12 (\(N = 30\)), ratio 10 gives about \(N = 24\). Running the loop at \(\mathrm{ratio} = 10\) confirms it.</p>
<p>(b) The criterion \(\omega_r/\omega_z \gtrsim 0.73 N^{0.86}\) at \(N = 20\) gives 9.6, so \(f_r \gtrsim 1.9\) MHz; take 2 MHz. The minimum transverse spacing is \(\approx \omega_z^2/2\omega_r = (0.2)^2/(2\times2) = 0.01\) MHz = 10 kHz.</p>
<p>(c) A gate detuning must be well inside 10 kHz, so \(\tau \gtrsim 1/10\) kHz \(= 100\ \mu\)s, and in practice several times that once the loop-closure condition \(\delta\tau = 2\pi K\) and the spectator modes are accounted for. Meanwhile the heating rate at 200 kHz is 25 times worse than at 1 MHz (Example 7), and the gate is ten times longer, so the absorbed quanta go up by more than two orders of magnitude. Weak axial confinement makes the chain long and the physics bad in three ways at once. This is the quantitative case for short chains plus transport.</p>
</details>

#### Exercise 4: Cooling Budget

(a) Using Example 5, what effective linewidth is needed to reach $\bar{n} = 0.01$ in a 1 MHz trap? (b) At the $R_c$ that follows, how long does cooling from the Doppler limit take? (c) Why is quenching a narrow transition (deliberately broadening it) the standard technique rather than using the natural linewidth?

<details><summary>Solution</summary>
<p>(a) \(\bar{n}_\mathrm{ss} = (\Gamma/4\omega)^2 = 0.01\) needs \(\Gamma = 0.4\omega\), i.e. \(\Gamma/2\pi = 400\) kHz at \(f = 1\) MHz.</p>
<p>(b) From the printed formula \(R_c = 4\eta^2\Omega^2/\Gamma\) with \(\Omega/2\pi = 50\) kHz, \(\eta = 0.097\) at 1 MHz (Example 4), \(\Gamma/2\pi = 400\) kHz: \(R_c = 4\times0.0094\times(3.14\times10^5)^2/(2.51\times10^6) = 1.48\times10^3\) s\(^{-1}\). The trajectory of Example 5 shows the approach taking roughly \(10/R_c\), so about 7 ms.</p>
<p>(c) Because the two requirements pull in opposite directions. A narrow line gives a deep limit \((\Gamma/4\omega)^2\) but a small \(R_c \propto \Gamma^{-1}\) at fixed \(\Omega\) — and, more importantly, the <em>cycle</em> rate is bounded by how fast the excited state can be recycled, which for a second-long metastable state is hopeless. Quenching with an auxiliary laser sets \(\Gamma_\mathrm{eff}\) to whatever value optimizes the product of depth and speed, typically a few hundred kHz. It is a tunable parameter, which the natural linewidth is not.</p>
</details>

#### Exercise 5: The MS Gate at Two Loops

Repeat Example 6 with $K = 2$ at the same gate duration $\tau = 100\ \mu$s. (a) What are $\delta$ and $g$? (b) Verify numerically that the Bell fidelity is still 1 in the Lamb-Dicke limit. (c) What is the practical argument for $K > 1$, and what is the argument against?

<details><summary>Solution</summary>
<p>(a) \(\delta = K/\tau = 20\) kHz, and \(g = 2\pi\delta/(4\sqrt{K}) = 2\pi\times20\,\mathrm{kHz}/5.657 = 2\pi\times3.54\) kHz, i.e. \(\eta\Omega/2\pi = 7.07\) kHz and \(\Omega/2\pi = 103\) kHz. Setting <code>K = 2</code> in the script does all of this.</p>
<p>(b) It is, to nine digits: the analytic \(2\Phi\) is again \(-\pi/4\), \(|\alpha(\tau)|\) is again zero at machine precision, and the spin block is again unitary. Nothing about the derivation privileged \(K = 1\).</p>
<p>(c) For: a larger \(\delta\) is further from every unwanted mode, so residual coupling to spectator modes is suppressed and the phase-space loops of those modes are closer to closing too. Also, the larger detuning relaxes the requirement on mode-frequency stability. Against: \(g\) must grow as \(\sqrt{K}\) at fixed \(\tau\), so the laser intensity grows linearly in \(K\), which brings more off-resonant carrier excitation, more photon scattering, and more AC Stark shift to calibrate. \(K = 1\) or 2 is the usual compromise.</p>
</details>

#### Exercise 6: Heating and the Choice of Ion

Using Example 7: (a) why does Be$^+$ heat 4.4 times faster than Ca$^+$ at the same $S_E$ and frequency? (b) Be$^+$ also has $\eta = 0.55$ against 0.068, so its gates can be much faster. Which effect wins for the absorbed-quanta budget? (c) What does this say about choosing a species?

<details><summary>Solution</summary>
<p>(a) \(d\bar{n}/dt \propto 1/m\), and \(40.078/9.012 = 4.45\). A light ion is easier to shake for the same field noise.</p>
<p>(b) The gate condition is \(\eta\Omega = \delta/2\sqrt{K}\) and \(\tau = 2\pi K/\delta\), so at fixed laser intensity a larger \(\eta\) permits a larger \(\delta\) and hence a shorter \(\tau\), roughly \(\tau \propto 1/\eta\) at fixed \(\Omega\). Be\(^+\) gains a factor of 8 in gate time against a factor of 4.4 in heating rate, so the absorbed quanta \(\dot{\bar{n}}\tau\) improve by about a factor of 2. The light ion wins — but only barely, and the margin depends entirely on how much laser power is available.</p>
<p>(c) That the species choice is a multi-way compromise with no dominant term: mass sets the heating rate and \(\eta\); the level structure decides hyperfine against optical and therefore the coherence mechanism; the wavelengths decide whether the lasers are commercial or heroic (Be\(^+\) needs 313 nm, generated by frequency mixing; Ca\(^+\) and Sr\(^+\) sit near convenient diode wavelengths); and the mass ratio matters again if two species are co-trapped for sympathetic cooling and mid-circuit readout. Different groups have optimized different terms, which is why the field has not converged on one ion.</p>
</details>

* * *

## Summary

### Key Takeaways

**1\. A Paul trap works because the saddle rotates**

  * Earnshaw forbids static trapping; an RF quadrupole confines dynamically, and the radial equation is the canonical Mathieu equation.
  * Floquet analysis gives the stability boundary at $a = 0$ as $q = 0.908046$ (textbook 0.90800), with $\det M = 1$ as a built-in check.
  * $\beta \approx q/\sqrt{2}$ is good to 1.8% at $q = 0.3$; a 0.5 mm trap at 20 MHz and 245 V gives 2.16 MHz secular motion and a 9.6 eV ($10^5$ K) well.

**2\. Micromotion is a surface problem in disguise**

  * The exact Floquet solution has sidebands at $\pm\Omega$ with relative amplitude $q/2$; verified as 0.02502 against 0.025 at $q = 0.05$.
  * A 1 V/m stray field displaces the ion 13 nm and gives a $7\times10^{-5}$ sideband-to-carrier ratio; at 100 V/m the sideband is stronger than the carrier.
  * The stray field comes from charged insulating patches and drifts, so compensation is a routine, not a calibration.

**3\. The ion crystal supplies the bus, and limits the speed**

  * Two ions: separation $2\times4^{-1/3}$, modes at 1 and $\sqrt{3}$ — reproduced to nine digits, which licenses the $N = 30$ results.
  * Linear stability requires $\omega_r/\omega_z \gtrsim 0.73N^{0.86}$, reproduced numerically from the sign of the lowest transverse eigenvalue.
  * At the stability margin the minimum transverse mode spacing falls as $N^{-0.8}$, so the minimum gate time grows as $N^{0.8}$: 10 $\mu$s at $N = 2$, 36 $\mu$s at $N = 30$. This, not the electrode count, is why single long chains lost.

**4\. Cooling has two stages with unrelated limits**

  * Doppler: $T = \hbar\Gamma/2k_B$, so every 20 MHz cooling line stops at 0.5 mK and $\bar{n} \approx 5$ in a 2 MHz trap.
  * Sideband: $\bar{n}_\mathrm{ss} = (\Gamma/4\omega)^2$, confirmed to three digits over four decades; the limit is spectral resolution, not temperature.
  * Sideband-ratio thermometry inverts exactly, by detailed balance rather than by the Lamb-Dicke approximation — no absolute calibration needed.

**5\. The Mølmer-Sørensen gate is a geometric phase, and it is exact**

  * The Magnus expansion terminates, giving $U = e^{-i\Phi S_y^2}D(\alpha S_y)$; numerics match to $6\times10^{-12}$ and the loop closes to $10^{-16}$.
  * $\Phi$ contains no $n$: the same spin propagator to $2\times10^{-11}$ from $n = 0$, 1, 3, 8. That is why this gate replaced Cirac-Zoller.
  * Beyond Lamb-Dicke the Debye-Waller factor costs $5.6\times10^{-5}$ at $n = 0$ and $4.3\times10^{-3}$ at $n = 8$, so the insensitivity is leading-order only.
  * Failing to close the loop is not a calibration error: a 2% timing error leaves residual spin-motion entanglement and costs $3\times10^{-3}$ of fidelity.

**6\. Anomalous heating is the bottleneck, and it is a surface**

  * $d\bar{n}/dt = e^2S_E/4m\hbar\omega$; the reference room-temperature surface trap gives 1455 quanta/s for Ca$^+$ at 1 MHz.
  * $S_E \propto d^{-4}$ (uncorrelated patches), $\propto 1/f$ (broad activation-energy distribution), suppressed $10^2$ by cooling to 10 K, suppressed $10^2$ by argon-ion milling — and recovering over days.
  * Error budget: a $10^{-4}$ target needs $S_E$ a factor of 1455 below the reference. Room-temperature surface traps cannot reach it; cryogenic ones at 3 MHz just can; shrinking $d$ from 50 to 30 $\mu$m loses a factor of eight.
  * The signature is the same as the two-level defects of Chapter 2, in a system with nothing else in common.

**Practical implications**

  * Quote $q$, $\Omega$ and $\beta$, not just the secular frequency; the stability margin is part of the specification.
  * Verify a gate design by closing the phase-space loop numerically, not by checking the pulse area.
  * Treat $\bar{n}$ and $\dot{\bar{n}}\tau$ as the two motional figures of merit; the second is the one that scales badly.
  * When comparing traps, ask for $d$, the electrode temperature, and the surface preparation before asking for the fidelity.

### Where This Leads

Chapter 4 keeps the atom and throws away the charge. Neutral atoms held in optical tweezers can be arranged in arbitrary two- and three-dimensional geometries and rearranged between shots, which removes the linear-chain constraint of Section 3.5 entirely. The interaction is not a shared phonon but the Rydberg blockade — a dipole-dipole coupling so strong that one excited atom forbids its neighbours from being excited at all — and the $n^{11}$ scaling of Rydberg properties will look extreme until it is computed.

The materials story changes character too. There is no electrode near a neutral atom and therefore no anomalous heating; the limits come from atom loss, from the finite lifetime of the Rydberg state, and from the optical system. Whether that counts as escaping the surface problem or merely relocating it is a question worth holding onto through Chapter 4 and settling in Chapter 5.

[← Chapter 2: Superconducting Qubits](<chapter-2.html>) [Chapter 4: Neutral Atoms →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The trap parameters, linewidths, noise spectral densities and heating rates quoted in this chapter are representative literature-scale values used for educational estimates. Verify against primary sources before using them in a proposal or a paper.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
