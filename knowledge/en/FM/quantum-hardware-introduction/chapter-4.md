---
title: "Chapter 4: Neutral Atoms"
chapter_title: "Chapter 4: Neutral Atoms"
subtitle: ⚛️ Tweezers, Rydberg Blockade, and a Register You Assemble Atom by Atom
reading_time: 40-45 minutes
difficulty: Advanced
code_examples: 6
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/quantum-hardware-introduction/chapter-4.html>) | Last sync: 2026-08-13

[Fundamental Mathematics Dojo](<../index.html>) > [Introduction to Quantum Hardware](<index.html>) > Chapter 4

The two platforms of Chapters 2 and 3 sit at opposite ends of a spectrum. A superconducting qubit is a fabricated object: fast, tightly integrated, and limited by defects in the materials it is made of. A trapped ion is a natural object: slow, identical to every other ion in the universe, and limited by what happens at the surface of the electrodes that hold it. This chapter is about a platform that takes the second half of that trade and pushes it further. A neutral atom in an optical tweezer is held by nothing but light. There is no electrode nearby, no junction, no oxide, and — this is the striking part — no fabrication step anywhere in the qubit itself.

That sounds like it should solve the materials problem, and in one sense it does. The consequence is instructive rather than triumphant: with the materials problem removed, the limits that remain are *atomic physics*, and atomic physics does not negotiate. The lifetime of a Rydberg state, the polarizability of a ground-state atom, the recoil of a single photon — these set an error floor that no amount of process engineering can lower. Understanding where that floor is, and what it is made of, is the point of this chapter.

We will also meet a genuinely different way of using quantum hardware. Everything in the [algorithms companion](<../quantum-computing-introduction/index.html>) to this course assumed the digital model: a register, a sequence of discrete gates, a measurement. A neutral-atom array can be run that way, but it can also be run in *analog* mode, in which the interaction Hamiltonian of the atoms is itself the model you want to study. That is not a simulation of an Ising model. It is an Ising model, made of atoms, whose parameters you set with a laser.

**Units and conventions.** As in Chapters 1 through 3, Hamiltonians are written with $\hbar = 1$, so a Hamiltonian has the dimensions of angular frequency and a coupling quoted as "$V/h = 10$ MHz" means $V = 2\pi\hbar \times 10^7\ \mathrm{s^{-1}}$. The customary unit in cold-atom work is MHz for laser couplings and interactions (compare GHz for superconducting circuits in Chapter 2, and MHz for ions in Chapter 3), µK for temperature, and µm for distance. The conversion used throughout is $h \times 1\ \text{MHz} = k_B \times 48.0\ \mu\mathrm{K} = 4.14\ \mathrm{neV}$. The definitions of $T_1$, $T_2$ and $T_2^\ast$ fixed in Chapter 1 are used unchanged. Qubit ordering is big-endian as in the whole series: for a two-atom state $|q_0 q_1\rangle$, atom 0 is the leftmost symbol and the most significant bit of the basis index.

## Learning Objectives

After completing this chapter, you will be able to:

  * Derive the radiation-pressure force on a two-level atom, compute the Doppler cooling limit and the recoil limit, and explain why a laser-cooled beam still needs a slower a metre long
  * Compute the depth, trap frequencies and photon-scattering rate of a Gaussian-beam optical dipole trap from the atomic polarizability, and state why a tweezer holds an atom in a room-temperature vacuum chamber
  * State the scaling of the van der Waals coefficient, lifetime and polarizability of a Rydberg state with principal quantum number, and use them to explain why the useful range of $n$ is bounded from both sides
  * Compute the blockade radius from $C_6$ and the Rabi frequency, and predict how weakly it depends on both
  * Solve the two-atom Rydberg problem by unitary evolution, measure the $\sqrt{2}$ collective Rabi frequency, and show that blockade leakage falls as $(\Omega/V)^2$
  * Construct the blockade CZ gate, verify that it is locally equivalent to CZ, and derive the intrinsic error floor $\sim(\gamma/V)^{2/3}$ that Rydberg decay and finite blockade impose together
  * Distinguish digital and analog operation of an atom array, and show by exact diagonalization that a Rydberg chain realizes a long-range transverse-field Ising model with crystalline ordered phases
  * Explain quantitatively why atom loss, destructive readout and the reload duty cycle are architectural constraints rather than engineering details

* * *

## 4.1 Making a Force Out of Light

### Two forces, and only one of them traps

A laser beam exerts two distinct forces on an atom, and the distinction between them organizes this whole chapter.

The first is the **scattering force**, or radiation pressure. The atom absorbs a photon from the beam, receives its momentum $\hbar k$ along the beam direction, and re-emits into a random direction. Averaged over many cycles the emission contributes nothing, so the net force is $\hbar k$ times the scattering rate:

$$ F_{\text{scatt}} = \hbar k \, \Gamma_{\text{sc}}, \qquad \Gamma_{\text{sc}} = \frac{\Gamma}{2}\,\frac{s_0}{1 + s_0 + \left(2\delta/\Gamma\right)^2} $$

Here $\Gamma$ is the natural linewidth of the transition, $s_0 = I/I_{\text{sat}}$ is the saturation parameter, and $\delta = \omega_L - \omega_0 - \mathbf{k}\cdot\mathbf{v}$ is the detuning as seen by a moving atom. This force is dissipative — it carries entropy away with the spontaneously emitted photons — and it saturates: at $s_0 \to \infty$ the atom spends half its time in the excited state and $\Gamma_{\text{sc}} \to \Gamma/2$.

The second is the **dipole force**, treated in §4.2. It is conservative, derives from a potential, does not saturate, and is what actually holds an atom in a tweezer. The two forces have different scalings with detuning — the scattering force falls as $1/\delta^2$ and the dipole force as $1/\delta$ — which is the entire reason a far-detuned trap can hold an atom for minutes while scattering almost no photons.

### Doppler cooling and the two limits

Point two counter-propagating red-detuned beams at an atom. An atom moving toward one beam sees it Doppler-shifted closer to resonance, absorbs more from it, and is pushed back. Expanding the two-beam force to first order in velocity gives a viscous damping,

$$ F \simeq -\alpha v, \qquad \alpha = -\frac{8\hbar k^2 s_0 (\delta/\Gamma)}{\left[1 + s_0 + (2\delta/\Gamma)^2\right]^2} $$

which is positive for $\delta < 0$. That is **optical molasses**, and in three dimensions it is what a magneto-optical trap (MOT) is built from, with a magnetic field gradient added so that the restoring force is also positional rather than only velocity-dependent.

Cooling cannot continue indefinitely, because the same photons that damp the motion also kick it. Balancing the damping against the momentum diffusion of random spontaneous emission gives the **Doppler limit**

$$ k_B T_D = \frac{\hbar\Gamma}{2} $$

and there is a lower floor still, the **recoil limit** $k_B T_{\text{rec}} = \hbar^2 k^2 / m$, set by the momentum of a single photon. Both are properties of the atom and the transition — not of the apparatus — which is the first appearance in this chapter of a limit you cannot engineer away. (Sub-Doppler mechanisms such as polarization-gradient cooling do beat $T_D$; they do not beat $T_{\text{rec}}$.)

The thermodynamics of this is worth a moment. Laser cooling is not refrigeration in the sense of the [classical statistical mechanics](<../classical-statistical-mechanics/index.html>) course — there is no cold reservoir. It is a scattering process that removes entropy by sending it away in a directed-in, isotropic-out photon flux. That is why it works in a room-temperature vacuum chamber, and it is also why the "temperature" of a cloud of $10^4$ atoms is a statement about a velocity distribution and nothing else.

### Code Example 1: Radiation Pressure and the Doppler Limit

```python
"""Radiation pressure, the Doppler limit, and the length of a slower.

Constants in SI; results printed in the units a cold-atom laboratory uses.
"""
import numpy as np

# --- fundamental constants (SI) ------------------------------------------
hbar = 1.054571817e-34        # J s
kB = 1.380649e-23             # J / K
u = 1.66053906660e-27         # kg
g_earth = 9.80665             # m / s^2

# --- Rb-87 D2 line -------------------------------------------------------
m = 86.909180527 * u          # kg
lam = 780.241209e-9           # m
Gamma = 2 * np.pi * 6.0666e6  # s^-1, natural linewidth
k = 2 * np.pi / lam           # m^-1

v_rec = hbar * k / m                    # single-photon recoil velocity
a_max = hbar * k * Gamma / (2 * m)      # deceleration at full saturation
T_dopp = hbar * Gamma / (2 * kB)        # Doppler limit
T_rec = hbar ** 2 * k ** 2 / (m * kB)   # recoil temperature

print("Rb-87 D2 line")
print(f"  wavelength                   {lam * 1e9:.2f} nm")
print(f"  linewidth Gamma/2pi          {Gamma / (2 * np.pi) / 1e6:.3f} MHz")
print(f"  excited-state lifetime 1/G   {1 / Gamma * 1e9:.1f} ns")
print(f"  recoil velocity hbar k / m   {v_rec * 1e3:.3f} mm/s")
print(f"  max deceleration hbar k G/2m {a_max:.3e} m/s^2 = {a_max / g_earth:.3e} g")
print(f"  Doppler limit hbar G / 2 kB  {T_dopp * 1e6:.1f} uK")
print(f"  recoil temperature           {T_rec * 1e6:.3f} uK")


def scattering_rate(s0, detuning, v):
    """Photon scattering rate from one beam; detuning and v in SI units."""
    delta = detuning - k * v      # Doppler-shifted detuning seen by the atom
    return 0.5 * Gamma * s0 / (1.0 + s0 + (2.0 * delta / Gamma) ** 2)


# --- how long must a slower be? ------------------------------------------
T_source = 300.0
v_th = np.sqrt(3 * kB * T_source / m)
print()
print(f"Thermal beam at {T_source:.0f} K: v_rms = {v_th:.1f} m/s, "
      f"{v_th / v_rec:.3e} photons to stop")
print(f"{'eta = a/a_max':>14}{'a (m/s^2)':>12}{'stop time (ms)':>16}{'stop length (m)':>17}")
for eta in [0.3, 0.5, 0.7, 1.0]:
    a = eta * a_max
    print(f"{eta:>14.1f}{a:>12.3e}{v_th / a * 1e3:>16.2f}{v_th ** 2 / (2 * a):>17.3f}")

# --- optical molasses: damping coefficient and capture velocity ----------
print()
print("Optical molasses (two counter-propagating beams per axis, s0 = 0.2):")
s0 = 0.2
print(f"{'detuning/Gamma':>15}{'alpha (kg/s)':>14}{'m/alpha (us)':>14}{'v_cap (m/s)':>13}")
for d in [-0.25, -0.5, -1.0, -2.0]:
    D = 1.0 + s0 + (2.0 * d) ** 2
    alpha = -8.0 * hbar * k ** 2 * s0 * d / D ** 2   # F = -alpha v to first order
    print(f"{d:>15.2f}{alpha:>14.3e}{m / alpha * 1e6:>14.2f}{abs(d) * Gamma / k:>13.2f}")

# --- numerical check of the small-v expansion ----------------------------
d, v = -0.5, 0.01
F_exact = hbar * k * (scattering_rate(s0, d * Gamma, v) - scattering_rate(s0, d * Gamma, -v))
D = 1.0 + s0 + (2.0 * d) ** 2
alpha = -8.0 * hbar * k ** 2 * s0 * d / D ** 2
print()
print(f"check at v = {v} m/s, detuning = {d} Gamma:")
print(f"  exact two-beam force  {F_exact:+.4e} N")
print(f"  -alpha v              {-alpha * v:+.4e} N")
print(f"  ratio                 {F_exact / (-alpha * v):.5f}")
print()
print("Capture velocity: metres per second. Thermal velocity: hundreds of them.")
print("That gap, not the final temperature, is what a Zeeman slower or 2D-MOT buys.")
```

```text
Rb-87 D2 line
  wavelength                   780.24 nm
  linewidth Gamma/2pi          6.067 MHz
  excited-state lifetime 1/G   26.2 ns
  recoil velocity hbar k / m   5.885 mm/s
  max deceleration hbar k G/2m 1.122e+05 m/s^2 = 1.144e+04 g
  Doppler limit hbar G / 2 kB  145.6 uK
  recoil temperature           0.362 uK

Thermal beam at 300 K: v_rms = 293.4 m/s, 4.986e+04 photons to stop
 eta = a/a_max   a (m/s^2)  stop time (ms)  stop length (m)
           0.3   3.365e+04            8.72            1.280
           0.5   5.608e+04            5.23            0.768
           0.7   7.851e+04            3.74            0.548
           1.0   1.122e+05            2.62            0.384

Optical molasses (two counter-propagating beams per axis, s0 = 0.2):
 detuning/Gamma  alpha (kg/s)  m/alpha (us)  v_cap (m/s)
          -0.25     1.301e-21        110.92         1.18
          -0.50     1.130e-21        127.67         2.37
          -1.00     4.047e-22        356.63         4.73
          -2.00     7.397e-23       1950.94         9.47

check at v = 0.01 m/s, detuning = -0.5 Gamma:
  exact two-beam force  -1.1304e-23 N
  -alpha v              -1.1304e-23 N
  ratio                 1.00000

Capture velocity: metres per second. Thermal velocity: hundreds of them.
That gap, not the final temperature, is what a Zeeman slower or 2D-MOT buys.
```

**What to look for.** Three numbers in that output are worth committing to memory.

**The deceleration is enormous and the stopping distance is still large.** A rubidium atom at full saturation decelerates at $1.1\times 10^5\ \mathrm{m/s^2}$ — eleven thousand times gravity — which sounds like it should stop anything instantly. But a thermal atom leaves a 300 K source at 293 m/s, and $v^2/2a$ is then 38 cm even at the theoretical maximum. At the $\eta \approx 0.5$ design margin a real slower uses, it is 77 cm. That is why the front end of a cold-atom apparatus is a metre of vacuum tube wrapped in magnet coils: the number is not an engineering compromise, it is $v_{\text{th}}^2 m / (\hbar k \Gamma)$.

**Fifty thousand photons per atom.** Stopping one atom takes $v_{\text{th}}/v_{\text{rec}} \approx 5\times10^4$ absorption-emission cycles, each of which must return the atom to the same ground state or the cycle breaks. This is why laser cooling works on alkalis and alkaline earths with closed cycling transitions and is hard on almost everything else — including, notably, most molecules.

**The capture velocity is metres per second.** The molasses damping is strong, with $m/\alpha$ of order 100 µs, but it only acts on atoms whose Doppler shift is comparable to the detuning, i.e. $v \lesssim |\delta|/k \approx$ a few m/s. The thermal distribution is hundreds. The entire purpose of a Zeeman slower or a two-dimensional MOT is to close that factor of a hundred; the final temperature is almost an afterthought by comparison.

Notice what has *not* appeared: any property of any solid. The two constants that set every number above, $\Gamma$ and $k$, belong to the rubidium atom.

* * *

## 4.2 The Optical Tweezer

### The dipole force

An atom in an oscillating electric field acquires an induced dipole moment $\mathbf{p} = \alpha \mathbf{E}$, and the interaction energy of that dipole with the field it induced is, after averaging over the optical cycle,

$$ U_{\text{dip}}(\mathbf{r}) = -\frac{1}{2\epsilon_0 c}\,\mathrm{Re}\,\alpha(\omega_L)\, I(\mathbf{r}) $$

For a laser tuned below all relevant resonances ("red detuned"), $\mathrm{Re}\,\alpha > 0$, the energy is lowest where the intensity is highest, and the atom is pulled into the focus. That is the whole mechanism: a focused laser beam is a potential well whose shape is the intensity profile of the beam.

For an alkali atom driven far from both D lines but not so far that the fine structure is irrelevant, the polarizability is dominated by those two transitions, and the standard result is

$$ U_{\text{dip}}(\mathbf{r}) = -\frac{\pi c^2}{2}\left[\frac{2\Gamma_{D2}}{\omega_{D2}^3}\left(\frac{1}{\omega_{D2} - \omega_L} + \frac{1}{\omega_{D2} + \omega_L}\right) + \frac{\Gamma_{D1}}{\omega_{D1}^3}\left(\frac{1}{\omega_{D1} - \omega_L} + \frac{1}{\omega_{D1} + \omega_L}\right)\right] I(\mathbf{r}) $$

The weights 2 and 1 are the line strengths of the $D_2$ and $D_1$ transitions, and the second term inside each bracket is the counter-rotating contribution, which matters at optical trapping wavelengths and is often dropped without comment. The corresponding photon scattering rate carries an extra factor of $\Gamma/\delta$, which is the crucial asymmetry:

$$ \frac{U_{\text{dip}}}{\hbar\Gamma_{\text{sc}}} \sim \frac{\delta}{\Gamma} $$

At a trap wavelength of 1064 nm and a rubidium D line near 780 nm, $\delta/\Gamma \sim 10^7$. That factor of ten million is what makes an optical tweezer a trap rather than a heater.

### The geometry of a focused beam

A Gaussian beam of power $P$ and waist $w_0$ has intensity

$$ I(r, z) = \frac{2P}{\pi w(z)^2}\exp\!\left(-\frac{2r^2}{w(z)^2}\right), \qquad w(z) = w_0\sqrt{1 + (z/z_R)^2}, \qquad z_R = \frac{\pi w_0^2}{\lambda} $$

Expanding about the focus gives a three-dimensional harmonic trap with

$$ \omega_r = \sqrt{\frac{4U_0}{m w_0^2}}, \qquad \omega_z = \sqrt{\frac{2U_0}{m z_R^2}}, \qquad \frac{\omega_r}{\omega_z} = \sqrt{2}\,\frac{z_R}{w_0} = \frac{\sqrt{2}\pi w_0}{\lambda} $$

The aspect ratio depends only on $w_0/\lambda$, so a diffraction-limited tweezer is *always* several times weaker along the beam than across it. That single geometric fact governs how atom arrays are laid out: two-dimensional arrays in the plane transverse to the beam are natural, and extending into the third dimension means either multiple beam directions or accepting a much softer axis.

### Loading, and the 50% problem

An atom is loaded into a tweezer from a cold cloud, and then the trap is deliberately made *lossy*: light-assisted collisions eject atoms in pairs, so the trap ends up containing either one atom or none, each with probability close to one half. This "collisional blockade" is what makes single-atom occupancy reliable, and it is also why a freshly loaded array of $N$ sites contains roughly $N/2$ atoms in random positions.

The fix is the technique that made the platform practical: image the array, work out which sites are filled, and then move atoms one at a time with a steerable tweezer until the target geometry is defect-free. Rearrangement is a real-time classical control problem — detect, plan a set of non-colliding moves, execute — and it costs time, which reappears in §4.5 as the duty cycle. What it buys is remarkable: the geometry of the register is *software*. A chain, a square lattice, a triangular lattice, a Kagome lattice, or an arbitrary graph are all the same apparatus with a different hologram.

### Code Example 2: Tweezer Depth, Frequencies and Heating

```python
"""The optical tweezer: Gaussian-beam dipole potential, depth, frequencies, heating.

Continues from Example 1 (same session).
"""
import numpy as np

c = 2.99792458e8

# --- Rb-87 D1 and D2 lines: (wavelength, linewidth, line-strength weight) --
lines = [(794.978851e-9, 2 * np.pi * 5.7500e6, 1.0),    # D1, weight 1
         (780.241209e-9, 2 * np.pi * 6.0666e6, 2.0)]    # D2, weight 2

lam_L = 1064e-9                      # trapping laser: far red of both D lines
omega_L = 2 * np.pi * c / lam_L
k_L = 2 * np.pi / lam_L
E_rec = hbar ** 2 * k_L ** 2 / (2 * m)          # recoil energy at the trap wavelength


def dipole_coefficients():
    """U/I (J per W/m^2) and Gamma_sc/I (s^-1 per W/m^2) summed over the D lines."""
    u_sum, g_sum = 0.0, 0.0
    for lam_i, Gam_i, w_i in lines:
        w0 = 2 * np.pi * c / lam_i
        # rotating + counter-rotating terms; red detuning makes the bracket positive
        br = 1.0 / (w0 - omega_L) + 1.0 / (w0 + omega_L)
        u_sum += w_i * Gam_i / w0 ** 3 * br
        g_sum += w_i * (Gam_i ** 2 / w0 ** 3) * (omega_L / w0) ** 3 * br ** 2
    return -0.5 * np.pi * c ** 2 * u_sum, 0.5 * np.pi * c ** 2 / hbar * g_sum


U_per_I, Gsc_per_I = dipole_coefficients()
alpha_au = -U_per_I * 2 * 8.8541878128e-12 * c / 1.64877727436e-41  # cross-check


def tweezer(P, w0):
    """Peak intensity, depth (K), Rayleigh range, trap frequencies (Hz), Gamma_sc."""
    I0 = 2.0 * P / (np.pi * w0 ** 2)
    U0 = U_per_I * I0                       # J, negative
    zR = np.pi * w0 ** 2 / lam_L
    nu_r = np.sqrt(4.0 * abs(U0) / (m * w0 ** 2)) / (2 * np.pi)
    nu_z = np.sqrt(2.0 * abs(U0) / (m * zR ** 2)) / (2 * np.pi)
    return I0, -U0 / kB, zR, nu_r, nu_z, Gsc_per_I * I0


print(f"trap laser {lam_L * 1e9:.0f} nm")
print(f"  U/I        = {U_per_I:.4e} J per (W/m^2)"
      f"  = {-U_per_I / kB * 1e7 * 1e6:.3f} uK per (kW/cm^2)")
print(f"  ground-state polarizability implied: {alpha_au:.1f} atomic units")
print(f"  Gamma_sc/I = {Gsc_per_I:.4e} s^-1 per (W/m^2)")
print(f"  (|U|/hbar) / Gamma_sc = {abs(U_per_I) / (hbar * Gsc_per_I):.3e}"
      f"   (trap depth in rad/s per scattered photon per s)")
print(f"  recoil energy at {lam_L * 1e9:.0f} nm: {E_rec / kB * 1e6:.4f} uK")
print()
hdr = (f"{'P (mW)':>7}{'w0 (um)':>9}{'I0 (kW/cm2)':>13}{'depth (mK)':>12}"
       f"{'zR (um)':>9}{'nu_r (kHz)':>11}{'nu_z (kHz)':>11}{'G_sc (1/s)':>11}")
print(hdr)
print("-" * len(hdr))
for P, w0 in [(2e-3, 1.0e-6), (5e-3, 1.0e-6), (5e-3, 1.5e-6), (20e-3, 2.0e-6)]:
    I0, depth, zR, nu_r, nu_z, gsc = tweezer(P, w0)
    print(f"{P * 1e3:>7.1f}{w0 * 1e6:>9.1f}{I0 * 1e-7:>13.3f}{depth * 1e3:>12.4f}"
          f"{zR * 1e6:>9.2f}{nu_r * 1e-3:>11.3f}{nu_z * 1e-3:>11.3f}{gsc:>11.3f}")

# --- the reference tweezer in detail --------------------------------------
P, w0 = 5e-3, 1.0e-6
I0, depth, zR, nu_r, nu_z, gsc = tweezer(P, w0)
U0 = depth * kB
dTdt = 2.0 * gsc * E_rec / (3.0 * kB)      # 3 kB T of energy in a 3D harmonic trap
print()
print(f"Reference tweezer: P = {P * 1e3:.0f} mW, w0 = {w0 * 1e6:.1f} um, "
      f"depth = {depth * 1e6:.1f} uK")
print(f"  radial harmonic length   {np.sqrt(hbar / (m * 2 * np.pi * nu_r)) * 1e9:.1f} nm"
      f"  = {np.sqrt(hbar / (m * 2 * np.pi * nu_r)) / w0:.4f} w0")
print(f"  bound levels             {U0 / (2 * np.pi * hbar * nu_r):.0f} radial, "
      f"{U0 / (2 * np.pi * hbar * nu_z):.0f} axial")
print(f"  recoil heating           {dTdt * 1e6:.3f} uK/s")
print(f"  time to heat by depth/10 {0.1 * depth / dTdt:.0f} s")
print(f"  atom at T = depth/10:    {0.1 * depth * 1e6:.1f} uK, "
      f"mean radial quanta {0.1 * depth * kB / (2 * np.pi * hbar * nu_r):.1f}")

print()
print("Aspect ratio: a diffraction-limited tweezer is always weaker along the beam.")
for w0 in [0.7e-6, 1.0e-6, 1.5e-6, 2.5e-6]:
    zR = np.pi * w0 ** 2 / lam_L
    _, _, _, nu_r, nu_z, _ = tweezer(5e-3, w0)
    print(f"  w0 = {w0 * 1e6:.1f} um -> zR/w0 = {zR / w0:5.2f}, "
          f"nu_r/nu_z = {nu_r / nu_z:5.2f} (predicted {np.sqrt(2) * zR / w0:5.2f})")
```

```text
trap laser 1064 nm
  U/I        = -2.1034e-36 J per (W/m^2)  = 1.524 uK per (kW/cm^2)
  ground-state polarizability implied: 677.3 atomic units
  Gamma_sc/I = 5.5019e-10 s^-1 per (W/m^2)
  (|U|/hbar) / Gamma_sc = 3.625e+07   (trap depth in rad/s per scattered photon per s)
  recoil energy at 1064 nm: 0.0973 uK

 P (mW)  w0 (um)  I0 (kW/cm2)  depth (mK)  zR (um) nu_r (kHz) nu_z (kHz) G_sc (1/s)
-----------------------------------------------------------------------------------
    2.0      1.0      127.324      0.1940     2.95     43.362     10.385      0.701
    5.0      1.0      318.310      0.4849     2.95     68.562     16.419      1.751
    5.0      1.5      141.471      0.2155     6.64     30.472      4.865      0.778
   20.0      2.0      318.310      0.4849    11.81     34.281      4.105      1.751

Reference tweezer: P = 5 mW, w0 = 1.0 um, depth = 484.9 uK
  radial harmonic length   41.2 nm  = 0.0412 w0
  bound levels             147 radial, 615 axial
  recoil heating           0.114 uK/s
  time to heat by depth/10 427 s
  atom at T = depth/10:    48.5 uK, mean radial quanta 14.7

Aspect ratio: a diffraction-limited tweezer is always weaker along the beam.
  w0 = 0.7 um -> zR/w0 =  2.07, nu_r/nu_z =  2.92 (predicted  2.92)
  w0 = 1.0 um -> zR/w0 =  2.95, nu_r/nu_z =  4.18 (predicted  4.18)
  w0 = 1.5 um -> zR/w0 =  4.43, nu_r/nu_z =  6.26 (predicted  6.26)
  w0 = 2.5 um -> zR/w0 =  7.38, nu_r/nu_z = 10.44 (predicted 10.44)
```

**What to look for.** The polarizability cross-check is the first thing to read: the two-D-line sum returns 677 atomic units against a literature value near 690, so a formula with two transitions in it reproduces the full sum over the rubidium spectrum to a few per cent. That is a useful calibration of how much atomic structure a trap calculation actually needs.

**A 5 mW beam makes a 485 µK trap.** Five milliwatts is a laser pointer, and the trap it makes is 485 µK deep — about 3300 times deeper than the recoil limit and about three times deeper than the Doppler limit, which is exactly the margin you want if the atom arrives from a MOT. The reason such a modest power suffices is the $1/w_0^2$ in the peak intensity: focusing to a micron converts 5 mW into 318 kW/cm².

**The trap is enormously anharmonic and it does not matter.** There are 147 bound radial levels, and the harmonic length is 4% of the waist. An atom in the lowest few levels is deep in the harmonic region; an atom at $T = U_0/10$ occupies about fifteen quanta and is still fine. Harmonic approximations for tweezers are safe not because the potential is harmonic but because the occupied part of it is.

**Photon scattering is 1.75 per second, and that is the trap's clock.** Each scattering event deposits about two recoil energies, giving 0.11 µK/s of heating, so the atom takes several hundred seconds to warm by a tenth of the trap depth. Compare that with the 26 ns excited-state lifetime that produced the cooling force in Example 1: the same atom-light interaction, evaluated seven orders of magnitude off resonance, has gone from the fastest process in the problem to the slowest. In practice background-gas collisions usually get there first, which is why these experiments live at ultra-high vacuum — a materials and surface-science problem, but of the chamber, not of the qubit.

* * *

## 4.3 Rydberg States and the Blockade

### Why one quantum number changes everything

A ground-state alkali atom is, for our purposes, inert: two atoms 5 µm apart interact through a van der Waals tail so weak that nothing happens on any experimental timescale. That is exactly what you want for storing quantum information and exactly what you cannot have for a two-qubit gate. Neutral atoms solve this by *switching the interaction on*: promote the atom to a state of high principal quantum number $n$, a **Rydberg state**, whose orbital radius grows as $n^2 a_0$ — so the atom becomes enormous, and enormously polarizable.

The scalings are all powers of $n$, and they are the design parameters of the platform:

| Quantity | Scaling | Consequence |
| --- | --- | --- |
| Orbital radius | $n^2$ | at $n = 70$ the atom is $\sim 0.3\ \mu$m across |
| Dipole matrix element between neighbouring Rydberg states | $n^2$ | strong coupling to microwaves and to other atoms |
| van der Waals coefficient $C_6$ | $n^{11}$ | the interaction is the fastest-growing quantity in the table |
| Radiative lifetime | $n^3$ | higher $n$ lives *longer*, which is unusual and helpful |
| dc polarizability | $n^7$ | stray electric fields shift the level, and badly |
| Blackbody-induced transfer rate | $n^{-2}$ | at 300 K this shortens the effective lifetime |

Two atoms in Rydberg states with no permanent dipole interact by the second-order van der Waals shift

$$ V(R) = \frac{C_6}{R^6} $$

and because $C_6 \propto n^{11}$, moving from $n = 40$ to $n = 80$ multiplies the interaction by $2^{11} = 2048$.

### The blockade condition

Now drive two atoms with a laser resonant on the $|g\rangle \to |r\rangle$ transition at Rabi frequency $\Omega$. If the pair is close enough that

$$ V(R) = \frac{C_6}{R^6} \gg \hbar\Omega $$

then the doubly excited state $|rr\rangle$ is shifted out of resonance and cannot be populated. One atom can be excited; two cannot. This is the **Rydberg blockade**, and the distance at which the two sides of the inequality balance defines the **blockade radius**

$$ R_b = \left(\frac{C_6}{\hbar\Omega}\right)^{1/6} $$

The sixth root is the important part of that formula. It means $R_b$ scales as $n^{11/6}$, not $n^{11}$, and as $\Omega^{-1/6}$, which is barely at all. The steepness of the van der Waals interaction that makes the blockade sharp in space also makes the blockade *radius* an insensitive knob: you cannot tune it much, and you do not have to.

### Code Example 3: Rydberg Scaling Laws

```python
"""Rydberg scaling laws: how one quantum number buys an interaction."""
import numpy as np

# Anchors for Rb nS Rydberg states. These are order-of-magnitude reference
# values; everything interesting below is a *ratio*, which the scalings fix.
n_ref = 70
C6_ref_GHz_um6 = 858.0        # C6/h for Rb |70S>, in GHz um^6
tau_ref_us = 150.0            # |70S> lifetime at room temperature, in us


def C6(n):
    """C6/h in GHz um^6:  C6 ~ n^11."""
    return C6_ref_GHz_um6 * (n / n_ref) ** 11


def lifetime(n):
    """Rydberg lifetime in us: radiative n^3, blackbody-limited n^2; use n^3."""
    return tau_ref_us * (n / n_ref) ** 3


def blockade_radius(n, Omega_MHz):
    """R_b such that C6/R_b^6 = hbar*Omega, i.e. (C6/h)/R_b^6 = Omega/2pi."""
    return (C6(n) * 1e3 / Omega_MHz) ** (1.0 / 6.0)     # GHz -> MHz


print("Rb nS scaling: C6 ~ n^11, tau ~ n^3, dipole moment ~ n^2, "
      "polarizability ~ n^7")
print(f"anchor: n = {n_ref}, C6/h = {C6_ref_GHz_um6:.0f} GHz um^6, "
      f"tau = {tau_ref_us:.0f} us")
print()
Om = 2.0     # MHz, a representative Rabi frequency on the ground-Rydberg transition
hdr = (f"{'n':>5}{'C6/h (GHz um^6)':>17}{'R_b (um)':>10}{'tau (us)':>10}"
       f"{'Rabi cycles':>13}{'dc-Stark rel.':>15}")
print(hdr)
print("-" * len(hdr))
for n in [40, 50, 60, 70, 80, 100, 120]:
    print(f"{n:>5}{C6(n):>17.3e}{blockade_radius(n, Om):>10.2f}{lifetime(n):>10.1f}"
          f"{Om * lifetime(n):>13.0f}{(n / n_ref) ** 7:>15.2f}")

print()
print(f"At Omega/2pi = {Om:.0f} MHz the blockade radius grows only as n^(11/6):")
for n in [50, 100]:
    print(f"  n = {n:>3}: R_b = {blockade_radius(n, Om):5.2f} um")
print(f"  ratio {blockade_radius(100, Om) / blockade_radius(50, Om):.3f}"
      f"  vs 2^(11/6) = {2 ** (11 / 6):.3f}")

print()
print("Blockade radius vs Rabi frequency: R_b ~ Omega^(-1/6), a very weak knob.")
print(f"{'Omega/2pi (MHz)':>17}{'R_b at n=70 (um)':>19}")
for Om_i in [0.2, 1.0, 2.0, 5.0, 20.0]:
    print(f"{Om_i:>17.1f}{blockade_radius(70, Om_i):>19.2f}")

print()
print("Van der Waals is steep: the interaction over one lattice spacing.")
n, a_um = 70, 5.0
print(f"  n = {n}, nearest-neighbour spacing a = {a_um:.1f} um")
for r in [1, 2, 3, 4]:
    R = r * a_um
    V_MHz = C6(n) * 1e3 / R ** 6
    print(f"    R = {R:5.1f} um ({r} a): V/h = {V_MHz:11.4f} MHz"
          f"   V/(hbar Om) = {V_MHz / Om:10.4f}")

print()
print("Two competing requirements pin the useful range of n from both sides:")
print(f"  small n: R_b < spacing, so no blockade "
      f"(n = 40 gives R_b = {blockade_radius(40, Om):.2f} um)")
print(f"  large n: dc-Stark sensitivity ~ n^7 and next-nearest neighbours also blockade")
print(f"  n = 40 -> 120 multiplies field sensitivity by "
      f"{(120 / 40) ** 7:.0f} at fixed field noise")
```

```text
Rb nS scaling: C6 ~ n^11, tau ~ n^3, dipole moment ~ n^2, polarizability ~ n^7
anchor: n = 70, C6/h = 858 GHz um^6, tau = 150 us

    n  C6/h (GHz um^6)  R_b (um)  tau (us)  Rabi cycles  dc-Stark rel.
----------------------------------------------------------------------
   40        1.820e+00      3.11      28.0           56           0.02
   50        2.119e+01      4.69      54.7          109           0.09
   60        1.574e+02      6.55      94.5          189           0.34
   70        8.580e+02      8.68     150.0          300           1.00
   80        3.727e+03     11.09     223.9          448           2.55
  100        4.339e+04     16.70     437.3          875          12.14
  120        3.224e+05     23.33     755.7         1511          43.51

At Omega/2pi = 2 MHz the blockade radius grows only as n^(11/6):
  n =  50: R_b =  4.69 um
  n = 100: R_b = 16.70 um
  ratio 3.564  vs 2^(11/6) = 3.564

Blockade radius vs Rabi frequency: R_b ~ Omega^(-1/6), a very weak knob.
  Omega/2pi (MHz)   R_b at n=70 (um)
              0.2              12.75
              1.0               9.75
              2.0               8.68
              5.0               7.45
             20.0               5.92

Van der Waals is steep: the interaction over one lattice spacing.
  n = 70, nearest-neighbour spacing a = 5.0 um
    R =   5.0 um (1 a): V/h =     54.9120 MHz   V/(hbar Om) =    27.4560
    R =  10.0 um (2 a): V/h =      0.8580 MHz   V/(hbar Om) =     0.4290
    R =  15.0 um (3 a): V/h =      0.0753 MHz   V/(hbar Om) =     0.0377
    R =  20.0 um (4 a): V/h =      0.0134 MHz   V/(hbar Om) =     0.0067

Two competing requirements pin the useful range of n from both sides:
  small n: R_b < spacing, so no blockade (n = 40 gives R_b = 3.11 um)
  large n: dc-Stark sensitivity ~ n^7 and next-nearest neighbours also blockade
  n = 40 -> 120 multiplies field sensitivity by 2187 at fixed field noise
```

**What to look for.** The table is a design space with a hard boundary on each side.

**From below: no interaction.** At $n = 40$, $C_6/h$ is under 2 GHz·µm⁶ and the blockade radius at $\Omega/2\pi = 2$ MHz is 3.1 µm. Tweezers cannot be packed much closer than a few µm — the diffraction limit and crosstalk between traps both push back — so at low $n$ the blockade radius falls below the achievable spacing and there is no gate.

**From above: everything else gets worse.** Going to $n = 120$ multiplies $C_6$ by $1.8\times10^5$ and gives a 23 µm blockade radius, which is *too large*: at a 5 µm lattice spacing, atoms four sites apart are still blockaded, and the register loses its addressability. Simultaneously the dc polarizability grows as $n^7$, so the same stray field that was negligible at $n = 40$ produces a 2187 times larger level shift at $n = 120$. Rydberg experiments are built inside carefully field-controlled enclosures for this reason, and the sensitivity is why they cannot simply climb to arbitrarily high $n$.

**The interaction is steep, which is the whole trick.** At $n = 70$ and a 5 µm spacing, the nearest-neighbour shift is 55 MHz and the next-nearest is 0.86 MHz — a factor of 64, because $2^6 = 64$. That contrast is what lets a global laser pulse blockade every nearest-neighbour pair while leaving second neighbours essentially free, which is precisely the structure that produces ordered phases in §4.4.

**The lifetime scaling is the one piece of good luck.** In almost every quantum platform, the state you use for interactions decays faster than the state you store in. Here the Rydberg lifetime grows as $n^3$, so the number of Rabi cycles available within the lifetime grows too: 56 at $n = 40$ against 1511 at $n = 120$. It is still a finite number, and §4.4 turns it into an error floor.

### Two atoms, exactly

The blockade is easy to state and easy to get slightly wrong, so it is worth solving exactly. Two atoms driven resonantly, with $|g\rangle = |0\rangle$ and $|r\rangle = |1\rangle$, have the Hamiltonian

$$ \frac{H}{\hbar} = \frac{\Omega}{2}\sum_{i=0,1} X_i - \Delta \sum_{i=0,1} n_i + V\, n_0 n_1, \qquad n_i = \frac{I - Z_i}{2} $$

on the four-dimensional space $\lbrace |gg\rangle, |gr\rangle, |rg\rangle, |rr\rangle \rbrace$. This is a $4\times4$ matrix, which means we can use exactly the machinery of the algorithms course: build the matrix, exponentiate it, read Born-rule probabilities. In the limit $V \to \infty$ the state $|rr\rangle$ is removed and the drive connects $|gg\rangle$ only to the symmetric singly-excited state

$$ |W\rangle = \frac{|gr\rangle + |rg\rangle}{\sqrt{2}} $$

with matrix element $\langle W| H |gg\rangle = \Omega/\sqrt{2}$. The pair therefore behaves as a two-level system with Rabi frequency $\sqrt{2}\,\Omega$ — measurably *faster* than a single atom driven by the same laser. That $\sqrt{2}$ is the cleanest experimental signature of the blockade there is, because it is a frequency rather than a population and does not require calibrating any efficiency.

### Code Example 4: Two-Atom Blockade Dynamics

```python
"""Two-atom Rydberg blockade: the sqrt(2) collective Rabi frequency.

Unitary evolution in the state-vector idiom of the algorithms course: build the
Hamiltonian as a matrix, exponentiate it, read Born-rule probabilities.
Units: hbar = 1, every frequency in units of Omega. The basis is big-endian,
|q0 q1> with |0> = |g> and |1> = |r>, so the index is 2*q0 + q1.
"""
import numpy as np
from scipy.linalg import expm

BASIS = ["gg", "gr", "rg", "rr"]
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
NR = np.array([[0, 0], [0, 1]], dtype=complex)      # |r><r|


def H_two_atom(Omega, Delta, V):
    """H/hbar = (Omega/2) sum_i X_i - Delta sum_i n_i + V n_0 n_1."""
    H = 0.5 * Omega * (np.kron(X, I2) + np.kron(I2, X))
    H -= Delta * (np.kron(NR, I2) + np.kron(I2, NR))
    H += V * np.kron(NR, NR)
    return H


def populations(psi0, H, times):
    """Populations of the four basis states on a whole time grid, one eigh call."""
    w, U = np.linalg.eigh(H)
    c = U.conj().T @ psi0
    psis = (np.exp(-1j * np.outer(times, w)) * c) @ U.T
    return np.abs(psis) ** 2


psi_gg = np.zeros(4, dtype=complex)
psi_gg[0] = 1.0
Omega = 1.0            # sets the unit of time

print("Resonant drive of both atoms from |gg>; hbar = 1, Omega = 1.")
print()
ts = np.array([0.0, np.pi / 4, np.pi / 2, np.pi / np.sqrt(2), np.pi, 2 * np.pi])
for label, V in [("V = 0 (independent atoms)", 0.0),
                 ("V = 1000 Omega (blockaded)", 1000.0)]:
    pops = populations(psi_gg, H_two_atom(Omega, 0.0, V), ts)
    print(label)
    print(f"  {'Omega t':>10}" + "".join(f"{'P(' + b + ')':>11}" for b in BASIS)
          + f"{'P(1 exc.)':>11}")
    for t, p in zip(ts, pops):
        print(f"  {t:>10.5f}" + "".join(f"{x:>11.6f}" for x in p)
              + f"{p[1] + p[2]:>11.6f}")
    print()

# --- exact Bohr frequencies from the spectrum -----------------------------
print("Eigenvalues of H (units of Omega) and the driven Bohr frequency:")
print(f"  {'V/Omega':>10}{'eigenvalues':>44}{'omega_Bohr':>12}{'expected':>11}")
for V in [0.0, 1.0, 5.0, 1000.0]:
    H = H_two_atom(Omega, 0.0, V)
    w, U = np.linalg.eigh(H)
    weight = np.abs(U.conj().T @ psi_gg) ** 2
    live = w[weight > 1e-9]
    gap = live.max() - live.min()
    exp = {0.0: "2 (= 2 Om)", 1000.0: "1.414214"}.get(V, "-")
    print(f"  {V:>10.1f}" + "".join(f"{x:>11.5f}" for x in w)
          + f"{gap:>12.6f}{exp:>11}")
print(f"  sqrt(2) = {np.sqrt(2):.6f}: the blockaded pair is a two-level system")
print("  spanned by |gg> and |W> = (|gr> + |rg>)/sqrt(2), coupled at Omega/sqrt(2),")
print("  hence a Rabi frequency of sqrt(2) Omega - measurably faster than one atom.")

# --- the collective pi-time, measured -------------------------------------
def first_max_time(V, Omega=1.0):
    """Time of the first maximum of the singly-excited population, refined."""
    ts = np.linspace(1e-3, 8.0, 400001)
    sig = populations(psi_gg, H_two_atom(Omega, 0.0, V), ts)[:, 1:3].sum(axis=1)
    i = np.argmax(sig[1:-1] >= np.maximum(sig[:-2], sig[2:])) + 1
    a, b, cc = sig[i - 1], sig[i], sig[i + 1]
    dt = ts[1] - ts[0]
    return ts[i] + 0.5 * dt * (a - cc) / (a - 2 * b + cc)


print()
print("Time of the first singly-excited maximum (units of 1/Omega):")
print(f"  {'V/Omega':>10}{'t_max':>12}{'pi/t_max':>12}")
for V in [0.0, 3.0, 10.0, 100.0, 1000.0]:
    t = first_max_time(V)
    print(f"  {V:>10.1f}{t:>12.6f}{np.pi / t:>12.6f}")
print(f"  blockaded prediction  t = pi/sqrt(2) = {np.pi / np.sqrt(2):.6f}")
print(f"  independent prediction t = pi/2      = {np.pi / 2:.6f}")

# --- how much probability leaks into |rr>? --------------------------------
print()
print("Blockade leakage: maximum P(rr) within one collective period.")
print(f"  {'V/Omega':>10}{'max P(rr)':>14}{'(Omega/V)^2/2':>16}{'ratio':>9}")
Vs = [2.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
pmaxes = []
for V in Vs:
    ts = np.linspace(0.0, 2 * np.pi / np.sqrt(2), 8001)
    pmax = populations(psi_gg, H_two_atom(Omega, 0.0, V), ts)[:, 3].max()
    pmaxes.append(pmax)
    print(f"  {V:>10.1f}{pmax:>14.3e}{0.5 * (Omega / V) ** 2:>16.3e}"
          f"{pmax / (0.5 * (Omega / V) ** 2):>9.3f}")
slope = np.polyfit(np.log(Vs[3:]), np.log(pmaxes[3:]), 1)[0]
print(f"  fitted exponent: max P(rr) ~ (V/Omega)^{slope:.4f}")
print()
print("Leakage falls as the square of the interaction, so 'more blockade' is cheap")
print("in principle - but P(rr) ~ (Omega/V)^2 and V ~ 1/R^6 make P(rr) ~ R^12, so")
print(f"a factor 100 in leakage costs only a factor {100 ** (1 / 12):.3f} in spacing")
print("- and that factor is where the trouble is.")
```

```text
Resonant drive of both atoms from |gg>; hbar = 1, Omega = 1.

V = 0 (independent atoms)
     Omega t      P(gg)      P(gr)      P(rg)      P(rr)  P(1 exc.)
     0.00000   1.000000   0.000000   0.000000   0.000000   0.000000
     0.78540   0.728553   0.125000   0.125000   0.021447   0.250000
     1.57080   0.250000   0.250000   0.250000   0.250000   0.500000
     2.22144   0.038868   0.158282   0.158282   0.644568   0.316564
     3.14159   0.000000   0.000000   0.000000   1.000000   0.000000
     6.28319   1.000000   0.000000   0.000000   0.000000   0.000000

V = 1000 Omega (blockaded)
     Omega t      P(gg)      P(gr)      P(rg)      P(rr)  P(1 exc.)
     0.00000   1.000000   0.000000   0.000000   0.000000   0.000000
     0.78540   0.722008   0.138996   0.138996   0.000000   0.277992
     1.57080   0.197150   0.401425   0.401425   0.000000   0.802849
     2.22144   0.000000   0.500000   0.500000   0.000001   0.999999
     3.14159   0.366872   0.316564   0.316564   0.000000   0.633128
     6.28319   0.070892   0.464554   0.464554   0.000000   0.929107

Eigenvalues of H (units of Omega) and the driven Bohr frequency:
     V/Omega                                 eigenvalues  omega_Bohr   expected
         0.0   -1.00000    0.00000    0.00000    1.00000    2.000000 2 (= 2 Om)
         1.0   -0.85464    0.00000    0.40303    1.45161    2.306244          -
         5.0   -0.75191    0.00000    0.65194    5.09996    5.851867          -
      1000.0   -0.70736    0.00000    0.70686 1000.00050    1.414213   1.414214
  sqrt(2) = 1.414214: the blockaded pair is a two-level system
  spanned by |gg> and |W> = (|gr> + |rg>)/sqrt(2), coupled at Omega/sqrt(2),
  hence a Rabi frequency of sqrt(2) Omega - measurably faster than one atom.

Time of the first singly-excited maximum (units of 1/Omega):
     V/Omega       t_max    pi/t_max
         0.0    1.570796    2.000000
         3.0    2.296593    1.367936
        10.0    2.218955    1.415798
       100.0    2.221439    1.414215
      1000.0    2.221441    1.414214
  blockaded prediction  t = pi/sqrt(2) = 2.221441
  independent prediction t = pi/2      = 1.570796

Blockade leakage: maximum P(rr) within one collective period.
     V/Omega     max P(rr)   (Omega/V)^2/2    ratio
         2.0     1.841e-01       1.250e-01    1.473
         5.0     2.554e-02       2.000e-02    1.277
        10.0     5.685e-03       5.000e-03    1.137
        30.0     5.817e-04       5.556e-04    1.047
       100.0     5.069e-05       5.000e-05    1.014
       300.0     5.582e-06       5.556e-06    1.005
      1000.0     5.007e-07       5.000e-07    1.001
  fitted exponent: max P(rr) ~ (V/Omega)^-2.0123

Leakage falls as the square of the interaction, so 'more blockade' is cheap
in principle - but P(rr) ~ (Omega/V)^2 and V ~ 1/R^6 make P(rr) ~ R^12, so
a factor 100 in leakage costs only a factor 1.468 in spacing
- and that factor is where the trouble is.
```

**What to look for.** The two limits are qualitatively different, not quantitatively different.

**Independent atoms reach $|rr\rangle$ with certainty; blockaded atoms reach $|W\rangle$ with certainty.** At $V = 0$ and $\Omega t = \pi$ the output is $|rr\rangle$ with probability 1.000000, because each atom independently completed a $\pi$ pulse. At $V = 1000\,\Omega$ and $\Omega t = \pi/\sqrt{2} = 2.221$ the output is the maximally entangled $|W\rangle$ state with probability 0.999999 and $P(rr) = 10^{-6}$. The same laser pulse, the same duration in units of $1/\Omega$, and the interaction has changed which state is produced. Note also that the entangled state arrives *earlier*: 2.221 rather than $\pi$.

**The $\sqrt{2}$ is exact and it is in the spectrum.** The eigenvalues at $V = 1000$ are $-0.70736$, $0$, $0.70686$ and $1000.0005$: the two states that carry any weight from $|gg\rangle$ are split by $1.414213$, against $\sqrt2 = 1.414214$. The first-maximum timing agrees to six figures. This is the same kind of check as validating a mapping against a closed form in the algorithms course — a number that comes out of a diagonalization and a number that comes out of an argument, agreeing.

**Leakage is quadratic, with a factor of one half.** The maximum $P(rr)$ within one collective period is $\tfrac12(\Omega/V)^2$ to within 5% for $V/\Omega \gtrsim 30$ (the ratio in the printed table is 1.047 at $V/\Omega = 30$ and 1.014 at 100, reaching a per cent only by $V/\Omega \approx 100$), and the fitted exponent is $-2.0123$. The quadratic law is the ordinary result of second-order perturbation theory in $\Omega/V$; the useful part is the prefactor, because it lets you convert a required gate error directly into a required $V/\Omega$ without simulating anything.

**And here is the difficulty hiding in the good news.** Since $P(rr) \propto (\Omega/V)^2$ and $V \propto R^{-6}$, the leakage goes as $R^{12}$: reducing it by a factor of 100 needs the spacing smaller by only $100^{1/12} = 1.468$ — equivalently, $V$ ten times larger, which the same $10^{1/6} = 1.468$ in spacing delivers. The exponent works in your favour. What does not work in your favour is that shrinking the spacing brings the tweezers themselves within a couple of microns of one another, and increasing $\Omega$ needs laser power that also drives off-resonant transitions. Section 4.4 puts numbers on the resulting compromise.

* * *

## 4.4 Two Ways to Compute With the Same Array

### Digital: the blockade gate

To turn the blockade into a gate we need a qubit that is *not* the Rydberg state, because a Rydberg state lives for microseconds and a qubit should live much longer. The standard choice is two hyperfine ground states, $|0\rangle$ and $|1\rangle$, with only $|1\rangle$ laser-coupled to $|r\rangle$. Each atom is now a three-level system and a pair spans nine dimensions.

The original protocol (Jaksch and co-workers, 2000) is three pulses:

  1. a $\pi$ pulse on the **control** atom, taking $|1\rangle \to |r\rangle$;
  2. a $2\pi$ pulse on the **target** atom, driving $|1\rangle \to |r\rangle \to |1\rangle$;
  3. a $\pi$ pulse on the control, returning $|r\rangle \to |1\rangle$.

Follow the four computational inputs through it. If the control is in $|0\rangle$, step 1 does nothing, and the target's complete $2\pi$ rotation returns it to $|1\rangle$ with the geometric phase $-1$ that any $2\pi$ rotation of a two-level system acquires. If the control is in $|1\rangle$, it is in $|r\rangle$ during step 2; if the target is *also* in $|1\rangle$, the blockade shifts $|rr\rangle$ out of resonance, the target's rotation is suppressed, and it picks up no phase — while the control's two $\pi$ pulses supply $(-i)^2 = -1$. Collecting the four cases gives

$$ U = \mathrm{diag}(1, -1, -1, -1) = (Z \otimes Z)\cdot \mathrm{CZ} $$

which is CZ up to single-qubit $Z$ gates, hence a perfectly good entangling gate. The mechanism is worth stating plainly: **the gate works by an interaction that is only switched on while the atoms are excited, and the phase it produces is not the interaction energy but the *absence* of a rotation.** That structural feature — the blockade suppresses rather than shifts — is why the gate error depends on $V$ only through the leakage, and why it can be made insensitive to the exact value of $V$.

Two physical mechanisms then set the error. Finite blockade lets $|rr\rangle$ be populated, with the $\tfrac12(\Omega/V)^2$ we measured. Finite Rydberg lifetime lets the atom decay while it is up there, with a probability proportional to the time spent in $|r\rangle$, hence to $\gamma/\Omega$. One error wants $\Omega$ small, the other wants $\Omega$ large, so there is an optimum, and at the optimum

$$ \varepsilon_{\min} \sim \left(\frac{\gamma}{V}\right)^{2/3} \sim \frac{R^4}{(\tau C_6)^{2/3}} $$

### Code Example 5: The Blockade CZ Gate and Its Error Floor

```python
"""The blockade CZ gate, and the error floor that physics puts under it.

Three levels per atom: |0> and |1> are the qubit (hyperfine ground states,
only |1> is laser-coupled), |r> is the Rydberg state. Index = 3*a0 + a1.

Continues from Example 3 (same session), which supplies C6(n) and lifetime(n).
"""
import numpy as np
from scipy.linalg import expm

LEV = ["0", "1", "r"]
QUBIT_IDX = [0, 1, 3, 4]           # |00>, |01>, |10>, |11> inside the 9-dim space
I3 = np.eye(3, dtype=complex)
SIG = np.zeros((3, 3), dtype=complex)
SIG[1, 2] = SIG[2, 1] = 1.0        # |1><r| + |r><1|, the laser coupling
NR = np.zeros((3, 3), dtype=complex)
NR[2, 2] = 1.0                     # |r><r|

VV = np.kron(NR, NR)               # Rydberg-Rydberg interaction operator
NR_TOT = np.kron(NR, I3) + np.kron(I3, NR)
DRIVE = [np.kron(SIG, I3), np.kron(I3, SIG)]


def gate_matrix(Omega, V, gamma):
    """4x4 block of the Jaksch pi - 2pi - pi sequence on the qubit subspace."""
    H0 = V * VV - 0.5j * gamma * NR_TOT       # non-Hermitian decay out of |r>
    U = np.eye(9, dtype=complex)
    for atom, area in [(0, np.pi), (1, 2 * np.pi), (0, np.pi)]:
        H = H0 + 0.5 * Omega * DRIVE[atom]
        U = expm(-1j * H * (area / Omega)) @ U
    return U[np.ix_(QUBIT_IDX, QUBIT_IDX)]


def avg_gate_error(M, G, d=4):
    """1 - average gate fidelity of a (possibly leaky) M against the target G."""
    F = (abs(np.trace(G.conj().T @ M)) ** 2 + np.trace(M.conj().T @ M).real) / (d * (d + 1))
    return 1.0 - F


G_ideal = np.diag([1.0, -1.0, -1.0, -1.0]).astype(complex)   # locally equal to CZ

print("Ideal blockade CZ: the Jaksch pi - 2pi - pi sequence, V -> infinity, no decay.")
M = gate_matrix(1.0, 1e7, 0.0)
print("  resulting 4x4 block (rounded):")
for row in np.round(M, 6):
    print("   " + "  ".join(f"{z.real:+.4f}{z.imag:+.4f}j" for z in row))
print(f"  error against diag(1,-1,-1,-1): {avg_gate_error(M, G_ideal):.3e}")
print("  diag(1,-1,-1,-1) = (Z x Z) . CZ, so this is CZ up to single-qubit Z gates.")

# --- the two error mechanisms, separately ---------------------------------
print()
print("Blockade leakage alone (gamma = 0), error vs V/Omega:")
print(f"  {'V/Omega':>10}{'gate error':>14}{'(Omega/V)^2':>14}")
for V in [10.0, 30.0, 100.0, 300.0, 1000.0]:
    print(f"  {V:>10.1f}{avg_gate_error(gate_matrix(1.0, V, 0.0), G_ideal):>14.3e}"
          f"{(1.0 / V) ** 2:>14.3e}")
print()
print("Rydberg decay alone (V -> infinity), error vs gamma/Omega:")
print(f"  {'gamma/Omega':>13}{'gate error':>14}{'gamma/Omega':>14}")
for g in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
    print(f"  {g:>13.1e}{avg_gate_error(gate_matrix(1.0, 1e7, g), G_ideal):>14.3e}{g:>14.3e}")
print("  Decay error is linear in gamma/Omega: the gate lasts 4pi/Omega and the")
print("  Rydberg level is occupied for a fixed fraction of it, so faster is better.")
print("  Leakage error is quadratic in Omega/V: slower is better. Hence an optimum.")


# --- the optimum, for real Rb numbers -------------------------------------
def total_error(Omega_ang, V_ang, gamma):
    return avg_gate_error(gate_matrix(Omega_ang, V_ang, gamma), G_ideal)


def optimize(n, R_um, n_scan=61):
    """Best Omega and the resulting error floor for Rb |nS> atoms R_um apart."""
    V_ang = 2 * np.pi * C6(n) * 1e9 / R_um ** 6            # rad/s
    gamma = 1.0 / (lifetime(n) * 1e-6)                     # s^-1
    grid = np.logspace(np.log10(1e-4 * V_ang), np.log10(0.3 * V_ang), n_scan)
    errs = np.array([total_error(1.0, V_ang / Om, gamma / Om) for Om in grid])
    i = int(np.argmin(errs))
    return V_ang, gamma, grid[i], errs[i]


print()
print("Rb |nS> atoms in a tweezer array: the intrinsic gate error floor.")
hdr = (f"{'n':>5}{'R (um)':>8}{'V/h (MHz)':>12}{'tau (us)':>10}"
       f"{'Om_opt/2pi (MHz)':>18}{'T_gate (us)':>13}{'error':>11}")
print(hdr)
print("-" * len(hdr))
rows = []
for n, R in [(70, 3.0), (70, 4.0), (70, 5.0), (70, 6.0), (70, 8.0),
             (50, 4.0), (100, 4.0), (100, 8.0)]:
    V_ang, gamma, Om, err = optimize(n, R)
    rows.append((n, R, err))
    print(f"{n:>5}{R:>8.1f}{V_ang / (2 * np.pi) / 1e6:>12.3f}{1e6 / gamma:>10.1f}"
          f"{Om / (2 * np.pi) / 1e6:>18.4f}{4 * np.pi / Om * 1e6:>13.4f}{err:>11.3e}")

# --- how the floor scales -------------------------------------------------
print()
print("Scaling check: error ~ (gamma/V)^(2/3) ~ R^4 / (tau C6)^(2/3).")
Rs = np.array([3.0, 4.0, 5.0, 6.0, 8.0])
errs = np.array([optimize(70, R)[3] for R in Rs])
print(f"  fitted exponent in R: {np.polyfit(np.log(Rs), np.log(errs), 1)[0]:.3f}"
      f"   (predicted 4)")
print(f"  error at R = 4 um, n = 70: {optimize(70, 4.0)[3]:.3e}")
print(f"  error at R = 4 um, n = 100: {optimize(100, 4.0)[3]:.3e}"
      f"   improvement {optimize(70, 4.0)[3] / optimize(100, 4.0)[3]:.1f}x")

# --- the error that no Hamiltonian contains: losing the atom --------------
print()
print("Array survival: a per-atom loss probability p_loss per cycle compounds.")
print(f"  {'p_loss':>9}" + "".join(f"{'N=' + str(N):>11}" for N in [10, 50, 100, 500, 1000]))
for p in [1e-4, 1e-3, 3e-3, 1e-2]:
    print(f"  {p:>9.0e}" + "".join(f"{(1 - p) ** N:>11.4f}"
                                   for N in [10, 50, 100, 500, 1000]))
print("  A defect-free 1000-atom array needs p_loss well below 1e-3 per atom per")
print("  cycle, or else most shots start with a hole in the register.")
print()
print("Destructive readout: fluorescence imaging of |1> heats the atom out of the")
print("trap, so each shot ends with reloading. The duty cycle is set by the MOT.")
for t_load_ms, t_gate_us in [(100.0, 100.0), (300.0, 100.0), (100.0, 1000.0)]:
    rate = 1.0 / (t_load_ms * 1e-3 + t_gate_us * 1e-6)
    print(f"  reload {t_load_ms:5.0f} ms + circuit {t_gate_us:6.0f} us"
          f" -> {rate:7.2f} shots/s, duty cycle {t_gate_us * 1e-6 * rate:.2e}")
```

```text
Ideal blockade CZ: the Jaksch pi - 2pi - pi sequence, V -> infinity, no decay.
  resulting 4x4 block (rounded):
   +1.0000+0.0000j  +0.0000+0.0000j  +0.0000+0.0000j  +0.0000+0.0000j
   +0.0000+0.0000j  -1.0000+0.0000j  +0.0000+0.0000j  +0.0000+0.0000j
   +0.0000+0.0000j  +0.0000+0.0000j  -1.0000+0.0000j  +0.0000+0.0000j
   +0.0000+0.0000j  +0.0000+0.0000j  +0.0000+0.0000j  -1.0000-0.0000j
  error against diag(1,-1,-1,-1): 7.139e-11
  diag(1,-1,-1,-1) = (Z x Z) . CZ, so this is CZ up to single-qubit Z gates.

Blockade leakage alone (gamma = 0), error vs V/Omega:
     V/Omega    gate error   (Omega/V)^2
        10.0     3.699e-03     1.000e-02
        30.0     4.112e-04     1.111e-03
       100.0     3.701e-05     1.000e-04
       300.0     4.112e-06     1.111e-05
      1000.0     3.701e-07     1.000e-06

Rydberg decay alone (V -> infinity), error vs gamma/Omega:
    gamma/Omega    gate error   gamma/Omega
        1.0e-02     5.300e-02     1.000e-02
        3.0e-03     1.631e-02     3.000e-03
        1.0e-03     5.477e-03     1.000e-03
        3.0e-04     1.648e-03     3.000e-04
        1.0e-04     5.496e-04     1.000e-04
  Decay error is linear in gamma/Omega: the gate lasts 4pi/Omega and the
  Rydberg level is occupied for a fixed fraction of it, so faster is better.
  Leakage error is quadratic in Omega/V: slower is better. Hence an optimum.

Rb |nS> atoms in a tweezer array: the intrinsic gate error floor.
    n  R (um)   V/h (MHz)  tau (us)  Om_opt/2pi (MHz)  T_gate (us)      error
-----------------------------------------------------------------------------
   70     3.0    1176.955     150.0           21.4234       0.0934  3.973e-04
   70     4.0     209.473     150.0            5.6900       0.3515  1.348e-03
   70     5.0      54.912     150.0            2.9068       0.6880  3.092e-03
   70     6.0      18.390     150.0            1.6601       1.2048  6.718e-03
   70     8.0       3.273     150.0            0.3376       5.9235  2.235e-02
   50     4.0       5.173      54.7            1.0399       1.9233  2.968e-02
  100     4.0   10593.730     437.3           66.3073       0.0302  4.898e-05
  100     8.0     165.527     437.3            3.9346       0.5083  7.255e-04

Scaling check: error ~ (gamma/V)^(2/3) ~ R^4 / (tau C6)^(2/3).
  fitted exponent in R: 4.086   (predicted 4)
  error at R = 4 um, n = 70: 1.348e-03
  error at R = 4 um, n = 100: 4.898e-05   improvement 27.5x

Array survival: a per-atom loss probability p_loss per cycle compounds.
     p_loss       N=10       N=50      N=100      N=500     N=1000
      1e-04     0.9990     0.9950     0.9900     0.9512     0.9048
      1e-03     0.9900     0.9512     0.9048     0.6064     0.3677
      3e-03     0.9704     0.8605     0.7405     0.2226     0.0496
      1e-02     0.9044     0.6050     0.3660     0.0066     0.0000
  A defect-free 1000-atom array needs p_loss well below 1e-3 per atom per
  cycle, or else most shots start with a hole in the register.

Destructive readout: fluorescence imaging of |1> heats the atom out of the
trap, so each shot ends with reloading. The duty cycle is set by the MOT.
  reload   100 ms + circuit    100 us ->    9.99 shots/s, duty cycle 9.99e-04
  reload   300 ms + circuit    100 us ->    3.33 shots/s, duty cycle 3.33e-04
  reload   100 ms + circuit   1000 us ->    9.90 shots/s, duty cycle 9.90e-03
```

**What to look for.** This is the central quantitative result of the chapter.

**The gate is exactly CZ in the ideal limit.** The $4\times4$ block is $\mathrm{diag}(1,-1,-1,-1)$ to $7\times10^{-11}$, which validates the whole nine-dimensional construction against the argument given above. Nothing about the sequence was fitted.

**Two error channels with opposite slopes.** Blockade leakage is $\propto (\Omega/V)^2$: 3.7×10⁻³ at $V/\Omega = 10$, falling by 100 for each factor of 10 in $V$. Rydberg decay is $\propto \gamma/\Omega$, with the fitted coefficient close to 5.5. One is quadratic and decreasing in $V/\Omega$, the other linear and increasing — so the total has a minimum, and it is a genuine physical optimum rather than a modelling artefact.

**The floor is around $10^{-3}$, and it moves as $R^4$.** For rubidium $|70S\rangle$ atoms 4 µm apart the optimum sits at $\Omega/2\pi = 5.7$ MHz, giving a 0.35 µs gate and an error of $1.3\times10^{-3}$. The fitted exponent in the spacing is 4.086 against the predicted 4. Move the atoms to 8 µm and the error is $2.2\times10^{-2}$; move to 3 µm and it is $4.0\times10^{-4}$. **A factor of two in interatomic spacing costs a factor of sixteen in gate error.** No other platform in this course has a geometric sensitivity that steep, and it is the reason tweezer positioning stability is a first-order concern rather than a detail.

**Higher $n$ helps, but read the fine print.** Going from $|70S\rangle$ to $|100S\rangle$ at fixed 4 µm spacing improves the floor 27-fold, to $4.9\times10^{-5}$. That is real: $C_6$ grew faster than $\gamma$ shrank. But at $n = 100$ the blockade radius is 16.7 µm, so at a 4 µm spacing the *fourth* neighbour is still blockaded, and the "two-atom gate" is no longer a two-atom operation. The honest statement is that $n$ trades gate fidelity against addressability, and the optimum depends on the array geometry you want.

**Then there is the error that no Hamiltonian contains.** The survival table is the arithmetic of a defect-free array: at $10^{-3}$ loss per atom per cycle, a 100-atom array survives 90% of the time and a 1000-atom array 37%. Atom loss is not a small correction to a gate error — it is a *leakage out of the computational space entirely*, and it must be handled by detection and reload rather than by a better pulse. Together with the reload duty cycle in the last block, this is the structural cost of the platform: circuits are fast (microseconds) and shots are slow (tens per second at best), so the sampling budget analysed in the [algorithms course](<../quantum-computing-introduction/index.html>) is limited by the MOT, not by the gates.

### Analog: when the Hamiltonian is the point

Now step back from gates entirely. Take $N$ atoms in a lattice, illuminate them all with the same laser, and write down what the array is doing:

$$ \frac{H}{\hbar} = \frac{\Omega}{2}\sum_i X_i - \Delta \sum_i n_i + \sum_{i<j}\frac{C_6}{R_{ij}^6}\, n_i n_j $$

Substituting $n_i = (I - Z_i)/2$ — the same convention as §4.3, with $|r\rangle = |1\rangle$ — turns the last two terms into $ZZ$ couplings and longitudinal fields, while the first is a transverse field. Writing the result as $H/\hbar = h_x\sum_i X_i + \sum_{i<j} J_{ij} Z_iZ_j - \sum_i h_z Z_i$ plus a constant, this is a **long-range transverse-field Ising model**, with

$$ J_{ij} = \frac{V_{ij}}{4}, \qquad h_z = \frac{1}{4}\sum_{j \neq i} V_{ij} - \frac{\Delta}{2}, \qquad h_x = \frac{\Omega}{2} $$

The $1/4$ is worth checking rather than trusting: each pair term $V_{ij}n_in_j$ contributes $-(V_{ij}/4)Z_i$ *and* $-(V_{ij}/4)Z_j$, so site $i$ collects one quarter of $V_{ij}$ from each of its neighbours. For the nearest-neighbour ring of Code Example 6, where each atom has exactly two neighbours at the same distance, that is $h_z = V/2 - \Delta/2$, which is the form the code prints — a coincidence of the geometry, not the general rule.

and the parameters are set by knobs on the optical table: $\Omega$ and $\Delta$ by the laser, $J_{ij}$ by where you put the atoms. Nothing has been Trotterized, no gate has been compiled, and no circuit depth accumulates. The atoms are not simulating an Ising model; within the stated approximations, they *are* one.

This is a genuinely different resource from the digital model, and it is worth being precise about the trade. What analog operation gains: no gate compilation, no Trotter error, no depth budget, and a system size limited by how many atoms you can arrange rather than by how many gates you can afford. What it loses: universality, error correction, and — most importantly — the ability to certify the answer. A digital circuit can in principle be error-corrected and its output verified; an analog simulator gives you the ground state of *its own* Hamiltonian, including whatever calibration errors and stray fields are in it.

### Code Example 6: A Rydberg Ring Is an Ising Model

```python
"""Analog mode: a Rydberg chain *is* an Ising model, and its ground state knows it.

H/hbar = (Omega/2) sum_i X_i - Delta sum_i n_i + sum_{i<j} C6/R_ij^6 n_i n_j

Ground states of a periodic 12-atom chain by sparse exact diagonalization, all
energies in units of Omega. Big-endian bit order: atom 0 is the most significant
bit of the basis index.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

N = 12                      # atoms; 12 accommodates periods 2, 3 and 4 exactly
DIM = 2 ** N
IDX = np.arange(DIM)
BITS = ((IDX[:, None] >> np.arange(N - 1, -1, -1)) & 1).astype(float)
DIST = np.array([[min(abs(i - j), N - abs(i - j)) for j in range(N)] for i in range(N)])


def build_H(Delta, Rb_over_a, kmax=N // 2):
    """Sparse H in units of Omega; Rb is set by C6/Rb^6 = hbar Omega."""
    diag = -Delta * BITS.sum(axis=1)
    for i in range(N):
        for j in range(i + 1, N):
            if 0 < DIST[i, j] <= kmax:
                diag += (Rb_over_a / DIST[i, j]) ** 6 * BITS[:, i] * BITS[:, j]
    H = sp.diags(diag, format="csr")
    for i in range(N):
        flip = IDX ^ (1 << (N - 1 - i))
        H = H + sp.csr_matrix((np.full(DIM, 0.5), (IDX, flip)), shape=(DIM, DIM))
    return H


def ground_state(Delta, Rb_over_a, kmax=N // 2):
    w, v = eigsh(build_H(Delta, Rb_over_a, kmax), k=1, which="SA",
                 v0=np.ones(DIM) / np.sqrt(DIM), tol=0.0)
    return float(w[0]), v[:, 0]


def observables(psi):
    """Density, nearest-neighbour correlation, and the crystalline order S(k).

    S(k) = (1/N) <|sum_i exp(i k i) n_i|^2>. A perfect period-p crystal on N = 12
    sites gives S(2pi/2) = 3.0, S(2pi/3) = 4/3, S(2pi/4) = 0.75.
    """
    p = np.abs(psi) ** 2
    dens = p @ BITS
    nn = float(np.mean([p @ (BITS[:, i] * BITS[:, (i + 1) % N]) for i in range(N)]))
    C = np.array([[p @ (BITS[:, i] * BITS[:, j]) for j in range(N)] for i in range(N)])
    Sk = {}
    for name, k in [("Z2", np.pi), ("Z3", 2 * np.pi / 3), ("Z4", np.pi / 2)]:
        ph = np.exp(1j * k * np.arange(N))
        Sk[name] = float((np.conj(ph)[:, None] * C * ph[None, :]).sum().real / N)
    return float(dens.mean()), nn, Sk, dens


print(f"Rydberg ring, N = {N} atoms, periodic, energies in units of Omega.")
print("Rb is the blockade radius: C6/Rb^6 = hbar Omega, so V(a) = (Rb/a)^6 Omega.")
print()
print("Detuning scan at Rb/a = 1.5:")
hdr = (f"{'Delta/Omega':>13}{'E0/Omega':>12}{'<n>':>9}{'<n_i n_i+1>':>13}"
       f"{'S(Z2)':>9}")
print(hdr)
print("-" * len(hdr))
for D in [-2.0, 0.0, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0]:
    E0, psi = ground_state(D, 1.5)
    dens, nn, Sk, _ = observables(psi)
    print(f"{D:>13.1f}{E0:>12.5f}{dens:>9.4f}{nn:>13.3e}{Sk['Z2']:>9.4f}")
print(f"  perfect Z2 crystal would give <n> = 0.5 and S(Z2) = {(N / 2) ** 2 / N:.1f}")

print()
print(f"At Rb/a = 1.5 the nearest-neighbour shift is {1.5 ** 6:.2f} Omega and the")
print(f"next-nearest is {(1.5 / 2) ** 6:.4f} Omega: blockaded at one spacing and")
print("nearly free at two. That asymmetry is what selects period 2.")

print()
print("Blockade-range scan at Delta/Omega = 6: which period does the ring choose?")
hdr = (f"{'Rb/a':>7}{'V(a)/Om':>11}{'V(2a)/Om':>11}{'<n>':>8}"
       f"{'S(Z2)':>8}{'S(Z3)':>8}{'S(Z4)':>8}{'ordering':>12}")
print(hdr)
print("-" * len(hdr))
for Rb in [0.5, 1.0, 1.3, 1.5, 1.8, 2.3, 2.8, 3.5]:
    E0, psi = ground_state(6.0, Rb)
    dens, nn, Sk, _ = observables(psi)
    best = max(Sk, key=Sk.get)
    ideal = {"Z2": 3.0, "Z3": 4 / 3, "Z4": 0.75}[best]
    label = best if Sk[best] > 0.5 * ideal else "uniform"
    print(f"{Rb:>7.1f}{Rb ** 6:>11.3f}{(Rb / 2) ** 6:>11.4f}{dens:>8.4f}"
          f"{Sk['Z2']:>8.3f}{Sk['Z3']:>8.3f}{Sk['Z4']:>8.3f}{label:>12}")
print("  ideal values: S(Z2) = 3.000, S(Z3) = 1.333, S(Z4) = 0.750")

for Rb, name in [(1.5, "Z2"), (2.3, "Z3"), (3.5, "Z4")]:
    E0, psi = ground_state(6.0, Rb)
    dens, nn, Sk, prof = observables(psi)
    p = np.abs(psi) ** 2
    cfg = format(int(np.argmax(p)), f"0{N}b").replace("0", ".").replace("1", "r")
    print()
    print(f"Delta/Omega = 6, Rb/a = {Rb} -> {name}:  <n> = {dens:.4f}, "
          f"S({name}) = {Sk[name]:.3f}")
    print("  site densities " + " ".join(f"{x:.3f}" for x in prof))
    print(f"  most probable configuration {cfg}  (probability {p.max():.4f}); the "
          f"ground state is\n  a symmetric superposition of all {name} translations, "
          "so every site looks alike.")

# --- the same physics, written as an Ising model ---------------------------
print()
print("The substitution n_i = (1 + Z_i)/2 turns the blockaded ring into Ising:")
print("  V sum_i n_i n_i+1 - Delta sum_i n_i")
print("   = (V/4) sum_i Z_i Z_i+1 + (V/2 - Delta/2) sum_i Z_i + const,")
print("  with (Omega/2) sum_i X_i the transverse field: J = V/4, h_z = V/2 - Delta/2,")
print("  h_x = Omega/2. Nothing was approximated except truncating the tail of V.")
print("  The atoms *are* the model; they are not a circuit that imitates it.")

print()
print("How much of the tail matters? Delta/Omega = 6, Rb/a = 1.5, range k kept:")
print(f"{'range k':>9}{'E0/Omega':>13}{'<n>':>9}{'S(Z2)':>9}{'|dE0| vs full':>15}")
E_full, _ = ground_state(6.0, 1.5)
for k in [1, 2, 3, N // 2]:
    E0, psi = ground_state(6.0, 1.5, k)
    dens, nn, Sk, _ = observables(psi)
    print(f"{k:>9}{E0:>13.6f}{dens:>9.4f}{Sk['Z2']:>9.4f}{abs(E0 - E_full):>15.2e}")
print()
print("A nearest-neighbour truncation misses the energy by a visible amount, and the")
print("miss is one-sided. An analog Rydberg machine does not implement the textbook")
print("nearest-neighbour Ising model - it implements the 1/R^6 one. That is a feature")
print("if long-range Ising is what you want, and a systematic error if it is not.")
```

```text
Rydberg ring, N = 12 atoms, periodic, energies in units of Omega.
Rb is the blockade radius: C6/Rb^6 = hbar Omega, so V(a) = (Rb/a)^6 Omega.

Detuning scan at Rb/a = 1.5:
  Delta/Omega    E0/Omega      <n>  <n_i n_i+1>    S(Z2)
--------------------------------------------------------
         -2.0    -1.32015   0.0425    1.653e-04   0.0439
          0.0    -3.65004   0.1858    9.678e-04   0.2460
          1.0    -6.76151   0.3495    1.185e-03   1.2479
          2.0   -11.84326   0.4623    7.056e-04   2.5837
          4.0   -23.40264   0.4913    7.437e-04   2.8963
          6.0   -35.26971   0.4966    9.064e-04   2.9521
         10.0   -59.18794   0.4994    1.537e-03   2.9766
         20.0  -119.51342   0.5144    2.941e-02   2.8354
  perfect Z2 crystal would give <n> = 0.5 and S(Z2) = 3.0

At Rb/a = 1.5 the nearest-neighbour shift is 11.39 Omega and the
next-nearest is 0.1780 Omega: blockaded at one spacing and
nearly free at two. That asymmetry is what selects period 2.

Blockade-range scan at Delta/Omega = 6: which period does the ring choose?
   Rb/a    V(a)/Om   V(2a)/Om     <n>   S(Z2)   S(Z3)   S(Z4)    ordering
-------------------------------------------------------------------------
    0.5      0.016     0.0002  0.9931   0.007   0.007   0.007     uniform
    1.0      1.000     0.0156  0.9850   0.015   0.015   0.015     uniform
    1.3      4.827     0.0754  0.5055   2.849   0.014   0.014          Z2
    1.5     11.391     0.1780  0.4966   2.952   0.004   0.004          Z2
    1.8     34.012     0.5314  0.4950   2.945   0.005   0.005          Z2
    2.3    148.036     2.3131  0.3347   0.041   1.229   0.012          Z3
    2.8    481.890     7.5295  0.3295   0.004   1.306   0.004          Z3
    3.5   1838.266    28.7229  0.2474   0.733   0.004   0.735          Z4
  ideal values: S(Z2) = 3.000, S(Z3) = 1.333, S(Z4) = 0.750

Delta/Omega = 6, Rb/a = 1.5 -> Z2:  <n> = 0.4966, S(Z2) = 2.952
  site densities 0.497 0.497 0.497 0.497 0.497 0.497 0.497 0.497 0.497 0.497 0.497 0.497
  most probable configuration .r.r.r.r.r.r  (probability 0.4745); the ground state is
  a symmetric superposition of all Z2 translations, so every site looks alike.

Delta/Omega = 6, Rb/a = 2.3 -> Z3:  <n> = 0.3347, S(Z3) = 1.229
  site densities 0.335 0.335 0.335 0.335 0.335 0.335 0.335 0.335 0.335 0.335 0.335 0.335
  most probable configuration ..r..r..r..r  (probability 0.2944); the ground state is
  a symmetric superposition of all Z3 translations, so every site looks alike.

Delta/Omega = 6, Rb/a = 3.5 -> Z4:  <n> = 0.2474, S(Z4) = 0.735
  site densities 0.247 0.247 0.247 0.247 0.247 0.247 0.247 0.247 0.247 0.247 0.247 0.247
  most probable configuration .r...r...r..  (probability 0.2406); the ground state is
  a symmetric superposition of all Z4 translations, so every site looks alike.

The substitution n_i = (1 + Z_i)/2 turns the blockaded ring into Ising:
  V sum_i n_i n_i+1 - Delta sum_i n_i
   = (V/4) sum_i Z_i Z_i+1 + (V/2 - Delta/2) sum_i Z_i + const,
  with (Omega/2) sum_i X_i the transverse field: J = V/4, h_z = V/2 - Delta/2,
  h_x = Omega/2. Nothing was approximated except truncating the tail of V.
  The atoms *are* the model; they are not a circuit that imitates it.

How much of the tail matters? Delta/Omega = 6, Rb/a = 1.5, range k kept:
  range k     E0/Omega      <n>    S(Z2)  |dE0| vs full
        1   -36.339291   0.4970   2.9571       1.07e+00
        2   -35.287036   0.4966   2.9522       1.73e-02
        3   -35.286862   0.4966   2.9522       1.72e-02
        6   -35.269706   0.4966   2.9521       0.00e+00

A nearest-neighbour truncation misses the energy by a visible amount, and the
miss is one-sided. An analog Rydberg machine does not implement the textbook
nearest-neighbour Ising model - it implements the 1/R^6 one. That is a feature
if long-range Ising is what you want, and a systematic error if it is not.
```

**What to look for.** This block does three things: it exhibits the ordered phases, it identifies the mechanism that selects them, and it quantifies what the $1/R^6$ tail costs.

**The detuning scan is a phase transition, seen at 12 sites.** At $\Delta/\Omega = -2$ the ground state is nearly empty ($\langle n\rangle = 0.04$); by $\Delta/\Omega = 4$ it is a period-2 crystal with $\langle n\rangle = 0.49$ and $S(Z_2) = 2.90$ against the perfect-crystal value 3.0. As Chapter 5 of the algorithms course argued for the Ising chain, a 12-site system cannot have a true transition — the free energy of a finite system is analytic — so what you are seeing is a crossover that would sharpen with $N$. It is nonetheless the physics, and the crossover is visible with 4096 basis states on a laptop.

**The blockade radius selects the period, and the mechanism is a ratio.** At $R_b/a = 1.5$ the nearest-neighbour shift is 11.4 $\Omega$ and the next-nearest is 0.18 $\Omega$ — blockaded at one spacing, free at two — and the ground state is a $\mathbb{Z}_2$ crystal. Push $R_b/a$ to 2.3 and the second neighbour is also blockaded at 2.3 $\Omega$, and the ground state becomes $\mathbb{Z}_3$: $\langle n\rangle = 0.33$, $S(Z_3) = 1.23$ against the ideal 1.333. At $R_b/a = 3.5$ it is $\mathbb{Z}_4$. **The lattice constant and the principal quantum number together choose the ordered phase.** Note the harmonic subtlety: the $\mathbb{Z}_4$ state also shows order at $k = \pi$, since period 4 is commensurate with period 2, so the discriminator between them is the density, not $S(k)$ alone.

**Every site looks identical, and that is the quantum part.** In each crystalline phase the site-resolved density is perfectly uniform (0.497 everywhere in the $\mathbb{Z}_2$ phase) while the most probable single measurement outcome is a perfect crystal with probability 0.47. The ground state of the ring is a symmetric superposition of all translations of the crystal, so no local observable sees the order and only the correlation function does. Any experiment that reports the density profile and stops has reported nothing; the order is in the two-point function.

**The tail is not a detail.** Truncating the interaction at nearest neighbours moves the ground-state energy by 1.07 out of 35.3 — three per cent, and one-sided. Keeping second neighbours reduces the error to 0.017. So an analog Rydberg machine is a very good simulator of the $1/R^6$ Ising model and a mediocre one of the textbook nearest-neighbour model. If your research question is about the long-range model, this is an advantage. If you wanted the short-range model, it is a systematic error you must either correct for or design around, and there is no dial that removes it.

* * *

## 4.5 What Scales, What Does Not, and Why

### The case for the platform, stated precisely

Three things about neutral atoms are structurally good, and they are worth separating from enthusiasm.

**There is no fabrication in the qubit.** Every rubidium-87 atom in the universe has the same $\Gamma$, the same hyperfine splitting, and the same $C_6$ at a given $n$. The qubit-to-qubit variability that dominates superconducting hardware (Chapter 2) simply does not exist here. Variability enters only through the *light*: tweezer depths differ, so light shifts differ, so single-qubit frequencies differ — but that is a calibration of a hologram, not a wafer.

**Control wiring scales better than in any other platform in this course.** A superconducting processor needs at least one coaxial line per qubit going into a dilution refrigerator. An atom array needs one global Rydberg laser, one trapping laser, and a spatial light modulator whose pixel count grows with the number of sites but whose *wiring* does not. Arrays of many hundreds of sites are a matter of laser power and field of view rather than of connector density.

**The geometry is programmable.** Because the register is defined by where the light is, the interaction graph is software. For analog quantum simulation this is not a convenience but the central capability: studying frustrated magnetism on a triangular lattice and on a Kagome lattice is the same apparatus and a different hologram.

### The limits, stated equally precisely

**The Rydberg error floor is atomic physics.** Example 5 gave $\varepsilon \sim (\gamma/V)^{2/3}$, of order $10^{-3}$ at practical spacings and $n$. There is no fabrication improvement that changes $\gamma$; it is a radiative lifetime. The available knobs are $n$ (bounded above by polarizability and blockade crosstalk), spacing (bounded below by optics), and $\Omega$ (bounded above by available power and off-resonant scattering). Improved pulse shapes and multi-photon schemes do better than the naive $\pi$-$2\pi$-$\pi$ sequence, but they operate inside the same $(\gamma/V)^{2/3}$ envelope.

**Atom loss is leakage, not error.** An atom that leaves its trap has not suffered a Pauli error; it has left the Hilbert space. This is a qualitatively harder failure mode than the depolarizing noise the algorithms course modelled, because it is not covered by a stabilizer code that assumes qubits stay put. It is handled by *detecting* the loss and reloading, which turns a coherence problem into a duty-cycle problem.

**Readout is destructive, and the duty cycle is set by the MOT.** State-selective detection works by pushing atoms in one state out of the trap and imaging fluorescence from the rest. The measurement therefore ends the experiment, and every shot begins with a fresh load — tens to hundreds of milliseconds. Example 5 puts the resulting shot rate at roughly 10 per second, with a duty cycle of $10^{-3}$: the atoms spend a tenth of a per cent of their existence computing. Recall the shot budgets from the algorithms course: a variational calculation needing $10^{10}$ measurements is out of reach here by many orders of magnitude, and the constraint is the magneto-optical trap, not the gate fidelity. Mid-circuit measurement and atom reservoirs attack this, and they do so at the level of the architecture rather than the qubit.

**Analog operation is not error-corrected, and cannot easily be.** Error correction requires discrete syndrome extraction, which requires gates. An analog simulator has no such structure, so its errors — miscalibrated $\Delta$, inhomogeneous $\Omega$, stray fields shifting the Rydberg level, atom position jitter feeding into $V \propto R^{-6}$ — appear directly in the answer. This does not make analog results worthless; it makes them results about a Hamiltonian you must characterize independently.

### The comparison, at the level of mechanism

| Axis | Neutral atoms | Contrast with Chapters 2 and 3 |
| --- | --- | --- |
| Qubit reproducibility | perfect (atoms are identical); light shifts are the variability | superconducting: every qubit differs; ions: also identical |
| Gate speed | sub-µs, set by $\Omega$ and the blockade | superconducting: tens of ns; ions: tens of µs |
| Intrinsic gate error floor | $\sim(\gamma/V)^{2/3}$, set by the Rydberg lifetime | superconducting: dielectric loss; ions: motional heating |
| Connectivity | reconfigurable graph, range set by $R_b$ | superconducting: fixed planar; ions: all-to-all in a chain |
| Operating temperature | room-temperature chamber, µK atoms | superconducting: 10 mK; ions: room-temperature electrodes work for small registers, but 4-10 K is required once low heating at small $d$ is needed (§3.6) |
| Control wiring | one global laser plus a hologram — best in this course | superconducting: one line per qubit |
| Dominant failure mode | atom loss and destructive readout | superconducting: $T_1$/$T_2$; ions: heating and optics |
| Materials bottleneck | **none in the qubit** — the knobs are in the apparatus (vacuum surfaces, optics), not in the qubit | superconducting: interface TLS; ions: electrode surfaces |

The last row is the one to think about. Removing the materials problem from the qubit is a genuine architectural advantage, and it comes with a genuine cost: when the limit is a radiative lifetime, there is no oxide to clean, no interface to improve, and no purification to attempt. The materials science of this platform lives entirely in the *apparatus* — vacuum surfaces, optical coatings, the electrode geometry that controls stray fields, the wavefront quality of the trapping optics — and Chapter 5 will argue that this makes neutral atoms the exception that proves the rule.

* * *

## Exercises

Work through these with the code from this chapter in front of you. Solutions follow each question.

#### Exercise 1: Slower Length and Photon Budget

A source of sodium atoms ($m = 23\ \mathrm{u}$, $\lambda = 589$ nm, $\Gamma/2\pi = 9.79$ MHz) operates at 600 K. (a) Compute $v_{\text{rms}}$, the recoil velocity, and the number of photons needed to stop one atom. (b) Compute the maximum deceleration and the stopping length at the $\eta = 0.5$ design margin. (c) Sodium is lighter than rubidium and its transition is stronger. Does that make the slower shorter or longer than rubidium's, and why is the answer not obvious from either fact alone?

<details><summary>Solution</summary>
<p><strong>(a)</strong> \(v_{\rm rms} = \sqrt{3k_BT/m} = \sqrt{3(1.381\times10^{-23})(600)/(23\times1.661\times10^{-27})} = 806.7\) m/s. The recoil velocity is \(v_{\rm rec} = h/(m\lambda) = 6.626\times10^{-34}/(3.82\times10^{-26}\times5.89\times10^{-7}) = 2.95\times10^{-2}\) m/s = 29.5 mm/s. The photon count is \(806.7/0.0295 = 2.74\times10^{4}\) — fewer than rubidium's \(5\times10^4\) despite the higher velocity, because each photon carries more velocity for a lighter atom.</p>
<p><strong>(b)</strong> \(a_{\max} = \hbar k \Gamma/(2m) = v_{\rm rec}\Gamma/2 = 0.0295 \times 2\pi(9.79\times10^6)/2 = 9.07\times10^{5}\) m/s\(^2\), about eight times rubidium's. The stopping length at \(\eta = 0.5\) is \(v^2/(2\eta a_{\max}) = 806.7^2/(9.07\times10^{5}) = 0.718\) m.</p>
<p><strong>(c)</strong> Almost the same length as rubidium's 0.77 m, which is a coincidence worth understanding. The stopping length is \(v_{\rm th}^2/(2\eta a_{\max}) = 3k_BT/(m \cdot \eta v_{\rm rec}\Gamma) = 3k_BT\lambda/(\eta h \Gamma)\): the mass cancels completely. A lighter atom is faster but also decelerates harder, and the two effects cancel exactly. What is left is \(T\lambda/\Gamma\), so the slower length is set by the source temperature, the wavelength and the linewidth — not by the mass. Sodium's shorter wavelength and larger linewidth roughly compensate its hotter source.</p>
</details>

#### Exercise 2: Designing a Tweezer

You need a tweezer at least 1 mK deep with a radial trap frequency above 100 kHz, using the 1064 nm coefficients from Code Example 2 ($U/I = -2.1034\times10^{-36}$ J per W/m²). (a) At $w_0 = 1.0$ µm, what power is needed for 1 mK? (b) What is the radial frequency at that power, and does it meet the specification? (c) You cannot afford that power. Show that reducing $w_0$ is more effective than raising $P$ for both requirements simultaneously, and quantify the scaling of each. (d) What sets the floor on $w_0$?

<details><summary>Solution</summary>
<p><strong>(a)</strong> Depth \(= |U/I| \cdot 2P/(\pi w_0^2)/k_B\). Code Example 2 gives 484.9 µK at 5 mW, and the depth is linear in \(P\), so 1 mK needs \(5 \times 1000/484.9 = 10.3\) mW.</p>
<p><strong>(b)</strong> \(\nu_r \propto \sqrt{U_0}\), so \(\nu_r = 68.56 \times \sqrt{1000/484.9} = 98.4\) kHz. That just misses 100 kHz — a useful reminder that depth and frequency are not independent specifications.</p>
<p><strong>(c)</strong> At fixed \(P\), \(U_0 \propto w_0^{-2}\) and \(\nu_r = \sqrt{4U_0/(m w_0^2)} \propto w_0^{-2}\). At fixed \(w_0\), \(U_0 \propto P\) and \(\nu_r \propto \sqrt{P}\). So halving the waist multiplies the depth by 4 and the frequency by 4, whereas quadrupling the power multiplies the depth by 4 and the frequency by only 2. The waist is the better knob for frequency by a factor of two in the exponent, which is why tweezer experiments spend their effort on high-numerical-aperture optics rather than on laser power.</p>
<p><strong>(d)</strong> Diffraction: \(w_0 \gtrsim \lambda/(\pi\,\mathrm{NA})\), so at 1064 nm and NA = 0.7 the waist cannot go far below 0.5 µm. Beyond that, a tighter waist shortens the Rayleigh range as \(w_0^2\), making the axial confinement relatively worse (Code Example 2's aspect-ratio table), and it also increases the intensity at fixed depth, hence the photon scattering rate. There is no free direction.</p>
</details>

#### Exercise 3: Choosing a Principal Quantum Number

Using the scalings of Code Example 3 with the $n = 70$ anchors ($C_6/h = 858$ GHz·µm⁶, $\tau = 150$ µs): (a) At a 4 µm lattice spacing and $\Omega/2\pi = 2$ MHz, what is the smallest $n$ for which nearest neighbours are blockaded ($R_b > a$)? (b) At that $n$, what is $V(2a)/\hbar\Omega$, and is the second neighbour blockaded too? (c) You want a $\mathbb{Z}_3$ ordered phase at $a = 4$ µm. Roughly what $n$ do you need? (d) State the two independent reasons not to simply use $n = 150$.

<details><summary>Solution</summary>
<p><strong>(a)</strong> \(R_b = [(C_6/h)/(\Omega/2\pi)]^{1/6}\) with \(C_6/h = 858(n/70)^{11}\) GHz·µm⁶. Setting \(R_b = 4\) µm at \(\Omega/2\pi = 2\) MHz requires \(C_6/h = 2\times10^{-3} \times 4^6 = 8.19\) GHz·µm⁶, hence \((n/70)^{11} = 8.19/858 = 9.54\times10^{-3}\), \(n/70 = (9.54\times10^{-3})^{1/11} = 0.6520\), \(n = 45.6\). So \(n = 46\) is the threshold and the table's \(n = 50\) row (\(R_b = 4.69\) µm) is the first comfortable choice.</p>
<p><strong>(b)</strong> At \(n = 50\), \(C_6/h = 21.19\) GHz·µm⁶ and \(V(8\,\mu\mathrm{m})/h = 21190/8^6 = 0.0808\) MHz, so \(V(2a)/\hbar\Omega = 0.040\). Firmly unblockaded: this is the \(\mathbb{Z}_2\) regime of Code Example 6, with \(R_b/a = 1.17\).</p>
<p><strong>(c)</strong> \(\mathbb{Z}_3\) needs \(R_b/a \approx 2.3\), i.e. \(R_b = 9.2\) µm, hence \(C_6/h = 2\times10^{-3}\times9.2^6 = 1214\) GHz·µm⁶ and \((n/70)^{11} = 1.415\), \(n = 70 \times 1.415^{1/11} = 72.2\). So \(n \approx 72\): only a 44% increase in \(n\) moves the system from \(\mathbb{Z}_2\) to \(\mathbb{Z}_3\), because \(n^{11}\) is brutal. This extreme sensitivity is also why \(n\) is a coarse knob and the lattice spacing is the fine one.</p>
<p><strong>(d)</strong> First, the dc polarizability grows as \(n^7\), so \(n = 150\) is \((150/70)^7 = 210\) times more sensitive to stray electric fields than \(n = 70\); at some point the field noise in the chamber shifts the Rydberg level by more than \(\Omega\) and the drive is no longer resonant. Second, \(R_b\) would be \((150/70)^{11/6} \times 8.68 = 35\) µm, blockading eight neighbours in each direction at a 4 µm spacing — the array stops being a set of addressable qubits and becomes one large blockaded blob. Neither reason has anything to do with fabrication quality.</p>
</details>

#### Exercise 4: Budgeting a Gate Error

Using the error floor $\varepsilon \simeq A(\gamma/V)^{2/3}$ calibrated on the $n = 70$, $R = 4$ µm point of Code Example 5 ($\varepsilon = 1.348\times10^{-3}$, $V/h = 209.5$ MHz, $\tau = 150$ µs): (a) Determine $A$. (b) What interatomic spacing would give $\varepsilon = 10^{-4}$ at $n = 70$? Is it achievable? (c) At what spacing would $\varepsilon$ reach $10^{-4}$ for $n = 100$? (d) You need $10^4$ two-qubit gates in a circuit. Which of the two failure mechanisms of §4.5 binds first, and at what array size?

<details><summary>Solution</summary>
<p><strong>(a)</strong> \(\gamma = 1/\tau = 6.667\times10^{3}\) s\(^{-1}\) and \(V = 2\pi \times 2.095\times10^{8} = 1.316\times10^{9}\) s\(^{-1}\), so \(\gamma/V = 5.066\times10^{-6}\) and \((\gamma/V)^{2/3} = 2.963\times10^{-4}\). Hence \(A = 1.348\times10^{-3}/2.963\times10^{-4} = 4.55\).</p>
<p><strong>(b)</strong> \(\varepsilon \propto R^4\), so \(R = 4\,\mu\mathrm{m} \times (10^{-4}/1.348\times10^{-3})^{1/4} = 4 \times 0.5217 = 2.09\) µm. Two atoms 2.1 µm apart at a 1 µm tweezer waist are barely two traps, and at \(n = 70\) the Rydberg orbit itself is about 0.3 µm across. This is the regime where the van der Waals expansion and the two-level treatment of the Rydberg pair both begin to fail. So: not achievable by shrinking the spacing.</p>
<p><strong>(c)</strong> Code Example 5 gives \(7.255\times10^{-4}\) at \(n = 100\), \(R = 8\) µm, so \(R = 8 \times (10^{-4}/7.255\times10^{-4})^{1/4} = 8 \times 0.6096 = 4.88\) µm. Comfortable optically — and this is the real reason to go to higher \(n\). The cost is the blockade radius of 16.7 µm from Exercise 3, so the array would need a scheme that tolerates long-range blockade (for instance, local addressing that only excites the pair being operated on).</p>
<p><strong>(d)</strong> At \(\varepsilon = 10^{-3}\), \(10^4\) gates give a total gate error of order 10 — the circuit is already meaningless from gate error alone, before loss is considered. Comparing the two mechanisms at equal severity: gate error per gate is \(10^{-3}\), and atom loss per cycle is of the same order, so for a circuit with \(G\) gates on \(N\) atoms the two contributions are \(G\varepsilon\) and \(Np_{\rm loss}\). They are comparable at \(G \approx N\); for the deep circuits fault tolerance requires (\(G \gg N\)) the gate error binds, while for a shallow analog-style protocol on a large array (\(N \gg G\)) loss binds. Both point to the same conclusion: this platform's near-term strength is wide-and-shallow, which is exactly the shape of an analog simulation.</p>
</details>

#### Exercise 5: Reading an Analog Simulation Honestly

A paper reports that a 200-atom Rydberg array was used to observe a quantum phase transition in a two-dimensional Ising model, with the measured order parameter agreeing with theory to 5%. (a) Which model, exactly, did the machine implement? (b) What is the classical baseline for 200 spins? (c) The paper reports site-resolved Rydberg densities and a structure factor. Which of those two is evidence of the ordered phase, and why? (d) Give three calibration errors that would produce a systematically wrong order parameter, and say which of them a nearest-neighbour Ising theory curve would not reveal. (e) State a defensible one-sentence summary of what such an experiment demonstrates.

<details><summary>Solution</summary>
<p><strong>(a)</strong> Not the nearest-neighbour Ising model: the machine implements \(\sum_{i&lt;j} (C_6/R_{ij}^6) n_i n_j\) with every pair coupled. Code Example 6 measured the difference — 3% in the ground-state energy in one dimension at \(R_b/a = 1.5\) — and in two dimensions the number of second and third neighbours is larger, so the tail matters more, not less.</p>
<p><strong>(b)</strong> A 200-spin Ising model is \(2^{200}\) states, so exact diagonalization is out. But that is not the baseline. Quantum Monte Carlo is essentially exact for this Hamiltonian in the absence of a sign problem, tensor networks handle two dimensions well at moderate entanglement, and mean-field plus finite-size scaling gets the transition location. The honest baseline is "what does the best classical method give with real effort", and for an unfrustrated Ising model that is a high bar.</p>
<p><strong>(c)</strong> Only the structure factor. Code Example 6 showed the site densities are perfectly uniform in every crystalline phase, because the ground state is a symmetric superposition of all translations of the crystal. A density profile is consistent with order but cannot demonstrate it; the two-point correlation function is the observable that distinguishes a crystal from a uniform paramagnet.</p>
<p><strong>(d)</strong> (i) An error in \(\Delta\), which moves the system along the phase boundary and directly shifts the apparent transition point. (ii) Inhomogeneous \(\Omega\) across the array from a non-uniform laser profile, which smears the effective \(h_x\) site to site. (iii) Atom position jitter, which enters \(V\) as \(R^{-6}\), so a 2% position error is a 12% coupling error — and, being random, it acts like disorder in \(J_{ij}\). A nearest-neighbour Ising theory curve would not reveal any of them, because the comparison is already against the wrong model: a fit that absorbs the \(1/R^6\) tail into an effective \(J\) will also absorb part of these errors.</p>
<p><strong>(e)</strong> "A 200-atom Rydberg array realized a long-range transverse-field Ising Hamiltonian and reproduced the expected ordered phase to 5%, demonstrating control of a programmable many-body Hamiltonian at a size where classical methods remain accurate." That is a real and publishable result. "Quantum advantage" is not the same sentence.</p>
</details>

* * *

## Summary

### Key Takeaways

**1\. Light provides two forces, and only one of them is a trap**

  * The scattering force $\hbar k \Gamma_{\text{sc}}$ is dissipative, saturates at $\hbar k \Gamma/2$, and cools; the dipole force is conservative, does not saturate, and traps.
  * They scale differently with detuning ($1/\delta^2$ versus $1/\delta$), and that ratio — $10^7$ for a 1064 nm trap on rubidium — is why a tweezer holds an atom for minutes while scattering under two photons a second.
  * The Doppler limit $\hbar\Gamma/2k_B = 146$ µK and the recoil limit 0.36 µK are properties of the atom, not the apparatus.
  * A slower's length is $3k_BT\lambda/(\eta h\Gamma)$ — independent of mass, and about a metre for any thermal alkali source.

**2\. A tweezer is a fully specified harmonic trap you can compute from a polarizability**

  * 5 mW at $w_0 = 1$ µm gives a 485 µK trap with 69 kHz radial and 16 kHz axial frequencies, 147 bound radial levels, and 1.75 photon scattering events per second.
  * The aspect ratio $\sqrt{2}\pi w_0/\lambda$ depends only on the optics, so tweezers are always weaker along the beam; two-dimensional arrays are the natural geometry.
  * Collisional blockade gives 50% loading, and rearrangement turns a random half-filled array into a defect-free one — which makes the register geometry software.

**3\. One quantum number buys the interaction, and the exponents set the design space**

  * $C_6 \propto n^{11}$, $\tau \propto n^3$, dc polarizability $\propto n^7$: the interaction grows fastest, the lifetime helpfully grows too, and the field sensitivity is the price.
  * $R_b = (C_6/\hbar\Omega)^{1/6}$ scales as $n^{11/6}$ and $\Omega^{-1/6}$ — an insensitive knob, which is why it need not be tuned precisely.
  * Low $n$ fails because $R_b <$ spacing; high $n$ fails because $R_b >$ several spacings and because $n^7$ field sensitivity takes over. The useful window is a decade wide.

**4\. The blockade is exactly solvable, and the exact answer has a $\sqrt2$ in it**

  * A blockaded pair is a two-level system spanned by $|gg\rangle$ and $|W\rangle$, coupled at $\Omega/\sqrt2$, hence oscillating at $\sqrt2\,\Omega$ — measured to six figures, and the cleanest signature of blockade there is.
  * Leakage into $|rr\rangle$ is $\tfrac12(\Omega/V)^2$, which converts a target gate error directly into a required $V/\Omega$.
  * The $\pi$-$2\pi$-$\pi$ sequence gives $\mathrm{diag}(1,-1,-1,-1) = (Z\otimes Z)\,\mathrm{CZ}$ exactly, and it works by *suppressing* a rotation rather than by accumulating an interaction phase.

**5\. The gate error floor is atomic physics, and it goes as $R^4$**

  * Blockade leakage $\propto(\Omega/V)^2$ and Rydberg decay $\propto\gamma/\Omega$ have opposite slopes, so the optimum error is $\sim(\gamma/V)^{2/3} \sim R^4/(\tau C_6)^{2/3}$.
  * For $|70S\rangle$ at 4 µm the floor is $1.3\times10^{-3}$ with a 0.35 µs gate; a factor of two in spacing costs a factor of sixteen in error.
  * Higher $n$ lowers the floor (27-fold from $n=70$ to $n=100$) at the cost of a blockade radius that spans several lattice sites.
  * No process improvement changes $\gamma$. This is the chapter's central point: removing the materials problem does not remove the limit, it relocates it into the atom.

**6\. The same array runs digitally or analogously, and the analog mode is a different resource**

  * $n_i = (I+Z_i)/2$ turns the driven array into a long-range transverse-field Ising model with $J_{ij} = V_{ij}/4$, $h_x = \Omega/2$ — implemented, not compiled.
  * A 12-atom ring shows $\mathbb{Z}_2$, $\mathbb{Z}_3$ and $\mathbb{Z}_4$ crystalline phases selected by $R_b/a$, with uniform site densities and order visible only in the correlation function.
  * The $1/R^6$ tail is 3% of the energy at $R_b/a = 1.5$ and is not removable: the machine simulates the long-range model, faithfully, whether or not that was the intention.
  * Analog operation has no Trotter error and no depth budget, and also no error correction and no certification.

**Practical implications**

  * Atom loss is leakage out of the computational space, not a Pauli error; a defect-free 1000-atom array needs loss well below $10^{-3}$ per atom per cycle.
  * Destructive readout plus MOT reload gives a shot rate of order 10 s⁻¹ and a duty cycle of $10^{-3}$: the measurement budgets of the algorithms course are the binding constraint here, not the gates.
  * When reading an analog result, ask which Hamiltonian was implemented, what the best classical method gives, and whether the reported observable can distinguish order from disorder at all.
  * Position stability is a first-order specification, because $V \propto R^{-6}$ turns a 2% position error into a 12% coupling error.

### Where This Leads

This chapter has covered the third of the three platforms that dominate current effort, and it has done so by taking the materials problem away and watching what remained. Chapter 5 completes the survey with three modalities that are structurally different from all three of these: photons, which have no idle decoherence at all and pay for it in nondeterminism; semiconductor spins, which are the only platform that can borrow the entire CMOS process stack, and which pay for that in charge noise from the oxide; and topological qubits, whose protection is a theorem rather than an engineering achievement, and whose distance from demonstration deserves to be stated precisely. Chapter 5 then puts all six side by side on the axes of Chapter 1 — and argues that the reason no modality has won is that each is stopped by a different materials problem.

[← Chapter 3: Trapped Ions](<chapter-3.html>) [Chapter 5: Photons, Spins, Topology — and the Scorecard →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * Atomic parameters, $C_6$ coefficients, lifetimes and trap numbers quoted here are illustrative order-of-magnitude values chosen for teaching; they are not device or apparatus specifications and must be verified against primary sources before use in any design or proposal.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
