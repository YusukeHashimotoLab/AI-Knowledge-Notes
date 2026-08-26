---
title: "Chapter 4: The Electrochemical Interface"
chapter_title: "Chapter 4: The Electrochemical Interface"
subtitle: "What Happens in the First Nanometre of Electrolyte, Why It Takes Three Electrodes to Watch It, and How to Read a Cyclic Voltammogram"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/VC95kMKrp8Y"
    title="Electrochemistry Ch.4: The Electrochemical Interface"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/MS/electrochemistry-introduction/chapter-4.html>) | Last sync: 2026-08-20

[Materials Science Dojo](<../index.html>) > [Electrochemistry Introduction](<index.html>) > Chapter 4

Chapters 2 and 3 were built on a phrase used casually and never examined: **the potential of the electrode**. Thermodynamics gave us \\(E_{\text{eq}}\\); kinetics gave us \\(\eta = E_{\text{applied}} - E_{\text{eq}}\\). Both assumed there is a well-defined electrical potential at the electrode, that we know its value, and that we can set it. Every one of those assumptions is more delicate than it sounds.

Where exactly is "the potential"? Not in the metal, which is an equipotential body, and not in the bulk solution, which is also nearly one. It lives in the transition between them, and that transition is astonishingly thin — a region of a few nanometres across which essentially the whole potential difference is dropped. That region is the **electric double layer**, and it is where all the physics of Chapter 3 actually takes place.

How do you measure a potential difference at a single interface? You cannot, not with two wires: any circuit has two interfaces in it, and a voltmeter reports their sum. Getting around this requires a third electrode and a specific trick, and understanding the trick is what separates people who can interpret electrochemical data from people who can only collect it.

And once you can control a single interface, what should you do with it? Overwhelmingly the first experiment anyone runs is **cyclic voltammetry**: sweep the potential up and down, record the current, and look at the shape. The resulting graph is the field's universal diagnostic. It is also routinely over-interpreted, and the last part of this chapter is about reading it honestly — including a NumPy simulation that produces the classic peaked shape from nothing but diffusion and the Nernst equation.

## 4.1 The First Nanometre: The Electric Double Layer

Take a metal electrode and charge it slightly negative. What does the electrolyte do?

The bulk solution is electrically neutral — every cation is statistically balanced by an anion. But near a charged surface, that neutrality breaks down. The negative electrode attracts cations and repels anions, so the layer of solution immediately adjacent to the metal acquires a net positive charge that mirrors the negative charge on the metal. Two sheets of opposite charge, separated by a very small distance. Hence **double layer**.

Two features of this arrangement matter enormously, and both follow from the geometry.

**The separation is molecular.** The counter-ions cannot press themselves flat against the metal. They arrive carrying a shell of solvent molecules, and the metal itself is covered in adsorbed solvent. The closest approach for the centres of those ions therefore sits roughly one solvation shell out from the surface — a plane conventionally called the **Helmholtz plane**. The distance involved is on the order of a fraction of a nanometre. That is not a metaphor for "small"; it is a real, physical, molecular length.

**Therefore the electric field is enormous.** Drop even one volt across a fraction of a nanometre and the field strength is on the order of \\(10^9\\) V/m. This is an extreme environment by any standard of ordinary chemistry, and it explains a great deal:

  * The electrode potential can shift reaction barriers strongly — the tunable activation energy that made the Butler–Volmer equation exponential in Chapter 3.
  * The interface stores charge very effectively, because capacitance grows as the plate separation shrinks. A double layer behaves like a capacitor with a plate separation of a molecular diameter, which is why electrochemical capacitors can store far more charge per unit area than conventional ones.
  * Solvent and ions in that region are not in their bulk state. Their orientation, their dielectric response and their reactivity all differ from the bulk, and this is one of the harder parts of the subject to model honestly.

**The double layer is a capacitor, and that has a consequence you cannot avoid.** Every time you change the electrode potential, you must move charge in or out of that capacitor. That charge movement is a current. It flows whether or not any chemistry is happening. Section 4.5 returns to this, because separating it from the current that does chemistry is one of the recurring practical problems of the field.

### 📚 How Far Does This Go? An Honest Limitation

The picture just given — two sheets of charge, one molecular diameter apart — is the simplest useful model, and it is the level this series stays at. Real double layers are more complicated in at least two ways worth naming.

  * **The counter-charge is not confined to a single plane.** Thermal motion pulls ions away from the surface, so the compensating charge is partly a compact layer and partly a diffuse cloud reaching further into the solution. Its extent depends strongly on ionic strength: concentrated electrolyte screens the electrode within a nanometre or so, dilute electrolyte leaves a field extending much further. This is one reason a **supporting electrolyte** at high concentration is added in almost every measurement — it compresses the diffuse region so the potential falls essentially entirely across the compact layer where the reacting species sits.
  * **Some ions shed their solvation shell and touch the metal directly**, adsorbing specifically rather than merely being attracted electrostatically. That can change the local field, block reactive sites, and shift kinetics in ways a purely electrostatic model cannot capture.

There is a long history of successively more detailed models of this region, and we skip it because the extra structure changes no conclusion in this chapter. Carry forward: **a molecular-scale capacitor, an enormous field, and a charging current that flows whenever the potential changes.**

## 4.2 Why Two Electrodes Are Not Enough

Here is the problem that makes electrochemistry an experimentally awkward science.

You want the potential of one interface — say, the platinum working electrode where the chemistry you care about is happening. So you connect a voltmeter. One lead goes to the platinum. The other lead has to go somewhere, and the only somewhere available is into the solution, through another piece of metal.

Now count the interfaces in your circuit. There are two: platinum-solution and second-metal-solution. The voltmeter reports the **sum** of the potential differences across both. It has no way to attribute the reading to either one, and no amount of careful wiring changes this, because a circuit must close.

It is worse than an accounting problem. Pass current to drive a reaction and that current flows through both interfaces in series, each developing its own overpotential by its own Butler–Volmer relation. The second interface is not a passive reference; **it moves while you are trying to measure**, by an amount that depends on the current you happen to be drawing. And the current must also cross the electrolyte, which has resistance, adding an \\(iR\\) drop — the ohmic term from Chapter 3 — to whatever the voltmeter reports.

So a two-electrode measurement gives you the interface you want, plus an interface you do not want that drifts with current, plus an ohmic term. Three unknowns, one number. This is not a precision problem; it is a structural impossibility.

**The three-electrode cell solves it by splitting the two jobs that the second electrode was doing badly.**

| Electrode | Job | Carries current? | Design priority |
|---|---|---|---|
| **Working (WE)** | The interface under study | Yes | It is the sample; everything else exists to serve it |
| **Counter (CE)** | Completes the circuit so current can flow | Yes — all of it | Large area, inert, and kept from contaminating the WE |
| **Reference (RE)** | Provides a fixed potential to measure against | **No — essentially zero** | Stable, reproducible, positioned close to the WE |

The insight is in the third row. The potentiostat measures the potential between working and reference with an input impedance so high that essentially **no current flows through the reference electrode**. Because no current flows, the reference passes no net reaction, develops no overpotential, and contributes no ohmic drop of its own. It sits at its equilibrium potential, unmoving, providing a fiducial mark.

Meanwhile all the current flows between working and counter. The counter electrode does develop a large and unknown overpotential — nobody cares, because nobody is measuring it. The potentiostat simply adjusts the working-counter voltage, continuously, by whatever amount holds the working-reference voltage at the value you asked for. That feedback loop is what a potentiostat *is*, and it is why you can specify a potential at a single interface, on an absolute scale, independently of how much current is flowing. Everything in Chapters 2 and 3 is measurable because of this arrangement.

## 4.3 Reference Electrodes in Practice

Chapter 2 defined the potential scale using the **standard hydrogen electrode (SHE)**: hydrogen gas at standard pressure bubbling over platinised platinum in acid at unit hydrogen-ion activity, declared to be \\(0\\) V by convention. It is a perfectly good definition and a thoroughly impractical instrument. It requires a supply of hydrogen gas, it needs a carefully maintained platinised surface that is easily poisoned, its activity condition is not something you can prepare by weighing something out, and nobody wants a hydrogen cylinder next to a routine experiment.

So the scale is defined by SHE and the measurements are made against something else. In practice:

| Reference | What it is | Rough position vs SHE | Typical use |
|---|---|---|---|
| **Ag/AgCl** | Silver wire coated in AgCl, in a chloride solution | Around \\(+0.2\\) V, depending on chloride concentration | The everyday workhorse; compact, robust, cheap |
| **SCE** (saturated calomel) | Mercury / mercury(I) chloride in saturated KCl | Also around \\(+0.2\\) V | Long-established, very stable; mercury disposal is a drawback |
| **RHE** (reversible hydrogen) | The same hydrogen reaction, but in *your* electrolyte at *its* pH | \\(0\\) V by construction, at any pH | Standard for water splitting and pH-varying work |

The values in the middle column deliberately say "around". A real Ag/AgCl reference's potential depends on the chloride concentration in its internal filling solution, and there are several common filling solutions in use. Reported values also shift slightly with temperature and with the liquid junction between the reference's filling solution and your electrolyte. Quoting a precise number without stating the filling solution would be false precision, and this series does not do it. What is reliable is the practical rule: **an Ag/AgCl or SCE reference sits a couple of hundred millivolts positive of SHE, and any potential you report must state which reference it was measured against.** A potential without a stated reference is not data.

### 📚 Why RHE Exists, and Why Water-Splitting Papers Use It

The reversible hydrogen electrode is the one worth understanding properly, because its purpose is subtle and it solves a real problem.

The hydrogen reaction \\(2\text{H}^+ + 2e^- \rightleftharpoons \text{H}_2\\) involves protons, so by the Nernst equation its equilibrium potential depends on pH. At 25 °C the shift is the familiar **59 mV per pH unit** — the same \\(2.303RT/F\\) that has now appeared in every chapter of this series — moving negative as pH rises.

The awkward consequence: water oxidation and water reduction *both* involve protons, so their equilibrium potentials **both** shift by that same 59 mV per pH unit. Measure a water-splitting catalyst against a fixed reference such as Ag/AgCl and changing the electrolyte pH moves the measured potential *and* the target you compare it against, by the same amount in the same direction. The overpotential — the quantity you actually care about — does not change at all, but the raw numbers wander and cross-pH comparison becomes bookkeeping.

RHE eliminates the bookkeeping by moving the zero of the scale along with the chemistry. Because RHE is the hydrogen reaction *in your own electrolyte*, it shifts by the same 59 mV per pH unit the reaction of interest does. So on an RHE scale the pH dependence cancels: hydrogen evolution sits at \\(0\\) V and water oxidation at \\(1.23\\) V, at every pH. **A potential quoted versus RHE is, to a good approximation, an overpotential with a constant offset** — exactly what you want when comparing catalysts measured in different electrolytes.

Two caveats. RHE only cancels the pH dependence for reactions with the same proton-to-electron ratio as hydrogen evolution; other stoichiometries still show pH dependence on an RHE scale. And "RHE" in a paper is sometimes a genuine hydrogen electrode in the working electrolyte and sometimes a conversion applied afterwards to an Ag/AgCl reading using a measured or assumed pH. Those are not equally trustworthy, and a careful methods section says which was done.

## 4.4 Reading a Cyclic Voltammogram

With a three-electrode cell you can command a potential. The most informative thing to do with that ability, and by a wide margin the most common, is to sweep it.

**Cyclic voltammetry** is exactly what the name says. Start at a potential where nothing happens, ramp linearly to a switching potential, ramp linearly back, and record current throughout. Plot current against potential — not against time, which is the first thing that confuses newcomers, because time is the hidden variable running around the loop. The result is a closed curve, and for a simple dissolved species reduced and then re-oxidised it has a shape distinctive enough that electrochemists call it a duck. Read it one feature at a time.

**The flat part at the start.** You are at a potential where the species is stable in its current oxidation state. No reaction. The only current is the small capacitive one from charging the double layer, discussed in Section 4.5.

**The rise.** As the potential approaches the formal potential of the couple, the Nernst equation begins to demand a different ratio of oxidised to reduced species at the surface. The surface obliges by reacting, and current flows. Because the demand is exponential in potential, the rise is steep.

**The peak — the feature that surprises people.** The current goes through a maximum and then *falls*, even though you are pushing harder and harder. The reason is that you are consuming the reactant: near the formal potential its surface concentration is driven toward zero, so the current is set by how fast fresh material arrives, and it arrives by diffusion across a depletion layer that thickens steadily as the experiment proceeds. A thicker layer means a shallower gradient, which means slower delivery. Past the peak, the potential no longer limits anything — **transport does** — and the current decays as the depletion layer spreads. This is the concentration overpotential of Chapter 3, visible as a shape.

**The reverse sweep, and the second peak.** Turn the potential around. The product you just made is still near the electrode, not yet diffused away, and as the potential comes back past the formal potential it is converted back, producing a peak of opposite sign. The reverse peak is therefore a direct report on **whether the product survived**: if it decomposed, reacted, or escaped, the peak is small or absent. Chemists exploit this constantly — cyclic voltammetry is one of the quickest ways to learn whether an oxidised or reduced form of a molecule is stable on the timescale of the experiment.

### 📚 The Three Numbers People Read Off a Voltammogram

| Feature | What it is | What it is used for | The honest caveat |
|---|---|---|---|
| **Peak positions** \\(E_{pc}\\), \\(E_{pa}\\) | Potentials of the cathodic and anodic maxima | Their midpoint approximates the formal potential of the couple | Only for a well-behaved, reversible couple with equal diffusion coefficients |
| **Peak separation** \\(\Delta E_p\\) | \\(E_{pa} - E_{pc}\\) | Diagnostic of how fast the electron transfer is | Inflated by uncompensated resistance just as convincingly as by slow kinetics — see Section 4.6 |
| **Peak current** \\(i_p\\) | Height of the peak | Scales with concentration and with the square root of scan rate | The scan-rate exponent is the useful part; converting a height to a diffusion coefficient requires assumptions the simulation in Section 4.7 makes explicit |

The peak separation deserves a specific comment because it is the most over-read number in the field. For a fast (**reversible**) one-electron couple, \\(\Delta E_p\\) takes a characteristic small value that our simulation will compute from first principles. Slow (**quasi-reversible**) electron transfer widens it, and the widening grows with scan rate, because a faster sweep gives the interface less time to keep up. That is a genuine and useful diagnostic. The trouble is that **uncompensated resistance widens \\(\Delta E_p\\) in a very similar way** — also increasing with scan rate, because faster scans mean bigger currents mean bigger \\(iR\\). Reporting slow kinetics on the basis of a wide \\(\Delta E_p\\), without having dealt with resistance, is one of the most common errors in the literature.

## 4.5 Capacitive and Faradaic Current

Two physically distinct currents flow in every electrochemical measurement, and telling them apart is a permanent practical concern.

**Faradaic current** crosses the interface. Electrons are transferred to or from a chemical species, a reaction occurs, and the charge passed is related to the amount of substance converted by Faraday's laws — the relation Chapter 1 used. This is the current that does the thing you want.

**Capacitive (non-faradaic) current** does not cross the interface. It is the charging and discharging of the double-layer capacitor from Section 4.1: charge accumulates on the metal and counter-charge in the solution, but nothing reacts and nothing is produced. A rearrangement, not a conversion.

Their scan-rate behaviour is completely different, and that difference is the handle you have on them. For a capacitor charged by a linear potential ramp, the current is the capacitance times the rate of change of voltage:

\\[ i_{\text{cap}} \;=\; C_{\text{dl}} A \, v \\]

**linear in scan rate**. For a diffusion-controlled faradaic peak, the peak current instead goes as

\\[ i_p \;\propto\; \sqrt{v} \\]

which the simulation in Section 4.7 will confirm to three decimal places without assuming it. The physical reason is that a faster sweep gives the depletion layer less time to grow; a thinner layer means a steeper gradient, and the gradient goes as \\(1/\sqrt{Dt}\\).

The consequence is unpleasant and unavoidable: **the ratio of the signal you want to the background you do not falls as \\(1/\sqrt{v}\\)**. Fast scans, which are exactly what you need to catch a short-lived intermediate, are also the conditions in which capacitive current does the most to bury the faradaic peak. Section 4.7 puts numbers on this trade-off.

## 4.6 iR Compensation: The Voltage You Never Applied

The last assumption to examine is the one Chapter 3 quietly relied on: that the potential you commanded is the potential the interface felt.

It is not, quite. Between the working electrode surface and the tip of the reference electrode sits a certain amount of electrolyte, and that electrolyte has resistance. When current flows through it, Ohm's law takes its cut. The potential actually experienced by the interface is:

\\[ E_{\text{true}} \;=\; E_{\text{applied}} \;-\; i R_u \\]

where \\(R_u\\) is the **uncompensated resistance** — the portion of the solution resistance that lies between the working electrode and the reference tip, and therefore inside the measurement loop.

Three properties of this error make it treacherous.

**It grows with current.** At microamps it is invisible. At hundreds of milliamps, on the same cell, it can be hundreds of millivolts — comparable to the entire overpotential you are trying to measure. Section 4.7 tabulates this, and the numbers are sobering.

**It always distorts in the same direction.** The interface always feels less driving force than you commanded, so measured currents come out low at any given applied potential, Tafel slopes come out too large, and voltammetric peaks separate more than they should. Every artefact points toward "worse kinetics than reality".

**It depends on geometry as much as chemistry.** \\(R_u\\) is set by electrolyte conductivity, by the distance from working electrode to reference tip, and by the shape of the current path. So the first line of defence is physical: **place the reference tip close to the working electrode** — classically with a fine drawn capillary positioned near the surface — and use a conductive supporting electrolyte. Every millimetre of solution excluded from the loop is resistance you never have to correct.

What cannot be removed physically is handled two ways. **Positive feedback**, in which the potentiostat adds an estimated \\(iR_u\\) back into its output in real time, is fast but can drive the control loop into oscillation if over-applied. **Post-measurement correction**, in which \\(R_u\\) is measured independently — high-frequency impedance is the usual route — and each data point corrected afterwards, is safer and fully auditable.

Whichever is used, the professional obligation is the same: **state \\(R_u\\), state how it was determined, and state how much of it was compensated.** A Tafel slope or a peak separation reported without that information is a measurement of the cell as much as of the chemistry.

## 4.7 Hands-On: Building a Voltammogram from Diffusion Alone

The simulation below builds a cyclic voltammogram from the smallest set of assumptions that can produce one: a single dissolved species undergoing a one-electron reduction, with electron transfer assumed **fast** so the surface obeys the Nernst equation at every instant — meaning there is no kinetic model in the code at all. The only remaining physics is semi-infinite diffusion, solved by explicit finite differences on a one-dimensional grid.

That austerity is the point. If the characteristic peaked shape, the peak separation and the scan-rate scaling all emerge from **diffusion plus Nernst and nothing else**, those features are not evidence of anything more exotic. They are the baseline any real voltammogram must be compared against.

Two conventions before reading the code. Reduction current is plotted as positive here, one of the two conventions in use — check the axis of any voltammogram you meet. And the transport parameters (a diffusion coefficient of \\(10^{-5}\\) cm²/s, a 1 mM solution, a 1 cm² electrode) are conventional orders of magnitude for a small ion in water; they set the *scale* of the currents but none of the *shape* conclusions, all of which are dimensionless.

```python
import numpy as np

# ---------------------------------------------------------------
# A minimal cyclic voltammogram, from diffusion alone.
#
# One dissolved species O, one electron, O + e- <-> R.
# Electron transfer is assumed FAST, so the electrode surface obeys
# the Nernst equation at every instant. The only physics left is
# semi-infinite diffusion, solved by explicit finite differences.
#
# Nothing here is fitted to an experiment. The duck shape, the peak
# separation and the sqrt(scan-rate) scaling all fall out of
# diffusion + Nernst and nothing else.
# ---------------------------------------------------------------
F = 96485.0        # C/mol
R_GAS = 8.314      # J/(mol K)
T = 298.15         # K
f = F / (R_GAS * T)

D = 1.0e-5         # cm^2/s   conventional order of magnitude for a small ion in water
C_BULK = 1.0e-6    # mol/cm^3 = 1 mM
AREA = 1.0         # cm^2
E0 = 0.0           # V, formal potential placed at zero for convenience

E_START = 0.4      # V, well positive of E0: nothing happens yet
E_SWITCH = -0.4    # V, well negative of E0: reduction has run its course


def simulate_cv(scan_rate, n_time=40000, lam=0.40):
    """One full cycle. Returns (E, i, cO at the switching potential, dx).

    lam = D dt/dx^2 is the explicit-scheme stability number and must stay <= 0.5.
    """
    span = abs(E_START - E_SWITCH)
    t_total = 2.0 * span / scan_rate
    dt = t_total / n_time
    dx = np.sqrt(D * dt / lam)

    # The depletion layer never grows past a few sqrt(D t); add margin.
    n_x = int(6.0 * np.sqrt(D * t_total) / dx) + 3
    cO = np.full(n_x, C_BULK)
    cR = np.zeros(n_x)

    E = np.empty(n_time)
    i = np.empty(n_time)
    cO_switch = None

    for k in range(n_time):
        t = (k + 1) * dt
        # triangular sweep: down to E_SWITCH, then straight back up
        if t <= t_total / 2.0:
            e = E_START - scan_rate * t
        else:
            e = E_SWITCH + scan_rate * (t - t_total / 2.0)

        # interior: one explicit diffusion step for each species
        cO[1:-1] += lam * (cO[2:] - 2.0 * cO[1:-1] + cO[:-2])
        cR[1:-1] += lam * (cR[2:] - 2.0 * cR[1:-1] + cR[:-2])

        # surface: equal D means O and R trade one-for-one, so cO+cR is
        # continuous across the boundary; Nernst fixes their ratio.
        theta = np.exp(f * (e - E0))          # = cO(0)/cR(0)
        total = cO[1] + cR[1]
        cR[0] = total / (1.0 + theta)
        cO[0] = total - cR[0]

        # surface gradient, three-point (second-order) form
        grad = (-3.0 * cO[0] + 4.0 * cO[1] - cO[2]) / (2.0 * dx)
        i[k] = F * AREA * D * grad            # A, reduction taken as positive
        E[k] = e
        if k == n_time // 2 - 1:              # snapshot at the switching potential
            cO_switch = cO.copy()

    return E, i, cO_switch, dx


# --- 1. one voltammogram, and its landmarks --------------------
v = 0.100                                     # V/s
E, i, cO_switch, dx = simulate_cv(v)
half = len(E) // 2

kc = int(np.argmax(i[:half]))                 # cathodic peak, forward sweep
ka = int(np.argmin(i[half:])) + half          # anodic peak, reverse sweep
ipc, Epc = i[kc], E[kc]
ipa, Epa = i[ka], E[ka]

print("Step 1: landmarks of the simulated voltammogram")
print(f"  scan rate                 : {v * 1000:.0f} mV/s")
print(f"  cathodic peak ipc         : {ipc * 1e6:.2f} uA at Epc = {Epc * 1000:+.1f} mV")
print(f"  anodic  peak ipa          : {ipa * 1e6:.2f} uA at Epa = {Epa * 1000:+.1f} mV")
print(f"  peak separation dEp       : {(Epa - Epc) * 1000:.1f} mV")
print(f"  midpoint (Epa+Epc)/2      : {(Epa + Epc) / 2 * 1000:+.1f} mV  (E0 was set to {E0 * 1000:+.0f} mV)")
print(f"  RT/F                      : {R_GAS * T / F * 1000:.1f} mV")
print(f"  (Epc - E0) / (RT/F)       : {Epc / (R_GAS * T / F):.2f}")
print(f"  raw |ipa / ipc|           : {abs(ipa / ipc):.3f}  (measured from zero, NOT baseline-corrected)")
print()

# --- 2. why the current falls after the peak -------------------
print("Step 2: after the cathodic peak the current decays")
for offset_mV in [0, -25, -50, -100, -200]:
    j = int(np.argmin(np.abs(E[:half] - (Epc + offset_mV / 1000.0))))
    print(f"  E = Epc {offset_mV:+5d} mV : i = {i[j] * 1e6:7.2f} uA "
          f"({100.0 * i[j] / ipc:5.1f} % of peak)")
t_switch = 0.8 / v
depleted = int(np.argmax(cO_switch > 0.99 * C_BULK)) * dx
print(f"  at the switching potential, O is depleted out to ~{depleted * 1e4:.0f} um")
print(f"  for comparison sqrt(D t) at that moment = {np.sqrt(D * t_switch) * 1e4:.0f} um")
print()

# --- 3. does the peak scale with the square root of scan rate? -
print("Step 3: peak current vs scan rate")
print(f"{'v (mV/s)':>10} {'ipc (uA)':>12} {'ipc/v':>12} {'ipc/sqrt(v)':>14} {'dEp (mV)':>10}")
print("-" * 62)
rates = [0.025, 0.050, 0.100, 0.200, 0.400]
peaks = []
for vr in rates:
    Er, ir, _, _ = simulate_cv(vr)
    h = len(Er) // 2
    c = int(np.argmax(ir[:h]))
    a = int(np.argmin(ir[h:])) + h
    peaks.append(ir[c])
    print(f"{vr * 1000:10.0f} {ir[c] * 1e6:12.3f} {ir[c] / vr * 1e6:12.1f} "
          f"{ir[c] / np.sqrt(vr) * 1e6:14.3f} {(Er[a] - Er[c]) * 1000:10.1f}")
peaks = np.array(rates), np.array(peaks)
const = peaks[1] / np.sqrt(peaks[0])
print(f"  spread of ipc/sqrt(v) over a 16x range of scan rate: "
      f"{100.0 * (const.max() - const.min()) / const.mean():.2f} %")
slope = np.polyfit(np.log10(peaks[0]), np.log10(peaks[1]), 1)[0]
print(f"  fitted exponent of ipc vs v (log-log): {slope:.3f}")
print()

# --- 4. capacitive current does not behave that way ------------
# A double layer of capacitance Cdl charged at scan rate v draws
# i_cap = Cdl * A * v -- linear in v, not sqrt(v).
CDL = 20.0e-6      # F/cm^2, textbook order of magnitude for a metal in aqueous electrolyte
print("Step 4: faradaic peak vs capacitive background")
print(f"{'v (mV/s)':>10} {'faradaic (uA)':>15} {'capacitive (uA)':>17} {'ratio':>9}")
print("-" * 55)
for vr, ipk in zip(*peaks):
    icap = CDL * AREA * vr
    print(f"{vr * 1000:10.0f} {ipk * 1e6:15.3f} {icap * 1e6:17.3f} {ipk / icap:9.1f}")
for vr in [10.0, 100.0, 1000.0]:
    ipk = const.mean() * np.sqrt(vr)
    icap = CDL * AREA * vr
    print(f"{vr * 1000:10.0f} {ipk * 1e6:15.3f} {icap * 1e6:17.3f} {ipk / icap:9.1f}")
print()
print("  faradaic ~ sqrt(v), capacitive ~ v, so their RATIO falls as 1/sqrt(v):")
print("  fast scans progressively bury the peak in charging current.")
print()

# --- 5. how much voltage does the solution eat? ----------------
# Ohmic drop is a property of the cell, not of the reaction.
print("Step 5: uncompensated resistance turns current into lost potential")
print(f"{'Ru (ohm)':>10} {'10 uA':>12} {'1 mA':>12} {'100 mA':>12}")
print("-" * 48)
for Ru in [1.0, 10.0, 100.0]:
    a1, a2, a3 = (Ru * cur * 1000.0 for cur in (10e-6, 1e-3, 100e-3))
    print(f"{Ru:10.0f} {a1:9.3f} mV {a2:9.3f} mV {a3:9.3f} mV")
```

**Output:**

```
Step 1: landmarks of the simulated voltammogram
  scan rate                 : 100 mV/s
  cathodic peak ipc         : 268.65 uA at Epc = -28.5 mV
  anodic  peak ipa          : -208.78 uA at Epa = +29.0 mV
  peak separation dEp       : 57.5 mV
  midpoint (Epa+Epc)/2      : +0.2 mV  (E0 was set to +0 mV)
  RT/F                      : 25.7 mV
  (Epc - E0) / (RT/F)       : -1.11
  raw |ipa / ipc|           : 0.777  (measured from zero, NOT baseline-corrected)

Step 2: after the cathodic peak the current decays
  E = Epc    +0 mV : i =  268.65 uA (100.0 % of peak)
  E = Epc   -25 mV : i =  249.03 uA ( 92.7 % of peak)
  E = Epc   -50 mV : i =  214.49 uA ( 79.8 % of peak)
  E = Epc  -100 mV : i =  161.81 uA ( 60.2 % of peak)
  E = Epc  -200 mV : i =  115.99 uA ( 43.2 % of peak)
  at the switching potential, O is depleted out to ~232 um
  for comparison sqrt(D t) at that moment = 89 um

Step 3: peak current vs scan rate
  v (mV/s)     ipc (uA)        ipc/v    ipc/sqrt(v)   dEp (mV)
--------------------------------------------------------------
        25      134.324       5373.0        849.540       57.5
        50      189.963       3799.3        849.540       57.5
       100      268.648       2686.5        849.540       57.5
       200      379.926       1899.6        849.540       57.5
       400      537.296       1343.2        849.540       57.5
  spread of ipc/sqrt(v) over a 16x range of scan rate: 0.00 %
  fitted exponent of ipc vs v (log-log): 0.500

Step 4: faradaic peak vs capacitive background
  v (mV/s)   faradaic (uA)   capacitive (uA)     ratio
-------------------------------------------------------
        25         134.324             0.500     268.6
        50         189.963             1.000     190.0
       100         268.648             2.000     134.3
       200         379.926             4.000      95.0
       400         537.296             8.000      67.2
     10000        2686.481           200.000      13.4
    100000        8495.398          2000.000       4.2
   1000000       26864.808         20000.000       1.3

  faradaic ~ sqrt(v), capacitive ~ v, so their RATIO falls as 1/sqrt(v):
  fast scans progressively bury the peak in charging current.

Step 5: uncompensated resistance turns current into lost potential
  Ru (ohm)        10 uA         1 mA       100 mA
------------------------------------------------
         1     0.010 mV     1.000 mV   100.000 mV
        10     0.100 mV    10.000 mV  1000.000 mV
       100     1.000 mV   100.000 mV 10000.000 mV
```

**Reading the result.** Six observations, in increasing order of importance.

  * **The peaks land where the thermal scale says they should.** The cathodic peak sits at \\(-28.5\\) mV relative to the formal potential, which the code divides by \\(RT/F = 25.7\\) mV to get \\(-1.11\\): the peak is displaced by very close to **one thermal voltage**, with no adjustable parameter anywhere in the model that could have been tuned to produce it. The anodic peak lands symmetrically at \\(+29.0\\) mV, and their midpoint, \\(+0.2\\) mV, recovers the formal potential we set to zero. That is the justification for taking the midpoint of two peaks as the formal potential — it is correct, and now you know under precisely what assumptions.

  * **The peak separation comes out at 57.5 mV.** This is the number quoted as the signature of a fast, reversible one-electron couple, and here it is derived rather than remembered. Note how close it sits to the 59 mV Nernst slope that has appeared in every chapter of this series — near enough that many texts quote 59 mV for both, though they are related rather than identical and our simulation is precise enough to distinguish them. Anything appreciably larger in a real measurement means slow electron transfer, uncompensated resistance, or both — and Section 4.6 explains why the second is the more common culprit and the easier one to overlook.

  * **The current after the peak decays, and slowly.** Step 2 tracks it: 200 mV past the peak, pushing far harder than at the peak itself, the current has fallen to \\(43.2\\)% of maximum. Nothing about the driving force caused that. At the switching potential the reactant has been stripped out to roughly 232 μm from the surface — several times the \\(\sqrt{Dt}\\) of 89 μm at that moment — and the gradient feeding the electrode has flattened accordingly. **Past the peak, the potentiostat is no longer in charge. Diffusion is.**

  * **The square-root law is exact, not approximate.** Step 3 varies the scan rate over a 16-fold range. The ratio \\(i_p / \sqrt{v}\\) is constant to \\(0.00\\)% across it, and a log-log fit returns an exponent of \\(0.500\\); meanwhile \\(i_p / v\\) varies by a factor of four, so the peak is definitively **not** proportional to scan rate. This is the diagnostic that distinguishes a freely diffusing species from a surface-attached one, whose signal would scale linearly with \\(v\\) instead — and it drops out of a model that contains no such distinction as an assumption. The peak separation, meanwhile, does not budge from 57.5 mV at any scan rate: electron transfer here is infinitely fast, so there is nothing for a faster sweep to outrun.

  * **The capacitive background wins in the end.** Step 4 sets a double-layer capacitance of a conventional order of magnitude against the simulated faradaic peaks. At 25 mV/s the peak is 269 times the charging current — the background is invisible. At 10 V/s the ratio has fallen to \\(13.4\\); at 1000 V/s it is \\(1.3\\), and the peak you are hunting is the same size as the background under it. The two currents obey different power laws in \\(v\\), so their ratio falls as \\(1/\sqrt{v}\\) forever and no amplifier fixes it. **This is the fundamental limit on fast-scan voltammetry** — chasing short-lived intermediates by scanning faster runs into a wall, not a cost.

  * **Ohmic drop is small until suddenly it is not.** Step 5 is Ohm's law and nothing more, included because the numbers are easy to under-imagine. A modest 10 Ω of uncompensated resistance costs \\(0.1\\) mV at 10 μA, which is nothing; the same 10 Ω costs 10 mV at 1 mA, and **1000 mV at 100 mA**. Nothing about the cell changed between those rows — only the current. This is why voltammetry on a microelectrode at nanoamps can ignore resistance entirely, and why an electrolyser at hundreds of milliamps per square centimetre cannot report any kinetic parameter without correcting for it.

One honest wrinkle worth pointing out, because it is a real trap. The printed \\(|i_{pa}/i_{pc}|\\) is \\(0.777\\), not \\(1.000\\), even though this simulated couple is perfectly reversible and perfectly stable — the product cannot decompose, because the model contains no chemistry that would let it. The ratio is not \\(1\\) because the reverse peak was measured from **zero current**, while it actually sits on top of the still-decaying tail of the forward reaction. Measuring it properly requires extrapolating that decaying forward current under the reverse peak and using it as the baseline. A ratio near \\(0.78\\) measured naively from zero is therefore what a *stable* product looks like, and someone who expects \\(1.0\\) from a raw plot will conclude their product is decomposing when it is not. **The baseline is part of the measurement**, and this is exactly the kind of thing a simulation with no hidden chemistry is good for: any deviation must be an artefact of how we measured, because there is nothing else it could be.

Two experiments worth running on this code. Set `E_SWITCH = -0.1`, turning around before the reduction has run its course, and the raw peak ratio falls from \\(0.777\\) to \\(0.572\\) while the forward peak is untouched at \\(268.65\\) μA — a reminder that the switching potential is an experimental choice with consequences, and another reason a raw peak ratio needs its measurement conditions attached. Then set `lam = 0.6`, violating the stability condition of the explicit scheme, and the concentration arrays blow up: overflow warnings, then `nan` everywhere. The second is a useful lesson in its own right — a numerical method that is fine up to a threshold and catastrophically wrong beyond it will not warn you in advance, and the check has to be yours.

### 🎯 Exercise Problems

  1. **Counting interfaces.** Draw a two-electrode cell and mark every location where a potential difference exists between the terminals of the voltmeter. Then draw the three-electrode version and explain, in terms of your diagram, precisely which of those potential differences the potentiostat has removed from the measurement and by what mechanism.

  2. **Which reference?** You are asked to compare a water-oxidation catalyst at pH 1, pH 7 and pH 13. Decide whether to report versus Ag/AgCl or versus RHE, and justify the choice quantitatively using the 59 mV per pH unit shift. Then state one situation in which the opposite choice would be the better one.

  3. **Reading a scan-rate series.** A colleague measures peak current at 10, 40 and 160 mV/s and finds 12, 24 and 48 μA. Is the species freely diffusing or surface-attached? Show the reasoning using the two power laws in Section 4.5, and state what additional measurement would confirm your conclusion.

  4. **Diagnosing a wide peak separation.** A voltammogram shows \\(\Delta E_p = 150\\) mV at 100 mV/s, growing to 320 mV at 1 V/s. Give two explanations consistent with this observation, then design a single experiment that distinguishes them. (Consider what you could change about the cell without changing the chemistry.)

  5. **The fast-scan wall.** Using the Step 4 output, estimate the scan rate at which the faradaic peak would fall to one tenth of the capacitive background for this system. Then, given that a faradaic peak can be recovered from a capacitive background if the background is reproducible, argue what property of the double layer would have to hold for background subtraction to rescue the measurement — and why that property is hard to guarantee.

## Summary

This chapter examined the assumption the previous two chapters rested on: that "the potential of the electrode" is a well-defined thing we know and control.

The potential difference lives almost entirely in the **electric double layer**, a region a few molecular diameters thick in which counter-ions accumulate to mirror the charge on the metal. The closest approach of the solvated ions defines the **Helmholtz plane**, and because a volt or so is dropped across a fraction of a nanometre, the field there is on the order of \\(10^9\\) V/m. That extreme field is what makes electrode potential a knob on activation energy, and the same geometry makes the interface an efficient **capacitor** — so a current flows whenever the potential changes, whether or not anything reacts.

**A two-electrode measurement cannot isolate one interface**, because a circuit contains two of them plus an ohmic path and a voltmeter reports the sum. The **three-electrode cell** fixes this by splitting the jobs: the **counter electrode** carries all the current, its own unmeasured overpotential being irrelevant, while the **reference electrode** carries essentially none and therefore stays pinned at its equilibrium potential. The potentiostat holds the working-reference voltage where you asked, adjusting the working-counter voltage by whatever it takes.

The scale is defined by SHE and measured against something practical. **Ag/AgCl** and **SCE** sit at roughly \\(+0.2\\) V versus SHE, with the exact value depending on filling solution and temperature — a potential quoted without its reference is not data. **RHE** is the important special case: because the hydrogen reaction shifts by **59 mV per pH unit** exactly as water splitting does, reporting versus RHE cancels the pH dependence, putting hydrogen evolution at \\(0\\) V and water oxidation at \\(1.23\\) V at every pH.

**Cyclic voltammetry** sweeps the potential and records current. The rise comes from the Nernstian demand for a new surface composition; the **peak** comes not from any change in driving force but from **reactant depletion**, as the diffusion layer thickens and the gradient feeding the electrode flattens; the **reverse peak** reports on whether the product survived long enough to be converted back.

Our simulation built all of this from **diffusion plus Nernst and nothing else**: no kinetic model, no fitted parameter. It placed the cathodic peak at \\(-28.5\\) mV, precisely \\(1.11 \times RT/F\\) from the formal potential; recovered the formal potential as the peak midpoint to within \\(0.2\\) mV; produced a peak separation of \\(57.5\\) mV; showed \\(i_p/\sqrt{v}\\) constant to \\(0.00\\)% over a 16-fold scan-rate range with a fitted exponent of \\(0.500\\); showed the current still at \\(43.2\\)% of peak a full 200 mV past it, with the reactant depleted out to 232 μm; and demonstrated that the faradaic-to-capacitive ratio falls from 269 at 25 mV/s to \\(1.3\\) at 1000 V/s — the hard limit on fast-scan work. It also produced a raw peak-current ratio of \\(0.777\\) for a perfectly stable product, a warning that **the baseline is part of the measurement**.

Finally, **uncompensated resistance** silently subtracts \\(iR_u\\) from every potential you apply. The same 10 Ω costs \\(0.1\\) mV at 10 μA and 1000 mV at 100 mA, and every distortion it causes points the same way: kinetics that look worse than they are. Move the reference close, use conductive electrolyte, compensate what remains, and report what you did.

Chapter 5 spends all of this. We build the full voltage budget of a water electrolyser — the \\(1.23\\) V from Chapter 2, plus the two activation overpotentials from Chapter 3, plus the \\(iR\\) from this chapter — and see which term dominates and why the oxygen side is the expensive one. We look at \\(\text{CO}_2\\) electrolysis, where multi-electron pathways make selectivity rather than rate the central problem, and re-read battery charge and discharge as a thermodynamic quantity with a kinetic tax. Along the way we connect to the [OER Computational Chemistry](<../../MI/oer-computational-chemistry/index.html>) series, which computes the overpotential of an oxygen-evolution catalyst from first principles before anyone synthesises it.

[← Chapter 3: Kinetics — Overpotential and Tafel Analysis](<chapter-3.html>) [Chapter 5: Applications — From Electrolysis to Batteries →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
