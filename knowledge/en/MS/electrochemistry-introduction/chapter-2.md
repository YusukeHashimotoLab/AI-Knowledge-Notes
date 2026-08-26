---
title: "Chapter 2: Electrode Potentials and Thermodynamics"
chapter_title: "Chapter 2: Electrode Potentials and Thermodynamics"
subtitle: "How a Table of Numbers Predicts Every Cell Voltage, Why Water Splitting Cannot Cost Less Than 1.23 V, and What the Nernst Equation Is Really Saying"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/KMzZJz8J10g"
    title="Electrochemistry Ch.2: Electrode Potentials and Thermodynamics"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/MS/electrochemistry-introduction/chapter-2.html>) | Last sync: 2026-08-20

[Materials Science Dojo](<../index.html>) > [Electrochemistry Introduction](<index.html>) > Chapter 2

Chapter 1 ended with Faraday's laws, which answer the question *how much*. Pass a known charge and you know exactly how many moles of product appear. What Faraday's laws are completely silent about is *whether the reaction happens at all*, and *at what voltage*.

Those two questions have one answer, and it is thermodynamic. A cell voltage is not an empirical property of a device; it is a free energy divided by a charge. Once you see that — once the relation \\(\Delta G = -nFE\\) is genuinely familiar rather than merely memorized — a large amount of electrochemistry stops being a collection of facts and becomes arithmetic on a single table.

This chapter builds that table and then uses it four times. We construct the potential scale from the standard hydrogen electrode, connect it to Gibbs free energy, compute the Daniell cell from two tabulated numbers, derive the Nernst equation and the famous 59 millivolts per decade that falls out of it at room temperature, and finally derive the 1.23 V floor that every water electrolyzer on Earth operates above.

One warning before we start, because it shapes how the rest of the series should be read. **Everything in this chapter is about what is possible, not about what is fast.** A reaction with a favourable voltage may be so slow as to be undetectable; a reaction at exactly its thermodynamic voltage produces no current at all. That gap is Chapter 3's subject, and it is where catalysis and most of materials science live. Thermodynamics tells you the floor. Kinetics tells you the rent.

## 2.1 The Problem With Half of a Reaction

We want to assign a number to a half-reaction — a measure of how strongly it pulls electrons. The trouble is that no measurement of a single electrode is possible, and the reason is not technological.

To measure a voltage you need two probes. Put one on your zinc electrode and the other in the solution, and you have not measured the zinc electrode alone — you have inserted a second piece of metal into the electrolyte, creating a second interface with its own potential difference, and your voltmeter reports the sum. Replace the probe with a different metal and the reading changes. **There is no way to measure the potential of one interface in isolation, because every complete circuit contains at least two of them.**

The response is the one physics uses whenever an absolute quantity is inaccessible: pick a reference, assign it zero by convention, and report everything relative to it. In mechanics we do this with gravitational potential energy and nobody objects. In electrochemistry the chosen reference is the **standard hydrogen electrode (SHE)**:

\\[
2\,\mathrm{H^+}(aq) + 2e^{-} \rightleftharpoons \mathrm{H_2}(g), \qquad E^{\circ} \equiv 0.000\ \mathrm{V}
\\]

realized as a platinized platinum electrode in an acid solution of unit hydrogen-ion activity, with hydrogen gas bubbled over it at standard pressure, at a specified temperature — 25 °C for standard tables. The value zero is **defined**, not measured. Every other electrode potential in every table you will ever use is a voltage measured against this electrode, at standard conditions.

Two immediate consequences.

**Standard potentials are differences, and only differences are physical.** If someone offered you a new convention in which the SHE were assigned +5.00 V, every tabulated potential would shift by exactly +5.00 V and every *cell voltage* — being a difference of two of them — would come out unchanged. The individual numbers are bookkeeping; the differences are what the voltmeter reads.

**The SHE is a definition, not a laboratory habit.** It requires flammable gas at controlled pressure, a fragile platinized surface, and an acid of known activity, and it is easily poisoned. Practical work uses secondary reference electrodes that are stable, sealed, and convenient, and whose potential relative to the SHE is known — a silver/silver chloride electrode sits at roughly +0.2 V vs SHE, depending on the chloride concentration inside it. That is why experimental papers write potentials "vs Ag/AgCl" or "vs RHE" rather than "vs SHE". Chapter 4 covers reference electrodes properly, including why converting between scales is a routine source of error in the literature.

### 📚 What the Sign of a Standard Potential Tells You

Read a tabulated \\(E^{\circ}\\) as an answer to one question: *compared with hydrogen, how badly does this species want electrons?*

  * **Positive \\(E^{\circ}\\)** — the half-reaction, written as a reduction, is more favourable than hydrogen reduction. The oxidized form is a **stronger oxidizing agent** than \\(\mathrm{H^+}\\). Copper(II) at \\(+0.34\\) V will take electrons from hydrogen gas.
  * **Negative \\(E^{\circ}\\)** — the reduction is less favourable than hydrogen's. The reduced form is a **stronger reducing agent** than \\(\mathrm{H_2}\\). Zinc metal at \\(-0.76\\) V will give electrons to acid, which is precisely why zinc dissolves in hydrochloric acid and copper does not.
  * **The magnitude is a free energy per electron.** A potential is joules per coulomb. A half-reaction at \\(-0.76\\) V is not "0.76 units bad"; it is a specific amount of free energy per mole of electrons transferred, and Section 2.3 converts it.

Note the direction of the reasoning in the second bullet. The fact that zinc dissolves in acid and copper does not is an *observation*, known long before anyone tabulated potentials. The potential table is a compressed, quantitative encoding of thousands of such observations.

## 2.2 The Potential Series and the Activity Series

Arrange the half-reactions in order of their standard reduction potentials and you get a ladder. Every entry is written as a reduction, with electrons on the left, following the **IUPAC convention** established in Chapter 1.

| Half-reaction (written as reduction) | \\(E^{\circ}\\) (V vs SHE) |
|---|---|
| \\(\mathrm{Li^+} + e^- \rightleftharpoons \mathrm{Li}\\) | \\(-3.04\\) |
| \\(\mathrm{Al^{3+}} + 3e^- \rightleftharpoons \mathrm{Al}\\) | \\(-1.66\\) |
| \\(\mathrm{Zn^{2+}} + 2e^- \rightleftharpoons \mathrm{Zn}\\) | \\(-0.76\\) |
| \\(\mathrm{Fe^{2+}} + 2e^- \rightleftharpoons \mathrm{Fe}\\) | \\(-0.44\\) |
| \\(2\mathrm{H^+} + 2e^- \rightleftharpoons \mathrm{H_2}\\) | \\(0.00\\) (defined) |
| \\(\mathrm{Cu^{2+}} + 2e^- \rightleftharpoons \mathrm{Cu}\\) | \\(+0.34\\) |
| \\(\mathrm{Ag^+} + e^- \rightleftharpoons \mathrm{Ag}\\) | \\(+0.80\\) |
| \\(\mathrm{O_2} + 4\mathrm{H^+} + 4e^- \rightleftharpoons 2\mathrm{H_2O}\\) | \\(+1.23\\) |
| \\(\mathrm{Cl_2} + 2e^- \rightleftharpoons 2\mathrm{Cl^-}\\) | \\(+1.36\\) |
| \\(\mathrm{F_2} + 2e^- \rightleftharpoons 2\mathrm{F^-}\\) | \\(+2.87\\) |

These are the standard tabulated values, conventionally quoted to two decimal places at 25 °C; different compilations occasionally differ in the last digit, and none of the reasoning below depends on that digit.

Read the table from the bottom up and you have a ranking of oxidizing power: fluorine is the strongest common oxidant, which is why fluorine chemistry is difficult and dangerous. Read it from the top down and you have a ranking of reducing power: lithium is the strongest common reductant, which is why lithium is the metal of choice for high-voltage batteries — its position on this ladder is most of the reason lithium-ion cells deliver the voltages they do.

Chemistry students meet a close relative of this table long before they meet electrochemistry: the **activity series** (or *ionization tendency*) of metals, memorized as an ordering of which metals displace which from solution, which react with acid, and which are found native in the ground. **The activity series is the potential series, restricted to metal/metal-ion couples, with the numbers removed.** A metal displaces the ion of any metal below it in the table; that is exactly the statement that the pairing produces a positive cell voltage.

### 📚 Reading Reactions Off the Table

Two rules extract predictions, and both are simple statements about differences.

  * **Any species on the left of a higher entry will oxidize any species on the right of a lower entry.** Silver ions (left side, \\(+0.80\\)) will oxidize copper metal (right side, \\(+0.34\\)) — drop a copper wire into silver nitrate and silver crystals grow on it while the solution turns blue.
  * **The cell voltage of any pairing is the difference of the two tabulated values**, cathode minus anode, both read as reductions. Silver against copper gives \\(0.80 - 0.34 = 0.46\\) V. That subtraction is the whole method.

A caution belongs with the second rule: \\(E^{\circ}\\) values are intensive, not extensive. If you multiply a half-reaction by two, its potential does **not** double. Potential is energy *per unit charge*, and doubling the reaction doubles both the energy and the charge. This is a common early mistake, and it disappears the moment you internalize Section 2.3.

The table also contains a warning that will grow into Chapter 3. Look at the oxygen entry at \\(+1.23\\) V and the chlorine entry at \\(+1.36\\) V. In a brine electrolysis cell, thermodynamics says oxygen should evolve preferentially — its potential is lower, so oxidizing water is *easier*. Industrially, chlorine is what comes out. Thermodynamics was not violated; it was simply outvoted by kinetics, because oxygen evolution is a four-electron process that is very slow on most surfaces while chlorine evolution is fast. **A table of thermodynamic potentials predicts what is allowed, and is frequently wrong about what happens.**

## 2.3 The Bridge: \\(\Delta G = -nFE\\)

Now the central equation of the chapter, and it is worth deriving rather than quoting, because the derivation explains every one of its parts.

Thermodynamics says that for a process at constant temperature and pressure, the Gibbs free energy change equals the **maximum non-expansion work** the process can deliver:

\\[
\Delta G = w_{\text{elec, max}}
\\]

with the sign convention that work done *by* the system is negative. In an electrochemical cell, that non-expansion work is electrical work, and electrical work is elementary: moving charge \\(Q\\) through a potential difference \\(E\\) costs \\(Q \cdot E\\). From Chapter 1, transferring \\(n\\) moles of electrons per mole of reaction means \\(Q = nF\\). Therefore the maximum work the cell can do on the outside world is \\(nFE\\), and

\\[
\Delta G = -nFE
\\]

Three things are packed into that short expression.

**The minus sign is a convention made honest.** A spontaneous reaction has \\(\Delta G < 0\\) and, by this equation, a positive cell voltage. So *positive \\(E\\) means the cell runs by itself*, which is the intuition everyone already has about batteries. The minus sign exists to make those two statements agree.

**\\(n\\) resolves the intensive/extensive puzzle of Section 2.2.** Double the reaction and \\(n\\) doubles, so \\(\Delta G\\) doubles — as an extensive quantity must — while \\(E\\) is unchanged. Free energy is extensive; voltage is free energy per unit charge, and is intensive. The equation carries both facts.

**"Maximum" is doing real work in that sentence.** The relation gives the work obtainable in the *reversible* limit — the limit of infinitesimally slow operation, drawing essentially zero current. Any real cell delivering real current delivers less, and the difference is dissipated as heat. This is not a small caveat; it is the entire economic problem of electrolysis, and Chapter 3 quantifies it.

At standard conditions the same relation reads \\(\Delta G^{\circ} = -nFE^{\circ}\\), which connects the potential table to the equilibrium constant through \\(\Delta G^{\circ} = -RT \ln K\\). Setting the two equal gives

\\[
\ln K = \frac{nFE^{\circ}}{RT}
\\]

so a cell voltage and an equilibrium constant are the same information in different units. Because \\(F/RT\\) is a large number at room temperature, a modest voltage corresponds to an enormous equilibrium constant: this is why a cell voltage of a volt or so describes a reaction that, at equilibrium, has gone essentially to completion.

## 2.4 Building the Daniell Cell

We can now compute the cell from Chapter 1 rather than measuring it. Zinc metal in zinc sulfate, copper metal in copper sulfate, wire and salt bridge between them.

**Step one: identify the two half-reactions and look them up.** From the table, \\(E^{\circ}(\mathrm{Zn^{2+}/Zn}) = -0.76\\) V and \\(E^{\circ}(\mathrm{Cu^{2+}/Cu}) = +0.34\\) V.

**Step two: decide which is the cathode.** The species that is reduced is the one with the higher reduction potential — that is what "higher reduction potential" means. Copper wins, so copper is the cathode and zinc is the anode. Zinc is oxidized, consistent with the observation in Chapter 1 that zinc metal dissolves when copper ions are available.

**Step three: subtract.**

\\[
E^{\circ}_{\text{cell}} = E^{\circ}_{\text{cathode}} - E^{\circ}_{\text{anode}} = (+0.34) - (-0.76) = 1.10\ \mathrm{V}
\\]

Note carefully what was *not* done. The zinc half-reaction runs backwards in this cell — it is an oxidation — but we did **not** flip the sign of its tabulated potential before using it. The subtraction handles the reversal. Flipping the sign *and* subtracting is the single most common arithmetic error in the subject, and it produces \\(-0.42\\) V, a number that should trigger immediate suspicion because it predicts that zinc and copper sulfate do not react, contradicting a demonstration you can do in a beaker.

**Step four: convert to free energy.** With \\(n = 2\\), the code in Section 2.7 gives \\(\Delta G^{\circ} = -212.3\\) kJ/mol — a substantial release, comparable to the enthalpies of ordinary combustion reactions per mole. The same code re-expresses it as **3.25 kJ of maximum electrical work per gram of zinc consumed**, and as **53.6 ampere-hours of charge per mole of zinc**, which are the forms a battery engineer actually wants: energy density and capacity.

### 📚 The Cell Diagram Notation

Electrochemists write cells in a compact line notation, and you should be able to read it:

\\[
\mathrm{Zn}(s) \mid \mathrm{Zn^{2+}}(aq) \parallel \mathrm{Cu^{2+}}(aq) \mid \mathrm{Cu}(s)
\\]

The conventions are fixed: **anode on the left, cathode on the right**; a single vertical bar is a phase boundary; a double bar is a salt bridge. Because the anode is always written on the left, the cell voltage computed as right-minus-left is automatically the correct sign — a positive number means the cell as written is spontaneous. If you write a cell diagram and get a negative voltage, you have written it backwards, and the spontaneous reaction is the reverse of the one you described.

## 2.5 The Nernst Equation: What Happens Away From Standard Conditions

Standard potentials assume unit activity for everything in solution, which is a condition that essentially no real system satisfies. A battery approaching the end of its discharge has consumed most of one reactant and accumulated a great deal of product. Its voltage sags, and we should be able to predict by how much.

The derivation is three lines. From general thermodynamics, the free energy at arbitrary composition relates to the standard value through the reaction quotient \\(Q\\):

\\[
\Delta G = \Delta G^{\circ} + RT \ln Q
\\]

Substitute \\(\Delta G = -nFE\\) on the left and \\(\Delta G^{\circ} = -nFE^{\circ}\\) on the right:

\\[
-nFE = -nFE^{\circ} + RT \ln Q
\\]

Divide through by \\(-nF\\):

\\[
E = E^{\circ} - \frac{RT}{nF} \ln Q
\\]

This is the **Nernst equation**. It says the cell voltage falls below its standard value as products accumulate (large \\(Q\\)) and rises above it when reactants are in excess (small \\(Q\\)), logarithmically in both directions. Note the endpoint: when the reaction reaches equilibrium, \\(Q = K\\) and \\(E = 0\\). **A dead battery is a battery at equilibrium.** It is not empty of chemicals; it has simply reached the composition at which the reaction has no remaining drive.

The logarithm has a practical consequence worth stating plainly: **concentration is a weak lever on voltage.** A tenfold change in a concentration ratio moves the cell voltage by tens of millivolts, not by volts. If you want a different voltage, change the chemistry — pick a different couple from the table. If you want a small, precise, reproducible shift, change the concentration.

Chemists usually prefer base-10 logarithms, and converting gives the form you will see everywhere:

\\[
E = E^{\circ} - \frac{2.303\,RT}{nF} \log_{10} Q
\\]

Evaluate the prefactor at 25 °C, which the code does exactly: \\(RT/F = 25.69\\) mV, and multiplying by \\(\ln 10\\) gives **59.16 mV**. This is the origin of the number every electrochemist carries around as "**59 millivolts per decade**". For a one-electron process, a tenfold change in the reaction quotient shifts the potential by 59 mV. For the two-electron Daniell cell it is half that, **29.58 mV per decade**, and the code's table confirms exactly that slope.

The same 59 mV appears in a place that looks unrelated but is not: the pH dependence of any half-reaction involving protons. Because \\(\mathrm{H^+}\\) enters the reaction quotient, and pH is by definition a base-10 logarithm of \\(\mathrm{H^+}\\) activity, a half-reaction consuming one proton per electron shifts by **59 mV per pH unit** at 25 °C. Oxygen evolution and hydrogen evolution both do this, which is why their potentials move in parallel with pH — and why the reversible hydrogen electrode (RHE) scale, which absorbs that shift, is so convenient in catalysis work. Chapter 4 develops this.

### 📚 The Concentration Cell: Voltage From Nothing But a Gradient

Push the Nernst equation to its strangest conclusion. Build a cell with **the same electrode and the same chemistry on both sides**, differing only in concentration: copper in dilute copper sulfate on the left, copper in concentrated copper sulfate on the right.

The standard cell voltage is exactly zero, because the two half-reactions are identical and their potentials cancel. Yet the cell produces a voltage. Reading the code's Step 5: a tenfold concentration ratio gives **29.6 mV**, a hundredfold gives **59.2 mV**, a thousandfold gives **88.7 mV**.

The driving force is not chemical but entropic — the system's tendency to equalize concentrations. Copper dissolves on the dilute side and deposits on the concentrated side, moving material in the direction that erases the gradient, and delivers electrical work in the process. Two lessons follow. First, **a concentration difference is a form of stored free energy**, which is the operating principle of ion-gradient devices from nerve cells to salinity-gradient power. Second, and more practically, **unintended concentration gradients generate unintended voltages**. A single metal object with an oxygen-rich region and an oxygen-poor region is a short-circuited concentration cell, and that is a substantial part of why localized corrosion happens where it does.

## 2.6 Why Water Splitting Cannot Cost Less Than 1.23 Volts

We can now derive the most-quoted number in the electrochemistry of energy, and it comes from ordinary tabulated thermodynamics rather than from anything electrochemical.

Liquid water forms from its elements with a standard Gibbs energy of formation of about \\(-237\\) kJ/mol. Electrolysis runs that reaction backwards, so decomposing one mole of water demands **at least \\(+237\\) kJ** of free energy:

\\[
\mathrm{H_2O}(l) \longrightarrow \mathrm{H_2}(g) + \tfrac{1}{2}\,\mathrm{O_2}(g), \qquad \Delta G^{\circ} = +237\ \mathrm{kJ/mol}
\\]

Two electrons pass per water molecule decomposed. Rearranging \\(\Delta G = -nFE\\) into \\(E = -\Delta G/(nF)\\) and putting in the numbers gives, as the code confirms, **1.2282 V**, universally quoted as **1.23 V**. That is the *reversible* or *thermodynamic* voltage for water splitting at standard conditions.

The identical number appears in the potential table as \\(E^{\circ}(\mathrm{O_2/H_2O}) = +1.23\\) V, paired against the SHE at \\(0.00\\) V. This is not a coincidence and it is a good consistency check: the oxygen half-reaction's tabulated potential *is* the free energy of water formation, divided by \\(nF\\), expressed in volts.

Now the sharper statement. **1.23 V is a floor, and it cannot be lowered.** No catalyst, no electrode material, no cell architecture, no clever engineering will split water at 1.0 V under standard conditions, for the same reason that no engine converts heat to work at greater than Carnot efficiency. A catalyst that appeared to do so would be a perpetual motion machine of the second kind. Everything a catalyst can do — and it can do a great deal — happens *above* this floor.

### 📚 The Second Voltage: Thermoneutral, About 1.48 V

There is a second characteristic voltage for water electrolysis, and it exists because \\(\Delta G\\) and \\(\Delta H\\) are different quantities.

Water's standard enthalpy of formation is about \\(-286\\) kJ/mol, larger in magnitude than its \\(-237\\) kJ/mol Gibbs energy. The gap, \\(T\Delta S\\), is entropy: splitting liquid water into two gases increases entropy substantially, so the surroundings can supply part of the required energy as **heat** rather than as electrical work.

Applying the same conversion to the enthalpy gives the code's **1.4821 V**, conventionally quoted as **about 1.48 V** and called the **thermoneutral voltage**. Its meaning is thermal:

  * Operating **between 1.23 V and 1.48 V**, the cell must absorb heat from its surroundings and will cool down.
  * Operating **at about 1.48 V**, the electrical input exactly covers the total enthalpy — the cell neither heats nor cools.
  * Operating **above 1.48 V**, as every practical electrolyzer does, the excess is dissipated as heat and the stack requires active cooling.

The difference between the two voltages, **0.2539 V** from the code, is the entropy term expressed in volts. This is why electrolyzer engineers care about both numbers: 1.23 V sets the efficiency ceiling, 1.48 V sets the thermal management problem. The territory between the two is a rare case where a device can run at greater than 100% *electrical* efficiency by drawing in ambient heat — at the cost of running impractically slowly.

## 2.7 Hands-On: Potentials, Free Energy, and the Nernst Equation

The code below carries out every calculation in this chapter from the same handful of constants. Nothing is quoted from memory; everything is arithmetic on \\(F\\), \\(R\\), \\(T\\), two standard potentials, and two formation thermodynamics values.

```python
import numpy as np

# ---------------------------------------------------------------
# Electrode potentials, free energy, and the Nernst equation.
#
# Fixed inputs, all standard tabulated values:
#   F = 96485 C/mol            Faraday constant
#   R = 8.314 J/(mol K)        gas constant
#   T = 298.15 K               25 degrees Celsius
#   E0(Zn2+/Zn) = -0.76 V      vs SHE
#   E0(Cu2+/Cu) = +0.34 V      vs SHE
#   dG0_f(H2O, liquid) = -237 kJ/mol   Gibbs energy of formation
# Everything printed below is arithmetic on those numbers.
# ---------------------------------------------------------------
F = 96485.0  # C/mol
R = 8.314    # J/(mol K)
T = 298.15   # K

E0_ZN = -0.76  # V vs SHE
E0_CU = +0.34  # V vs SHE

# --- 1. Build the Daniell cell from two half-cell potentials -----
E0_cell = E0_CU - E0_ZN
n = 2  # Zn + Cu2+ -> Zn2+ + Cu transfers two electrons
dG0 = -n * F * E0_cell

print("Step 1: the Daniell cell from its two half-reactions")
print(f"  cathode  Cu2+ + 2e- -> Cu     E0 = {E0_CU:+.2f} V")
print(f"  anode    Zn -> Zn2+ + 2e-     E0 = {E0_ZN:+.2f} V (as a reduction)")
print(f"  E0_cell = E0(cathode) - E0(anode) = {E0_cell:.2f} V")
print(f"  dG0 = -nFE0 = {dG0:.0f} J/mol = {dG0 / 1000:.1f} kJ/mol")
print()

# --- 2. How much energy is that, per gram of zinc? ---------------
M_zn = 65.38  # g/mol
print("Step 2: the same number, expressed as usable work")
print(f"  maximum electrical work per mole of Zn = {-dG0 / 1000:.1f} kJ")
print(f"  per gram of Zn                          = {-dG0 / 1000 / M_zn:.2f} kJ/g")
print(f"  charge delivered per mole of Zn         = {n * F / 3600:.1f} A.h")
print()

# --- 3. The Nernst slope at 25 C ---------------------------------
slope_ln = R * T / F                  # volts per unit of ln(Q)
slope_log10 = R * T * np.log(10) / F  # volts per DECADE of Q

print("Step 3: where '59 mV per decade' comes from")
print(f"  RT/F            = {slope_ln:.6f} V         = {slope_ln * 1000:.2f} mV")
print(f"  RT ln(10) / F   = {slope_log10:.6f} V      = {slope_log10 * 1000:.2f} mV per decade")
print(f"  for n = 2 electrons: {slope_log10 * 1000 / 2:.2f} mV per decade")
print()

# --- 4. Nernst equation for the Daniell cell ---------------------
# E = E0 - (RT / nF) ln Q,  with Q = [Zn2+] / [Cu2+]
ratios = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4])
E = E0_cell - (R * T / (n * F)) * np.log(ratios)

print("Step 4: Daniell cell EMF vs the concentration ratio Q = [Zn2+]/[Cu2+]")
print(f"{'Q':>12} {'log10 Q':>10} {'E (V)':>10} {'E - E0 (mV)':>14}")
print("-" * 50)
for q, e_val in zip(ratios, E):
    print(f"{q:12.0e} {np.log10(q):10.1f} {e_val:10.4f} {(e_val - E0_cell) * 1000:14.1f}")
print()
print(f"  slope of the table: {(E[0] - E[-1]) / 8 * 1000:.2f} mV per decade of Q")
print()

# --- 5. A concentration cell: same metal, both sides -------------
# Two Cu electrodes, identical except for [Cu2+]. E0_cell = 0 exactly.
print("Step 5: a concentration cell (Cu | Cu2+ (c1) || Cu2+ (c2) | Cu)")
print(f"{'c2/c1':>10} {'E (V)':>10}")
print("-" * 22)
for ratio in [1.0, 10.0, 100.0, 1000.0]:
    E_conc = (R * T / (n * F)) * np.log(ratio)
    print(f"{ratio:10.0f} {E_conc:10.4f}")
print()

# --- 6. Why water splitting needs at least 1.23 V ----------------
# 2 H2O -> 2 H2 + O2 runs the formation reaction backwards.
dG0_f_H2O = -237e3  # J/mol, liquid water, standard conditions
dH0_f_H2O = -286e3  # J/mol, liquid water, standard conditions
n_water = 2         # electrons per H2O decomposed

E_rev = -dG0_f_H2O / (n_water * F)
E_tn = -dH0_f_H2O / (n_water * F)

print("Step 6: the 1.23 V floor for water electrolysis")
print(f"  dG0_f(H2O, l) = {dG0_f_H2O / 1000:.0f} kJ/mol")
print(f"  E_rev = -dG0 / (nF) = {E_rev:.4f} V     (reversible / thermodynamic)")
print(f"  dH0_f(H2O, l) = {dH0_f_H2O / 1000:.0f} kJ/mol")
print(f"  E_tn  = -dH0 / (nF) = {E_tn:.4f} V      (thermoneutral)")
print(f"  gap (the T dS term, as a voltage) = {E_tn - E_rev:.4f} V")
print()

# --- 7. The same reaction as a Gibbs energy per mole of H2 -------
print("Step 7: cross-check via the two half-reactions at pH 0")
E_her = 0.00  # 2 H+ + 2e- -> H2, defines the SHE
E_oer = 1.23  # O2 + 4 H+ + 4e- -> 2 H2O, from the same thermodynamics
print(f"  E0(H+/H2)  = {E_her:.2f} V  (definition of the SHE)")
print(f"  E0(O2/H2O) = {E_oer:.2f} V  (rounded value of E_rev above)")
print(f"  minimum cell voltage = {E_oer - E_her:.2f} V")
```

**Output:**

```
Step 1: the Daniell cell from its two half-reactions
  cathode  Cu2+ + 2e- -> Cu     E0 = +0.34 V
  anode    Zn -> Zn2+ + 2e-     E0 = -0.76 V (as a reduction)
  E0_cell = E0(cathode) - E0(anode) = 1.10 V
  dG0 = -nFE0 = -212267 J/mol = -212.3 kJ/mol

Step 2: the same number, expressed as usable work
  maximum electrical work per mole of Zn = 212.3 kJ
  per gram of Zn                          = 3.25 kJ/g
  charge delivered per mole of Zn         = 53.6 A.h

Step 3: where '59 mV per decade' comes from
  RT/F            = 0.025691 V         = 25.69 mV
  RT ln(10) / F   = 0.059156 V      = 59.16 mV per decade
  for n = 2 electrons: 29.58 mV per decade

Step 4: Daniell cell EMF vs the concentration ratio Q = [Zn2+]/[Cu2+]
           Q    log10 Q      E (V)    E - E0 (mV)
--------------------------------------------------
       1e-04       -4.0     1.2183          118.3
       1e-03       -3.0     1.1887           88.7
       1e-02       -2.0     1.1592           59.2
       1e-01       -1.0     1.1296           29.6
       1e+00        0.0     1.1000            0.0
       1e+01        1.0     1.0704          -29.6
       1e+02        2.0     1.0408          -59.2
       1e+03        3.0     1.0113          -88.7
       1e+04        4.0     0.9817         -118.3

  slope of the table: 29.58 mV per decade of Q

Step 5: a concentration cell (Cu | Cu2+ (c1) || Cu2+ (c2) | Cu)
     c2/c1      E (V)
----------------------
         1     0.0000
        10     0.0296
       100     0.0592
      1000     0.0887

Step 6: the 1.23 V floor for water electrolysis
  dG0_f(H2O, l) = -237 kJ/mol
  E_rev = -dG0 / (nF) = 1.2282 V     (reversible / thermodynamic)
  dH0_f(H2O, l) = -286 kJ/mol
  E_tn  = -dH0 / (nF) = 1.4821 V      (thermoneutral)
  gap (the T dS term, as a voltage) = 0.2539 V

Step 7: cross-check via the two half-reactions at pH 0
  E0(H+/H2)  = 0.00 V  (definition of the SHE)
  E0(O2/H2O) = 1.23 V  (rounded value of E_rev above)
  minimum cell voltage = 1.23 V
```

**Reading the result.** Five observations, in increasing order of importance.

  * **1.10 V came out of two table lookups and a subtraction.** No measurement, no fitting, no adjustable parameter. This is the payoff of the SHE convention: the modularity promised in Chapter 1 is fully realized, and the same two lines of code compute the voltage of any pairing you can find in a table.

  * **The free energy is large, and its practical forms are more informative than the voltage.** \\(-212.3\\) kJ/mol is a real chemical energy release. Divided by the mass of zinc consumed it is 3.25 kJ/g; expressed as charge it is 53.6 A·h per mole of zinc. Battery specifications are written in exactly these currencies, and the conversion between "a voltage from a table" and "watt-hours per kilogram" is nothing more than the arithmetic in Step 2.

  * **59 mV per decade is not a measured constant.** It is \\(RT \ln(10)/F\\) evaluated at 298.15 K, and the code returns 59.16 mV. Because \\(T\\) appears in the numerator, the slope is temperature-dependent — a hot cell has a steeper Nernstian response — and because \\(n\\) divides it, a two-electron reaction responds at half the rate. The Daniell table's own slope, extracted from its endpoints, is 29.58 mV per decade, exactly \\(59.16/2\\). When Chapter 3 introduces the Tafel slope, resist the temptation to conflate the two: they look similar and mean entirely different things, one thermodynamic and one kinetic.

  * **Four orders of magnitude in concentration bought about 0.24 V.** Look at Step 4's endpoints: sweeping \\(Q\\) from \\(10^{-4}\\) to \\(10^{4}\\) — a factor of one hundred million in the concentration ratio — moved the cell from 1.2183 V to 0.9817 V. That is under a quarter of a volt for a change no laboratory could actually sustain. **The logarithm is a brutal compressor**, and this is the quantitative reason that concentration is a fine-tuning knob and chemistry is the coarse one.

  * **The 1.23 V floor is thermodynamics, not engineering.** \\(237\\) kJ/mol divided by \\(2F\\) is 1.2282 V, and it agrees with the tabulated oxygen potential to the precision of the input data — two independent routes to the same number. The thermoneutral 1.4821 V sits 0.2539 V above it, that gap being the entropy of turning a liquid into two gases, expressed as a voltage. Every real electrolyzer operates above 1.48 V, and the amount by which it exceeds that is the entire subject of Chapters 3 and 5.

A worthwhile modification: change `T` in the code from 298.15 K to 353.15 K (80 °C, a common electrolyzer operating temperature) and rerun Steps 3 and 6. The Nernst slope grows, and if you also supply the temperature-corrected formation values, the reversible voltage falls. That is the thermodynamic reason high-temperature electrolysis is attractive, and Chapter 5 returns to it.

### 🎯 Exercise Problems

  1. **Cells from the table.** Using only the potential table of Section 2.2, compute the standard cell voltage and identify the anode for each pairing: (a) silver against zinc, (b) copper against iron, (c) lithium against copper. For each, write the cell diagram in the notation of Section 2.4 and confirm your voltage comes out positive.

  2. **The sign-flip trap.** A student computes the Daniell cell by first reversing the zinc half-reaction to \\(\mathrm{Zn} \to \mathrm{Zn^{2+}} + 2e^-\\), changing its potential to \\(+0.76\\) V, and then subtracting it from the copper value to get \\(-0.42\\) V. Identify precisely which step is wrong, state the correct rule in one sentence, and describe a physical observation that would immediately tell the student their answer is wrong.

  3. **Free energy and capacity.** For pairing (c) of Exercise 1, compute \\(\Delta G^{\circ}\\) and the maximum electrical work per gram of lithium consumed (molar mass 6.94 g/mol). Compare with the 3.25 kJ/g the code obtained for zinc, and explain in terms of both the voltage and the molar mass why lithium is the more attractive battery anode.

  4. **Nernst by hand.** A Daniell cell is built with \\([\mathrm{Zn^{2+}}] = 1.0\\) M and \\([\mathrm{Cu^{2+}}] = 0.001\\) M. Predict the cell voltage using the 29.58 mV-per-decade slope from the code, then verify against the Step 4 table. Does the cell voltage go up or down, and does that direction match Le Chatelier's principle?

  5. **The dead battery.** Using \\(\ln K = nFE^{\circ}/RT\\), compute the equilibrium constant of the Daniell reaction at 25 °C. Then state the ratio \\([\mathrm{Zn^{2+}}]/[\mathrm{Cu^{2+}}]\\) at which the cell voltage would finally reach zero, and comment on whether a real cell would ever reach it.

  6. **Corrosion as a concentration cell.** A steel pipe is buried so that part of it lies in well-aerated sandy soil and part in oxygen-poor clay. Using the concentration-cell reasoning of Section 2.5, argue which region corrodes and why. Then explain, in terms of the potential table, why attaching a block of zinc to the pipe protects it.

  7. **The floor and the ceiling.** An electrolyzer operates at 1.85 V. Compute its voltage efficiency relative to 1.23 V and relative to 1.48 V, state which of the two is the meaningful figure for reporting energy efficiency, and explain what happens to the difference between the applied voltage and 1.48 V.

## Summary

A single electrode's potential cannot be measured, because every circuit contains at least two interfaces. Electrochemistry solves this the way physics always solves an inaccessible absolute: it defines a reference. The **standard hydrogen electrode is assigned exactly 0.000 V**, and every tabulated potential is a measured difference against it. Only differences are physical; the individual numbers are bookkeeping that cancels out of every real prediction.

Arranged by value, those numbers form the **potential series**, which is the familiar **activity series of metals** with numbers attached and extended beyond metals. Two rules extract predictions: a species higher in the table oxidizes one lower in it, and the cell voltage of any pairing is \\(E^{\circ}_{\text{cathode}} - E^{\circ}_{\text{anode}}\\) with both values read as reductions. The table is a statement about what is *allowed*, and brine electrolysis producing chlorine rather than oxygen — despite chlorine's higher potential at \\(+1.36\\) V against oxygen's \\(+1.23\\) V — is the standing reminder that thermodynamic permission is not a prediction of outcome.

The bridge between the electrical and chemical descriptions is \\(\Delta G = -nFE\\), derived from the identification of free energy with maximum non-expansion work. It explains the sign convention (spontaneous means positive voltage), it reconciles voltage being intensive with free energy being extensive, and its word "maximum" carries the warning that every real, current-carrying cell falls short. Applied to the **Daniell cell**, two table lookups give \\((+0.34) - (-0.76) = \mathbf{1.10\ V}\\), and the code converted that to \\(\Delta G^{\circ} = \mathbf{-212.3}\\) **kJ/mol**, equivalently **3.25 kJ/g** of zinc and **53.6 A·h** per mole.

Away from standard conditions the **Nernst equation** \\(E = E^{\circ} - (RT/nF)\ln Q\\) governs the voltage, and it reaches zero exactly at equilibrium — a dead battery is a battery at equilibrium, not an empty one. In base-10 form its prefactor at 25 °C is \\(RT\ln(10)/F = \mathbf{59.16\ mV}\\), the famous **59 mV per decade**, halved to **29.58 mV per decade** for the two-electron Daniell cell as the code's own table confirms. The logarithm makes concentration a weak lever: eight decades of concentration ratio moved the cell only from 1.2183 V to 0.9817 V. The same equation predicts the **concentration cell**, which generates **29.6 mV per tenfold gradient** from no chemical difference at all — a stored free energy that powers biological ion gradients and drives localized corrosion.

Finally, the number that anchors the rest of the series. Water's Gibbs energy of formation, \\(-237\\) kJ/mol, divided by \\(2F\\), gives \\(\mathbf{1.2282\ V}\\) — the **1.23 V** floor for water splitting, confirmed independently by the tabulated oxygen potential. It cannot be lowered by any catalyst. Using the enthalpy \\(-286\\) kJ/mol instead gives the **thermoneutral voltage, 1.4821 V**, about **1.48 V**, above which an electrolyzer produces net heat; the **0.2539 V** gap between them is the entropy of gas formation written as a voltage.

Everything so far has assumed reversibility — infinitely slow operation, zero current, perfect efficiency. Chapter 3 removes that assumption, and the removal is expensive. We will define **overpotential** as the excess voltage a real reaction demands, break it into activation, concentration, and resistive contributions, meet the **exchange current density** that measures how easily a reaction proceeds at equilibrium, build the **Butler–Volmer equation**, and extract the **Tafel slope** from it. The central lesson waiting there is the one this chapter has been setting up by omission: a catalyst never changes any number computed in this chapter. It changes only how much voltage you must pay above them.

[← Chapter 1: Why Electrochemistry?](<chapter-1.html>) [Chapter 3: Kinetics — Overpotential and Tafel Analysis →](<chapter-3.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
