---
title: "Chapter 5: Applications — From Electrolysis to Batteries"
chapter_title: "Chapter 5: Applications — From Electrolysis to Batteries"
subtitle: "Where the Thermodynamics of Chapter 2 and the Kinetics of Chapter 3 Meet a Real Cell, a Real Current, and a Real Electricity Bill"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/sdFvXHF-ToQ"
    title="Electrochemistry Ch.5: Applications — From Electrolysis to Batteries"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/MS/electrochemistry-introduction/chapter-5.html>) | Last sync: 2026-08-20

[Materials Science Dojo](<../index.html>) > [Electrochemistry Introduction](<index.html>) > Chapter 5

Four chapters of machinery are now on the table. Chapter 1 established that all of this is bookkeeping about electrons changing hands, and that the same two electrodes can be read in two directions — a cell that gives you electricity, or a cell that consumes it. Chapter 2 gave the thermodynamics: an electrode potential is a free energy per electron, \\(\Delta G = -nFE\\) converts between the two languages, and the difference of two electrode potentials is the voltage a cell will show when nothing is happening. Chapter 3 insisted that "nothing is happening" is exactly the problem, and introduced overpotential as the price of making something happen. Chapter 4 put us at the interface where that price is actually paid, and showed how a three-electrode measurement isolates it.

This chapter spends all of it. We take three technologies — **water electrolysis**, **CO₂ electrolysis**, and **batteries** — and read each one entirely in the vocabulary already built. There is no new theory here, which is precisely the point. If the previous four chapters did their job, then a hydrogen plant, a carbon-recycling reactor, and the cell in your laptop are all the same diagram with different labels, and the differences between them are differences of number and of engineering, not of principle.

The organizing quantity for the whole chapter is a single equation you can write on the back of an envelope:

\\[ E_{\text{cell}} = E_{\text{rev}} + \eta_{\text{anode}} + \left|\eta_{\text{cathode}}\right| + iR_{\text{cell}} \\]

The first term is thermodynamics and you cannot negotiate with it. The rest is everything the field actually works on. Section 5.5 turns that equation into code and makes the four terms compete for space in a real voltage budget.

## 5.1 Water Electrolysis: The 1.23 V Floor and Everything Above It

Water electrolysis is the cleanest example in electrochemistry of the gap between what thermodynamics permits and what kinetics charges you for. The overall reaction is as simple as chemistry gets:

\\[ 2\text{H}_2\text{O} \;\longrightarrow\; 2\text{H}_2 + \text{O}_2 \\]

Split it into half-reactions at the two electrodes. In acid, the cathode runs the **hydrogen evolution reaction (HER)** and the anode runs the **oxygen evolution reaction (OER)**:

\\[ \text{HER (cathode):}\quad 2\text{H}^+ + 2e^- \;\longrightarrow\; \text{H}_2 \\]

\\[ \text{OER (anode):}\quad 2\text{H}_2\text{O} \;\longrightarrow\; \text{O}_2 + 4\text{H}^+ + 4e^- \\]

In alkaline conditions the same two transformations are written with hydroxide instead of protons — HER becomes \\(2\text{H}_2\text{O} + 2e^- \rightarrow \text{H}_2 + 2\text{OH}^-\\) and OER becomes \\(4\text{OH}^- \rightarrow \text{O}_2 + 2\text{H}_2\text{O} + 4e^-\\) — but the electron counts and the overall reaction are identical. The convention changes; the chemistry does not.

Balance the electrons and the stoichiometry falls out immediately: OER releases four electrons per O₂, HER consumes two per H₂, so the cell produces **two moles of hydrogen for every mole of oxygen**. That is not a measured ratio. It is forced by the electron bookkeeping, and it is the same Faraday's-law argument Chapter 1 used to convert charge into mass.

### 📚 The Two Voltages You Have to Keep Apart

Two numbers get quoted for water splitting and confusing them is the most common error in the field.

  * **The reversible voltage, 1.23 V.** This comes from \\(\Delta G^\circ = -nFE^\circ\\) applied to the water-splitting reaction at 25 °C. It is the voltage at which the reaction is at equilibrium — the point where, in Chapter 3's language, forward and reverse rates balance and the *net* current is zero. Below it, electrolysis is thermodynamically forbidden. At it, electrolysis proceeds infinitely slowly, which is to say not at all.
  * **The thermoneutral voltage, about 1.48 V.** This comes from the *enthalpy* rather than the free energy. Splitting water is endothermic: part of the energy the products carry away was supposed to come from the surroundings as heat. A cell operating exactly at the thermoneutral voltage neither absorbs nor releases net heat. Below it, a working cell would need heat from outside; above it, the excess appears as waste heat inside the stack.

Neither number is a target you can hit. Both are landmarks that tell you where you stand. A cell running at 1.23 V produces nothing; a cell running above 1.48 V is warming itself up with the difference.

The real cell voltage is the sum written at the top of this chapter, and every term above the first is a loss:

  * \\(\eta_{\text{OER}}\\) — the activation overpotential at the anode, obeying Tafel behaviour over its useful range, as Chapter 3 derived.
  * \\(\eta_{\text{HER}}\\) — the same thing at the cathode, with its own exchange current density and its own Tafel slope.
  * \\(iR_{\text{cell}}\\) — the ohmic drop across electrolyte, membrane, contacts, and current collectors, exactly the quantity Chapter 4 taught you to correct for in a measurement and which here you cannot correct away, because in a working device the energy really is lost as heat.

Chapter 3's concentration overpotential belongs here too, appearing at high rates when reactant supply or bubble removal cannot keep up with the current. In a real electrolyser it is largely a mass-transport and gas-management engineering problem, and Section 5.5 leaves it out of the model rather than pretending to a number for it.

### 📚 Why OER Is the Bottleneck

Compare the two half-reactions as *processes* rather than as equations, and the asymmetry is stark.

HER moves **two electrons**. Its intermediates are few — hydrogen adsorbed on the surface, then combined and released — and on a good catalyst the pathway is short and forgiving. It is one of the most facile electrochemical reactions known.

OER moves **four electrons and four protons**, and it has to form an **oxygen–oxygen bond** that did not exist in either water molecule. In the standard mechanistic picture the surface passes through a sequence of bound intermediates — a hydroxyl, an oxo, and a hydroperoxo species — with one electron and one proton removed at each step. Four sequential charge transfers, three intermediates, and a new bond, all at one site, all driven by the same electrode potential.

That last clause is the source of the deepest difficulty. A single knob — the potential — sets the driving force for **all four** steps simultaneously. You cannot tune step 3 without also tuning step 1. And because the intermediates bind to the surface through the same oxygen atom, their binding energies tend to move together when you change the catalyst: strengthen the binding of one and you strengthen the others. This correlation is why decades of catalyst development have not driven the OER overpotential to zero, and why the search for better anodes is a genuinely hard optimization rather than a matter of trying more materials.

The computational side of this story — how binding energies of those intermediates are calculated, why they are correlated, and how the correlation puts a floor under the overpotential of an entire class of materials — is exactly the subject of [Computational Chemistry of OER](<../../MI/oer-computational-chemistry/index.html>). That series picks up where this section stops. If Section 5.5's budget makes you want to know *why* the anode term is so stubborn, that is where to go next.

The practical consequence is a rule of thumb worth carrying: **in a well-built water electrolyser, the anode overpotential is usually the single largest controllable term in the voltage budget.** Section 5.5 makes that concrete with an illustrative model, and shows what improving each term is worth.

## 5.2 CO₂ Electrolysis: The Same Cell, a Harder Question

Now change the reactant. Instead of reducing protons to hydrogen at the cathode, feed the cell carbon dioxide and reduce *that*. The reaction is called the **CO₂ reduction reaction (CO₂RR)**, and it is the electrochemical route to a carbon-recycling society: take CO₂ that would otherwise be emitted, drive it with renewable electricity, and get back a molecule you can use as a fuel or a chemical feedstock.

The anode does not change. It still runs OER, still needs its four electrons, still dominates the anodic side of the budget. Everything interesting happens at the cathode.

### 📚 A Ladder of Products, Ordered by Electrons

CO₂ reduction is not one reaction but a family, indexed by how many electrons and protons you deliver:

  * **Two electrons** gives **carbon monoxide** or **formate** — the shallowest rung, and the one that is most controllable. CO is directly useful as a feedstock for established industrial chemistry.
  * **Six to eight electrons** gives **methanol** or **methane** — fully reduced single-carbon molecules, usable as fuels.
  * **Twelve or more electrons**, together with **carbon–carbon bond formation**, gives **ethylene, ethanol, and other multi-carbon products** — the most valuable targets and by far the hardest, because now the catalyst must not only deliver a dozen charge transfers to one site but also join two carbon atoms while doing it.

Read that list against Section 5.1 and the difficulty is already visible: OER was hard with four electrons and one new bond, and CO₂RR to a multi-carbon product asks for three times the electrons plus a carbon–carbon bond, at one site, driven by one potential.

### 📚 The Selectivity Problem, Stated Plainly

The multi-electron difficulty is real but it is not the defining problem. The defining problem is **competition**.

CO₂ reduction happens in water. Water contains protons. And protons reduce to hydrogen — the HER of Section 5.1, one of the most facile reactions in electrochemistry — at potentials that sit **close to those of the CO₂ reduction pathways**. The equilibrium potentials of the various CO₂RR products cluster near 0 V on the reversible-hydrogen-electrode scale, which is to say near HER's own equilibrium potential by construction. Sources differ on the precise values, and this chapter deliberately does not tabulate them, because the qualitative statement is the one that matters and it is not in dispute: **thermodynamics does not separate these reactions.**

So when you apply a potential negative enough to drive CO₂ reduction, you are simultaneously applying a potential more than negative enough to drive hydrogen evolution. Two reactions compete for the same electrons at the same surface. If HER wins, your renewable electricity has produced hydrogen from water while the CO₂ flowed through untouched.

This reframes the entire design problem. In water electrolysis you ask a catalyst to be **fast**. In CO₂ electrolysis you ask it to be **fast and discriminating** — and discrimination is the harder request, because it cannot come from thermodynamics. If the equilibrium potentials do not separate the products, then only the *kinetics* can. Selectivity in CO₂RR is a purely kinetic phenomenon: which pathway the surface stabilizes, which intermediate it binds, which transition state it lowers. It is Chapter 3's message in its strongest form. Thermodynamics says what is allowed; here it allows everything, and the catalyst decides what actually happens.

Hence the metric the field lives by: **Faradaic efficiency**, the fraction of the charge you passed that ended up in the product you wanted. It is a pure electron-bookkeeping quantity, defined exactly the way Chapter 1 defined charge-to-product conversion, and it makes the competition auditable. A cell with high current density and poor Faradaic efficiency is an expensive hydrogen generator. And there is a third axis beyond efficiency and rate: **stability**, whether the catalyst and the electrolyte still behave the same way after hundreds of hours. Rate, selectivity, and stability form a trade-off surface that no material has yet cleared on all three axes at industrial scale.

> **A dedicated series is coming.** CO₂ electrolysis deserves far more than one section — gas-diffusion electrode architectures, the carbonate problem, local pH effects, product separation, and the techno-economics that decide whether any of it makes sense. A follow-on series on CO₂ electrolysis and carbon recycling is planned, and it will take this chapter's five paragraphs as its starting assumption rather than its content. What you should take from here is the shape of the problem: **same cell, same OER anode, same overpotential vocabulary, plus one new and dominating requirement — selectivity against a competitor that thermodynamics refuses to eliminate.**

## 5.3 Batteries, Re-read Through Chapters 2 and 3

Batteries feel like a different subject. They are not. A battery is the same two-electrode cell run in both directions, and everything you know about electrolysers transfers with the signs flipped.

**Discharge is a galvanic process.** The cell reaction is spontaneous, \\(\Delta G < 0\\), and the cell pushes current through an external load. Chemistry does work on the circuit.

**Charge is an electrolytic process.** You force the reaction backwards with an external power supply, exactly as you force water to split. The circuit does work on the chemistry.

This is Chapter 1's galvanic/electrolytic distinction, and it is why the anode/cathode naming trips people up in batteries specifically: the definitions are tied to *oxidation and reduction*, not to physical terminals, so the electrode that is the anode during discharge becomes the cathode during charge. The physical electrode did not move. The direction of the reaction did. Practitioners usually sidestep the confusion by naming battery electrodes after their discharge roles and keeping that convention throughout — a pragmatic choice, and one worth stating explicitly whenever you write it down.

### 📚 Where the Cell Voltage Comes From

Chapter 2 said a cell's equilibrium voltage is the difference between two electrode potentials. That statement is the entire design principle of a battery: **pick two redox couples far apart on the potential scale**. Chapter 2 built the Daniell cell this way, from a zinc couple at −0.76 V and a copper couple at +0.34 V, giving 1.10 V. A modern battery replaces those couples with materials chosen for many other reasons at once — how much charge they store per unit mass, how reversibly they cycle, whether they survive contact with the electrolyte — but the voltage still comes from the same subtraction, and from nothing else.

The Nernst equation from Chapter 2 then explains something every phone user has seen: **the voltage sags as the battery empties.** As discharge proceeds the composition of both electrodes changes, and the Nernst relation says the electrode potentials shift with those activities. The open-circuit voltage of a partly discharged cell is genuinely different from that of a full one. This is thermodynamics, not degradation.

In lithium-ion cells the chemistry has a distinctive shape worth naming qualitatively. Both electrodes are **hosts**: layered or framework materials into which lithium ions insert and from which they are extracted, with the electrode's oxidation state changing to compensate. Lithium ions shuttle back and forth through the electrolyte between the two hosts while the compensating electrons travel the external circuit — the reason the design is often called a **rocking-chair** cell. This chapter quotes no electrode potentials, capacities, or cell voltages for any specific chemistry; those numbers depend on the exact material, the state of charge, and the measurement convention, and getting them subtly wrong is the classic way an introduction misleads. The *structure* of the explanation is what generalizes, and the structure is: two hosts at different potentials, ions inside, electrons outside.

### 📚 Overpotential You Can See on a Graph

Here is where Chapter 3 pays off most vividly. Charge a cell and then discharge it at the same rate, and plot voltage against capacity. The two curves do not lie on top of each other. **The charge curve sits above the discharge curve**, and the vertical distance between them is the **voltage gap**, or hysteresis.

That gap is not a mystery. It is the same three losses, appearing twice with opposite signs:

  * **Activation overpotential** at both electrodes, from the finite rate of charge transfer.
  * **Concentration overpotential**, from ion transport in the electrolyte and diffusion inside the host particles — usually the dominant term at high rates, and the reason fast charging is hard.
  * **Ohmic drop** \\(iR\\) across the electrolyte, separator, current collectors, and contacts.

During discharge these losses **subtract** from the thermodynamic voltage: you get less out than thermodynamics promised. During charge they **add**: you must put more in. The energy corresponding to the gap does not go anywhere useful — it becomes heat, which is why batteries warm up under heavy use and why thermal management is a first-class engineering problem rather than an afterthought.

Two consequences follow directly, and they explain most of what a battery datasheet is trying to tell you:

  * **Round-trip efficiency is a kinetics number, not a thermodynamics number.** The ratio of energy out to energy in is set by the size of that gap, and the gap is overpotential plus \\(iR\\). Improve the kinetics and you improve the efficiency, without touching the cell chemistry's voltage at all.
  * **The gap grows with current.** Every term in it does: activation overpotential grows logarithmically with current in the Tafel regime, ohmic drop grows linearly, and concentration overpotential grows faster still as transport hits its limits. This is why a cell delivers less usable energy when discharged quickly than when discharged slowly, and why "capacity" is only meaningful when quoted with a rate.

Notice that the identical sentence describes an electrolyser: the voltage you must apply exceeds the thermodynamic minimum by the overpotentials plus \\(iR\\), and the excess becomes heat. **An electrolyser is a battery you only ever charge.** The budget in Section 5.5 is written for water splitting, but the accounting is the same accounting a battery engineer does when explaining where the round-trip losses went.

## 5.4 Where the Field Is Going

Three directions are worth naming, because each is a direct extension of something in this series and each is where a newcomer can actually contribute.

**Electrocatalyst design is becoming a data problem.** The historical method — synthesize a material, measure a Tafel plot, iterate — is limited by how many materials a laboratory can make. The modern approach adds computed descriptors: calculate how strongly a surface binds the reaction intermediates, use that to predict activity, and screen far more candidates than could ever be synthesized. That is the [Computational Chemistry of OER](<../../MI/oer-computational-chemistry/index.html>) programme applied to a specific reaction, and the broader data-driven layer above it — how to build models from heterogeneous catalyst datasets, and where those models mislead you — is the subject of [MI Applications to Catalyst Design](<../../MI/catalyst-mi-application/index.html>). Both are honest about the limitation that matters: a computed descriptor is a prediction, and predictions still have to be verified with the three-electrode measurement of Chapter 4.

**Operando measurement is closing the gap between the model and the surface.** Every measurement in Chapter 4 was electrical — current, voltage, time — and electrical measurements tell you the rate but not the identity of what is on the surface. The growing set of **operando** techniques, spectroscopy and diffraction performed *while* the cell runs at potential and under current, asks what the surface actually looks like during the reaction. The answers have repeatedly been surprising, because catalysts restructure, dissolve and redeposit, and change oxidation state under working conditions. A catalyst characterized only before and after a run is one you have described in two states it never occupied while working.

**Scale-up is its own science, not an engineering afterthought.** A laboratory electrode is small, flat, well-stirred, and at a uniform potential. An industrial stack is none of those: current distributes unevenly across large electrodes, gas bubbles block active area and add resistance, and materials that survived a few hundred hours must survive years. Section 5.5's budget shows why this is not a detail — a tenth of a volt of extra ohmic drop is a permanent tax on every kilogram of product the plant will ever make.

## 5.5 Hands-On: Building a Cell-Voltage Budget

Everything in Section 5.1 can be turned into arithmetic. We will build the voltage budget of a water electrolyser term by term, see which term dominates, watch all of them get worse as we push more current, and price three different improvements against each other.

**Read this warning before reading the numbers.** Exactly one quantity in the code is an established physical value: \\(E_{\text{rev}} = 1.23\\) V, together with the constants \\(F\\) and \\(R\\), and the thermoneutral voltage of about 1.48 V used for comparison. **Every kinetic parameter below — the exchange current densities, the Tafel slopes, and the cell resistance — is ILLUSTRATIVE.** They were chosen to be plausible in order of magnitude and to make the structure of the budget visible. They are not measurements of any particular catalyst, membrane, or cell, and no number in the output should ever be quoted as the performance of a real device. What is real is the *shape* of the result: which terms are large, how they scale with current, and which improvements are worth what.

```python
import numpy as np

# ---------------------------------------------------------------
# A cell-voltage budget for water electrolysis.
#
#   E_cell(i) = E_rev + eta_OER(i) + eta_HER(i) + i * R_cell
#
# E_rev = 1.23 V is the established thermodynamic value for water
# splitting at 25 C. EVERYTHING ELSE below is an ILLUSTRATIVE model
# parameter, chosen to be plausible in order of magnitude. They are
# NOT measured values of any specific catalyst or any specific cell.
# ---------------------------------------------------------------
F = 96485.0          # C/mol, Faraday constant
R_GAS = 8.314        # J/(mol K)
T = 298.15           # K
E_REV = 1.23         # V, reversible water-splitting voltage at 25 C
E_TN = 1.48          # V, thermoneutral voltage (approx)

# --- 0. The Tafel slope anchor from Chapter 3 -------------------
# b = 2.303 R T / (alpha F). With alpha = 0.5 this is the textbook
# 118 mV/decade. This one IS derived, not assumed.
alpha_ref = 0.5
b_ref = 2.303 * R_GAS * T / (alpha_ref * F)
print("Step 0: Tafel slope anchor (derived, not assumed)")
print(f"  b = 2.303 R T / (alpha F), alpha = {alpha_ref}")
print(f"  b = {b_ref * 1000:.1f} mV/decade at T = {T:.2f} K")
print()

# --- 1. ILLUSTRATIVE kinetic parameters -------------------------
# Units: exchange current densities in A/cm^2, Tafel slopes in V/dec,
# cell resistance in ohm cm^2.
I0_OER = 1.0e-7      # ILLUSTRATIVE: sluggish 4-electron anode
B_OER = 0.060        # ILLUSTRATIVE: 60 mV/dec
I0_HER = 1.0e-3      # ILLUSTRATIVE: fast 2-electron cathode
B_HER = 0.040        # ILLUSTRATIVE: 40 mV/dec
R_CELL = 0.20        # ILLUSTRATIVE: 0.20 ohm cm^2 (membrane + contacts)


def tafel_overpotential(i, i0, b):
    """Magnitude of the activation overpotential, Tafel form."""
    return b * np.log10(i / i0)


# --- 2. The budget at one operating point -----------------------
I_OP = 1.0           # A/cm^2  (= 1000 mA/cm^2), a demanding duty
eta_o = tafel_overpotential(I_OP, I0_OER, B_OER)
eta_h = tafel_overpotential(I_OP, I0_HER, B_HER)
v_ir = I_OP * R_CELL
e_cell = E_REV + eta_o + eta_h + v_ir

print(f"Step 1: voltage budget at i = {I_OP:.2f} A/cm^2 (ILLUSTRATIVE parameters)")
print(f"{'term':<28}{'volts':>10}{'share':>10}")
print("-" * 48)
for name, val in [
    ("E_rev (thermodynamics)", E_REV),
    ("eta_OER (anode kinetics)", eta_o),
    ("eta_HER (cathode kinetics)", eta_h),
    ("i R_cell (ohmic)", v_ir),
]:
    print(f"{name:<28}{val:>10.3f}{val / e_cell * 100:>9.1f}%")
print("-" * 48)
print(f"{'E_cell (total)':<28}{e_cell:>10.3f}{100.0:>9.1f}%")
print()

# --- 3. Efficiency ----------------------------------------------
eff_rev = E_REV / e_cell
eff_tn = E_TN / e_cell
print("Step 2: voltage efficiency")
print(f"  vs E_rev = 1.23 V : {eff_rev * 100:.1f}%")
print(f"  vs E_tn  = 1.48 V : {eff_tn * 100:.1f}%")
print(f"  overhead above thermodynamics: {e_cell - E_REV:.3f} V "
      f"({(e_cell - E_REV) / E_REV * 100:.1f}% of 1.23 V)")
print()

# --- 4. Sweep the current density -------------------------------
currents = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 2.0])
print("Step 3: how the budget changes with current density")
print(f"{'i (A/cm^2)':>11}{'eta_OER':>10}{'eta_HER':>10}{'iR':>8}"
      f"{'E_cell':>9}{'eff(1.23)':>11}")
print("-" * 59)
for i in currents:
    eo = tafel_overpotential(i, I0_OER, B_OER)
    eh = tafel_overpotential(i, I0_HER, B_HER)
    ir = i * R_CELL
    ec = E_REV + eo + eh + ir
    print(f"{i:>11.2f}{eo:>10.3f}{eh:>10.3f}{ir:>8.3f}{ec:>9.3f}"
          f"{E_REV / ec * 100:>10.1f}%")
print()

# --- 5. Which knob is worth turning? ----------------------------
# Three separate one-at-a-time interventions at i = 1 A/cm^2.
print(f"Step 4: value of three interventions at i = {I_OP:.2f} A/cm^2")
base = e_cell


def budget(i0_oer=I0_OER, b_oer=B_OER, i0_her=I0_HER, b_her=B_HER,
           r=R_CELL, i=I_OP):
    return (E_REV + tafel_overpotential(i, i0_oer, b_oer)
            + tafel_overpotential(i, i0_her, b_her) + i * r)


scenarios = [
    ("baseline", budget()),
    ("OER i0 x100 (better anode catalyst)", budget(i0_oer=I0_OER * 100)),
    ("HER i0 x100 (better cathode catalyst)", budget(i0_her=I0_HER * 100)),
    ("R_cell halved (thinner membrane)", budget(r=R_CELL / 2)),
]
for name, val in scenarios:
    print(f"  {name:<40}{val:>7.3f} V  "
          f"(saves {base - val:+.3f} V, eff {E_REV / val * 100:.1f}%)")
print()

# --- 6. What the extra volts cost, per kilogram of hydrogen -----
# Two electrons per H2 molecule. Energy per mole = 2 F E_cell joules.
M_H2 = 2.016e-3      # kg/mol, molar mass of H2
JOULES_PER_KWH = 3.6e6


def kwh_per_kg(e):
    return (2 * F * e / JOULES_PER_KWH) / M_H2


print("Step 5: electrical energy per kilogram of H2 (2 F E_cell per mole)")
for label, e in [("at E_rev = 1.23 V (thermodynamic floor)", E_REV),
                 ("at E_tn = 1.48 V (thermoneutral)", E_TN),
                 (f"at E_cell = {base:.3f} V (this budget)", base)]:
    print(f"  {label:<42}{kwh_per_kg(e):>7.1f} kWh/kg")
print(f"  overhead of this budget over the floor: "
      f"{kwh_per_kg(base) - kwh_per_kg(E_REV):.1f} kWh/kg")
```

**Output:**

```
Step 0: Tafel slope anchor (derived, not assumed)
  b = 2.303 R T / (alpha F), alpha = 0.5
  b = 118.3 mV/decade at T = 298.15 K

Step 1: voltage budget at i = 1.00 A/cm^2 (ILLUSTRATIVE parameters)
term                             volts     share
------------------------------------------------
E_rev (thermodynamics)           1.230     62.4%
eta_OER (anode kinetics)         0.420     21.3%
eta_HER (cathode kinetics)       0.120      6.1%
i R_cell (ohmic)                 0.200     10.2%
------------------------------------------------
E_cell (total)                   1.970    100.0%

Step 2: voltage efficiency
  vs E_rev = 1.23 V : 62.4%
  vs E_tn  = 1.48 V : 75.1%
  overhead above thermodynamics: 0.740 V (60.2% of 1.23 V)

Step 3: how the budget changes with current density
 i (A/cm^2)   eta_OER   eta_HER      iR   E_cell  eff(1.23)
-----------------------------------------------------------
       0.01     0.300     0.040   0.002    1.572      78.2%
       0.05     0.342     0.068   0.010    1.650      74.6%
       0.10     0.360     0.080   0.020    1.690      72.8%
       0.50     0.402     0.108   0.100    1.840      66.9%
       1.00     0.420     0.120   0.200    1.970      62.4%
       2.00     0.438     0.132   0.400    2.200      55.9%

Step 4: value of three interventions at i = 1.00 A/cm^2
  baseline                                  1.970 V  (saves +0.000 V, eff 62.4%)
  OER i0 x100 (better anode catalyst)       1.850 V  (saves +0.120 V, eff 66.5%)
  HER i0 x100 (better cathode catalyst)     1.890 V  (saves +0.080 V, eff 65.1%)
  R_cell halved (thinner membrane)          1.870 V  (saves +0.100 V, eff 65.8%)

Step 5: electrical energy per kilogram of H2 (2 F E_cell per mole)
  at E_rev = 1.23 V (thermodynamic floor)      32.7 kWh/kg
  at E_tn = 1.48 V (thermoneutral)             39.4 kWh/kg
  at E_cell = 1.970 V (this budget)            52.4 kWh/kg
  overhead of this budget over the floor: 19.7 kWh/kg
```

**Reading the result.** Five observations, in increasing order of importance.

  * **The Tafel anchor is the only kinetic number here that is derived.** Step 0 recomputes Chapter 3's result: with a transfer coefficient of 0.5 at 25 °C, \\(b = 2.303RT/(\alpha F) = 118.3\\) mV per decade. Everything after Step 0 uses *illustrative* slopes instead, because real slopes vary with mechanism and material; the anchor is there to remind you what a slope means and roughly how large one is.

  * **The anode is the largest loss.** In this budget the OER overpotential contributes 0.420 V against the HER's 0.120 V — more than three times as much, from the same current, in the same cell. That ratio is not an accident of the parameters chosen; it follows from the exchange current densities differing by four orders of magnitude, which is the modelling shorthand for "four electrons and a new O–O bond versus two electrons and a hydrogen molecule". Section 5.1's claim that OER is the bottleneck is visible directly in the ledger.

  * **Thermodynamics is not the majority of your electricity bill for long.** At 1 A/cm² the reversible voltage is 62.4% of the applied voltage, and the other 37.6% is loss. Push to 2 A/cm² and the total reaches 2.200 V, with voltage efficiency down to 55.9%. Every industrial electrolyser lives on this trade-off: higher current density means more hydrogen per unit of capital equipment, and a worse efficiency on every kilogram of it. There is no setting at which both are optimal, which is why the choice is an economic one rather than a scientific one.

  * **The two loss types scale differently, and that changes which one matters.** Activation overpotential grows with the *logarithm* of current — from 0.01 to 2 A/cm², the OER term rises only from 0.300 V to 0.438 V. The ohmic term grows *linearly*, from 0.002 V to 0.400 V, a factor of two hundred. At low current the catalyst is essentially the whole problem; at high current the ohmic drop has caught up with it and is still climbing. This is the quantitative reason the field's attention shifts from catalysis to cell and stack engineering as devices scale toward industrial current densities.

  * **Improvements are worth less than they look, and that is the Tafel slope's doing.** Making the anode catalyst a *hundred times* more active saves 0.120 V. Halving the cell resistance — a much more modest-sounding change — saves 0.100 V. The reason is in the logarithm: two decades of improvement in exchange current density buys you exactly two Tafel slopes of overpotential, and with a 60 mV/dec slope that is 0.120 V, no matter how impressive the hundredfold sounds in an abstract. **Catalysis pays logarithmically; engineering pays linearly.** Anyone budgeting a research programme should know which of the two they are buying.

The last step prices the whole thing in physical units. Two electrons per hydrogen molecule and \\(F = 96485\\) C/mol give the energy per mole directly, and the molar mass converts to a kilogram basis: the thermodynamic floor is 32.7 kWh per kilogram of hydrogen, and this illustrative cell needs 52.4 kWh — an overhead of 19.7 kWh/kg, every kilogram, forever. That is what a volt of overpotential costs when you multiply it by an industrial production rate, and it is why a hundred millivolts is a serious result rather than a rounding error.

**Try this.** Set `B_OER = 0.120` — roughly the 118 mV/dec anchor instead of the illustrative 60 — and rerun. The anode term and the whole budget worsen, and, more interestingly, the *value* of the hundredfold catalyst improvement in Step 4 doubles. Improving a catalyst is worth twice as much when its Tafel slope is twice as steep. Then set `R_CELL = 0.05` and watch which term the 2 A/cm² row is dominated by. Two edits, and you have reproduced the two main arguments an electrolyser engineer has with a catalyst chemist.

### 🎯 Exercise Problems

  1. **The two voltages.** Explain, in your own words and without looking back, why a water electrolyser operating at exactly 1.23 V produces no hydrogen, and what physically changes between 1.23 V and the thermoneutral 1.48 V. Then state what an operator should conclude if a real stack is running below 1.48 V.

  2. **Electron counting.** Using only the two half-reactions in Section 5.1, derive the 2:1 molar ratio of H₂ to O₂ without consulting the overall equation. Then explain why the *same* current flows through both electrodes despite them producing different numbers of moles.

  3. **Why selectivity cannot come from thermodynamics.** Section 5.2 states that the equilibrium potentials of the CO₂ reduction products cluster near those of hydrogen evolution. Explain what would have to be true instead for selectivity to be a thermodynamic problem, and state precisely which quantity from Chapter 3 the catalyst must therefore control.

  4. **The battery gap.** Sketch a charge and a discharge voltage curve for the same cell at the same rate, label the gap, and attribute it to the three loss terms of Section 5.3. Then predict — with reasons — how the sketch changes if the same cell is cycled at ten times the rate, and which of the three terms you expect to grow fastest.

  5. **Budgeting a research programme.** Using the code in Section 5.5, find the improvement in `I0_OER` needed to save 0.300 V at 1 A/cm², and separately the reduction in `R_CELL` needed to save the same 0.300 V. State whether each is plausible, and use the comparison to argue for how you would split a fixed research budget between catalyst discovery and cell engineering. (Note whether the ohmic answer is even physically attainable, and what that tells you.)

## Summary

Water electrolysis splits into **HER at the cathode** (two electrons per H₂) and **OER at the anode** (four electrons per O₂), and the electron bookkeeping alone forces the 2:1 product ratio. Its thermodynamic floor is **1.23 V**, the voltage at which nothing happens; the **thermoneutral voltage of about 1.48 V** marks where the cell stops needing heat from outside and starts producing it. A working cell runs at \\(E_{\text{rev}} + \eta_{\text{OER}} + \eta_{\text{HER}} + iR_{\text{cell}}\\), and every term after the first is loss. **OER is the bottleneck** because it moves four electrons and four protons through three surface intermediates and must form an O–O bond, all driven by a single potential that sets the driving force for every step at once.

**CO₂ electrolysis** keeps the same anode and changes the question at the cathode. Its products form a ladder ordered by electron count — CO and formate at two electrons, methanol and methane at six to eight, multi-carbon products at twelve or more plus a carbon–carbon bond. The defining difficulty is not the electron count but **competition**: hydrogen evolution runs at potentials clustered near those of the CO₂ pathways, so thermodynamics does not separate them and **selectivity becomes a purely kinetic property of the catalyst**, audited by Faradaic efficiency.

**Batteries are the same cell read in both directions** — discharge is galvanic, charge is electrolytic — which is why an electrode's anode/cathode label flips between them. The cell voltage is a difference of electrode potentials, the voltage sag on discharge is the Nernst equation acting on changing compositions, and the **charge–discharge voltage gap is overpotential plus \\(iR\\) appearing twice with opposite signs**. Round-trip efficiency is therefore a kinetics number, and it degrades with rate because every loss term grows with current.

The **hands-on budget** made the structure quantitative with clearly illustrative kinetic parameters: at 1 A/cm² the anode overpotential was 0.420 V against the cathode's 0.120 V, the applied voltage was 1.970 V, and voltage efficiency was 62.4% against the 1.23 V reference. Activation losses grow logarithmically with current while ohmic losses grow linearly, so the dominant problem shifts from catalysis to cell engineering as devices scale. A hundredfold better anode catalyst bought 0.120 V; halving the cell resistance bought 0.100 V. **Catalysis pays logarithmically, engineering pays linearly** — and in energy terms, this illustrative cell needed 52.4 kWh per kilogram of hydrogen against a thermodynamic floor of 32.7.

## Series Conclusion: One Argument, Five Chapters

Look back at the five chapters and they make a single argument with a single shape.

**Chapter 1** established the accounting. Electrochemistry is oxidation and reduction with the two halves physically separated, so that the electrons have to travel through a wire you own — and once they do, you can count them, and counting them is Faraday's law. It also drew the map: galvanic cells give you electricity, electrolytic cells consume it, and the same hardware does both.

**Chapter 2** established what is *possible*. Electrode potentials are free energies per electron; \\(\Delta G = -nFE\\) is the exchange rate between chemistry's currency and electricity's; the difference of two potentials is a cell voltage; and the Nernst equation says how that voltage moves when concentrations do. The 1.23 V of water splitting entered here, as a thermodynamic fact.

**Chapter 3** established what is *fast*. Equilibrium is not a rate, and a reaction that thermodynamics permits may proceed imperceptibly slowly. Overpotential is what you pay to make it go; exchange current density measures how much of a bargain a given surface offers; Butler–Volmer becomes Tafel in the useful regime; and a catalyst changes the kinetics without ever touching the thermodynamics. That last sentence is the single most useful idea in this series.

**Chapter 4** established what is *measurable*. Everything above happens at an interface with a double layer, a potential drop over nanometres, and no way to measure the potential of a single electrode in isolation — hence the three-electrode cell, the reference electrode, the voltammogram, and the \\(iR\\) correction that separates what the interface did from what the electrolyte cost you.

**Chapter 5** spent it. Every technology in this chapter is those four ideas with different labels: a thermodynamic floor you cannot go below, kinetic overpotentials you pay at both electrodes, an ohmic tax that grows with ambition, and a measurement discipline that tells you which is which. The reason a water electrolyser, a CO₂ reactor, and a battery can be taught in one chapter is that at this level of description **they are the same device**.

Three routes lead onward from here. If you want to know why the OER overpotential resists improvement — and how binding-energy calculations both explain and bound it — go to [Computational Chemistry of OER](<../../MI/oer-computational-chemistry/index.html>). If you want the data-driven layer above catalyst discovery, with its methods and its honest failure modes, go to [MI Applications to Catalyst Design](<../../MI/catalyst-mi-application/index.html>). And if Section 5.2 left you wanting the full story of turning carbon dioxide back into something useful, that series is coming, and this one was written to be its prerequisite.

[← Chapter 4: The Electrochemical Interface](<chapter-4.html>) [Series Top →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
