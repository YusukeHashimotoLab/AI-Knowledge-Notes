---
title: "Chapter 1: Why OER Is the Bottleneck"
chapter_title: "Chapter 1: Why OER Is the Bottleneck"
subtitle: "Water Electrolysis, the Four-Electron Anode, and the Overpotential That Sets the Price of Green Hydrogen"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/4p0lkDnLqwI"
    title="OER Comp Chem Ch.1: Why OER Is the Bottleneck"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/MI/oer-computational-chemistry/chapter-1.html>) | Last sync: 2026-08-18

[Materials Informatics Dojo](<../index.html>) > [Computational Chemistry of OER](<index.html>) > Chapter 1

Green hydrogen has a simple recipe: put electricity into water, take hydrogen out. The chemistry has been understood for two centuries, the engineering is commercially deployed, and the input — renewable electricity — is getting cheaper every year. And yet the electricity bill for a kilogram of hydrogen is still stubbornly higher than thermodynamics says it needs to be, and most of the excess is being spent at one electrode.

That electrode is the **anode**, where the **oxygen evolution reaction (OER)** happens. This chapter explains why the reaction that produces the *by-product* is the one that limits the whole device, defines the two quantities the rest of the series is built on — the equilibrium potential and the **overpotential** — and shows, with code you can run, exactly how much energy each extra millivolt costs.

This series is a focused companion to the [MI Applications to Catalyst Design](<../catalyst-mi-application/index.html>) series. That series covers how machine learning and Bayesian optimization search catalyst composition space; this one goes one level down, into the *physical quantity* those models are trying to predict. If you have already read it, treat this as the chemistry underneath its descriptors.

## 1.1 Splitting Water Takes Two Electrodes

Water electrolysis is one overall reaction driven by two half-reactions at two separate electrodes:

\\[ 2\text{H}_2\text{O} \;\longrightarrow\; 2\text{H}_2 + \text{O}_2 \\]

The electrons do not travel through the water. They leave one electrode, go around an external circuit through the power supply, and arrive at the other. Ions carry the charge through the electrolyte to close the loop. Writing the halves separately, in the **acidic convention** (protons as the mobile ion), makes the asymmetry visible immediately.

At the **cathode**, the **hydrogen evolution reaction (HER)**:

\\[ 2\text{H}^+ + 2e^- \;\longrightarrow\; \text{H}_2 \\]

At the **anode**, the **oxygen evolution reaction (OER)**:

\\[ 2\text{H}_2\text{O} \;\longrightarrow\; \text{O}_2 + 4\text{H}^+ + 4e^- \\]

Add them in the ratio 2:1 and the protons cancel, leaving the overall reaction above. Nothing is lost or invented in the bookkeeping — but the two halves are not comparable in difficulty.

### 📚 Two Electrons Versus Four

**HER moves two electrons and forms one bond.** Two protons pick up two electrons and pair into H₂. On a good surface the sequence is short: adsorb a hydrogen atom, then either combine it with a second adsorbed hydrogen or attack it with another proton. There is essentially **one intermediate** — an adsorbed H atom — and therefore essentially one binding energy to get right.

**OER moves four electrons and forms an O=O double bond.** Two water molecules must be stripped of four protons and four electrons, and two oxygen atoms that started on different molecules must be joined. Doing this in one concerted event is not physically plausible on a surface; the reaction proceeds through a **sequence of steps**, each transferring one proton and one electron, passing through a chain of surface-bound intermediates before O₂ finally leaves.

The consequence is structural, not incidental. **Each additional coupled transfer is another place to lose energy.** Every intermediate must be bound to the surface — bound too weakly and the surface will not form it; bound too strongly and it will not let go. A catalyst designer facing HER has roughly one binding energy to tune. A designer facing OER has three intermediates whose binding energies must all be right *simultaneously*, on the same surface, at the same site.

Chapter 3 shows that this is worse than it sounds, because those binding energies are not independent of one another. For now, hold the counting argument: more coupled transfers means more intermediates, and more intermediates means more constraints that a single surface must satisfy at once.

## 1.2 What a Potential Actually Does

Readers coming from materials informatics rather than electrochemistry often meet potentials as numbers in a dataset. It is worth spending a paragraph on what the number physically does, because the entire computational method in Chapter 2 rests on it.

An electrode potential is a **control knob on the energy of electrons in the electrode**. Making the electrode more positive lowers the energy of its electrons; making it more negative raises them. An electron transfer from an adsorbed species to the electrode is therefore uphill or downhill depending on where you have set that knob.

Quantitatively, moving one electron across a potential difference \\(U\\) changes its free energy by \\(-eU\\). For a reaction that transfers \\(n\\) electrons, the free-energy change and the cell potential are linked by

\\[ \Delta G = -nFE \\]

where \\(F\\) is the Faraday constant — the charge of one mole of electrons. The minus sign encodes the convention that a spontaneous reaction (negative \\(\Delta G\\)) produces a positive cell potential.

### 📚 Where 1.23 V Comes From

Read the relation backwards. Water splitting is *not* spontaneous — you have to push it — so its \\(\Delta G\\) is positive and the potential you need is the one that exactly cancels it.

Splitting one mole of liquid water into hydrogen and oxygen at standard conditions costs a positive standard Gibbs free energy, and the reaction moves two electrons per H₂ molecule. Dividing that energy by \\(nF\\) gives the **standard equilibrium potential of water splitting**:

\\[ E^{0} = 1.23\ \text{V} \\]

This series treats 1.23 V as a **standard-state definition** — the conventional reference every OER discussion is measured against. The code in Section 1.5 runs \\(\Delta G = nFE\\) forward from that definition to show what 1.23 V is worth in kilojoules, and the result lands where the standard Gibbs energy of water formation says it should. That is a consistency check, not a new measurement.

Two readings of the same number matter for later chapters:

  * **Per mole of H₂**: 1.23 V × 2 electrons × \\(F\\) is the thermodynamic minimum energy input.
  * **Per electron**: exactly **1.23 eV**. A full OER turnover moves four electrons, so the oxygen side of the ledger must account for **4 × 1.23 = 4.92 eV**. Chapter 2 turns that sentence into a hard constraint on any free-energy diagram you draw.

### 📚 Overpotential: The Extra Push

At exactly 1.23 V, thermodynamics permits water splitting — and nothing happens at a useful rate. Equilibrium potentials describe where a reaction *stops being forbidden*, not how fast it goes. To drive real current through a real electrode you must apply more:

\\[ \eta \;=\; U_{\text{applied}} - E^{0} \\]

The quantity \\(\eta\\) is the **overpotential**: the extra voltage beyond the thermodynamic price that a real electrode demands. It is the central figure of merit of this entire series. Every catalyst comparison, every volcano plot, every screening campaign in computational OER research is ultimately a competition to make \\(\eta\\) smaller.

A practical cell must supply the sum of several such penalties: an anode overpotential, a cathode overpotential, and ohmic losses in the electrolyte and hardware. Two facts justify this series' focus on the anode. First, the OER contribution is typically the largest of the electrochemical terms, for exactly the four-electron reason in Section 1.1. Second, it is the term most directly addressable by **catalyst design** — you cannot redesign the conductivity of water, but you can redesign a surface.

## 1.3 Why Overpotential Costs Money

The economics follow from a conservation law, not from a market forecast.

The number of electrons required per kilogram of hydrogen is **fixed by stoichiometry**. Faraday's law says each mole of H₂ needs exactly two moles of electrons, and a kilogram of H₂ is a fixed number of moles. So the *charge* you must push per kilogram cannot be negotiated by any catalyst, any engineer, or any subsidy.

Energy, however, is charge multiplied by voltage. Since the charge is fixed, the **energy per kilogram is directly proportional to the cell voltage** — and the cell voltage is \\(1.23 + \eta\\) volts. Which gives the sentence that motivates this whole field:

> Every volt of overpotential is a volt of pure waste, paid on every single electron, for the entire operating life of the plant.

It also explains a scaling behaviour that surprises people expecting diminishing returns. Because energy is linear in \\(\eta\\), reducing overpotential by 100 mV saves the *same* number of kilowatt-hours per kilogram whether you go from 400 mV to 300 mV or from 200 mV to 100 mV. There is no point on the curve where further improvement stops paying. The code below makes this exact.

We will state no electricity prices. Prices vary by region, contract, and year, and inventing one would be the kind of false precision this series avoids. Energy per kilogram is a physical quantity we can compute honestly; multiply it by whatever price your own situation carries.

## 1.4 What Computation Offers

Here is the case for spending the rest of this series on quantum chemistry rather than on experiments.

**The experimental loop is slow and the search space is enormous.** Testing a candidate OER catalyst means synthesizing it, mounting it in a cell, and measuring it — and the space of plausible oxide surfaces, dopants, terminations, and facets is combinatorially large. Even a well-run high-throughput laboratory samples a vanishing fraction of it.

**Computation can ask the deciding question before synthesis.** The insight that makes this tractable is that OER activity is governed largely by **how strongly a surface binds the reaction intermediates**. Those binding energies are computable from first principles for a surface that has never been made. So the screening question becomes: *given a hypothetical surface, does it bind \*OH, \*O and \*OOH in the pattern that makes all four steps easy?* Chapter 2 turns that question into a calculable number — a theoretical overpotential — using the computational hydrogen electrode.

**This is where the MI connection lives.** Once a descriptor is calculable, it can be predicted, and once it can be predicted, a machine learning model can screen millions of candidates that were never computed at all. That pipeline — DFT for a training set, a model for the sweep, experiment for the survivors — is what the [catalyst MI series](<../catalyst-mi-application/index.html>) builds. This series supplies the physically meaningful target variable that such a pipeline needs. A model trained on a descriptor nobody can interpret is a model nobody can trust.

### 📚 The Honest Caveat, Stated Early

Everything above comes with conditions attached, and stating them now is better than discovering them later.

The models we will build treat an idealized surface: a flat, perfect, static facet, with the solvent handled approximately and the electron transfers assumed to be thermodynamically limited rather than kinetically limited. Real catalysts have steps, kinks, defects, and surfaces that **restructure while operating**. The theoretical overpotential we compute is a *thermodynamic lower bound on the difficulty*, not a prediction of a measured polarization curve.

That is still enormously useful. It ranks candidates, explains why whole families of materials underperform, and tells you which direction to move. It is not a substitute for measurement. Chapter 5 examines these limits properly, including the cases where the model is known to mislead. Read this series as building a good map, not a photograph.

## 1.5 Hands-On: The Price of an Overpotential

Let us compute the bookkeeping. The code takes the 1.23 V definition and Faraday's law as its only inputs and derives everything else: the energy behind the equilibrium potential, the energy needed per kilogram of hydrogen, and how that scales with overpotential.

**A word on the numbers.** The overpotential values swept below are **illustrative teaching values** chosen to span a plausible range. They are not measurements and are not attributed to any material. The constants — the Faraday constant, the molar mass of H₂, the 1.23 V convention — are definitions, and everything else is derived from them.

```python
import numpy as np

# ---------------------------------------------------------------
# The thermodynamic price of splitting water, and what an
# overpotential adds to the bill.
#
# Constants below are SI-defined or CODATA values, plus ONE
# convention: the standard equilibrium cell potential of water
# electrolysis, E0 = 1.23 V. Everything else is derived.
# ---------------------------------------------------------------
FARADAY = 96485.33212        # C/mol, exact by SI definition
M_H2 = 2.01588e-3            # kg/mol, molar mass of H2
E0_WATER = 1.23              # V, standard equilibrium cell potential (convention)
N_ELECTRONS_PER_H2 = 2       # 2 H+ + 2 e- -> H2

# --- 1. Where 1.23 V comes from: dG = n F E ---------------------
dG_per_mol_H2 = N_ELECTRONS_PER_H2 * FARADAY * E0_WATER   # J per mol H2
print("Step 1: the 1.23 V convention, read as an energy")
print(f"  dG = n F E0 = {N_ELECTRONS_PER_H2} x {FARADAY:.2f} C/mol x {E0_WATER} V")
print(f"     = {dG_per_mol_H2/1000:.1f} kJ per mol H2 split from liquid water")
print(f"  Per electron transferred: {E0_WATER:.2f} eV")
print(f"  A full OER turnover moves 4 electrons: {4*E0_WATER:.2f} eV")
print()

# --- 2. Electrical energy per kilogram of hydrogen --------------
def energy_per_kg_H2(cell_voltage):
    """Electrical energy (kWh) to make 1 kg of H2 at a given cell voltage.

    Faraday's law only: 1 mol H2 needs 2 mol of electrons, so the
    charge per kg of H2 is fixed and the energy is strictly linear
    in the cell voltage.
    """
    charge_per_kg = N_ELECTRONS_PER_H2 * FARADAY / M_H2   # C per kg H2
    joules = charge_per_kg * cell_voltage                 # J per kg H2
    return joules / 3.6e6                                 # -> kWh per kg

# --- 3. Illustrative overpotentials (TEACHING VALUES) -----------
# These eta values are chosen to span a plausible teaching range.
# They are NOT measurements and are NOT tied to any real material.
eta_values = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50])

cell_voltage = E0_WATER + eta_values
voltage_efficiency = E0_WATER / cell_voltage
energy = np.array([energy_per_kg_H2(u) for u in cell_voltage])
excess = energy - energy[0]

print("Step 2: cost of an overpotential (ILLUSTRATIVE eta values)")
print(f"{'eta (V)':>8} {'U_cell (V)':>11} {'efficiency':>11} "
      f"{'kWh per kg H2':>15} {'excess kWh/kg':>15}")
print("-" * 64)
for e, u, f, w, x in zip(eta_values, cell_voltage, voltage_efficiency, energy, excess):
    print(f"{e:8.2f} {u:11.2f} {100*f:10.1f}% {w:15.2f} {x:15.2f}")
print()

# --- 4. The scaling is linear, exactly ---------------------------
slope = np.diff(energy) / np.diff(eta_values)
print("Step 3: is the cost linear in eta?")
print(f"  d(kWh/kg) / d(eta) for each interval: {np.round(slope, 4)}")
print(f"  spread across intervals: {slope.max() - slope.min():.2e} kWh/kg per V")
print(f"  -> every extra 0.10 V of overpotential costs "
      f"{0.10*slope.mean():.2f} kWh per kg H2, at any starting point")
print()

# --- 5. Halving the overpotential, in efficiency terms -----------
for e in [0.50, 0.25]:
    u = E0_WATER + e
    print(f"  eta = {e:.2f} V -> U_cell = {u:.2f} V, "
          f"voltage efficiency {100*E0_WATER/u:.1f}%, "
          f"{energy_per_kg_H2(u):.2f} kWh/kg")
```

**Output:**

```
Step 1: the 1.23 V convention, read as an energy
  dG = n F E0 = 2 x 96485.33 C/mol x 1.23 V
     = 237.4 kJ per mol H2 split from liquid water
  Per electron transferred: 1.23 eV
  A full OER turnover moves 4 electrons: 4.92 eV

Step 2: cost of an overpotential (ILLUSTRATIVE eta values)
 eta (V)  U_cell (V)  efficiency   kWh per kg H2   excess kWh/kg
----------------------------------------------------------------
    0.00        1.23      100.0%           32.71            0.00
    0.10        1.33       92.5%           35.37            2.66
    0.20        1.43       86.0%           38.02            5.32
    0.30        1.53       80.4%           40.68            7.98
    0.40        1.63       75.5%           43.34           10.64
    0.50        1.73       71.1%           46.00           13.30

Step 3: is the cost linear in eta?
  d(kWh/kg) / d(eta) for each interval: [26.5904 26.5904 26.5904 26.5904 26.5904]
  spread across intervals: 1.56e-13 kWh/kg per V
  -> every extra 0.10 V of overpotential costs 2.66 kWh per kg H2, at any starting point

  eta = 0.50 V -> U_cell = 1.73 V, voltage efficiency 71.1%, 46.00 kWh/kg
  eta = 0.25 V -> U_cell = 1.48 V, voltage efficiency 83.1%, 39.35 kWh/kg
```

**Reading the result.** Four things, in increasing order of importance.

  * **The consistency check passes.** Running \\(\Delta G = nFE\\) forward from 1.23 V gives 237.4 kJ per mole of H₂ — which is where the standard Gibbs energy of forming liquid water sits. We did not look that number up; we derived it from the potential and it landed in the right place. The 1.23 V convention and the thermochemistry of water are the same statement in different units.

  * **Efficiency falls fast.** At zero overpotential the voltage efficiency is 100% by construction — that is the definition of the reference. At 0.30 V of overpotential it is already down to 80.4%, and at 0.50 V to 71.1%. Roughly three tenths of the electricity bought for the plant is being converted to heat rather than to fuel.

  * **The slope is a constant, to machine precision.** The five interval slopes are identical to four decimal places, and the spread across them is \\(10^{-13}\\) — floating-point noise, not physics. This is the linearity argument made numerically. Every 0.10 V of overpotential costs 2.66 kWh per kilogram of H₂, and it costs that whether it is the first 0.10 V or the fifth.

  * **A catalyst improvement never stops paying.** Cutting overpotential from 0.50 V to 0.25 V takes the energy from 46.00 to 39.35 kWh/kg — a saving of 6.65 kWh on every kilogram produced, forever. Multiply by your own electricity price and your own plant output; the physics has done its part of the arithmetic.

One caution about the efficiency column. The 1.23 V reference is a *free-energy* reference, so "100% efficient" here means "no electrical work wasted relative to the thermodynamic minimum" — a real cell operating reversibly would also need to absorb heat from its surroundings. The comparison is exact for what we are doing (ranking overpotentials) and is not a complete thermal accounting of an electrolyser.

Try replacing the illustrative sweep with a finer grid — `np.arange(0, 0.61, 0.05)` — and confirm that the constant-slope result is unchanged. That invariance is the point: the conclusion does not depend on which teaching values we picked.

### 🎯 Exercise Problems

  1. **Counting the transfers.** Write out HER and OER in the acidic convention and verify by hand that combining them in a 2:1 ratio cancels all protons and electrons, leaving 2H₂O → 2H₂ + O₂. State how many electrons cross the circuit per O₂ molecule produced.

  2. **Reading the knob.** An electrode is made more positive by 0.20 V. By how much does the free energy of transferring one electron *from* an adsorbed species *to* that electrode change, in eV? State the sign and explain it in words.

  3. **The fixed charge.** Using only Faraday's law and the molar mass of H₂, compute the charge in coulombs required to produce 1 kg of hydrogen. Explain why no catalyst can change this number, and which factor in the energy expression a catalyst *can* change.

  4. **Where the savings are.** Using the code, compute the kWh/kg saved by reducing overpotential from 0.45 V to 0.35 V, and from 0.15 V to 0.05 V. Are they the same? Explain, in terms of the algebra of \\(U = 1.23 + \eta\\), why the answer had to come out that way.

  5. **Auditing a claim.** A press release states that a new catalyst "achieves 95% efficiency". List four questions you would need answered before that number means anything — including which reference potential the efficiency is measured against, and whether it refers to one electrode or a full cell.

## Summary

Water electrolysis splits into two half-reactions, and they are not equally hard. **HER** at the cathode moves two electrons through essentially one intermediate; **OER** at the anode moves four electrons and four protons through a chain of surface-bound intermediates, and must form an O=O bond from oxygen atoms that began on separate water molecules. More coupled transfers means more intermediates, and more intermediates means more binding energies that a single surface must get right at once — which is why the by-product electrode is the bottleneck.

An electrode potential is a control knob on electron energies, connected to the free energy of a reaction by \\(\Delta G = -nFE\\). The **standard equilibrium potential of water splitting, 1.23 V**, is the thermodynamic price of the reaction, taken in this series as a standard-state definition; our code ran \\(\Delta G = nFE\\) forward from it and recovered 237.4 kJ per mole of H₂, exactly where the thermochemistry of water says it should be. Read per electron, that same number is **1.23 eV**, so a four-electron OER turnover must account for **4.92 eV** — a constraint Chapter 2 will enforce explicitly.

The **overpotential** \\(\eta = U - E^0\\) is the extra push a real electrode demands, and it is the figure of merit for the rest of this series. Because Faraday's law fixes the charge per kilogram of hydrogen, the energy cost is strictly **linear** in \\(\eta\\): our code found the slope constant to within \\(10^{-13}\\), at 2.66 kWh per kilogram for every 0.10 V. Overpotential is therefore waste paid on every electron for the life of the plant, and improvements never stop paying. Computation earns its place by predicting how a hypothetical surface binds the OER intermediates *before* anyone synthesizes it — with the honest caveat that the idealized surfaces we model give a thermodynamic bound on difficulty, not a measured polarization curve.

The next chapter builds the tool that makes those predictions possible. We write out the four proton-coupled electron transfer steps explicitly, introduce the **computational hydrogen electrode** — the trick that lets us price a proton-electron pair without ever simulating a solvated proton — and construct free-energy diagrams that turn a set of binding energies into a single number: the theoretical overpotential.

[← Series Top](<index.html>) [Chapter 2: The Computational Hydrogen Electrode →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
