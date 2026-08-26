---
title: "Chapter 1: Why Electrochemistry?"
chapter_title: "Chapter 1: Why Electrochemistry?"
subtitle: "Reactions That Move Electrons Through a Wire, and Why a Carbon-Neutral Society Runs on Them"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/MV6Adc8gntY"
    title="Electrochemistry Ch.1: Why Electrochemistry?"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/MS/electrochemistry-introduction/chapter-1.html>) | Last sync: 2026-08-20

[Materials Science Dojo](<../index.html>) > [Electrochemistry Introduction](<index.html>) > Chapter 1

Almost every chemical reaction you met in an introductory course happens in a single place. Two molecules collide, bonds rearrange, and whatever energy is released comes out as heat, warming the flask. The reaction is over before you can get an instrument near the interesting part of it.

Electrochemistry is what happens when you refuse to let that happen. You take a reaction that would have released its energy as heat, split it into two halves, put the halves in two different places, and force the electrons that would have jumped directly between them to travel through a wire instead. Suddenly the reaction is no longer a private transaction inside a beaker. It is a current you can measure, a voltage you can control, and a rate you can dial up or down with a knob.

That single structural change — separating the two halves of a redox reaction in space — is the whole subject. Everything else in this series is a consequence: why a battery has a particular voltage, why splitting water costs at least a certain number of volts, why a catalyst changes how fast the reaction goes but never changes what voltage it needs, and why a laboratory measurement of an electrode requires three electrodes rather than the obvious two.

This chapter builds the vocabulary. It also makes the case for why a materials scientist in 2026 should care, which is not a matter of taste: the technologies a carbon-neutral society depends on — water electrolysis, carbon dioxide conversion, and every rechargeable battery — are all electrochemical, and all of them are limited by materials rather than by principles. If you are coming from the computational side, this series is the physical grounding underneath [OER Computational Chemistry](<../../MI/oer-computational-chemistry/index.html>) and [Catalyst Informatics](<../../MI/catalyst-mi-application/index.html>); if you are coming from thermodynamics, [Materials Thermodynamics](<../materials-thermodynamics-introduction/index.html>) develops the free-energy machinery that Chapter 2 will put to work.

## 1.1 Redox, Reconsidered: Chemistry That Moves Electrons

Start with a reaction that has nothing electrical about it. Drop a strip of zinc metal into a solution of copper sulfate. Within minutes the zinc surface darkens, a spongy layer of copper metal appears on it, and the blue colour of the solution fades. Written as one equation:

\\[
\mathrm{Zn}(s) + \mathrm{Cu}^{2+}(aq) \longrightarrow \mathrm{Zn}^{2+}(aq) + \mathrm{Cu}(s)
\\]

Nothing here looks like electricity. The beaker gets slightly warm and that is the end of it. But look at what actually moved. Zinc began as a neutral metal atom and ended as a doubly charged ion in solution; it lost two electrons. Copper began as a doubly charged ion and ended as a neutral metal atom; it gained two electrons. **Two electrons were transferred from a zinc atom to a copper ion**, at the point of contact, and the energy of that transfer came out as heat.

This is what *redox* means, stripped of the mnemonic devices. **Oxidation is loss of electrons. Reduction is gain of electrons.** The two always occur together, because electrons do not accumulate anywhere — whatever one species loses, another must gain, in the same instant and in exactly the same number.

The word "oxidation" is a historical accident worth acknowledging once and then setting aside. It originally meant *combination with oxygen*, because burning and rusting were the reactions people had in mind. The modern definition — loss of electrons — includes those cases but is far broader, and it has nothing necessarily to do with oxygen. Zinc dissolving in copper sulfate involves no oxygen at all, and it is a textbook oxidation.

### 📚 Three Ways to Recognize a Redox Reaction

Not every reaction is redox, and telling them apart quickly is a useful skill.

  * **Oxidation states change.** Assign oxidation numbers to every atom on both sides. If any of them changed, electrons moved. If none changed, the reaction is not redox — acid-base neutralization and most precipitations fall in this category.
  * **A free element appears or disappears.** Elemental forms (\\(\mathrm{Zn}\\), \\(\mathrm{Cu}\\), \\(\mathrm{H_2}\\), \\(\mathrm{O_2}\\), \\(\mathrm{Cl_2}\\)) have oxidation state zero by definition. Any reaction that consumes or produces one is necessarily redox.
  * **The reaction could, in principle, be split.** This is the electrochemist's test and the most useful one. If you can imagine physically separating the electron donor from the electron acceptor and connecting them with a wire, the reaction is redox. If you cannot — if the two partners must touch for anything to happen — it is not.

That third test is the bridge to everything that follows. The zinc-and-copper-sulfate reaction passes it, and separating the two halves is exactly what turns a warm beaker into a battery.

## 1.2 Half-Reactions: Splitting the Bookkeeping

Because oxidation and reduction always occur together, it is convenient to write them apart. Each piece is called a **half-reaction**, and each carries electrons explicitly as a reactant or a product:

\\[
\mathrm{Zn}(s) \longrightarrow \mathrm{Zn}^{2+}(aq) + 2e^{-} \qquad \text{(oxidation)}
\\]

\\[
\mathrm{Cu}^{2+}(aq) + 2e^{-} \longrightarrow \mathrm{Cu}(s) \qquad \text{(reduction)}
\\]

Add them and the electrons cancel, returning the overall reaction. A half-reaction with free electrons in it is not a reaction that can happen alone — electrons do not float around in solution waiting to be used. It is an accounting device, and a remarkably powerful one.

Its power comes from **modularity**. There are perhaps a few hundred half-reactions of practical interest, and any oxidation can be paired with any reduction to give a complete reaction. Instead of tabulating every possible pairing, you tabulate the halves once and combine them as needed. Chapter 2 turns this into a quantitative tool: each half-reaction gets a number, a *standard electrode potential*, and the voltage of any pairing is a subtraction.

Balancing a half-reaction follows a fixed procedure, and it is worth having it in your hands because you will do it constantly.

### 📚 Balancing a Half-Reaction in Acidic Solution

Take the reduction of permanganate, a standard laboratory oxidant, as the worked example.

  1. **Balance the atoms other than H and O.** \\(\mathrm{MnO_4^-} \to \mathrm{Mn}^{2+}\\) — manganese is already balanced, one on each side.
  2. **Balance O by adding \\(\mathrm{H_2O}\\).** Four oxygens on the left, none on the right, so add four waters on the right: \\(\mathrm{MnO_4^-} \to \mathrm{Mn}^{2+} + 4\,\mathrm{H_2O}\\).
  3. **Balance H by adding \\(\mathrm{H^+}\\).** Eight hydrogens on the right now, so add eight protons on the left: \\(\mathrm{MnO_4^-} + 8\,\mathrm{H^+} \to \mathrm{Mn}^{2+} + 4\,\mathrm{H_2O}\\).
  4. **Balance charge by adding electrons.** The left side carries \\(-1 + 8 = +7\\); the right carries \\(+2\\). Add five electrons to the left to bring it to \\(+2\\).

\\[
\mathrm{MnO_4^-} + 8\,\mathrm{H^+} + 5e^{-} \longrightarrow \mathrm{Mn}^{2+} + 4\,\mathrm{H_2O}
\\]

In basic solution, the same procedure works, followed by adding \\(\mathrm{OH^-}\\) to both sides to neutralize any \\(\mathrm{H^+}\\) that appears. The presence of \\(\mathrm{H^+}\\) in step 3 is not a technicality — it is why the potentials of many half-reactions depend on pH, a dependence Chapter 2 quantifies through the Nernst equation.

Notice also that the electron count is not a free choice. Permanganate reduction is a **five-electron** process; copper deposition is a **two-electron** process; the reduction of carbon dioxide to methane is an **eight-electron** process. This number, conventionally called \\(n\\), appears in every quantitative relation in the subject, and it is the reason some electrochemical reactions are easy and others are notoriously difficult.

## 1.3 Two Kinds of Cell: Galvanic and Electrolytic

Now perform the separation. Put the zinc strip in a beaker of zinc sulfate solution and the copper strip in a separate beaker of copper sulfate solution. Connect the two metals by a wire, and connect the two solutions by a **salt bridge** — a tube of inert electrolyte that lets ions pass but keeps the solutions from mixing.

The reaction still wants to happen; its chemistry has not changed. But the zinc atom and the copper ion are now centimetres apart, and electrons cannot jump that far through water. The only available route is the wire. Zinc dissolves, pushing electrons into the wire; those electrons travel to the copper electrode and reduce copper ions there. **A current flows, and it flows because a chemical reaction is driving it.**

This is a **galvanic cell** (also called a voltaic cell), and the arrangement just described is the **Daniell cell**, the standard teaching example we will make quantitative in Chapter 2. The essential feature is that the reaction is spontaneous: the cell does electrical work on the outside world, and the free energy released comes from the chemistry.

The salt bridge deserves a moment, because students often treat it as decoration. It is not. As zinc dissolves, positive charge builds up in the left beaker; as copper deposits, positive charge is depleted from the right. Within a fraction of a second the resulting electric field would stop the reaction dead. The salt bridge lets ions migrate to cancel that charge imbalance, closing the circuit through the solution. **A cell without an ionic path is not a slow cell; it is an open circuit.**

Now run the same apparatus backwards. Disconnect the wire and connect a power supply across the two electrodes instead, pushing current the other way. If you push hard enough, copper dissolves and zinc plates out — the reverse of the spontaneous reaction. This is an **electrolytic cell**: an arrangement in which electrical energy from outside drives a reaction that would not happen on its own.

### 📚 Galvanic vs Electrolytic, Side by Side

| | Galvanic (voltaic) cell | Electrolytic cell |
|---|---|---|
| Driving force | Spontaneous chemical reaction | External power supply |
| Energy flow | Chemistry → electricity | Electricity → chemistry |
| Sign of \\(\Delta G\\) | Negative | Positive (for the forced direction) |
| Cell voltage | Produced by the cell | Applied to the cell |
| Everyday example | Battery discharging, fuel cell | Water electrolysis, electroplating, battery charging |
| Anode polarity in the external circuit | Negative terminal | Positive terminal |
| Where oxidation happens | At the anode | At the anode |

The last two rows are the ones that cause trouble, and Section 1.4 is devoted to them.

There is a beautiful symmetry hiding here that is worth stating early. A rechargeable battery is **both kinds of cell in one device**: galvanic when discharging, electrolytic when charging. The same electrode is the site of oxidation in one mode and reduction in the other. This is why the terminology matters — a naming convention that flips when you plug the device in is a naming convention that will confuse you at the worst possible moment.

## 1.4 Anode and Cathode: Settling the Confusion Once

Here is the single most reliable source of error for newcomers to this subject, and it is entirely avoidable.

**The definition is about chemistry, not about polarity:**

  * **The anode is the electrode where oxidation occurs.**
  * **The cathode is the electrode where reduction occurs.**

That is all. These definitions never change, in any device, in any mode of operation. Memorize them in this form and you will never be wrong.

The confusion arises because people also remember a *polarity* rule — "the anode is the negative terminal" — and that rule is only true for a galvanic cell. Work through both cases and the reason becomes obvious.

**In a galvanic cell (a battery discharging).** The anode is where zinc is oxidizing, pumping electrons into the external circuit. An electrode that is a source of electrons for the external circuit is, from the outside, the **negative terminal**. So here: anode = negative.

**In an electrolytic cell (electrolysis, or a battery charging).** An external power supply is pulling electrons out of one electrode and pushing them into the other. The electrode from which electrons are pulled must replace them by taking electrons from something in the solution — and taking electrons from a species is oxidizing it, so this electrode is the anode. But it is connected to the *positive* side of the power supply. So here: anode = positive.

The polarity flipped. The chemistry did not. In both cases the anode is where oxidation happens; only its relationship to the external circuit changed, because in one case the cell is the source of energy and in the other it is the load.

### 📚 A Mnemonic That Actually Survives Both Cases

Two letter-matching tricks work regardless of cell type, because both point at chemistry rather than polarity:

  * **An Ox** — **An**ode, **Ox**idation.
  * **Red Cat** — **Red**uction, **Cat**hode.

And one more that helps with current direction: the words share initials with the ions that move toward them. **An**ions (negative ions) migrate toward the **an**ode; **cat**ions (positive ions) migrate toward the **cat**hode. This is true in both cell types, and it follows from the chemistry: the anode is producing positive charge in solution, so negative ions are drawn in to balance it.

Alongside this sits a second convention that must be nailed down before Chapter 2, because half the textbooks in circulation once did it the other way. **All electrode potentials in this series, and in modern practice, are quoted as reduction potentials.** Every half-reaction is written in the direction of reduction, with electrons on the left, and its tabulated potential refers to that direction. This is the **IUPAC convention**, and it has been standard for decades. If you encounter an old table of "oxidation potentials", every sign in it is flipped relative to what we use. Chapter 2 leans on this convention completely: a cell voltage is computed as \\(E^{\circ}_{\text{cathode}} - E^{\circ}_{\text{anode}}\\), with both values read from the same reduction-potential table, and the subtraction handles the direction reversal for you.

## 1.5 Why This Matters Now: Energy, Storage, and Carbon

Electrochemistry has been a respectable corner of physical chemistry since **Volta built the first battery in 1800** and **Faraday established the quantitative laws of electrolysis in the 1830s**. What changed recently is not the science but the stakes.

The reason is a mismatch. Renewable electricity is now cheap, and in many places abundant — but it is also **intermittent** and it is **electricity**, not fuel and not chemicals. A society that decarbonizes by building solar and wind farms immediately faces two problems that generation alone cannot solve: what to do when the sun is not shining, and what to do about the enormous fraction of industrial emissions that come from making *materials* rather than from making power. Steel, cement, ammonia, plastics, and fuels are chemical products, and no amount of clean electricity decarbonizes them until there is a way to turn electrons into chemical bonds.

**Electrochemistry is that way.** It is the only general-purpose interface between the electrical world and the chemical world. Three application families make the point.

**Water electrolysis.** Pass current through water and it decomposes into hydrogen and oxygen. The hydrogen is a storable fuel and a chemical feedstock; combined with nitrogen it becomes ammonia, and ammonia is fertilizer. Thermodynamics sets a floor on the voltage required — **1.23 V at standard conditions**, a number Chapter 2 derives from the free energy of water formation. Real electrolyzers run well above that floor, and the excess is almost entirely a **materials problem**: the oxygen-evolving electrode is slow, and making it faster without making it expensive or unstable is one of the central open questions in the field. This is precisely the territory of the [OER Computational Chemistry](<../../MI/oer-computational-chemistry/index.html>) series.

**Carbon dioxide reduction.** Instead of reducing protons to hydrogen, reduce carbon dioxide to carbon monoxide, formate, ethylene, or alcohols. In principle this closes the carbon loop: emissions become feedstock. In practice it is far harder than water splitting, for a reason that Section 1.2 already hinted at. Producing hydrogen takes two electrons. Producing ethylene from carbon dioxide takes twelve, along with a specific sequence of proton transfers and carbon-carbon bond formation, and the competing hydrogen reaction is always available as an escape route. **Selectivity, not just activity, is the bottleneck** — the catalyst must not merely be fast, it must be fast at one thing out of many possibilities.

**Batteries.** A rechargeable battery is a device for storing electricity as chemistry and getting it back. Its usefulness is governed by three quantities that electrochemistry supplies directly: the **voltage**, set by the thermodynamics of the two electrode reactions; the **capacity**, set by how much material can be cycled and by Faraday's laws; and the **rate**, set by kinetics and transport. Chapter 5 revisits batteries once we have the tools to read all three at once.

### 📚 The Common Structure Behind All Three

Different as they look, these applications share one architecture, and it is the architecture of this entire series.

| Question | What sets the answer | Covered in |
|---|---|---|
| What voltage does the reaction require or deliver? | Thermodynamics (\\(\Delta G\\), electrode potentials) | Chapter 2 |
| How much extra voltage does it cost in practice? | Kinetics (overpotential, catalysis) | Chapter 3 |
| How much product per unit of charge? | Faraday's laws and reaction selectivity | This chapter, and Chapter 5 |
| How do we measure any of this reliably? | The three-electrode cell, reference electrodes | Chapter 4 |

Read the table downward and you have the working method of an electrochemist: establish what thermodynamics demands, measure what the system actually does, attribute the difference, and then attack the largest term.

## 1.6 A Map of This Series

Five chapters, each answering one of those questions.

**Chapter 1 — Why Electrochemistry?** (this chapter) Redox as electron transfer, half-reactions, the two cell types, the anode/cathode convention, and Faraday's laws relating charge to amount of substance.

**Chapter 2 — Electrode Potentials and Thermodynamics.** The standard hydrogen electrode and the potential scale built on it; the relation \\(\Delta G = -nFE\\) that connects voltage to free energy; the Daniell cell computed from tabulated potentials; the Nernst equation and its 59 mV-per-decade concentration dependence; the origin of the 1.23 V water-splitting floor.

**Chapter 3 — Kinetics: Overpotential and Tafel Analysis.** Why a thermodynamically favourable reaction can still be immeasurably slow. Overpotential and its three contributions, exchange current density, the Butler–Volmer equation, and the Tafel slope. The central message: **a catalyst changes kinetics, never thermodynamics.**

**Chapter 4 — The Electrochemical Interface.** What actually exists at the surface of a charged electrode: the electrical double layer, why potential is only meaningful relative to a reference, why serious measurements use three electrodes, how to read a cyclic voltammogram, and what \\(iR\\) correction is for.

**Chapter 5 — Applications: From Electrolysis to Batteries.** Water electrolysis broken into its voltage budget, carbon dioxide electrolysis and the selectivity problem, and batteries reread through both thermodynamics and kinetics.

The dependencies are strictly linear: each chapter uses the previous one. Chapter 2 is the most important to get right, because the thermodynamic framework it builds is the reference point against which every real measurement in Chapters 3 to 5 is judged.

## 1.7 Hands-On: Faraday's Laws, or Charge as a Reagent

We finish with the one quantitative result that belongs in this chapter, and it is arguably the most practically useful equation in the subject.

Faraday's two laws, in modern language, say something simple: **the amount of substance transformed at an electrode is proportional to the charge passed, and the constant of proportionality is fixed by the electron count of the half-reaction.** In one equation,

\\[
n_{\text{product}} = \frac{Q}{nF} = \frac{I \cdot t}{nF}
\\]

where \\(Q\\) is charge in coulombs, \\(I\\) is current in amperes, \\(t\\) is time in seconds, \\(n\\) is the electrons per formula unit, and \\(F\\) is the **Faraday constant, 96485 C/mol** — the charge carried by one mole of electrons.

The conceptual step worth pausing on: this equation lets you treat **charge as a reagent**. In ordinary chemistry you measure out reactants by mass or volume. In electrochemistry one of the reactants is delivered by a wire, and you meter it out with an ammeter and a clock. That is an unusually convenient kind of control — currents can be set precisely, changed instantly, and integrated exactly.

The code below computes the standard example — how much copper 1 ampere deposits in 1 hour — and then uses the same machinery to make three further points: that the electron count \\(n\\) is what distinguishes different products, that hydrogen is expensive in charge terms for reasons that are purely stoichiometric, and that the Faraday constant is nothing more mysterious than Avogadro's number of elementary charges.

```python
import numpy as np

# ---------------------------------------------------------------
# Faraday's laws of electrolysis: charge in, substance out.
#
# Fixed inputs (all standard constants or IUPAC atomic weights):
#   F = 96485 C/mol       Faraday constant
#   M(Cu) = 63.55 g/mol   molar mass of copper
#   M(H2) = 2.016 g/mol   molar mass of hydrogen gas
#   M(Al) = 26.98 g/mol   molar mass of aluminium
# Everything printed below is arithmetic on those numbers.
# ---------------------------------------------------------------
F = 96485.0  # C/mol

# --- 1. The headline case: copper plating at 1 A for 1 hour -----
current = 1.0     # ampere
time_s = 3600.0   # one hour
charge = current * time_s  # coulomb

n_cu = 2       # Cu(2+) + 2 e- -> Cu
M_cu = 63.55   # g/mol

mol_electrons = charge / F
mol_cu = mol_electrons / n_cu
mass_cu = mol_cu * M_cu

print("Step 1: copper deposited by 1.0 A for 1 hour")
print(f"  charge Q = I * t          = {charge:.0f} C")
print(f"  moles of electrons  Q/F   = {mol_electrons:.6f} mol")
print(f"  moles of Cu  (n = 2)      = {mol_cu:.6f} mol")
print(f"  mass of Cu                = {mass_cu:.4f} g")
print()

# --- 2. The same charge, three different products ---------------
# The ONLY thing that changes is n, the electrons per formula unit.
print("Step 2: the same 3600 C, spent on three different reactions")
print(f"{'product':>10} {'n (e- per unit)':>16} {'moles':>12} {'mass (g)':>12}")
print("-" * 54)
for name, n, M in [("Cu", 2, 63.55), ("H2", 2, 2.016), ("Al", 3, 26.98)]:
    moles = charge / (n * F)
    print(f"{name:>10} {n:16d} {moles:12.6f} {moles * M:12.4f}")
print()

# --- 3. Scaling up: how much charge does 1 kg of product cost? ---
print("Step 3: charge required to make 1 kg of each product")
print(f"{'product':>10} {'charge (C)':>14} {'A.h':>12} {'kWh at 2.0 V':>14}")
print("-" * 54)
CELL_VOLTAGE = 2.0  # a round illustrative operating voltage, not a measurement
for name, n, M in [("Cu", 2, 63.55), ("H2", 2, 2.016), ("Al", 3, 26.98)]:
    moles = 1000.0 / M
    q = moles * n * F
    print(f"{name:>10} {q:14.3e} {q / 3600:12.1f} {q * CELL_VOLTAGE / 3.6e6:14.2f}")
print()

# --- 4. Faraday's law is linear: doubling current doubles mass ---
currents = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
masses = currents * time_s / (n_cu * F) * M_cu
print("Step 4: copper mass after 1 hour vs current (linear in I)")
print(f"{'I (A)':>8} {'mass Cu (g)':>14}")
print("-" * 24)
for i, m in zip(currents, masses):
    print(f"{i:8.1f} {m:14.4f}")
print()

# --- 5. One electron at a time -----------------------------------
e = 1.602176634e-19  # C, exact SI definition of the elementary charge
N_A_derived = F / e
print("Step 5: F is just Avogadro's number of elementary charges")
print(f"  F / e = {N_A_derived:.4e} per mol   (Avogadro's number)")
```

**Output:**

```
Step 1: copper deposited by 1.0 A for 1 hour
  charge Q = I * t          = 3600 C
  moles of electrons  Q/F   = 0.037311 mol
  moles of Cu  (n = 2)      = 0.018656 mol
  mass of Cu                = 1.1856 g

Step 2: the same 3600 C, spent on three different reactions
   product  n (e- per unit)        moles     mass (g)
------------------------------------------------------
        Cu                2     0.018656       1.1856
        H2                2     0.018656       0.0376
        Al                3     0.012437       0.3356

Step 3: charge required to make 1 kg of each product
   product     charge (C)          A.h   kWh at 2.0 V
------------------------------------------------------
        Cu      3.037e+06        843.5           1.69
        H2      9.572e+07      26588.7          53.18
        Al      1.073e+07       2980.1           5.96

Step 4: copper mass after 1 hour vs current (linear in I)
   I (A)    mass Cu (g)
------------------------
     0.5         0.5928
     1.0         1.1856
     2.0         2.3711
     5.0         5.9279
    10.0        11.8557

Step 5: F is just Avogadro's number of elementary charges
  F / e = 6.0221e+23 per mol   (Avogadro's number)
```

**Reading the result.** Four observations, in increasing order of importance.

  * **One ampere for one hour deposits about 1.19 grams of copper.** That is the number to keep in your head as a sense of scale. An ampere is a substantial current for a benchtop experiment, an hour is a long experiment, and the product is a gram — roughly a fifth of a teaspoon of metal. Electrochemistry moves material slowly unless you use large electrodes and large currents. Industrial electroplating and electrowinning operate at thousands of amperes for exactly this reason.

  * **Copper and hydrogen consume identical charge and give wildly different masses.** Both are two-electron products, so 3600 C gives 0.018656 mol of each. But a mole of copper weighs 63.55 g and a mole of hydrogen gas weighs 2.016 g, so the masses differ by more than a factor of thirty. Aluminium changes the picture the other way: it takes three electrons per atom, so the same charge gives only 0.012437 mol. **The electron count \\(n\\) and the molar mass pull in opposite directions**, and neither can be ignored.

  * **Hydrogen is cheap by mass and expensive by charge.** Making one kilogram of hydrogen requires \\(9.57 \times 10^{7}\\) coulombs, about 26589 ampere-hours — more than thirty times the charge that one kilogram of copper needs. This is not a deficiency of any particular technology. It is stoichiometry: hydrogen's molar mass is tiny, so a kilogram of it is an enormous number of moles, and every mole costs two moles of electrons. Any discussion of green hydrogen economics starts from this arithmetic, and no catalyst improvement can change it.

  * **The Faraday constant is not a new constant.** Dividing \\(F\\) by the elementary charge gives \\(6.0221 \times 10^{23}\\) per mole, which is Avogadro's number. \\(F\\) is simply the charge of one mole of electrons, and Faraday's laws are, in modern terms, the statement that electrons are counted one at a time and that the stoichiometric coefficient tells you how many go into each product molecule. The laws were established in the 1830s, decades before anyone knew the electron existed — which makes them one of the better examples in science of a correct quantitative law preceding its own explanation.

One further use of this code is worth noting because it appears constantly in real work. If you measure the current, integrate it to get charge, and then weigh or analyze how much product you actually obtained, the ratio of the two is the **Faradaic efficiency** — the fraction of the charge that went where you wanted it to go. For copper plating from a clean bath this is often close to one. For carbon dioxide reduction, where hydrogen evolution competes at every moment, it can be far lower, and reporting it is mandatory. Try modifying Step 1 to accept a measured mass and print the implied efficiency; that is a ten-line change and it turns the script into a laboratory tool.

### 🎯 Exercise Problems

  1. **Classify the reactions.** For each of the following, decide whether it is a redox reaction and, if so, write the two half-reactions: (a) \\(\mathrm{HCl} + \mathrm{NaOH} \to \mathrm{NaCl} + \mathrm{H_2O}\\), (b) \\(\mathrm{Fe} + 2\,\mathrm{HCl} \to \mathrm{FeCl_2} + \mathrm{H_2}\\), (c) \\(\mathrm{AgNO_3} + \mathrm{NaCl} \to \mathrm{AgCl} + \mathrm{NaNO_3}\\), (d) \\(2\,\mathrm{H_2O} \to 2\,\mathrm{H_2} + \mathrm{O_2}\\). For each redox case, state which species is oxidized and how many electrons are transferred per formula unit.

  2. **Balance a hard half-reaction.** Using the four-step procedure of Section 1.2, balance the reduction of dichromate, \\(\mathrm{Cr_2O_7^{2-}} \to \mathrm{Cr}^{3+}\\), in acidic solution. State \\(n\\). Then rewrite the same half-reaction for basic solution.

  3. **The naming trap.** A lead-acid car battery is being charged by an alternator. Identify which electrode is the anode, which is the cathode, and which is connected to the positive terminal of the alternator. Then answer the same three questions for the same battery while it is starting the engine. Explain in two sentences why two of your six answers changed and four did not.

  4. **Faraday in reverse.** An electroplating run passes a steady 2.5 A for 45 minutes through a nickel bath (\\(\mathrm{Ni}^{2+} + 2e^{-} \to \mathrm{Ni}\\), molar mass 58.69 g/mol). Compute the expected mass of nickel. The part actually gained 1.90 g. Compute the Faradaic efficiency and propose two physical explanations for the shortfall.

  5. **Sizing an electrolyzer.** Using Step 3 of the code as a starting point, estimate the current required for a water electrolyzer that produces 1 kg of hydrogen per hour. Comment on what that number implies about the physical size of the cell, and on why industrial electrolyzers are built as stacks of many cells in series rather than one enormous cell.

  6. **Where the salt bridge matters.** Suppose you build the Daniell cell of Section 1.3 but forget the salt bridge. Predict what the voltmeter reads at the instant of connection and one second later, and explain the difference in terms of charge accumulation. Then explain why the salt bridge does not simply short-circuit the cell.

## Summary

Electrochemistry begins with a structural trick rather than a new kind of chemistry. A **redox reaction** — one in which electrons transfer from a species that is **oxidized** (loses electrons) to one that is **reduced** (gains electrons) — normally happens on contact and releases its energy as heat. Separate the two partners in space, connect them with a wire for electrons and a salt bridge for ions, and the same reaction becomes a measurable, controllable electric current.

Writing the two halves separately as **half-reactions**, with electrons explicit, gives the subject its modularity: a few hundred tabulated halves generate every pairing of interest. The electron count \\(n\\) of a half-reaction is not incidental — it appears in every quantitative relation that follows, and it is why a two-electron process like hydrogen evolution and a twelve-electron process like ethylene formation from carbon dioxide present entirely different difficulties.

Cells come in two kinds. A **galvanic cell** runs a spontaneous reaction and delivers electrical work; an **electrolytic cell** consumes electrical work to drive a reaction uphill. A rechargeable battery is both, in alternation. Because polarity flips between the two modes while the chemistry does not, the naming convention must be anchored to chemistry: **the anode is where oxidation happens and the cathode is where reduction happens**, always, in every device. All potentials in this series follow the **IUPAC reduction-potential convention**, which Chapter 2 uses to compute cell voltages by simple subtraction.

The subject matters now because renewable electricity is abundant and intermittent while industrial demand is for fuels and materials, and **electrochemistry is the general-purpose interface between the two**. Water electrolysis, carbon dioxide reduction, and rechargeable batteries are all limited by materials rather than by principle — by how much extra voltage a real electrode demands, and by whether it produces the product you asked for.

**Faraday's laws** supply the quantitative anchor: \\(n_{\text{product}} = It/(nF)\\), with \\(F = 96485\\) C/mol. Our code showed that 1 A for 1 hour deposits **1.19 g of copper**, that the same 3600 C yields the same 0.018656 mol of any two-electron product regardless of what it weighs, that one kilogram of hydrogen demands about **26589 A·h** of charge for purely stoichiometric reasons, and that \\(F\\) divided by the elementary charge returns Avogadro's number — the laws are electron counting, established in the **1830s**, decades before the electron was known.

Chapter 2 supplies the missing half of the picture. Faraday's laws tell you *how much* product a given charge makes; they say nothing about *what voltage* it takes to make it, or whether the reaction runs on its own at all. For that we need the **standard hydrogen electrode**, the table of standard potentials built on it, the bridge \\(\Delta G = -nFE\\) between voltage and free energy, and the **Nernst equation** that tells us how potential responds to concentration. With those in hand we will build the Daniell cell's 1.10 V from two tabulated numbers, and derive the 1.23 V that water splitting can never go below.

[← Series Top](<index.html>) [Chapter 2: Electrode Potentials and Thermodynamics →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
