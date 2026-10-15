---
title: "Chapter 5: Real Fluids and Equations of State"
chapter_title: "Chapter 5: Real Fluids and Equations of State"
subtitle: From Ideal Gas to Process Simulator
reading_time: 20-25 minutes
difficulty: Intermediate
code_examples: 1
exercises: 3
version: 1.0
created_at: 2026-08-23
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/tGOpNey5U9E?start=2925"
    title="Chemical Engineering Thermodynamics Ch.5: Real Fluids and Equations of State"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

# Chapter 5: Real Fluids and Equations of State

This final chapter replaces the ideal gas law with the equations of state that real process simulators use, and shows why the thermodynamic model buried in a simulator's settings dialog quietly determines whether its answer is worth anything.

**From Ideal Gas to Process Simulator**

## Learning Objectives

By completing this chapter, you will be able to:

  * ✅ Explain when the ideal gas law is adequate and when it fails
  * ✅ Define the compressibility factor Z and interpret values above and below 1
  * ✅ Describe what the van der Waals parameters a and b represent
  * ✅ Name the cubic equations of state used in industry and state the properties they need
  * ✅ Choose between a cubic EOS and an activity-coefficient model for a given system
  * ✅ Explain why physics-based simulators, digital twins, and soft sensors depend on a thermodynamic model

**Reading Time**: 20-25 minutes **Code Examples**: 1 **Exercises**: 3

* * *

## 5.1 Where the Ideal Gas Fails

Every chapter so far has leaned on the ideal gas law:

$$ PV = nRT $$

It rests on two assumptions: molecules occupy no volume, and they exert no forces on one another. Both are excellent approximations when molecules are far apart — **low pressure and high temperature** — which covers a great deal of introductory work and a fair amount of real plant operation.

Both assumptions weaken under exactly the conditions engineers care most about. Near condensation, attractive interactions become decisive; at high pressure, molecules are packed closely enough that their own volume matters. That is not an exotic corner of the operating map. It covers **compressor suction and discharge wherever the gas is dense or near condensation** (dilute service is another matter), the **150–300 bar of an ammonia synthesis loop** (Chapter 4), and **liquefaction and most high-pressure or phase-equilibrium duties in natural-gas processing**. Using $PV = nRT$ there can produce errors far outside any acceptable tolerance.

The standard way to keep score is the **compressibility factor**:

$$ Z = \frac{PV}{nRT} $$

$Z$ is the ratio of the real molar volume to the ideal one, so $Z = 1$ means the molar volume equals the ideal-gas value at that state — it does not by itself certify ideal enthalpies, entropies, or fugacities — and any departure measures how badly the ideal gas law is lying about volume. Two regimes matter:

- **$Z < 1$** — attraction dominates. Molecules pull on each other, so the real gas is *more* compact than ideal. This is common for many gases at temperatures where attraction governs their moderate-pressure departures; hydrogen and helium at ambient temperature, above their Boyle temperatures, behave the other way.
- **$Z > 1$** — repulsion dominates. At very high pressure the finite size of the molecules themselves resists further compression, and the gas is *less* compact than ideal.

The table below shows methane at 300 K, computed with the code in Section 5.3:

| Pressure | Z | Interpretation |
|---|---|---|
| 1 bar | 0.998 | Effectively ideal |
| 50 bar | 0.924 | Attraction; 8% error if ignored |
| 150 bar | 0.851 | Lowest $Z$ of those tabulated |
| 300 bar | 0.951 | Repulsion pushing back |
| 400 bar | 1.070 | Repulsion dominant, Z > 1 |

An 8% volume error at 50 bar matters for volumetric flow, vessel sizing, inventory, and compressor work — but it does not carry across into each of them one-for-one: inventory scales inversely with molar volume, vessel sizing carries other criteria, and compressor work depends on the whole compression path and on caloric properties, not on a single-state volume. Each has to be evaluated with its own balance. The question is no longer whether to correct for non-ideality, but how.

## 5.2 The van der Waals Insight

A landmark answer came from Johannes Diderik van der Waals in his **1873** doctoral thesis, work that later won him the Nobel Prize. His idea was to patch the ideal gas law with one term for each failed assumption:

$$ \left(P + \frac{a}{V_m^2}\right)\left(V_m - b\right) = RT $$

where $V_m$ is molar volume.

| Parameter | Corrects for | Physical meaning |
|---|---|---|
| **a** | Intermolecular attraction | Molecules pull inward, so the pressure the container feels is *lower* than the repulsive-reference value $RT/(V_m - b)$; add $a/V_m^2$ back |
| **b** | Finite molecular volume | Molecules cannot be compressed into zero space; the free volume is $V_m - b$, not $V_m$ |

Both parameters are substance-specific, and both can be estimated from the critical temperature and pressure through the model's critical-point conditions — though they can equally be fitted to other property data.

What makes this more than a curve fit is what falls out of it. Because the equation is **cubic in volume**, below the critical temperature it can return three real roots for a single pressure — the largest vapor-like, the smallest liquid-like, the middle one physically unstable; which of the outer two is stable, or whether both coexist, follows from comparing fugacities. In other words, a single equation written for a gas spontaneously predicts **condensation**, a **liquid phase**, and a **critical point** above which the two phases become indistinguishable. That was a genuine discovery, and it is why van der Waals is the ancestor of everything in this chapter.

It is also generally too inaccurate for quantitative industrial property work. Modern practice keeps the structure — a repulsion term and an attraction term, cubic in volume — and replaces the details.

## 5.3 Cubic Equations of State in Practice

Two descendants dominate industrial process simulation:

- **Soave–Redlich–Kwong (SRK)**, published by Soave in **1972**, modifying the 1949 Redlich–Kwong equation
- **Peng–Robinson (PR)**, published in **1976**

Both remain cubic in volume, which matters practically: a cubic can be solved fast and reliably for millions of flash calculations, and its multiple roots keep the vapor/liquid behavior that van der Waals discovered. Both take exactly three inputs per pure component — the **critical temperature $T_c$**, the **critical pressure $P_c$**, and the **acentric factor $\omega$**, defined from the reduced saturation pressure as $\omega = -\log_{10}(P^{sat}/P_c) - 1$ at $T/T_c = 0.7$ and read heuristically as a measure of non-sphericity. In practice PR often predicts hydrocarbon liquid densities better than SRK — neither ordinary cubic is reliable close to the critical point without validation — SRK has a long tradition in gas processing, and for many hydrocarbon systems the two agree closely away from the difficult regions, though liquid densities and phase boundaries can still differ materially.

Behind this sits the **principle of corresponding states**: when compared at the same **reduced conditions**

$$ T_r = \frac{T}{T_c} \qquad P_r = \frac{P}{P_c} $$

different fluids behave remarkably similarly. Both reduced coordinates have to match: methane at $T_r = 1.5$ looks much like nitrogen at $T_r = 1.5$ **only if** their reduced pressures $P_r$ match too, and even then their differing acentric factors leave a residual difference. This is why one equation with three parameters can cover hundreds of compounds, and why the acentric factor exists: it adjusts corresponding-states behaviour using a substance's reduced vapour pressure, non-sphericity being the usual informal reading of it.

The code below computes $Z$ for methane from the SRK equation by solving the cubic numerically.

```python
import numpy as np

def srk_Z(T, P, Tc, Pc, omega):
    """Compressibility factor Z from the Soave–Redlich–Kwong EOS."""
    Tr, Pr = T / Tc, P / Pc                       # reduced temperature and pressure
    m = 0.480 + 1.574 * omega - 0.176 * omega**2  # Soave's acentric correction
    alpha = (1.0 + m * (1.0 - np.sqrt(Tr)))**2    # temperature dependence of attraction
    A = 0.42748 * alpha * Pr / Tr**2              # dimensionless attraction term
    B = 0.08664 * Pr / Tr                         # dimensionless volume term
    # Cubic in Z:  Z^3 - Z^2 + (A - B - B^2) Z - A*B = 0
    roots = np.roots([1.0, -1.0, A - B - B**2, -A * B])
    real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
    real = real[real > B]                         # discard unphysical roots
    return real[-1]                               # vapor root = largest real root

# Methane: Tc = 190.6 K, Pc = 45.99 bar, omega = 0.011
Tc, Pc, omega = 190.6, 45.99, 0.011

print(f"{'T (K)':>7} {'P (bar)':>8} {'Tr':>6} {'Pr':>6} {'Z (vapor)':>10}")
for T, P in [(300.0, 1.0), (300.0, 50.0), (300.0, 150.0), (300.0, 300.0), (300.0, 400.0)]:
    print(f"{T:7.1f} {P:8.1f} {T/Tc:6.3f} {P/Pc:6.3f} {srk_Z(T, P, Tc, Pc, omega):10.4f}")

#   T (K)  P (bar)     Tr     Pr  Z (vapor)
#   300.0      1.0  1.574  0.022     0.9983
#   300.0     50.0  1.574  1.087     0.9238
#   300.0    150.0  1.574  3.262     0.8508
#   300.0    300.0  1.574  6.523     0.9511
#   300.0    400.0  1.574  8.698     1.0703
```

At 300 K and 50 bar, SRK gives **Z = 0.924**: methane occupies about 8% less volume than the ideal gas law predicts. The run also reproduces the qualitative arc of Section 5.1 — $Z$ falls as attraction takes hold, bottoms out, then climbs past 1 as repulsion wins.

The same equation of state yields more than a volume. From it one computes each species' **fugacity coefficient** $\phi_i$ — an effective-pressure correction factor, defined so that the fugacity of Chapter 3 is $f_i = \phi_i y_i P$ and $\phi_i \to 1$ as the gas becomes ideal. That is what makes equilibrium calculations survive high pressure. The equilibrium constant of Chapter 4, written there in partial pressures alone, strictly carries a $K_\phi = \prod_i \phi_i^{\nu_i}$ factor alongside it, and in the **150–300 bar ammonia loop** that factor is a real correction rather than a formality — evaluate $Z$ and the individual $\phi_i$ at the loop's actual temperature, pressure, and composition rather than reading either off a pure-component table. This is how Chapter 4's ideal-gas equilibrium arithmetic is repaired at industrial pressures: the equilibrium problem is re-solved with a validated mixture model whose fugacity coefficients depend on the composition being solved for.

## 5.4 Beyond Cubics: Choosing a Property Package

Cubic equations of state are useful engineering models for nonpolar and weakly polar molecules — hydrocarbons, nitrogen, carbon dioxide, hydrogen — across many gas and high-pressure conditions, subject to fluid- and property-specific validation (cryogenic hydrogen, near-critical CO₂, and demanding density work can need more). Conventional SRK or PR with simple mixing rules performs poorly for **strongly polar and hydrogen-bonding liquid mixtures**, where the interactions are specific and directional rather than an averaged attraction. Water, alcohols, and organic acids are exactly the cases whose full property behaviour a single averaged attraction term $a$ cannot reproduce accurately over broad conditions.

For those liquid mixtures, engineers use **activity-coefficient models** instead — **NRTL**, **UNIQUAC**, and **Wilson** are the standard names, with **UNIFAC** available when a system has no binary data of its own, provided the group-interaction parameters it needs have already been fitted from other systems. These models describe deviations from ideal-solution behavior in the liquid phase and are the usual way to reproduce an azeotrope such as ethanol-water. A cubic is not fundamentally incapable of azeotropy — with suitable mixing rules and fitted mixture parameters it can represent one — but default SRK or PR with simple mixing rules generally gives inadequate ethanol-water VLE.

```mermaid
flowchart TD
    A["What am I modeling?"] --> B["Hydrocarbons, gases,<br/>high pressure, supercritical"]
    A --> C["Polar or hydrogen-bonding<br/>liquids at low pressure"]
    B --> D["Cubic EOS: SRK or PR"]
    C --> E["Activity model: NRTL, UNIQUAC, Wilson"]
    D --> F["Validate against binary VLE data"]
    E --> F
```

This choice — the **property package** — is an early and consequential step in setting up a simulation, and the most consequential thing most users click past. It deserves the attention because **a wrong property package is the classic silent killer of simulation accuracy**. The simulation does not crash. It converges, produces a full heat and material balance, and reports column profiles to four decimal places that are simply wrong. Model an ethanol-water column with a default cubic EOS and the azeotrope may vanish, at which point the simulator will cheerfully tell you that a modest number of trays takes a water-rich feed to an ethanol overhead beyond the azeotropic composition — which ordinary fixed-pressure binary distillation cannot do.

The practical guidance is short. Polar liquids at low pressure: an activity-coefficient model. Hydrocarbons and light gases, including at high pressure: a cubic EOS. At high pressure generally, choose an equation of state suited to the actual fluid, phases, and properties — high pressure does not cancel the association, electrolyte, or critical-region limitations above, and those systems may need a specialized formulation rather than a standard cubic. When in doubt, or when money depends on the answer, **check the model against measured binary vapor-liquid equilibrium data** for the key pair in your system before trusting any downstream result. Regressing binary interaction parameters against real data is routine professional practice, not an advanced technique.

## 5.5 Thermodynamics as the Foundation of the Digital Plant

Much of physics-based digital process engineering sits on top of this chapter. A **flowsheet simulator** is, structurally, three things: mass and energy balances, unit-operation models, and a thermodynamic package supplying every enthalpy, entropy, density, and equilibrium constant those models request. The balances are exact and the unit models are approximations — and while conservation, kinetics, transport, and equipment geometry all carry physical content too, the thermodynamics is one essential source of it, and the deepest and least visible layer.

The same is true one level up. A **digital twin** (see Chapter 5 of our *Chemical Engineering Introduction* series) is a digital representation kept synchronized with the operating plant — physics-based, empirical, or both; where it is built from a flowsheet model, it inherits that model's property package unless the package is deliberately changed. A **soft sensor** that predicts a density, a dew point, or a composition is predicting a thermodynamic quantity, and it learns the convention — and any bias — of the target data it was trained against.

This has a sharp consequence for machine learning. **An ML surrogate trained on simulator output inherits the simulator's thermodynamics — its accuracy and its errors alike.** The surrogate cannot distinguish a systematic property-package error from physics; a well-fitted one reproduces that error quickly and without any warning attached, unless independent validation or diagnostics reveal it. A surrogate that matches its training simulator to 0.1% is not accurate to 0.1%; it is *consistent* to 0.1% with a model whose own accuracy is a separate question nobody asked.

Which is the closing point of this series. Thermodynamic literacy is not about memorizing equations of state — a simulator solves those better than you will. It is about knowing that at 200 bar real-gas effects have to be evaluated against the actual temperature, composition, and tolerance rather than assumed away, that an azeotrope is a real physical constraint rather than a numerical artifact, and that a converged simulation is a statement about a model, not about a plant. **Thermodynamic literacy is what lets an engineer distrust a simulation for the right reasons.**

## 5.6 Series Summary

Chapter 1 established **energy accounting**: the First Law, enthalpy, and the fact that energy is conserved while its availability for useful work is not. Chapter 2 added **direction and limits** through entropy and the Second Law — why heat flows spontaneously from hotter to colder bodies, why a cyclic heat engine between finite temperatures cannot convert all the heat it absorbs into work, and why every real process destroys some capacity to do work. Chapter 3 applied both to **phase equilibrium**, which underpins distillation, absorption, extraction, and the other equilibrium-staged separations — though filtration, screening, centrifugation, and many kinetic or membrane separations rest on other principles. Chapter 4 extended the same equilibrium logic to **reacting systems**, showing how $\Delta G$ sets the ceiling no catalyst can raise. Chapter 5 discarded the ideal gas and arrived at the **real-fluid models** that industrial simulation actually runs on.

Taken together these five chapters describe what is possible before anyone asks what is practical — which is why thermodynamics is a foundational part of a chemical engineering education and stays useful for the whole career that follows.

Where to go next: revisit our *Chemical Engineering Introduction* series — its unit operations, reactors, and design decisions should now read differently with the thermodynamics in place — and continue to *Process Informatics Introduction* for the data layer built on top of these models and *Introduction to Bayesian Optimization* for making decisions when each experiment is expensive.

Thank you for learning with us.

## Exercises

1. **Conceptual — when is the ideal gas law acceptable?**: A colleague uses $PV = nRT$ for (a) air in a ventilation duct at 1 bar and 25 °C, (b) steam at 40 bar in a boiler drum, and (c) hydrogen at 200 bar feeding an ammonia loop. Which uses are defensible, and what single pair of numbers would you check in each case?
   *Hint*: think in reduced conditions — how far is each fluid from its own critical point, and is it near condensation?
   *Answer*: **(a) is fine.** Air at ambient conditions is far above its critical temperature and at very low reduced pressure, so $Z \approx 1.00$ and the ideal gas law is accurate to a fraction of a percent. **(b) is not defensible.** Steam at 40 bar in a boiler drum is saturated — it sits *on* the condensation line, where attractive forces dominate and $Z$ departs substantially from 1; water is also strongly hydrogen-bonded. Use steam tables or a proper property model. **(c) normally calls for a real-gas treatment as well**, not because hydrogen is polar (it is not), but because at 200 bar molecular volume matters; hydrogen characteristically shows $Z > 1$ there, so the ideal gas law *underestimates* the volume. The exercise gives no temperature or accuracy target, so quantify the error at the actual conditions before calling any shortcut acceptable. The pair to check in every case is the reduced conditions $T_r = T/T_c$ and $P_r = P/P_c$: comfortably above $T_r \approx 2$ and below $P_r \approx 0.1$, ideal is usually safe.

2. **Quantitative — compute Z from PVT data**: For a hypothetical gas, take it that at **300 K** and **50 bar** 1 mol occupies **0.46 L** (an illustrative input; no substance or experimental source is attached to it). Take $R = 0.08314$ L·bar·mol⁻¹·K⁻¹. Compute (a) the ideal molar volume, (b) $Z$, and (c) state which molecular effect dominates.
   *Hint*: $Z = PV/(nRT)$; compute $RT$ first, in L·bar/mol.
   *Answer*: $RT = 0.08314 \times 300 = 24.94$ L·bar/mol. (a) Ideal molar volume $= RT/P = 24.94/50 = \mathbf{0.499\ L/mol}$. (b) $Z = PV/(RT) = (50 \times 0.46)/24.94 = 23.0/24.94 = \mathbf{0.922}$. (c) $Z < 1$, so this gas is **more compact than ideal** and **attraction dominates** — a common situation for many gases at moderate pressure and suitable temperature, though not a universal one. Note how close this is to the SRK result for methane in Section 5.3 ($Z = 0.924$ at the same conditions). That is an illustrative comparison only: Section 5.3's figures are SRK *predictions*, not measurements, and this exercise's gas is unidentified, so the agreement shows two calculations landing in the same place, not an equation of state validated against data. Validating an EOS requires identified experimental PVT data with a stated uncertainty.

3. **Discussion — property package selection**: You must set up two simulations: (a) an **ethanol-water distillation column** at 1 atm, and (b) a **natural-gas compressor train** taking mostly methane from 10 to 100 bar. Choose a property model for each, justify it, and name one failure mode you would guard against in each case.
   *Hint*: ask whether the difficult behavior lives in the liquid phase or in a highly compressed gas phase.
   *Answer*: **(a) Ethanol-water: an activity-coefficient model — NRTL or UNIQUAC.** At atmospheric pressure the vapour can usually be approximated as ideal — subject to the accuracy required, since associating vapours are not automatically ideal at low pressure — and the main equilibrium-model difficulty is in the **liquid**, where hydrogen bonding produces strong deviations from ideal-solution behavior and a **minimum-boiling azeotrope at 95.6% ethanol by weight**. A default cubic EOS with simple mixing rules averages those specific interactions away and can miss the azeotrope entirely, yielding a column design that promises, from a water-rich feed, an ethanol overhead beyond the azeotropic composition that no tray count in an ordinary atmospheric binary column can deliver. Guard against it by **regressing the binary interaction parameters against measured ethanol-water VLE data** and confirming the model reproduces the azeotropic composition and temperature before designing anything. **(b) Natural-gas compressor train: a cubic EOS — SRK or PR.** Methane is nonpolar, so a cubic handles it well, and at typical compressor temperatures non-ideality at 100 bar is significant enough to need evaluating: as Section 5.3 shows, SRK gives $Z \approx 0.92$ for methane at 300 K and 50 bar, so ignoring compressibility misstates volumetric flow — and, through the compression path and caloric properties, the stage work, which has to be evaluated at the actual inlet, outlet, and path conditions rather than scaled from a single $Z$. The failure mode to guard against is **heavier components**: real natural gas carries ethane, propane, CO₂, and water, and the risk is **retrograde condensation** (condensation on an isothermal pressure drop, within the relevant region of a multicomponent phase envelope) **or hydrate formation** (ice-like water-gas solids) at intermediate conditions. Model the actual multicomponent mixture rather than pure methane; check the vapour-liquid envelope, and any aqueous phase, along the compression and cooling path so no stage is unknowingly operating with liquid present; and assess hydrate stability separately, since a vapour-liquid envelope says nothing about a solid phase.
