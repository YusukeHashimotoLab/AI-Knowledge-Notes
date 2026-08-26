---
title: "Chapter 3: Multiple Reactions and Selectivity"
chapter_title: "Chapter 3: Multiple Reactions and Selectivity"
subtitle: Making the Right Product, Not Just More Product
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
    src="https://www.youtube.com/embed/GIrdjPDTjwY?start=1704"
    title="Chemical Engineering Reaction Engineering Ch.3: Multiple Reactions and Selectivity"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

# Chapter 3: Multiple Reactions and Selectivity

[Chapter 1](chapter-1.html) built the rate law and [Chapter 2](chapter-2.html) sized the reactors that exploit it. Both worked with a single reaction, where the only question worth asking was *how much* of the reactant disappeared. Real feedstocks are not so obliging: they react along several paths at once, and the engineer's job shifts from maximizing conversion to steering the molecules toward the product that pays.

**Making the Right Product, Not Just More Product**

## Learning Objectives

By completing this chapter, you will be able to:

  * ✅ Distinguish selectivity from yield and from conversion, and state which convention you are using
  * ✅ Explain why a point of selectivity is paid for twice — once in raw material and again in separation
  * ✅ Use the instantaneous selectivity ratio to choose a concentration policy for parallel reactions
  * ✅ Choose a temperature policy from the two activation energies, while recognizing that the optimum is economic rather than kinetic
  * ✅ Compute the optimal time and maximum intermediate concentration for a first-order series reaction
  * ✅ Explain why a CSTR damages an intermediate product and why semi-batch operation is the standard selectivity tool in fine chemicals

**Reading Time**: 20-25 minutes **Code Examples**: 1 **Exercises**: 3

* * *

## 3.1 Why Selectivity Beats Conversion

A reactor does not produce a product. It produces a **mixture**, and everything in that mixture becomes someone else's problem downstream. Conversion measures how much reactant left; it says nothing about where the reactant went. A reactor at 99% conversion that sends a third of its carbon to a by-product is commercially worse than one at 70% conversion that sends almost none — unconverted reactant can be recovered and recycled, whereas a by-product must be separated, then disposed of or sold at a discount.

The economics are blunt. Every point of selectivity lost is **paid twice**. First in raw material: the atoms that became the wrong molecule were bought at feedstock price and cannot be bought back. Second in separation: the by-product must now be removed to specification, which is column height, reflux, steam, and sometimes a unit operation the flowsheet would not otherwise contain. The separation train of [Mass Transfer Chapter 5](../chemical-engineering-mass-transfer/chapter-5.html) has the shape it does largely because of decisions made upstream in the reactor — a poorly selective reactor writes the specification for an expensive column, and no amount of clever distillation design undoes that.

There is a further asymmetry: conversion can usually be raised later by adding volume or a recycle loop, both capital problems with known solutions, whereas once the by-product has formed no downstream unit turns it back into feedstock.

### Definitions, and the Conventions That Vary

Three quantities do the work, and they are routinely confused.

| Quantity | Definition used here | What it answers |
|---|---|---|
| **Conversion** $X$ | reactant consumed / reactant fed | How far did the reaction go? |
| **Selectivity** $S$ | moles of desired product formed / moles of undesired product formed | Where did the reacted material go? |
| **Yield** $Y$ | moles of desired product formed / moles of reactant fed | How much product did the feed actually buy? |

**This chapter uses selectivity as the ratio of desired to undesired product**, because it then appears directly as a ratio of rates. That convention is not universal: many texts and most plant data sheets define selectivity *per mole of key reactant consumed*, which is bounded by 1 and behaves like an efficiency. In particular, [Introduction Chapter 2](../chemical-engineering-introduction/chapter-2.html) — the overview this series expands — uses the per-reactant-consumed form; this chapter uses the product ratio because it appears directly as a ratio of rates, so convert before comparing numbers between the two series. Both are defensible; the failure mode is not choosing one, but sliding between them mid-calculation. The three quantities are linked — roughly, yield is conversion times the fractional (per-reactant-consumed) selectivity — which is why a plant can raise conversion and *lose* yield, a trap examined in Section 3.4.

Distinguish also **instantaneous** selectivity, evaluated from the rates at a point in the reactor, from **overall** selectivity, integrated over the whole reactor. The first is the design lever; the second is what the plant reports.

## 3.2 Parallel Reactions: The Concentration Policy

The first canonical case is **parallel** (or competing) reactions, where one reactant has two fates:

$$ \mathrm{A} \xrightarrow{k_1} \mathrm{D} \quad (\text{desired}), \qquad r_D = k_1 C_A^{a_1} $$

$$ \mathrm{A} \xrightarrow{k_2} \mathrm{U} \quad (\text{undesired}), \qquad r_U = k_2 C_A^{a_2} $$

Divide one rate by the other and the whole design argument falls out in a single line:

$$ \frac{r_D}{r_U} = \frac{k_1}{k_2}\, C_A^{\,a_1 - a_2} $$

The rate constants set a baseline the engineer cannot move at fixed temperature. The **exponent difference $a_1 - a_2$** is the handle, because it decides whether the ratio rises or falls with concentration. Only its sign matters for the policy.

**If $a_1 > a_2$** — the desired reaction is the more strongly concentration-dependent one — high $C_A$ favors D. Keep the reactant concentrated everywhere: undiluted feed, high pressure for gas-phase systems, minimal inerts, and a reactor whose concentration profile stays high, which means **batch or plug flow**. A solvent recycle that dilutes the feed is now actively harmful.

**If $a_1 < a_2$** the policy inverts. Low $C_A$ favors D, so keep the reactant dilute: a diluted feed, low pressure in the gas phase, and a **CSTR**, whose entire contents sit at the low outlet concentration rather than passing through a high-concentration entrance region. Where dilution is unattractive because it inflates the separation load, **semi-batch** operation achieves the same end — adding A slowly to a vessel already containing everything else, so A is consumed nearly as fast as it enters and never builds up. Section 3.6 returns to this.

**If $a_1 = a_2$** the exponents cancel, the ratio is $k_1/k_2$ regardless of concentration, and no concentration policy exists. Only temperature remains.

With two reactants the logic is unchanged: each species gets its own exponent difference and its own policy, and the two can point in opposite directions — high A but low B, which is exactly what a semi-batch reactor with A charged and B fed slowly delivers.

## 3.3 The Temperature Policy

Temperature acts through the rate constants, via the Arrhenius expression of [Chapter 1](chapter-1.html):

$$ \frac{k_1}{k_2} = \frac{A_1}{A_2}\exp\!\left[-\frac{E_1 - E_2}{RT}\right] $$

The rule reads off the sign of $E_1 - E_2$ and is genuinely as simple as it looks:

| Case | Effect of raising $T$ | Policy |
|---|---|---|
| $E_1 > E_2$ (desired reaction more temperature-sensitive) | $k_1/k_2$ rises | **Run hot** |
| $E_1 < E_2$ | $k_1/k_2$ falls | **Run cold** |
| $E_1 \approx E_2$ | ratio nearly fixed | Temperature is not a selectivity lever |

The magnitudes involved are large enough to matter. With $E_1 - E_2 = 30$ kJ/mol, moving from 300 K to 350 K multiplies $k_1/k_2$ by roughly **5.6** — a selectivity change no catalyst tweak would deliver as cheaply.

The caveat is where the honesty lives. **"Run cold" costs rate**, and rate is reactor volume: a temperature that doubles selectivity while cutting the overall rate tenfold buys a purer product in a reactor several times larger. The optimum is therefore **economic, not kinetic** — it balances recovered selectivity against capital, utility duty, and throughput, and it moves with prices, so two plants running identical chemistry can rationally sit at different temperatures. Practical limits bite as well: catalyst deactivation, thermal decomposition of a heat-sensitive product, materials of construction, and for exothermic systems the runaway concerns that make "just run hotter" a safety question rather than an optimization one. Treat the Arrhenius rule as the direction of the gradient, not the location of the optimum.

## 3.4 Series Reactions: Time Becomes a Selectivity Variable

The second canonical case is **series** (consecutive) reactions, where the desired product is an intermediate that reacts onward:

$$ \mathrm{A} \xrightarrow{k_1} \mathrm{B} \xrightarrow{k_2} \mathrm{C}, \qquad \mathrm{B\ desired} $$

with both steps first order. For a batch reactor fed pure A, the closed-form solutions are standard:

$$ C_A = C_{A0}\,e^{-k_1 t}, \qquad C_B = \frac{k_1 C_{A0}}{k_2 - k_1}\left(e^{-k_1 t} - e^{-k_2 t}\right) $$

valid for $k_1 \neq k_2$, with $C_C$ following by difference. The shape of $C_B(t)$ is the entire story: it starts at zero, rises while A is plentiful, peaks, and then **falls** as the supply of A runs out while the second reaction keeps consuming what is left. Time — space time in a flow reactor — has become a selectivity variable in its own right. In [Chapter 2](chapter-2.html) more residence time was always more conversion and therefore always better; here, residence time past the peak destroys product.

Differentiating $C_B$ and setting the derivative to zero gives the optimum:

$$ t_{opt} = \frac{\ln(k_1/k_2)}{k_1 - k_2}, \qquad \frac{C_{B,max}}{C_{A0}} = \left(\frac{k_1}{k_2}\right)^{k_2/(k_2 - k_1)} $$

Note what the second expression does *not* contain: any reference to time, or to the individual rate constants. **The best achievable intermediate yield depends only on the ratio $k_1/k_2$.** No amount of reactor engineering beats it; only chemistry — a better catalyst, a different solvent, a temperature that shifts the ratio — moves that ceiling.

### Worked Example: $k_1 = 1.0$ /h, $k_2 = 0.5$ /h

$$ t_{opt} = \frac{\ln(1.0/0.5)}{1.0 - 0.5} = \frac{\ln 2}{0.5} = \frac{0.693}{0.5} \approx \mathbf{1.39\ \text{hours}} $$

$$ \frac{C_{B,max}}{C_{A0}} = \left(\frac{1.0}{0.5}\right)^{0.5/(0.5-1.0)} = 2^{-1} = \mathbf{0.50} $$

Read the two numbers together. Even with the desired step running **twice as fast** as the destructive one, the very best a perfectly operated batch reactor achieves is **half the feed converted to product**, the rest split between unconverted A and over-reacted C. That is the ceiling, and it is reached only if you stop on time.

Stopping on time is easy to get wrong, for an unintuitive reason: at $t_{opt}$ the conversion of A is only 75%. Every operator instinct, and every incentive that rewards conversion, pushes toward running longer — which converts more A and produces **less B**. By $t = 5$ h conversion is over 99%, an excellent-looking number on a shift report, while $C_B$ has collapsed to about 30% of its peak with over 84% of the carbon sitting in C. **Overcooking destroys product.** Conversion and yield are pulling against each other, and a plant optimizing the wrong one loses money while its metrics improve.

The peak is mercifully flat: at $t = 1.2$ h and again at $t = 1.6$ h, $C_B/C_{A0}$ is still about 0.495, within about 1% of the maximum. Control need not be surgical — it needs to be on the correct side of the disaster, and the disaster is downstream in time.

## 3.5 Why a CSTR Hurts an Intermediate

For a series reaction with the intermediate desired, **plug flow or batch beats a single CSTR** at the same space time. The reason is residence time distribution. A PFR gives every fluid element the same time in the reactor — every molecule of A sees exactly $t_{opt}$ and is then removed. A CSTR mixes its feed instantly into the whole vessel, so residence times are distributed exponentially — the subject of [Chapter 4](chapter-4.html): some material short-circuits to the outlet almost unreacted, while some **overstays** and is carried through to C. Both tails cost B — the short one never makes it, the long one destroys what it made.

A second penalty compounds the first. A CSTR operates entirely at its outlet composition, where B is most abundant relative to A, so the whole vessel runs continuously at the condition most favorable to the *second* reaction. A PFR spends its entrance region rich in A and nearly free of B, where the destructive step has almost nothing to work on.

Section 3.7 computes the penalty rather than asserting it, and the arithmetic shows this is not a marginal preference. The claim to carry away is **PFR or batch for a desired intermediate, unless something else forces the choice**. Where a CSTR is unavoidable — heat removal, solids handling, viscosity — a **cascade of CSTRs** recovers much of the loss, since a series of stirred tanks approaches plug-flow behavior as their number grows.

## 3.6 Semi-Batch as a Selectivity Tool

Section 3.2 identified a policy — hold $C_A$ low — without saying how to achieve it without dilution. **Semi-batch operation** is the answer: charge the reactor with everything except one reagent, then feed that reagent slowly over hours. If the feed rate stays well below the rate at which the reaction can consume it, the reagent is destroyed essentially as fast as it arrives and its concentration never builds, without a drop of added solvent.

This is not a niche technique. It is close to the default mode in **fine chemicals and pharmaceutical manufacture**, where products are complex, side reactions are expensive, and batch sizes are small enough that a slow addition is affordable in time. Part of why it is so common is that one maneuver serves several purposes: it controls selectivity, it limits the instantaneous heat release of an exothermic step (a slow feed keeps the accumulated reagent — and therefore the adiabatic temperature rise available to a runaway — small), and it gives the operator a control handle, the feed pump, that can be stopped instantly.

The trade is throughput. A reaction that could finish in twenty minutes at full concentration may be fed over eight hours to protect selectivity, with the vessel occupied throughout. Whether that is worth doing is again economic — and in fine chemicals, where product value per kilogram is high, the answer is usually yes.

## 3.7 Code Example: The Series Reaction, Quantified

```python
"""Series reaction A -> B -> C, both steps first order, B desired.
Analytical solution for a batch reactor (equivalently a PFR in space time).
k1 = 1.0 / h, k2 = 0.5 / h, feed pure A.
"""
import math

k1, k2 = 1.0, 0.5      # per hour
CA0 = 1.0              # mol/L, basis


def c_a(t):
    """Reactant A, first-order decay."""
    return CA0 * math.exp(-k1 * t)


def c_b(t):
    """Intermediate B, closed form for k1 != k2."""
    return CA0 * k1 / (k2 - k1) * (math.exp(-k1 * t) - math.exp(-k2 * t))


def c_c(t):
    """Over-reacted product C, by difference."""
    return CA0 - c_a(t) - c_b(t)


def cstr_b(tau):
    """Single CSTR at steady state, same kinetics, space time tau."""
    return CA0 * k1 * tau / ((1.0 + k1 * tau) * (1.0 + k2 * tau))


t_opt = math.log(k1 / k2) / (k1 - k2)
cb_max = CA0 * (k1 / k2) ** (k2 / (k2 - k1))
print(f"t_opt    = ln({k1/k2:.1f}) / {k1-k2:.1f} = {t_opt:.4f} h")
print(f"C_B,max  = (k1/k2)^(k2/(k2-k1)) = {cb_max:.4f} mol/L")
print(f"check by direct evaluation:      {c_b(t_opt):.4f} mol/L\n")

print(f"{'t [h]':>7} {'C_A':>8} {'C_B':>8} {'C_C':>8}  {'note':<20}")
for t in [0.5, 1.0, 1.2, t_opt, 1.6, 2.0, 3.0, 5.0]:
    note = "<-- optimum" if abs(t - t_opt) < 1e-9 else ""
    print(f"{t:7.3f} {c_a(t):8.4f} {c_b(t):8.4f} {c_c(t):8.4f}  {note:<20}")

print(f"\nOvercooking: at t = 5 h, C_B has fallen to "
      f"{100*c_b(5.0)/cb_max:.0f}% of its peak while C_A is nearly gone.")

pfr = c_b(t_opt)
cstr = cstr_b(t_opt)
print(f"\nSame space time tau = {t_opt:.4f} h:")
print(f"  PFR  (plug flow) C_B = {pfr:.4f} mol/L")
print(f"  CSTR (well mixed) C_B = {cstr:.4f} mol/L")
print(f"  CSTR delivers {100*cstr/pfr:.0f}% of the PFR yield")

tau_star = 1.0 / math.sqrt(k1 * k2)
print(f"  even at its own best tau = {tau_star:.4f} h, CSTR reaches only "
      f"{cstr_b(tau_star):.4f} mol/L")

# t_opt    = ln(2.0) / 0.5 = 1.3863 h
# C_B,max  = (k1/k2)^(k2/(k2-k1)) = 0.5000 mol/L
# check by direct evaluation:      0.5000 mol/L
#
#   t [h]      C_A      C_B      C_C  note
#   0.500   0.6065   0.3445   0.0489
#   1.000   0.3679   0.4773   0.1548
#   1.200   0.3012   0.4952   0.2036
#   1.386   0.2500   0.5000   0.2500  <-- optimum
#   1.600   0.2019   0.4949   0.3032
#   2.000   0.1353   0.4651   0.3996
#   3.000   0.0498   0.3467   0.6035
#   5.000   0.0067   0.1507   0.8426
#
# Overcooking: at t = 5 h, C_B has fallen to 30% of its peak while C_A is nearly gone.
#
# Same space time tau = 1.3863 h:
#   PFR  (plug flow) C_B = 0.5000 mol/L
#   CSTR (well mixed) C_B = 0.3431 mol/L
#   CSTR delivers 69% of the PFR yield
#   even at its own best tau = 1.4142 h, CSTR reaches only 0.3431 mol/L
```

The steady-state CSTR expression comes from a mole balance rather than an integration:

$$ \frac{C_B}{C_{A0}} = \frac{k_1 \tau}{(1 + k_1 \tau)(1 + k_2 \tau)} $$

Three things in that output deserve attention. The **optimum row** confirms the hand calculation exactly: 0.5000 mol/L of B at 1.3863 h, with A at 0.2500 — 75% conversion, not 99%. The **overcooking line** prices that missing 99%: B down to 30% of its peak, 84% of the carbon in C. And the **reactor comparison** puts a number on Section 3.5: at the same space time the CSTR gives 0.343 against the PFR's 0.500, about **69%** of plug-flow performance. Tuning does not rescue it — at its own optimal space time of 1.414 h the CSTR still reaches only 0.343, because the residence time distribution, not the space time, is what costs the product.

## 3.8 Chapter Summary

1. A reactor's real product is a **mixture**; every point of selectivity lost is paid twice, once in raw material and again in the separation train downstream
2. **Selectivity** here means desired product formed over undesired product formed; **yield** is desired product per mole of reactant fed. Conventions vary across textbooks — the per-reactant-consumed definition is equally common — so state yours and hold to it
3. For **parallel** reactions the ratio $r_D/r_U = (k_1/k_2)C_A^{a_1-a_2}$ carries the whole argument: $a_1 > a_2$ means high $C_A$, so batch or PFR with concentrated feed; $a_1 < a_2$ means low $C_A$, so CSTR, dilution, or slow semi-batch addition; $a_1 = a_2$ leaves no concentration policy
4. **Temperature**: run hot if $E_1 > E_2$, cold if $E_1 < E_2$ — but running cold costs rate and therefore reactor volume, so the optimum is economic rather than kinetic
5. For **series** reactions $\mathrm{A} \to \mathrm{B} \to \mathrm{C}$ with B desired, $C_B$ rises then falls, making time or space time a selectivity variable: $t_{opt} = \ln(k_1/k_2)/(k_1-k_2)$, $C_{B,max}/C_{A0} = (k_1/k_2)^{k_2/(k_2-k_1)}$. For $k_1 = 1.0$ /h, $k_2 = 0.5$ /h that is **1.39 h** and **0.50** — half the feed at best, and only if you stop on time. Overcooking destroys product
6. The ceiling depends **only on the ratio $k_1/k_2$**, not on the individual rate constants: reactor engineering cannot beat it, only better chemistry can
7. **PFR or batch beats a single CSTR** for an intermediate, because mixed residence times let part of the fluid overstay — 0.343 against 0.500 in the worked case, about 69%. A CSTR cascade recovers much of the loss
8. **Semi-batch** addition holds a reactant concentration low without dilution and is close to the default in fine chemicals, buying selectivity and thermal control at the cost of throughput

**Next chapter**: every design equation so far has trusted the reactor to keep its flow-pattern promise — perfect mixing in the CSTR, plug flow in the tube. Real vessels break those promises with bypassing, dead zones, and channeling, and the selectivity argument just made about mixed residence times shows how much breaking them can cost. [Chapter 4](chapter-4.html) puts the promise to the test with a tracer: the **residence-time distribution** $E(t)$, the signatures of the two ideals, the pathologies a measured curve reveals, and the tanks-in-series model that summarizes how mixed a real vessel actually is.

## Exercises

1. **Quantitative — choosing a concentration policy**: A liquid-phase reactant A forms a desired product D and an undesired product U in parallel, with $r_D = k_1 C_A^2$ and $r_U = k_2 C_A$, where $k_1/k_2 = 0.5$ L/mol at the operating temperature. (a) Write the instantaneous selectivity ratio and evaluate it at $C_A = 4$ mol/L and at $C_A = 0.5$ mol/L. (b) Convert each to the fraction of reacting A that becomes D. (c) State the reactor type and feed policy you would specify, and say what would change if the orders were reversed ($r_D = k_1 C_A$, $r_U = k_2 C_A^2$). (d) The activation energies are $E_1 = 80$ kJ/mol and $E_2 = 50$ kJ/mol. Which direction should temperature move, and by roughly what factor does $k_1/k_2$ change between 300 K and 350 K?
   *Hint*: $r_D/r_U = (k_1/k_2)C_A^{a_1-a_2}$; the fraction to D is $(r_D/r_U)/(1 + r_D/r_U)$; for (d) use $k_1/k_2 \propto \exp[-(E_1-E_2)/RT]$ with $R = 8.314$ J/(mol·K).
   *Answer*: (a) $a_1 - a_2 = 1$, so $r_D/r_U = 0.5\,C_A$: **2.0** at $C_A = 4$, **0.25** at $C_A = 0.5$ — an eightfold change in concentration gives an eightfold change in the ratio. (b) Fraction to D $= 2.0/3.0 = $ **0.667** at the high concentration and $0.25/1.25 = $ **0.20** at the low one: two thirds of the reacting A becomes product in one case, one fifth in the other, with identical chemistry. (c) Since $a_1 > a_2$, **high $C_A$ favors D**: a **batch or plug-flow reactor** with an **undiluted feed**, no diluting solvent recycle, and no CSTR, whose whole volume would sit at the low outlet concentration. With the orders reversed, $a_1 - a_2 = -1$ and every recommendation inverts — **CSTR**, diluted feed, or **semi-batch** addition. (d) $E_1 > E_2$, so **run hot**: with $E_1 - E_2 = 30$ kJ/mol the ratio changes by $\exp[-(30000/8.314)(1/350 - 1/300)] \approx \exp(1.72) \approx$ **5.6** over that 50 K rise. Whether to take it depends on whether D and the catalyst survive 350 K — the kinetics give only the direction.

2. **Quantitative — stopping on time**: A series reaction $\mathrm{A} \to \mathrm{B} \to \mathrm{C}$ has first-order steps with $k_1 = 3.0$ /h and $k_2 = 1.0$ /h, and B is the desired product. (a) Compute $t_{opt}$ and $C_{B,max}/C_{A0}$. (b) Compare with the $k_1 = 1.0$, $k_2 = 0.5$ case of Section 3.4: which feature of the kinetics controls the maximum yield, and which controls the timing? (c) A colleague proposes doubling both rate constants by raising the temperature. What happens to $t_{opt}$ and to $C_{B,max}$?
   *Hint*: $t_{opt} = \ln(k_1/k_2)/(k_1-k_2)$ and $C_{B,max}/C_{A0} = (k_1/k_2)^{k_2/(k_2-k_1)}$. For (c), substitute $k_1 \to 2k_1$, $k_2 \to 2k_2$ and see which expression changes.
   *Answer*: (a) $t_{opt} = \ln(3.0/1.0)/(3.0 - 1.0) = 1.0986/2 = $ **0.549 h**, about 33 minutes. $C_{B,max}/C_{A0} = 3^{1/(1-3)} = 3^{-0.5} = 1/\sqrt{3} = $ **0.577**. Checking directly: $\frac{3}{1-3}\left(e^{-1.648} - e^{-0.549}\right) = -1.5(0.1925 - 0.5774) = 0.577$ ✓. (b) The **ratio $k_1/k_2$ alone sets the ceiling** — it is the only thing in the $C_{B,max}$ expression — while the **difference $k_1 - k_2$ sets the timing**, through the denominator of $t_{opt}$. Here the ratio is 3 rather than 2, so the ceiling rises from 0.50 to 0.577; the difference is 2.0 rather than 0.5, so the optimum arrives sooner — by a factor of $1.386/0.549 \approx 2.5$, not the naive $2.0/0.5 = 4$, because the numerator $\ln(k_1/k_2)$ moved as well as the denominator. A faster reaction is not a more selective one. (c) Doubling both leaves the ratio at 3, so $C_{B,max}$ is **unchanged at 0.577**, while $t_{opt}$ **halves to 0.275 h** — a smaller reactor for the same throughput at no cost in yield. The caveat: "doubling both" quietly assumes equal activation energies; in general the ratio moves with temperature too, which is why the temperature question in a series system is really about the *ratio* of activation energies rather than about rate.

3. **Conceptual — why a CSTR hurts an intermediate**: A pilot plant makes an intermediate B from A via $\mathrm{A} \to \mathrm{B} \to \mathrm{C}$ in a batch reactor and achieves the expected yield. Scale-up to continuous operation is proposed, and the process engineer suggests a single CSTR sized for the same space time as the batch reaction time. (a) Explain physically why the CSTR will underperform, using the residence time distribution. (b) Give a second, independent reason based on the composition the CSTR operates at. (c) Suggest two modifications that recover most of the lost yield, and state the trade each involves. (d) Would the same objection apply if B were the *final* product of a single reaction $\mathrm{A} \to \mathrm{B}$?
   *Hint*: think about what fraction of the fluid in a stirred tank leaves almost immediately, and what fraction is still there after several space times.
   *Answer*: (a) A batch or plug-flow reactor gives every molecule the **same** time in the reactor, so all of the fluid can be removed at $t_{opt}$. A CSTR mixes its feed instantly into the bulk, producing an **exponential distribution** of residence times: some material short-circuits to the outlet nearly unreacted and never becomes B, while some remains for several space times and is carried through to C. Both tails destroy yield, and no choice of average space time removes them, because the spread is inherent to the mixing — 0.343 against 0.500 at the same space time in the worked case, and still 0.343 at the CSTR's own optimum. (b) A CSTR operates **entirely at its outlet composition**, the composition richest in B relative to A, so the whole vessel runs continuously at the condition most favorable to the destructive second step. A PFR spends its entrance region rich in A and almost free of B, approaching the unfavorable composition only near the exit. (c) A **cascade of CSTRs in series**, whose residence time distribution narrows toward plug flow as the number of tanks rises — a handful recovers most of the gap, at the cost of more vessels, agitators, and instrumentation. Or a **tubular plug-flow reactor**, which eliminates the problem rather than mitigating it, at the cost of harder heat removal, possible fouling, and difficulty with solids or viscous fluids. Since CSTRs are usually proposed for heat transfer or simplicity in the first place, the choice is rarely made on yield alone. (d) **No.** For a single reaction $\mathrm{A} \to \mathrm{B}$ extra residence time means more conversion, never less product. A CSTR still needs more volume than a PFR for the same conversion at positive order, since it operates at the low outlet concentration and therefore the low rate, but that is a **capital** penalty, not a **yield** penalty. With a single reaction, volume buys conversion; with a series reaction, volume past the optimum trades yield away for it.
