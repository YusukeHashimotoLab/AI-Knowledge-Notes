---
title: "Chapter 2: The Computational Hydrogen Electrode"
chapter_title: "Chapter 2: The Computational Hydrogen Electrode"
subtitle: "Four Proton-Coupled Electron Transfers, the Trick That Prices a Proton, and the Free-Energy Diagram That Predicts an Overpotential"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/4pvU7yXB3IU"
    title="OER Comp Chem Ch.2: The Computational Hydrogen Electrode"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/MI/oer-computational-chemistry/chapter-2.html>) | Last sync: 2026-08-18

[Materials Informatics Dojo](<../index.html>) > [Computational Chemistry of OER](<index.html>) > Chapter 2

Chapter 1 left us with a target and a difficulty. The target is the **overpotential** — the extra voltage a real anode demands beyond the 1.23 V thermodynamic price. The difficulty is that computing it appears to require simulating an electrified interface: a charged metal surface, a liquid electrolyte, solvated protons swimming about, and an electrode potential that is not a natural quantity in a quantum chemistry code at all.

This chapter presents the idea that made computational OER screening practical anyway. The **computational hydrogen electrode (CHE)**, introduced by Nørskov, Rossmeisl and co-workers in the mid-2000s, sidesteps the entire problem with a single well-chosen reference. By the end you will be able to take four numbers describing how a surface binds its intermediates, draw the free-energy diagram, and read a theoretical overpotential straight off it — with code that enforces the one thermodynamic constraint no catalyst can escape.

## 2.1 The Four Steps

OER on a surface does not happen in one concerted event. It proceeds through a sequence of **proton-coupled electron transfers (PCETs)**: each step removes one proton and one electron and leaves behind a different oxygen-containing species bound to the active site.

Write \\(\ast\\) for a clean active site on the surface, and \\(\ast\text{OH}\\), \\(\ast\text{O}\\), \\(\ast\text{OOH}\\) for that site carrying a hydroxyl, an oxygen atom, and a hydroperoxo group respectively. The standard four-step mechanism in the acidic convention is:

\\[ \text{(1)}\quad \ast + \text{H}_2\text{O} \;\longrightarrow\; \ast\text{OH} + \text{H}^+ + e^- \\]

\\[ \text{(2)}\quad \ast\text{OH} \;\longrightarrow\; \ast\text{O} + \text{H}^+ + e^- \\]

\\[ \text{(3)}\quad \ast\text{O} + \text{H}_2\text{O} \;\longrightarrow\; \ast\text{OOH} + \text{H}^+ + e^- \\]

\\[ \text{(4)}\quad \ast\text{OOH} \;\longrightarrow\; \ast + \text{O}_2 + \text{H}^+ + e^- \\]

Add the four together. The three intermediates cancel — each is produced by one step and consumed by the next — and the site \\(\ast\\) is regenerated. What remains is

\\[ 2\text{H}_2\text{O} \;\longrightarrow\; \text{O}_2 + 4\text{H}^+ + 4e^- \\]

which is exactly the OER half-reaction from Chapter 1. The mechanism is a *decomposition* of that half-reaction, not an addition to it, and this fact will shortly become the constraint that disciplines everything else.

### 📚 Reading the Mechanism Physically

Three observations are worth making before we start computing.

**Step 3 is where the O–O bond forms.** Steps 1 and 2 progressively strip hydrogen from a water molecule adsorbed at the site. Step 3 brings a *second* water molecule in and joins its oxygen to the one already there, creating the \\(\ast\text{OOH}\\) species. This is the chemically hardest thing OER has to do, and it is why the reaction needs a third intermediate at all.

**Every step transfers exactly one proton and one electron.** This is not a coincidence of notation; it is the property that CHE exploits. Because the counts are identical in every step, an applied potential acts on all four steps in the same way.

**The site returns to where it started.** After step 4 the surface is clean again and ready for another turnover. A catalyst is, by definition, something that comes back — and the closure of this cycle is what lets us treat the four \\(\Delta G\\) values as a complete description of one turnover.

## 2.2 The Problem CHE Solves

To compute the free energy of step 2, say, you need the free energy of the products minus that of the reactants:

\\[ \Delta G_2 = \left[G(\ast\text{O}) + \mu(\text{H}^+) + \mu(e^-)\right] - G(\ast\text{OH}) \\]

The surface terms \\(G(\ast\text{O})\\) and \\(G(\ast\text{OH})\\) are ordinary electronic structure calculations — hard work, but standard work. The trouble is the other two terms.

**The proton.** \\(\mu(\text{H}^+)\\) is the chemical potential of a proton in solution. Getting it from first principles means simulating a proton solvated in liquid water, with the hydrogen-bond network reorganizing around it, averaged over configurations — a research project in itself, whose error bars would swamp the differences we are trying to resolve. **The electron.** \\(\mu(e^-)\\) is the chemical potential of an electron in the electrode, which depends on the applied potential — a quantity a standard periodic DFT calculation has no way to set.

Both terms are hard. Neither is optional. And notice that both appear in *every one of the four steps*, always together, always exactly once.

### 📚 The Trick

That last observation is the whole idea. We never need \\(\mu(\text{H}^+)\\) and \\(\mu(e^-)\\) separately — only their **sum**. And the sum has a reference system in which it is trivially known.

Consider the hydrogen electrode at equilibrium. By definition of the reversible hydrogen electrode (RHE) scale, at \\(U = 0\\) V vs RHE the reaction

\\[ \text{H}^+ + e^- \;\rightleftharpoons\; \tfrac{1}{2}\text{H}_2(g) \\]

is at equilibrium. Equilibrium means the chemical potentials of the two sides are equal:

\\[ \mu(\text{H}^+) + \mu(e^-) \;=\; \tfrac{1}{2}\,\mu(\text{H}_2) \\]

And \\(\mu(\text{H}_2)\\) is the free energy of a hydrogen **molecule in the gas phase** — one of the easiest calculations in all of quantum chemistry. No solvation, no charge, no electrode. The awkward pair of terms has been replaced by half of a gas-phase molecule.

**Then the potential.** Each of the four steps releases one electron into the electrode. Raising the electrode potential to \\(U\\) lowers the energy of that electron by \\(eU\\), so every step's free energy shifts by the same amount:

\\[ \Delta G_i(U) \;=\; \Delta G_i(U=0) - eU \\]

This is the second half of CHE and it is remarkably cheap: you compute each step *once*, at \\(U = 0\\), and then the entire potential dependence is a subtraction. No recalculation, no charged supercell, no electrolyte model. The applied potential enters as arithmetic.

### 📚 What Goes Into ΔG in Practice

The \\(\Delta G\\) values used below are free energies, not raw electronic energies, and the difference matters. In a real study each step's \\(\Delta G\\) is assembled from several contributions:

  * **DFT electronic energies** of the surface with and without each adsorbate, and of the gas-phase reference molecules.
  * **Zero-point energy (ZPE)** corrections. Bound O–H and O–O vibrations have substantial zero-point energies, and they differ between \\(\ast\text{OH}\\), \\(\ast\text{O}\\) and \\(\ast\text{OOH}\\), so they do not cancel.
  * **Entropy** terms. Adsorbed species have far less entropy than the gas-phase molecules they came from, so \\(-T\Delta S\\) is a real contribution, especially for steps that release O₂.

These corrections are obtained from a vibrational analysis of each adsorbed species and from standard thermochemistry for the gas-phase references. **This chapter will not quote values for them** — they depend on the surface, the adsorbate, the functional and the vibrational treatment, and inventing representative numbers would be exactly the kind of false precision this series avoids. What you should take away is structural: the four numbers we feed the code below are *corrected free energies*, and a study that reported raw electronic energies as if they were free energies would be making a real error, not a cosmetic one.

## 2.3 The Free-Energy Diagram

With CHE in hand, one turnover is described by four numbers: \\(\Delta G_1, \Delta G_2, \Delta G_3, \Delta G_4\\) at \\(U = 0\\). Plotting their running sum gives the **free-energy diagram** — a staircase whose steps are the four PCETs.

Now the constraint promised earlier. Section 2.1 showed that the four steps sum to the overall OER half-reaction. The free energy of that overall reaction is fixed by thermodynamics — it is the water-splitting free energy from Chapter 1, read per electron as 1.23 eV. Therefore, for **any** catalyst whatsoever:

\\[ \sum_{i=1}^{4} \Delta G_i(U=0) \;=\; 4 \times 1.23\ \text{eV} \;=\; 4.92\ \text{eV} \\]

Read that carefully, because it is the most important sentence in the chapter. The catalyst controls **how the 4.92 eV is divided among the four steps**. It does not control the total. A surface cannot make OER cheaper; it can only make the four instalments more evenly sized.

### 📚 From the Diagram to an Overpotential

Apply a potential \\(U\\) and every step drops by \\(eU\\). For the reaction to run downhill all the way — no thermodynamic barrier anywhere in the cycle — every step must satisfy \\(\Delta G_i - eU \le 0\\), which means

\\[ eU \;\ge\; \max_i \Delta G_i \\]

The **largest single step therefore sets the potential you must apply**. Call it \\(\Delta G_{\max}\\); the step it belongs to is the **potential-limiting step**. The potential at which the whole cycle first becomes downhill is \\(\Delta G_{\max}/e\\), and the **theoretical overpotential** is the excess of that over the thermodynamic price:

\\[ \eta \;=\; \frac{\Delta G_{\max}}{e} - 1.23\ \text{V} \\]

Two consequences follow immediately from the fixed sum.

**A perfect catalyst is a perfectly even one.** If all four steps were equal, each would be 4.92/4 = 1.23 eV, \\(\Delta G_{\max}\\) would be 1.23 eV, and \\(\eta\\) would be zero. Every real deviation from evenness makes some step larger than 1.23 eV and pushes \\(\eta\\) up.

**Improving one step is not free.** Because the total is fixed, lowering one step's \\(\Delta G\\) necessarily raises the others' sum. Catalyst design under this model is not "make everything easier" — it is a **redistribution problem**. Chapter 3 shows that the redistribution is further constrained by scaling relations between the intermediates, which is why the volcano plot exists and why \\(\eta\\) has a stubborn floor.

## 2.4 Hands-On: A Free-Energy Diagram Calculator

The code below implements the whole scheme: it takes four step free energies at \\(U = 0\\), checks the thermodynamic sum rule, shifts every step by an arbitrary applied potential, builds the cumulative profile, and reports the theoretical overpotential and the potential-limiting step.

**A word on the numbers — read this before the code.** Real DFT for catalytic surfaces is not something numpy can do, so the step energies below are **illustrative teaching values assigned to fictitious surfaces called Catalyst A, B and C**. They were chosen to sum to 4.92 eV and to make three different points; they are not measurements, not DFT results, and are **not attributable to any real material**. Nothing in this chapter ranks real catalysts. What is real is the machinery — feed it four genuine numbers and it does the genuine calculation.

```python
import numpy as np

# ---------------------------------------------------------------
# A computational hydrogen electrode (CHE) free-energy diagram.
#
# The four OER steps on an active site *, in the acidic convention:
#   1)  * + H2O    -> *OH  + H+ + e-
#   2)  *OH        -> *O   + H+ + e-
#   3)  *O + H2O   -> *OOH + H+ + e-
#   4)  *OOH       -> *  + O2 + H+ + e-
#
# Every step releases one proton and one electron, so an applied
# potential U lowers each step by exactly eU.
#
# ALL numbers below are ILLUSTRATIVE TEACHING VALUES for fictitious
# catalysts. They are not measurements and not DFT results, and they
# are not attributed to any real material.
# ---------------------------------------------------------------
E0_WATER = 1.23          # V, standard equilibrium potential (convention)
N_STEPS = 4
TOTAL_dG = N_STEPS * E0_WATER    # eV, fixed by thermodynamics

STEP_LABELS = [
    "*  + H2O -> *OH  + H+ + e-",
    "*OH      -> *O   + H+ + e-",
    "*O + H2O -> *OOH + H+ + e-",
    "*OOH     -> *  + O2 + H+ + e-",
]

CATALYSTS = {
    "Catalyst A": np.array([1.00, 1.60, 1.42, 0.90]),
    "Catalyst B": np.array([0.85, 1.75, 1.35, 0.97]),
    "Catalyst C": np.array([1.20, 1.28, 1.25, 1.19]),
}


def check_sum(dG0, name):
    """The four steps must sum to 4 x 1.23 eV, for every catalyst."""
    total = dG0.sum()
    ok = np.isclose(total, TOTAL_dG, atol=1e-9)
    print(f"  {name}: sum of steps = {total:.4f} eV "
          f"(required {TOTAL_dG:.2f} eV) -> {'OK' if ok else 'INCONSISTENT'}")
    return ok


def steps_at_U(dG0, U):
    """Step free energies at applied potential U (each step loses eU)."""
    return dG0 - U


def profile_at_U(dG0, U):
    """Cumulative free-energy profile, starting from the clean site at 0."""
    return np.concatenate(([0.0], np.cumsum(steps_at_U(dG0, U))))


def theoretical_overpotential(dG0):
    """eta = max step / e - 1.23 V. The largest step is potential-limiting."""
    dG_max = dG0.max()
    limiting = int(dG0.argmax()) + 1
    return dG_max, dG_max - E0_WATER, limiting


print("Constraint check: the sum is fixed by thermodynamics, not by the catalyst")
for name, dG0 in CATALYSTS.items():
    check_sum(dG0, name)
print()

# --- Catalyst A walked through three potentials ------------------
name = "Catalyst A"
dG0 = CATALYSTS[name]
dG_max, eta, limiting = theoretical_overpotential(dG0)
U_onset = dG_max          # the potential at which no step is uphill

print(f"{name} (ILLUSTRATIVE teaching values) — step free energies")
for i, (lab, g) in enumerate(zip(STEP_LABELS, dG0), start=1):
    mark = "  <-- potential-limiting" if i == limiting else ""
    print(f"  step {i}  {lab:30s} dG(U=0) = {g:5.2f} eV{mark}")
print(f"  largest step dG_max = {dG_max:.2f} eV  (step {limiting})")
print(f"  theoretical overpotential eta = {dG_max:.2f} - {E0_WATER:.2f} = {eta:.2f} V")
print(f"  onset potential (all steps downhill) U = {U_onset:.2f} V")
print()

print(f"{'':>10} " + " ".join(f"{'step '+str(i):>9}" for i in range(1, N_STEPS + 1))
      + f" {'uphill?':>9}")
print("-" * 62)
for U in [0.00, E0_WATER, U_onset]:
    s = steps_at_U(dG0, U)
    uphill = "yes" if s.max() > 1e-12 else "no"
    print(f"U={U:5.2f} V " + " ".join(f"{v:9.2f}" for v in s) + f" {uphill:>9}")
print()

print("Cumulative free-energy profiles (eV), state 0 = clean site *")
print(f"{'':>10} " + " ".join(f"{'G'+str(i):>9}" for i in range(N_STEPS + 1)))
print("-" * 62)
for U in [0.00, E0_WATER, U_onset]:
    p = profile_at_U(dG0, U)
    print(f"U={U:5.2f} V " + " ".join(f"{v:9.2f}" for v in p))
print()

# --- Ranking the illustrative catalysts --------------------------
print("Theoretical overpotentials of the illustrative catalysts")
print(f"{'catalyst':>12} {'dG_max (eV)':>12} {'limiting step':>14} {'eta (V)':>9}")
print("-" * 51)
for cname, g in sorted(CATALYSTS.items(), key=lambda kv: kv[1].max()):
    m, e, lim = theoretical_overpotential(g)
    print(f"{cname:>12} {m:12.2f} {lim:14d} {e:9.2f}")
print()
print("An ideal catalyst would split 4.92 eV into four equal steps of "
      f"{TOTAL_dG/N_STEPS:.2f} eV, giving eta = 0.00 V.")
```

**Output:**

```
Constraint check: the sum is fixed by thermodynamics, not by the catalyst
  Catalyst A: sum of steps = 4.9200 eV (required 4.92 eV) -> OK
  Catalyst B: sum of steps = 4.9200 eV (required 4.92 eV) -> OK
  Catalyst C: sum of steps = 4.9200 eV (required 4.92 eV) -> OK

Catalyst A (ILLUSTRATIVE teaching values) — step free energies
  step 1  *  + H2O -> *OH  + H+ + e-     dG(U=0) =  1.00 eV
  step 2  *OH      -> *O   + H+ + e-     dG(U=0) =  1.60 eV  <-- potential-limiting
  step 3  *O + H2O -> *OOH + H+ + e-     dG(U=0) =  1.42 eV
  step 4  *OOH     -> *  + O2 + H+ + e-  dG(U=0) =  0.90 eV
  largest step dG_max = 1.60 eV  (step 2)
  theoretical overpotential eta = 1.60 - 1.23 = 0.37 V
  onset potential (all steps downhill) U = 1.60 V

              step 1    step 2    step 3    step 4   uphill?
--------------------------------------------------------------
U= 0.00 V      1.00      1.60      1.42      0.90       yes
U= 1.23 V     -0.23      0.37      0.19     -0.33       yes
U= 1.60 V     -0.60      0.00     -0.18     -0.70        no

Cumulative free-energy profiles (eV), state 0 = clean site *
                  G0        G1        G2        G3        G4
--------------------------------------------------------------
U= 0.00 V      0.00      1.00      2.60      4.02      4.92
U= 1.23 V      0.00     -0.23      0.14      0.33      0.00
U= 1.60 V      0.00     -0.60     -0.60     -0.78     -1.48

Theoretical overpotentials of the illustrative catalysts
    catalyst  dG_max (eV)  limiting step   eta (V)
---------------------------------------------------
  Catalyst C         1.28              2      0.05
  Catalyst A         1.60              2      0.37
  Catalyst B         1.75              2      0.52

An ideal catalyst would split 4.92 eV into four equal steps of 1.23 eV, giving eta = 0.00 V.
```

**Reading the result.** Work down the three potentials in the tables; each says something different.

**At U = 0 V, everything is uphill.** The cumulative profile climbs monotonically to 4.92 eV — the full thermodynamic cost of one turnover, exactly as the sum rule requires. This is the diagram with no help from the power supply, and it simply says what Chapter 1 said: water does not split on its own.

**At U = 1.23 V, the endpoints balance but the middle does not.** Look at the cumulative row: it starts at 0.00 and *ends* at 0.00. That is the sum rule expressing itself again — four steps, each shifted down by 1.23 eV, subtract exactly 4.92 eV from a total of 4.92 eV. Thermodynamically the overall reaction is now at equilibrium, and if the catalyst were ideal the story would end here.

It does not, because the *path* between those endpoints is not flat. Step 2 is still +0.37 eV uphill, and the profile rises from −0.23 to +0.33 before coming back down. **The reaction cannot run at 1.23 V not because the overall thermodynamics forbids it, but because one step in the sequence still does.** This single row is the reason overpotential exists on a well-behaved catalyst at all.

**At U = 1.60 V, the last uphill step flattens.** Step 2 is exactly zero and every other step is negative — the entire cycle is downhill. That potential is \\(\Delta G_{\max}/e\\), and the gap between it and 1.23 V is the theoretical overpotential: 0.37 V for this illustrative surface.

**The ranking table shows the redistribution problem.** All three fictitious catalysts carry the same 4.92 eV total, but they divide it differently, and their overpotentials span 0.05 V to 0.52 V. Catalyst C is the near-ideal case — its four steps are 1.20, 1.28, 1.25, 1.19, close to the perfectly even 1.23 eV each — and its \\(\eta\\) is correspondingly tiny. Catalyst B has a single 1.75 eV step and pays 0.52 V for it, *even though its other three steps are the easiest of the set*. In this model a catalyst is only as good as its worst step; excellence elsewhere earns nothing.

Notice also that step 2 happens to be limiting for all three of our illustrative surfaces. That is a property of the teaching values we chose, not a law of nature — different binding patterns make different steps limiting, and identifying which step limits a given surface is one of the more actionable outputs of a real CHE study.

Two honest limitations before moving on. First, this is **pure thermodynamics**: it finds the potential at which no step is uphill, and says nothing about activation barriers, which can make a thermodynamically downhill step slow anyway. Second, the sum rule is enforced here as a *check*, but in a real calculation the four \\(\Delta G\\) values come from independent DFT calculations and their sum will miss 4.92 eV by some amount. That residual is one of the most useful diagnostics available — a large one means an error in the references, the corrections, or the surface model, and Chapter 5 returns to how it is used.

Try editing `CATALYSTS` to add a surface of your own. Give it four steps that sum to 4.92 eV and see how small you can make \\(\eta\\); then try making one step very small and watch what the constraint does to the others.

### 🎯 Exercise Problems

  1. **Closing the cycle.** Add the four PCET steps by hand and verify that \\(\ast\\), \\(\ast\text{OH}\\), \\(\ast\text{O}\\) and \\(\ast\text{OOH}\\) all cancel, leaving the OER half-reaction. Explain in one sentence why this cancellation is what forces the sum rule.

  2. **The reference, restated.** In your own words, explain why \\(\mu(\text{H}^+) + \mu(e^-) = \tfrac{1}{2}\mu(\text{H}_2)\\) holds at 0 V vs RHE, and why it is only ever the *sum* of the two chemical potentials that the CHE method needs.

  3. **Breaking the rule on purpose.** Modify `CATALYSTS` so that one entry sums to 5.20 eV instead of 4.92 eV. Confirm that `check_sum` flags it, then explain what a physically real calculation producing that sum would tell you about the calculation — not about the catalyst.

  4. **Designing the ideal case.** Using the constraint that the four steps sum to 4.92 eV, prove that \\(\Delta G_{\max} \ge 1.23\\) eV for every possible catalyst, and hence that \\(\eta \ge 0\\) always. Under what condition is equality reached?

  5. **The cost of a single bad step.** Take Catalyst C and raise its step 3 by 0.30 eV, compensating by lowering step 1 by the same amount so the sum still holds. Recompute \\(\eta\\) using the code. Which step is now limiting, and what does the result say about optimizing one intermediate at a time?

  6. **Connecting to Chapter 1.** Catalyst A's theoretical overpotential is 0.37 V. Using the energy-cost function from Chapter 1, compute the extra kWh per kilogram of hydrogen that this anode overpotential alone would add, and state clearly why this is a lower bound on the real penalty.

## Summary

OER on a surface proceeds through **four proton-coupled electron transfers** via the intermediates \\(\ast\text{OH}\\), \\(\ast\text{O}\\) and \\(\ast\text{OOH}\\), and the four steps add up exactly to the OER half-reaction — the intermediates cancel and the active site is regenerated. Computing each step's free energy appears to require the chemical potential of a solvated proton and of an electron at a set potential, neither of which a standard calculation can supply.

The **computational hydrogen electrode** removes both obstacles with one reference. Because the two awkward terms always appear together, and because \\(\text{H}^+ + e^- \rightleftharpoons \tfrac{1}{2}\text{H}_2(g)\\) is at equilibrium at 0 V vs RHE, their sum equals \\(\tfrac{1}{2}\mu(\text{H}_2)\\) — the free energy of a gas-phase hydrogen molecule. The applied potential then enters as pure arithmetic: each step shifts by \\(-eU\\), so one calculation at \\(U = 0\\) yields the diagram at every potential. In practice the \\(\Delta G\\) values assembled this way combine DFT electronic energies with zero-point energy and entropy corrections, which differ between the three intermediates and therefore do not cancel.

The **sum rule** is the discipline of the whole method: for any catalyst, \\(\sum \Delta G_i = 4 \times 1.23 = 4.92\\) eV. A surface cannot change the total, only its division into four instalments, and the **largest step is potential-limiting**, giving \\(\eta = \Delta G_{\max}/e - 1.23\\) V. Our calculator walked the fictitious Catalyst A (illustrative steps 1.00, 1.60, 1.42, 0.90 eV) through three potentials: uphill everywhere at 0 V; endpoints balanced at 1.23 V but step 2 still +0.37 eV uphill, which is precisely why 1.23 V does not suffice; and fully downhill at 1.60 V, giving \\(\eta = 0.37\\) V. Across the three illustrative surfaces \\(\eta\\) ranged from 0.05 V to 0.52 V on the same fixed 4.92 eV total — a catalyst is only as good as its worst step, and design is a **redistribution problem**, not a reduction problem.

The next chapter asks the obvious follow-up question, and finds an uncomfortable answer. If the ideal catalyst just needs four equal steps, why has nobody built one? The reason is that the binding energies of \\(\ast\text{OH}\\), \\(\ast\text{O}\\) and \\(\ast\text{OOH}\\) are not independent — they are locked together by **scaling relations** that make some divisions of the 4.92 eV physically unreachable. That constraint produces the **volcano plot** and imposes a floor on the overpotential that no amount of composition tuning within a family can break through.

[← Chapter 1: Why OER Is the Bottleneck](<chapter-1.html>) [Chapter 3: Scaling Relations and the Volcano →](<chapter-3.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
