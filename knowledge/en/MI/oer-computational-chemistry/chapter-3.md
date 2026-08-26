---
title: "Chapter 3: Scaling Relations and the Volcano"
chapter_title: "Chapter 3: Scaling Relations and the Volcano"
subtitle: "Why Two of the Four Steps Refuse to Move Independently, and What That Costs Every Catalyst"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/9ngyvEnf70Q"
    title="OER Comp Chem Ch.3: Scaling Relations and the Volcano"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/MI/oer-computational-chemistry/chapter-3.html>) | Last sync: 2026-08-18

[Materials Informatics Dojo](<../index.html>) > [Computational Chemistry of OER](<index.html>) > Chapter 3

Chapter 2 handed us a machine. Feed the computational hydrogen electrode four adsorption free energies — one each for \\(\ast\text{OH}\\), \\(\ast\text{O}\\), \\(\ast\text{OOH}\\), plus the arithmetic that closes the cycle — and it returns a theoretical overpotential. The obvious next move is to turn the crank on every surface you can imagine and keep the winners.

That plan runs into a wall almost immediately, and the wall is the subject of this chapter. The three intermediates are not free to be whatever a chemist would like them to be. Two of them are bound to each other by a regularity so stubborn that it survives changes of metal, of oxide, of facet, and of coordination environment. Once you accept that regularity, a *floor* appears underneath the overpotential of every catalyst that obeys it — a number no material can go below, no matter how clever the synthesis. The purpose of this chapter is to derive that floor rather than quote it, and then to show how the same constraint collapses the whole four-dimensional search into a single number per surface.

That last collapse is the good news buried inside the bad news, and it is the door through which materials informatics walks.

> **A note on every number in this chapter**
>
> The step energies attached to "Catalyst A" through "Catalyst E" in Chapter 4, and the sweeps in this chapter, are **illustrative teaching values on fictitious surfaces**. They are chosen to make the geometry of the argument visible. They are not measurements, not DFT results, and must never be attributed to a real material. The one empirical input is the scaling constant of Section 3.2, and it is stated as an approximate, widely reproduced regularity. Everything downstream of it — the floor, the apex, the shape of the volcano — is computed by the code below.

## 3.1 Three Intermediates, Two Kinds of Bond

Recall the four-step associative mechanism from Chapter 2. In acid,

\\[ \text{H}_2\text{O} + \ast \rightarrow \ast\text{OH} + \text{H}^+ + e^- \\]
\\[ \ast\text{OH} \rightarrow \ast\text{O} + \text{H}^+ + e^- \\]
\\[ \ast\text{O} + \text{H}_2\text{O} \rightarrow \ast\text{OOH} + \text{H}^+ + e^- \\]
\\[ \ast\text{OOH} \rightarrow \ast + \text{O}_2 + \text{H}^+ + e^- \\]

Four proton-coupled electron transfers, and the sum of their free energies is fixed by the overall reaction: the four steps must add up to the free energy of splitting two waters into oxygen and hydrogen, which is \\(4 \times 1.23 = 4.92\\) eV. That total is not adjustable. It is thermodynamics, and it is the same for platinum, for an iridium oxide, and for a surface that exists only inside a simulation.

Now look at *how* each intermediate attaches to the surface. \\(\ast\text{OH}\\) sits on the surface through a single oxygen atom, with a hydrogen hanging off it. \\(\ast\text{OOH}\\) also sits on the surface through a single oxygen atom, with an \\(\text{OOH}\\) tail hanging off it. \\(\ast\text{O}\\), by contrast, is a bare oxygen with two dangling valences, and it engages the surface far more aggressively.

### 📚 One Bond, Two Species

The structural point is worth stating slowly, because everything that follows is a consequence of it.

A surface's chemistry, from an adsorbate's point of view, is largely summarized by *how strongly it grabs an oxygen atom through one bond*. Make the surface more oxophilic — by changing the metal, by doping, by straining — and it grabs oxygen harder. But \\(\ast\text{OH}\\) and \\(\ast\text{OOH}\\) both hold on through **the same kind of single bond to the same kind of oxygen**. If the surface pulls harder, it pulls harder on both.

So the two species do not respond to a change of material independently. They move **together**. Their binding energies rise and fall in near-lockstep, offset by a roughly constant amount that reflects the difference between what is hanging off the oxygen — a hydrogen in one case, an \\(\text{OOH}\\) fragment in the other.

\\(\ast\text{O}\\) is the exception, and it is the exception for a clean reason: it binds through *two* valences rather than one, so it responds about twice as steeply to the same change in surface oxophilicity. This is why \\(\ast\text{O}\\) can be moved relative to the other two, and why the descriptor of Section 3.4 is built out of it.

## 3.2 The Scaling Relation

Compute \\(\Delta G_{\ast\text{OH}}\\) and \\(\Delta G_{\ast\text{OOH}}\\) on surface after surface — different metals, different oxides, different facets, different coverages — and plot one against the other. The points do not scatter. They fall close to a line of slope one, displaced upward by a roughly constant offset:

\\[ \Delta G_{\ast\text{OOH}} \approx \Delta G_{\ast\text{OH}} + \Delta_{\text{scal}} \\]

The offset \\(\Delta_{\text{scal}}\\) is **a widely reproduced empirical result of roughly 3.2 eV**. Treat that number the way this series treats every empirical input: as an approximate regularity with real scatter around it, useful because it is robust, not because it is exact. Nothing in this chapter depends on the third decimal place; everything depends on the fact that the offset is *large* and *not easily changed*.

### 📚 Why the Relation Is So Hard to Break

Three properties make this particular scaling relation unusually stubborn.

**It is geometric, not electronic in any tunable sense.** The offset is set by the chemistry of what is attached to the binding oxygen. Changing the surface changes where *both* species sit on the line; it does not change the line.

**It survives averaging.** Individual surfaces scatter around the relation, sometimes substantially. But the scatter is roughly symmetric and the systematic offset persists across whole families of materials, which is exactly the situation in which a simple linear model becomes a useful design constraint rather than a curiosity.

**It is not a law.** This matters for Chapter 5. The relation is an empirical regularity about a particular class of surfaces treated with a particular level of theory. Strategies that deliberately break it — stabilizing \\(\ast\text{OOH}\\) with a second binding site, using a proton relay, changing the mechanism so that \\(\ast\text{OOH}\\) never forms — are precisely the strategies that could beat the floor we are about to derive. The floor is a floor *for catalysts that obey the relation*.

Here is why the relation is bad news. Chapter 2's four step energies looked like four independent design variables. They are not. The second and third steps are

\\[ \Delta G_2 = \Delta G_{\ast\text{O}} - \Delta G_{\ast\text{OH}}, \qquad \Delta G_3 = \Delta G_{\ast\text{OOH}} - \Delta G_{\ast\text{O}} \\]

and their **sum telescopes**: \\(\Delta G_{\ast\text{O}}\\) cancels, leaving \\(\Delta G_2 + \Delta G_3 = \Delta G_{\ast\text{OOH}} - \Delta G_{\ast\text{OH}}\\) — which is exactly the quantity the scaling relation pins down. Two of the four steps have been welded together. You may slide energy between them by moving \\(\ast\text{O}\\), but you cannot reduce their sum.

## 3.3 The Floor, Derived

That is enough to compute a hard limit, and the derivation is short enough to do in your head — which is why we will make the code do it, and then check the code against a brute-force scan.

If two non-negative numbers have a fixed sum \\(S\\), the smallest their *maximum* can be is \\(S/2\\), achieved when they are equal. The limiting step of the whole reaction is at least as large as the larger of \\(\Delta G_2\\) and \\(\Delta G_3\\). And the theoretical overpotential is the limiting step, expressed as a potential, minus the equilibrium potential. Chain those three sentences together and the floor falls out.

```python
import numpy as np

# ------------------------------------------------------------------
# THE TWO CONSTRAINTS
#
#   E_EQ     : the equilibrium potential of the four-electron OER.
#              A DEFINITION (the standard potential of the reaction),
#              not something this code measures.
#   SUM_TOTAL: 4 x E_EQ. Pure arithmetic -- the four step energies
#              must add up to the free energy of splitting two waters.
#   SCALING  : the *OOH / *OH offset. An EMPIRICAL regularity, widely
#              reproduced at roughly 3.2 eV across oxide surfaces.
#              It is an INPUT here; every consequence of it below is
#              computed, never quoted.
# ------------------------------------------------------------------
E_EQ = 1.23
N_STEPS = 4
SUM_TOTAL = N_STEPS * E_EQ
SCALING = 3.2

print("Constraint 1 (thermodynamic sum)")
print(f"  dG1 + dG2 + dG3 + dG4 = {N_STEPS} x {E_EQ} eV = {SUM_TOTAL:.2f} eV")
print("Constraint 2 (scaling relation, empirical and approximate)")
print(f"  dG_OOH - dG_OH        = {SCALING:.2f} eV")
print()

# dG2 + dG3 = (dG_O - dG_OH) + (dG_OOH - dG_O) = dG_OOH - dG_OH.
# The middle two steps telescope: dG_O cancels, and what is left is
# exactly the quantity the scaling relation pins down.
middle_pair = SCALING
best_max_step = middle_pair / 2.0        # best possible max of two
                                         # numbers with a fixed sum
eta_floor = best_max_step - E_EQ

print("Telescoping the middle pair")
print(f"  dG2 + dG3 = dG_OOH - dG_OH = {middle_pair:.2f} eV  (dG_O cancels)")
print(f"  the best a catalyst can do is split it evenly:")
print(f"    max(dG2, dG3) >= {middle_pair:.2f} / 2 = {best_max_step:.3f} eV")
print(f"    eta_floor      = {best_max_step:.3f} eV - {E_EQ} V "
      f"= {eta_floor:.3f} V")
print()

# Brute force: no split of the pair does better than the even one.
grid = np.linspace(0.0, SCALING, 320001)
limiting = np.maximum(grid, SCALING - grid)
i = int(np.argmin(limiting))
print("Brute-force check over every possible split of the pair")
print(f"  grid points scanned         : {grid.size}")
print(f"  best split, dG2             : {grid[i]:.5f} eV")
print(f"  best split, dG3             : {SCALING - grid[i]:.5f} eV")
print(f"  smallest achievable dG_max  : {limiting[i]:.5f} eV")
print(f"  implied overpotential floor : {limiting[i] - E_EQ:.5f} V")
print(f"  matches the analytic value  : "
      f"{np.isclose(limiting[i] - E_EQ, eta_floor)}")
print()

# What would have to be true for the floor to vanish?
needed = E_EQ * 2.0
print("Inverting the argument: which scaling constant would give eta = 0?")
print(f"  we would need dG_OOH - dG_OH = 2 x {E_EQ} = {needed:.2f} eV")
print(f"  nature appears to offer      ~ {SCALING:.2f} eV")
print(f"  the excess to be removed     = {SCALING - needed:.2f} eV")
```

**Output:**

```
Constraint 1 (thermodynamic sum)
  dG1 + dG2 + dG3 + dG4 = 4 x 1.23 eV = 4.92 eV
Constraint 2 (scaling relation, empirical and approximate)
  dG_OOH - dG_OH        = 3.20 eV

Telescoping the middle pair
  dG2 + dG3 = dG_OOH - dG_OH = 3.20 eV  (dG_O cancels)
  the best a catalyst can do is split it evenly:
    max(dG2, dG3) >= 3.20 / 2 = 1.600 eV
    eta_floor      = 1.600 eV - 1.23 V = 0.370 V

Brute-force check over every possible split of the pair
  grid points scanned         : 320001
  best split, dG2             : 1.60000 eV
  best split, dG3             : 1.60000 eV
  smallest achievable dG_max  : 1.60000 eV
  implied overpotential floor : 0.37000 V
  matches the analytic value  : True

Inverting the argument: which scaling constant would give eta = 0?
  we would need dG_OOH - dG_OH = 2 x 1.23 = 2.46 eV
  nature appears to offer      ~ 3.20 eV
  the excess to be removed     = 0.74 eV
```

**Reading the result.** Three observations.

  * **The floor is 0.370 V under these assumptions**, and it came out of two lines of arithmetic on two inputs: the definition \\(1.23\\) V and the empirical offset \\(3.2\\) eV. It is not a fitted number and it is not quoted from anywhere; it is what those two inputs imply. Change the offset and the floor moves with it — that sensitivity is the point of the last block of output.
  * **The brute-force scan finds nothing better.** Scanning every way of splitting the welded pair between steps 2 and 3 — 320,001 of them — the best is the even split, exactly as the algebra says. This is a check on the reasoning, not on the arithmetic: it confirms that no clever redistribution of energy between the two coupled steps buys you anything.
  * **The inverted question is the sharpest way to see the problem.** A perfect catalyst needs the \\(\ast\text{OOH}\\)/\\(\ast\text{OH}\\) offset to be \\(2 \times 1.23 = 2.46\\) eV, so that the two welded steps can each be exactly \\(1.23\\) eV. Nature offers roughly \\(3.2\\). The gap of about \\(0.74\\) eV is the entire problem of OER catalysis stated as a single number, and it is a *chemical* target: any strategy that closes it must change what the offset is, not how well the surface is optimized.

One caution before we build the volcano. The floor applies to catalysts that obey the scaling relation and proceed through this mechanism. It is a statement about a *family*, not a theorem about oxygen evolution. Chapter 5 returns to exactly this point.

## 3.4 The Volcano

The floor tells you the best case. The volcano tells you what happens when you are not in the best case — and it does so using **one number per catalyst**.

That number is the **descriptor**

\\[ x \equiv \Delta G_{\ast\text{O}} - \Delta G_{\ast\text{OH}} \\]

which is exactly \\(\Delta G_2\\), the second step. Why is one number enough? Because the scaling relation has already fixed the sum of the middle pair. Knowing \\(\Delta G_2 = x\\), the third step is forced: \\(\Delta G_3 = \Delta_{\text{scal}} - x\\). The two welded steps are now a one-parameter family, and the limiting step of the pair is \\(\max(x,\; \Delta_{\text{scal}} - x)\\).

Two branches, meeting at a point. Plot \\(-\eta\\) — activity, higher is better — against \\(x\\), and you get the inverted V that gives the volcano plot its name.

The code below sweeps the descriptor, tabulates both branches, locates the apex numerically, and checks the numerical apex against the algebra. It then does one more thing that is usually left implicit: it computes the conditions under which the *other* two steps stay out of the way, because the one-descriptor picture is only valid while they do.

```python
from math import isclose

# ------------------------------------------------------------------
# THE VOLCANO, built on the descriptor x = dG_O - dG_OH.
#
# With the scaling relation in force the two middle steps are
#     dG2 = dG_O   - dG_OH = x
#     dG3 = dG_OOH - dG_O  = SCALING - x
# so their maximum -- the potential-determining step of the coupled
# pair -- is max(x, SCALING - x), and the predicted overpotential is
# that maximum minus the equilibrium potential.
#
# Steps 1 and 4 are NOT on this axis. They depend on dG_OH itself,
# and the window in which they stay out of the way is computed below.
# ------------------------------------------------------------------
def limiting_step(x):
    """Return (dG_max in eV, label of the potential-determining step)."""
    d2, d3 = x, SCALING - x
    return (d2, "step 2  *OH -> *O") if d2 >= d3 else (d3, "step 3  *O -> *OOH")

def volcano_eta(x):
    return limiting_step(x)[0] - E_EQ

print(f"{'x = dG_O - dG_OH':>17} {'dG2':>7} {'dG3':>7} {'dG_max':>8} "
      f"{'eta (V)':>8} {'activity -eta':>14}   potential-determining step")
for k in range(17):
    x = 0.80 + 0.10 * k
    dmax, label = limiting_step(x)
    eta = dmax - E_EQ
    print(f"{x:17.2f} {x:7.2f} {SCALING - x:7.2f} {dmax:8.2f} "
          f"{eta:8.3f} {-eta:14.3f}   {label}")
print()

# Apex located numerically on a fine grid, then compared with algebra.
N = 400001
xs = [0.0 + 4.0 * j / (N - 1) for j in range(N)]
etas = [volcano_eta(x) for x in xs]
j_best = min(range(N), key=lambda j: etas[j])
x_apex_numeric, eta_apex_numeric = xs[j_best], etas[j_best]
x_apex_analytic = SCALING / 2.0
eta_apex_analytic = x_apex_analytic - E_EQ

print("Apex of the volcano")
print(f"  numerical scan ({N} points)")
print(f"    x_apex   = {x_apex_numeric:.5f} eV")
print(f"    eta_apex = {eta_apex_numeric:.5f} V")
print(f"  algebra")
print(f"    x_apex   = SCALING / 2          = {x_apex_analytic:.5f} eV")
print(f"    eta_apex = SCALING / 2 - E_EQ   = {eta_apex_analytic:.5f} V")
print(f"  numerical apex matches algebra   : "
      f"{isclose(x_apex_numeric, x_apex_analytic, abs_tol=1e-5)} / "
      f"{isclose(eta_apex_numeric, eta_apex_analytic, abs_tol=1e-5)}")
print(f"  every scanned point is at or above the floor: "
      f"{all(e >= eta_apex_analytic - 1e-12 for e in etas)}")
print()

# The volcano has a closed form: a V in eta, an inverted V in -eta.
worst = max(abs(volcano_eta(x) - (eta_apex_analytic + abs(x - x_apex_analytic)))
            for x in xs)
print("Closed form of the whole curve")
print(f"  eta(x) = eta_floor + |x - x_apex|")
print(f"  largest deviation over {N} scanned points: {worst:.2e} V")
print()

# ------------------------------------------------------------------
# WHEN IS THE TWO-BRANCH VOLCANO VALID?
# Steps 1 and 4 are dG1 = dG_OH and dG4 = SUM_TOTAL - dG_OOH, and with
# the scaling relation dG4 = SUM_TOTAL - SCALING - dG_OH. Neither may
# exceed dG_max, which brackets dG_OH in a window we can compute.
# ------------------------------------------------------------------
tail_pair = SUM_TOTAL - SCALING
print(f"dG1 + dG4 = SUM_TOTAL - (dG_OOH - dG_OH) = {tail_pair:.2f} eV, "
      f"so dG4 = {tail_pair:.2f} - dG_OH")
print()
print(f"{'x':>6} {'dG_max':>8}   dG_OH window in which steps 1 and 4 stay out of the way")
for k in range(9):
    x = 0.80 + 0.20 * k
    dmax, _ = limiting_step(x)
    lo, hi = max(0.0, tail_pair - dmax), dmax
    print(f"{x:6.2f} {dmax:8.2f}   {lo:.2f} eV <= dG_OH <= {hi:.2f} eV"
          f"   (width {hi - lo:.2f} eV)")
print()
dmax_apex, _ = limiting_step(x_apex_analytic)
print(f"At the apex the window is "
      f"{max(0.0, tail_pair - dmax_apex):.2f} eV <= dG_OH <= {dmax_apex:.2f} eV")
print(f"  width = {dmax_apex - max(0.0, tail_pair - dmax_apex):.2f} eV")
```

**Output:**

```
 x = dG_O - dG_OH     dG2     dG3   dG_max  eta (V)  activity -eta   potential-determining step
             0.80    0.80    2.40     2.40    1.170         -1.170   step 3  *O -> *OOH
             0.90    0.90    2.30     2.30    1.070         -1.070   step 3  *O -> *OOH
             1.00    1.00    2.20     2.20    0.970         -0.970   step 3  *O -> *OOH
             1.10    1.10    2.10     2.10    0.870         -0.870   step 3  *O -> *OOH
             1.20    1.20    2.00     2.00    0.770         -0.770   step 3  *O -> *OOH
             1.30    1.30    1.90     1.90    0.670         -0.670   step 3  *O -> *OOH
             1.40    1.40    1.80     1.80    0.570         -0.570   step 3  *O -> *OOH
             1.50    1.50    1.70     1.70    0.470         -0.470   step 3  *O -> *OOH
             1.60    1.60    1.60     1.60    0.370         -0.370   step 2  *OH -> *O
             1.70    1.70    1.50     1.70    0.470         -0.470   step 2  *OH -> *O
             1.80    1.80    1.40     1.80    0.570         -0.570   step 2  *OH -> *O
             1.90    1.90    1.30     1.90    0.670         -0.670   step 2  *OH -> *O
             2.00    2.00    1.20     2.00    0.770         -0.770   step 2  *OH -> *O
             2.10    2.10    1.10     2.10    0.870         -0.870   step 2  *OH -> *O
             2.20    2.20    1.00     2.20    0.970         -0.970   step 2  *OH -> *O
             2.30    2.30    0.90     2.30    1.070         -1.070   step 2  *OH -> *O
             2.40    2.40    0.80     2.40    1.170         -1.170   step 2  *OH -> *O

Apex of the volcano
  numerical scan (400001 points)
    x_apex   = 1.60000 eV
    eta_apex = 0.37000 V
  algebra
    x_apex   = SCALING / 2          = 1.60000 eV
    eta_apex = SCALING / 2 - E_EQ   = 0.37000 V
  numerical apex matches algebra   : True / True
  every scanned point is at or above the floor: True

Closed form of the whole curve
  eta(x) = eta_floor + |x - x_apex|
  largest deviation over 400001 scanned points: 2.22e-16 V

dG1 + dG4 = SUM_TOTAL - (dG_OOH - dG_OH) = 1.72 eV, so dG4 = 1.72 - dG_OH

     x   dG_max   dG_OH window in which steps 1 and 4 stay out of the way
  0.80     2.40   0.00 eV <= dG_OH <= 2.40 eV   (width 2.40 eV)
  1.00     2.20   0.00 eV <= dG_OH <= 2.20 eV   (width 2.20 eV)
  1.20     2.00   0.00 eV <= dG_OH <= 2.00 eV   (width 2.00 eV)
  1.40     1.80   0.00 eV <= dG_OH <= 1.80 eV   (width 1.80 eV)
  1.60     1.60   0.12 eV <= dG_OH <= 1.60 eV   (width 1.48 eV)
  1.80     1.80   0.00 eV <= dG_OH <= 1.80 eV   (width 1.80 eV)
  2.00     2.00   0.00 eV <= dG_OH <= 2.00 eV   (width 2.00 eV)
  2.20     2.20   0.00 eV <= dG_OH <= 2.20 eV   (width 2.20 eV)
  2.40     2.40   0.00 eV <= dG_OH <= 2.40 eV   (width 2.40 eV)

At the apex the window is 0.12 eV <= dG_OH <= 1.60 eV
  width = 1.48 eV
```

**Reading the result.** Four points, and the last is the one that carries into Chapter 5.

  * **The two legs have different bottlenecks.** On the left, at small \\(x\\), the surface holds \\(\ast\text{O}\\) too tightly relative to \\(\ast\text{OH}\\): converting that over-stabilized \\(\ast\text{O}\\) into \\(\ast\text{OOH}\\) is the expensive move, and **step 3 is potential-determining**. On the right, at large \\(x\\), \\(\ast\text{O}\\) is destabilized relative to \\(\ast\text{OH}\\), so the expensive move is the earlier one — stripping the second proton to make \\(\ast\text{O}\\) out of \\(\ast\text{OH}\\) — and **step 2 is potential-determining**. Both legs are the same failure seen from opposite sides: the welded pair is split unevenly, and whichever half got the larger share sets the price.
  * **The apex is at \\(x = 1.60\\) eV with \\(\eta = 0.370\\) V**, and the numerical scan over 400,001 grid points agrees with the algebra to five decimals. The apex is the even split of the welded pair — the same configuration the brute-force scan found in Section 3.3, now seen as a location on a curve rather than as an inequality.
  * **The whole curve is \\(\eta = \eta_{\text{floor}} + |x - x_{\text{apex}}|\\)**, verified to \\(2 \times 10^{-16}\\) V across the scan. That closed form is worth memorizing, because it says something blunt: under these assumptions the overpotential is the floor **plus** your distance from the apex, in a one-to-one exchange. Mis-place your descriptor by 0.3 eV and you pay 0.3 V. There is no forgiveness region, no plateau near the top. The volcano is sharp.
  * **The one-descriptor picture has a validity window, and the code states it.** Steps 1 and 4 are not on the descriptor axis at all; they depend on \\(\Delta G_{\ast\text{OH}}\\) itself, and the scaling relation ties them together too — their sum is \\(4.92 - 3.2 = 1.72\\) eV. As long as \\(\Delta G_{\ast\text{OH}}\\) lies in the printed window, neither can outgrow the welded pair, and the volcano is the whole story. At the apex that window is \\(0.12\\) to \\(1.60\\) eV, a width of \\(1.48\\) eV — comfortable, which is *why* the one-descriptor picture works so well in practice. But it is a window, not a guarantee, and a screening pipeline that never checks it will eventually mis-rank something.

Notice what the volcano did *not* require: it never needed \\(\Delta G_{\ast\text{O}}\\) and \\(\Delta G_{\ast\text{OH}}\\) separately, only their difference. That is the collapse, and it is worth naming.

## 3.5 Why No One Sits Above the Apex

It is tempting to read the volcano as an empirical observation — people computed a lot of catalysts, the points happened to fall under an inverted V, and someone drew the envelope. That reading gets the logic backwards.

The apex is not the best point anyone *found*. It is the best point that *exists*, given the constraints, and the code above proved it twice: once by algebra and once by exhaustive scan. Any catalyst obeying the scaling relation is a point on the curve, not near it. The scatter you see in a real volcano plot is scatter in the scaling relation itself — surfaces that deviate from the line sit off the curve, and a surface that deviated *in the right direction*, with a smaller \\(\ast\text{OOH}\\)/\\(\ast\text{OH}\\) offset, would sit above the apex.

That is not a loophole in the argument; it is the argument's own prescription. The way above the volcano is to break the relation. Everything else — better facets, better dopants, better nanostructuring, more careful synthesis — moves you *along* the curve toward the apex, which is worth doing and which stops paying at \\(0.370\\) V.

> **The optimist's reading and the pessimist's reading**
>
> The pessimist says: a floor of roughly 0.37 V means several hundred millivolts of wasted energy in every water electrolyser built on this chemistry, forever.
>
> The optimist says: the floor is a statement about a *constraint*, and constraints identified this precisely are targets. The number \\(0.74\\) eV computed in Section 3.3 — the excess in the scaling offset — is a design specification. It tells you how much stabilization of \\(\ast\text{OOH}\\), relative to \\(\ast\text{OH}\\), a genuinely new catalyst architecture would have to deliver.
>
> Both readings are correct, and they are the same sentence.

## 3.6 One Number Per Material

Step back from the electrochemistry and look at the shape of what just happened.

We began with a four-dimensional problem: three adsorption energies and a mechanism. We ended with a **function of one variable**. The reduction was not a numerical approximation and not a fit — it came from two constraints, one thermodynamic and one empirical, each of which removed a degree of freedom.

This is the **descriptor** idea, and it is the single most transferable thing in this chapter. A descriptor is one number, cheap to obtain, that predicts a performance metric that is expensive to obtain. Here the descriptor is \\(\Delta G_{\ast\text{O}} - \Delta G_{\ast\text{OH}}\\) and the metric is the theoretical overpotential. Elsewhere in materials science it might be a *d*-band centre predicting adsorption, a tolerance factor predicting whether a perovskite will form, or a formation energy predicting stability.

### 📚 What a Descriptor Buys, and What It Costs

**It buys screening.** Instead of running a full four-intermediate calculation on every candidate, you compute one quantity and read the answer off a curve. Chapter 4 does exactly this for five surfaces and checks that the shortcut reproduces the long calculation exactly.

**It buys interpretability.** A one-dimensional model can be plotted, argued about, and used to reason backwards: "we need to move \\(x\\) down by 0.2 eV" is an actionable sentence in a way that "we need a better catalyst" is not.

**It costs generality, silently.** The descriptor is only as good as the constraints that produced it. Feed it a material that breaks the scaling relation and it will confidently return a wrong answer, because nothing in the formula knows the relation has failed. It will not warn you. The validity window computed in Section 3.4 is one instance of a check you can build in; there are others, and most screening pipelines contain none of them.

That last item is where machine learning enters and where it gets dangerous. Once you accept that one number can stand in for an expensive calculation, the natural next thought is: why stop at one number, and why insist that a human derive it? Learn the mapping from structure straight to activity, over thousands of candidates, and let the model find its own descriptors.

That is Chapter 5's subject, and its warning. A learned descriptor inherits every assumption baked into the data it was trained on — including the computational hydrogen electrode's own approximations from Chapter 2, and including the scaling relation itself. A model trained on surfaces that all obey the relation will confidently reproduce the volcano, floor and all, and will have no way to tell you that the interesting materials are the ones it has never seen.

### 🎯 Exercise Problems

  1. **Move the constant.** Re-run the floor calculation with scaling offsets of \\(3.0\\), \\(3.2\\), and \\(3.4\\) eV. Tabulate the resulting floor and apex position. How many millivolts of floor does each 0.1 eV of offset cost, and why is the answer exactly what it is?
  2. **The forgiveness question.** Using the closed form \\(\eta = \eta_{\text{floor}} + |x - x_{\text{apex}}|\\), compute how far the descriptor may stray from the apex before the overpotential exceeds \\(0.60\\) V. Compare that tolerance with the typical spread of DFT adsorption energies and comment on what it implies for screening accuracy.
  3. **Breaking the window.** Construct an illustrative surface with \\(x = 1.60\\) eV but \\(\Delta G_{\ast\text{OH}} = 1.90\\) eV. Compute all four step energies and the overpotential directly. Which step limits, and by how much does the one-descriptor volcano under-predict \\(\eta\\)? Relate your answer to the printed validity window.
  4. **A different weld.** Suppose a new class of surfaces obeyed \\(\Delta G_{\ast\text{OOH}} \approx \Delta G_{\ast\text{OH}} + 2.8\\) eV instead. Recompute the floor and the apex, then state clearly what would have to be physically true of such a surface for that offset to be achievable.
  5. **Descriptor audit.** Pick any descriptor used in your own field. Write down (a) the constraint or approximation that justifies collapsing the problem onto it, (b) a class of materials for which that constraint fails, and (c) a check you could compute cheaply that would flag the failure. If you cannot answer (c), say what that implies about using the descriptor for screening.

## Summary

This chapter turned Chapter 2's four-step machinery into a constraint and then into a map. The **scaling relation** \\(\Delta G_{\ast\text{OOH}} \approx \Delta G_{\ast\text{OH}} + \Delta_{\text{scal}}\\), with \\(\Delta_{\text{scal}}\\) a widely reproduced empirical offset of roughly \\(3.2\\) eV, holds because \\(\ast\text{OH}\\) and \\(\ast\text{OOH}\\) both attach to the surface through a single oxygen and therefore respond together to any change in surface oxophilicity; \\(\ast\text{O}\\), binding through two valences, is the one intermediate that can be moved independently. The consequence is that steps 2 and 3 **telescope into a welded pair** whose sum equals \\(\Delta_{\text{scal}}\\) — two of the four design variables are gone. From the fixed total \\(4 \times 1.23 = 4.92\\) eV and that welded sum, our code derived an overpotential **floor of \\(0.370\\) V**, confirmed by a brute-force scan over 320,001 splits, and inverted the argument to show that a perfect catalyst would need an offset of \\(2.46\\) eV — an excess of about \\(0.74\\) eV to be removed by chemistry, not by optimization. Collapsing onto the single **descriptor** \\(x = \Delta G_{\ast\text{O}} - \Delta G_{\ast\text{OH}}\\) produced the **volcano**: step 3 limits on the strong-binding leg, step 2 limits on the weak-binding leg, and the apex sits at \\(x = 1.60\\) eV with the floor value, matched by a 400,001-point numerical scan. The whole curve obeys \\(\eta = \eta_{\text{floor}} + |x - x_{\text{apex}}|\\) to machine precision, so distance from the apex converts one-for-one into overpotential. We also computed the **validity window** on \\(\Delta G_{\ast\text{OH}}\\) — \\(0.12\\) to \\(1.60\\) eV at the apex — inside which the first and fourth steps stay out of the way and the one-descriptor picture is complete.

Chapter 4 puts this to work. We build a small screening pipeline in NumPy on five fictitious catalysts, construct their free-energy diagrams at zero and at the equilibrium potential, rank them by overpotential, and then verify that the single-descriptor volcano reproduces that ranking exactly — the shortcut and the long calculation agreeing, digit for digit, because they must.

[← Chapter 2: The Computational Hydrogen Electrode](<chapter-2.html>) [Chapter 4: Hands-On: Screening in Python →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
