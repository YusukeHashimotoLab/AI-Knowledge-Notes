---
title: "Chapter 5: ML Screening and the Limits of CHE"
chapter_title: "Chapter 5: ML Screening and the Limits of CHE"
subtitle: "From One Descriptor to a Learned Model, and an Honest Accounting of What the Computational Hydrogen Electrode Cannot See"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/WjNEABrgXc0"
    title="OER Comp Chem Ch.5: ML Screening and the Limits of CHE"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/MI/oer-computational-chemistry/chapter-5.html>) | Last sync: 2026-08-18

[Materials Informatics Dojo](<../index.html>) > [Computational Chemistry of OER](<index.html>) > Chapter 5

## 5.1 From One Descriptor to Many

Chapter 3 performed a remarkable compression. A four-electron reaction with three intermediates, four free-energy steps, and a whole surface's worth of electronic structure was reduced to **one number** — the binding free energy of the \\(\ast\mathrm{OH}\\) intermediate — and the theoretical overpotential was read off a curve. That is what a descriptor is: a lossy but useful projection of a complicated system onto an axis you can actually compute.

The volcano earns that compression from a scaling relation. Because \\(\Delta G_{\mathrm{OOH}}\\) and \\(\Delta G_{\mathrm{OH}}\\) move together with a nearly constant offset — the roughly 3.2 eV we hedged so carefully in Chapter 3 — fixing one intermediate very nearly fixes the other, and the whole diagram collapses onto a single axis.

Machine learning is the same move, generalized. Instead of insisting that one physically chosen quantity carries all the information, you let a model learn which combination of *many* cheap quantities predicts the expensive one. The inputs are things you can read off a composition or a structure without solving anything self-consistently: which elements are present and in what proportion, their electronegativities and atomic radii, coordination numbers, a d-band descriptor if you have one, oxidation states, lattice geometry. The output is \\(\Delta G_{\mathrm{OH}}\\) — the very number that Chapter 4's screening loop paid a full slab calculation to obtain.

If the model is any good, you stop paying for most of those calculations.

> **What the model is and is not learning**
>
> It is not learning chemistry. It is learning a correlation between features and a label, on the set of materials you happened to compute. Everything in this chapter that sounds like a warning is downstream of that one sentence.

## 5.2 The Screening Funnel

The practical architecture that follows is a **funnel**. Each stage is cheaper per candidate and larger in population than the stage below it; each stage's job is to pass a manageable shortlist down.

  * **Stage 1 — the learned model.** Millions of compositions are conceivable; scoring one costs a fraction of a millisecond. Run it on everything.
  * **Stage 2 — DFT with CHE.** The Chapter 4 machinery, applied only to the shortlist. Expensive, physically grounded, still an approximation.
  * **Stage 3 — experiment.** Synthesis and electrochemical testing. The only stage that can tell you whether any of the preceding was true.

Here is that funnel as arithmetic. **Every unit cost below is invented by us** to make the ratios legible; no cluster, code, or laboratory is being described.

| Stage | Candidates | Invented unit cost | Stage cost |
|---|---|---|---|
| ML prediction | 100,000 | 1.0 × 10⁻⁶ | 0.1 |
| DFT / CHE (best 1%) | 1,000 | 20 | 20,000 |
| Experiment (best 5% of those) | 50 | 200 | 10,000 |
| **Total** | | | **30,000** |

Running DFT on every candidate instead would cost 2,000,000 in the same invented units — a factor of roughly **67** more. And if there are, say, 40 genuinely good candidates hidden in the space, a stage-1 model with 90% recall passes 36 of them on and destroys 4; at 50% recall it destroys 20.

**Reading the table.** Two lessons, and the second matters more than the first.

  * **The cheap stage is free and the expensive stages dominate.** The ML column contributes essentially nothing to the total. All the cost lives where the physics lives — which is exactly why the model is worth having, and also why nobody should describe the funnel as "AI discovering a catalyst". The model narrows; DFT and the bench decide.
  * **Recall is the quantity that actually matters at stage 1, and it is asymmetric.** A false positive is forwarded to DFT, which rejects it — you paid one wasted calculation. A false negative is deleted from the universe. It never reaches DFT, never reaches the bench, and nothing downstream can recover it. This is why a screening model should be tuned to be *generous*, and why reporting only \\(R^2\\) on a random test split tells you almost nothing about whether the funnel works.

## 5.3 A Ridge Model in NumPy — and Where It Fails

Let us build the stage-1 model, in the same spirit as the rest of this series: no scikit-learn, no black box, just the normal equation.

**Ridge regression** fits \\(y \approx \mathbf{w}^\top \mathbf{x} + b\\) by minimizing squared error plus a penalty \\(\lambda \lVert \mathbf{w} \rVert^2\\) that discourages large coefficients. The penalty has a closed-form solution: with a design matrix \\(A\\) whose first column is ones,

\\[ \boldsymbol{\theta} = (A^\top A + \lambda P)^{-1} A^\top \mathbf{y} \\]

where \\(P\\) is the identity with its first diagonal entry zeroed so the bias is not penalized. That is the entire algorithm.

The data below is **entirely synthetic**. We invent five feature columns, invent a linear rule that turns them into a fictitious binding energy, add invented noise, and fit. Nothing here is a material, and no coefficient corresponds to any real descriptor. The purpose is to watch the *shape* of the error.

```python
import numpy as np

# --- Ridge regression on SYNTHETIC data: interpolation vs extrapolation ---
#
# Nothing below describes a real material. We invent five "descriptor"
# columns, invent a linear rule that turns them into a fictitious binding
# energy, add noise, and then see how well a fitted model recovers the rule.
# The point is the SHAPE of the error, not the values.

rng = np.random.default_rng(0)

N_FEATURES = 5
TRUE_W = np.array([0.80, -0.45, 0.30, -0.15, 0.05])   # invented rule
TRUE_B = 1.20                                          # invented offset
NOISE = 0.05                                           # invented scatter [eV]


def make_data(n, low, high, rng):
    """n fictitious candidates whose descriptors live in [low, high]."""
    X = rng.uniform(low, high, size=(n, N_FEATURES))
    y = X @ TRUE_W + TRUE_B + rng.normal(0.0, NOISE, size=n)
    return X, y


def ridge_fit(X, y, lam):
    """Normal equation with an L2 penalty. Bias column is NOT penalised."""
    n, d = X.shape
    A = np.hstack([np.ones((n, 1)), X])          # design matrix with bias
    P = np.eye(d + 1) * lam
    P[0, 0] = 0.0                                # leave the bias alone
    theta = np.linalg.solve(A.T @ A + P, A.T @ y)
    return theta


def predict(theta, X):
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X]) @ theta


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --- Training set: descriptors drawn from the INTERIOR range [-1, 1] -----
X_train, y_train = make_data(200, -1.0, 1.0, rng)
X_test, y_test = make_data(200, -1.0, 1.0, rng)      # same range = interpolation

theta = ridge_fit(X_train, y_train, lam=1e-2)

print("Ridge fit on synthetic data (all values invented for teaching)")
print(f"  training candidates : {X_train.shape[0]}")
print(f"  descriptors         : {N_FEATURES}")
print(f"  injected noise sd   : {NOISE:.3f} (fictitious units)")
print()
print(f"{'coefficient':>12} {'true':>8} {'fitted':>8}")
print(f"{'bias':>12} {TRUE_B:>8.3f} {theta[0]:>8.3f}")
for i in range(N_FEATURES):
    print(f"{'w' + str(i + 1):>12} {TRUE_W[i]:>8.3f} {theta[1 + i]:>8.3f}")
print()

print("Errors inside the training range (interpolation)")
print(f"  train RMSE = {rmse(predict(theta, X_train), y_train):.4f}")
print(f"  test  RMSE = {rmse(predict(theta, X_test), y_test):.4f}")
print()

# --- Now ask the same model about candidates OUTSIDE the training range --
print("Errors outside the training range (extrapolation)")
print(f"{'descriptor range':>18} {'RMSE':>8} {'x interp':>9}")
base = rmse(predict(theta, X_test), y_test)
for lo, hi in [(-1.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]:
    Xo, yo = make_data(200, lo, hi, rng)
    e = rmse(predict(theta, Xo), yo)
    label = f"[{lo:+.0f}, {hi:+.0f}]"
    print(f"{label:>18} {e:>8.4f} {e / base:>9.2f}")
print()

# --- Why it grows: the fitted rule is only approximately the true rule,
# --- and the small coefficient error is multiplied by how far out you go.
dw = theta[1:] - TRUE_W
print(f"coefficient error |w_fit - w_true|_2 = {np.linalg.norm(dw):.4f}")
print("Error at distance R from the origin grows roughly like R x that number:")
for R in [1.0, 3.0, 6.0]:
    print(f"  R = {R:>3.0f} -> predicted drift ~ {R * np.linalg.norm(dw):.4f}")
```

**Output:**

```
Ridge fit on synthetic data (all values invented for teaching)
  training candidates : 200
  descriptors         : 5
  injected noise sd   : 0.050 (fictitious units)

 coefficient     true   fitted
        bias    1.200    1.203
          w1    0.800    0.797
          w2   -0.450   -0.464
          w3    0.300    0.283
          w4   -0.150   -0.156
          w5    0.050    0.049

Errors inside the training range (interpolation)
  train RMSE = 0.0487
  test  RMSE = 0.0524

Errors outside the training range (extrapolation)
  descriptor range     RMSE  x interp
          [-1, +1]   0.0499      0.95
          [+1, +2]   0.0753      1.44
          [+2, +4]   0.1301      2.48
          [+4, +8]   0.2478      4.73

coefficient error |w_fit - w_true|_2 = 0.0224
Error at distance R from the origin grows roughly like R x that number:
  R =   1 -> predicted drift ~ 0.0224
  R =   3 -> predicted drift ~ 0.0672
  R =   6 -> predicted drift ~ 0.1344
```

**Reading the result.** Three observations, and the third is the one to carry out of this series.

  * **Inside the training range, the model is as good as the data.** Train and test RMSE both sit near the injected noise level. The fit has recovered essentially all of the learnable signal; the residual is the scatter we put in on purpose. This is the happy case, and it is the case every parity plot in every screening paper is showing you.
  * **The coefficients are close but not exact, and ridge biases them slightly toward zero.** That is the penalty doing its job: it trades a little bias for stability. On real, correlated descriptors this trade is usually worth taking.
  * **Outside the training range the error grows without any warning signal.** The model does not know it is extrapolating. It returns a number with the same confident format at every input. The final block shows why: a small coefficient error, multiplied by distance, becomes a large prediction error, and the growth is roughly linear in how far out you go. **Interpolation is a measurement; extrapolation is a guess in the shape of a measurement.**

This is the honest lesson of ML screening. The model is superb at "which of these known-type materials should I compute first" and unreliable at "what completely new chemistry should I try" — which is, awkwardly, the question everyone actually wants answered. The defence is procedural, not algorithmic: define an **applicability domain**, check whether a candidate falls inside it before trusting the prediction, and treat everything outside as a hypothesis to be tested rather than a result.

## 5.4 Breaking the Scaling Relation

The volcano of Chapter 3 has a ceiling because of the scaling relation, not because of any individual material. If \\(\Delta G_{\mathrm{OOH}} - \Delta G_{\mathrm{OH}}\\) is pinned near roughly 3.2 eV on essentially every flat oxide surface, then the best possible theoretical overpotential is set by how far that constant sits from the ideal 2 × 1.23 eV, and no amount of composition tuning within the family escapes it. Getting past the ceiling means breaking the correlation itself.

Three strategies are actively pursued. All three are **research directions, not solved problems**, and each is stated qualitatively here on purpose.

**Stabilizing \\(\ast\mathrm{OOH}\\) differently from \\(\ast\mathrm{OH}\\).** The scaling relation exists because both intermediates bind through a single oxygen atom to the same site, so both feel nearly the same local electronic environment. If the environment can be made to distinguish them — geometrically rather than electronically — the correlation loosens. The larger \\(\ast\mathrm{OOH}\\) species could in principle be stabilized by a second interaction that \\(\ast\mathrm{OH}\\) is too small to reach: a nearby hydrogen-bond donor, a confining pocket in a framework material, or a second metal site at an appropriate distance. The idea is intuitive; realizing it in a material that is also stable and conductive under OER conditions is the hard part.

**Bifunctional mechanisms.** If the four proton-electron transfers do not all occur at the same site, the single-site scaling argument does not apply as written. A surface with two chemically distinct sites — one favouring O–H bond activation, another favouring O–O coupling — is not constrained to place all intermediates on one binding-energy axis. This is an appealing design principle and a genuinely difficult one to verify, because establishing that two sites cooperate (rather than that one of them simply dominates) requires evidence beyond a thermodynamic diagram.

**Lattice-oxygen participation.** The mechanism assumed throughout this series is adsorbate-evolving: every oxygen in the product \\(\mathrm{O}_2\\) comes from water, and the surface is a spectator. An alternative family of mechanisms has oxygen atoms *from the oxide lattice itself* incorporated into the evolved \\(\mathrm{O}_2\\), with the resulting vacancy refilled from solution. Because the intermediates and elementary steps differ, the conventional scaling relation need not hold. It is also a mechanism with an obvious tension built in: a lattice that gives up its oxygen is a lattice that is being chemically restructured, which puts activity and stability directly at odds. Distinguishing this pathway experimentally requires isotope labelling and careful controls, and the assignment is not always uncontroversial.

Notice what all three have in common: each buys freedom from the scaling relation by making the system *more complicated than the model that produced the volcano*. That is the honest price. The moment you leave the one-site, one-descriptor picture, the CHE analysis that made screening tractable becomes harder to apply — and the ML model you trained on flat-surface data is now being asked to extrapolate.

## 5.5 The Limits of CHE, Honestly

This is the section the whole series has been building toward. The computational hydrogen electrode is a genuinely elegant piece of physical reasoning, and it is used far beyond what it can support. Five limits, in rough order of how often they are forgotten.

**1. It is thermodynamics only.** CHE gives you free-energy *differences* between intermediates. It says nothing about the **barriers between them**. A reaction path can be downhill at every step and still be slow, because the transition state connecting two stable intermediates may sit high above both. The theoretical overpotential from a free-energy diagram is therefore a *lower bound on the difficulty*, not a rate. Two materials with identical diagrams can differ by orders of magnitude in measured current. Whenever you see a free-energy diagram used to explain why one catalyst is faster than another, the kinetic step has been assumed, not shown.

**2. It assumes an ideal, static surface.** The calculation is performed on a clean, periodic, low-index slab, relaxed in vacuum or with a thin water layer, and frozen in that geometry. Real electrodes at OER potentials are not that. Surfaces reconstruct; near-surface layers oxidize further; some species dissolve and redeposit; amorphous oxyhydroxide layers form on top of the crystalline phase you modelled. The unsettling consequence is that **the active site may not be the surface you computed** — it may be a defect, a step edge, a dissolved-and-redeposited species, or a phase that only exists under polarization. A perfectly executed calculation on the wrong structure is still the wrong answer.

**3. Solvent and electric-field effects are heavily simplified.** The electrode–electrolyte interface is a dense, structured, dynamic region with an enormous local field across a few ångströms, specifically adsorbed ions, and a hydrogen-bond network that reorganizes as intermediates change. Standard practice replaces all of this with an implicit continuum, a static water bilayer, or nothing at all, plus an empirical correction for hydrogen bonding to \\(\ast\mathrm{OOH}\\). These corrections are reasonable and they are not small. The pH dependence, the identity of the electrolyte cation, and the field-dependence of binding energies all enter here, and CHE in its basic form absorbs them into constants.

**4. Stability is at least as important as activity, and CHE is silent about it.** A catalyst that is superb for an hour is not a catalyst. OER operates at strongly oxidizing potentials in aqueous electrolyte — conditions that dissolve many of the elements one would most like to use. Nothing in the free-energy diagram of the four reaction steps says whether the material survives them. Screening on activity alone systematically produces shortlists whose top entries fail for reasons the screen never examined, and this failure mode is common enough that it should be assumed until checked.

**5. Descriptor errors compound.** The whole edifice rests on computed adsorption energies, and those depend on the exchange-correlation functional. The dependence is systematic rather than random: different functional families lean differently on transition-metal oxides, on localized d-electrons, and on the description of the O–O bond. That error then propagates through the scaling relation into the predicted overpotential, and — if you built one — into the training labels of your ML model, which inherits the bias in full and reports it with the confident precision of Section 5.3. Comparing two numbers computed with the *same* setup is far safer than comparing either to an absolute experimental value, and comparing across setups is not a comparison at all.

> **The honest summary of CHE.** It is a tool for **ranking candidates within a family under a fixed set of assumptions**. It is not a predictor of catalytic rate, not a statement about the material's real surface, and not an assessment of whether the material survives. Used for what it does, it is one of the most productive ideas in computational electrocatalysis. Used for what it does not do, it produces confident, well-formatted, wrong shortlists.

## 5.6 What the Field Does About Each Limit

None of the five limits is being ignored. Each has a body of methodology aimed at it, and knowing the names is enough to read the literature critically.

**For thermodynamics-only:** explicit **transition-state searches** for the elementary steps, and **microkinetic modelling**, which assembles elementary rate constants into a predicted current and can reveal that the rate-determining step is not the thermodynamically largest one. Microkinetics is also where the potential and coverage dependence of the rate can be handled properly.

**For ideal surfaces:** **in-situ and operando characterization** — spectroscopic and diffraction methods applied while the electrode is running rather than before and after — is how the field learns what the surface actually is under potential. Computationally, the counterpart is to model reconstructed, defective, amorphous, or vacancy-containing structures rather than the pristine slab, which multiplies the number of calculations and makes the ML stage more attractive, not less.

**For solvent and field:** **explicit-solvent methods** with molecular water and sampling of configurations, **constant-potential** (grand-canonical) electronic structure approaches that hold the electrode potential fixed rather than the electron number, and treatments of the double layer and specifically adsorbed ions. These are considerably more expensive than the CHE recipe of Chapter 2, which is precisely why the CHE recipe remains in use.

**For stability:** **Pourbaix-style analysis**, which asks which phase is thermodynamically preferred at a given potential and pH and therefore whether your candidate is even the stable species under operating conditions. Computed Pourbaix diagrams are approximate and carry their own assumptions, but a screen that includes one is asking a question that an activity-only screen never asks.

**For functional dependence:** benchmarking against higher-level methods on small model systems, comparing across functional families rather than trusting one, and — in the same spirit as this whole series — reporting *differences* computed consistently rather than absolutes.

## 5.7 Where This Meets Materials Informatics

If you came to this series from the MI side of AI Terakoya, here is the join.

The [MI Applications to Catalyst Design](<../catalyst-mi-application/index.html>) series takes the data-driven layer as its subject: descriptor engineering, regression and classification models for activity, uncertainty quantification, Bayesian optimization over composition space, and the active-learning loop that closes computation and experiment. This series has been the layer underneath it — where the descriptors come from, what \\(\Delta G_{\mathrm{OH}}\\) means, why a volcano has that shape, and what a computed overpotential is and is not.

The two layers need each other in a specific way.

**Physics gives the ML model its features.** A model trained on raw composition alone must rediscover binding-energy trends from data. A model given \\(\Delta G_{\mathrm{OH}}\\), d-band descriptors, coordination numbers, and oxidation states is being handed the structure of the problem, and it can learn from far fewer examples because of it. Feature engineering informed by mechanism is the single most reliable way to make a small materials dataset go further.

**ML gives the physics its reach.** Chapter 4's screening loop is limited by how many slab calculations you can afford. Section 5.2's funnel removes that limit for the ranking stage, and Bayesian optimization goes further: rather than scoring a fixed list, it chooses *which calculation or experiment to run next*, using the model's uncertainty to balance exploiting the current best against exploring where the model is ignorant. That is the natural next step after this series, and it is exactly what the catalyst MI series develops.

**And the caution transfers in both directions.** A model trained on CHE labels inherits every one of the five limits in Section 5.5 — including the ones that are systematic rather than random, which no amount of data cures. Uncertainty estimates from the model describe scatter in the *labels*, not error in the *physics that produced the labels*. A screening pipeline that reports a tight confidence interval around a prediction of a quantity that is itself the wrong quantity is not being rigorous; it is being precise about the wrong thing. The MI discipline of holding out a test set, comparing against a strong baseline, and reporting honest error applies here word for word — with one addition specific to this domain: **state the applicability domain, and state what the label actually is.**

## 5.8 What You Have, and What to Do With It

Take stock of the toolkit this series built.

You can write the four-step OER mechanism and explain why the four-electron requirement makes it the bottleneck of water splitting. You can construct a free-energy diagram using the computational hydrogen electrode, including the reference trick that replaces an intractable proton-electron pair with half a hydrogen molecule, and the corrections that turn computed electronic energies into free energies. You can read a potential-dependent diagram and extract a theoretical overpotential from it. You know why the scaling relation exists, why it caps the volcano, and why the offset is quoted as roughly 3.2 eV rather than as an exact constant. You have implemented the screening loop in Python, and now a stage-1 model to sit in front of it.

More importantly, you can read a computational electrocatalysis paper and ask the questions that decide whether it means anything. What surface was modelled, and is there any reason to think it is the surface that operates? Is the claim thermodynamic or kinetic, and if it is presented as an explanation of rate, where did the barriers come from? How was solvation handled? Was stability examined at all? Which functional, and were the compared numbers computed the same way? If there is a machine-learning model, what were its labels, and does the highlighted candidate fall inside the training distribution?

That habit is the durable part. The specific numbers in this field will move; the questions will not.

**Calibration over allegiance.** Computational screening for OER catalysts is neither the solved pipeline that enthusiastic abstracts imply nor the wishful exercise that its critics describe. It is a genuinely useful ranking tool with clearly stated limits, embedded in a workflow whose expensive stages are still DFT and the bench. Knowing exactly where the tool stops being trustworthy is not a reason to distrust it — it is the condition for using it well.

### 🎯 Exercise Problems

  1. **Recall versus precision** : reproduce the funnel arithmetic of Section 5.2 in a few lines of Python, then argue quantitatively why a stage-1 model should be tuned for high recall even at the cost of many false positives. Compute the extra DFT cost of loosening the cut from the best 1% to the best 5%, and compare it with the cost of losing one genuinely good candidate.
  2. **The extrapolation cliff** : modify the ridge code so the training range is \\([-2, 2]\\) instead of \\([-1, 1]\\), leaving everything else fixed. Report how the extrapolation table changes and explain the result in one sentence.
  3. **The penalty's effect** : sweep `lam` over several orders of magnitude and tabulate the fitted coefficients, the interpolation RMSE, and the extrapolation RMSE. At which end of the sweep does the model most resemble ordinary least squares, and what does a very large penalty do to the predictions?
  4. **Applicability domain** : propose a concrete numerical test that flags a candidate as outside the training distribution using only the training feature matrix. Implement it and verify that it fires on the \\([+4, +8]\\) block and stays quiet on the \\([-1, +1]\\) block.
  5. **Diagram versus rate** : sketch two free-energy diagrams that are identical at every intermediate but describe catalysts with very different measured currents. State exactly what physical quantity your sketch is varying and why CHE cannot see it.
  6. **Which limit bites?** : for each of the five limits in Section 5.5, name one experimental observation that would reveal that the limit had been reached — that is, an observation the CHE prediction could not explain.
  7. **Reading a claim** : find a public computational screening study for OER catalysts and answer the six questions listed in Section 5.8. Note explicitly which questions the paper does not give you enough information to answer.

## Summary

This chapter generalized the descriptor idea and then stated the honest limits of everything built on it. **ML screening is the volcano's logic extended** : instead of one physically chosen descriptor, a model learns which combination of cheap composition and structure features predicts the expensive quantity, so most candidates never need a calculation. **The funnel arithmetic** — with invented unit costs — showed that the cheap stage contributes essentially nothing to the total and that all cost lives in DFT and experiment; the quantity that actually matters at stage 1 is **recall**, because a false positive costs one wasted calculation while a false negative deletes a candidate from the universe. **A ridge model in plain NumPy** reproduced the invented rule to within the injected noise inside the training range, and its error grew steadily and silently outside it — roughly linearly in distance, driven by a small coefficient error multiplied by how far out you ask. **Interpolation is a measurement; extrapolation is a guess in the shape of a measurement**, and the only defence is an explicitly stated applicability domain. **Breaking the scaling relation** — differential stabilization of \\(\ast\mathrm{OOH}\\), bifunctional sites, lattice-oxygen participation — is how the volcano ceiling might be raised, and all three are active research directions that buy their freedom by making the system more complicated than the model that produced the volcano. **The five limits of CHE** are the core of the chapter: it is thermodynamics only and says nothing about barriers; it assumes an ideal static surface when real electrodes reconstruct, dissolve, and re-oxidize under OER conditions, so the active site may not be what you computed; solvent and field effects are heavily simplified; stability matters as much as activity and CHE is silent about it; and functional-dependent descriptor errors compound through the scaling relation into predicted overpotentials and into ML training labels. **The field's answers to each** are named rather than developed: microkinetic modelling and transition-state searches, in-situ and operando characterization, explicit-solvent and constant-potential methods, Pourbaix-style stability analysis, and cross-functional benchmarking. **The join with materials informatics** runs both ways — physics supplies features that let small datasets go further, ML supplies reach through screening and Bayesian optimization — with the caution that a model inherits its labels' systematic errors and cannot quantify them.

This completes the *Computational Chemistry of the Oxygen Evolution Reaction* series. You can build a CHE free-energy diagram, read an overpotential off it, explain the scaling relation that caps the volcano, run the screening loop in Python, and put a learned model in front of it. Just as importantly, you can say precisely what each of those results is not. Calibration over allegiance: it is the part of this toolkit that will still be right when the numbers change.

[← Chapter 4: Hands-On: Screening in Python](<chapter-4.html>) [Series Top →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
