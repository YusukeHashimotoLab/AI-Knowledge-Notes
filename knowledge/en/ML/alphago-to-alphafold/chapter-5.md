---
title: "Chapter 5: The Legacy: From Games to Science"
chapter_title: "Chapter 5: The Legacy: From Games to Science"
subtitle: "What Transfers from a Board Game to a Nobel Prize, and What Was Never Solved at All"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/ATsZQZRvZck"
    title="AlphaGo to AlphaFold Ch.5: The Legacy: From Games to Science"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/ML/alphago-to-alphafold/chapter-5.html>) | Last sync: 2026-08-19

[Machine Learning Dojo](<../index.html>) > [AlphaGo to AlphaFold](<index.html>) > Chapter 5

## 5.1 Two Arcs That Appear to Contradict Each Other

Put the two stories of this series side by side and they look like opposites.

**AlphaGo removed human knowledge.** Chapter 2's system began from human expert games; Chapter 3's successor threw them away, along with the hand-built features, the rollout policy, and eventually the game-specific machinery itself. Each subtraction made the system stronger. The narrative arc is one of *deletion*: fewer priors, more self-play, better play.

**AlphaFold injected domain structure.** Chapter 4's system is not a general sequence model pointed at protein data. It is built around multiple sequence alignments — the evolutionary record of which residues mutate together — and around an explicit geometric representation of the protein, with the physical symmetries of three-dimensional space wired into how the network reasons about pairs of residues. The narrative arc is one of *addition*: more structure, more priors, better predictions.

If "remove the priors" were the lesson of AlphaGo, AlphaFold would be a step backwards. It is not. So the lesson has been stated wrong.

Here is the resolution, and it is the single most useful idea in this chapter.

> **Neither system is about priors. Both are about the structure of the problem — specifically, about where a cheap and reliable feedback signal comes from.**
>
> Go can verify itself. Play a game to the end and the rules tell you who won, exactly, for free, as many times as you like. Self-play works because the feedback loop is closed inside the problem. Priors were removable because something else — the rules — was supplying ground truth at unlimited volume.
>
> Protein structure cannot verify itself. You cannot ask a computer whether a predicted fold is correct; only an experiment can say, and experiments are slow and expensive. But nature had already run the experiment, billions of times, and written the results down: **evolution is the labelled dataset**. Sequences that fold have survived; residues in contact mutate in correlated ways. The MSA is not a hand-crafted feature in the old sense. It is a channel through which an enormous amount of already-performed verification reaches the model.

Seen this way, the two systems are the same move. Both located an abundant source of ground truth and built an architecture that could drink from it. AlphaGo found it in the rules; AlphaFold found it in the evolutionary record and in the decades of experimentally solved structures that CASP had accumulated into a benchmark. What looks like "remove priors" versus "add priors" is really "the feedback was already free" versus "the feedback had to be routed in".

**The bottleneck was never the algorithm.** It was, and remains, the availability of a cheap and reliable signal that tells the system whether it was right.

## 5.2 Making the Point Quantitative

The claim above is qualitative, so let us make it concrete on a synthetic search problem — no proteins, no board positions, just a space of candidates with a hidden quality score and three different kinds of feedback available to whoever is searching it.

  * **Regime A — the verifier is free and exact.** This is Go. Evaluate every candidate; keep the best.
  * **Regime B — the verifier is exact but rationed.** This is the wet lab. You may run a small number of definitive tests, and nothing else tells you anything.
  * **Regime C — a cheap noisy proxy, plus a small exact budget.** This is the AlphaFold-shaped situation, and also the shape of every screening pipeline in materials science. Something cheap ranks the whole space imperfectly; the expensive verifier confirms only the top of the ranking.

**Every value below is invented for teaching.** No number corresponds to any real system.

```python
import numpy as np

# --- Three feedback regimes over the SAME synthetic search space ----------
#
# Nothing below is a protein, a board position, or a material. We invent a
# space of candidates with a hidden "quality" score, then ask what a search
# can achieve under three different kinds of feedback:
#
#   A. free exact verifier      -- the game case (self-play scores itself)
#   B. expensive exact verifier -- the bench case (few experiments allowed)
#   C. cheap noisy proxy + a few exact checks -- the AlphaFold-shaped case
#
# The point is the ORDERING of the three, and how C depends on proxy quality.

rng = np.random.default_rng(0)

N_CANDIDATES = 20000    # invented size of the search space
EXACT_BUDGET = 50       # invented number of affordable exact evaluations

# Hidden quality of every candidate (invented; standard normal).
truth = rng.normal(0.0, 1.0, size=N_CANDIDATES)
best_possible = truth.max()


def regime_a():
    """Verifier is free: evaluate everything, keep the best."""
    return truth.max()


def regime_b(rng):
    """Verifier is exact but rationed: sample at random, keep the best."""
    idx = rng.choice(N_CANDIDATES, size=EXACT_BUDGET, replace=False)
    return truth[idx].max()


def regime_c(noise_sd, rng):
    """Cheap noisy proxy ranks everything; the exact budget checks the top."""
    proxy = truth + rng.normal(0.0, noise_sd, size=N_CANDIDATES)
    shortlist = np.argsort(proxy)[-EXACT_BUDGET:]      # proxy's top picks
    return truth[shortlist].max()                      # verified exactly


N_TRIALS = 200

a = regime_a()
b = np.mean([regime_b(rng) for _ in range(N_TRIALS)])

print("Search under three feedback regimes (all values invented)")
print(f"  candidates             : {N_CANDIDATES}")
print(f"  exact-evaluation budget: {EXACT_BUDGET}")
print(f"  trials averaged        : {N_TRIALS}")
print()
print(f"{'regime':<38}{'best found':>11}{'% of ceiling':>14}")
print(f"{'A. free exact verifier':<38}{a:>11.3f}{100 * a / best_possible:>13.0f}%")
print(f"{'B. rationed exact verifier':<38}{b:>11.3f}{100 * b / best_possible:>13.0f}%")

for noise_sd in [0.25, 0.5, 1.0, 2.0, 4.0]:
    c = np.mean([regime_c(noise_sd, rng) for _ in range(N_TRIALS)])
    label = f"C. proxy (noise sd = {noise_sd:.2f}) + budget"
    print(f"{label:<38}{c:>11.3f}{100 * c / best_possible:>13.0f}%")

print()
print(f"ceiling (true best in the space) = {best_possible:.3f}")
```

**Output:**

```
Search under three feedback regimes (all values invented)
  candidates             : 20000
  exact-evaluation budget: 50
  trials averaged        : 200

regime                                 best found  % of ceiling
A. free exact verifier                      3.946          100%
B. rationed exact verifier                  2.261           57%
C. proxy (noise sd = 0.25) + budget         3.946          100%
C. proxy (noise sd = 0.50) + budget         3.939          100%
C. proxy (noise sd = 1.00) + budget         3.770           96%
C. proxy (noise sd = 2.00) + budget         3.293           83%
C. proxy (noise sd = 4.00) + budget         2.846           72%

ceiling (true best in the space) = 3.946
```

**Reading the result.** Three things, and the third is the reason the chapter opened the way it did.

  * **The free verifier wins trivially.** Regime A finds the exact optimum, because nothing stops it from looking at everything. This is the situation self-play enjoys, and it is why the AlphaGo line could afford to delete human knowledge: the deletion cost nothing that the rules were not already supplying for free.
  * **The same budget, spent blindly, recovers barely half the ceiling.** Regime B is not a bad algorithm. It is a good algorithm with nothing to steer it. Fifty exact measurements chosen at random out of twenty thousand candidates is the ordinary condition of experimental science, and it is why progress there is slow — not because the search is stupid, but because the signal is scarce.
  * **A cheap proxy converts regime B into regime A, and how completely it does so depends entirely on how good the proxy is.** With a low-noise proxy, fifty verifications land essentially on the optimum. As the proxy degrades, the advantage decays smoothly back toward the blind baseline. **The proxy is the whole story.** AlphaFold's achievement, in this framing, was building an extraordinarily low-noise proxy for a quantity that previously required an experiment — not inventing a better search.

Everything in the rest of this chapter is a consequence of that last sentence.

## 5.3 What AlphaFold Actually Changed

The prediction system is the part that gets written about. The part that changed daily practice is the **database**.

**Structures on demand.** In 2021, DeepMind and EMBL-EBI launched the AlphaFold Protein Structure Database, releasing predicted structures openly; it was subsequently expanded to cover hundreds of millions of predicted structures across a very large sample of known protein sequences. The practical effect is a change of category. A predicted structure stopped being a project and became a lookup. Before, a researcher with an uncharacterized protein faced a decision — is this worth months of crystallography, or years? After, the same researcher opens a browser and starts from a three-dimensional model with per-residue confidence attached.

That confidence attachment matters more than it sounds. The model does not merely output coordinates; it outputs a per-residue estimate of how much to trust them. Used properly, a low-confidence region is itself information — often a signal that the region is flexible or disordered rather than that the model failed. Used improperly, the confidence is ignored and a low-confidence loop is discussed as though it were a crystal structure.

**Drug discovery: hypothesis generation, not replacement.** This is the claim most often overstated, so state it carefully. A predicted structure helps you *form hypotheses*: where a binding pocket might be, which residues line it, whether a candidate molecule could plausibly fit, which mutations might sit at an interface. Those are real accelerations of the early stage, and they are widely used.

What a predicted structure does not do is tell you whether a molecule binds, whether binding does anything useful in a cell, whether the compound is absorbed, whether it is toxic, or whether the target was the right target. Those questions are answered by assays, by animal models, and eventually by clinical trials, and none of that has been shortened. DeepMind's drug-discovery spinout, **Isomorphic Labs**, exists precisely because turning structural prediction into medicines is a separate, long, and largely experimental undertaking. The honest framing is that structure prediction moved a bottleneck; it did not remove the pipeline behind it.

**Enzyme and protein design.** The adjacent and arguably more transformative direction runs the arrow backwards: rather than predicting the structure of a sequence that exists, design a sequence that will fold into a structure you specify. Computational protein design has produced proteins with no natural counterpart, and the availability of fast, accurate structure prediction feeds it directly — a designed sequence can be checked computationally before anyone synthesizes it. This is regime C from Section 5.2, applied to design instead of search.

**The 2024 Nobel Prize in Chemistry** recognized both halves of that picture, and the split is worth getting right because it is frequently misreported. One half went to **David Baker**, for computational protein design. The other half was awarded **jointly to Demis Hassabis and John Jumper**, for protein structure prediction. Design and prediction, recognized together, as two sides of the same problem.

It is a short arc. A program learned to play a board game; less than a decade later, the line of work it started shared a Nobel Prize in Chemistry.

## 5.4 Ripples into Materials Science

The template that AlphaFold demonstrated — **predict the expensive quantity cheaply, then verify a shortlist** — is not specific to biology. It is the architecture of computational screening in general, and materials science adopted it enthusiastically.

The pattern reads the same way in every domain: a learned model estimates a property that would otherwise require an expensive calculation or a synthesis; the model ranks an enormous candidate space; the expensive method is spent only on the top of the ranking. DeepMind's **GNoME** work on inorganic crystal structure prediction is the most visible example, and follow-on efforts have applied similar pipelines to autonomous synthesis. These efforts **announced large numbers of candidate structures, with independent assessments urging caution** — in particular about how many announced candidates are genuinely new, genuinely stable, and genuinely useful, as opposed to variants of known compounds or entries that a careful materials chemist would not count. That caution is not a dismissal of the work; it is the ordinary and healthy process of a claim meeting its field. We deliberately quote no counts here, because the counts are exactly what is disputed.

The deeper point is the one Section 5.2 already made numerically. In materials science, the verifier is regime B: synthesis and characterization, slow and rationed. A model can only convert that into regime C if the proxy is good, and "good" is measured against the specific decision being made — a model that predicts formation energy well may say nothing about whether a compound can actually be made, whether it survives operating conditions, or whether it is affordable.

**If you came from the MI side of AI Terakoya**, the join is direct. The [Computational Chemistry of OER](<../../MI/oer-computational-chemistry/index.html>) series works through this exact structure in one concrete domain: a physically motivated descriptor stands in for an expensive calculation, a volcano relation turns that descriptor into a ranking, and the final chapter is an unsparing account of what the underlying approximation cannot see. [MI Applications to Catalyst Design](<../../MI/catalyst-mi-application/index.html>) takes the data-driven layer above it — learned property models, uncertainty, Bayesian optimization, active learning. Read either alongside this chapter and the shared skeleton is unmistakable: cheap proxy, expensive verifier, and an argument about how much the proxy can be trusted.

The one asymmetry worth naming: AlphaFold had CASP. Materials science has no single, decades-old, blind, community-wide benchmark of comparable authority. That absence is not a minor administrative gap — it is why claims in the field are harder to adjudicate, and it is the part of the AlphaFold story most worth copying.

## 5.5 Honest Limits: What "Solved" Does Not Mean

The phrase "protein folding is solved" did real damage, in both directions — it oversold the result to outsiders and invited a backlash from people who knew better. The precise version is narrower and more defensible.

> **"Solved" is a claim about a benchmark, not about biology.** The benchmark was: given a single sequence, predict the coordinates of the folded structure of a single chain, evaluated against an experimental structure. On that task, performance reached a level competitive with experiment. Biology is not that task.

Here is what remains open.

**Dynamics and conformational ensembles.** A protein is not a statue. Many function by moving — opening and closing, shifting between states, transmitting a change at one site to a distant one. A single predicted structure is one snapshot of an ensemble, and it is generally the snapshot most consistent with the evolutionary signal rather than a description of the distribution. The question "which conformations does this protein visit, and with what probabilities" is not the question that was answered, and it is often the question that determines function.

**Complexes and interactions.** Proteins act in assemblies and in partnership with nucleic acids, small molecules, ions, and each other. Predicting a chain in isolation is a different and easier problem than predicting what it does when bound. **AlphaFold 3**, announced in 2024, extends the approach beyond single chains toward complexes including interactions with other molecular classes — a genuine and substantial extension of scope. It is still prediction, still benchmarked, and still subject to every caveat in this section; treat its outputs the way you would treat any confident model output, which is to say as a hypothesis with a confidence estimate attached.

**Intrinsically disordered regions.** A large fraction of the proteome does not adopt a single stable fold at all, and disorder is functional rather than defective — it is how many regulatory and signalling proteins work. For these regions the correct output is not a structure. A model that returns low confidence there is behaving well; a user who renders that region as a definite shape and reasons about it is not.

**Mutation effects.** "Does this single amino-acid change break the protein?" is one of the most clinically valuable questions in the field, and it is not the question the model was trained on. Structure prediction is heavily driven by the evolutionary record of a family, and a single point mutation barely perturbs that record — so predictions for a variant and for the wild type can look nearly identical even when the biological consequences are severe. Predicting stability change, loss of function, or pathogenicity is an adjacent problem with its own methods, its own benchmarks, and considerably more modest performance.

**And the input dependence.** Prediction quality is tied to the depth and quality of the available evolutionary information. Proteins from sparsely sampled families, engineered sequences with no natural relatives, and designed proteins are the harder cases precisely because the channel that carries the free verification is thin.

None of this diminishes what was accomplished. It locates it. A benchmark was conquered; a large and important family of adjacent questions was not.

## 5.6 What Carries Over to Your Own Work

Strip away the compute, the specialized architectures, and the institutional resources, and three transferable habits remain. They cost nothing and they are the actual content of this series.

**1. Build verification-rich loops.** Before choosing a model, ask what will tell you whether an output is right, how much each check costs, and how many you can afford. If the answer is "a slow human review, occasionally", no architecture will rescue the project — you are stuck in regime B. Effort spent making feedback cheaper, faster, or more automatic usually beats the same effort spent on the model. Where an exact check is unaffordable, the highest-value engineering task is constructing a proxy and then honestly measuring its noise, because Section 5.2 says the noise level is what determines the payoff.

**2. Use domain priors as scaffolding, not as decoration.** AlphaFold's geometric and symmetry structure was not ornamental; it encoded the fact that the answer lives in three-dimensional space and does not change when you rotate it. Good priors express something true and constraining about the problem, and they let a model learn from far less data. Bad priors are guesses about the solution dressed up as architecture. The test is whether you can state, in one sentence, the fact about the world that the prior encodes. If you cannot, it is decoration.

**3. Take benchmarks seriously, and take their limits seriously too.** CASP is the underrated character in this story. A blind, independently assessed, community-run evaluation held on a regular cadence, with targets whose answers nobody in the competition knows — that is the institution that made the result *credible* rather than merely claimed. Any field that wants comparable clarity has to build comparable infrastructure. And the same discipline demands the other half: state plainly what the benchmark measures and what it does not, because the gap between "state of the art on this benchmark" and "solved" is exactly where overclaiming lives.

A fourth, less comfortable habit belongs here too. **Notice which of your problems is genuinely a regime-A problem.** Games, simulations, code that either compiles or does not, theorems a checker can verify — these have cheap, exact feedback and are the natural homes for large-scale learned search. Most real problems do not. Recognizing which kind you have in front of you, early, is worth more than any technique in this series.

## 5.7 Series Conclusion: Calibration over Allegiance

Trace the line from the beginning. Chapter 1 set up games as a proving ground and explained why Go was the one that would not fall to the methods that beat chess — a branching factor that defeats exhaustive search and a position-evaluation problem nobody knew how to write down. Chapter 2 showed how AlphaGo combined a learned evaluation with Monte Carlo tree search so that the network told the search where to look and the search told the network what mattered. Chapter 3 removed the human games and then the game-specific machinery, and the systems got stronger; AlphaGo itself was retired in 2017, the same year a documentary put the Lee Sedol matches in front of a general audience. Chapter 4 took the same institution — DeepMind — into a problem where self-play is impossible, and found that evolution had already run the experiments. This chapter has argued that these are one story, not two.

In under a decade, a line of work that began with a board game contributed to a Nobel Prize in Chemistry. That is a remarkable fact and it deserves to be stated plainly. It also deserves the qualifications this chapter has spent its length assembling: a benchmark was conquered rather than a science completed; the drug pipeline was accelerated at its front end rather than replaced; the materials-science analogues are promising and contested; and the questions that remain — dynamics, complexes, disorder, variants — are not footnotes but the substance of what proteins actually do.

**Calibration over allegiance.** There are two easy positions available on all of this, and both are lazy. One treats each announcement as evidence that the remaining problems are a formality. The other treats every result as marketing and every limitation as a refutation. Neither requires you to read anything carefully.

The useful position is harder and duller: figure out exactly what was demonstrated, exactly what was measured, exactly what the measurement omits, and exactly which of your own problems shares the structure that made the demonstration work. That habit outlasts every specific system named in this series. The results will be superseded — some already have been. The questions you learn to ask about them will not.

### 🎯 Exercise Problems

  1. **The apparent contradiction** : in your own words, in under 150 words, explain to a skeptical colleague why "AlphaGo removed priors, AlphaFold added them" is not a contradiction. Your answer must mention where the ground truth comes from in each case.
  2. **Proxy quality** : modify the Section 5.2 code so that the proxy is correlated with truth only over part of the space — say, accurate for candidates whose true value is below the median and pure noise above it. Predict what happens to regime C before running it, then run it and explain any discrepancy.
  3. **Budget versus proxy** : hold the proxy noise fixed at 2.0 and sweep `EXACT_BUDGET` over several values. Compare the gain from doubling the budget against the gain from halving the noise. Which is the better investment, and does the answer depend on where you start?
  4. **Regime audit** : take one problem from your own work. Write down what the exact verifier is, what it costs, what cheap proxy exists (if any), and how noisy that proxy is. Classify the problem as A, B, or C, and state what would move it one letter to the left.
  5. **Reading a claim** : find a public announcement of an AI-driven discovery in any scientific field. Identify (a) what quantity was predicted, (b) what quantity was verified, (c) how many candidates were verified versus announced, and (d) whether an independent party did any of the verifying. Note which of the four you cannot determine from the announcement.
  6. **The benchmark question** : CASP made AlphaFold's result credible. Design, in half a page, a CASP-equivalent for a problem in your own field. Specify who holds the answers, how targets are chosen, how entries are blinded, and what would make the results uninformative.
  7. **What "solved" excludes** : for each of the four open problems in Section 5.5, write one sentence describing a concrete biological situation in which a single high-confidence predicted structure would lead a researcher to the wrong conclusion.

## Summary

This chapter resolved the apparent contradiction between the two arcs of the series and then drew the honest boundary around what was achieved. **AlphaGo removed human knowledge and AlphaFold injected domain structure**, and both were right, because neither story is really about priors: **self-play worked because games verify themselves, and AlphaFold worked because evolution had already run the experiment**. The shared principle is search and learning shaped by the structure of the problem, and the binding constraint in both cases is **the availability of a cheap, reliable feedback signal**. **A synthetic three-regime simulation** made this quantitative with invented values: a free exact verifier reaches the ceiling, the same budget spent blindly recovers barely half of it, and a cheap proxy recovers nearly all of it — with the recovery decaying smoothly as the proxy gets noisier, which is the entire engineering problem in one line. **What AlphaFold changed in practice** was largely the database: the AlphaFold DB launched in 2021 with EMBL-EBI and later expanded to hundreds of millions of predicted structures, turning a project into a lookup; drug discovery gained hypothesis generation at the front of the pipeline rather than a replacement for experiments, as the existence of Isomorphic Labs illustrates; and the **2024 Nobel Prize in Chemistry** recognized one half to David Baker for computational protein design and the other half jointly to Demis Hassabis and John Jumper for protein structure prediction. **The template rippled into materials science** as predict-then-verify screening, with GNoME and its successors announcing large numbers of candidate structures and independent assessments urging caution — which is why this chapter quotes no counts, and why the missing piece is a CASP-grade community benchmark rather than a better model. **The honest limits** are that "solved" describes a benchmark and not biology: dynamics and conformational ensembles, complexes and interactions (extended toward by AlphaFold 3 in 2024), intrinsically disordered regions, and mutation effects all remain open, and prediction quality still depends on how much evolutionary information is available. **What carries over** to any reader's own work is three habits — build verification-rich loops and invest in making feedback cheap, use domain priors that encode a stateable fact about the world rather than decoration, and treat benchmarks as the infrastructure that makes results credible while stating plainly what they omit.

This completes the *AlphaGo to AlphaFold* series. In under a decade, a line of work that began with a board game contributed to a Nobel Prize in Chemistry — and the useful response to that is neither enthusiasm nor suspicion, but the discipline of asking what was demonstrated, what was measured, what the measurement leaves out, and which of your own problems shares the structure that made it work. Calibration over allegiance.

[← Chapter 4: AlphaFold: The Protein Folding Breakthrough](<chapter-4.html>) [Series Top →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
