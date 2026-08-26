---
title: "Chapter 4: AlphaFold: The Protein Folding Breakthrough"
chapter_title: "Chapter 4: AlphaFold: The Protein Folding Breakthrough"
subtitle: "From a String of Amino Acids to a Three-Dimensional Shape, and How Evolution Turned Out to Be the Training Data"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/Z9WEyF6_NY8"
    title="AlphaGo to AlphaFold Ch.4: AlphaFold: The Protein Folding Breakthrough"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/ML/alphago-to-alphafold/chapter-4.html>) | Last sync: 2026-08-19

[Machine Learning Dojo](<../index.html>) > [AlphaGo to AlphaFold](<index.html>) > Chapter 4

Chapter 3 ended on a boundary. The self-play arc works where experience is cheap and the reward is clean — and almost nothing in the natural sciences is like that. There is no simulator for biology you can run a billion times. There is no score at the end of an experiment that tells you unambiguously that you won.

So when the same organisation turned to one of biology's oldest open problems, none of the machinery from the first half of this series transferred. No self-play. No reward function. No search over moves. What transferred was something less specific and more durable: a willingness to ask what the *right representation of the problem* is, and the discovery that for this problem, **the data had already been generated — by evolution, over several billion years, and stored in the sequences themselves**.

## 4.1 The Problem: From a String to a Shape

A protein is manufactured as a linear chain. A gene specifies a sequence of **amino acids**, twenty of them in the standard set, strung together in order like letters in a word. That chain is the protein's *primary structure*, and writing it down is easy — sequencing technology has made amino acid sequences abundant and cheap.

But a linear chain does almost nothing. Within moments of being made, the chain **folds** — spontaneously, in three dimensions, into a specific and reproducible shape. And it is the shape that does the work.

**Structure determines function.** An enzyme works because its folded form creates a pocket geometrically and chemically complementary to the molecule it acts on. An antibody binds its target because a particular arrangement of loops presents a matching surface. A membrane channel lets one ion through and not another because of the precise dimensions of the passage it forms. Change the shape and the function changes or disappears; a great many diseases are, at bottom, proteins that have adopted the wrong shape.

This means that knowing a protein's structure is often the difference between having a name for a molecule and understanding what it does. Which brings the obvious question: given the sequence, can you predict the shape?

For decades the reliable answer was: not really — go and measure it. And measuring is hard.

  * **X-ray crystallography** requires persuading the protein to form an ordered crystal, then inferring the atomic arrangement from a diffraction pattern. Crystallisation is notoriously unpredictable; some proteins have resisted it for decades, and membrane proteins are especially uncooperative.
  * **Cryo-electron microscopy** freezes many copies of the molecule and reconstructs a three-dimensional density from enormous numbers of noisy two-dimensional images. It sidesteps crystallisation and has transformed the field, but it demands specialised instruments and substantial expertise.
  * **NMR spectroscopy** works in solution and gives dynamic information, but becomes progressively harder as the protein gets larger.

All of them are slow, expensive, and skilled work. The consequence was a widening gap: **sequences accumulated far faster than structures**. Vast numbers of proteins were known only as strings, with no picture of what they were.

### 📚 Anfinsen: The Information Is in the Sequence

The theoretical licence to attempt prediction at all comes from a set of experiments by Christian Anfinsen and colleagues, carried out through the 1960s and into the 1970s, and recognised with a share of the Nobel Prize in Chemistry in 1972.

The experiment is simple to describe. Take a folded, functional protein. Treat it so that it unfolds completely and loses its activity. Then carefully remove the treatment and restore ordinary conditions.

The protein **folds back up on its own** — into the same structure, recovering its function — with no cellular machinery present to guide it.

The conclusion drawn from this is **Anfinsen's thermodynamic hypothesis**: for such a protein, the native structure is the one that is thermodynamically favoured under physiological conditions, and **all the information needed to specify it is already contained in the amino acid sequence**. Nothing external is dictating the fold.

This is the permission slip for everything that follows. If the sequence did not determine the structure, sequence-to-structure prediction would be attempting the impossible. Anfinsen says the information is there. He does not say it is easy to read.

### 📚 Levinthal's Paradox: Why Search Is Not the Answer

Cyrus Levinthal pointed out an apparent contradiction that shaped how everyone thought about the problem afterwards.

Consider a protein chain of realistic length. Every bond along the backbone can adopt several distinct orientations. The number of possible three-dimensional configurations is therefore **exponential in the length of the chain** — an astronomically large number, vastly beyond anything that could be sampled.

Yet real proteins fold in a fraction of a second.

The two statements cannot both describe a random search. If a protein tried configurations one after another until it found the right one, folding would take longer than the age of the universe. It plainly does not.

The resolution is that **folding is not a search — it is a descent**. The energy landscape is not flat with a single hidden hole in it; it is shaped like a **funnel**, biased so that partially correct arrangements are already energetically favourable and the chain is guided downhill toward the native state. Local structure forms fast, and it constrains what can happen next.

Levinthal's paradox is stated here for a specific reason. It says that **brute enumeration of configurations is the wrong tool** — and by extension that a physics simulation trying to watch every atom move is fighting the problem the hard way. Something else was needed. It turned out to be statistics.

## 4.2 CASP: The Field's Honest Scoreboard

Structure prediction had a problem that plagues many computational fields: it is very easy to fool yourself. If you know the answer, you will tune your method until it produces the answer, and you will believe you have a method.

**CASP** — the Critical Assessment of Structure Prediction — was created to make that impossible, and it has run **every two years since 1994**.

The design is what makes it trustworthy. Experimentalists who have solved a structure but not yet published it contribute the sequence. Predictors receive only the sequence and submit models **blind** — nobody outside the experimental group knows the answer. Independent assessors then compare predictions against the withheld experimental structure.

You cannot tune to the answer, because you do not have it. CASP is therefore one of the more honest scoreboards in computational science, and it turned structure prediction into a field with a clear, community-wide, unfakeable measure of progress. Reading its history is reading the field's real trajectory rather than its press releases — and for a long time that trajectory was **steady, hard-won, incremental improvement**, with prediction useful for proteins resembling something already solved and unreliable otherwise.

Then two CASPs changed it.

**CASP13, in 2018**, was led by **AlphaFold** — the first version. Its approach was recognisably in the tradition that preceded it: use deep learning to predict, for each pair of residues, whether they are in **contact** in the folded structure (later refined to predicting distributions over distances), then use those predicted constraints to build a three-dimensional model. What was new was how much better the deep-learning-based prediction of those pairwise relationships had become. AlphaFold 1 was a leading system, and it was clearly a step forward — but it was a step, and the field's overall picture had not changed.

**CASP14, in 2020**, was different in kind. **AlphaFold 2** produced predictions that, for a large fraction of targets, were competitive with experimental structures. The assessors' verdict was strong enough that the problem was **widely described as essentially solved for single protein domains** — a phrase worth handling carefully, and Section 4.4 does exactly that. The method was published in *Nature* in 2021.

## 4.3 How AlphaFold 2 Works, at the Level of Principle

What follows describes the *ideas*, not the implementation. The architecture has many components and a great deal of engineering detail; five principles carry most of the conceptual weight.

### 📚 (1) Evolution as Data: Reading Contacts from an Alignment

This is the deepest idea, and it is the one that connects the two halves of this series. AlphaFold 2 does not receive only the sequence you asked about. It receives a **multiple sequence alignment (MSA)**: that sequence together with many evolutionarily related sequences — the same protein as it appears across a wide range of organisms — lined up so that corresponding positions sit in the same column.

Why does this help so enormously? Because of **co-evolution**. Here is the logic, with a concrete pair.

Suppose that in the folded structure, position 30 and position 85 sit right next to each other, even though they are far apart along the chain. Suppose further that in this organism, position 30 carries a **positively charged** residue and position 85 a **negatively charged** one, and the attraction between them — a salt bridge — helps hold the fold together.

Now let evolution run across millions of years and thousands of species.

  * A mutation that flips position 30 from positive to negative puts two negative charges next to each other. They repel. The fold is destabilised, the protein works less well, and selection removes that variant.
  * But a variant that flips **position 30 to negative and position 85 to positive** restores the attraction. The fold survives. Selection lets it through.

So across the alignment you observe something distinctive: **neither column is conserved** — both positions vary freely across species — **but they do not vary independently**. Their changes are correlated. When one flips, the other flips to compensate.

That correlation is a fossil record of a spatial constraint. Two positions that are far apart in the chain but adjacent in space leave a **statistical signature in the alignment**, and it can be read out without knowing anything about the structure at all.

This is why the problem was tractable despite having relatively few experimental structures. The training data was never limited to solved structures. **Every sequenced genome is an evolutionary experiment whose results have already been recorded**, and there are a great many of them. Section 4.5 makes this signal concrete and measurable.

Two cautions come with it, both of which the real system must handle and both of which appear in our code. First, related sequences are related **by descent**, so any two positions may look correlated simply because the sequences share ancestry rather than because they touch — a confound that must be corrected for. Second, a position that never varies carries no signal at all: perfectly conserved columns tell you nothing about who they touch.

### 📚 (2) The Pair Representation and the Evoformer

AlphaFold 2 maintains **two representations simultaneously**, and keeping them separate is central to the design.

  * The **MSA representation** describes the aligned sequences — what is happening at each position, across all those evolutionary variants.
  * The **pair representation** describes **relationships between pairs of positions** — an entry for every pair \\((i, j)\\), encoding the network's current belief about how those two residues relate to each other in space.

The pair representation is where the structure implicitly lives before any coordinates exist. It is the network's evolving hypothesis about the geometry.

The **Evoformer** is the block that processes both. Its defining feature is that the two representations **exchange information repeatedly, in both directions**, using attention.

  * Information flows from the MSA into the pair representation: co-evolutionary signals between columns become evidence about which pairs are close.
  * Information flows back from the pair representation into the MSA: current geometric beliefs change how the alignment should be read.
  * The pair representation is also updated against **itself** — and this is the subtle and important part. Geometry has to be self-consistent. If \\(i\\) is close to \\(j\\), and \\(j\\) is close to \\(k\\), that constrains how far \\(i\\) can be from \\(k\\). The triangle inequality is not optional. Updates that let triples of pairs reason about one another push the pair representation toward geometrically coherent configurations rather than a bag of independent guesses.

Repeating this exchange many times constitutes **iterative refinement**. A first pass produces a rough, partly inconsistent picture; each further pass reconciles it, propagating strong local evidence outward to resolve weaker constraints elsewhere. The network is not computing an answer in one shot — it is converging on one.

### 📚 (3) The Structure Module: From Relationships to Coordinates

Everything so far is abstract: representations of positions and pairs. At some point an actual three-dimensional object must come out.

The **structure module** takes the refined representations and produces **explicit 3D coordinates**. The key design decision is that it does not predict coordinates as unconstrained numbers. It works with the geometry directly — treating each residue as a rigid local frame with a position and an orientation, and predicting how those frames are arranged in space.

This matters because **structures obey constraints that arbitrary lists of numbers do not**. Bond lengths do not vary freely. Atoms cannot occupy the same space. And a protein's structure is the same structure regardless of how you rotate or translate it — so the representation should not depend on an arbitrary choice of coordinate frame. Building this geometry into the module means the network spends its capacity on the question that is actually open — how the pieces are arranged relative to one another — rather than on re-deriving facts about chemistry and Euclidean space that were never in doubt.

The parallel to MuZero in Chapter 3 is worth noticing. In both cases the win comes from choosing a representation aligned with what the system actually needs to produce, and refusing to spend capacity on anything else.

### 📚 (4) Recycling: Running the Whole Thing Again

There is a further loop wrapped around everything above. The outputs of a complete pass — the refined representations and the predicted structure — are fed **back into the input** of another complete pass. This is called **recycling**.

The effect is that the network gets to revise its own answer with the benefit of having seen it. A first pass may place one region confidently and leave another ambiguous; on the next pass, the confident region is available as context that constrains the ambiguous one. It is iterative refinement again, at the scale of the whole network rather than a single block.

This is a genuinely useful engineering idea beyond biology: **let the model see its own draft**, and give it the opportunity to reconcile the parts that do not fit.

### 📚 (5) Confidence: Knowing What It Does Not Know

A prediction without a reliability estimate is dangerous, because users cannot tell a confident answer from a guess. AlphaFold 2 outputs, alongside the structure, a **per-residue confidence score called pLDDT**.

Roughly: for each residue, how confident is the model that it has placed this part of the structure correctly? High confidence marks regions to be taken seriously; low confidence marks regions to be treated with caution — and, in practice, low-confidence regions often correspond to parts of the protein that are genuinely **disordered**, having no single well-defined structure to predict.

The value of this cannot be overstated for scientific use. **A model that reports where it is unreliable can be used responsibly**; one that presents every prediction with equal apparent authority cannot. A researcher can build on the confident regions and design experiments around the uncertain ones. This is a large part of why the predictions became genuinely usable rather than merely impressive.

That usability was then delivered at scale. In 2021, DeepMind and **EMBL-EBI** launched the **AlphaFold Protein Structure Database**, making predicted structures openly available — turning a method into infrastructure that any researcher could consult.

## 4.4 What It Solved, and What It Did Not

The phrase "the protein folding problem is solved" needs unpacking, because the honest version is more interesting than the headline.

**What genuinely changed.** For **single, well-behaved protein domains** with sufficient evolutionary information available, prediction moved from *often unreliable* to *usually accurate enough to be scientifically useful*. Combined with open release of predictions at scale, that is a change in what routine biological research can assume it has access to. It is difficult to overstate the practical significance.

**What remains hard.** All of the following are active problems, and none was resolved by CASP14.

  * **Complexes.** Proteins mostly do their jobs by binding other things — other proteins, nucleic acids, small molecules. Predicting how two proteins dock together is a harder problem than predicting either one alone, and later systems have addressed it with real progress rather than closure.
  * **Dynamics.** A predicted structure is a single static picture. Real proteins move, and the motion is frequently the mechanism: enzymes open and close, transporters alternate between conformations, signalling proteins switch between states. A method that outputs one shape does not tell you which shapes a molecule alternates between, or how it gets from one to the other.
  * **Disordered regions.** A substantial fraction of protein sequence has no single stable fold — sometimes adopting structure only on binding a partner, sometimes remaining flexible throughout. "Predict the structure" is not a well-posed question for these regions. Confidence scores flag them, which is the right behaviour, but flagging is not solving.
  * **Mutation effects.** A single amino acid change can destroy a protein's function while barely altering its predicted structure. Since the method leans heavily on evolutionary statistics across many sequences, reasoning about the consequences of *one specific variant in one individual* is not what it was built to do, and it should not be assumed to do it well.
  * **The dependence on evolutionary depth.** The co-evolution signal in Section 4.3 requires many related sequences. Proteins with few known relatives — including designed proteins and some rapidly evolving families — provide a much thinner alignment, and performance is correspondingly less assured.

The correct summary is: **a long-standing problem was reduced to a much smaller one, and the remaining problem is still real.** That framing is not deflationary. It is what makes the achievement legible — and it is what tells the next researcher where the open questions are.

## 4.5 Hands-On: Reading Contacts Out of Evolution

The co-evolution argument in Section 4.3 is the conceptual heart of this chapter, and it is the kind of claim that deserves to be checked rather than accepted. Here it is as something measurable.

We build a synthetic family of related sequences over a small alphabet of four residue classes — hydrophobic, polar, basic, acidic — in which three chosen **position pairs** mutate together, exactly as a salt bridge would force them to. Then we throw the structure away, keep only the alignment, and try to recover which pairs were in contact using **mutual information** alone:

\\[ I(i, j) \;=\; H(i) + H(j) - H(i, j) \\]

which is zero when two columns vary independently and large when knowing one tells you about the other.

We also plant a trap. Sequences come from three **lineages**, and eight columns drift according to lineage. Every pair among those eight will look correlated without ever touching in space — the shared-ancestry confound named in Section 4.3. The **average product correction (APC)** is the standard remedy: subtract from each pair the correlation attributable to each column's general tendency to correlate with everything.

```python
import numpy as np

# ---------------------------------------------------------------
# Evolution as data: reading spatial contacts out of an alignment.
# We build a synthetic family of related sequences in which a few
# chosen position PAIRS mutate together, the way two residues that
# touch in the folded structure must. Then we throw the structure
# away, keep only the alignment, and try to recover which pairs were
# in contact -- from statistics alone. This is the co-evolution
# signal that AlphaFold's input is built on.
# ---------------------------------------------------------------
rng = np.random.default_rng(20201130)

N_SEQ, L = 800, 24                 # sequences, positions (columns)
ALPHABET = np.array(list("hp+-"))  # reduced classes: hydrophobic/polar/basic/acidic
Q = len(ALPHABET)

# Ground truth. The number is the probability that the partner residue
# "answers" the first one -- how tightly the pair is constrained.
CONTACTS = {(4, 19): 0.95,         # a tight pair
            (9, 14): 0.90,         # another tight pair
            (2, 21): 0.60}         # a loose pair: real, but noisy
PARTNER = np.array([2, 3, 0, 1])   # h<->+, p<->- : the "answering" rule

# A confound with no structural meaning: shared ancestry. Sequences come
# from three lineages, and these columns drift by lineage. Every pair
# among them looks correlated without ever touching in space.
LINEAGES, LINEAGE_COLS = 3, [0, 3, 6, 7, 11, 15, 18, 22]


def make_alignment():
    """Generate the (N_SEQ, L) alignment of integer-coded residues."""
    # per-column background composition; small Dirichlet alpha => conserved
    conc = rng.lognormal(mean=0.0, sigma=1.0, size=L)
    for col in [c for pair in CONTACTS for c in pair]:
        conc[col] = 2.0            # a column that never varies has no signal
    background = np.array([rng.dirichlet(np.full(Q, c)) for c in conc])
    lineage_comp = {(col, lin): rng.dirichlet(np.full(Q, 0.6))
                    for col in LINEAGE_COLS for lin in range(LINEAGES)}
    lineage_of = rng.integers(0, LINEAGES, size=N_SEQ)

    msa = np.zeros((N_SEQ, L), dtype=np.int8)
    for n in range(N_SEQ):
        for col in range(L):
            p = (lineage_comp[(col, lineage_of[n])] if col in LINEAGE_COLS
                 else background[col])
            msa[n, col] = rng.choice(Q, p=p)
        for (i, j), strength in CONTACTS.items():
            if rng.random() < strength:      # the partner answers
                msa[n, j] = PARTNER[msa[n, i]]
    return msa


def entropy(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def joint_counts(col_i, col_j):
    return np.bincount(col_i.astype(int) * Q + col_j.astype(int),
                       minlength=Q * Q).reshape(Q, Q)


def mutual_information(col_i, col_j):
    """Plug-in MI in bits: H(i) + H(j) - H(i, j)."""
    jc = joint_counts(col_i, col_j)
    return entropy(jc.sum(axis=1)) + entropy(jc.sum(axis=0)) - entropy(jc.ravel())


msa = make_alignment()
print(f"Alignment: {N_SEQ} sequences x {L} positions, alphabet '{''.join(ALPHABET)}'")
for n in range(5):
    print("   " + "".join(ALPHABET[msa[n]]))

# --- The co-evolution story, made concrete ----------------------
i, j = 4, 19
jc = joint_counts(msa[:, i], msa[:, j])
print(f"\nJoint counts for the true contact ({i}, {j}); rows = position {i}:")
print("        " + "".join(f"{c:>7}" for c in ALPHABET))
for a in range(Q):
    print(f"     {ALPHABET[a]}  " + "".join(f"{jc[a, b]:>7}" for b in range(Q)))
print(f"   Marginals: position {i} {jc.sum(axis=1)}, position {j} {jc.sum(axis=0)}")
print("   Each row has ONE dominant column: when position 4 changes,")
print("   position 19 changes with it. Neither column alone is conserved.")

# --- Score every pair -------------------------------------------
mi = np.zeros((L, L))
for a in range(L):
    for b in range(a + 1, L):
        mi[a, b] = mi[b, a] = mutual_information(msa[:, a], msa[:, b])

# Average product correction: subtract the MI a pair would show purely
# from each column's general tendency to correlate with everything.
row_mean = mi.sum(axis=1) / (L - 1)
mi_apc = mi - np.outer(row_mean, row_mean) / row_mean.mean()
np.fill_diagonal(mi_apc, 0.0)

PAIRS = [(a, b) for a in range(L) for b in range(a + 1, L)]


def label(pair):
    if pair in CONTACTS:
        return f"TRUE CONTACT (coupling {CONTACTS[pair]:.2f})"
    if pair[0] in LINEAGE_COLS and pair[1] in LINEAGE_COLS:
        return "lineage artefact"
    return ""


METHODS = [("raw MI", "raw mutual information", mi),
           ("MI-APC", "MI after average product correction", mi_apc)]
rankings = {}
for short, name, score in METHODS:
    order = sorted(PAIRS, key=lambda ab: score[ab], reverse=True)
    rankings[short] = order
    print(f"\nTop 8 pairs by {name}:")
    print(f"   {'rank':>4} {'pair':>10} {'score':>9}   note")
    for r, pair in enumerate(order[:8], 1):
        print(f"   {r:>4} {str(pair):>10} {score[pair]:>9.4f}   {label(pair)}")

print(f"\nRank of each true contact, out of {len(PAIRS)} candidate pairs:")
print(f"   {'pair':>10} {'coupling':>9} " + " ".join(f"{s:>8}" for s in rankings))
for pair, strength in CONTACTS.items():
    ranks = " ".join(f"{order.index(pair) + 1:>8}" for order in rankings.values())
    print(f"   {str(pair):>10} {strength:>9.2f} {ranks}")

off = mi[np.triu_indices(L, 1)]
print(f"\nBackground MI over all {len(PAIRS)} pairs: median {np.median(off):.4f}, "
      f"95th pct {np.percentile(off, 95):.4f} bits")
print(f"Strongest true contact (9, 14): {mi[9, 14]:.4f} bits "
      f"-- {mi[9, 14]/np.median(off):.0f}x the median")
```

**Output:**

```
Alignment: 800 sequences x 24 positions, alphabet 'hp+-'
   ---+h-h-ph+++-h---++hp-h
   -phh-+p+ph+p+-+---hpp+hh
   pphhh-+pph+h+-+h--++p+-h
   --+h-+p-+php+--h--+pp+hh
   -h-+h++-p++p+-hh--++pp+h

Joint counts for the true contact (4, 19); rows = position 4:
              h      p      +      -
     h        3      3    448     11
     p        0      1      0    109
     +        9      1      0      0
     -        1    207      4      3
   Marginals: position 4 [465 110  10 215], position 19 [ 13 212 452 123]
   Each row has ONE dominant column: when position 4 changes,
   position 19 changes with it. Neither column alone is conserved.

Top 8 pairs by raw mutual information:
   rank       pair     score   note
      1    (9, 14)    1.3888   TRUE CONTACT (coupling 0.90)
      2    (4, 19)    1.2343   TRUE CONTACT (coupling 0.95)
      3    (2, 21)    0.5431   TRUE CONTACT (coupling 0.60)
      4    (6, 15)    0.2733   lineage artefact
      5    (6, 22)    0.2575   lineage artefact
      6     (6, 7)    0.2383   lineage artefact
      7     (3, 6)    0.2303   lineage artefact
      8    (6, 11)    0.2265   lineage artefact

Top 8 pairs by MI after average product correction:
   rank       pair     score   note
      1    (9, 14)    1.2302   TRUE CONTACT (coupling 0.90)
      2    (4, 19)    1.1050   TRUE CONTACT (coupling 0.95)
      3    (2, 21)    0.5110   TRUE CONTACT (coupling 0.60)
      4    (6, 15)    0.1664   lineage artefact
      5    (6, 22)    0.1547   lineage artefact
      6     (6, 7)    0.1399   lineage artefact
      7     (3, 6)    0.1398   lineage artefact
      8    (6, 11)    0.1369   lineage artefact

Rank of each true contact, out of 276 candidate pairs:
         pair  coupling   raw MI   MI-APC
      (4, 19)      0.95        2        2
      (9, 14)      0.90        1        1
      (2, 21)      0.60        3        3

Background MI over all 276 pairs: median 0.0071, 95th pct 0.1184 bits
Strongest true contact (9, 14): 1.3888 bits -- 195x the median
```

**Reading the result.** Five observations, and the last two are the ones that keep you honest.

  * **The joint count table is the whole argument in one picture.** Look at the marginals: position 4 takes all four residue classes across the alignment (465, 110, 10, 215), and so does position 19 (13, 212, 452, 123). Neither column is conserved — either one, read alone, looks like a freely varying position with nothing to say. But the joint table has **one dominant entry per row**: when position 4 is `h`, position 19 is almost always `+` (448 of 465); when position 4 is `-`, position 19 is almost always `p` (207 of 215). The constraint is invisible in either column and unmistakable in the pair. That is the salt-bridge story of Section 4.3, in counts.

  * **All three true contacts top the ranking.** Out of **276** candidate pairs, the three planted contacts occupy ranks **1, 2 and 3** — including the deliberately loose pair (2, 21), whose partner answers only 60% of the time. Its score (0.5431 bits) is well below the tight pairs but still far above everything else. The signal survives substantial noise, which is exactly the property that makes it usable on real alignments where constraints are soft and partially satisfied.

  * **The confound is real, and the correction works.** Ranks 4 through 8 are entirely **lineage artefacts** — pairs of columns that share nothing but ancestry. Raw MI puts the strongest of them at 0.2733 bits, comfortably above the background median of 0.0071. If you had no ground truth, you would take them for contacts. Applying APC cuts that pair from 0.2733 to 0.1664 — a reduction of roughly 39% — while the top true contact falls only from 1.3888 to 1.2302, about 11%. The correction suppresses the confound roughly three times harder than it suppresses the signal, which is why the real systems use it.

  * **Note that the ranking is not ordered by coupling strength.** Pair (9, 14) has coupling 0.90 and outranks (4, 19) at coupling 0.95. This is not a bug and it is worth understanding: **mutual information mixes the strength of the coupling with how variable the columns are**. A slightly weaker constraint on a more evenly varying pair of columns carries more bits than a slightly stronger constraint on a lopsided pair. Any single-number contact score has this ambiguity baked in, so a high score is evidence of a constraint but not a measurement of how tight it is.

  * **This is easier than the real problem, in four specific ways.** Our alphabet has four classes rather than twenty, so the joint distribution has few enough cells to estimate from 800 sequences. Real alignments are much sparser per cell, and plug-in MI is biased upward when counts are thin. Our sequences are independent within a lineage; real ones sit on a phylogenetic tree with far more structure than three flat groups. Our couplings are direct and pairwise; real correlations propagate — if \\(i\\) touches \\(j\\) and \\(j\\) touches \\(k\\), then \\(i\\) and \\(k\\) will appear correlated without touching, and disentangling direct from indirect coupling is a substantial problem in its own right. And most importantly, **a list of likely contacts is not a structure.** Turning pairwise constraints into coordinates is the job of everything in Sections 4.3(2) and 4.3(3), and it is where the difficulty actually concentrates.

Try setting the loose pair's coupling from `0.60` down to `0.35` and re-running. Watch where it lands in the ranking, and how close its score gets to the lineage artefacts. That crossover — where a genuine but weak constraint becomes indistinguishable from a confound — is the practical limit of the whole approach, and it is the reason a real system needs far more than mutual information.

### 🎯 Exercise Problems

  1. **The permission slip.** State Anfinsen's thermodynamic hypothesis in one sentence, and explain precisely why sequence-to-structure prediction would be ill-posed without it. Then explain why the hypothesis does *not* imply the prediction is easy.

  2. **Why not simulate?** Levinthal's paradox is often summarised as "folding cannot be a random search". Explain the funnel resolution in your own words, and then explain what the paradox implies about the prospects of predicting structure by simulating every atom's motion.

  3. **The co-evolution logic.** Invent your own residue pair — different chemistry from the salt bridge in Section 4.3 — and tell the same story: why a mutation at one position alone is selected against, why a compensating pair of mutations is not, and what pattern this leaves in an alignment. State explicitly what you would see in the *individual* columns.

  4. **Instrument the demo.** Modify the code so the number of sequences `N_SEQ` sweeps over 50, 100, 200, 400 and 800, and record the rank of each true contact at each size. At what alignment depth does the loose pair stop being recoverable? Relate your answer to the last bullet of Section 4.4.

  5. **Auditing a prediction.** A colleague shows you an AlphaFold-predicted structure and proposes designing an inhibitor for a pocket visible in it. List five questions you would ask before agreeing — including at least one about confidence scores, one about whether the protein is likely to be dynamic, and one about the depth of the available alignment.

## Summary

A protein is made as a linear chain of amino acids and folds spontaneously into a specific three-dimensional shape, and it is the **shape that determines the function**. Measuring that shape — by X-ray crystallography, cryo-EM, or NMR — is slow, expensive, skilled work, so sequences accumulated far faster than structures. **Anfinsen's refolding experiments** (1960s–70s; Nobel Prize 1972) established the **thermodynamic hypothesis** that all the information needed is already in the sequence, making prediction well-posed. **Levinthal's paradox** established that it cannot be found by enumeration: the number of configurations is exponential in chain length, yet folding takes a fraction of a second, so the landscape must be **funnelled** rather than searched.

**CASP**, running **every two years since 1994**, made progress measurable by having predictors submit models **blind** against withheld experimental structures. Progress was incremental for decades. **AlphaFold 1 led CASP13 in 2018** with deep learning applied to pairwise contact prediction. **AlphaFold 2, at CASP14 in 2020**, changed the field, with the problem **widely described as essentially solved for single domains**; the method was published in *Nature* in 2021, and the **AlphaFold Protein Structure Database** launched with **EMBL-EBI** in 2021.

Five principles carry the design. **Evolution as data**: multiple sequence alignments expose **co-evolution**, where two positions in spatial contact vary in a correlated way because compensating mutations preserve the fold. The **pair representation** holds beliefs about every pair of positions, and the **Evoformer** repeatedly exchanges information between it and the MSA using attention, including updates that enforce geometric self-consistency. The **structure module** turns those relationships into actual coordinates while building in the constraints real geometry obeys. **Recycling** feeds a completed pass back as input so the network can revise its own draft. And **pLDDT** reports per-residue confidence, which is what makes the output safe to build on.

Our co-evolution demo made the first principle measurable: from an 800-sequence alignment, all three planted contacts ranked **1, 2 and 3 out of 276 candidate pairs**, with the strongest at **1.3888 bits against a background median of 0.0071** — while the joint count table showed the constraint to be **invisible in either column alone**. The planted shared-ancestry confound occupied ranks 4 through 8, and the **average product correction** cut it by about 39% while costing the true signal only about 11%.

The honest boundary matters as much as the achievement. **Single static domains** with deep alignments moved from unreliable to routinely useful. **Complexes, dynamics, disordered regions, single-mutation effects, and proteins with few known relatives remain hard.** A long-standing problem was reduced to a smaller one, and the smaller one is still real.

The next chapter steps back to ask what the arc from Go to proteins actually taught — what carried across from games to science, what did not, and what a field should learn from a decade in which the same organisation solved two problems that had almost nothing in common.

[← Chapter 3: Zero and Beyond: Learning Without Humans](<chapter-3.html>) [Chapter 5: The Legacy: From Games to Science →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
