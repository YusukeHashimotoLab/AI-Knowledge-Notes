---
title: "Chapter 1: Games as the Proving Ground"
chapter_title: "Chapter 1: Games as the Proving Ground"
subtitle: "Why Artificial Intelligence Grew Up on Chessboards, and Why the Recipe That Beat Kasparov Could Not Touch Go"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/aIDn9Jyv7j8"
    title="AlphaGo to AlphaFold Ch.1: Games as the Proving Ground"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/ML/alphago-to-alphafold/chapter-1.html>) | Last sync: 2026-08-19

[Machine Learning Dojo](<../index.html>) > [AlphaGo to AlphaFold](<index.html>) > Chapter 1

This series tells one story in two halves. The first half is a machine that learned to play a board game better than any human. The second half is a machine that predicted the three-dimensional shape of proteins. They look like unrelated achievements, and the popular telling usually treats them that way. They are not unrelated. The same laboratory built both, within a few years of each other, and the second was possible because of what the first taught its builders about combining **search** with **learned intuition**.

To see the connection you have to start where the field started: on a game board. This chapter explains why artificial intelligence research spent decades there, what the winning recipe for chess actually was, and why that recipe hit a wall in the game of Go that no amount of faster hardware could break through. The wall is the interesting part. It is not really about the size of a number.

If you have not met reinforcement learning before, this series does not require it — but it will occasionally point at ideas that the [Introduction to Reinforcement Learning](<../reinforcement-learning-introduction/index.html>) series develops properly. Read them as companions: that series builds the algorithms, this one tells the story of what was built with them.

## 1.1 Why Artificial Intelligence Went to the Game Board

Playing games looks like a frivolous thing for a serious research field to spend forty years on. It was not. Games have three properties that almost nothing else in the world has at the same time, and each one removes a different obstacle that makes real problems hard.

**Perfect information.** In chess and in Go, both players see the entire position at all times. Nothing is hidden, nothing is noisy, and no measurement is required. Compare this with almost any real task a machine might be asked to do: a robot sees a partial, blurry, and out-of-date picture of its surroundings; a medical model sees a handful of measurements from a body with thousands of relevant variables. Games hand the algorithm the complete state of the world for free, which means a failure can be blamed on the reasoning rather than on the perception.

**An unambiguous outcome.** A game ends in a win, a loss, or a draw. There is no argument about which happened and no expert panel needed to score it. This matters enormously, because most of machine learning depends on having a target to move toward, and in most real domains the target is contested. Was this diagnosis correct? Was this translation good? Was this design manufacturable? Experts disagree, labels are expensive, and the label itself carries the disagreement into the model. A game gives you a clean, cheap, and uncontested reward signal.

**Unlimited self-generated data.** This is the property that ends up mattering most. A chess program can play a chess program. Nobody has to be paid, no experiment has to be run, no sample has to be synthesized. The data supply is bounded only by computation. Contrast that with a materials laboratory, where a single new data point may require a week of synthesis and characterization, or with a medical dataset, where the data cannot be created at all — it must be waited for.

### 📚 The Real-World Contrast, Stated Plainly

Put the three properties side by side with a typical scientific problem and the appeal becomes obvious.

| | Board game | Typical real problem |
|---|---|---|
| State | Fully visible, exact | Partial, noisy, indirect |
| Objective | Win / lose / draw, uncontested | Multi-objective, disputed, sometimes undefined |
| Data | Generated on demand, unlimited | Slow, expensive, and finite |
| Feedback delay | Bounded by game length | Months to years, sometimes never |

A game is therefore not a toy version of a hard problem. It is a **laboratory instrument**: a setting engineered to isolate one variable — the quality of the decision-making itself — from every other difficulty that would otherwise confound the measurement. When a program beats a human champion, the claim being tested is narrow and precise, which is exactly what makes it useful as science.

The honest caveat belongs here too, and the rest of this series is partly an argument about it. Success on a game proves that a method can make good decisions *given* a perfect world model and an uncontested objective. It proves nothing, by itself, about what happens when those gifts are withdrawn. Chapter 4 is where the gifts get withdrawn.

## 1.2 The Chess Recipe: Look Ahead, Then Judge

The first landmark most people remember is **Deep Blue defeating Garry Kasparov in 1997** — the reigning world chess champion, in a match, on the board. It was a genuine milestone and it is worth understanding precisely what kind of machine did it, because the *kind* is the thing that later failed at Go.

The chess recipe has three parts, and only three.

**Part one: minimax.** Build a tree of possible futures. From the current position, list every legal move; from each of those, list every legal reply; and so on. The player to move picks the branch that maximizes their outcome, and the opponent picks the branch that minimizes it — hence *minimax*. If you could build the tree all the way to the end of the game, this procedure would play perfectly, because every leaf would be a known win, loss, or draw.

**Part two: alpha-beta pruning.** You cannot build the whole tree, but you can avoid building most of it. Alpha-beta is the observation that once a branch has been shown to be worse than one you have already fully examined, you can stop examining it — the opponent will never let you go there, so its exact value is irrelevant. With good move ordering, alpha-beta lets a search reach roughly **twice the depth** for the same amount of work. Section 1.6 makes this concrete.

**Part three: a handcrafted evaluation function.** Since the tree still cannot reach the end of the game, the search stops at some depth and asks: *how good is this position?* In chess, a surprisingly good answer can be written down by hand. Count the material with standard piece values. Add terms for king safety, pawn structure, control of the centre, mobility of the pieces, and so on. Weight them, sum them, return a number. Human chess masters helped tune those terms, and the result is a static evaluator: a function from a position to a score, computed without looking ahead at all.

That is the whole recipe. Deep Blue combined it with purpose-built hardware and an enormous amount of engineering, and at the top of chess it was enough.

### 📚 Why Chess Was Vulnerable to This Approach

Three features of chess made the recipe work, and it is worth naming them because Go has none of them.

  * **The branching factor is modest.** A chess position offers **roughly 35 legal moves** on average. Thirty-five is small enough that a deep search stays within reach of fast hardware.
  * **Material is a real signal.** In chess, pieces have durable, largely position-independent worth. A queen is better than a pawn in almost every position on almost every board. This gives the evaluation function a strong, cheap, and reliable backbone to build on.
  * **Tactics dominate.** A great deal of chess strength consists of seeing forced sequences a few moves deep. Search is *precisely* the tool for that, and a machine that never gets tired and never miscalculates has a structural advantage over a human in exactly this component of the game.

Notice what the recipe did **not** contain: any learning. Deep Blue did not improve by playing. Its knowledge was placed there by its authors, and its strength came from applying that fixed knowledge across an enormous number of positions. This distinction — knowledge supplied by humans versus knowledge acquired by the machine — is the axis the entire series runs along.

## 1.3 Why Go Broke the Recipe

After 1997, the obvious next target was **Go**, the East Asian board game played with black and white stones on a 19×19 grid. Go resisted for nearly two decades. The usual explanation is that Go is bigger. That is true, and it is also the less interesting half of the answer.

**The size problem.** A Go position offers **roughly 250 legal moves** on average, against chess's 35. Games run much longer, commonly on the order of a hundred and fifty moves. The number of legal board positions is enormous — commonly quoted as being **on the order of \\(10^{170}\\)**, a figure so far beyond physical enumeration that comparisons to the number of atoms in the observable universe understate it. Section 1.6 computes what these numbers do to a search tree, and the answer is that they do not merely make the problem harder. They make search-to-the-end permanently impossible, for any conceivable machine, by a margin that has no upper bound in sight.

But if size were the only obstacle, faster computers and better pruning would have eventually delivered. They did not, and the reason is the second wall.

**The evaluation problem.** Nobody could write down a static evaluation function for Go.

This is worth sitting with, because it is a strange kind of failure. In chess, "count the material" gets you most of the way to a usable evaluator. In Go, the equivalent — count the stones, or count the territory — is close to useless in the middle game. Stones are not captured often; the board fills up steadily; and a group of stones that looks alive can turn out to be dead a hundred moves later because of a subtle liberty count. Territory is not owned until the very end. A position that a professional player calls "clearly winning" may look, to any straightforward counting procedure, almost exactly like the position they call "clearly losing".

### 📚 The Words Professionals Use, and Why They Resisted Formalization

Ask a strong Go player why a move is good and you will hear a vocabulary that is precise among practitioners and almost impossible to code:

  * **Thickness** (*atsumi*): a wall of stones that is not itself territory but that makes fighting anywhere nearby favourable, for a long time to come.
  * **Influence**: the diffuse tendency of a group to make a whole region of the board more valuable to its owner, without claiming any of it.
  * **Aji** (literally *taste*): latent possibilities left over in a dead-looking group, which may become relevant many moves later.
  * **Shape**: local configurations that are efficient or inefficient in ways experienced players recognize instantly and cannot fully reduce to rules.

Every one of these is a statement about a **long-range, whole-board, and highly non-local** property of a position. Every attempt to hand-code them ran into the same failure: the rules had exceptions, the exceptions had exceptions, and the resulting evaluators were both slow and unreliable. Professional intuition here is not vagueness. It is a compressed judgement, learned over a lifetime of games, that nobody had found a language to write down.

So Go presented two walls at once. **The tree is too wide to search**, and **the leaves cannot be judged**. Removing either one alone does not help — a search you cannot evaluate is useless, and an evaluator you cannot search from is a static guess. That combination is why Go was, for a long time, the standing embarrassment of game-playing artificial intelligence.

## 1.4 Two Ingredients, One Balance

Step back from both games and a general structure appears. **Any agent that plays a game well needs exactly two capabilities.**

**A way to look ahead.** Given a position, consider possible continuations and their consequences. This is *search*. Search is generic — it needs nothing but the rules of the game — and it converts computation directly into skill. Its weakness is that its cost grows exponentially with how far ahead you look.

**A way to judge.** Given a position, say how good it is without playing the rest of the game out. This is *evaluation*, or *knowledge*. Evaluation is cheap per position and can encode arbitrarily deep understanding. Its weakness is that somebody has to supply it, and it can be wrong in ways search would have caught.

The two are complements, and the entire history of game-playing programs can be read as **a shifting balance between them**:

| Era | Search | Evaluation | Result |
|---|---|---|---|
| Classical chess programs | Deep, alpha-beta pruned | Handcrafted by humans | Superhuman at chess |
| Classical Go programs | Shallow (tree too wide) | Handcrafted, and poor | Far below professional play |
| Monte Carlo Go | Random playouts | Statistical, from playouts | Strong amateur play |
| AlphaGo (Chapter 2) | Guided by learned policy | Learned from data and self-play | Superhuman at Go |

Read down the *evaluation* column and the story is clear: the progress came from changing where the judgement comes from. Human-written, then statistical, then learned. The search column changes too, but it changes in service of the other — a better evaluator lets you search less and better.

Keep this frame. It is the frame that survives the jump from Go to protein structure in Chapter 5, where "search" becomes something quite different but the division of labour stays recognisable.

## 1.5 The Monte Carlo Turn

The idea that eventually cracked the evaluation wall arrived from an unexpected direction, and in its first form it looks like a joke.

**Suppose you cannot judge a position. Play it out at random and see what happens.**

Take the position. Have both sides make uniformly random legal moves until the game ends. Record who won. Do this many times. The fraction of random playouts that end in a win for you becomes your estimate of how good the position is.

The immediate objection is the right one: random play is nothing like real play, so how can the outcome of random games say anything about a position between two experts? Three answers, in increasing order of importance.

**It requires no knowledge whatsoever.** The procedure needs the rules of the game and nothing else. No piece values, no shape library, no professional consultation. For a game where the knowledge was the bottleneck, a knowledge-free evaluator is not a compromise — it is the entire point.

**Errors are symmetric and partially cancel.** Both sides play badly in a random playout. A position that is genuinely good tends to remain good under random continuation, because there are simply more ways for the game to end well from a good position than from a bad one. The estimate is biased and noisy, but it is *correlated with the truth*, which is all an evaluator needs to be useful inside a search.

**It converts computation into accuracy, smoothly.** More playouts means a tighter estimate. There is no threshold to cross and no wall to hit; the evaluator improves continuously with the compute you give it. That property — that you can always spend more and get a little more — is what makes the idea worth building on.

On its own, random playout evaluation is crude. The breakthrough came from a second idea: **do not spread the playouts evenly**. Use the early results to decide where to spend the later ones, concentrating effort on the moves that look promising while still occasionally checking the ones that do not. That combination — a tree that grows toward the promising regions, with random playouts at its frontier — is **Monte Carlo Tree Search**, and it lifted computer Go from weak to strong amateur play in a few years.

It did not, by itself, reach professional strength. Chapter 2 explains the algorithm properly, implements it, and then shows what AlphaGo added to it.

## 1.6 Hands-On: Where Brute Force Dies

Let us put numbers on the width of the two trees. The code takes exactly two inputs — the approximate branching factors, roughly 35 for chess and roughly 250 for Go — and derives everything else by arithmetic. Both figures are conventional approximations, not measurements; the conclusions are about orders of magnitude and do not depend on the second digit.

```python
import math

import numpy as np

# ---------------------------------------------------------------
# How big is a game tree?
#
# Two inputs only, both approximate and widely quoted:
#   b ~ 35  average legal moves per position in chess
#   b ~ 250 average legal moves per position in Go (19x19)
# Everything else below is arithmetic on those two numbers.
# ---------------------------------------------------------------
B_CHESS = 35
B_GO = 250

depths = np.arange(1, 11)
nodes_chess = np.power(float(B_CHESS), depths)
nodes_go = np.power(float(B_GO), depths)

print("Step 1: leaf nodes at depth d  (b^d)")
print(f"{'depth':>6} {'chess (b=35)':>16} {'Go (b=250)':>16} {'ratio Go/chess':>18}")
print("-" * 60)
for d, c, g in zip(depths, nodes_chess, nodes_go):
    print(f"{d:6d} {c:16.2e} {g:16.2e} {g / c:18.2e}")
print()

# --- 2. What machine would you need to search depth 10? --------
# Give the machine one second of thinking time.
SECONDS = 1.0
print("Step 2: nodes per second needed to enumerate depth 10 in 1 second")
for name, b in [("chess", B_CHESS), ("Go", B_GO)]:
    need = b**10 / SECONDS
    print(f"  {name:>5}: {need:.2e} nodes/s")
print()

# --- 3. How deep can a fixed machine actually go? --------------
# A deliberately generous machine: one billion positions per second,
# far beyond any single evaluator ever built.
RATE = 1e9  # positions per second

print(f"Step 3: depth reachable in one second at {RATE:.0e} positions/s")
for name, b in [("chess", B_CHESS), ("Go", B_GO)]:
    d_max = math.log(RATE * SECONDS) / math.log(b)
    print(f"  {name:>5}: full-width depth {d_max:.2f} plies")
print()

# --- 4. Alpha-beta pruning: the square-root effect --------------
# With perfect move ordering, alpha-beta visits about b^(d/2)
# nodes instead of b^d -- i.e. an EFFECTIVE branching factor of
# sqrt(b). It is a large win, and it is still exponential.
print("Step 4: alpha-beta at best case visits ~b^(d/2)")
print(f"{'depth':>6} {'chess plain':>14} {'chess a-b':>14} {'Go plain':>14} {'Go a-b':>14}")
print("-" * 66)
for d in [4, 8, 12, 16]:
    print(
        f"{d:6d} {B_CHESS**d:14.2e} {B_CHESS ** (d / 2):14.2e} "
        f"{B_GO**d:14.2e} {B_GO ** (d / 2):14.2e}"
    )
print()

print(f"Step 5: depth reachable in one second at {RATE:.0e} positions/s, WITH alpha-beta")
for name, b in [("chess", B_CHESS), ("Go", B_GO)]:
    d_max = 2 * math.log(RATE * SECONDS) / math.log(b)
    print(f"  {name:>5}: effective depth {d_max:.2f} plies")
print()

# --- 6. The wall, stated as a time ------------------------------
# A typical Go game runs on the order of 150 moves. Enumerating
# even a fraction of that tree is not a hardware problem.
GAME_LENGTH = 150
seconds_per_year = 365.25 * 24 * 3600
# Work in log10 throughout: these integers overflow a float otherwise.
log_tree = GAME_LENGTH * math.log10(B_GO)
log_years = log_tree - math.log10(RATE) - math.log10(seconds_per_year)
log_atoms_universe = 80  # order-of-magnitude figure, for scale only
print("Step 6: brute force to the end of a Go game")
print(f"  crude tree size for a {GAME_LENGTH}-move game: 250^{GAME_LENGTH} ~ 1e{log_tree:.0f}")
print(f"  time at {RATE:.0e} positions/s: ~1e{log_years:.0f} years")
print(f"  for scale, atoms in the observable universe: ~1e{log_atoms_universe}")
```

**Output:**

```
Step 1: leaf nodes at depth d  (b^d)
 depth     chess (b=35)       Go (b=250)     ratio Go/chess
------------------------------------------------------------
     1         3.50e+01         2.50e+02           7.14e+00
     2         1.22e+03         6.25e+04           5.10e+01
     3         4.29e+04         1.56e+07           3.64e+02
     4         1.50e+06         3.91e+09           2.60e+03
     5         5.25e+07         9.77e+11           1.86e+04
     6         1.84e+09         2.44e+14           1.33e+05
     7         6.43e+10         6.10e+16           9.49e+05
     8         2.25e+12         1.53e+19           6.78e+06
     9         7.88e+13         3.81e+21           4.84e+07
    10         2.76e+15         9.54e+23           3.46e+08

Step 2: nodes per second needed to enumerate depth 10 in 1 second
  chess: 2.76e+15 nodes/s
     Go: 9.54e+23 nodes/s

Step 3: depth reachable in one second at 1e+09 positions/s
  chess: full-width depth 5.83 plies
     Go: full-width depth 3.75 plies

Step 4: alpha-beta at best case visits ~b^(d/2)
 depth    chess plain      chess a-b       Go plain         Go a-b
------------------------------------------------------------------
     4       1.50e+06       1.22e+03       3.91e+09       6.25e+04
     8       2.25e+12       1.50e+06       1.53e+19       3.91e+09
    12       3.38e+18       1.84e+09       5.96e+28       2.44e+14
    16       5.07e+24       2.25e+12       2.33e+38       1.53e+19

Step 5: depth reachable in one second at 1e+09 positions/s, WITH alpha-beta
  chess: effective depth 11.66 plies
     Go: effective depth 7.51 plies

Step 6: brute force to the end of a Go game
  crude tree size for a 150-move game: 250^150 ~ 1e360
  time at 1e+09 positions/s: ~1e343 years
  for scale, atoms in the observable universe: ~1e80
```

**Reading the result.** Four observations, in increasing order of importance.

  * **The gap opens immediately and never closes.** At depth 1 Go is only about 7 times wider than chess. By depth 10 it is \\(3.5 \times 10^{8}\\) times wider. This is what exponential means: a modest ratio in the base becomes an unbridgeable ratio in the result. The relevant quantity is \\((250/35)^{d}\\), and it compounds every single ply.

  * **A machine that plays chess well is helpless at Go.** Give the same generous billion-positions-per-second machine one second of thought. In chess it enumerates a full-width tree to about 5.8 plies; in Go, to about 3.75. Under three moves each. A Go position at that depth is barely distinguishable from the starting position — the consequences of a move in Go typically play out over dozens of plies.

  * **Alpha-beta is a big win and it is not enough.** Pruning roughly halves the exponent, which is the same as taking the square root of the tree — an enormous saving. Our machine goes from 5.83 to 11.66 plies in chess, which is genuinely deep chess. The same trick takes Go from 3.75 to 7.51 plies, which is still nothing. **Halving an exponent that is far too large leaves an exponent that is far too large.** No pruning technique, and no plausible hardware, changes that conclusion, because the requirement grows exponentially while hardware grows at best geometrically in time.

  * **The end of the game is not reachable in any sense of the word.** A crude count of a 150-move Go game gives about \\(10^{360}\\) sequences, requiring on the order of \\(10^{343}\\) years to enumerate at a billion per second. For scale, the observable universe contains something like \\(10^{80}\\) atoms. The gap is not a hardware problem, a budget problem, or a patience problem. It is a statement that search alone will never solve Go, and therefore that **something other than search must supply the judgement**.

That last sentence is the reason this chapter exists. The chess recipe had a human-written evaluator to fall back on when the search ran out. Go had no such thing. Every path forward from here involves getting the evaluation from somewhere else.

Try changing `RATE` to \\(10^{15}\\) — a million times more powerful than anything reasonable — and rerun Step 5. The Go depth improves from about 7.5 to about 12.5 plies. Six orders of magnitude of hardware buys five more plies. That single experiment is the whole argument of Section 1.3 in one line.

### 🎯 Exercise Problems

  1. **The three properties.** For each of the three properties in Section 1.1 (perfect information, unambiguous outcome, unlimited self-generated data), name one real scientific or engineering problem that has it and one that clearly lacks it. For the ones that lack it, state what would have to be built to supply a substitute.

  2. **Auditing the chess recipe.** Deep Blue's evaluation function was written by people. List three specific things a hand-written chess evaluator can encode easily, and two things it would find genuinely hard. Then argue whether the hard ones matter enough to have cost it the match.

  3. **The exponent, by hand.** Using only the branching factors, compute the ratio of Go's tree to chess's tree at depth 6 without running the code, then check yourself against the output. Explain in one sentence why this ratio is itself exponential in the depth.

  4. **When does pruning save you?** Alpha-beta turns \\(b^{d}\\) into roughly \\(b^{d/2}\\). Suppose a new technique promised to turn it into \\(b^{d/4}\\). Using the code, find the Go search depth this would reach in one second at \\(10^{9}\\) positions per second. Is it enough to play Go well? Justify your answer by reference to how long the consequences of a Go move take to appear.

  5. **Why random play is not obviously absurd.** Section 1.5 claims a random playout gives an estimate that is biased and noisy but *correlated with the truth*. Construct a position type in any game you know where this claim would fail badly — that is, where a genuinely winning position gives poor random-playout results. What does your example suggest a Monte Carlo player would need in order to be trustworthy?

## Summary

Artificial intelligence spent decades on board games for reasons that were methodological rather than recreational. Games offer **perfect information**, an **unambiguous win/loss signal**, and **unlimited self-generated data** — three gifts that no realistic problem provides at once, and which together isolate the quality of decision-making from every other difficulty. That makes a game a laboratory instrument, and it also means a victory on a game proves something narrower than headlines usually suggest.

The chess recipe that culminated in **Deep Blue's 1997 defeat of Garry Kasparov** had exactly three ingredients: **minimax** search over possible futures, **alpha-beta pruning** to skip branches that cannot matter, and a **handcrafted evaluation function** to score positions where the search stops. It contained no learning at all. It worked because chess offers a modest branching factor of roughly 35, because material value is a strong and durable signal, and because tactical calculation — the thing search does best — is a large fraction of chess strength.

Go broke the recipe twice over. Its branching factor of **roughly 250** and its long games put a full search permanently out of reach: our code showed a generous billion-positions-per-second machine reaching under 4 plies in one second, and only about 7.5 plies even with the square-root saving that alpha-beta provides at best. Six orders of magnitude of extra hardware would buy five more plies. But the **deeper wall was evaluation**: nobody could write down a static scoring function for a Go position. The concepts professionals actually use — thickness, influence, *aji*, shape — are long-range, whole-board judgements that resisted every attempt at formalization.

Read as a general structure, any game player needs two things: **a way to look ahead** (search) and **a way to judge** (evaluation). The history of the field is a shifting balance between them, and progress in Go came from changing *where the judgement comes from* — first from humans, then from statistics, then from learning. The statistical step is the **Monte Carlo idea**: if you cannot judge a position, play it out at random many times and count the wins. It needs no knowledge, its errors partially cancel, and it converts computation into accuracy smoothly.

The next chapter turns that idea into a real algorithm. We build **Monte Carlo Tree Search** from its four steps, derive and explain the **UCB** rule that decides where to spend the next simulation, and implement a complete working search that goes from playing randomly to playing essentially perfect tic-tac-toe as we give it more thinking time. Then we add the piece that made AlphaGo: two neural networks, one that narrows the search and one that judges the positions, and the matches that followed.

[← Series Top](<index.html>) [Chapter 2: AlphaGo: Search Meets Learning →](<chapter-2.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
