---
title: "Chapter 3: Zero and Beyond: Learning Without Humans"
chapter_title: "Chapter 3: Zero and Beyond: Learning Without Humans"
subtitle: "AlphaGo Zero, AlphaZero, MuZero, and the Systematic Removal of Everything a Human Put In"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/cS-YTfdUprY"
    title="AlphaGo to AlphaFold Ch.3: Zero and Beyond: Learning Without Humans"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/ML/alphago-to-alphafold/chapter-3.html>) | Last sync: 2026-08-19

[Machine Learning Dojo](<../index.html>) > [AlphaGo to AlphaFold](<index.html>) > Chapter 3

The version of AlphaGo that played Lee Sedol was, in an important sense, a collaboration. It had been shown a large collection of games between strong human players and had learned to imitate them before it ever learned to compete. Human judgement was in the system's foundations — in what it was first taught to consider plausible, and in the hand-designed features that described a board position to it.

That raises a question that sounds philosophical but turns out to be an engineering question with a testable answer: **was the human knowledge helping, or was it a ceiling?**

This chapter follows the answer through three systems, each of which removed one more thing a human had supplied. **AlphaGo Zero** (published in *Nature* in 2017) removed the human games. **AlphaZero** (announced in 2017) removed the assumption that the recipe was about Go at all. **MuZero** (announced in 2019 and published in *Nature* in 2020) removed the rules themselves. The arc is one of the cleanest demonstrations in machine learning that *less built-in knowledge, plus a mechanism for generating your own experience, can beat more built-in knowledge* — and the final section is equally honest about where that lesson stops applying.

## 3.1 AlphaGo Zero: The Loop That Needs No Teacher

AlphaGo Zero starts from **random play**. It is given the rules of Go and a description of the board position, and nothing else — no game records, no opening book, no hand-designed features encoding what a good shape looks like. Chapter 2 described AlphaGo as a system with several specialised networks trained in stages; AlphaGo Zero replaces that arrangement with **a single neural network with two heads**.

The network takes a board position and outputs two things:

  * a **policy** — a probability distribution over the legal moves, expressing *which moves look worth considering here*
  * a **value** — a single number estimating *how this position is likely to turn out for the player to move*

Both heads sit on top of the same shared body, so everything the network learns about reading a position serves both jobs at once. But this network, on its own, is not the player. The player is the network **plus search**, and the relationship between the two is the whole idea.

### 📚 Search as a Policy Improvement Operator

This is the concept worth slowing down for, because everything else in the chapter is a consequence of it.

Take the network's policy at some position. It is a guess — early in training, a bad one. Now run a Monte Carlo tree search from that position, using the policy to decide which branches are worth exploring and the value head to judge the positions the search reaches. The search spends real computation: it looks ahead, it discovers that some move the policy liked runs into trouble three moves later, it discovers that a move the policy nearly ignored is actually strong.

When the search finishes, look at how it distributed its attention — which moves it visited most. That **visit distribution is a better policy than the one you started with**. Not by assumption: by construction. The search took the raw policy and refined it with actual lookahead, and the refinement is exactly what the extra computation bought.

Search is therefore a **policy improvement operator**: a procedure that turns any policy into a better one, at the cost of computation.

Now close the loop. If search reliably produces a better policy than the network's, then **the search result is a training target**. Train the network to output what the search decided. The network absorbs, into a single fast forward pass, judgement that previously required a whole tree of lookahead. And because the network is now better, the next search — which is guided by the network — is also better, which produces still better targets.

  * The network guides the search.
  * The search improves on the network.
  * The improvement is distilled back into the network.
  * Repeat.

The value head learns in the same loop from a different signal: each self-play game ends in a win or a loss, and every position from that game is trained toward its eventual outcome. No human ever labels a position as good or bad. The game's own result does it.

### 📚 Why the Curriculum Takes Care of Itself

A system that learns only from self-play faces an obvious hazard: at the beginning it is terrible, so it can only generate terrible data. Where does the bootstrap come from?

The answer is that **self-play automatically supplies an opponent of exactly the right strength — itself**. A random player against a random player produces games that are noise, but they are *decidable* noise: someone wins. That is enough signal to learn something slightly better than random. Slightly-better-than-random then plays slightly-better-than-random, and the games become marginally more meaningful, and so on.

This is a **curriculum that generates itself**. It never presents problems too hard to learn from, because the difficulty is pinned to the learner's own current ability. A fixed human opponent could not do this — too strong and every game is an incomprehensible loss, too weak and there is nothing left to learn. Self-play tracks the learner automatically, forever.

## 3.2 Why Removing Human Knowledge Helped

AlphaGo Zero, trained this way, went on to defeat the earlier version of AlphaGo that had beaten Lee Sedol. A system given strictly *less* information beat a system given more. Three reasons are worth separating.

**Human priors are a ceiling as well as a floor.** Initialising a policy by imitating human games gives you a fast start: you begin somewhere sensible instead of at random. But you also begin *inside the distribution of moves humans play*, and moves that no strong human plays are, by construction, moves your initial policy considers unlikely. A search guided by that policy will rarely look at them. Human knowledge accelerates the early climb and then quietly narrows the space you climb in.

**Hand-designed features encode assumptions about what matters.** Every hand-built feature is a decision that some property of the board is worth computing and, implicitly, that everything not computed is not worth attention. A network reading the raw position is free to discover its own features — including ones nobody thought to name.

**The objective becomes purely the objective.** A system trained partly to imitate humans is optimising two things at once: winning, and looking like a human player. Those overlap heavily, but not perfectly, and where they diverge, imitation is a distraction. Remove it and the only remaining pressure is the game's own result.

### 📚 The Student Surpassed the Textbook

The most striking evidence for the ceiling argument came from what AlphaGo Zero learned about **joseki** — the standard corner sequences that human Go players have refined over centuries and study as settled theory.

Training from scratch, the system rediscovered many of these sequences on its own. That is already remarkable: patterns that took a human tradition generations to isolate emerged from self-play with no exposure to that tradition.

Then it did something more interesting. As training continued, **it discarded some of them**, preferring alternatives that human theory had not settled on.

Both halves of that story matter. The rediscovery says human Go theory was genuinely finding real structure in the game — the joseki were not arbitrary conventions. The subsequent departure says the theory was *incomplete*, and that a system which had been taught it as ground truth would have had a much harder time noticing. A student initialised on the textbook tends to stay near the textbook. A student that derives everything itself can notice that a page is wrong.

## 3.3 AlphaZero: One Recipe, Three Games

If the recipe really contains no Go-specific knowledge, it should work on other games. **AlphaZero**, announced in 2017, tested exactly that: the same self-play-plus-search algorithm applied to **chess** and **shogi** as well as Go, learning each from self-play alone.

This mattered because chess in particular had a long and highly optimised alternative tradition. Classical chess engines are built the other way round: on a hand-crafted evaluation function encoding decades of accumulated chess understanding, driven by a search that examines an enormous number of positions and prunes them with carefully tuned heuristics. It is a formidable design, and it had been refined for a very long time.

The contrast in search style is worth stating qualitatively, because it is the clearest expression of the difference between the two philosophies.

  * A **classical engine searches broadly**. Its evaluation function is cheap, so it can afford to look at a great many positions and rely on volume plus pruning heuristics to find the right move.
  * **AlphaZero searches narrowly and deeply**. Its evaluation is a neural network — far more expensive per position, so it can only afford to examine a small fraction as many. It compensates by using its policy head to decide *which* lines deserve attention, and following those far fewer lines further.

One system compensates for a shallow evaluation with breadth; the other compensates for limited breadth with a deep evaluation. AlphaZero showed that the second trade could be made to work against the first — and, more importantly for this series, that **the same code could be pointed at three different games and learn all of them**. The Go result might have been a Go result. Three games make it a method.

## 3.4 MuZero: Giving Up the Rules

AlphaZero still received one substantial gift: **the rules**. To run a tree search you must be able to ask "if I play this move, what position results?" — and AlphaZero was handed a perfect simulator that answered instantly and exactly.

That is a real limitation, because most interesting problems do not come with one. **MuZero**, announced in 2019 and published in *Nature* in 2020, removed it.

MuZero learns its own **model of the environment** from experience. Three learned components work together:

  * a **representation** function, which maps the observations it has seen so far into an internal state
  * a **dynamics** function, which takes an internal state and an action and predicts *the next internal state and the immediate reward*
  * a **prediction** function, which reads an internal state and outputs a policy and a value

With these, MuZero can run a tree search entirely inside its own head: start from the current internal state, apply the dynamics function to imagine the consequence of an action, apply it again, and evaluate the imagined states with the prediction function. It plans without ever being told the rules.

### 📚 A Model of What Matters

The crucial design decision is what the learned model is *not* asked to do.

A conventional model-based agent learns a model that **reconstructs the environment**: given a state and an action, predict what you will actually observe next — the next screen, the next configuration. This is intuitive and it is a trap. Most of what an observation contains is irrelevant to good decisions. A model trained to reproduce observations spends its capacity on visual detail that has nothing to do with whether you are winning, and its errors compound over the multiple steps a search requires.

MuZero's internal states are **not trained to reconstruct anything**. There is no requirement that you can decode an internal state back into an observation. The only training pressure on the dynamics function is that the quantities the search actually consumes — **reward, value, and policy** — come out right along the imagined trajectory.

This is the shift from *a model of everything* to **a model of what matters**. The internal state need not be interpretable, or complete, or faithful in any sense a human would recognise. It only needs to be **sufficient for planning**. Capacity that a reconstructive model would spend on appearances is spent instead on consequences.

The payoff is generality. Because it never needed rules, MuZero also learned to play **Atari** games, where the agent is given pixels and a score and no description whatsoever of the game's mechanics — a setting where AlphaZero's approach simply cannot be applied. The same system covered both the precise, rule-defined world of board games and the messy, rules-unknown world of video games.

## 3.5 The Arc, and Where It Stops

Read the three systems in order and the pattern is a subtraction:

| System | What it was given | What was removed |
|---|---|---|
| AlphaGo | Human games, hand-designed features, the rules | — |
| AlphaGo Zero | Hand-designed features minimised, the rules | Human game records |
| AlphaZero | The rules | Any Go-specific tailoring |
| MuZero | Observations and rewards | The rules themselves |

Each step removed something a human had supplied, and each step produced a system that was more general than the one before. The general lesson people took from this — often stated as a preference for methods that scale with computation over methods that encode human insight — is genuinely supported here.

### 📚 The Honest Caveat

It would be easy to end the chapter there, and it would be misleading. The conditions that make this arc work are demanding, and most real problems fail them.

**Self-play requires a cheap, fast, perfect simulator — or a way to learn one.** Go can be played millions of times at negligible cost, with no consequence to being wrong. A drug candidate cannot be synthesised millions of times. A manufacturing line cannot be crashed to find out what happens. Where experience is expensive, the engine that drives this entire arc stalls.

**The reward must be clean and unambiguous.** Board games hand you a perfect, undisputed, immediate-at-the-end signal: you won, or you did not. Most real objectives are nothing like this. They are delayed, partially observed, contested between stakeholders, or only definable as a proxy for what you actually want — and a proxy is exactly the thing a powerful optimiser will exploit.

**The problem must be self-contained.** Go is a closed world whose complete state is visible and whose dynamics never change. Real environments are open, partially observed, and non-stationary.

**MuZero relaxes the first condition without removing it.** Learning its own model means it does not need a *given* simulator — but it still needs enough interaction with the environment to *learn* one, which is not the same as needing no experience.

So the correct reading is not "self-play solves things". It is: **where you can generate your own experience cheaply and score it unambiguously, built-in human knowledge is a ceiling worth removing.** Chapter 4 turns to a problem that satisfies none of these conditions — no simulator, no self-play, no reward function — and had to be attacked in an entirely different way. The bridge between the two halves of this series is not a technique. It is the discovery that **evolution had already generated the data**.

## 3.6 Hands-On: Policy Improvement You Can Watch

The claim that search improves a policy, and that the improvement can be trained back into the policy, is easy to state and easy to nod along to. Here it is as something you can run.

The setting is tic-tac-toe — small enough that the whole thing takes a few seconds, and small enough that we can inspect what was learned. The code implements the AlphaGo Zero loop honestly, with two deliberate simplifications: the "neural network" is a lookup table, and the search expands only the root node. Everything else is the real structure. Generation 0 is uniform random play. Each generation plays self-play games in which **every move is chosen by search**, records the search's visit distribution at every position, and then trains the next policy toward those distributions.

The measurement is the important part. Each generation is scored **without search**, against the fixed generation-0 random policy. Any improvement therefore has to be improvement the search successfully taught the *policy* to keep — not improvement the search is doing live at evaluation time.

```python
import numpy as np

# ---------------------------------------------------------------
# A miniature AlphaGo Zero loop, on tic-tac-toe. Not a strong player --
# a demonstration that SEARCH IS A POLICY IMPROVEMENT OPERATOR: run a
# search on top of the current policy, then train the next policy to
# imitate what the search decided. Input: the rules, and nothing else.
# ---------------------------------------------------------------
rng = np.random.default_rng(20171018)

N_GENERATIONS = 6      # times round the loop
GAMES_PER_GEN = 60     # self-play games per generation
SIMS_PER_MOVE = 80     # search budget at each move
TEMPERATURE = 0.5      # <1 sharpens visit counts into the training target
EXPLORE = 0.25         # chance of a random self-play move (keeps coverage)
ALPHA = 0.7            # how far each training step moves toward the target
EVAL_GAMES = 600       # games used to score a generation against gen 0

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
         (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def result_of(board):
    """(is_over, score) with score from player 1's point of view."""
    for a, b, c in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return True, (1 if board[a] == 1 else -1)
    return (True, 0) if not (board == 0).any() else (False, 0)


def policy_of(table, board):
    """Policy at a state. Unseen states fall back to uniform-random,
    so an EMPTY table is exactly the generation-0 random player."""
    p = table.get(board.tobytes())
    if p is None:
        p = np.zeros(9)
        p[np.flatnonzero(board == 0)] = 1.0
    return p / p.sum()


def rollout(table, board, player, rng):
    """Play to the end using the CURRENT policy; score is player-1-relative."""
    b, turn = board.copy(), player
    while True:
        over, score = result_of(b)
        if over:
            return score
        b[rng.choice(9, p=policy_of(table, b))] = turn
        turn = 3 - turn


def search(table, board, player, rng, c=1.4):
    """Minimal MCTS: expand the root only. Each simulation picks a root
    move by UCB and evaluates it with a rollout played by the current
    policy. The output that matters is the VISIT COUNT distribution."""
    moves = np.flatnonzero(board == 0)
    counts, totals = np.zeros(len(moves)), np.zeros(len(moves))
    sign = 1 if player == 1 else -1

    for sim in range(SIMS_PER_MOVE):
        unvisited = np.flatnonzero(counts == 0)
        if len(unvisited):
            i = int(unvisited[0])
        else:
            i = int(np.argmax(totals / counts
                              + c * np.sqrt(np.log(sim) / counts)))
        child = board.copy()
        child[moves[i]] = player
        over, score = result_of(child)
        counts[i] += 1
        totals[i] += sign * (score if over
                             else rollout(table, child, 3 - player, rng))

    sharp = counts ** (1.0 / TEMPERATURE)
    pi = np.zeros(9)
    pi[moves] = sharp / sharp.sum()
    return pi


def self_play_game(table, rng):
    """One game against itself; every search output is kept as a target."""
    board, turn, records = np.zeros(9, dtype=np.int8), 1, []
    while True:
        if result_of(board)[0]:
            return records
        pi = search(table, board, turn, rng)
        records.append((board.tobytes(), pi))
        # occasional random moves keep the self-play curriculum broad --
        # the job Dirichlet noise does at the root in the real system
        board[int(rng.choice(np.flatnonzero(board == 0)) if rng.random() < EXPLORE
                  else rng.choice(9, p=pi))] = turn
        turn = 3 - turn


def train(table, batch):
    """Move each state's policy toward what the search decided there.
    A real system fits a neural network to these targets and thereby
    generalises to states it never saw. Here the table IS the model, so
    fitting reduces to a weighted average -- but the loop is the same."""
    new = dict(table)
    for key, pis in batch.items():
        target = np.mean(pis, axis=0)
        new[key] = (1 - ALPHA) * new[key] + ALPHA * target if key in new else target
    return new


def play_match(table_a, table_b, seed):
    """RAW policies, no search. A plays first in half the games."""
    rng = np.random.default_rng(seed)
    wins = draws = 0
    for g in range(EVAL_GAMES):
        board, a_first, turn = np.zeros(9, dtype=np.int8), g % 2 == 0, 1
        while True:
            over, res = result_of(board)
            if over:
                a_side = 1 if a_first else 2
                draws += res == 0
                wins += res != 0 and (res == 1) == (a_side == 1)
                break
            table = table_a if ((turn == 1) == a_first) else table_b
            board[rng.choice(9, p=policy_of(table, board))] = turn
            turn = 3 - turn
    return wins / EVAL_GAMES, draws / EVAL_GAMES


# --- The loop ---------------------------------------------------
gen0 = {}          # an empty table IS the uniform-random player
table = gen0

print("Gen 0 = uniform random play (the table is empty). Each generation:")
print("self-play with search, then fit the next policy to the search's")
print("visit counts. Scoring opponent is ALWAYS gen 0, and the scored")
print("policy plays WITHOUT search -- so any gain is improvement kept.\n")
print(f"{'gen':>4} {'states known':>13} {'win %':>8} {'draw %':>8} {'score':>8}")
print("-" * 45)

for gen in range(N_GENERATIONS + 1):
    if gen > 0:
        batch = {}
        for _ in range(GAMES_PER_GEN):
            for key, pi in self_play_game(table, rng):
                batch.setdefault(key, []).append(pi)
        table = train(table, batch)
    win, draw = play_match(table, gen0, seed=1000 + gen)
    print(f"{gen:>4} {len(table):>13} {100*win:>7.1f}% {100*draw:>7.1f}% "
          f"{win + 0.5*draw:>8.3f}")

# --- What did it actually learn? --------------------------------
opening = policy_of(table, np.zeros(9, dtype=np.int8))
print("\nOpening preference of the final policy (P of each first move):")
for row in opening.reshape(3, 3):
    print("   " + "   ".join(f"{x:.3f}" for x in row))
print(f"   centre {opening[4]:.3f}   corner avg {opening[[0, 2, 6, 8]].mean():.3f}"
      f"   edge avg {opening[[1, 3, 5, 7]].mean():.3f}"
      f"   (uniform = {1/9:.3f})")

tactic = np.array([1, 1, 0, 2, 2, 0, 0, 0, 0], dtype=np.int8)  # cell 2 wins now
print("\nTactical test    X X .     (X to move; cell 2 wins on the spot)")
print("                 O O .")
print("                 . . .")
print(f"   P(winning move) = {policy_of(table, tactic)[2]:.3f}"
      f"   vs uniform-random {1/6:.3f}")
```

**Output:**

```
Gen 0 = uniform random play (the table is empty). Each generation:
self-play with search, then fit the next policy to the search's
visit counts. Scoring opponent is ALWAYS gen 0, and the scored
policy plays WITHOUT search -- so any gain is improvement kept.

 gen  states known    win %   draw %    score
---------------------------------------------
   0             0    43.5%    12.2%    0.496
   1           283    50.2%    14.0%    0.572
   2           482    54.8%    12.0%    0.608
   3           662    55.8%    13.8%    0.628
   4           843    58.8%    10.5%    0.641
   5           994    58.8%    11.8%    0.648
   6          1138    60.0%    11.7%    0.658

Opening preference of the final policy (P of each first move):
   0.168   0.058   0.141
   0.042   0.187   0.083
   0.131   0.073   0.117
   centre 0.187   corner avg 0.139   edge avg 0.064   (uniform = 0.111)

Tactical test    X X .     (X to move; cell 2 wins on the spot)
                 O O .
                 . . .
   P(winning move) = 0.959   vs uniform-random 0.167
```

**Reading the result.** Four things, in increasing order of interest.

  * **The control behaves.** Generation 0 is the random policy playing the random policy, and it scores 0.496 — a coin flip, as it must, since the two sides are identical and each plays first in half the games. That line is there to confirm the measurement is not flattering us.

  * **Improvement is monotone, and it is improvement the policy kept.** The score climbs 0.496 → 0.572 → 0.608 → 0.628 → 0.641 → 0.648 → 0.658 across six generations. Remember that the evaluated policy runs **without any search at all**. The search happened during training and then went away; what remains is a lookup table that has absorbed some of what the search knew. That is the policy improvement operator doing its job, measured directly.

  * **It rediscovered opening theory.** Nobody told this code anything about tic-tac-toe strategy — only how to detect three in a row. The final opening preference puts the **centre highest (0.187)**, the **four corners next (0.139 average)**, and the **four edges last (0.064 average)**, against a uniform baseline of 0.111. Centre, then corners, then edges is exactly the standard human ordering for this game. It is a miniature of the joseki story from Section 3.2: derive the game from scratch and you converge on the same structure the textbook found, because the structure was in the game rather than in the tradition.

  * **It learned to take a win.** In the tactical position, where one specific move wins immediately, the final policy plays that move with probability **0.959**, against a random baseline of 0.167. The value signal — games end in wins and losses, and positions are trained toward their outcomes — propagated backwards into the policy with no human ever annotating the position as "winning move here".

**What this demo does not show, and should not be read as showing.** Three limits are worth naming, because each corresponds to something the real systems had to solve.

The score plateaus, and it plateaus well below 1.0. Part of that is the opponent — a random player occasionally stumbles into a draw or a win no matter how good you are — and part is a real limitation: the lookup table has **no generalisation**. It knows only the 1138 positions it happened to visit; a neural network's ability to transfer what it learned in one position to a similar position it has never seen is precisely the ingredient this demo throws away. That growing "states known" column is not a feature, it is the cost of not having a network.

The search here expands only the root node. A real MCTS builds a tree, and the deep, selective lookahead described in Section 3.3 is exactly what root-only search cannot do.

And tic-tac-toe is trivially small. The reason the arc in this chapter was a research achievement rather than an exercise is that the same loop had to be made to work on a game with an astronomically larger space, where nothing can be enumerated and everything depends on the network generalising well.

Try changing `EXPLORE` to `0.0` and re-running. The self-play games become narrower, coverage grows more slowly, and the policy improves less against an opponent that wanders into positions the training never explored — the small, practical reason the real system injects noise at the root of every search.

### 🎯 Exercise Problems

  1. **The loop, in your own words.** Explain in three sentences why the visit distribution produced by a search is a better policy than the policy that guided that search. Then explain what would go wrong if it were not — that is, what breaks in the training loop if search fails to improve on the network.

  2. **The ceiling argument.** AlphaGo Zero was given less information than AlphaGo and ended up stronger. Give the strongest version of the counter-argument — why one would *expect* initialising from human games to help — and then explain what specifically about self-play defeats it.

  3. **Reconstruction versus planning.** MuZero's learned dynamics are not trained to predict observations. Describe a concrete situation in which a model that perfectly predicts the next observation is nonetheless useless for choosing an action, and one in which a model that cannot reconstruct anything is still sufficient.

  4. **Instrument the demo.** Modify the code to record, for each generation, the average agreement between the raw policy and the search's visit distribution at the positions visited that generation. Does agreement increase over generations? Explain what increasing agreement means about how much the search still has left to teach.

  5. **Auditing a transfer claim.** A team proposes applying AlphaZero-style self-play to optimising a chemical synthesis route. Using the conditions in Section 3.5, list the four questions you would ask before believing it can work, and state which of the four you expect to be the hardest to answer honestly.

## Summary

AlphaGo Zero replaced AlphaGo's staged, human-initialised design with **a single network with a policy head and a value head**, trained from **random play** with no human game records and no hand-designed features beyond the board itself. The mechanism that makes this possible is that **search is a policy improvement operator**: a tree search guided by the current policy produces a visit distribution better than that policy, so the search's own output becomes the training target, the improved network guides a better search, and the loop compounds. Self-play supplies a curriculum that tracks the learner's ability automatically, and the game's result supplies the value signal without any human labelling. Trained this way, AlphaGo Zero surpassed the version that had defeated Lee Sedol — and, tellingly, **rediscovered many human joseki and then discarded some of them**: the tradition had found real structure, but it was incomplete, and a system initialised on it would have found that far harder to notice.

**AlphaZero** showed the recipe was not about Go, learning **chess and shogi** by the same method, and trading classical engines' broad, cheap, heuristic search for a narrow, expensive, deeply selective one. **MuZero** removed the last human input — the rules — by learning its own **dynamics, reward, value and policy** in an internal state that is never asked to reconstruct observations, only to be **sufficient for planning**. That "model of what matters" is what let the same system extend to Atari, where the mechanics are never disclosed to the agent.

Our tic-tac-toe implementation made the loop concrete: scored without search against a fixed random opponent, the policy climbed monotonically from 0.496 to 0.658 over six generations, converged on the textbook opening ordering of **centre (0.187) > corners (0.139) > edges (0.064)**, and learned to take an immediate win with probability **0.959** against a 0.167 baseline — all from the rules alone. The demo's plateau also illustrated what it deliberately omits: without a network there is no generalisation, only the positions actually visited.

The honest boundary is the part to carry forward. This arc works where **experience is cheap to generate and the reward is clean** — conditions board games satisfy perfectly and most real problems satisfy not at all. The next chapter turns to a problem with no simulator, no self-play, and no reward function: predicting a protein's three-dimensional structure from its amino acid sequence. The data that made it tractable was not generated by an agent. It had been accumulating in the sequences themselves for several billion years.

[← Chapter 2: AlphaGo: Search Meets Learning](<chapter-2.html>) [Chapter 4: AlphaFold: The Protein Folding Breakthrough →](<chapter-4.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
