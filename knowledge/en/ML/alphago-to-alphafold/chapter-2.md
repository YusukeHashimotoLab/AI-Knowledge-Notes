---
title: "Chapter 2: AlphaGo: Search Meets Learning"
chapter_title: "Chapter 2: AlphaGo: Search Meets Learning"
subtitle: "Monte Carlo Tree Search, Two Neural Networks, and the Move That Made a World Champion Leave the Room"
---

## Video Lecture

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/f8yUJ99dm6A"
    title="AlphaGo to AlphaFold Ch.2: AlphaGo: Search Meets Learning"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> This video covers the same content as the text below. Choose your preferred learning format.

---

🌐 EN | [🇯🇵 JP](<../../../jp/ML/alphago-to-alphafold/chapter-2.html>) | Last sync: 2026-08-19

[Machine Learning Dojo](<../index.html>) > [AlphaGo to AlphaFold](<index.html>) > Chapter 2

Chapter 1 left Go looking impossible for two independent reasons. The tree is too wide to search, and the leaves cannot be judged. Either wall alone would be serious; together they had held the game against forty years of effort.

This chapter is about how both walls came down at once, and the shape of the solution is worth stating before we start, because it is the template the rest of this series follows:

> **Search where a learned intuition tells you to look, and stop searching when a learned judgement tells you what you would have found.**

We build the machinery in order. First **Monte Carlo Tree Search (MCTS)** from nothing — its four steps, the formula that steers it, and why it needs no evaluation function to get started. Then the two neural networks AlphaGo added and precisely where they plug in. Then the matches, including the two most-discussed moves in the history of computer game-playing. And finally a complete working MCTS you can run, which goes from playing randomly to playing essentially perfectly as we give it more thinking time.

Readers who want the reinforcement learning machinery developed properly — value functions, policy gradients, the exploration-exploitation trade-off in its general form — should read the [Introduction to Reinforcement Learning](<../reinforcement-learning-introduction/index.html>) series alongside this one. Here we use those ideas; there they are built.

## 2.1 Monte Carlo Tree Search, From Zero

Classical minimax grows the tree **uniformly**: every branch is explored to the same depth, whether or not it is worth it. That is what makes it die at a branching factor of 250. MCTS abandons uniformity completely. It grows a tree that is **deep where the game is interesting and shallow everywhere else**, and it decides which is which from its own accumulated statistics.

The algorithm keeps one tree in memory. Each node stores a position and two numbers: a **visit count** \\(N\\), and an accumulated **total value** \\(W\\). One iteration of the algorithm — one *simulation* — walks from the root to a leaf, plays a game out, and updates the numbers on the way back. That is all it does. It repeats until the clock runs out.

Each simulation has four steps.

**Step 1 — Selection.** Start at the root. As long as the current node has had all of its children created already, choose one of them and descend. The choice is not random and not greedy; it uses the UCB rule of Section 2.2, which balances following the best-looking move against checking a move that has not been tried much. Keep descending until you reach a node that still has unexplored moves, or a finished game.

**Step 2 — Expansion.** At that node, take one move that has never been tried, create the child position it leads to, and add it to the tree. The tree therefore grows by exactly **one node per simulation** — which is what keeps memory under control and makes the growth adaptive rather than uniform.

**Step 3 — Simulation (rollout).** From the new node, play the game to its end using a cheap policy. In the plainest version that policy is *uniformly random*. Record the result: win, loss, or draw. This is the Monte Carlo evaluator of Chapter 1, and it is where a classical program would have called a handcrafted evaluation function instead.

**Step 4 — Backup.** Walk back up the path you just descended, from the new node to the root. At every node on that path, increment \\(N\\) by one and add the result to \\(W\\). One detail decides whether the algorithm works at all: the result must be recorded **from the point of view of the player who chose to move into that node**. A win for Black is a win at the nodes Black chose and a loss at the nodes White chose. Get this backwards and the search will confidently find the worst move available.

After the last simulation, the program plays a move from the root. Notably, the move chosen is usually the **most-visited** child, not the highest-scoring one. A high average from three visits is noise; a high visit count means the selection rule kept returning there under scrutiny. Visit count is the more robust statistic, and the code in Section 2.7 uses it.

### 📚 The Anytime Property

MCTS has a feature that alpha-beta search does not, and it matters more in practice than any single technical detail.

**You can stop it whenever you like and it will give you its best answer so far.** After ten simulations it plays badly; after ten thousand it plays well; after ten million, better still. There is no partial state that is invalid, no depth that must be completed before the numbers mean anything, no iteration that must be finished before the tree is consistent. Each simulation is a complete, self-contained improvement to the estimates.

This is called the **anytime property**, and it has three consequences worth naming.

  * **Time control is trivial.** Give the search a deadline. Whatever it has when the deadline arrives is usable. A depth-limited alpha-beta search, by contrast, has to commit to a depth in advance and may be caught mid-iteration with an unusable partial result.
  * **Scaling is smooth.** More computation buys more strength continuously, with no thresholds and no cliffs. This is the same property that made random playouts attractive in Chapter 1, now inherited by the whole search.
  * **The search is asymmetric by construction.** Simulations pile up in the promising subtrees, so the effective depth in the lines that matter is far greater than the average depth of the tree. The 250-wide branching factor stops being fatal, because MCTS never had any intention of expanding all 250.

## 2.2 UCB: How the Search Decides Where to Look

Selection is the heart of the algorithm, and it faces a dilemma familiar from the multi-armed bandit problem: **exploit** the move that currently looks best, or **explore** a move whose estimate is still uncertain?

Pure exploitation is a trap. Suppose the strongest move happened to lose its first two random playouts through bad luck. A greedy searcher abandons it forever, and the mistake is unrecoverable because it never gathers the evidence that would correct it. Pure exploration is equally useless: spreading simulations evenly over 250 moves is the uniform search that we established cannot work.

The **Upper Confidence Bound** rule resolves this. At a node \\(s\\), among its children \\(a\\), select the child maximizing

\\[ \mathrm{UCB}(s,a) \;=\; \underbrace{\frac{W(s,a)}{N(s,a)}}_{\text{exploitation}} \;+\; \underbrace{c\,\sqrt{\frac{\ln N(s)}{N(s,a)}}}_{\text{exploration}} \\]

where \\(N(s,a)\\) is the number of simulations that went through child \\(a\\), \\(W(s,a)\\) their total value, \\(N(s)\\) the visits to the parent, and \\(c\\) a constant controlling the balance.

Read the two terms separately.

**The first term is the empirical win rate** of that move, \\(W/N\\) — everything the search has learned about the move so far, and nothing more.

**The second term is a measure of ignorance.** It grows when \\(N(s,a)\\) is small (this move is under-explored relative to its siblings) and it grows, slowly, as the parent accumulates visits (the whole node is being studied harder, so the untested siblings deserve another look). The name comes from its origin: the sum of the two terms is an optimistic upper bound on the move's true value, and selecting by it implements the principle **"optimism in the face of uncertainty"**.

### 📚 Why the Square Root and the Logarithm

The shape of the exploration term is not arbitrary, and the two pieces do different jobs.

The \\(1/\sqrt{N(s,a)}\\) factor is the shape of a **statistical error bar**. The uncertainty in an average of \\(N\\) samples shrinks like \\(1/\sqrt{N}\\), so the bonus is, up to a constant, the width of the confidence interval around the win rate. A move sampled four times carries twice the bonus of one sampled sixteen times.

The \\(\sqrt{\ln N(s)}\\) factor makes the bonus grow with the parent's visit count, but **only logarithmically** — extremely slowly. This is what makes the guarantee work: the bonus never stops growing, so no move is abandoned permanently and every child will eventually be revisited, but it grows so slowly that the search still spends the overwhelming majority of its effort on the good moves. The consequence is that the fraction of simulations wasted on a bad move grows only like \\(\ln N\\) while total simulations grow like \\(N\\).

The constant \\(c\\) sets the balance directly: small \\(c\\) makes a search that commits early and may miss things; large \\(c\\) makes one that keeps second-guessing itself. A common default is \\(c = \sqrt{2}\\), which is what the code in Section 2.7 uses, and it is worth experimenting with.

**The crucial property**, and the reason this whole scheme beats a handcrafted evaluator, is what happens as the simulations accumulate. Early on, the estimates come from random playouts and are poor. But every simulation that descends deeply enough replaces a random guess with an actual game outcome propagated up the tree, and the selection rule keeps pushing simulations into the lines that matter. **The evaluator improves itself.** No human wrote any Go knowledge into it; the knowledge is manufactured on the spot, from the rules and from computation.

## 2.3 Where MCTS Was Still Weak

Monte Carlo Tree Search lifted computer Go from feeble to strong amateur level. It did not reach professional strength on its own, and its two limitations point exactly at what AlphaGo added.

**The rollouts are too dumb.** Uniformly random play produces games that no Go player would recognise. In positions whose value depends on a precise sequence — a life-and-death fight, a delicate capturing race — random continuation destroys the very structure that determines the answer, and the statistics converge confidently to the wrong number. Hand-written "fast policies" that made rollouts slightly less random helped, but they reintroduced exactly the hand-crafted knowledge problem that Monte Carlo was supposed to eliminate.

**The branching factor is still 250 at every node.** MCTS explores unevenly, which is a large improvement, but before it can decide that a move is bad it must expand it and sample it at least once. With 250 legal moves at every node, a large fraction of every simulation budget is spent discovering that obviously terrible moves are terrible.

State those two weaknesses as needs and the design of AlphaGo writes itself:

  * We need something that says, before any search, **which moves are even worth considering**.
  * We need something that says, at a leaf, **how good this position is**, without playing 200 random moves.

Those are two different learned functions, and AlphaGo trained one for each.

## 2.4 Two Networks: Narrow the Search, Judge the Leaves

AlphaGo, described in a paper **published in Nature in 2016**, added two deep neural networks to MCTS. Both take a representation of the board as input. They answer different questions.

### The Policy Network: What Should I Consider?

The **policy network** takes a position and outputs a **probability distribution over legal moves** — its estimate of how likely each move is to be the move played. It is, in effect, a model of *plausibility*, and it was built in two stages.

**Stage one: learn from humans.** The network was first trained by supervised learning on a large collection of positions from games played by strong human players, with the target being the move the human actually chose. This is ordinary classification: input a board, predict a move. What it produces is a network with something recognisable as **Go intuition** — the ability to look at a position and immediately suggest the handful of moves a strong player would consider. Notice what has just happened. The knowledge that resisted hand-coding for decades — thickness, shape, influence — was never written down at all. It was *absorbed from examples*, implicitly, as whatever internal features let the network predict expert moves.

**Stage two: improve by self-play.** A network trained to imitate humans is capped at imitating humans, and imitation is not the objective — winning is. So a copy of the policy network was then improved through **reinforcement learning**: the network played games against previous versions of itself, and the moves in games it won were made more likely while the moves in games it lost were made less likely. This is the **policy gradient** idea, developed properly in the [reinforcement learning series](<../reinforcement-learning-introduction/index.html>). The crucial shift is in the training signal: from "what would a human have played here?" to "what actually wins?"

**How it plugs into MCTS**: the policy network's probabilities bias the **selection** step, adding a prior that steers simulations toward moves it considers plausible. A move the network gives a negligible probability is not formally forbidden, but it will be visited rarely. The effective branching factor collapses from 250 to something a search can actually work with — *without* any move being permanently excluded, because the UCB exploration term is still there underneath.

### The Value Network: How Good Is This Position?

The **value network** takes a position and outputs a single number: an estimate of the **probability that the current player will win** from it. This is the static evaluation function that nobody could write by hand — obtained, again, not by writing it but by learning it, from the outcomes of a very large number of self-play games.

**How it plugs into MCTS**: it replaces or augments the **rollout** step. Instead of playing 200 random moves and observing who won, the search asks the value network directly. AlphaGo did not discard rollouts entirely; it combined the two estimates, and there is a good reason for keeping both. They fail in different ways. The value network can be confidently wrong about a position unlike anything in its training; the rollout is unbiased in a certain sense but extremely noisy and blind to precise sequences. Two flawed estimators with **uncorrelated** failure modes beat either one alone.

### 📚 The Division of Labour, In One Table

| | Policy network | Value network |
|---|---|---|
| Question answered | Which moves are worth considering? | Who is winning here? |
| Output | Distribution over legal moves | One number: win probability |
| Trained on | Human expert moves, then self-play | Outcomes of self-play games |
| Role inside MCTS | Biases **selection** — narrows the tree | Replaces/augments **rollout** — evaluates the leaf |
| Attacks which wall | Branching factor (width) | Missing evaluation function (depth) |

The last row is the point of the whole chapter. Chapter 1 identified **two** independent walls, and this is a system with **one network aimed at each**, joined by a search that was already good at converting computation into strength. Neither network plays Go by itself. The policy network alone would play plausible-looking moves with no calculation behind them; the value network alone would have no way to consider a move. Search alone, as we saw, cannot get off the ground. **The combination is the achievement**, and it is worth being precise that the intuition and the search reinforce each other: the networks make the search efficient enough to be deep, and the search corrects the networks' mistakes by actually playing the lines out.

## 2.5 The Matches

Three matches, in three years, took the system from "interesting research" to a moment that most of the public actually witnessed.

**Fan Hui, 2015.** AlphaGo played the European champion Fan Hui and won **5-0**. This was the first time a program had beaten a professional Go player in an even game on a full board, and it was announced together with the Nature paper. Within the Go world the reaction was measured — European professional strength is some way below the top of the game, and many strong players expected the gap to the very best to hold for years.

**Lee Sedol, 2016.** AlphaGo played Lee Sedol, one of the strongest players of his generation, in a five-game match, and won **4-1**. Two moves from that match are still discussed.

The first is **Move 37 of game 2**. AlphaGo played a shoulder hit on the fifth line — a move that violates a piece of conventional wisdom every Go student learns early, and one which the commentators initially assumed was a mistake or a bug. It was neither. As the game developed, the stone's influence proved decisive, and the move is now widely regarded as a genuinely creative contribution to Go theory rather than a computer's oddity. It is the moment that gave the match its cultural weight: not a machine calculating faster than a human, but a machine playing something a human would not have thought of, and being *right*.

The second is **Move 78 of game 4**, played by Lee Sedol — a wedge into the middle of AlphaGo's position that the program had evidently not weighted seriously. AlphaGo's subsequent play deteriorated and Lee won the game, the only game anyone would win against that version of the system in the match. Both moves belong in the story, and leaving out the second one gets the lesson wrong. AlphaGo was not infallible; it had regions of the game where its learned judgement was thin, and a world-class human found one of them under enormous pressure.

**Ke Jie, 2017.** A later version of the system played Ke Jie, then the world's top-ranked player, and won. Shortly after that match the program was **retired from competitive play**. There was, by then, not much left to establish by playing more human opponents, and the research had already moved on to the question Chapter 3 takes up: what happens if you remove the human games from the training entirely?

### 📚 Reading the Matches Honestly

Three cautions, because this is a story that invites overstatement.

**"Superhuman at Go" is a narrow claim.** It means: in this game, with these rules, under these time controls, this system beat these players. It does not mean the system understood Go, could explain a move, could play a variant with slightly different rules, or could do anything else at all.

**Move 37 is evidence about training, not about consciousness.** The reason AlphaGo could play a move no human would consider is that its policy had been shaped by self-play rather than only by imitation. A network trained purely to imitate human moves is, by construction, unlikely to produce a move humans do not play. Once the objective became *winning* rather than *matching expert moves*, the space of reachable strategies grew beyond the human distribution. That is a precise, mechanical explanation of "creativity" here, and it is more useful than a mystical one.

**Move 78 is evidence about coverage.** A learned evaluator is reliable in the regions of position-space its training visited and unreliable elsewhere. Lee Sedol found a position type that was thin in AlphaGo's experience. This failure mode — confident, fluent, and wrong outside the training distribution — is not specific to Go. It is the single most important caveat to carry into Chapter 5, where the same architecture is pointed at proteins and the question "is this input like the training data?" becomes a scientific question with real consequences.

## 2.6 Why This Mattered Beyond Go

Strip out the Go and a general template remains, and it is the reason this chapter belongs at the front of a series that ends in structural biology.

**Learned intuition proposes; search disposes.** A neural network turns an intractably large space of options into a short list of plausible ones. A search procedure then examines that short list properly, and — critically — its results can *overrule* the network. The network is fast and sometimes wrong; the search is slow and grounded in the actual rules. Each covers the other's failure mode.

Three lessons generalize beyond board games.

  * **Intuition can be learned rather than written.** The concepts that defeated forty years of hand-coding were captured, implicitly, by a network trained to predict expert choices. Any field where practitioners can *do* something they cannot fully *explain* is a candidate for the same treatment — and that describes a great deal of experimental science.
  * **Self-play beats imitation, when you can define winning.** Imitation caps you at the teacher. An objective you can measure lets you exceed the teacher. The catch is in the condition, and it is the hinge of this entire series: *when you can define winning*. Go hands you the definition for free. Chapter 4 is about what happens when you have to construct one.
  * **Search is how learned models are made trustworthy.** A network's output, taken alone, is a guess. The same output used to guide a search that actually plays the lines out is a guess that gets checked. This pattern — a fast learned proposal followed by a slower grounded verification — recurs constantly in applied machine learning, and it is a good default whenever a model is fluent but not reliable.

The remaining question is the one that makes the second half of this series possible. AlphaGo needed human games to start from. If the whole method depends on a large corpus of expert human decisions, its reach is limited to the handful of domains that have one. **Does it?** Chapter 3 answers that, and the answer is what turns a game-playing result into a scientific tool.

## 2.7 Hands-On: A Complete MCTS in Eighty Lines

Talking about MCTS is much less convincing than running one. Below is a complete implementation for tic-tac-toe — all four steps, the UCB rule exactly as written in Section 2.2, and random rollouts. There is **no neural network and no evaluation function of any kind**. The only Go-style knowledge in the program is the rules of the game.

The experiment: play the search against an opponent that moves uniformly at random, at several simulation budgets, and see whether skill actually tracks computation. The zero-simulation row is the same random policy on both sides, included as a baseline for what "no search" looks like.

```python
import math

import numpy as np

# ---------------------------------------------------------------
# A complete Monte Carlo Tree Search, in about 80 lines.
# Game: tic-tac-toe. No neural network, no handcrafted evaluation
# function -- the ONLY source of knowledge is random playouts.
# ---------------------------------------------------------------
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
         (0, 3, 6), (1, 4, 7), (2, 5, 8),
         (0, 4, 8), (2, 4, 6)]


def winner(board):
    """Return 1 or 2 if that player has three in a row, else 0."""
    for i, j, k in LINES:
        if board[i] != 0 and board[i] == board[j] == board[k]:
            return board[i]
    return 0


def legal(board):
    return [i for i in range(9) if board[i] == 0]


def apply_move(board, move, player):
    nb = list(board)
    nb[move] = player
    return tuple(nb)


class Node:
    """One position in the search tree.

    `player` is the side to move here. `W` accumulates results from
    the viewpoint of the side that moved INTO this node -- that is,
    the side to move at the parent. That convention is what makes
    the UCB comparison at the parent an apples-to-apples one.
    """

    def __init__(self, board, player, parent=None, move=None):
        self.board, self.player = board, player
        self.parent, self.move = parent, move
        self.children = []
        self.result = winner(board)
        self.terminal = self.result != 0 or not legal(board)
        self.untried = [] if self.terminal else legal(board)
        self.N, self.W = 0, 0.0

    def ucb(self, c):
        """Upper Confidence Bound: exploitation + exploration."""
        return self.W / self.N + c * math.sqrt(math.log(self.parent.N) / self.N)


def reward(win, mover):
    """Score a finished game from `mover`'s point of view."""
    if win == 0:
        return 0.5
    return 1.0 if win == mover else 0.0


def mcts_move(board, player, n_sims, rng, c=math.sqrt(2)):
    root = Node(board, player)
    for _ in range(n_sims):
        node = root

        # --- 1. SELECTION: descend by UCB while the node is fully expanded
        while not node.terminal and not node.untried:
            node = max(node.children, key=lambda ch: ch.ucb(c))

        # --- 2. EXPANSION: add one unexplored child
        if not node.terminal:
            mv = node.untried.pop(rng.integers(len(node.untried)))
            child = Node(apply_move(node.board, mv, node.player),
                         3 - node.player, parent=node, move=mv)
            node.children.append(child)
            node = child

        # --- 3. SIMULATION: play the rest of the game at random
        b, p = node.board, node.player
        win = winner(b)
        while win == 0 and 0 in b:
            mv = legal(b)[rng.integers(len(legal(b)))]
            b = apply_move(b, mv, p)
            win, p = winner(b), 3 - p

        # --- 4. BACKUP: push the result up the path just travelled
        while node is not None:
            node.N += 1
            if node.parent is not None:
                node.W += reward(win, node.parent.player)
            node = node.parent

    # Play the most-visited move, not the highest-scoring one:
    # visit count is the more robust statistic.
    return max(root.children, key=lambda ch: ch.N).move


def random_move(board, player, rng):
    mv = legal(board)
    return mv[rng.integers(len(mv))]


def play_game(n_sims, rng):
    """MCTS (player 1, moves first) vs a uniformly random opponent."""
    board, player = (0,) * 9, 1
    while True:
        w = winner(board)
        if w or not legal(board):
            return w
        if player == 1:
            mv = random_move(board, player, rng) if n_sims == 0 \
                else mcts_move(board, player, n_sims, rng)
        else:
            mv = random_move(board, player, rng)
        board, player = apply_move(board, mv, player), 3 - player


# --- The experiment: does more search actually mean more skill? ---
N_GAMES = 200
rng = np.random.default_rng(0)

print(f"MCTS (X, first) vs random (O) -- {N_GAMES} games per setting")
print(f"{'simulations/move':>18} {'win':>8} {'draw':>8} {'loss':>8}")
print("-" * 46)
for n_sims in [0, 10, 100, 1000]:
    results = np.array([play_game(n_sims, rng) for _ in range(N_GAMES)])
    win = np.mean(results == 1) * 100
    draw = np.mean(results == 0) * 100
    loss = np.mean(results == 2) * 100
    label = "0 (pure random)" if n_sims == 0 else str(n_sims)
    print(f"{label:>18} {win:7.1f}% {draw:7.1f}% {loss:7.1f}%")
```

**Output:**

```
MCTS (X, first) vs random (O) -- 200 games per setting
  simulations/move      win     draw     loss
----------------------------------------------
   0 (pure random)    58.5%    12.5%    29.0%
                10    87.5%     5.5%     7.0%
               100    97.5%     2.5%     0.0%
              1000    98.0%     2.0%     0.0%
```

**Reading the result.** Four observations, in increasing order of importance.

  * **The baseline confirms the game is not trivially won.** Random against random, moving first, wins 58.5% and *loses 29.0%*. First-move advantage exists but does not carry you. Every improvement in the rows below is attributable to the search and to nothing else.

  * **Even ten simulations per move is a large improvement.** Ten random playouts is an absurdly small budget — the tree has barely more nodes than the board has squares — and yet the win rate jumps from 58.5% to 87.5% and losses drop from 29.0% to 7.0%. This is the anytime property being useful at the very bottom of the range: a nearly-free search is already much better than none.

  * **The losses go to zero, and stay there.** At 100 simulations per move the program loses **0.0%** of 200 games, and the same at 1000. This is the qualitative change that matters. Winning more often is a quantitative improvement; *never losing* means the search is reliably finding the forced tactics — the immediate wins and the blocks of the opponent's threats — that decide tic-tac-toe. It has, in effect, worked out how to play the game, from the rules and random playouts alone.

  * **The returns flatten, for a good reason.** Going from 100 to 1000 simulations moves the win rate only from 97.5% to 98.0%: a factor of ten in computation for half a percentage point. The search has essentially solved the game, and the remaining draws are not mistakes — they are games in which the random opponent happened to stumble into correct defensive moves, and a drawn game against correct defence is the best result available. **The ceiling here is the game, not the algorithm.** That is exactly what you want to see, and it is why this experiment is a fair demonstration rather than a flattering one.

Three things to try, each of which teaches something specific.

  * **Change the exploration constant.** Set `c=0.0` in `mcts_move` for pure exploitation, and `c=5.0` for near-pure exploration, and re-run the 100-simulation row. Both should be worse than \\(\sqrt{2}\\), and they will be worse in different ways.
  * **Make the search play second.** Swap the roles in `play_game` so the random player moves first. Results will be worse across the board — tic-tac-toe favours the first player — but the losses should still vanish at a high enough budget.
  * **Point it at a real evaluator.** Replace step 3 with a call to any function that returns a win probability for `node.board`, and you have converted this program into the skeleton of AlphaGo. That substitution — a learned judgement in place of the random playout — is the single largest idea in this chapter, and it is a five-line change.

### 🎯 Exercise Problems

  1. **The sign convention.** Section 2.1 warns that the backup step must record results from the viewpoint of the player who moved *into* each node. In the code, find the line that implements this. Then predict, in words, exactly how the program would play if you changed `node.parent.player` to `node.player` — and explain why that is worse than merely playing randomly.

  2. **Reading the UCB formula.** A node has been visited 100 times. Child A has \\(W/N = 0.6\\) from 50 visits; child B has \\(W/N = 0.8\\) from 4 visits. With \\(c = \sqrt{2}\\), compute both UCB scores by hand and state which is selected. Then find the number of additional visits to B, holding its win rate fixed, after which A would be selected instead.

  3. **Which wall, which network.** For each of the following, state whether it addresses the width problem or the evaluation problem, and which of AlphaGo's two networks (if either) plays that role: (a) alpha-beta pruning; (b) the UCB exploration term; (c) random rollouts; (d) the policy network's move probabilities; (e) the value network's win estimate.

  4. **Imitation versus objective.** Explain, in terms of training signals, why a policy network trained *only* on human games would be very unlikely to produce a move like Move 37, and why self-play training makes such a move reachable. Then state one risk that self-play introduces which imitation learning does not have.

  5. **Diagnosing Move 78.** Lee Sedol's game-4 wedge exposed a position type that AlphaGo evaluated poorly. Describe two concrete things you could measure about a learned evaluator, *before* deploying it, that would give you warning about such blind spots. State clearly what each measurement can and cannot detect.

## Summary

**Monte Carlo Tree Search** grows a tree that is deep where the game is interesting and shallow everywhere else, one node per simulation, through four steps: **selection** down the existing tree, **expansion** of one new child, **simulation** of a rollout to the end of the game, and **backup** of the result along the path travelled — recorded from the viewpoint of the player who moved into each node. It has the **anytime property**: interrupt it whenever you like and its current best answer is valid, so computation converts to strength smoothly, with no thresholds.

Selection uses the **UCB rule**, \\(W/N + c\sqrt{\ln N(s) / N(s,a)}\\), whose first term is the empirical win rate and whose second is a measure of ignorance shaped like a statistical error bar. The \\(1/\sqrt{N(s,a)}\\) factor makes under-sampled moves attractive; the \\(\ln N(s)\\) growth guarantees no move is abandoned permanently while keeping the wasted fraction of simulations small. Crucially, **the evaluator improves itself** as simulations accumulate — no human knowledge is required to start.

MCTS alone plateaued at strong amateur Go for two reasons: random rollouts destroy the precise sequences that decide many positions, and every node still has roughly 250 moves that must each be sampled before they can be dismissed. **AlphaGo, published in Nature in 2016, added one network for each problem.** The **policy network** — trained first on human expert moves, then improved by self-play policy gradient — outputs a distribution over plausible moves and biases *selection*, collapsing the effective branching factor without forbidding anything. The **value network** — trained on self-play outcomes — outputs a win probability and replaces or augments the *rollout*, supplying the static evaluation function that decades of hand-coding could not produce. The two were combined with rollouts rather than replacing them, because estimators that fail differently are stronger together.

The results followed quickly: **Fan Hui in 2015 (5-0)**, the first professional beaten in an even game; **Lee Sedol in 2016 (4-1)**, remembered for **Move 37 of game 2**, the shoulder hit that contradicted conventional wisdom and turned out to be right, and for **Move 78 of game 4**, Lee's wedge into a blind spot that won the only game against AlphaGo in the match; and **Ke Jie in 2017**, after which the system was retired from competitive play. Read honestly, Move 37 is evidence about the training objective — self-play frees a policy from the human distribution — and Move 78 is evidence about coverage, the confident-and-wrong failure of a learned evaluator outside its training experience.

The template that generalizes is **learned intuition proposes, search disposes**: a fast network reduces an intractable space to a short list, and a grounded search checks it and can overrule it. Our tic-tac-toe implementation demonstrated the search half of that with no learning at all — losses fell from 29.0% at zero simulations to 7.0% at ten and to **0.0% at one hundred**, with the remaining 2% draws being the game's own ceiling rather than the algorithm's.

One dependency remains, and it is the one that decides whether any of this reaches science: AlphaGo learned its intuition from a large corpus of human expert games. The next chapter removes that corpus entirely, examines what a system that starts from nothing but the rules discovers on its own, and asks what is left of the method when the game itself is taken away.

[← Chapter 1: Games as the Proving Ground](<chapter-1.html>) [Chapter 3: Zero and Beyond: Learning Without Humans →](<chapter-3.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
