---
title: "Chapter 2: Advanced Sampling Methods"
chapter_title: "Chapter 2: Advanced Sampling Methods"
---

🌐 EN | [🇯🇵 JP](<../../../jp/FM/computational-statistical-mechanics/chapter-2.html>) | Last sync: 2025-11-16

[Fundamentals of Mathematics Dojo](<../index.html>) > [Computational Statistical Mechanics](<index.html>) > Chapter 2 

## Learning Objectives

In Chapter 1 we learned how to sample configurations from the Boltzmann distribution using the Metropolis method. In this chapter we study advanced techniques that handle situations where naive Monte Carlo struggles: state spaces separated by high energy barriers, and high-weight regions that are visited only rarely. The ideas developed here carry over directly to the Molecular Dynamics method of Chapter 3, for example as replica-exchange molecular dynamics. 

  * Understand why naive sampling breaks down, and how importance sampling reduces variance
  * Explain the acceptance rule and temperature ladder of extended-ensemble methods, especially replica exchange
  * Understand how the Wang-Landau method estimates the density of states directly
  * Understand the principle of free-energy calculation via umbrella sampling and WHAM
  * Compare the strengths and weaknesses of each method and select the right one for a problem

**Chapter metadata**  
Reading time: 30-35 minutes / Difficulty: Advanced / Code examples: 3 (all executed) / Exercises: 5 

## 2.1 Limits of Simple Sampling and Importance Sampling

### Why does naive Monte Carlo break down?

In statistical mechanics, at inverse temperature \\( \beta = 1/(k_B T) \\) a system follows the Boltzmann distribution \\[ p(\mathbf{x}) = \frac{1}{Z}\, e^{-\beta U(\mathbf{x})}, \qquad Z = \int e^{-\beta U(\mathbf{x})}\, d\mathbf{x}. \\] Our goal is to evaluate the expectation of an observable \\( A \\), \\( \langle A \rangle = \int A(\mathbf{x})\, p(\mathbf{x})\, d\mathbf{x} \\). However, if we draw points uniformly from configuration space ("simple sampling"), the probability of visiting the high-weight region where \\( p \\) is concentrated becomes exponentially small, and the variance of the estimator explodes. 

**Importance sampling** instead generates points from a proposal distribution \\( q(\mathbf{x}) \\) that is easy to sample, and corrects with the weight \\( w = p/q \\): \\[ \langle A \rangle = \int A(\mathbf{x})\, \frac{p(\mathbf{x})}{q(\mathbf{x})}\, q(\mathbf{x})\, d\mathbf{x} \approx \frac{1}{N}\sum_{i=1}^{N} A(\mathbf{x}_i)\, \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}, \qquad \mathbf{x}_i \sim q. \\] If \\( q \\) is shifted toward the important region, a high-accuracy estimate is obtained with far fewer samples. 

### Code Example 1: Variance Reduction by Importance Sampling

We estimate the tail probability \\( P(X \gt 4) \\) of a standard normal distribution. This is the smallest model of a "rare event" in which the high-weight region is seldom visited. We compare naive Monte Carlo with importance sampling. 

import numpy as np from math import erfc, sqrt np.random.seed(42) # Goal: estimate the tail probability P(X > a) for X ~ N(0,1). # This is the archetype of a "rare event": a Boltzmann average dominated # by a small, high-weight region of configuration space. a = 4.0 N = 200000 # --- Naive Monte Carlo: sample from N(0,1), count the fraction above a --- x_naive = np.random.randn(N) hits = (x_naive > a).astype(float) p_naive = hits.mean() se_naive = np.sqrt(hits.var(ddof=1) / N) # standard error of the estimator # --- Importance sampling: draw from a shifted Gaussian q = N(a, 1) --- # Estimator: E_p[1_{x>a}] = E_q[1_{x>a} * p(x)/q(x)] x_is = a + np.random.randn(N) w = np.exp(-0.5 * x_is**2) / np.exp(-0.5 * (x_is - a)**2) # weight p/q contrib = (x_is > a).astype(float) * w p_is = contrib.mean() se_is = np.sqrt(contrib.var(ddof=1) / N) p_true = 0.5 * erfc(a / sqrt(2.0)) # reference via complementary error function print(f"Reference P(X>{a}) = {p_true:.6e}") print(f"Naive MC: estimate = {p_naive:.6e}, std.err = {se_naive:.2e}") print(f"Importance sampling: estimate = {p_is:.6e}, std.err = {se_is:.2e}") print(f"Variance reduction factor = {se_naive**2 / se_is**2:.1f}x") print(f"Naive samples above a: {int(hits.sum())} / {N}")

Reference P(X>4.0) = 3.167124e-05 Naive MC: estimate = 1.500000e-05, std.err = 8.66e-06 Importance sampling: estimate = 3.163344e-05, std.err = 1.50e-07 Variance reduction factor = 3326.3x Naive samples above a: 3 / 200000

Naive Monte Carlo reaches the tail with only 3 of 200,000 samples; its estimate is far from the true value and its standard error is as large as the estimate itself. Importance sampling, with the proposal shifted into the tail, reduces the standard error by a factor of about 3300 for the same sample size and reproduces the true value well. This is the core idea of advanced sampling: guide the samples toward the important region. 

## 2.2 Extended-Ensemble Methods (Replica Exchange)

### Energy barriers and broken ergodicity

Because the Metropolis method of Chapter 1 relies on local proposal moves, a system with several metastable states separated by high energy barriers becomes trapped in one state and cannot sample the full configuration space. This is called **broken ergodicity**. The lower the temperature (larger \\( \beta \\)), the more the probability of crossing a barrier decays exponentially as \\( e^{-\beta \Delta U} \\). 

The **replica exchange method** (parallel tempering) places \\( M \\) replicas on a temperature ladder of inverse temperatures \\( \beta_1 \gt \beta_2 \gt \cdots \gt \beta_M \\), evolves each with independent Metropolis updates, and stochastically swaps configurations between neighbouring replicas. The hot replicas cross barriers easily, and through the swaps this mobility is transferred to the cold replicas. 

A swap of configurations between replicas \\( m \\) and \\( n \\) is accepted with the following probability so as to satisfy detailed balance: \\[ P_{\text{acc}} = \min\\!\left(1,\; \exp\big[(\beta_m - \beta_n)\,(U_m - U_n)\big]\right), \\] where \\( U_m \\) is the energy of the configuration currently held by replica \\( m \\). The temperature ladder is typically set in a geometric progression so that the energy distributions of neighbouring replicas overlap sufficiently. 

### Code Example 2: Barrier Crossing in a Double-Well Potential

For the double-well potential \\( U(x) = h\,(x^2-1)^2 \\) we compare plain Metropolis at low temperature with replica exchange. At low temperature the barrier \\( \beta h = 20 \\) is high, so the naive method should be unable to escape its initial well. 

import numpy as np # Double-well potential U(x) = h * (x^2 - 1)^2 # Two minima at x = +/-1, a barrier of height h at x = 0. h = 5.0 def U(x): return h * (x**2 - 1.0)**2 def metropolis_run(beta, n_steps, step=0.3, x0=-1.0, rng=None): """Single-temperature plain Metropolis. Returns trajectory and #crossings.""" if rng is None: rng = np.random.default_rng() x = x0 traj = np.empty(n_steps) crossings = 0 prev_sign = np.sign(x) for i in range(n_steps): xp = x + step * rng.standard_normal() if rng.random() < np.exp(-beta * (U(xp) - U(x))): x = xp s = np.sign(x) if s != 0 and s != prev_sign: crossings += 1 prev_sign = s traj[i] = x return traj, crossings def replica_exchange(betas, n_steps, step=0.3, swap_every=10, rng=None): """Replica exchange (parallel tempering) on a temperature ladder.""" if rng is None: rng = np.random.default_rng() M = len(betas) xs = np.full(M, -1.0) # all replicas start in the left well coldest = np.empty(n_steps) # trajectory of the lowest-T replica crossings = 0 prev_sign = np.sign(xs[0]) swap_attempts = 0 swap_accepts = 0 for i in range(n_steps): # local Metropolis update for every replica for m in range(M): xp = xs[m] + step * rng.standard_normal() if rng.random() < np.exp(-betas[m] * (U(xp) - U(xs[m]))): xs[m] = xp # exchange of neighbouring replicas if i % swap_every == 0: for m in range(M - 1): swap_attempts += 1 delta = (betas[m] - betas[m+1]) * (U(xs[m]) - U(xs[m+1])) if rng.random() < np.exp(delta): xs[m], xs[m+1] = xs[m+1], xs[m] swap_accepts += 1 # track the coldest replica (index 0 = largest beta) s = np.sign(xs[0]) if s != 0 and s != prev_sign: crossings += 1 prev_sign = s coldest[i] = xs[0] acc = swap_accepts / max(swap_attempts, 1) return coldest, crossings, acc n_steps = 40000 beta_cold = 4.0 # low temperature => tall effective barrier (beta*h = 20) # plain Metropolis at the cold temperature traj_plain, cross_plain = metropolis_run(beta_cold, n_steps, x0=-1.0, rng=np.random.default_rng(7)) frac_right_plain = np.mean(traj_plain > 0) # replica exchange with a geometric temperature ladder betas = np.array([4.0, 2.4, 1.44, 0.864, 0.5]) # beta ladder (cold -> hot) traj_re, cross_re, swap_acc = replica_exchange(betas, n_steps, rng=np.random.default_rng(7)) frac_right_re = np.mean(traj_re > 0) print(f"Barrier height h = {h}, cold beta = {beta_cold} (beta*h = {beta_cold*h:.0f})") print(f"Temperature ladder (beta): {betas}") print() print("Plain Metropolis at the cold temperature:") print(f" barrier crossings = {cross_plain}") print(f" fraction of time in the right well (x>0) = {frac_right_plain:.3f}") print() print("Replica exchange (coldest replica):") print(f" barrier crossings = {cross_re}") print(f" fraction of time in the right well (x>0) = {frac_right_re:.3f}") print(f" mean neighbour-swap acceptance = {swap_acc:.2f}") print() print("By symmetry the exact fraction in each well is 0.500.")

Barrier height h = 5.0, cold beta = 4.0 (beta*h = 20) Temperature ladder (beta): [4. 2.4 1.44 0.864 0.5 ] Plain Metropolis at the cold temperature: barrier crossings = 0 fraction of time in the right well (x>0) = 0.000 Replica exchange (coldest replica): barrier crossings = 1622 fraction of time in the right well (x>0) = 0.494 mean neighbour-swap acceptance = 0.82 By symmetry the exact fraction in each well is 0.500.

Plain Metropolis never crosses the barrier in 40,000 steps and stays completely trapped in the initial left well (x<0), giving a right-well fraction of 0.000. Replica exchange, by contrast, lets the coldest replica cross the barrier 1622 times and samples both wells almost symmetrically (0.494 versus the exact value 0.500). The neighbour-swap acceptance of 0.82 confirms that the temperature ladder is well designed. 

## 2.3 The Wang-Landau Method

### Estimating the density of states directly

The methods so far sampled at a fixed temperature, but the **Wang-Landau method** estimates a temperature-independent quantity directly: the **density of states** \\( g(E) \\). Once \\( g(E) \\) is known, the partition function at any temperature \\( Z(\beta) = \sum_E g(E)\, e^{-\beta E} \\) and the free energy follow from a single calculation. 

The idea is to achieve a **flat histogram** in energy space. If transitions are accepted with probability \\[ P(E \to E') = \min\\!\left(1,\; \frac{g(E)}{g(E')}\right), \\] the system visits each energy in proportion to \\( 1/g(E) \\), and the visitation frequencies begin to level out. On each visit \\( g(E) \\) is updated with a modification factor \\( f \\) as \\( \ln g(E) \leftarrow \ln g(E) + \ln f \\); every time the histogram becomes flat enough, \\( f \\) is refined to \\( \sqrt{f} \\). As \\( f \\) approaches 1, \\( g(E) \\) converges to the true density of states. 

### Code Example 3: Density of States of the 2D Ising Model

For the \\( 4\times4 \\) two-dimensional Ising model we estimate the density of states with the Wang-Landau method. With 16 spins we can enumerate all \\( 2^{16} \\) states to compute the exact \\( g(E) \\), so the accuracy of the estimate can be verified directly. 

import numpy as np L = 4 # 4x4 periodic Ising lattice (16 spins) N = L * L def energy(state): """Total energy of a 2D Ising configuration with periodic boundaries, J=1.""" s = state.reshape(L, L) e = 0 e -= np.sum(s * np.roll(s, 1, axis=0)) e -= np.sum(s * np.roll(s, 1, axis=1)) return int(e) # ---- Exact density of states by brute-force enumeration (2^16 states) ---- exact = {} for idx in range(2**N): bits = (idx >> np.arange(N)) & 1 state = 2 * bits - 1 # map {0,1} -> {-1,+1} E = energy(state) exact[E] = exact.get(E, 0) + 1 energies = sorted(exact.keys()) # all energies allowed on the 4x4 lattice # ---- Wang-Landau estimation of the density of states ---- def wang_landau(seed=0, flat=0.90, f_final=1e-6): rng = np.random.default_rng(seed) e_index = {E: i for i, E in enumerate(energies)} nbins = len(energies) lng = np.zeros(nbins) # ln g(E), updated in place state = rng.integers(0, 2, size=N) * 2 - 1 E = energy(state) f = 1.0 # ln modification factor starts at ln(e)=1 sweeps = 0 while f > f_final: hist = np.zeros(nbins) flat_enough = False while not flat_enough: for _ in range(N * 100): k = rng.integers(N) s = state.reshape(L, L) i, j = divmod(k, L) nb = (s[(i+1) % L, j] + s[(i-1) % L, j] \+ s[i, (j+1) % L] + s[i, (j-1) % L]) dE = 2 * s[i, j] * nb Enew = E + dE iold, inew = e_index[E], e_index[Enew] if np.log(rng.random()) < lng[iold] - lng[inew]: state[k] *= -1 E = Enew lng[e_index[E]] += f hist[e_index[E]] += 1 sweeps += 100 nonzero = hist[hist > 0] if nonzero.min() > flat * hist.mean(): flat_enough = True f *= 0.5 # refine the modification factor return lng, sweeps lng, sweeps = wang_landau(seed=1) # Normalise: the smallest g equals 2 (the two fully ordered ground states). lng -= lng.min() lng += np.log(2.0) # Rescale so that sum g(E) = 2^N. lng += np.log(2**N) - np.log(np.sum(np.exp(lng))) print(f"{'E':>5} {'exact g(E)':>12} {'WL g(E)':>16} {'rel.err':>10}") for E in energies: ge = exact[E] gw = np.exp(lng[energies.index(E)]) print(f"{E:>5} {ge:>12d} {gw:>16.1f} {abs(gw-ge)/ge:>9.2%}") print() print(f"Total Wang-Landau sweeps: {sweeps}") print(f"Sum of exact g(E) = {sum(exact.values())} (should equal 2^16 = {2**N})")

E exact g(E) WL g(E) rel.err -32 2 2.0 0.83% -24 32 33.3 4.01% -20 64 65.1 1.73% -16 424 425.3 0.32% -12 1728 1737.9 0.57% -8 6688 6649.8 0.57% -4 13568 13647.4 0.59% 0 20524 20487.4 0.18% 4 13568 13595.8 0.21% 8 6688 6672.1 0.24% 12 1728 1706.3 1.26% 16 424 417.4 1.55% 20 64 63.0 1.60% 24 32 31.1 2.79% 32 2 2.0 0.22% Total Wang-Landau sweeps: 71300 Sum of exact g(E) = 65536 (should equal 2^16 = 65536)

The estimated density of states reproduces \\( g(E) \\), which varies over seven orders of magnitude, to within a few percent across the whole energy range. Importantly, the ground state that governs the low-temperature physics \\( (E=-32,\; g=2) \\) is captured correctly. Obtaining the thermodynamics at all temperatures from a single calculation is the greatest advantage of the Wang-Landau method. 

## 2.4 Umbrella Sampling and Free-Energy Calculation

### Free energy along a reaction coordinate (PMF)

In chemical reactions and phase transitions we often want the free-energy surface along a **reaction coordinate** \\( \xi \\), the so-called potential of mean force (PMF), \\[ F(\xi) = -\frac{1}{\beta}\, \ln p(\xi), \qquad p(\xi) = \int \delta(\xi(\mathbf{x}) - \xi)\, p(\mathbf{x})\, d\mathbf{x}. \\] If \\( F(\xi) \\) contains a high barrier, ordinary sampling yields almost no data for \\( p(\xi) \\) near the barrier. 

**Umbrella sampling** divides the reaction coordinate into several windows and adds a harmonic bias potential to each window \\( i \\), \\[ w_i(\xi) = \frac{1}{2}\, k_i\, (\xi - \xi_i^{0})^2, \\] holding the system near the target \\( \xi_i^{0} \\) with an "umbrella". Each window then obtains sufficient statistics even near the barrier. Removing the bias from the biased histograms \\( p_i^{b}(\xi) \\) of each window and stitching the windows together to recover the original \\( F(\xi) \\) is the job of **WHAM (the Weighted Histogram Analysis Method)**. 

WHAM determines the free-energy shift \\( f_i \\) of each window and the unbiased probability \\( p(\xi) \\) by iterating the following coupled equations self-consistently: \\[ p(\xi) = \frac{\sum_{i} n_i\, p_i^{b}(\xi)} {\sum_{i} N_i\, e^{-\beta [w_i(\xi) - f_i]}}, \qquad e^{-\beta f_i} = \int e^{-\beta w_i(\xi)}\, p(\xi)\, d\xi. \\] Here \\( N_i \\) is the total number of samples in window \\( i \\) and \\( n_i \\) is the per-bin count. It can be understood as a generalization of the weight correction of importance sampling (Section 2.1) to many windows. 

A modern framework that combines the weights of configurations obtained from umbrella sampling or replica exchange and merges estimators across several thermodynamic states is MBAR (the Multistate Bennett Acceptance Ratio). WHAM corresponds to its histogram version. 

## 2.5 Comparison of Methods and Practical Guidance

All the methods in this chapter share the common goal of guiding samples into regions that naive sampling cannot reach, but the problem settings they excel at differ. The table below summarizes them. 

Method | Main purpose | Well suited to | Main caveats  
---|---|---|---  
Importance sampling | Estimating expectations and rare events | Low-dimensional problems with a good proposal | A poor proposal makes the weight variance diverge  
Replica exchange | Equilibrium sampling across barriers | Systems with several metastable states | Ladder design and the cost of many replicas  
Wang-Landau | Direct estimation of the density of states | Discrete-energy models, all-temperature thermodynamics | Hard to extend to continuous or large systems  
Umbrella sampling + WHAM | Free-energy surface along a reaction coordinate | Reactions/transitions with a known good coordinate | Success hinges on the coordinate and window overlap  
  
In practice, deciding whether you need "an average at a specific temperature", "a free-energy landscape", or "thermodynamics at all temperatures" quickly points to the right method. Moreover, the ideas of extended ensembles and biasing learned here are not limited to discrete Monte Carlo updates. They enter the molecular dynamics of the next chapter naturally, as replica-exchange molecular dynamics (REMD) and metadynamics. The very notion of accelerating sampling is a common pillar running through computational statistical mechanics. 

## Exercises

### Exercise 2.1: The proposal distribution in importance sampling

In Code Example 1, how does the standard error change if the proposal is switched from \\( q = N(a, 1) \\) to \\( q = N(0, 1) \\) (i.e. identical to naive MC)? Explain the reason from the variance expression, and discuss whether an optimum exists as the variance \\( \sigma^2 \\) of \\( q = N(a, \sigma^2) \\) is varied. 

### Exercise 2.2: Deriving the replica-exchange acceptance rule

When two replicas at inverse temperatures \\( \beta_m, \beta_n \\) hold configurations of energies \\( U_m, U_n \\), derive the acceptance probability \\( \min(1, \exp[(\beta_m-\beta_n)(U_m-U_n)]) \\) from the ratio of Boltzmann weights before and after the swap, on the basis of detailed balance. 

### Exercise 2.3: Designing the temperature ladder

If you reduce the ladder in Code Example 2 to \\( M=3 \\), or widen the spacing between rungs, how do the neighbour-swap acceptance and the number of barrier crossings change? Vary the parameters, run the code, and analyze the trade-off between acceptance and sampling efficiency quantitatively. 

### Exercise 2.4: Thermodynamics from Wang-Landau

Using the density of states \\( g(E) \\) from Code Example 3, compute the internal energy \\( \langle E \rangle(\beta) \\) and the specific heat \\( C(\beta) \\) as functions of temperature from the partition function \\( Z(\beta)=\sum_E g(E)e^{-\beta E} \\), and find the temperature at which the specific heat peaks. 

### Exercise 2.5: Applying umbrella sampling

For the double-well potential of Section 2.2, implement umbrella sampling with harmonic biases in several windows using the reaction coordinate \\( \xi = x \\), and recover the free energy \\( F(x) \\) with WHAM or a simple weight correction. Check that the obtained \\( F(x) \\) is consistent with the shape of \\( U(x) \\) (two wells and a barrier). 

## Checking the Learning Objectives

  * Have you understood, through the measured 3300x factor, why simple sampling fails to visit the high-weight region and how importance sampling reduces variance?
  * Can you explain the acceptance rule and the role of the temperature ladder in replica exchange, and show the difference from plain Metropolis with experimental results?
  * Can you explain how the Wang-Landau method estimates the density of states directly via a flat histogram?
  * Have you understood how umbrella sampling and WHAM recover the free-energy surface along a reaction coordinate?
  * Can you compare the strengths and weaknesses of the four methods and choose the one suited to a given problem?

## Summary

  * When sampling the Boltzmann distribution, the high-weight region is visited only rarely and naive methods break down; importance sampling overcomes this with a proposal distribution and weight correction.
  * Replica exchange stochastically swaps replicas on a temperature ladder, achieving equilibrium sampling across high energy barriers.
  * The Wang-Landau method estimates the temperature-independent density of states directly, yielding the thermodynamics at all temperatures from a single calculation.
  * Umbrella sampling and WHAM recover the free-energy surface along a reaction coordinate through bias potentials and weight correction.
  * All these methods rest on the common idea of guiding and accelerating sampling toward important regions, and they carry over to molecular dynamics.

## Next Steps

In this chapter we learned probabilistic sampling methods that explore state space efficiently. Chapter 3 changes perspective and treats the Molecular Dynamics method, which generates configurations by integrating the equations of motion in time. Deterministic time evolution and the probabilistic sampling of this chapter may seem to be opposites, but combining them, as in replica-exchange molecular dynamics, enables even more powerful sampling. Let us move on to the next chapter. 

## References

  1. D. Frenkel and B. Smit, _Understanding Molecular Simulation: From Algorithms to Applications_ , 2nd ed., Academic Press, 2002.
  2. M. E. J. Newman and G. T. Barkema, _Monte Carlo Methods in Statistical Physics_ , Oxford University Press, 1999.
  3. K. Hukushima and K. Nemoto, "Exchange Monte Carlo Method and Application to Spin Glass Simulations," _J. Phys. Soc. Jpn._ 65, 1604 (1996).
  4. F. Wang and D. P. Landau, "Efficient, Multiple-Range Random Walk Algorithm to Calculate the Density of States," _Phys. Rev. Lett._ 86, 2050 (2001).
  5. S. Kumar et al., "The Weighted Histogram Analysis Method for Free-Energy Calculations on Biomolecules," _J. Comput. Chem._ 13, 1011 (1992).

[Previous](<chapter-1.html>) [Contents](<index.html>) [Next](<chapter-3.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
