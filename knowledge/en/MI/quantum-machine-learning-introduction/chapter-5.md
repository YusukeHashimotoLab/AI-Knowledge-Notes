---
title: "Chapter 5: An Honest Assessment"
chapter_title: "Chapter 5: An Honest Assessment"
subtitle: Dequantization, Classical Surrogates, Benchmark Hygiene, and the One Thing That Would Change the Answer
reading_time: 45-50 minutes
difficulty: Advanced
code_examples: 6
exercises: 5
---

🌐 EN | [🇯🇵 JP](<../../../jp/MI/quantum-machine-learning-introduction/chapter-5.html>) | Last sync: 2026-08-13

[Materials Informatics Dojo](<../index.html>) > [Introduction to Quantum Machine Learning](<index.html>) > Chapter 5

Four chapters have built quantum machine-learning models and measured them. None of them won. Chapter 3's entangling fidelity kernel came out 1.16 times worse than a tuned radial basis function under an identical protocol, and the one quantum kernel there that matched the RBF was a *product* angle encoding with no entanglement in it at all — a classical kernel in quantum clothing, as that chapter said. Chapter 4's variational circuit lost to a NumPy network of the same size by a factor of 3.9 — a loss its paired bootstrap interval resolves against the smaller of the two matched networks and calls a tie against the larger — and lost to five-parameter linear regression as well. None of those losses was caused by hardware: every number in this course came from an exact, noiseless simulator, so no error rate, coherence time or qubit count was ever in the way.

This chapter explains why that happened, and — more usefully — what it would take for it not to happen. The explanation has a name, *dequantization*, and it is not a slogan: it is a family of constructions that take a quantum model and produce a classical model with comparable performance and polynomial cost. Section 5.2 builds two of them, from scratch, and measures how close they get. They get very close, and one of them beats the quantum model it was built to imitate.

That result then has to be read carefully, because the temptation in both directions is strong. "Quantum machine learning does not work" is as unjustified by this evidence as "quantum machine learning will revolutionize materials discovery". What the evidence supports is narrower and more useful: for *classical* data of the kind that sits in a materials database, the models in this course have no advantage to offer, the obstruction is mathematical rather than technological, and the specific thing that changes the argument is not a better machine but a different kind of data. Sections 5.3 and 5.4 make that case and say what a materials researcher should do about it now.

## Learning Objectives

After completing this chapter, you will be able to:

  * State what dequantization does and does not prove, and distinguish the formal complexity-theoretic result from the loose sense in which the word is usually used
  * Construct a low-weight Pauli (classical shadow) surrogate of a quantum kernel, count its features as a function of qubit number, and measure how much of the kernel it reproduces
  * Construct a truncated-Fourier surrogate from the encoding's own frequency spectrum, verify numerically that the spectrum is finite, and compare the surrogate's test error against the quantum model's
  * Demonstrate, on one data set, three arithmetically true and mutually contradictory summaries of the same experiment, and name the manipulation behind each
  * Apply an eight-point hygiene checklist to a quantum-advantage claim and identify which points a given claim leaves unanswered
  * Explain why quantum *data* removes the input problem that dominates the classical-data case, what new costs it introduces, and which materials-science measurements plausibly qualify
  * Decide, for a specific research programme, which parts of this subject are worth learning now and which are worth waiting on

* * *

## 5.1 What Dequantization Actually Says

### Two different claims wearing one word

In the technical literature, *dequantization* names a specific kind of theorem. Given a quantum algorithm that achieves some accuracy with some resource count, a dequantization result exhibits a classical algorithm achieving comparable accuracy with cost polynomial in the same parameters — under a matched assumption about how the input is provided, namely **sample-and-query access**: the ability to draw an index with probability proportional to the squared entry, and to read any entry, which is the classical analogue of the quantum algorithm's assumed efficient state preparation. That assumption is doing the work, and so is a second one that is easy to miss: the results apply where the relevant matrix is **low rank** or otherwise has a small effective description. The best-known examples concern the quantum linear-algebra algorithms that many QML proposals were built on top of: recommendation systems, principal component analysis, low-rank linear systems. In each case the exponential speedup turned out to depend on an input model that, when granted equally to a classical algorithm, made the classical algorithm fast too.

What has *not* been dequantized is worth naming in the same breath, because "quantum linear algebra was dequantized" is routinely over-read. The original quantum linear-system algorithm for **sparse, well-conditioned** matrices — where the input is a sparse-access oracle rather than a low-rank sampling structure — has no known classical analogue, and the dequantization technique does not reach it: it needs the low-rank structure that sparse well-conditioned systems do not have. So the correct summary is that a particular *input model* was equalised for a particular *matrix class*, not that quantum linear algebra was refuted.

In casual use, "dequantized" means something weaker and more practical: *someone built a classical model that does as well*. That is not a theorem, it is an experiment, and it is what this chapter does. The weaker statement is the one that matters for a decision about where to spend a research year, because it does not require the classical construction to be provably general — it only has to work on the problem in front of you.

Both senses share a moral, and it is worth stating before any code. A quantum model's advantage is never a property of the quantum model alone. It is a *gap*, and a gap has two sides. Most reported gaps in this field have been closed from the classical side, by someone writing down the classical model that the quantum construction was implicitly describing.

### Code Example 1: The Quantum Kernel, the Protocol, and the Baselines

Everything in this chapter runs in one Python session with NumPy alone. The simulator is the one from Chapter 2 of the sister course [Introduction to Quantum Computing](<../../FM/quantum-computing-introduction/index.html>), with the functions this chapter needs re-listed verbatim and `sample()` omitted because nothing here draws measurement outcomes; the data set is Chapter 1's, reproduced with its seed; the kernel is the fidelity kernel $k(\mathbf{x},\mathbf{x}') = |\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2$ of Chapter 3. The protocol — five-fold cross-validation on the training set to select every hyperparameter, one look at the test set at the end — is applied to every model in the chapter without exception.

One deliberate change from Chapter 3: the feature map here is the plain angle encoding with two re-uploading layers and a CNOT ring, not Chapter 3's ZZ map. The reason is stated now so that nothing later looks like a convenience. Chapter 3's map puts a *pairwise* angle $(\pi - \pi x_j)(\pi - \pi x_k)$ into each two-qubit rotation, which is quadratic in the descriptors, so the resulting kernel is not band-limited in $\mathbf{x}$ and its frequency set cannot be enumerated. The map used here is band-limited, its spectrum is finite and measurable, and §5.2 needs that to build one of the two surrogates. The other surrogate — the Pauli one — works on any feature map, Chapter 3's included.

```python
"""Chapter 5 setup, in one block: the mini-simulator functions this chapter needs,
re-listed verbatim (no sample(): nothing here draws outcomes), the dataset, the
quantum kernel of Chapter 3, and closed-form kernel ridge regression.
Nothing but NumPy; no results from earlier chapters are assumed.
"""
import numpy as np

# ---- single-qubit gates -------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta):
    e = np.exp(-1j * theta / 2)
    return np.array([[e, 0], [0, np.conj(e)]], dtype=complex)


# ---- states -------------------------------------------------------------
def ket(bits: str) -> np.ndarray:
    """'01' -> the 4-dimensional basis state |01> (big-endian)."""
    n = len(bits)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[int(bits, 2)] = 1.0
    return psi


def apply_gate(state, U, targets, n):
    """Apply the 2^k x 2^k unitary U to the listed target qubits of an n-qubit state."""
    k = len(targets)
    psi = state.reshape([2] * n)          # 1. view as an n-index tensor
    psi = np.moveaxis(psi, targets, range(k))   # 2. bring targets to the front
    rest = psi.shape[k:]
    psi = psi.reshape(2 ** k, -1)         # 3. flatten and multiply
    psi = U @ psi
    psi = psi.reshape(list((2,) * k) + list(rest))
    psi = np.moveaxis(psi, range(k), targets)   # 4. put the axes back
    return psi.reshape(-1)


CNOT4 = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=complex)


def cnot(state, control, target, n):
    """CNOT with the given control and target; any pair of qubits, any order."""
    return apply_gate(state, CNOT4, [control, target], n)


def probs(state):
    """Born-rule probabilities of all 2^n outcomes."""
    return np.abs(state) ** 2


PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}


def expval(state, pauli, coeff_map=None):
    """Expectation value of a Pauli string such as 'ZZ', 'XI' (one character per qubit).

    If coeff_map is given, the result is multiplied by coeff_map[pauli], so that a
    whole Hamiltonian is one line:  sum(expval(psi, p, terms) for p in terms).
    """
    n = len(pauli)
    phi = state.copy()
    for q, ch in enumerate(pauli):
        if ch != 'I':
            phi = apply_gate(phi, PAULI[ch], [q], n)
    val = np.vdot(state, phi).real
    if coeff_map is not None:
        val *= coeff_map.get(pauli, 1.0)
    return val


# ---- the dataset, identical in every chapter of this course --------------
def make_materials_dataset(n=60, seed=7):
    """Synthetic composition-descriptor -> formation-energy-like regression set.
    4 descriptors in [0,1]; smooth nonlinear target + mild noise. Deterministic."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (n, 4))
    y = (np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1])
         + 0.5 * X[:, 2]**2 - 0.3 * X[:, 3]
         + 0.05 * rng.standard_normal(n))
    return X, y


Xall, yall = make_materials_dataset()
Xtr, ytr = Xall[:40], yall[:40]      # train = first 40 rows
Xte, yte = Xall[40:], yall[40:]      # test  = last 20 rows

# ---- the quantum kernel of Chapter 3 ------------------------------------
K_LAYERS = 2                          # data re-uploading depth of the feature map


def feature_state(x, layers=K_LAYERS):
    """|phi(x)>: angle encoding, re-uploaded `layers` times, with a CNOT ring."""
    n = len(x)
    psi = ket('0' * n)
    for _ in range(layers):
        for q in range(n):
            psi = apply_gate(psi, ry(np.pi * x[q]), [q], n)
        for q in range(n):
            psi = cnot(psi, q, (q + 1) % n, n)
    return psi


def feature_matrix(Xs, layers=K_LAYERS):
    return np.array([feature_state(x, layers) for x in Xs])


def quantum_kernel(XA, XB, layers=K_LAYERS):
    """k(x, x') = |<phi(x)|phi(x')>|^2, the fidelity kernel of Chapter 3."""
    SA, SB = feature_matrix(XA, layers), feature_matrix(XB, layers)
    return np.abs(SA.conj() @ SB.T) ** 2


def rbf_kernel(XA, XB, gamma):
    d2 = ((XA[:, None, :] - XB[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d2)


# ---- kernel ridge regression, closed form -------------------------------
def krr_fit(K, y, lam):
    """alpha = (K + lam I)^{-1} (y - mean), with the mean handled separately."""
    mu = y.mean()
    return np.linalg.solve(K + lam * np.eye(len(y)), y - mu), mu


def krr_predict(Kcross, alpha, mu):
    return Kcross @ alpha + mu


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def cv_select(kernel_fn, X, y, folds=5, lambdas=LAMBDAS):
    """Choose lam by k-fold CV on the TRAINING set only. Returns (lam, cv_mse)."""
    K = kernel_fn(X, X)
    idx = np.arange(len(y))
    cut = np.array_split(idx, folds)
    best = (None, np.inf)
    for lam in lambdas:
        errs = []
        for f in range(folds):
            va = cut[f]
            tr = np.setdiff1d(idx, va)
            a, mu = krr_fit(K[np.ix_(tr, tr)], y[tr], lam)
            errs.append(mse(krr_predict(K[np.ix_(va, tr)], a, mu), y[va]))
        m = float(np.mean(errs))
        if m < best[1]:
            best = (lam, m)
    return best


print("Quantum kernel on the 60-point materials set")
print("-" * 76)
Kqq = quantum_kernel(Xall, Xall)
off = Kqq[~np.eye(len(Kqq), dtype=bool)]
print(f"  feature map           {K_LAYERS} re-uploading layers, 4 qubits,"
      f" state dimension {2**4}")
print(f"  diagonal              min {np.diag(Kqq).min():.6f}"
      f"  max {np.diag(Kqq).max():.6f}   (must be exactly 1)")
print(f"  off-diagonal mean     {off.mean():.6f}")
print(f"  off-diagonal std      {off.std():.6f}")
print(f"  off-diagonal range    [{off.min():.6f}, {off.max():.6f}]")
ev = np.linalg.eigvalsh(Kqq)[::-1]
print(f"  eigenvalues (top 6)   " + " ".join(f"{v:.4f}" for v in ev[:6]))
print(f"  effective rank        {ev.sum()**2 / (ev**2).sum():.2f} of {len(ev)}")

print("\nThe protocol every model in this chapter is held to")
print("-" * 88)
GAMMAS = (0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 10.0)
lam_q, cvq = cv_select(lambda a, b: quantum_kernel(a, b), Xtr, ytr)
Kt = quantum_kernel(Xtr, Xtr)
aq, muq = krr_fit(Kt, ytr, lam_q)
mse_q = mse(krr_predict(quantum_kernel(Xte, Xtr), aq, muq), yte)


def rbf_run(g):
    """Return (lam, CV MSE, test MSE) for one bandwidth, lam chosen by CV."""
    lam, cv = cv_select(lambda a, b, g=g: rbf_kernel(a, b, g), Xtr, ytr)
    a, mu = krr_fit(rbf_kernel(Xtr, Xtr, g), ytr, lam)
    return lam, cv, mse(krr_predict(rbf_kernel(Xte, Xtr, g), a, mu), yte)


rbf_table = {g: rbf_run(g) for g in GAMMAS}
g_star = min(GAMMAS, key=lambda g: rbf_table[g][1])          # honest selection
lam_r, cvr, mse_r = rbf_table[g_star]
g_orc = min(GAMMAS, key=lambda g: rbf_table[g][2])           # oracle bound
lam_o, cvo, mse_orc = rbf_table[g_orc]
A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
coef = np.linalg.lstsq(A, ytr, rcond=None)[0]
mse_ols = mse(np.hstack([Xte, np.ones((len(Xte), 1))]) @ coef, yte)
mse_mean = mse(np.full(len(yte), ytr.mean()), yte)
print(f"  {'model':<40}{'selected by 5-fold CV':>24}{'CV MSE':>9}"
      f"{'test MSE':>10}{'test RMSE':>11}")
rows = [("quantum fidelity kernel ridge", f"lam={lam_q:g}", cvq, mse_q),
        ("RBF kernel ridge", f"gamma={g_star:g}, lam={lam_r:g}", cvr, mse_r),
        ("RBF kernel ridge, best gamma on test", f"gamma={g_orc:g} (not a selection)",
         cvo, mse_orc),
        ("ordinary least squares", "-", None, mse_ols),
        ("predict the training mean", "-", None, mse_mean),
        ("irreducible noise floor", "-", None, 0.05 ** 2)]
for nm, sel, cv, te in rows:
    c = f"{cv:9.4f}" if cv is not None else f"{'-':>9}"
    print(f"  {nm:<40}{sel:>24}{c}{te:10.4f}{te**0.5:11.4f}")
print("  The third row is not a model anyone is allowed to select; it is an upper")
print("  bound on what the classical side can do on this split, and every verdict")
print("  below is checked against it as well as against the honest second row.")
print("  The RMSE column exists so these numbers can be read against the anchors of")
print("  Chapters 1 and 3: linear ridge at 0.2146 and the best classical RBF at 0.1453.")
```

```text
Quantum kernel on the 60-point materials set
----------------------------------------------------------------------------
  feature map           2 re-uploading layers, 4 qubits, state dimension 16
  diagonal              min 1.000000  max 1.000000   (must be exactly 1)
  off-diagonal mean     0.153000
  off-diagonal std      0.191101
  off-diagonal range    [0.000000, 0.977952]
  eigenvalues (top 6)   11.8973 5.7725 5.0626 3.7772 3.4988 3.0388
  effective rank        13.23 of 60

The protocol every model in this chapter is held to
----------------------------------------------------------------------------------------
  model                                      selected by 5-fold CV   CV MSE  test MSE  test RMSE
  quantum fidelity kernel ridge                           lam=0.01   0.0722    0.0680     0.2607
  RBF kernel ridge                            gamma=0.3, lam=1e-06   0.0232    0.0422     0.2055
  RBF kernel ridge, best gamma on test    gamma=2 (not a selection)   0.0308    0.0208     0.1443
  ordinary least squares                                         -        -    0.0465     0.2157
  predict the training mean                                      -        -    0.2747     0.5242
  irreducible noise floor                                        -        -    0.0025     0.0500
  The third row is not a model anyone is allowed to select; it is an upper
  bound on what the classical side can do on this split, and every verdict
  below is checked against it as well as against the honest second row.
  The RMSE column exists so these numbers can be read against the anchors of
  Chapters 1 and 3: linear ridge at 0.2146 and the best classical RBF at 0.1453.
```

**What to look for.** Four readings set up the whole chapter.

The **effective rank is 13.23 out of 60**. The quantum feature space has $2^4 = 16$ dimensions, so a 60-point Gram matrix could in principle have rank 16; the participation ratio says the spectrum is dominated by about thirteen directions. That is a small, classical object. Whatever this kernel is doing, it is not exploiting a space too large to write down.

The **quantum kernel loses to the RBF kernel**, 0.0680 against 0.0422, under a protocol that tuned both by the same cross-validation on the same folds. It loses to five-parameter linear regression at 0.0465, and it loses by a factor of 3.3 to the strongest RBF available on this split, 0.0208. That last number is worth pausing on: as an RMSE it is 0.1443, which reproduces Chapter 3's best-classical anchor of 0.1453 almost exactly, and the least-squares row's 0.2157 reproduces Chapter 1's linear-ridge anchor of 0.2146. The classical side of this chapter is the same classical side the rest of the course measured.

The **cross-validation error and the test error disagree**, and not only in magnitude — five-fold CV on 40 points selects $\gamma = 0.3$ while the test set prefers $\gamma = 2$, which is why the third row exists. The quantum kernel's CV error is 0.0722 against the CV-selected RBF's 0.0232, a factor of three, while that pair's test gap is a factor of 1.6. With 40 training points and 20 test points, neither number carries three significant figures of meaning, which is itself one of the findings of Section 5.3 and the reason that section runs 20 splits instead of one.

### There is less in the feature space than advertised

The intuition that motivates quantum kernels is that a state on $n$ qubits lives in a $2^n$-dimensional space, so an encoding into that space gives access to an exponentially rich feature map. The intuition is correct and irrelevant, and the reason is a two-line calculation.

Take two states $|\phi\rangle, |\phi'\rangle$ drawn independently at random from the uniform (Haar) measure on a $d$-dimensional space. Then $|\langle\phi|\phi'\rangle|^2$ has mean $1/d$ and standard deviation of order $1/d$. As $d = 2^n$ grows, the fidelity between any two distinct states concentrates exponentially around zero. A fidelity kernel matrix on such states is the identity plus a vanishing off-diagonal part — and kernel ridge regression on the identity matrix is a look-up table that memorizes the training set and predicts the mean everywhere else.

Real encodings are not Haar-random, so the concentration is slower. But it is still there, it is still geometric, and it is measurable.

### Code Example 2: Exponential Concentration, Measured

```python
"""Exponential concentration: what happens to the kernel as the register grows.
Continues from Example 1 (same session).
"""


def widen(Xs, n):
    """Fill an n-qubit register with the 4 descriptors, repeating them as needed."""
    reps = int(np.ceil(n / Xs.shape[1]))
    return np.tile(Xs, (1, reps))[:, :n]


print("Kernel statistics against register width (same 60 points throughout)")
print("-" * 92)
print(f"  {'n':>3}{'dim':>7}{'off-diag mean':>15}{'off-diag std':>14}"
      f"{'max off-diag':>14}{'eff. rank':>11}{'test MSE':>10}")
NS = list(range(2, 13))
stats = {}
for n in NS:
    Xw = widen(Xall, n)
    Kw = quantum_kernel(Xw, Xw)
    off = Kw[~np.eye(len(Kw), dtype=bool)]
    ev = np.linalg.eigvalsh(Kw)[::-1]
    er = ev.sum() ** 2 / (ev ** 2).sum()
    kf = lambda a, b, nn=n: quantum_kernel(a, b)
    lam, _ = cv_select(kf, Xw[:40], ytr)
    a, mu = krr_fit(Kw[:40, :40], ytr, lam)
    te = mse(krr_predict(Kw[40:, :40], a, mu), yte)
    stats[n] = (off.mean(), off.std(), off.max(), er, te)
    print(f"  {n:>3}{2**n:>7}{off.mean():15.6f}{off.std():14.6f}"
          f"{off.max():14.6f}{er:11.2f}{te:10.4f}")

print("\nThe decay is geometric, and 2^-n is the reference line")
print("-" * 92)
print(f"  {'n':>3}{'mean':>13}{'2^-n':>13}{'mean/2^-n':>12}"
      f"{'std':>13}{'std ratio n-1 -> n':>21}")
prev = None
for n in NS:
    m, s, _, _, _ = stats[n]
    r = "-" if prev is None else f"{s/prev:.3f}"
    print(f"  {n:>3}{m:13.6f}{2.0**-n:13.6f}{m/2.0**-n:12.3f}{s:13.6f}{r:>21}")
    prev = s

print("\nWhat the concentration costs in shots")
print("-" * 92)
print("  A fidelity kernel entry is estimated from a Bernoulli experiment: the count")
print("  of all-zero strings in an inversion test. One shot therefore has variance")
print("  k(1-k), NOT 1, and S = k(1-k)/eps^2. Resolving structure at a tenth of the")
print("  spread means eps = sigma/10, which is Chapter 3's convention, so")
print("  S = 100 k(1-k)/sigma^2. A Gram matrix of N points needs N(N-1)/2 entries.")
print(f"  {'n':>3}{'mean k':>12}{'std of entries':>16}{'shots per entry':>18}"
      f"{'shots for 60x60 Gram':>22}")
for n in NS:
    m, s = stats[n][0], stats[n][1]
    per = 100.0 * m * (1.0 - m) / s ** 2
    print(f"  {n:>3}{m:12.6f}{s:16.6f}{per:18,.0f}{per * 60 * 59 / 2:22,.0f}")
print("  The column is nearly FLAT, and that is a fact about this encoding rather")
print("  than about fidelity kernels: k falls faster than sigma here, so the numerator")
print("  shrinks with the denominator. Quoting 1/sigma^2 instead would have given 190")
print("  shots at n = 12 and, extrapolated, an exponential; Exercise 1 shows what the")
print("  cost does become once the map is expressive enough for k ~ sigma ~ 2^-n.")

print("\nThe honest reading")
print("-" * 92)
print("  Nothing here is a hardware limitation. The simulator is exact and noiseless.")
print("  The kernel flattens because a fidelity between two states in a 2^n-dimensional")
print("  space is typically of order 2^-n, and a kernel matrix that is the identity plus")
print("  a vanishing off-diagonal part carries no information about the data: kernel")
print("  ridge regression on it can only reproduce the training mean.")
print("  Note the direction of the test-MSE column. It gets worse, not better, as the")
print("  register grows -- the opposite of the 'exponentially large feature space'")
print("  intuition that motivates quantum kernels in the first place.")
```

```text
Kernel statistics against register width (same 60 points throughout)
--------------------------------------------------------------------------------------------
    n    dim  off-diag mean  off-diag std  max off-diag  eff. rank  test MSE
    2      4       0.359838      0.309325      0.998156       4.20    0.0362
    3      8       0.246025      0.262934      0.986398       6.94    0.0284
    4     16       0.153000      0.191101      0.977952      13.23    0.0680
    5     32       0.109725      0.167800      0.974733      17.80    0.0628
    6     64       0.077986      0.137245      0.880540      24.29    0.0831
    7    128       0.056518      0.119306      0.867474      29.58    0.0838
    8    256       0.044784      0.105990      0.857991      33.69    0.1064
    9    512       0.036139      0.094839      0.856278      37.32    0.1235
   10   1024       0.029538      0.085575      0.773427      40.44    0.1243
   11   2048       0.024784      0.078636      0.767840      42.82    0.1301
   12   4096       0.021315      0.072502      0.759562      44.88    0.1431

The decay is geometric, and 2^-n is the reference line
--------------------------------------------------------------------------------------------
    n         mean         2^-n   mean/2^-n          std   std ratio n-1 -> n
    2     0.359838     0.250000       1.439     0.309325                    -
    3     0.246025     0.125000       1.968     0.262934                0.850
    4     0.153000     0.062500       2.448     0.191101                0.727
    5     0.109725     0.031250       3.511     0.167800                0.878
    6     0.077986     0.015625       4.991     0.137245                0.818
    7     0.056518     0.007812       7.234     0.119306                0.869
    8     0.044784     0.003906      11.465     0.105990                0.888
    9     0.036139     0.001953      18.503     0.094839                0.895
   10     0.029538     0.000977      30.246     0.085575                0.902
   11     0.024784     0.000488      50.757     0.078636                0.919
   12     0.021315     0.000244      87.308     0.072502                0.922

What the concentration costs in shots
--------------------------------------------------------------------------------------------
  A fidelity kernel entry is estimated from a Bernoulli experiment: the count
  of all-zero strings in an inversion test. One shot therefore has variance
  k(1-k), NOT 1, and S = k(1-k)/eps^2. Resolving structure at a tenth of the
  spread means eps = sigma/10, which is Chapter 3's convention, so
  S = 100 k(1-k)/sigma^2. A Gram matrix of N points needs N(N-1)/2 entries.
    n      mean k  std of entries   shots per entry  shots for 60x60 Gram
    2    0.359838        0.309325               241               426,127
    3    0.246025        0.262934               268               474,916
    4    0.153000        0.191101               355               628,088
    5    0.109725        0.167800               347               614,066
    6    0.077986        0.137245               382               675,667
    7    0.056518        0.119306               375               663,082
    8    0.044784        0.105990               381               674,010
    9    0.036139        0.094839               387               685,470
   10    0.029538        0.085575               391               692,844
   11    0.024784        0.078636               391               691,819
   12    0.021315        0.072502               397               702,448
  The column is nearly FLAT, and that is a fact about this encoding rather
  than about fidelity kernels: k falls faster than sigma here, so the numerator
  shrinks with the denominator. Quoting 1/sigma^2 instead would have given 190
  shots at n = 12 and, extrapolated, an exponential; Exercise 1 shows what the
  cost does become once the map is expressive enough for k ~ sigma ~ 2^-n.

The honest reading
--------------------------------------------------------------------------------------------
  Nothing here is a hardware limitation. The simulator is exact and noiseless.
  The kernel flattens because a fidelity between two states in a 2^n-dimensional
  space is typically of order 2^-n, and a kernel matrix that is the identity plus
  a vanishing off-diagonal part carries no information about the data: kernel
  ridge regression on it can only reproduce the training mean.
  Note the direction of the test-MSE column. It gets worse, not better, as the
  register grows -- the opposite of the 'exponentially large feature space'
  intuition that motivates quantum kernels in the first place.
```

**What to look for.** The off-diagonal mean falls from 0.360 at two qubits to 0.021 at twelve, and the standard deviation from 0.309 to 0.073 — a factor between 0.73 and 0.92 per added qubit and tending towards 0.9, slower than the Haar prediction of 0.5 because a two-layer encoding is far from a random circuit, but geometric all the same. Exercise 1 works out the Haar comparison.

The column that matters is the last one. **The test error gets worse as the register grows**: 0.036 at two qubits, 0.143 at twelve. This is the opposite of what the exponential-feature-space intuition predicts, and it is not a subtle effect — it is a factor of four over ten qubits, on an exact simulator, with the hyperparameter re-tuned at every width. Adding qubits does not add model capacity in any useful sense. It adds distance between every pair of data points until the kernel can no longer tell them apart.

The shot column converts this into an experimental cost, and it has to be computed correctly or it says the opposite of the truth. A fidelity kernel entry is estimated as the probability of an all-zero outcome in an inversion test, so a single shot is a *bit* with variance $k(1-k)$, not a bounded observable with variance one. Resolving structure at a tenth of the spread therefore costs $S = 100\,k(1-k)/\sigma^2$ shots per entry, and on this encoding that column is essentially flat in $n$: about 240 at two qubits and 400 at twelve, because $k$ falls faster than $\sigma$ does and the numerator shrinks along with the denominator. A 60-point Gram matrix needs 1770 entries, so about $7\times10^5$ shots at twelve qubits — and that number does not grow appreciably with the register.

The naive $1/\sigma^2$ would have read 190 shots at twelve qubits and would have grown geometrically, which is a more dramatic story and the wrong one. Where the exponential shot cost genuinely lives is the regime this encoding never reaches: once the feature map is expressive enough that $k$ and $\sigma$ both fall as $2^{-n}$, the ratio $k(1-k)/\sigma^2$ is itself of order $2^{n}$, and *then* a 30-qubit fidelity kernel is unmeasurable for reasons that have nothing to do with error rates. Exercise 1 does that arithmetic against the Haar prediction, and Chapter 3's Section 3.6 measures the exponent on a map that is in the regime. The honest summary of *this* table is narrower than the one usually offered: the register width destroys the model's accuracy long before it destroys the measurement budget.

* * *

## 5.2 Two Surrogates, Built and Measured

A classical surrogate is a classical model built to imitate a specific quantum model. There is no general recipe, but there are two constructions that apply to almost any kernel of the form used in this course, and they attack from opposite directions.

**The Pauli (classical shadow) surrogate** exploits the fact that a fidelity kernel is an inner product of density matrices. Writing $\rho(\mathbf{x}) = |\phi(\mathbf{x})\rangle\langle\phi(\mathbf{x})|$ in the Pauli basis,

$$
\rho(\mathbf{x}) = \frac{1}{2^n}\sum_{P} c_P(\mathbf{x})\,P,\qquad c_P(\mathbf{x}) = \mathrm{Tr}\bigl[\rho(\mathbf{x})P\bigr] = \langle\phi(\mathbf{x})|P|\phi(\mathbf{x})\rangle
$$

and therefore

$$
k(\mathbf{x},\mathbf{x}') = \bigl|\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle\bigr|^2 = \mathrm{Tr}\bigl[\rho(\mathbf{x})\rho(\mathbf{x}')\bigr] = \frac{1}{2^n}\sum_P c_P(\mathbf{x})\,c_P(\mathbf{x}')
$$

The sum has $4^n$ terms, which is the exponential cost. Truncate it at Pauli weight $w$ — keeping only strings with at most $w$ non-identity factors — and the feature vector has $\sum_{k\le w}\binom{n}{k}3^k$ components, which is $O(n^w)$. Each surviving component is the expectation of a low-weight Pauli operator, and that is precisely the quantity a *classical shadow* estimates from randomized measurements with a shot count independent of $n$. So this surrogate is not a mathematical abstraction: it is a description of a measurement protocol, and once the features are in hand the model is a classical ridge regression.

**The Fourier surrogate** exploits the structure of the encoding instead. Chapter 2 established the rule: $L$ encoding gates whose generator is a Pauli over two — $Z/2$ there, $Y/2$ here, and only the eigenvalue spacing matters — give the frequency set $\Omega = \lbrace -L,\ldots,L\rbrace$ in the rotation angle, so expectation values are trigonometric polynomials in the descriptors with a *finite* set of frequencies — here in multiples of $1/2$ cycle per unit descriptor, because the angle is $\pi x_q$. If that set can be enumerated, a classical model carrying exactly those basis functions spans the same space of functions as the quantum model, and it never touches a quantum device at all — not in training, not at inference, not to build the features.

That last sentence needs a bound on "enumerated", and the bound is the same one the Pauli surrogate carries. With two layers each descriptor contributes five band-limited functions — $1$, $\cos\pi x_q$, $\sin\pi x_q$, $\cos 2\pi x_q$, $\sin 2\pi x_q$ — so the *full* band-limited product basis on $d = 4$ descriptors has $5^4 = 625$ members, and on $d$ descriptors it has $5^d$: exponential in the descriptor count, exactly as $4^n$ is exponential in the qubit count. The surrogate built below keeps 41 of those 625, namely the products of *total degree at most two*. That truncation is not free and it is not neutral: it is a **bet that the target has no high-order interactions between descriptors**, the same bet the weight-$w$ Pauli truncation makes about high-weight correlations. On this target the bet wins, and Code Example 4 shows by how much. On a target built from a four-way interaction it would lose, and the honest statement of the method is "enumerate the band-limited basis and truncate it by interaction order", not "enumerate the band-limited basis".

### Code Example 3: The Pauli Surrogate

```python
"""A classical surrogate built from the kernel's own structure: low-weight Paulis.
Continues from Example 1 (same session).
"""
from itertools import product


def pauli_strings(n, max_weight):
    """All Pauli strings on n qubits with at most `max_weight` non-identity factors."""
    out = []
    for letters in product('IXYZ', repeat=n):
        if sum(c != 'I' for c in letters) <= max_weight:
            out.append(''.join(letters))
    return out


def shadow_features(Xs, max_weight, n=4, layers=K_LAYERS):
    """Phi(x)_P = <phi(x)| P |phi(x)> / 2^(n/2) for every Pauli P of low weight.

    The full set of 4^n Paulis reproduces the fidelity kernel exactly, because
    rho = 2^-n sum_P <P> P and k(x,x') = Tr[rho(x) rho(x')]. Truncating the sum at
    weight w leaves a classical feature map of size O(n^w) -- and each feature is
    exactly what a classical shadow of the state estimates.
    """
    ps = pauli_strings(n, max_weight)
    S = feature_matrix(Xs, layers)
    Phi = np.empty((len(Xs), len(ps)))
    for i, psi in enumerate(S):
        Phi[i] = [expval(psi, p) for p in ps]
    return Phi / np.sqrt(2.0 ** n), ps


Kq_all = quantum_kernel(Xall, Xall)
print("Truncating the exact Pauli expansion of the quantum kernel (4 qubits)")
print("-" * 90)
off_mask = ~np.eye(len(Kq_all), dtype=bool)
print(f"  {'max weight':>11}{'features':>10}{'rel. error (all)':>18}"
      f"{'rel. error (off-diag)':>23}{'alignment':>12}")
surrogates = {}
for w in range(0, 5):
    Phi, ps = shadow_features(Xall, w)
    Kw = Phi @ Phi.T
    err = np.linalg.norm(Kq_all - Kw) / np.linalg.norm(Kq_all)
    erro = (np.linalg.norm(Kq_all[off_mask] - Kw[off_mask])
            / np.linalg.norm(Kq_all[off_mask]))
    align = (np.sum(Kq_all * Kw)
             / (np.linalg.norm(Kq_all) * np.linalg.norm(Kw)))
    surrogates[w] = Phi
    print(f"  {w:>11}{len(ps):>10}{err:18.6f}{erro:23.6f}{align:12.6f}")
print("  Weight 4 is the whole expansion, so the last row is an identity check: the")
print("  surrogate and the quantum kernel agree to machine precision. Note that the")
print("  all-entries column is dominated by the diagonal, which truncation always")
print("  shrinks below 1; the off-diagonal column is the one that carries the data.")

print("\nWhat each truncation is worth on the actual task")
print("-" * 90)


def ridge_features(Phi, ytr_, lam):
    """Primal ridge on an explicit feature map: w = (Phi'Phi + lam I)^-1 Phi'(y-mu)."""
    mu = ytr_.mean()
    G = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    return np.linalg.solve(G, Phi.T @ (ytr_ - mu)), mu


def cv_features(Phi, y, folds=5, lambdas=LAMBDAS):
    idx = np.arange(len(y))
    cut = np.array_split(idx, folds)
    best = (None, np.inf)
    for lam in lambdas:
        errs = []
        for f in range(folds):
            va = cut[f]
            tr = np.setdiff1d(idx, va)
            wv, mu = ridge_features(Phi[tr], y[tr], lam)
            errs.append(mse(Phi[va] @ wv + mu, y[va]))
        m = float(np.mean(errs))
        if m < best[1]:
            best = (lam, m)
    return best


print(f"  {'model':<38}{'features':>10}{'lam':>9}{'CV MSE':>10}{'test MSE':>11}")
for w in range(0, 5):
    Phi = surrogates[w]
    lam, cv = cv_features(Phi[:40], ytr)
    wv, mu = ridge_features(Phi[:40], ytr, lam)
    te = mse(Phi[40:] @ wv + mu, yte)
    print(f"  {'Pauli surrogate, weight <= ' + str(w):<38}"
          f"{Phi.shape[1]:>10}{lam:9g}{cv:10.4f}{te:11.4f}")
print(f"  {'quantum fidelity kernel ridge':<38}{'2^4 dim':>10}{lam_q:9g}"
      f"{cvq:10.4f}{mse_q:11.4f}")
print(f"  {'RBF kernel ridge (tuned)':<38}{'-':>10}{lam_r:9g}{cvr:10.4f}"
      f"{mse_r:11.4f}")

print("\nThe dequantization argument, in one paragraph")
print("-" * 90)
print("  Weight <= 2 needs 1 + 3n + 9 n(n-1)/2 features, which is O(n^2), not 4^n. On")
print("  four qubits that is 67 numbers per data point. It does NOT reproduce the Gram")
print("  matrix closely -- the off-diagonal relative error is still tens of per cent --")
print("  and yet its test error already matches the full quantum kernel, and weight 3")
print("  beats it. Predictive performance is not the same quantity as kernel fidelity,")
print("  and it is predictive performance that an advantage claim is about.")
print("  Each of those 67 numbers is the expectation of a")
print("  low-weight Pauli, which is exactly what a classical shadow measures with a")
print("  number of shots independent of n. So the model can be built, trained and")
print("  deployed classically once the features are in hand -- and if the features")
print("  themselves are computable classically, the quantum computer never enters.")
print("  This is what 'dequantization' means in practice: not a proof that quantum")
print("  models are useless, but a demonstration that a specific quantum model's")
print("  advantage evaporates once you write down its classical shadow.")
```

```text
Truncating the exact Pauli expansion of the quantum kernel (4 qubits)
------------------------------------------------------------------------------------------
   max weight  features  rel. error (all)  rel. error (off-diag)   alignment
            0         1          0.880535               0.863743    0.607811
            1        13          0.834832               0.819586    0.689051
            2        67          0.673329               0.666768    0.822302
            3       175          0.311714               0.318940    0.968583
            4       256          0.000000               0.000000    1.000000
  Weight 4 is the whole expansion, so the last row is an identity check: the
  surrogate and the quantum kernel agree to machine precision. Note that the
  all-entries column is dominated by the diagonal, which truncation always
  shrinks below 1; the off-diagonal column is the one that carries the data.

What each truncation is worth on the actual task
------------------------------------------------------------------------------------------
  model                                   features      lam    CV MSE   test MSE
  Pauli surrogate, weight <= 0                   1    1e-06    0.3167     0.2747
  Pauli surrogate, weight <= 1                  13      0.1    0.2089     0.2655
  Pauli surrogate, weight <= 2                  67      0.1    0.1058     0.0642
  Pauli surrogate, weight <= 3                 175     0.01    0.0768     0.0379
  Pauli surrogate, weight <= 4                 256     0.01    0.0722     0.0680
  quantum fidelity kernel ridge            2^4 dim     0.01    0.0722     0.0680
  RBF kernel ridge (tuned)                       -    1e-06    0.0232     0.0422

The dequantization argument, in one paragraph
------------------------------------------------------------------------------------------
  Weight <= 2 needs 1 + 3n + 9 n(n-1)/2 features, which is O(n^2), not 4^n. On
  four qubits that is 67 numbers per data point. It does NOT reproduce the Gram
  matrix closely -- the off-diagonal relative error is still tens of per cent --
  and yet its test error already matches the full quantum kernel, and weight 3
  beats it. Predictive performance is not the same quantity as kernel fidelity,
  and it is predictive performance that an advantage claim is about.
  Each of those 67 numbers is the expectation of a
  low-weight Pauli, which is exactly what a classical shadow measures with a
  number of shots independent of n. So the model can be built, trained and
  deployed classically once the features are in hand -- and if the features
  themselves are computable classically, the quantum computer never enters.
  This is what 'dequantization' means in practice: not a proof that quantum
  models are useless, but a demonstration that a specific quantum model's
  advantage evaporates once you write down its classical shadow.
```

**What to look for.** The weight-4 row is an identity check: keeping all 256 Paulis reproduces the kernel to machine precision and gives exactly the quantum kernel's test error, 0.0680, confirming the expansion is right.

Above that, the numbers separate two things that are easy to conflate. **Kernel fidelity and predictive performance are different quantities.** The weight-2 surrogate reproduces the Gram matrix badly — a relative error of 67% on the off-diagonal entries, an alignment of 0.822 — and yet its test error, 0.0642, is already slightly *better* than the full quantum kernel's 0.0680. The weight-3 surrogate reaches 0.0379, comfortably beating it. Truncation is not only a cost saving here; it is a regularizer, removing the high-weight directions in which the fidelity kernel puts weight it cannot support with 40 data points.

For an advantage claim, only the predictive column is relevant. An argument of the form "our kernel cannot be approximated classically to within $\epsilon$ in Frobenius norm" says nothing about whether a classical model predicts as well, and the two are not monotonically related.

The feature count is the other half of the argument: 67 features at four qubits, and $1 + 3n + 9n(n-1)/2$ at $n$ qubits, which is 11,176 at $n = 50$ against $4^{50} \approx 1.3\times10^{30}$. Exercise 2 tabulates it.

### Code Example 4: The Fourier Surrogate, Which Needs No Quantum Computer

```python
"""A surrogate that never touches a quantum computer: truncated Fourier features.
Continues from Examples 1 and 3 (same session).
"""

print("First, what frequencies the quantum kernel actually contains")
print("-" * 90)
xref = np.array([0.31, 0.62, 0.17, 0.85])
grid = np.arange(64) / 64.0 * 2.0                # two periods of x1 -> period 2
Kline = np.array([quantum_kernel(np.array([[t, xref[1], xref[2], xref[3]]]),
                                 xref[None, :])[0, 0] for t in grid])
amp = np.abs(np.fft.rfft(Kline)) / len(grid)
print(f"  k(x, x_ref) as a function of x1 alone, sampled on 64 points over x1 in [0,2)")
print(f"  {'harmonic (cycles per unit x1)':<34}{'amplitude':>12}")
for j in range(0, 7):
    print(f"  {j/2.0:<34.1f}{amp[j]:12.6f}")
print(f"  total power above harmonic 1.0: {np.sum(amp[3:]**2)**0.5:.3e}")
print("  The spectrum stops, exactly. Two re-uploading layers of Ry(pi x) produce")
print("  harmonics at 0, 1/2 and 1 cycle per unit of x1 and nothing beyond, so a")
print("  classical model carrying exactly those basis functions spans the same space")
print("  of functions -- this is Chapter 2's frequency argument turned into an attack.")
print("  Five functions per descriptor (1, cos, sin, cos2, sin2) means the FULL")
print(f"  band-limited product basis has 5^4 = {5**4} members, and 5^d in general.")
print("  The surrogate below keeps only total degree <= 2, which is 41 of those 625:")
print("  a bet that the target has no high-order interactions, not a free lunch.")


def fourier_features(Xs, dmax, n=4):
    """cos/sin basis with total degree <= dmax; frequencies j * pi * x_q."""
    cols = [np.ones(len(Xs))]
    names = ['1']

    def rec(q, degs):
        if q == n:
            if sum(degs) == 0:
                return
            for signs in product((0, 1), repeat=n):
                if any(s and d == 0 for s, d in zip(signs, degs)):
                    continue
                col = np.ones(len(Xs))
                nm = []
                for qq, (d, s) in enumerate(zip(degs, signs)):
                    if d == 0:
                        continue
                    ang = d * np.pi * Xs[:, qq]
                    col = col * (np.sin(ang) if s else np.cos(ang))
                    nm.append(('sin' if s else 'cos') + f'({d}pi x{qq+1})')
                cols.append(col)
                names.append('*'.join(nm))
            return
        for d in range(0, dmax + 1):
            if sum(degs) + d <= dmax:
                rec(q + 1, degs + [d])

    rec(0, [])
    return np.column_stack(cols), names


print("\nThe surrogate against the model it is imitating")
print("-" * 90)
print(f"  {'model':<40}{'features':>10}{'lam':>9}{'CV MSE':>10}"
      f"{'test MSE':>11}{'align':>8}")
for dmax in (1, 2, 3):
    Phi, names = fourier_features(Xall, dmax)
    Kf = Phi @ Phi.T / Phi.shape[1]
    align = (np.sum(Kq_all * Kf)
             / (np.linalg.norm(Kq_all) * np.linalg.norm(Kf)))
    lam, cv = cv_features(Phi[:40], ytr)
    wv, mu = ridge_features(Phi[:40], ytr, lam)
    te = mse(Phi[40:] @ wv + mu, yte)
    print(f"  {'Fourier surrogate, total degree <= ' + str(dmax):<40}"
          f"{Phi.shape[1]:>10}{lam:9g}{cv:10.4f}{te:11.4f}{align:8.3f}")
Phi2, _ = shadow_features(Xall, 2)
lam2, cv2 = cv_features(Phi2[:40], ytr)
w2, mu2 = ridge_features(Phi2[:40], ytr, lam2)
te2 = mse(Phi2[40:] @ w2 + mu2, yte)
print(f"  {'Pauli surrogate, weight <= 2':<40}{Phi2.shape[1]:>10}{lam2:9g}"
      f"{cv2:10.4f}{te2:11.4f}{'-':>8}")
print(f"  {'quantum fidelity kernel ridge':<40}{'2^4 dim':>10}{lam_q:9g}"
      f"{cvq:10.4f}{mse_q:11.4f}{1.0:8.3f}")
print(f"  {'RBF kernel ridge (tuned)':<40}{'-':>10}{lam_r:9g}{cvr:10.4f}"
      f"{mse_r:11.4f}{'-':>8}")
print(f"  {'ordinary least squares':<40}{'5':>10}{'-':>9}{'-':>10}"
      f"{mse_ols:11.4f}{'-':>8}")

print("\nThe gap, stated as a number")
print("-" * 90)
Phi, _ = fourier_features(Xall, 2)
lam, _ = cv_features(Phi[:40], ytr)
wv, mu = ridge_features(Phi[:40], ytr, lam)
pred_f = Phi[40:] @ wv + mu
te_f = mse(pred_f, yte)
print(f"  quantum kernel ridge                     test MSE {mse_q:.4f}")
print(f"  degree-2 Fourier surrogate, no quantum    test MSE {te_f:.4f}"
      f"   ({te_f/mse_q:.2f}x)")
print(f"  weight-2 Pauli surrogate                  test MSE {te2:.4f}"
      f"   ({te2/mse_q:.2f}x)")
print(f"  tuned RBF                                 test MSE {mse_r:.4f}"
      f"   ({mse_r/mse_q:.2f}x)")


# ---- R6: those ratios are point estimates on 20 rows. Pair them. ------------
def paired_bootstrap_mse(y_true, pred_a, pred_b, B=10000, seed=0, alpha=0.05):
    """95% interval for MSE(a) - MSE(b), resampling the SAME test rows for both.

    Chapter 1's R6. Every ratio printed above is a ratio of two point estimates
    measured on twenty rows; this is the statistic that says whether it is real.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pa, pb = np.asarray(pred_a), np.asarray(pred_b)
    m = len(y_true)
    d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, m, m)
        d[b] = (np.mean((y_true[i] - pa[i]) ** 2)
                - np.mean((y_true[i] - pb[i]) ** 2))
    return (float(d.mean()), float(np.quantile(d, alpha / 2)),
            float(np.quantile(d, 1 - alpha / 2)))


aq2, muq2 = krr_fit(quantum_kernel(Xtr, Xtr), ytr, lam_q)
pred_q = krr_predict(quantum_kernel(Xte, Xtr), aq2, muq2)
ar2, mur2 = krr_fit(rbf_kernel(Xtr, Xtr, g_star), ytr, lam_r)
pred_r = krr_predict(rbf_kernel(Xte, Xtr, g_star), ar2, mur2)
pred_p = Phi2[40:] @ w2 + mu2
P = {'quantum fidelity kernel': pred_q, 'Fourier surrogate d<=2': pred_f,
     'Pauli surrogate w<=2': pred_p, 'tuned RBF': pred_r}

print("\n  R6: the same gaps with a paired bootstrap interval on the same 20 rows")
print(f"  {'MSE(A) - MSE(B)':<50}{'mean':>10}{'95% interval':>21}{'verdict':>10}")
for a, b in [('Fourier surrogate d<=2', 'quantum fidelity kernel'),
             ('Pauli surrogate w<=2', 'quantum fidelity kernel'),
             ('tuned RBF', 'quantum fidelity kernel'),
             ('Fourier surrogate d<=2', 'tuned RBF')]:
    m_, lo, hi = paired_bootstrap_mse(yte, P[a], P[b])
    v = "A better" if hi < 0.0 else ("B better" if lo > 0.0 else "no call")
    print(f"  {a + '  -  ' + b:<50}{m_:+10.4f}   [{lo:+.4f}, {hi:+.4f}]{v:>10}")
print("  A surrogate that reproduces the quantum model to within the width of the")
print("  experiment is a surrogate that removes the reason to run the quantum model.")
print("  Nothing about this argument needs a hardware improvement to be answered; it")
print("  needs a problem whose kernel has no small classical description.")
```

```text
First, what frequencies the quantum kernel actually contains
------------------------------------------------------------------------------------------
  k(x, x_ref) as a function of x1 alone, sampled on 64 points over x1 in [0,2)
  harmonic (cycles per unit x1)        amplitude
  0.0                                   0.375000
  0.5                                   0.250000
  1.0                                   0.062500
  1.5                                   0.000000
  2.0                                   0.000000
  2.5                                   0.000000
  3.0                                   0.000000
  total power above harmonic 1.0: 1.227e-16
  The spectrum stops, exactly. Two re-uploading layers of Ry(pi x) produce
  harmonics at 0, 1/2 and 1 cycle per unit of x1 and nothing beyond, so a
  classical model carrying exactly those basis functions spans the same space
  of functions -- this is Chapter 2's frequency argument turned into an attack.
  Five functions per descriptor (1, cos, sin, cos2, sin2) means the FULL
  band-limited product basis has 5^4 = 625 members, and 5^d in general.
  The surrogate below keeps only total degree <= 2, which is 41 of those 625:
  a bet that the target has no high-order interactions, not a free lunch.

The surrogate against the model it is imitating
------------------------------------------------------------------------------------------
  model                                     features      lam    CV MSE   test MSE   align
  Fourier surrogate, total degree <= 1             9        1    0.0841     0.0546   0.759
  Fourier surrogate, total degree <= 2            41     0.01    0.0236     0.0172   0.846
  Fourier surrogate, total degree <= 3           129    1e-06    0.0337     0.0363   0.844
  Pauli surrogate, weight <= 2                    67      0.1    0.1058     0.0642       -
  quantum fidelity kernel ridge              2^4 dim     0.01    0.0722     0.0680   1.000
  RBF kernel ridge (tuned)                         -    1e-06    0.0232     0.0422       -
  ordinary least squares                           5        -         -     0.0465       -

The gap, stated as a number
------------------------------------------------------------------------------------------
  quantum kernel ridge                     test MSE 0.0680
  degree-2 Fourier surrogate, no quantum    test MSE 0.0172   (0.25x)
  weight-2 Pauli surrogate                  test MSE 0.0642   (0.94x)
  tuned RBF                                 test MSE 0.0422   (0.62x)

  R6: the same gaps with a paired bootstrap interval on the same 20 rows
  MSE(A) - MSE(B)                                         mean         95% interval   verdict
  Fourier surrogate d<=2  -  quantum fidelity kernel   -0.0507   [-0.1127, -0.0107]  A better
  Pauli surrogate w<=2  -  quantum fidelity kernel     -0.0039   [-0.0464, +0.0387]   no call
  tuned RBF  -  quantum fidelity kernel                -0.0257   [-0.0976, +0.0362]   no call
  Fourier surrogate d<=2  -  tuned RBF                 -0.0250   [-0.0613, +0.0015]   no call
  A surrogate that reproduces the quantum model to within the width of the
  experiment is a surrogate that removes the reason to run the quantum model.
  Nothing about this argument needs a hardware improvement to be answered; it
  needs a problem whose kernel has no small classical description.
```

**What to look for.** The spectrum check comes first and it is exact: sampling $k(\mathbf{x},\mathbf{x}_\mathrm{ref})$ along one descriptor and taking an FFT gives amplitudes 0.375, 0.250 and 0.0625 at harmonics 0, 1/2 and 1 cycle per unit, and $1.2\times10^{-16}$ of power above that. The kernel is a trigonometric polynomial with three terms per descriptor. Nothing is hidden in high frequencies, because there are no high frequencies.

Given that, the surrogate is a two-line construction: enumerate $\cos(j\pi x_q)$ and $\sin(j\pi x_q)$ up to total degree 2, fit ridge regression with $\lambda$ chosen by the same cross-validation as everything else. Forty-one features, no quantum device, test MSE **0.0172** against the quantum kernel's **0.0680** — a factor of four *better*, and better than the tuned RBF's 0.0422 as well. The total degree is itself cross-validated rather than chosen: degree 1 scores 0.0841 in CV and degree 3 scores 0.0337, against degree 2's 0.0236, so the winning row is the one the training data selects.

Two facts limit what that number means, and both belong here rather than in a caveat at the end. The first is the interaction-order bet: 41 features is a total-degree-$\le 2$ slice of the $5^4 = 625$-member band-limited basis, and the full basis grows as $5^d$ in the descriptor count. This surrogate is cheap because the target happens to be dominated by pairwise structure — which it is, by construction, since the generator is $\sin(\pi x_0)\cos(\pi x_1) + \tfrac12 x_2^2 - 0.3 x_3$. The second is the R6 point: 0.0172 against 0.0680 is a ratio of two point estimates on 20 test rows, and the paired interval printed with it is what decides whether the gap is real. Here it survives — the interval on MSE(Fourier) $-$ MSE(quantum) is $[-0.113, -0.011]$, entirely below zero — and that is the chapter's load-bearing result. Two of the neighbouring comparisons do *not* survive, and it is worth saying so in the same breath: the weight-2 Pauli surrogate against the quantum kernel is $[-0.046, +0.039]$, a tie, and the tuned RBF against the quantum kernel is $[-0.098, +0.036]$, so the $1.61\times$ gap that Code Example 1 reported is **not** resolvable on this single split. That last interval is precisely why Section 5.3 runs twenty splits instead of one, and why its verdict rests on winning all twenty rather than on the size of the gap.

The result should also be read precisely in the other direction, because it is easy to over- and under-claim. It does **not** show that quantum kernels are useless. It shows that *this* quantum kernel, on *this* data, had a function class that a classical model could enumerate and use more efficiently — and that the enumeration was easy because the encoding was shallow, the descriptors few, and the interactions low-order. The general statement in the direction of pessimism is the one Chapter 4 arrived at from the other side: a circuit shallow enough to train, and local enough to have non-vanishing gradients, tends also to be simple enough to write down classically. That tension, not any hardware limit, is the central difficulty of the field.

### What a surrogate does and does not prove

  * **It does prove** that a particular reported advantage is not an advantage: if the surrogate matches, the quantum resource bought nothing on that problem.
  * **It does not prove** that no quantum model can help on that problem. A different encoding may induce a kernel with no small classical description.
  * **It does not scale automatically.** The Fourier construction needs the frequency support to be enumerable, which fails for deep circuits and also for Chapter 3's ZZ map, whose pairwise angle is quadratic in the descriptors and therefore not band-limited. Even when it is enumerable it grows as $5^d$ in the number of descriptors for this two-layer map, so what is actually built is a truncation by interaction order — 41 features out of 625 here — and that truncation fails on a target with genuine high-order interactions. The Pauli construction has the matching limitation in the other variable: it needs low-weight features to carry the signal, and its count grows as $O(n^w)$ in the qubit number. Both are bets on low-order structure, both apply to any feature map, and both failure modes are the interesting case — which is also where Chapter 4's trainability cost bites.
  * **It is cheap, and therefore obligatory.** Building both surrogates for the models in this course took a few hundred lines and under a second of compute. Any claim of quantum advantage on classical data that has not been checked against them has not been checked.

* * *

## 5.3 How to Read a Quantum-Advantage Claim

Two models, one data set, and a great deal of freedom in what to report. This section spends that freedom deliberately, to show what it buys.

### Code Example 5: Three True Summaries of One Experiment

```python
"""Benchmark hygiene, made quantitative: how an advantage claim gets manufactured.
Continues from Examples 1, 3 and 4 (same session).
"""
print("Hygiene item 1 -- one split is not a result")
print("-" * 92)
print("  20 random 40/20 splits of the same 60 points. Both models re-tune lambda by")
print("  5-fold CV inside each split's training set. Nothing else changes.")
diffs, qs, rs = [], [], []
for rep in range(20):
    perm = np.random.default_rng(500 + rep).permutation(len(yall))
    tr, te = perm[:40], perm[40:]
    Xa, ya, Xb, yb = Xall[tr], yall[tr], Xall[te], yall[te]
    lq, _ = cv_select(lambda a, b: quantum_kernel(a, b), Xa, ya)
    aq_, mq_ = krr_fit(quantum_kernel(Xa, Xa), ya, lq)
    eq = mse(krr_predict(quantum_kernel(Xb, Xa), aq_, mq_), yb)
    best = (None, np.inf)
    for g in GAMMAS:
        lr_, c_ = cv_select(lambda a, b, g=g: rbf_kernel(a, b, g), Xa, ya)
        if c_ < best[1]:
            best = ((g, lr_), c_)
    (gs, ls), _ = best
    ar_, mr_ = krr_fit(rbf_kernel(Xa, Xa, gs), ya, ls)
    er = mse(krr_predict(rbf_kernel(Xb, Xa, gs), ar_, mr_), yb)
    qs.append(eq); rs.append(er); diffs.append(eq - er)
qs, rs, diffs = np.array(qs), np.array(rs), np.array(diffs)
print(f"  {'statistic':<44}{'quantum':>12}{'RBF':>12}")
print(f"  {'mean test MSE over 20 splits':<44}{qs.mean():12.4f}{rs.mean():12.4f}")
print(f"  {'median test MSE':<44}{np.median(qs):12.4f}{np.median(rs):12.4f}")
print(f"  {'best single split':<44}{qs.min():12.4f}{rs.min():12.4f}")
print(f"  {'worst single split':<44}{qs.max():12.4f}{rs.max():12.4f}")
print(f"  {'splits won':<44}{int((diffs < 0).sum()):12d}"
      f"{int((diffs > 0).sum()):12d}")
se = diffs.std(ddof=1) / np.sqrt(len(diffs))
print(f"\n  paired difference (quantum - RBF): mean {diffs.mean():+.4f}"
      f"  s.e. {se:.4f}  t = {diffs.mean()/se:+.2f}")
print("  That t is NOT a valid t statistic and this course will not pretend it is.")
print("  The 20 splits are resamples of the same 60 points, so their training sets")
print("  overlap heavily and the 20 differences are positively correlated; the naive")
print("  s.e. is too small and t is inflated. This is Dietterich's objection to")
print("  resampled t tests, and it applies to every 'repeated random split' table.")
print("  What does carry the conclusion here is distribution-free: the quantum arm")
print(f"  loses {int((diffs > 0).sum())} of {len(diffs)} splits, and under any null in which the two")
print(f"  models are equal that has probability 2^-{len(diffs)} = {2.0**-len(diffs):.1e} by the sign test.")
j = int(np.argmin(diffs))
print(f"  most favourable split for the quantum arm: #{j}, quantum {qs[j]:.4f}"
      f" vs RBF {rs[j]:.4f}")
print(f"\n  Three sentences, all arithmetically true of this one experiment:")
print(f"    'the quantum kernel reaches a test MSE of {qs.min():.4f}, against"
      f" {rs.max():.4f} for the")
print(f"     RBF baseline' -- best quantum split against worst classical split,"
      f" a {rs.max()/qs.min():.2f}x win")
print(f"    'on the standard split the quantum kernel scores {mse_q:.4f} against"
      f" {mse_r:.4f}' -- a {mse_q/mse_r:.2f}x loss")
print(f"    'over 20 paired splits the quantum kernel loses every one, mean"
      f" {qs.mean():.4f} against {rs.mean():.4f}'")
print("  Only the third is an honest summary, and only the third is falsifiable by")
print("  re-running the experiment. Mixing splits between the two arms of a comparison")
print("  is the single most effective way to manufacture a result.")

print("\nHygiene item 2 -- baseline strength, with the quantum arm held fixed")
print("-" * 92)
print("  The same quantum result, compared against RBF baselines of varying effort.")
print(f"  {'baseline':<44}{'test MSE':>12}{'quantum looks':>16}")
Kq_tr = quantum_kernel(Xtr, Xtr)
for label, g, lam in [("RBF, gamma=10, lambda=1 (barely tuned)", 10.0, 1.0),
                      ("RBF, gamma=10, lambda=1e-3", 10.0, 1e-3),
                      ("RBF, gamma=1, lambda=1e-3 (a guess)", 1.0, 1e-3),
                      ("RBF, gamma and lambda by 5-fold CV", g_star, lam_r)]:
    a_, m_ = krr_fit(rbf_kernel(Xtr, Xtr, g), ytr, lam)
    e = mse(krr_predict(rbf_kernel(Xte, Xtr, g), a_, m_), yte)
    verdict = f"{e/mse_q:.2f}x better" if e > mse_q else f"{mse_q/e:.2f}x worse"
    print(f"  {label:<44}{e:12.4f}{verdict:>16}")
print(f"  quantum fidelity kernel, lambda by the same CV: {mse_q:.4f}")
print("  The quantum model did not change between these four rows. The verdict moved")
print("  from '2.25x better' to '3.50x worse' entirely through the choice of opponent.")
print(f"  Note also that the CV-selected RBF is not the best RBF here: gamma = {g_orc:g}")
print(f"  reaches {mse_orc:.4f} on this split, and 40 points are too few for CV to identify")
print("  gamma reliably. That cuts both ways, and it is the reason a single number")
print("  without a paired error bar is not evidence -- including the numbers above.")

print("\nHygiene item 3 -- data set size")
print("-" * 92)
print(f"  {'train size':>11}{'quantum':>10}{'RBF':>10}{'q/RBF':>8}{'OLS':>10}"
      f"{'Fourier d2':>12}{'mean of y':>11}")
Phi_all, _ = fourier_features(Xall, 2)
for ntr in (10, 15, 20, 30, 40):
    Xa, ya = Xall[:ntr], yall[:ntr]
    lq, _ = cv_select(lambda a, b: quantum_kernel(a, b), Xa, ya, folds=5)
    aq_, mq_ = krr_fit(quantum_kernel(Xa, Xa), ya, lq)
    eq = mse(krr_predict(quantum_kernel(Xte, Xa), aq_, mq_), yte)
    best = (None, np.inf)
    for g in GAMMAS:
        lr_, c_ = cv_select(lambda a, b, g=g: rbf_kernel(a, b, g), Xa, ya, folds=5)
        if c_ < best[1]:
            best = ((g, lr_), c_)
    (gs, ls), _ = best
    ar_, mr_ = krr_fit(rbf_kernel(Xa, Xa, gs), ya, ls)
    er = mse(krr_predict(rbf_kernel(Xte, Xa, gs), ar_, mr_), yte)
    Aa = np.hstack([Xa, np.ones((ntr, 1))])
    ca = np.linalg.lstsq(Aa, ya, rcond=None)[0]
    eo = mse(np.hstack([Xte, np.ones((len(yte), 1))]) @ ca, yte)
    lf, _ = cv_features(Phi_all[:ntr], ya, folds=5)
    wf, mf = ridge_features(Phi_all[:ntr], ya, lf)
    ef = mse(Phi_all[40:] @ wf + mf, yte)
    print(f"  {ntr:>11}{eq:10.4f}{er:10.4f}{eq/er:8.2f}{eo:10.4f}{ef:12.4f}"
          f"{mse(np.full(len(yte), ya.mean()), yte):11.4f}")
print("  Read the ratio column, not the absolute one. Every model improves with data,")
print("  so the absolute gap shrinks from 0.08 to 0.03 -- which a favourably worded")
print("  abstract could describe as 'the classical advantage vanishes with scale'. The")
print("  ratio wanders between 1.6 and 3.2 with no trend at all: the quantum kernel is")
print("  a constant factor worse everywhere, and the wandering is sampling noise on 20")
print("  test points. Publish the curve, not a point, and put an error bar on it.")

print("\nA checklist for reading a quantum-advantage claim")
print("-" * 92)
CHECKLIST = [
    ("Data set size",
     "Below a few hundred points no comparison of two models separates them.",
     "Ask for the paired standard error, never two isolated point estimates."),
    ("Baseline strength",
     "A tuned RBF, a gradient-boosted tree and plain linear regression are the",
     "minimum. 'We beat an untuned SVM' is not a measurement of anything."),
    ("Tuning parity",
     "Count the hyperparameters searched and the budget spent on each side.",
     "Unequal search effort is the most common silent thumb on the scale."),
    ("Selection count",
     "How many splits, seeds, initialisations, feature maps and qubit counts",
     "were tried, and is the reported one the best of them? Ask for the spread."),
    ("Data provenance",
     "If the features come from a spreadsheet or a classical simulation, the",
     "input problem and dequantization apply in full force."),
    ("Surrogate ruled out?",
     "Low-weight Pauli features and the encoding's own Fourier basis are cheap.",
     "If either matches the quantum model, there is nothing left to explain."),
    ("Resources reported?",
     "Shots, circuit evaluations, wall-clock, and the classical pre- and",
     "post-processing the pipeline needs anyway. Advantage is a ratio, not a win."),
    ("Bigger than the noise?",
     "Compare the claimed gap with the label noise of the data set itself.",
     "Most reported gaps are smaller than the noise they are measured through."),
]
for i, (head, l1, l2) in enumerate(CHECKLIST, start=1):
    print(f"  {i}. {head}")
    print(f"     {l1}")
    print(f"     {l2}")
```

```text
Hygiene item 1 -- one split is not a result
--------------------------------------------------------------------------------------------
  20 random 40/20 splits of the same 60 points. Both models re-tune lambda by
  5-fold CV inside each split's training set. Nothing else changes.
  statistic                                        quantum         RBF
  mean test MSE over 20 splits                      0.0843      0.0242
  median test MSE                                   0.0777      0.0238
  best single split                                 0.0375      0.0108
  worst single split                                0.1833      0.0431
  splits won                                             0          20

  paired difference (quantum - RBF): mean +0.0600  s.e. 0.0080  t = +7.54
  That t is NOT a valid t statistic and this course will not pretend it is.
  The 20 splits are resamples of the same 60 points, so their training sets
  overlap heavily and the 20 differences are positively correlated; the naive
  s.e. is too small and t is inflated. This is Dietterich's objection to
  resampled t tests, and it applies to every 'repeated random split' table.
  What does carry the conclusion here is distribution-free: the quantum arm
  loses 20 of 20 splits, and under any null in which the two
  models are equal that has probability 2^-20 = 9.5e-07 by the sign test.
  most favourable split for the quantum arm: #4, quantum 0.0487 vs RBF 0.0302

  Three sentences, all arithmetically true of this one experiment:
    'the quantum kernel reaches a test MSE of 0.0375, against 0.0431 for the
     RBF baseline' -- best quantum split against worst classical split, a 1.15x win
    'on the standard split the quantum kernel scores 0.0680 against 0.0422' -- a 1.61x loss
    'over 20 paired splits the quantum kernel loses every one, mean 0.0843 against 0.0242'
  Only the third is an honest summary, and only the third is falsifiable by
  re-running the experiment. Mixing splits between the two arms of a comparison
  is the single most effective way to manufacture a result.

Hygiene item 2 -- baseline strength, with the quantum arm held fixed
--------------------------------------------------------------------------------------------
  The same quantum result, compared against RBF baselines of varying effort.
  baseline                                        test MSE   quantum looks
  RBF, gamma=10, lambda=1 (barely tuned)            0.1528    2.25x better
  RBF, gamma=10, lambda=1e-3                        0.1091    1.61x better
  RBF, gamma=1, lambda=1e-3 (a guess)               0.0194     3.50x worse
  RBF, gamma and lambda by 5-fold CV                0.0422     1.61x worse
  quantum fidelity kernel, lambda by the same CV: 0.0680
  The quantum model did not change between these four rows. The verdict moved
  from '2.25x better' to '3.50x worse' entirely through the choice of opponent.
  Note also that the CV-selected RBF is not the best RBF here: gamma = 2
  reaches 0.0208 on this split, and 40 points are too few for CV to identify
  gamma reliably. That cuts both ways, and it is the reason a single number
  without a paired error bar is not evidence -- including the numbers above.

Hygiene item 3 -- data set size
--------------------------------------------------------------------------------------------
   train size   quantum       RBF   q/RBF       OLS  Fourier d2  mean of y
           10    0.1977    0.1197    1.65    0.1234      0.0704     0.2497
           15    0.1475    0.0495    2.98    0.1272      0.0527     0.2964
           20    0.1682    0.0549    3.06    0.0548      0.0739     0.2763
           30    0.0977    0.0305    3.21    0.0464      0.0251     0.2742
           40    0.0680    0.0422    1.61    0.0465      0.0172     0.2747
  Read the ratio column, not the absolute one. Every model improves with data,
  so the absolute gap shrinks from 0.08 to 0.03 -- which a favourably worded
  abstract could describe as 'the classical advantage vanishes with scale'. The
  ratio wanders between 1.6 and 3.2 with no trend at all: the quantum kernel is
  a constant factor worse everywhere, and the wandering is sampling noise on 20
  test points. Publish the curve, not a point, and put an error bar on it.

A checklist for reading a quantum-advantage claim
--------------------------------------------------------------------------------------------
  1. Data set size
     Below a few hundred points no comparison of two models separates them.
     Ask for the paired standard error, never two isolated point estimates.
  2. Baseline strength
     A tuned RBF, a gradient-boosted tree and plain linear regression are the
     minimum. 'We beat an untuned SVM' is not a measurement of anything.
  3. Tuning parity
     Count the hyperparameters searched and the budget spent on each side.
     Unequal search effort is the most common silent thumb on the scale.
  4. Selection count
     How many splits, seeds, initialisations, feature maps and qubit counts
     were tried, and is the reported one the best of them? Ask for the spread.
  5. Data provenance
     If the features come from a spreadsheet or a classical simulation, the
     input problem and dequantization apply in full force.
  6. Surrogate ruled out?
     Low-weight Pauli features and the encoding's own Fourier basis are cheap.
     If either matches the quantum model, there is nothing left to explain.
  7. Resources reported?
     Shots, circuit evaluations, wall-clock, and the classical pre- and
     post-processing the pipeline needs anyway. Advantage is a ratio, not a win.
  8. Bigger than the noise?
     Compare the claimed gap with the label noise of the data set itself.
     Most reported gaps are smaller than the noise they are measured through.
```

**What to look for.** Each of the three blocks is a different way to manufacture a result, and each is common in print.

**Mixing the arms of a comparison.** Twenty random splits, both models re-tuned inside each split, and the quantum kernel loses **all twenty**. And yet the best quantum split, 0.0375, is better than the worst classical split, 0.0431, so a sentence comparing those two numbers is arithmetically true and describes a $1.15\times$ win. Nothing in that sentence is a fabrication. The manipulation is entirely in the pairing.

The block also prints the paired difference, $+0.0600$ with a nominal standard error of 0.0080 and $t = 7.5$, and then tells the reader not to believe the $t$. That is deliberate, because the same defect sits in most "repeated random split" tables in this literature. Twenty splits of the same 60 points share most of their training rows, so the twenty per-split differences are positively correlated, the naive standard error is too small, and $t$ is inflated by an unknown factor — Dietterich's objection to resampled $t$ tests, and it applies to this course's own table as much as to anyone else's. What carries the conclusion is the distribution-free statement next to it: twenty losses out of twenty, which under any null where the two models are equally good has probability $2^{-20} \approx 10^{-6}$ by the sign test. The sign test throws away the magnitudes and keeps only the directions, which is exactly the information the overlap does not corrupt. Quote that, not the $t$.

**Choosing the opponent.** The quantum arm is held completely fixed while the classical baseline varies in effort. Against an RBF with $\gamma = 10, \lambda = 1$ the quantum kernel looks $2.25\times$ better; against a cross-validated RBF it looks $1.61\times$ worse; against a lucky guess of $\gamma = 1, \lambda = 10^{-3}$ it looks $3.50\times$ worse. A range of a factor of eight in the reported verdict, with the quantum model unchanged. This is why "we compared against a standard SVM" is not a sentence that carries information: the question is how much effort went into the baseline relative to the proposal, and the answer is almost never reported.

Note also the honest complication in that block: the cross-validated RBF is *not* the best RBF here. Forty points are too few for five-fold CV to identify $\gamma$ reliably, so the protocol picked a worse setting than a guess would have. That cuts both ways and is exactly why a single point estimate without a paired error bar is not evidence — a lesson that applies to this course's own numbers.

**Choosing the scale.** Across training sizes from 10 to 40 the quantum kernel's *absolute* gap to the RBF shrinks from 0.078 to 0.026, which a favourably worded abstract could describe as an advantage emerging with scale. The *ratio* wanders between 1.6 and 3.2 with no trend, which is the correct reading: the quantum kernel is a constant factor worse everywhere, and the wandering is sampling noise on 20 test points. Publish the curve, and put an error bar on it.

### The checklist

The eight points printed by Code Example 5 are the practical distillation. Two of them deserve emphasis because they are the ones most often missing and most decisive.

**"Is the data set classical?"** If the features arrive from a spreadsheet, a DFT calculation or an instrument's data-reduction pipeline, then loading them into a quantum state is itself the bottleneck — the input problem of Chapter 1 — and the dequantization arguments of this chapter apply in full force. Almost every published QML benchmark uses classical data.

**"Is a classical surrogate ruled out?"** Section 5.2 is a template. Building the two surrogates is a day's work for anyone who can already run the quantum model, and if either matches, there is nothing left to explain. A claim that has not been checked against them is not a measurement of anything.

A useful habit when reading: before looking at the reported numbers, write down what the *baseline* should be and what the *paired* statistic should be. Then see whether the paper contains them. The absence is more informative than the numbers.

* * *

## 5.4 When Quantum Data Changes the Picture

Everything negative in this course has one shared premise: the data is classical. Four real numbers per material, sitting in a table, that must be loaded into a quantum register before a quantum model can look at them. That premise is doing all the work, and it is worth seeing exactly how.

### The input problem, and why it disappears

Loading a classical vector into a quantum state costs something. Angle encoding is cheap in depth but uses one qubit per feature and gives a restricted feature map; amplitude encoding is compact but needs a state-preparation circuit whose depth grows with the dimension. Either way, the loading is a *classical-to-quantum* conversion that a classical algorithm never has to perform, and its cost is charged entirely to the quantum side of the comparison. Worse, whatever structure the encoding imposes is now the model's inductive bias, and Section 5.2 showed how easily that bias can be written down and used classically instead.

Now suppose the data is already a quantum state. A pair of entangled photons from a parametric down-conversion source; the state of a nuclear spin ensemble part-way through a pulse sequence; the many-body state of a cold-atom array at the end of a quench; the output of a quantum simulation of a candidate catalyst. In these cases there is nothing to load. The quantum computer receives quantum states directly, and the classical competitor cannot receive them at all — it can only receive whatever classical measurement record someone chose to extract, which is a lossy projection of the state.

This is the CQ/QQ distinction of Chapter 1's four-quadrant picture, and it is the one place where the asymmetry runs the other way. A number of results in this direction are genuinely rigorous, and the precise form of the statement matters. They are **sample-complexity** separations, not computational ones: there are learning tasks defined on quantum states for which *any* strategy restricted to single copies of the state — however clever the classical post-processing, and with unlimited computation — needs exponentially many experiments, while a strategy that can process two or more copies coherently needs polynomially many. The bound is on the number of experiments, not on running time, and it holds against every single-copy measurement scheme rather than merely against the classical algorithms someone has thought of. The mechanism is that a coherent processor can measure properties of the *ensemble* — overlaps between states from different experimental settings, for instance — that no sequence of independent single-copy measurements can reconstruct efficiently.

### What it costs instead

The input problem is replaced by three new ones, and it would be dishonest to present the quantum-data case as free of obstacles.

  * **States cannot be stored.** A classical data set can be re-used indefinitely; a quantum state is consumed by measurement and cannot be copied. "Ten thousand training examples" means running the experiment ten thousand times, per epoch, and any algorithm that needs to revisit a training point needs to re-prepare it.
  * **The experiment must be co-located with the processor.** The states have to arrive coherently. In practice this means the sensor or the simulation and the quantum processor are the same apparatus, or are connected by a quantum channel. This is a serious engineering constraint and it excludes most existing instruments.
  * **The classical baseline is not weak.** Classical shadows — the same construction as Section 5.2's surrogate features — let a classical algorithm learn a great deal from randomized measurements of quantum states, with sample complexity independent of the number of qubits for low-weight observables. Several proposed quantum-data advantages have been dequantized by exactly that route. The separations that survive are the ones where the quantity of interest is not a low-weight observable.

### What this means for materials research, concretely

The honest position is that quantum data is the direction in which the argument changes in kind rather than in degree, and that the near-term instances of it in materials science are narrow but real:

  * **Quantum simulation output as training data.** If a fault-tolerant machine one day computes the ground state of a strongly correlated material, learning a property from the state itself — rather than from a handful of extracted numbers — is a QQ-quadrant task with no classical loading step.
  * **Quantum sensing and metrology.** Measurements whose signal lives in the coherence of a quantum probe rather than in a classical readout produce quantum data in the relevant sense. Learning to discriminate or estimate from such probes is an active area, and a treatment of it belongs to a course on quantum sensing rather than to this one.
  * **Characterization of quantum devices themselves.** Learning the noise model or Hamiltonian of a processor from its own states is the most immediately practical instance, and it is a materials problem: the answer is a statement about defects, interfaces and two-level fluctuators. The sister course [Introduction to Quantum Hardware](<../../FM/quantum-hardware-introduction/index.html>) treats the physics; the learning problem sits on top of it.

None of these are pipelines a materials informatics group can build this year. All of them are reasons the subject is worth understanding rather than dismissing.

* * *

## 5.5 The Whole Argument in One Sweep

### Code Example 6: Concentration and Dequantization, Together

```python
"""Capstone: concentration and dequantization in one sweep.
Continues from Examples 1, 2, 3, 4 and 5 (same session).
"""


def widen(Xs, n):
    """Fill an n-qubit register with the 4 descriptors, repeating them as needed."""
    reps = int(np.ceil(n / Xs.shape[1]))
    return np.tile(Xs, (1, reps))[:, :n]


def krr_test_mse(kernel_fn, Xa, ya, Xb, yb):
    lam, cv = cv_select(kernel_fn, Xa, ya)
    a, mu = krr_fit(kernel_fn(Xa, Xa), ya, lam)
    return mse(krr_predict(kernel_fn(Xb, Xa), a, mu), yb), lam, cv


print("Concentration and its classical surrogate, side by side")
print("-" * 96)
print("  shots/entry = 100 k(1-k)/sigma^2, i.e. eps = sigma/10 on a Bernoulli entry")
print(f"  {'n':>3}{'off-diag std':>14}{'quantum test':>14}"
      f"{'Pauli w<=2':>12}{'features':>10}{'shots/entry':>13}{'quantum/Pauli':>15}")
NSW = list(range(2, 9))
sweep = {}
for n in NSW:
    Xw = widen(Xall, n)
    Kw = quantum_kernel(Xw, Xw)
    off = Kw[~np.eye(len(Kw), dtype=bool)]
    kf = lambda a, b: quantum_kernel(a, b)
    te_q, lam_qn, _ = krr_test_mse(kf, Xw[:40], ytr, Xw[40:], yte)
    Phi, ps = shadow_features(Xw, 2, n=n)
    lam_s, _ = cv_features(Phi[:40], ytr)
    wv, mu = ridge_features(Phi[:40], ytr, lam_s)
    te_s = mse(Phi[40:] @ wv + mu, yte)
    sweep[n] = (off.std(), te_q, te_s, Phi.shape[1])
    kb = off.mean()
    print(f"  {n:>3}{off.std():14.6f}{te_q:14.4f}{te_s:12.4f}"
          f"{Phi.shape[1]:>10}{100 * kb * (1 - kb) / off.std()**2:13,.0f}"
          f"{te_q/te_s:15.2f}")

print("\nThe two purely classical references, which do not depend on n at all")
print("-" * 96)
Phi_f, _ = fourier_features(Xall, 2)
lam_f, cv_f = cv_features(Phi_f[:40], ytr)
wf, mf = ridge_features(Phi_f[:40], ytr, lam_f)
te_fourier = mse(Phi_f[40:] @ wf + mf, yte)
print(f"  {'Fourier surrogate, total degree <= 2':<44}{Phi_f.shape[1]:>10} features"
      f"   test MSE {te_fourier:.4f}")
print(f"  {'RBF kernel ridge, gamma and lambda by CV':<44}{'-':>10}          "
      f"   test MSE {mse_r:.4f}")
print(f"  {'RBF kernel ridge, best gamma on test (bound)':<44}{'-':>10}          "
      f"   test MSE {mse_orc:.4f}")
print(f"  {'ordinary least squares':<44}{'5':>10} features"
      f"   test MSE {mse_ols:.4f}")
print(f"  {'best quantum result anywhere in the sweep':<44}{'-':>10}          "
      f"   test MSE {min(v[1] for v in sweep.values()):.4f}"
      f"  (n = {min(sweep, key=lambda k: sweep[k][1])})")

print("\nSummary of the whole course, in numbers")
print("-" * 96)
rows = [
    ("predict the training mean", 1, mse_mean),
    ("ordinary least squares", 5, mse_ols),
    ("RBF kernel ridge, tuned by CV", 40, mse_r),
    ("RBF kernel ridge, best gamma on test (bound)", 40, mse_orc),
    ("Fourier surrogate of the quantum feature map", Phi_f.shape[1], te_fourier),
    ("weight-2 Pauli surrogate, 4 qubits", 67, sweep[4][2]),
    ("quantum fidelity kernel ridge, 4 qubits", 40, sweep[4][1]),
    ("quantum fidelity kernel ridge, best n", 40,
     min(v[1] for v in sweep.values())),
]
print(f"  {'model':<46}{'params/features':>17}{'test MSE':>11}{'vs best':>10}")
best = min(r[2] for r in rows)
for nm, npar, e in rows:
    print(f"  {nm:<46}{npar:>17}{e:11.4f}{e/best:10.2f}x")
print(f"  {'irreducible noise floor':<46}{'-':>17}{0.05**2:11.4f}"
      f"{0.05**2/best:10.2f}x")

print("\nWhat the sweep establishes")
print("-" * 96)
print("  1. Widening the register makes the quantum model worse. Not monotonically --")
print("     n = 3 is the best point and n = 8 the worst, and 20 test points cannot")
print("     resolve the wiggles in between -- but the trend is unambiguous over the")
print("     sweep. The kernel concentrates, the Gram matrix approaches the identity, and")
print("     kernel ridge on an identity matrix is a look-up table with no generalisation.")
print("     More qubits is not more model; on classical data it is less model.")
print("  2. A classical surrogate with O(n^2) features tracks the quantum model to within")
print("     a factor of about two in both directions at every width, with no systematic")
print("     advantage on the quantum side. Where the quantum kernel has concentrated,")
print("     truncation even helps: it acts as a regulariser the fidelity kernel lacks.")
print("  3. The best number in the whole table belongs to a classical model with an")
print("     explicit feature map built from the quantum encoding's own frequencies.")
print("     Everything the quantum model could express was expressible without it.")
print("  4. None of this is a hardware statement. The simulator is exact and noiseless,")
print("     so no error rate, no coherence time and no qubit count changes any row.")
print("     The obstruction is the problem, not the machine -- and that is the reason")
print("     to look for problems with quantum data rather than better quantum hardware.")
```

```text
Concentration and its classical surrogate, side by side
------------------------------------------------------------------------------------------------
  shots/entry = 100 k(1-k)/sigma^2, i.e. eps = sigma/10 on a Bernoulli entry
    n  off-diag std  quantum test  Pauli w<=2  features  shots/entry  quantum/Pauli
    2      0.309325        0.0362      0.0362        16          241           1.00
    3      0.262934        0.0284      0.0345        37          268           0.82
    4      0.191101        0.0680      0.0642        67          355           1.06
    5      0.167800        0.0628      0.0735       106          347           0.85
    6      0.137245        0.0831      0.1955       154          382           0.43
    7      0.119306        0.0838      0.0506       211          375           1.65
    8      0.105990        0.1064      0.0667       277          381           1.60

The two purely classical references, which do not depend on n at all
------------------------------------------------------------------------------------------------
  Fourier surrogate, total degree <= 2                41 features   test MSE 0.0172
  RBF kernel ridge, gamma and lambda by CV             -             test MSE 0.0422
  RBF kernel ridge, best gamma on test (bound)         -             test MSE 0.0208
  ordinary least squares                               5 features   test MSE 0.0465
  best quantum result anywhere in the sweep            -             test MSE 0.0284  (n = 3)

Summary of the whole course, in numbers
------------------------------------------------------------------------------------------------
  model                                           params/features   test MSE   vs best
  predict the training mean                                     1     0.2747     15.94x
  ordinary least squares                                        5     0.0465      2.70x
  RBF kernel ridge, tuned by CV                                40     0.0422      2.45x
  RBF kernel ridge, best gamma on test (bound)                 40     0.0208      1.21x
  Fourier surrogate of the quantum feature map                 41     0.0172      1.00x
  weight-2 Pauli surrogate, 4 qubits                           67     0.0642      3.73x
  quantum fidelity kernel ridge, 4 qubits                      40     0.0680      3.94x
  quantum fidelity kernel ridge, best n                        40     0.0284      1.65x
  irreducible noise floor                                       -     0.0025      0.15x

What the sweep establishes
------------------------------------------------------------------------------------------------
  1. Widening the register makes the quantum model worse. Not monotonically --
     n = 3 is the best point and n = 8 the worst, and 20 test points cannot
     resolve the wiggles in between -- but the trend is unambiguous over the
     sweep. The kernel concentrates, the Gram matrix approaches the identity, and
     kernel ridge on an identity matrix is a look-up table with no generalisation.
     More qubits is not more model; on classical data it is less model.
  2. A classical surrogate with O(n^2) features tracks the quantum model to within
     a factor of about two in both directions at every width, with no systematic
     advantage on the quantum side. Where the quantum kernel has concentrated,
     truncation even helps: it acts as a regulariser the fidelity kernel lacks.
  3. The best number in the whole table belongs to a classical model with an
     explicit feature map built from the quantum encoding's own frequencies.
     Everything the quantum model could express was expressible without it.
  4. None of this is a hardware statement. The simulator is exact and noiseless,
     so no error rate, no coherence time and no qubit count changes any row.
     The obstruction is the problem, not the machine -- and that is the reason
     to look for problems with quantum data rather than better quantum hardware.
```

**What to look for.** This is the course's summary experiment, so it is worth reading slowly.

**Widening the register makes the quantum model worse.** Best at three qubits, worst at eight. Not monotone — 20 test points cannot resolve the wiggles — but the trend across the sweep is unambiguous, and the mechanism is in the second column: the off-diagonal spread falls by a factor of three from $n=2$ to $n=8$, the Gram matrix drifts towards the identity, and there is progressively less information in it to regress on.

**The classical surrogate tracks the quantum model at every width**, within about a factor of two in either direction and with no systematic advantage on the quantum side. Its feature count is $O(n^2)$: 67 at four qubits, 277 at eight, against $4^n$ for the exact expansion.

**The best number in the table belongs to a classical model**, the 41-feature Fourier surrogate at 0.0172, built from nothing but the quantum encoding's own frequency list. The best quantum result anywhere in the sweep is 0.0284, at three qubits, and even that is a configuration nobody would have proposed on the grounds of quantum advantage — and it is still worse than the strongest RBF's 0.0208. The bound row matters here: a reader who suspects the cross-validated RBF of being a weak opponent can use the oracle row instead, and every conclusion in this chapter survives the substitution.

**None of it is a hardware statement.** The simulator is exact and noiseless. No error rate, coherence time or qubit count changes any entry. That is the single most important sentence in this course, because "it will work when the hardware improves" is the standard response to results like these, and here it is simply not applicable.

* * *

## 5.6 Practical Guidance, and What the Series Was For

### What is worth learning now

The conclusion of five chapters of negative results is not "ignore this subject". It is that the *reason* to learn it is different from the reason usually given, and the different reason is durable.

**Learn the kernel view, because it is portable.** Chapter 3's material — feature maps, induced inner products, kernel ridge regression in closed form, the geometry of a Gram matrix, effective rank — is classical machine learning that happens to have been sharpened by quantum questions. It will still be useful if quantum computing stalls for twenty years. The habit of asking "what inner product does my representation induce, and does that inner product respect the physics?" is worth more to a materials informatics practitioner than any circuit.

**Learn the frequency view, because it is diagnostic.** Chapter 2's picture — a model as a truncated Fourier series whose frequency support is set by the representation, not by the fitting procedure — is the cleanest available way to think about what a feature representation can and cannot express. It transfers directly to descriptor design, to Fourier features in classical kernels, and to positional encodings in transformers.

**Learn the benchmark discipline, because the field needs it.** Matched parameter counts, one optimiser, one selection protocol, paired statistics, every restart reported, baselines that include the trivial ones. Sections 4.3 and 5.3 are a template that applies to every comparison a materials informatics group will ever run, quantum or not. The most transferable skill in this course is the ability to look at a reported gap and ask which of the eight checklist points it fails.

**Understand the concentration and plateau arguments, because they are the fastest available filter.** Any proposal involving a fidelity kernel at 30 qubits, or a deep unstructured ansatz with a global read-out, is answerable in one line and without running anything. That is a useful thing to be able to do in a review meeting.

### What is worth waiting on

  * **SDK fluency.** Framework APIs change yearly and the skill transfers poorly. Learn one when there is a specific calculation to run, not before.
  * **Hardware access.** Nothing in this course needed it, and nothing in this course would have changed if it had been available. Time on a device is worth having when the bottleneck is a device.
  * **Building a QML pipeline for tabular materials data.** This is the specific activity the evidence argues against. The models lose, the surrogates match, the obstruction is mathematical, and the cost ratio is five orders of magnitude.

### A decision table

| If your problem is | Then the relevant question is | And the near-term answer is |
| --- | --- | --- |
| Property prediction from descriptors in a database | Does a quantum model beat a tuned classical one? | No, and the reason is not the hardware |
| Ground-state energies of correlated materials | Can a quantum computer produce the data at all? | The sister course's VQE and its successors; not a learning problem yet |
| Learning from a quantum sensor or simulator | Is the data quantum, and is the processor co-located? | The one direction with a real asymmetry, and the least mature |
| Characterizing a quantum device from its own behaviour | What does the noise say about the material? | Practical now, and a materials problem |
| Combinatorial optimisation dressed as ML | Is the classical solver actually strong? | Usually yes, and usually not compared against |

### The series in one page

Chapter 1 laid out the four quadrants and warned that the course would live almost entirely in one of them, classical data with quantum processing, because that is where the field's activity and its overclaiming are concentrated.

Chapter 2 showed that the encoding, not the variational part, is where a quantum model's expressivity comes from, and that a re-uploading circuit is a truncated Fourier series with a frequency support you can enumerate. That enumeration became this chapter's most effective classical attack.

Chapter 3 built the fidelity kernel, connected it to kernel ridge regression, and measured exponential concentration: more qubits means a flatter kernel, and a flatter kernel means less learning. Its entangling kernel came out 1.16 times worse than a tuned RBF, and the only quantum kernel there that matched the classical baseline had no entanglement in it.

Chapter 4 trained a variational circuit against a matched classical network under a protocol designed to be defensible, and lost — with the lowest training loss and the highest test loss of the three models, which is overfitting, not underfitting. Its paired intervals resolve the loss against the 25-parameter network and call the 31-parameter comparison a tie, which is the shape of most honest results at this data size. It also measured barren plateaus in their QML form: $2.07\times$ variance decay per qubit for a local read-out, $4.03\times$ for a global one, $1.05\times$ with entanglement removed.

Chapter 5 built two classical surrogates and found that both match the quantum kernel and one beats it, showed three contradictory-but-true summaries of a single experiment, and located the one premise whose failure would change the argument: classical data.

If the course has one message, it is that **the honest number is the useful number**. The number to carry away is not the single-split 1.6, which is one draw of a noisy quantity; it is the better-powered one — over twenty paired splits the quantum kernel loses every one, mean test MSE 0.0843 against the cross-validated RBF's 0.0242, a factor of 3.5 — together with the single-split 3.3 against the strongest RBF available on this data. A materials researcher who knows that the quantum kernel loses by about a factor of three to a properly tuned classical kernel, and why, is in a far better position than one who has read that quantum machine learning offers exponential speedups. The first can plan; the second can only wait. And the discipline that produced that number — fix the protocol first, include the trivial baselines, report every run, publish the curve — is the part of this course that will still be earning its keep long after every specific result in it has been superseded.

* * *

## Exercises

#### Exercise 1: The Haar Prediction, Against the Measurement

For two independent Haar-random pure states in dimension $d$, the fidelity $|\langle\phi|\phi'\rangle|^2$ follows a Beta$(1, d-1)$ distribution.

  1. Write down its mean and variance in terms of $d$.
  2. Tabulate the predicted mean and standard deviation for $n = 2$ to $12$ qubits, and compare with the measured values of Code Example 2.
  3. The measured mean exceeds the Haar prediction by a factor that grows from 1.44 at $n=2$ to 87 at $n=12$. What does that say about the encoding?
  4. Does the discrepancy make the concentration problem better or worse for a practitioner? Answer in terms of the shot count, remembering that a kernel entry is a Bernoulli variable and so $S = k(1-k)/\varepsilon^2$.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Mean \(1/d\); variance \((d-1)/\bigl(d^2(d+1)\bigr)\), so the standard deviation is \(\sqrt{d-1}/\bigl(d\sqrt{d+1}\bigr) \approx 1/d\) for large \(d\). Both scale as \(2^{-n}\).</p>

<p><strong>2.</strong> See the table in the code. At \(n = 12\), \(d = 4096\), the Haar mean is \(2.44\times10^{-4}\) against a measured \(2.13\times10^{-2}\).</p>

<p><strong>3.</strong> The two-layer angle encoding is very far from Haar-random. It is shallow, its entangling structure is a single CNOT ring per layer, and the encoded states occupy a low-dimensional manifold of the full Hilbert space parameterised by only four real numbers. The states are therefore much more similar to each other than random states would be, and the kernel retains more structure than the Haar estimate allows.</p>

<p><strong>4.</strong> Better, and the size of the improvement is worth computing correctly. A kernel entry is a Bernoulli variable, so \(S = k(1-k)/\varepsilon^2\); at the ten-fold margin \(\varepsilon = \sigma/10\) this is \(100\,k(1-k)/\sigma^2\). In the Haar regime \(k \approx \sigma \approx 1/d\), which gives \(S \approx 100\,d\) — that is, \(4.1\times10^{5}\) at twelve qubits, or \(d = 4096\) if one drops the margin and only asks the shot noise to match the whole spread. The measured encoding needs about \(4\times10^{2}\), a factor of a thousand less, because its \(\sigma\) is far larger than the Haar value. Note that the wrong formula, \(1/\sigma^2\), would have put the Haar demand at \(d^2 = 1.7\times10^{7}\) and overstated it by a further factor of \(d\). But the same fact that saves the shots is bad news for the advantage argument: a kernel that stays classically structured because its circuit is shallow is a kernel with a small classical description, which is exactly what Code Examples 3 and 4 exploit. Escaping the shot problem and escaping dequantization pull in opposite directions.</p>

```python
"""Exercise 1. Continues from Examples 1 and 2 (same session)."""
print(f"  {'n':>3}{'d':>7}{'Haar mean':>12}{'Haar std':>12}{'measured mean':>15}"
      f"{'measured std':>14}{'mean ratio':>12}")
for n in NS:
    d = 2 ** n
    hm = 1.0 / d
    hs = np.sqrt((d - 1) / (d ** 2 * (d + 1)))
    m, s = stats[n][0], stats[n][1]
    print(f"  {n:>3}{d:>7}{hm:12.6f}{hs:12.6f}{m:15.6f}{s:14.6f}{m/hm:12.2f}")
# S = k(1-k)/eps^2 with eps = sigma/10. A kernel entry is a bit, not a bounded
# observable, so the numerator is k(1-k) and not 1.
d = 2 ** 12
hm, hs = 1.0 / d, np.sqrt((d - 1) / (d ** 2 * (d + 1)))
mm, ms = stats[12][0], stats[12][1]
print(f"\n  shots per kernel entry at n = 12, eps = sigma/10")
print(f"    Haar:      {100*hm*(1-hm)/hs**2:,.0f}   (~ 100 d = {100*d:,d})")
print(f"    measured:  {100*mm*(1-mm)/ms**2:,.0f}")
print(f"    the wrong formula 1/sigma^2 would say Haar {1/hs**2:,.0f}"
      f" and measured {1/ms**2:,.0f}")
```

```text
    n      d   Haar mean    Haar std  measured mean  measured std  mean ratio
    2      4    0.250000    0.193649       0.359838      0.309325        1.44
    3      8    0.125000    0.110240       0.246025      0.262934        1.97
    4     16    0.062500    0.058709       0.153000      0.191101        2.45
    5     32    0.031250    0.030288       0.109725      0.167800        3.51
    6     64    0.015625    0.015383       0.077986      0.137245        4.99
    7    128    0.007812    0.007752       0.056518      0.119306        7.23
    8    256    0.003906    0.003891       0.044784      0.105990       11.46
    9    512    0.001953    0.001949       0.036139      0.094839       18.50
   10   1024    0.000977    0.000976       0.029538      0.085575       30.25
   11   2048    0.000488    0.000488       0.024784      0.078636       50.76
   12   4096    0.000244    0.000244       0.021315      0.072502       87.31

  shots per kernel entry at n = 12, eps = sigma/10
    Haar:      409,700   (~ 100 d = 409,600)
    measured:  397
    the wrong formula 1/sigma^2 would say Haar 16,785,410 and measured 190
```

</details>

#### Exercise 2: Polynomial Against Exponential, With Numbers

  1. Give the number of Pauli strings on $n$ qubits with weight at most $w$.
  2. Tabulate it for $w = 1, 2, 3$ and $n = 4, 10, 20, 50$, next to $4^n$.
  3. At $n = 50$ and $w = 2$, what fraction of the full expansion is retained?
  4. A referee objects that a truncated surrogate is "not the same kernel". Answer the objection using the numbers from Code Example 3.

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(\sum_{k=0}^{w}\binom{n}{k}3^k\).</p>

<p><strong>2-3.</strong> See the table. At \(n = 50, w = 2\): 11,176 features against \(4^{50} = 1.27\times10^{30}\), a fraction of \(8.8\times10^{-27}\).</p>

<p><strong>4.</strong> The objection is correct and irrelevant. The truncated surrogate is demonstrably a different kernel — Code Example 3 measures the difference, 67% relative error on the off-diagonal entries at weight 2 — and its test error is nevertheless equal to or better than the exact kernel's (0.0642 against 0.0680 at weight 2; 0.0379 at weight 3). An advantage claim is a claim about predictive performance under a fixed protocol, not about operator norms. If the surrogate predicts as well, the exact kernel bought nothing, whatever the distance between them.</p>

```python
"""Exercise 2. NumPy and math.comb only."""
import numpy as np
from math import comb


def npauli(n, w):
    """Number of Pauli strings on n qubits with weight at most w."""
    return sum(comb(n, k) * 3 ** k for k in range(w + 1))


print(f"  {'n':>4}{'w=1':>10}{'w=2':>12}{'w=3':>14}{'4^n':>14}{'w=2 fraction':>16}")
for n in (4, 10, 20, 50):
    total = 4.0 ** n
    print(f"  {n:>4}{npauli(n,1):>10,}{npauli(n,2):>12,}{npauli(n,3):>14,}"
          f"{total:>14.3e}{npauli(n,2)/total:>16.2e}")
```

```text
     n       w=1         w=2           w=3           4^n    w=2 fraction
     4        13          67           175     2.560e+02        2.62e-01
    10        31         436         3,676     1.049e+06        4.16e-04
    20        61       1,771        32,551     1.100e+12        1.61e-09
    50       151      11,176       540,376     1.268e+30        8.82e-27
```

</details>

#### Exercise 3: How Often Does Cherry-Picking Work?

Suppose two models are genuinely equal, and their per-split test errors differ only by noise.

  1. Simulate 20 paired splits from a null model in which both arms have the same mean and a realistic spread, and compute the distribution of $\max_j(\text{classical}_j)/\min_j(\text{quantum}_j)$ — the "best of mine against worst of yours" ratio.
  2. What fraction of null experiments permit a claimed improvement of $1.5\times$ or more by that route?
  3. Repeat with the correct paired statistic. What fraction of null experiments give $|t| > 2$?
  4. State the reporting rule that follows.

<details>
<summary>Solution</summary>

<p><strong>1-2.</strong> With 20 splits, the best-against-worst ratio exceeds \(1.5\) in essentially every null experiment: 0.998 at the smallest spread tried and 1.000 at the two larger ones. The median ratio is 2.18, 4.78 and 10.47 for coefficients of variation 0.15, 0.30 and 0.45, and the middle row reproduces the 4.9-fold within-arm range actually measured in Code Example 5. The reason is that the ratio compares two order statistics from opposite tails, so its expectation grows with the number of splits even when the means are identical.</p>

<p><strong>3.</strong> The paired \(t\) statistic exceeds 2 in absolute value in about 5% of null experiments, which is what it is designed to do.</p>

<p><strong>4.</strong> Report the paired difference and its standard error over all splits, and report the number of splits run. Never compare an extremum of one arm with an extremum of the other. If a paper reports a best-case number, the honest reader's first question is how many cases there were.</p>

```python
"""Exercise 3. NumPy only."""
import numpy as np
rng = np.random.default_rng(21)
trials, splits = 20000, 20
print(f"  null model: both arms identically distributed, {splits} paired splits,"
      f" {trials:,} trials")
print("  the quantum arm of Code Example 5 spanned 0.0375 to 0.1833 over 20 splits,")
print("  a range of 4.9x -- which the cv = 0.30 row below reproduces")
print(f"  {'cv':>6}{'range within one arm':>22}{'median best/worst':>19}"
      f"{'P(ratio>=1.5)':>15}{'P(|t|>2)':>10}")
for cv in (0.15, 0.30, 0.45):
    ratio = np.empty(trials)
    tstat = np.empty(trials)
    within = np.empty(trials)
    for t in range(trials):
        base = rng.lognormal(0.0, cv, splits)        # shared split difficulty
        a = base * rng.lognormal(0.0, cv, splits)    # arm A
        b = base * rng.lognormal(0.0, cv, splits)    # arm B, same distribution
        ratio[t] = b.max() / a.min()                 # best of A against worst of B
        within[t] = a.max() / a.min()
        d = a - b
        tstat[t] = d.mean() / (d.std(ddof=1) / np.sqrt(splits))
    print(f"  {cv:6.2f}{np.median(within):22.2f}{np.median(ratio):19.2f}"
          f"{np.mean(ratio >= 1.5):15.3f}{np.mean(np.abs(tstat) > 2):10.3f}")
```

```text
  null model: both arms identically distributed, 20 paired splits, 20,000 trials
  the quantum arm of Code Example 5 spanned 0.0375 to 0.1833 over 20 splits,
  a range of 4.9x -- which the cv = 0.30 row below reproduces
      cv  range within one arm  median best/worst  P(ratio>=1.5)  P(|t|>2)
    0.15                  2.18               2.18          0.998     0.060
    0.30                  4.79               4.78          1.000     0.058
    0.45                 10.49              10.47          1.000     0.055
```

</details>

#### Exercise 4: Audit a Claim

An abstract reads: *"We introduce a quantum kernel for formation-energy prediction. On a benchmark of 80 alloy compositions our method achieves an RMSE of 42 meV/atom, outperforming a support vector regressor (58 meV/atom) by 28%. Results are obtained on a 12-qubit feature map."*

  1. Which of the eight checklist items does the abstract leave unanswered?
  2. The improvement is quoted as 28% in RMSE. Convert to MSE, and estimate how many test points would be needed for that gap to be resolvable at $t = 2$, assuming the per-point squared errors have a coefficient of variation of 1.5.
  3. What single additional experiment would settle the claim most cheaply?
  4. What would the concentration measurement of Code Example 2 predict for a 12-qubit fidelity kernel, and what does that imply about the reported result?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> Data set size is given but is far too small; the baseline is a single SVR with no statement of tuning effort; there is no mention of tuning parity, of how many feature maps or qubit counts were tried, of splits or seeds, of a classical surrogate, or of shots and wall-clock. Data provenance is classical (alloy compositions), which the abstract does not flag as relevant. Six of eight are unanswered, and the two that are answered are answered unfavourably.</p>

<p><strong>2.</strong> RMSE 42 against 58 meV/atom is MSE \(1764\) against \(3364\) (meV/atom)\(^2\), a relative gap of 0.476 in MSE. With coefficient of variation \(c = 1.5\) on the per-point squared errors and a paired comparison, \(t \approx \sqrt{N}\,\Delta/(c\,\bar{\mu})\) gives \(N \approx (2 \times 1.5/0.476)^2 \approx 40\) test points — so 80 points split 60/20 is marginal but not hopeless, provided the comparison is paired and the split is not chosen after the fact. The estimate is optimistic because it ignores the selection over feature maps.</p>

<p><strong>3.</strong> Build the weight-2 Pauli surrogate of the same feature map and run it under the same protocol. It is a day's work, requires no quantum device, and either removes the claim or substantially strengthens it.</p>

<p><strong>4.</strong> Code Example 2 measures an off-diagonal standard deviation of 0.0725 at twelve qubits and a test error nearly four times worse than at two qubits. A 12-qubit fidelity kernel on 80 points is in the concentrated regime, so the most likely explanation for a good reported result is that the kernel was not actually evaluated at full width — for instance that the encoding is a product over few effective features — or that the comparison was favourable for one of the reasons in part 1.</p>

```python
"""Exercise 4. NumPy only."""
import numpy as np
rmse_q, rmse_c = 42.0, 58.0
mse_q, mse_c = rmse_q ** 2, rmse_c ** 2
rel = (mse_c - mse_q) / mse_c
print(f"  RMSE {rmse_q:.0f} vs {rmse_c:.0f} meV/atom  ->  MSE {mse_q:.0f} vs {mse_c:.0f}")
print(f"  relative gap in MSE: {rel:.3f}")
for c in (1.0, 1.5, 2.0):
    print(f"  coefficient of variation {c:.1f}  ->  need N ~ "
          f"{(2*c/rel)**2:6.0f} paired test points at t = 2")
print("  80 points split 60/20 give 20 test points")
```

```text
  RMSE 42 vs 58 meV/atom  ->  MSE 1764 vs 3364
  relative gap in MSE: 0.476
  coefficient of variation 1.0  ->  need N ~     18 paired test points at t = 2
  coefficient of variation 1.5  ->  need N ~     40 paired test points at t = 2
  coefficient of variation 2.0  ->  need N ~     71 paired test points at t = 2
  80 points split 60/20 give 20 test points
```

</details>

#### Exercise 5: The Shot Budget for Quantum Data

A quantum sensor produces one copy of a state $\rho_i$ per run, labelled by an experimental setting $i$. You want to learn a property from $N$ such states.

  1. For a fidelity kernel over the $N$ states, how many state preparations does the Gram matrix cost, if each entry needs $S$ shots and each shot consumes one copy of each of the two states?
  2. Evaluate for $N = 100$ and $S = 1000$, and compare with the classical case where each data point is loaded from a descriptor vector as often as needed.
  3. A classical-shadow approach measures $M$ randomized single-copy observables per state instead. How does the cost scale, and what does it buy?
  4. Which of the two routes would you choose for a first experiment, and what does the choice depend on?

<details>
<summary>Solution</summary>

<p><strong>1.</strong> \(N(N-1)/2\) entries, \(S\) shots each, two copies per shot: \(N(N-1)S\) state preparations.</p>

<p><strong>2.</strong> \(100 \times 99 \times 1000 = 9.9\times10^{6}\) runs of the experiment. For classical data the corresponding number is zero — the descriptor vector is re-read from memory for free, as often as required. This is the input problem running in reverse, and it is the price of the quantum-data setting.</p>

<p><strong>3.</strong> \(NM\) preparations in total, linear rather than quadratic in \(N\), and the shadow features can then be reused for any number of downstream models, kernels and cross-validation folds without re-running the experiment. What it buys is exactly the classical surrogate of Section 5.2: low-weight observables with a sample complexity independent of qubit number. What it gives up is any property that is not a low-weight observable — which is where the provable quantum-data separations live.</p>

<p><strong>4.</strong> Shadows first, for a first experiment, on two grounds: the cost is linear in \(N\) rather than quadratic, and the resulting features are a reusable data set rather than a single model's Gram matrix. Move to coherent processing of the states only when there is a specific quantity of interest that is provably not captured by low-weight observables — because that is the only situation in which the quadratic cost and the co-location requirement buy something a classical post-processor cannot have.</p>

```python
"""Exercise 5. NumPy only."""
N, S, M = 100, 1000, 500
print(f"  fidelity Gram matrix: {N*(N-1)//2:,} entries x {S:,} shots x 2 copies")
print(f"    -> {N*(N-1)*S:,} runs of the experiment")
print(f"  classical descriptors: 0 runs (the vector is re-read from memory)")
print(f"  classical shadows: N x M = {N*M:,} runs, reusable for every downstream model")
print(f"  ratio, coherent kernel against shadows: {N*(N-1)*S/(N*M):,.0f}x")
```

```text
  fidelity Gram matrix: 4,950 entries x 1,000 shots x 2 copies
    -> 9,900,000 runs of the experiment
  classical descriptors: 0 runs (the vector is re-read from memory)
  classical shadows: N x M = 50,000 runs, reusable for every downstream model
  ratio, coherent kernel against shadows: 198x
```

</details>

* * *

## Summary

### Key Takeaways

**1\. Dequantization is two claims, and the weak one is the one that decides research budgets**

  * The theorems are about input models: granting a classical algorithm the same sample-and-query access removed several celebrated exponential speedups.
  * The practical version is an experiment — build the classical model that the quantum construction implicitly describes — and it costs a day.
  * An advantage is a gap, and a gap has two sides. Most reported gaps in this field were closed from the classical side.

**2\. The exponential feature space is real and useless**

  * Haar-random fidelities concentrate as $2^{-n}$; measured on this encoding, the off-diagonal spread falls by a factor of 0.73 to 0.92 per qubit, from 0.309 at two qubits to 0.073 at twelve.
  * Test error goes *up* with register width, 0.036 to 0.143, on an exact simulator with the regularizer re-tuned at every width. On classical data, more qubits is less model.
  * The shot cost is the one thing that does *not* blow up here: an entry is a Bernoulli count, so $S = 100\,k(1-k)/\sigma^2$ at a tenfold margin, which is 241 shots at two qubits and 397 at twelve — flat, because $k$ falls faster than $\sigma$. About $7\times10^5$ shots buys the whole 60-point Gram matrix at twelve qubits. The exponential shot cost belongs to the Haar regime, where $k \approx \sigma \approx 2^{-n}$ and $S \approx 100\,d$; the wrong formula $1/\sigma^2$ manufactures it a register too early.

**3\. Two surrogates, both cheap, both sufficient**

  * Low-weight Pauli truncation: $O(n^w)$ features, each one a classical-shadow observable. At weight 2 it reproduces the Gram matrix badly (67% off-diagonal error) and the test error better than exactly (0.0642 against 0.0680).
  * The encoding's own Fourier basis: the spectrum is provably finite — harmonics 0, 1/2, 1 with $10^{-16}$ above — and 41 classical features score 0.0172, a factor of four better than the quantum kernel and better than the tuned RBF. The total degree is cross-validated: 2 beats 1 and 3 on CV error. This is the one gap in the chapter that a paired bootstrap resolves, $[-0.113, -0.011]$; the quantum kernel's single-split loss to the tuned RBF does not, which is why Section 5.3 runs twenty splits.
  * Neither surrogate is a free lunch, and the caveat is the same for both. The 41 Fourier features are a total-degree-$\le2$ slice of a band-limited basis with $5^4 = 625$ members, growing as $5^d$; the Pauli features are a weight-$\le2$ slice of $4^n$. Each is a bet that the target's structure is low-order, and each fails on a target where it is not.
  * Kernel fidelity and predictive performance are different quantities. Only the second is what an advantage claim is about — and it is a difference of two estimates on 20 rows, so it carries a paired interval or it carries nothing.

**4\. Three true summaries, one experiment**

  * Best quantum split against worst classical split: a $1.15\times$ win. Standard split: a $1.61\times$ loss. Twenty paired splits: a loss in all twenty, mean 0.0843 against 0.0242, a factor of 3.5.
  * The $t = 7.5$ that those twenty splits produce is not usable — overlapping training sets correlate the differences and inflate it (Dietterich). The sign test is: 20 losses out of 20 is $2^{-20}$ under the null, and that is what the conclusion rests on.
  * Holding the quantum arm fixed and varying only the baseline moves the verdict from "$2.25\times$ better" to "$3.50\times$ worse".
  * Absolute gaps shrink with data while the ratio does not move — so quote ratios and curves, not absolute differences at one size.

**5\. Quantum data is the premise that matters**

  * Everything negative here assumes classical data and therefore a loading step charged to the quantum side.
  * With states supplied directly, the loading step vanishes and rigorous separations exist — *sample-complexity* separations, holding against every single-copy measurement strategy rather than merely against known classical algorithms, and not statements about running time — at the price of consuming a state per shot, co-locating the experiment with the processor, and beating a classical-shadow baseline that is stronger than it looks.
  * For materials science the near-term instances are narrow: output of quantum simulation, quantum-probe measurements, and characterizing quantum devices as materials.

**6\. What to take from five chapters of negative results**

  * The kernel view and the frequency view are classical machine learning sharpened by quantum questions, and they transfer.
  * The benchmark discipline — fixed protocol, matched budgets, trivial baselines, paired statistics, every run reported — is the most valuable exportable skill in the course.
  * "It will work when the hardware improves" is answerable: every negative number here came from an exact noiseless simulator.

**Practical implications**

  * Before proposing a quantum model for tabular materials data, build the two surrogates. If either matches, the proposal is finished, and finishing it cheaply is a success.
  * Ask of any advantage claim: how many points, how strong the baseline, how equal the tuning, how many configurations tried, is the data quantum, was a surrogate checked, what were the shots, and is the gap bigger than the label noise.
  * Spend learning time on representation-induced inner products, frequency support, and comparison methodology. Spend device time only when the bottleneck is a device.
  * Watch the quantum-data quadrant, and watch it specifically — not quantum machine learning in general.

[← Chapter 4: Variational Quantum Models](<chapter-4.html>) [Series Top →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The comparisons and the audit exercise in this chapter are measured on, or constructed around, one synthetic 60-point data set; the fictional abstract in Exercise 4 is an invented illustration and refers to no real publication, method or group.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
