import numpy as np

# The TOY Hamiltonian. Round teaching coefficients, no units, no molecule.
# What is realistic is the STRUCTURE: a weighted sum of Pauli strings.
I2, X, Z = np.eye(2), np.array([[0., 1.], [1., 0.]]), np.array([[1., 0.], [0., -1.]])
c1, c2, c3, c4 = 0.5, 0.5, 0.25, 0.30
terms = [(c1, "Z0", np.kron(Z, I2)), (c2, "Z1", np.kron(I2, Z)),
         (c3, "Z0Z1", np.kron(Z, Z)), (c4, "X0X1", np.kron(X, X))]
H = sum(coeff * op for coeff, _, op in terms)

# The classical reference, affordable only because this is a 4x4.
eigvals = np.linalg.eigvalsh(H)
E_exact = float(eigvals[0])
print("Exact diagonalization (numpy.linalg.eigvalsh)")
print("  spectrum      = " + ", ".join(f"{v:+.9f}" for v in eigvals))
print(f"  ground energy = {E_exact:.9f}\n")

# Ansatz:  |psi(theta)> = CNOT . (Ry(theta0) (x) Ry(theta1)) |00>
CNOT = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                 [0., 0., 0., 1.], [0., 0., 1., 0.]])


def ry(t):
    c, s = np.cos(t / 2.0), np.sin(t / 2.0)
    return np.array([[c, -s], [s, c]])


def ansatz_state(theta):
    psi = np.array([1.0, 0.0, 0.0, 0.0])                    # |00>
    return CNOT @ (np.kron(ry(theta[0]), ry(theta[1])) @ psi)


def energy(theta):
    psi = ansatz_state(theta)
    return float(psi @ H @ psi)


# Parameter-shift gradient. Each angle sits in exactly one Ry gate whose
# generator has eigenvalues +-1/2, so the rule is EXACT, not approximate.
def parameter_shift_gradient(theta):
    grad = np.zeros_like(theta)
    for k in range(len(theta)):
        shift = np.zeros_like(theta)
        shift[k] = np.pi / 2.0
        grad[k] = 0.5 * (energy(theta + shift) - energy(theta - shift))
    return grad


probe, h = np.array([0.7, -1.3]), 1e-6
fd = np.array([(energy(probe + h * e) - energy(probe - h * e)) / (2 * h) for e in np.eye(2)])
ps = parameter_shift_gradient(probe)
print("Parameter shift vs central finite difference at theta = (0.70, -1.30)")
print(f"  parameter shift   = [{ps[0]:+.9f}, {ps[1]:+.9f}]")
print(f"  finite difference = [{fd[0]:+.9f}, {fd[1]:+.9f}]")
print(f"  max abs deviation = {np.max(np.abs(ps - fd)):.2e}")
# The hybrid loop: quantum energy evaluations inside classical descent.
def descend(theta0, label, n_steps=400, learning_rate=0.25):
    theta = np.array(theta0, dtype=float)
    print(label + "\n  step       theta0       theta1         energy    |gradient|")
    for step in range(n_steps + 1):
        g = parameter_shift_gradient(theta)
        if step % 200 == 0:
            print(f"  {step:4d}  {theta[0]:+10.6f}  {theta[1]:+10.6f}  "
                  f"{energy(theta):+12.9f}  {np.linalg.norm(g):.3e}")
        theta = theta - learning_rate * g
    print(f"  converged energy = {energy(theta):.9f}   (exact = {E_exact:.9f})\n")
    return theta


# Run A: an arbitrary starting point. The optimizer behaves perfectly.
descend([0.30, 0.90], "Run A: descent from an arbitrary start")

# Run B: seed the same descent with a coarse scan first.
grid = np.linspace(0.0, 2.0 * np.pi, 13)
best = min(((energy(np.array([a, b])), a, b) for a in grid for b in grid))
print(f"Grid scan over {len(grid)}x{len(grid)} angles: "
      f"theta = ({best[1]:.6f}, {best[2]:.6f}), energy = {best[0]:.9f}\n")
theta = descend([best[1], best[2]], "Run B: descent seeded by the grid scan")
print(f"  E_vqe - E_exact = {energy(theta) - E_exact:+.3e}   (machine epsilon)\n")

# What the device would actually have to measure, term by term.
psi_opt = ansatz_state(theta)
print("Pauli-term expectation values at the optimum")
for coeff, label, op in terms:
    ev = float(psi_opt @ op @ psi_opt)
    print(f"  <{label:5s}> = {ev:+.9f}   contribution = {coeff * ev:+.9f}")
print()

# Shot noise: the device samples the +-1 eigenvalue and averages.
rng = np.random.default_rng(2026)
probs = psi_opt ** 2                                # Z-basis outcome probabilities
z0 = np.array([+1.0, +1.0, -1.0, -1.0])             # Z0 on |00>,|01>,|10>,|11>
exact_z0 = float(probs @ z0)
spread = np.sqrt(1.0 - exact_z0 ** 2)               # single-shot standard deviation
print(f"Finite-shot estimation of <Z0>   (exact = {exact_z0:+.9f}, "
      f"single-shot sd = {spread:.6f})")
print("     shots    mean of 400 runs    sd of 400 runs    sd/sqrt(N) predicted    ratio")
previous = None
for n_shots in [100, 400, 1600, 6400]:
    est = np.array([float(np.mean(z0[rng.choice(4, size=n_shots, p=probs)]))
                    for _ in range(400)])
    sd = est.std(ddof=1)
    ratio = "    -" if previous is None else f"{sd / previous:.3f}"
    print(f"  {n_shots:8d}  {est.mean():+17.6f}  {sd:16.6f}  {spread / np.sqrt(n_shots):22.6f}  {ratio:>7s}")
    previous = sd
