import math
import numpy as np
from itertools import product

np.set_printoptions(precision=6, suppress=True)

# The STO-3G parameters for hydrogen: the published basis-set DEFINITION.
EXP_H = np.array([3.42525091, 0.62391373, 0.16885540])
COEF_H = np.array([0.15432897, 0.53532814, 0.44463454])

R_BOHR = 1.4                                    # chosen H-H separation, in bohr
NUC = [(1.0, np.array([0.0, 0.0, 0.0])), (1.0, np.array([0.0, 0.0, R_BOHR]))]
BASIS = [(EXP_H, COEF_H, NUC[0][1]), (EXP_H, COEF_H, NUC[1][1])]
NBF = len(BASIS)

def F0(x):                                      # Boys function, math.erf only
    return 1.0 - x / 3.0 if x < 1e-12 else 0.5 * math.sqrt(math.pi / x) * math.erf(math.sqrt(x))

def Ns(a):                                      # normalization of an s primitive
    return (2.0 * a / math.pi) ** 0.75

def gp(a, A, b, B):                             # Gaussian product theorem
    p = a + b
    return p, (a * A + b * B) / p, math.exp(-a * b / p * float(np.dot(A - B, A - B)))

def s_prim(a, A, b, B):
    p, _, K = gp(a, A, b, B)
    return Ns(a) * Ns(b) * K * (math.pi / p) ** 1.5

def t_prim(a, A, b, B):
    p, _, K = gp(a, A, b, B)
    mu, AB2 = a * b / p, float(np.dot(A - B, A - B))
    return Ns(a) * Ns(b) * mu * (3.0 - 2.0 * mu * AB2) * K * (math.pi / p) ** 1.5

def v_prim(a, A, b, B, Zc, C):
    p, P, K = gp(a, A, b, B)
    return -Ns(a) * Ns(b) * Zc * (2.0 * math.pi / p) * K * F0(p * float(np.dot(P - C, P - C)))

def eri_prim(a, A, b, B, c, C, d, D):
    p, P, Kab = gp(a, A, b, B)
    q, Q, Kcd = gp(c, C, d, D)
    pref = 2.0 * math.pi ** 2.5 / (p * q * math.sqrt(p + q))
    return (Ns(a) * Ns(b) * Ns(c) * Ns(d) * pref * Kab * Kcd
            * F0(p * q / (p + q) * float(np.dot(P - Q, P - Q))))

def contract2(fn):                              # contract a 2-index integral
    M = np.zeros((NBF, NBF))
    for i, (ea, ca, A) in enumerate(BASIS):
        for j, (eb, cb, B) in enumerate(BASIS):
            M[i, j] = sum(wa * wb * fn(a, A, b, B)
                          for a, wa in zip(ea, ca) for b, wb in zip(eb, cb))
    return M

S, T = contract2(s_prim), contract2(t_prim)
V = sum(contract2(lambda a, A, b, B, Z=Z, C=C: v_prim(a, A, b, B, Z, C)) for Z, C in NUC)
H_core = T + V

ERI = np.zeros((NBF,) * 4)
for i, j, k, l in product(range(NBF), repeat=4):
    (ea, ca, A), (eb, cb, B) = BASIS[i], BASIS[j]
    (ec, cc, C), (ed, cd, D) = BASIS[k], BASIS[l]
    ERI[i, j, k, l] = sum(wa * wb * wc * wd * eri_prim(a, A, b, B, c, C, d, D)
                          for a, wa in zip(ea, ca) for b, wb in zip(eb, cb)
                          for c, wc in zip(ec, cc) for d, wd in zip(ed, cd))

E_nuc = NUC[0][0] * NUC[1][0] / float(np.linalg.norm(NUC[0][1] - NUC[1][1]))

for name, M in [("S", S), ("T", T), ("V", V), ("H_core = T + V", H_core)]:
    print(f"{name:16s} = [{M[0,0]:+.6f} {M[0,1]:+.6f} ; {M[1,0]:+.6f} {M[1,1]:+.6f}]")
print(f"\n(00|00) = {ERI[0,0,0,0]:.9f}   (00|11) = {ERI[0,0,1,1]:.9f}")
print(f"(01|01) = {ERI[0,1,0,1]:.9f}   (00|01) = {ERI[0,0,0,1]:.9f}")
print(f"\nS symmetric: {np.allclose(S, S.T)}   "
      f"max |S_ii - 1| = {np.max(np.abs(np.diag(S) - 1.0)):.2e}   "
      f"(ij|kl) = (kl|ij): {np.allclose(ERI, ERI.transpose(2, 3, 0, 1))}")
print(f"nuclear repulsion Z_A Z_B / R = {E_nuc:.9f} hartree")
sv, sc = np.linalg.eigh(S)
X = sc @ np.diag(1.0 / np.sqrt(sv)) @ sc.T      # symmetric orthogonalizer S^(-1/2)
N_OCC = 1                                       # 2 electrons, 2 per spatial orbital

def fock(P):
    F = H_core.copy()
    for m, n in product(range(NBF), repeat=2):
        F[m, n] += float(np.sum(P * (ERI[m, n] - 0.5 * ERI[m, :, n, :])))
    return F

# A deliberately lopsided starting guess: both electrons parked on atom A.
P, E_elec = np.array([[2.0, 0.0], [0.0, 0.0]]), 0.0
print(f"X^T S X = I: {np.allclose(X.T @ S @ X, np.eye(NBF))}   "
      f"overlap eigenvalues = {sv}\n")
print("  iter          E_elec (hartree)        change     max |dP|")
for it in range(1, 51):
    eps, Co = np.linalg.eigh(X.T @ fock(P) @ X)
    C = X @ Co
    P_new = 2.0 * (C[:, :N_OCC] @ C[:, :N_OCC].T)
    E_new = 0.5 * float(np.sum(P_new * (H_core + fock(P_new))))
    dE, dP = E_new - E_elec, float(np.max(np.abs(P_new - P)))
    print(f"  {it:4d}   {E_new:+22.12f}   {dE:+11.3e}   {dP:.3e}")
    P, E_elec = P_new, E_new
    if abs(dE) < 1e-12 and dP < 1e-12:
        break

E_rhf = E_elec + E_nuc
print(f"\nFock hermitian: {np.allclose(fock(P), fock(P).T)}   "
      f"C^T S C = I: {np.allclose(C.T @ S @ C, np.eye(NBF))}   "
      f"orbital energies = {eps}")
print(f"MO coefficients C (columns are the molecular orbitals) =")
print(C)
print(f"\nelectronic energy = {E_elec:+.9f}   nuclear repulsion = {E_nuc:+.9f}")
print(f"TOTAL RHF ENERGY  = {E_rhf:+.9f} hartree")
h_mo = C.T @ H_core @ C
eri_mo = np.einsum('ip,jq,kr,ls,ijkl->pqrs', C, C, C, C, ERI, optimize=True)

print("h_pq =")
print(h_mo)
print(f"(00|00) = {eri_mo[0,0,0,0]:.9f}   (00|11) = {eri_mo[0,0,1,1]:.9f}")
print(f"(01|01) = {eri_mo[0,1,0,1]:.9f}   (11|11) = {eri_mo[1,1,1,1]:.9f}")
print(f"(00|01) = {eri_mo[0,0,0,1]:.9f}   (vanishes by symmetry)")

E_check = 2.0 * h_mo[0, 0] + eri_mo[0, 0, 0, 0] + E_nuc
print(f"\nRHF energy rebuilt from the MO integrals = {E_check:+.9f} hartree")
print(f"agrees with the SCF total: {abs(E_check - E_rhf) < 1e-10}   "
      f"(difference {abs(E_check - E_rhf):.2e})")
N_SO, DIM = 4, 16
Zp = np.array([[1.0, 0.0], [0.0, -1.0]])
LOW = np.array([[0.0, 1.0], [0.0, 0.0]])            # sigma^- = |0><1|

def jw(p):                                          # a_p = Z_0 ... Z_{p-1} sigma^-_p
    op = np.array([[1.0]])
    for q in range(N_SO):
        op = np.kron(op, Zp if q < p else (LOW if q == p else np.eye(2)))
    return op

def so(p, sigma):                                   # 0a, 0b, 1a, 1b -> qubits 0..3
    return 2 * p + sigma

a = [jw(p) for p in range(N_SO)]
ad = [op.T for op in a]
err = max(float(np.max(np.abs(a[p] @ ad[q] + ad[q] @ a[p]
                              - (np.eye(DIM) if p == q else 0.0))))
          for p, q in product(range(N_SO), repeat=2))
print(f"max deviation in the anticommutator (a_p, a_q^dag) = delta_pq : {err:.2e}")

H_q = E_nuc * np.eye(DIM)
for (p, q), sg in product(product(range(2), repeat=2), (0, 1)):
    H_q += h_mo[p, q] * (ad[so(p, sg)] @ a[so(q, sg)])
for (p, q, r, s), (sg, tau) in product(product(range(2), repeat=4),
                                       product((0, 1), repeat=2)):
    H_q += 0.5 * eri_mo[p, q, r, s] * (ad[so(p, sg)] @ ad[so(r, tau)]
                                       @ a[so(s, tau)] @ a[so(q, sg)])

HF, DOUBLE = 0b1100, 0b0011
print(f"H is {H_q.shape[0]}x{H_q.shape[1]}, hermitian: {np.allclose(H_q, H_q.T)}")
print(f"<1100|H|1100> = {H_q[HF, HF]:+.9f}   equals the SCF total: "
      f"{abs(H_q[HF, HF] - E_rhf) < 1e-10}")

evals = np.linalg.eigvalsh(H_q)
print("\nfull 16x16 spectrum (hartree)")
for k in range(0, DIM, 4):
    print("  " + "  ".join(f"{v:+14.9f}" for v in evals[k:k + 4]))

sectors = {}
for i in range(DIM):
    o = [(i >> (N_SO - 1 - q)) & 1 for q in range(N_SO)]
    sectors.setdefault((sum(o), o[0] + o[2] - o[1] - o[3]), []).append(i)

print("\n   N   2*Sz   dim      lowest energy (hartree)")
low, allv = {}, []
for key in sorted(sectors):
    vals = np.linalg.eigvalsh(H_q[np.ix_(sectors[key], sectors[key])])
    low[key] = float(vals[0])
    allv.extend(vals.tolist())
    print(f"  {key[0]:2d}   {key[1]:+4d}   {len(sectors[key]):3d}   {vals[0]:+22.9f}")
print(f"\nsector eigenvalues reproduce the full spectrum: "
      f"{np.allclose(np.sort(np.array(allv)), evals)}")

E_fci = low[(2, 0)]
print(f"\nTOTAL RHF ENERGY   = {E_rhf:+.9f} hartree")
print(f"FCI GROUND ENERGY  = {E_fci:+.9f} hartree")
print(f"CORRELATION ENERGY = {E_fci - E_rhf:+.9f} hartree")
print(f"FCI <= RHF: {E_fci <= E_rhf}   "
      f"FCI is the global minimum of the 16x16: {abs(E_fci - evals[0]) < 1e-12}")

dets = [0b1100, 0b1001, 0b0110, 0b0011]
print("\nthe N=2, Sz=0 block in the basis |1100>, |1001>, |0110>, |0011>")
print(H_q[np.ix_(dets, dets)])
print(f"<1100|H|1001> = {H_q[HF, 0b1001]:+.2e}      "
      f"<1100|H|0011> = {H_q[HF, DOUBLE]:+.9f}")

# How many Pauli strings does this Hamiltonian actually contain?
PAULI = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]], dtype=complex),
         'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.array([[1, 0], [0, -1]], dtype=complex)}
n_terms = 0
for st in product('IXYZ', repeat=N_SO):
    M = np.array([[1.0]], dtype=complex)
    for ch in st:
        M = np.kron(M, PAULI[ch])
    if abs(complex(np.trace(M.conj().T @ H_q)) / DIM) > 1e-12:
        n_terms += 1
print(f"\nnonzero Pauli strings out of the 256 four-qubit strings: {n_terms}")
def ansatz(t):                                  # cos(t/2)|1100> + sin(t/2)|0011>
    psi = np.zeros(DIM)
    psi[HF], psi[DOUBLE] = math.cos(t / 2.0), math.sin(t / 2.0)
    return psi

def energy(t):
    psi = ansatz(t)
    return float(psi @ H_q @ psi)

def grad(t):                                    # parameter-shift rule, exact here
    return 0.5 * (energy(t + math.pi / 2) - energy(t - math.pi / 2))

print("   theta      E(theta) (hartree)")
for t in np.linspace(-1.0, 0.5, 7):
    print(f"  {t:+.4f}   {energy(t):+18.9f}")

theta = 0.0                                     # start at the Hartree-Fock determinant
print("\n step        theta        E(theta) (hartree)     E - E_FCI     dE/dtheta")
for step in range(13):
    if step % 2 == 0:
        print(f"  {step:3d}   {theta:+11.8f}   {energy(theta):+18.9f}   "
              f"{energy(theta) - E_fci:+11.3e}   {grad(theta):+11.3e}")
    theta = theta - grad(theta)

E_vqe, psi = energy(theta), ansatz(theta)
print(f"\noptimal theta = {theta:+.9f} rad")
print(f"VQE ENERGY    = {E_vqe:+.9f} hartree      FCI ENERGY = {E_fci:+.9f} hartree")
print(f"|VQE - FCI|   = {abs(E_vqe - E_fci):.3e} hartree   "
      f"VQE >= FCI: {E_vqe - E_fci > -1e-12}")
print(f"amplitudes: |1100> = {psi[HF]:+.9f}   |0011> = {psi[DOUBLE]:+.9f}   "
      f"HF weight = {psi[HF] ** 2:.9f}")
print(f"\nRHF {E_rhf:+.9f}    FCI {E_fci:+.9f}    VQE {E_vqe:+.9f}    "
      f"correlation {E_fci - E_rhf:+.9f}")
