"""Bab 4: persamaan Bellman, operator Bellman, dan kebijakan optimal.

Dipakai bab-bab berikutnya:
  T_pi(mdp, pi, v)       operator Bellman ekspektasi
  T_opt(mdp, v)          operator Bellman optimal
  greedy(mdp, v)         kebijakan greedy terhadap v (seri: aksi kecil)
  v_opt_lp(mdp)          V* dari program linear (pembanding)
"""
import numpy as np
from scipy.optimize import linprog

from bab01_lorong import BENIH
from bab03_mdp import acak, mdp_lorong, semua_det


def T_pi(mdp, pi, v):
    Ppi, rpi = mdp.rantai(pi)
    return np.where(mdp.terminal, 0.0, rpi + mdp.gamma * Ppi @ v)


def T_opt(mdp, v):
    return np.where(mdp.terminal, 0.0, mdp.nilai_q(v).max(axis=1))


def greedy(mdp, v, tol=1e-12):
    q = mdp.nilai_q(v)
    pi = np.zeros_like(q)
    a = np.argmax(q > q.max(axis=1, keepdims=True) - tol, axis=1)
    pi[np.arange(mdp.nS), a] = 1.0
    return pi


def v_opt_lp(mdp):
    """min sum_s v(s) dengan v(s) >= r(s,a) + gamma sum p v(s')."""
    n = mdp.nS
    A, b = [], []
    for s in range(n):
        for a in range(mdp.nA):
            baris = mdp.gamma * mdp.P[s, a].copy()
            baris[s] -= 1.0
            A.append(baris)
            b.append(-mdp.r_sa()[s, a])
    batas = [(0, 0) if mdp.terminal[s] else (None, None) for s in range(n)]
    h = linprog(np.ones(n), A_ub=A, b_ub=b, bounds=batas, method="highs")
    return h.x


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    m = mdp_lorong()
    pi = acak(m)

    print("(1) Q^pi dan A^pi = Q^pi - V^pi, kebijakan acak:")
    print("    petak  Q(kiri) Q(kanan)  V^pi   A(kiri) A(kanan)")
    v_eksak = m.nilai(pi)
    q = m.nilai_q(v_eksak)
    for s in (1, 2, 3):
        a_ = q[s] - v_eksak[s]
        print(f"    {s:4d} {q[s, 0]:8.4f} {q[s, 1]:8.4f} {v_eksak[s]:7.4f}"
              f" {a_[0]:8.4f} {a_[1]:8.4f}")

    print("(2) Evaluasi kebijakan acak: v_{k+1} = T_pi v_k, v_0 = 0")
    print("     k   v(1)    v(2)    v(3)    galat     rasio")
    v = np.zeros(5)
    gal_lama = np.abs(v - v_eksak).max()
    for k in range(1, 13):
        v = T_pi(m, pi, v)
        gal = np.abs(v - v_eksak).max()
        if k <= 4 or k % 4 == 0:
            print(f"    {k:2d}" + "".join(f" {x:7.4f}" for x in v[1:4])
                  + f"  {gal:.2e}  {gal / gal_lama:.3f}")
        gal_lama = gal
    print("    eksak" + "".join(f" {x:7.4f}" for x in v_eksak[1:4]))

    print("(3) V* tiga cara, Q*, dan kebijakan greedy:")
    for g in (0.5, 0.9):
        mg = mdp_lorong(gamma=g)
        brute = max((mg.nilai(p)[1:4] for _, p in semua_det(mg)),
                    key=np.sum)
        lp = v_opt_lp(mg)[1:4]
        v = np.zeros(5)
        for _ in range(200):
            v = T_opt(mg, v)
        pg = greedy(mg, v)
        kode = "".join("-+"[int(pg[s, 1])] for s in (1, 2, 3))
        q = mg.nilai_q(v)
        print(f"    gamma = {g}")
        for nama, x in (("coba 8", brute), ("LP", lp),
                        ("iterasi T*", v[1:4]), ("Q*(s, kiri)", q[1:4, 0]),
                        ("Q*(s, kanan)", q[1:4, 1])):
            print(f"      {nama:13s}" + "".join(f" {y:8.4f}" for y in x))
        print(f"      greedy terhadap V*: {kode}")

    print("(4) Kontraksi: maks ||T u - T w|| / ||u - w||, 20 000 pasangan")
    rng = np.random.default_rng(BENIH)
    U = rng.normal(0, 10, (20_000, 5))
    W = rng.normal(0, 10, (20_000, 5))
    U[:, [0, 4]] = 0
    W[:, [0, 4]] = 0
    for nama, f in (("T_pi (acak)", lambda x: T_pi(m, pi, x)),
                    ("T*", lambda x: T_opt(m, x))):
        r = [np.abs(f(u) - f(w)).max() / np.abs(u - w).max()
             for u, w in zip(U, W)]
        print(f"    {nama:12s} maks rasio = {max(r):.4f} (gamma = 0.5)")
