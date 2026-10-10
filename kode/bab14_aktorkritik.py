"""Bab 14: aktor-kritik satu langkah dan GAE pada lorong licin.

Dipakai bab-bab berikutnya:
  gae(delta, g, lam)          advantage GAE untuk satu trajektori
  entropi(p)                  entropi sebaran diskret
"""
import numpy as np

from bab01_lorong import BENIH
from bab03_mdp import mdp_lorong
from bab06_mc import EPISODE_MINI
from bab07_td import langkah
from bab13_reinforce import J, episode_pi, grad_logpi, softmax_kebijakan
from bab13_reinforce import latih as latih_reinforce


def gae(delta, g, lam):
    A, acc = np.zeros(len(delta)), 0.0
    for t in range(len(delta) - 1, -1, -1):
        acc = delta[t] + g * lam * acc
        A[t] = acc
    return A


def entropi(p):
    p = np.asarray(p, float)
    return float(-(p[p > 0] * np.log(p[p > 0])).sum())


def aktor_kritik(n_ep, alpha, alpha_w, benih, g=0.5):
    m = mdp_lorong()
    rng = np.random.default_rng(benih)
    rng_l = np.random.default_rng(benih + 1)
    theta = np.zeros((5, 2))
    w = np.zeros(5)
    kurva = []
    for _ in range(n_ep):
        ep = episode_pi(softmax_kebijakan(theta), rng, rng_l)
        I = 1.0
        for (s, a, r), (_, _, s2) in zip(ep, langkah(ep)):
            v2 = 0.0 if s2 in (0, 4) else w[s2]
            d = r + g * v2 - w[s]
            w[s] += alpha_w * d
            theta += alpha * I * d * grad_logpi(theta, s, a)
            I *= g
        kurva.append(J(theta, m))
    return np.array(kurva)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    m = mdp_lorong()
    v = m.nilai(np.full((5, 2), 0.5))
    tr = langkah(EPISODE_MINI[1])
    delta = np.array([r + (0 if s2 in (0, 4) else 0.5 * v[s2]) - v[s]
                      for s, r, s2 in tr])
    print("(1) Episode E2 dengan V = V^pi acak, gamma = 1/2:")
    print("    galat TD =", delta)
    for lam in (0.0, 0.5, 1.0):
        print(f"    GAE lambda = {lam}: {gae(delta, 0.5, lam)}")
    print(f"    G_0 - V(2) = {1 - v[2]:.4f}")
    print(f"    entropi (1/2, 1/2) = {entropi([.5, .5]):.4f},"
          f" (0.9, 0.1) = {entropi([.9, .1]):.4f}")

    print("(2) Lorong: J(theta), rata-rata 20 benih, alpha = 0.5")
    print("    episode   REINFORCE+baseline   aktor-kritik")
    k0 = np.mean([latih_reinforce(1000, 0.5, BENIH + u, baseline=True)[0]
                  for u in range(20)], 0)
    k1 = np.mean([aktor_kritik(1000, 0.5, 0.1, BENIH + u)
                  for u in range(20)], 0)
    for e in (10, 50, 100, 300, 1000):
        print(f"    {e:7d} {k0[e - 1]:20.4f} {k1[e - 1]:14.4f}")
