"""Bab 6: importance sampling biasa dan berbobot, dan MC kontrol pada
grid es.
"""
import numpy as np

from bab01_lorong import ACAK, BENIH, OPTIMUM, Lorong, jalankan_episode
from bab05_dp import iterasi_kebijakan, mdp_grid
from bab06_mc import EPISODE_MINI, mc_kontrol, rasio_is


def episode_paralel(b, pi, n, rng, rng_l, g=0.5, licin=0.25):
    """n episode lorong sekaligus dari petak 2 dengan perilaku b.

    Mengembalikan G_0 dan rho_{0:T-1} untuk kebijakan sasaran pi."""
    s = np.full(n, 2)
    G = np.zeros(n)
    rho = np.ones(n)
    diskon = np.ones(n)
    jalan = np.ones(n, bool)
    while jalan.any():
        i = np.where(jalan)[0]
        a = (rng.random(len(i)) < b[s[i], 1]).astype(int)
        rho[i] *= pi[s[i], a] / b[s[i], a]
        arah = np.where(a == 1, 1, -1)
        balik = (s[i] == 2) & (rng_l.random(len(i)) < licin)
        s[i] = s[i] + np.where(balik, -arah, arah)
        r = np.where(s[i] == 0, 2.0, np.where(s[i] == 4, 8.0, 0.0))
        G[i] += diskon[i] * r
        diskon[i] *= g
        jalan[i] = (s[i] != 0) & (s[i] != 4)
    return G, rho


def is_dari_larik(G, rho, berbobot=False):
    if berbobot:
        return (rho * G).sum() / rho.sum() if rho.sum() > 0 else 0.0
    return (rho * G).mean()


def is_estimasi(episode, pi, b, g=0.5, berbobot=False):
    """Taksiran V^pi(S_0) dari episode yang dibangkitkan b."""
    rho = np.array([rasio_is(ep, pi, b) for ep in episode])
    G = np.array([sum(g ** t * r for t, (_, _, r) in enumerate(ep))
                  for ep in episode])
    if berbobot:
        return (rho * G).sum() / rho.sum() if rho.sum() > 0 else 0.0
    return (rho * G).mean()


if __name__ == "__main__":
    print("(4) Off-policy: sasaran -++ (V(2) = 3.25), perilaku acak")
    for nama, bb in (("biasa", False), ("berbobot", True)):
        print(f"    episode mini, IS {nama}:"
              f" {is_estimasi(EPISODE_MINI, OPTIMUM, ACAK, berbobot=bb):.4f}")
    print("      n   rata IS biasa  RMSE   rata berbobot  RMSE")
    rng = np.random.default_rng(BENIH)
    rng_l = np.random.default_rng(BENIH + 1)
    ulang, nmaks = 2000, 1000
    G, rho = episode_paralel(ACAK, OPTIMUM, ulang * nmaks, rng, rng_l)
    G, rho = G.reshape(ulang, nmaks), rho.reshape(ulang, nmaks)
    for n in (10, 100, 1000):
        baris = f"    {n:4d}"
        for bb in (False, True):
            est = np.array([is_dari_larik(G[u, :n], rho[u, :n], bb)
                            for u in range(ulang)])
            gal = est - 3.25
            baris += f" {est.mean():11.4f} {np.sqrt((gal ** 2).mean()):7.4f}"
        print(baris)
    print(f"    bagian episode dengan rho > 0 = {(rho > 0).mean():.4f},"
          f" var(rho G) = {(rho * G).var():.2f}")

    print("(5) MC kontrol pada grid es, gamma = 0.99, eps = 0.05:")
    print("    V(awal) kebijakan greedy; optimal = 0.5938")
    print("    episode    Q0 = 0    Q0 = 1")
    mg = mdp_grid(gamma=0.99)
    catat = (1000, 5000, 20000)
    k0 = mc_kontrol(mg, max(catat), eps=0.05, catat=catat, q0=0.0)[1]
    k1 = mc_kontrol(mg, max(catat), eps=0.05, catat=catat, q0=1.0)[1]
    for n, a, b in zip(catat, k0, k1):
        print(f"    {n:7d} {a + 0.0:9.4f} {b:9.4f}")
