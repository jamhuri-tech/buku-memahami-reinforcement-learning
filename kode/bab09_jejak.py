"""Bab 9: return n-langkah, lambda-return, dan TD(lambda).

Dipakai bab-bab berikutnya:
  return_n(imb, V, s_akhir, g, n)    G_{t:t+n} untuk semua t
  return_lambda(imb, V, ...)         lambda-return untuk semua t
  JalanAcak(n)                       random walk n petak, imbalan -1/+1
  td_n_episode(...), td_lambda_episode(...)
"""
import numpy as np

from bab01_lorong import BENIH
from bab06_mc import EPISODE_MINI
from bab07_td import langkah


def return_n(trans, V, g, n):
    """trans: (s, r, s2); V terminal 0. Kembalikan G_{t:t+n}."""
    T = len(trans)
    hasil = []
    for t in range(T):
        G = 0.0
        for k in range(t, min(t + n, T)):
            G += g ** (k - t) * trans[k][1]
        if t + n < T:
            G += g ** n * V[trans[t + n][0]]
        hasil.append(G)
    return hasil


def return_lambda(trans, V, g, lam):
    T = len(trans)
    hasil = []
    for t in range(T):
        Gn = [return_n(trans, V, g, n)[t] for n in range(1, T - t + 1)]
        G = (1 - lam) * sum(lam ** (n - 1) * Gn[n - 1]
                            for n in range(1, T - t)) + lam ** (T - t - 1) * Gn[-1]
        hasil.append(G)
    return hasil


def jejak_episode(trans, V, g, lam, alpha, ganti=False, offline=True):
    """TD(lambda) arah mundur. Mengembalikan V baru dan jejak tiap t."""
    V = V.copy()
    e = np.zeros_like(V)
    inc = np.zeros_like(V)
    rekam = []
    for s, r, s2 in trans:
        v2 = 0.0 if s2 in (0, len(V) - 1) else V[s2]
        d = r + g * v2 - V[s]
        e *= g * lam
        e[s] = 1.0 if ganti else e[s] + 1.0
        rekam.append(e.copy())
        if offline:
            inc += alpha * d * e
        else:
            V += alpha * d * e
    return (V + inc if offline else V), rekam


class JalanAcak:
    """Petak 0..n+1; 0 dan n+1 terminal (imbalan -1 dan +1); mulai di
    tengah; setiap langkah kiri atau kanan dengan peluang sama."""

    def __init__(self, n=19):
        self.n = n
        self.v_benar = np.arange(-n + 1, n + 1, 2) / (n + 1)

    def episode(self, rng):
        s = (self.n + 1) // 2
        trans = []
        while 0 < s <= self.n:
            s2 = s + (1 if rng.random() < 0.5 else -1)
            r = 1.0 if s2 == self.n + 1 else (-1.0 if s2 == 0 else 0.0)
            trans.append((s, r, s2))
            s = s2
        return trans


def td_n_online(trans, V, n, alpha):
    """n-step TD daring (Sutton dan Barto 2018, algoritma 7.1), g = 1."""
    T = len(trans)
    for tau in range(T):
        G = sum(trans[k][1] for k in range(tau, min(tau + n, T)))
        if tau + n < T:
            G += V[trans[tau + n][0]]
        s = trans[tau][0]
        V[s] += alpha * (G - V[s])


def td_lambda_online(trans, V, lam, alpha):
    e = np.zeros_like(V)
    for s, r, s2 in trans:
        v2 = 0.0 if s2 in (0, len(V) - 1) else V[s2]
        d = r + v2 - V[s]
        e *= lam
        e[s] += 1.0
        V += alpha * d * e


def galat_rata(metode, par, alpha, ulang=50, n_ep=10, benih=BENIH):
    env = JalanAcak()
    rng = np.random.default_rng(benih)
    total = 0.0
    for _ in range(ulang):
        V = np.zeros(env.n + 2)
        for _ in range(n_ep):
            tr = env.episode(rng)
            if metode == "n":
                td_n_online(tr, V, par, alpha)
            else:
                td_lambda_online(tr, V, par, alpha)
            total += np.sqrt(((V[1:-1] - env.v_benar) ** 2).mean())
    return total / (ulang * n_ep)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    V = np.array([0, 0, 0, 4, 0.])
    tr = langkah(EPISODE_MINI[1])
    print("(1) Episode E2, V = (0, 0, 4), gamma = 1/2:")
    for n in (1, 2, 3, 4):
        print(f"    n = {n}: G_t:t+n = {np.round(return_n(tr, V, 0.5, n), 4)}")
    print("    G^lambda (lambda = 1/2) =",
          np.round(return_lambda(tr, V, 0.5, 0.5), 4))
    for ganti in (False, True):
        Vb, rek = jejak_episode(tr, V, 0.5, 0.5, 1.0, ganti=ganti)
        print(f"    jejak {'ganti' if ganti else 'akumulasi'}:"
              f" e(2) = {[round(e[2], 4) for e in rek]}")
        print(f"      kenaikan offline V(2), alpha = 1: {Vb[2] - V[2]:.4f}")

    alphas = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    print("(2) Jalan acak 19 petak: RMS rata-rata 10 episode, 50 ulangan")
    print("    alpha    " + "".join(f" {a:6}" for a in alphas))
    for metode, pars in (("n", (1, 2, 4, 8, 16)),
                         ("l", (0.0, 0.4, 0.8, 0.9, 1.0))):
        for p in pars:
            nama = f"n = {p}" if metode == "n" else f"lam = {p}"
            with np.errstate(all="ignore"):
                hasil = [galat_rata(metode, p, a) for a in alphas]
            print(f"    {nama:9s}" + "".join(
                f" {h:6.3f}" if h < 2 else "   div" for h in hasil))
