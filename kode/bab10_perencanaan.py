"""Bab 10: Dyna-Q dan MCTS (UCT).

Dipakai bab-bab berikutnya:
  Labirin            labirin deterministik 6 x 9 (gaya Gymnasium)
  dyna_q(...)        Dyna-Q tabular; langkah per episode
  uct(...)           pencarian pohon Monte Carlo dengan UCB1
"""
import math

import numpy as np

from bab01_lorong import BENIH, dinamika

PETA = ["...X....G",
        "...X..X..",
        "S..X..X..",
        "......X..",
        "...X.....",
        "........."]
ARAH = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class Labirin:
    nA = 4

    def __init__(self, peta=PETA):
        self.peta = peta
        self.n, self.m = len(peta), len(peta[0])
        self.nS = self.n * self.m
        for i, baris in enumerate(peta):
            for j, c in enumerate(baris):
                if c == "S":
                    self.awal = i * self.m + j
                if c == "G":
                    self.tujuan = i * self.m + j

    def reset(self):
        self.s = self.awal
        return self.s

    def langkah_model(self, s, a):
        i, j = divmod(s, self.m)
        i2, j2 = i + ARAH[a][0], j + ARAH[a][1]
        if not (0 <= i2 < self.n and 0 <= j2 < self.m) \
                or self.peta[i2][j2] == "X":
            i2, j2 = i, j
        s2 = i2 * self.m + j2
        return s2, (1.0 if s2 == self.tujuan else 0.0), s2 == self.tujuan

    def step(self, a):
        self.s, r, selesai = self.langkah_model(self.s, a)
        return self.s, r, selesai


def _pilih(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    return int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))


def dyna_q(env, n_rencana, n_ep=50, alpha=0.1, g=0.95, eps=0.1,
           benih=BENIH):
    rng = np.random.default_rng(benih)
    Q = np.zeros((env.nS, env.nA))
    model = {}
    langkah_ep = []
    for _ in range(n_ep):
        s, k = env.reset(), 0
        while True:
            a = _pilih(Q, s, eps, rng)
            s2, r, selesai = env.step(a)
            k += 1
            Q[s, a] += alpha * (r + (0 if selesai else g * Q[s2].max())
                                - Q[s, a])
            model[(s, a)] = (r, s2, selesai)
            kunci = list(model)
            for _ in range(n_rencana):
                ps, pa = kunci[rng.integers(len(kunci))]
                pr, ps2, pselesai = model[(ps, pa)]
                Q[ps, pa] += alpha * (pr + (0 if pselesai else
                                            g * Q[ps2].max()) - Q[ps, pa])
            s = s2
            if selesai:
                break
        langkah_ep.append(k)
    return Q, np.array(langkah_ep)


class Simpul:
    def __init__(self):
        self.N = np.zeros(2)
        self.W = np.zeros(2)
        self.anak = {}


def _rollout(s, rng, g, licin):
    G, diskon = 0.0, 1.0
    while s not in (0, 4):
        a = int(rng.integers(2))
        s, r = _transisi(s, a, rng, licin)
        G += diskon * r
        diskon *= g
    return G


def _transisi(s, a, rng, licin):
    u = rng.random()
    for p, s2, r in dinamika(s, a, licin):
        if u < p:
            return s2, r
        u -= p
    return s2, r


def uct(s0, n_sim, c=2.0, g=0.5, licin=0.25, benih=BENIH):
    """UCT pada lorong licin dengan simulator dinamika. Mengembalikan
    rata-rata Q dan banyaknya kunjungan setiap aksi di akar."""
    rng = np.random.default_rng(benih)
    akar = Simpul()

    def simulasi(simpul, s):
        if s in (0, 4):
            return 0.0
        n = simpul.N.sum()
        if (simpul.N == 0).any():
            a = int(np.flatnonzero(simpul.N == 0)[0])
        else:
            skor = simpul.W / simpul.N + c * np.sqrt(np.log(n) / simpul.N)
            a = int(skor.argmax())
        s2, r = _transisi(s, a, rng, licin)
        if simpul.N[a] == 0:
            G = r + g * _rollout(s2, rng, g, licin)
        else:
            anak = simpul.anak.setdefault((a, s2), Simpul())
            G = r + g * simulasi(anak, s2)
        simpul.N[a] += 1
        simpul.W[a] += G
        return G

    for _ in range(n_sim):
        simulasi(akar, s0)
    return akar.W / np.maximum(akar.N, 1), akar.N


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("(1) Labirin 6 x 9: langkah per episode, rata-rata 30 ulangan")
    print("    rencana n   ep 1    ep 2    ep 5   ep 10   ep 50")
    for n in (0, 5, 50):
        L = np.mean([dyna_q(Labirin(), n, benih=BENIH + u)[1]
                     for u in range(30)], axis=0)
        print(f"    {n:9d}" + "".join(f" {L[e - 1]:7.1f}"
                                       for e in (1, 2, 5, 10, 50)))

    print("(2) UCT dari petak 2, gamma = 1/2 (Q* = 1.75 kiri, 3.25 kanan)")
    print("    simulasi   Q(kiri)  Q(kanan)  N(kiri)  N(kanan)")
    for n in (10, 100, 1000, 10000):
        q, N = uct(2, n)
        print(f"    {n:8d} {q[0]:9.4f} {q[1]:9.4f} {N[0]:8.0f} {N[1]:9.0f}")
