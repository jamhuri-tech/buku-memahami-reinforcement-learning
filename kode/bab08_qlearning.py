"""Bab 8: SARSA, Q-learning, expected SARSA, dan double Q-learning.

Dipakai bab-bab berikutnya:
  Tebing                    cliff walking 4 x 12 (gaya Gymnasium)
  pilih_eps(Q, s, eps, rng) aksi epsilon-greedy, seri dipecah acak
  sasaran(metode, ...)      sasaran TD SARSA / Q / expected SARSA
  latih(env, metode, ...)   kontrol TD tabular; return per episode
"""
import numpy as np

from bab01_lorong import BENIH, Lorong

KIRI, KANAN = 0, 1


class Tebing:
    """Grid 4 x 12. Awal (3, 0), tujuan (3, 11), tebing (3, 1..10).
    Setiap langkah -1; jatuh ke tebing -100 dan kembali ke awal."""
    nS, nA = 48, 4
    ARAH = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # atas kanan bawah kiri

    def reset(self):
        self.s = 36
        return self.s

    def step(self, a):
        i, j = divmod(self.s, 12)
        di, dj = self.ARAH[a]
        i, j = min(max(i + di, 0), 3), min(max(j + dj, 0), 11)
        if i == 3 and 1 <= j <= 10:
            self.s = 36
            return self.s, -100.0, False
        self.s = i * 12 + j
        return self.s, -1.0, self.s == 47


class LorongQ:
    """Lorong licin dengan keadaan 0..4 untuk tabel Q (5 x 2)."""
    nS, nA = 5, 2

    def __init__(self, benih=BENIH + 1):
        self.env = Lorong(benih=benih)

    def reset(self):
        return self.env.reset()

    def step(self, a):
        return self.env.step(a)


def pilih_eps(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def peluang_eps(q, eps):
    maks = q == q.max()
    return eps / len(q) + (1 - eps) * maks / maks.sum()


def sasaran(metode, Q, r, s2, a2, selesai, g, eps):
    if selesai:
        return r
    if metode == "sarsa":
        return r + g * Q[s2, a2]
    if metode == "q":
        return r + g * Q[s2].max()
    return r + g * peluang_eps(Q[s2], eps) @ Q[s2]      # expected SARSA


def latih(env, metode, n_ep, alpha=0.5, eps=0.1, g=1.0, benih=BENIH,
          Q=None, alpha_n=None, maks=10_000):
    """Mengembalikan Q dan jumlah imbalan (tanpa diskon) per episode."""
    rng = np.random.default_rng(benih)
    Q = np.zeros((env.nS, env.nA)) if Q is None else Q
    N = np.zeros_like(Q)
    total = []
    for _ in range(n_ep):
        s = env.reset()
        a = pilih_eps(Q, s, eps, rng)
        jumlah = 0.0
        for _ in range(maks):
            s2, r, selesai = env.step(a)
            jumlah += r
            a2 = pilih_eps(Q, s2, eps, rng)
            N[s, a] += 1
            lr = N[s, a] ** -alpha_n if alpha_n else alpha
            Q[s, a] += lr * (sasaran(metode, Q, r, s2, a2, selesai, g, eps)
                             - Q[s, a])
            s, a = s2, a2
            if selesai:
                break
        total.append(jumlah)
    return Q, np.array(total)


def jalur_greedy(Q, env, maks=100):
    s, jalur = env.reset(), []
    for _ in range(maks):
        jalur.append(s)
        s, r, selesai = env.step(int(Q[s].argmax()))
        if selesai:
            jalur.append(s)
            break
    return jalur


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("(1) Q-learning pada lorong licin, gamma = 1/2, eps = 0.2")
    print("    maks |Q - Q*| sesudah n episode, laju langkah 1/N^w")
    print("        n       w = 1    w = 0.6")
    Qs = np.array([[0, 0], [2, 1.625], [1.75, 3.25], [1.625, 8], [0, 0]])
    for n_ep in (100, 1000, 10000):
        baris = f"    {n_ep:5d}"
        for w in (1.0, 0.6):
            Q, _ = latih(LorongQ(), "q", n_ep, eps=0.2, g=0.5, alpha_n=w)
            baris += f" {np.abs(Q - Qs).max():10.4f}"
        print(baris)
    print("    greedy (w = 0.6):", "".join(
        "-+"[int(Q[s].argmax())] for s in (1, 2, 3)))

    print("(2) Tebing 4 x 12, eps = 0.1, alpha = 0.5, 50 ulangan:")
    print("    metode          imbalan per episode  jalur greedy")
    print("                    ep 1-50  ep 451-500   (langkah)")
    for nama, met in (("SARSA", "sarsa"), ("Q-learning", "q"),
                      ("expected SARSA", "esarsa")):
        kurva, panjang = [], []
        for u in range(50):
            Q, tot = latih(Tebing(), met, 500, benih=BENIH + u)
            kurva.append(tot)
            panjang.append(len(jalur_greedy(Q, Tebing())) - 1)
        kurva = np.array(kurva)
        print(f"    {nama:15s} {kurva[:, :50].mean():9.1f}"
              f" {kurva[:, 450:].mean():9.1f}"
              f" {np.median(panjang):12.0f}")
