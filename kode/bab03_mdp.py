"""Bab 3: MDP tabular, rantai Markov sebuah kebijakan, dan pembentukan
imbalan.

Dipakai bab-bab berikutnya:
  MDP                    kelas MDP tabular: P[s, a, s'], R[s, a, s']
  mdp_lorong(...)        lorong licin sebagai MDP lima keadaan
  MDP.r_sa()             imbalan harapan r(s, a)
  MDP.rantai(pi)         P^pi dan r^pi (keadaan terminal menyerap)
  MDP.nilai(pi)          V^pi dari (I - gamma P^pi) v = r^pi (Bab 4)
  semua_det(mdp)         semua kebijakan deterministik
"""
import itertools

import numpy as np

from bab01_lorong import BENIH, GAMMA, IMBALAN, LICIN, dinamika


class MDP:
    """MDP berhingga. Keadaan terminal menyerap dengan imbalan 0."""

    def __init__(self, P, R, gamma, terminal=(), awal=0, nama_aksi=None):
        self.P = np.asarray(P, float)          # (nS, nA, nS)
        self.R = np.asarray(R, float)          # (nS, nA, nS)
        self.gamma = gamma
        self.terminal = np.zeros(self.P.shape[0], bool)
        self.terminal[list(terminal)] = True
        self.awal = awal
        self.nS, self.nA = self.P.shape[:2]
        self.nama_aksi = nama_aksi
        assert np.allclose(self.P.sum(axis=2), 1)

    def r_sa(self):
        """r(s, a) = sum_s' p(s' | s, a) r(s, a, s')."""
        return (self.P * self.R).sum(axis=2)

    def rantai(self, pi):
        """P^pi[s, s'] dan r^pi[s] untuk kebijakan pi[s, a]."""
        Ppi = np.einsum("sa,sat->st", pi, self.P)
        rpi = (pi * self.r_sa()).sum(axis=1)
        return Ppi, rpi

    def nilai(self, pi, gamma=None):
        g = self.gamma if gamma is None else gamma
        Ppi, rpi = self.rantai(pi)
        n = ~self.terminal                     # keadaan tak-terminal
        v = np.zeros(self.nS)
        A = np.eye(n.sum()) - g * Ppi[np.ix_(n, n)]
        v[n] = np.linalg.solve(A, rpi[n])
        return v

    def nilai_q(self, v, gamma=None):
        """Q(s, a) = r(s, a) + gamma sum_s' p(s' | s, a) v(s')."""
        g = self.gamma if gamma is None else gamma
        return self.r_sa() + g * self.P @ v


def mdp_lorong(licin=LICIN, gamma=GAMMA, imbalan=None):
    """Lorong licin: keadaan 0..4, aksi 0 = kiri, 1 = kanan."""
    imbalan = IMBALAN if imbalan is None else imbalan
    P = np.zeros((5, 2, 5))
    R = np.zeros((5, 2, 5))
    for s in range(5):
        for a in range(2):
            if s in (0, 4):
                P[s, a, s] = 1.0
                continue
            for p, s2, _ in dinamika(s, a, licin):
                P[s, a, s2] += p
                R[s, a, s2] = imbalan.get(s2, 0.0)
    return MDP(P, R, gamma, terminal=(0, 4), awal=2,
               nama_aksi=("kiri", "kanan"))


def semua_det(mdp, keadaan=(1, 2, 3)):
    """Pasangan (kode, pi) untuk semua kebijakan deterministik."""
    for aksi in itertools.product(range(mdp.nA), repeat=len(keadaan)):
        pi = np.zeros((mdp.nS, mdp.nA))
        pi[:, 0] = 1.0
        for s, a in zip(keadaan, aksi):
            pi[s] = 0.0
            pi[s, a] = 1.0
        yield "".join("-+"[a] for a in aksi), pi


def acak(mdp):
    return np.full((mdp.nS, mdp.nA), 1.0 / mdp.nA)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    m = mdp_lorong()
    pi = acak(m)
    Ppi, rpi = m.rantai(pi)
    T = ~m.terminal
    Pt = Ppi[np.ix_(T, T)]

    print("(1) Kebijakan acak: sebaran S_t dari petak 2")
    print("     t   P(0)    P(1)    P(2)    P(3)    P(4)   masih jalan")
    d = np.zeros(5)
    d[2] = 1.0
    for t in range(7):
        print(f"    {t:2d}" + "".join(f" {x:7.4f}" for x in d)
              + f" {d[1:4].sum():9.4f}")
        d = d @ Ppi
    Nf = np.linalg.inv(np.eye(3) - Pt)
    print("    harapan panjang episode dari petak 1, 2, 3 =",
          Nf.sum(axis=1))
    rng = np.random.default_rng(BENIH)
    panjang = []
    for _ in range(100_000):
        s, n = 2, 0
        while s not in (0, 4):
            s = rng.choice(5, p=Ppi[s])
            n += 1
        panjang.append(n)
    print(f"    simulasi 100 000 episode: rata-rata {np.mean(panjang):.4f},"
          f" maks {max(panjang)}")

    print("(2) Deret sum_t gamma^t (P^pi)^t r^pi, kebijakan acak:")
    v_eksak = m.nilai(pi)[1:4]
    print("    eksak =", v_eksak)
    jumlah = np.zeros(3)
    suku = rpi[1:4].copy()
    for K in range(1, 33):
        jumlah += suku
        suku = m.gamma * Pt @ suku
        if K in (1, 2, 4, 8, 16, 32):
            gal = np.abs(jumlah - v_eksak).max()
            print(f"    K = {K:2d}  jumlah = {jumlah}  galat = {gal:.1e}")
    rho = np.abs(np.linalg.eigvals(Pt)).max()
    print(f"    jari-jari spektral P = {rho:.4f},"
          f" gamma * rho = {m.gamma * rho:.4f}")
