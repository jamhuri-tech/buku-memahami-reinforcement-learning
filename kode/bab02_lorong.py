"""Bab 2: tiga kebijakan lorong licin sebagai tiga lengan bandit.

Satu tarikan = satu episode dari petak 2 dengan kebijakan itu. Return
dibangkitkan dari sebarannya yang eksak (Contoh Soal 1.4), setara
dengan menjalankan Lorong satu episode, lalu dibagi 4 supaya berada
di [0, 1]. Lengan: -++ (rata-rata 13/4), +++ (16/5), --+ (7/4).
"""
import numpy as np

from bab01_lorong import (KIRI, KANAN, Lorong, jalankan_episode,
                          kebijakan_det, nilai_eksak, return_)
from bab02_bandit import BENIH, simulasi

NAMA = ("-++", "+++", "--+")
EKSAK = np.array([13 / 4, 16 / 5, 7 / 4])


def tarik(a, rng):
    """Return episode dibagi 4 untuk lengan a (vektor)."""
    u = rng.random(len(a))
    g = np.where(a == 0, np.where(u < 0.75, 4.0, 1.0),
                 np.where(u < 0.75, 1.0, 4.0))
    k = rng.geometric(0.75, size=len(a)) - 1     # banyaknya tergelincir
    g = np.where(a == 1, 4.0 * 0.25 ** k, g)
    return g / 4


if __name__ == "__main__":
    # sebaran eksak = sebaran return episode Lorong
    rng = np.random.default_rng(BENIH)
    env = Lorong(benih=BENIH + 1)
    for i, aksi in enumerate(((KIRI, KANAN, KANAN), (KANAN,) * 3,
                              (KIRI, KIRI, KANAN))):
        pi = kebijakan_det(aksi)
        assert abs(nilai_eksak(pi)[1] - EKSAK[i]) < 1e-12
        g = [return_([r for _, _, r in jalankan_episode(env, pi, rng)])[0]
             for _ in range(20_000)]
        h = 4 * tarik(np.full(20_000, i), rng)
        assert abs(np.mean(g) - np.mean(h)) < 0.05

    print("(4) Kebijakan lorong sebagai lengan, T = 2000, R = 500:")
    print("    strategi     regret (satuan return)   memilih -++"
          " (t>1900)")
    for nama, s, kw in (("eps = 0,1", "eps", {"eps": 0.1}),
                        ("UCB1", "ucb", {"c": 2.0})):
        reg, tepat = simulasi(EKSAK / 4, s, T=2000, R=500, tarik=tarik,
                              **kw)
        print(f"    {nama:12s} {4 * reg[-1]:14.2f}"
              f" {100 * tepat[1900:].mean():17.1f} %")
