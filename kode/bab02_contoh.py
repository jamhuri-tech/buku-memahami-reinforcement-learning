"""Memeriksa setiap bilangan Contoh Soal Bab 2."""
from fractions import Fraction as F

import numpy as np
from scipy import integrate, stats

from bab02_bandit import MU_MINI, RIWAYAT, rata_inkremental

# Riwayat data mini
N = np.zeros(3, int)
S = np.zeros(3, int)
for a, r in RIWAYAT:
    N[a - 1] += 1
    S[a - 1] += r
assert list(N) == [2, 2, 4] and list(S) == [1, 1, 3]
Q = S / N
assert np.allclose(Q, [0.5, 0.5, 0.75])

# Contoh Soal 2.1: rata-rata inkremental lengan 3
r3 = [r for a, r in RIWAYAT if a == 3]
assert r3 == [1, 0, 1, 1]
assert np.allclose(rata_inkremental(r3), [1, 0.5, 2 / 3, 0.75])
assert F(2, 3) + F(1, 4) * (1 - F(2, 3)) == F(3, 4)

# Contoh Soal 2.2: laju langkah tetap 1/2
assert np.allclose(rata_inkremental(r3, alpha=0.5),
                   [0.5, 0.25, 0.625, 13 / 16])
bobot = [F(1, 16), F(1, 16), F(1, 8), F(1, 4), F(1, 2)]
assert sum(bobot) == 1
assert F(1, 16) * 1 + F(1, 8) * 0 + F(1, 4) + F(1, 2) == F(13, 16)
for a in (0.1, 0.3, 0.5):                       # jumlah bobot = 1
    n = 20
    w = a * (1 - a) ** (n - np.arange(1, n + 1))
    assert abs(w.sum() + (1 - a) ** n - 1) < 1e-12

# Contoh Soal 2.3: epsilon-greedy
eps, k = 0.1, 3
p_g = 1 - eps + eps / k
assert round(p_g, 4) == 0.9333 and round(eps / k, 4) == 0.0333
harap = p_g * 0.75 + eps / k * 0.5 + eps / k * 0.25
assert abs(harap - 0.725) < 1e-12
assert abs((1 - eps) * 0.75 + eps * MU_MINI.mean() - 0.725) < 1e-12
assert abs(eps / k * (0.5 + 0.25) - 0.025) < 1e-12

# Contoh Soal 2.4: UCB1 pada t = 8
assert round(np.log(8), 4) == 2.0794 and round(2 * np.log(8), 4) == 4.1589
ucb = Q + np.sqrt(2 * np.log(8) / N)
assert np.allclose(np.round(ucb, 4), [1.9420, 1.9420, 1.7697])
assert round(np.sqrt(2 * np.log(8) / 2), 4) == 1.4420
assert round(np.sqrt(2 * np.log(8) / 4), 4) == 1.0197
assert round(1.4420 - 1.0197, 2) == 0.42

# Contoh Soal 2.5: Thompson, P(Beta(4,2) > Beta(2,2)) = 5/7
assert F(4, 6) == F(2, 3)
assert F(5, 6) - F(9, 7) + F(1, 2) == F(1, 21)
assert 1 - 6 * F(1, 21) == F(5, 7) and round(5 / 7, 4) == 0.7143
p, _ = integrate.quad(lambda y: stats.beta.pdf(y, 2, 2)
                      * stats.beta.sf(y, 4, 2), 0, 1)
assert abs(p - 5 / 7) < 1e-10
y = 0.37                                          # fungsi sebaran Beta(4,2)
assert abs(stats.beta.cdf(y, 4, 2) - (5 * y ** 4 - 4 * y ** 5)) < 1e-12

# Contoh Soal 2.6: varians return dua kebijakan lorong
v1 = F(3, 4) * F(1, 4) * 9
assert v1 == F(27, 16) and float(v1) == 1.6875
eg2 = 12 / (1 - F(1, 64))
assert eg2 == F(768, 63)
v2 = eg2 - F(16, 5) ** 2
assert v2 == F(3072, 1575) and round(float(v2), 4) == 1.9505
sd = float((v1 + v2)) ** 0.5
assert round(sd, 4) == 1.9073
n = (2 * sd / 0.05) ** 2
assert round(2 * sd / 0.05, 2) == 76.29 and round(n) == 5821
print("Bab 2: semua Contoh Soal cocok")
