"""Memeriksa setiap bilangan Contoh Soal Bab 6."""
from fractions import Fraction as F
from math import log

import numpy as np

from bab01_lorong import ACAK, OPTIMUM
from bab06_mc import EPISODE_MINI, eps_greedy, mc_prediksi, rasio_is
from bab06_offpolicy import is_estimasi

h = F(1, 2)

# Contoh Soal 6.1: first-visit dan every-visit pada tiga episode
v1, n1 = mc_prediksi(EPISODE_MINI, pertama=True)
v2, n2 = mc_prediksi(EPISODE_MINI, pertama=False)
assert np.allclose(v1[1:4], [2, 2, 8]) and np.allclose(n1[1:4], [2, 3, 2])
assert np.allclose(v2[1:4], [2, 2.5, 8]) and np.allclose(n2[1:4], [2, 4, 2])
assert (4 + 1 + 1) / 3 == 2 and (4 + 1 + 4 + 1) / 4 == 2.5

# Contoh Soal 6.2: harapan taksiran every-visit dari satu episode
# K ~ Geometrik(1/2) kunjungan ke petak 2; E[G_K] = 5/2
EGK = h * (h * 2 + h * 8)
assert EGK == F(5, 2)
e_every = sum(h ** k * EGK * (1 - F(1, 4) ** k) / (k * F(3, 4))
              for k in range(1, 200))
assert abs(float(e_every) - 10 / 3 * log(7 / 4)) < 1e-12
assert round(10 / 3 * log(7 / 4), 4) == 1.8654
assert round(10 / 3 * log(7 / 4) - 10 / 7, 4) == 0.4368
e_first = sum(h ** k * F(1, 4) ** (k - 1) * EGK for k in range(1, 200))
assert abs(float(e_first) - 10 / 7) < 1e-15
assert abs(log(2) - log(8 / 7) - log(7 / 4)) < 1e-15

# Contoh Soal 6.3: epsilon-greedy dua aksi
pi = eps_greedy(np.array([[0.0, 1.0]]), 0.1)
assert np.allclose(pi, [[0.05, 0.95]])

# Contoh Soal 6.4: IS pada tiga episode mini
rho = [rasio_is(ep, OPTIMUM, ACAK) for ep in EPISODE_MINI]
assert rho == [4.0, 0.0, 0.0]
assert abs(is_estimasi(EPISODE_MINI, OPTIMUM, ACAK) - 16 / 3) < 1e-12
assert is_estimasi(EPISODE_MINI, OPTIMUM, ACAK, berbobot=True) == 4.0

# Contoh Soal 6.5: sebaran rho di bawah perilaku acak
p_kanan = h * F(3, 4) * h          # 2 -> 3 -> 4, G = 4
p_kiri = h * F(1, 4) * h           # 2 -> 1 (tergelincir) -> 0, G = 1
assert (p_kanan, p_kiri) == (F(3, 16), F(1, 16))
assert p_kanan + p_kiri == F(1, 4) and 4 * (p_kanan + p_kiri) == 1
EG = 4 * (p_kanan * 4 + p_kiri * 1)
assert EG == F(13, 4)
E2 = 16 * (p_kanan * 16 + p_kiri * 1)
assert E2 == 49 and E2 - EG ** 2 == F(615, 16)
assert float(F(615, 16)) == 38.4375
assert round(float(F(615, 16) / F(27, 16)), 1) == 22.8
assert round(float(F(615, 16) / 100) ** 0.5, 4) == 0.6200
print("Bab 6: semua Contoh Soal cocok")
