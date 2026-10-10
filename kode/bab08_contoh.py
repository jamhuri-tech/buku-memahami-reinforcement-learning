"""Memeriksa setiap bilangan Contoh Soal Bab 8."""
from fractions import Fraction as F
from math import pi, sqrt

import numpy as np
from scipy import integrate, stats

from bab08_qlearning import peluang_eps

h = F(1, 2)
Q = {(2, 0): F(7, 4), (1, 0): F(2), (1, 1): F(13, 8), (3, 1): F(8)}

# Contoh Soal 8.1: satu pembaruan dari Q*
t_sarsa = h * Q[1, 1]
t_q = h * max(Q[1, 0], Q[1, 1])
p = peluang_eps(np.array([2.0, 1.625]), 0.2)
assert np.allclose(p, [0.9, 0.1])
t_e = h * (F(9, 10) * Q[1, 0] + F(1, 10) * Q[1, 1])
assert (t_sarsa, t_q, float(t_e)) == (F(13, 16), 1, 0.98125)
baru = [Q[2, 0] + h * (t - Q[2, 0]) for t in (t_sarsa, t_q, t_e)]
assert baru == [F(41, 32), F(11, 8), F(437, 320)]
assert [float(b) for b in baru] == [1.28125, 1.375, 1.365625]
assert F(3, 4) * 1 + F(1, 4) * h * Q[3, 1] == Q[2, 0]

# Contoh Soal 8.2: peluang jatuh di tepi tebing
assert round(0.975 ** 10, 4) == 0.7763 and round(1 - 0.975 ** 10, 4) == 0.2237

# Contoh Soal 8.3: E[max(X1, X2)] = mu + 1/sqrt(pi)
assert round(1 / sqrt(pi), 4) == 0.5642 and round(-0.1 + 1 / sqrt(pi), 4) == 0.4642
f = lambda x: x * 2 * stats.norm.pdf(x, -0.1) * stats.norm.cdf(x, -0.1)
em, _ = integrate.quad(f, -12, 12)
assert abs(em - (-0.1 + 1 / sqrt(pi))) < 1e-9
assert abs(sqrt(2) * sqrt(2 / pi) / 2 - 1 / sqrt(pi)) < 1e-15
print("Bab 8: semua Contoh Soal cocok")
