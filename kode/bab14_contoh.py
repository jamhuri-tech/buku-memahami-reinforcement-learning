"""Memeriksa setiap bilangan Contoh Soal Bab 14."""
from fractions import Fraction as F
from math import log

import numpy as np

from bab14_aktorkritik import entropi, gae

h = F(1, 2)
V = {1: F(19, 14), 2: F(10, 7), 3: F(61, 14)}

# Contoh Soal 14.1: E[delta | 2, kanan] = A(2, kanan)
d3, d1 = h * V[3] - V[2], h * V[1] - V[2]
assert (d3, d1) == (F(21, 28), F(-21, 28))
m = F(3, 4) * d3 + F(1, 4) * d1
assert m == F(3, 8) and F(101, 56) - F(80, 56) == F(3, 8)
var = F(3, 4) * d3 ** 2 + F(1, 4) * d1 ** 2 - m ** 2
assert var == F(27, 64) and round(float(var), 4) == 0.4219

# Contoh Soal 14.2: GAE pada E2
delta = [h * V[1] - V[2], h * V[2] - V[1], h * V[3] - V[2], 8 - V[3]]
assert delta == [F(-3, 4), F(-9, 14), F(3, 4), F(51, 14)]
dl = np.array([float(x) for x in delta])
assert gae(dl, 0.5, 0.0)[0] == -0.75
assert round(gae(dl, 0.5, 0.5)[0], 4) == -0.8069
assert round(gae(dl, 0.5, 1.0)[0], 4) == round(1 - 10 / 7, 4) == -0.4286
g05 = sum(F(1, 4) ** l * d for l, d in enumerate(delta))
assert round(float(g05), 4) == -0.8069

# Contoh Soal 14.3: entropi
assert round(entropi([.5, .5]), 4) == round(log(2), 4) == 0.6931
assert round(-0.9 * log(0.9), 4) == 0.0948 and round(-0.1 * log(0.1), 4) == 0.2303
assert round(entropi([.9, .1]), 4) == 0.3251
print("Bab 14: semua Contoh Soal cocok")
