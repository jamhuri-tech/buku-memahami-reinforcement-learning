"""Memeriksa setiap bilangan Contoh Soal Bab 15."""
from fractions import Fraction as F
from math import exp, log

import numpy as np

from bab03_mdp import mdp_lorong
from bab13_reinforce import softmax_kebijakan
from bab15_natural import fisher, gradien_natural, surrogat_clip

# Contoh Soal 15.1: KL kuadratik lawan eksak
B = np.diag([.5, .5]) - np.outer([.5, .5], [.5, .5])
assert np.allclose(B, 0.25 * np.array([[1, -1], [-1, 1]]))
d = np.array([-0.1, 0.1])
assert abs(0.5 * d @ B @ d - 0.005) < 1e-15
p = softmax_kebijakan(np.array([[-0.1, 0.1]]))[0]
assert round(p[1], 6) == 0.549834 and round(p[0], 6) == 0.450166
kl = 0.5 * log(0.5 / p[0]) + 0.5 * log(0.5 / p[1])
assert round(kl, 6) == 0.004992 and abs(kl / 0.005 - 1) < 0.002
assert np.allclose(B @ np.ones(2), 0)

# Contoh Soal 15.2: gradien natural = advantage
m = mdp_lorong()
gn = gradien_natural(np.zeros((5, 2)), m)
assert np.allclose(gn[1:4], [[9 / 14, -9 / 14], [-3 / 8, 3 / 8],
                             [-51 / 14, 51 / 14]], atol=1e-4)

# Contoh Soal 15.3: selisih kinerja dan surrogat
eta2 = [F(1, 8), F(1), F(3, 8)]
A = [F(9, 14), F(3, 8), F(51, 14)]
assert sum(e * a for e, a in zip(eta2, A)) == F(51, 28)
assert F(13, 4) - F(10, 7) == F(51, 28) and round(51 / 28, 4) == 1.8214
eta = [F(2, 7), F(8, 7), F(2, 7)]
L = sum(e * a for e, a in zip(eta, A))
assert L == F(162, 98) and round(float(L), 4) == 1.6531

# Contoh Soal 15.4: langkah TRPO
gFg = sum(e * a ** 2 for e, a in zip(eta, A))
assert round(float(gFg), 4) == 4.0703
assert [round(float(e * a ** 2), 4) for e, a in zip(eta, A)] == [0.1181, 0.1607, 3.7915]
assert round((0.02 / float(gFg)) ** 0.5, 4) == 0.0701

# Contoh Soal 15.5: surrogat PPO
nilai = [surrogat_clip(r, a) for r, a in ((0.5, 1), (1.1, 1), (1.5, 1),
                                          (0.5, -1), (1.5, -1))]
assert np.allclose(nilai, [0.5, 1.1, 1.2, -0.8, -1.5])
h = 1e-6
turunan = [(surrogat_clip(r + h, a) - surrogat_clip(r - h, a)) / (2 * h)
           for r, a in ((1.5, 1), (0.5, -1), (1.5, -1))]
assert np.allclose(turunan, [0, 0, -1])
print("Bab 15: semua Contoh Soal cocok")
