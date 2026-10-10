"""Memeriksa setiap bilangan Contoh Soal Bab 13."""
from fractions import Fraction as F

import numpy as np

from bab03_mdp import mdp_lorong
from bab13_reinforce import grad_eksak, grad_logpi, taksiran_reinforce

m = mdp_lorong()
th = np.zeros((5, 2))

# Contoh Soal 13.1: grad log pi pada theta = 0
assert np.allclose(grad_logpi(th, 2, 1)[2], [-0.5, 0.5])
assert np.allclose(grad_logpi(th, 2, 0)[2], [0.5, -0.5])
assert np.allclose(0.5 * grad_logpi(th, 2, 1) + 0.5 * grad_logpi(th, 2, 0), 0)

# Contoh Soal 13.2: gradien eksak
e2 = 1 / (1 - F(1, 8))
assert e2 == F(8, 7) and e2 / 4 == F(2, 7)
g, eta = grad_eksak(th, m)
assert np.allclose(eta[1:4], [2 / 7, 8 / 7, 2 / 7])
A = {1: F(9, 14), 2: F(3, 8), 3: F(51, 14)}    # advantage aksi yang naik
assert F(2, 7) * F(1, 2) * A[1] == F(9, 98)
assert F(8, 7) * F(1, 2) * A[2] == F(3, 14)
assert F(2, 7) * F(1, 2) * A[3] == F(51, 98)
assert np.allclose(g[1:4, 0], [9 / 98, -3 / 14, -51 / 98])
assert np.allclose(np.round(g[1:4, 1], 4), [-0.0918, 0.2143, 0.5204])
# advantage dari Bab 4: A(1, kiri), A(2, kanan), A(3, kanan)
q1 = (F(2), F(5, 7)); v1 = F(19, 14)
assert q1[0] - v1 == F(9, 14)
assert F(101, 56) - F(10, 7) == F(3, 8) and 8 - F(61, 14) == F(51, 14)

# Contoh Soal 13.3: taksiran satu episode
ep = [(2, 1, 0.0), (3, 1, 8.0)]
t = taksiran_reinforce(th, ep)
assert np.allclose(t[2], [-2, 2]) and np.allclose(t[3], [-2, 2])
assert round(2 / 0.2143, 1) == 9.3 and round(2 / 0.5204, 1) == 3.8

# Contoh Soal 13.4: dengan baseline V
v = np.array([0, 19 / 14, 10 / 7, 61 / 14, 0])
tb = taksiran_reinforce(th, ep, baseline=v)
assert np.allclose(tb[2], [-9 / 7, 9 / 7]) and np.allclose(tb[3], [-51 / 56, 51 / 56])
assert 4 - F(10, 7) == F(18, 7) and F(1, 2) * (8 - F(61, 14)) == F(51, 28)
assert round(9 / 7, 4) == 1.2857 and round(51 / 56, 4) == 0.9107
tk = taksiran_reinforce(th, [(2, 0, 0.0), (1, 0, 2.0)], baseline=v)
assert tk[2, 0] < 0                     # 1 < 10/7: kiri didorong turun
print("Bab 13: semua Contoh Soal cocok")
