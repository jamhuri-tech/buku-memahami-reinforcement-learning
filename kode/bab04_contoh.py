"""Memeriksa setiap bilangan Contoh Soal Bab 4."""
from fractions import Fraction as F

import numpy as np

from bab03_mdp import acak, mdp_lorong, semua_det
from bab04_bellman import T_opt, T_pi, greedy, v_opt_lp

m = mdp_lorong()
pi = acak(m)
h = F(1, 2)

# Contoh Soal 4.1: Bellman ekspektasi kebijakan acak, pecahan eksak
V2 = F(5, 4) / (1 - F(1, 8))
assert V2 == F(10, 7)
V1, V3 = 1 + F(1, 4) * V2, 4 + F(1, 4) * V2
assert (V1, V3) == (F(19, 14), F(61, 14))
assert V2 == h * (h * V1 + h * V3)
assert np.allclose(m.nilai(pi)[1:4], [19 / 14, 10 / 7, 61 / 14])

# Contoh Soal 4.2: Q^pi di petak 2
qk = h * (F(3, 4) * V1 + F(1, 4) * V3)
qn = h * (F(3, 4) * V3 + F(1, 4) * V1)
assert qk == F(59, 56) and qn == F(101, 56)
assert round(float(qk), 4) == 1.0536 and round(float(qn), 4) == 1.8036
assert h * (qk + qn) == V2 and F(160, 112) == F(10, 7)
q = m.nilai_q(m.nilai(pi))
assert np.allclose(q[2], [59 / 56, 101 / 56])
assert np.allclose((pi * (q - m.nilai(pi)[:, None])).sum(1), 0)

# Contoh Soal 4.3: kontraksi T^pi
u = np.zeros(5)
w = np.array([0, 1, 1, 1, 0.])
Tu, Tw = T_pi(m, pi, u), T_pi(m, pi, w)
assert np.allclose(Tu[1:4], [1, 0, 4])
assert np.allclose(Tw[1:4], [1.25, 0.5, 4.25])
assert np.allclose((Tw - Tu)[1:4], [0.25, 0.5, 0.25])
assert np.abs(Tw - Tu).max() == 0.5 * np.abs(w - u).max()

# Contoh Soal 4.4: V* dan Q* untuk gamma = 1/2
Vs = {1: F(2), 2: F(13, 4), 3: F(8)}
Q = {(1, 0): F(2), (1, 1): h * Vs[2],
     (2, 0): h * (F(3, 4) * Vs[1] + F(1, 4) * Vs[3]),
     (2, 1): h * (F(3, 4) * Vs[3] + F(1, 4) * Vs[1]),
     (3, 0): h * Vs[2], (3, 1): F(8)}
assert [Q[k] for k in sorted(Q)] == [2, F(13, 8), F(7, 4), F(13, 4),
                                     F(13, 8), 8]
for s in (1, 2, 3):
    assert max(Q[s, 0], Q[s, 1]) == Vs[s]
A = [Q[s, a] - Vs[s] for s in (1, 2, 3) for a in (0, 1)]
assert A == [0, F(-3, 8), F(-3, 2), 0, F(-51, 8), 0]
v = np.zeros(5)
for _ in range(50):
    v = T_opt(m, v)
assert np.allclose(v[1:4], [2, 3.25, 8])
assert np.allclose(m.nilai_q(v)[1:4], [[2, 13 / 8], [7 / 4, 13 / 4],
                                        [13 / 8, 8]])
assert np.allclose(v_opt_lp(m)[1:4], [2, 3.25, 8])

# Contoh Soal 4.5: +++ optimal untuk gamma = 0,9
m9 = mdp_lorong(gamma=0.9)
v9 = m9.nilai(dict(semua_det(m9))["+++"])
assert np.allclose(np.round(v9[1:4], 4), [6.0940, 6.7712, 8])
q9 = m9.nilai_q(v9)
assert round(0.75 * 6.0940 + 2, 4) == 6.5705
assert round(q9[2, 0], 4) == 5.9135 and round(0.9 * 6.5705, 4) == 5.9135
assert round(q9[3, 0], 4) == 6.0940
assert np.allclose(T_opt(m9, v9), v9)
assert not np.allclose(T_opt(m, m.nilai(dict(semua_det(m))["+++"])),
                       m.nilai(dict(semua_det(m))["+++"]))
assert np.array_equal(greedy(m9, v9)[1:4].argmax(1), [1, 1, 1])
# teks: laju sesudah kebijakan +++, gamma rho = 0,45
P, _ = m9.rantai(dict(semua_det(m9))["+++"])
assert abs(np.abs(np.linalg.eigvals(P[1:4, 1:4])).max() - 0.5) < 1e-12
print("Bab 4: semua Contoh Soal cocok")
