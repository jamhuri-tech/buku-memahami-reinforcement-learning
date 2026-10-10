"""Memeriksa setiap bilangan Contoh Soal Bab 5."""
from fractions import Fraction as F

import numpy as np

from bab03_mdp import acak, mdp_lorong, semua_det
from bab04_bellman import T_opt, greedy
from bab05_dp import (PETA_GRID, evaluasi_iteratif, iterasi_kebijakan,
                      kode_lorong, mdp_grid)

m = mdp_lorong()
q = F(1, 4)

# Contoh Soal 5.1: serentak lawan di tempat, dua sapuan
v = [F(0)] * 3
for _ in range(2):
    v = [1 + q * v[1], q * (v[0] + v[2]), 4 + q * v[1]]
assert v == [1, F(5, 4), 4]
w = [F(0)] * 3
riw = []
for _ in range(2):
    w[0] = 1 + q * w[1]
    w[1] = q * (w[0] + w[2])
    w[2] = 4 + q * w[1]
    riw.append(list(w))
assert riw[0] == [1, F(1, 4), F(65, 16)] and float(riw[0][2]) == 4.0625
assert riw[1][0] == F(17, 16) and float(riw[1][0]) == 1.0625
assert riw[1][1] == F(41, 32) and float(riw[1][1]) == 1.28125
assert round(float(riw[1][2]), 4) == 4.3203
ve = m.nilai(acak(m))[1:4]
assert round(max(abs(float(x) - y) for x, y in zip(v, ve)), 4) == 0.3571
assert round(max(abs(float(x) - y) for x, y in zip(riw[1], ve)), 4) == 0.2946
v2, k = evaluasi_iteratif(m, acak(m), ditempat=True)
assert np.allclose(v2[1:4], ve, atol=1e-9)

# teks: faktor gamma / (1 - gamma)
assert [round(g / (1 - g)) for g in (0.5, 0.99, 0.999)] == [1, 99, 999]

# Contoh Soal 5.2: policy iteration dari ---
pi0 = np.zeros((5, 2))
pi0[:, 0] = 1
_, _, r = iterasi_kebijakan(m, pi0)
assert [kode_lorong(p) for p, _ in r] == ["---", "--+", "-++"]
assert np.allclose(r[0][1][1:4], [2, 0.8, 0.4])
assert np.allclose(r[1][1][1:4], [2, 1.75, 8])
qv = m.nilai_q(r[0][1])
assert np.allclose(qv[1:4], [[2, 0.4], [0.8, 0.4], [0.4, 8]])
qv = m.nilai_q(r[1][1])
assert np.allclose(qv[2], [1.75, 3.25])

# Contoh Soal 5.3: langkah Newton = evaluasi kebijakan greedy
rng = np.random.default_rng(0)
for g in (0.5, 0.9):
    mg = mdp_lorong(gamma=g)
    for _ in range(20):
        v = np.zeros(5)
        v[1:4] = rng.normal(0, 5, 3)
        pv = greedy(mg, v)
        P, rv = mg.rantai(pv)
        n = ~mg.terminal
        Pt, Ft = P[np.ix_(n, n)], (T_opt(mg, v) - v)[n]
        baru = v[n] - np.linalg.solve(g * Pt - np.eye(3), Ft)
        assert np.allclose(baru, mg.nilai(pv)[n])

# Contoh Soal 5.4: value iteration dua sapuan
v1 = T_opt(m, np.zeros(5))
assert np.allclose(v1[1:4], [2, 0, 8])
assert kode_lorong(greedy(m, v1)) == "-++"
assert np.allclose(m.nilai_q(v1)[2], [1.75, 3.25])
v2 = T_opt(m, v1)
assert np.allclose(v2[1:4], [2, 3.25, 8])
assert np.allclose(T_opt(m, v2), v2)

# teks: grid es
assert sum(r.count("L") for r in PETA_GRID) == 9
mg = mdp_grid()
assert mg.nS == 64 and (~mg.terminal).sum() == 54
assert round(0.9 ** 14, 2) == 0.23
print("Bab 5: semua Contoh Soal cocok")
