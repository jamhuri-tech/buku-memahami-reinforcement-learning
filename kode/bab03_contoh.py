"""Memeriksa setiap bilangan Contoh Soal Bab 3."""
from fractions import Fraction as F

import numpy as np

from bab01_lorong import return_
from bab03_bentuk import bentuk_bonus, bentuk_potensial, terbaik
from bab03_mdp import MDP, acak, mdp_lorong, semua_det

m = mdp_lorong()
r = m.r_sa()

# Contoh Soal 3.1: transisi dan imbalan harapan
assert np.allclose(m.P[2, 0], [0, 0.75, 0, 0.25, 0])
assert r[2, 0] == 0 and r[1, 0] == 2 and r[3, 1] == 8
# pintu kanan acak 16 / 0: p(s'|s,a) dan r(s,a) tetap
assert F(1, 2) + F(1, 2) == 1 and F(1, 2) * 16 + F(1, 2) * 0 == 8

# Contoh Soal 3.2: P^pi dan r^pi kebijakan acak
Ppi, rpi = m.rantai(acak(m))
Pt = Ppi[1:4, 1:4]
assert np.allclose(Pt, [[0, .5, 0], [.5, 0, .5], [0, .5, 0]])
assert np.allclose(rpi[1:4], [1, 0, 4])
assert np.allclose(Pt.sum(1), [0.5, 1, 0.5])
assert np.allclose(Ppi.sum(1), 1)
assert F(1, 2) * F(3, 4) + F(1, 2) * F(1, 4) == F(1, 2)

# Contoh Soal 3.3: harapan panjang episode
mvec = np.linalg.solve(np.eye(3) - Pt, np.ones(3))
assert np.allclose(mvec, [3, 4, 3])
assert np.allclose(np.linalg.inv(np.eye(3) - Pt).sum(1), [3, 4, 3])

# Contoh Soal 3.4: deret nilai kebijakan -++
pi = dict(semua_det(m))["-++"]
P2, r2 = m.rantai(pi)
Pt2 = P2[1:4, 1:4]
assert np.allclose(Pt2, [[0, 0, 0], [.25, 0, .75], [0, 0, 0]])
assert np.allclose(r2[1:4], [2, 0, 8])
suku2 = 0.5 * Pt2 @ r2[1:4]
assert np.allclose(suku2, [0, 3.25, 0])
assert np.allclose(Pt2 @ Pt2, 0)
assert np.allclose(r2[1:4] + suku2, m.nilai(pi)[1:4])
assert np.allclose(m.nilai(pi)[1:4], [2, 3.25, 8])
# teks: nilai eigen kebijakan acak
ev = np.sort(np.linalg.eigvals(Pt).real)
assert np.allclose(ev, [-2 ** -0.5, 0, 2 ** -0.5])
assert round(0.5 * 2 ** -0.5, 4) == 0.3536

# Contoh Soal 3.5: pembentukan berpotensial Phi(s) = s
Phi = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0}
jalur = [2, 1, 2, 3, 4]
Fs = [F(1, 2) * Phi[b] - Phi[a] for a, b in zip(jalur, jalur[1:])]
assert Fs == [F(-3, 2), 0, F(-1, 2), -3]
R = [0, 0, 0, 8]
Rb = [x + f for x, f in zip(R, Fs)]
assert Rb == [F(-3, 2), 0, F(-1, 2), 5]
Gb = sum(F(1, 2) ** t * x for t, x in enumerate(Rb))
assert Gb == -1
assert return_(R, 0.5)[0] - Phi[2] == -1
mb = bentuk_potensial(m, [0, 1, 2, 3, 0])
for kode, p in semua_det(m):                    # V' = V - Phi
    assert np.allclose(mb.nilai(p), m.nilai(p) - np.array(
        [0, 1, 2, 3, 0]))
assert terbaik(mb)[0] == "-++"
kode, v = terbaik(bentuk_bonus(m, 2, 1.0))
assert kode == "+++"
assert np.allclose(m.nilai(dict(semua_det(m))["+++"])[1:4], [1.6, 3.2, 8])
print("Bab 3: semua Contoh Soal cocok")
