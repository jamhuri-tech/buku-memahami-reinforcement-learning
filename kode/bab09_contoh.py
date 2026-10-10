"""Memeriksa setiap bilangan Contoh Soal Bab 9."""
from fractions import Fraction as F

import numpy as np

from bab06_mc import EPISODE_MINI
from bab07_td import langkah
from bab09_jejak import jejak_episode, return_lambda, return_n

h = F(1, 2)
V = np.array([0, 0, 0, 4, 0.])
tr = langkah(EPISODE_MINI[1])

# Contoh Soal 9.1: return n langkah pada t = 0
G = [return_n(tr, V, 0.5, n)[0] for n in (1, 2, 3, 4)]
assert G == [0, 0, 0.5, 1]

# Contoh Soal 9.2: lambda-return dan jumlah galat TD
Gl = h * (0 + h * 0 + h * h * F(1, 2)) + F(1, 8) * 1
assert Gl == F(3, 16) and float(Gl) == 0.1875
assert return_lambda(tr, V, 0.5, 0.5)[0] == 0.1875
delta = [0, 0, 2, 4]
assert sum(F(1, 4) ** k * d for k, d in enumerate(delta)) == F(3, 16)

# Contoh Soal 9.3: jejak akumulasi dan ganti untuk petak 2
Va, ra = jejak_episode(tr, V, 0.5, 0.5, 1.0)
Vg, rg = jejak_episode(tr, V, 0.5, 0.5, 1.0, ganti=True)
assert [e[2] for e in ra] == [1, 0.25, 1.0625, 0.265625]
assert [e[2] for e in rg] == [1, 0.25, 1, 0.25]
assert Va[2] - V[2] == 3.1875 and Vg[2] - V[2] == 3.0
assert 0.1875 + (2 + 0.25 * 4) == 3.1875
# kesetaraan offline: jumlah pembaruan maju every-visit
Gl_all = return_lambda(tr, V, 0.5, 0.5)
maju = sum(Gl_all[t] - V[s] for t, (s, _, _) in enumerate(tr) if s == 2)
assert abs(maju - 3.1875) < 1e-12
print("Bab 9: semua Contoh Soal cocok")
