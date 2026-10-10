"""Memeriksa setiap bilangan Contoh Soal Bab 7."""
from fractions import Fraction as F

import numpy as np

from bab03_mdp import acak, mdp_lorong
from bab06_mc import EPISODE_MINI
from bab07_td import batch_td, langkah, td0_episode

h = F(1, 2)

# Contoh Soal 7.1: TD(0) pada tiga episode
V = np.zeros(5)
d = [td0_episode(V, ep, 0.5) for ep in EPISODE_MINI]
assert d == [[0, 8], [0, 0, 2, 4], [-1, 2]]
assert np.allclose(V[1:4], [1, 0.5, 6])

# Contoh Soal 7.2: galat MC = jumlah galat TD berdiskon (V tetap)
Vt = {0: 0, 1: 0, 2: 0, 3: 4, 4: 0}
tr = langkah(EPISODE_MINI[1])
delta = [r + h * Vt[s2] * (s2 not in (0, 4)) - Vt[s] for s, r, s2 in tr]
assert delta == [0, 0, 2, 4]
assert sum(h ** k * x for k, x in enumerate(delta)) == 1
G0 = sum(h ** k * r for k, (_, r, _) in enumerate(tr))
assert G0 - Vt[2] == 1

# Contoh Soal 7.3: batch TD = kepastian setara
V2 = F(9, 4) / (1 - F(1, 16))
assert V2 == F(12, 5)
V1 = 1 + F(1, 4) * V2
assert V1 == F(8, 5)
assert V2 == h * h * V1 + h * h * 8
assert np.allclose(batch_td(EPISODE_MINI)[1:4], [1.6, 2.4, 8], atol=1e-9)

# Contoh Soal 7.4: varians sasaran TD dan MC
v = mdp_lorong().nilai(acak(mdp_lorong()))
t1, t3 = h * F(19, 14), h * F(61, 14)
assert (t1, t3) == (F(19, 28), F(61, 28)) and h * (t1 + t3) == F(10, 7)
assert h * h * (t3 - t1) ** 2 == F(9, 16)
EG2 = F(17, 2) * sum(h ** k * F(1, 16) ** (k - 1) for k in range(1, 300))
assert abs(float(EG2) - 136 / 31) < 1e-15
assert round(136 / 31 - (10 / 7) ** 2, 4) == 2.3463
print("Bab 7: semua Contoh Soal cocok")
