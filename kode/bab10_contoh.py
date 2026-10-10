"""Memeriksa setiap bilangan Contoh Soal Bab 10."""
import math
from fractions import Fraction as F

import numpy as np

from bab10_perencanaan import Labirin, uct

# Contoh Soal 10.1: satu langkah perencanaan Dyna-Q (alpha = 1)
Q2, Q3 = 0 + F(1, 2) * 0, F(8)
assert (Q2, Q3) == (0, 8)
assert 0 + F(1, 2) * Q3 == 4

# Contoh Soal 10.2: rollout acak = Q^acak(2, .)
assert round(59 / 56, 4) == 1.0536 and round(101 / 56, 4) == 1.8036
assert F(13, 4) > F(7, 4)

# Contoh Soal 10.3: skor UCT dan kunjungan ulang
q, N = uct(2, 10)
assert np.array_equal(N, [5, 5])
assert np.allclose(np.round(q, 4), [0.8008, 1.8531])
bonus = 2 * math.sqrt(math.log(10) / 5)
assert round(math.log(10) / 5, 4) == 0.4605 and round(bonus, 4) == 1.3572
assert round(0.8008 + bonus, 4) == 2.1580 and round(1.8531 + bonus, 4) == 3.2103


def cari(qk, qn, nk, nn):
    m = 0
    while not (qk + 2 * math.sqrt(math.log(nk + nn + m) / nk)
               >= qn + 2 * math.sqrt(math.log(nk + nn + m) / (nn + m))):
        m += 1
    return m


assert cari(0.8008, 1.8531, 5, 5) == 29
q, N = uct(2, 100)
assert round(q[1], 4) == 3.1640 and np.array_equal(N, [5, 95])
assert cari(0.8008, 3.1640, 5, 95) == 2104

# teks: panjang jalur terpendek labirin (BFS)
env = Labirin()
jarak = {env.awal: 0}
antri = [env.awal]
while antri:
    s = antri.pop(0)
    for a in range(4):
        s2, _, _ = env.langkah_model(s, a)
        if s2 not in jarak:
            jarak[s2] = jarak[s] + 1
            antri.append(s2)
assert jarak[env.tujuan] == 12
print("Bab 10: semua Contoh Soal cocok")
