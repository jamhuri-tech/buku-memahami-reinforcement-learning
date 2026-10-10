"""Memeriksa setiap bilangan Contoh Soal Bab 1."""
from fractions import Fraction as F

import numpy as np

from bab01_lorong import (ACAK, KANAN, KANAN_SEMUA, KIRI, OPTIMUM,
                          dinamika, kebijakan_det, nilai_eksak, return_)

# Tabel dinamika: setiap (s, a) berpeluang total 1
for s in (1, 2, 3):
    for a in (KIRI, KANAN):
        assert abs(sum(p for p, _, _ in dinamika(s, a)) - 1) < 1e-12
assert dinamika(1, KIRI) == [(1.0, 0, 2.0)]
assert dinamika(3, KANAN) == [(1.0, 4, 8.0)]

# Contoh Soal 1.1: kanan di petak 2
assert dinamika(2, KANAN) == [(0.75, 3, 0.0), (0.25, 1, 0.0)]
assert sum(p * r for p, _, r in dinamika(2, KANAN)) == 0
assert F(3, 4) * 1 == F(3, 4)

# Contoh Soal 1.2: return episode 2 -> 1 -> 2 -> 3 -> 4
G = return_([0, 0, 0, 8], 0.5)
assert G == [1.0, 2.0, 4.0, 8.0]
assert round(return_([0, 0, 0, 8], 0.9)[0], 3) == 5.832
assert return_([0, 8], 0.5)[0] == 4 and return_([0, 8], 0.9)[0] == 7.2
assert round(0.9 ** 2, 2) == 0.81

# Contoh Soal 1.3: peluang trajektori
ling = F(1, 4) * 1 * F(3, 4) * 1
assert ling == F(3, 16) and float(ling) == 0.1875
acak = F(1, 2) ** 4 * ling
assert acak == F(3, 256) and round(float(acak), 4) == 0.0117

# Contoh Soal 1.4: V(2) kebijakan +++ dua cara
deret = sum(F(3, 4) * F(1, 4) ** k * 8 * F(1, 2) ** (1 + 2 * k)
            for k in range(60))
assert abs(float(deret) - 3.2) < 1e-12
assert 3 / (1 - F(1, 16)) == F(16, 5)
v2 = F(3) / (1 - F(1, 16))
assert v2 == F(16, 5) and v2 / 2 == F(8, 5)
assert np.allclose(nilai_eksak(KANAN_SEMUA), [1.6, 3.2, 8])
assert F(1, 4) / F(3, 4) == F(1, 3) and 2 + 2 * F(1, 3) == F(8, 3)

# Teks: nilai eksak tiga kebijakan
assert np.allclose(nilai_eksak(ACAK), [19 / 14, 10 / 7, 61 / 14])
assert round(10 / 7, 4) == 1.4286
assert np.allclose(nilai_eksak(OPTIMUM), [2, 3.25, 8])
assert F(13, 4) - F(16, 5) == F(1, 20)
assert F(1, 2) * F(13, 4) == F(13, 8) < 2

# Contoh Soal 1.5: gamma ambang di petak 1
def kanan_di_1(g):
    return 6 * g ** 2 / (1 - g ** 2 / 4)


g0 = 2 / np.sqrt(13)
assert abs(kanan_di_1(g0) - 2) < 1e-12 and round(g0, 4) == 0.5547
assert kanan_di_1(0.5) == 1.5 / 0.9375 == 1.6
assert round(6 * 0.81, 2) == 4.86 and round(1 - 0.81 / 4, 4) == 0.7975
assert round(kanan_di_1(0.9), 4) == 6.0940
for g in (0.5, 0.9):
    assert np.isclose(nilai_eksak(KANAN_SEMUA, g)[0], kanan_di_1(g))
assert np.allclose(nilai_eksak(OPTIMUM, 0.9), [2, 5.85, 8])

# Kebijakan yang tidak pernah keluar bernilai 0
for aksi in ((KANAN, KIRI, KIRI), (KANAN, KANAN, KIRI)):
    assert np.allclose(nilai_eksak(kebijakan_det(aksi)), 0)
print("Bab 1: semua Contoh Soal cocok")
