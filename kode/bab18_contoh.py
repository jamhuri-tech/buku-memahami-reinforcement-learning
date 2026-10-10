"""Memeriksa setiap bilangan Contoh Soal Bab 18."""
import itertools
from math import comb, sqrt

import numpy as np

from bab18_riset import benih_perlu, iqm

# Contoh Soal 18.1: rata-rata, median, IQM
x = [120, 300, 480, 500, 500]
assert np.mean(x) == 380 and np.median(x) == 480
assert round(iqm(x), 1) == 426.7 and abs(iqm(x) - (300 + 480 + 500) / 3) < 1e-12
assert iqm([1, 5, 9]) == 5.0 and iqm([0, 0, 900]) == 300.0   # n = 3: = rata

# Contoh Soal 18.2: nilai-p permutasi terkecil
def p_min(n):
    a, b = list(range(n)), list(range(n, 2 * n))      # B selalu lebih besar
    data = a + b
    d0 = abs(np.mean(a) - np.mean(b))
    hit = tot = 0
    for idx in itertools.combinations(range(2 * n), n):
        g1 = [data[i] for i in idx]
        g2 = [data[i] for i in range(2 * n) if i not in idx]
        tot += 1
        hit += abs(np.mean(g1) - np.mean(g2)) >= d0 - 1e-12
    return hit / tot


assert comb(4, 2) == 6 and p_min(2) == 1 / 3
assert comb(6, 3) == 20 and p_min(3) == 0.1

# Contoh Soal 18.3: banyaknya benih
assert round(2 * (1.96 + 0.8416) ** 2, 2) == 15.70
assert round(benih_perlu(150, 100), 1) == 35.3
s = sqrt((0.5606 ** 2 + 0.0682 ** 2) / 2)
assert round(s, 4) == 0.3993 and round(3.1613 - 2.9303, 4) == 0.2310
assert round(benih_perlu(s, 0.2310), 1) == 46.9
print("Bab 18: semua Contoh Soal cocok")
