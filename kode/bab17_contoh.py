"""Memeriksa setiap bilangan Contoh Soal Bab 17."""
from math import exp, log

import numpy as np

from bab17_rlhf import PI_REF, R_BENAR, KALIMAT, kl, optimum_kl

sig = lambda z: 1 / (1 + exp(-z))

# Contoh Soal 17.1: Bradley-Terry
assert round(sig(1), 4) == 0.7311 and round(-log(sig(1)), 4) == 0.3133
assert round(1 - sig(1), 4) == 0.2689

# Contoh Soal 17.2: acuan model mini
assert abs(PI_REF @ R_BENAR - 0.6875) < 1e-12
assert abs(PI_REF[KALIMAT.index("aaa")] - 1 / 64) < 1e-15
assert round(2000 / 64, 2) == 31.25

# Contoh Soal 17.3: optimum KL tertutup
ref = np.array([0.5, 0.25, 0.25])
r = np.array([0.0, 1.0, 2.0])
w = ref * np.exp(r)
assert np.allclose(np.round(w, 4), [0.5, 0.6796, 1.8473])
assert round(w.sum(), 4) == 3.0268
p = optimum_kl(ref, r, 1.0)
assert np.allclose(np.round(p, 4), [0.1652, 0.2245, 0.6103])
assert round(kl(p, ref), 4) == 0.3376 and round(p @ r, 4) == 1.4451
assert round(log(w.sum()), 4) == 1.1075
assert abs(p @ r - kl(p, ref) - log(w.sum())) < 1e-12

# Contoh Soal 17.4: DPO satu pasangan
h = 0.5 * (0.2 + 0.1)
assert round(h, 2) == 0.15 and round(sig(h), 4) == 0.5374
assert round(-log(sig(h)), 4) == 0.6210
assert round(-(1 - sig(h)), 4) == -0.4626 and round(0.5 * (1 - sig(h)), 4) == 0.2313
print("Bab 17: semua Contoh Soal cocok")
