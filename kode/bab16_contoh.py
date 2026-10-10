"""Memeriksa setiap bilangan Contoh Soal Bab 16."""
from math import log, pi, tanh

import numpy as np
from scipy import integrate, stats

from bab16_gauss import Q, grad_log_gauss, koreksi_tanh
from bab16_pendulum import Pendulum

# Contoh Soal 16.1: fungsi skor dan identitas Stein
gm, gs = grad_log_gauss(1.5, 0.5, 1.0)
assert gm == 1.0 and gs == 0.0 and Q(1.5) == -0.25
e, _ = integrate.quad(lambda a: Q(a) * (a - 0.5) * stats.norm.pdf(a, 0.5),
                      -12, 12)
assert abs(e - 3.0) < 1e-9

# Contoh Soal 16.2: sasaran DDPG dan TD3
assert abs(-1 + 0.99 * -10 - (-10.9)) < 1e-12
assert abs(-1 + 0.99 * min(-10, -12) - (-12.88)) < 1e-12
em, _ = integrate.quad(lambda x: x * 2 * stats.norm.pdf(x) * stats.norm.sf(x),
                       -12, 12)
assert abs(em + 1 / pi ** 0.5) < 1e-9          # E min = -1/sqrt(pi)

# Contoh Soal 16.3: koreksi tanh
assert round(2 * tanh(1), 4) == 1.5232
lN = -0.5 - 0.5 * log(2 * pi)
assert round(lN, 4) == -1.4189 and round(koreksi_tanh(1.0), 4) == -0.8676
assert round(lN - koreksi_tanh(1.0) - log(2), 4) == -1.2445
assert round(koreksi_tanh(3.0), 4) == -4.6187

# Contoh Soal 16.4: satu langkah Pendulum
env = Pendulum()
for u, w in ((-2, 0.45), (0, 0.75), (2, 1.05)):
    env.th, env.w, env.t = np.pi / 2, 0.0, 0
    _, r, _, _ = env.step(u)
    assert abs(env.w - w) < 1e-12
assert round(r, 4) == -2.4714 and round(-((pi / 2) ** 2 + 0.004), 4) == -2.4714
print("Bab 16: semua Contoh Soal cocok")
