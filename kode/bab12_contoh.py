"""Memeriksa setiap bilangan Contoh Soal Bab 12."""
import math

import numpy as np

from bab12_cartpole import CartPole, turunan
from bab12_minibatch import (AKHIR, G, Q_ONLINE_S2, Q_SA, Q_TARGET_S2, R,
                             dueling, gradien_huber, huber)

# Contoh Soal 12.1: sasaran, loss kuadrat, loss Huber, gradien
y = R + G * (1 - AKHIR) * Q_TARGET_S2.max(1)
assert np.allclose(y, [3.7, 1.9, 1.0])
e = Q_SA - y
assert np.allclose(e, [-1.2, -0.7, -0.6])
assert round(np.mean(e ** 2), 4) == 0.7633
assert np.allclose(huber(e), [0.7, 0.245, 0.18])
assert abs(huber(e).mean() - 0.375) < 1e-12
assert np.allclose(np.round(gradien_huber(e) / 3, 4), [-0.3333, -0.2333, -0.2])

# Contoh Soal 12.2: double DQN
a2 = Q_ONLINE_S2.argmax(1)
yd = R + G * (1 - AKHIR) * Q_TARGET_S2[np.arange(3), a2]
assert list(a2[:2]) == [0, 1] and np.allclose(yd, [2.8, 1.45, 1.0])

# teks: dueling
assert np.allclose(np.round(dueling(2.0, np.array([1, -1, 0.5])), 4),
                   [2.8333, 0.8333, 2.3333])

# Contoh Soal 12.3: satu langkah CartPole, F = +10
s = np.array([0.0, 0.0, 0.05, 0.0])
assert round(math.cos(0.05), 5) == 0.99875 and round(math.sin(0.05), 5) == 0.04998
assert round(10 / 1.1, 4) == 9.0909
num = 9.8 * math.sin(0.05) - math.cos(0.05) * 10 / 1.1
den = 0.5 * (4 / 3 - 0.1 * math.cos(0.05) ** 2 / 1.1)
assert round(num, 4) == -8.5898 and round(den, 4) == 0.6213
xdd, thdd = turunan(s, 10.0)
assert round(thdd, 4) == -13.8249 and round(xdd, 4) == 9.7185
env = CartPole()
env.s, env.t = s.copy(), 0
s2, *_ = env.step(1)
assert np.allclose(np.round(s2, 4), [0, 0.1944, 0.05, -0.2765])
print("Bab 12: semua Contoh Soal cocok")
