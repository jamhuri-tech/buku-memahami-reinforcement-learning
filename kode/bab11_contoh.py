"""Memeriksa setiap bilangan Contoh Soal Bab 11."""
from fractions import Fraction as F

import numpy as np
import sympy as sp

from bab11_linear import baird

R = sp.Rational
P = sp.Matrix([[0, R(1, 2), 0], [R(1, 2), 0, R(1, 2)], [0, R(1, 2), 0]])
Nf = (sp.eye(3) - P).inv()
mu = Nf.row(1) / sum(Nf.row(1))
assert list(Nf.row(1)) == [1, 2, 1] and list(mu) == [R(1, 4), R(1, 2), R(1, 4)]
v = sp.Matrix([R(19, 14), R(10, 7), R(61, 14)])
X = sp.Matrix([[1, -1], [1, 0], [1, 1]])
D = sp.diag(*mu)

# Contoh Soal 11.1: gradien MC = kuadrat terkecil berbobot
assert X.T * D * X == sp.Matrix([[1, 0], [0, R(1, 2)]])
assert X.T * D * v == sp.Matrix([R(15, 7), R(3, 4)])
wmc = (X.T * D * X).inv() * X.T * D * v
assert list(wmc) == [R(15, 7), R(3, 2)]
assert list(X * wmc) == [R(9, 14), R(15, 7), R(51, 14)]
assert list(X * wmc - v) == [R(-5, 7), R(5, 7), R(-5, 7)]
VE = lambda w: ((X * w - v).T * D * (X * w - v))[0]
assert VE(wmc) == R(25, 49) and round(float(VE(wmc)), 4) == 0.5102

# Contoh Soal 11.2: titik tetap TD
g = R(1, 2)
A = X.T * D * (X - g * P * X)
b = X.T * D * sp.Matrix([1, 0, 4])
assert A == sp.Matrix([[R(5, 8), 0], [0, R(1, 2)]])
assert b == sp.Matrix([R(5, 4), R(3, 4)])
wtd = A.inv() * b
assert list(wtd) == [2, R(3, 2)]
assert list(X * wtd) == [R(1, 2), 2, R(7, 2)]
assert list(X * wtd - v) == [R(-6, 7), R(4, 7), R(-6, 7)]
assert VE(wtd) == R(26, 49) and round(float(VE(wtd)), 4) == 0.5306
assert VE(wmc) / (1 - g) == R(50, 49) and VE(wtd) <= R(50, 49)

# Contoh Soal 11.3: dua keadaan w, 2w
for gm in (0.4, 0.5, 0.6, 0.9):
    w = 1.0
    for _ in range(100):
        w += 0.1 * (gm * 2 * w - w)
    assert (abs(w) > 1) == (gm > 0.5)

# teks: Baird divergen
_, j = baird(1000)
assert j[-1] > 40 * j[0]
print("Bab 11: semua Contoh Soal cocok")
