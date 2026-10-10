"""Bab 11: hampiran fungsi linear, LSTD, dan triad maut.

Dipakai bab-bab berikutnya:
  fitur_lorong(s)          x(s) = (1, s - 2), x(terminal) = 0
  JalanAcak1000            jalan acak 1000 petak, lompatan 1..100
  fitur_agregasi/poli/fourier(s, ...)
  gradien_mc, semi_td      pembaruan bobot linear per episode
  lstd(trans, fitur, g)    w = A^-1 b dari data
  baird(...)               contoh tandingan Baird
"""
import numpy as np

from bab01_lorong import ACAK, BENIH, Lorong, jalankan_episode
from bab07_td import langkah


def fitur_lorong(s):
    if s in (0, 4):
        return np.zeros(2)
    return np.array([1.0, s - 2.0])


class JalanAcak1000:
    n = 1000

    def episode(self, rng):
        s, tr = 500, []
        while True:
            lompat = int(rng.integers(1, 101)) * (1 if rng.random() < 0.5
                                                   else -1)
            s2 = s + lompat
            if s2 < 1:
                tr.append((s, -1.0, None))
                return tr
            if s2 > self.n:
                tr.append((s, 1.0, None))
                return tr
            tr.append((s, 0.0, s2))
            s = s2

    def nilai_benar(self):
        n = self.n
        P = np.zeros((n, n))
        r = np.zeros(n)
        for s in range(1, n + 1):
            for d in range(1, 101):
                for arah in (1, -1):
                    s2 = s + arah * d
                    if s2 < 1:
                        r[s - 1] -= 1 / 200
                    elif s2 > n:
                        r[s - 1] += 1 / 200
                    else:
                        P[s - 1, s2 - 1] += 1 / 200
        return np.linalg.solve(np.eye(n) - P, r), P


def fitur_agregasi(s, k=10, n=1000):
    x = np.zeros(k)
    x[(s - 1) * k // n] = 1.0
    return x


def fitur_poli(s, orde=5, n=1000):
    u = (s - 1) / (n - 1)
    return u ** np.arange(orde + 1)


def fitur_fourier(s, orde=5, n=1000):
    u = (s - 1) / (n - 1)
    return np.cos(np.pi * np.arange(orde + 1) * u)


def gradien_mc(w, tr, fitur, alpha, g=1.0):
    G = 0.0
    for s, r, _ in reversed(tr):
        G = r + g * G
        x = fitur(s)
        w += alpha * (G - w @ x) * x


def semi_td(w, tr, fitur, alpha, g=1.0, terminal=lambda s: s is None):
    for s, r, s2 in tr:
        x = fitur(s)
        v2 = 0.0 if terminal(s2) else w @ fitur(s2)
        w += alpha * (r + g * v2 - w @ x) * x


def lstd(tr_semua, fitur, g, d, terminal=lambda s: s is None, eps=0.0):
    A = eps * np.eye(d)
    b = np.zeros(d)
    for tr in tr_semua:
        for s, r, s2 in tr:
            x = fitur(s)
            x2 = np.zeros(d) if terminal(s2) else fitur(s2)
            A += np.outer(x, x - g * x2)
            b += r * x
    return np.linalg.solve(A, b)


def baird(n_sapuan, g=0.99, alpha=0.01):
    """Pembaruan semi-gradien TD harapan (gaya DP) pada contoh Baird.

    Tujuh keadaan; keadaan 1..6: v = 2 w_i + w_8; keadaan 7: v = w_7 +
    2 w_8. Sasaran: selalu ke keadaan 7. Pembaruan diberi bobot sebaran
    seragam (sebaran perilaku), imbalan 0."""
    X = np.zeros((7, 8))
    for i in range(6):
        X[i, i], X[i, 7] = 2, 1
    X[6, 6], X[6, 7] = 1, 2
    w = np.array([1, 1, 1, 1, 1, 1, 10, 1.0])
    jejak = [np.linalg.norm(w)]
    for _ in range(n_sapuan):
        delta = g * X[6] @ w - X @ w            # sasaran: ke keadaan 7
        w = w + alpha / 7 * X.T @ delta
        jejak.append(np.linalg.norm(w))
    return w, np.array(jejak)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("(1) Lorong, kebijakan acak, x(s) = (1, s - 2), 20 000 episode")
    rng = np.random.default_rng(BENIH)
    env = Lorong(benih=BENIH + 1)
    data = [langkah(jalankan_episode(env, ACAK, rng)) for _ in range(20_000)]
    term = lambda s: s in (0, 4)
    wm, wt = np.zeros(2), np.zeros(2)
    rm, rt = np.zeros(2), np.zeros(2)          # rata-rata Polyak
    for i, tr in enumerate(data, 1):
        gradien_mc(wm, tr, fitur_lorong, 0.01, g=0.5)
        semi_td(wt, tr, fitur_lorong, 0.01, g=0.5, terminal=term)
        if i > 10_000:
            rm += wm / 10_000
            rt += wt / 10_000
    wl = lstd(data, fitur_lorong, 0.5, 2, terminal=term)
    print("    alpha = 0.01, w dirata-ratakan atas episode 10 001..20 000")
    print(f"    gradien MC   w = {rm}  (teori 2.1429, 1.5)")
    print(f"    semi-grad TD w = {rt}  (teori 2, 1.5)")
    print(f"    LSTD         w = {wl}")

    print("(2) Jalan acak 1000 petak: akar VE sesudah 2000 episode")
    jl = JalanAcak1000()
    v, P = jl.nilai_benar()
    N = np.linalg.inv(np.eye(1000) - P)
    mu = N[499] / N[499].sum()
    print("    fitur          d  gradien MC     semi-grad TD     LSTD")
    print("                       alpha  VE^.5  alpha  VE^.5")
    rng = np.random.default_rng(BENIH)
    data = [jl.episode(rng) for _ in range(2000)]
    # alpha terbaik dari kisi 3e-5 .. 0.3 (Latihan 11.2)
    for nama, f, d, a_mc, a_td in (
            ("agregasi 10", fitur_agregasi, 10, 3e-4, 0.1),
            ("polinomial 5", fitur_poli, 6, 1e-4, 1e-2),
            ("Fourier 5", fitur_fourier, 6, 3e-5, 3e-3)):
        baris = f"    {nama:13s} {d:2d}"
        for metode, a in ((gradien_mc, a_mc), (semi_td, a_td)):
            w = np.zeros(d)
            for tr in data:
                metode(w, tr, f, a)
            vh = np.array([w @ f(s) for s in range(1, 1001)])
            baris += f" {a:6.0e} {np.sqrt(mu @ (vh - v) ** 2):6.4f}"
        w = lstd(data, f, 1.0, d, eps=1e-6)
        vh = np.array([w @ f(s) for s in range(1, 1001)])
        baris += f" {np.sqrt(mu @ (vh - v) ** 2):8.4f}"
        print(baris)
    for nama, f, d in (("agregasi 10", fitur_agregasi, 10),
                       ("polinomial 5", fitur_poli, 6),
                       ("Fourier 5", fitur_fourier, 6)):
        Phi = np.array([f(s) for s in range(1, 1001)])
        D = mu[:, None]
        w = np.linalg.solve(Phi.T @ (D * Phi), Phi.T @ (mu * v))
        print(f"    min VE^.5 {nama:13s} = {np.sqrt(mu @ (Phi @ w - v) ** 2):.4f}")

    print("(3) Contoh Baird: norma w, semi-gradien TD off-policy")
    _, j = baird(1000)
    for k in (0, 10, 100, 500, 1000):
        print(f"    sapuan {k:4d}: ||w|| = {j[k]:.4g}")
