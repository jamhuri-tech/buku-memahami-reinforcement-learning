"""Bab 12: CartPole dari persamaan geraknya (gaya Gymnasium).

Persamaan gerak kereta dan tiang (Barto, Sutton, dan Anderson 1983;
koreksi tanda Florian 2007), diintegrasikan dengan metode Euler
eksplisit, dt = 0.02 detik. Keadaan (x, x', theta, theta').

Dipakai bab-bab berikutnya:
  CartPole                reset(rng) -> s; step(a) -> (s, r, terminal,
                          terpotong)
  turunan(s, F)           (x'', theta'') untuk gaya F
"""
import numpy as np

G = 9.8            # percepatan gravitasi (m/s^2)
M_KERETA = 1.0     # kg
M_TIANG = 0.1      # kg
L = 0.5            # setengah panjang tiang (m)
GAYA = 10.0        # N
DT = 0.02          # s
BATAS_X = 2.4
BATAS_SUDUT = 12 * 2 * np.pi / 360
MAKS_LANGKAH = 500


def turunan(s, F):
    x, xd, th, thd = s
    m = M_KERETA + M_TIANG
    c, si = np.cos(th), np.sin(th)
    temp = (F + M_TIANG * L * thd ** 2 * si) / m
    thdd = (G * si - c * temp) / (L * (4.0 / 3.0 - M_TIANG * c ** 2 / m))
    xdd = temp - M_TIANG * L * thdd * c / m
    return xdd, thdd


class CartPole:
    nA = 2
    dim = 4

    def reset(self, rng):
        self.s = rng.uniform(-0.05, 0.05, 4)
        self.t = 0
        return self.s.copy()

    def step(self, a):
        F = GAYA if a == 1 else -GAYA
        xdd, thdd = turunan(self.s, F)
        x, xd, th, thd = self.s
        self.s = np.array([x + DT * xd, xd + DT * xdd,
                           th + DT * thd, thd + DT * thdd])
        self.t += 1
        terminal = abs(self.s[0]) > BATAS_X or abs(self.s[2]) > BATAS_SUDUT
        terpotong = self.t >= MAKS_LANGKAH
        return self.s.copy(), 1.0, bool(terminal), terpotong


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    print("(3) Satu langkah CartPole dari s = (0, 0, 0.05, 0):")
    for a in (0, 1):
        env = CartPole()
        env.s, env.t = np.array([0.0, 0.0, 0.05, 0.0]), 0
        xdd, thdd = turunan(env.s, GAYA if a else -GAYA)
        s2, *_ = env.step(a)
        print(f"    aksi {a}: x'' = {xdd:8.4f}, theta'' = {thdd:8.4f}")
        print(f"            s' = {s2}")
    rng = np.random.default_rng(20261009)
    panjang = []
    for _ in range(1000):
        env = CartPole()
        env.reset(rng)
        k = 0
        while True:
            k += 1
            _, _, term, potong = env.step(int(rng.integers(2)))
            if term or potong:
                break
        panjang.append(k)
    print(f"    kebijakan acak, 1000 episode: panjang rata-rata"
          f" {np.mean(panjang):.2f},")
    print(f"    terpanjang {max(panjang)}")
