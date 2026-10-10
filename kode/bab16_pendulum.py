"""Bab 16: Pendulum dari persamaan geraknya (gaya Gymnasium).

theta'' = 3 g / (2 l) sin(theta) + 3 / (m l^2) u, dengan g = 10, m = 1,
l = 1, dt = 0.05; |u| <= 2, |theta'| <= 8. theta = 0 berarti tegak.
Imbalan -(theta^2 + 0.1 theta'^2 + 0.001 u^2), theta dinormalkan ke
[-pi, pi]. Episode dipotong sesudah 200 langkah (tidak ada terminal).
Pengamatan (cos theta, sin theta, theta').
"""
import numpy as np

G, M, L, DT = 10.0, 1.0, 1.0, 0.05
U_MAKS, W_MAKS = 2.0, 8.0


def normal_sudut(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class Pendulum:
    dim, dim_aksi = 3, 1

    def reset(self, rng):
        self.th = rng.uniform(-np.pi, np.pi)
        self.w = rng.uniform(-1, 1)
        self.t = 0
        return self.amati()

    def amati(self):
        return np.array([np.cos(self.th), np.sin(self.th), self.w],
                        np.float32)

    def step(self, u):
        u = float(np.clip(u, -U_MAKS, U_MAKS))
        r = -(normal_sudut(self.th) ** 2 + 0.1 * self.w ** 2 + 0.001 * u ** 2)
        w = self.w + (3 * G / (2 * L) * np.sin(self.th)
                      + 3.0 / (M * L ** 2) * u) * DT
        self.w = float(np.clip(w, -W_MAKS, W_MAKS))
        self.th = self.th + self.w * DT
        self.t += 1
        return self.amati(), r, False, self.t >= 200


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    env = Pendulum()
    env.th, env.w, env.t = np.pi / 2, 0.0, 0
    print("(3) Pendulum: satu langkah dari theta = pi/2, theta' = 0")
    for u in (-2.0, 0.0, 2.0):
        env.th, env.w = np.pi / 2, 0.0
        s, r, *_ = env.step(u)
        print(f"    u = {u:4.1f}: r = {r:.4f}, theta' baru = {env.w:.4f}")
    rng = np.random.default_rng(20261009)
    tot = []
    for _ in range(100):
        env.reset(rng)
        jumlah = 0.0
        for _ in range(200):
            _, r, _, _ = env.step(rng.uniform(-2, 2))
            jumlah += r
        tot.append(jumlah)
    print(f"    kebijakan acak, 100 episode: return rata-rata"
          f" {np.mean(tot):.1f}")
