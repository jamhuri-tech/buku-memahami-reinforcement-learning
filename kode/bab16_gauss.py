"""Bab 16: kebijakan Gaussian dan gradien kebijakan deterministik pada
bandit kontinu satu langkah: Q(a) = -(a - 2)^2.

Dipakai bab-bab berikutnya:
  grad_log_gauss(a, mu, sd)    (d/dmu, d/dsd) log N(a; mu, sd^2)
  koreksi_tanh(u)              log(1 - tanh(u)^2)
"""
import numpy as np

BENIH = 20261009


def Q(a):
    return -(a - 2.0) ** 2


def grad_log_gauss(a, mu, sd):
    return (a - mu) / sd ** 2, ((a - mu) ** 2 - sd ** 2) / sd ** 3


def koreksi_tanh(u):
    return np.log(1 - np.tanh(u) ** 2)


if __name__ == "__main__":
    mu, sd = 0.5, 1.0
    print("(1) Bandit kontinu Q(a) = -(a - 2)^2, mu = 0.5, sigma = 1")
    print(f"    J = -((mu - 2)^2 + sigma^2) = {-((mu - 2) ** 2 + sd ** 2):.4f}")
    print(f"    eksak: dJ/dmu = {-2 * (mu - 2):.4f},"
          f" dJ/dsigma = {-2 * sd:.4f}")
    print("    taksiran fungsi skor dJ/dmu, 2000 ulangan:")
    print("        n     rata   sd tanpa b   sd dengan b")
    rng = np.random.default_rng(BENIH)
    for n in (10, 100, 10_000):
        tak = []
        for _ in range(2000):
            a = rng.normal(mu, sd, n)
            gm, gs = grad_log_gauss(a, mu, sd)
            tak.append([(Q(a) * gm).mean(), (Q(a) * gs).mean(),
                        ((Q(a) - Q(a).mean()) * gm).mean()])
        tak = np.array(tak)
        print(f"    {n:5d} {tak[:, 0].mean():8.4f} {tak[:, 0].std():12.4f}"
              f" {tak[:, 2].std():13.4f}")
    print(f"    DPG: dQ/da di a = mu = {-2 * (mu - 2):.4f} (tanpa varians)")
    print("(2) Koreksi tanh: log(1 - tanh(u)^2)")
    for u in (0.0, 1.0, 3.0):
        print(f"    u = {u:3.1f}: {koreksi_tanh(u):8.4f}")
