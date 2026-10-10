"""Bab 15: matriks Fisher, gradien natural, lema selisih kinerja, dan
langkah TRPO pada lorong licin dengan kebijakan softmax tabular.

Dipakai bab-bab berikutnya:
  fisher(theta, m)           matriks Fisher berbobot eta (10 x 10)
  gradien_natural(theta, m)  F^+ grad J
  kl_kebijakan(p, q, eta)    sum_s eta(s) KL(p(.|s) || q(.|s))
  surrogat_clip(r, A, eps)   tujuan PPO
"""
import numpy as np

from bab03_mdp import mdp_lorong, semua_det
from bab13_reinforce import J, grad_eksak, softmax_kebijakan


def fisher(theta, m):
    pi = softmax_kebijakan(theta)
    _, eta = grad_eksak(theta, m)
    F = np.zeros((10, 10))
    for s in (1, 2, 3):
        p = pi[s]
        blok = np.diag(p) - np.outer(p, p)
        F[2 * s:2 * s + 2, 2 * s:2 * s + 2] = eta[s] * blok
    return F


def gradien_natural(theta, m, redam=1e-6):
    """(F + redam I)^-1 grad J. F singular (softmax tak berubah bila
    semua preferensi satu keadaan ditambah konstanta), sehingga perlu
    redaman seperti Levenberg-Marquardt."""
    g, _ = grad_eksak(theta, m)
    F = fisher(theta, m) + redam * np.eye(10)
    return np.linalg.solve(F, g.ravel()).reshape(5, 2)


def kl_kebijakan(p, q, eta):
    return float(sum(eta[s] * np.sum(p[s] * np.log(p[s] / q[s]))
                     for s in (1, 2, 3)))


def surrogat_clip(r, A, eps=0.2):
    return np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A)


def lintasan(theta0, langkah, alpha, natural, m):
    th, hasil = theta0.copy(), []
    for _ in range(langkah):
        if natural == "tanpa_redam":
            g0 = grad_eksak(th, m)[0].ravel()
            g = (np.linalg.pinv(fisher(th, m)) @ g0).reshape(5, 2)
        elif natural:
            g = gradien_natural(th, m)
        else:
            g = grad_eksak(th, m)[0]
        th = th + alpha * g
        hasil.append(J(th, m))
    return np.array(hasil)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    m = mdp_lorong()
    th = np.zeros((5, 2))
    print("(1) Di theta = 0: gradien biasa dan gradien natural")
    g, eta = grad_eksak(th, m)
    gn = gradien_natural(th, m)
    print("    grad J (kiri, kanan) per petak:\n   ", g[1:4].round(4).tolist())
    print("    F^+ grad J per petak:\n   ", gn[1:4].round(4).tolist())
    A = m.nilai_q(m.nilai(softmax_kebijakan(th))) - \
        m.nilai(softmax_kebijakan(th))[:, None]
    print("    advantage A^pi per petak:\n   ", A[1:4].round(4).tolist())
    gFg = g.ravel() @ gn.ravel()
    beta = np.sqrt(2 * 0.01 / gFg)
    th1 = th + beta * gn
    print(f"    g^T F^-1 g = {gFg:.4f}; TRPO delta = 0.01:"
          f" beta = {beta:.4f}")
    print(f"    KL sesudah langkah = "
          f"{kl_kebijakan(softmax_kebijakan(th), softmax_kebijakan(th1), eta):.5f},"
          f" J: {J(th, m):.4f} -> {J(th1, m):.4f}")

    print("(2) Lema selisih kinerja: pi = acak, pi' = -++")
    pi = softmax_kebijakan(th)
    pi2 = dict(semua_det(m))["-++"]
    P2, _ = m.rantai(pi2)
    n = ~m.terminal
    e2 = np.zeros(5)
    e2[n] = np.linalg.solve((np.eye(3) - 0.5 * P2[np.ix_(n, n)]).T,
                            np.array([0, 1.0, 0]))
    kiri = (e2[:, None] * pi2 * A).sum()
    L = (eta[:, None] * pi2 * A).sum()
    print(f"    eta' = {e2[1:4]}")
    print(f"    sum eta'(s) A(s, pi'(s)) = {kiri:.4f},"
          f" J(pi') - J(pi) = {13 / 4 - 10 / 7:.4f}")
    print(f"    surrogat dengan eta lama = {L:.4f}")

    print("(3) Naik gradien eksak dari pi(kanan) = 0.1 di setiap petak,")
    print("    J sesudah k langkah (J* = 3.25)")
    th0 = np.zeros((5, 2))
    th0[:, 0] = np.log(9)                      # pi(kiri) = 0.9
    print("    k     biasa (a=1)   natural (a=0.5)   tanpa redaman")
    b = lintasan(th0, 200, 1.0, False, m)
    nt = lintasan(th0, 200, 0.5, True, m)
    nt0 = lintasan(th0, 200, 0.5, "tanpa_redam", m)
    for k in (1, 5, 20, 50, 200):
        print(f"    {k:3d} {b[k - 1]:13.4f} {nt[k - 1]:15.4f}"
              f" {nt0[k - 1]:15.4f}")

    print("(4) Surrogat PPO, eps = 0.2:")
    for r, a in ((0.5, 1.0), (1.1, 1.0), (1.5, 1.0), (0.5, -1.0),
                 (1.5, -1.0)):
        print(f"    r = {r:3.1f}, A = {a:4.1f}: L_CLIP = "
              f"{surrogat_clip(r, a):6.3f}")
