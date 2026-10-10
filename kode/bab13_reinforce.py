"""Bab 13: teorema gradien kebijakan dan REINFORCE pada lorong licin.

Kebijakan softmax tabular: pi(a | s) = exp(theta[s, a]) / sum_b
exp(theta[s, b]). J(theta) = V^pi(2), dihitung eksak dengan kelas MDP.

Dipakai bab-bab berikutnya:
  softmax_kebijakan(theta)        pi (5 x 2)
  J(theta, mdp), grad_eksak(theta, mdp)
  grad_logpi(theta, s, a)         turunan log pi terhadap theta
  episode_pi(pi, rng, rng_l)      episode lorong (s, a, r)
"""
import numpy as np

from bab01_lorong import BENIH, dinamika
from bab03_mdp import mdp_lorong


def softmax_kebijakan(theta):
    z = theta - theta.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def J(theta, m):
    return m.nilai(softmax_kebijakan(theta))[2]


def grad_eksak(theta, m):
    """Teorema gradien kebijakan: eta(s) pi(a|s) (Q(s,a) - V(s))."""
    pi = softmax_kebijakan(theta)
    v = m.nilai(pi)
    q = m.nilai_q(v)
    Ppi, _ = m.rantai(pi)
    n = ~m.terminal
    eta = np.zeros(5)
    awal = np.zeros(n.sum())
    awal[1] = 1.0                             # petak 2 di antara 1, 2, 3
    eta[n] = np.linalg.solve((np.eye(n.sum()) - m.gamma
                              * Ppi[np.ix_(n, n)]).T, awal)
    g = eta[:, None] * pi * (q - v[:, None])
    g[m.terminal] = 0.0
    return g, eta


def grad_logpi(theta, s, a):
    g = np.zeros_like(theta)
    g[s] = -softmax_kebijakan(theta)[s]
    g[s, a] += 1.0
    return g


def episode_pi(pi, rng, rng_l, s=2):
    jejak = []
    while s not in (0, 4):
        a = int(rng.random() < pi[s, 1])
        u = rng_l.random()
        for p, s2, r in dinamika(s, a):
            if u < p:
                break
            u -= p
        jejak.append((s, a, r))
        s = s2
    return jejak


def taksiran_reinforce(theta, ep, g=0.5, baseline=None):
    """sum_t gamma^t (G_t - b(S_t)) grad log pi(A_t | S_t)."""
    G, hasil = 0.0, np.zeros_like(theta)
    returns = []
    for s, a, r in reversed(ep):
        G = r + g * G
        returns.append(G)
    returns.reverse()
    for t, ((s, a, r), Gt) in enumerate(zip(ep, returns)):
        b = 0.0 if baseline is None else baseline[s]
        hasil += g ** t * (Gt - b) * grad_logpi(theta, s, a)
    return hasil


def latih(n_ep, alpha, benih, baseline=False, alpha_w=0.1, g=0.5):
    m = mdp_lorong()
    rng = np.random.default_rng(benih)
    rng_l = np.random.default_rng(benih + 1)
    theta = np.zeros((5, 2))
    w = np.zeros(5)
    kurva = []
    for i in range(n_ep):
        ep = episode_pi(softmax_kebijakan(theta), rng, rng_l)
        G, ret = 0.0, []
        for s, a, r in reversed(ep):
            G = r + g * G
            ret.append(G)
        ret.reverse()
        for t, ((s, a, r), Gt) in enumerate(zip(ep, ret)):
            d = Gt - (w[s] if baseline else 0.0)
            if baseline:
                w[s] += alpha_w * d
            theta += alpha * g ** t * d * grad_logpi(theta, s, a)
        kurva.append(J(theta, m))
    return np.array(kurva), theta


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    m = mdp_lorong()
    th = np.zeros((5, 2))
    g, eta = grad_eksak(th, m)
    print("(1) Gradien J di theta = 0 (kebijakan acak):")
    print("    eta (kunjungan berdiskon) =", eta[1:4])
    print("    teorema: dJ/dtheta(s, kiri)  =", g[1:4, 0])
    print("             dJ/dtheta(s, kanan) =", g[1:4, 1])
    h = 1e-6
    fd = np.zeros((5, 2))
    for s in (1, 2, 3):
        for a in (0, 1):
            d = np.zeros((5, 2))
            d[s, a] = h
            fd[s, a] = (J(th + d, m) - J(th - d, m)) / (2 * h)
    print(f"    selisih maks dengan beda hingga = {np.abs(fd - g).max():.1e}")

    print("(2) Taksiran REINFORCE di theta = 0, 100 000 episode")
    print("    komponen kanan untuk petak 1, 2, 3")
    rng = np.random.default_rng(BENIH)
    rng_l = np.random.default_rng(BENIH + 1)
    pi = softmax_kebijakan(th)
    v = m.nilai(pi)
    tak0, tak1 = [], []
    for _ in range(100_000):
        ep = episode_pi(pi, rng, rng_l)
        tak0.append(taksiran_reinforce(th, ep)[1:4, 1])
        tak1.append(taksiran_reinforce(th, ep, baseline=v)[1:4, 1])
    for nama, x in (("tanpa baseline", tak0), ("baseline V", tak1)):
        x = np.array(x)
        print(f"    {nama:15s} rata = {x.mean(0)}")
        print(f"    {'':15s} var  = {x.var(0)}")

    print("(3) Belajar REINFORCE, alpha = 0.5, rata-rata 20 benih")
    print("    episode   tanpa baseline   baseline (alpha_w = 0.1)")
    k0 = np.mean([latih(1000, 0.5, BENIH + u)[0] for u in range(20)], 0)
    k1 = np.mean([latih(1000, 0.5, BENIH + u, baseline=True)[0]
                  for u in range(20)], 0)
    for e in (10, 50, 100, 300, 1000):
        print(f"    {e:7d} {k0[e - 1]:16.4f} {k1[e - 1]:16.4f}")
    print("    J(theta*) = V*(2) = 3.25")
