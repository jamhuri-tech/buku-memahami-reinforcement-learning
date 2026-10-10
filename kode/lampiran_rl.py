"""Lampiran B: lorong licin, value iteration, Q-learning, dan REINFORCE
dalam satu berkas NumPy."""
import numpy as np

KIRI, KANAN, LICIN, GAMMA = 0, 1, 0.25, 0.5
IMBALAN = {0: 2.0, 4: 8.0}            # pintu kiri dan kanan


def dinamika(s, a):
    """(peluang, s_baru, imbalan) untuk petak s = 1, 2, 3."""
    arah = -1 if a == KIRI else 1
    if s == 2:
        keluar = [(1 - LICIN, s + arah),
                  (LICIN, s - arah)]
    else:
        keluar = [(1.0, s + arah)]
    return [(p, s2, IMBALAN.get(s2, 0.0)) for p, s2 in keluar]


def langkah(s, a, rng):
    u = rng.random()
    for p, s2, r in dinamika(s, a):
        if u < p:
            return s2, r
        u -= p
    return s2, r


def value_iteration(n=50):
    v = np.zeros(5)                # v[0] = v[4] = 0, terminal
    for _ in range(n):
        q = np.array([[sum(p * (r + GAMMA * v[s2])
                           for p, s2, r in dinamika(s, a))
                       for a in (KIRI, KANAN)]
                      for s in (1, 2, 3)])
        v[1:4] = q.max(1)
    return v[1:4], q


def q_learning(n_ep, rng, rng_l, eps=0.2):
    Q, N = np.zeros((5, 2)), np.zeros((5, 2))
    for _ in range(n_ep):
        s = 2
        while s not in (0, 4):
            acak = rng.random() < eps
            a = rng.integers(2) if acak else Q[s].argmax()
            s2, r = langkah(s, a, rng_l)
            N[s, a] += 1
            d = r + GAMMA * Q[s2].max() - Q[s, a]
            Q[s, a] += d / N[s, a]
            s = s2
    return Q[1:4]


def reinforce(n_ep, rng, rng_l, alpha=0.5):
    th = np.zeros((5, 2))
    for _ in range(n_ep):
        s, jejak = 2, []
        while s not in (0, 4):
            p = np.exp(th[s]) / np.exp(th[s]).sum()
            a = int(rng.random() < p[1])
            s2, r = langkah(s, a, rng_l)
            jejak.append((s, a, r, p))
            s = s2
        G = 0.0
        for t in reversed(range(len(jejak))):
            s, a, r, p = jejak[t]
            G = r + GAMMA * G
            g = np.eye(2)[a] - p          # turunan log pi
            th[s] += alpha * GAMMA ** t * G * g
    e = np.exp(th[1:4])
    return e / e.sum(1, keepdims=True)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    v, q = value_iteration()
    print("V* =", v)
    print("Q* (kiri, kanan) =", q.round(4).tolist())
    rng, rng_l = np.random.default_rng(20261009), np.random.default_rng(1)
    print("Q-learning 20 000 episode:")
    print("   ", q_learning(20_000, rng, rng_l).round(3).tolist())
    print("REINFORCE 2000 episode, pi(kanan):",
          reinforce(2000, rng, rng_l)[:, 1].round(3))
