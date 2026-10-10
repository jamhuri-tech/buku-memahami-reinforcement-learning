"""Bab 6: metode Monte Carlo untuk prediksi, kontrol, dan off-policy.

Dipakai bab-bab berikutnya:
  EPISODE_MINI                 tiga episode Contoh Soal (s, a, r)
  mc_prediksi(eps, g, pertama) taksiran V dari daftar episode
  episode_env(env, pi, rng)    episode untuk MDP tabular (kelas MDP)
  mc_kontrol(mdp, n, eps, ...) MC on-policy epsilon-greedy, Q tabel
  rasio_is(ep, pi, b)          rasio importance sampling rho_{0:T-1}
"""
import numpy as np

from bab01_lorong import (ACAK, BENIH, KANAN, KIRI, OPTIMUM, Lorong,
                          jalankan_episode)
from bab03_mdp import mdp_lorong

# (S_t, A_t, R_{t+1}); kebijakan acak, gamma = 1/2
EPISODE_MINI = [
    [(2, KANAN, 0), (3, KANAN, 8)],
    [(2, KIRI, 0), (1, KANAN, 0), (2, KIRI, 0), (3, KANAN, 8)],
    [(2, KIRI, 0), (1, KIRI, 2)],
]


def mc_prediksi(episode, g=0.5, pertama=True, nS=5):
    """Rata-rata return per keadaan, first-visit atau every-visit."""
    jumlah = np.zeros(nS)
    n = np.zeros(nS)
    for ep in episode:
        G, hasil = 0.0, []
        for s, a, r in reversed(ep):
            G = r + g * G
            hasil.append((s, G))
        hasil.reverse()
        dilihat = set()
        for s, G in hasil:
            if pertama and s in dilihat:
                continue
            dilihat.add(s)
            jumlah[s] += G
            n[s] += 1
    with np.errstate(invalid="ignore"):
        return jumlah / n, n


def episode_env(mdp, pi, rng, s0=None, maks=10_000):
    """Episode pada MDP tabular dengan kebijakan pi[s, a]."""
    s = mdp.awal if s0 is None else s0
    jejak = []
    for _ in range(maks):
        a = rng.choice(mdp.nA, p=pi[s])
        s2 = rng.choice(mdp.nS, p=mdp.P[s, a])
        jejak.append((s, a, mdp.R[s, a, s2]))
        s = s2
        if mdp.terminal[s]:
            break
    return jejak


def eps_greedy(Q, eps):
    """Kebijakan epsilon-greedy dari tabel Q. Peluang 1 - eps dibagi
    rata di antara aksi yang seri di maksimum."""
    nS, nA = Q.shape
    maks = Q == Q.max(axis=1, keepdims=True)
    return eps / nA + (1 - eps) * maks / maks.sum(axis=1, keepdims=True)


def mc_kontrol(mdp, n_ep, eps=0.1, benih=BENIH, catat=None, q0=0.0):
    """MC first-visit on-policy dengan kebijakan epsilon-greedy.

    catat: daftar nomor episode; pada setiap nomor itu dicatat V^pi
    kebijakan greedy (dihitung eksak dengan mdp.nilai)."""
    rng = np.random.default_rng(benih)
    Q = np.full((mdp.nS, mdp.nA), q0)
    Q[mdp.terminal] = 0.0
    N = np.zeros((mdp.nS, mdp.nA))
    kurva = []
    for i in range(1, n_ep + 1):
        ep = episode_env(mdp, eps_greedy(Q, eps), rng)
        G = 0.0
        pertama = {}
        for t in range(len(ep) - 1, -1, -1):
            s, a, r = ep[t]
            G = r + mdp.gamma * G
            pertama[(s, a)] = G            # tertimpa sampai yang pertama
        for (s, a), G in pertama.items():
            N[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / N[s, a]
        if catat and i in catat:
            kurva.append(mdp.nilai(eps_greedy(Q, 0.0))[mdp.awal])
    return Q, kurva


def rasio_is(ep, pi, b):
    """rho_{0:T-1} = prod pi(A_t | S_t) / b(A_t | S_t)."""
    return float(np.prod([pi[s, a] / b[s, a] for s, a, _ in ep]))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("(1) Tiga episode mini, gamma = 1/2:")
    for nama, pertama in (("first-visit", True), ("every-visit", False)):
        v, n = mc_prediksi(EPISODE_MINI, pertama=pertama)
        print(f"    {nama}: V = {v[1:4]}, n = {n[1:4]}")

    print("(2) Taksiran V(2) kebijakan acak (eksak 1.4286),"
          " 2000 ulangan:")
    print("     n   bias first  RMSE first  bias every  RMSE every")
    rng = np.random.default_rng(BENIH)
    env = Lorong(benih=BENIH + 1)
    ulang = 2000
    ns = (1, 3, 10, 30, 100)
    data = [[jalankan_episode(env, ACAK, rng) for _ in range(max(ns))]
            for _ in range(ulang)]
    for n in ns:
        baris = f"    {n:3d}"
        for pertama in (True, False):
            est = np.array([mc_prediksi(d[:n], pertama=pertama)[0][2]
                            for d in data])
            gal = est - 10 / 7
            baris += f" {gal.mean():11.4f} {np.sqrt((gal ** 2).mean()):11.4f}"
        print(baris)

    print("(3) MC kontrol eps-greedy (eps = 0.1), lorong,"
          " 5000 episode:")
    m = mdp_lorong()
    Q, _ = mc_kontrol(m, 5000)
    print("    Q(s, kiri)  =", Q[1:4, 0])
    print("    Q(s, kanan) =", Q[1:4, 1])
    kode = "".join("-+"[int(Q[s].argmax())] for s in (1, 2, 3))
    print(f"    kebijakan greedy: {kode}")
