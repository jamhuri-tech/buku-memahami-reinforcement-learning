"""Bab 7: temporal difference TD(0), batch TD lawan batch MC.

Dipakai bab-bab berikutnya:
  td0_episode(V, ep, alpha, g)  memperbarui V di tempat; galat TD
  batch_td(episode, g)          titik tetap TD pada data tetap
  kurva_rmse(...)               RMSE V lorong per episode, TD atau MC
"""
import numpy as np

from bab01_lorong import ACAK, BENIH, Lorong, jalankan_episode
from bab03_mdp import acak, mdp_lorong
from bab06_mc import EPISODE_MINI, mc_prediksi

TERMINAL = (0, 4)


def langkah(ep):
    """Episode (s, a, r) -> daftar transisi (s, r, s_baru)."""
    hasil = []
    for t, (s, a, r) in enumerate(ep):
        s2 = ep[t + 1][0] if t + 1 < len(ep) else (0 if r == 2 else 4)
        hasil.append((s, r, s2))
    return hasil


def td0_episode(V, ep, alpha, g=0.5):
    delta = []
    for s, r, s2 in langkah(ep):
        target = r + (0.0 if s2 in TERMINAL else g * V[s2])
        d = target - V[s]
        V[s] += alpha * d
        delta.append(d)
    return delta


def batch_td(episode, g=0.5, alpha=0.01, tol=1e-12):
    """Ulangi semua transisi; pembaruan dijumlahkan per sapuan."""
    V = np.zeros(5)
    trans = [x for ep in episode for x in langkah(ep)]
    while True:
        inc = np.zeros(5)
        for s, r, s2 in trans:
            target = r + (0.0 if s2 in TERMINAL else g * V[s2])
            inc[s] += target - V[s]
        V += alpha * inc
        if np.abs(inc).max() < tol:
            return V


def kurva_rmse(metode, alpha, n_ep=100, ulang=100, benih=BENIH, v0=0.0):
    """RMSE (petak 1..3) terhadap V acak sesudah setiap episode."""
    v_benar = mdp_lorong().nilai(acak(mdp_lorong()))[1:4]
    rng = np.random.default_rng(benih)
    env = Lorong(benih=benih + 1)
    hasil = np.zeros(n_ep)
    for _ in range(ulang):
        V = np.full(5, v0)
        V[[0, 4]] = 0.0
        for i in range(n_ep):
            ep = jalankan_episode(env, ACAK, rng)
            if metode == "td":
                td0_episode(V, ep, alpha)
            else:
                G = 0.0
                for s, a, r in reversed(ep):
                    G = r + 0.5 * G
                    V[s] += alpha * (G - V[s])
            hasil[i] += np.sqrt(((V[1:4] - v_benar) ** 2).mean())
    return hasil / ulang


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("(1) TD(0) pada tiga episode mini, alpha = 1/2, V0 = 0:")
    V = np.zeros(5)
    for i, ep in enumerate(EPISODE_MINI, 1):
        d = td0_episode(V, ep, 0.5)
        print(f"    E{i}: galat TD = {np.round(d, 4).tolist()}"
              f"  V(1..3) = {V[1:4]}")

    print("(2) Data tetap tiga episode, diulang sampai konvergen:")
    print("    batch MC every-visit =",
          mc_prediksi(EPISODE_MINI, pertama=False)[0][1:4])
    print("    batch TD(0)          =", batch_td(EPISODE_MINI)[1:4])
    print("    nilai sebenarnya     =",
          mdp_lorong().nilai(acak(mdp_lorong()))[1:4])

    print("(3) Varians sasaran di petak 2 (V benar), 200 000 sampel:")
    rng = np.random.default_rng(BENIH)
    env = Lorong(benih=BENIH + 1)
    v = mdp_lorong().nilai(acak(mdp_lorong()))
    sas_td, sas_mc = [], []
    for _ in range(200_000):
        ep = jalankan_episode(env, ACAK, rng)
        s, r, s2 = langkah(ep)[0]
        sas_td.append(r + 0.5 * v[s2])
        sas_mc.append(sum(0.5 ** t * x for t, (_, _, x) in enumerate(ep)))
    print(f"    sasaran TD: rata {np.mean(sas_td):.4f}, var"
          f" {np.var(sas_td):.4f}")
    print(f"    sasaran MC: rata {np.mean(sas_mc):.4f}, var"
          f" {np.var(sas_mc):.4f}")

    print("(4) RMSE V(1..3), kebijakan acak, rata-rata 100 ulangan:")
    print("    metode  alpha   ep 10   ep 30  ep 100")
    for metode, alphas in (("TD", (0.05, 0.1, 0.2)),
                           ("MC", (0.05, 0.1, 0.2))):
        for a in alphas:
            k = kurva_rmse(metode.lower(), a)
            print(f"    {metode:6s} {a:5.2f} {k[9]:7.4f} {k[29]:7.4f}"
                  f" {k[99]:7.4f}")
