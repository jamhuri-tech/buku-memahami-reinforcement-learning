"""Bab 5: pemrograman dinamis pada lorong licin dan grid es 8 x 8.

Dipakai bab-bab berikutnya:
  PETA_GRID                   peta grid es (S awal, L lubang, G tujuan)
  mdp_grid(peta, licin, g)    grid es sebagai MDP (empat aksi)
  evaluasi_iteratif(...)      evaluasi kebijakan, serentak atau di tempat
  iterasi_kebijakan(mdp)      policy iteration; riwayat kebijakan
  iterasi_nilai(mdp, ...)     value iteration; riwayat residu
"""
import time

import numpy as np

from bab03_mdp import MDP, mdp_lorong
from bab04_bellman import T_opt, greedy

PETA_GRID = ["S.......",
             "..L.....",
             "....L...",
             ".L....L.",
             "...L....",
             "......L.",
             ".L..L...",
             "...L...G"]
ARAH = [(-1, 0), (0, 1), (1, 0), (0, -1)]       # atas, kanan, bawah, kiri


def mdp_grid(peta=PETA_GRID, licin=0.2, gamma=0.99):
    """Aksi berhasil dengan peluang 1 - licin; sisanya dibagi rata ke
    dua arah tegak lurus. Menabrak dinding berarti diam. Masuk G memberi
    imbalan 1; G dan L terminal."""
    n, m = len(peta), len(peta[0])
    nS = n * m
    P = np.zeros((nS, 4, nS))
    R = np.zeros((nS, 4, nS))
    terminal = [i * m + j for i in range(n) for j in range(m)
                if peta[i][j] in "LG"]
    for i in range(n):
        for j in range(m):
            s = i * m + j
            for a in range(4):
                if s in terminal:
                    P[s, a, s] = 1.0
                    continue
                for b, p in ((a, 1 - licin), ((a + 1) % 4, licin / 2),
                             ((a + 3) % 4, licin / 2)):
                    di, dj = ARAH[b]
                    i2 = min(max(i + di, 0), n - 1)
                    j2 = min(max(j + dj, 0), m - 1)
                    s2 = i2 * m + j2
                    P[s, a, s2] += p
                    R[s, a, s2] = 1.0 if peta[i2][j2] == "G" else 0.0
    return MDP(P, R, gamma, terminal=terminal, awal=0,
               nama_aksi=("atas", "kanan", "bawah", "kiri"))


def evaluasi_iteratif(mdp, pi, theta=1e-10, ditempat=False, v0=None,
                      maks=100_000):
    """Mengembalikan v dan banyaknya sapuan."""
    Ppi, rpi = mdp.rantai(pi)
    v = np.zeros(mdp.nS) if v0 is None else v0.copy()
    for k in range(1, maks + 1):
        if ditempat:
            delta = 0.0
            for s in np.where(~mdp.terminal)[0]:
                baru = rpi[s] + mdp.gamma * Ppi[s] @ v
                delta = max(delta, abs(baru - v[s]))
                v[s] = baru
        else:
            baru = np.where(mdp.terminal, 0.0, rpi + mdp.gamma * Ppi @ v)
            delta = np.abs(baru - v).max()
            v = baru
        if delta < theta:
            return v, k
    return v, maks


def iterasi_kebijakan(mdp, pi0=None):
    """Policy iteration dengan evaluasi eksak. Riwayat: daftar (pi, v)."""
    pi = greedy(mdp, np.zeros(mdp.nS)) if pi0 is None else pi0
    riwayat = []
    while True:
        v = mdp.nilai(pi)
        riwayat.append((pi, v))
        pi_baru = greedy(mdp, v)
        if np.array_equal(pi_baru, pi):
            return pi, v, riwayat
        pi = pi_baru


def iterasi_nilai(mdp, theta=1e-10, ditempat=False, maks=100_000):
    """Value iteration. Mengembalikan v, banyaknya sapuan, residu."""
    v = np.zeros(mdp.nS)
    residu = []
    r_sa = mdp.r_sa()
    for k in range(1, maks + 1):
        if ditempat:
            delta = 0.0
            for s in np.where(~mdp.terminal)[0]:
                baru = (r_sa[s] + mdp.gamma * mdp.P[s] @ v).max()
                delta = max(delta, abs(baru - v[s]))
                v[s] = baru
        else:
            baru = T_opt(mdp, v)
            delta = np.abs(baru - v).max()
            v = baru
        residu.append(delta)
        if delta < theta:
            break
    return v, k, residu


def kode_lorong(pi):
    return "".join("-+"[int(pi[s, 1])] for s in (1, 2, 3))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    m = mdp_lorong()
    print("(1) Policy iteration pada lorong licin dari ---:")
    pi0 = np.zeros((5, 2))
    pi0[:, 0] = 1.0
    _, _, riw = iterasi_kebijakan(m, pi0)
    for i, (p, v) in enumerate(riw):
        print(f"    iterasi {i}: {kode_lorong(p)}  V = {v[1:4]}")

    print("(2) Grid es 8 x 8: banyaknya sapuan sampai residu < 1e-10")
    print("    gamma  metode              sapuan   V(awal)")
    for g in (0.9, 0.99):
        mg = mdp_grid(gamma=g)
        pi, v_pi, riw = iterasi_kebijakan(mg)
        print(f"    {g:<5}  policy iteration  {len(riw):8d}   {v_pi[0]:.6f}")
        for nama, dt in (("VI serentak", False), ("VI di tempat", True)):
            v, k, _ = iterasi_nilai(mg, ditempat=dt)
            print(f"    {g:<5}  {nama:17s} {k:8d}   {v[0]:.6f}")
        v_vi, _, _ = iterasi_nilai(mg, theta=1e-12)
        sama = np.array_equal(greedy(mg, v_vi).argmax(1), pi.argmax(1))
        print(f"           selisih maks V = {np.abs(v_vi - v_pi).max():.1e},"
              f" kebijakan sama: {sama}")

    print("(3) VI serentak, gamma = 0.99: nilai lawan kebijakan greedy")
    print("     k   ||v_k - V*||  rugi greedy  batas 2g/(1-g)||.||")
    mg = mdp_grid(gamma=0.99)
    _, v_opt, _ = iterasi_kebijakan(mg)
    v = np.zeros(mg.nS)
    for k in range(1, 201):
        v = T_opt(mg, v)
        if k in (1, 5, 10, 20, 40, 80, 160, 200):
            gal = np.abs(v - v_opt).max()
            rugi = np.abs(mg.nilai(greedy(mg, v)) - v_opt).max()
            print(f"    {k:3d}   {gal:11.2e}  {rugi:11.2e}"
                  f"  {2 * 0.99 * gal / 0.01:11.2e}")
