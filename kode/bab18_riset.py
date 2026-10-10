"""Bab 18: ringkasan statistik eksperimen RL lintas benih.

Dipakai bab-bab berikutnya:
  iqm(x)                       rata-rata interkuartil (25 % dipangkas)
  ci_bootstrap(x, f, rng)      selang persentil bootstrap 95 %
  uji_permutasi(x, y, rng)     nilai-p dua sisi untuk selisih rata-rata
  benih_perlu(sd, d)           banyaknya benih per kelompok (uji z)
"""
import numpy as np
from scipy import stats

from pathlib import Path

from bab13_reinforce import latih as latih_reinforce

DATA = Path(__file__).resolve().parent.parent / "data"

BENIH = 20261009


def iqm(x):
    return float(stats.trim_mean(np.asarray(x, float), 0.25))


def ci_bootstrap(x, f, rng, B=10_000):
    x = np.asarray(x, float)
    sampel = [f(rng.choice(x, len(x))) for _ in range(B)]
    return np.percentile(sampel, [2.5, 97.5])


def uji_permutasi(x, y, rng, B=20_000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d0 = abs(x.mean() - y.mean())
    gab = np.concatenate([x, y])
    hit = 0
    for _ in range(B):
        p = rng.permutation(gab)
        hit += abs(p[:len(x)].mean() - p[len(x):].mean()) >= d0 - 1e-12
    return hit / B


def benih_perlu(sd, d, z_a=1.96, z_b=0.8416):
    return 2 * (z_a + z_b) ** 2 * sd ** 2 / d ** 2


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    x = [120, 300, 480, 500, 500]
    print("(1) Lima benih: return akhir", x)
    print(f"    rata {np.mean(x):.1f}, median {np.median(x):.1f},"
          f" IQM {iqm(x):.1f}")
    lo, hi = ci_bootstrap(x, np.mean, rng)
    print(f"    selang bootstrap 95 % rata-rata: [{lo:.1f}, {hi:.1f}]")

    print("(2) REINFORCE lorong, episode 50, 40 benih per metode")
    a = np.array([latih_reinforce(50, 0.5, BENIH + u)[0][-1]
                  for u in range(40)])
    b = np.array([latih_reinforce(50, 0.5, BENIH + u, baseline=True)[0][-1]
                  for u in range(40)])
    for nama, v in (("tanpa baseline", a), ("baseline", b)):
        lo, hi = ci_bootstrap(v, np.mean, rng)
        print(f"    {nama:14s} rata {v.mean():.4f}, sd {v.std(ddof=1):.4f},"
              f" CI [{lo:.3f}, {hi:.3f}]")
    print(f"    benih dengan J < 2: tanpa baseline {(a < 2).sum()},"
          f" baseline {(b < 2).sum()}")
    print(f"    median: tanpa baseline {np.median(a):.4f},"
          f" baseline {np.median(b):.4f}")
    print(f"    uji permutasi 40 lawan 40: p = {uji_permutasi(a, b, rng):.4f}")
    salah = 0
    pval = []
    for _ in range(2000):
        i = rng.choice(40, 3, replace=False)
        j = rng.choice(40, 3, replace=False)
        salah += a[i].mean() >= b[j].mean()
    print(f"    3 lawan 3 benih: urutan terbalik {100 * salah / 2000:.1f} %")
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = b.mean() - a.mean()
    print(f"    benih per metode untuk kuasa 80 %: {benih_perlu(sd, d):.1f}")

    print("(3) Hasil akhir buku ini per benih (3 benih)")
    print("    percobaan                rata    IQM   CI rata-rata 95 %")
    h12 = np.load(DATA / "bab12_dqn.npz")
    h14 = np.load(DATA / "bab14_a2c.npz")
    h15 = np.load(DATA / "bab15_ppo.npz")
    h16 = np.load(DATA / "bab16_kontinu.npz")
    baris = [
        ("DQN lengkap, 60k", [h12[f"dqn_{u}_evalk"][-1] for u in range(3)]),
        ("A2C lambda 0", [h14[f"lam0_{u}"][-100:, 1].mean() for u in range(3)]),
        ("A2C lambda 0.95", [h14[f"lam095_{u}"][-100:, 1].mean()
                             for u in range(3)]),
        ("PPO agresif clip", [h15[f"agresif_clip_{u}"][-100:, 1].mean()
                              for u in range(3)]),
        ("SAC Pendulum", [h16[f"sac_{u}"][-1] for u in range(3)]),
        ("TD3 Pendulum", [h16[f"td3_{u}"][-1] for u in range(3)]),
    ]
    for nama, v in baris:
        lo, hi = ci_bootstrap(v, np.mean, rng)
        print(f"    {nama:20s} {np.mean(v):8.1f} {iqm(v):6.1f}"
              f"  [{lo:7.1f}, {hi:7.1f}]")
