"""Bab 16: ringkasan DDPG, TD3, SAC pada Pendulum dari
data/bab16_kontinu.npz."""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "bab16_kontinu.npz"

if __name__ == "__main__":
    h = np.load(DATA)
    print("(4) Pendulum, 15 000 langkah, 3 benih; return greedy rata-rata")
    print("    5 episode setiap 1000 langkah (acak: -1241, maks 0)")
    print("    algo    1k     3k     5k    10k    15k  > -300 sejak")
    for algo in ("ddpg", "td3", "sac"):
        k = np.array([h[f"{algo}_{u}"] for u in range(3)])
        m = k.mean(0)
        lulus = np.flatnonzero(m > -300)
        sejak = f"{1000 * (lulus[0] + 1)}" if len(lulus) else "-"
        print(f"    {algo:5s}" + "".join(f" {m[i]:6.0f}"
                                      for i in (0, 2, 4, 9, 14))
              + f" {sejak:>9s}")
