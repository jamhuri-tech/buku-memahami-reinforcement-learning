"""Bab 15: ringkasan PPO CartPole dari data/bab15_ppo.npz."""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "bab15_ppo.npz"


def blok(k, a, b):
    m = (k[:, 0] >= a) & (k[:, 0] < b)
    return k[m, 1].mean()


if __name__ == "__main__":
    h = np.load(DATA)
    print("(5) PPO pada CartPole, N = 8, n = 128, 300 000 langkah")
    print("    rata-rata 3 benih per blok 60 ribu langkah; rasio r maks")
    print("    konfigurasi           0-60 60-120 120-180 240-300  r maks")
    for konfig in ("std", "agresif"):
        for varian in ("clip", "tanpa_clip"):
            ks = [h[f"{konfig}_{varian}_{u}"] for u in range(3)]
            b = [np.mean([blok(k, a, a + 60_000) for k in ks])
                 for a in (0, 60_000, 120_000, 240_000)]
            r = max(h[f"{konfig}_{varian}_{u}_rasio"].max() for u in range(3))
            nama = f"{konfig}, {varian.replace('_', ' ')}"
            print(f"    {nama:19s}" + "".join(f" {x:6.0f}" for x in b)
                  + f" {r:7.1f}")
