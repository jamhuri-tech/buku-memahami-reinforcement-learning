"""Bab 14: ringkasan A2C CartPole dari data/bab14_a2c.npz."""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "bab14_a2c.npz"
VARIAN = [("lam0", "lambda = 0"), ("lam095", "lambda = 0.95"),
          ("lam1", "lambda = 1"), ("berbagi", "berbagi, 0.95")]


def blok(k, a, b):
    m = (k[:, 0] >= a) & (k[:, 0] < b)
    return k[m, 1].mean()


if __name__ == "__main__":
    h = np.load(DATA)
    print("(3) A2C pada CartPole, N = 8, n = 16, 300 000 langkah, 3 benih")
    print("    return episode rata-rata per blok langkah (ribu)")
    print("    varian          0-60  60-120 180-240 240-300  akhir/benih")
    for kunci, nama in VARIAN:
        ks = [h[f"{kunci}_{u}"] for u in range(3)]
        b = [np.mean([blok(k, a, a + 60_000) for k in ks])
             for a in (0, 60_000, 180_000, 240_000)]
        akhir = [k[-100:, 1].mean() for k in ks]
        print(f"    {nama:14s}" + "".join(f" {x:6.0f}" for x in b[:2])
              + "".join(f" {x:7.0f}" for x in b[2:]) + "  "
              + " ".join(f"{x:3.0f}" for x in akhir))
