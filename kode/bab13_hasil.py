"""Bab 13: ringkasan REINFORCE CartPole dari data/bab13_reinforce.npz."""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "bab13_reinforce.npz"

if __name__ == "__main__":
    h = np.load(DATA)
    print("(4) REINFORCE pada CartPole, 1000 episode, 3 benih")
    print("    panjang episode rata-rata per blok 100 episode")
    print("    varian       1-100  401-500  901-1000   per benih (akhir)")
    for kunci, nama in (("tanpa", "tanpa"), ("baseline", "baseline")):
        k = np.array([h[f"{kunci}_{u}"] for u in range(3)])
        blok = [k[:, a:a + 100].mean() for a in (0, 400, 900)]
        akhir = k[:, 900:].mean(1)
        print(f"    {nama:10s}" + "".join(f" {b:8.1f}" for b in blok)
              + "   " + " ".join(f"{x:5.0f}" for x in akhir))
