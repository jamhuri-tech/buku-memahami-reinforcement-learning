"""Bab 12: ringkasan pelatihan DQN dari data/bab12_dqn.npz."""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "bab12_dqn.npz"
VARIAN = [("dqn", "DQN lengkap"), ("double", "double DQN"),
          ("tanpa_target", "tanpa target"), ("tanpa_replay", "tanpa replay")]

if __name__ == "__main__":
    h = np.load(DATA)
    print("(4) CartPole, 60 000 langkah, 3 benih (return maks 500)")
    print("    return greedy rata-rata 10 episode, rata-rata 3 benih")
    print("    varian          20k    30k    40k    50k    60k  episode")
    for kunci, nama in VARIAN:
        ev = np.array([h[f"{kunci}_{u}_evalk"] for u in range(3)])
        m = ev.mean(0)
        n_ep = np.mean([len(h[f"{kunci}_{u}_kurva"]) for u in range(3)])
        print(f"    {nama:13s}" + "".join(f" {m[k]:6.0f}"
                                         for k in (3, 5, 7, 9, 11))
              + f" {n_ep:8.0f}")
