"""Bab 13: REINFORCE dengan PyTorch pada CartPole. (Pelatihan +-10 menit.)

Varian: "tanpa" (tanpa baseline) dan "baseline" (jaringan nilai V(s)).
Hasil disimpan di data/bab13_reinforce.npz; bab13_hasil.py mencetaknya.
# cek_keluaran: lewati
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from bab12_cartpole import CartPole

BENIH = 20261009
DATA = Path(__file__).resolve().parent.parent / "data" / "bab13_reinforce.npz"


def mlp(keluar):
    return nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64),
                         nn.Tanh(), nn.Linear(64, keluar))


def latih(varian, benih, n_ep=1000, lr=1e-3, lr_v=1e-3, g=0.99):
    torch.manual_seed(benih)
    rng = np.random.default_rng(benih)
    env = CartPole()
    pi = mlp(2)
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    V = mlp(1)
    opt_v = torch.optim.Adam(V.parameters(), lr=lr_v)
    kurva = []
    for _ in range(n_ep):
        s, S, A, Rw = env.reset(rng), [], [], []
        while True:
            with torch.no_grad():
                p = torch.softmax(pi(torch.as_tensor(s, dtype=torch.float32)),
                                  -1).numpy()
            a = int(rng.random() < p[1])
            s2, r, term, potong = env.step(a)
            S.append(s)
            A.append(a)
            Rw.append(r)
            s = s2
            if term or potong:
                break
        G = np.zeros(len(Rw))
        acc = 0.0
        for t in range(len(Rw) - 1, -1, -1):
            acc = Rw[t] + g * acc
            G[t] = acc
        tS = torch.as_tensor(np.array(S), dtype=torch.float32)
        tG = torch.as_tensor(G, dtype=torch.float32)
        diskon = torch.as_tensor(g ** np.arange(len(G)), dtype=torch.float32)
        if varian == "baseline":
            v = V(tS).squeeze(1)
            loss_v = ((tG - v) ** 2).mean()
            opt_v.zero_grad()
            loss_v.backward()
            opt_v.step()
            adv = tG - v.detach()
        else:
            adv = tG
        logp = torch.log_softmax(pi(tS), -1)[torch.arange(len(A)),
                                               torch.as_tensor(A)]
        loss = -(diskon * adv * logp).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        kurva.append(len(Rw))
    return np.array(kurva)


if __name__ == "__main__":
    torch.set_num_threads(1)
    hasil = {}
    for varian in ("tanpa", "baseline"):
        for u in range(3):
            t0 = time.time()
            k = latih(varian, BENIH + u)
            hasil[f"{varian}_{u}"] = k
            print(f"{varian} {u}: rata 100 terakhir {k[-100:].mean():.1f},"
                  f" {time.time() - t0:.0f} dtk", flush=True)
    DATA.parent.mkdir(exist_ok=True)
    np.savez(DATA, **hasil)
