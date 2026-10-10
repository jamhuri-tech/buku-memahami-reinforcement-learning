"""Bab 15: PPO-clip pada CartPole (PyTorch). (Pelatihan +-8 menit.)

Varian: "clip" (eps = 0.2) dan "tanpa_clip" (tujuan r A tanpa
pemotongan, epoch dan minibatch sama), masing-masing dengan konfigurasi
standar (lr 2.5e-4, 4 epoch) dan agresif (lr 1e-3, 10 epoch). Hasil di data/bab15_ppo.npz;
bab15_hasil.py mencetaknya.
# cek_keluaran: lewati
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from bab12_cartpole import CartPole
from bab14_a2c import AktorKritik

BENIH = 20261009
DATA = Path(__file__).resolve().parent.parent / "data" / "bab15_ppo.npz"


def latih(varian, benih, N=8, n=128, langkah_total=300_000, lr=2.5e-4,
          g=0.99, lam=0.95, eps=0.2, epoch=4, mb=256, c_ent=0.01):
    torch.manual_seed(benih)
    rng = np.random.default_rng(benih)
    envs = [CartPole() for _ in range(N)]
    S = np.array([e.reset(rng) for e in envs], np.float32)
    net = AktorKritik()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ret = np.zeros(N)
    kurva, rasio_maks = [], []
    t = 0
    while t < langkah_total:
        bS, bA, bR, bD, bT, bLP, bV = [], [], [], [], [], [], []
        for _ in range(n):
            with torch.no_grad():
                logit, v = net(torch.as_tensor(S))
                logp = torch.log_softmax(logit, -1).numpy()
            A = (rng.random(N) < np.exp(logp[:, 1])).astype(int)
            S2 = np.zeros_like(S)
            D = np.zeros(N, np.float32)
            T = np.zeros(N, np.float32)
            for i, e in enumerate(envs):
                s2, r, term, potong = e.step(A[i])
                ret[i] += r
                D[i] = term
                if term or potong:
                    T[i] = 1
                    kurva.append((t + i, ret[i]))
                    ret[i] = 0
                    s2 = e.reset(rng)
                S2[i] = s2
            bS.append(S.copy()); bA.append(A); bR.append(np.ones(N))
            bD.append(D); bT.append(T)
            bLP.append(logp[np.arange(N), A]); bV.append(v.numpy())
            S = S2
            t += N
        with torch.no_grad():
            v_berikut = net(torch.as_tensor(S))[1].numpy()
        adv = np.zeros((n, N), np.float32)
        acc = np.zeros(N, np.float32)
        for k in range(n - 1, -1, -1):
            delta = bR[k] + g * (1 - bD[k]) * v_berikut - bV[k]
            acc = delta + g * lam * (1 - bT[k]) * acc
            adv[k] = acc
            v_berikut = bV[k]
        sasaran = adv + np.array(bV)
        tS = torch.as_tensor(np.array(bS).reshape(-1, 4))
        tA = torch.as_tensor(np.array(bA).reshape(-1))
        tLP = torch.as_tensor(np.array(bLP).reshape(-1), dtype=torch.float32)
        tAdv = torch.as_tensor(adv.reshape(-1))
        tG = torch.as_tensor(sasaran.reshape(-1), dtype=torch.float32)
        jumlah = n * N
        for _ in range(epoch):
            urut = rng.permutation(jumlah)
            for a in range(0, jumlah, mb):
                i = torch.as_tensor(urut[a:a + mb])
                logit, v = net(tS[i])
                logp = torch.log_softmax(logit, -1)
                lp = logp.gather(1, tA[i][:, None]).squeeze(1)
                r = torch.exp(lp - tLP[i])
                Ab = tAdv[i]
                Ab = (Ab - Ab.mean()) / (Ab.std() + 1e-8)
                if varian == "clip":
                    obj = torch.min(r * Ab, torch.clamp(r, 1 - eps, 1 + eps)
                                    * Ab)
                else:
                    obj = r * Ab
                ent = -(logp.exp() * logp).sum(-1).mean()
                loss = (-obj.mean() + 0.5 * ((tG[i] - v) ** 2).mean()
                        - c_ent * ent)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                rasio_maks.append(float(r.max()))
    return np.array(kurva), np.array(rasio_maks)


if __name__ == "__main__":
    torch.set_num_threads(1)
    hasil = {}
    for konfig, kw in (("std", {}), ("agresif", {"lr": 1e-3, "epoch": 10})):
        for varian in ("clip", "tanpa_clip"):
            for u in range(3):
                t0 = time.time()
                k, rm = latih(varian, BENIH + u, **kw)
                hasil[f"{konfig}_{varian}_{u}"] = k
                hasil[f"{konfig}_{varian}_{u}_rasio"] = rm
                print(f"{konfig} {varian} {u}: 100 episode terakhir"
                      f" {k[-100:, 1].mean():.1f}, rasio maks {rm.max():.1f},"
                      f" {time.time() - t0:.0f} dtk", flush=True)
    DATA.parent.mkdir(exist_ok=True)
    np.savez(DATA, **hasil)
