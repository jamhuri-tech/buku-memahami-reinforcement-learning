"""Bab 14: A2C dengan GAE pada CartPole (PyTorch). (Pelatihan +-4 menit.)

N lingkungan berjalan serentak; setiap pembaruan memakai n langkah dari
setiap lingkungan. Hasil disimpan di data/bab14_a2c.npz; bab14_hasil.py
mencetaknya.
# cek_keluaran: lewati
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from bab12_cartpole import CartPole

BENIH = 20261009
DATA = Path(__file__).resolve().parent.parent / "data" / "bab14_a2c.npz"


class AktorKritik(nn.Module):
    """Aktor dan kritik dengan dua jaringan terpisah."""

    def __init__(self):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(4, 64), nn.Tanh(),
                                nn.Linear(64, 64), nn.Tanh(),
                                nn.Linear(64, 2))
        self.v = nn.Sequential(nn.Linear(4, 64), nn.Tanh(),
                               nn.Linear(64, 64), nn.Tanh(),
                               nn.Linear(64, 1))

    def forward(self, s):
        return self.pi(s), self.v(s).squeeze(-1)


class AktorKritikBerbagi(nn.Module):
    """Aktor dan kritik berbagi badan jaringan (untuk perbandingan)."""

    def __init__(self):
        super().__init__()
        self.tubuh = nn.Sequential(nn.Linear(4, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh())
        self.kepala_pi = nn.Linear(64, 2)
        self.kepala_v = nn.Linear(64, 1)

    def forward(self, s):
        h = self.tubuh(s)
        return self.kepala_pi(h), self.kepala_v(h).squeeze(-1)


def latih(lam, benih, N=8, n=16, langkah_total=300_000, lr=2e-3, g=0.99,
          c_v=0.5, c_ent=0.01, normal=True, klip=0.5, berbagi=False):
    torch.manual_seed(benih)
    rng = np.random.default_rng(benih)
    envs = [CartPole() for _ in range(N)]
    S = np.array([e.reset(rng) for e in envs], np.float32)
    net = AktorKritikBerbagi() if berbagi else AktorKritik()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ret = np.zeros(N)
    selesai_ep, kurva = [], []
    t = 0
    while t < langkah_total:
        bS, bA, bR, bD, bT = [], [], [], [], []
        for _ in range(n):
            with torch.no_grad():
                logit, _ = net(torch.as_tensor(S))
                p = torch.softmax(logit, -1).numpy()
            A = (rng.random(N) < p[:, 1]).astype(int)
            S2 = np.zeros_like(S)
            R = np.ones(N, np.float32)
            D = np.zeros(N, np.float32)          # terminal sungguhan
            T = np.zeros(N, np.float32)          # akhir episode (+ potong)
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
            bS.append(S.copy()); bA.append(A); bR.append(R); bD.append(D)
            bT.append(T)
            S = S2
            t += N
        with torch.no_grad():
            _, v_akhir = net(torch.as_tensor(S))
        tS = torch.as_tensor(np.array(bS))                # n x N x 4
        logit, v = net(tS)
        vn = v.detach().numpy()
        adv = np.zeros((n, N), np.float32)
        acc = np.zeros(N, np.float32)
        v_berikut = v_akhir.numpy()
        for k in range(n - 1, -1, -1):
            # bootstrap kecuali terminal; reset memutus rantai GAE
            delta = bR[k] + g * (1 - bD[k]) * v_berikut - vn[k]
            acc = delta + g * lam * (1 - bT[k]) * acc
            adv[k] = acc
            v_berikut = vn[k]
        sasaran = torch.as_tensor(adv + vn)
        if normal:                       # advantage dibakukan per batch
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logp = torch.log_softmax(logit, -1)
        tA = torch.as_tensor(np.array(bA))
        lp = logp.gather(-1, tA[..., None]).squeeze(-1)
        ent = -(logp.exp() * logp).sum(-1).mean()
        loss = (-(torch.as_tensor(adv) * lp).mean()
                + c_v * ((sasaran - v) ** 2).mean() - c_ent * ent)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), klip)
        opt.step()
    return np.array(kurva)


if __name__ == "__main__":
    torch.set_num_threads(1)
    hasil = {}
    for nama, lam, bb in (("lam0", 0.0, False), ("lam095", 0.95, False),
                          ("lam1", 1.0, False), ("berbagi", 0.95, True)):
        for u in range(3):
            t0 = time.time()
            k = latih(lam, BENIH + u, berbagi=bb)
            hasil[f"{nama}_{u}"] = k
            print(f"{nama} benih {u}: 100 episode terakhir"
                  f" {k[-100:, 1].mean():.1f}, {time.time() - t0:.0f} dtk",
                  flush=True)
    DATA.parent.mkdir(exist_ok=True)
    np.savez(DATA, **hasil)
