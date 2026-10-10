"""Bab 16: DDPG, TD3, dan SAC pada Pendulum (PyTorch). (+-15 menit.)

Hasil disimpan di data/bab16_kontinu.npz; bab16_hasil.py mencetaknya.
# cek_keluaran: lewati
"""
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from bab16_pendulum import U_MAKS, Pendulum

BENIH = 20261009
DATA = Path(__file__).resolve().parent.parent / "data" / "bab16_kontinu.npz"


def mlp(masuk, keluar, h=128):
    return nn.Sequential(nn.Linear(masuk, h), nn.ReLU(), nn.Linear(h, h),
                         nn.ReLU(), nn.Linear(h, keluar))


class AktorGauss(nn.Module):
    """SAC: a = U_MAKS tanh(u), u ~ N(mu, sigma)."""

    def __init__(self):
        super().__init__()
        self.net = mlp(3, 2)

    def forward(self, s, deterministik=False):
        mu, log_sd = self.net(s).chunk(2, -1)
        log_sd = log_sd.clamp(-5, 2)
        if deterministik:
            return U_MAKS * torch.tanh(mu), None
        u = mu + log_sd.exp() * torch.randn_like(mu)
        logp = (-0.5 * ((u - mu) / log_sd.exp()) ** 2 - log_sd
                - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - (2 * (np.log(2) - u - nn.functional.softplus(-2 * u))
                       ).sum(-1) - np.log(U_MAKS)
        return U_MAKS * torch.tanh(u), logp


def evaluasi(pilih, rng, n=5):
    env = Pendulum()
    tot = []
    for _ in range(n):
        s, jumlah = env.reset(rng), 0.0
        for _ in range(200):
            s, r, _, _ = env.step(pilih(s))
            jumlah += r
        tot.append(jumlah)
    return float(np.mean(tot))


def latih(algo, benih, langkah_total=15_000, mulai=1000, batch=256,
          g=0.99, tau=0.005, lr=1e-3, tiap_eval=1000):
    torch.manual_seed(benih)
    rng = np.random.default_rng(benih)
    rng_eval = np.random.default_rng(benih + 7)
    env = Pendulum()
    if algo == "sac":
        aktor = AktorGauss()
        log_alfa = torch.zeros(1, requires_grad=True)
        opt_alfa = torch.optim.Adam([log_alfa], lr=lr)
    else:
        aktor = nn.Sequential(mlp(3, 1), nn.Tanh())
    aktor_t = copy.deepcopy(aktor)
    Q1, Q2 = mlp(4, 1), mlp(4, 1)
    Q1t, Q2t = copy.deepcopy(Q1), copy.deepcopy(Q2)
    opt_a = torch.optim.Adam(aktor.parameters(), lr=lr)
    opt_q = torch.optim.Adam(list(Q1.parameters()) + list(Q2.parameters()),
                             lr=lr)
    n = langkah_total
    S = np.zeros((n, 3), np.float32)
    A = np.zeros((n, 1), np.float32)
    R = np.zeros(n, np.float32)
    S2 = np.zeros((n, 3), np.float32)
    s = env.reset(rng)
    kurva = []

    def aksi_det(x):
        with torch.no_grad():
            xt = torch.as_tensor(x)
            if algo == "sac":
                return float(aktor(xt, True)[0])
            return float(U_MAKS * aktor(xt))

    for t in range(n):
        if t < mulai:
            a = rng.uniform(-U_MAKS, U_MAKS)
        elif algo == "sac":
            with torch.no_grad():
                a = float(aktor(torch.as_tensor(s))[0])
        else:
            a = float(np.clip(aksi_det(s) + 0.1 * U_MAKS * rng.normal(),
                              -U_MAKS, U_MAKS))
        s2, r, _, potong = env.step(a)
        S[t], A[t], R[t], S2[t] = s, a, r, s2
        s = env.reset(rng) if potong else s2
        if t >= mulai:
            idx = rng.integers(t + 1, size=batch)
            tS, tA, tR, tS2 = (torch.as_tensor(x[idx]) for x in (S, A, R, S2))
            with torch.no_grad():
                if algo == "sac":
                    a2, lp2 = aktor(tS2)
                    alfa = log_alfa.exp()
                    q2 = torch.min(Q1t(torch.cat([tS2, a2], -1)),
                                   Q2t(torch.cat([tS2, a2], -1))).squeeze(-1)
                    y = tR + g * (q2 - alfa * lp2)
                else:
                    a2 = U_MAKS * aktor_t(tS2)
                    if algo == "td3":
                        bising = (0.2 * U_MAKS * torch.randn_like(a2)).clamp(
                            -0.5 * U_MAKS, 0.5 * U_MAKS)
                        a2 = (a2 + bising).clamp(-U_MAKS, U_MAKS)
                        q2 = torch.min(Q1t(torch.cat([tS2, a2], -1)),
                                       Q2t(torch.cat([tS2, a2], -1)))
                    else:
                        q2 = Q1t(torch.cat([tS2, a2], -1))
                    y = tR + g * q2.squeeze(-1)
            sa = torch.cat([tS, tA], -1)
            loss_q = ((Q1(sa).squeeze(-1) - y) ** 2).mean()
            if algo != "ddpg":
                loss_q = loss_q + ((Q2(sa).squeeze(-1) - y) ** 2).mean()
            opt_q.zero_grad()
            loss_q.backward()
            opt_q.step()
            perbarui_aktor = algo != "td3" or t % 2 == 0
            if perbarui_aktor:
                if algo == "sac":
                    a_baru, lp = aktor(tS)
                    sa2 = torch.cat([tS, a_baru], -1)
                    q = torch.min(Q1(sa2), Q2(sa2)).squeeze(-1)
                    loss_a = (log_alfa.exp().detach() * lp - q).mean()
                    loss_alfa = -(log_alfa * (lp.detach() + 1.0)).mean()
                    opt_alfa.zero_grad()
                    loss_alfa.backward()
                    opt_alfa.step()
                else:
                    loss_a = -Q1(torch.cat([tS, U_MAKS * aktor(tS)], -1)).mean()
                opt_a.zero_grad()
                loss_a.backward()
                opt_a.step()
                with torch.no_grad():
                    pasangan = [(Q1, Q1t), (Q2, Q2t)]
                    if algo != "sac":
                        pasangan.append((aktor, aktor_t))
                    for net, net_t in pasangan:
                        for p, pt in zip(net.parameters(), net_t.parameters()):
                            pt.mul_(1 - tau).add_(tau * p)
        if (t + 1) % tiap_eval == 0:
            kurva.append(evaluasi(aksi_det, rng_eval))
    return np.array(kurva)


if __name__ == "__main__":
    torch.set_num_threads(1)
    hasil = {}
    for algo in ("ddpg", "td3", "sac"):
        for u in range(3):
            t0 = time.time()
            k = latih(algo, BENIH + u)
            hasil[f"{algo}_{u}"] = k
            print(f"{algo} {u}: {np.round(k[-5:])}, {time.time() - t0:.0f} dtk",
                  flush=True)
    DATA.parent.mkdir(exist_ok=True)
    np.savez(DATA, **hasil)
