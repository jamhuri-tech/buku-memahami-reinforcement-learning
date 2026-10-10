"""Bab 12: DQN dengan PyTorch pada CartPole. (Pelatihan +-20 menit.)

Varian: "dqn" (replay + jaringan target), "tanpa_target", "tanpa_replay",
"double". Hasil disimpan di data/bab12_dqn.npz; bab12_hasil.py
mencetaknya. Jalankan: python3 kode/bab12_dqn.py
# cek_keluaran: lewati
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from bab12_cartpole import CartPole

BENIH = 20261009
DATA = Path(__file__).resolve().parent.parent / "data" / "bab12_dqn.npz"


def jaringan(dueling=False):
    return nn.Sequential(nn.Linear(4, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))


def sasaran_dqn(r, s2, akhir, Q_target, Q_online=None, g=0.99):
    """y = r + g max_a Q_target(s2, a), atau double DQN bila Q_online
    diberikan: aksi dipilih Q_online, dinilai Q_target."""
    with torch.no_grad():
        if Q_online is None:
            q2 = Q_target(s2).max(1).values
        else:
            a2 = Q_online(s2).argmax(1, keepdim=True)
            q2 = Q_target(s2).gather(1, a2).squeeze(1)
        return r + g * (1 - akhir) * q2


def nilai_greedy(Q, rng, n):
    env = CartPole()
    nilai = []
    for _ in range(n):
        s, ret = env.reset(rng), 0.0
        while True:
            with torch.no_grad():
                a = int(Q(torch.as_tensor(s, dtype=torch.float32)).argmax())
            s, r, term, potong = env.step(a)
            ret += r
            if term or potong:
                break
        nilai.append(ret)
    return float(np.mean(nilai))


def latih(varian, benih, langkah_total=60_000, ukuran_replay=50_000,
          batch=128, lr=3e-4, g=0.99, sinkron=500, eps_akhir=0.05,
          eps_langkah=10_000, mulai=1000, tiap_eval=5000, klip=10.0,
          tiap_latih=1, tau=0.005):
    torch.manual_seed(benih)
    rng = np.random.default_rng(benih)
    env = CartPole()
    Q = jaringan()
    Qt = jaringan()
    Qt.load_state_dict(Q.state_dict())
    opt = torch.optim.Adam(Q.parameters(), lr=lr)
    if varian == "tanpa_replay":
        ukuran_replay = batch
    S = np.zeros((ukuran_replay, 4), np.float32)
    A = np.zeros(ukuran_replay, np.int64)
    Rw = np.zeros(ukuran_replay, np.float32)
    S2 = np.zeros((ukuran_replay, 4), np.float32)
    Akhir = np.zeros(ukuran_replay, np.float32)
    isi, ptr = 0, 0
    s = env.reset(rng)
    ret, kurva, ep_langkah, evaluasi = 0.0, [], [], []
    rng_eval = np.random.default_rng(benih + 7)
    for t in range(langkah_total):
        eps = max(eps_akhir, 1 - (1 - eps_akhir) * t / eps_langkah)
        if rng.random() < eps:
            a = int(rng.integers(2))
        else:
            with torch.no_grad():
                a = int(Q(torch.as_tensor(s, dtype=torch.float32)).argmax())
        s2, r, term, potong = env.step(a)
        S[ptr], A[ptr], Rw[ptr], S2[ptr], Akhir[ptr] = s, a, r, s2, term
        ptr = (ptr + 1) % ukuran_replay
        isi = min(isi + 1, ukuran_replay)
        ret += r
        s = s2
        if term or potong:
            kurva.append(ret)
            ep_langkah.append(t + 1)
            ret, s = 0.0, env.reset(rng)
        if (t >= mulai or (varian == "tanpa_replay" and isi == batch)) \
                and t % tiap_latih == 0:
            if varian == "tanpa_replay":
                idx = np.arange(isi)
            else:
                idx = rng.integers(isi, size=batch)
            tb = [torch.as_tensor(x[idx]) for x in (S, A, Rw, S2, Akhir)]
            target_net = Q if varian == "tanpa_target" else Qt
            y = sasaran_dqn(tb[2], tb[3], tb[4], target_net,
                            Q if varian == "double" else None, g)
            q = Q(tb[0]).gather(1, tb[1][:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(Q.parameters(), klip)
            opt.step()
            if tau is not None:
                with torch.no_grad():
                    for pt, po in zip(Qt.parameters(), Q.parameters()):
                        pt.mul_(1 - tau).add_(tau * po)
        if tau is None and (t + 1) % sinkron == 0:
            Qt.load_state_dict(Q.state_dict())
        if (t + 1) % tiap_eval == 0:
            evaluasi.append(nilai_greedy(Q, rng_eval, 10))
    return np.array(kurva), np.array(ep_langkah), np.array(evaluasi)

if __name__ == "__main__":
    torch.set_num_threads(1)
    hasil = {}
    for varian in ("dqn", "double", "tanpa_target", "tanpa_replay"):
        for u in range(3):
            t0 = time.time()
            kurva, langkah, ev = latih(varian, BENIH + u)
            hasil[f"{varian}_{u}_kurva"] = kurva
            hasil[f"{varian}_{u}_langkah"] = langkah
            hasil[f"{varian}_{u}_evalk"] = ev
            print(f"{varian} {u}: {len(kurva)} episode, eval {ev.round()},"
                  f" {time.time() - t0:.0f} dtk", flush=True)
    DATA.parent.mkdir(exist_ok=True)
    np.savez(DATA, **hasil)
