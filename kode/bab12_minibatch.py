"""Bab 12: sasaran DQN dan double DQN, loss Huber, dan dueling pada
satu minibatch tiga transisi; NumPy dicocokkan dengan PyTorch.

Dipakai bab-bab berikutnya:
  huber(e), gradien_huber(e)    loss Huber (delta = 1) dan turunannya
  dueling(V, A)                 Q = V + A - rata-rata A
"""
import numpy as np
import torch

G = 0.9
R = np.array([1.0, 1.0, 1.0])
AKHIR = np.array([0.0, 0.0, 1.0])
Q_SA = np.array([2.5, 1.2, 0.4])              # Q_online(s, a)
Q_TARGET_S2 = np.array([[2.0, 3.0], [1.0, 0.5], [0.0, 0.0]])
Q_ONLINE_S2 = np.array([[4.0, 1.0], [0.0, 2.0], [0.0, 0.0]])


def huber(e):
    a = np.abs(e)
    return np.where(a <= 1, 0.5 * e ** 2, a - 0.5)


def gradien_huber(e):
    return np.clip(e, -1, 1)


def dueling(V, A):
    return V + A - A.mean()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    y_dqn = R + G * (1 - AKHIR) * Q_TARGET_S2.max(1)
    a2 = Q_ONLINE_S2.argmax(1)
    y_dbl = R + G * (1 - AKHIR) * Q_TARGET_S2[np.arange(3), a2]
    print("(1) Minibatch tiga transisi, gamma = 0.9:")
    print("    sasaran DQN        y =", y_dqn)
    print("    sasaran double DQN y =", y_dbl)
    e = Q_SA - y_dqn
    print("    galat q - y (DQN)    =", e)
    print(f"    loss kuadrat rata-rata = {np.mean(e ** 2):.4f},"
          f" loss Huber = {huber(e).mean():.4f}")
    print("    dL_Huber/dq          =", gradien_huber(e) / 3)
    q = torch.tensor(Q_SA, requires_grad=True)
    L = torch.nn.functional.smooth_l1_loss(q, torch.tensor(y_dqn))
    L.backward()
    print(f"    PyTorch: loss = {L.item():.4f}, grad = "
          f"{q.grad.numpy()}")
    print("(2) Dueling: V = 2, A = (1, -1, 0.5)")
    print("    Q =", dueling(2.0, np.array([1.0, -1.0, 0.5])))
