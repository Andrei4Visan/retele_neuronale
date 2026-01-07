# FILE: model_hopfield.py
# Hopfield Network (discrete) FROM SCRATCH - demo memorie asociativa
# Nu foloseste datasetul economic, e un demo clasic cerut de tema.

import numpy as np
import matplotlib.pyplot as plt


def to_bipolar(x01: np.ndarray) -> np.ndarray:
    # 0/1 -> -1/+1
    return np.where(x01 > 0, 1, -1).astype(int)


def to_01(xpm: np.ndarray) -> np.ndarray:
    # -1/+1 -> 0/1
    return np.where(xpm > 0, 1, 0).astype(int)


def train_hopfield(patterns_pm1: np.ndarray) -> np.ndarray:
    # Hebb rule: W = sum(p p^T), diag=0, optional normalize
    n = patterns_pm1.shape[1]
    W = np.zeros((n, n), dtype=float)
    for p in patterns_pm1:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0.0)
    W /= n
    return W


def energy(W: np.ndarray, s: np.ndarray) -> float:
    return float(-0.5 * s @ W @ s)


def recall(W: np.ndarray, s0: np.ndarray, steps=30, async_update=True, seed=42):
    rng = np.random.default_rng(seed)
    s = s0.copy()
    energies = [energy(W, s)]

    n = len(s)
    for _ in range(steps):
        if async_update:
            order = rng.permutation(n)
            for i in order:
                h = W[i] @ s
                s[i] = 1 if h >= 0 else -1
        else:
            h = W @ s
            s = np.where(h >= 0, 1, -1)

        energies.append(energy(W, s))

    return s, energies


def add_noise(x01: np.ndarray, noise_rate=0.2, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = x01.copy()
    n = x.size
    k = int(noise_rate * n)
    idx = rng.choice(n, size=k, replace=False)
    x_flat = x.reshape(-1)
    x_flat[idx] = 1 - x_flat[idx]
    return x_flat.reshape(x.shape)


def show_pattern(ax, x01, title):
    ax.imshow(x01, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


def main():
    # 3 patternuri simple 10x10 (litere stilizate)
    A = np.array([
        [0,0,0,1,1,1,1,0,0,0],
        [0,0,1,1,0,0,1,1,0,0],
        [0,1,1,0,0,0,0,1,1,0],
        [0,1,1,0,0,0,0,1,1,0],
        [0,1,1,1,1,1,1,1,1,0],
        [0,1,1,0,0,0,0,1,1,0],
        [0,1,1,0,0,0,0,1,1,0],
        [0,1,1,0,0,0,0,1,1,0],
        [0,1,1,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0],
    ], dtype=int)

    H = np.array([
        [0,1,1,0,0,0,0,1,1,0],
        [0,1,1,0,0,0,0,1,1,0],
        [0,1,1,0,0,0,0,1,1,0],
        [0,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,0],
        [0,1,1,0,0,0,0,1,1,0],
        [0,1,1,0,0,0,0,1,1,0],
        [0,1,1,0,0,0,0,1,1,0],
        [0,1,1,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0],
    ], dtype=int)

    T = np.array([
        [0,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
    ], dtype=int)

    patterns01 = np.array([A, H, T])
    P = patterns01.shape[0]
    n = patterns01[0].size

    patterns_pm1 = np.array([to_bipolar(p.reshape(-1)) for p in patterns01])
    W = train_hopfield(patterns_pm1)

    # test: luam pattern A si il corupem
    A_noisy = add_noise(A, noise_rate=0.25, seed=1)
    s0 = to_bipolar(A_noisy.reshape(-1))

    s_rec, energies = recall(W, s0, steps=25, async_update=True, seed=1)
    A_rec = to_01(s_rec).reshape(A.shape)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    show_pattern(axes[0], A, "Original A")
    show_pattern(axes[1], A_noisy, "Noisy A")
    show_pattern(axes[2], A_rec, "Recovered")
    plt.tight_layout()
    plt.savefig("hopfield_recall_demo.png", dpi=150)
    plt.show()

    plt.figure()
    plt.plot(energies)
    plt.title("Hopfield: Energy decreases (should go down)")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.savefig("hopfield_energy_curve.png", dpi=150)
    plt.show()

    print("Saved: hopfield_recall_demo.png, hopfield_energy_curve.png")


if __name__ == "__main__":
    main()
