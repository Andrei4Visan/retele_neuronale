# FILE: model_hamming_maxnet.py
# Hamming network + MAXNET (winner-take-all) - demo simplu
# Clasic: similaritate Hamming intre input si prototipuri, apoi MAXNET alege castigatorul.

import numpy as np


def to_bipolar(x01: np.ndarray) -> np.ndarray:
    return np.where(x01 > 0, 1, -1).astype(int)


def hamming_similarity(x_pm1: np.ndarray, prototypes_pm1: np.ndarray) -> np.ndarray:
    # Pentru -1/+1: similaritate = nr potriviri
    # potrivire cand x_i == p_i
    matches = (prototypes_pm1 == x_pm1).sum(axis=1)
    return matches.astype(float)


def maxnet(a: np.ndarray, eps=0.1, steps=50):
    # MAXNET iterative inhibition: a_i(t+1)=max(0, a_i(t) - eps*sum_{j!=i} a_j(t))
    a = a.astype(float).copy()
    n = len(a)
    for _ in range(steps):
        a_next = a.copy()
        s = a.sum()
        for i in range(n):
            a_next[i] = max(0.0, a[i] - eps * (s - a[i]))
        a = a_next
        if (a > 0).sum() <= 1:
            break
    return a


def main():
    # prototipuri binare scurte (8 biti) doar pentru demo
    P1 = np.array([1,0,1,0,1,0,1,0])
    P2 = np.array([1,1,1,0,0,0,1,1])
    P3 = np.array([0,0,0,1,1,1,0,0])

    prototypes = np.stack([P1, P2, P3], axis=0)
    prot_pm1 = np.array([to_bipolar(p) for p in prototypes])

    x = np.array([1,1,1,0,0,0,1,0])  # input aproape de P2
    x_pm1 = to_bipolar(x)

    sim = hamming_similarity(x_pm1, prot_pm1)
    print("Hamming similarity:", sim)

    a_final = maxnet(sim, eps=0.2, steps=50)
    winner = int(np.argmax(a_final))
    print("MAXNET activations:", a_final)
    print("Winner index:", winner, "(0=P1, 1=P2, 2=P3)")


if __name__ == "__main__":
    main()
