import numpy as np

def step(x):
    return 1 if x >= 0 else 0

class Perceptron:
    def __init__(self, lr=0.1, n_epochs=50, seed=42):
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0, 0.1, size=X.shape[1])
        self.b = 0.0

        for epoch in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                y_hat = step(np.dot(self.w, xi) + self.b)
                err = yi - y_hat
                if err != 0:
                    self.w += self.lr * err * xi
                    self.b += self.lr * err
                    errors += 1
            if errors == 0:
                break

    def predict(self, X):
        return np.array([step(np.dot(self.w, xi) + self.b) for xi in X])

def run_gate(name, X, y):
    p = Perceptron(lr=0.1, n_epochs=50, seed=42)
    p.fit(X, y)
    pred = p.predict(X)
    acc = (pred == y).mean()
    print(f"{name} -> pred={pred.tolist()} acc={acc:.2f} w={p.w} b={p.b:.3f}")

def main():
    print("=== PERCEPTRON FROM SCRATCH (LOGIC GATES) ===")

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)

    y_and = np.array([0, 0, 0, 1])
    y_or  = np.array([0, 1, 1, 1])
    y_xor = np.array([0, 1, 1, 0])

    run_gate("AND", X, y_and)
    run_gate("OR",  X, y_or)
    run_gate("XOR (should fail)", X, y_xor)

if __name__ == "__main__":
    main()
