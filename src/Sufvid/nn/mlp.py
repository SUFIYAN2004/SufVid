import numpy as np

def make_data(kind="moons", n=300, noise=0.12, seed=0):
    """Two-class 2-D datasets: 'moons', 'circles', 'spiral'. Returns X (n, 2), y (n,) of 0./1."""
    rng = np.random.default_rng(seed)
    h = n // 2
    if kind == "moons":
        a, b = rng.uniform(0, np.pi, h), rng.uniform(0, np.pi, n - h)
        X = np.vstack([np.column_stack([np.cos(a), np.sin(a)]),
                       np.column_stack([1 - np.cos(b), 0.5 - np.sin(b)])]) - [0.5, 0.25]
    elif kind == "circles":
        a = rng.uniform(0, 2 * np.pi, n)
        r = np.where(np.arange(n) < h, 0.45, 1.0)
        X = np.column_stack([r * np.cos(a), r * np.sin(a)])
    elif kind == "spiral":
        parts = []
        for c, m in ((0, h), (1, n - h)):
            r = np.sqrt(rng.uniform(0.03, 1.0, m))
            a = 3.2 * np.pi * r + c * np.pi
            parts.append(1.3 * np.column_stack([r * np.cos(a), r * np.sin(a)]))
        X = np.vstack(parts)
    else:
        raise KeyError(f"unknown dataset {kind!r}; choose from ['circles', 'moons', 'spiral']")
    X = X + rng.normal(0.0, noise, X.shape)
    y = (np.arange(n) >= h).astype(float)
    return X, y

def _forward(Ws, bs, X):
    acts = [X]
    for i in range(len(Ws) - 1):
        acts.append(np.tanh(acts[-1] @ Ws[i] + bs[i]))
    z = (acts[-1] @ Ws[-1] + bs[-1])[:, 0]
    return acts, z

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

def _loss_grad(Ws, bs, X, y):
    acts, z = _forward(Ws, bs, X)
    N = len(X)
    p = _sigmoid(z)
    loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
    d = ((p - y) / N)[:, None]
    gW, gb = [None] * len(Ws), [None] * len(Ws)
    for i in range(len(Ws) - 1, -1, -1):
        gW[i] = acts[i].T @ d
        gb[i] = d.sum(0)
        if i > 0:
            d = (d @ Ws[i].T) * (1.0 - acts[i] ** 2)
    return loss, gW, gb, p

def train_mlp(data="moons", hidden=(16,), epochs=600, lr=0.05, snapshots=40, res=64,
              bounds=None, n=300, noise=0.12, seed=0):
    """Train a small classifier and record everything needed to animate it.
    data: 'moons' | 'circles' | 'spiral' | (X, y) with X of shape (N, 2), y in {0, 1}.
    hidden=() gives plain logistic regression. Returns a dict:
      X, y, loss[epochs+1], acc[epochs+1]  (index = number of updates done),
      snap_epochs[S], grids[S, res, res] (P(class 1); row 0 = lowest y), bounds, weights."""
    if isinstance(data, str):
        X, y = make_data(data, n, noise, seed)
    else:
        X, y = data
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[1] != 2 or len(y) != len(X):
            raise ValueError("data must be a dataset name or (X, y) with X of shape (N, 2)")
    if epochs < 1 or snapshots < 2:
        raise ValueError("need epochs >= 1 and snapshots >= 2")
    if bounds is None:
        bounds = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    x0, x1, y0, y1 = bounds
    gx, gy = np.meshgrid(np.linspace(x0, x1, res), np.linspace(y0, y1, res))
    G = np.column_stack([gx.ravel(), gy.ravel()])

    rng = np.random.default_rng(seed)
    sizes = [2, *hidden, 1]
    Ws = [rng.normal(0.0, np.sqrt(1.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
    bs = [np.zeros(b) for b in sizes[1:]]
    P = Ws + bs
    m1 = [np.zeros_like(a) for a in P]
    m2 = [np.zeros_like(a) for a in P]
    b1, b2, eps = 0.9, 0.999, 1e-8

    snap = np.unique(np.round(epochs * np.linspace(0.0, 1.0, snapshots) ** 2).astype(int))
    snap_set = set(snap.tolist())
    loss_hist, acc_hist, grids = [], [], []
    for e in range(epochs + 1):
        loss, gW, gb, p = _loss_grad(Ws, bs, X, y)
        loss_hist.append(loss)
        acc_hist.append(float(np.mean((p > 0.5) == (y > 0.5))))
        if e in snap_set:
            _, zg = _forward(Ws, bs, G)
            grids.append(_sigmoid(zg).reshape(res, res))
        if e < epochs:
            step = e + 1
            for j, (a, g) in enumerate(zip(P, gW + gb)):
                m1[j] = b1 * m1[j] + (1 - b1) * g
                m2[j] = b2 * m2[j] + (1 - b2) * g * g
                a -= lr * (m1[j] / (1 - b1 ** step)) / (np.sqrt(m2[j] / (1 - b2 ** step)) + eps)
    return {"X": X, "y": y, "loss": np.array(loss_hist), "acc": np.array(acc_hist),
            "snap_epochs": snap, "grids": np.array(grids), "bounds": tuple(bounds),
            "weights": {"W": Ws, "b": bs}, "hidden": tuple(hidden), "seed": seed}
