import numpy as np
class Policy:
    """numpy MLP (ELU) from bc_fit.py: obs -> (cmd - q, tau_ff)."""
    def __init__(self, path):
        z = np.load(path); self.mask = z["mask"] if "mask" in z.files else 1.0; self.mu, self.sd, self.ymu, self.ysd = z["mu"], z["sd"], z["ymu"], z["ysd"]
        self.W = [z[f"W{i}"] for i in range(int(z["nl"]))]; self.b = [z[f"b{i}"] for i in range(int(z["nl"]))]
    def __call__(self, x):
        h = (x * self.mask - self.mu) / self.sd
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = W @ h + b
            if i < len(self.W) - 1: h = np.where(h > 0, h, np.exp(np.minimum(h, 0)) - 1)
        return h * self.ysd + self.ymu
