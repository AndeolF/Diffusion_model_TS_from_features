import torch  # type: ignore
import numpy as np


class TGiverFromError:
    def __init__(self, t_vals, errors, alpha=1.0):
        self.device = None  # Défini dynamiquement plus tard
        errors = np.array(errors) + 1e-6
        weights = errors**alpha
        weights /= weights.sum()

        self.t_vals = torch.tensor(t_vals, dtype=torch.float32)
        self.probs = torch.tensor(weights, dtype=torch.float32)

    def sample(self, batch_size, device):
        self.device = device
        indices = torch.multinomial(
            self.probs, num_samples=batch_size, replacement=True
        )
        return self.t_vals[indices].to(device)

    def __call__(self, batch_size):
        if self.device is None:
            raise RuntimeError(
                "Device not set. Call sample(batch_size, device) instead of __call__ the first time."
            )
        return self.sample(batch_size, self.device)

    def save(self, path):
        torch.save(
            {
                "t_vals": self.t_vals,
                "probs": self.probs,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        data = torch.load(path)
        obj = cls.__new__(cls)
        obj.t_vals = data["t_vals"]
        obj.probs = data["probs"]
        obj.device = None
        return obj
