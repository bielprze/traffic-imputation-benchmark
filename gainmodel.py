"""
GAIN-only numerical branch ported from pinned author gain.py.
Training wrapper uses validation early stopping and is strictly
inductive across time splits.
"""

import copy
import json
import sys
import time
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Keep imports after required runtime/environment initialization.
from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

MODULE_DIR = Path(__file__).resolve().parent


def setup(seed):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


class GainNet(nn.Module):

    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.Sigmoid(),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(
                    layer.weight, std=(2 / layer.in_features) ** 0.5
                )
                nn.init.zeros_(layer.bias)

    def forward(self, x, m):
        return self.net(torch.cat([x, m], -1))


class Neural:

    def __init__(self, d, method, cfg, seed):
        if method != "GAIN":
            raise ValueError("Use grinmodel.ReferenceGRIN for GRIN")
        setup(seed)
        self.d, self.method, self.cfg, self.seed = (d, method, cfg, seed)
        self.device = torch.device(cfg.get("device", "cpu"))
        n = len(d["names"])
        tr = d["x"][: d["a"]]
        self.offset = np.nanmin(tr, 0)
        self.scale = np.nanmax(tr, 0) - self.offset + 1e-06
        self.model = GainNet(n)
        self.discriminator = GainNet(n)
        self.model.to(self.device)
        self.discriminator.to(self.device)

    def predict(self, x, index=None):
        self.model.eval()
        z = (x - self.offset) / self.scale
        obs = np.isfinite(z)
        clean = np.nan_to_num(z).astype("float32")
        with torch.no_grad():
            rng = np.random.default_rng(self.seed + 810000)
            v = np.where(obs, clean, rng.uniform(0, 0.01, z.shape)).astype(
                "float32"
            )
            result = (
                self.model(
                    torch.tensor(v, device=self.device),
                    torch.tensor(obs.astype("float32"), device=self.device),
                )
                .cpu()
                .numpy()
            )
        result = np.maximum(result * self.scale + self.offset, 0)
        result[obs] = x[obs]
        return (result, {})

    def fit(self, val_mask, path):
        path.mkdir(parents=True, exist_ok=True)
        c = self.cfg
        d = self.d
        tr = d["x"][: d["a"]]
        va = d["x"][d["a"] : d["b"]]
        vx = va.copy()
        vx[val_mask] = np.nan
        opt = torch.optim.Adam(self.model.parameters(), lr=c["lr"])
        rng = np.random.default_rng(self.seed + 120000)
        dop = torch.optim.Adam(self.discriminator.parameters(), lr=c["lr"])
        clean = np.nan_to_num((tr - self.offset) / self.scale).astype(
            "float32"
        )
        observed = np.isfinite(tr)
        best = np.inf
        best_state = None
        stale = 0
        history = []
        start = time.perf_counter()
        for epoch in range(c["epochs"]):
            self.model.train()
            losses = []
            steps = c["gain_steps_per_epoch"]
            for step in range(steps):
                rows = rng.choice(
                    len(tr), min(c["gain_batch_size"], len(tr)), replace=False
                )
                m = observed[rows] & (
                    rng.random((len(rows), tr.shape[1]))
                    >= c["training_mask_rate"]
                )
                mt = torch.tensor(m.astype("float32"), device=self.device)
                xt = torch.tensor(
                    np.where(
                        m, clean[rows], rng.uniform(0, 0.01, m.shape)
                    ).astype("float32"),
                    device=self.device,
                )
                hint = mt * torch.tensor(
                    (rng.random(m.shape) < c["hint_rate"]).astype("float32"),
                    device=self.device,
                )
                g = self.model(xt, mt)
                completed = mt * xt + (1 - mt) * g
                prob = self.discriminator(completed.detach(), hint)
                dl = -(
                    mt * torch.log(prob + 1e-08)
                    + (1 - mt) * torch.log(1 - prob + 1e-08)
                ).mean()
                dop.zero_grad()
                dl.backward()
                dop.step()
                for p in self.discriminator.parameters():
                    p.requires_grad_(False)
                g = self.model(xt, mt)
                prob = self.discriminator(mt * xt + (1 - mt) * g, hint)
                loss = -((1 - mt) * torch.log(prob + 1e-08)).mean() + c[
                    "gain_alpha"
                ] * ((mt * xt - mt * g) ** 2).mean() / mt.mean().clamp_min(
                    1e-08
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                for p in self.discriminator.parameters():
                    p.requires_grad_(True)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                losses.append(float(loss.detach()))
            pred, _ = self.predict(vx)
            score = float(np.mean(np.abs(pred[val_mask] - va[val_mask])))
            history.append(
                {
                    "epoch": epoch + 1,
                    "training_loss": float(np.mean(losses)),
                    "validation_MAE": score,
                    "seconds": time.perf_counter() - start,
                }
            )
            if score < best:
                best = score
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
            (path / "history.json").write_text(json.dumps(history, indent=2))
            if epoch + 1 == 20:
                torch.save(best_state, path / "best_through_epoch20.pt")
            print(
                self.method,
                "seed",
                self.seed,
                "epoch",
                epoch + 1,
                "val",
                round(score, 4),
                flush=True,
            )
            if stale >= c["patience"]:
                break
        self.model.load_state_dict(best_state)
        torch.save(best_state, path / "weights.pt")
        metadata = {
            "method": self.method,
            "seed": self.seed,
            "config": c,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "validation_MAE": best,
            "training_seconds": time.perf_counter() - start,
            "early_stopping": (
                "validation MAE, min_delta=0, patience=" + str(c["patience"])
            ),
            "offset": self.offset.tolist(),
            "scale": self.scale.tolist(),
            "budget_limited": (
                len(history) == c["epochs"] and stale < c["patience"]
            ),
        }
        (path / "fit.json").write_text(json.dumps(metadata, indent=2))
        return metadata

    def restore(self, path):
        self.model.load_state_dict(
            torch.load(
                path / "weights.pt",
                weights_only=True,
                map_location=self.device,
            )
        )
        return json.loads((path / "fit.json").read_text())
