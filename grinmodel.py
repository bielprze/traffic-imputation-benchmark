"""
GRIN author architecture and objective, with a split-safe resumable
trainer.

Source: Graph-Machine-Learning-Group/grin @
4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e.
The copied GRINet/BiGRIL implementation is unchanged. See docs/GRIN.md.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Keep imports after required runtime/environment initialization.
import copy  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor/grin"))
from lib.nn.models.grin import GRINet  # noqa: E402


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temp, path)


def atomic_torch(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def masked_mae(prediction, target, observed):
    # Indexing avoids ever multiplying a NaN target by zero.
    return torch.abs(prediction - target)[observed.bool()].mean()


def training_objective(
    imputation, auxiliary, target, original_observed, weight=1.0
):
    """
    GraphFiller: main MAE + four auxiliary MAEs on PRE-whitening mask.
    """
    if auxiliary.shape[0] != 4:
        raise ValueError("Expected four author auxiliary heads")
    loss = masked_mae(imputation, target, original_observed)
    for prediction in auxiliary:
        loss = loss + weight * masked_mae(
            prediction, target, original_observed
        )
    return loss


class ReferenceGRIN:
    def __init__(self, train, adjacency, config, seed):
        # Constructor has no validation/test argument: scaler can only
        # see train.
        seed_all(seed)
        self.cfg, self.seed = copy.deepcopy(config), seed
        self.device = torch.device(config["device"])
        train = np.asarray(train, dtype=np.float32)
        self.offset = float(np.nanmean(train))
        self.scale = float(np.nanstd(train))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Invalid global training scale")
        self.model = GRINet(
            adj=adjacency,
            d_in=1,
            d_hidden=config["hidden"],
            d_ff=config["ff_size"],
            ff_dropout=config["ff_dropout"],
            n_layers=config["n_layers"],
            kernel_size=config["kernel_size"],
            decoder_order=config["decoder_order"],
            global_att=False,
            d_u=0,
            d_emb=config["embedding_size"],
            layer_norm=config["layer_norm"],
            merge="mlp",
            impute_only_holes=True,
        ).to(self.device)

    def tensors(self, x):
        original = np.asarray(x, dtype=np.float32)
        mask = np.isfinite(original)
        clean = np.where(mask, (original - self.offset) / self.scale, 0.0)
        return (
            torch.tensor(clean[..., None], device=self.device),
            torch.tensor(mask[..., None], device=self.device),
        )

    def batches(self, values, masks, starts, batch_size=None):
        size = batch_size or self.cfg["batch_size"]
        offsets = torch.arange(self.cfg["window"], device=self.device)
        for first in range(0, len(starts), size):
            ss = torch.as_tensor(
                starts[first : first + size], device=self.device
            )
            ix = ss[:, None] + offsets
            yield ss, values[ix], masks[ix]

    def validation_score(self, truth, hidden):
        """
        Reference val_mae: masked, window-weighted, unscaled; no
        clipping.
        """
        if (
            hidden.shape != truth.shape
            or np.any(hidden & ~np.isfinite(truth))
            or not hidden.any()
        ):
            raise ValueError("Invalid validation mask")
        x = truth.copy()
        x[hidden] = np.nan
        values, mask = self.tensors(x)
        target = torch.tensor(
            np.nan_to_num(truth).astype("float32")[..., None],
            device=self.device,
        )
        evaluation = torch.tensor(hidden[..., None], device=self.device)
        starts = np.arange(len(x) - self.cfg["window"] + 1)
        if not len(starts):
            raise ValueError("Split shorter than window")
        total = torch.zeros((), device=self.device, dtype=torch.float64)
        count = torch.zeros((), device=self.device, dtype=torch.int64)
        self.model.eval()
        with torch.no_grad():
            for ss, y, m in self.batches(values, mask, starts):
                ix = ss[:, None] + torch.arange(
                    self.cfg["window"], device=self.device
                )
                pred = self.model(y, mask=m) * self.scale + self.offset
                keep = evaluation[ix]
                total += torch.abs(pred - target[ix])[keep].double().sum()
                count += keep.sum()
        return float((total / count).cpu())

    def predict(self, x, index=None):
        """
        Stride-one windows, mean overlap; preserve observations exactly.

        Only windows touching missing cells need evaluation: windows are
        independent.
        Benchmark clipping is after aggregation and is not used for
        selection.
        """
        self.model.eval()
        values, observed = self.tensors(x)
        w = self.cfg["window"]
        if len(x) < w:
            raise ValueError("Split shorter than window")
        missing = ~np.isfinite(x)
        cumulative = np.r_[0, np.cumsum(missing.any(axis=1))]
        starts = np.arange(len(x) - w + 1)
        starts = starts[(cumulative[starts + w] - cumulative[starts]) > 0]
        sums = np.zeros_like(x, dtype=np.float64)
        counts = np.zeros((len(x), 1), dtype=np.int64)
        with torch.no_grad():
            for ss, y, m in self.batches(values, observed, starts):
                predictions = self.model(y, mask=m).cpu().numpy()[..., 0]
                for s, prediction in zip(ss.cpu().numpy(), predictions):
                    sums[s : s + w] += prediction
                    counts[s : s + w] += 1
        if np.any(missing & (counts == 0)):
            raise AssertionError("Uncovered missing position")
        raw = sums / np.maximum(counts, 1) * self.scale + self.offset
        pred = np.maximum(raw, 0.0) if self.cfg["clip_nonnegative"] else raw
        pred[~missing] = x[~missing]
        return pred, {
            "window_stride": 1,
            "aggregation": "mean",
            "evaluated_windows": len(starts),
            "clipped_missing_values": (
                int(np.sum(missing & (raw < 0)))
                if self.cfg["clip_nonnegative"]
                else 0
            ),
        }

    def fit(
        self, train, validation, val_mask, path, on_epoch=None, stop_after=None
    ):
        """
        Resume exactly at saved epoch boundaries, including optimizer
        and RNG.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        c = self.cfg
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=c["lr"], weight_decay=0.0
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=c["epochs"], eta_min=c["eta_min"]
        )
        values, original_mask = self.tensors(train)
        sampler = torch.Generator().manual_seed(self.seed + 120000)
        history, best, stale, best_epoch, best_state = (
            [],
            float("inf"),
            0,
            0,
            None,
        )
        elapsed = 0.0
        checkpoint = path / "resume.pt"
        if checkpoint.exists():
            # Own checkpoint, never an untrusted downloaded pickle.
            state = torch.load(
                checkpoint, map_location=self.device, weights_only=False
            )
            if state["config"] != c or state["seed"] != self.seed:
                raise ValueError("Resume config mismatch")
            self.model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            history, best, stale = (
                state["history"],
                state["best"],
                state["stale"],
            )
            best_epoch, best_state, elapsed = (
                state["best_epoch"],
                state["best_state"],
                state["elapsed"],
            )
            sampler.set_state(state["sampler_rng"].cpu())
            torch.set_rng_state(state["torch_rng"].cpu())
            if self.device.type == "cuda":
                torch.cuda.set_rng_state_all(
                    [s.cpu() for s in state["cuda_rng"]]
                )
        start = time.perf_counter()
        for epoch in range(len(history), c["epochs"]):
            if stale >= c["patience"]:
                break
            self.model.train()
            epoch_start = time.perf_counter()
            losses = []
            rate = optimizer.param_groups[0]["lr"]
            starts = torch.randint(
                len(train) - c["window"] + 1,
                (c["samples_per_epoch"],),
                generator=sampler,
            ).numpy()
            for _, target, original in self.batches(
                values, original_mask, starts
            ):
                visible = torch.bernoulli(
                    original.float() * (1 - c["whiten_prob"])
                ).bool()
                # Author layers replace all unavailable values before
                # spatial messages.
                main, auxiliary = self.model(target, mask=visible)
                loss = training_objective(
                    main, auxiliary, target, original, c["pred_loss_weight"]
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    c["grad_clip"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
                losses.append(float(loss.detach()))
            val_mae = self.validation_score(validation, val_mask)
            if not np.isfinite(val_mae):
                raise RuntimeError("Nonfinite validation MAE")
            if val_mae < best:
                best, best_epoch, stale = val_mae, epoch + 1, 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                atomic_torch(path / "weights.pt", best_state)
            else:
                stale += 1
            scheduler.step()
            history.append(
                {
                    "epoch": epoch + 1,
                    "training_loss": float(np.mean(losses)),
                    "validation_MAE": val_mae,
                    "lr": rate,
                    "optimizer_updates": (epoch + 1) * len(losses),
                    "sampled_windows": (epoch + 1) * c["samples_per_epoch"],
                    "epoch_seconds": time.perf_counter() - epoch_start,
                    "seconds": elapsed + time.perf_counter() - start,
                }
            )
            atomic_torch(
                checkpoint,
                {
                    "config": c,
                    "seed": self.seed,
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                    "best": best,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "stale": stale,
                    "elapsed": elapsed + time.perf_counter() - start,
                    "sampler_rng": sampler.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (
                        torch.cuda.get_rng_state_all()
                        if self.device.type == "cuda"
                        else []
                    ),
                },
            )
            atomic_json(path / "history.json", history)
            print(
                f"GRIN-reference seed={self.seed} epoch={epoch+1}"
                f" val={val_mae:.6f} "
                f"best={best_epoch}"
                f' seconds={history[-1]["epoch_seconds"]:.1f}',
                flush=True,
            )
            if on_epoch:
                on_epoch(history[-1], best_epoch, stale)
            if stop_after is not None and epoch + 1 >= stop_after:
                return None
        if best_state is None:
            raise RuntimeError("No valid checkpoint")
        self.model.load_state_dict(best_state)
        atomic_torch(path / "weights.pt", best_state)
        metadata = {
            "method": "GRIN",
            "variant": "author-reference-v3",
            "seed": self.seed,
            "config": c,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "validation_MAE": best,
            "selection_metric": (
                "unclipped window-weighted validation MAE on artificial hidden"
                " values"
            ),
            "training_seconds": elapsed + time.perf_counter() - start,
            "optimizer_updates": history[-1]["optimizer_updates"],
            "sampled_windows": history[-1]["sampled_windows"],
            "early_stopping": (
                f'validation MAE, min_delta=0, patience={c["patience"]}'
            ),
            "offset": self.offset,
            "scale": self.scale,
            "scaler_source": "train only, global float32",
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "budget_limited": (
                len(history) == c["epochs"] and stale < c["patience"]
            ),
        }
        atomic_json(path / "fit.json", metadata)
        return metadata

    def restore(self, path):
        path = Path(path)
        metadata = json.loads((path / "fit.json").read_text())
        if metadata["config"] != self.cfg or metadata["seed"] != self.seed:
            raise ValueError("Fit configuration mismatch")
        if (
            metadata["offset"] != self.offset
            or metadata["scale"] != self.scale
        ):
            raise ValueError("Train scaler mismatch")
        self.model.load_state_dict(
            torch.load(
                path / "weights.pt",
                map_location=self.device,
                weights_only=True,
            )
        )
        return metadata
