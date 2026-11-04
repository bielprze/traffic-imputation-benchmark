"""
Sage-Voronoi, faithful GraphConv/ReLU-LSTM with corrected node ordering.

Source: original gcnnmodel.py and train_GNN.py in the supplied
benchmark.
See docs/SAGE.md for deliberate preprocessing/evaluation corrections.
"""

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Keep imports after required runtime/environment initialization.
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402
from scipy.sparse.csgraph import shortest_path  # noqa: E402
from pyproj import Transformer  # noqa: E402
import tensorflow as tf  # noqa: E402

ROOT = Path(__file__).resolve().parent


def geometry(names, path=None, max_depth=5):
    path = Path(path) if path else ROOT / "sensors_location.geojson"
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    lookup = {}
    for f in raw["features"]:
        key = f["properties"]["sensor_id"]
        if key in lookup:
            raise ValueError("Duplicate geographical sensor ID: " + key)
        if f["geometry"]["type"] != "Point":
            raise ValueError("Expected point geometry")
        lookup[key] = f["geometry"]["coordinates"][:2]
    missing = sorted(set(names) - set(lookup))
    if missing:
        raise ValueError("Missing sensor coordinates: " + str(missing))
    lonlat = np.asarray([lookup[k] for k in names], float)
    if len(np.unique(lonlat, axis=0)) != len(names):
        raise ValueError(
            "Duplicate sensor coordinates need explicit resolution"
        )
    # GeoJSON defaults to WGS84 longitude/latitude; Darmstadt is UTM
    # zone 32N.
    if not (
        (lonlat[:, 0] > 6)
        & (lonlat[:, 0] < 12)
        & (lonlat[:, 1] > 47)
        & (lonlat[:, 1] < 55)
    ).all():
        raise ValueError(
            "Unexpected geographic extent: verify CRS before constructing"
            " graph"
        )
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    points = np.column_stack(tr.transform(lonlat[:, 0], lonlat[:, 1]))
    tri = Delaunay(points)
    ptr, nbr = tri.vertex_neighbor_vertices
    n = len(names)
    direct = np.zeros((n, n), float)
    for i in range(n):
        direct[i, nbr[ptr[i] : ptr[i + 1]]] = 1
    hops = shortest_path(direct, directed=False, unweighted=True)
    adj = np.zeros_like(hops)
    active = (hops > 0) & (hops <= max_depth)
    adj[active] = 1 / hops[active]
    meta = {
        "geojson_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sensor_order": list(names),
        "projection": "EPSG:4326 -> EPSG:32632 (always_xy)",
        "max_depth": max_depth,
        "adj_scaler": 0,
        "directed_edges": int(active.sum()),
        "duplicate_coordinates": False,
    }
    return adj, points, meta


class GraphLSTM(tf.keras.layers.Layer):
    def __init__(self, adj, out_feat=10, lstm_units=64, seed=0):
        super().__init__()
        n = len(adj)
        self.n = n
        self.out_feat = out_feat
        # Original unsorted_segment_mean divides weighted sums by edge
        # COUNT.
        # Sparse multiplication is mathematically identical and avoids
        # E*B*W storage.
        count = (adj > 0).sum(1)
        normalized = adj / np.maximum(count[:, None], 1)
        row, col = np.where(normalized != 0)
        self.operator = tf.sparse.reorder(
            tf.SparseTensor(
                np.c_[row, col], normalized[row, col].astype("float32"), [n, n]
            )
        )
        self.weight = self.add_weight(
            name="weight",
            shape=(1, out_feat),
            initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        )
        self.lstm = tf.keras.layers.LSTM(lstm_units, activation="relu")
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = tf.transpose(inputs, [2, 0, 1, 3])
        shape = tf.shape(x)
        batch, window = shape[1], shape[2]
        messages = tf.reshape(
            tf.sparse.sparse_dense_matmul(
                self.operator, tf.reshape(x, [self.n, -1])
            ),
            tf.shape(x),
        )
        graph = tf.concat(
            [tf.matmul(x, self.weight), tf.matmul(messages, self.weight)],
            axis=-1,
        )
        z = self.dense(
            self.lstm(
                tf.reshape(graph, [self.n * batch, window, 2 * self.out_feat])
            )
        )
        # Node-major flattening MUST be undone node-major before
        # transposition.
        return tf.transpose(tf.reshape(z, [self.n, batch, 1]), [1, 2, 0])


class Sage:
    def __init__(self, d, cfg, seed):
        self.d, self.cfg, self.seed = d, cfg, int(seed)
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(
                cfg.get("threads", 2)
            )
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass
        tf.config.experimental.enable_op_determinism()
        self.window = int(cfg.get("window", 12))
        self.batch = int(cfg.get("batch_size", 64))
        self.adj, self.points, self.geo = geometry(
            d["names"], cfg.get("geojson_path"), cfg.get("max_depth", 5)
        )
        layer = GraphLSTM(
            self.adj, cfg.get("out_feat", 10), cfg.get("lstm_units", 64), seed
        )
        inp = tf.keras.Input((self.window, len(d["names"]), 1))
        self.model = tf.keras.Model(inp, layer(inp))
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=cfg.get("learning_rate", 2e-4)
        )
        self._infer = tf.function(
            lambda x: self.model(x, training=False),
            input_signature=[
                tf.TensorSpec(
                    [None, self.window, len(d["names"]), 1], tf.float32
                )
            ],
        )

    def _windows(self, x):
        z = ((x - self.d["mu"]) / self.d["sd"]).astype("float32")
        obs = ~np.isnan(z)
        if np.isinf(z).any():
            raise ValueError("Infinite inputs are invalid, not missing")
        z = np.where(obs, z, 0)
        indices = np.arange(len(z) - self.window)[:, None] + np.arange(
            self.window
        )
        return (
            z[indices, ..., None],
            z[self.window :, None, :],
            obs[self.window :, None, :].astype("float32"),
        )

    def fit(self, val_mask, path):
        """
        Choose checkpoint by causal imputation MAE on fixed masked
        validation.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        validation = self.d["x"][self.d["a"] : self.d["b"]]
        val_mask = np.asarray(val_mask) if val_mask is not None else None
        if (
            val_mask is None
            or val_mask.dtype != bool
            or val_mask.shape != validation.shape
            or not val_mask.any()
        ):
            raise ValueError(
                "Sage requires a nonempty fixed boolean artificial validation"
                " mask"
            )
        if np.isnan(validation[val_mask]).any():
            raise ValueError("Validation mask includes unknown values")
        validation_masked = validation.copy()
        validation_masked[val_mask] = np.nan
        tr = self._windows(self.d["x"][: self.d["a"]])
        va = self._windows(validation)
        if not tr[2].sum() or not va[2].sum():
            raise ValueError("No known train/validation targets")

        @tf.function
        def step(x, y, m):
            with tf.GradientTape() as tape:
                pred = self.model(x, training=True)
                num = tf.reduce_sum(tf.square(pred - y) * m)
                den = tf.reduce_sum(m)
                loss = tf.math.divide_no_nan(num, den)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables)
            )
            return num, den

        best = np.inf
        best_forecast = np.inf
        wait = 0
        history = []
        rng = np.random.default_rng(self.seed)
        best_epoch = 0
        epochs = self.cfg.get("epochs", 200)
        patience = self.cfg.get("patience", 20)
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(tr[0]))
            num = den = 0.0
            for begin in range(0, len(order), self.batch):
                ids = order[begin : begin + self.batch]
                a, b = step(*(v[ids] for v in tr))
                num += float(a)
                den += float(b)
            vn = vd = 0.0
            for begin in range(0, len(va[0]), self.batch):
                x, y, m = (v[begin : begin + self.batch] for v in va)
                p = self._infer(x).numpy()
                vn += float(np.sum((p - y) ** 2 * m))
                vd += float(m.sum())
            forecast_score = vn / vd
            imputed, _ = self.predict(validation_masked)
            score = float(
                np.abs(imputed[val_mask] - validation[val_mask]).mean()
            )
            if not np.isfinite(score) or not np.isfinite(num / den):
                raise RuntimeError("Nonfinite Sage training loss")
            history.append(
                {
                    "epoch": epoch,
                    "train_mse": num / den,
                    "validation_forecast_mse": forecast_score,
                    "validation_masked_mae": score,
                }
            )
            if score < best:
                best = score
                best_forecast = forecast_score
                best_epoch = epoch
                wait = 0
                self.model.save_weights(path / "sage.weights.h5")
            else:
                wait += 1
            (path / "history.json").write_text(json.dumps(history, indent=2))
            print(
                f"Sage seed={self.seed} epoch={epoch} train={num/den:.6g}"
                f" val_masked_MAE={score:.6g}"
                f" val_forecast_MSE={forecast_score:.6g}",
                flush=True,
            )
            if wait >= patience:
                break
        self.model.load_weights(path / "sage.weights.h5")
        meta = {
            "method": "sage_voronoi",
            "seed": self.seed,
            "config": self.cfg,
            "epochs": len(history),
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_validation_forecast_mse": best_forecast,
            "best_validation_MAE": best,
            "stopped_early": len(history) < epochs,
            "hit_epoch_cap": len(history) == epochs,
            "budget_limited": len(history) == epochs,
            "validation_mask_sha256": (
                hashlib.sha256(
                    np.ascontiguousarray(val_mask).tobytes()
                ).hexdigest()
            ),
            "early_stopping": (
                "minimum causal imputation MAE on fixed artificially masked"
                " validation cells; clean forecast MSE diagnostic only"
            ),
            "training_seconds": time.perf_counter() - start,
            "tensorflow": tf.__version__,
            "geometry": self.geo,
        }
        (path / "fit.json").write_text(json.dumps(meta, indent=2))
        np.savez_compressed(
            path / "graph.npz", adj=self.adj, projected_points=self.points
        )
        return meta

    def restore(self, path):
        path = Path(path)
        meta = json.loads((path / "fit.json").read_text())
        if (
            meta["geometry"] != self.geo
            or meta["seed"] != self.seed
            or meta["config"] != self.cfg
        ):
            raise ValueError("Sage checkpoint identity mismatch")
        self.model.load_weights(path / "sage.weights.h5")
        return meta

    def predict(self, x, index=None):
        """
        Causal recursion; fill missing cells only, never peek at future
        input.
        """
        obs = ~np.isnan(x)
        if np.isinf(x).any():
            raise ValueError("Infinite inputs are invalid")
        z = np.where(obs, x, self.d["mu"]).copy()
        fallback = int((~obs[: self.window]).sum())
        calls = 0
        for t in range(self.window, len(z)):
            if obs[t].all():
                continue
            window = (
                (z[t - self.window : t] - self.d["mu"]) / self.d["sd"]
            ).astype("float32")[None, ..., None]
            p = self._infer(window).numpy()[0, 0] * self.d["sd"] + self.d["mu"]
            p = np.maximum(p, 0)
            if not np.isfinite(p).all():
                raise RuntimeError("Nonfinite Sage recursive prediction")
            z[t, ~obs[t]] = p[~obs[t]]
            calls += 1
        return z, {
            "mode": "causal_recursive",
            "initial_window_train_mean_cells": fallback,
            "recursive_rows": calls,
        }

    def forecast(self, x):
        """
        One-step forecast using past observed values only; natural NaN
        -> train mean.

        Leading window predictions are NaN and MUST be excluded from
        forecast scoring.
        This is separate from recursively imputing an artificial gap.
        """
        inputs, _, _ = self._windows(x)
        out = np.full_like(x, np.nan, dtype=float)
        for begin in range(0, len(inputs), self.batch):
            pred = self._infer(inputs[begin : begin + self.batch]).numpy()[
                :, 0
            ]
            out[self.window + begin : self.window + begin + len(pred)] = (
                np.maximum(pred * self.d["sd"] + self.d["mu"], 0)
            )
        return out
