"""
Classical imputers and graph-regularized solvers; validation selects
parameters.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.impute import KNNImputer


def graph_solve(x, A, space=1.0, temporal=0.0, ridge=1e-6):
    """
    Min ||M(z-y)||²+space*z'Lz+temporal*||Dz||²+ridge*||z||².
    Ridge anchors unobserved components to train mean in normalized
    units.
    """
    obs = np.isfinite(x)
    y = np.nan_to_num(x)
    T, N = x.shape
    L = sparse.csr_matrix(np.diag(A.sum(1)) - A)

    def mv(v):
        z = v.reshape(T, N)
        o = (obs + ridge) * z + space * (L @ z.T).T
        if temporal:
            diff = z[1:] - z[:-1]
            o[1:] += temporal * diff
            o[:-1] -= temporal * diff
        return o.ravel()

    op = LinearOperator((T * N, T * N), matvec=mv, dtype=float)
    deg = np.full(T, 2.0)
    deg[[0, -1]] = 1
    diag = obs + ridge + space * A.sum(1)[None, :] + temporal * deg[:, None]
    pre = LinearOperator(
        op.shape, matvec=lambda v: v / diag.ravel(), dtype=float
    )
    rhs = (obs * y).ravel()
    count = [0]

    def tick(_):
        count[0] += 1

    z, info = cg(
        op, rhs, M=pre, rtol=1e-7, atol=1e-9, maxiter=2000, callback=tick
    )
    residual = np.linalg.norm(mv(z) - rhs) / max(np.linalg.norm(rhs), 1e-12)
    if info != 0 or not np.isfinite(z).all() or residual > 1e-5:
        raise RuntimeError(
            f"CG convergence failure: {info}, residual={residual}"
        )
    pred = z.reshape(T, N)
    pred[obs] = x[obs]
    return pred, {"iterations": count[0], "relative_residual": float(residual)}


class Classical:
    def __init__(self, d, method, params):
        self.d, self.method, self.params = d, method, params
        tr = d["x"][: d["a"]]
        if method == "knn":
            self.imp = KNNImputer(n_neighbors=params.get("k", 5)).fit(
                (tr - d["mu"]) / d["sd"]
            )
        if method == "knn_unscaled":
            self.imp = KNNImputer(n_neighbors=params.get("k", 5)).fit(tr)
        if method == "historical":
            slots = (
                d["index"][: d["a"]].hour * 6
                + d["index"][: d["a"]].minute // 10
            )
            self.slots = (
                pd.DataFrame(tr)
                .groupby(slots)
                .mean()
                .reindex(range(144))
                .fillna(pd.Series(d["med"]))
                .values
            )

    def predict(self, x, index):
        d = self.d
        m = self.method
        p = self.params
        meta = {}
        frame = pd.DataFrame(x, index=index)
        if m == "median":
            z = np.where(np.isfinite(x), x, d["med"])
        elif m == "linear":
            z = (
                frame.interpolate(method="time")
                .ffill()
                .bfill()
                .fillna(pd.Series(d["med"]))
                .values
            )
        elif m == "ffill_bfill":
            z = frame.ffill().bfill().fillna(pd.Series(d["med"])).values
        elif m == "rolling_mean":
            z = (
                frame.fillna(
                    frame.rolling(p.get("window", 5), min_periods=1).mean()
                )
                .interpolate(method="time")
                .ffill()
                .bfill()
                .fillna(pd.Series(d["med"]))
                .values
            )
        elif m == "historical":
            z = np.where(
                np.isfinite(x),
                x,
                self.slots[index.hour * 6 + index.minute // 10],
            )
        elif m == "knn":
            z = self.imp.transform((x - d["mu"]) / d["sd"]) * d["sd"] + d["mu"]
        elif m == "knn_unscaled":
            z = self.imp.transform(x)
        elif m.startswith("graph"):
            A = d["A"].copy()
            if p.get("graph") == "none":
                A[:] = 0
            if p.get("graph") == "correlation":
                tr = d["x"][: d["a"]]
                C = pd.DataFrame(tr).corr().fillna(0).values
                np.fill_diagonal(C, 0)
                A = np.zeros_like(C)
                for c in range(len(C)):
                    js = np.argsort(C[c])[-5:]
                    A[c, js] = np.maximum(C[c, js], 0)
                A = np.maximum(A, A.T)
            norm = p.get("normalize", True)
            mu = d["mu"] if norm else np.zeros_like(d["mu"])
            sd = d["sd"] if norm else np.ones_like(d["sd"])
            z, meta = graph_solve(
                (x - mu) / sd,
                A,
                p.get("space", 1),
                p.get("temporal", 1) if m == "graph_temporal" else 0,
            )
            z = z * sd + mu
        else:
            raise ValueError(m)
        z = np.maximum(z, 0)
        z[np.isfinite(x)] = x[np.isfinite(x)]
        return z, meta


if __name__ == "__main__":
    import sys
    from run_experiments import main

    methods = [
        "median",
        "linear",
        "ffill_bfill",
        "rolling_mean",
        "historical",
        "knn",
        "knn_unscaled",
        "graph_laplacian",
        "graph_temporal",
    ]
    main(["--stage", "evaluate", "--methods", *methods, *sys.argv[1:]])
