import unittest
import sys
import numpy as np
import pandas as pd
from helper import make_mask, profile_missing, metrics, digest
from baseline_imputation import graph_solve, Classical


class MethodologyTests(unittest.TestCase):
    def test_graph_analytic(self):
        y = np.array([[1.0, np.nan, 3.0]])
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0.0]])
        p, info = graph_solve(y, A, 1, 0, ridge=1e-10)
        self.assertAlmostEqual(p[0, 1], 2, places=6)
        np.testing.assert_array_equal(p[np.isfinite(y)], y[np.isfinite(y)])

    def test_temporal_has_effect(self):
        x = np.array([[0.0, 0.0], [np.nan, 10.0], [0.0, 0.0]])
        a = np.array([[0, 1], [1, 0.0]])
        p, _ = graph_solve(x, a, 1, 0)
        q, _ = graph_solve(x, a, 1, 10)
        self.assertGreater(abs(p[1, 0] - q[1, 0]), 1)

    def test_time_zero_equivalence(self):
        x = np.array([[0.0, 0.0], [np.nan, 10.0], [0.0, 0.0]])
        a = np.array([[0, 1], [1, 0.0]])
        full, _ = graph_solve(x, a, 1, 0)
        one = np.concatenate([graph_solve(row[None], a, 1, 0)[0] for row in x])
        np.testing.assert_allclose(full, one, atol=1e-5)

    def test_mask_eligibility_reproducibility(self):
        x = np.arange(1000, dtype=float).reshape(100, 10)
        x[2:6, 3] = np.nan
        for sc in [
            {"kind": "block", "length": 8},
            {"kind": "mcar", "rate": 0.3},
            {"kind": "synchronous", "length": 8},
        ]:
            a = make_mask(x, sc, 10)
            b = make_mask(x, sc, 10)
            np.testing.assert_array_equal(a, b)
            self.assertFalse((a & ~np.isfinite(x)).any())
            self.assertNotEqual(digest(a), digest(make_mask(x, sc, 11)))

    def test_hidden_truth_does_not_change_prediction(self):
        train = np.arange(80, dtype=float).reshape(40, 2)
        x = train[-10:].copy()
        mask = make_mask(x, {"kind": "block", "length": 3}, 2)
        d = {
            "x": train,
            "a": 30,
            "mu": train[:30].mean(0),
            "sd": train[:30].std(0),
            "med": np.median(train[:30], 0),
            "A": np.array([[0, 1], [1, 0]]),
            "index": pd.date_range("2020-01-01", periods=40, freq="10min"),
        }
        poisoned = x.copy()
        poisoned[mask] = 1e12
        x[mask] = np.nan
        poisoned[mask] = np.nan
        for method in [
            "linear",
            "median",
            "knn",
            "historical",
            "graph_temporal",
        ]:
            model = Classical(d, method, {})
            p, _ = model.predict(x, d["index"][-10:])
            q, _ = model.predict(poisoned, d["index"][-10:])
            np.testing.assert_array_equal(p, q)

    def test_metrics(self):
        m = metrics(np.array([0.0, 2.0]), np.array([1.0, 4.0]))
        self.assertEqual(m["MAE"], 1.5)
        self.assertAlmostEqual(m["RMSE"], np.sqrt(2.5))
        self.assertEqual(m["MRE"], 1)
        with self.assertRaises(ValueError):
            metrics(np.array([1]), np.array([np.nan]))

    def test_legacy_axis_defect(self):
        node_major = np.array([[0], [1], [10], [11], [20], [21]])  # N=3,B=2
        bad = node_major.reshape(2, 1, 3)
        good = node_major.reshape(3, 2).T[:, None, :]
        self.assertFalse(np.array_equal(bad, good))
        np.testing.assert_array_equal(good[:, 0], [[0, 10, 20], [1, 11, 21]])

    def test_linear_endpoint(self):
        x = pd.DataFrame({"x": [0.0, np.nan, np.nan, 9.0]})
        self.assertFalse(
            np.array_equal(x.interpolate().values, x.ffill().values)
        )
        x = pd.DataFrame({"x": [0.0, np.nan, np.nan]})
        np.testing.assert_array_equal(x.interpolate().values, x.ffill().values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
