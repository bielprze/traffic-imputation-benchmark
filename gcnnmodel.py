'''
gcnnmodel.py

Graph convolutional model definitions and preprocessing utilities
for Spatio-Temporal Graph Convolutional Networks (S-TGCN)
and Voronoi Neighborhood Weighted ST-GCN (VNWS-TGCN).
'''

import math
import typing

import numpy as np
from scipy.spatial import Delaunay
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Custom timeseries dataset function (imported from local module)
from timeseriesdatasetfromarray import timeseries_dataset_from_array


typing_NDArray = np.ndarray


class GraphInfo:
    """
    Simple container for graph edges and node count.

    Attributes:
        edges: Tuple of (source_indices, target_indices, edge_weights).
        num_nodes: Total number of nodes in the graph.
    """

    def __init__(
        self,
        edges: typing.Tuple[typing.List[int], typing.List[int], typing.List[float]],
        num_nodes: int,
    ) -> None:
        self.edges = edges
        self.num_nodes = num_nodes


class GraphConv(layers.Layer):
    """
    Graph convolution layer.

    Applies graph-based message passing followed by a linear transform and activation.
    """

    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        graph_info: GraphInfo,
        aggregation_type: str = "mean",
        combination_type: str = "concat",
        activation: typing.Optional[str] = None,
        seed: typing.Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type

        initializer = keras.initializers.GlorotUniform(seed=seed)
        self.weight = self.add_weight(
            name="weight",
            shape=(in_feat, out_feat),
            initializer=initializer,
            trainable=True,
            dtype="float32",
        )
        self.activation = layers.Activation(activation)

    def _aggregate(
        self,
        neighbour_representations: tf.Tensor,
    ) -> tf.Tensor:
        """
        Aggregate neighbour messages per node.
        """
        func_map = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }
        agg_fn = func_map.get(self.aggregation_type)
        if not agg_fn:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

        return agg_fn(
            neighbour_representations,
            self.graph_info.edges[0],  # segment IDs (source nodes)
            num_segments=self.graph_info.num_nodes,
        )

    def _compute_nodes_representation(
        self,
        features: tf.Tensor,
    ) -> tf.Tensor:
        """Linear transform of node features."""
        # features shape: (num_nodes, batch, seq_len, in_feat)
        return tf.matmul(features, self.weight)

    def _compute_aggregated_messages(
        self,
        features: tf.Tensor,
    ) -> tf.Tensor:
        """
        Gather neighbour features, weight by edge influence, and aggregate.
        """
        # gather features for target nodes
        nbr_feats = tf.gather(features, self.graph_info.edges[1])
        influence = tf.convert_to_tensor(self.graph_info.edges[2], dtype=nbr_feats.dtype)

        # reshape to multiply by influence: (..., neighbours)
        nbr_feats = tf.transpose(nbr_feats, [1, 3, 2, 0])
        nbr_feats = nbr_feats * influence
        nbr_feats = tf.transpose(nbr_feats, [3, 0, 2, 1])

        aggregated = self._aggregate(nbr_feats)
        return tf.matmul(aggregated, self.weight)

    def _combine(
        self,
        nodes_repr: tf.Tensor,
        agg_messages: tf.Tensor,
    ) -> tf.Tensor:
        """Combine node representation with aggregated messages."""
        if self.combination_type == "concat":
            h = tf.concat([nodes_repr, agg_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_repr + agg_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}")
        return self.activation(h)

    def call(self, features: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Args:
            features: Tensor of shape (num_nodes, batch, seq_len, in_feat)
        Returns:
            Tensor of shape (num_nodes, batch, seq_len, out_feat)
        """
        nodes_repr = self._compute_nodes_representation(features)
        agg_msgs = self._compute_aggregated_messages(features)
        return self._combine(nodes_repr, agg_msgs)


class LSTMGC(layers.Layer):
    """
    Combined GraphConv + LSTM model layer.

    Applies graph convolution, then per-node LSTM and dense forecasting.
    """

    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        lstm_units: int,
        input_seq_len: int,
        output_seq_len: int,
        graph_info: GraphInfo,
        graph_conv_params: typing.Optional[dict] = None,
        seed: typing.Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        params = graph_conv_params or {}
        self.graph_conv = GraphConv(
            in_feat,
            out_feat,
            graph_info,
            seed=seed,
            **params,
        )
        self.lstm = layers.LSTM(lstm_units, activation="relu")
        self.dense = layers.Dense(output_seq_len)
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Args:
            inputs: Tensor of shape (batch, seq_len, nodes, in_feat)
        Returns:
            Tensor of shape (batch, output_seq_len, nodes)
        """
        # transpose to (nodes, batch, seq_len, in_feat)
        x = tf.transpose(inputs, [2, 0, 1, 3])
        gcn_out = self.graph_conv(x)

        # reshape for LSTM: (batch*nodes, seq_len, out_feat)
        shape = tf.shape(gcn_out)
        num_nodes, batch, seq_len, feat = shape[0], shape[1], shape[2], shape[3]
        gcn_reshaped = tf.reshape(gcn_out, (batch * num_nodes, seq_len, feat))

        lstm_out = self.lstm(gcn_reshaped)
        dense_out = self.dense(lstm_out)

        # reshape back to (batch, output_seq_len, nodes)
        out = tf.reshape(dense_out, (batch, self.output_seq_len, num_nodes))
        return out


def compute_adjacency_matrix(
    route_distances: typing_NDArray,
    sigma2: float,
    epsilon: float,
) -> typing_NDArray:
    """
    Compute adjacency via Gaussian kernel on distances.

    Args:
        route_distances: (N, N) array of distances.
        sigma2: kernel width parameter.
        epsilon: threshold for edge existence.
    Returns:
        (N, N) binary adjacency matrix.
    """
    num = route_distances.shape[0]
    dist_scaled = route_distances / 10000.0
    w2 = dist_scaled * dist_scaled
    mask = np.ones((num, num)) - np.eye(num)
    return (np.exp(-w2 / sigma2) >= epsilon).astype(float) * mask


def calculate_route_distance(
    sensors: typing.Dict[str, typing.Tuple[float, float]]
) -> typing.Tuple[typing_NDArray, typing.List[typing.Tuple[float, float]]]:
    """
    Compute pairwise Euclidean distances and return sensor locations list.

    Args:
        sensors: mapping sensor_id → (x, y)
    Returns:
        (dist_matrix, all_points_list)
    """
    keys = list(sensors.keys())
    n = len(keys)
    dist_mat = np.zeros((n, n), dtype=float)
    points = [sensors[k] for k in keys]

    for i in range(n):
        x_i, y_i = points[i]
        for j in range(i + 1, n):
            x_j, y_j = points[j]
            d = math.hypot(x_i - x_j, y_i - y_j)
            dist_mat[i, j] = dist_mat[j, i] = d

    return dist_mat, points


def preprocess(
    data_array: typing_NDArray,
    train_size: float,
    val_size: float,
) -> typing.Tuple[typing_NDArray, typing_NDArray, typing_NDArray, StandardScaler]:
    """
    Split time series array into train/val/test and standardize.

    Returns:
        train_arr, val_arr, test_arr, fitted StandardScaler
    """
    total = data_array.shape[0]
    n_train = int(total * train_size)
    n_val = int(total * val_size)

    train_arr = data_array[:n_train]
    val_arr = data_array[n_train : n_train + n_val]
    test_arr = data_array[n_train + n_val :]

    scaler = StandardScaler()
    train_arr = scaler.fit_transform(train_arr)
    val_arr = scaler.transform(val_arr)
    test_arr = scaler.transform(test_arr)

    return train_arr, val_arr, test_arr, scaler


def create_tf_dataset(
    data_array: typing_NDArray,
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    shuffle: bool = True,
    multi_horizon: bool = True,
    seed: typing.Optional[int] = None,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset of (inputs, targets) pairs for timeseries forecasting.
    """
    inputs = timeseries_dataset_from_array(
        data=np.expand_dims(data_array[:-forecast_horizon], -1),
        targets=None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
        seed=seed,
    )
    offset = (
        input_sequence_length if multi_horizon else input_sequence_length + forecast_horizon - 1
    )
    target_len = forecast_horizon if multi_horizon else 1
    targets = timeseries_dataset_from_array(
        data_array[offset:],
        None,
        sequence_length=target_len,
        shuffle=False,
        batch_size=batch_size,
        seed=seed,
    )
    ds = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=100)
    return ds.prefetch(tf.data.AUTOTUNE).cache()


def _fill_matrix(
    src: int,
    current: int,
    neighbours_map: typing.Dict[int, typing.List[int]],
    matrix: typing_NDArray,
    depth: int,
    max_depth: int,
) -> None:
    """
    Recursive helper to compute shortest path length up to max_depth.
    """
    for nbr in neighbours_map.get(current, []):
        if nbr == src:
            continue
        if matrix[src, nbr] < 0.000001 or matrix[src, nbr] > depth:
            matrix[src, nbr] = matrix[nbr, src] = depth
        if depth < max_depth:
            _fill_matrix(src, nbr, neighbours_map, matrix, depth + 1, max_depth)


def calculate_adjacency_matrix_Delaunay(
    points: typing.List[typing.Tuple[float, float]],
    max_depth: int,
    adj_scaler: int = 0,
) -> typing_NDArray:
    """
    Build adjacency matrix based on Delaunay graph distances up to max_depth,
    then scale weights according to adj_scaler.
    """
    delaunay = Delaunay(points)
    indptr, nbrs = delaunay.vertex_neighbor_vertices
    neighbours_map = {
        i: nbrs[indptr[i] : indptr[i + 1]].tolist() for i in range(len(points))
    }

    n = len(points)
    mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        _fill_matrix(i, i, neighbours_map, mat, 1, max_depth)

    # scale adjacency
    mask = mat > 0.0001
    if adj_scaler == 0:
        mat[mask] = 1 / mat[mask]
    elif adj_scaler == 1:
        mat[mask] = np.exp(1 - mat[mask])
    else:
        mat[mask] = 1.0

    return mat
