import argparse
import time

import numpy as np
import ray

from nums import numpy as nps
from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.optimizer.cluster_sim import ClusterState
from nums.core.optimizer.comp_graph import GraphArray
from nums.core.optimizer.tree_search import RandomTS
from nums.core.systems.filesystem import FileSystem
from nums.core.systems.gpu_systems import (
    NumpySerialSystem,
    CupySerialSystem,
    NumpyRaySystem,
    CupyRaySystem,
    TorchCPURaySystem,
    TorchGPURaySystem,
    CupyOsActorSystem,
    CupyNcclActorSystem,
    CupyParallelSystem,
)
from nums.models.glms import LogisticRegression
from nums.core import application_manager as am
from utils import benchmark_func, get_number_of_gpus
import lr_opt as opt

random_seed = 1337


def cupy_used_bytes():
    import cupy as cp

    mempool = cp.get_default_memory_pool()
    return mempool.used_bytes()


# global app


def forward(app, X, W, B):
    Z = opt.collapse_graph_array(app, X @ W + B)
    return Z


def update_weight(app, LR, W, D, X):
    return W - LR * (D @ X).T


def update_bias(app, LR, B, D):
    return B - LR * D.T


def relu(app, X):
    return X * (X > app.zero)


def relu_deriv(app, X):
    return (X > app.zero) * app.one


def sigmoid(app, X):
    return app.one / (app.one + app.exp(-X))


def sigmoid_deriv(app, Z):
    return Z * (app.one - Z)


def np_sigmoid(app, X):
    return 1 / (1 + app.exp(-X))


def np_sigmoid_deriv(app, Z):
    return Z * (1 - Z)


def one_step_fit_np(np, X, y, W_in_1, W_1_2, W_2_out, B_1, B_2, B_out):
    # Initialize bias
    LR = 1
    # B_1 = np.zeros(shape=(X.shape[0], dim_1), dtype=X.dtype)
    # B_2 = np.zeros(shape=(X.shape[0], dim_2), dtype=X.dtype)
    # B_out = np.zeros(shape=(X.shape[0], y.shape[1]), dtype=X.dtype)
    Z_1 = X @ W_in_1 + B_1
    # print(f"S_1.shape {S_1.shape} S_1.block_shape {S_1.block_shape}")
    # Z_1 = relu(app, S_1)
    # print(f"Z_1.shape {Z_1.shape} Z_1.block_shape {Z_1.block_shape}")
    # F_1 = relu_deriv(app, S_1).T
    S_1 = np_sigmoid(app, Z_1)
    F_1 = np_sigmoid_deriv(app, Z_1).T

    Z_2 = S_1 @ W_1_2 + B_2
    # Z_2 = relu(app, S_2)
    # F_2 = relu_deriv(app, S_2).T
    S_2 = np_sigmoid(app, Z_2)
    F_2 = np_sigmoid_deriv(app, Z_2).T
    # print(f"S_2.shape {S_2.shape} S_2.block_shape {S_2.block_shape}")

    # y_predict = relu(app, Z_2 @ W_2_out + B_out)
    Z_out = S_2 @ W_2_out + B_out
    F_out = np_sigmoid_deriv(app, Z_out).T
    y_predict = np_sigmoid(app, Z_out)
    # print("start forward proprogation")
    # --back proprogation--
    D_out = F_out * (y_predict - y).T
    D_2 = F_2 * (W_2_out @ D_out)
    D_1 = F_1 * (W_1_2 @ D_2)

    W_in_1 -= LR * (D_1 @ X).T
    W_1_2 -= LR * (D_2 @ S_1).T
    W_2_out -= LR * (D_out @ S_2).T

    B_1 -= LR * D_1.T
    B_2 -= LR * D_2.T
    B_out -= LR * D_out.T
    endtime = time.time()
    return endtime


def np_feedforward(app, X, W_in_1, W_1_2, W_2_out, B_1, B_2, B_out):
    Z_1 = X @ W_in_1 + B_1
    S_1 = np_sigmoid(app, Z_1)

    Z_2 = S_1 @ W_1_2 + B_2
    S_2 = np_sigmoid(app, Z_2)

    Z_out = S_2 @ W_2_out + B_out
    y_predict = np_sigmoid(app, Z_out)
    endtime = time.time()

    return endtime


def feedforward(app, X, W_in_1, W_1_2, W_2_out, B_1, B_2, B_out):
    print("Z_1 = X @ W_in_1 + B_1")
    Z_1 = X @ W_in_1 + B_1
    print("S_1 = sigmoid(app, Z_1)")
    S_1 = sigmoid(app, Z_1)
    print("Z_2 = S_1 @ W_1_2 + B_2")
    Z_2 = S_1 @ W_1_2 + B_2
    print("S_2 = sigmoid(app, Z_2)")
    S_2 = sigmoid(app, Z_2)
    print("Z_out = S_2 @ W_2_out + B_out")
    Z_out = S_2 @ W_2_out + B_out
    print("y_predict = sigmoid(app, Z_out)")
    y_predict = sigmoid(app, Z_out)
    endtime = time.time()

    y_predict.touch()
    return endtime


def feedforward_opt(app, X, W_in_1, W_1_2, W_2_out, B_1, B_2, B_out, num_gpus):
    # Section 1
    # LR = app.one
    cluster_state = ClusterState((num_gpus, 1), app.system)
    one_ga: GraphArray = GraphArray.from_ba(app.one, cluster_state)
    X_ga = GraphArray.from_ba(X, cluster_state)
    W_in_1_ga = GraphArray.from_ba(W_in_1, cluster_state)
    W_1_2_ga = GraphArray.from_ba(W_1_2, cluster_state)
    W_2_out_ga = GraphArray.from_ba(W_2_out, cluster_state)
    B_1_ga = GraphArray.from_ba(B_1, cluster_state)
    B_2_ga = GraphArray.from_ba(B_2, cluster_state)
    B_out_ga = GraphArray.from_ba(B_out, cluster_state)
    initend = time.time()

    # Section 2
    Z_1_ga: GraphArray = forward(app, X_ga, W_in_1_ga, B_1_ga)
    S_1_ga: GraphArray = opt.sigmoid(app, Z_1_ga, one_ga)

    Z_2_ga: GraphArray = forward(app, S_1_ga, W_1_2_ga, B_2_ga)
    S_2_ga: GraphArray = opt.sigmoid(app, Z_2_ga, one_ga)

    Z_out_ga: GraphArray = forward(app, S_2_ga, W_2_out_ga, B_out_ga)
    y_predict_ga: GraphArray = opt.sigmoid(app, Z_out_ga, one_ga)
    endtime = time.time()

    y_predict_ga_ba: BlockArray = opt.compute_graph_array(app, y_predict_ga)
    y_predict_ga_ba.touch()

    return initend, endtime


def one_step_fit_opt(app, X, y, W_in_1, W_1_2, W_2_out, B_1, B_2, B_out, num_gpus):
    # --forward proprogation--
    # print("start forward proprogation")
    LR = app.one
    cluster_state = ClusterState((num_gpus, 1), app.system)
    one_ga: GraphArray = GraphArray.from_ba(app.one, cluster_state)
    X_ga = GraphArray.from_ba(X, cluster_state)
    # print(f"X_ga block_shape {X_ga.block_shape}")
    y_ga = GraphArray.from_ba(y, cluster_state)
    W_in_1_ga = GraphArray.from_ba(W_in_1, cluster_state)
    W_1_2_ga = GraphArray.from_ba(W_1_2, cluster_state)
    W_2_out_ga = GraphArray.from_ba(W_2_out, cluster_state)
    B_1_ga = GraphArray.from_ba(B_1, cluster_state)
    B_2_ga = GraphArray.from_ba(B_2, cluster_state)
    B_out_ga = GraphArray.from_ba(B_out, cluster_state)
    initend = time.time()

    Z_1_ga: GraphArray = forward(app, X_ga, W_in_1_ga, B_1_ga)
    S_1_ga: GraphArray = opt.sigmoid(app, Z_1_ga, one_ga)
    F_1_ga: GraphArray = opt.sigmoid_deriv(app, Z_1_ga, one_ga)
    # print(f"S_1.shape {S_1.shape} S_1.block_shape {S_1.block_shape}")
    # Z_1_ga: GraphArray = opt.relu(S_1_ga, zero_ga)
    # print(f"Z_1.shape {Z_1.shape} Z_1.block_shape {Z_1.block_shape}")
    # F_1_ga: GraphArray = opt.relu_deriv(S_1_ga, zero_ga, one_ga)

    Z_2_ga: GraphArray = forward(app, S_1_ga, W_1_2_ga, B_2_ga)
    S_2_ga: GraphArray = opt.sigmoid(app, Z_2_ga, one_ga)
    F_2_ga: GraphArray = opt.sigmoid_deriv(app, Z_2_ga, one_ga)
    # Z_2_ga: GraphArray = opt.relu(S_2_ga, zero_ga)
    # print(f"S_2.shape {S_2.shape} S_2.block_shape {S_2.block_shape}")
    # F_2_ga: GraphArray = opt.relu_deriv(S_2_ga, zero_ga, one_ga)

    Z_out_ga: GraphArray = forward(app, S_2_ga, W_2_out_ga, B_out_ga)
    y_predict_ga: GraphArray = opt.sigmoid(app, Z_out_ga, one_ga)
    F_out_ga: GraphArray = opt.sigmoid_deriv(app, Z_out_ga, one_ga)
    # y_predict_ga: GraphArray = opt.relu(S_out_ga, zero_ga)

    # print("start back propogation")
    # --back propogation--
    D_out_ga = opt.collapse_graph_array(app, F_out_ga.T * (y_predict_ga - y_ga).T )
    D_2_ga = opt.collapse_graph_array(app, F_2_ga.T * (W_2_out_ga @ D_out_ga))
    D_1_ga = opt.collapse_graph_array(app, F_1_ga.T * (W_1_2_ga @ D_2_ga))

    endtime = time.time()

    D_out_ga_ba = opt.compute_graph_array(app, D_out_ga)
    D_2_ga_ba = opt.compute_graph_array(app, D_2_ga)
    D_1_ga_ba = opt.compute_graph_array(app, D_1_ga)

    S_1_ga_ba = opt.compute_graph_array(app, S_1_ga)
    S_2_ga_ba = opt.compute_graph_array(app, S_2_ga)

    W_in_1: BlockArray = update_weight(app, LR, W_in_1, D_1_ga_ba, X)
    W_1_2: BlockArray = update_weight(app, LR, W_1_2, D_2_ga_ba, S_1_ga_ba)
    W_2_out: BlockArray = update_weight(app, LR, W_2_out, D_out_ga_ba, S_2_ga_ba)

    B_1: BlockArray = update_bias(app, LR, B_1, D_1_ga_ba)
    B_2: BlockArray = update_bias(app, LR, B_2, D_2_ga_ba)
    B_out: BlockArray = update_bias(app, LR, B_out, D_out_ga_ba)
    # print("Start touching")

    W_in_1.touch()
    W_1_2.touch()
    W_2_out.touch()
    B_1.touch()
    B_2.touch()
    B_out.touch()

    return initend, endtime


def one_step_fit(app, X, y, W_in_1, W_1_2, W_2_out, B_1, B_2, B_out):
    # --forward proprogation--
    # print("start forward proprogation")
    LR = app.one
    Z_1 = X @ W_in_1 + B_1
    # print(f"S_1.shape {S_1.shape} S_1.block_shape {S_1.block_shape}")
    # Z_1 = relu(app, S_1)
    # print(f"Z_1.shape {Z_1.shape} Z_1.block_shape {Z_1.block_shape}")
    # F_1 = relu_deriv(app, S_1).T
    S_1 = sigmoid(app, Z_1)
    # F_1 = sigmoid_deriv(app, Z_1).T

    Z_2 = S_1 @ W_1_2 + B_2
    # Z_2 = relu(app, S_2)
    # F_2 = relu_deriv(app, S_2).T
    S_2 = sigmoid(app, Z_2)
    # F_2 = sigmoid_deriv(app, Z_2).T
    # print(f"S_2.shape {S_2.shape} S_2.block_shape {S_2.block_shape}")

    # y_predict = relu(app, Z_2 @ W_2_out + B_out)
    Z_out = S_2 @ W_2_out + B_out
    y_predict = sigmoid(app, Z_out)
    # print("start forward proprogation")
    # --back proprogation--
    # Wrong? D_out = (y_predict - y).T
    D_out = (y_predict - y) * sigmoid_deriv(app, Z_out)
    D_2 = (D_out @ W_2_out.T) * sigmoid_deriv(app, Z_2)
    # D_2 = F_2 * (W_2_out @ D_out)
    # D_1 = F_1 * (W_1_2 @ D_2)
    D_1 = (D_2 @ W_1_2.T) * sigmoid_deriv(app, Z_1)

    W_in_1 -= LR * (X.T @ D_1)
    W_1_2 -= LR * (S_1.T @ D_2)
    W_2_out -= LR * (S_2.T @ D_out)
    # W_in_1 -= LR * (D_1 @ X).T
    # W_1_2 -= LR * (D_2 @ S_1).T
    # W_2_out -= LR * (D_out @ S_2).T

    # B_1 -= LR * D_1.T
    # B_2 -= LR * D_2.T
    # B_out -= LR * D_out.T
    B_1 -= LR * D_1
    B_2 -= LR * D_2
    B_out -= LR * D_out
    # print("Start touching")
    endtime = time.time()
    W_in_1.touch()
    W_1_2.touch()
    W_2_out.touch()
    B_1.touch()
    B_2.touch()
    B_out.touch()

    return endtime


def np_init_weights(app, X, y):
    dim_1 = 2048  # neurons in the first layer
    dim_2 = 2048  # neurons in the second layer
    
    W_in_1 = app.random.normal(size=(X.shape[1], dim_1))
    W_1_2 = app.random.normal(size=(dim_1, dim_2))
    W_2_out = app.random.normal(size=(dim_2, y.shape[1]))
    # Initialize bias
    B_1 = app.zeros((X.shape[0], dim_1))
    B_2 = app.zeros((X.shape[0], dim_2))
    B_out = app.zeros((X.shape[0], y.shape[1]))
    return W_in_1, W_1_2, W_2_out, B_1, B_2, B_out


def init_weights(app: ArrayApplication, num_gpus, X, y, verbose=False):
    dim_1 = 2048  # neurons in the first layer
    dim_2 = 2048  # neurons in the second layer
    if num_gpus == 2:
        # W_in_1 = app.random.normal(shape=(X.shape[1], dim_1), block_shape=(X.block_shape[1], dim_1), dtype=X.dtype)
        # # print(f"W_in_1.shape {W_in_1.shape} W_in_1.block_shape {W_in_1.block_shape}")
        # W_1_2 = app.random.normal(shape=(dim_1, dim_2), block_shape=(dim_1, dim_2), dtype=X.dtype)
        # # print(f"W_1_2.shape {W_1_2.shape} W_1_2.block_shape {W_1_2.block_shape}")
        # W_2_out = app.random.normal(shape=(dim_2, y.shape[1]), block_shape=(dim_2, y.shape[1]), dtype=X.dtype)
        # # print(f"W_2_out.shape {W_2_out.shape} W_2_out.block_shape {W_2_out.block_shape}")

        # # Initialize bias
        print("init weights")

        # B_1 = app.zeros((X.shape[0], dim_1), (X.block_shape[0], dim_1), dtype=X.dtype)
        # # print(f"B_1.shape {B_1.shape} B_1.block_shape {B_1.block_shape}")
        # B_2 = app.zeros((X.shape[0], dim_2), (X.block_shape[0], dim_2), dtype=X.dtype)
        # # print(f"B_2.shape {B_2.shape} B_2.block_shape {B_2.block_shape}")
        # B_out = app.zeros((X.shape[0], y.shape[1]), (X.block_shape[0], y.shape[1]), dtype=X.dtype)
        # # print(f"B_out.shape {B_out.shape} B_out.block_shape {B_out.block_shape}")

        W_in_1 = app.random.normal(shape=(X.shape[1], dim_1), block_shape=(X.block_shape[1], dim_1 // 2), dtype=X.dtype)
        print(f"W_in_1.shape {W_in_1.shape} W_in_1.block_shape {W_in_1.block_shape}")
        W_1_2 = app.random.normal(shape=(dim_1, dim_2), block_shape=(dim_1 // 2, dim_2 // 2), dtype=X.dtype)
        print(f"W_1_2.shape {W_1_2.shape} W_1_2.block_shape {W_1_2.block_shape}")
        W_2_out = app.random.normal(shape=(dim_2, y.shape[1]), block_shape=(dim_2 // 2, y.block_shape[1]), dtype=X.dtype)
        print(f"W_2_out.shape {W_2_out.shape} W_2_out.block_shape {W_2_out.block_shape}")

        # Initialize bias

        B_1 = app.zeros((X.shape[0], dim_1), (X.block_shape[0], dim_1 // 2), dtype=X.dtype)
        print(f"B_1.shape {B_1.shape} B_1.block_shape {B_1.block_shape}")
        B_2 = app.zeros((X.shape[0], dim_2), (X.block_shape[0], dim_2 // 2), dtype=X.dtype)
        print(f"B_2.shape {B_2.shape} B_2.block_shape {B_2.block_shape}")
        B_out = app.zeros((X.shape[0], y.shape[1]), (X.block_shape[0], y.block_shape[1]), dtype=X.dtype)
        print(f"B_out.shape {B_out.shape} B_out.block_shape {B_out.block_shape}")
    else:
        assert num_gpus is 1

        W_in_1 = app.random.normal(shape=(X.shape[1], dim_1), block_shape=(X.block_shape[1], dim_1), dtype=X.dtype)
        W_1_2 = app.random.normal(shape=(dim_1, dim_2), block_shape=(dim_1, dim_2), dtype=X.dtype)
        W_2_out = app.random.normal(shape=(dim_2, y.shape[1]), block_shape=(dim_2, y.shape[1]), dtype=X.dtype)
        # Initialize bias
        B_1 = app.zeros((X.shape[0], dim_1), (X.block_shape[0], dim_1), dtype=X.dtype)
        B_2 = app.zeros((X.shape[0], dim_2), (X.block_shape[0], dim_2), dtype=X.dtype)
        B_out = app.zeros((X.shape[0], y.shape[1]), (X.block_shape[0], y.shape[1]), dtype=X.dtype)

    if verbose:
        print(f"W_in_1.shape {W_in_1.shape} W_in_1.block_shape {W_in_1.block_shape}")
        print(f"W_1_2.shape {W_1_2.shape} W_1_2.block_shape {W_1_2.block_shape}")
        print(f"W_2_out.shape {W_2_out.shape} W_2_out.block_shape {W_2_out.block_shape}")
        print(f"B_1.shape {B_1.shape} B_1.block_shape {B_1.block_shape}")
        print(f"B_2.shape {B_2.shape} B_2.block_shape {B_2.block_shape}")
        print(f"B_out.shape {B_out.shape} B_out.block_shape {B_out.block_shape}")
    return W_in_1, W_1_2, W_2_out, B_1, B_2, B_out


def np_sample(app, sample_size, feature):
    # print(sample_size)
    X_train = app.random.normal(size=(sample_size, feature))
    # print(X_train.shape)
    y_train = app.ones((sample_size, 1))
    return X_train, y_train


def sample(app: ArrayApplication, sample_size, feature, num_gpus, augment=False):
    if num_gpus == 2:
        # X_train = nps.concatenate([app.random.normal(shape=(sample_size // 2, feature), block_shape=(sample_size // 2, feature), dtype=np.float64),
        #                         app.random.normal(shape=(sample_size // 2, feature), block_shape=(sample_size // 2, feature), dtype=np.float64) + 2.0],
        #                           axis=0)
        # y_train = nps.concatenate([app.zeros(shape=(sample_size // 2,),
        #                                          block_shape=(sample_size // 2,),
        #                                          dtype=nps.int),
        #                            app.ones(shape=(sample_size // 2,),
        #                                      block_shape=(sample_size // 2,),
        #                                      dtype=nps.int)], axis=0)
        print("X_train")
        X_train = app.random.normal(shape=(sample_size, feature), block_shape=(sample_size // 2, feature),
                                    dtype=np.float64)
        print("y_train")
        y_train = app.ones(shape=(sample_size, 1), block_shape=(sample_size // 2, 1), dtype=nps.int)
    else:
        assert num_gpus is 1
        # X_train = app.random.normal(shape=(sample_size, feature), block_shape=(sample_size, feature), dtype=np.float64)
        # y_train = app.ones(shape=(sample_size,), block_shape=(sample_size,),
        #                                          dtype=nps.int)
        X_train = app.random.normal(shape=(sample_size, feature), block_shape=(sample_size, feature), dtype=np.float64)

        y_train = app.ones(shape=(sample_size, 1), block_shape=(sample_size, 1), dtype=nps.int)
    # We augment X with 1s for intercept term.
    if augment:
        X_train = app.concatenate([X_train, app.ones(shape=(X_train.shape[0], 1),
                                                     block_shape=(X_train.block_shape[0], 1),
                                                     dtype=X_train.dtype)],
                                  axis=1,
                                  axis_block_size=X_train.block_shape[1] + 1)
    return X_train, y_train
    # return X_train


def benchmark_mlp(num_gpus, N_list, system_class_list, d=1000, optimizer=True, dtype=np.float32):
    format_string = "%20s,%10s,%10s,%10s,%10s,%10s"
    print(format_string % ("Library", "N", "Cost", "CostOpt", "CostInit", "CV"))
    global app

    for N in N_list:
        N = int(N)
        N_block = N // num_gpus
        d_block = d // 1

        for system_class in system_class_list:
            # try:
            if True:
                if system_class in ["Cupy", "Numpy"]:
                    name = system_class
                    import cupy as cp

                    arr_lib = cp if system_class == "Cupy" else np
                    arr_lib.inv = arr_lib.linalg.inv
                    app = arr_lib

                    X, y = np_sample(np, sample_size=N, feature=1000)
                    W_in_1, W_1_2, W_2_out, B_1, B_2, B_out = np_init_weights(np, X, y)

                    # X = arr_lib.zeros((N, d), dtype=dtype)
                    # y = arr_lib.ones((N,), dtype=dtype)

                    # Prevent the Singular matrix Error in np.linalg.inv
                    # arange = arr_lib.arange(N)
                    # X[arange, arange % d] = 1
                    X = cp.asarray(X)
                    y = cp.asarray(y)
                    # print("initialize weights")
                    W_in_1 = cp.asarray(W_in_1)
                    W_1_2 = cp.asarray(W_1_2)
                    W_2_out = cp.asarray(W_2_out)
                    # print("initialize bias")
                    B_1 = cp.asarray(B_1)
                    B_2 = cp.asarray(B_2)
                    B_out = cp.asarray(B_out)
                    cp.cuda.Device(0).synchronize()
                    # W_in_1 = app.random.normal(shape=(X.shape[1], dim_1), dtype=X.dtype)
                    # W_1_2 = app.random.normal(shape=(dim_1, dim_2), dtype=X.dtype)
                    # W_2_out = app.random.normal(shape=(dim_2, y.shape[1]), dtype=X.dtype)

                    # Initialize bias

                    # B_1 = app.zeros((X.shape[0], dim_1), dtype=X.dtype)
                    # B_2 = app.zeros((X.shape[0], dim_2), dtype=X.dtype)
                    # B_out = app.zeros((X.shape[0], y.shape[1]), dtype=X.dtype)
                    # print("done initialize bias")
                    # cp.cuda.Device(0).synchronize()

                    # Benchmark one step mlp
                    def func():
                        tic = time.time()
                        # toc_end = np_feedforward(app, X, W_in_1, W_1_2, W_2_out, B_1, B_2, B_out)
                        toc_end = one_step_fit_np(arr_lib, X, y, W_in_1, W_1_2, W_2_out, B_1, B_2, B_out)
                        cp.cuda.Device(0).synchronize()
                        toc = time.time()
                        return toc - tic, toc_end - tic, 0, None

                    # func()
                    # exit()

                    costs, costs_opt, costs_init = benchmark_func(func)
                    del (X, y, W_in_1, W_1_2, W_2_out, B_1, B_2, B_out)
                else:
                    # Init system
                    name = system_class.__name__
                    # system = system_class(num_gpus)
                    app = am.instance()
                    app.system.num_gpus = num_gpus
                    app.system.cluster_shape = (num_gpus, 1)

                    app.system.optimizer = optimizer
                    # system.init()
                    # app = ArrayApplication(system=system, filesystem=FileSystem(system))

                    # Make dataset
                    # print("hi there")
                    nps.random.seed(0)
                    # print("a", flush=True)
                    X, y = sample(app, sample_size=N, feature=1000, num_gpus=num_gpus)
                    # print(f"X.shape {X.shape} X.block_shape {X.block_shape}")
                    # print(f"y.shape {y.shape} y.block_shape {y.block_shape}")
                    W_in_1, W_1_2, W_2_out, B_1, B_2, B_out = init_weights(app, num_gpus, X, y)

                    # X = sample(app, sample_size=N, feature=1000, num_gpus=num_gpus)
                    # print("b", flush=True)

                    # X = app.ones((N, d), block_shape=(N_block, d_block), dtype=dtype)
                    # y = app.ones((N,), block_shape=(N_block,), dtype=dtype)

                    # Benchmark one step MLP
                    def func():
                        tic = time.time()
                        if optimizer:
                            toc_init, toc_opt = one_step_fit_opt(app, X, y, W_in_1, W_1_2, W_2_out, B_1, B_2,
                                                                       B_out, num_gpus)
                            # toc_init, toc_opt = feedforward_opt(app, X, W_in_1, W_1_2, W_2_out, B_1, B_2,
                            # B_out, num_gpus)
                        else:
                            toc_opt = feedforward(app, X, W_in_1, W_1_2, W_2_out, B_1, B_2, B_out)
                            toc_init = tic
                            # toc_opt = one_step_fit(app, X, y, W_in_1, W_1_2, W_2_out, B_1, B_2, B_out)

                        toc = time.time()
                        return toc - tic, toc_opt - tic, toc_init - tic, None

                    costs, costs_opt, costs_init = benchmark_func(func)

                    del (X, y, app, W_in_1, W_1_2, W_2_out, B_1, B_2, B_out)
                    # del (X, app)
            # except Exception:
            else:
                costs = [-1]
                costs_opt = [-1]
                costs_init = [-1]

            log_str = format_string % (
                # system_class.__name__,
                name,
                "%d" % N,
                "%.4f" % np.mean(costs),
                "%.4f" % np.mean(costs_opt),
                "%.4f" % np.mean(costs_init),
                "%.2f" % (np.std(costs) / np.mean(costs)),
            )
            print(log_str)
            with open("result_lr.csv", "a") as f:
                f.write(log_str + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument('--optimizer',
                        help='This is a boolean flag.',
                        type=eval,
                        choices=[True, False],
                        default='True')

    args = parser.parse_args()
    num_gpus = args.num_gpus or get_number_of_gpus()
    optimizer = args.optimizer
    # try:
    #     ray.init(address="auto")
    # except ConnectionError:
    #     ray.init()

    benchmark_mlp(
        num_gpus,
        N_list=[
            1000,
            # 0.5e6 / 4,
            # 1e6 / 4,
            # 2e6 / 4,
            # 3e6 / 4,
            # 5e6 / 4,
            # 10e6 / 4,
            # 20e6 / 4,
            # 40e6 / 4,
            # 80e6 / 4,
            # 160e6 / 4,
            # 200e6 / 4,
        ],
        system_class_list=[
            # NumpySerialSystem,
            # CupySerialSystem,
            # NumpyRaySystem,
            # CupyRaySystem,
            # TorchGPURaySystem,
            # CupyOsActorSystem,
            # CupyNcclActorSystem,
            CupyParallelSystem,
            # "Cupy",
            # "Numpy",
        ],
        optimizer=optimizer,
    )


