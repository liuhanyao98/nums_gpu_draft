import argparse
import time

import numpy as np
import ray

from nums import numpy as nps
from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.array.base import Block
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


def forward(app, X, W):
    Z = opt.collapse_graph_array(app, X @ W)
    return Z


# W_in_1: BlockArray = update_weight(app, LR, W_in_1, D_1_ga_ba, X)
def update_weight(app, LR, W, D, X):
    return W - LR * (D @ X).T


# def update_weight(app, LR, W, D, X):
#     return W - LR * (X.T @ D)

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


def one_step_fit_np(np, X, y, W_in_1, W_1_2, W_2_out):
    # Initialize bias
    LR = 1
    Z_1 = X @ W_in_1
    # print(f"S_1.shape {S_1.shape} S_1.block_shape {S_1.block_shape}")
    # Z_1 = relu(app, S_1)
    # print(f"Z_1.shape {Z_1.shape} Z_1.block_shape {Z_1.block_shape}")
    # F_1 = relu_deriv(app, S_1).T
    S_1 = np_sigmoid(app, Z_1)
    F_1 = np_sigmoid_deriv(app, Z_1).T

    Z_2 = S_1 @ W_1_2
    # Z_2 = relu(app, S_2)
    # F_2 = relu_deriv(app, S_2).T
    S_2 = np_sigmoid(app, Z_2)
    F_2 = np_sigmoid_deriv(app, Z_2).T
    # print(f"S_2.shape {S_2.shape} S_2.block_shape {S_2.block_shape}")

    # y_predict = relu(app, Z_2 @ W_2_out + B_out)
    Z_out = S_2 @ W_2_out
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

    # B_1 -= LR * D_1.T
    # B_2 -= LR * D_2.T
    # B_out -= LR * D_out.T
    endtime = time.time()
    return endtime


def np_feedforward(app, X, W_in_1, W_1_2, W_2_out):
    Z_1 = X @ W_in_1
    S_1 = np_sigmoid(app, Z_1)

    Z_2 = S_1 @ W_1_2
    S_2 = np_sigmoid(app, Z_2)

    Z_out = S_2 @ W_2_out
    y_predict = np_sigmoid(app, Z_out)
    endtime = time.time()

    return endtime


def feedforward_data(app, X, W_in_1, W_1_2, W_2_out):
    # print("Z_1 = X @ W_in_1 ")
    Z_1 = X @ W_in_1
    # print("S_1 = sigmoid(app, Z_1)")
    S_1 = sigmoid(app, Z_1)
    # print("Z_2 = S_1 @ W_1_2")
    Z_2 = S_1 @ W_1_2
    # print("S_2 = sigmoid(app, Z_2)")
    S_2 = sigmoid(app, Z_2)
    # print("Z_out = S_2 @ W_2_out")
    Z_out = S_2 @ W_2_out
    # print("y_predict = sigmoid(app, Z_out)")
    y_predict = sigmoid(app, Z_out)
    endtime = time.time()

    y_predict.touch()
    return endtime


def distribute_weights(W, cluster_state):
    for node_id in cluster_state.get_cluster_node_ids():
        # print(f"node_id{node_id}")
        for grid_entry in W.grid.get_entry_iterator():
            # from nums.core.array.base import Block
            block: Block = W.blocks[grid_entry]
            if node_id not in cluster_state.get_block_node_ids(block.id):
                dst_actor = node_id[0]
                # print(f"dst_actor{dst_actor}")
                app.system.distribute_to(block.oid, dst_actor)  # copy for compute
                cluster_state.commit_copy_block(block.id, node_id)  # copy for optimizer


def distribute_graph_array(G, cluster_state):
    for node_id in cluster_state.get_cluster_node_ids():
        for grid_entry in G.grid.get_entry_iterator():
            block: Block = cluster_state.get_block(G.graphs[grid_entry].block_id)
            if node_id not in cluster_state.get_block_node_ids(block.id):
                dst_actor = node_id[0]
                # print(f"dst_actor{dst_actor}")
                app.system.distribute_to(block.oid, dst_actor)  # copy for compute
                cluster_state.commit_copy_block(block.id, node_id)  # copy for optimizer


def feedforward_opt(app, X, W_in_1, W_1_2, W_2_out, num_gpus):
    # Section 1
    # LR = app.one
    cluster_state = ClusterState((num_gpus, 1), app.system)
    one_ga: GraphArray = GraphArray.from_ba(app.one, cluster_state)
    X_ga = GraphArray.from_ba(X, cluster_state)
    # print(f"X_ga block_shape {X_ga.block_shape}")
    # y_ga = GraphArray.from_ba(y, cluster_state)
    W_in_1_ga = GraphArray.from_ba(W_in_1, cluster_state)
    W_1_2_ga = GraphArray.from_ba(W_1_2, cluster_state)
    W_2_out_ga = GraphArray.from_ba(W_2_out, cluster_state)

    # Distribute Weights
    distribute_weights(app.one, cluster_state)
    distribute_weights(X, cluster_state)
    # distribute_weights(y, cluster_state)
    initend = time.time()

    # Section 2
    # print(f"forward Z_1_ga")
    # print(f"W_in_1_ga block_shape {W_in_1_ga.block_shape}")
    Z_1_ga: GraphArray = forward(app, X_ga, W_in_1_ga)  # --> 0/1
    S_1_ga: GraphArray = opt.sigmoid(app, Z_1_ga, one_ga)  # --> 0/1
    # distribute_weights(S_1_ga, cluster_state)

    # print(f"forward Z_2_ga")
    Z_2_ga: GraphArray = forward(app, S_1_ga, W_1_2_ga)
    S_2_ga: GraphArray = opt.sigmoid(app, Z_2_ga, one_ga)

    # print("forward Z_out_ga")
    Z_out_ga: GraphArray = forward(app, S_2_ga, W_2_out_ga)  # --> 0/1
    # print("forward y_predict_ga")
    y_predict_ga: GraphArray = opt.sigmoid(app, Z_out_ga, one_ga)  # --> 0/1
    endtime = time.time()

    y_predict_ga_ba: BlockArray = opt.compute_graph_array(app, y_predict_ga)
    y_predict_ga_ba.touch()

    return initend, endtime


def one_step_fit_opt(app, X, y, W_in_1, W_1_2, W_2_out, num_gpus, verbose=False):
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

    # Distribute Weights
    distribute_weights(app.one, cluster_state)
    # distribute_weights(X, cluster_state)
    # distribute_weights(y, cluster_state)

    if verbose:
        print("forward Z_1_ga")
    Z_1_ga: GraphArray = forward(app, X_ga, W_in_1_ga)  # --> 0/1
    if verbose:
        print("forward S_1_ga")
    S_1_ga: GraphArray = opt.sigmoid(app, Z_1_ga, one_ga)  # --> 0/1
    # distribute_weights(S_1_ga, cluster_state)
    if verbose:
        print("forward F_1_ga")
    F_1_ga: GraphArray = opt.sigmoid_deriv(app, Z_1_ga, one_ga)  # --> 0/1
    # print(f"S_1.shape {S_1.shape} S_1.block_shape {S_1.block_shape}")
    # Z_1_ga: GraphArray = opt.relu(S_1_ga, zero_ga)
    # print(f"Z_1.shape {Z_1.shape} Z_1.block_shape {Z_1.block_shape}")
    # F_1_ga: GraphArray = opt.relu_deriv(S_1_ga, zero_ga, one_ga)
    if verbose:
        print("forward Z_2_ga")
    Z_2_ga: GraphArray = forward(app, S_1_ga, W_1_2_ga)
    S_2_ga: GraphArray = opt.sigmoid(app, Z_2_ga, one_ga)
    F_2_ga: GraphArray = opt.sigmoid_deriv(app, Z_2_ga, one_ga)
    # Z_2_ga: GraphArray = opt.relu(S_2_ga, zero_ga)
    # print(f"S_2.shape {S_2.shape} S_2.block_shape {S_2.block_shape}")
    # F_2_ga: GraphArray = opt.relu_deriv(S_2_ga, zero_ga, one_ga)
    if verbose:
        print("forward Z_out_ga")
    Z_out_ga: GraphArray = forward(app, S_2_ga, W_2_out_ga)  # --> 0/1
    if verbose:
        print("forward y_predict_ga")
    y_predict_ga: GraphArray = opt.sigmoid(app, Z_out_ga, one_ga)  # --> 0/1
    if verbose:
        print("forward F_out_ga")
    F_out_ga: GraphArray = opt.sigmoid_deriv(app, Z_out_ga, one_ga)  # --> 0/1
    # print(F_out_ga.shape) -> (1000,)
    # y_predict_ga: GraphArray = opt.relu(S_out_ga, zero_ga)
    initend = time.time()
    if verbose:
        print("-----------------------------start back propogation-------------------------------")
        print("-----------------------------start back propogation-------------------------------")
        print("-----------------------------start back propogation-------------------------------")
    # --back propogation--
    if verbose:
        print("collapse D_out_ga")
    D_out_ga = opt.collapse_graph_array(app, F_out_ga.T * (y_predict_ga - y_ga).T)  # --> 0/1
    # D_out_ga = opt.collapse_graph_array(app, (y_predict_ga - y_ga) * F_out_ga)
    if verbose:
        print("collapse D_2_ga")
    # print(f"W_2_out_ga shape {W_2_out_ga.shape}") -> (2048,)
    # print(f"D_out_ga shape {D_out_ga.shape}") -> (1000,)
    # F_2_ga.shape -> (1000, 2048)
    D_2_ga = opt.collapse_graph_array(app, F_2_ga.T * (W_2_out_ga @ D_out_ga))
    # D_2_ga = opt.collapse_graph_array(app, (D_out_ga @ W_2_out_ga.T) * F_2_ga)
    if verbose:
        print("collapse D_1_ga")
    D_1_ga = opt.collapse_graph_array(app, F_1_ga.T * (W_1_2_ga @ D_2_ga))  # --> 0/1
    distribute_graph_array(D_1_ga, cluster_state)
    # print(D_1_ga.shape)
    # D_1_ga = opt.collapse_graph_array(app, (D_2_ga @ W_1_2_ga.T) * F_1_ga)

    # print("-----------------------------start computing weights-------------------------------")
    # print("-----------------------------start computing weights-------------------------------")
    # print("-----------------------------start computing weights-------------------------------")
    if verbose:
        print("collapse_graph_array dW_in_1_ga")
    dW_in_1_ga = opt.collapse_graph_array(app, (D_1_ga @ X_ga).T) # --> now all exeucted on GPU 0 
    if verbose:
        print("collapse_graph_array dW_1_2_ga")
    dW_1_2_ga = opt.collapse_graph_array(app, (D_2_ga @ S_1_ga).T)
    if verbose:
        print("collapse_graph_array dW_2_out_ga")
    dW_2_out_ga = opt.collapse_graph_array(app, (D_out_ga @ S_2_ga).T)

    endtime = time.time()
    dW_in_1_ga_ba: BlockArray = opt.compute_graph_array(app, dW_in_1_ga)
    dW_1_2_ga_ba: BlockArray = opt.compute_graph_array(app, dW_1_2_ga)
    dW_2_out_ga_ba: BlockArray = opt.compute_graph_array(app, dW_2_out_ga)

    # W_in_1_ga = opt.collapse_graph_array(app, W_in_1_ga - one_ga * (D_1_ga @ X_ga).T)
    # print("collapse_graph_array W_1_2_ga")
    # W_1_2_ga = opt.collapse_graph_array(app, W_1_2_ga - one_ga * (D_2_ga @ S_1_ga).T)
    # print("collapse_graph_array W_2_out_ga")
    # W_2_out_ga = opt.collapse_graph_array(app, W_2_out_ga - one_ga * (D_out_ga @ S_2_ga).T)

    # W_in_1: BlockArray = opt.compute_graph_array(app, W_in_1_ga)
    # W_1_2: BlockArray = opt.compute_graph_array(app, W_1_2_ga)
    # W_2_out: BlockArray = opt.compute_graph_array(app, W_2_out_ga)
    if verbose:
        print("update W_in_1")
    W_in_1 -= dW_in_1_ga_ba
    if verbose:
        print("update W_1_2")
    W_1_2 -= dW_1_2_ga_ba
    if verbose:
        print("update W_2_out")
    W_2_out -= dW_2_out_ga_ba

    # D_out_ga_ba = opt.compute_graph_array(app, D_out_ga)
    # D_2_ga_ba = opt.compute_graph_array(app, D_2_ga)
    # D_1_ga_ba = opt.compute_graph_array(app, D_1_ga)

    # S_1_ga_ba = opt.compute_graph_array(app, S_1_ga)
    # S_2_ga_ba = opt.compute_graph_array(app, S_2_ga)
    # W_in_1: BlockArray = update_weight(app, LR, W_in_1, D_1_ga_ba, X)
    # W_1_2: BlockArray = update_weight(app, LR, W_1_2, D_2_ga_ba, S_1_ga_ba)
    
    # W_2_out: BlockArray = update_weight(app, LR, W_2_out, D_out_ga_ba, S_2_ga_ba)
    # W - LR * (D @ X).T
    # print("Start touching")
    W_in_1.touch()
    W_1_2.touch()
    W_2_out.touch()

    return initend, endtime


def one_step_fit(app, X, y, W_in_1, W_1_2, W_2_out):
    # --forward proprogation--
    # print("start forward proprogation")
    LR = app.one

    Z_1 = X @ W_in_1
    # print(f"S_1.shape {S_1.shape} S_1.block_shape {S_1.block_shape}")
    # Z_1 = relu(app, S_1)
    # print(f"Z_1.shape {Z_1.shape} Z_1.block_shape {Z_1.block_shape}")
    # F_1 = relu_deriv(app, S_1).T
    S_1 = sigmoid(app, Z_1)
    F_1 = sigmoid_deriv(app, Z_1).T

    Z_2 = S_1 @ W_1_2
    S_2 = sigmoid(app, Z_2)
    F_2 = sigmoid_deriv(app, Z_2).T
    # print(f"S_2.shape {S_2.shape} S_2.block_shape {S_2.block_shape}")

    # y_predict = relu(app, Z_2 @ W_2_out + B_out)
    Z_out = S_2 @ W_2_out
    F_out = sigmoid_deriv(app, Z_out).T
    y_predict = sigmoid(app, Z_out)
    # print("start forward proprogation")
    # --back proprogation--
    D_out = F_out * (y_predict - y).T
    D_2 = F_2 * (W_2_out @ D_out)
    D_1 = F_1 * (W_1_2 @ D_2)

    W_in_1 -= LR * (D_1 @ X).T
    W_1_2 -= LR * (D_2 @ S_1).T
    W_2_out -= LR * (D_out @ S_2).T

    endtime = time.time()
    # print("Start touching")
    W_in_1.touch()
    W_1_2.touch()
    W_2_out.touch()

    return endtime


def np_init_weights(app, X, y):
    dim_1 = 8192  # neurons in the first layer
    dim_2 = 8192  # neurons in the second layer

    W_in_1 = app.random.normal(size=(X.shape[1], dim_1))
    W_1_2 = app.random.normal(size=(dim_1, dim_2))
    W_2_out = app.random.normal(size=(dim_2, y.shape[1]))
    return W_in_1, W_1_2, W_2_out


def model_init_weights(app: ArrayApplication, num_gpus, X, y, verbose=False):
    dim_1 = 8192  # neurons in the first layer
    dim_2 = 8192  # neurons in the second layer
    if num_gpus == 2:
        W_in_1 = app.random.normal(shape=(X.shape[1], dim_1), block_shape=(X.block_shape[1], dim_1), dtype=X.dtype)
        # print(f"W_in_1.shape {W_in_1.shape} W_in_1.block_shape {W_in_1.block_shape}")
        W_1_2 = app.random.normal(shape=(dim_1, dim_2), block_shape=(dim_1, dim_2 // 2), dtype=X.dtype)
        # print(f"W_1_2.shape {W_1_2.shape} W_1_2.block_shape {W_1_2.block_shape}")
        W_2_out = app.random.normal(shape=(dim_2, y.shape[1]), block_shape=(dim_2 // 2, y.block_shape[1]),
                                    dtype=X.dtype)
        # W_2_out = app.random.normal(shape=(dim_2, 1), block_shape=(dim_2, 1),
        #                             dtype=X.dtype)
        # print(f"W_2_out.shape {W_2_out.shape} W_2_out.block_shape {W_2_out.block_shape}")
    else:
        assert num_gpus is 1

        W_in_1 = app.random.normal(shape=(X.shape[1], dim_1), block_shape=(X.block_shape[1], dim_1), dtype=X.dtype)
        W_1_2 = app.random.normal(shape=(dim_1, dim_2), block_shape=(dim_1, dim_2), dtype=X.dtype)
        W_2_out = app.random.normal(shape=(dim_2, y.shape[1]), block_shape=(dim_2, y.shape[1]), dtype=X.dtype)

    if verbose:
        print(f"W_in_1.shape {W_in_1.shape} W_in_1.block_shape {W_in_1.block_shape}")
        print(f"W_1_2.shape {W_1_2.shape} W_1_2.block_shape {W_1_2.block_shape}")
        print(f"W_2_out.shape {W_2_out.shape} W_2_out.block_shape {W_2_out.block_shape}")
    return W_in_1, W_1_2, W_2_out


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
        # print("X_train")
        X_train = app.random.normal(shape=(sample_size, feature), block_shape=(sample_size, feature // 2),
                                    dtype=np.float64)
        # print("y_train")
        # y_train = app.ones(shape=(sample_size, 1), block_shape=(sample_size // 2, 1), dtype=nps.int)
        y_train = app.ones(shape=(sample_size, 1), block_shape=(sample_size, 1), dtype=nps.int)
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


def benchmark_mlp(num_gpus, N_list, system_class_list, d=8192, optimizer=True, dtype=np.float32):
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

                    X, y = np_sample(np, sample_size=N, feature=d)
                    W_in_1, W_1_2, W_2_out = np_init_weights(np, X, y)

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
                        # toc_end = np_feedforward(app, X, W_in_1, W_1_2, W_2_out)
                        toc_end = one_step_fit_np(arr_lib, X, y, W_in_1, W_1_2, W_2_out)
                        cp.cuda.Device(0).synchronize()
                        toc = time.time()
                        return toc - tic, toc_end - tic, 0, None

                    # func()
                    # exit()

                    costs, costs_opt, costs_init = benchmark_func(func)
                    del (X, y, W_in_1, W_1_2, W_2_out)
                else:
                    # Init system
                    name = system_class.__name__
                    # system = system_class(num_gpus)
                    app = am.instance(num_gpus, optimizer)
                    # system.init()
                    # app = ArrayApplication(system=system, filesystem=FileSystem(system))

                    # Make dataset
                    # print("hi there")
                    nps.random.seed(0)
                    # print("a", flush=True)
                    X, y = sample(app, sample_size=N, feature=d, num_gpus=num_gpus)
                    # print(f"X.shape {X.shape} X.block_shape {X.block_shape}")
                    # print(f"y.shape {y.shape} y.block_shape {y.block_shape}")
                    W_in_1, W_1_2, W_2_out = model_init_weights(app, num_gpus, X, y)

                    # X = sample(app, sample_size=N, feature=1000, num_gpus=num_gpus)
                    # print("b", flush=True)

                    # X = app.ones((N, d), block_shape=(N_block, d_block), dtype=dtype)
                    # y = app.ones((N,), block_shape=(N_block,), dtype=dtype)

                    # Benchmark one step MLP
                    def func():
                        tic = time.time()
                        if optimizer:
                            # print("------------------one_step_fit_opt-----------------------------------")
                            toc_init, toc_opt = one_step_fit_opt(app, X, y, W_in_1, W_1_2, W_2_out, num_gpus, True)
                            # print("feedforward_opt")
                            # toc_init, toc_opt = feedforward_opt(app, X, W_in_1, W_1_2, W_2_out, num_gpus)
                        else:
                            # toc_opt = feedforward_data(app, X, W_in_1, W_1_2, W_2_out)
                            toc_init = tic
                            toc_opt = one_step_fit(app, X, y, W_in_1, W_1_2, W_2_out)

                        toc = time.time()
                        return toc - tic, toc_opt - tic, toc_init - tic, None

                    costs, costs_opt, costs_init = benchmark_func(func)

                    del (X, y, app, W_in_1, W_1_2, W_2_out)
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
            # CupyParallelSystem,
            "Cupy",
            # "Numpy",
        ],
        optimizer=optimizer,
    )


