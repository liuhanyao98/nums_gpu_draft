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


def matmul_opt(app, W, D, num_gpus):
    # Section 1
    cluster_state = ClusterState((num_gpus, 1), app.system)
    W_ga: GraphArray = GraphArray.from_ba(W, cluster_state)
    D_ga: GraphArray = GraphArray.from_ba(D, cluster_state)
    initend = time.time()

    # Section 2
    Z_ga: GraphArray = opt.collapse_graph_array(app, W_ga @ D_ga)
    endtime = time.time()

    Z: BlockArray = opt.compute_graph_array(app, Z_ga)
    Z.touch()
    del Z
    return initend, endtime


# def matmul_opt(app, X, num_gpus):
#     # Section 1
#     cluster_state = ClusterState((num_gpus, 1), app.system)
#     X_ga: GraphArray = GraphArray.from_ba(X, cluster_state)
#     # W_ga: GraphArray = GraphArray.from_ba(W, cluster_state)
#     # D_ga: GraphArray = GraphArray.from_ba(D, cluster_state)
#     initend = time.time()

#     # Section 2
#     # Z_ga: GraphArray = opt.collapse_graph_array(app, W_ga @ D_ga)
#     Z_ga: GraphArray = opt.collapse_graph_array(app, X_ga.T @ X_ga)
#     endtime = time.time()

#     Z: BlockArray = opt.compute_graph_array(app, Z_ga)
#     Z.touch()
#     return initend, endtime


def benchmark_bop(num_gpus, N_list, system_class_list, d=400000, optimizer=True, dtype=np.float32):
    format_string = "%20s,%10s,%10s,%10s,%10s,%10s"
    print(format_string % ("Library", "N", "Cost", "CostOpt", "CostInit", "CV"))
    global app

    for N in N_list:
        N = int(N)
        d1 = N
        d2 = d
        for system_class in system_class_list:
            # try:
            if True:
                if system_class in ["Cupy", "Numpy"]:
                    name = system_class
                    import cupy as cp

                    arr_lib = cp if system_class == "Cupy" else np
                    arr_lib.inv = arr_lib.linalg.inv
                    app = arr_lib

                    # X = arr_lib.ones((N, d), dtype=dtype)

                    W = arr_lib.ones(shape=(d1, d2), dtype=dtype)
                    D = arr_lib.ones(shape=(d2, N), dtype=dtype)
                    # Prevent the Singular matrix Error in np.linalg.inv
                    # arange = arr_lib.arange(N)
                    # X[arange, arange % d] = 1
                    cp.cuda.Device(0).synchronize()

                    # Benchmark bop
                    def func():
                        tic = time.time()
                        Z = W @ D
                        # Z = X.T @ X
                        cp.cuda.Device(0).synchronize()
                        toc = time.time()
                        return toc - tic, 0, 0, None

                    costs, costs_opt, costs_init = benchmark_func(func)
                    # del (X, app)
                    del (W, D, app)
                else:
                    # Init system
                    name = system_class.__name__
                    app = am.instance(num_gpus, optimizer)

                    W = app.ones(shape=(d1, d2), block_shape=(d1, d2 // num_gpus), dtype=dtype)
                    # print("W", W)
                    D = app.ones(shape=(d2, N), block_shape=(d2 // num_gpus, N), dtype=dtype)
                    # print("D", D)
                    # X = app.ones((N, d), block_shape=(N // num_gpus, d), dtype=dtype)
                    # Benchmark bop
                    def func():
                        tic = time.time()
                        if optimizer:
                            toc_init, toc_opt = matmul_opt(app, W, D, num_gpus)
                            # toc_init, toc_opt = matmul_opt(app, X, num_gpus)
                        else:
                            Z = (W @ D).touch()
                            # Z = (X.T @ X).touch()

                        toc = time.time()
                        return toc - tic, 0, 0, None

                    costs, costs_opt, costs_init = benchmark_func(func)
                    
                    # del (X, app)
                    del (W, D, app)
            #except Exception:
            else:
                costs = [-1]
                costs_opt = [-1]
                costs_init = [-1]

            log_str = format_string % (
                name,
                "%d" % N,
                "%.4f" % np.mean(costs),
                "%.4f" % np.mean(costs_opt),
                "%.4f" % np.mean(costs_init),
                "%.2f" % (np.std(costs) / np.mean(costs)),
            )
            print(log_str)
            with open("result_bop.csv", "a") as f:
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

    benchmark_bop(
        num_gpus,
        N_list=[
            4000,
            #50000,
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
            "Cupy",
            # "Numpy",
        ],
        optimizer=optimizer,
    )


