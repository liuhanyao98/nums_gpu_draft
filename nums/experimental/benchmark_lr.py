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


def one_step_fit_opt(app, X, y, theta, num_gpus):
    # Section 1
    cluster_state = ClusterState((num_gpus, 1), app.system)
    one_ga: GraphArray = GraphArray.from_ba(app.one, cluster_state)
    X_ga = GraphArray.from_ba(X, cluster_state)
    y_ga = GraphArray.from_ba(y, cluster_state)
    theta_ga = GraphArray.from_ba(theta, cluster_state)
    initend = time.time()

    # Section 2
    mu_ga: GraphArray = opt.forward(app, X_ga, theta_ga, one_ga)
    grad_ga: GraphArray = opt.grad(app, X_ga, y_ga, mu_ga)
    hess_ga: GraphArray = opt.hessian(app, X_ga, one_ga, mu_ga)
    endtime = time.time()

    grad_ga_ba: BlockArray = opt.compute_graph_array(app, grad_ga)
    hess_ga_ba: BlockArray = opt.compute_graph_array(app, hess_ga)
    theta: BlockArray = opt.update_theta(app, grad_ga_ba, hess_ga_ba, theta)

    theta.touch()

    return initend, endtime

def forward(app, X, theta, one):
    Z = X @ theta
    mu = one / (one + app.exp(-Z))
    return mu


def grad(X, y, mu):
    return X.T @ (mu - y)


def hessian(X, one, mu):
    s = mu * (one - mu)
    t = X.T * s
    return t @ X


def update_theta(app, g, hess, local_theta):
    return local_theta - app.inv(hess) @ g


def one_step_fit(app, X, y, theta):
    one = app.one
    mu = forward(app, X, theta, one)
    grad_ = grad(X, y, mu)
    hess_ = hessian(X, one, mu)
    theta = update_theta(app, grad_, hess_, theta)
    endtime = time.time()
    theta.touch()
    return endtime


def one_step_fit_np(np, X, y):
    theta = np.zeros((X.shape[1],), dtype=X.dtype)
    one = 1
    mu = forward(np, X, theta, one)
    grad_ = grad(X, y, mu)
    hess_ = hessian(X, one, mu)
    theta = update_theta(np, grad_, hess_, theta)


def benchmark_lr(num_gpus, N_list, system_class_list, d=1000, optimizer=True, dtype=np.float32):
    format_string = "%20s,%10s,%10s,%10s,%10s,%10s"
    print(format_string % ("Library", "N", "Cost", "CostOpt", "CostInit", "CV"))
    global app

    for N in N_list:
        N = int(N)

        for system_class in system_class_list:
            # try:
            if True:
                if system_class in ["Cupy", "Numpy"]:
                    name = system_class
                    import cupy as cp

                    arr_lib = cp if system_class == "Cupy" else np
                    arr_lib.inv = arr_lib.linalg.inv
                    app = arr_lib

                    X = arr_lib.zeros((N, d), dtype=dtype)
                    y = arr_lib.ones((N,), dtype=dtype)

                    # Prevent the Singular matrix Error in np.linalg.inv
                    arange = arr_lib.arange(N)
                    X[arange, arange % d] = 1
                    cp.cuda.Device(0).synchronize()

                    # Benchmark one step LR
                    def func():
                        tic = time.time()
                        one_step_fit_np(arr_lib, X, y)
                        cp.cuda.Device(0).synchronize()
                        toc = time.time()
                        return toc - tic, 0, 0, None

                    costs, costs_opt, costs_init = benchmark_func(func)
                    del (X, y, app)
                else:
                    # Init system
                    name = system_class.__name__
                    app = am.instance(num_gpus, optimizer)

                    # Make dataset
                    nps.random.seed(0)
                    X = app.ones((N, d), block_shape=(N // num_gpus, d), dtype=dtype)
                    y = app.ones((N,), block_shape=(N // num_gpus,), dtype=dtype)
                    theta = app.zeros((X.shape[1],), (X.block_shape[1],), dtype=X.dtype)

                    # Benchmark one step LR
                    def func():
                        tic = time.time()
                        if optimizer:
                            toc_init, toc_opt = one_step_fit_opt(app, X, y, theta, num_gpus=num_gpus)
                        else:
                            toc_init = tic
                            toc_opt = one_step_fit(app, X, y, theta)
                        toc = time.time()
                        return toc - tic, toc_opt - tic, toc_init - tic, None

                    costs, costs_opt, costs_init = benchmark_func(func)
                    
                    del (X, y, app)
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

    benchmark_lr(
        num_gpus,
        N_list=[
            # 1,
            # 0.5e6 / 4,
            # 1e6 / 4,
            # 2e6 / 4,
            # 3e6 / 4,
            5e6 / 4,
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

