import time

import numpy as np
from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.optimizer.cluster_sim import ClusterState
from nums.core.optimizer.comp_graph import GraphArray
from nums.core.optimizer.tree_search import RandomTS
from nums.core import application_manager as am
import nums.numpy as nps

random_seed = 1337


def compute_graph_array(app: ArrayApplication, ga: GraphArray) -> BlockArray:
    result_ga: GraphArray = RandomTS(
        seed=random_seed,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True).solve(ga)
    result_ga.grid, result_ga.to_blocks()
    return BlockArray(result_ga.grid, app.system, result_ga.to_blocks())


def collapse_graph_array(app: ArrayApplication, ga: GraphArray) -> GraphArray:
    # print("collapse")
    return RandomTS(
        seed=random_seed,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True).solve(ga)


def forward(app: ArrayApplication, X, theta, one):
    # print("X @ theta")
    Z = collapse_graph_array(app, X @ theta)
    # print("one / (one + app.exp(-Z)")
    mu = collapse_graph_array(app, one / (one + app.exp(-Z)))
    return mu


def sigmoid(app, X, one):
    return collapse_graph_array(app, one / (one + app.exp(-X)))


def sigmoid_deriv(app, Z, one):
    return collapse_graph_array(app, Z * (one - Z))
# def relu(app, X, two):
#     return collapse_graph_array(app, (app.abs(X) + X) / two)
    # return collapse_graph_array(app, X * (X > zero))


# def relu_deriv(app, X, zero, one):
#     return collapse_graph_array(app, app.where(X <= 0, zero, one))
    # return collapse_graph_array(app, (X > zero) * one)


def grad(app: ArrayApplication, X, y, mu):
    # print("X.T @ (mu - y)")
    return collapse_graph_array(app, X.T @ (mu - y))


def hessian(app: ArrayApplication, X, one, mu):
    # print("mu * (one - mu)")
    s = collapse_graph_array(app, mu * (one - mu))
    # print("(X.T * s) @ X")
    return collapse_graph_array(app, (X.T * s) @ X)


def update_theta(app: ArrayApplication, g, hess, local_theta):
    # print("local_theta - app.inv(hess) @ g")
    return local_theta - app.inv(hess) @ g


class LogisticRegression(object):

    def __init__(self, app, cluster_shape, fit_intercept):
        self.app = app
        self.cluster_shape = cluster_shape
        self.fit_intercept = fit_intercept
        assert not self.fit_intercept
        self.theta = None

    def init(self, sample: BlockArray):
        self.theta: BlockArray = self.app.zeros((sample.shape[1],), (sample.shape[1],),
                                                dtype=sample.dtype)

    def partial_fit(self, X, y):
        app = self.app
        cluster_state = ClusterState(self.cluster_shape, app.system)
        one_ga: GraphArray = GraphArray.from_ba(app.one, cluster_state)
        Xc = X
        theta = self.theta
        X_ga = GraphArray.from_ba(Xc, cluster_state)
        y_ga = GraphArray.from_ba(y, cluster_state)
        theta_ga = GraphArray.from_ba(theta, cluster_state)
        mu_ga: GraphArray = forward(app, X_ga, theta_ga, one_ga)
        # print("mu scheduled.")
        grad_ga: GraphArray = grad(app, X_ga, y_ga, mu_ga)
        # print("grad scheduled.")
        hess_ga: GraphArray = hessian(app, X_ga, one_ga, mu_ga)
        # print("hess scheduled.")
        grad_ga_ba: BlockArray = compute_graph_array(app, grad_ga)
        hess_ga_ba: BlockArray = compute_graph_array(app, hess_ga)
        self.theta: BlockArray = update_theta(app, grad_ga_ba, hess_ga_ba, theta)

    def predict(self, X: BlockArray) -> BlockArray:
        # Evaluate on training set.
        app = self.app
        cluster_state = ClusterState(self.cluster_shape, app.system)
        one_ga: GraphArray = GraphArray.from_ba(app.one, cluster_state)
        X_ga = GraphArray.from_ba(X, cluster_state)
        theta_ga = GraphArray.from_ba(self.theta, cluster_state)
        return (compute_graph_array(app, forward(app, X_ga, theta_ga, one_ga)) > 0.5).astype(np.int)


def sample(app: ArrayApplication, sample_size):
    # print("hi X_train")
    X_train = nps.concatenate([nps.random.randn(sample_size // 2, 2),
                               nps.random.randn(sample_size // 2, 2) + 2.0], axis=0)
    # print("hi y_train")
    y_train = nps.concatenate([nps.zeros(shape=(sample_size // 2,), dtype=nps.int),
                               nps.ones(shape=(sample_size // 2,), dtype=nps.int)], axis=0)
    # We augment X with 1s for intercept term.
    X_train = app.concatenate([X_train, app.ones(shape=(X_train.shape[0], 1),
                                                 block_shape=(X_train.block_shape[0], 1),
                                                 dtype=X_train.dtype)],
                              axis=1,
                              axis_block_size=X_train.block_shape[1] + 1)
    # print("sample bye")
    return X_train, y_train


def example(max_iters, batch_size):

    app = am.instance()
    model = LogisticRegression(app=app,
                               cluster_shape=(1, 1), fit_intercept=False)
    X, y = sample(app, sample_size=8)
    model.init(X)

    for i in range(max_iters):
        # Take a step.
        X, y = sample(app, batch_size)
        model.partial_fit(X, y)
        print("train accuracy", (nps.sum(y == model.predict(X)) / X.shape[0]).get())


if __name__ == "__main__":
    example(10, 10**3)
