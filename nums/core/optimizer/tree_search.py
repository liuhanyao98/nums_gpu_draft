# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Union
import numpy as np

from nums.core.optimizer.comp_graph import GraphArray, TreeNode, BinaryOp, ReductionOp, Leaf, UnaryOp


random_state = np.random.RandomState(1337)


class ProgramState(object):

    def __init__(self, arr: GraphArray,
                 max_reduction_pairs=None,
                 force_final_action=True,
                 unique_reduction_pairs=False):
        self.arr: GraphArray = arr
        self.force_final_action = force_final_action
        self.get_action_kwargs = {"max_reduction_pairs": max_reduction_pairs,
                                  "unique_reduction_pairs": unique_reduction_pairs}
        self.tnode_map = {}
        self.init_frontier()

    def num_nodes(self):
        r = 0
        for grid_entry in self.arr.grid.get_entry_iterator():
            root: TreeNode = self.arr.graphs[grid_entry]
            r += root.num_nodes()
        return r

    def init_frontier(self):
        for grid_entry in self.arr.grid.get_entry_iterator():
            self.add_frontier_tree(self.arr.graphs[grid_entry])

    def add_frontier_tree(self, start_node: TreeNode):
        for tnode in start_node.get_frontier():
            self.add_frontier_node(tnode)

    def get_bc_action(self, tnode: TreeNode):
        # This is hacky, but no good way to do it w/ current abstractions.
        if isinstance(tnode, BinaryOp):
            grid_entry = self.get_tnode_grid_entry(tnode)
            node_id = self.arr.cluster_state.get_cluster_entry(grid_entry, self.arr.grid.grid_shape)
            actions = [(tnode.tree_node_id, {"node_id": node_id})]
        elif isinstance(tnode, ReductionOp):
            leaf_ids = tuple(tnode.leafs_dict.keys())[:2]
            grid_entry = self.get_tnode_grid_entry(tnode)
            node_id = self.arr.cluster_state.get_cluster_entry(grid_entry, self.arr.grid.grid_shape)
            actions = [(tnode.tree_node_id, {"node_id": node_id,
                                             "leaf_ids": leaf_ids})]
        elif isinstance(tnode, UnaryOp):
            grid_entry = self.get_tnode_grid_entry(tnode)
            node_id = self.arr.cluster_state.get_cluster_entry(grid_entry, self.arr.grid.grid_shape)
            actions = [(tnode.tree_node_id, {"node_id": node_id})]
        else:
            raise Exception()
        return actions

    def add_frontier_node(self, tnode: TreeNode):
        # This is a frontier node.
        actions = None
        if self.force_final_action and tnode.parent is None:
            if isinstance(tnode, (BinaryOp, UnaryOp)) or (isinstance(tnode, ReductionOp)
                                                          and len(tnode.children_dict) == 2):
                # This is a root frontier binary op or reduction op with 2 children.
                # The next action is the last action,
                # so intercept action to force computation on root node entry.
                actions = self.get_bc_action(tnode)
        if actions is None:
            actions = tnode.get_actions(**self.get_action_kwargs)
        self.tnode_map[tnode.tree_node_id] = (tnode, actions)

    def copy(self):
        return ProgramState(self.arr.copy())

    def commit_action(self, action):
        tnode_id, kwargs = action
        entry = self.tnode_map[tnode_id]
        old_node: TreeNode = entry[0]
        new_node: TreeNode = old_node.execute_on(**kwargs)
        # The frontier needs to be updated, so remove the current node from frontier.
        del self.tnode_map[tnode_id]
        if old_node.parent is None and old_node is not new_node:
            # We operated on a root node, so update the array.
            self.update_root(old_node, new_node)
        if isinstance(new_node, Leaf):
            # If it's a leaf node, its parent may now be a frontier node.
            new_node_parent: TreeNode = new_node.parent
            if new_node_parent is not None and new_node_parent.is_frontier():
                self.add_frontier_node(new_node_parent)
        else:
            # There's still work that needs to be done to compute this node.
            # Add the returned node to the frontier.
            # Either a BinaryOp or ReductionOp.
            if new_node.is_frontier():
                self.add_frontier_node(new_node)
        # That's it. This program state is now updated.
        return self.objective(self.arr.cluster_state.resources)

    def simulate_action(self, action):
        tnode_id, kwargs = action
        entry = self.tnode_map[tnode_id]
        node: TreeNode = entry[0]
        new_resources: np.ndarray = node.simulate_on(**kwargs)
        return self.objective(new_resources)

    def objective(self, resources):
        # Our simple objective.
        return np.sum(resources[1:])

    def get_tnode_grid_entry(self, tnode: TreeNode):
        if tnode.parent is None:
            root: TreeNode = tnode
        else:
            root: TreeNode = tnode.get_root()
        tree_root_grid_entry = None
        for grid_entry in self.arr.grid.get_entry_iterator():
            tree_node: TreeNode = self.arr.graphs[grid_entry]
            if tree_node is root:
                tree_root_grid_entry = grid_entry
                break
        if tree_root_grid_entry is None:
            raise Exception("Bad tree.")
        return tree_root_grid_entry

    def update_root(self, old_root, new_root):
        tree_root_grid_entry = self.get_tnode_grid_entry(old_root)
        self.arr.graphs[tree_root_grid_entry] = new_root

    def get_all_actions(self):
        # This is not deterministic due to hashing of children for reduction nodes.
        actions = []
        for tnode_id in self.tnode_map:
            actions += self.tnode_map[tnode_id][1]
        return actions


class TreeSearch(object):

    def __init__(self,
                 seed: Union[int, np.random.RandomState] = 1337,
                 max_samples_per_step=None,
                 max_reduction_pairs=None,
                 force_final_action=True):
        if isinstance(seed, np.random.RandomState):
            self.rs = seed
        else:
            assert isinstance(seed, (int, np.int))
            self.rs = np.random.RandomState(seed)
        self.max_samples_per_step = max_samples_per_step
        self.max_reduction_pairs = max_reduction_pairs
        self.force_final_action = force_final_action

    def step(self, state: ProgramState):
        raise NotImplementedError()

    def solve(self, arr: GraphArray):
        state: ProgramState = ProgramState(arr,
                                           max_reduction_pairs=self.max_reduction_pairs,
                                           force_final_action=self.force_final_action)
        num_steps = 0
        while True:
            num_steps += 1
            state, cost, is_done = self.step(state)
            if is_done:
                break
        return state.arr


class BlockCyclicTS(TreeSearch):

    def __init__(self,
                 seed: Union[int, np.random.RandomState] = 1337,
                 max_samples_per_step=None,
                 max_reduction_pairs=None,
                 force_final_action=True):
        super().__init__(seed,
                         max_samples_per_step,
                         max_reduction_pairs,
                         force_final_action)

    def step(self, state: ProgramState):
        if len(state.tnode_map) == 0:
            # We're done.
            return state, state.objective(state.arr.cluster_state.resources), True
        action = None
        for tnode_id in state.tnode_map:
            action = state.get_bc_action(state.tnode_map[tnode_id][0])[0]
            break
        curr_cost = state.commit_action(action)
        return state, curr_cost, False


class RandomTS(TreeSearch):
    def __init__(self,
                 seed: Union[int, np.random.RandomState] = 1337,
                 max_samples_per_step=None,
                 max_reduction_pairs=None,
                 force_final_action=True):
        super().__init__(seed,
                         max_samples_per_step,
                         max_reduction_pairs,
                         force_final_action)

    def sample_actions(self, state: ProgramState) -> list:
        if self.max_samples_per_step is None:
            return state.get_all_actions()
        # Subsample a set of frontier nodes to try next.
        tnode_ids = list(state.tnode_map.keys())
        num_tnodes = len(tnode_ids)
        if num_tnodes <= self.max_samples_per_step:
            tnode_id_sample = tnode_ids
        else:
            idx_set = set()
            tnode_id_sample = []
            while len(idx_set) < self.max_samples_per_step:
                i = self.rs.randint(0, num_tnodes)
                if i not in idx_set:
                    idx_set.add(i)
                    tnode_id_sample.append(tnode_ids[i])
        actions = []
        for tnode_id in tnode_id_sample:
            actions += state.tnode_map[tnode_id][1]
        return actions

    def step(self, state: ProgramState):
        # Sampling slows things down because for some reason,
        # the lowest cost computations are the sums, so
        # an algorithm that finds local optima keeps the number of leafs for reductions
        # small by computing them whenever they occur.
        actions = self.sample_actions(state)
        if len(actions) == 0:
            # We're done.
            return state, state.objective(state.arr.cluster_state.resources), True
        min_action = None
        min_cost = np.float64("inf")
        for i in range(len(actions)):
            action = actions[i]
            action_cost = state.simulate_action(action)
            if action_cost < min_cost:
                min_action = action
                min_cost = action_cost
        curr_cost = state.commit_action(min_action)
        return state, curr_cost, False
