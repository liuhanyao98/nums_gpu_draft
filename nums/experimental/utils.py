import os
import gc

def check_block_integrity(arr):
    for grid_entry in arr.grid.get_entry_iterator():
        assert arr.blocks[grid_entry].grid_entry == grid_entry
        assert arr.blocks[grid_entry].rect == arr.grid.get_slice_tuples(grid_entry)
        assert arr.blocks[grid_entry].shape == arr.grid.get_block_shape(grid_entry)


def benchmark_func(func, repeat=2, warmup=1):
    for i in range(warmup):
        gc.collect()
        func()

    costs = []
    costs_opt = []
    costs_init = []
    for i in range(repeat):
        gc.collect()
        cost, cost_opt, cost_init, _ = func()
        costs.append(cost)
        costs_opt.append(cost_opt)
        costs_init.append(cost_init)


    return costs, costs_opt, costs_init


def benchmark_print(func, repeat=2, warmup=1):
    for i in range(warmup):
        func()

    costs = []
    new_W_in_1 = None
    new_W_1_2 = None
    new_W_2_out = None
    new_B_1 = None
    new_B_2 = None
    new_B_out = None
    for i in range(repeat):
        cost, new_W_in_1, new_W_1_2, new_W_2_out, new_B_1, new_B_2, new_B_out = func()
        costs.append(cost)

    print("W_in_1:\n")
    print(new_W_in_1.get())
    print("W_1_2:\n")
    print(new_W_1_2.get())
    print("W_2_out:\n")
    print(new_W_2_out.get())

    print("B_1:\n")
    print(new_B_1.get())
    print("B_2:\n")
    print(new_B_2.get())
    print("B_out:\n")
    print(new_B_out.get())

    return costs

def get_number_of_gpus():
    val = os.popen('nvidia-smi --query-gpu=name --format=csv,noheader | wc -l').read()
    return int(val)


def cupy_used_bytes():
    import cupy as cp

    mempool = cp.get_default_memory_pool()
    return mempool.used_bytes()


if __name__ == "__main__":
    print(f"Number of GPUS: {get_number_of_gpus()}")


