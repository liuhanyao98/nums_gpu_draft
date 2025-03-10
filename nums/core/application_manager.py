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


import logging
import sys
import time
from nums.core import settings
from nums.core.systems.filesystem import FileSystem
from nums.core.systems import numpy_compute
from nums.core.systems.systems import System, SerialSystem, RaySystem
from nums.core.systems.gpu_systems import CupyParallelSystem, CupyOsActorSystem, CupyNcclActorSystem
from nums.core.systems.schedulers import RayScheduler, TaskScheduler, BlockCyclicScheduler
from nums.core.array.application import ArrayApplication


# pylint: disable=global-statement


_instance: ArrayApplication = None


def is_initialized():
    return _instance is not None


def instance(num_gpus, optimizer):
    # Lazy-initialize to initialize on use instead of initializing on import.
    global _instance
    if _instance is None:
        _instance = create(num_gpus, optimizer)
    return _instance


def create(num_gpus, optimizer):
    configure_logging()

    global _instance

    if _instance is not None:
        raise Exception("create() called more than once.")

    system_name = settings.system_name

    compute_module = {
        "numpy": numpy_compute
    }[settings.compute_name]

    if system_name == "serial":
        system: System = SerialSystem(compute_module=compute_module)
    elif system_name == "ray-task":
        scheduler: RayScheduler = TaskScheduler(compute_module=compute_module,
                                                use_head=settings.use_head)
        system: System = RaySystem(compute_module=compute_module,
                                   scheduler=scheduler)
    elif system_name == "ray-cyclic":
        cluster_shape = settings.cluster_shape
        scheduler: RayScheduler = BlockCyclicScheduler(compute_module=compute_module,
                                                       cluster_shape=cluster_shape,
                                                       use_head=settings.use_head)
        system: System = RaySystem(compute_module=compute_module,
                                   scheduler=scheduler)
    elif system_name == "cupy-parallel":
        system = CupyParallelSystem()
        system.num_gpus = num_gpus
        system.cluster_shape = (num_gpus, 1)
        system.optimizer = optimizer
    elif system_name == "cupy-os-actor":
        system = CupyOsActorSystem(num_gpus)
        system.optimizer = optimizer
    elif system_name == "cupy-nccl-actor":
        system = CupyNcclActorSystem(num_gpus)
        system.optimizer = optimizer
    else:
        raise Exception()
    system.init()
    return ArrayApplication(system=system, filesystem=FileSystem(system))


def destroy():
    global _instance
    if _instance is None:
        return
    # This will shutdown ray if ray was started by NumS.

    system = _instance.system
    del _instance.one_half
    del _instance.two
    del _instance.one
    del _instance.zero
    del _instance
    system.shutdown()
    # _instance.system.shutdown()
    

def configure_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)
