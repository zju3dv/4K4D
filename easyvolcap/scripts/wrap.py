import os
import sys

# fmt: off
sys.path.append('.')
from easyvolcap.utils.console_utils import *
# fmt: on


# Calls torchrun under the hood


def configurable_entrypoint(SEPERATION='--', RUNNER='torchrun', EASYVV='easyvolcap/scripts/main.py',
                            default_runner_args=['--nproc_per_node', 'auto'],
                            extra_easyvv_args=['distributed=True'],
                            ):
    # Prepare for args
    args = sys.argv
    if SEPERATION in args:
        runner_args = args[1:args.index(SEPERATION)]
        easyvv_args = args[args.index(SEPERATION) + 1:]
    else:
        runner_args = default_runner_args  # no extra arguments for torchrun (auto communimation, all available gpus)
        easyvv_args = args[1:]
    easyvv_args += extra_easyvv_args

    # Prepare for invokation
    args = []
    args.append(RUNNER)
    if runner_args: args.append(' '.join(runner_args))
    args.append(EASYVV)
    if easyvv_args: args.append(' '.join(easyvv_args))
    run(' '.join(args))


def dist_entrypoint():
    configurable_entrypoint(RUNNER='torchrun', default_runner_args=['--nproc_per_node', 'auto'], extra_easyvv_args=['distributed=True'])


def ipdb_entrypoint():
    configurable_entrypoint(RUNNER='ipdb3', default_runner_args=['-c', 'continue'], extra_easyvv_args=[])


def prof_entrypoint():
    configurable_entrypoint(RUNNER='python', default_runner_args=[], extra_easyvv_args=['profiler_cfg.enabled=True'])
