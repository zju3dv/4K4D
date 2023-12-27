import os
import sys

# fmt: off
sys.path.append('.')
from easyvolcap.utils.console_utils import *
# fmt: on


def configurable_entrypoint(SEPERATION='--', LAUNCHER='', EASYVOLCAP='evc',
                            default_launcher_args=[],
                            extra_launcher_args=[],
                            default_easyvolcap_args=[],
                            extra_easyvolcap_args=[],
                            ):
    # Prepare for args
    args = sys.argv
    if SEPERATION in args:
        launcher_args = args[1:args.index(SEPERATION)]
        easyvolcap_args = args[args.index(SEPERATION) + 1:]
    else:
        launcher_args = default_launcher_args  # no extra arguments for torchrun (auto communimation, all available gpus)
        easyvolcap_args = args[1:] if len(args[1:]) else default_easyvolcap_args
    launcher_args += extra_launcher_args
    easyvolcap_args += extra_easyvolcap_args

    # Prepare for invokation
    args = []
    args.append(LAUNCHER)
    if launcher_args: args.append(' '.join(launcher_args))
    args.append(EASYVOLCAP)
    if easyvolcap_args: args.append(' '.join(easyvolcap_args))

    # The actual invokation
    run(' '.join(args))


def dist_entrypoint():
    # Distribuated training
    configurable_entrypoint(LAUNCHER='torchrun', EASYVOLCAP='easyvolcap/scripts/main.py', default_launcher_args=['--nproc_per_node', 'auto'], extra_easyvolcap_args=['distributed=True'])


def prof_entrypoint():
    # Profiling
    configurable_entrypoint(extra_easyvolcap_args=['profiler_cfg.enabled=True'])


def gui_entrypoint():
    # Directly run GUI without external requirements
    configurable_entrypoint(EASYVOLCAP='evc -t gui', default_easyvolcap_args=['-c', 'configs/specs/gui.yaml'])
