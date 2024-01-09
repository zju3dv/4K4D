from typing import Callable, List, Dict
from multiprocessing.pool import ThreadPool, Pool
from easyvolcap.utils.console_utils import *


def parallel_execution(*args, action: Callable, num_workers=32, print_progress=False, sequential=False, async_return=False, desc=None, use_process=False, **kwargs):
    """
    Executes a given function in parallel using threads or processes.
    When using threads, the parallelism is achieved during IO blocking (i.e. when loading images from disk or writing something to disk).
    If your task is compute intensive, consider using packages like numpy or torch since they release the GIL during heavy lifting.

    Args:
        *args: Variable length argument list.
        action (Callable): The function to execute in parallel.
        num_workers (int): The number of worker threads or processes to use.
        print_progress (bool): Whether to print a progress bar.
        sequential (bool): Whether to execute the function sequentially instead of in parallel.
        async_return (bool): Whether to return a pool object for asynchronous results.
        desc (str): The description to use for the progress bar.
        use_process (bool): Whether to use processes instead of threads.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        If `async_return` is False, returns a list of the results of executing the function on each input argument.
        If `async_return` is True, returns a pool object for asynchronous results.
    """

    # https://superfastpython.com/threadpool-python/
    # Python threads are well suited for use with IO-bound tasks
    # NOTE: we expect first arg / or kwargs to be distributed

    def get_length(args: List, kwargs: Dict):
        for a in args:
            if isinstance(a, list):
                return len(a)
        for v in kwargs.values():
            if isinstance(v, list):
                return len(v)
        raise NotImplementedError

    def get_action_args(length: int, args: List, kwargs: Dict, i: int):
        action_args = [(arg[i] if isinstance(arg, list) and len(arg) == length else arg) for arg in args]
        # TODO: Support all types of iterable
        action_kwargs = {key: (kwargs[key][i] if isinstance(kwargs[key], list) and len(kwargs[key]) == length else kwargs[key]) for key in kwargs}
        return action_args, action_kwargs

    if not sequential:
        # Create ThreadPool
        if use_process:
            pool = Pool(processes=num_workers)
        else:
            pool = ThreadPool(processes=num_workers)

        # Spawn threads
        results = []
        asyncs = []
        length = get_length(args, kwargs)
        for i in range(length):
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = pool.apply_async(action, action_args, action_kwargs)
            asyncs.append(async_result)

        # Join threads and get return values
        if not async_return:
            for async_result in tqdm(asyncs, back=3, desc=desc, disable=not print_progress):  # log previous frame
                results.append(async_result.get())  # will sync the corresponding thread
            pool.close()
            pool.join()
            return results
        else:
            return pool
    else:
        results = []
        length = get_length(args, kwargs)
        for i in tqdm(range(length), back=3, desc=desc, disable=not print_progress):  # log previous frame
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = action(*action_args, **action_kwargs)
            results.append(async_result)
        return results
