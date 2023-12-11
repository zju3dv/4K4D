from easyvolcap.utils.console_utils import *


@catch_throw
def my_tests(globals: dict = globals(), prefix: str = 'test'):
    # extract testing functions
    tests = {name: func for name, func in globals.items() if name.startswith(prefix)}
    # run tests
    pbar = tqdm(total=len(tests))
    for name, func in tests.items():
        pbar.desc = name
        pbar.refresh()

        func()
        log(f'{name}: {green("OK")}')

        pbar.update(n=1)
        pbar.refresh()
