from easyvolcap.utils.console_utils import *


def serialize(data, path=""):
    if isinstance(data, dict):
        return {k: serialize(v, f"{path}.{k}" if path else k) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize(v, f"{path}[{i}]" if path else f"[{i}]") for i, v in enumerate(data)]
    elif isinstance(data, (str, int, float, bool)):
        return data
    else:
        # message = f"Object of type {magenta(type(data))} at path '{yellow(path)}' is not JSON serializable"
        # log(red(message))
        error = f"Object of type {type(data)} at path '{path}' is not JSON serializable"
        raise TypeError(error)
