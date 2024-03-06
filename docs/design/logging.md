# Logging System

## Core Features

### Colored & Unified Logging & Printing

Built upon [`rich`](https://github.com/Textualize/rich), we aim to provide a comprehensive and easy-to-access logging system where you don't have to deal with confusing logging levels.
We simply want to print something to screen and the `log` and `print` functions are built just for that.

```python
# Print this message, record where and when it was printed, make the path blue
log(f'Video generated: {blue(output_path)}')
```

### Call External Function with Grace

The `run` and `read` function will cause `os.system` but record your commands and its output to the console.
Note that if peak performance is required, try running the command directly in Python like replacing `os.system('mv xxx yyy')` with `os.rename('xxx', 'yyy')`.
There will be a notable throughput difference.

```python
# Call ffmpeg using the `run` interface
def generate_video(...):
    ...
    run(cmd)
    return output
```

### Loss Printing

For easier logging and monitoring of the training process, we create and maintain a `rich` table for the loss information.
The table will be updated every time you call `display_table` and will be updated on screen according to the frequency you set by replacing the global variable `live`.

```python
# Render table to screen
log_stats = dotdict(psnr=32.0, ssim=1.0, loss=0.0)
display_table(log_stats)  # render dict as a table (live, console, table)
```

### Progress Bar

We also provide a progress bar for you to monitor the progress of your training.
We speicifically modified the progress bar to have similar APIs like `tqdm`, while at the same time print messages with the global `console` object.
Since all printing actions are managed by the same `console`, the logging, loss table printing and progress bar update will not interfere with each other.

```python
for i in tqdm(range(10), back=2, desc='Progress', disable=False):
    log(f'index: {i}')
```

### Exception Traceback & Debugging

The integrated way for debugging a ***EasyVolcap*** application is to use the command-line debugger `pdbr`.
By placing a `breakpoint()` statement at the line you want the interpreter to stop at, you can run the original command anywhere (like on a remote server with only a terminal).

Another useful trick is to wrap your main function with the decorator `catch_throw`, which we've monkey-patched to stop if some unsolved exception was raised in that function. All main entry points of ***EasyVolcap*** are wrapped with this decorator.

```python
@catch_throw
def main():
    ...
```

## Using ***EasyVolcap***'s Logging Elsewhere

We specifically designed the logging system of ***EasyVolcap*** to be light-weight and flexible enough.
To use it elsewhere, like in an external script, simply do:

```python
from easyvolcap.utils.console_utils import *
```

This will create a global `console` variable along with a set of convenience functions.
See some of my latest scripts for examples ([`depth_fusion.py`](../../scripts/fusion/depth_fusion.py), [`volume_fusion`](../../scripts/fusion/volume_fusion.py)).