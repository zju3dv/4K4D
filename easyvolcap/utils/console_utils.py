# fmt: off
class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"

essential_packages = [
    'pdbr', # will also install rich
    'tqdm',
    'ujson',
    'ruamel.yaml',
]

try:
    for package in essential_packages:
        __import__(package)
except ImportError as e:
    print(f'{Colors.YELLOW}{Colors.BOLD}Missing package: {Colors.RED}{Colors.BOLD}{e}{Colors.YELLOW}{Colors.BOLD}, trying to hot install using pip...{Colors.END}')
    import sys
    import subprocess
    subprocess.call([sys.executable, '-m', 'ensurepip'])
    subprocess.call([sys.executable, '-m', 'pip', 'install', *essential_packages])


# This file should serve as a drop in replacement for log_utils.py
import os
import re
import sys
import time
import pdbr  # looks much better than ipdb...
import rich
import shutil
import argparse
import builtins
import warnings
import readline  # if you need to update stuff from input
import numpy as np
import ujson as json
from bdb import BdbQuit
from pdbr import RichPdb
from ruamel.yaml import YAML
from functools import partial
from collections import deque
from io import StringIO, BytesIO
from typing import List, Dict, Union, Optional, IO, Callable, Any, Iterable, Type, cast
from os.path import join, exists, dirname, split, splitext, abspath, relpath, isdir, isfile, basename

from rich import traceback, pretty
from rich.live import Live
from rich.text import Text
from rich.style import Style
from rich.control import Control
from rich.console import Console
from rich.progress import Progress
from rich.table import Table, Column
from rich.pretty import Pretty, pretty_repr, pprint
from rich.progress import BarColumn, TimeElapsedColumn, TimeRemainingColumn, filesize, ProgressColumn
from tqdm.std import tqdm as std_tqdm
from tqdm.rich import tqdm_rich, FractionColumn, RateColumn
from easyvolcap.utils.base_utils import default_dotdict, dotdict, DoNothing

pdbr_theme = 'ansi_dark'
pdbr.utils.set_traceback(pdbr_theme)
RichPdb._theme = pdbr_theme
# fmt: on


class MyYAML(YAML):
    def dumps(self, obj: Union[dict, dotdict]):
        if isinstance(obj, dotdict): obj = obj.to_dict()
        buf = BytesIO()
        self.dump(obj, buf)  # ?: is the dumping also in utf-8?
        return buf.getvalue().decode(encoding='utf-8', errors='strict')[:-1]  # remove \n


yaml = MyYAML()
yaml.default_flow_style = None

warnings.filterwarnings("ignore")  # ignore disturbing warnings
os.environ["PYTHONBREAKPOINT"] = "easyvolcap.utils.console_utils.set_trace"

slim_width = 140
verbose_width = None
slim_log_time = False
slim_log_path = False
slim_time_format = '%H:%M:%S'
# slim_time_format = ''
verbose_time_format = '%Y-%m-%d %H:%M:%S.%f'
do_nothing_console = Console(file=StringIO(), stderr=StringIO())
console = Console(soft_wrap=True, tab_size=4, log_time_format=slim_time_format, width=slim_width, log_time=slim_log_time, log_path=slim_log_path)  # shared
progress = Progress(console=console, expand=True)  # destroyed
live = Live(console=console, refresh_per_second=10)  # destroyed
traceback.install(console=console, width=slim_width)  # for colorful tracebacks
pretty.install(console=console)

NoneType = type(None)


# NOTE: we use console.log for general purpose logging
# Need to check its reliability and integratability
# Since the string markup might mess things up
# TODO: maybe make the console object confiugrable?


class without_live:
    def __enter__(self):
        stop_live()
        stop_prog()

    def __exit__(self, exc_type, exc_val, exc_tb):
        start_live()
        start_prog()


def stop_live():
    global live
    if live is None: return
    live.stop()
    live = None


def start_live():
    global live
    if live is not None: return
    live = Live(console=console, refresh_per_second=1)
    live.start()


def stop_prog():
    global progress
    if progress is None: return
    progress.stop()
    progress = None


def start_prog():
    global progress
    if progress is not None: return
    progress = Progress(console=console, expand=True)


def stacktrace(extra_lines=0, **kwargs):  # be consise
    # Print colorful stacktrace
    kwargs.update(dict(extra_lines=extra_lines))
    console.print_exception(**kwargs, width=slim_width)  # messy formats


breakpoint_disabled = False  # gives user a ways to manually enable or disable all breakpoints in the project


def disable_breakpoint():
    global breakpoint_disabled
    breakpoint_disabled = True


def enable_breakpoint():
    global breakpoint_disabled
    breakpoint_disabled = False


progress_disabled = False
standard_console = console


def disable_console():
    global console, standard_console, do_nothing_console
    console = do_nothing_console


def enable_console():
    global console, standard_console, do_nothing_console
    console = standard_console


def disable_progress():
    global progress_disabled
    progress_disabled = True


def enable_progress():
    global progress_disabled
    progress_disabled = False


verbose_log = True


def disable_verbose_log():
    global verbose_log
    verbose_log = False
    console.width = slim_width
    console._log_render.show_time = slim_log_time
    console._log_render.show_path = slim_log_path
    console._log_render.time_format = slim_time_format


def enable_verbose_log():
    global verbose_log
    verbose_log = True
    console.width = verbose_width
    console._log_render.show_time = True
    console._log_render.show_path = True
    console._log_render.time_format = verbose_time_format


enable_verbose_log()


def set_trace(*args, **kwargs):
    if breakpoint_disabled: return
    stop_live()
    stop_prog()
    rich_pdb = RichPdb()
    rich_pdb.set_trace(sys._getframe(1))


def post_mortem(*args, **kwargs):
    stop_live()
    stop_prog()
    pdbr.post_mortem()  # break on the last exception's stack for inpection


def line(obj):
    """
    Represent objects in oneline for prettier printing
    """
    s = pretty_repr(obj, indent_size=0)
    s = s.replace('\n', ' ')
    s = re.sub(' {2,}', ' ', s)
    return s


def path(string):  # add path markup
    string = str(string)
    if exists(string):
        return Text(string, style=Style(bold=True, color='blue', link=f'file://{abspath(string)}'))
    else:
        # return Text(string, style=Style(bold=True, color='blue'))
        return blue(string)


def red(string: str) -> str: return f'[red bold]{string}[/]'
def blue(string: str) -> str: return f'[blue bold]{string}[/]'
def cyan(string: str) -> str: return f'[cyan bold]{string}[/]'
def pink(string: str) -> str: return f'[bright_magenta bold]{string}[/]'
def green(string: str) -> str: return f'[green bold]{string}[/]'
def yellow(string: str) -> str: return f'[yellow bold]{string}[/]'
def magenta(string: str) -> str: return f'[magenta bold]{string}[/]'
def color(string: str, color: str): return f'[{color} bold]{string}[/]'


def red_slim(string: str) -> str: return f'[red]{string}[/]'
def blue_slim(string: str) -> str: return f'[blue]{string}[/]'
def cyan_slim(string: str) -> str: return f'[cyan]{string}[/]'
def pink_slim(string: str) -> str: return f'[bright_magenta]{string}[/]'
def green_slim(string: str) -> str: return f'[green]{string}[/]'
def yellow_slim(string: str) -> str: return f'[yellow]{string}[/]'
def magenta_slim(string: str) -> str: return f'[magenta]{string}[/]'
def color_slim(string: str, color: str): return f'[{color}]{string}[/]'


def markup_to_ansi(string: str) -> str:
    """
    Convert rich-style markup to ANSI sequences for command-line formatting.

    Args:
        string: Text with rich-style markup.

    Returns:
        Text formatted via ANSI sequences.
    """
    with console.capture() as out:
        console.print(string, soft_wrap=True)
    return out.get()


def get_log_prefix(back=2,
                   module_color=blue,
                   func_color=green,
                   ):
    frame = sys._getframe(back)  # with another offset
    func = frame.f_code.co_name
    module = frame.f_globals['__name__'] if frame is not None else ''
    return module_color(module) + " -> " + func_color(func) + ":"


def log(*stuff,
        back=1,
        file: Optional[IO[str]] = None,
        no_prefix=False,
        module_color=blue,
        func_color=green,
        console: Optional[Console] = console,
        **kwargs):
    """
    Perform logging using the built in shared logger
    """
    writer = console if file is None else Console(file=file, soft_wrap=True, tab_size=4, log_time_format=verbose_time_format)  # shared
    writer._log_render.time_format = verbose_time_format if verbose_log else slim_time_format
    if no_prefix or not verbose_log: writer.log(*stuff, _stack_offset=2, **kwargs)
    else: writer.log(get_log_prefix(back + 1, module_color, func_color), *stuff, _stack_offset=2, **kwargs)


def run(cmd,
        quite=False,
        dry_run=False,
        skip_failed=False,
        invokation=os.system,  # or subprocess.run
        ):
    """
    Run a shell command and print the command to the console.

    Args:
        cmd (str or list): The command to run. If a list, it will be joined with spaces.
        quite (bool): If True, suppress console output.
        dry_run (bool): If True, print the command but do not execute it.

    Raises:
        RuntimeError: If the command returns a non-zero exit code.

    Returns:
        None
    """
    if isinstance(cmd, list):
        cmd = ' '.join(list(map(str, cmd)))
    func = sys._getframe(1).f_code.co_name
    if not quite:
        cmd_color = 'cyan' if not cmd.startswith('rm') else 'red'
        cmd_color = 'green' if dry_run else cmd_color
        dry_msg = magenta('[dry_run]: ') if dry_run else ''
        log(yellow(func), '->', green(invokation.__name__) + ":", dry_msg + color(cmd, cmd_color), no_prefix=True)
        # print(color(cmd, cmd_color), soft_wrap=False)
    if not dry_run:
        code = invokation(cmd)
    else:
        code = 0
    if code != 0 and not skip_failed:
        log(red(code), "<-", yellow(func) + ":", red(cmd), no_prefix=True)
        # print(red(cmd), soft_wrap=True)
        raise RuntimeError(f'{code} <- {func}: {cmd}')
    else:
        return code  # or output


def read(cmd: str, *args, **kwargs):
    def get_output(cmd: str):
        # return subprocess.run(cmd.split(' '), stdout=subprocess.PIPE, shell=True).stdout
        return os.popen(cmd).read()
    return run(cmd, invokation=get_output, skip_failed=True, *args, **kwargs)


def run_if_not_exists(cmd, outname, *args, **kwargs):
    # whether a file exists, whether a directory has more than 3 elements
    # if (os.path.exists(outname) and os.path.isfile(outname)) or (os.path.isdir(outname) and len(os.listdir(outname)) >= 3):
    if os.path.exists(outname):
        log(yellow('Skipping:'), cyan(cmd))
    else:
        run(cmd, *args, **kwargs)


def catch_throw(func: Callable):
    # This function catches errors and stops the execution for easier inspection
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, BdbQuit): return
            log(red(f'Runtime exception: {e}'))
            stacktrace()
            post_mortem()
    return inner


def print(*stuff,
          sep: str = " ",
          end: str = "\n",
          file: Optional[IO[str]] = None,
          flush: bool = False,
          console: Optional[Console] = console,
          **kwargs,
          ):
    r"""
    Print object(s) supplied via positional arguments.
    This function has an identical signature to the built-in print.
    For more advanced features, see the :class:`~rich.console.Console` class.

    Args:
        sep (str, optional): Separator between printed objects. Defaults to " ".
        end (str, optional): Character to write at end of output. Defaults to "\\n".
        file (IO[str], optional): File to write to, or None for stdout. Defaults to None.
        flush (bool, optional): Has no effect as Rich always flushes output. Defaults to False.

    """
    writer = console if file is None else Console(file=file, soft_wrap=True, tab_size=4, log_time_format=verbose_time_format)  # shared
    writer.print(*stuff, sep=sep, end=end, **kwargs)


# https://github.com/tqdm/tqdm/blob/master/tqdm/rich.py
# this is really nice
# if we want to integrate this into our own system, just import the tqdm from here


class PathColumn(ProgressColumn):
    def __init__(self, **kwargs):
        filename, line_no, locals = console._caller_frame_info(2)
        link_path = None if filename.startswith("<") else os.path.abspath(filename)
        path = filename.rpartition(os.sep)[-1]
        path_text = Text(style='log.path')
        path_text.append(
            path, style=f"link file://{link_path}" if link_path else ""
        )
        if line_no:
            path_text.append(":")
            path_text.append(
                f"{line_no}",
                style=f"link file://{link_path}#{line_no}" if link_path else "",
            )
        self.path_text = path_text
        super().__init__(**kwargs)

    def render(self, task):
        return self.path_text


class RateColumn(ProgressColumn):
    """Renders human readable transfer speed."""

    def __init__(self, unit="", unit_scale=False, unit_divisor=1000, **kwargs):
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__(**kwargs)

    def render(self, task):
        """Show data transfer speed."""
        speed = task.speed
        if speed is None:
            return Text(f"  ?  {self.unit}/s", style="progress.data.speed")
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(
                speed,
                ["", "K", "M", "G", "T", "P", "E", "Z", "Y"],
                self.unit_divisor,
            )
        else:
            unit, suffix = filesize.pick_unit_and_suffix(speed, [""], 1)
        # precision = 3 if unit == 1 else 6
        ratio = speed / unit

        precision = 3 - int(np.log(ratio) / np.log(10))
        precision = max(0, precision)
        return Text(f"{ratio:,.{precision}f} {suffix}{self.unit}/s",
                    style="progress.data.speed")
        # ratio = speed / unit
        # if ratio > 1:
        #     return Text(f"{ratio:,.{precision}f} {suffix}{self.unit}/s",
        #                 style="progress.data.speed")
        # else:
        #     return Text(f"{1/ratio:,.{precision}f} {suffix}s/{self.unit}",
        #                 style="progress.data.speed")


class PrefixColumn(ProgressColumn):
    def __init__(self, content: str = None, **kwargs):
        self.content = content
        super().__init__(**kwargs)

    def render(self, task):
        if self.content is not None:
            return self.content
        else:
            log_prefix = get_log_prefix(back=3)
            return log_prefix


class TimeColumn(ProgressColumn):
    def render(self, task):
        log_time = console.get_datetime()
        log_time_display = Text(log_time.strftime(verbose_time_format if verbose_log else slim_time_format), style='log.time')
        return log_time_display


class tqdm_rich(std_tqdm):
    def __init__(self, *args, back=2, **kwargs):
        # This popping happens before initiating tqdm
        _prog = kwargs.pop('progress', None)

        # Thanks! tqdm!
        super().__init__(*args, **kwargs)
        self.disable = self.disable or progress_disabled
        if self.disable: return

        # Whatever for now
        stop_live()
        start_prog()
        _prog = _prog if _prog is not None else progress

        # Use the predefined progress object
        d = self.format_dict
        self._prog = _prog
        self._prog.columns = \
            ((TimeColumn(), ) if verbose_log or slim_log_time else ()) + \
            ((PrefixColumn(content=get_log_prefix(back=back)),) if verbose_log else ()) + \
            (
                "[progress.description]{task.description}"
                "[progress.percentage]{task.percentage:>4.0f}%",
                BarColumn(),
                FractionColumn(unit_scale=d['unit_scale'], unit_divisor=d['unit_divisor']),
                TimeElapsedColumn(), "<", TimeRemainingColumn(),
                RateColumn(unit=d['unit'], unit_scale=d['unit_scale'], unit_divisor=d['unit_divisor']),
            ) + \
            ((PathColumn(table_column=Column(ratio=1.0, justify='right')),) if verbose_log or slim_log_path else ())
        self._prog.start()
        self._task_id = self._prog.add_task(self.desc or "", **d)

    def close(self):
        if self.disable: return
        self.disable = True  # prevent multiple closures

        self.display(refresh=True)
        if self._prog.finished:
            self._prog.stop()
            for task_id in self._prog.task_ids:
                self._prog.remove_task(task_id)

            # Whatever for now
            stop_prog()

    def clear(self, *_, **__):
        pass

    def display(self, refresh=True, *_, **__):
        if not hasattr(self, '_prog'):
            return
        if self._task_id not in self._prog.task_ids:
            return
        self._prog.update(self._task_id, completed=self.n, description=self.desc, refresh=refresh)

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        if hasattr(self, '_prog'):
            self._prog.reset(self._task_id, total=total)
        super(tqdm_rich, self).reset(total=total)


tqdm = tqdm_rich


def time_function(sync_cuda: bool = True):
    """Decorator: time a function call"""
    def inner(*args, func: Callable = lambda x: x, **kwargs):
        start = time.perf_counter()
        ret = func(*args, **kwargs)
        if sync_cuda:
            import torch  # don't want to place this outside
            torch.cuda.synchronize()
        end = time.perf_counter()
        name = getattr(func, '__name__', repr(func))
        log(name, f"{(end - start) * 1000:8.3f} ms", back=2)
        return ret

    def wrapper(func: Callable):
        return partial(inner, func=func)

    if isinstance(sync_cuda, bool):
        return wrapper
    else:
        func = sync_cuda
        sync_cuda = True
        return partial(inner, func=func)


class Timer:
    def __init__(self,
                 name='base',
                 exp_name='',
                 record_dir: str = 'data/timing',
                 disabled: bool = False,
                 sync_cuda: bool = True,
                 record_to_file: bool = False,
                 ):
        self.sync_cuda = sync_cuda
        self.disabled = disabled
        self.name = name
        self.exp_name = exp_name
        self.start_time = time.perf_counter()  # manually record another start time incase timer is disabled during initialization
        self.start()  # you can always restart multiple times to reuse this timer

        self.record_to_file = record_to_file
        if self.record_to_file:
            self.timing_record = dotdict()

    def __enter__(self):
        self.start()

    def start(self):
        if self.disabled: return self
        if self.sync_cuda:
            import torch  # don't want to place this outside
            try: torch.cuda.synchronize()
            except: pass
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        self.stop()

    def stop(self, print=True, back=2):
        if self.disabled: return 0
        if self.sync_cuda:
            import torch  # don't want to place this outside
            try: torch.cuda.synchronize()
            except: pass
        start = self.start_time
        end = time.perf_counter()
        if print: log(f"{(end - start) * 1000:8.3f} ms", self.name, back=back)  # 3 decimals, 3 digits
        return end - start  # return the difference

    def record(self, event: str = ''):
        if self.disabled: return 0
        self.name = event
        diff = self.stop(print=bool(event), back=3)
        if self.record_to_file and event:
            if event not in self.timing_record:
                self.timing_record[event] = []
            self.timing_record[event].append(diff)

            with open(join(self.record_dir, f'{self.exp_name}.json'), 'w') as f:
                json.dump(self.timing_record, f, indent=4)
        self.start()
        return diff


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


rows = None


def display_table(states: dotdict,
                  styles=default_dotdict(
                      NoneType,
                      {
                          'eta': 'cyan',
                          'epoch': 'cyan',
                          'img_loss': 'magenta',
                          'psnr': 'magenta',
                          'loss': 'magenta',
                          'data': 'blue',
                          'batch': 'blue',
                      }
                  ),
                  maxlen=5,
                  ):

    def create_table(columns: List[str],
                     rows: List[List[str]] = [],
                     styles=default_dotdict(NoneType),
                     ):
        try:
            from easyvolcap.engine import cfg
            title = cfg.exp_name  # MARK: global config & circular imports
        except Exception as e:
            title = None
        table = Table(title=title, show_footer=True, show_header=False, box=None)  # move the row names down at the bottom
        for col in columns:
            table.add_column(footer=Text(col, styles[col]), style=styles[col], justify="center")
        for row in rows:
            table.add_row(*row)
        return table

    keys = list(states.keys())
    values = list(map(str, states.values()))
    width, height = shutil.get_terminal_size(fallback=(120, 50))
    maxlen = max(min(height - 8, maxlen), 1)  # 5 would fill the terminal

    global rows
    if rows is None:
        rows = deque(maxlen=maxlen)
    if rows.maxlen != maxlen:
        rows = deque(list(rows)[-maxlen + 1:], maxlen=maxlen)  # save space for header and footer
    rows.append(values)

    # MARK: check performance hit of these calls
    start_live()
    table = create_table(keys, rows, styles)
    live.update(table)  # disabled autorefresh
    return table


def build_parser(d: dict, parser: argparse.ArgumentParser = None, **kwargs):
    """
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    """
    if 'description' in kwargs:
        kwargs['description'] = markup_to_ansi(green(kwargs['description']))

    if parser is None:
        parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=slim_width), **kwargs)

    help_pattern = f'default = {blue("{}")}'

    for k, v in d.items():
        if isinstance(v, dict):
            if 'default' in v:
                # Use other params as kwargs
                d = v.pop('default')
                t = v.pop('type', type(d))
                h = v.pop('help', markup_to_ansi(help_pattern.format(d)))
                parser.add_argument(f'--{k}', default=d, type=t, help=h, **v)
            else:
                # TODO: Add argparse group here
                pass
        elif isinstance(v, list):
            parser.add_argument(f'--{k}', type=type(v[0]) if len(v) else str, default=v, nargs='+', help=markup_to_ansi(help_pattern.format(v)))
        elif isinstance(v, bool):
            t = 'no_' + k if v else k
            parser.add_argument(f'--{t}', action='store_false' if v else 'store_true', dest=k, help=markup_to_ansi(help_pattern.format(v)))
        else:
            parser.add_argument(f'--{k}', type=type(v), default=v, help=markup_to_ansi(help_pattern.format(v)))

    return parser
