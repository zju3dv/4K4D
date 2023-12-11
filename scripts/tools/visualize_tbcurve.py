# this file loads tensorboard log from data/record, draws stuff using matplotlib
# we'd like to visualize the validation lpips from the log
import os
import json
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from os.path import join
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from tensorboard.backend.event_processing import event_accumulator


def parse_tensorboard(path, scalars):

    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )

    ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"

    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


parser = argparse.ArgumentParser()
parser.add_argument('--logdirs', type=str, nargs='+', default=['data/record/if_nerf'])  # will read everything from this folder
parser.add_argument('--metrics', type=str, nargs='+', default=['val/lpips', 'val/psnr', 'val/ssim'])  # in the form of tensorboard scalars
parser.add_argument('--ymins', type=float, nargs='+', default=[0.022, 26.00, 0.955])  # in the form of tensorboard scalars
parser.add_argument('--ymaxs', type=float, nargs='+', default=[0.042, 32.00, 0.995])  # in the form of tensorboard scalars
parser.add_argument('--xmins', type=float, nargs='+', default=[0.000, 0.000, 0.000])  # in the form of tensorboard scalars
parser.add_argument('--xmaxs', type=float, nargs='+', default=[120.0, 120.0, 120.0])  # in the form of tensorboard scalars
parser.add_argument('--ep_iter_map', type=json.loads, default='{"aninerf_377": 50, "neuralbody_377": 50}')  # in the form of tensorboard scalars
parser.add_argument('--batch_time_key', type=str, default='train/batch')  # in the form of tensorboard scalars
parser.add_argument('--batch_time_correction', type=json.loads, default='{"aninerf_377": 2.0, "neuralbody_377": 3.0}', help='this should include the code modification correction')  # in the form of tensorboard scalars
parser.add_argument('--output', type=str, default='data/record')  # save stuff here
args = parser.parse_args()

paths = []
for logdir in args.logdirs:  # every logdir
    for path in glob(join(logdir, '*')):  # every subfolder of every logdir
        paths.append(path)  # the tbevent files are in these subfolders


# TODO: prepare the figure here, draw the figure in the next loop
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 15
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.size'] = 5.0
plt.rcParams['xtick.minor.size'] = 3.0
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 3.0
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Avenir'
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.monospace'] = 'Cascadia Code'

# xsize, ysize = 8, 5
# plt.figure(figsize=(xsize, ysize))
# plt.title('Title size = ')
# plt.legend()
# plt.xlabel('$x$', labelpad=10)
# plt.ylabel('$\phi$', labelpad=10)
plt.style.use('ggplot')

metric_data = {m: {} for m in args.metrics}  # every metric has its own figure
scalars = args.metrics + [args.batch_time_key]  # for recording times
for path in tqdm(paths):
    name = os.path.basename(path)
    scalars = parse_tensorboard(path, scalars)
    batch_time = scalars[args.batch_time_key].value.mean()  # average of overall batchtime for this run
    # draw user required metrics one by one on the given figure
    for metric in args.metrics:
        # prepare the values and times for plotting
        # metric_time = scalars[metric].step * args.ep_iter_map[name]  # convert the step to time
        metric_time = scalars[metric].step * batch_time * args.ep_iter_map[name] * args.batch_time_correction[name]  # convert the step to time
        metric_value = scalars[metric].value
        metric_data[metric][name] = (metric_time, metric_value)

# ema python implementation
# def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
#     last = scalars[0]  # First value in the plot (first timestep)
#     smoothed = list()
#     for point in scalars:
#         smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
#         smoothed.append(smoothed_val)                        # Save it
#         last = smoothed_val                                  # Anchor the last smoothed value

#     return smoothed


def ewma(x, smooth=0.75):
    '''
    Returns the exponentially weighted moving average of x.

    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1} (smooth = 1 - alpha)

    Returns:
    --------
    ewma: numpy array
          the exponentially weighted moving average
    '''
    # Coerce x to an array
    x = np.array(x)
    n = x.size

    # Create an initial weight matrix of (1-alpha), and a matrix of powers
    # to raise the weights by
    w0 = np.ones(shape=(n, n)) * smooth
    p = np.vstack([np.arange(i, i - n, -1) for i in range(n)])

    # Create the weight matrix
    with np.errstate(over='ignore'):
        w = np.tril(w0**p, 0)

    # Calculate the ewma
    return np.dot(w, x[::np.newaxis]) / w.sum(axis=1)


for i, metric in enumerate(tqdm(metric_data.keys())):
    xsize, ysize = 8, 5
    metric_name = metric.split("/")[-1].upper()
    plt.figure(figsize=(xsize, ysize))
    plt.title(f'Validataion {metric_name}')
    # plt.xlabel('time (second)', labelpad=10)
    plt.xlabel('Training Time (Minutes)', labelpad=10)
    plt.ylabel(f'Novel Views {metric_name}', labelpad=10)

    xdiff = (args.xmaxs[i] - args.xmins[i]) / 12
    plt.xlim(args.xmins[i], args.xmaxs[i])
    plt.xticks(np.arange(args.xmins[i], args.xmaxs[i] + xdiff, xdiff))

    ydiff = (args.ymaxs[i] - args.ymins[i]) / 10
    plt.ylim(args.ymins[i], args.ymaxs[i])
    plt.yticks(np.arange(args.ymins[i], args.ymaxs[i] + ydiff, ydiff))  # 120 minutesS

    for name, (metric_time, metric_value) in metric_data[metric].items():
        metric_time = metric_time / 60  # seconds -> minutes

        transparent = plt.plot(metric_time, metric_value, alpha=0.25)  # transparent original data

        # metric_time, ind = np.unique(metric_time, return_index=True)
        # metric_value = metric_value[ind]
        # f_cubic = interp1d(metric_time, metric_value, kind='cubic')
        # metric_time_smooth = np.linspace(metric_time.min(), metric_time.max(), num=len(metric_time) * 5, endpoint=True, dtype=np.float64)
        # plt.plot(metric_time_smooth, f_cubic(metric_time_smooth), label=name, alpha=1.0)  # plot the interpolated data

        color = transparent[0].get_color()
        plt.plot(metric_time, ewma(metric_value), label=name, alpha=1.0, color=color)  # solid interpolated data

    plt.legend()
    plt.savefig(join(args.output, f'{metric.replace("/", "_")}.pdf'), bbox_inches='tight')
