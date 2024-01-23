# Read a metrics.json and convert it to time.json

from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    args = dotdict()
    args.metrics_json = '/mnt/data/home/xuzhen/projects/large_gaussian/demo/levels/3dgs/metrics.json'
    args.time_json = '/mnt/data/home/xuzhen/projects/large_gaussian/demo/levels/3dgs/time.json'
    args.main_axis = 'camera'
    args = dotdict(vars(build_parser(args).parse_args()))

    metrics = dotdict(json.load(open(args.metrics_json, 'r')))
    times = dotdict()
    for value in metrics.metrics:
        times[value[args.main_axis]] = value.time
    json.dump(times, open(args.time_json, 'w'), indent=4)
    log(yellow(f'Converted json saved to: {blue(args.time_json)}'))


if __name__ == '__main__':
    main()
