# Load bounds, find union, report them
import csv
import argparse
import numpy as np
from easyvolcap.utils.console_utils import *


def read_aabbs_csv(input_csv_path: str) -> List[List]:
    """Read camera intrinsics and extrinsics from a calibration CSV file.

    Args:
        input_csv_path (Path): Path to a CSV file that contains camera calibration data.

    Returns:
        List[CameraData]: A list of `CameraData` objects that describe multiple camera intrinsics and extrinsics.
    """
    bounds = []
    with open(input_csv_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row = dotdict(row)
            bounds.append([[float(row.aabb_min_x), float(row.aabb_min_y), float(row.aabb_min_z)], [float(row.aabb_max_x), float(row.aabb_max_y), float(row.aabb_max_z)]])
    return bounds


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="data/actorshq/Actor01/Sequence01/1x")
    args = parser.parse_args()
    data_root = args.data_root

    aabbs_path = join(data_root, "..", 'aabbs.csv')
    bounds = np.array(read_aabbs_csv(aabbs_path))  # F, 2, 3
    bounds_min = bounds[:, 0].min(axis=0)  # 3,
    bounds_max = bounds[:, 1].max(axis=0)  # 3,

    bounds = np.stack([bounds_min, bounds_max])
    log(f'{data_root}: {line(bounds)}')


if __name__ == '__main__':
    main()
