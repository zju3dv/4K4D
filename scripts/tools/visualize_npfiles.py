import os
import numpy as np
import pandas as pd
from functools import reduce
# This script serves the purpose of reading the content and meaningfully visualize the data a numpy dict
# that is, a dict whose keys are numpy arrays

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input', default='data/xuzhen36/talk/lbs/smpl_params.npy')
parser.add_argument('-o', '--output', default='')
args = parser.parse_args()

if not args.output:
    args.output = os.path.splitext(args.input)[0] + '.xlsx'

d = np.load(args.input, allow_pickle=True)
if isinstance(d, np.ndarray) and np.squeeze(d).size == 1:
    # the outer most scope is already a dict
    d = d.item()
    assert isinstance(d, dict)
elif isinstance(d, np.lib.npyio.NpzFile):
    d = {**d}
else:
    # should apply an augmentation
    n = os.path.splitext(os.path.basename(args.input))[0]
    d = {n: d}


w = pd.ExcelWriter(args.output)


def get_indices(*shapes):
    inds = np.stack(np.meshgrid(*[np.arange(s) for s in shapes], indexing='ij'), axis=-1)
    inds = reduce(np.char.add, np.split(inds.astype(str), inds.shape[-1], axis=-1))
    return inds.ravel()


def traverse(d, w, key_prefix=''):
    for key, item in d.items():
        key_full = key_prefix + '.' + key if key_prefix else key
        if isinstance(item, dict):
            traverse(item, w, key_full)
        else:
            item = np.array(item)
            if item.ndim == 1:
                item = item[:, None]
            # item might be a high-d np array
            df = pd.DataFrame(item.reshape(item.shape[0], -1), columns=get_indices(*item.shape[1:]))
            df.to_excel(w, key_full)

# apply tree traversal
traverse(d, w)
print(f'writing to: {args.output}')
w.close()
