import json
import torch
import lpips
import numpy as np
import collections.abc

from tqdm import tqdm
from termcolor import colored
from rich.pretty import pprint
from skimage.metrics import structural_similarity as compare_ssim

# fmt: off
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from easyvolcap.utils.console_utils import log
from easyvolcap.utils.relight_utils import linear2srgb, cart2sph, srgb2linear
from easyvolcap.utils.math_utils import normalize
# fmt: on

lpips_model = lpips.LPIPS(verbose=False, net='vgg').to('cuda', non_blocking=True)

# x,y -> scalar


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def sort_dict(od):
    res = dict()
    for k, v in sorted(od.items()):
        if isinstance(v, dict):
            res[k] = sort_dict(v)
        else:
            res[k] = v
    return res


def compute_psnr(x, y, mask, gt_mask): 
    x = x[gt_mask]
    y = y[gt_mask] # only valid pixels for PSNR
    return 10 * np.log(1 / ((x - y)**2).mean()) / np.log(10)
def compute_ssim(x, y, mask, gt_mask): 
    a, b, w, h = cv2.boundingRect(gt_mask.astype(np.uint8))
    x = x[b:b+h, a:a+w]
    y = y[b:b+h, a:a+w]
    return compare_ssim(x.astype(np.float32), y.astype(np.float32), channel_axis=-1).item()
def compute_lpips(x, y, mask, gt_mask): 
    a, b, w, h = cv2.boundingRect(gt_mask.astype(np.uint8))
    x = x[b:b+h, a:a+w]
    y = y[b:b+h, a:a+w]
    return lpips_model(torch.from_numpy(x).to('cuda', non_blocking=True).float().permute(2, 0, 1)[None], torch.from_numpy(y).to('cuda', non_blocking=True).float().permute(2, 0, 1)[None])[0].item()


def compute_degree(x, y, mask, gt_mask):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    # msk = (y.sum(dim=-1) != 0) & (x.sum(dim=-1) != 0)  # use gt degrees as mask
    msk = mask & gt_mask
    # only compute degrees on valid regions
    x = x[msk]
    y = y[msk]
    x = (x - 0.5) * 2
    y = (y - 0.5) * 2
    x = normalize(x)
    y = normalize(y)
    dot = (x * y).sum(dim=-1)
    dot = dot.clip(-1, 1)
    return torch.rad2deg(torch.arccos(dot).mean()).item()

# x -> x


def trans_identity(x): return x
def trans_pi(x): return x * np.pi
def trans_linear2srgb(x): return linear2srgb(torch.from_numpy(x)).numpy()
def trans_srgb2linear(x): return srgb2linear(torch.from_numpy(x)).numpy()
def trans_flip(x): return (x[..., [2, 1, 0]] - 0.5) * 2

# x, y -> x, y


def align_identity(x, y): return x, y


def align_median(x, y):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    msk = (y.sum(dim=-1) != 0) & (x.sum(dim=-1) != 0)  # use gt degrees as mask
    # only compute degrees on valid regions

    y_msk = y[msk]
    x_msk = x[msk]
    if len(x_msk) and len(y_msk):
        y_median = torch.median(y[msk], dim=0)[0]
        x_median = torch.median(x[msk], dim=0)[0]
        return (x * y_median / x_median).numpy(), (y).numpy()
    else:
        return x.numpy(), y.numpy()

def align_median_backward(x, y):
    return align_median(y, x)[::-1]


metrics = [['degree'],
           ['psnr', 'ssim', 'lpips'],
           ['psnr', 'ssim', 'lpips'],
           ['psnr', 'ssim', 'lpips'],
           ]
keys = ['normal', 'albedo', 'rendering', 'shading']
gt_keys = ['normals', 'albedo_diffuse', 'rgb', 'rgb']
exts = ['.exr', '.exr', '.exr', '.exr']
gt_exts = ['.png', '.png', '.exr', '.exr']
# gt_outs = ['output', 'output', 'output', 'output_vis']
gt_outs = ['output_hdr_test', 'output_hdr_test', 'output_hdr_test', 'output_vis_test']
# out normal is as is
# out albedo needs tonemapping
# out rendering needs tonemapping
# out shading needs tonemapping
pre_trans = ['identity', 'identity', 'identity', 'identity']  # transform colorspace
# post_trans = ['identity', 'linear2srgb', 'linear2srgb', 'linear2srgb']  # transform colorspace
post_trans = ['identity', 'identity', 'identity', 'identity']  # transform colorspace
# post_trans = ['identity', 'pi', 'identity', 'identity']  # transform colorspace
# post_trans = ['identity', 'identity', 'linear2srgb', 'identity']  # transform colorspace
# gt normal should be flipped
# gt albedo needs to be tonemapped
# gt rendering are .pngs, as is
# gt vis needs tonemapping
# pre_gt_trans = ['flip', 'identity', 'srgb2linear', 'identity']  # transform colorspace
pre_gt_trans = ['flip', 'identity', 'identity', 'identity']  # transform colorspace
# post_gt_trans = ['identity', 'linear2srgb', 'linear2srgb', 'linear2srgb']  # transform colorspace
post_gt_trans = ['identity', 'identity', 'identity', 'identity']  # transform colorspace
# post_gt_trans = ['identity', 'pi', 'identity', 'identity']  # transform colorspace
# post_gt_trans = ['identity', 'identity', 'linear2srgb', 'identity']  # transform colorspace
# use median values to align them
# aligns = ['identity', 'identity', 'identity', 'identity']  # align albedo and rendered images
aligns = ['identity', 'median', 'median', 'median']  # align albedo and rendered images

# exp_templates = ['relight_{data}_{human}', 'relighting4d_{data}_{human}', 'bruteforce_{data}_{human}']
exp_templates = [
    # 'relight_{data}_{human}',
    # 'ablation_relight_{data}_{human}',
    # 'relighting4d_{data}_{human}',
    'nerfactor_{data}_{human}_1f',
    # 'bruteforce_{data}_{human}'
]
datas = ['synthetic']
humans = [
    # 'jody',
    # 'josh',
    # 'malcolm',
    'megan',
    # 'nathan'
]

frames = list(range(100))[::21]  # 5 images ?
views = [12, 15, 19]  # 3 views
# views = [0, 4, 8]  # 3 views

frames_list = [
    list(range(100))[::21],
    list(range(100))[::21],
    list(range(1, 70))[::13],
    list(range(0, 100))[::21],
    list(range(1, 69))[::13],
]
views_list = [
    views,
    views,
    views,
    views,
    views
]

# lights = ['gym_entrance', 'shanghai_bund', 'peppermint_powerplant_blue', 'olat0002-0027', 'olat0004-0019', 'olat0004-0017']
# gt_lights = ['gym', 'shanghai', 'peppermint', 'olat2', 'olat9', 'olat7']

# lights = ['olat0002-0027', 'olat0004-0019', 'olat0004-0017']
# gt_lights = ['olat2', 'olat9', 'olat7']

lights = ['gym_entrance', 'shanghai_bund', 'olat0002-0027', 'olat0004-0019']
gt_lights = ['gym', 'shanghai', 'olat2', 'olat9']

nerfactor_1f_path_template = 'data/novel_light/{exp}/eval/frame{frame:04d}/{key}/{light}/{view:04d}{ext}'
pred_path_template = 'data/novel_light/{exp}/eval/motion/{key}/{light}/frame{frame:04d}_view{view:04d}{ext}'  # read exr
gt_path_template = '/nas/home/gengchen/develop/blenderproc_render/{gt_out}/{human}_{gt_light}/train/images/{frame:06d}_{view:02d}_{gt_key}{gt_ext}'

# this dict should be saved to a log file for later viewing
# result_dict = {
#     e: {
#         d: {
#             h: {

#             } for h in humans
#         } for d in datas
#     } for e in exps
# }
result_dict = {}

for exp_template in exp_templates:  # 4
    for data in datas:  # 1
        for human, frames, views in zip(humans, frames_list, views_list):  # 5
            exp = exp_template.format(human=human, data=data)

            result_dict[exp] = {}

            # if 'megan' in human and 'nerfactor' in exp and '1f' in exp:
            #     log(f'Skipping megan for nerfactor_1f, did not converge')
            #     continue  # skip megan for nerfactor_1f, did not converge

            # find the experiment data root
            for key_index in range(len(keys)):
                key = keys[key_index]
                ext = exts[key_index]
                gt_ext = gt_exts[key_index]
                gt_key = gt_keys[key_index]
                gt_out = gt_outs[key_index]

                metric_list = metrics[key_index]
                pre_tran = pre_trans[key_index]
                post_tran = post_trans[key_index]
                pre_gt_tran = pre_gt_trans[key_index]
                post_gt_tran = post_gt_trans[key_index]
                align = aligns[key_index]
                pre_transform = globals()[f'trans_{pre_tran}']
                post_transform = globals()[f'trans_{post_tran}']
                pre_gt_transform = globals()[f'trans_{pre_gt_tran}']
                post_gt_transform = globals()[f'trans_{post_gt_tran}']
                alignment = globals()[f'align_{align}']

                metric_value_dict = {k: [] for k in metric_list}  # stores the value for this metric

                # will take mean of the following computation's output
                pbar = tqdm(total=len(frames) * len(views) * len(lights))
                for frame in frames:  # 5
                    for view in views:  # 3
                        for light_index in range(len(lights)):  # 4
                            # if ('normal' in key or 'albedo' in key) and light_index >= 1:
                            #     continue

                            light = lights[light_index]
                            gt_light = gt_lights[light_index]
                            pbar.desc = f'{exp}:{key}:frame{frame}:view{view}:{light}'

                            if 'nerfactor' in exp and '1f' in exp:
                                pred_path = nerfactor_1f_path_template.format(exp=exp, key=key, light=light, frame=frame, view=view, ext=ext)
                            else:
                                pred_path = pred_path_template.format(exp=exp, key=key, light=light, frame=frame, view=view, ext=ext)
                            gt_path = gt_path_template.format(gt_out=gt_out, human=human, gt_light=gt_light, frame=frame, view=view, gt_key=gt_key, gt_ext=gt_ext)
                            gt_mask_path = os.path.splitext(gt_path)[0].replace(gt_key, 'mask') + '.png'

                            if not os.path.exists(pred_path):
                                pred_path = pred_path.replace('frame', 'frame_')  # errornous?
                            if not os.path.exists(pred_path):
                                log(f'{pred_path} does not exist. Skipping evaluation for this frame', 'red')
                                continue
                            if not os.path.exists(gt_path):
                                log(f'{gt_path} does not exist. Skipping evaluation for this frame', 'red')
                                continue
                            if not os.path.exists(gt_mask_path):
                                log(f'{gt_mask_path} does not exist. Skipping evaluation for this frame', 'red')
                                continue

                            pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
                            gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_UNCHANGED)

                            H, W = pred.shape[:2]
                            gt = cv2.resize(gt, (W, H), cv2.INTER_AREA)  # down sampling
                            gt_mask = cv2.resize(gt_mask, (W, H), cv2.INTER_AREA)  # down sampling

                            gt_mask = cv2.dilate(gt_mask, np.ones((3, 3)))
                            gt_mask = cv2.erode(gt_mask, np.ones((3, 3)))
                            
                            if gt_mask.sum() < 1000:
                                log(f'{gt_mask_path} is invalid. Skipping evaluation for this frame', 'red')
                                continue

                            # mask = gt_mask # fill background with zeros
                            mask = pred[..., -1] > 0
                            gt_mask = gt_mask > 0  # fill background pixels with zeros before computing metrics
                            pred = pred[..., :3]  # only compute metrics on color channels
                            gt = gt[..., :3]  # only compute metrics on color channels
                            pred[~mask] = 0
                            gt[~gt_mask] = 0

                            if pred.dtype == np.uint8:
                                pred = pred / 255
                            elif pred.dtype == np.uint16:
                                pred = pred / 65535
                            pred = pred.astype(np.float32)  # normalized to 0, 1 (exr or png or whatever)

                            if gt.dtype == np.uint8:
                                gt = gt / 255
                            elif gt.dtype == np.uint16:
                                gt = gt / 65535
                            gt = gt.astype(np.float32)  # normalized to 0, 1 (exr or png or whatever)

                            pred = pre_transform(pred)
                            gt = pre_gt_transform(gt)  # gt transform sometimes violates 0 for normals
                            pred, gt = alignment(pred, gt)
                            pred = post_transform(pred)
                            gt = post_gt_transform(gt)  # gt transform sometimes violates 0 for normals
                            pred[~mask] = 0
                            gt[~gt_mask] = 0

                            for metric in metric_list:
                                compute_metric = globals()[f'compute_{metric}']
                                value = compute_metric(pred, gt, mask, gt_mask)
                                metric_value_dict[metric].append(value)

                            pbar.update(1)
                pbar.close()
                metric_value_mean = {k: np.mean(v) for k, v in metric_value_dict.items()}
                # result_dict[exp][data][human][key] = metric_value_mean
                result_dict[exp][key] = metric_value_mean

                # log(f'\n{colored(exp, "cyan")}:{colored(exp, "magenta")}\n{metric_value_mean}')
                # pprint(f'\n{colored(exp, "cyan")}:{colored(exp, "magenta")}\n{metric_value_mean}')
                print(f'{colored(exp, "blue")}: {colored(key, "magenta")}')
                pprint(metric_value_mean)

                try:
                    if os.path.exists("data/metrics.json"):
                        metrics_json = json.load(open('data/metrics.json', 'r'))
                        update(metrics_json, result_dict)
                        json.dump(sort_dict(metrics_json), open('data/metrics.json', 'w'), indent=4)
                except:
                    json.dump(sort_dict(result_dict), open('data/metrics.json', 'w'), indent=4)


# __import__('ipdb').set_trace()
# json.dump(result_dict, open('data/metrics.json', 'w'))
