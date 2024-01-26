import torch
import numpy as np
from torch import nn
from os.path import join, dirname
from typing import List, Tuple, Union, Type
from multiprocessing.pool import ThreadPool

from easyvolcap.engine import cfg, args  # global
from easyvolcap.engine import VISUALIZERS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.color_utils import colormap
from easyvolcap.utils.depth_utils import depth_curve_fn
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import save_image, generate_video, Visualization


@VISUALIZERS.register_module()
class VolumetricVideoVisualizer:  # this should act as a base class for other types of visualizations (need diff dataset)
    def __init__(self,
                 uncrop_output_images: bool = True,  # will try to find crop_h, crop_w...
                 store_alpha_channel: bool = True,  # store rendered acc in alpha channel
                 store_ground_truth: bool = True,  # store the ground truth rendered values
                 store_image_error: bool = True,  # render the error map (usually mse)
                 store_video_output: bool = False,  # whether to construct .mp4 from .png s

                 vis_ext: str = '.png',  # faster saving, faster viewing, not good for evaluation (metrics)
                 result_dir: str = 'data/result',
                 img_pattern: str = f'{{type}}/frame{{frame:04d}}_camera{{camera:04d}}',  # the formatting of the output
                 save_tag: str = '',
                 types: List[str] = [
                     Visualization.RENDER.name,
                     Visualization.DEPTH.name,
                     Visualization.ALPHA.name,
                 ],

                 stream_delay: int = 2,  # after this number of pending copy, start synchronizing the stream and saving to disk
                 pool_limit: int = 10,  # maximum number of pending tasks in the thread pool, keep this small to avoid using too much resource
                 video_fps: int = 60,
                 verbose: bool = True,

                 dpt_curve: str = 'normalize',  # looks good
                 dpt_mult: float = 1.0,
                 dpt_cm: str = 'virdis' if args.type != 'gui' else 'linear',  # looks good
                 ):
        super().__init__()

        self.uncrop_output_images = uncrop_output_images
        self.store_alpha_channel = store_alpha_channel
        self.store_ground_truth = store_ground_truth
        self.store_video_output = store_video_output
        self.store_image_error = store_image_error

        result_dir = join(result_dir, cfg.exp_name)  # MARK: global configuration # TODO: unify the global config, currently a hack for orbit.yaml here
        result_dir = join(result_dir, str(save_tag)) if save_tag != '' else result_dir  # could be a pure number
        self.vis_ext = vis_ext
        self.save_tag = save_tag
        self.result_dir = result_dir
        self.types = [Visualization[t] for t in types]  # types of visualization

        self.img_pattern = img_pattern + self.vis_ext
        self.img_gt_pattern = self.img_pattern.replace(self.vis_ext, f'_gt{self.vis_ext}')
        self.img_error_pattern = self.img_pattern.replace(self.vis_ext, f'_error{self.vis_ext}')

        self.thread_pools: List[ThreadPool] = []
        self.cuda_streams: List[torch.cuda.Stream] = []
        self.cpu_buffers: List[torch.Tensor] = []
        self.stream_delay = stream_delay
        self.pool_limit = pool_limit

        self.video_fps = video_fps
        self.verbose = verbose
        self.dpt_curve = dpt_curve
        self.dpt_mult = dpt_mult
        self.dpt_cm = dpt_cm

        if self.verbose:
            types = '{' + ','.join([t.name for t in self.types]) + '}'
            log(f'Visualization output: {blue(join(self.result_dir, dirname(self.img_pattern).format(type=types)))}')  # use yellow for output path

    def generate_type(self, output: dotdict, batch: dotdict, type: Visualization = Visualization.RENDER):
        # Extract the renderable image from output and batch
        img: torch.Tensor = None
        img_gt: Union[torch.Tensor, None] = None
        img_error: Union[torch.Tensor, None] = None

        if type == Visualization.NORMAL:
            def norm_curve_fn(norm):
                norm = normalize(norm)
                norm = norm @ batch.R.mT
                norm[..., 1] *= -1
                norm[..., 2] *= -1
                norm = norm * 0.5 + 0.5
                norm = norm * output.acc_map  # norm is different when blending
                return norm

            img = norm_curve_fn(output.norm_map)
            if self.store_ground_truth and 'norm' in batch:
                img_gt = norm_curve_fn(batch.norm)

        elif type == Visualization.DEPTH:
            if self.dpt_curve == 'linear':
                img = output.dpt_map
            else:
                img = depth_curve_fn(output.dpt_map, cm=self.dpt_cm)
            # img = (img - 0.5) * self.dpt_mult + 0.5
            img = img * self.dpt_mult
            if self.store_ground_truth and 'dpt' in batch:
                if self.dpt_curve == 'linear':
                    img_gt = batch.dpt
                else:
                    img_gt = depth_curve_fn(batch.dpt, cm=self.dpt_cm)
                # img_gt = (img_gt - 0.5) * self.dpt_mult + 0.5
                img_gt = img_gt * self.dpt_mult

        elif type == Visualization.FEATURE:
            # This visualizes the xyzt + xyz feature output
            def feat_curve_fn(feat: torch.Tensor):
                B, P, C = feat.shape
                N = C // 3  # ignore last few feature channels
                feat = torch.stack(feat[..., :3 * N].chunk(3, dim=-1), dim=-1).mean(dim=-2)  # now in rgb
                return feat
            img = feat_curve_fn(output.feat_map)
            # No gt for this

        elif type == Visualization.SURFACE:
            img = output.surf_map  # rgb, maybe add multiplier

        elif type == Visualization.DEFORM:
            img = output.resd_map  # rgb, maybe add multiplier

        elif type == Visualization.ALPHA:
            img = output.acc_map.expand(output.acc_map.shape[:-1] + (3,))
            if self.store_ground_truth and 'msk' in batch:
                img_gt = batch.msk.expand(batch.msk.shape[:-1] + (3,))

        # ... implement more
        elif type == Visualization.RENDER:
            img = output.rgb_map
            if self.store_ground_truth and 'rgb' in batch:
                img_gt = batch.rgb

        elif type == Visualization.SRCINPS:
            # src_inps, only for per-command visualization
            img = batch.src_inps.permute(0, 1, 3, 4, 2)
            img = torch.cat([img[:, i] for i in range(img.shape[1])], dim=-2)
            return img, None, None

        else:
            raise NotImplementedError(f'Unimplemented visualization type: {type}')

        if img_gt is not None and 'bg_color' in output and 'msk' in batch:
            # Fill gt with input BG colors
            img_gt = img_gt + output.bg_color * (1 - batch.msk)

        if self.store_image_error and img_gt is not None:
            img_error = (img - img_gt).pow(2).sum(dim=-1).clip(0, 1)[..., None].expand(img.shape)

        if self.store_alpha_channel:
            msk = output.acc_map
            img = torch.cat([img, msk], dim=-1)
            if img_gt is not None:
                msk_gt = batch.msk
                img_gt = torch.cat([img_gt, msk_gt], dim=-1)
            if img_error is not None:
                msk_gt = batch.msk
                img_error = torch.cat([img_error, (msk_gt + msk).clip(0, 1)], dim=-1)

        B, P, C = img.shape
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
        img = img.view(B, H, W, C).float()
        if img_gt is not None: img_gt = img_gt.view(B, H, W, C).float()
        if img_error is not None: img_error = img_error.view(B, H, W, C).float()

        if self.uncrop_output_images:  # necessary for GUI applications
            if 'orig_h' in batch.meta:  # !: BATCH: Removed
                x, y, w, h = batch.meta.crop_x[0].item(), batch.meta.crop_y[0].item(), batch.meta.W[0].item(), batch.meta.H[0].item()
                H, W = batch.meta.orig_h[0].item(), batch.meta.orig_w[0].item()
                img_full = img.new_zeros(B, H, W, C)  # original size
                img_full[:, y:y + h, x:x + w, :] = img
                img = img_full
                if img_gt is not None:
                    img_gt_full = img_gt.new_zeros(B, H, W, C)  # original size
                    img_gt_full[:, y:y + h, x:x + w, :] = img_gt
                    img_gt = img_gt_full
                if img_error is not None:
                    img_error_full = img_error.new_zeros(B, H, W, C)  # original size
                    img_error_full[:, y:y + h, x:x + w, :] = img_error
                    img_error = img_error_full

        return img, img_gt, img_error

    def visualize_type(self, output: dotdict, batch: dotdict, type: Visualization = Visualization.RENDER):
        # Starting a new stream is a small overhead for the GPU, but it allows us to run the visualization in parallel
        dft_stream: torch.cuda.Stream = torch.cuda.current_stream()  # default stream
        vis_stream: torch.cuda.Stream = torch.cuda.Stream()
        vis_stream.wait_stream(dft_stream)
        torch.cuda.set_stream(vis_stream)

        # Prepare for recoreder and storing some stuff to disk
        imgs, img_gts, img_errors = self.generate_type(output, batch, type)  # can be batched
        image_stats = dotdict()
        camera_index: torch.Tensor = batch.meta.camera_index
        frame_index: torch.Tensor = batch.meta.frame_index
        img_paths = []
        img_arrays = []

        for i in range(len(imgs)):
            frame = frame_index[i].item()
            camera = camera_index[i].item()

            # For shared values # TODO: fix this hacky implementation
            self.camera = camera  # for generating video
            self.frame = frame  # for generating video
            img_path = self.img_pattern.format(type=type.name, camera=camera, frame=frame)
            img_gt_path = self.img_gt_pattern.format(type=type.name, camera=camera, frame=frame)
            img_error_path = self.img_error_pattern.format(type=type.name, camera=camera, frame=frame)

            # Images
            img = imgs[i]
            if img_gts is not None: img_gt = img_gts[i]
            if img_errors is not None: img_error = img_errors[i]

            # For recorder
            image_stats[img_path] = img
            if img_gts is not None: image_stats[img_gt_path] = img_gt
            if img_errors is not None: image_stats[img_error_path] = img_error

            # Saving images to disk
            img_paths.append(join(self.result_dir, img_path))
            img_arrays.append(img.detach().to('cpu', non_blocking=True))  # start moving
            if img_gts is not None:
                img_paths.append(join(self.result_dir, img_gt_path))
                img_arrays.append(img_gt.detach().to('cpu', non_blocking=True))  # start moving
            if img_errors is not None:
                img_paths.append(join(self.result_dir, img_error_path))
                img_arrays.append(img_error.detach().to('cpu', non_blocking=True))  # start moving

        self.cuda_streams.append(vis_stream)
        self.cpu_buffers.append((img_paths, img_arrays))
        self.limit_cuda_streams()
        self.limit_thread_pools()  # maybe clear some of the taskes in the thread pool

        dft_stream.wait_stream(vis_stream)  # wait for the copy in this stream to finish before any other cuda operations on the default stream begins
        torch.cuda.set_stream(dft_stream)  # restore the original state
        return image_stats  # it's OK to return this

    # We need an interface for constructing final output paths
    # Along with paths (keys) for tensorboard logging system
    # GT values are stored separatedly in another entry
    # Same for `error` values

    def visualize(self, output: dotdict, batch: dotdict):
        image_stats = dotdict()
        for type in self.types:
            image_stats.update(self.visualize_type(output, batch, type))
        return image_stats

    def limit_cuda_streams(self):
        stream_cnt = len(self.cuda_streams)
        if stream_cnt > self.stream_delay:
            excess_streams = self.cuda_streams[:stream_cnt - self.stream_delay]
            excess_buffers = self.cpu_buffers[:stream_cnt - self.stream_delay]
            for stream, buffer in zip(excess_streams, excess_buffers):
                stream.synchronize()  # wait for the copy in this stream to finish
                img_paths, img_arrays = buffer
                img_arrays = [im.numpy() for im in img_arrays]
                pool = parallel_execution(img_paths, img_arrays, action=save_image, async_return=True, num_workers=3)  # actual writing to disk (async)
                self.thread_pools.append(pool)
            self.cpu_buffers = self.cpu_buffers[stream_cnt - self.stream_delay:]
            self.cuda_streams = self.cuda_streams[stream_cnt - self.stream_delay:]

    def limit_thread_pools(self):
        pool_length = len(self.thread_pools)
        if pool_length > self.pool_limit:
            for pool in self.thread_pools[:pool_length - self.pool_limit]:
                pool.close()
                pool.join()
            self.thread_pools = self.thread_pools[pool_length - self.pool_limit:]

    def summarize(self):
        for stream, buffer in zip(self.cuda_streams, self.cpu_buffers):
            stream.synchronize()  # wait for the copy in this stream to finish
            img_paths, img_arrays = buffer
            img_arrays = [im.numpy() for im in img_arrays]
            pool = parallel_execution(img_paths, img_arrays, action=save_image, async_return=True, num_workers=3)  # actual writing to disk (async)
            self.thread_pools.append(pool)

        for pool in self.thread_pools:  # finish all pending taskes before generating videos
            pool.close()
            pool.join()
        self.thread_pools.clear()  # remove all pools for this evaluation

        if self.store_video_output:
            for type in self.types:
                result_dir = dirname(join(self.result_dir, self.img_pattern)).format(type=type.name, camera=self.camera, frame=self.frame)
                result_str = f'"{result_dir}/*{self.vis_ext}"'
                output_path = result_str[1:].split('*')[0][:-1] + '.mp4'
                try:
                    generate_video(result_str, output_path, self.video_fps)  # one video for one type?
                except RuntimeError as e:
                    log(yellow('Error encountered during video composition, will retry without hardware encoding'))
                    generate_video(result_str, output_path, self.video_fps, hwaccel='none', preset='veryslow', vcodec='libx265')  # one video for one type?
                log(f'Video generated: {blue(output_path)}')
                # TODO: use timg/tiv to visaulize the video / image on disk to the commandline

        if self.verbose:
            types = '{' + ','.join([t.name for t in self.types]) + '}'
            log(yellow(f'Visualization output: {blue(join(self.result_dir, dirname(self.img_pattern).format(type=types)))}'))  # use yellow for output path
        return dotdict()
