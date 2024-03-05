"""
This file may save mesh, point cloud, volume bit mask or other volumetric representation to disk
"""
import torch
import mcubes
import numpy as np
from torch import nn
from os.path import join
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import List, Dict, Tuple, Union, Type, Callable

from easyvolcap.engine import cfg
from easyvolcap.engine import VISUALIZERS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import Visualization, export_pts, export_mesh
from easyvolcap.utils.chunk_utils import multi_gather, multi_scatter_


@VISUALIZERS.register_module()
class GeometryVisualizer:
    def __init__(self,
                 result_dir: str = f'data/geometry',
                 save_tag: str = '',
                 types: List[str] = [
                     Visualization.MESH.name,
                 ],
                 exts: Dict[str, str] = {
                     Visualization.MESH.name: ".ply",
                     Visualization.POINT.name: ".ply",
                 },
                 exports: Dict[str, Callable] = {
                     Visualization.MESH.name: (lambda mesh, filename: export_mesh(**mesh, filename=filename)),
                     Visualization.POINT.name: (lambda mesh, filename: export_pts(**mesh, filename=filename)),
                 },
                 verbose: bool = True,
                 pool_limit: int = 10,  # maximum number of pending tasks in the thread pool

                 occ_thresh: float = 0.5,
                 sdf_thresh: float = 0.0,
                 **kwargs,
                 ):
        self.occ_thresh = occ_thresh
        self.sdf_thresh = sdf_thresh

        result_dir = join(result_dir, cfg.exp_name)  # MARK: global configuration
        result_dir = join(result_dir, save_tag) if save_tag != '' else result_dir
        self.result_dir = result_dir
        self.types = [Visualization[t] for t in types]  # types of visualization
        self.exts = exts  # file extensions for each type of visualization
        self.exports = exports
        self.verbose = verbose

        self.thread_pools: List[ThreadPool] = []
        self.pool_limit = pool_limit
        self.geo_pattern = f'{{type}}/frame{{frame:04d}}_camera{{camera:04d}}{{ext}}'

        if verbose:
            log(f'Visualization output: {yellow(join(result_dir, dirname(self.geo_pattern)))}')  # use yellow for output path
            log(f'Visualization types:', line(types))

    def generate_type(self, output: dotdict, batch: dotdict, type: Visualization = Visualization.MESH):
        if type == Visualization.MESH:
            if 'sdf' in output:
                occ = -output.sdf + 0.5
                self.occ_thresh = -self.sdf_thresh + 0.5
            else:
                occ = output.occ
            voxel_size = batch.meta.voxel_size
            W, H, D = batch.meta.W[0].item(), batch.meta.H[0].item(), batch.meta.D[0].item()  # !: BATCH
            cube = torch.full((np.prod(batch.valid.shape),), -10.0, dtype=occ.dtype, device='cpu')[None]  # 1, WHD
            cube = multi_scatter_(cube[..., None], batch.inds.cpu(), occ.cpu())  # dim = -2 # B, WHD, 1 assigned B, P, 1
            cube = cube.view(-1, W, H, D)  # B, W, H, D

            # We leave the results on CPU but as tensors instead of numpy arrays
            torch.cuda.synchronize()  # some of the batched data are asynchronously moved to the cpu
            verts, faces = mcubes.marching_cubes(cube.float().numpy()[0], self.occ_thresh)
            verts = torch.as_tensor(verts, dtype=torch.float)[None]
            faces = torch.as_tensor(faces.astype(np.int32), dtype=torch.int)[None]
            verts = verts * voxel_size.to(verts.dtype) + batch.meta.bounds[:, 0].to(verts.dtype)  # !: BATCH

            mesh = dotdict()
            mesh.verts = verts
            mesh.faces = faces
            log(f'Number of vertices: {verts.numel()}, faces: {faces.numel()} (frame: {batch.meta.frame_index.item()})')
        else:
            raise NotImplementedError(f'Unimplemented visualization type: {type}')
        return mesh

    def visualize_type(self, output: dotdict, batch: dotdict, type: Visualization = Visualization.MESH):
        geos = self.generate_type(output, batch, type)  # can be batched

        geo_stats = dotdict()
        camera_index: torch.Tensor = batch.meta.camera_index
        frame_index: torch.Tensor = batch.meta.frame_index
        geo_paths = []
        geo_arrays = []

        for i in range(len(frame_index)):
            frame = frame_index[i].item()
            camera = camera_index[i].item()

            # For shared values
            geo_path = self.geo_pattern.format(type=type.name, camera=camera, frame=frame, ext=self.exts[type.name])

            # Images
            geo = dotdict({k: geos[k][i] for k in geos.keys()})

            # For recorder
            geo_stats[geo_path] = geo

            # Saving images to disk
            geo_paths.append(join(self.result_dir, geo_path))
            geo_arrays.append(geo)

        pool = parallel_execution(geo_arrays, geo_paths, action=self.exports[type.name], async_return=True, num_workers=3)  # actual writing to disk (async)
        self.thread_pools.append(pool)
        self.limit_thread_pools()  # maybe clear some of the taskes in the thread pool
        return geo_stats

    def visualize(self, output: dotdict, batch: dotdict):
        geo_stats = dotdict()
        for type in self.types:
            geo_stats.update(self.visualize_type(output, batch, type))
        return geo_stats

    def limit_thread_pools(self):
        if len(self.thread_pools) > self.pool_limit:
            for pool in self.thread_pools[:self.pool_limit]:
                pool.close()
                pool.join()
            self.thread_pools = self.thread_pools[self.pool_limit:]

    def summarize(self):
        for pool in self.thread_pools:  # finish all pending taskes before generating videos
            pool.close()
            pool.join()
        self.thread_pools.clear()  # remove all pools for this evaluation
        return dotdict()
