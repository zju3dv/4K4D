# Running *EasyVolcap* on Static Multi-View Dataset

In the following sections, we'll provide an example of the [data-preparation](#colmap-for-camera-poses) process using COLMAP and ffmpeg](#colmap-for-camera-poses), and then show how to train and render [Instant-NGP+T](#our-instant-ngpt-implementation), [3DGS+T](#our-3dgst-implementation) and [ENeRF](#our-enerf-implementation-and-generalization) models inside ***EasyVolcap***.

Note that although the provided example is on a static dataset.
**The same preparation, training and rendering process can also be done on a dynamic dataset (e.g. multi-view video) with the same or similar commands.**
For now, we only provide examples of static datasets since most dynamic datasets are very large in size and hard to download.

For multi-view datasets with static cameras (suppose a sufficient number of cameras for COLMAP to run), run [COLMAP](#colmap-for-camera-poses) and [Instant-NGP+T](#our-instant-ngpt-implementation) on the first frame should give a rough camera pose.
The training and rendering process for models like [Instant-NGP+T](#our-instant-ngpt-implementation), [3DGS+T](#our-3dgst-implementation) and [ENeRF](#our-enerf-implementation-and-generalization) is almost completely the same as the static ones. Take notice of the `dataloader_cfg.dataset_cfg.frame_sample` and `val_dataloader_cfg.dataset_cfg.frame_sample` parameter to control the number of frames to use for training and evaluation. Please refer to [`config.md`](docs/design/config.md) for more details on how to pass in arguments using ***EasyVolcap***'s configuration system.

The example dataset used from this section can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1ZXji-2npuMqZRLkxXMWKUNS9kPikDThb/view?usp=sharing). After extracting the dataset, place the content inside `data/zju` such that you see files like:
- `data/zju/zju3dv/images`
- `data/zju/zju3dv/optimized`

### [COLMAP](https://github.com/colmap/colmap) for Camera Poses

In this section, we provide a basic example of extracting video frames, extracting camera parameters and preparing the dataset structure for ***EasyVolcap*** for a static scene.

For now, we only implemented the pipeline of extracting video frames and running COLMAP on it.
Foreground segmentation is also supported and documented in [`segmentation.md`](docs/misc/segmentation.md). 
[NeRFStudio](https://github.com/nerfstudio-project/nerfstudio) provides guidance on how to utilize ARKit of LiDAR-enabled iPhone or iPad devices to quickly get access to the camera parameters, and we're looking forward to incorporating those data formats into our pipeline.

COLMAP is robust, but extremely slow to run. 
Thus we wrote a script to help you with those annoying parameter tuning and database management with pretty prints of stuff. 
The parameter choices provided in the script were tested with a recording of [*Xiaowei Zhou's Lab*](https://xzhou.me).

Here we assume the user has the following directory structure after completing the data preprocessing step.

```shell
data/mww/zju3dv # data_root & data_root
├── intri.yml # required
├── extri.yml # required
├── images # required
├── colmap # optional, required during colmap pre-processing
│   ├── colmap.db
│   ├── colmap_ply
│   │   └── 000000.ply
│   ├── colmap_sparse
│   │   └── 0
│   │       ├── cameras.bin
│   │       ├── images.bin
│   │       ├── points3D.bin
│   │       └── project.ini
│   └── colmap_text
│       ├── cameras.txt
│       ├── images.txt
│       └── points3D.txt
└── videos # optional, required before extracting images
    └── 00
        └── output.mp4
```

To begin with, place your video inside the `data/zju/zju3dv/videos/00` folder. 

And run the following command to

1. Extract frames from the video
2. Discard blurry frames
3. Run COLMAP
4. Convert to easyvolcap camera format

Let's assume you've stored the location of `data/zju/zju3dv` in a variable `data_root` and also used shell arguments for other parameter choices.

```shell
expname="zju3dv"
data_root="data/zju/zju3dv"
video="output.mp4"
ffmpeg="/usr/bin/ffmpeg"
fps="20"
resolution="1920:1080"
ffmpeg_vf="zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,"
```

After the preparation, you can extract the video frames using:

```shell
# Extra video frames in images/00, make sure to tune fps and change the scale to match your video (or omit them to use the video's default)
mkdir -p "${data_root}/images/00"
${ffmpeg} -i "${data_root}/videos/00/${video}" -vf "${ffmpeg_vf}fps=${fps},scale=${resolution}" -q:v 1 -qmin 1 -start_number 0 "${data_root}/images/00/%06d.jpg"
```
Before running COLMAP, it's also recommended to discard blurry images to aid the reconstruction.
After which, you can run COLMAP using the following command:


```shell
# Discard blurry images, maybe tune the threshold, check the script for more details
python scripts/colmap/discard_blurry.py --data_root "${data_root}/images/00"

# Run COLMAP, this took roughly 3 hours on ~700 images
python scripts/colmap/run_colmap.py --data_root "${data_root}"

# Convert to easymocap dataformat
python scripts/colmap/colmap_to_easyvolcap.py --data_root "${data_root}"

# Flatten the images folder
python scripts/colmap/unflatten_dataset.py --data_root "${data_root}"
```

### Our [Instant-NGP](https://github.com/NVlabs/instant-ngp)+T Implementation

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/41d8cc39-deac-4955-8342-f7c531717ddc

Now the data preparation is completed, you've got an `images` folder and a pair of `intri.yml` and `extri.yml` files, you can run the l3mhet model.
- **L3**: 3 levels of importance sampling.
- **MHE**: Multi-resolution hash encoding.
- **T**: Supports time dimension input.

You need to write a config file for this model
1. Write the data-folder-related stuff inside configs/datasets. Just copy and paste [`configs/datasets/zju/zju3dv.yaml`](configs/datasets/zju/zju3dv.yaml) and modify the `data_root` and `bounds` (bounding box), or maybe add the camera's `near` `far` threshold.
2. Write the experiment config inside configs/exps. Just copy and paste [`configs/exps/l3mhet/l3mhet_zju3dv.yaml`](configs/exps/l3mhet/l3mhet_zju3dv.yaml) and modify the `dataset`-related line in `configs`.

The training should convert to a meaningful stage after 10-20 mins on a 3090 (actually the first 500 iter already produces a reasonable result).

```shell
# With your config files ready, you can run the following command to train the model
evc-train -c configs/exps/l3mhet/l3mhet_${expname}.yaml

# Now run the following command to render some output
evc-test -c configs/exps/l3mhet/l3mhet_${expname}.yaml,configs/specs/spiral.yaml
```
[`configs/specs/spiral.yaml`](configs/specs/spiral.yaml): please check this file for more details, it's a collection of configurations to tell the dataloader and visualizer to generate a spiral path by interpolating the given cameras

### Our [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)+T Implementation

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/09c4ebf3-88a5-49ba-8b2a-c6212f2d0004

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/f2f583df-f7e9-41ab-a4bb-2b6578b7a781

After preparing the dataset and model with COLMAP and Instant-NGP following the tutorial in the previous sections, we're ready to train some representation with better representational power and better rendering quality.

The original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) uses the sparse reconstruction result of COLMAP for initialization.
However, we found that the sparse reconstruction result often contains a lot of floating points, which is hard to prune for 3DGS and could easily make the model fail to converge.
Thus, we opted to use the "dense" reconstruction result of our Instant-NGP+T implementation by computing the RGBD image for input views and concatenating them as the input of 3DGS. The script [`volume_fusion.py`](scripts/fusion/volume_fusion.py) controls this process and it should work similarly on all models that support depth output.

Following the official [Instant-NGP](https://github.com/NVlabs/instant-ngp) implementation, we also perform camera parameter optimizations during the training of the Instant-NGP+T model (extrinsic parameters for now).
The script responsible for extracting the camera parameters is [`extract_optimized_cameras.py`](scripts/tools/extract_optimized_cameras.py).
An `optimized` folder will be created inside your dataset root, where a pair of `intri.yml` and `extri.yml` will be stored there.
After extraction, use [`optimized.yaml`](configs/specs/optimized.yaml) in your experiment config for applying the optimized parameters.

The following script block provides examples of how to extract the optimized camera parameters and prepare an initialization for our 3DGS+T implementation.

```shell
# Extract optimized cameras from the l3mhet model
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_${expname}.yaml

# Extract geometry (point cloud) for initialization from the l3mhet model
# Tune image sample rate and resizing ratio for a denser or sparser estimation
python scripts/fusion/volume_fusion.py -- -c configs/exps/l3mhet/l3mhet_${expname}.yaml val_dataloader_cfg.dataset_cfg.ratio=0.25 val_dataloader_cfg.dataset_cfg.view_sample=0,null,25

# Move the rendering results to the dataset folder
mkdir -p ${data_root}/vhulls
cp data/geometry/l3mhet_${expname}/POINT/frame0000.ply ${data_root}/vhulls/000000.ply
```

Our convention for storing initialization point clouds:
- Raw point clouds extracted using Instant-NGP or Space Carving are placed inside the `vhulls` folder. These files might be large. It's OK to directly optimize 3DGS+T on these.
- We might perform some clean-up of the point clouds and store them in the `surfs` folder.
  - For 3DGS+T, the cleaned-up point clouds might be easier to optimize since 3DGS is good at growing details but not so good at dealing with floaters (removing or splitting).
  - For other representations, the cleaned-up point clouds work better than the visual hull (from Space Carving) but might not work so well compared to the raw point clouds of Instant-NGP.

Then, prepare a nexperiment config like [`configs/exps/gaussiant/gaussiant_zju3dv.yaml`](configs/exps/gaussiant/gaussiant_zju3dv.yaml).
The [`colmap.yaml`](configs/specs/colmap.yaml) provides some heuristics for large-scale static scenes. Remove these if you're not planning on using COLMAP's parameters directly.

```shell
# Train a 3DGS model on the ${expname} dataset
evc-train -c configs/exps/gaussiant/gaussiant_${expname}.yaml

# Perform rendering on the trained ${expname} dataset
evc-test -c configs/exps/gaussiant/gaussiant_${expname}.yaml,configs/specs/superm.yaml,configs/specs/spiral.yaml

# Perform rendering with GUI, do this on a machine with monitor, tested on Windows and Ubuntu
evc-gui -c configs/exps/gaussiant/gaussiant_${expname}.yaml,configs/specs/superm.yaml
```

The [`superm.yaml`](configs/specs/superm.yaml) skips the loading of input images and other initializations for network-only rendering since all the information we need is contained inside the trained model.

### Our Improved [ENeRF](https://github.com/zju3dv/ENeRF) Implementation and Generalization

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/24aaaade-e9e7-4ab4-a6d7-0a18a9905331

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/9e6d65cb-8977-4ded-81d7-769b86d3aa95

Pre-trained model for ENeRFi on the DTU dataset can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1OFBFxes9kje02RARFpYpQ6SkmYlulYca/view?usp=sharing). After downloading, rename the model to `latest.npz` place it in `data/trained_model/enerfi_dtu`.

```shell
# Render ENeRF with GUI on zju3dv dataset
evc-gui -c configs/base.yaml,configs/models/enerfi.yaml,configs/datasets/zju/zju3dv.yaml,configs/specs/optimized.yaml exp_name=enerfi_dtu val_dataloader_cfg.dataset_cfg.cache_raw=False val_dataloader_cfg.dataset_cfg.ratio=1.0 val_dataloader_cfg.dataset_cfg.view_sample=0,None,5 val_dataloader_cfg.dataset_cfg.force_sparse_view=True val_dataloader_cfg.dataset_cfg.n_srcs_list=3, val_dataloader_cfg.dataset_cfg.use_vhulls=True

# Render ENeRF with GUI on zju3dv dataset using fp16, higher performance, worse results on some views
evc-gui -c configs/base.yaml,configs/models/enerfi.yaml,configs/datasets/zju/zju3dv.yaml,configs/specs/optimized.yaml,configs/specs/fp16.yaml exp_name=enerfi_dtu val_dataloader_cfg.dataset_cfg.cache_raw=False val_dataloader_cfg.dataset_cfg.ratio=1.0 val_dataloader_cfg.dataset_cfg.view_sample=0,None,5 val_dataloader_cfg.dataset_cfg.force_sparse_view=True val_dataloader_cfg.dataset_cfg.n_srcs_list=3, val_dataloader_cfg.dataset_cfg.use_vhulls=True # only cache ten images

# Render ENeRF on zju3dv dataset with spiral paths
evc-test -c configs/base.yaml,configs/models/enerfi.yaml,configs/datasets/zju/zju3dv.yaml,configs/specs/optimized.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml exp_name=enerfi_dtu val_dataloader_cfg.dataset_cfg.cache_raw=False val_dataloader_cfg.dataset_cfg.ratio=1.0 val_dataloader_cfg.dataset_cfg.view_sample=0,None,5 val_dataloader_cfg.dataset_cfg.force_sparse_view=True val_dataloader_cfg.dataset_cfg.n_srcs_list=3, val_dataloader_cfg.dataset_cfg.use_vhulls=True val_dataloader_cfg.dataset_cfg.render_size=768,1366 runner_cfg.visualizer_cfg.save_tag=zju3dv

# Render ENeRF with GUI on RenBody dataset
evc-gui -c configs/base.yaml,configs/models/enerfi.yaml,configs/datasets/renbody/0013_01_obj.yaml,configs/specs/optimized.yaml,configs/specs/mask.yaml,configs/specs/vf0.yaml exp_name=enerfi_dtu val_dataloader_cfg.dataset_cfg.n_srcs_list=4, val_dataloader_cfg.dataset_cfg.use_vhulls=True val_dataloader_cfg.dataset_cfg.ratio=1.0
evc-gui -c configs/base.yaml,configs/models/enerfi.yaml,configs/datasets/renbody/0008_01_obj.yaml,configs/specs/optimized.yaml,configs/specs/mask.yaml,configs/specs/vf0.yaml exp_name=enerfi_dtu val_dataloader_cfg.dataset_cfg.n_srcs_list=4, val_dataloader_cfg.dataset_cfg.use_vhulls=True val_dataloader_cfg.dataset_cfg.ratio=1.0
```
