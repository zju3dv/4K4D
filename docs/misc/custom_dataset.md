# Preparing Custom Datasets For Dynamic Volumetric Videos

## Dataset Structure

***EasyVolcap*** has built-in support for both multi-view datasets and monocular datasets. Once you organize your dataset in one of the data formats listed below, you can directly use ***EasyVolcap*** for model training on your own dataset. You can also use the provided scripts to convert the exsiting multi-view or monocular datasets to the required format, you can refer to the [nerf_to_easyvolcap.py](../../scripts/nerf/nerf_to_easyvolcap.py) for an example of converting multi-view dataset, and the [colmap_to_easyvolcap.py](../../scripts/colmap/colmap_to_easyvolcap.py) for an example of converting monocular dataset, more preprocessing scripts can be found in the [scripts/preprocess](../../scripts/preprocess) directory.

### Multi-view Dataset

Basic dataset structure for multi-view dataset:

```shell
data/dataset/sequence # data_root & data_root
├── intri.yml # required: intrinsics
├── extri.yml # required: extrinsics
└── images # required: source images
    ├── 00 # camera / frame
    │   ├── 000000.jpg # image
    │   ├── 000001.jpg # for dynamic dataset, more images can be placed here
    │   ...
    ├── 01
    ├── 02
    ...
```

### Monocular Dataset

Basic dataset structure for monocular dataset:

```shell
data/dataset/sequence # data_root & data_root
├── cameras  # required: camera parameter for the whole sequence
│   ├── 00 # only one camera
│   │   ├── intri.yml # required: intrinsics
│   │   └── extri.yml # required: extrinsics
└── images # required: source images captured by the monocular camera
    ├── 00 # only one camera
    │   ├── 000000.jpg # image
    │   ├── 000001.jpg
    │   ...
```

Compared to multi-view datasets, the difference in data organization format for monocular datasets is that it includes a separate `cameras` folder, and both the `cameras` and `images` folders have only one subdirectory, indicating the presence of a single camera.

## Dataset Configuration

Before you can use ***EasyVolcap*** to train on your own dataset and get a reasonable result, you need to prepare a dataset configuration file for your dataset. The dataset configuration file is a YAML file that contains the basic information of your dataset.

Among all the configuration items, the most important ones that you need to pay attention to are the `near`, `far`, `bounds`, take the [enerf_outdoor.yaml](../../configs/datasets/enerf_outdoor/enerf_outdoor.yaml) as an example:

```yaml
dataloader_cfg:
    dataset_cfg:
        near: 3.5 # global near for the specific scene
        far: 9.0 # global far for the specific scene
        bounds: [[-3.0, -3.0, -2.0], [3.0, 3.0, 2.0]] # global bounds for the specific scene
```

Most algorithms require a reasonable sampling range to achieve good results. Therefore, you should provide approximate `near` and `far` values based on the dataset you captured. These values will serve as a unified sampling range for each viewpoint in the current scene. Additionally, set the `bounds` parameter to a bounding box that roughly encompasses the entire scene, using the calibrated scene center as the origin. ***EasyVolcap*** will utilize this `bounds` parameter to calculate the intersection points between rays and the scene, enabling further refinement of the near and far sampling ranges for each ray.

Dataset variables:

```shell
# Linux
expname=0013_01
data_root=data/renbody/0013_01

# Another dataset
expname=0013_01
data_root=data/renbody/0013_01

# Windows
$expname="0013_01"
$data_root="data/renbody/0013_01"
```

<!-- ## Camera Synchronization

- [ ] TODO: Board

## Static Calibration

Please refer to [static.md](../../docs/misc/static.md#colmap-for-camera-poses). -->

## Sparse Calibration

For sparse calibration, we utilized the method proposed in [EasyMocap](https://chingswy.github.io/easymocap-public-doc/quickstart/calibration/wild-sparse.html), which has been thoroughly validated for its robustness and accuracy through multiple iterations in our research.

Assuming you have successfully installed the **EasyVolcap** environment following the installation instructions in the [main readme](../../readme.md#installation), you can install [**EasyMocap**](https://chingswy.github.io/easymocap-public-doc/) simply via

```shell
# Clone EasyMocap somewhere outside the main directory of EasyVolcap
cd ~
git clone https://github.com/zju3dv/EasyMocap.git

# Install EasyMocap
cd EasyMocap
conda activate easyvolcap
python3 setup.py develop
```

After installation, first, you need to capture the `ground1f` and `background1f` images using your sparse set of cameras as described in the [wild + multiple sparse](https://chingswy.github.io/easymocap-public-doc/quickstart/calibration/wild-sparse.html), where `background1f` is a set of static background-only images captured by your multi-view cameras, and `ground1f` is the same set of background with a chessboard inside captured by your multi-view cameras which is used to align the scale and center of the COLMAP calibrated world coordinate system to the real world scale and to the place of the chessboard. For example, if you have a sparse set of webcams, `eg.` a Logitech webcam, you can perform the capture by:

```shell
cd ~/EasyMocap

# Capture the pure background, remember to change the camera config file to your own if you have more 6 webcams
python3 apps/camera/realtime_display.py --cfg_cam config/camera/usb-logitech.yml --display --num 1 --out ./data/background1f

# Capture the background with the chessboard in the scene, put the chessboard to the place where you want the world center to be
python3 apps/camera/realtime_display.py --cfg_cam config/camera/usb-logitech.yml --display --num 1 --out ./data/ground1f
```

Then, you can use your cell phone to record a video alongside your set of cameras, you can see how it goes in the example provided in [wild + multiple sparse](https://chingswy.github.io/easymocap-public-doc/quickstart/calibration/wild-sparse.html), and load the video to the `~/EasyMocap/data/background1f/` directory. After all the data is ready, you can run the following command to perform the sparse calibration using the [script](../../scripts/colmap/sparse_calibration.sh):

```shell
cd ~/easyvolcap
source scripts/colmap/sparse_calibration.sh
CUDA_VISIBLE_DEVICES=0 calibration 
```

Remember to change the paths of your `EasyMocap` and `scene_root` if you put it somewhere different than the default path. After the calibration is done, you can find the **EasyVolcap** required format camera parameters `extri.yml` and `intri.yml` under the `colmap/align/static/images` sub-directory of your `scene_root` directory. Create the symbolic link to any dataset root directory that is captured by the same set of cameras, and you can use the camera parameters for the dataset.

## Dense Calibration

**There's a [dedicated section in static.md](../../docs/misc/static.md#colmap-for-camera-poses) detailing the full calibration process from captured videos to a full dataset.**
**Following that, there's also a [detailed description about finetuning the camera parameters](../../docs/misc/static.md#our-instant-ngpt-implementation) using a modified instant-ngp model.**

Note that the dense calibration process on a multi-view dynamic system is pretty much the same, where we view the first frame (or any other meaning frame) as a static scene and run COLMAP on it.

Let's assume you've arranged your images inside the `images` folder like: `images/00/000000.jpg`, `images/00/000001.jpg` ... `images/01/000000.jpg`, `images/01/000001.jpg`. We'll first need to rearange the images into a unified directory for running COLMAP:

```shell
data_root=data/neural3dv/sear_steak

# Rearrange multi-view dataset (soft link for images for running COLMAP)
python scripts/colmap/arrange_images.py --data_root "${data_root}" # run with -h for more info

# Run COLMAP, this took roughly 3 hours on ~700 images
python scripts/colmap/run_colmap.py --data_root "${data_root}"

# Convert to easymocap dataformat
python scripts/colmap/colmap_to_easyvolcap.py --data_root "${data_root}"
```

Here we detail the process of the using the modified instant-ngp model to finetune the camera parameters on a multi-view dynamic system.
Prepare config files like [`neural3dv.yaml`](../../configs/datasets/neural3dv/neural3dv.yaml) and [`sear_steak.yaml`](../../configs/datasets/neural3dv/sear_steak.yaml).
Then prepare experiment configs like [`l3mhet_sear_steak_static.yaml`](../../configs/exps/l3mhet/l3mhet_sear_steak_static.yaml).

There are generally two cases about the camera setup of a multi-view dynamic system:
1. The provided dataset contains a meaningful background and you want to optimize the camera parameters using the full images.
   Check [`l3mhet_sear_steak_static.yaml`](../../configs/exps/l3mhet/l3mhet_sear_steak_static.yaml) for an example.
```yaml
configs:
    - configs/base.yaml # default arguments for the whole codebase
    - configs/models/l3mhet.yaml # network model configuration
    - configs/datasets/neural3dv/sear_steak.yaml # dataset usage configuration
    - configs/specs/static.yaml
    - configs/specs/optcam.yaml
    - configs/specs/transient.yaml
```
1. The provided dataset comes with masks and you do not want the background to influence the foreground optimization. 
   Check [`l3mhet_0013_01_static.yaml`](../../configs/exps/l3mhet/l3mhet_0013_01_static.yaml) for an example. Add a [`mask.yaml`](../../config/specs/mask.yaml) to the parent list.

Run the ***L3MHET*** model with camera parameter optimization:

```shell
# Train l3mhet on static frame
evc -c configs/exps/l3mhet/l3mhet_${expname}_static.yaml

# Extract camera parameters
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_${expname}_static.yaml

# Check optimized camera
python scripts/tools/visualize_cameras.py --data_root ${data_root}/optimized
```


## Video to Images

Videos should be arranged like this before extracting images from it:

```shell
data/dataset/sequence # data_root & data_root
└── videos # optional, required before extracting images
    ├── 00.mp4
    └── 01.mp4
    ...
```

Extraction command:

```shell
python scripts/preprocess/extract_videos.py --data_root ${data_root}
```

## Segmentation

Masks will be put into the dataset folder with similar structures to the `images`` folder:

```shell
data/dataset/sequence # data_root & data_root
└── masks
    ├── 00 # camera / frame
    │   ├── 000000.jpg # image
    │   ├── 000001.jpg # for dynamic dataset, more images can be placed here
    │   ...
    ├── 01
    ├── 02
    ...
```

Background images will be arranged like (or should be placed as) this:

```shell
data/dataset/sequence # data_root & data_root
└── bkgd # optional
    ├── 00.jpg
    └── 01.jpg
    ...
```

```shell
python scripts/segmentation/inference_robust_video_matting.py --data_root ${data_root}
python scripts/segmentation/extract_backgrounds.py --data_root ${data_root} --masks_dir rvm --dilation 100 --jump 1 --bkgd_dir bkgd_d100_j1
python scripts/segmentation/extract_backgrounds.py --data_root ${data_root} --masks_dir rvm --dilation 20 --jump 1 --bkgd_dir bkgd_d20_j1
python scripts/segmentation/merge_backgrounds.py --data_root ${data_root} --source_dir bkgd_d100_j1 --paste_dir bkgd_d20_j1 --bkgd_dir bkgd
python scripts/segmentation/inference_bkgdmattev2.py --data_root ${data_root}
```

## Space Carving

Visual hulls and processed visual hulls will be arranged like:

```shell
data/dataset/sequence # data_root & data_root
└── vhulls # optional
    ├── 000000.ply
    └── 000001.ply
    ...
```

```shell
data/dataset/sequence # data_root & data_root
└── surfs # optional
    ├── 000000.ply
    └── 000001.ply
    ...
```

Prepare config files like [`renbody.yaml`](../../configs/datasets/renbody/renbody.yaml) and [`0013_01.yaml`](../../configs/datasets/renbody/0013_01.yaml).
Note that we expect the camera parameters to have been properly prepared before extracting visual hulls:
- Converted to ***EasyVolcap*** format
- Optimized using ***L3MHET*** model (maybe also enable `use_aligned_cameras`)
- If previous optimizations extracted visual hulls for some of the frames, remove them before extracting on the whole sequence

Space carving scripts:

```shell
# Remove previous visual hull estimation by static training
rm -r ${data_root}/vhulls

# Extract visual hulls
evc -t test -c configs/base.yaml,configs/models/point_planes.yaml,configs/datasets/renbody/0013_01.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5

# Preprocess visual hulls
evc -t test -c configs/base.yaml,configs/models/point_planes.yaml,configs/datasets/renbody/0013_01.yaml,configs/specs/surfs.yaml

# Extract on optimized cameras

# Extract visual hulls
evc -t test -c configs/base.yaml,configs/models/point_planes.yaml,configs/datasets/renbody/0013_01.yaml,configs/specs/optimized.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5

# Preprocess visual hulls
evc -t test -c configs/base.yaml,configs/models/point_planes.yaml,configs/datasets/renbody/0013_01.yaml,configs/specs/optimized.yaml,configs/specs/surfs.yaml
```