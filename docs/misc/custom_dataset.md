# Preparing Custom Datasets For Dynamic Volumetric Videos

## Dataset Structure

Basic dataset structure:

```shell
data/dataset/sequence # data_root & datadir
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

Dataset variables:

```shell
# Linux
expname=skateboard
datadir=data/volcano/skateboard

# Another dataset
expname=wy02
datadir=data/sensetime4d/wy02

# Windows
$expname="skateboard"
$datadir="data/volcano/skateboard"
```

## Camera Synchronization

- [ ] TODO: Board

## Static Calibration

Please refer to [static.md](../../docs/misc/static.md#colmap-for-camera-poses).

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

After installation, first, you need to capture the `ground1f` and `background1f` images using your sparse set of cameras as described in the [wild + multiple sparse](https://chingswy.github.io/easymocap-public-doc/quickstart/calibration/wild-sparse.html), where `background1f` is a set of static background only images captured by your multi-view cameras, and `ground1f` is the same set of background with a chessboard inside captured by your multi-view cameras which is used to align the scale and center of the colmap calibrated world coordinate system to the real world scale and to the place of the chessboard. For example, if you have a sparse set of webcam, `eg.` logitech webcam, you can perform the capture by:

```shell
cd ~/EasyMocap

# Capture the pure background, remember to change the camera config file to your own if you have more 6 webcams
python3 apps/camera/realtime_display.py --cfg_cam config/camera/usb-logitech.yml --display --num 1 --out ./data/background1f

# Capture the background with the chessboard in the scene, put the chessboard to the place where you want the world center to be
python3 apps/camera/realtime_display.py --cfg_cam config/camera/usb-logitech.yml --display --num 1 --out ./data/ground1f
```

Then, you can use you cell iphone to record a video alongside your set of cameras, you can see how it goes in the example provided in [wild + multiple sparse](https://chingswy.github.io/easymocap-public-doc/quickstart/calibration/wild-sparse.html), and load the video to the `~/EasyMocap/data/background1f/` directory. After all the data is ready, you can run the following command to perform the sparse calibration using the [script](../../scripts/colmap/sparse_calibration.sh):

```shell
cd ~/easyvolcap
source scripts/colmap/sparse_calibration.sh
CUDA_VISIBLE_DEVICES=0 calibration 
```

Remember to change the paths of your `EasyMocap` and `scene_root` if you put it somewhere different than the default path. After the calibration is done, you can find the **EasyVolcap** required format camera parameters `extri.yml` and `intri.yml` under the `colmap/align/static/images` sub-directory of your `scene_root` directory. Create the symbolic link to any dataset root directory that are captured by the same set of cameras, and you can use the camera parameters for the dataset.

## Dense Calibration

```shell
# Prepare colmap ply model
mkdir -p ${datadir}/colmap/colmap_ply
colmap model_converter --input_path ${datadir}/colmap/colmap_text --output_path ${datadir}/colmap/colmap_ply/000000.ply --output_type PLY

# Use bbox of ply for configs
# colmap gui ...

# Convert to easyvolcap camera format
python scripts/colmap/colmap_to_easymocap.py --data_root ${datadir} --colmap colmap/colmap_text

# Check camera
python scripts/tools/visualize_cameras.py --data_root ${datadir}
```

Prepare config files like [`volcano.yaml`](../../configs/datasets/volcano/volcano.yaml) and [`skateboard.yaml`](../../configs/datasets/volcano/skateboard.yaml).

Then prepare experiment configs like [`l3mhet_skateboard_static.yaml`](../../configs/exps/l3mhet/l3mhet_skateboard_static.yaml).

Two cases:
1. The provided dataset contains meaningful background and you want to optimize the camera parameters using the full images.
   Check [`l3mhet_skateboard_static.yaml`](../../configs/exps/l3mhet/l3mhet_skateboard_static.yaml) for an example.
```yaml
configs:
    - configs/base.yaml # default arguments for the whole codebase
    - configs/models/l3mhet.yaml # network model configuration
    - configs/datasets/volcano/skateboard.yaml # dataset usage configuration
    - configs/specs/static.yaml
    - configs/specs/optcam.yaml
    - configs/specs/transient.yaml
```
2. The provided dataset comes with masks and you do not want the background to influence the foreground optimization. 
   Check [`l3mhet_wy02_static.yaml`](../../configs/exps/l3mhet/l3mhet_wy02_static.yaml) for an example. Add a [`mask.yaml`](../../config/specs/mask.yaml) to the parent list.

Run the ***L3MHET*** model with camera parameter optimization:

```shell
# Train l3mhet on static frame
evc -c configs/exps/l3mhet/l3mhet_${expname}_static.yaml

# Extract camera parameters
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_${expname}_static.yaml

# Check optimized camera
python scripts/tools/visualize_cameras.py --data_root ${datadir}/optimized
```


## Video to Images

Videos should be arranged like this before extracting images from it:

```shell
data/dataset/sequence # data_root & datadir
└── videos # optional, required before extracting images
    ├── 00.mp4
    └── 01.mp4
    ...
```

Extraction command:

```shell
python scripts/preprocess/extract_videos.py --data_root ${datadir}
```

## Segmentation

Masks will be put into the dataset folder with similar structures of the `images` folder:

```shell
data/dataset/sequence # data_root & datadir
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
data/dataset/sequence # data_root & datadir
└── bkgd # optional
    ├── 00.jpg
    └── 01.jpg
    ...
```

```shell
python scripts/segmentation/inference_robust_video_matting.py --data_root ${datadir}
python scripts/segmentation/extract_backgrounds.py --data_root ${datadir} --masks_dir rvm --dilation 100 --jump 1 --bkgd_dir bkgd_d100_j1
python scripts/segmentation/extract_backgrounds.py --data_root ${datadir} --masks_dir rvm --dilation 20 --jump 1 --bkgd_dir bkgd_d20_j1
python scripts/segmentation/merge_backgrounds.py --data_root ${datadir} --source_dir bkgd_d100_j1 --paste_dir bkgd_d20_j1 --bkgd_dir bkgd
python scripts/segmentation/inference_bkgdmattev2.py --data_root ${datadir}
```

## Space Carving

Visual hulls and processed visual hulls will be arranged like:

```shell
data/dataset/sequence # data_root & datadir
└── vhulls # optional
    ├── 000000.ply
    └── 000001.ply
    ...
```

```shell
data/dataset/sequence # data_root & datadir
└── surfs # optional
    ├── 000000.ply
    └── 000001.ply
    ...
```

Prepare config files like [`volcano.yaml`](../../configs/datasets/volcano/volcano.yaml) and [`skateboard.yaml`](../../configs/datasets/volcano/skateboard.yaml).
Note that we expect the camera parameters to have been properly prepared before extracting visual hulls:
- Converted to ***EasyVolcap*** format
- Optimized using ***L3MHET*** model (maybe also enable `use_aligned_cameras`)
- If previous optimzations extracted visual hulls for some of the frames, remove them before extracting on the whole sequence

Space carving scripts:

```shell
# Remove previous visual hull estimation by static training
rm -r ${datadir}/vhulls

# Extract visual hulls
evc -t test -c configs/base.yaml,configs/models/point_planes.yaml,configs/datasets/volcano/skateboard.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5

# Preprocess visual hulls
evc -t test -c configs/base.yaml,configs/models/point_planes.yaml,configs/datasets/volcano/skateboard.yaml,configs/specs/surfs.yaml

# Extract on optimized cameras

# Extract visual hulls
evc -t test -c configs/base.yaml,configs/models/point_planes.yaml,configs/datasets/sensetime4d/wy02.yaml,configs/specs/optimized.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5

# Preprocess visual hulls
evc -t test -c configs/base.yaml,configs/models/point_planes.yaml,configs/datasets/sensetime4d/wy02.yaml,configs/specs/optimized.yaml,configs/specs/surfs.yaml
```