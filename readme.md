# 4K4D: Real-Time 4D View Synthesis at 4K Resolution

[Paper](https://drive.google.com/file/d/1Y-C6ASIB8ofvcZkyZ_Vp-a2TtbiPw1Yx/view?usp=sharing) | [Project Page](https://zju3dv.github.io/4k4d) | [arXiv](https://arxiv.org/abs/2310.11448) | [Pretrained Models](https://drive.google.com/drive/folders/1mBMsYeXawU_sF3NFyuWC1hnfrYbSfDfi?usp=sharing) | [Minimal Datasets](https://drive.google.com/drive/folders/1pH-SWwbt01raqZ74dvcOvYFxDbGGUcxu?usp=sharing) | [***EasyVolcap***](https://github.com/zju3dv/EasyVolcap)

Repository for our paper *4K4D: Real-Time 4D View Synthesis at 4K Resolution*.

![python](https://img.shields.io/github/languages/top/zju3dv/4K4D)
![star](https://img.shields.io/github/stars/zju3dv/4K4D)
[![license](https://img.shields.io/badge/license-zju3dv-white)](license)

***News***:

- 24.03.27: The [training code](easyvolcap/models/samplers/r4dv_sampler.py) for *4K4D* has been open-sourced along with [documentations](readme.md#Training).
- 24.02.27: *4K4D* has been accepted to CVPR 2024.
- 23.12.18: The backbone of *4K4D*, our volumetric video framework [***EasyVolcap***](https://github.com/zju3dv/EasyVolcap) has been open-sourced!
- 23.12.18: The [inference code](easyvolcap/models/samplers/super_charged_r4dv.py) for *4K4D* has also been open-sourced along with [documentations](readme.md#Rendering).

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/faf60953-fd7f-4309-a1f5-758eaa1182ca

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/b03247d6-07c7-48d7-8e28-2d59ee3c37af

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/379b2793-eda2-4720-a1ad-cbfd2205af4d

**For more high-resolution results and more real-time demos, please visit our [project page](https://zju3dv.github.io/4k4d).**

## Installation

Please refer to the [installation guide of ***EasyVolcap***](https://github.com/zju3dv/EasyVolcap/tree/main/readme.md#installation) for basic environment setup.

After setting up the environment, you should execute the installation command in this repository's root directory to register the modules:

```shell
# Run this inside the 4K4D repo
pip install -e . --no-build-isolation --no-deps
```

Note that it's not necessary for all requirements present in [environment.yml](environment.yml) and [requirements.txt](requirements.txt) to be installed on your system as they contain dependencies for other parts of ***EasyVolcap***. Thanks to the modular design of ***EasyVolcap***, this missing packages will not hinder the rendering and training of *4K4D*.
After the installation process, we're expecting *PyTorch*, *PyTorch3D* and *tiny-cuda-nn* to be present in the current system for the rendering of *4K4D* to work properly.
For the training of *4K4D*, you should also make sure that *Open3D* is properly installed.
Other packages can be easily installed using `pip` if errors about their import are encountered.
Check that this is the case with:

```shell
python -c "from easyvolcap.utils.console_utils import *" # Check for easyvolcap installation. 4K4D is a fork of EasyVolcap
python -c "import torch; print(torch.rand(3,3,device='cuda'))" # Check for pytorch installation
python -c "from pytorch3d.io import load_ply" # Check for pytorch3d installation
python -c "import tinycudann" # Check for tinycudann installation
python -c "import open3d" # Check for open3d installation. open3d is only required for training (extracting visual hulls)
```

## Datasets

In this section, we provide instructions on downloading the full dataset for DNA-Rendering, ZJU-Mocap, NHR, ENeRF-Outdoor and Mobile-Stage dataset.
If you only want to preview the pretrained models in the interactive GUI without any need for training, we recommend checking out the [minimal dataset](#rendering-with-minimal-dataset-only-encoded-videos) section because the full datasets are quite large in size.
Note that for full quality rendering, you still need to download the full dataset as per the instructions below.
<!-- Rendering with the encoded video dataset will lead to almost no visual quality loss -->

*4K4D* follows the typical dataset setup of ***EasyVolcap***, where we group similar sequences into sub-directories of a particular dataset.
Inside those sequences, the directory structure should generally remain the same:
For example, after downloading and preparing the *0013_01* sequence of the *DNA-Rendering* dataset, the directory structure should look like this:

```shell
# data/renbody/0013_01:
# Required:
images # raw images, cameras inside: images/00, images/01 ...
masks # foreground masks, cameras inside: masks/00, masks/01 ...
extri.yml # extrinsic camera parameters, not required if the optimized folder is present
intri.yml # intrinsic camera parameters, not required if the optimized folder is present

# Optional:
optimized # OPTIONAL: optimized camera parameters: optimized/extri.yml, optimized: intri.yml
vhulls # OPTIONAL: extracted visual hull: vhulls/000000.ply, vhulls/000001.ply ... not required if the optimized folder and surfs folder are present
surfs # OPTIONAL: processed visual hull: surfs/000000.ply, surfs/000001.ply ...
```

### DNA-Rendering, NHR and ZJU-MoCap Datasets

Please refer to [*Im4D*'s guide](https://github.com/zju3dv/im4d#set-up-datasets) to download ZJU-MoCap, NHR and DNA-Rendering datasets.
After downloading, the extracted files should be placed in to `data/my_zjumocap`, `data/NHR` and `data/renbody` respectively.
If someone is interested in the processed data, please email me at [zhenx@zju.edu.cn](mailto://zhenx@zju.edu.cn) and CC [xwzhou@zju.edu.cn](xwzhou@zju.edu.cn) and [pengsida@zju.edu.cn](pengsida@zju.edu.cn) to request the processing guide.
For ZJU-MoCap, you can fill in [this Google form](https://docs.google.com/forms/d/e/1FAIpQLSdmXJkGB8E4DoSZ80VyB9vfnIJZeaGf8NS2nkLozHSU1MaQzA/viewform?usp=sf_link) to request the download link.
Note that you should cite the corresponding papers if you use these datasets.

### ENeRF-Outdoor Dataset

If someone is interested in downloading the ENeRF-Outdoor dataset, please fill in [this Google form](https://docs.google.com/forms/d/e/1FAIpQLSeFAgcnJEDYEaco4vA2QsD-bM8V8CRFuiR0QWbi_3uXmXtUKg/viewform?usp=sf_link) to request the download link. Note that this dataset is for non-commercial use only.
After downloading, the extracted files should be placed in `data/enerf_outdoor`.

### Mobile-Stage Dataset

If someone is interested in downloading the Mobile-Stage dataset, please fill in [this Google form](https://docs.google.com/forms/d/e/1FAIpQLSeEbjuTV7w0lfryl-9FPX1VteuPGkbqjDvxXebY02Tm6BMejQ/viewform?usp=sf_link) to request the download link. Note that this dataset is for non-commercial use only.
After downloading, the extracted files should be placed in `data/mobile_stage`.

## Rendering

First, download the [pretrained models](https://drive.google.com/drive/folders/1mBMsYeXawU_sF3NFyuWC1hnfrYbSfDfi?usp=sharing).

After downloading, place them into `data/trained_model` (e.g. `data/trained_model/4k4d_0013_01/1599.npz`, `data/trained_model/4k4d_0013_01_r4/latest.pt` and `data/trained_model/4k4d_0013_01_mb/-1.npz`).

Note: The pre-trained models were created with the release codebase. This code base has been cleaned up and includes bug fixes, hence the metrics you get from evaluating them will differ from those in the paper.
If yor're interested in reproducing the error metrics reported in the paper, please consider downloading the [reference images](https://drive.google.com/file/d/1xES9EoH7DwPaMcHbL8_g4HqCm7S1jIS2/view?usp=sharing).

Here we provide their naming convensions which corresponds to their respective config files:

1. `4k4d_0013_01` (without any postfixes) is the real-time 4K4D model, corresponding to `configs/projects/realtime4dv/rendering/4k4d_0013_01.yaml`. This model can only be used for rendering. When combined with the full dataset mentioned above, this is the full official *4K4D* implementation.
2. `4k4d_0013_01_r4` (with the `_r4` postfix) is the full pretrained model used during training, corresponding to `configs/projects/realtime4dv/training/4k4d_0013_01_r4.yaml`. This model can only be used for training. `r4` is short for *realtime4dv*.
3. `4k4d_0013_01_mb` (with the `_mb` postfix) is [an extension to 4K4D](https://github.com/dendenxu/Web4K4D) (Note: to be open-sourced) where we distill the IBR + SH appearance model into a set of low-degree SH parameters. This model can only be used for rendering and do not require pre-computation. `mb` is short for *mobile*.

### Rendering of Trained Model

After placing the models and datasets in their respective places, you can run ***EasyVolcap*** with configs located in [configs/projects/realtime4dv/rendering](configs/projects/realtime4dv/rendering) to perform rendering operations with *4K4D*.

For example, to render the *0013_01* sequence of the *DNA-Rendering* dataset, you can run:

```shell
# GUI Rendering
evc -t gui -c configs/projects/realtime4dv/rendering/4k4d_0013_01.yaml,configs/specs/vf0.yaml # Only load, precompute and render the first frame
evc -t gui -c configs/projects/realtime4dv/rendering/4k4d_0013_01.yaml # Precompute and render all 150 frames, this could take a minute or two

# Testing with input views
evc -t test -c configs/projects/realtime4dv/rendering/4k4d_0013_01.yaml,configs/specs/eval.yaml,configs/specs/vf0.yaml # Only render some of the view of the first frame
evc -t test -c configs/projects/realtime4dv/rendering/4k4d_0013_01.yaml,configs/specs/eval.yaml # Only rendering some selected testing views and frames

# Rendering rotating novel views
evc -t test -c configs/projects/realtime4dv/rendering/4k4d_0013_01.yaml,configs/specs/eval.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml,configs/specs/vf0.yaml # Render a static rotating novel view
evc -t test -c configs/projects/realtime4dv/rendering/4k4d_0013_01.yaml,configs/specs/eval.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml # Render a dynamic rotating novel view
```

### Rendering With Minimal Dataset (Only Encoded Videos)

We provide a minimal dataset for 4K4D to render with its full pipeline by encoding the input images and masks into videos **(typically less than 100MiB each)**.

This leads to almost no visual quality loss, but if you have access to the full dataset, it's recommended to run the model on the full dataset instead (Sec. [**Rendering**](#rendering)).

Here we provide instructions on setting up the minimal dataset and rendering with it:

1. Download the [pretrained models](https://drive.google.com/drive/folders/1mBMsYeXawU_sF3NFyuWC1hnfrYbSfDfi?usp=sharing). If you've already done so for the [**Rendering**](#rendering) section, this step can be skipped.
   1. Pretrained models should be placed directly into the `data/trained_model` directory (e.g. `data/trained_model/4k4d_0013_01/1599.npz`).
2. Downlaod the [minimal datasets](https://drive.google.com/drive/folders/1pH-SWwbt01raqZ74dvcOvYFxDbGGUcxu?usp=sharing). 
   1. Place the compressed files inside their respective `data_root` (e.g. `0013_01_libx265.tar.gz` should be placed into `data/renbody/0013_01`) and uncompressed them. 
   2. Note that if you've already downloaded the full dataset with raw images as per the [**Rendering**](#rendering) section, there's no need for redownloading the minimal dataset with encoded videos. 
   3. However, if you continue, no files should be replaced and you can safely run both kinds of rendering.
   4. After the uncompression, you should see two folders: `videos_libx265` and `optimized`. The former contains the encoded videos, and the latter contains the optionally optimized camera parameters. For some dataset, you'll see `intri.yml` and `extri.yml` instead of the `optimized` folder. And for some others, you'll see a `videos_masks_libx265` for storing the masks separatedly.
3. Process the minimal datasets using these two scripts:
   1. [scripts/realtime4dv/extract_images.py](scripts/realtime4dv/extract_images.py): Extract images from the encoded videos. Use `--data_root` to control which dataset to extract.
   2. [scripts/realtime4dv/extract_masks.py](scripts/realtime4dv/extract_masks.py): Extract masks from the encoded videos. Use `--data_root` to control which dataset to extract.
   3. After the extraction (preprocessing), you should see a `images_libx265` and a `masks_libx265` inside your `data_root`.
    Example processing scripts:

```shell
# For foreground datasets with masks and masked images (DNA-Rendering, NHR, ZJU-Mocap)
python scripts/realtime4dv/extract_images.py --data_root data/renbody/0013_01 --vcodec none --hwaccel none
python scripts/realtime4dv/extract_masks.py --data_root data/renbody/0013_01 --vcodec none --hwaccel none

# For datasets with masks and full images (ENeRF-Outdoor and dance3 of MobileStage)
python scripts/realtime4dv/extract_images.py --data_root data/mobile_stage/dance3 --vcodec none --hwaccel none
python scripts/realtime4dv/extract_images.py --data_root data/mobile_stage/dance3 --vcodec none --hwaccel none --videos_dir videos_masks_libx265 --images_dir masks_libx265 --single_channel
```

4. Now, the minimal dataset has been prepared and you can render a model with it. The only change is to append a new config onto the command: `configs/specs/video.yaml`.
    Example rendering scripts:

```shell
# See configs/projects/realtime4dv/rendering for more
evc -t gui -c configs/projects/realtime4dv/rendering/4k4d_0013_01.yaml,configs/specs/video.yaml
evc -t gui -c configs/projects/realtime4dv/rendering/4k4d_sport1.yaml,configs/specs/video.yaml
evc -t gui -c configs/projects/realtime4dv/rendering/4k4d_my_313.yaml,configs/specs/video.yaml
evc -t gui -c configs/projects/realtime4dv/rendering/4k4d_dance3.yaml,configs/specs/video.yaml
evc -t gui -c configs/projects/realtime4dv/rendering/4k4d_actor1_4.yaml,configs/specs/video.yaml
```

### Rendering Without Dataset (Mobile 4K4D)

- [ ] TODO: Finish up the web viewer for the mobile 4k4d.


## Training

- [ ] TODO: Finish up the training doc for 4k4d.

### Pretrained Models for Training

- [ ] TODO: Finish up the training doc for 4k4d.

### Training on DNA-Rendering (ZJU-MoCap and NHR)

Training *4K4D* on the prepared dataset is simple, just one command and you're all set.
However, it's recommended to train a single-frame version of *4K4D* to identify potential issues (5 mins) before running on the full sequence (24 hours).
If any missing pacakges are reporteds, please `pip install` them before going any further.
Note that if you've requested the prepared dataset, I should have already prepared the visual hulls and optimized camera parameters for you.
If they are not present in the dataset, you should first check the [Initialization section of the Custom Dataset section](#initialization-space-carving--visual-hull) for instructions on how to prepare the visual hulls and optimize the camera parameters (if needed of course).

You can use this script to test the installation and training process of *4K4D*.
This script trains a single-frame version of *4K4D* on the first frame of the *0013_01* sequence of the *DNA-Rendering* dataset.

```shell
evc -c configs/exps/4k4d/4k4d_0013_01_r4.yaml,configs/specs/static.yaml,configs/specs/tiny.yaml exp_name=4k4d_0013_01_r4_static
```

During the first 100-200 iterations, you should see that the training PSNR increase to 24-25 dB. Otherwise there might be bugs during your dataset preparation or installation process.

The actual training of the full model is more straight forward:

```shell
evc -c configs/exps/4k4d/4k4d_0013_01_r4.yaml
```

For training on other sequences or dataset, change this line:

```yaml
- configs/datasets/renbody/0013_01_obj.yaml # dataset usage configuration
```

to something else like:

```yaml
- configs/datasets/NHR/sport2_obj.yaml # dataset usage configuration
```

and save the modified config file separatedly to something like: `configs/exps/4k4d/4k4d_sport2_r4.yaml`

Example configurations files can be found in `configs/exps/4k4d` and `configs/projects/realtime4dv/training`. 
The `4k4d` folder contains some more example configuration files. If you're starting from scratch, it's recommended to use and extend files inside this folder.
The `training` folder contains configuration files matching our provided pretrained model, some of which are legacy model files that requires special configurations.
For (a) visual hull initialization and (b) converting the trained models for real-time rendering, please see the [Custom Datasets](#custom-datasets) section.

### Training on ENeRF-Outdoor

- [ ] TODO: Finish up the training doc for 4k4d.

Training *4K4D* on a background dataset (*dance3* or *ENeRF-Outdoor*) is a little bit more involved than the foreground-only ones.
For the foreground-only ones, you only need to train one model and maybe later convert them for realtime rendering.
For the background-enabled ones, you'll need to train a separated model for the background and foreground and then jointly optimize them.
Note that if you've requested the processed data, I should have already placed the processed visual hulls and background initializations inside your dataset.
If they're not present, please check the [Initialization section of the Custom Dataset section](#initialization-space-carving--visual-hull) for instructions.

## Custom Datasets

- [ ] TODO: Finish up the training doc for 4k4d.

### Dataset Preparation

The *4K4D* project is an effort around creating a real-time-renderable neural volumetric video.

In the following, we'll be walking throught the process of training our method on a custom multi-view dataset. 
Lets call the dataset `renbody` and call the sequence `0013_01` for notation. Note that you can change out the *0013_01* and *renbody* parts for other names for you custom dataset. Other namings like *4k4d* should remain the same.
Let's assume a typical input contains calibrated camera parameters compatible with [EasyMocap](https://github.com/zju3dv/EasyMocap), where the folder & directory structure looks like this:

```shell
data/renbody/0013_01
│── extri.yml
│── intri.yml
├── images
│   ├── 00
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   ...
│   │   ...
│   └── 01
│   ...
└── masks
    ├── 00
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    │   ...
    │   ...
    └── 01
    ...
```

We assume the foreground mask has already been segmented. If not, you could also checkout the scripts contained in [`scripts/segmentation`](../../scripts/segmentation/inference_robust_video_matting.py). My experience is that [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference.md) typically gets the job done. If you've also got access to ground truth background images (with minimal lighting changes), you could also give [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) a try. A relevant script can also be found in [`scripts/segmentation`](../../scripts/segmentation/inference_bkgdmattev2.py).

### Configurations

Given the dataset, you're now prepared for creating your correcponding configuration file for *4K4D*.
The first file is corresponding to the dataset itself, where data loading paths and input ratios or view numbers are defined. Lets put it in [`configs/datasets/renbody/0013_01.yaml`](../../configs/datasets/renbody/0013_01.yaml). You can look at the actual file to get a grasp of what info this file should contain. At the minimum, you should specify the data loading root for the dataset. If you feel unfamilier with the configuration system, feel free to check out the specific [documentation](../../docs/design/config.md) for that part. The content of the `0013_01.yaml` (and its parent `renbody.yaml`) file should look something like this:

```yaml
# Content of configs/datasets/renbody/0013_01.yaml
configs: configs/datasets/renbody/renbody.yaml

dataloader_cfg: # we see the term "dataloader" as one word?
    dataset_cfg: &dataset_cfg
        data_root: data/renbody/0013_01
        images_dir: images_calib

val_dataloader_cfg:
    dataset_cfg:
        <<: *dataset_cfg
```
```yaml
# Content of configs/datasets/renbody/renbody.yaml
dataloader_cfg: # we see the term "dataloader" as one word?
    dataset_cfg: &dataset_cfg
        masks_dir: masks
        ratio: 0.5
        bounds: [[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]] # thinner?

        force_sparse_view: True
        view_sample: [0, 60, 1]
        frame_sample: [0, 150, 1] # only train for a thousand frames

model_cfg:
    sampler_cfg:
        bg_brightness: 0.0
    renderer_cfg:
        bg_brightness: 0.0

val_dataloader_cfg:
    dataset_cfg:
        <<: *dataset_cfg
        frame_sample: [0, 150, 1]
    sampler_cfg:
        view_sample: [0, 60, 20]

```

Here you'll see I created a general description file `configs/datasets/renbody/renbody.yaml` for the whole *DNA-Rendering* dataset, which is a good practice if your multi-view dataset contains multiple different sequences but they are under roughly the same setting (view count, lighting condition, mask quality etc.). You'll also note I explicitly specified how many views and frames this dataset has. The number you put in here should not exceed the actual amount. If you're feeling lazy you can also just write `[0, null, 1]` for `view_sample` and `frame_sample`, however doing this means a trained model will still require access to the original dataset to perform some loading and rendering.

Until now, such data preparation are generalizable across all multi-view dataset supported by ***EasyVolcap***, you should always create the corresponding dataset configuraitons for you custom ones as this helps in reproducibility.

Our next step is to create the corresponding *4K4D* configuration for running experiments on the `0013_01` sequence. You can create a [`configs/exps/4k4d/4k4d_0013_01_r4.yaml`](../../configs/exps/4k4d/4k4d_0013_01_r4.yaml) to hold such information:

```yaml
configs:
    - configs/base.yaml # default arguments for the whole codebase
    - configs/models/r4dv.yaml # network model configuration
    - configs/datasets/renbody/0013_01_obj.yaml # dataset usage configuration
    - configs/specs/mask.yaml # specific usage configuration
    - configs/specs/optimized.yaml # specific usage configuration

# prettier-ignore
exp_name: {{fileBasenameNoExtension}}
```

You'll notice I placed the configurations in and order of `base`, `model`, `dataset` and then `specs`. This is typically the best practice as you get more and more specific about the experiment you want to perform here. The `mask.yaml` file will provide necessary configurations for reading, parsing and using provided foreground masks, and it is essential for all experiments on *4K4D*. 

For some datasets (like `4k4d_sport2_r4.yaml`), you'll notice I included a spec called `prior.yaml`. The `prior.yaml` file slightly modifies the optimizaion hyperparameters (increase weight of input masks and increate input view count) since the mask of *NHR* is quite accurate and the focal length is quite high. You can also check out the actual file to get a better understanding of what it does (this applies to almost all files under the [`specs`](../../configs/specs) directory).

For now, there's no `0013_01_obj.yaml`. This file should contain a tighter bounding box aggregated from the visual hulls and we will describe how to prepare this in the next section.

### Initialization (Space Carving || Visual Hull)

The next step for *4K4D* is initialization. With the correct dataset placement and configuration files in place, you should be able to run the following script to extract the visull hull from input foreground masks:

```shell
# Extract visual hulls
evc -t test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/renbody/0013_01.yaml,configs/specs/optimized.yaml,configs/specs/vhulls.yaml

# Preprocess visual hulls
evc -t test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/renbody/0013_01.yaml,configs/specs/optimized.yaml,configs/specs/surfs.yaml
```

This won't take long. Typically a few minutes should suffices. After which you'll notice a folder `vhulls` created in your dataset folder (`data/renbody/renbody` in this example). Note that we also support directly running the training script and let ***EasyVolcap*** lazily take care of this initialization process, but since *4K4D* could benefit from a tighter bounding box when defininig its 4D feature volumes (two modified [K-Planes](https://github.com/sarafridov/K-Planes)), I figure one extra hoop should be worth it.

- [ ] TODO: Make this bbox summarization process automatic

You‘ll also notice the above script will output an aggregated bounding box value. Copy it and create a new file [`configs/datasets/renbody/0013_01_obj.yaml`](../../configs/datasets/renbody/0013_01_obj.yaml) to hold it.

```yaml
configs: configs/datasets/renbody/0013_03.yaml

dataloader_cfg: &dataloader_cfg
    dataset_cfg: &dataset_cfg # ratio: 0.5
        bounds: [[-0.5152, -0.6097, -0.9667], [0.5948, 0.8103, 0.7933]] # !: BATCH
        vhull_thresh: 0.90

val_dataloader_cfg:
    dataset_cfg:
        <<: *dataset_cfg
```

And modify your [`configs/exps/4k4d/4k4d_0013_01_r4.yaml`](../../configs/exps/4k4d/4k4d_0013_01_r4.yaml) to replace `0013_01.yaml` with `0013_01_obj.yaml`. This will make sure we use the tightest bound when defining the dimensions of our 4D feature volumes (two modified [K-Planes](https://github.com/sarafridov/K-Planes)).

### Training

Next, we're ready to perform the actual training:

```shell
evc -c configs/exps/4k4d/4k4d_0013_01_r4.yaml
```

This will take a day two depending on your machine. Please pay attention to the console logs and keep and eye out for the loss and metrics. All records and training time evalution will be saved to `data/record` and `data/result` respectively. So launch your tensorboard or other viewing tools for training inspection.
After training, we'd want to perform post processing on the trained model for various uses.

### Rendering

Firstly, we want to generate the version that supports realtime rendering while maintaining almost no performance loss (dubbed *super charged*):

```shell
python scripts/realtime4dv/charger.py --sampler SuperChargedR4DV --exp_name 4k4d_0013_01 -- -c configs/exps/4k4d/4k4d_0013_01_r4.yaml,configs/specs/super.yaml
```

Now you can render the converted model in the interactive ***EasyVolcap*** viewer with:

```shell
evc -t gui -c configs/exps/4k4d/4k4d_0013_01_r4.yaml,configs/specs/superf.yaml exp_name=4k4d_0013_01
```

The preloading of all frames is quite heavy on system memory and also time intensive. So we advise controlling the number of frames to load via:

```shell
evc -t gui -c configs/exps/4k4d/4k4d_0013_01_r4.yaml,configs/specs/superf.yaml,configs/specs/vf0.yaml exp_name=4k4d_0013_01
```

`0` is the starting frame, `1` is the ending frame id and `1` indicates how many frames to jump. For example, `0,30,5` will load 6 frames (`0,5,10,15,20,25`). You can also use `None` to indicate the default value for the total. For example, `val_dataloader_cfg.dataset_cfg.frame_sample=0,None,1` will load all frames.
Note that this variant is the default variant used in the paper.

### Optimizing the Cameras

Inheriting from ***EasyVolcap***, *4K4D* also supports loading optimized camera parameters for each dataset (sequence).
This is done by training an improved Instant-NGP model on one frame of the sequence. And then extracting the optimized camera parameters to the ***EasyVolcap*** format.
However, if your dataset's provided camera parameters are already good enough, you can skip this step.
For the NHR dataset, the provided cameras are good enough thus we did not mention this step in the previous sections.
So, from now on, we will be using the *0013_01* sequence of the *DNA-Rendering* dataset as an example.

More details on how to perform the camera parameters finetune can be found in [camera.md](../design/camera.md).
For now, let's assume you've got a `l3mhet_0013_01_static.yaml` file setup and ready for use.

After training the single frame camera optimization model using this script:

```shell
evc -c configs/specs/exps/l3mhet/l3mhet_0013_01_static.yaml
```

You're expected to finded a model `data/trained_model/l3mhet_0013_01_static.yaml`

- [ ] TODO: Continues this.


## Custom Full-Scene Datasets

Prepare the dynamic dataset as in [`custom_dataset.md`](../../docs/misc/custom_dataset.md).
Specifically, you should follow the guide in [the dense calibration section](../../docs/misc/custom_dataset.md#dense-calibration) to prepare the optimized camera parameters using NGP (`optimized`).
And then, you can follow the guide in [the space carving section](../../docs/misc/custom_dataset.md#space-carving) to prepare the visual hulls (`vhulls` and `surfs`).
After that, we can continue on the training of custom full-scene dataset.

First of all, rearrange the background images folder and train a background NGP:

```shell
# Prepare dataset variables
expname=actor1_4
data_root=data/enerf_outdoor/actor1_4

# Rearrange background images
python scripts/segmentation/link_backgrounds.py --data_root ${data_root}

# Train background NGP
evc -c configs/exps/l3mhet/l3mhet_${expname}_bkgd.yaml

# Extract point clouds from trained background NGP
python scripts/fusion/volume_fusion.py -- -c configs/exps/l3mhet/l3mhet_${expname}_bkgd.yaml val_dataloader_cfg.dataset_cfg.ratio=0.15 val_dataloader_cfg.dataset_cfg.view_sample=0,None,3 # 50W should be ok

# Prepare for initialization of bg 4k4d
mkdir -p ${data_root}/bkgd/boost
cp data/geometry/l3mhet_${expname}_bkgd/POINT/frame0000.ply ${data_root}/bkgd/boost/000000.ply
```

Prepare 3 configs like:

- [`4k4d_actor1_4_r4_bg.yaml`](../../configs/exps/4k4d/4k4d_actor1_4_r4_bg.yaml): Background 4K4D training config
- [`4k4d_actor1_4_r4_fg.yaml`](../../configs/exps/4k4d/4k4d_actor1_4_r4_fg.yaml): Foreground 4K4D training config
- [`4k4d_actor1_4_r4.yaml`](../../configs/exps/4k4d/4k4d_actor1_4_r4.yaml): Joint 4K4D training config

Optionally, create static version of the `fg` model to validate the implementation like [`4k4d_actor1_4_r4_fg_static.yaml`](../../configs/exps/4k4d/4k4d_actor1_4_r4_fg_static.yaml):

```shell
# Train static foreground model
evc -c configs/exps/4k4d/4k4d_${expname}_r4_fg_static.yaml
```

And optionally, validate the rendering of the static frame:

```shell
# Convert to real-time format
python scripts/realtime4dv/charger.py --exp_name 4k4d_${expname}_fg_static --sampler SuperChargedR4DV -- -c configs/exps/4k4d/4k4d_${expname}_r4_fg_static.yaml,configs/specs/super.yaml

# Real-time rendering in GUI
evc -t gui -c configs/exps/4k4d/4k4d_${expname}_r4_fg_static.yaml,configs/specs/superf.yaml exp_name=4k4d_${expname}_fg_static
```

Train the `bg` and `fg` model seperatedly:

```shell
# Train background model
evc -c configs/exps/4k4d/4k4d_${expname}_r4_bg.yaml

# Train foreground model, this could take a long time
evc -c configs/exps/4k4d/4k4d_${expname}_r4_fg.yaml

# Joint training
evc -c configs/exps/4k4d/4k4d_${expname}_r4.yaml
```

Real-time rendering of backgrounds:

```shell
# Convert to real-time format
python scripts/realtime4dv/charger.py --exp_name 4k4d_${expname}_bg --sampler SuperChargedR4DV -- -c configs/exps/4k4d/4k4d_${expname}_r4_bg.yaml,configs/specs/super.yaml

# Non-real-time rendering
evc -t test -c configs/exps/4k4d/4k4d_${expname}_r4_bg.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml val_dataloader_cfg.dataset_cfg.focal_ratio=0.65 val_dataloader_cfg.dataset_cfg.n_render_views=600

# Real-time rendering of spiral path
evc -t test -c configs/exps/4k4d/4k4d_${expname}_r4_bg.yaml,configs/specs/superf.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml,configs/specs/eval.yaml exp_name=4k4d_${expname}_bg val_dataloader_cfg.dataset_cfg.focal_ratio=0.65 val_dataloader_cfg.dataset_cfg.n_render_views=600

# Real-time rendering in GUI
evc -t gui -c configs/exps/4k4d/4k4d_${expname}_r4_bg.yaml,configs/specs/superf.yaml exp_name=4k4d_${expname}_bg
```

Real-time rendering of foregrounds:

```shell
# Convert to real-time format
python scripts/realtime4dv/charger.py --exp_name 4k4d_${expname}_fg --sampler SuperChargedR4DV -- -c configs/exps/4k4d/4k4d_${expname}_r4_fg.yaml,configs/specs/super.yaml

# Non-real-time rendering
evc -t test -c configs/exps/4k4d/4k4d_${expname}_r4_fg.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml val_dataloader_cfg.dataset_cfg.focal_ratio=0.65 val_dataloader_cfg.dataset_cfg.n_render_views=600

# Real-time rendering of spiral path
evc -t test -c configs/exps/4k4d/4k4d_${expname}_r4_fg.yaml,configs/specs/superf.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml,configs/specs/eval.yaml exp_name=4k4d_${expname}_fg val_dataloader_cfg.dataset_cfg.focal_ratio=0.65 val_dataloader_cfg.dataset_cfg.n_render_views=600

# Real-time rendering in GUI
evc -t gui -c configs/exps/4k4d/4k4d_${expname}_r4_fg.yaml,configs/specs/superf.yaml exp_name=4k4d_${expname}_fg
```

Real-time rendering of joint results:

```shell
# Convert to real-time format
python scripts/realtime4dv/charger.py --exp_name 4k4d_${expname} --sampler SuperChargedR4DV -- -c configs/exps/4k4d/4k4d_${expname}_r4.yaml,configs/specs/super.yaml

# Non-real-time rendering
evc -t test -c configs/exps/4k4d/4k4d_${expname}_r4.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml val_dataloader_cfg.dataset_cfg.focal_ratio=0.65 val_dataloader_cfg.dataset_cfg.n_render_views=600

# Real-time rendering of spiral path
evc -t test -c configs/exps/4k4d/4k4d_${expname}_r4.yaml,configs/specs/superf.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml,configs/specs/eval.yaml exp_name=4k4d_${expname} val_dataloader_cfg.dataset_cfg.focal_ratio=0.65 val_dataloader_cfg.dataset_cfg.n_render_views=600

# Real-time rendering in GUI
evc -t gui -c configs/exps/4k4d/4k4d_${expname}_r4.yaml,configs/specs/superf.yaml exp_name=4k4d_${expname}
```

Using static scene to check implementation and avoid OOM errors:

```shell
# Run training on first frame
evc -c configs/exps/4k4d/4k4d_${expname}_r4_static.yaml

# Run rendering on first frame of non-real-time model
evc -t test -c configs/exps/4k4d/4k4d_${expname}_r4_static.yaml

# Convert to real-time model
python scripts/realtime4dv/charger.py --exp_name 4k4d_${expname}_static --sampler SuperChargedR4DVB -- -c configs/exps/4k4d/4k4d_${expname}_r4_static.yaml,configs/specs/super.yaml

# Run rendering with real-time model
evc -t test -c configs/exps/4k4d/4k4d_${expname}_r4_static.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml,configs/specs/superb.yaml,configs/specs/eval.yaml exp_name=4k4d_${expname}_static

# Run GUI rendering with real-time model
evc -t gui -c configs/exps/4k4d/4k4d_${expname}_r4_static.yaml,configs/specs/superb.yaml exp_name=4k4d_${expname}_static
```

## Acknowledgements

We would like to acknowledge the following inspiring prior work:

- [EasyVolcap: Accelerating Neural Volumetric Video Research](https://github.com/zju3dv/EasyVolcap) (Xu et al.)
- [IBRNet: Learning Multi-View Image-Based Rendering](https://ibrnet.github.io/) (Wang et al.)
- [ENeRF: Efficient Neural Radiance Fields for Interactive Free-viewpoint Video](https://zju3dv.github.io/enerf) (Lin et al.)
- [K-Planes: Explicit Radiance Fields in Space, Time, and Appearance](https://sarafridov.github.io/K-Planes/) (Fridovich-Keil et al.)

## Citation

If you find this code useful for your research, please cite us using the following BibTeX entry.

```bibtex
@inproceedings{xu20234k4d,
  title={4K4D: Real-Time 4D View Synthesis at 4K Resolution},
  author={Xu, Zhen and Peng, Sida and Lin, Haotong and He, Guangzhao and Sun, Jiaming and Shen, Yujun and Bao, Hujun and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2024}
}

@article{xu2023easyvolcap,
  title={EasyVolcap: Accelerating Neural Volumetric Video Research},
  author={Xu, Zhen and Xie, Tao and Peng, Sida and Lin, Haotong and Shuai, Qing and Yu, Zhiyuan and He, Guangzhao and Sun, Jiaming and Bao, Hujun and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia 2023 Technical Communications},
  year={2023}
}
```
****
