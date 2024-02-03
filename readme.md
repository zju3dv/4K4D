# 4K4D: Real-Time 4D View Synthesis at 4K Resolution

[Paper](https://drive.google.com/file/d/1Y-C6ASIB8ofvcZkyZ_Vp-a2TtbiPw1Yx/view?usp=sharing) | [Project Page](https://zju3dv.github.io/4k4d) | [arXiv](https://arxiv.org/abs/2310.11448) | [Pretrained Models](https://drive.google.com/drive/folders/1mBMsYeXawU_sF3NFyuWC1hnfrYbSfDfi?usp=sharing) | [Minimal Datasets](https://drive.google.com/drive/folders/1pH-SWwbt01raqZ74dvcOvYFxDbGGUcxu?usp=sharing) | [***EasyVolcap***](https://github.com/zju3dv/EasyVolcap)

Repository for our paper *4K4D: Real-Time 4D View Synthesis at 4K Resolution*.

![python](https://img.shields.io/github/languages/top/zju3dv/4K4D)
![star](https://img.shields.io/github/stars/zju3dv/4K4D)
[![license](https://img.shields.io/badge/license-zju3dv-white)](license)

***News***:

- 23.12.18: The backbone of *4K4D*, our volumetric video framework [***EasyVolcap***](https://github.com/zju3dv/EasyVolcap) has been open-sourced!
- 23.12.18: The inference code for *4K4D* has also been open-sourced along with documentations.

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

- [ ] TODO: Add trainable models & training examples 

### Pretrained Models for Training

### Training on DNA-Rendering (ZJU-MoCap and NHR)

### Training on ENeRF-Outdoor


## Custom Datasets

- [ ] TODO: Add trainable models & examples on custom datasets

### Dataset Preparation

### Configurations

### Initialization (Space Carving || Visual Hull)

### Training

### Rendering

### Optimizing the Cameras


## Custom Full-Scene Datasets

- [ ] TODO: Add trainable models & examples on custom full-scene datasets


## Acknowledgements

We would like to acknowledge the following inspiring prior work:

- [EasyVolcap: Accelerating Neural Volumetric Video Research](https://zju3dv.github.io/easyvolcap) (Xu et al.)
- [IBRNet: Learning Multi-View Image-Based Rendering](https://ibrnet.github.io/) (Wang et al.)
- [ENeRF: Efficient Neural Radiance Fields for Interactive Free-viewpoint Video](https://zju3dv.github.io/enerf) (Lin et al.)
- [K-Planes: Explicit Radiance Fields in Space, Time, and Appearance](https://sarafridov.github.io/K-Planes/) (Fridovich-Keil et al.)

## Citation

If you find this code useful for your research, please cite us using the following BibTeX entry.

```bibtex
@article{xu20234k4d,
  title={4K4D: Real-Time 4D View Synthesis at 4K Resolution},
  author={Xu, Zhen and Peng, Sida and Lin, Haotong and He, Guangzhao and Sun, Jiaming and Shen, Yujun and Bao, Hujun and Zhou, Xiaowei},
  booktitle={arXiv preprint arXiv:2310.11448},
  year={2023}
}

@article{xu2023easyvolcap,
  title={EasyVolcap: Accelerating Neural Volumetric Video Research},
  author={Xu, Zhen and Xie, Tao and Peng, Sida and Lin, Haotong and Shuai, Qing and Yu, Zhiyuan and He, Guangzhao and Sun, Jiaming and Bao, Hujun and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia 2023 Technical Communications},
  year={2023}
}
```
