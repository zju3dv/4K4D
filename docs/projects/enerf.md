# ENeRF: Efficient Neural Radiance Fields for Interactive Free-viewpoint Video

[Paper](https://arxiv.org/abs/2112.01517) | [Project Page](https://zju3dv.github.io/enerf) | [Original Code](https://github.com/zju3dv/ENeRF) | [Data](https://github.com/zju3dv/ENeRF/blob/master/docs/enerf_outdoor.md)

This is an official implementation of paper *ENeRF: Efficient Neural Radiance Fields for Interactive Free-viewpoint Video* in ***EasyVolcap*** Codebase.

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/ef021cb0-7860-4816-8d07-92fd764cdf34

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/11325dfc-801d-4142-906c-9fd3797176e3

## Installation

Please refer to the [installation guide of ***EasyVolcap***](../../readme.md#installation) for basic environment setup.

Note that it's not necessary for all requirements present in [environment.yml](../../environment.yml) and [requirements.txt](../../requirements.txt) to be installed on your system as they contain dependencies for other parts of ***EasyVolcap***. Thanks to the modular design of ***EasyVolcap***, these missing packages will not hinder the rendering and training of *ENeRF*.

After the installation process, we're expecting *PyTorch* to be present in the current system for the rendering of *ENeRF* to work properly. Other packages can be easily installed using `pip` if errors about their import are encountered. Check that this is the case with:

```shell
python -c "from easyvolcap.utils.console_utils import *" # Check for easyvolcap installation. 4K4D is a fork of EasyVolcap
python -c "import torch; print(torch.rand(3, 3, device='cuda'))" # Check for pytorch installation
```


## Datasets

In this section, we provide instructions on downloading the full dataset for DTU, DNA-Rendering, ZJU-Mocap, NHR, ENeRF-Outdoor, and Mobile-Stage dataset.
<!--If you only want to preview the pretrained models in the interactive GUI without any need for training, we recommend checking out the [minimal dataset](#rendering-with-minimal-dataset-only-encoded-videos) section because the full datasets are quite large in size.
Note that for full quality rendering, you still need to download the full dataset as per the instructions below. -->

*ENeRF* follows the typical dataset setup of ***EasyVolcap***, where we group similar sequences into sub-directories of a particular dataset. Inside those sequences, the directory structure should generally remain the same. To be more specific, ***EasyVolcap*** supports two types of datasets: (a) **multi-view dataset** which is captured by a synchronized multi-view camera rig, and (b) **monocular dataset** which is captured by a single moving camera. Both of them are supported by ***EasyVolcap***, and the difference between them is that the organization of the camera parameters and images sub-directories.

For example, the *ZJU-Mocap* dataset is a multi-view dataset, while we re-organize the *DTU* dataset as a monocular dataset, their directory structures should look like this:
```shell
# Multi-view dataset: data/my_zjumocap/my_313
# Required:
images # raw images, cameras inside: images/00, images/01 ...
    00
        000000.jpg
        000001.jpg
        ...
    01
        000000.jpg
        000001.jpg
        ...
    ...
extri.yml # extrinsic camera parameters in the data root, not required if the optimized folder is present
intri.yml # intrinsic camera parameters in the data root, not required if the optimized folder is present

# Monocular dataset: data/dtu/scan1
# Required:
images/00 # raw images, cameras inside only in images/00
    000000.jpg
    000001.jpg
    ...
cameras/00 # contains both extrinsic and intrinsic camera parameters
    extri.yml
    intri.yml

# Optional:
masks # OPTIONAL: foreground masks, cameras inside: masks/00, ...
optimized # OPTIONAL: optimized camera parameters: optimized/extri.yml, optimized: intri.yml
vhulls # OPTIONAL: extracted visual hull: vhulls/000000.ply, vhulls/000001.ply ... not required if the optimized folder and surfs folder are present
surfs # OPTIONAL: processed visual hull: surfs/000000.ply, surfs/000001.ply ...
```

### DTU Dataset

Follow Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) from original [MVSNet repo](https://github.com/YoYo000/MVSNet) and unzip. Once downloaded, use the [script](../../scripts/preprocess/dtu_to_easyvolcap.py) we provide to organize the data into the required format:
```shell
python3 scripts/preprocess/dtu_to_easyvolcap.py --dtu_root ${dtu_root} --easyvolcap_root data/dtu
```
where `$dtu_root` is the path to the downloaded raw DTU dataset. After processing, the extracted files should be placed in `data/dtu`. Here we provide a [preprocessed scan](https://drive.google.com/file/d/12wPt_uVS1EuoIRit4A3rUgYFtui6da-t/view) of the DTU dataset used by ***EasyVolcap***.

### DNA-Rendering, ZJU-Mocap, NHR, ENeRF-Outdoor and Mobile-Stage Datasets

For these datasets, please refer to the [4K4D Datasets](./realtime4dv.md#datasets) section for instructions on downloading and organizing the datasets.
Note that you should cite the corresponding papers if you use these datasets.


## Rendering

### Rendering of Pretrained Model

We provide [dtu pretrained model](https://drive.google.com/file/d/1i-5lydVohmm1u9YN6qVfM6Ivm2LZD5tc/view?usp=sharing) for you. After downloading, place them into `data/trained_model` (`data/trained_model/enerf_dtu/latest.pt` to be more specific).

After placing the trained models and datasets in their respective places, you can run ***EasyVolcap*** with configs located in [configs/exps/enerf](../../configs/exps/enerf) to perform rendering operations with *ENeRF*.

For example, to render the *my_313* sequence of the *ZJU-Mocap* dataset, you can run:

```shell
# GUI Rendering
evc -t gui -c configs/exps/enerf/enerf_my_313_static.yaml exp_name=enerf_dtu # Render the first frame of the sequence
evc -t gui -c configs/exps/enerf/enerf_my_313.yaml exp_name=enerf_dtu # Render the whole sequence

# Testing with input views
evc -t test -c configs/exps/enerf/enerf_my_313_static.yaml,configs/specs/eval.yaml exp_name=enerf_dtu runner_cfg.visualizer_cfg.save_tag=enerf_my_313_static # Only render some of the view of the first frame
evc -t test -c configs/exps/enerf/enerf_my_313.yaml,configs/specs/eval.yaml exp_name=enerf_dtu runner_cfg.visualizer_cfg.save_tag=enerf_my_313 # Render some selected testing views and frames

# Rendering rotating novel views
evc -t test -c configs/exps/enerf/enerf_my_313_static.yaml,configs/specs/eval.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml exp_name=enerf_dtu runner_cfg.visualizer_cfg.save_tag=enerf_my_313_static # Render a static rotating novel view
evc -t test -c configs/exps/enerf/enerf_my_313.yaml,configs/specs/eval.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml val_dataloader_cfg.dataset_cfg.frame_sample=0,200,1 exp_name=enerf_dtu runner_cfg.visualizer_cfg.save_tag=enerf_my_313 # Render a dynamic rotating novel view of 200 frames sequence
```

#### Useful CLI Configurable Arguments

There are many CLI arguments you can use to customize the rendering process (both test and gui), here we list some of them, for example, to render the *0013_01* sequence of the *DNA-Rendering* dataset using the [dtu pretrained model](https://drive.google.com/file/d/1i-5lydVohmm1u9YN6qVfM6Ivm2LZD5tc/view?usp=sharing) with different configurations, you can run:

```shell
# Render ENeRF with GUI on 1/4 resolution of dna-rendering dataset using only 1/2 of the views and 3 source views for each target view, choice 1
evc -t gui -c configs/exps/enerf/enerf_0013_01.yaml exp_name=enerf_dtu val_dataloader_cfg.dataset_cfg.cache_raw=False val_dataloader_cfg.dataset_cfg.ratio=0.25 val_dataloader_cfg.dataset_cfg.src_view_sample=0,None,2 val_dataloader_cfg.dataset_cfg.n_srcs_list=3, val_dataloader_cfg.dataset_cfg.use_vhulls=True model_cfg.sampler_cfg.cache_size=10 # only cache ten images

# Render ENeRF with GUI on 1/4 resolution of dna-rendering dataset using only 1/2 of the views and 3 source views for each target view, choice 2
evc -t gui -c configs/exps/enerf/enerf_0013_01.yaml exp_name=enerf_dtu val_dataloader_cfg.dataset_cfg.cache_raw=False val_dataloader_cfg.dataset_cfg.ratio=0.25 val_dataloader_cfg.dataset_cfg.view_sample=0,None,2 val_dataloader_cfg.dataset_cfg.force_sparse_view=True val_dataloader_cfg.dataset_cfg.n_srcs_list=3, val_dataloader_cfg.dataset_cfg.use_vhulls=True model_cfg.sampler_cfg.cache_size=10 # only cache ten images

# Render ENeRF with GUI on full resolution of dna-rendering dataset using only all views as source views and 3 source views for each target view, using fp16, higher performance, worse results on some views
evc -t gui -c configs/exps/enerf/enerf_0013_01.yaml,configs/specs/fp16.yaml exp_name=enerf_dtu val_dataloader_cfg.dataset_cfg.cache_raw=False val_dataloader_cfg.dataset_cfg.ratio=1.0 val_dataloader_cfg.dataset_cfg.n_srcs_list=3, val_dataloader_cfg.dataset_cfg.use_vhulls=True model_cfg.sampler_cfg.cache_size=10 # only cache ten images

# Render ENeRF on resolution 768,1336 of dna-rendering dataset with rotating paths
evc -t test -c configs/exps/enerf/enerf_0013_01.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml exp_name=enerf_dtu val_dataloader_cfg.dataset_cfg.cache_raw=False val_dataloader_cfg.dataset_cfg.ratio=1.0 val_dataloader_cfg.dataset_cfg.n_srcs_list=3, val_dataloader_cfg.dataset_cfg.use_vhulls=True model_cfg.sampler_cfg.cache_size=10 val_dataloader_cfg.dataset_cfg.render_size=768,1366 val_dataloader_cfg.dataset_cfg.frame_sample=0,150,1 val_dataloader_cfg.dataset_cfg.n_render_views=150 runner_cfg.visualizer_cfg.save_tag=spiral

# Render ENeRF on dna-rendering dataset with a pre-generated path (stored in data/paths/0013_01/*.yml)
evc -t test -c configs/exps/enerf/enerf_0013_01.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml exp_name=enerf_dtu val_dataloader_cfg.dataset_cfg.render_size=-1,-1 val_dataloader_cfg.dataset_cfg.save_interp_path=False val_dataloader_cfg.dataset_cfg.camera_path_intri=data/paths/0013_01/intri.yml val_dataloader_cfg.dataset_cfg.camera_path_extri=data/paths/0013_01/extri.yml val_dataloader_cfg.dataset_cfg.frame_sample=0,16,1 val_dataloader_cfg.dataset_cfg.n_render_views=16 val_dataloader_cfg.dataset_cfg.interp_type=NONE val_dataloader_cfg.dataset_cfg.interp_cfg.smoothing_term=-1.0 val_dataloader_cfg.dataset_cfg.use_vhulls=True model_cfg.sampler_cfg.cache_size=10 runner_cfg.visualizer_cfg.save_tag=path
```

- `val_dataloader_cfg.dataset_cfg.cache_raw`: whether to cache the raw images in memory as bytes object or just as `UnstructuredTensors`, you can change it to `True` to cache the raw images in memory as bytes object if you have enough memory;

- `val_dataloader_cfg.dataset_cfg.ratio`: controls the resolution of the rendered images, you can change it to `0.25` to render at 1/4 resolution if you have limited CUDA memory;

- `val_dataloader_cfg.dataset_cfg.n_srcs_list`: this controls the number of source views that will be used for each target view rendering, you can change it to `3,` to use 3 source views for each target view rendering;

- `val_dataloader_cfg.dataset_cfg.src_view_sample`: this controls the set of views that will be used as source views for target view rendering, you can change it to `0,None,2` to use only 1/2 of the views for target view rendering or to `0,2,4,6,8,10,12` to use only these specific views as source views;

- `val_dataloader_cfg.sampler_cfg.view_sample`: this controls the set of views that will be seen by the model during training or testing, this is useful when you want to train or test on a subset of the views, you can change it to `0,None,2` to use only 1/2 of the views for training or testing or to `0,2,4,6,8,10,12` to use only these specific views for training or testing;

- `val_dataloader_cfg.dataset_cfg.view_sample`: by default, this is set to `0,None,1`, which means the dataset will see all views, and you can control the set of source views you want to use for target view rendering or the set of views the model will see during training or testing by changing the two arguments above respectively, but if you want to achieve these two requirements at the same time, you can change this argument to `0,None,2` to constriain the model to see and use only 1/2 of the views;
  - `val_dataloader_cfg.dataset_cfg.force_sparse_view`: do not forget to set this to `True` if you modifies the `val_dataloader_cfg.dataset_cfg.view_sample` to a sparse view sampling pattern, otherwise there will be an assertion error;

- `model_cfg.sampler_cfg.cache_size`: this is a acceleration trick we use to make testing or rendering faster by cache the images and its corresponding features in CUDA memory, you can change it to `10` to cache only 10 images in memory if you have limited CUDA memory;

- `val_dataloader_cfg.dataset_cfg.render_size`: sets the resolution of the rendered images, you can change it to `768,1366` to render at 768x1366 resolution;

Note that `val_dataloader_cfg.dataset_cfg.n_render_views` should be set to the exact number of views in the path you generated, and `val_dataloader_cfg.dataset_cfg.camera_path_intri` and `val_dataloader_cfg.dataset_cfg.camera_path_extri` should be set to the path of the intri.yml and extri.yml of the path you generated. I strongly recommend you to follow every parameter configuration I list in the last example above if you want to render with a pre-generated path.


### Rendering of Config Specific Model

Please follow the instructions we provide in the [training](#training) section to train a config-specific model from scratch or from a pretrained model. After that, you can run ***EasyVolcap*** with configs located in [configs/exps/enerf](../../configs/exps/enerf) to perform rendering operations with *ENeRF*.

The difference between the commands listed in the [Rendering of DTU Pretrained Model](#rendering-of-dtu-pretrained-model) is that you do not need to explicitly assign `exp_name` since it is the same with the config name by default (remember to assign the `exp_name` to the one you set when training if you do not use the default one).

For example, to render the *actor1* sequence of the *ENeRF-Outdoor* dataset, you can run:
```shell
# GUI Rendering
evc -t gui -c configs/exps/enerf/enerf_actor1.yaml,configs/specs/static.yaml # Render the first frame of the sequence
evc -t gui -c configs/exps/enerf/enerf_actor1.yaml # Render the whole sequence

# Testing with input views
evc -t test -c configs/exps/enerf/enerf_actor1.yaml,configs/specs/static.yaml,configs/specs/eval.yaml # Only render some of the view of the first frame
evc -t test -c configs/exps/enerf/enerf_actor1.yaml,configs/specs/eval.yaml # Render some selected testing views and frames

# Rendering rotating novel views
evc -t test -c configs/exps/enerf/enerf_actor1.yaml,configs/specs/eval.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml # Render a dynamic rotating novel view
# Rendering cubic interpolating novel views
evc -t test -c configs/exps/enerf/enerf_actor1.yaml,configs/specs/eval.yaml,configs/specs/cubic.yaml,configs/specs/ibr.yaml # Render a dynamic cubic interpolating novel view
```


## Training

### Training from Scratch

Training *ENeRF* on the prepared dataset is simple from scratch, just one command and you're all set. If any missing pacakges are reporteds and causes the stop of the training process (sometimes, some warnings is fine), please `pip install` them before going any further.

You can use this script to test the installation and training process of *ENeRF*. This script trains a single-frame version of *ENeRF* on the first frame of the *actor1* sequence of the *ENeRF-Outdoor* dataset.

However, the same as [training guide of *4K4D*](./realtime4dv.md#training-on-dna-rendering-zju-mocap-and-nhr), it's recommended to spend like 5 minutes to train a single-frame version of *ENeRF* to identify potential issues before running on the full sequence (24 hours).

```shell
evc -c configs/exps/enerf/enerf_actor1.yaml,configs/specs/static.yaml exp_name=check
```

During the first 100-200 iterations, you should see that the training PSNR increase to 23-24 dB. Otherwise there might be bugs during your dataset preparation or installation process.

The actual training of the full model is more straight forward:

```shell
evc -c configs/exps/enerf/enerf_actor1.yaml
```

For training on other sequences or dataset, change this line `configs/exps/enerf/enerf_actor1.yaml` to something else like `configs/exps/enerf/enerf_my_394.yaml` and save the modified config file separatedly under the `configs/exps/enerf` directory.

Example configurations files of *ENeRF* can be found in `configs/exps/enerf`, which contains some more example configuration files. If you're starting from scratch, it's recommended to use and extend files inside this folder.

### Training from Pretrained Model

To train *ENeRF* from a pretrained model, eg. our provided [DTU pretrained model](https://drive.google.com/drive/folders/1VGXr0gx_W2yl_gwCU_cKmStSOqO_0z1o?usp=sharing), or a model you trained with other datasets, you first need to manually download or copy the pretrained model to `data/trained_model/$exp_name` where `$exp_name` is the name of the model you want to train (not the name of the pretrain model).

For example, if you want to train *ENeRF* on *actor1* of *ENeRF-Outdoor* dataset from the provided DTU pretrained model, you should download the model and place it in `data/trained_model/enerf_dtu_actor1` (here we set `$exp_name=enerf_dtu_actor1`), then, you can run the following command to train *ENeRF* from the pretrained model:

```shell
evc -c configs/exps/enerf/enerf_actor1.yaml exp_name=enerf_dtu_actor1 runner_cfg.resume=True runner_cfg.epochs=2000 # 2000 = 1600(pretrained epochs) + 400(finetuned epochs)
```

The finetuned model will be saved in `data/trained_model/enerf_dtu_actor1/latest.pt` after training, and you can follow the [Rendering](#rendering-of-pretrained-model) section to render the finetuned model, remember to change the `exp_name` to `enerf_dtu_actor1`.


## Custom Datasets

If you want to train *ENeRF* on your own dataset, you need to prepare the dataset in the format required by ***EasyVolcap***.

You can follow the instructions we provide in the [4K4D Custom Datasets](./realtime4dv.md#custom-datasets) section to prepare the dataset. Note that there is no need to prepare the foreground masks and visual hulls for *ENeRF*.

```shell
# Visualize camera
python scripts/tools/visualize_cameras.py --data_root ${data_root}

# Composable experiments
evc -t gui -c configs/base.yaml,configs/models/enerfi.yaml,configs/datasets/volcano/skateboard.yaml,configs/specs/mask.yaml exp_name=enerfi_dtu val_dataloader_cfg.dataset_cfg.frame_sample=0,1,1
```


## Acknowledgements

We would like to acknowledge the following inspring prior work:

- [ENeRF: Efficient Neural Radiance Fields for Interactive Free-viewpoint Video](https://zju3dv.github.io/enerf) (Lin et al.)
- [IBRNet: Learning Multi-View Image-Based Rendering](https://ibrnet.github.io/) (Wang et al.)


## Citation

If you find this code useful for your research, please cite us using the following BibTeX entry.

```bibtex
@inproceedings{lin2022efficient,
  title={Efficient Neural Radiance Fields for Interactive Free-viewpoint Video},
  author={Lin, Haotong and Peng, Sida and Xu, Zhen and Yan, Yunzhi and Shuai, Qing and Bao, Hujun and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2022}
}

@article{xu2023easyvolcap,
  title={EasyVolcap: Accelerating Neural Volumetric Video Research},
  author={Xu, Zhen and Xie, Tao and Peng, Sida and Lin, Haotong and Shuai, Qing and Yu, Zhiyuan and He, Guangzhao and Sun, Jiaming and Bao, Hujun and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia 2023 Technical Communications},
  year={2023}
}
```
