<img src="https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/de41df46-25e6-456c-a253-90d7807b2a9a" alt="logo" width="33%"/>

*****EasyVolcap***: Accelerating Neural Volumetric Video Research**

![python](https://img.shields.io/github/languages/top/zju3dv/EasyVolcap)
![star](https://img.shields.io/github/stars/zju3dv/EasyVolcap)
[![license](https://img.shields.io/badge/license-zju3dv-white)](license)

[Paper](https://dl.acm.org/doi/10.1145/3610543.3626173) | [arXiv](https://arxiv.org/abs/2312.06575) | [Example Dataset](https://drive.google.com/file/d/1XxeO7TnAPvDugnxguEF5Jp89ERS9CAia/view?usp=sharing) | [Pretrained Model](https://drive.google.com/file/d/1OFBFxes9kje02RARFpYpQ6SkmYlulYca/view?usp=sharing) | [4K4D](https://zju3dv.github.io/4k4d)

***News***:

- 24.02.27 [***4K4D***](https://zju3dv.github.io/4k4d) has been accepted to CVPR 2024.
- 23.12.13 ***EasyVolcap*** will be presented at SIGGRAPH Asia 2023, Sydney.
- 23.10.17 [***4K4D***](https://zju3dv.github.io/4k4d), a real-time 4D view synthesis algorithm developed using ***EasyVolcap***, has been made public.

***EasyVolcap*** is a PyTorch library for accelerating neural volumetric video research, particularly in areas of **volumetric video capturing**, reconstruction, and rendering.

https://github.com/zju3dv/EasyVolcap/assets/43734697/14fdfb46-5277-4963-ba75-067ea574c87a

## Installation

Copy-and-paste version of the installation process listed below. For a more thorough explanation, read on.
```shell
# Prepare conda environment
conda install -n base mamba -y -c conda-forge
conda create -n easyvolcap "python>=3.10" -y
conda activate easyvolcap

# Install conda dependencies
mamba env update

# Install pip dependencies
cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip install

# Register EasyVolcp for imports
pip install -e . --no-build-isolation --no-deps
```

We opted to use the latest `pyproject.toml` style packing system for exposing command line interfaces.
It creates a virtual environment for building dependencies by default, which could be quite slow. Disabled with `--no-build-isolation`.
You should create a `conda` or `mamba` (recommended) environment for development, and install the dependencies manually.
If the existing environment with `PyTorch` installed can be utilized, you can jump straight to installing the `pip` dependencies.
More details about installing on *Windows* or compiling *CUDA* modules can be found in [`install.md`](docs/design/install.md).

Note: `pip` dependencies can sometimes fail to install & build. However, not all of them are strictly required for ***EasyVolcap***.
  - The core ones include `tinycudann` and `pytorch3d`. Make sure those are built correctly and you'll be able to use most of the functionality of ***EasyVolcap***.
  - It's also OK to install missing packages manually when ***EasyVolcap*** reports that they are missing since we lazy load a lot of them (`tinycudann`, `diff_gauss`, `open3d` etc.). 
  - Just be sure to check how we listed the missing package in [`requirements.txt`](requirements.txt) before performing `pip install` on them. Some packages require to be installed from GitHub.
  - If the `mamba env update` step fails due to network issues, it is OK to proceed with pip installs since `PyTorch` will also be installed by pip.


## Usage

### Expanding Upon ***EasyVolcap***

Most of the time when we want to build a new set of algorithms on top of the framework, we only have to worry about the actual network itself.
Before writing your new volumetric video algorithm, we need a basic understanding of the network's input and output:

**We use Python dictionaries for passing in and out network input and output.**

1. The `batch` variable stores the network input you sampled from the dataset (e.g. camera parameters).
2. The `output` key of the `batch` variable should contain the network output. For each network module's output definition, please refer to the [design documents](docs/design/main.md) of them (`camera`, `sampler`, `network`, `renderer`) or just see the definitions in [`volumetric_video_model.py`](easyvolcap/models/volumetric_video_model.py) (the `render_rays` function).

<!-- There are generally two ways of developing a new algorithm: -->
**We support purely customized network construction & usage and also a unified NeRF-like pipeline.**

1. If your new network model's structure is similar to NeRF-based ones (i.e. with the separation of `sampler`, `network` and `renderer`), you can simply swap out parts of the [`volumetric_video_network.py`](easyvolcap/models/networks/volumetric_video_network.py) by writing a new config to swap the `type` parameter of the `***_cfg` dictionaries.
2. If you'd like to build a completely new network model: to save you some hassle, we grant the `sampler` classes the ability to directly output the core network output (`rgb_map` stored in `batch.output`). Define your rendering function and network structure however you like and reuse other parts of the codebase. An example: [`gaussiant_sampler.py`](easyvolcap/models/samplers/gaussiant_sampler.py).

**A miminal custom moduling using all other ***EasyVolcap*** components should look something like this:**

```python
from easyvolcap.engine import SAMPLERS
from easyvolcap.utils.net_utils import VolumetricVideoModule
from easyvolcap.utils.console_utils import *

@SAMPLERS.register_module() # make the custom module callable by class name
class CustomVolumetricVideoModule(VolumetricVideoModule):
    def __init__(self,
                 network, # ignore noop_network
                 ... # configurable parameters
                 ):
        # Initialize custom network parameters
        ...
    
    def forward(self, batch: dotdict):
        # Perform network forwarding
        ...

        # Store output for further processing
        batch.output.rgb_map = ... # store rendered image for loss (B, N, 3)
```

In the respective config, selecte this module with:

```yaml
model_cfg:
    sampler_cfg:
        type: CustomVolumetricVideoModule
```

### Importing ***EasyVolcap*** In Other Places

***EasyVolcap*** now supports direct import from other locations & codebases.
After installing, you can not only directly use utility modules and functions from `easyvolcap.utils`, but also import and build upon our core modules and classes.

```python
# Import the logging and debugging functions
from easyvolcap.utils.console_utils import * # log, tqdm, @catch_throw
from easyvolcap.utils.timer_utils import timer  # timer.record
from easyvolcap.utils.data_utils import export_pts, export_mesh, export_npz
...

# Import the OpenGL-based viewer and build upon it
from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer

class CustomViewer(VolumetricVideoViewer):
    ...
```
The import will work when actually running the code, but it might fail since some of the autocompletion modules [is not fully compatible with the newest editable install](https://code.visualstudio.com/docs/python/editing#_importresolvefailure).

If you see warnings when importing ***EasyVolcap*** in your editor like VSCode, you might want to add the path of your ***EasyVolcap*** codebase to the `python.autoComplete.extraPaths` and `python.analysis.extraPaths` like this:

```json
{
    "python.autoComplete.extraPaths": ["/home/zju3dv/code/easyvolcap"],
    "python.analysis.extraPaths": ["/home/zju3dv/code/easyvolcap"]
}
```

Another solution is to replace the installation command of ***EasyVolcap*** with a compatible one [using compatible editable install](https://microsoft.github.io/pyright/#/import-resolution?id=editable-installs):

```shell
pip install -e . --no-build-isolation --no-deps --config-settings editable_mode=compat
```

Note that this is [marked deprecated in the PEP specification](https://setuptools.pypa.io/en/latest/userguide/development_mode.html#legacy-behavior). Thus our recommendation is to change the setting of your editor instead.

### New Project Based on ***EasyVolcap***

If you're interested in developing or researching with ***EasyVolcap***, the recommended way is to fork the repository and modify or append to our source code directly instead of using ***EasyVolcap*** as a module.

After cloning and forking, add [https://github.com/zju3dv/EasyVolcap](https://github.com/zju3dv/EasyVolcap) as an `upstream` if you want to receive updates from our side. Use `git fetch upstream` to pull and merge our updates to ***EasyVolcap*** to your new project if needed. The following code block provides an example of this development process.

Our recent project [4K4D](https://github.com/zju3dv/4K4D) is developed in this fashion.

```shell
# Prepare the name and GitHub repo of your new project
project=4K4D
repo=https://github.com/zju3dv/${project}

# Clone EasyVolcap and add our repo as an upstream
git clone https://github.com/zju3dv/EasyVolcap ${project}

# Setup the remote of your new project
git set-url origin ${repo}

# Add EasyVolcap as an upstream
git remote add upstream https://github.com/zju3dv/EasyVolcap

# If EasyVolcap updates, fetch the updates and maybe merge with it
git fetch upstream
git merge upstream/main
```

Nevertheless, we still encourage you to read on and possibly follow the tutorials in the [Examples](#examples) section and maybe read our design documents in the [Design Docs](#design-docs) section to grasp an understanding of how ***EasyVolcap*** works as a project.

## Examples

In the following sections, we'll show examples of how to run ***EasyVolcap*** on a small multi-view video dataset with several of our implemented algorithms, including Instant-NGP+T, 3DGS+T, and ENeRFi (ENeRF Improved).
In the documentation [`static.md`](docs/misc/static.md), we also provide a complete example of how to prepare the dataset using COLMAP and run the above-mentioned three models using ***EasyVolcap***.

The example dataset for this section can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1XxeO7TnAPvDugnxguEF5Jp89ERS9CAia/view?usp=sharing). After downloading the example dataset, place the unzipped files inside `data/enerf_outdoor` such that you can see files like:
- `data/enerf_outdoor/actor1_4_subseq/images`
- `data/enerf_outdoor/actor1_4_subseq/intri.yml`
- `data/enerf_outdoor/actor1_4_subseq/extri.yml`

This dataset is a small subset of the [ENeRF-Outdoor](https://github.com/zju3dv/ENeRF/blob/master/docs/enerf_outdoor.md) dataset released by our team. For downloading the full dataset, please follow the guide in the [link]((https://github.com/zju3dv/ENeRF/blob/master/docs/enerf_outdoor.md)). 

### Dataset Structure

```shell
data/dataset/sequence # data_root & data_root
├── intri.yml # required: intrinsics
├── extri.yml # required: extrinsics
└── images # required: source images
    ├── 000000 # camera / frame
    │   ├── 000000.jpg # image
    │   ├── 000001.jpg # for dynamic dataset, more images can be placed here
    │   ...
    │   ├── 000298.jpg # for dynamic dataset, more images can be placed here
    │   └── 000299.jpg # for dynamic dataset, more images can be placed here
    ├── 000001
    ├── 000002
    ...
    ├── 000058
    └── 000059
```

***EasyVolcap*** is designed to work on the simplest data form: `images` and no more. The key data preprocessing are done in the `dataloader` and `dataset` modules. These steps are done in the dataloader's initialization
1. We might correct the camera pose with their center of attention and world-up vector (`dataloader_cfg.dataset_cfg.use_aligned_cameras=True`).
2. We undistort read images from the disk using the intrinsic poses and store them as jpeg bytes in memory.

Before running the model, let's first prepare some shell variables for easy access.

```shell
expname=actor1_4_subseq
data_root=data/enerf_outdoor/actor1_4_subseq
```

### Running Instant-NGP+T

We extend Instant-NGP to be time-aware, as a baseline method. With the data preparation completed, we've got an `images` folder and a pair of `intri.yml` and `extri.yml` files, and we can run the l3mhet model.
Note that this model is not built for dynamic scenes, we train it here mainly for extracting initialization point clouds and computing a tighter bounding box.
Similar procedures can be applied to other datasets if such initialization is required.

We need to write a config file for this model
1. Write the data-folder-related stuff inside configs/datasets. Just copy and paste [`configs/datasets/enerf_outdoor/actor1_4_subseq.yaml`](configs/datasets/enerf_outdoor/actor1_4_subseq.yaml) and modify the `data_root` and `bounds` (bounding box), or maybe add a camera near-far threshold.
2. Write the experiment config inside configs/exps. Just copy and paste [`configs/exps/l3mhet/l3mhet_actor1_4_subseq.yaml`](configs/exps/l3mhet/l3mhet_actor1_4_subseq.yaml) and modify the `dataset`-related line in `configs`.

```shell
# With your config files ready, you can run the following command to train the model
evc -c configs/exps/l3mhet/l3mhet_${expname}.yaml

# Now run the following command to render some output
evc -t test -c configs/exps/l3mhet/l3mhet_${expname}.yaml,configs/specs/spiral.yaml
```
[`configs/specs/spiral.yaml`](configs/specs/spiral.yaml): please check this file for more details, it's a collection of configs to tell the dataloader and visualizer to generate a spiral path by interpolating the given cameras


### Running 3DGS+T

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/acd83f13-ba34-449c-96ce-e7b7b0781de4

The original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) uses the sparse reconstruction result of COLMAP for initialization.
However, we found that the sparse reconstruction result often contains a lot of floating points, which is hard to prune for 3DGS and could easily make the model fail to converge.
Thus, we opted to use the "dense" reconstruction result of our Instant-NGP+T implementation by computing the RGBD image for input views and concatenating them as the input of 3DGS. The script [`volume_fusion.py`](scripts/tools/volume_fusion.py) controls this process and it should work similarly on all models that support depth output.

The following script block provides an example of how to prepare an initialization for our 3DGS+T implementation.

```shell
# Extract geometry (point cloud) for initialization from the l3mhet model
# Tune image sample rate and resizing ratio for a denser or sparser estimation
python scripts/tools/volume_fusion.py -- -c configs/exps/l3mhet/l3mhet_${expname}.yaml val_dataloader_cfg.dataset_cfg.ratio=0.15

# Move the rendering results to the dataset folder
source_folder="data/geometry/l3mhet_${expname}/POINT"
destination_folder="${data_root}/vhulls"

# Create the destination directory if it doesn't exist
mkdir -p ${destination_folder}

# Loop through all .ply files in the source directory
for file in ${source_folder}/*.ply; do
    number=$(echo $(basename ${file}) | sed -e 's/frame\([0-9]*\).ply/\1/')
    formatted_number=$(printf "%06d" ${number})
    destination_file="${destination_folder}/${formatted_number}.ply"
    cp ${file} ${destination_file}
done
```

Our conventions for storing initialization point clouds:
- Raw point clouds extracted using Instant-NGP or Space Carving are placed inside the `vhulls` folder. These files might be large. It's OK to directly optimize 3DGS+T on these.
- We might perform some cleanup of the point clouds and store them in the `surfs` folder.
  - For 3DGS+T, the cleaned-up point clouds might be easier to optimize since 3DGS is good at growing details but not so good at dealing with floaters (removing or splitting).
  - For other representations, the cleaned-up point clouds work better than the visual hull (from Space Carving) but might not work so well as the raw point clouds of Instant-NGP.

Then, prepare an experiment config like [`configs/exps/gaussiant/gaussiant_actor1_4_subseq.yaml`](configs/exps/gaussiant/gaussiant_actor1_4_subseq.yaml).
The [`colmap.yaml`](configs/specs/colmap.yaml) provides some heuristics for large-scale static scenes. Remove these if you're not planning on using COLMAP's parameters directly.

```shell
# Train a 3DGS model on the ${expname} dataset
evc -c configs/exps/gaussiant/gaussiant_${expname}.yaml # might run out of VRAM, try reducing densify until iter

# Perform rendering on the trained ${expname} dataset
evc -t test -c configs/exps/gaussiant/gaussiant_${expname}.yaml,configs/specs/superm.yaml,configs/specs/spiral.yaml

# Perform rendering with GUI, do this on a machine with monitor, tested on Windows and Ubuntu
evc -t gui -c configs/exps/gaussiant/gaussiant_${expname}.yaml,configs/specs/superm.yaml
```

The [`superm.yaml`](configs/specs/superm.yaml) skips the loading of input images and other initializations for network-only rendering since all the information we need is contained inside the trained model.

### Inferencing With ENeRFi

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/68401485-85fe-477f-9144-976bb2ee8d3c

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/6d60f2a4-6692-43e8-b682-aa27fcdf9516

The pre-trained model for ENeRFi on the DTU dataset can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1OFBFxes9kje02RARFpYpQ6SkmYlulYca/view?usp=sharing). After downloading, rename the model to `latest.npz` and place it in `data/trained_model/enerfi_dtu`.

```shell
# Render ENeRFi with pretrained model
evc -t test -c configs/exps/enerfi/enerfi_${expname}.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml runner_cfg.visualizer_cfg.save_tag=${expname} exp_name=enerfi_dtu

# Render ENeRFi with GUI
evc -t gui -c configs/exps/enerfi/enerfi_${expname}.yaml exp_name=enerfi_dtu # 2.5 FPS on 3060
```

If more performance is desired:

```shell
# Fine quality, faster rendering
evc -t gui -c configs/exps/enerfi/enerfi_actor1_4_subseq.yaml exp_name=enerfi_dtu model_cfg.sampler_cfg.n_planes=32,8 model_cfg.sampler_cfg.n_samples=4,1 # 3.6 FPS on 3060

# Worst quality, fastest rendering
evc -t gui -c configs/exps/enerfi/enerfi_actor1_4_subseq.yaml,configs/specs/fp16.yaml exp_name=enerfi_dtu model_cfg.sampler_cfg.n_planes=32,8 model_cfg.sampler_cfg.n_samples=4,1 # 5.0 FPS on 3060
```


## Documentations

- [ ] Documentations are still WIP. We'll gradually add more guides and examples, especially regarding the usage of ***EasyVolcap***'s various systems.

### Design Docs

The documentation contained in the [`docs/design`](docs/design) directory contains explanations of design choices and various best practices when developing with ***EasyVolcap***.

[`docs/design/main.md`](docs/design/main.md): Gives an overview of the structure of the ***EasyVolcap*** codebase.

[`docs/design/config.md`](docs/design/config.md): Thoroughly explains the commandline and configuration API of ***EasyVolcap***.

[`docs/design/logging.md`](docs/design/logging.md): Describes the functionalities of the logging system of ***EasyVolcap***.

[`docs/design/dataset.md`](docs/design/dataset.md)

[`docs/design/model.md`](docs/design/model.md)

[`docs/design/runner.md`](docs/design/runner.md)

[`docs/design/viewer.md`](docs/design/viewer.md)

### Project Docs

### Misc Docs

## Acknowledgments

We would like to acknowledge the following inspiring prior work:

- [EasyMocap: Make Human Motion Capture Easier](https://github.com/zju3dv/EasyMocap)
- [XRNeRF: OpenXRLab Neural Radiance Field (NeRF) Toolbox and Benchmark](https://github.com/openxrlab/xrnerf)
- [Nerfstudio: A Modular Framework for Neural Radiance Field Development](https://github.com/nerfstudio-project/nerfstudio)
- [Dear ImGui: Bloat-Free Graphical User Interface for C++ With Minimal Dependencies](https://github.com/ocornut/imgui)
- [Neural Body: Implicit Neural Representations with Structured Latent Codes](https://github.com/zju3dv/neuralbody)
- [ENeRF: Efficient Neural Radiance Fields for Interactive Free-Viewpoint Video](https://github.com/zju3dv/ENeRF)
- [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://github.com/NVlabs/instant-ngp)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting)

## License

***EasyVolcap***'s license can be found [here](license).

Note that the license of the algorithms or other components implemented in ***EasyVolcap*** might be different from the license of ***EasyVolcap*** itself. You will have to install their respective modules to use them in ***EasyVolcap*** following the guide in the [installation section](#installation).
Please refer to their respective licensing terms if you're planning on using them.

## Citation

If you find this code useful for your research, please cite us using the following BibTeX entry. 
If you used our implementation of other methods, please also cite them separately.

```bibtex
@article{xu2023easyvolcap,
  title={EasyVolcap: Accelerating Neural Volumetric Video Research},
  author={Xu, Zhen and Xie, Tao and Peng, Sida and Lin, Haotong and Shuai, Qing and Yu, Zhiyuan and He, Guangzhao and Sun, Jiaming and Bao, Hujun and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia 2023 Technical Communications},
  year={2023}
}

@article{xu20234k4d,
  title={4K4D: Real-Time 4D View Synthesis at 4K Resolution},
  author={Xu, Zhen and Peng, Sida and Lin, Haotong and He, Guangzhao and Sun, Jiaming and Shen, Yujun and Bao, Hujun and Zhou, Xiaowei},
  booktitle={arXiv preprint arXiv:2310.11448},
  year={2023}
}
```
