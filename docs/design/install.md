# Installation Guide

## Conda & Pip Dependencies

Here, we provide a more thorough explanation of the installation process listed in the [main readme](../../readme.md#installation).
Basically, what we does is create a separated conda environment, install `PyTorch` from conda and then install other dependencies one-by-one using `pip`.
The reason for the `cat` command is that `pip install -r requirements.txt` will fail even if only one of the dependencies failed to install and it quickly became infuriating to see an error message eating away 30 minutes of our lives. 

```shell
# Clone this codebase with:
git clone https://github.com/zju3dv/EasyVolcap
cd EasyVolcap

# If you haven't installed conda, 
# We recommend installing it from https://docs.conda.io/projects/miniconda/en/latest/
# On Ubuntu, these scripts can be used
# cd ~
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# chmod +x Miniconda3-latest-Linux-x86_64.sh
# ./Miniconda3-latest-Linux-x86_64.sh # and finish the installation as the guide

# Maybe install mamba through conda 
# The recommended way is to use mambaforge directly instead of installing miniconda3
conda install -n base mamba -c conda-forge -y

# Picks up on environment.yml and create a pytorch env
# mamba env create

# If this doesn't work, separate create and update as follows
# Note that as of 2023.04.21, wsl2 with python 3.10 could not correctly initialize or load opengl
# You need manually change the following command to accomodate for that (3.10 to 3.9)
# And update the environtment.yaml file to use python 3.9 instead of 3.10
# mamba create -n easyvolcap "python==3.10" -y
# mamba create -n easyvolcap "python>=3.10,<3.10" -y
mamba create -n easyvolcap "python>=3.10" -y
conda activate easyvolcap
mamba env update # picks up environment.yml

# With the environment created, install other dependencies 
# And possibly modify .zshrc to automatically activate this env
# echo conda activate easyvolcap >> ~/.zshrc
# source ~/.zshrc # actually not needed (already activated)

# Install all pip dependencies one by one in case some fails
# pip dependencies might fail to install, check the root cause and try this script again
# Possible caveats: cuda version? pytorch version? python version?
cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip install 

# Registers command-line interfaces like `evc` `evc-train` `evc-test` etc.
pip install -e . --no-build-isolation --no-deps
```
<!-- 
```shell
# To add autocompletion
# For zsh
echo "" >> ~/.zshrc
echo "###### EasyVolumetricVideo AutoCompletion Start ######" >> ~/.zshrc
echo "$(which shtab) --shell=zsh -u easyvolcap.utils.import_utils.prepare_shtab_parser | tee $PWD/data/zsh-functions/_evv > /dev/null" >> ~/.zshrc
echo "export fpath=($PWD/data/zsh-functions \$fpath)" >> ~/.zshrc
echo "autoload -Uz compaudit && compinit" >> ~/.zshrc
echo "###### EasyVolumetricVideo AutoCompletion End ######" >> ~/.zshrc
echo "" >> ~/.zshrc
echo "Please reload your shell for auto completion to work"
``` -->

Note that the provided examples in the [main readme](../../readme.md#examples) uses an HDR video, thus you might also need to install a version of `ffmpeg` that supports the `zscale` filter.
<!-- Note that the provided video `output.mp4` is in HDR, thus we need to nstall FFmpeg with zscale filter support for extracting images: -->
```shell
sudo add-apt-repository ppa:savoury1/ffmpeg4
sudo apt update
sudo apt install ffmpeg
```

## CUDA Related Compilations

To use any compiled CUDA modules, you need to have the CUDA Toolkit installed and configured.

Typically your system administration would have already done so if you're using a shared server for AI realted research. Check under `/usr/local` to find anything related to CUDA.

Then, add these lines to your `.zshrc` or `.bashrc` to expose related paths for compilation:
```shell
# CUDA related configs
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda"
export CUDA_DEVICE_ORDER=PCI_BUS_ID # OPTIONAL: defaults to capability order, might be different for GL and CUDA
```

Let's go through the compilation process with the `tinycudann` package.

To retain the compiled objects without starting over in case anything fails, we recommend first cloning `tinycudann` then perform the compilation and installation manually:

```shell
cd ~
git clone https://github.com/NVlabs/tiny-cuda-nn --recursive
cd tiny-cuda-nn/bindings/torch
python setup.py install
```

Or maybe try to install `tiny-cuda-nn` with:

```shell
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

This is automatically done when installing dependencies for ***EasyVolcap*** using this command above:

```shell
# Install all pip dependencies one by one in case some fails
cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip install
```

If you still encounter `cannot find -lcuda: No such file or directory` error after setting the above paths, try executing

```shell
# ln -s /usr/lib/x86_64-linux-gnu/libcuda.so ~/anaconda3/envs/easyvolcap/lib/libcuda.so
ln -s /usr/lib/x86_64-linux-gnu/libcuda.so ~/miniconda3/envs/easyvolcap/lib/libcuda.so
```

and then install `tiny-cuda-nn` with: `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`.

Sometimes (`diff-gaussian-rasterization`), on some systems, the import order of CUDA-compiled extension and PyTorch matters.
On some of our system, if you import PyTorch before the CUDA-compiled extension, it will fail to find some Python symbols.
Try changing the orders if that happens.

```shell
python -c "from diff_gaussian_rasterization import _C" # works
python -c "import torch; from diff_gaussian_rasterization import _C" # might fail
python -c "from diff_gaussian_rasterization import _C; import torch" # should work
```

## Windows

As discussed in [this issue](https://github.com/zju3dv/EasyVolcap/issues/10), the installation commands are tailored for Linux systems but ***EasyVolcap***'s dependency requirements are very loose. Aside from `PyTorch`, which you can install in anyway you like following [their official guide](https://pytorch.org/get-started/locally/) or just reuse your previous environment, also other packages are related to the specific functionality or algorithm you want to run.

During the initialization in the `main.py` script, we will try to recursively import user defined components from the `easyvolcap` folder (also check [`config.py`](../design/config.md#reusing-the-configuration-system)). Some warnings about missing packages might be raised, but as long as you don't use that functionality, it should be fine.

For example, the real-time viewer does not require packages like `pytorch3d` or `open3d` so you could ignore them during import.
We've also tested the viewer functionality (run `evc-gui` to test) on Windows, which requires the following OpenGL and ImGUI related packages:

```
glfw
PyGLM
pyperclip
pyopengl
imgui-bundle
opencv-python
cuda-python
pdbr
```

These packages should be able to be installed directly from the command-line using `pip`.

Sidenote: on macOS (Mac), there's no NVIDIA gpu. We use the CUDA-GL interface to transfer rendered images (a PyTorch tensor, which is a block of CUDA memory) onto the screen (a textured based framebuffer). Thus the real-time viewer isn't fully supported on Mac yet.