# CUDA-PyTorch Interop

## Debugging Nasty CUDA Errors

```python
# Saving dump file and making sure the file is fully written
with open('fwd.dump', 'wb') as f:
    torch.save(dotdict(xyz3=xyz3, rgb3=rgb3, cov6=cov6, occ1=occ1, camera=camera, batch=batch), f)
    os.fsync(f)
```

```shell
# Using cuda-gdb to debug the CUDA error
cuda-gdb -ex run --args python easyvolcap/scripts/main.py -c configs/exps/fdgs/fdgs_0319_f0.yaml runner_cfg.log_interval=1

# https://stackoverflow.com/questions/75973717/where-is-cuda-memcheck
# https://docs.nvidia.com/deeplearning/cudnn/installation/linux.html
# Using memcheck to locate illegal memory access issues
compute-sanitizer --tool memcheck --kernel-regex kne=identifyTileRanges python easyvolcap/scripts/main.py -c configs/exps/fdgs/fdgs_0319_f0.yaml runner_cfg.log_interval=1 # *: will complain about libcudnn
/usr/local/cuda-11.8/bin/cuda-memcheck python easyvolcap/scripts/main.py -c configs/exps/fdgs/fdgs_0319_f0.yaml runner_cfg.log_interval=1 # !: will not work...
```

```shell
# Installing CUDA Driver
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/525.116.04/NVIDIA-Linux-x86_64-525.116.04.run
chmod +x NVIDIA-Linux-x86_64-525.116.04.run
sudo ~/NVIDIA-Linux-x86_64-$(cat /sys/module/nvidia/version).run --accept-license --ui=none --no-kernel-module --no-questions

# Installing CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
chmod +x cuda_12.1.0_530.30.02_linux.run
sudo cuda_12.1.0_530.30.02_linux.run --accept-license --ui=none --no-kernel-module --no-questions --no-driver # ?: is it really --no-driver

# Installing CuDNN
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.0.131_cuda12-archive.tar.xz

tar -xvf cudnn-linux-x86_64-8.9.0.131_cuda12-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

```shell
# Testing speed
python -c "import torch; from tqdm import tqdm; [(torch.rand(2000, 2000, device='cuda')**10).ndim for i in tqdm(range(1000000000))]"
```

## PyTorch CUDA Extension