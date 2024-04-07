### Tricks

```shell
evc-test -c configs/exps/l3mhet/l3mhet_ipstage.yaml configs=configs/specs/geometry.yaml val_dataloader_cfg.dataset_cfg.voxel_size=0.05 val_dataloader_cfg.dataset_cfg.frame_sample=0,1,1
evc-test -c configs/exps/enerf/enerf_seq3.yaml exp_name=enerf_dtu configs=configs/specs/fp16.yaml,configs/specs/cubic.yaml,configs/specs/ibr.yaml val_dataloader_cfg.dataset_cfg.render_size=1080,1920 val_dataloader_cfg.dataset_cfg.smoothing_term=15.0 val_dataloader_cfg.dataset_cfg.n_render_views=300 runner_cfg.visualizer_cfg.save_tag=seq3
evc-test -c configs/exps/enerf/enerf_seq3.yaml exp_name=enerf_dtu configs=configs/specs/fp16.yaml,configs/specs/cubic.yaml,configs/specs/ibr.yaml runner_cfg.visualizer_cfg.save_tag=seq3 val_dataloader_cfg.dataset_cfg.render_size=1080,1920 val_dataloader_cfg.dataset_cfg.smoothing_term=15.0 val_dataloader_cfg.dataset_cfg.n_render_views=300 val_dataloader_cfg.batch_sampler_cfg.n_srcs_list=2,

python scripts/fusion/volume_fusion.py -- -c configs/exps/l3mhet/l3mhet_zju3dv.yaml configs=configs/specs/fp16.yaml,configs/specs/vis.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 val_dataloader_cfg.dataset_cfg.frame_sample=0,None,25
python scripts/fusion/volume_fusion.py -- -c configs/exps/l3mhet/l3mhet_dance3_static.yaml configs=configs/specs/fp16.yaml,configs/specs/vis.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 val_dataloader_cfg.view_sample=0,None,3

evc-test -c configs/projects/stableenerf/enerf/enerf_my_313_se.yaml runner_cfg.visualizer_cfg.save_tag=dtu_pretrain val_dataloader_cfg.dataset_cfg.immask_crop=False val_dataloader_cfg.dataset_cfg.imbound_crop=False val_dataloader_cfg.dataset_cfg.immask_fill=True
evc-test -c configs/exps/l3mhet/l3mhet_seq3.yaml exp_name=l3mhet_seq3 configs=configs/specs/fp16.yaml,configs/specs/interp.yaml val_dataloader_cfg.dataset_cfg.render_size=1080,1920 val_dataloader_cfg.dataset_cfg.smoothing_term=15.0 val_dataloader_cfg.dataset_cfg.n_render_views=300

python scripts/fusion/volume_fusion.py -- -c configs/exps/l3mhet/l3mhet_xuzhen_static.yaml configs=configs/specs/fp16.yaml,configs/specs/vis.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 val_dataloader_cfg.dataset_cfg.view_sample=0,None,6
python scripts/tools/extract_mesh.py --occ_thresh 0.3 -- -c configs/exps/l3mhet/l3mhet_0008_05_static.yaml

python scripts/realtime4dv/charger.py SuperChargedR4DV scr4dv_my_377_optcam -- -c configs/exps/r4dv/r4dv_my_377_optcam.yaml configs=configs/specs/super.yaml
python scripts/realtime4dv/charger.py SuperChargedR4DV scr4dv_my_387_optcam -- -c configs/exps/r4dv/r4dv_my_387_optcam.yaml configs=configs/specs/super.yaml
python scripts/realtime4dv/charger.py SuperChargedR4DV scr4dv_my_390_optcam -- -c configs/exps/r4dv/r4dv_my_390_optcam.yaml configs=configs/specs/super.yaml

# Start a long running profling job to debug training time issues
evc-prof -c configs/exps/r4dv/r4dv_${name}_optcam.yaml profiler_cfg.skip_first=0 profiler_cfg.wait=2490 profiler_cfg.warmup=5 profiler_cfg.active=5 profiler_cfg.repeat=0 # long running profiling job
evc-train -c configs/exps/r4dv/r4dv_${name}_optcam.yaml profiler_cfg.enabled=True profiler_cfg.skip_first=0 profiler_cfg.wait=2490 profiler_cfg.warmup=5 profiler_cfg.active=5 profiler_cfg.repeat=0 profiler_cfg.clear_previous=False runner_cfg.epochs=250 runner_cfg.decay_epochs=1600 # long running profiling job
evc-gui -c configs/exps/r4dv/r4dv_${name}_optcam.yaml exp_name=scr4dv_${name}_optcam configs=configs/specs/superf.yaml val_dataloader_cfg.dataset_cfg.frame_sample=0,1,1

# Profiling the l3mhet job
evc-train -c configs/exps/l3mhet/l3mhet_seq3.yaml exp_name=l3mhet_seq3_prof runner_cfg.ep_iter=50 runner_cfg.epochs=1 profiler_cfg.enabled=True profiler_cfg.skip_first=0 profiler_cfg.wait=40 profiler_cfg.warmup=5 profiler_cfg.active=5 profiler_cfg.repeat=1 profiler_cfg.clear_previous=False runner_cfg.resume=False

# Export video for easyvolcap technical communications
evc-test -c configs/exps/r4dv/r4dv_0013_01_optcam.yaml exp_name=scr4dv_0013_01_optcam configs=configs/specs/superf.yaml val_dataloader_cfg.dataset_cfg.frame_sample=0,150,1 val_dataloader_cfg.dataset_cfg.camera_path_intri=data/paths/0013_01/intri.yml val_dataloader_cfg.dataset_cfg.camera_path_extri=data/paths/0013_01/extri.yml configs=configs/specs/superf.yaml,configs/specs/cubic.yaml,configs/specs/ibr.yaml val_dataloader_cfg.dataset_cfg.smoothing_term=10.0 model_cfg.sampler_cfg.should_release_memory=False val_dataloader_cfg.dataset_cfg.ratio=1.0 val_dataloader_cfg.dataset_cfg.render_size=2160,3840 val_dataloader_cfg.dataset_cfg.n_render_views=600 runner_cfg.visualizer_cfg.vis_ext=.png runner_cfg.visualizer_cfg.video_fps=60 runner_cfg.visualizer_cfg.dpt_cm=virdis

# Prepare my server configuration
zip -r zsh_vim_tmux.zip .zshrc .oh-my-zsh .tmux .tmux.conf .tmux.conf.local .p10k.zsh .gitconfig .ssh .config/nvim software/exa software/nvim-linux64 software/vmtouch software/viu software/v2ray-sagernet software/aliyunpan -x .ssh/id_rsa
```

#### Swapfiles

```shell
# Removing existing swapfiles
sudo swapoff -v /swapfile
# Remove the swap file entry /swapfile swap swap defaults 0 0 from the /etc/fstab file.
sudo rm /swapfile

# Creating swapfile
sudo fallocate -l 512G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```