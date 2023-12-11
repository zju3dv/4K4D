# Using EasyVolcap For Camera Parameters Optimization (Finetuning)


Test scripts:

```shell
# Train static ngpt with mask
evc -c configs/exps/ngpt/ngpt_my_313_static_mask.yaml runner_cfg.eval_ep=1 runner_cfg.resume=False exp_name=ngpt_my_313_static_mask_$(git rev-parse --short=6 HEAD)

# Train static mhet without mask
evc -c configs/exps/mhet/mhet_sport1_static.yaml runner_cfg.eval_ep=1 runner_cfg.resume=False exp_name=mhet_sport1_static_$(git rev-parse --short=6 HEAD)

# Train static ngpt without mask
evc -c configs/exps/ngpt/ngpt_sport1_static.yaml runner_cfg.eval_ep=1 runner_cfg.resume=False exp_name=ngpt_sport1_static_$(git rev-parse --short=6 HEAD)

# Render static ngpt with mask using the default dataset
evc -t test -c configs/exps/ngpt/ngpt_my_313_static_mask.yaml exp_name=ngpt_my_313_static_mask_$(git rev-parse --short=6 HEAD)

# Render static ngpt with mask using oribt cameras
evc -t test -c configs/exps/ngpt/ngpt_my_313_static_mask.yaml exp_name=ngpt_my_313_static_mask_$(git rev-parse --short=6 HEAD) configs=configs/specs/orbit.yaml

# Render IBR from enerf
evc -t test -c configs/exps/enerf/enerf_my_313.yaml runner_cfg.eval_ep=1 exp_name=enerf_my_313_$(git rev-parse --short=6 HEAD) configs=configs/specs/orbit.yaml,configs/specs/ibr.yaml val_dataloader_cfg.dataset_cfg.frame_sample=0,1,1
```

Some examples:

```shell
evc -c configs/exps/mhet/mhet_basketball_static.yaml runner_cfg.eval_ep=1 runner_cfg.resume=False # train mhe on first frame of enerf_outdoor dataset
evc -t test -c configs/exps/mhet/mhet_basketball_static.yaml configs=configs/specs/orbit.yaml runner_cfg.visualizer_cfg.types=RENDER,DEPTH, val_dataloader_cfg.dataset_cfg.render_size=1024,1024 # render basketball from NHR in FHD

evc -c configs/exps/mhet/mhet_actor1_static.yaml runner_cfg.eval_ep=1 runner_cfg.resume=False # train mhe on first frame of enerf_outdoor dataset
evc -t test -c configs/exps/mhet/mhet_actor1_static.yaml configs=configs/specs/interp.yaml runner_cfg.visualizer_cfg.types=RENDER,DEPTH, val_dataloader_cfg.dataset_cfg.render_size=1024,1024 # render actor1 from enerf_outdoor in FHD

evc -c configs/exps/mhet/mhet_ip412_blurry.yaml runner_cfg.eval_ep=1 runner_cfg.resume=False # train mhe on 412
evc -t test -c configs/exps/mhet/mhet_ip412_blurry.yaml configs=configs/specs/orbit.yaml runner_cfg.visualizer_cfg.types=RENDER,DEPTH, val_dataloader_cfg.dataset_cfg.render_size=1024,1024 # render mhe on iphone_412 scene in FHD

evc -c configs/exps/enerf/enerf_actor1.yaml runner_cfg.eval_ep=1 # train enerf, render images every epoch
evc -c configs/exps/enerf/enerf_actor1_static.yaml runner_cfg.eval_ep=1 runner_cfg.resume=False # train enerf on one frame, render images every epoch, start from scratch
evc-prof -t test -c configs/exps/enerf/enerf_actor1.yaml # perform profling on testing of enerf

# GUI without any control
evc -t gui  -c configs/exps/enerf/enerf_actor1.yaml exp_name=enerf_actor1_caad45 val_dataloader_cfg.dataset_cfg.ratio=0.8 configs=configs/specs/cubic.yaml,configs/specs/ibr.yaml val_dataloader_cfg.dataset_cfg.view_sample=7,6,5,2,15,13,17,8,16,14,12,11,10,1,9,0,3,4 runner_cfg.visualizer_cfg.types=RENDER,DEPTH val_dataloader_cfg.dataset_cfg.n_render_views=600 val_dataloader_cfg.dataset_cfg.frame_sample=0,600,1 runner_cfg.visualizer_cfg.types="[]" viewer_cfg.window_size=800,1184

evc -t gui -c configs/exps/enerf/enerf_actor1.yaml exp_name=enerf_actor1_caad45 val_dataloader_cfg.dataset_cfg.ratio=0.8 configs=configs/specs/cubic.yaml,configs/specs/ibr.yaml val_dataloader_cfg.dataset_cfg.view_sample=7,6,5,2,15,13,17,8,16,14,12,11,10,1,9,0,3,4 runner_cfg.visualizer_cfg.types=RENDER,DEPTH val_dataloader_cfg.dataset_cfg.n_render_views=600 val_dataloader_cfg.dataset_cfg.frame_sample=0,1,1 runner_cfg.visualizer_cfg.types="[]" viewer_cfg.window_size=800,1184 # for faster testing

evc -t gui -c configs/exps/enerf/enerf_actor1.yaml exp_name=enerf_actor1_caad45 val_dataloader_cfg.dataset_cfg.ratio=0.4 configs=configs/specs/cubic.yaml,configs/specs/ibr.yaml val_dataloader_cfg.dataset_cfg.view_sample=7,6,5,2,15,13,17,8,16,14,12,11,10,1,9,0,3,4 runner_cfg.visualizer_cfg.types=RENDER,DEPTH val_dataloader_cfg.dataset_cfg.n_render_views=600 val_dataloader_cfg.batch_sampler_cfg.n_srcs_list=3, val_dataloader_cfg.dataset_cfg.frame_sample=0,1,1 runner_cfg.visualizer_cfg.types="[]" viewer_cfg.window_size=400,592 # for faster rendering of small images

# GUI with some control, slow
evc -t gui -c configs/exps/enerf/enerf_actor1.yaml exp_name=enerf_actor1_ce1fdd val_dataloader_cfg.dataset_cfg.ratio=0.8 val_dataloader_cfg.dataset_cfg.frame_sample=0,1,1 runner_cfg.visualizer_cfg.types="[]"

# Rendering test views for ENeRF
evc -t test -c configs/exps/enerf/enerf_actor1.yaml exp_name=enerf_actor1_$(git rev-parse --short=6 HEAD) val_dataloader_cfg.dataset_cfg.ratio=0.8 configs=configs/specs/cubic.yaml,configs/specs/ibr.yaml val_dataloader_cfg.dataset_cfg.view_sample=7,6,5,2,15,13,17,8,16,14,12,11,10,1,9,0,3,4 runner_cfg.visualizer_cfg.types=RENDER,DEPTH val_dataloader_cfg.dataset_cfg.n_render_views=600 val_dataloader_cfg.dataset_cfg.frame_sample=0,600,1 # for faster testing

# The commands before are obsolete now
evc -t gui -c configs/exps/enerf/enerf_my_313.yaml exp_name=enerf_my_313_0d7f34 val_dataloader_cfg.dataset_cfg.frame_sample=0,1,1 viewer_cfg.window_size=512,512 # for faster rendering of small images

# GUI for ngp
evc -t gui -c configs/exps/ngpt/ngpt_my_313_static_mask.yaml exp_name=ngpt_my_313_static_mask_a68c25 val_dataloader_cfg.dataset_cfg.ratio=0.5 val_dataloader_cfg.dataset_cfg.frame_sample=0,1,1 runner_cfg.visualizer_cfg.types="[]" viewer_cfg.window_size=256,256 model_cfg.renderer_cfg.bg_brightness=0.0 # for faster rendering of small images
```

We use similar strategies for optimizing camera parameters as in Instant-NGP.
For now, only extrinsic parameters are taken care of.
A script for extracting the optimized camera can be found at: `scripts/tools/extract_optimized_cameras.py`

```shell

# Optimized actor1_4
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_actor1_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_actor2_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_actor3_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_actor1_4_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_actor2_3_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_actor5_6_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_actor1_complex_static.yaml

# Optimized RenBody
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0007_01_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0008_01_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0008_03_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0008_05_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0011_11_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0012_01_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0012_03_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0012_11_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0013_01_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0013_03_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0013_06_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0013_09_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0013_11_static.yaml
# python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0014_09_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0019_08_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0019_09_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0021_01_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0021_02_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0021_03_static.yaml
# python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0021_08_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0022_04_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0022_07_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0023_01_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0023_04_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_0023_06_static.yaml

# Optimized ZJU-MoCap
# python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_my_313_static.yaml
# python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_my_315_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_my_377_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_my_386_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_my_387_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_my_390_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_my_392_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_my_393_static.yaml

# Optimized MobileStage
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_dance3_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_xuzhen_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_purple_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_black_static.yaml
python scripts/tools/extract_optimized_cameras.py -- -c configs/exps/l3mhet/l3mhet_white_static.yaml
```