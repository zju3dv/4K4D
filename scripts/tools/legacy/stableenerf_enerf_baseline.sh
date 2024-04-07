# SECTION: ZJUMOCAP
# Convert pretrained model
python scripts/tools/prepare_finetune.py enerf_dtu enerf_my_313_se
python scripts/tools/prepare_finetune.py enerf_dtu enerf_my_315_se
python scripts/tools/prepare_finetune.py enerf_dtu enerf_my_377_se
python scripts/tools/prepare_finetune.py enerf_dtu enerf_my_386_se
python scripts/tools/prepare_finetune.py enerf_dtu enerf_my_387_se
python scripts/tools/prepare_finetune.py enerf_dtu enerf_my_390_se
python scripts/tools/prepare_finetune.py enerf_dtu enerf_my_392_se
python scripts/tools/prepare_finetune.py enerf_dtu enerf_my_393_se
python scripts/tools/prepare_finetune.py enerf_dtu enerf_my_394_se

# Extrac visual hulls using all views
evc-test -c configs/projects/stableenerf/enerf/enerf_my_313_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain' dry_run=True val_dataloader_cfg.dataset_cfg.frame_sample=0,null,1
evc-test -c configs/projects/stableenerf/enerf/enerf_my_315_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain' dry_run=True val_dataloader_cfg.dataset_cfg.frame_sample=0,null,1
evc-test -c configs/projects/stableenerf/enerf/enerf_my_377_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain' dry_run=True val_dataloader_cfg.dataset_cfg.frame_sample=0,null,1
evc-test -c configs/projects/stableenerf/enerf/enerf_my_386_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain' dry_run=True val_dataloader_cfg.dataset_cfg.frame_sample=0,null,1
evc-test -c configs/projects/stableenerf/enerf/enerf_my_387_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain' dry_run=True val_dataloader_cfg.dataset_cfg.frame_sample=0,null,1
evc-test -c configs/projects/stableenerf/enerf/enerf_my_390_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain' dry_run=True val_dataloader_cfg.dataset_cfg.frame_sample=0,null,1
evc-test -c configs/projects/stableenerf/enerf/enerf_my_392_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain' dry_run=True val_dataloader_cfg.dataset_cfg.frame_sample=0,null,1
evc-test -c configs/projects/stableenerf/enerf/enerf_my_393_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain' dry_run=True val_dataloader_cfg.dataset_cfg.frame_sample=0,null,1
evc-test -c configs/projects/stableenerf/enerf/enerf_my_394_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain' dry_run=True val_dataloader_cfg.dataset_cfg.frame_sample=0,null,1

# Save pretrain rendering results
evc-test -c configs/projects/stableenerf/enerf/enerf_my_313_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain'
evc-test -c configs/projects/stableenerf/enerf/enerf_my_315_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain'
evc-test -c configs/projects/stableenerf/enerf/enerf_my_377_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain'
evc-test -c configs/projects/stableenerf/enerf/enerf_my_386_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain'
evc-test -c configs/projects/stableenerf/enerf/enerf_my_387_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain'
evc-test -c configs/projects/stableenerf/enerf/enerf_my_390_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain'
evc-test -c configs/projects/stableenerf/enerf/enerf_my_392_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain'
evc-test -c configs/projects/stableenerf/enerf/enerf_my_393_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain'
evc-test -c configs/projects/stableenerf/enerf/enerf_my_394_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain'

# Run finetuning and save finetuned results
evc-train -c configs/projects/stableenerf/enerf/enerf_my_313_se.yaml
evc-train -c configs/projects/stableenerf/enerf/enerf_my_315_se.yaml
evc-train -c configs/projects/stableenerf/enerf/enerf_my_377_se.yaml
evc-train -c configs/projects/stableenerf/enerf/enerf_my_386_se.yaml
evc-train -c configs/projects/stableenerf/enerf/enerf_my_387_se.yaml
evc-train -c configs/projects/stableenerf/enerf/enerf_my_390_se.yaml
evc-train -c configs/projects/stableenerf/enerf/enerf_my_392_se.yaml
evc-train -c configs/projects/stableenerf/enerf/enerf_my_393_se.yaml
evc-train -c configs/projects/stableenerf/enerf/enerf_my_394_se.yaml

# SECTION: NHR

# Convert pretrained model
python scripts/tools/prepare_finetune.py enerf_dtu enerf_sport1_se
python scripts/tools/prepare_finetune.py enerf_dtu enerf_sport2_se
python scripts/tools/prepare_finetune.py enerf_dtu enerf_sport3_se
python scripts/tools/prepare_finetune.py enerf_dtu enerf_basketball_se

# Extrac visual hulls using all views
evc-test -c configs/projects/stableenerf/enerf/enerf_sport1_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain' dry_run=True val_dataloader_cfg.dataset_cfg.frame_sample=0,null,1
evc-test -c configs/projects/stableenerf/enerf/enerf_sport2_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain' dry_run=True val_dataloader_cfg.dataset_cfg.frame_sample=0,null,1
evc-test -c configs/projects/stableenerf/enerf/enerf_sport3_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain' dry_run=True val_dataloader_cfg.dataset_cfg.frame_sample=0,null,1
evc-test -c configs/projects/stableenerf/enerf/enerf_basketball_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain' dry_run=True val_dataloader_cfg.dataset_cfg.frame_sample=0,null,1

# Save pretrain rendering results
evc-test -c configs/projects/stableenerf/enerf/enerf_sport1_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain'
evc-test -c configs/projects/stableenerf/enerf/enerf_sport2_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain'
evc-test -c configs/projects/stableenerf/enerf/enerf_sport3_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain'
evc-test -c configs/projects/stableenerf/enerf/enerf_basketball_se.yaml runner_cfg.visualizer_cfg.save_tag='pretrain'

# Run finetuning and save finetuned results
evc-train -c configs/projects/stableenerf/enerf/enerf_sport1_se.yaml
evc-train -c configs/projects/stableenerf/enerf/enerf_sport2_se.yaml
evc-train -c configs/projects/stableenerf/enerf/enerf_sport3_se.yaml
evc-train -c configs/projects/stableenerf/enerf/enerf_basketball_se.yaml
