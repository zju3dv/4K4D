python scripts/actorshq/actorshq2easyvolcap.py --data_root data/actorshq/Actor01/Sequence1/1x
python scripts/actorshq/actorshq2easyvolcap.py --data_root data/actorshq/Actor01/Sequence2/1x
python scripts/actorshq/actorshq2easyvolcap.py --data_root data/actorshq/Actor02/Sequence1/1x
python scripts/actorshq/actorshq2easyvolcap.py --data_root data/actorshq/Actor02/Sequence2/1x
python scripts/actorshq/actorshq2easyvolcap.py --data_root data/actorshq/Actor03/Sequence1/1x
# python scripts/actorshq/actorshq2easyvolcap.py --data_root data/actorshq/Actor04/Sequence1/1x
python scripts/actorshq/actorshq2easyvolcap.py --data_root data/actorshq/Actor04/Sequence2/1x
python scripts/actorshq/actorshq2easyvolcap.py --data_root data/actorshq/Actor05/Sequence1/1x
python scripts/actorshq/actorshq2easyvolcap.py --data_root data/actorshq/Actor05/Sequence2/1x
python scripts/actorshq/actorshq2easyvolcap.py --data_root data/actorshq/Actor06/Sequence1/1x
python scripts/actorshq/actorshq2easyvolcap.py --data_root data/actorshq/Actor06/Sequence2/1x
python scripts/actorshq/actorshq2easyvolcap.py --data_root data/actorshq/Actor07/Sequence1/1x
python scripts/actorshq/actorshq2easyvolcap.py --data_root data/actorshq/Actor08/Sequence1/1x
python scripts/actorshq/actorshq2easyvolcap.py --data_root data/actorshq/Actor08/Sequence2/1x

python scripts/actorshq/report_bounds.py --data_root data/actorshq/Actor01/Sequence1/1x
python scripts/actorshq/report_bounds.py --data_root data/actorshq/Actor01/Sequence2/1x
python scripts/actorshq/report_bounds.py --data_root data/actorshq/Actor02/Sequence1/1x
python scripts/actorshq/report_bounds.py --data_root data/actorshq/Actor02/Sequence2/1x
python scripts/actorshq/report_bounds.py --data_root data/actorshq/Actor03/Sequence1/1x
# python scripts/actorshq/report_bounds.py --data_root data/actorshq/Actor04/Sequence1/1x
python scripts/actorshq/report_bounds.py --data_root data/actorshq/Actor04/Sequence2/1x
python scripts/actorshq/report_bounds.py --data_root data/actorshq/Actor05/Sequence1/1x
python scripts/actorshq/report_bounds.py --data_root data/actorshq/Actor05/Sequence2/1x
python scripts/actorshq/report_bounds.py --data_root data/actorshq/Actor06/Sequence1/1x
python scripts/actorshq/report_bounds.py --data_root data/actorshq/Actor06/Sequence2/1x
python scripts/actorshq/report_bounds.py --data_root data/actorshq/Actor07/Sequence1/1x
python scripts/actorshq/report_bounds.py --data_root data/actorshq/Actor08/Sequence1/1x
python scripts/actorshq/report_bounds.py --data_root data/actorshq/Actor08/Sequence2/1x

evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0001_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0001_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/surfs.yaml

evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0001_02.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0001_02.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/surfs.yaml

evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0002_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0002_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/surfs.yaml

evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0002_02.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0002_02.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/surfs.yaml

evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0003_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0003_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/surfs.yaml

# evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0004_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
# evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0004_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/surfs.yaml

evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0004_02.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0004_02.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/surfs.yaml

evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0005_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0005_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/surfs.yaml

evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0005_02.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0005_02.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/surfs.yaml

evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0006_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0006_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/surfs.yaml

evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0006_02.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0006_02.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/surfs.yaml

evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0007_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0007_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/surfs.yaml

evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0008_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0008_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/surfs.yaml

evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0008_02.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0008_02.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/surfs.yaml

# Special cases for memory consideration
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0006_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml,configs/specs/vhulls.yaml val_dataloader_cfg.dataset_cfg.frame_sample=2000,None,1 val_dataloader_cfg.dataset_cfg.ratio=0.5 # jump 2
evc-test -c configs/base.yaml,configs/models/r4dv.yaml,configs/datasets/actorshq/0006_01.yaml,configs/specs/mask.yaml,configs/specs/vis.yaml dataloader_cfg.dataset_cfg.frame_sample=2000,None,1 val_dataloader_cfg.dataset_cfg.frame_sample=2000,None,1,configs/specs/surfs.yaml
