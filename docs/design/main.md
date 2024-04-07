# Structure of ***EasyVolcap***

## Core Modules

Here's a directory tree of the core modules of EasyVolcap.
I commended on the directory structure of the `easyvolcap` folder, which should give you a general sense about our codebase's organization.

```shell
easyvolcap
├── engine # configuration and module registration control
│   ├── __init__.py # the entry point of EasyVolcap
│   ├── config.py
│   ├── registry.py
│   └── ...
├── dataloaders # basic torch multi-process dataloader
│   ├── datasets
│   │   ├── volumetric_video_dataset.py # multi-view dataset, with sharding, memory storage, jpeg/png compression
│   │   └── ...
│   ├── datasamplers.py # controls the sampling process (view? frame?)
│   └── volumetric_video_dataloader.py
├── models # a `model` controls the flow of batch -> output
│   ├── cameras # make camera parameters optimizable
│   │   ├── optimizable_camera.py
│   │   └── ...
│   ├── samplers # sample points on rays (NeRF-like), or just output the full image here (custom network)
│   │   ├── importance_sampler.py
│   │   └── ...
│   ├── networks # NeRF-like networks or network components
│   │   ├── embedders # convert from physical properties to features, also used elsewhere
│   │   │   ├── positional_encoding_embedder.py
│   │   │   └── ...
│   │   ├── regressors # convert from features to physical properties, also used elsewhere
│   │   │   ├── mlp_regressor.py
│   │   │   └── ...
│   │   ├── volumetric_video_network.py
│   │   └── ...
│   ├── renderers # render sampled points to image tensors
│   │   ├── volume_renderer.py
│   │   └── ...
│   ├── supervisors # compute losses
│   │   ├── volumetric_video_supervisor.py
│   │   └── ...
│   ├── volumetric_video_model.py
│   └── ...
├── runners
│   ├── evaluators # compute metrics and the call visualizers
│   │   ├── volumetric_video_evaluator.py
│   │   └── ...
│   ├── visualizers # convert from tensors to disk images
│   │   ├── volumetric_video_visualizer.py
│   │   └── ...
│   ├── moderators.py # like schedulers, but is customizable
│   ├── optimizers.py # torch optimizers
│   ├── schedulers.py # lr schedulers
│   ├── recorders.py # record logs to tensorboard
│   ├── volumetric_video_runner.py # controls the training loop
│   ├── volumetric_video_viewer.py # high-performance viewer
│   └── ...
├── scripts # entry points for EasyVolcap, all command-line should start from here
│   ├── main.py # evc-test ...
│   └── wrap.py # evc-gui, evc-dist ...
└── utils # store utility functions here, should not import from other modules
    ├── shaders # we place the shaders here for easier loading from gl_utils
    │   ├── splat.vert
    │   ├── splat.frag
    │   └── ...
    ├── console_utils.py # the main logging utils, import this for better logging
    ├── timer_utils.py # a global timer
    ├── data_utils.py
    ├── net_utils.py
    ├── egl_utils.py
    ├── gl_utils.py
    └── ...
```

For more detailed descriptions and illustrations, please refer to our technical brief on [arXiv](https://arxiv.org/abs/2312.06575).
For more details on the individual systems, please look into their corresponding documentation in the [`docs`](../../docs) directory.

## Others

Aside from the core modules and functionalities, we also provide a set of [`scripts`](../../scripts) for common tasks, such as data preparation, mask extraction and external commands.
The [`tests`](../../tests) folder contains a set of unit tests for the core modules, which we plan on expanding and make more comprehensive in the future.
The [`configs`](../../configs) folder provides a set of experiment configurations to use for various systems.

```shell
configs
├── base.yaml # all configs should inherit this
├── datasets # contains dataset specific configurations, like where to find images and bbox etc -> corresponding to the `dataloaders` folder in easyvolcap
├── models # defines network modules' conbination -> corresponding to the `models` folder in easyvolcap
├── exps # combines datasets and models to form an experiment
├── specs # collection of config enties, so you don't have to type them in the command line everytime
└── projects # defines project specific experiments and dataset configurations (like splits)

scripts
├── enerf_outdoor # enerf_outdoor dataset integration
├── zjumocap # zjumocap dataset integration
├── easymocap # easymocap integration (mocap & calib), will expand
├── colmap # colmap integration (calib)
├── preprocess # convert dataset from one format to another (mostly to easyvolcap format)
├── segmentation # inference mask from images (or background)
└── tools # useful tools for visualization and easyvolcap management
```