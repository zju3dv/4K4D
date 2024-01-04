# Configuration & Registry System

## Features

We inherited and modified the configuration system from [`mmcv`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py), along with a [registry](https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py)-based module-building system.
The original `mmcv` config support parent (base) config, and an easy-to-use file / command-line interface.

Here's an example config chain:

```yaml
# Content of renbody.yaml (parent config)
dataloader_cfg: # we see the term "dataloader" as one word?
    dataset_cfg: &dataset_cfg
        masks_dir: masks # good naming ^_^
        ratio: 0.5
        bounds: [[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]] # thinner?

        force_sparse_view: True
        view_sample: [0, 60, 1]
        frame_sample: [0, 150, 1] # only train for a thousand frames

model_cfg:
    sampler_cfg:
        bg_brightness: 0.0
    renderer_cfg:
        bg_brightness: 0.0

val_dataloader_cfg:
    dataset_cfg:
        <<: *dataset_cfg
        frame_sample: [0, 150, 30]
    sampler_cfg:
        view_sample: [0, 60, 20]


# Content of 0013_01.yaml (inner config, child of renbody.yaml, parent of 0013_01_obj.yaml)
configs: configs/datasets/renbody/renbody.yaml

dataloader_cfg: # we see the term "dataloader" as one word?
    dataset_cfg: &dataset_cfg
        data_root: data/renbody/0013_01
        images_dir: images_calib

val_dataloader_cfg:
    dataset_cfg:
        <<: *dataset_cfg

# Content of 0013_01_obj.yaml (child of 0013_01.yaml)
configs: configs/datasets/renbody/0013_01.yaml

dataloader_cfg: &dataloader_cfg
    dataset_cfg: &dataset_cfg # ratio: 0.5
        bounds: [[-0.5352, -0.7697, -0.9967], [0.4148, 0.7203, 0.9533]] # !: BATCH

val_dataloader_cfg:
    dataset_cfg:
        <<: *dataset_cfg
```

We build upon `mmcv`'s system to support some of the missing features:

- Full inheritance support is added (personally find this very useful):
  - If a parent config has the same key with the parent, the parent's value will be recursively overwritten. For example, the `bounds` key in `0013_01_obj.yaml` will overwrite the `bounds` in `renbody.yaml`.
  - Special keywords can be used for acessing the parent's config. Check out [`configs/specs/solid.yaml`](../../configs/specs/solid.yaml) for some examples. Note that to use these special keys will lose their effects if formatted, thus we added a magic line `# prettier-ignore` to prevent this.
  - Adding a `_delete_: True` in the children will make sure the config system ignore the parent configs of this key.
  - Multiple parent can be used. This is most useful when building experiments configs like [`configs/exps/4k4d/4k4d_0013_01.yaml`](../../configs/exps/4k4d/4k4d_0013_01_r4.yaml).
  - Special keys are supported for extracting information from the config file itself, like `{{fileBasenameNoExtension}}`. I always put such a config at the end of an experiment config to match the config to the experiment records.
- Empty `yaml` files are allowed and ignored (will raise error in `mmcv`).
- Extra arguments are automatically ignored (possibly with warning messages).
- More robust path support:
  - Project root-relative paths. ([`configs/models/enerf.yaml`](../../configs/models/enerf.yaml))
  - Config file relative paths. (`enerf.yaml`)
  - Absolute config file paths. (`/home/<user>/easyvolcap/configs/models/enerf.yaml`)
- An `_append_` operator for appending to existing arrays.
- `_` configs are ignored.

Exists in `mmcv` and useful features:

- Dict-based command-line interface. (`model_cfg.supervisor_cfg.msk_loss_weight=0.01`)
- A `type`-based building system for registered modules (`build_from_cfg` will look up the registry for the `type` string key).
  - This allows for easy switching between modules. (`val_dataloader_cfg.dataset_cfg.type=InferenceDataset`)
- A `configs` key for specifying a file to inherit from (also available from the command line). (`configs=configs/specs/orbit.yaml`)
- A `_delete_` operator for preventing inheritance.
- Digital keys for replacing array elements. (`model_cfg.network_cfg.network_cfgs.0.type=...`)
- Arrays can be explicitly defined in the command line. (`val_dataloader_cfg.dataset_cfg.view_sample=[0, -1, 1]`)
- `.yaml`, `.py`, `.json` flavors config files.
  - I personally prefer `.yaml`s due to their simplicity, however, most modern projects use `.py`s.

Caveats:

- `None` in the command line will be parsed to actual None. `null` will also be parsed to `None`.
- `None` in python config files are straight forward.
- `None` in yaml config files are expected to be `null`.
- For array indexing `[0, -1, 1]` will **exclude** the last element, use `[0, null, 1]` instead.

## Using the Configuration System

Take away: 

- **Configurations are all python `dict` under the hood.**
- **Layer by layer we replace values from the inner most (functions default arguments) to the outer most (comman-line args).**
- **You can simply think of the config system as a fancier way of defining function arguments in Python. By fancier I mean adding some warning and command line interfaces.**

Here's an example of using the config system programmatically and simultaneously marking the function itself as callable from configs.

```python
@catch_throw
@callable_from_cfg
def gui(
    viewer_cfg: dotdict = dotdict(type="VolumetricVideoViewer"),  # use different naming for config here, is this good?
    invokation_type: str = 'test',  # TODO: implement camera and other dataset types

    # Reproducibility configuration
    base_device: str = 'cuda',
    dry_run: bool = False,  # return without hassle
    **kwargs,
):
    runner: "VolumetricVideoRunner" = globals()[invokation_type](kwargs,
                                                                 base_device=base_device,
                                                                 dry_run=True,
                                                                 )  # return the runner (trainer) immediately
    viewer: "VolumetricVideoViewer" = RUNNERS.build(viewer_cfg, runner=runner)  # will start the window
    if dry_run: return runner  # just construct everything, then return

    launcher(**kwargs, runner_function=viewer.run, runner_object=runner)
```

We provide interfaces for using the config-based building system (with or without registering) directly:

- Mark a function with the `@callable_from_cfg` decorator to enjoy the default argument substitution functionality.
- For a function (whose argument you want to make configurable), use `@callable_from_cfg`.
- For a module (or a function returning a module), which will be called later in runtime (i.e. a PyTorch `nn.Module`), use the registry system.
  - Define a corresponding registry for this type of module. (`EMBEDDERS = Registry('embedders')`)
  - Register the module with the registry (`@EMBEDDERS.register_module()`) to make it accessible from a string.
  - We defined a default `__init__.py` for folders to support fully automatic recursive imports to actually call `register_module`.
    - You can find such `__init__.py` lying around throughout the entire project. ([`easyvolcap/dataloaders/__init__.py`](../../easyvolcap/dataloaders/__init__.py))
    - When a new directory contains modules for registration, copy this file to the directory.
    - We register all modules in `main.py` before doing any compiling & building.
  - The imports for the `registry` and `config` system can be found in the official [`easyvolcap/engine/__init__.py`](../easyvolcap/engine/__init__.py). You can copy the imports of this file to utilized the registry and configuration system.

The configuration system is built in layers:

- The most default arguments are always written as **function defaults** in the python code, this is the recommended way for default args.
  - With a complete module, a complete list (physically meaningful) of default args should be provided in this fashion.
  - This allows for easy switching between different modules. (`model_cfg.network_cfg.sampler_cfg.type=CostVolumeDepthSampler`)
- The next level is defined in [`configs/base.yaml`](../../configs/base.yaml), which defines the default structure for a simple `nert+latent code` representation.
  - To make command-line configuration (and replacement-based configuration) possible, the full structure of the `dataloader`, `model`, and `runner` needs to be specified.
  - Otherwise, something like `val_dataloader_cfg.dataset_cfg.type=InferenceDataset` will not complain about no `val_dataloader` key found.
- Then, we settled on defining model-related and data-related configs separately in [`configs/models`](../../configs/models) and [`configs/datasets`](../../configs/datasets) respectively.
  - This `makes reusing `datasets ` on different` modules much easier.
  - [`configs/exps`](../../configs/exps) combines [`configs/models`](../../configs/models) and [`configs/datasets`](../../configs/datasets) to define a particular experiment.
  - We also created a [`configs/specs`](../../configs/specs) for easily applying a bunch of configs on some predefined methods.
  - For example, to make a method train on only the first frame, you add [`configs/specs/static.yaml`](../../configs/specs/static.yaml) at the end of `configs` of such a model (or exp).
- The command line arguments are the outermost layer of configuration.
  - Note that special entries are separately defined as `--config` and `--type` for the entry point (shorthand: `-c` and `-t`).
- It's also possible to use multiple parent configs (defined by the `configs` key or separated by `,` in the entrypoint arguments for `-c`)
  - Those parent configs will be parsed layer by layer, the latter overwritting the previous if duplicated keys are found. 


### Reusing the Configuration System

This is [a piece of code](../../easyvolcap/engine/__init__.py) that will be called everytime we import a core module from ***EasyVolcap***:

```python
parser = get_parser()
args, argv = parser.parse_known_args()  # commandline arguments
argv = [v.strip('-') for v in argv]  # arguments starting with -- will not automatically go to the ops dict, need to parse them again
argv = parser.parse_args(argv)  # the reason for -- arguments is that almost all shell completion requires a prefix for optional arguments
args.opts.update(argv.opts)
cfg = parse_cfg(args)
```

Generally, there are two ways to use the configuration system:

#### Direct Usage

The first way is to directly use our registration and commandline entry point.
This requires you to `from easyvolcap.engine import cfg`, which performs argument parsing and config loading.

```shell
# Run this
evc-gui not_exist.help=ok

# Invoke pdbr and check for the parameter
(Pdbr) cfg.not_exist
{'help': 'ok'}
(Pdbr) cfg.not_exist.help
'ok'
(Pdbr)
```

As long as there are import commands from ***EasyVolcap*** which implicitly calls `from easyvolcap.engine import cfg`, we will parse and store the arguments in the global variable `cfg`.
Those import commands typically includes network & system modules like `easyvolcap.runners`, `easyvolcap.models`, `easyvolcap.dataloaders`, `easyvolcap.engine`, etc.
And exclude the `easyvolcap.utils` modules, which is used for utility functions and classes.


#### Building on Top

Sometimes you would want to build your own command line argument parser, or you want to use the configuration system in a different way.
There's only **one** rule-of-thumb: *parse your arguments before importing `__init__.py` from `easyvolcap.engine` as stated in the previous section*

You could even use them in tandem like this:
```python
# fmt: off
import sys

# To use my own parser and EasyVolcap's dict based parser together
try:
    sep_ind = sys.argv.index('--')
    our_args = sys.argv[1:sep_ind]
    evv_args = sys.argv[sep_ind + 1:]
    sys.argv = [sys.argv[0]] + evv_args
except ValueError as e:
    pass # skip if no -- is present

# My own argument parser
args = dotdict(
    test_arg='hello',
    store_true=False,
)

args = args.update(vars(build_parser(args).parse_args()))

print(args) # will output {'test_arg': 'hello', 'store_true': False}

# Will implicitly import from easyvolcap.engine, thus parse args
from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer

# fmt: on
```
