# Contribution Guide for ***EasyVolcap***

## General Formulation

*Copied from the [main readme](../../readme.md#expanding--customizing-easyvolcap).*

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

## How Do We Usually Go About Creating & Reimplementing New Algorithms?

## Making Contributions Public
