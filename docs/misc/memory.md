### Common Debugging Techniques

```shell
python -c "import torch;print(torch.rand(3, 3, device='cuda'))" # Test whether torch and cuda works
python -c 'print("\x1b[?25h", end="")' # Print the show cursor escape sequence in case cursor is missing
python -c "from easyvolcap.utils.egl_utils import create_opengl_context; create_opengl_context()" # Test EGL context creation (works on a barebone conda environment)

python tests/online_param_update_tests.py # Test whether it's OK to update parameter's size
python tests/headless_opengl_tests.py # Test whether it's OK to render without a display using EGL

# Using GDB to debug a python program that throws at C++ level `gdb -ex "catch throw" -ex run --args`
gdb -ex "catch throw" -ex run --args python scripts/main.py -t gui -c configs/exps/enerf_fuzhizhi.yaml exp_name=enerf_zjumocap val_dataloader_cfg.dataset_cfg.frame_sample=0,1,1 val_dataloader_cfg.dataset_cfg.ratio=0.5
gdb -ex "catch throw" -ex run --args python -c "from easyvolcap.utils.egl_utils import create_opengl_context; create_opengl_context()"

# Show kernel calls (user kernel boundaries for isolating the bug)
strace -o ok.txt python -c "from easyvolcap.utils.egl_utils import create_opengl_context; create_opengl_context()" # store results to ok.txt

# Compare directories
diff --brief --recursive ~/miniconda3/envs/easyvolcap/lib/python3.10/site-packages/OpenGL ~/miniconda3/lib/python3.9/site-packages/OpenGL > diff.txt
git diff ~/miniconda3/envs/easyvolcap/lib/python3.10/site-packages/OpenGL ~/miniconda3/lib/python3.9/site-packages/OpenGL

# Fix weird ipython behavior
# https://github.com/ipython/ipython/issues/13993
echo "c.TerminalInteractiveShell.autosuggestions_provider = 'AutoSuggestFromHistory'" >> $HOME/.ipython/profile_default/ipython_config.py
```

```python
# Memory usage analysis
from pytorch_memlab import MemReporter
reporter = MemReporter(self)
reporter.report()

# Save model graph
from torchviz import make_dot
dot = make_dot(loss, dict(self.model.network.named_parameters()), show_attrs=True, show_saved=True)
dot.render('r4dv')
```
