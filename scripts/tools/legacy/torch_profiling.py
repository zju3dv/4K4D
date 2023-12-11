# fmt: off
import sys
sys.argv = sys.argv[1:] + ['profiling.enabled', 'True']
sys.path.append('.')
from train import main
main()
# fmt: on

