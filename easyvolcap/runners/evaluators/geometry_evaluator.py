"""
This file may compare metrics on mesh and volume data
"""

from easyvolcap.engine import EVALUATORS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict


@EVALUATORS.register_module()
class GeometryEvaluator:
    def evaluate(self, output: dotdict, batch: dotdict):
        return dotdict()

    def summarize(self):
        return dotdict()
