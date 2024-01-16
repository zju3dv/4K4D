import os
from os.path import join
import cv2
import numpy as np

from easymocap.config.baseconfig import load_object, Config
from easymocap.bodymodel.smpl import SMPLModel


def load_bodymodel(data_root, exp_file=None, device='cpu'):
    from __editable___myeasymocap_0_0_0_finder import MAPPING
    if exp_file:
        cfg_exp = Config.load(join(data_root, exp_file))
    else:
        cfg_exp = Config.load(data_root)
    cfg_model = cfg_exp.args.at_final.load_body_model
    myeasymocap_path = MAPPING['myeasymocap']
    cfg_model.args.model_path = join(myeasymocap_path, '..', cfg_model.args.model_path)
    cfg_model.args.regressor_path = join(myeasymocap_path, '..', cfg_model.args.regressor_path)
    cfg_model.args.device = device
    body_loader = load_object(cfg_model.module, cfg_model.args)
    body_model: SMPLModel = body_loader.smplmodel
    return body_model
