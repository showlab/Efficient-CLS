# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_ocdet_config(cfg):
    """
    Add config for densepose head.
    """
    _C = cfg

    _C.DATA_DIR = "datasets/"
    _C.MODEL.DENSEPOSE_ON = True

    _C.REPLAY = CN()
    _C.REPLAY.IMS_PER_STEP = 32
    _C.REPLAY.MEM_PER_CLASS = 5
    _C.REPLAY.APPLY_AUG = False

    _C.SSOD = CN()
    _C.SSOD.ORACLE_IMS_PER_STEP = 2
    _C.SSOD.PSEUDO_IMS_PER_STEP = 2
    _C.SSOD.PSEUDO_TRAINING = False
    _C.SSOD.PSEUDO_MODEL = 'teacher'
    _C.SSOD.PSEUDO_WARMUP_ITERS = 0
    _C.SSOD.PSEUDO_SCORE_THRESH = 0.7
    _C.SSOD.PSEUDO_NMS_THRESH = 0.5
    _C.SSOD.TRAIN_STRONG_AUG = False
    _C.SSOD.REPLAY_STRONG_AUG = False
    _C.SSOD.AUG_TYPES = ['geometric', 'color']
    _C.SSOD.FL_GAMMA = 1.5
    _C.SSOD.LAMBDA_U = 0.5
    _C.SSOD.EMA_ALPHA = 0.996