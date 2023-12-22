#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
import torchvision.transforms
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
# from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.structures import Boxes, Instances, pairwise_iou

import json
import time
import copy
import math
import shutil
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.utils.data as torchdata
import detectron2.data.transforms as T
from PIL import Image
from multiprocessing import Pipe, Manager, Lock
from detectron2.data import DatasetCatalog, get_detection_dataset_dicts
from detectron2.data.samplers import TrainingSampler
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.common import DatasetFromList, MapDataset

import sys
sys.path.insert(1, './efficient_cls')
from efficient_cls.build import build_model
from efficient_cls.memory import Memory, RandomMemory
from efficient_cls.config import add_ocdet_config
from efficient_cls.dataset_mapper import PseudoAugDatasetMapper
from efficient_cls.ts_ensemble import EnsembleTSModel


logger = logging.getLogger("detectron2")


def get_dataset_dicts(json_files, mapping):
    data_dicts = []
    for json_file in json_files:
        assert os.path.exists(json_file)
        with open(json_file, 'r') as f:
            raw_dict = json.load(f)
        annotations = []
        for obj in raw_dict['annotations']:
            assert 'category' in obj
            obj['category_id'] = mapping[obj['category']]
            annotations.append(obj)

        file_name = raw_dict['file_name']
        if not os.path.exists(file_name):
            file_name = json_file.replace('annotations',
                                          'images').replace('.json', os.path.splitext(raw_dict['file_name'])[-1])
        new_dict = {
            'file_name': file_name,
            'height': raw_dict['height'],
            'width': raw_dict['width'],
            'image_id': raw_dict['image_id'],
            'annotations': annotations,
        }
        data_dicts.append(new_dict)

    return data_dicts


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return COCOEvaluator(dataset_name, output_dir=output_folder)


def do_test(cfg, model, iteration, prefix=None):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        output_folder = os.path.join(cfg.OUTPUT_DIR, f'{prefix}_inference' if prefix is not None else 'inference',
                                     "inference_{:07d}".format(iteration), dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, output_folder
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

    return results


@torch.no_grad()
def update_teacher_model(model, model_teacher, keep_rate=0.996):
    student_model_dict = model.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    model_teacher.load_state_dict(new_teacher_dict)


def do_train(cfg, model, model_teacher):
    # For reproducibility, also let diff ranks have the same replay data
    # TODO: might have problem in data augmentation
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)

    # register dataset
    train_df = pd.read_csv(cfg.TRAIN_FILE)
    train_df['json_file'] = train_df['json_file'].apply(lambda x: os.path.join(cfg.DATA_DIR, x))

    test_df = pd.read_csv(cfg.TEST_FILE)
    test_df['json_file'] = test_df['json_file'].apply(lambda x: os.path.join(cfg.DATA_DIR, x))

    mapping = json.load(open(cfg.MAPPING_FILE, 'r'))

    DatasetCatalog.register(cfg.DATASETS.TEST[0], lambda: get_dataset_dicts(test_df['json_file'], mapping))
    MetadataCatalog.get(cfg.DATASETS.TEST[0]).evaluator_type = 'coco'
    MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes = list(mapping.keys())

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    scheduler = build_lr_scheduler(cfg, optimizer)

    ensem_ts_model = EnsembleTSModel(model_teacher, model)
    checkpointer = DetectionCheckpointer(
        ensem_ts_model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    max_step = len(sorted(pd.unique(train_df['step_name'])))
    start_step = 0

    if cfg.MEMORY_TYPE == 'balanced':
        memory = Memory(cfg.REPLAY.MEM_PER_CLASS, cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    elif cfg.MEMORY_TYPE == 'random':
        memory = RandomMemory(cfg.REPLAY.MEM_PER_CLASS, cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    else:
        raise NotImplementedError(cfg.MEMORY_TYPE)

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_step)
    writers = default_writers(cfg.OUTPUT_DIR, max_step) if comm.is_main_process() else []

    default_transform = [
        T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    ]
    pseudo_transform = [
        T.RandomCrop(crop_type='relative', crop_size=(0.9, 0.9)),
        T.RandomRotation([-5, 5]),
        T.RandomFlip(prob=0.5, horizontal=True),
    ] + default_transform


    with EventStorage(start_step) as storage:
        for step_iter in range(start_step, max_step):
            storage.iter = step_iter
            step_name = 'step_{:05d}'.format(step_iter)
            # Teacher student update with EMA.
            logger.info(f"[Step {step_iter}] EMA update with alpha={cfg.SSOD.EMA_ALPHA}")
            update_teacher_model(model, model_teacher, keep_rate=cfg.SSOD.EMA_ALPHA)
            # Load training data.
            step_df = train_df[train_df['step_name'] == step_name]
            if 'oracle' in step_df.columns:
                train_list_oracle = step_df[step_df['oracle'] == 1]['json_file'].to_list()
                train_list_pseudo = step_df[step_df['oracle'] == 0]['json_file'].to_list()
            else:
                train_list = step_df['json_file'].to_list()
                train_list_oracle = sorted(random.sample(train_list, min(len(train_list), cfg.SSOD.ORACLE_IMS_PER_STEP)))
                train_list_pseudo = sorted(set(train_list) - set(train_list_oracle))
            train_data_oracle = get_dataset_dicts(train_list_oracle, mapping)
            if cfg.SSOD.ORACLE_IMS_PER_STEP == 16:
                train_data_oracle += get_dataset_dicts(train_list_pseudo, mapping)
            train_data_pseudo = []
            if cfg.SSOD.PSEUDO_IMS_PER_STEP > 0 and len(train_list_pseudo) > 0 and step_iter > cfg.SSOD.PSEUDO_WARMUP_ITERS:
                logger.info(f"[Step {step_iter}] Pseudo labeling {len(train_list_pseudo)} train images.")
                dataset = DatasetFromList(get_dataset_dicts(train_list_pseudo, mapping))
                dataset = MapDataset(dataset, DatasetMapper(cfg, is_train=True, augmentations=default_transform))
                data_loader = torchdata.DataLoader(dataset, batch_size=4, collate_fn=lambda x: x)
                # generate pseudo label with Teacher model.
                train_data_pseudo = []
                vis = False # True if step_iter % 10 == 0 else False # cfg.TEST.EVAL_PERIOD
                with torch.no_grad():
                    for i, data in enumerate(data_loader):
                        if comm.get_world_size() > 1:
                            train_data_pseudo += \
                                model_teacher.module.generate_pseudo_label(data, phase='train_teacher')
                        else:
                            train_data_pseudo += \
                                model_teacher.generate_pseudo_label(data, phase='train_teacher')
                logger.info(f"[Step {step_iter}] Sampling {min(len(train_data_pseudo), cfg.SSOD.PSEUDO_IMS_PER_STEP)}/{len(train_data_pseudo)} pseudo images.")
                train_data_pseudo = random.sample(train_data_pseudo, min(len(train_data_pseudo), cfg.SSOD.PSEUDO_IMS_PER_STEP))
            for x in train_data_oracle:
                x['ann_type'] = 'oracle'
            for x in train_data_pseudo:
                x['ann_type'] = 'pseudo'
            train_data = train_data_oracle + train_data_pseudo      
            replay_data = memory.retrieve(max_num=cfg.REPLAY.IMS_PER_STEP, samples_per_class=2)
            # print('REPLAY DATA - {}, {}'.format(comm.get_rank(), [x['image_id'] for x in replay_data]))

            n = min(len(train_data_oracle), cfg.MAX_UPDATE)
            logger.info(f"[Step {step_iter}] Updating {n} oracle images to memory.")
            if cfg.SSOD.ORACLE_IMS_PER_STEP == 16:
                memory.update(random.sample(train_data_oracle[:n], n))
            else:
                memory.update(random.sample(train_data_oracle, n))

            # compared to "train_net.py", we do not support accurate timing and
            # precise BN here, because they are not trivial to implement in a small training loop

            logger.info(f"[Step {step_iter}] {len(train_data)} train images, {len(replay_data)} replay images.")
            assert model.training
            logger.info(f"[Step {step_iter}] Starting training {len(train_data)} images.")
            data_loader = build_detection_train_loader(
                dataset=train_data, sampler=TrainingSampler(len(train_data), shuffle=True), total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
                mapper=PseudoAugDatasetMapper(cfg, is_train=True, augmentations=default_transform, pseudo_augmentations=pseudo_transform)
            )
            # print('TRAIN DATA - {}, {}'.format(comm.get_rank(), [x['image_id'] for x in train_data]))
            num_iters = (len(train_data) - 1) // cfg.SOLVER.IMS_PER_BATCH + 1
            for data, iteration in zip(data_loader, range(0, num_iters)):
                # print(f"{iteration} / {num_iters}")
                if iteration == num_iters - 1:
                    # print([x['image_id'] for x in data])
                    last_n = len(train_data) - iteration * cfg.SOLVER.IMS_PER_BATCH  # cfg.SOLVER.IMS_PER_BATCH
                    last_n = math.ceil(last_n / comm.get_world_size())
                    data = data[:last_n]
                    if len(data) == 0:
                        break
                # print('TRAIN - {}, {}'.format(comm.get_rank(), [x['image_id'] for x in data]))
                # print([x['ann_type'] for x in data])
                loss_dict = model(data)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                # print('loss:', losses_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            # replaying memory
            if cfg.REPLAY.IMS_PER_STEP > 0 and len(replay_data) > 0:
                if cfg.REPLAY.APPLY_AUG:
                    aug_transform = [
                        T.RandomCrop(crop_type='relative', crop_size=(0.9, 0.9)),
                        T.RandomRotation([-5, 5]),
                        T.RandomFlip(prob=0.5, horizontal=True),
                        T.RandomBrightness(0.5, 1.5),
                        T.RandomContrast(0.8, 1.2),
                    ]
                    aug_transform = random.sample(aug_transform, 3)
                else:
                    aug_transform = []
                data_loader = build_detection_train_loader(
                    dataset=replay_data, sampler=TrainingSampler(len(replay_data), shuffle=False), total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
                    mapper=DatasetMapper(cfg, is_train=True, augmentations=default_transform + aug_transform)
                )
                assert model.training and model_teacher.training
                logger.info(f"[Step {step_iter}] Starting replaying {len(replay_data)} images.")
                for data, iteration in zip(data_loader, range(0, (len(replay_data) - 1) // cfg.SOLVER.IMS_PER_BATCH + 1)):
                    # print('REPLAY - {}, {}'.format(comm.get_rank(), [x['image_id'] for x in data]))
                    # print([len(x['instances']) for x in data])
                    loss_dict = model(data)
                    losses = sum(loss_dict.values())
                    assert torch.isfinite(losses).all(), loss_dict

                    replay_loss_dict_reduced = {k + '(replay)': v.item() for k, v in
                                                comm.reduce_dict(loss_dict).items()}
                    replay_losses_reduced = sum(loss for loss in replay_loss_dict_reduced.values())

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                loss_dict_reduced.update(replay_loss_dict_reduced)

            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and ((step_iter + 1) % cfg.TEST.EVAL_PERIOD == 0 or step_iter == max_step - 1)
            ):
                test_results = do_test(cfg, model_teacher, step_iter)
                if comm.is_main_process():
                    for dataset_name, results_i in test_results.items():
                        res = results_i['bbox']
                        for k, v in res.items():
                            if "-" in k:
                                continue
                            storage.put_scalar(f'{dataset_name}/{k}', v, smoothing_hint=False)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if (
                    (step_iter + 1) % 10 == 0
                    or step_iter == max_step - 1
            ):
                for writer in writers:
                    writer.write()
                logger.info(f'Writing to: {cfg.OUTPUT_DIR}')
            # periodic_checkpointer.step(step_iter)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ocdet_config(cfg)
    print(args.config_file)
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    cfg.TRAIN_FILE = os.path.join(cfg.DATA_DIR, args.dataset, f"train_anno{args.num_oracle}.csv")
    cfg.TEST_FILE = os.path.join(cfg.DATA_DIR, args.dataset, 'test.csv')
    cfg.MAPPING_FILE = os.path.join(cfg.DATA_DIR, args.dataset, 'mapping.json')
    cfg.DATASETS.TRAIN = (f'{args.dataset}_train',)
    cfg.DATASETS.TEST = (f'{args.dataset}_test',)
    cfg.INPUT.MIN_SIZE_TRAIN = 800
    cfg.INPUT.MIN_SIZE_TEST = 800

    mapping = json.load(open(cfg.MAPPING_FILE, 'r'))

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(mapping)
    cfg.CLASS_NAMES = list(mapping.keys())

    cfg.SSOD.ORACLE_IMS_PER_STEP = args.num_oracle
    cfg.SSOD.PSEUDO_IMS_PER_STEP = args.num_pseudo
    cfg.RANDOM_SEED = args.random_seed
    cfg.SSOD.PSEUDO_WARMUP_ITERS = args.pseudo_warmup_iters
    cfg.SSOD.LAMBDA_U = args.lambda_u
    cfg.MAX_UPDATE = args.max_update
    cfg.SSOD.EMA_ALPHA = args.ema_alpha
    cfg.SSOD.PSEUDO_SCORE_THRESH = args.pseudo_thresh
    cfg.LAMBDA_RPN = args.lambda_rpn
    cfg.REPLAY.IMS_PER_STEP = args.replay_size
    cfg.REPLAY.MEM_PER_CLASS = args.memory_size
    cfg.REPLAY.APPLY_AUG = args.replay_aug

    cfg.MEMORY_TYPE = args.memory_type

    os.makedirs(args.save_dir, exist_ok=True)
    cfg.OUTPUT_DIR = args.save_dir + f'/{args.dataset}/{args.exp}/' \
                     f'[{args.exp}]Pseudo-{cfg.SSOD.PSEUDO_IMS_PER_STEP}_{cfg.SSOD.PSEUDO_WARMUP_ITERS}' \
                     f'_C4_VOC_BS{cfg.SOLVER.IMS_PER_BATCH}' \
                     f'_TrainOracle-{cfg.SSOD.ORACLE_IMS_PER_STEP}' \
                     f'_Replay-{cfg.REPLAY.IMS_PER_STEP}' \
                     f'_Memory-{cfg.REPLAY.MEM_PER_CLASS}' \
                     f'_LambdaU-{cfg.SSOD.LAMBDA_U}' \
                     f'_EMA-{cfg.SSOD.EMA_ALPHA}' \
                     f'_ReplayAug-{cfg.REPLAY.APPLY_AUG}' \
                     f'_PseudoThresh-{cfg.SSOD.PSEUDO_SCORE_THRESH}' \
                     f'_Seed-{cfg.RANDOM_SEED}' \
                     f'_MemoryType{args.memory_type}'

    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)
    shutil.copy(os.path.basename(__file__), cfg.OUTPUT_DIR)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # load voc pretrained model (0~19 same classes)
    if args.dataset == 'oak':
        with open(cfg.MODEL.WEIGHTS, 'rb') as f:
            data = pickle.load(f, encoding="latin1")
        w = model.state_dict()
        for k in ['weight', 'bias']:
            w[f'roi_heads.box_predictor.cls_score.{k}'][:20] = \
                torch.FloatTensor(data['model'][f'roi_heads.box_predictor.cls_score.{k}'][:20])
            w[f'roi_heads.box_predictor.cls_score.{k}'][-1:] = \
                torch.FloatTensor(data['model'][f'roi_heads.box_predictor.cls_score.{k}'][-1:])
            w[f'roi_heads.box_predictor.bbox_pred.{k}'][:20 * 4] = \
                torch.FloatTensor(data['model'][f'roi_heads.box_predictor.bbox_pred.{k}'][:20 * 4])
        model.load_state_dict(w)

    model_teacher = build_model(cfg)
    model_teacher.load_state_dict(model.state_dict())
    # Fix teacher's backbone
    for name, parameters in model_teacher.named_parameters():
        parameters.requires_grad = False if ('backbone' in name) else True

    logger.info("Model:\n{}".format(model))

    distributed = comm.get_world_size() > 1
    # assert not distributed, 'distributed training is currently not supported.'
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
        model_teacher = DistributedDataParallel(
            model_teacher, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, model_teacher)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--exp', type=str, default="")
    parser.add_argument('--config', type=str, default="./configs/efficient_cls.yaml")
    parser.add_argument('--save_dir', type=str, default="./outputs/")
    parser.add_argument('--dataset', type=str, default="oak", choices=["oak", "egoobj"])
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_oracle', type=int, default=2)
    parser.add_argument('--num_pseudo', type=int, default=14)
    parser.add_argument('--replay_size', type=int, default=16)
    parser.add_argument('--memory_size', type=int, default=5)
    parser.add_argument('--pseudo_thresh', type=float, default=0.7)
    parser.add_argument('--max_update', type=int, default=4)
    parser.add_argument('--lambda_u', type=float, default=1.0)
    parser.add_argument('--lambda_rpn', type=float, default=0.0)
    parser.add_argument('--ema_alpha', type=float, default=0.99)
    parser.add_argument('--pseudo_warmup_iters', type=int, default=-1)
    parser.add_argument('--memory_type', type=str, default='balanced')
    parser.add_argument('--replay_aug', action='store_true')
    args = parser.parse_args()
    
    args.dist_url = 'auto'
    args.num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
