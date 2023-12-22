# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import json
import pdb
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances, Boxes, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from rpn import build_proposal_generator
from roi_heads import build_roi_heads

from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.utils.visualizer import Visualizer
from utils import vis_pseudo

__all__ = ["GeneralizedRCNN"]


def distillation(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y / T), F.softmax(teacher_scores / T)) * (
                T * T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


def _log_pseudo_stats():
    pass


# @META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        cfg,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "cfg": cfg,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # Calculate sample weights
        # weights = [1.0 if x['ann_type'] == 'oracle' else self.cfg.SSOD.LAMBDA_U for x in batched_inputs]
        ann_types = [x['ann_type'] for x in batched_inputs]

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, ann_types)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, ann_types)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def predict_logits_with_gtboxes(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        features = self.backbone(images.tensor)
        proposal_boxes = [x.gt_boxes for x in gt_instances]
        box_features = self.roi_heads._shared_roi_transform(
            [features[f] for f in self.roi_heads.in_features], proposal_boxes
        )
        logits, _ = self.roi_heads.box_predictor(box_features.mean(dim=[2, 3]))

        return logits

    @torch.no_grad()
    def generate_pseudo_label(self, batched_inputs: List[Dict[str, torch.Tensor]], phase='train'):
        self.eval()
        assert not self.training
        storage = get_event_storage()

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        proposals, _ = self.proposal_generator(images, features, None)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.roi_heads._shared_roi_transform(
            [features[f] for f in self.roi_heads.in_features], proposal_boxes
        )
        predictions = self.roi_heads.box_predictor(box_features.mean(dim=[2, 3]))

        boxes = self.roi_heads.box_predictor.predict_boxes(predictions, proposals)
        scores = self.roi_heads.box_predictor.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        results, _ = fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            score_thresh=self.cfg.SSOD.PSEUDO_SCORE_THRESH,
            nms_thresh=self.cfg.SSOD.PSEUDO_NMS_THRESH,
            topk_per_image=100,
        )

        for input_per_image, results_per_image in zip(batched_inputs, results):
            pseudo_boxes, pseudo_classes = results_per_image.pred_boxes, results_per_image.pred_classes
            gt_boxes, gt_classes = input_per_image['instances'].gt_boxes, input_per_image['instances'].gt_classes
            gt_boxes, gt_classes = gt_boxes.to(pseudo_boxes.device), gt_classes.to(pseudo_classes.device)
            num_pseudo, num_gt = len(pseudo_classes), len(gt_classes)
            storage.put_scalar("pseudo-labeling/num_pseudo", num_pseudo)
            storage.put_scalar("pseudo-labeling/num_gt", num_gt)
            if num_pseudo == 0 and num_gt == 0:
                miou = 1.0
                box_accuracy = 1.0
            elif num_pseudo == 0 or num_gt == 0:
                miou = 0.0
                box_accuracy = 0.0
            else:
                ious = pairwise_iou(pseudo_boxes, gt_boxes)
                matched_ious, matched_inds = torch.max(ious, dim=1)
                matched_gt_classes = gt_classes[matched_inds]
                matched_gt_classes[matched_ious < 0.5] = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
                miou = float(matched_ious.mean())
                box_accuracy = (pseudo_classes == matched_gt_classes).nonzero().numel() / pseudo_classes.numel()
            storage.put_scalar("pseudo-labeling/box_accuracy", box_accuracy)
            storage.put_scalar("pseudo-labeling/miou", miou)

        pseudo_outputs = []
        for input_per_image, results_per_image in zip(batched_inputs, results):
            pseudo_per_image = input_per_image.copy()
            pseudo_per_image.pop('image')
            pseudo_per_image.pop('instances')
            ori_height, ori_width = input_per_image['height'], input_per_image['width']
            cur_height, cur_width = results_per_image.image_size
            scales = (ori_height / cur_height, ori_width / cur_width)

            anns = []
            for bbox, cls_id in zip(results_per_image.pred_boxes.tensor.detach(),
                                    results_per_image.pred_classes.detach()):
                ori_bbox = [float(bbox[0] * scales[1]),
                            float(bbox[1] * scales[0]),
                            float(bbox[2] * scales[1]),
                            float(bbox[3] * scales[0])]
                anns.append({
                    'bbox': ori_bbox,
                    'bbox_mode': 0,
                    'category_id': int(cls_id),
                })
            pseudo_per_image['annotations'] = anns
            pseudo_outputs.append(pseudo_per_image)

        self.train()
        assert self.training

        return pseudo_outputs

    @torch.no_grad()
    def generate_proposals_and_predictions(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)

        proposals = self.roi_heads.label_and_sample_proposals(proposals, gt_instances)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.roi_heads._shared_roi_transform(
            [features[f] for f in self.roi_heads.in_features], proposal_boxes
        )
        predictions = self.roi_heads.box_predictor(box_features.mean(dim=[2, 3]))

        return proposals, predictions

    @torch.no_grad()
    def generate_uncertainty_scores(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        self.eval()
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)

        proposals = [x[:512] for x in proposals]
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.roi_heads._shared_roi_transform(
            [features[f] for f in self.roi_heads.in_features], proposal_boxes
        )
        predictions = self.roi_heads.box_predictor(box_features.mean(dim=[2, 3]))
        boxes = self.roi_heads.box_predictor.predict_boxes(predictions, proposals)
        scores = self.roi_heads.box_predictor.predict_probs(predictions, proposals)

        uncertainty_scores = []
        for score, box in zip(scores, boxes):
            s = - (score * torch.log(score)).sum()
            uncertainty_scores.append(float(s))

        self.train()
        assert self.training
        return uncertainty_scores

    def generate_ts_discrepancy(self, batched_inputs: List[Dict[str, torch.Tensor]], model_teacher):
        self.eval()
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        with torch.no_grad():
            try:
                proposals, teacher_predictions = model_teacher.generate_proposals_and_predictions(batched_inputs)
            except:
                proposals, teacher_predictions = model_teacher.module.generate_proposals_and_predictions(batched_inputs)
        _, proposal_losses = self.proposal_generator(images, features, None)  # TODO: not necessary?

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.roi_heads._shared_roi_transform(
            [features[f] for f in self.roi_heads.in_features], proposal_boxes
        )
        predictions = self.roi_heads.box_predictor(box_features.mean(dim=[2, 3]))

        dists = torch.linalg.norm(predictions[0] - teacher_predictions[0], dim=-1)
        scores = []
        start_i = 0
        for p in proposals:
            scores.append(dists[start_i:start_i+len(p)].sum())
            start_i += len(p)

        self.train()
        assert self.training
        return scores

    def forward_with_distillation(self, batched_inputs: List[Dict[str, torch.Tensor]], model_teacher):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # Calculate sample weights
        weights = [1 if x['ann_type'] == 'oracle' else self.cfg.SSOD.LAMBDA_U for x in batched_inputs]

        features = self.backbone(images.tensor)
        with torch.no_grad():
            proposals, teacher_predictions = model_teacher.module.generate_proposals_and_predictions(batched_inputs)
        _, proposal_losses = self.proposal_generator(images, features, gt_instances)    # TODO: not necessary?

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.roi_heads._shared_roi_transform(
            [features[f] for f in self.roi_heads.in_features], proposal_boxes
        )
        predictions = self.roi_heads.box_predictor(box_features.mean(dim=[2, 3]))

        detector_losses = self.roi_heads.box_predictor.losses(predictions, proposals, weights)
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        distill_losses = {'distill_loss_cls': distillation(predictions[0], gt_classes, teacher_predictions[0], T=20, alpha=0.7)}

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(distill_losses)
        return losses


