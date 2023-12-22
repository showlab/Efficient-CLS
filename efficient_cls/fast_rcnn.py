# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict, List, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn import giou_loss, smooth_l1_loss

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, _log_classification_stats


logger = logging.getLogger(__name__)


class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.cfg = cfg

    def losses(self, predictions, proposals, ann_types=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        # image level -> instance level
        cls_weights = [1.0 if x == 'oracle' else self.cfg.SSOD.LAMBDA_U for x in ann_types]
        reg_weights = [1.0 if x == 'oracle' else self.cfg.SSOD.LAMBDA_U for x in ann_types]  # for pseudo, ignore regression loss

        cls_weights = torch.cat([torch.full([len(p)], v) for p, v in zip(proposals, cls_weights)]).to(scores.device)
        reg_weights = torch.cat([torch.full([len(p)], v) for p, v in zip(proposals, reg_weights)]).to(scores.device)

        losses = {
            "loss_cls": self.cls_loss(scores, gt_classes, cls_weights),  # cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes, reg_weights
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes, weights=None):
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        anchors = [proposal_boxes[fg_inds]]
        pred_anchor_deltas = [fg_pred_deltas.unsqueeze(0)]
        gt_boxes = [gt_boxes[fg_inds]]
        fg_mask = ...

        if isinstance(anchors[0], Boxes):
            anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
        else:
            anchors = cat(anchors)
        assert self.box_reg_loss_type == "smooth_l1"
        gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
        loss_box_reg = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[fg_mask],
            gt_anchor_deltas[fg_mask],
            beta=self.smooth_l1_beta,
            reduction="none",
        )
        if weights is not None:
            loss_box_reg = loss_box_reg.sum(dim=[0, 2]) * weights[fg_inds]
        loss_box_reg = loss_box_reg.sum()

        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # max(torch.count_nonzero(weights), 1.0)  # return 0 if empty

    def cls_loss(self, scores, gt_classes, weights=None):
        total_loss = cross_entropy(scores, gt_classes, reduction="none")
        if weights is not None:
            total_loss = total_loss * weights
        # total_loss = total_loss.mean()

        return total_loss.mean()    # / max(torch.count_nonzero(weights), 1.0)

    def comput_focal_loss(self, scores, gt_classes, weights=None):
        FC_loss = FocalLoss(
            gamma=self.cfg.SSOD.FL_GAMMA,
            num_classes=self.num_classes,
        )
        total_loss = FC_loss(input=scores, target=gt_classes, weight=weights)
        total_loss = total_loss / gt_classes.shape[0]

        return total_loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target, weight=None):
        # focal loss
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum()