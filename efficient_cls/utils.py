import copy
import os
import json
import random
import logging
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import ImageFilter, Image
from detectron2.utils.visualizer import Visualizer
from detectron2.data import detection_utils as utils

import matplotlib
matplotlib.use('Agg')

def get_dataset_dicts(json_files, mapping):
    data_dicts = []
    for json_file in json_files:
        if not os.path.exists(json_file):
            json_file = json_file.replace('zhangjie', 'jay')
        raw_dict = json.load(open(json_file, 'r'))
        annotations = []
        for obj in raw_dict['annotations']:
            if 'category' not in obj:
                obj['category_id'] = mapping[obj['category_id']]
            else:
                obj['category_id'] = mapping[obj['category']]
            annotations.append(obj)

        new_dict = {
            'file_name': json_file.replace('annotations',
                                           'images').replace('.json', os.path.splitext(raw_dict['file_name'])[-1]),
            'height': raw_dict['height'],
            'width': raw_dict['width'],
            'image_id': raw_dict['image_id'],
            'annotations': annotations,
        }
        data_dicts.append(new_dict)

    return data_dicts


def save_img(img, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
    else:
        plt.show()


def vis_pseudo(input_per_image, results_per_image, class_names, save_path=None):
    cmap = plt.get_cmap('hsv')
    colors = cmap(np.linspace(0, 1, len(class_names)))
    im_bgr = input_per_image['image'].permute(1, 2, 0).cpu().numpy()
    im_rgb = im_bgr[:, :, ::-1]
    v = Visualizer(im_rgb)
    for bbox, cls_id in zip(input_per_image['instances'].gt_boxes,
                            input_per_image['instances'].gt_classes):
        v.draw_box(bbox.cpu(), edge_color='w', line_style='--')
        v.draw_text(class_names[int(cls_id)], tuple(bbox[:2]), font_size=6, color='w')
    for bbox, score, cls_id in zip(results_per_image.pred_boxes,
                                   results_per_image.scores,
                                   results_per_image.pred_classes):
        v.draw_box(bbox.cpu(), edge_color=colors[int(cls_id)])
        v.draw_text(class_names[int(cls_id)] + ' {:.1f}%'.format(score * 100),
                    tuple(bbox[:2]), font_size=6, color=colors[int(cls_id)])
    save_img(v.get_output().get_image(), save_path)


def vis_oracle(input_per_image, class_names, save_path=None):
    cmap = plt.get_cmap('hsv')
    colors = cmap(np.linspace(0, 1, len(class_names)))
    im_bgr = input_per_image['image'].permute(1, 2, 0).cpu().numpy()
    im_rgb = im_bgr[:, :, ::-1]
    v = Visualizer(im_rgb)
    for bbox, cls_id in zip(input_per_image['instances'].gt_boxes,
                            input_per_image['instances'].gt_classes):
        v.draw_box(bbox.cpu(), edge_color=colors[int(cls_id)], line_style='--')
        v.draw_text(class_names[int(cls_id)], tuple(bbox[:2]), font_size=6, color=colors[int(cls_id)])
    save_img(v.get_output().get_image(), save_path)


class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def build_strong_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

        randcrop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                ),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                ),
                transforms.RandomErasing(
                    p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                ),
                transforms.ToPILImage(),
            ]
        )
        augmentation.append(randcrop_transform)

        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)

