import json
import os
import numpy as np

from .map_utils import DetectionMAP

import logging
logger = logging.getLogger(__name__)

__all__ = ['eval_results', 'eval_json_results']

def eval_results(results,
                 class_num,
                 overlap_thresh=0.5,
                 map_type='11point',
                 is_bbox_normalized=False,
                 evaluate_difficult=False):
    """
    Bounding box evaluation for VOC dataset

    Args:
        results (list): prediction bounding box results.
        class_num (int): evaluation class number.
                        bbox overlap
        map_type (string): method for mAP calcualtion,
                        can only be '11point' or 'integral'
        is_bbox_normalized (bool): whether bbox is normalized
                        to range [0, 1].
        evaluate_difficult (bool): whether to evaluate
                        difficult gt bbox.
    """
    logger.info("Start evaluate...")

    detection_map = DetectionMAP(
        class_num=class_num,
        overlap_thresh=overlap_thresh,
        map_type=map_type,
        is_bbox_normalized=is_bbox_normalized,
        evaluate_difficult=evaluate_difficult)

    for b in results:
        bboxes = b['bbox'][0]
        bbox_lengths = b['bbox'][1][0]
        gt_boxes = b['gt_bbox'][0]
        gt_labels = b['gt_class'][0]
        difficults = b['is_difficult'][0] if not evaluate_difficult \
                            else None

        bbox_idx = 0
        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i]
            gt_label = gt_labels[i]
            difficult = None if difficults is None \
                            else difficults[i]
            bbox_num = bbox_lengths[i]
            bbox = bboxes[bbox_idx:bbox_idx + bbox_num]
            gt_box, gt_label, difficult = prune_zero_padding(
                gt_box, gt_label, difficult)
            detection_map.update(bbox, gt_box, gt_label, difficult)
            bbox_idx += bbox_num

    logger.info("Accumulating evaluatation results...")
    detection_map.accumulate()
    map_stat = 100. * detection_map.get_map()
    logger.info("mAP({:.2f}, {}) = {:.2f}".format(overlap_thresh, map_type,
                                                  map_stat))
    return [map_stat]


def prune_zero_padding(gt_box, gt_label, difficult=None):
    valid_cnt = 0
    for i in range(len(gt_box)):
        if gt_box[i, 0] == 0 and gt_box[i, 1] == 0 and \
                gt_box[i, 2] == 0 and gt_box[i, 3] == 0:
            break
        valid_cnt += 1
    return (gt_box[:valid_cnt], gt_label[:valid_cnt], difficult[:valid_cnt]
            if difficult is not None else None)


def eval_json_results(json_directory, dataset, num_classes,
                      overlap_thresh=0.5,
                      map_type='11point',
                      is_bbox_normalized=False,
                      evaluate_difficult=False):
    """
    评价json结果
    """
    assert os.path.isfile(json_directory), \
        "invalid json file"
    with open(json_directory, 'r') as f:
        results = json.load(f)

    records = dataset.get_roidb()
    records_dict = {}
    for record in records:
        k = os.path.basename(record['im_file']).split('.')[0]
        records_dict[k] = record

    detection_map = DetectionMAP(
        class_num=num_classes,
        overlap_thresh=overlap_thresh,
        map_type=map_type,
        is_bbox_normalized=is_bbox_normalized,
        evaluate_difficult=evaluate_difficult)

    logger.info("Start evaluate...")
    for im in results:
        bbox = np.array(im[1])
        record = records_dict[im[0]]
        gt_box = record['gt_bbox']
        gt_label = record['gt_class']
        difficult = record['difficult']
        detection_map.update(bbox, gt_box, gt_label, difficult)

    logger.info("Accumulating evaluatation results...")
    detection_map.accumulate()
    map_stat = 100. * detection_map.get_map()
    logger.info("mAP({:.2f}, {}) = {:.2f}".format(overlap_thresh, map_type,
                                                  map_stat))
    return map_stat