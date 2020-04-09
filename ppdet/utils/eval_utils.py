import os
import json
import numpy as np
import paddle.fluid as fluid

from .map_utils import DetectionMAP

__all__ = ['parse_fetches', 'eval_run'
           'eval_results', 'eval_json_results', 'bbox2out']

import logging
logger = logging.getLogger(__name__)


def parse_fetches(fetches, prog=None, extra_keys=None):
    """
    Parse fetch variable infos from model fetches,
    values for fetch_list and keys for stat
    """
    keys, values = [], []
    cls = []
    for k, v in fetches.items():
        if hasattr(v, 'name'):
            keys.append(k)
            #v.persistable = True
            values.append(v.name)
        else:
            cls.append(v)

    if prog is not None and extra_keys is not None:
        for k in extra_keys:
            try:
                v = fluid.framework._get_var(k, prog)
                keys.append(k)
                values.append(v.name)
            except Exception:
                pass

    return keys, values, cls

def eval_run(exe, compile_program, loader, keys, values):
    """
    Run evaluation program, return program outputs.
    """
    iter_id = 0
    images_num = 0
    results = []

    try:
        loader.start()
        while True:
            outs = exe.run(compile_program, fetch_list=values, return_numpy=False) 
            res = {
                k: (np.array(v), v.recursive_sequence_lengths())
                for k, v in zip(keys, outs)}
            results.append(res)

            if iter_id % 50 == 0:
                logger.info('Infer iter {}'.format(iter_id))
            iter_id += 1
            images_num += len(res['bbox'][1][0])
            
    except (StopIteration, fluid.core.EOFException):
        loader.reset()

    # 日志
    logger.info('Infer finish iter {}'.format(iter_id))
    logger.info('Infer total number of images: {}'.format(images_num))

    return results

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
    return map_stat


def prune_zero_padding(gt_box, gt_label, difficult=None):
    valid_cnt = 0
    for i in range(len(gt_box)):
        if gt_box[i, 0] == 0 and gt_box[i, 1] == 0 and \
                gt_box[i, 2] == 0 and gt_box[i, 3] == 0:
            break
        valid_cnt += 1
    return (gt_box[:valid_cnt], gt_label[:valid_cnt], difficult[:valid_cnt]
            if difficult is not None else None)


def eval_json_results(json_file, dataset, num_classes,
                      overlap_thresh=0.5,
                      map_type='11point',
                      is_bbox_normalized=False,
                      evaluate_difficult=False):
    """
    评价json结果
    """
    assert os.path.isfile(json_file), \
        "invalid json file"
    with open(json_file, 'r') as f:
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

def bbox2out(results):
    """
    Args:
        results: request a dict, should include: `bbox`, `im_id`,
                 if is_bbox_normalized=True, also need `im_shape`.
    """
    xywh_res = []
    for t in results:
        bboxes = t['bbox'][0]
        lengths = t['bbox'][1][0]
        im_ids = np.array(t['im_id'][0]).flatten()
        if bboxes.shape == (1, 1) or bboxes is None:
            continue

        k = 0
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i])
            for j in range(num):
                dt = bboxes[k]
                clsid, score, xmin, ymin, xmax, ymax = dt.tolist()
                w = xmax - xmin + 1
                h = ymax - ymin + 1

                bbox = [xmin, ymin, w, h]
                coco_res = {
                    'image_id': im_id,
                    'category_id': int(clsid),
                    'bbox': bbox,
                    'score': score
                }
                xywh_res.append(coco_res)
                k += 1
    return xywh_res




