import logging
import numpy as np
import os
import time
import json

import paddle.fluid as fluid

from .voc_eval import bbox_eval as voc_bbox_eval
from .post_process import mstest_box_post_process, mstest_mask_post_process, box_flip
from .map_utils import DetectionMAP

__all__ = ['parse_fetches', 'eval_run', 'eval_results', 'json_eval_results']

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

            if iter_id % 100 == 0:   # ztodo
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
                 num_classes,
                 is_bbox_normalized=False,
                 map_type='11point'):
    """Evaluation for evaluation program results"""
    box_ap = voc_bbox_eval(
        results,
        num_classes,
        is_bbox_normalized=is_bbox_normalized,
        map_type=map_type)

    return [box_ap]


def json_eval_results(json_directory, dataset, num_classes,
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



