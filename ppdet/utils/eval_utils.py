import logging
import numpy as np
import os
import time

import paddle.fluid as fluid

from .voc_eval import bbox_eval as voc_bbox_eval
from .post_process import mstest_box_post_process, mstest_mask_post_process, box_flip

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


def json_eval_results(metric, json_directory=None, dataset=None):
    """
    cocoapi eval with already exists proposal.json, bbox.json or mask.json
    """
    assert metric == 'COCO'
    from ppdet.utils.coco_eval import cocoapi_eval
    anno_file = dataset.get_anno()
    json_file_list = ['proposal.json', 'bbox.json', 'mask.json']
    if json_directory:
        assert os.path.exists(
            json_directory), "The json directory:{} does not exist".format(
                json_directory)
        for k, v in enumerate(json_file_list):
            json_file_list[k] = os.path.join(str(json_directory), v)

    coco_eval_style = ['proposal', 'bbox', 'segm']
    for i, v_json in enumerate(json_file_list):
        if os.path.exists(v_json):
            cocoapi_eval(v_json, coco_eval_style[i], anno_file=anno_file)
        else:
            logger.info("{} not exists!".format(v_json))
