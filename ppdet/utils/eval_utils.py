import numpy as np
import paddle.fluid as fluid

__all__ = ['parse_fetches', 'eval_run']

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





