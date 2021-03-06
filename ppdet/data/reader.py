import copy
import functools
import collections
import traceback
import numpy as np
import logging

from ppdet.core.workspace import register, serializable

from .parallel_map import ParallelMap
from .transform.batch_operators import Gt2YoloTarget

__all__ = ['Reader', 'create_reader']

logger = logging.getLogger(__name__)


class Compose(object):
    def __init__(self, transforms, ctx=None):
        self.transforms = transforms
        self.ctx = ctx

    def __call__(self, data):
        ctx = self.ctx if self.ctx else {}
        for f in self.transforms:
            try:
                data = f(data, ctx)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.info("fail to map op [{}] with error: {} and stack:\n{}".
                            format(f, e, str(stack_info)))
                raise e
        return data

def _has_empty(item):
    def empty(x):
        if isinstance(x, np.ndarray) and x.size == 0:
            return True
        elif isinstance(x, collections.Sequence) and len(x) == 0:
            return True
        else:
            return False

    if isinstance(item, collections.Sequence) and len(item) == 0:
        return True
    if item is None:
        return True
    if empty(item):
        return True
    return False


def _segm(samples):
    assert 'gt_poly' in samples
    segms = samples['gt_poly']
    if 'is_crowd' in samples:
        is_crowd = samples['is_crowd']
        if len(segms) != 0:
            assert len(segms) == is_crowd.shape[0]

    gt_masks = []
    valid = True
    for i in range(len(segms)):
        segm = segms[i]
        gt_segm = []
        if 'is_crowd' in samples and is_crowd[i]:
            gt_segm.append([[0, 0]])
        else:
            for poly in segm:
                if len(poly) == 0:
                    valid = False
                    break
                gt_segm.append(np.array(poly).reshape(-1, 2))
        if (not valid) or len(gt_segm) == 0:
            break
        gt_masks.append(gt_segm)
    return gt_masks


def batch_arrange(batch_samples, fields):
    def im_shape(samples, dim=3):
        # hard code
        assert 'h' in samples
        assert 'w' in samples
        if dim == 3:  # RCNN, ..
            return np.array((samples['h'], samples['w'], 1), dtype=np.float32)
        else:  # YOLOv3, ..
            return np.array((samples['h'], samples['w']), dtype=np.int32)

    arrange_batch = []
    for samples in batch_samples:
        one_ins = ()
        for i, field in enumerate(fields):
            if field == 'im_size':
                one_ins += (im_shape(samples, 2), )
            else:
                if field == 'is_difficult':
                    field = 'difficult'
                assert field in samples, '{} not in samples'.format(field)
                one_ins += (samples[field], )
        arrange_batch.append(one_ins)
    return arrange_batch


@register
@serializable
class Reader(object):
    """
    Args:
        dataset (DataSet): DataSet object
        sample_transforms (list of BaseOperator): a list of sample transforms
            operators.
        batch_transforms (list of BaseOperator): a list of batch transforms
            operators.
        batch_size (int): batch size.
        shuffle (bool): whether shuffle dataset or not. Default False.
        drop_last (bool): whether drop last batch or not. Default False.
        drop_empty (bool): whether drop sample when it's gt is empty or not.
            Default True.
        mixup_epoch (int): mixup epoc number. Default is -1, meaning
            not use mixup.
        worker_num (int): number of working threads/processes.
            Default -1, meaning not use multi-threads/multi-processes.
        use_process (bool): whether use multi-processes or not.
            It only works when worker_num > 1. Default False.
        bufsize (int): buffer size for multi-threads/multi-processes,
            please note, one instance in buffer is one batch data.
        memsize (str): size of shared memory used in result queue when
            use_process is true. Default 3G.
        inputs_def (dict): network input definition use to get input fields,
            which is used to determine the order of returned data.
    """

    def __init__(self,
                 dataset=None,
                 sample_transforms=None,
                 batch_transforms=None,
                 batch_size=None,
                 shuffle=False,
                 drop_last=False,
                 drop_empty=True,
                 mixup_epoch=-1,
                 worker_num=-1,
                 use_process=False,
                 use_fine_grained_loss=False,
                 num_classes=80,
                 bufsize=100,
                 memsize='3G',
                 inputs_def=None):
        self._dataset = dataset
        self._roidbs = self._dataset.get_roidb()

        self._fields = copy.deepcopy(inputs_def[
            'fields']) if inputs_def else None

        # transform
        self._sample_transforms = Compose(sample_transforms,
                                          {'fields': self._fields})
        self._batch_transforms = None
        if use_fine_grained_loss:
            for bt in batch_transforms:
                if isinstance(bt, Gt2YoloTarget):
                    bt.num_classes = num_classes
        elif batch_transforms:
            batch_transforms = [
                bt for bt in batch_transforms
                if not isinstance(bt, Gt2YoloTarget)
            ]
        if batch_transforms:
            self._batch_transforms = Compose(batch_transforms,
                                             {'fields': self._fields})

        # data
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._drop_empty = drop_empty

        # sampling
        self._mixup_epoch = mixup_epoch

        self._load_img = False
        self._sample_num = len(self._roidbs)

        self._indexes = None
        self._pos = -1
        self._epoch = -1

        # multi-process
        self._worker_num = worker_num
        self._parallel = None
        if self._worker_num > -1:
            task = functools.partial(self.worker, self._drop_empty)
            self._parallel = ParallelMap(self, task, worker_num, bufsize,
                                         use_process, memsize)

    def __call__(self):
        if self._worker_num > -1:
            return self._parallel
        else:
            return self

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._epoch < 0:
            self.reset()
        if self.drained():
            raise StopIteration
        batch = self._load_batch()
        if self._drop_last and len(batch) < self._batch_size:
            raise StopIteration
        if self._worker_num > -1:
            return batch
        else:
            return self.worker(self._drop_empty, batch)

    def _load_batch(self):
        batch = []
        bs = 0
        while bs != self._batch_size:
            if self._pos >= self.size():
                break
            pos = self.indexes[self._pos]
            sample = copy.deepcopy(self._roidbs[pos])
            self._pos += 1

            if self._drop_empty and self._fields and 'gt_bbox' in self._fields:
                if _has_empty(sample['gt_bbox']):
                    #logger.warn('gt_bbox {} is empty or not valid in {}, '
                    #   'drop this sample'.format(
                    #    sample['im_file'], sample['gt_bbox']))
                    continue

            if self._load_img:
                sample['image'] = self._load_image(sample['im_file'])

            if self._epoch < self._mixup_epoch:
                num = len(self.indexes)
                mix_idx = np.random.randint(1, num)
                mix_idx = self.indexes[(mix_idx + self._pos - 1) % num]
                sample['mixup'] = copy.deepcopy(self._roidbs[mix_idx])
                if self._load_img:
                    sample['mixup']['image'] = self._load_image(sample['mixup'][
                        'im_file'])

            batch.append(sample)
            bs += 1
        return batch

    def worker(self, drop_empty=True, batch_samples=None):
        """
        sample transform and batch transform.
        """
        batch = []
        for sample in batch_samples:
            sample = self._sample_transforms(sample)
            if drop_empty and 'gt_bbox' in sample:
                if _has_empty(sample['gt_bbox']):
                    #logger.warn('gt_bbox {} is empty or not valid in {}, '
                    #   'drop this sample'.format(
                    #    sample['im_file'], sample['gt_bbox']))
                    continue
            batch.append(sample)
        if len(batch) > 0 and self._batch_transforms:
            batch = self._batch_transforms(batch)
        if len(batch) > 0 and self._fields:
            batch = batch_arrange(batch, self._fields)
        return batch

    def reset(self):
        """implementation of Dataset.reset
        """
        self.indexes = [i for i in range(self.size())]

        if self._shuffle:
            np.random.shuffle(self.indexes)

        if self._mixup_epoch > 0 and len(self.indexes) < 2:
            logger.info("Disable mixup for dataset samples "
                        "less than 2 samples")
            self._mixup_epoch = -1

        if self._epoch < 0:
            self._epoch = 0
        else:
            self._epoch += 1

        self._pos = 0

    def _load_image(self, filename):
        with open(filename, 'rb') as f:
            return f.read()

    def size(self):
        """ implementation of Dataset.size
        """
        return self._sample_num

    def drained(self):
        """ implementation of Dataset.drained
        """
        assert self._epoch >= 0, 'The first epoch has not begin!'
        return self._pos >= self.size()

    def stop(self):
        if self._parallel:
            self._parallel.stop()


def create_reader(cfg, max_iter=0, global_cfg=None):
    """
    Return iterable data reader.

    Args:
        max_iter (int): number of iterations.
    """
    if not isinstance(cfg, dict):
        raise TypeError("The config should be a dict when creating reader.")

    # synchornize use_fine_grained_loss/num_classes from global_cfg to reader cfg
    if global_cfg:
        cfg['use_fine_grained_loss'] = getattr(global_cfg,
                                               'use_fine_grained_loss', False)
        cfg['num_classes'] = getattr(global_cfg, 'num_classes', 80)
    reader = Reader(**cfg)()

    def _reader():
        n = 0
        while True:
            for _batch in reader:
                if len(_batch) > 0:
                    yield _batch
                    n += 1
                if max_iter > 0 and n == max_iter:
                    return
            reader.reset()
            if max_iter <= 0:
                return

    return _reader
