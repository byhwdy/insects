import logging
import numpy as np

import paddle.fluid as fluid

__all__ = ['nms']

logger = logging.getLogger(__name__)

def nms(dets, thresh):
    """Apply classic DPM-style greedy NMS."""
    if dets.shape[0] == 0:
        return []
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    # nominal indices
    # _i, _j
    # sorted indices
    # i, j
    # temp variables for box i's (the box currently under consideration)
    # ix1, iy1, ix2, iy2, iarea

    # variables for computing overlap with box j (lower scoring box)
    # xx1, yy1, xx2, yy2
    # w, h
    # inter, ovr

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return np.where(suppressed == 0)[0]


def bbox_area(box):
    w = box[2] - box[0] + 1
    h = box[3] - box[1] + 1
    return w * h


def bbox_overlaps(x, y):
    N = x.shape[0]
    K = y.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        y_area = bbox_area(y[k])
        for n in range(N):
            iw = min(x[n, 2], y[k, 2]) - max(x[n, 0], y[k, 0]) + 1
            if iw > 0:
                ih = min(x[n, 3], y[k, 3]) - max(x[n, 1], y[k, 1]) + 1
                if ih > 0:
                    x_area = bbox_area(x[n])
                    ua = x_area + y_area - iw * ih
                    overlaps[n, k] = iw * ih / ua
    return overlaps
