import numpy as np
from numbers import Integral

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from ppdet.core.workspace import register, serializable

__all__ = ['DropBlock', 'MultiClassNMS', 'ConvNorm', 'MultiClassSoftNMS']

def ConvNorm(input,
             num_filters,
             filter_size,
             stride=1,
             groups=1,
             norm_decay=0.,
             norm_type='affine_channel',
             norm_groups=32,
             dilation=1,
             lr_scale=1,
             freeze_norm=False,
             act=None,
             norm_name=None,
             initializer=None,
             name=None):
    fan = num_filters
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=((filter_size - 1) // 2) * dilation,
        dilation=dilation,
        groups=groups,
        act=None,
        param_attr=ParamAttr(
            name=name + "_weights",
            initializer=initializer,
            learning_rate=lr_scale),
        bias_attr=False,
        name=name + '.conv2d.output.1')

    norm_lr = 0. if freeze_norm else 1.
    pattr = ParamAttr(
        name=norm_name + '_scale',
        learning_rate=norm_lr * lr_scale,
        regularizer=L2Decay(norm_decay))
    battr = ParamAttr(
        name=norm_name + '_offset',
        learning_rate=norm_lr * lr_scale,
        regularizer=L2Decay(norm_decay))

    if norm_type in ['bn', 'sync_bn']:
        global_stats = True if freeze_norm else False
        out = fluid.layers.batch_norm(
            input=conv,
            act=act,
            name=norm_name + '.output.1',
            param_attr=pattr,
            bias_attr=battr,
            moving_mean_name=norm_name + '_mean',
            moving_variance_name=norm_name + '_variance',
            use_global_stats=global_stats)
        scale = fluid.framework._get_var(pattr.name)
        bias = fluid.framework._get_var(battr.name)
    elif norm_type == 'gn':
        out = fluid.layers.group_norm(
            input=conv,
            act=act,
            name=norm_name + '.output.1',
            groups=norm_groups,
            param_attr=pattr,
            bias_attr=battr)
        scale = fluid.framework._get_var(pattr.name)
        bias = fluid.framework._get_var(battr.name)
    elif norm_type == 'affine_channel':
        scale = fluid.layers.create_parameter(
            shape=[conv.shape[1]],
            dtype=conv.dtype,
            attr=pattr,
            default_initializer=fluid.initializer.Constant(1.))
        bias = fluid.layers.create_parameter(
            shape=[conv.shape[1]],
            dtype=conv.dtype,
            attr=battr,
            default_initializer=fluid.initializer.Constant(0.))
        out = fluid.layers.affine_channel(
            x=conv, scale=scale, bias=bias, act=act)
    if freeze_norm:
        scale.stop_gradient = True
        bias.stop_gradient = True
    return out


def DropBlock(input, block_size, keep_prob, is_test):
    if is_test:
        return input

    def CalculateGamma(input, block_size, keep_prob):
        input_shape = fluid.layers.shape(input)
        feat_shape_tmp = fluid.layers.slice(input_shape, [0], [3], [4])
        feat_shape_tmp = fluid.layers.cast(feat_shape_tmp, dtype="float32")
        feat_shape_t = fluid.layers.reshape(feat_shape_tmp, [1, 1, 1, 1])
        feat_area = fluid.layers.pow(feat_shape_t, factor=2)

        block_shape_t = fluid.layers.fill_constant(
            shape=[1, 1, 1, 1], value=block_size, dtype='float32')
        block_area = fluid.layers.pow(block_shape_t, factor=2)

        useful_shape_t = feat_shape_t - block_shape_t + 1
        useful_area = fluid.layers.pow(useful_shape_t, factor=2)

        upper_t = feat_area * (1 - keep_prob)
        bottom_t = block_area * useful_area
        output = upper_t / bottom_t
        return output

    gamma = CalculateGamma(input, block_size=block_size, keep_prob=keep_prob)
    input_shape = fluid.layers.shape(input)
    p = fluid.layers.expand_as(gamma, input)

    input_shape_tmp = fluid.layers.cast(input_shape, dtype="int64")
    random_matrix = fluid.layers.uniform_random(
        input_shape_tmp, dtype='float32', min=0.0, max=1.0)
    one_zero_m = fluid.layers.less_than(random_matrix, p)
    one_zero_m.stop_gradient = True
    one_zero_m = fluid.layers.cast(one_zero_m, dtype="float32")

    mask_flag = fluid.layers.pool2d(
        one_zero_m,
        pool_size=block_size,
        pool_type='max',
        pool_stride=1,
        pool_padding=block_size // 2)
    mask = 1.0 - mask_flag

    elem_numel = fluid.layers.reduce_prod(input_shape)
    elem_numel_m = fluid.layers.cast(elem_numel, dtype="float32")
    elem_numel_m.stop_gradient = True

    elem_sum = fluid.layers.reduce_sum(mask)
    elem_sum_m = fluid.layers.cast(elem_sum, dtype="float32")
    elem_sum_m.stop_gradient = True

    output = input * mask * elem_numel_m / elem_sum_m
    return output

@register
@serializable
class MultiClassNMS(object):
    __op__ = fluid.layers.multiclass_nms
    __append_doc__ = True

    def __init__(self,
                 score_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 nms_threshold=.5,
                 normalized=False,
                 nms_eta=1.0,
                 background_label=0):
        super(MultiClassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.background_label = background_label


@register
@serializable
class MultiClassSoftNMS(object):
    def __init__(
            self,
            score_threshold=0.01,
            keep_top_k=300,
            softnms_sigma=0.5,
            normalized=False,
            background_label=0, ):
        super(MultiClassSoftNMS, self).__init__()
        self.score_threshold = score_threshold
        self.keep_top_k = keep_top_k
        self.softnms_sigma = softnms_sigma
        self.normalized = normalized
        self.background_label = background_label

    def __call__(self, bboxes, scores):
        def create_tmp_var(program, name, dtype, shape, lod_level):
            return program.current_block().create_var(
                name=name, dtype=dtype, shape=shape, lod_level=lod_level)

        def _soft_nms_for_cls(dets, sigma, thres):
            """soft_nms_for_cls"""
            dets_final = []
            while len(dets) > 0:
                maxpos = np.argmax(dets[:, 0])
                dets_final.append(dets[maxpos].copy())
                ts, tx1, ty1, tx2, ty2 = dets[maxpos]
                scores = dets[:, 0]
                # force remove bbox at maxpos
                scores[maxpos] = -1
                x1 = dets[:, 1]
                y1 = dets[:, 2]
                x2 = dets[:, 3]
                y2 = dets[:, 4]
                eta = 0 if self.normalized else 1
                areas = (x2 - x1 + eta) * (y2 - y1 + eta)
                xx1 = np.maximum(tx1, x1)
                yy1 = np.maximum(ty1, y1)
                xx2 = np.minimum(tx2, x2)
                yy2 = np.minimum(ty2, y2)
                w = np.maximum(0.0, xx2 - xx1 + eta)
                h = np.maximum(0.0, yy2 - yy1 + eta)
                inter = w * h
                ovr = inter / (areas + areas[maxpos] - inter)
                weight = np.exp(-(ovr * ovr) / sigma)
                scores = scores * weight
                idx_keep = np.where(scores >= thres)
                dets[:, 0] = scores
                dets = dets[idx_keep]
            dets_final = np.array(dets_final).reshape(-1, 5)
            return dets_final

        def _soft_nms(bboxes, scores):
            bboxes = np.array(bboxes)
            scores = np.array(scores)
            class_nums = scores.shape[-1]

            softnms_thres = self.score_threshold
            softnms_sigma = self.softnms_sigma
            keep_top_k = self.keep_top_k

            cls_boxes = [[] for _ in range(class_nums)]
            cls_ids = [[] for _ in range(class_nums)]

            start_idx = 1 if self.background_label == 0 else 0
            for j in range(start_idx, class_nums):
                inds = np.where(scores[:, j] >= softnms_thres)[0]
                scores_j = scores[inds, j]
                rois_j = bboxes[inds, j, :]
                dets_j = np.hstack((scores_j[:, np.newaxis], rois_j)).astype(
                    np.float32, copy=False)
                cls_rank = np.argsort(-dets_j[:, 0])
                dets_j = dets_j[cls_rank]

                cls_boxes[j] = _soft_nms_for_cls(
                    dets_j, sigma=softnms_sigma, thres=softnms_thres)
                cls_ids[j] = np.array([j] * cls_boxes[j].shape[0]).reshape(-1,
                                                                           1)

            cls_boxes = np.vstack(cls_boxes[start_idx:])
            cls_ids = np.vstack(cls_ids[start_idx:])
            pred_result = np.hstack([cls_ids, cls_boxes])

            # Limit to max_per_image detections **over all classes**
            image_scores = cls_boxes[:, 0]
            if len(image_scores) > keep_top_k:
                image_thresh = np.sort(image_scores)[-keep_top_k]
                keep = np.where(cls_boxes[:, 0] >= image_thresh)[0]
                pred_result = pred_result[keep, :]

            res = fluid.LoDTensor()
            res.set_lod([[0, pred_result.shape[0]]])
            if pred_result.shape[0] == 0:
                pred_result = np.array([[1]], dtype=np.float32)
            res.set(pred_result, fluid.CPUPlace())

            return res

        pred_result = create_tmp_var(
            fluid.default_main_program(),
            name='softnms_pred_result',
            dtype='float32',
            shape=[6],
            lod_level=1)
        fluid.layers.py_func(
            func=_soft_nms, x=[bboxes, scores], out=pred_result)
        return pred_result
