import os
import numpy as np
import xml.etree.ElementTree as ET
from collections import Sequence

from .dataset import DataSet
from ppdet.core.workspace import register, serializable

import logging
logger = logging.getLogger(__name__)

# 昆虫名称列表
INSECT_NAMES = ['Boerner', 'Leconte', 'Linnaeus', 
                'acuminatus', 'armandi', 'coleoptera', 'linnaeus']

def get_cname2cid():
    """昆虫cname2cid
    """
    insect_category2id = {}
    for i, item in enumerate(INSECT_NAMES):
        insect_category2id[item] = i

    return insect_category2id

def get_cid2cname():
    """昆虫cid2cname
    """
    insect_id2category = {}
    for i, item in enumerate(INSECT_NAMES):
        insect_id2category[i] = item

    return insect_id2category

@register
@serializable
class InsectsDataSet(DataSet):
    def __init__(self, 
                 dataset_dir, 
                 anno_dir):
        '''
        Args:
            dataset_dir (str): 数据集根目录
            anno_dir (str): 标注文件目录
        '''
        super(InsectsDataSet, self).__init__()

        self.dataset_dir = dataset_dir
        self.anno_dir = anno_dir

    def load_roidb_and_cname2cid(self):
        '''昆虫records
        '''
        cname2cid = get_cname2cid()
        datadir = os.path.join(self.dataset_dir, self.anno_dir)

        filenames = os.listdir(os.path.join(datadir, 'annotations', 'xmls'))
        records = []
        ct = 0
        for fname in filenames:
            fid = fname.split('.')[0]
            im_file = os.path.join(datadir, 'images', fid + '.jpeg')  # im_file
            im_id = np.array([ct])  # im_id

            fpath = os.path.join(datadir, 'annotations', 'xmls', fname)
            tree = ET.parse(fpath)

            im_w = float(tree.find('size').find('width').text)  # im_w
            im_h = float(tree.find('size').find('height').text)  # im_h

            objs = tree.findall('object')
            gt_bbox = np.zeros((len(objs), 4), dtype=np.float32) # gt_bbox
            gt_class = np.zeros((len(objs), 1), dtype=np.int32) # gt_class
            gt_score = np.zeros((len(objs), 1), dtype=np.float32) # gt_score
            difficult = np.zeros((len(objs), 1), dtype=np.int32) # difficult
            for i, obj in enumerate(objs):
                cname = obj.find('name').text
                gt_class[i][0] = cname2cid[cname]
                x1 = float(obj.find('bndbox').find('xmin').text)
                y1 = float(obj.find('bndbox').find('ymin').text)
                x2 = float(obj.find('bndbox').find('xmax').text)
                y2 = float(obj.find('bndbox').find('ymax').text)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(im_w - 1, x2)
                y2 = min(im_h - 1, y2)
                gt_bbox[i] = [x1, y1, x2, y2]
                gt_score[i][0] = 1.
                difficult[i][0]  = int(obj.find('difficult').text)
            
            voc_rec = {
                'im_file': im_file,
                'im_id': im_id,
                'h': im_h,
                'w': im_w,
                'gt_class': gt_class,
                'gt_bbox': gt_bbox,
                'gt_score': gt_score,
                'difficult': difficult
            }
            if len(objs) != 0:
                records.append(voc_rec)
            ct += 1

        logger.info('{} samples in file {}'.format(len(records), 
            os.path.join(datadir, 'annotations', 'xmls')))

        self.roidbs = records

@register
@serializable
class InsectsImages(object):
    def __init__(self, sample_num=-1):
        self.sample_num = sample_num

        self.images = None
        self.roidbs = None
        self._imid2path = {}
        self._cid2cname = get_cid2cname()

    def get_cid2cname(self):
        return self._cid2cname

    def get_imid2path(self):
        return self._imid2path

    # 模型输入
    def get_roidb(self):
        if not self.roidbs:
            self.roidbs = self._load_images()
        return self.roidbs

    def set_images(self, images):
        self.images = images
        self.roidbs = self._load_images()

    def _load_images(self):
        images = self.images
        
        ct = 0
        records = []
        for image in images:
            assert image != '' and os.path.isfile(image), \
                    "Image {} not found".format(image)
            if self.sample_num > 0 and ct >= self.sample_num:
                break

            self._imid2path[ct] = image
            rec = {
                'im_id': np.array([ct]), 
                'im_file': image
            }
            records.append(rec)
            ct += 1

        assert len(records) > 0, "No image file found"

        return records