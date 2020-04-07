import os
import numpy as np


from ppdet.core.workspace import register, serializable

@serializable
class DataSet(object):
    def __init__(self):
        self.roidbs = None

    def load_roidb_and_cname2cid(self):
        """load dataset"""
        raise NotImplementedError('%s.load_roidb_and_cname2cid not available' %
                                  (self.__class__.__name__))

    def get_roidb(self):
        if not self.roidbs:
            self.load_roidb_and_cname2cid()

        return self.roidbs

@register
@serializable
class ImagesFolder(object):
    def __init__(self):
        self.images = None
        self.roidbs = None
        self._imid2path = {}

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

            self._imid2path[ct] = image
            rec = {
                'im_id': np.array([ct]), 
                'im_file': image
            }
            records.append(rec)
            ct += 1

        assert len(records) > 0, "No image file found"

        return records