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

