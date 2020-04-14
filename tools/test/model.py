import sys
sys.path.append('/paddle/PaddleDetection')
sys.path.append('.')

import paddle.fluid as fluid
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.modeling.architectures import yolov3 
from ppdet.modeling.backbones import darknet 
from ppdet.modeling.anchor_heads import yolo_head 

yml_file = 'configs/yolov3_resnet.yml'
cfg = load_config(yml_file)

backbone = create('ResNet')

im = fluid.data(name='image',
            	shape=[None, 3, None, None],
            	dtype='float32',
            	lod_level=0)
out = backbone(im)
print(fluid.default_main_program());exit()
print(out)
# print(vars(model.yolo_head.yolo_loss))
# print(vars(model.yolo_head.nms))

