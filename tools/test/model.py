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

model = create('ResNet')
print(model)
# print(vars(model.yolo_head.yolo_loss))
# print(vars(model.yolo_head.nms))
