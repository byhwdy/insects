
import sys
sys.path.append('/paddle/PaddleDetection')
sys.path.append('.')

import paddle.fluid as fluid
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.modeling.architectures import yolov3 
from ppdet.modeling.backbones import darknet 
from ppdet.modeling.anchor_heads import yolo_head 

yml_file = 'configs/insects/yolov3_darknet.yml'
# yml_file = 'configs/yolov3_mobilenet_v1_fruit.yml'
cfg = load_config(yml_file)

main_arch = cfg.architecture
model = create(main_arch)
print(vars(model.yolo_head.yolo_loss))
print(vars(model.yolo_head.nms))
