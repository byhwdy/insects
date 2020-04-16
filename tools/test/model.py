import sys
sys.path.append('/paddle/PaddleDetection')
sys.path.append('.')

import paddle.fluid as fluid
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.modeling.architectures import yolov3 
from ppdet.modeling.backbones import darknet 
from ppdet.modeling.anchor_heads import yolo_head 

yml_file = 'configs/yolov3_r50vd_dcn_v3.yml'
cfg = load_config(yml_file)

model = create('ResNet')

im = fluid.data(name='image',
            	shape=[None, 3, None, None],
            	dtype='float32',
            	lod_level=0)
out = model(im)
print(fluid.default_main_program())
place = fluid.CPUPlace()
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())

fluid.io.save_inference_model(
    "./paddle_r50vd_dcn_model",
    feeded_var_names=[im.name],
    target_vars=list(out.values()),
    executor=exe)

