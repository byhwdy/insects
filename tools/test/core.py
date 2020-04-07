import sys
sys.path.append('/paddle/insects')

from ppdet.core.workspace import load_config, merge_config, create
from test import Person

# cfg = load_config('configs/yolov3_darknet_insects.yml')
cfg = load_config('configs/test.yml')
# print(cfg)
# cfg.myclass.run()
myobj = create(Person)
myobj.run()

# model = create('YOLOv3')
# print(type(model))