import sys
sys.path.append('/paddle/insects')

from ppdet.core.workspace import load_config, merge_config, create

cfg = load_config('configs/test.yml')
# merge_config(FLAGS.opt)
# print(cfg)
# myobj = create('MyClass')
# myobj.run()

model = create('YOLOv3')
print(type(model))