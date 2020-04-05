import os
import glob
import sys
sys.path.append('/paddle/PaddleDetection')
sys.path.append('.')

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.reader import create_reader

def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)
    
    images = []

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        images.append(infer_img)
        return images

    infer_dir = os.path.abspath(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.extend(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    assert len(images) > 0, "no image found in {}".format(infer_dir)

    # logger.info("Found {} inference images in total.".format(len(images)))

    return images


yml_file = 'configs/yolov3_darknet_insects.yml'
# yml_file = 'configs/yolov3_mobilenet_v1_fruit.yml'
cfg = load_config(yml_file)
# print(cfg.TrainReader)
# print(cfg.TestReader)
# exit()

# dataset = cfg.TrainReader['dataset']
# roidbs = dataset.get_roidb()
# print(roidbs[0])
# print(dataset.cname2cid)
# exit()

dataset = cfg.TestReader['dataset']
infer_dir = 'dataset/insects/val/images'
infer_img = 'dataset/insects/val/images/1998.jpeg'
infer_img = None
test_images = get_test_images(infer_dir, infer_img)
dataset.set_images(test_images)
# imid2path = dataset.get_imid2path()
# catid2name = dataset.get_cid2cname()
# roidb = dataset.get_roidb()
# print(imid2path)
# print(catid2name)
# print(roidb)



# train_reader = create_reader(cfg.TrainReader)
# eval_reader = create_reader(cfg.EvalReader)
infer_reader = create_reader(cfg.TestReader)

# batch = next(train_reader())
# batch = next(eval_reader())
batch = next(infer_reader())


print('batch_size: ', len(batch))
print('one_smaple: ')
one_smaple = batch
print(one_smaple)
# print(one_smaple[0].shape)
# print(type(batch))
# print(len(batch))
# print(type(batch[0]))
# print(len(batch[0]))
# print(batch[0][1])
# print(batch[0][2])