import logging
FORMAT = '%(asctime)s - <%(name)s> - %(levelname)s: %(message)s'
logging.basicConfig(filename='./log/log.txt', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

import os
import glob
import random
import numpy as np
from PIL import Image
import json

import sys
sys.path.append('/paddle/insects')

def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)

set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

from paddle import fluid

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.eval_utils import parse_fetches, bbox2out
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu, check_version
from ppdet.utils.visualizer import visualize_results
import ppdet.utils.checkpoint as checkpoint
from ppdet.data.reader import create_reader
from ppdet.data.source.insects import get_cid2cname

def get_test_images(infer_img, infer_dir, random_seed=None, num_img=-1):
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
        logger.info("Loaded img {}.".format(infer_img))
        return images

    infer_dir = os.path.abspath(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.extend(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    if random_seed:
        random.seed(random_seed)
        random.shuffle(images)
    if num_img > 0:
        images = images[:num_img]    

    assert len(images) > 0, "no image found in {}".format(infer_dir)

    logger.info("Found {} inference images in dir {}.".format(
        len(images), infer_dir))

    return images

def get_save_image_name(output_dir, image_path):
    """
    Get save image name from source image path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name = os.path.split(image_path)[-1]
    name, ext = os.path.splitext(image_name)
    return os.path.join(output_dir, "{}".format(name)) + ext

def save_json_results(results, imid2name, output_dir, json_name):
    """
    save results in json format
    """
    json_results = []
    for result in results:
        im_lengths = result['bbox'][1][0]
        start_ind = 0
        for i, im_length in enumerate(im_lengths):
            im = []
            im_name = imid2name[int(result['im_id'][0][i])]
            im.append(im_name)
            bbox = result['bbox'][0][start_ind:start_ind+im_length].tolist()
            im.append(bbox)
            json_results.append(im)
            start_ind += im_length
    file_path = os.path.join(output_dir, json_name)
    with open(file_path, 'w') as f:
        json.dump(json_results, f)
    logging.info("saved {}".format(file_path))

def main():
    ## 配置
    cfg = load_config(FLAGS.config)
    if 'architecture' in cfg:
        main_arch = cfg.architecture
    else:
        raise ValueError("'architecture' not specified in config file.")
    merge_config(FLAGS.opt)
    check_gpu(cfg.use_gpu)
    check_version()

    # 数据源
    dataset = cfg.TestReader['dataset']
    test_images = get_test_images(FLAGS.infer_img, 
        FLAGS.infer_dir, FLAGS.random_seed, FLAGS.num_img)
    dataset.set_images(test_images)
    imid2path = dataset.get_imid2path()
    catid2name = get_cid2cname()
    imid2name = {k: str(os.path.basename(v).split('.')[0])
                 for k, v in imid2path.items()}

    # 模型
    model = create(main_arch)
    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs_def = cfg['TestReader']['inputs_def']
            inputs_def['iterable'] = True
            feed_vars, loader = model.build_inputs(**inputs_def)
            test_fetches = model.test(feed_vars)
    infer_prog = infer_prog.clone(True)
    extra_keys = ['im_id']
    keys, values, _ = parse_fetches(test_fetches, infer_prog, extra_keys)

    # 执行器
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # 数据
    reader = create_reader(cfg.TestReader)
    loader.set_sample_list_generator(reader, place)

    #### 运行 ####
    exe.run(startup_prog)

    # 加载权重
    assert 'weights' in cfg, \
           'model can not load weights'        
    checkpoint.load_params(exe, infer_prog, cfg.weights)

    results = []
    im_ids = []
    for iter_id, data in enumerate(loader()):
        outs = exe.run(infer_prog, feed=data, fetch_list=values, return_numpy=False)
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(keys, outs)
        }
        logger.info('Infer iter {}'.format(iter_id))

        results.append(res)
        im_ids.extend([int(id) for id in res['im_id'][0]])
    
    ## 输出json结果
    if FLAGS.json_mode:
        save_json_results(results, imid2name, FLAGS.output_dir, FLAGS.json_name)
        return 

    # 输出画图结果
    bbox_results = bbox2out(results)
    for im_id in im_ids:
        image_path = imid2path[im_id]
        image = Image.open(image_path).convert('RGB')

        image = visualize_results(image,
                                  int(im_id), catid2name,
                                  FLAGS.draw_threshold, bbox_results)

        save_name = get_save_image_name(FLAGS.output_dir, image_path)
        logger.info("Detection bbox results save in {}".format(save_name))
        image.save(save_name, quality=95)



if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "--json_mode",
        action="store_true",
        default=False,
        help="Infer image to json result")
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="when infer in dir, select img randomly at seed")
    parser.add_argument(
        "--num_img",
        type=int,
        default=-1,
        help="when infer in dir, number of img selected")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization or json files.")
    parser.add_argument(
        "--json_name",
        type=str,
        default="prediction.json",
        help="Json files name.")
    FLAGS = parser.parse_args()
    main()
