import os
import glob
import random
import json

import numpy as np
from PIL import Image

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

from ppdet.utils.eval_utils import parse_fetches
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu, check_version
from ppdet.utils.visualizer import visualize_results
import ppdet.utils.checkpoint as checkpoint
from ppdet.utils.voc_eval import bbox2out

from ppdet.data.reader import create_reader
from ppdet.data.source.insects import get_cid2cname

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def get_test_images(infer_dir):
    """
    Get image path list in TEST mode
    """
    assert infer_dir is not None, \
        "--infer_dir should be set"
    assert os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)
    
    images = []

    infer_dir = os.path.abspath(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.extend(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    
    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in {}.".format(len(images), infer_dir))

    return images

def save_json_results(results, imid2name, output_dir):
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
    file_path = os.path.join(output_dir, 'preds.json')
    with open(file_path, 'w') as f:
        json.dump(json_results, f)
    logging.info("saved {}".format(file_path))

def main():
    ## 配置与检查
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
    test_images = get_test_images(FLAGS.infer_dir)
    dataset.set_images(test_images)
    imid2name = {k: str(os.path.basename(v).split('.')[0])
                 for k, v in dataset.get_imid2path().items()}

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
    for iter_id, data in enumerate(loader()):
        outs = exe.run(infer_prog, feed=data, fetch_list=values, return_numpy=False) 
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(keys, outs)
        }
        logger.info('Infer iter {}'.format(iter_id))

        results.append(res)
    save_json_results(results, imid2name, FLAGS.output_dir)

if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    FLAGS = parser.parse_args()
    main()
