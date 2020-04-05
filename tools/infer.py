import os
import glob

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


def get_save_image_name(output_dir, image_path):
    """
    Get save image name from source image path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name = os.path.split(image_path)[-1]
    name, ext = os.path.splitext(image_name)
    return os.path.join(output_dir, "{}".format(name)) + ext


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

    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def main():
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
    test_images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
    dataset.set_images(test_images)
    imid2path = dataset.get_imid2path()
    catid2name = dataset.get_cid2cname()

    # 执行器
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

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

    # 数据
    reader = create_reader(cfg.TestReader)
    loader.set_sample_list_generator(reader, place)

    # 运行
    exe.run(startup_prog)

    # 加载权重
    if cfg.weights:
        checkpoint.load_params(exe, infer_prog, cfg.weights)

    # use tb-paddle to log image
    if FLAGS.use_tb:
        from tb_paddle import SummaryWriter
        tb_writer = SummaryWriter(FLAGS.tb_log_dir)
        tb_image_step = 0
        tb_image_frame = 0  # each frame can display ten pictures at most. 

    for iter_id, data in enumerate(loader()):
        outs = exe.run(infer_prog,
                       feed=data,
                       fetch_list=values,
                       return_numpy=False)
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(keys, outs)
        }
        
        logger.info('Infer iter {}'.format(iter_id))

        bbox_results = None
        if 'bbox' in res:
            bbox_results = bbox2out([res])

        # visualize result
        im_ids = res['im_id'][0]
        for im_id in im_ids:
            image_path = imid2path[int(im_id)]
            image = Image.open(image_path).convert('RGB')

            # use tb-paddle to log original image           
            if FLAGS.use_tb:
                original_image_np = np.array(image)
                tb_writer.add_image(
                    "original/frame_{}".format(tb_image_frame),
                    original_image_np,
                    tb_image_step,
                    dataformats='HWC')

            image = visualize_results(image,
                                      int(im_id), catid2name,
                                      FLAGS.draw_threshold, bbox_results)

            # use tb-paddle to log image with bbox
            if FLAGS.use_tb:
                infer_image_np = np.array(image)
                tb_writer.add_image(
                    "bbox/frame_{}".format(tb_image_frame),
                    infer_image_np,
                    tb_image_step,
                    dataformats='HWC')
                tb_image_step += 1
                if tb_image_step % 10 == 0:
                    tb_image_step = 0
                    tb_image_frame += 1

            save_name = get_save_image_name(FLAGS.output_dir, image_path)
            logger.info("Detection bbox results save in {}".format(save_name))
            image.save(save_name, quality=95)


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--use_tb",
        type=bool,
        default=False,
        help="whether to record the data to Tensorboard.")
    parser.add_argument(
        '--tb_log_dir',
        type=str,
        default="tb_log_dir/image",
        help='Tensorboard logging directory for image.')
    FLAGS = parser.parse_args()
    main()
