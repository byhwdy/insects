import os
import glob
from argparse import ArgumentParser
import random

import numpy as np
from PIL import Image

import sys
sys.path.append('/paddle/insects')

from paddle import fluid

from ppdet.core.workspace import load_config, merge_config, create

from ppdet.utils.eval_utils import parse_fetches
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu, check_version
from ppdet.utils.visualizer import visualize_results
import ppdet.utils.checkpoint as checkpoint
from ppdet.utils.voc_eval import bbox2out

from ppdet.data.reader import create_reader
from ppdet.data.source.insects import get_cid2cname, get_bbox_out

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
    return os.path.join(output_dir, "{}_gt".format(name)) + ext


def get_test_images(infer_dir, infer_img, random_seed=None):
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
    if random_seed:
        random.seed(random_seed)
        random.shuffle(images)    
    assert len(images) > 0, "no image found in {}".format(infer_dir)

    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def main():
    # 数据源
    catid2name = get_cid2cname()
    test_images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img, FLAGS.random_seed)
    test_images = test_images[:FLAGS.sample_num]
    bbox_out = get_bbox_out(test_images)

    for im_id, im_path in enumerate(test_images):
        image = Image.open(im_path).convert('RGB')

        image = visualize_results(image,
                                  int(im_id), catid2name,
                                  FLAGS.draw_threshold, bbox_out)

        save_name = get_save_image_name(FLAGS.output_dir, im_path)
        image.save(save_name, quality=95)

        logger.info("Detection bbox results save in {}".format(save_name))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--sample_num",
        type=int,
        default=1,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1024,
        help="Image path, has higher priority over --rondom_seed")
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
    FLAGS = parser.parse_args()
    main()
