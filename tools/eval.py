import logging
FORMAT = '%(asctime)s - <%(name)s> - %(levelname)s: %(message)s'
logging.basicConfig(filename='./log/log.txt', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

import sys
sys.path.append('/paddle/insects')

import os
def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

import paddle.fluid as fluid

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.eval_utils import parse_fetches, eval_run, eval_results, eval_json_results
import ppdet.utils.checkpoint as checkpoint
from ppdet.utils.check import check_gpu, check_version
from ppdet.utils.cli import ArgsParser
from ppdet.data.reader import create_reader

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

    # json评价模式
    if FLAGS.json_file:
        logger.info("start evalute in json mode")
        dataset = cfg.EvalReader['dataset']
        if FLAGS.dataset == 'train':
            dataset = cfg.TrainReader['dataset']
        eval_json_results(FLAGS.json_file, 
            dataset=dataset, num_classes=cfg.num_classes)
        return

    ## 模型
    model = create(main_arch)   ####
    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs_def = cfg.EvalReader['inputs_def']
            feed_vars, loader = model.build_inputs(**inputs_def)
            fetches = model.eval(feed_vars)
    eval_prog = eval_prog.clone(True)
    extra_keys = ['gt_bbox', 'gt_class', 'is_difficult']
    keys, values, _ = parse_fetches(fetches, eval_prog, extra_keys)

    ## 执行器
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    ## 数据
    reader = create_reader(cfg.EvalReader)  ####
    loader.set_sample_list_generator(reader, place)

    #### 运行 ####
    exe.run(startup_prog)
    ## 加载参数
    assert 'weights' in cfg, \
           'model can not load weights'
    checkpoint.load_params(exe, eval_prog, cfg.weights)

    ## 评价
    results = eval_run(exe, eval_prog, loader, keys, values)
    eval_results(results, cfg.num_classes)

if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "-f",
        "--json_file",
        default=None,
        type=str,
        help="Evaluation file directory, default is current directory.")
    parser.add_argument(
        "-d",
        "--dataset",
        default='val',
        type=str,
        help="which dataset to use, support train/val")
    FLAGS = parser.parse_args()
    main()
