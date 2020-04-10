import logging
FORMAT = '%(asctime)s - <%(name)s> - %(levelname)s: %(message)s'
logging.basicConfig(filename='./log/log.txt', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

import os
import time
import datetime
from collections import deque
import numpy as np

from paddle import fluid
from visualdl import LogWriter

import sys
sys.path.append('/paddle/insects')
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.reader import create_reader
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu, check_version
from ppdet.utils.eval_utils import parse_fetches, eval_run, eval_results
from ppdet.utils.stats import TrainingStats
import ppdet.utils.checkpoint as checkpoint


def main():
    # 配置
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    if 'architecture' in cfg:
        main_arch = cfg.architecture
    else:
        raise ValueError("'architecture' not specified in config file.")
    check_gpu(cfg.use_gpu)
    check_version()

    # 执行器
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # 模型
    lr_builder = create('LearningRate')
    optim_builder = create('OptimizerBuilder')
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            model = create(main_arch)
            inputs_def = cfg.TrainReader['inputs_def']
            feed_vars, train_loader = model.build_inputs(**inputs_def)
            train_fetches = model.train(feed_vars)
            loss = train_fetches['loss']
            lr = lr_builder()
            optimizer = optim_builder(lr)
            optimizer.minimize(loss)
    train_keys, train_values, _ = parse_fetches(train_fetches)
    train_values.append(lr)
    if FLAGS.eval:
        eval_prog = fluid.Program()
        with fluid.program_guard(eval_prog, startup_prog):
            with fluid.unique_name.guard():
                model = create(main_arch)
                inputs_def = cfg.EvalReader['inputs_def']
                feed_vars, eval_loader = model.build_inputs(**inputs_def)
                fetches = model.eval(feed_vars)
        eval_prog = eval_prog.clone(True)
        extra_keys = ['gt_bbox', 'gt_class', 'is_difficult']
        eval_keys, eval_values, _ = parse_fetches(fetches, eval_prog, extra_keys)
        eval_reader = create_reader(cfg.EvalReader)
        eval_loader.set_sample_list_generator(eval_reader, place)

    ##### 运行 ####
    exe.run(startup_prog)

    ## 恢复与迁移
    ignore_params = cfg.finetune_exclude_pretrained_params \
                 if 'finetune_exclude_pretrained_params' in cfg else []
    start_iter = 0
    if FLAGS.resume_checkpoint:
        checkpoint.load_checkpoint(exe, train_prog, FLAGS.resume_checkpoint)
        start_iter = checkpoint.global_step()
    elif cfg.pretrain_weights:
        checkpoint.load_params(
            exe, train_prog, cfg.pretrain_weights, ignore_params=ignore_params)

    ## 数据迭代器
    train_reader = create_reader(cfg.TrainReader, cfg.max_iters - start_iter, cfg)
    train_loader.set_sample_list_generator(train_reader, place)

    ## 训练循环
    train_loader.start()

    # 过程跟踪
    train_stats = TrainingStats(cfg.log_smooth_window, train_keys)
    start_time = time.time()
    end_time = time.time()
    time_stat = deque(maxlen=cfg.log_smooth_window)
    cfg_name = os.path.basename(FLAGS.config).split('.')[0]
    save_dir = os.path.join(cfg.save_dir, cfg_name)
    best_box_ap_list = [0.0, 0]
    if FLAGS.use_vdl:
        log_writter = LogWriter(FLAGS.vdl_dir, sync_cycle=5)
        with log_writter.mode("train") as vdl_logger:
            scalar_loss = vdl_logger.scalar(tag="loss")
        with log_writter.mode("val") as vdl_logger:
            scalar_map = vdl_logger.scalar(tag="map")

    for it in range(start_iter, cfg.max_iters):
        # 运行程序
        outs = exe.run(train_prog, fetch_list=train_values)
        stats = {k: np.array(v).mean() for k, v in zip(train_keys, outs[:-1])}
        
        # 日志与可视化窗口
        start_time = end_time
        end_time = time.time()
        time_stat.append(end_time - start_time)
        time_cost = np.mean(time_stat)
        eta_sec = (cfg.max_iters - it) * time_cost
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        train_stats.update(stats)
        logs = train_stats.log()
        if it % cfg.log_iter == 0:
            # log
            strs = 'iter: {}, lr: {:.6f}, {}, time: {:.3f}, eta: {}'.format(
                it, np.mean(outs[-1]), logs, time_cost, eta)
            logger.info(strs)
            # vdl
            if FLAGS.use_vdl:
                scalar_loss.add_record(it//cfg.log_iter, stats['loss'])

        # 保存与评价窗口
        if (it > 0 and it % cfg.snapshot_iter == 0 or it == cfg.max_iters - 1):
            # 模型保存
            save_name = str(it) if it != cfg.max_iters - 1 else "final"
            checkpoint.save(exe, train_prog, os.path.join(save_dir, save_name))

            # 评价
            if FLAGS.eval:
                results = eval_run(exe, eval_prog, eval_loader,
                                   eval_keys, eval_values)
                box_ap_stats = eval_results(results, cfg.num_classes)

                ## 保存最佳模型
                if box_ap_stats > best_box_ap_list[0]:
                    best_box_ap_list[0] = box_ap_stats
                    best_box_ap_list[1] = it
                    checkpoint.save(exe, train_prog, os.path.join(save_dir, "best_model"))
                    logger.info("Best eval box ap: {}, in iter: {}".format(
                        best_box_ap_list[0], best_box_ap_list[1]))

                ## 记录map窗口
                if FLAGS.use_vdl:
                    step = it//cfg.snapshot_iter if it % cfg.snapshot_iter == 0 \
                           else it//cfg.snapshot_iter+1 
                    scalar_map.add_record(step, box_ap_stats)

    train_loader.reset()


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        default=None,
        type=str,
        help="Checkpoint path for resuming training.")
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")
    parser.add_argument(
        "--use_vdl",
        action='store_true',
        default=False,
        help="whether to record the data to Tensorboard.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="/home/aistudio/log",
        help='Tensorboard logging directory for scalar.')
    FLAGS = parser.parse_args()
    main()