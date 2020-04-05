import os
import time
import numpy as np
import datetime
from collections import deque

from paddle import fluid

import sys
sys.path.append('/paddle/insects')

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.reader import create_reader

from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu, check_version
from ppdet.utils.eval_utils import parse_fetches, eval_run, eval_results
from ppdet.utils.stats import TrainingStats
import ppdet.utils.checkpoint as checkpoint

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

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

    # build program
    lr_builder = create('LearningRate')
    optim_builder = create('OptimizerBuilder')
    # print(vars(lr_builder))
    # print(vars(lr_builder.schedulers[0]))
    # print(vars(lr_builder.schedulers[1]))
    # print(optim_builder.regularizer)
    # print(optim_builder.optimizer)
    # exit(0)
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            model = create(main_arch)
            # print(vars(model))
            # print(vars(model.backbone))
            # print(vars(model.yolo_head))
            # print(vars(model.yolo_head.yolo_loss))
            # print(vars(model.yolo_head.nms))
            # exit(0)
            inputs_def = cfg['TrainReader']['inputs_def']
            feed_vars, train_loader = model.build_inputs(**inputs_def)
            train_fetches = model.train(feed_vars)
            loss = train_fetches['loss']
            lr = lr_builder()
            optimizer = optim_builder(lr)
            optimizer.minimize(loss)
    # parse train fetches
    train_keys, train_values, _ = parse_fetches(train_fetches)
    train_values.append(lr)

    if FLAGS.eval:
        eval_prog = fluid.Program()
        with fluid.program_guard(eval_prog, startup_prog):
            with fluid.unique_name.guard():
                model = create(main_arch)
                inputs_def = cfg['EvalReader']['inputs_def']
                feed_vars, eval_loader = model.build_inputs(**inputs_def)
                fetches = model.eval(feed_vars)
        eval_prog = eval_prog.clone(True)
        eval_reader = create_reader(cfg.EvalReader)
        eval_loader.set_sample_list_generator(eval_reader, place)
        # parse eval fetches
        extra_keys = ['gt_bbox', 'gt_class', 'is_difficult']
        eval_keys, eval_values, eval_cls = parse_fetches(fetches, eval_prog,
                                                         extra_keys)


    exe.run(startup_prog)

    # 恢复训练与迁移学习
    ignore_params = cfg.finetune_exclude_pretrained_params \
                 if 'finetune_exclude_pretrained_params' in cfg else []
    start_iter = 0
    if FLAGS.resume_checkpoint:
        checkpoint.load_checkpoint(exe, train_prog, FLAGS.resume_checkpoint)
        start_iter = checkpoint.global_step()
    elif cfg.pretrain_weights:
        checkpoint.load_params(
            exe, train_prog, cfg.pretrain_weights, ignore_params=ignore_params)

    # 数据
    train_reader = create_reader(cfg.TrainReader, cfg.max_iters - start_iter, cfg)
    train_loader.set_sample_list_generator(train_reader, place)

    # whether output bbox is normalized in model output layer
    is_bbox_normalized = False
    # if map_type not set, use default 11point, only use in VOC eval
    map_type = cfg.map_type if 'map_type' in cfg else '11point'
    # 状态
    train_loader.start()
    train_stats = TrainingStats(cfg.log_smooth_window, train_keys)
    start_time = time.time()
    end_time = time.time()
    time_stat = deque(maxlen=cfg.log_smooth_window)
    cfg_name = os.path.basename(FLAGS.config).split('.')[0]
    save_dir = os.path.join(cfg.save_dir, cfg_name)
    best_box_ap_list = [0.0, 0]  #[map, iter]

    # tensorboard
    if FLAGS.use_tb:
        from tb_paddle import SummaryWriter
        tb_writer = SummaryWriter(FLAGS.tb_log_dir)
        tb_loss_step = 0
        tb_mAP_step = 0

    ### 训练循环 ###
    for it in range(start_iter, cfg.max_iters):
        start_time = end_time
        end_time = time.time()
        time_stat.append(end_time - start_time)
        time_cost = np.mean(time_stat)
        eta_sec = (cfg.max_iters - it) * time_cost
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        # 执行
        # outs = exe.run(train_prog, fetch_list=train_values)
        # stats = {k: np.array(v).mean() for k, v in zip(train_keys, outs[:-1])}

        # # tb
        # if FLAGS.use_tb and it % cfg.log_iter == 0:
        #     for loss_name, loss_value in stats.items():
        #         tb_writer.add_scalar(loss_name, loss_value, tb_loss_step)
        #     tb_loss_step += 1

        # # log
        # train_stats.update(stats)
        # logs = train_stats.log()
        # if it % cfg.log_iter == 0:
        #     strs = 'iter: {}, lr: {:.6f}, {}, time: {:.3f}, eta: {}'.format(
        #         it, np.mean(outs[-1]), logs, time_cost, eta)
        #     logger.info(strs)

        ## 验证
        if (it > 0 and it % cfg.snapshot_iter == 0 or it == cfg.max_iters - 1):
            ## 保存模型
            # save_name = str(it) if it != cfg.max_iters - 1 else "model_final"
            # checkpoint.save(exe, train_prog, os.path.join(save_dir, save_name))

            ## 验证
            if FLAGS.eval:
                # evaluation
                results = eval_run(exe, eval_prog, eval_loader,
                                   eval_keys, eval_values, eval_cls)
                box_ap_stats = eval_results(results, cfg.num_classes,
                                            is_bbox_normalized, map_type)
                print(box_ap_stats)
                exit()
                # use tb_paddle to log mAP
                if FLAGS.use_tb:
                    tb_writer.add_scalar("mAP", box_ap_stats[0], tb_mAP_step)
                    tb_mAP_step += 1

                ## 保存最佳模型
                if box_ap_stats[0] > best_box_ap_list[0]:
                    best_box_ap_list[0] = box_ap_stats[0]
                    best_box_ap_list[1] = it
                    checkpoint.save(exe, train_prog,
                                    os.path.join(save_dir, "best_model"))
                logger.info("Best test box ap: {}, in iter: {}".format(
                    best_box_ap_list[0], best_box_ap_list[1]))

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
        "--output_eval",
        default=None,
        type=str,
        help="Evaluation directory, default is current directory.")
    parser.add_argument(
        "--use_tb",
        type=bool,
        default=False,
        help="whether to record the data to Tensorboard.")
    parser.add_argument(
        '--tb_log_dir',
        type=str,
        default="tb_log_dir/scalar",
        help='Tensorboard logging directory for scalar.')
    FLAGS = parser.parse_args()
    main()