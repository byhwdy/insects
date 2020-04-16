import paddle.fluid as fluid

place = fluid.CPUPlace()

main_program = fluid.Program()
startup_program = fluid.Program()
with fluid.program_guard(main_program=main_program, startup_program=startup_program):
    x = fluid.layers.data(name="x", shape=[-1, 784], dtype='float32')
    y = fluid.layers.data(name="y", shape=[-1, 1], dtype='int32')
    # feeder = fluid.DataFeeder(feed_list=[x, y], place=place)
    z = fluid.layers.fc(input=x, size=10, act="relu")
    loss = fluid.layers.reduce_mean(z)
    optimizer = fluid.optimizer.SGD(learning_rate=0.5)
    optimizer.minimize(loss)

# 打印
print(main_program)

# 保存
dirname1 = './my_model_params'
dirname2 = './my_model_persistables'
dirname3 = './my_model'
dirname4 = './my_full_model/best_model'
exe = fluid.Executor(place=place)
exe.run(startup_program)
fluid.io.save_params(executor=exe,
                     dirname=dirname1,
                     main_program=main_program,
                     filename=None)
fluid.io.save_persistables(executor=exe,
                     dirname=dirname2,
                     main_program=main_program,
                     filename=None)
fluid.io.save_inference_model(dirname=dirname3,
                              feeded_var_names=['x'],
                              target_vars=[z],
                              executor=exe,
                              main_program=main_program)
fluid.save(main_program, dirname4)


