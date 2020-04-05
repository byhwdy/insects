import paddle.fluid as fluid

dir_path = "./my_paddle_model"
file_name = "persistables"
image = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
predict = fluid.layers.fc(input=image, size=10, act='softmax', name='fclayer')

loss = fluid.layers.cross_entropy(input=predict, label=label)
avg_loss = fluid.layers.mean(loss)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
fluid.io.save_persistables(executor=exe, dirname=dir_path, filename=None)
# 网络中fc层中的持久性变量weight和bia将会保存在路径“./my_paddle_model”下名为"persistables"的文件中。