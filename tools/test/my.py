import paddle.fluid as fluid

def lenet_5(img):
    '''
    定义神经网络结构
    '''
    conv1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")

    conv1_bn = fluid.layers.batch_norm(input=conv1)

    conv2 = fluid.nets.simple_img_conv_pool(
        input=conv1_bn,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")

    predition = fluid.layers.fc(input=conv2, size=10, act="softmax")
    return predition

# 变量赋值
image = fluid.layers.data(name="img", shape=[1, 28, 28], dtype="float32")
predition = lenet_5(image)

place = fluid.CPUPlace()
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())

# 使用函数 save_inference_model() 保存 paddle 模型
fluid.io.save_inference_model(
    "//home/aistudio/paddle_lenet_5_model",
    feeded_var_names=[image.name],
    target_vars=[predition],
    executor=exe)