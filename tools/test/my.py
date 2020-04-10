import sys
sys.path.append('/paddle/insects')

import paddle.fluid as fluid

boundaries = [100, 200]
lr_steps = [0.1, 0.01, 0.001]
learning_rate = fluid.layers.piecewise_decay(boundaries, lr_steps) #case1, Tensor
#learning_rate = 0.1  #case2, float32
warmup_steps = 50
start_lr = 1. / 3.
end_lr = 0.1
decayed_lr = fluid.layers.linear_lr_warmup(learning_rate,
    warmup_steps, start_lr, end_lr)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
for i in range(300):
	out, = exe.run(fetch_list=[decayed_lr.name])
	print(i, '======', out)
# case1: [0.33333334]
# case2: [0.33333334]