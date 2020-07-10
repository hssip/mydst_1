import paddle.fluid as fluid
import numpy as np
# data = fluid.data(name="data", shape=[4,10], dtype='float64')
# target_tensor = fluid.data(name="target_tensor", shape=[-1,20], dtype='float64')
# result = fluid.layers.expand_as(x=data, target_tensor=target_tensor)
# use_cuda = False
# place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
# exe = fluid.Executor(place)
# exe.run(fluid.default_startup_program())
# x = np.random.rand(4,10)
# y = np.random.rand(8,20)
# output= exe.run(feed={"data":x,"target_tensor":y},fetch_list=[result])
# print(output[0].shape)

a = np.array([-np.inf,2,0]).astype('float32')
print(type(a[0]))
