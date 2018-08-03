import os
import sys
import nnvm
from mxnet.gluon.model_zoo.vision import get_model
import tvm
from tvm.contrib import graph_runtime
import numpy as np

def main():
    dirname = sys.argv[1]
    name = 'resnet50_v1'
    block = get_model(name, pretrained=True)
    sym, params = nnvm.frontend.from_mxnet(block)
    sym = nnvm.sym.softmax(sym)
    target = 'llvm -mcpu=skylake-avx512 --system-lib'
    shape_dict = {'data': (1, 3, 224, 224)}
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    with open(os.path.join(dirname, 'net.json'), 'w') as f:
        f.write(graph.json())
    lib.export_library(os.path.join(dirname, 'net.so'))
    with open(os.path.join(dirname, 'net.params'), 'wb') as f:
        f.write(nnvm.compiler.save_param_dict(params))

    print('run model with arr.txt')
    with open('arr.txt', 'r') as f:
        arr = np.array(list(map(lambda s: float(s.replace('\n', '')), f)), dtype='float32').reshape((1, 3, 224, 224))

    ctx = tvm.cpu()
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input('data', tvm.nd.array(arr))
    module.run()
    out_shape = (1, 1000)
    out = module.get_output(0, out=tvm.nd.empty(out_shape))
    out = out.asnumpy().squeeze()
    with open('out_py.txt', 'w') as f:
        for i in range(len(out)):
            f.write('%.12f\n' % out[i])

if __name__ == '__main__':
    main()