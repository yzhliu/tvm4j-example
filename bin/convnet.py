import os
import tvm
import nnvm.compiler
import nnvm.testing
from tvm.contrib import cc, util, graph_runtime

import numpy as np

def test_e2e(target_dir):
    batch_size = 1
    num_class = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_class)

    net, params = nnvm.testing.resnet.get_workload(batch_size=batch_size, image_shape=image_shape)

    with nnvm.compiler.build_config(opt_level=0):
        graph, lib, params = nnvm.compiler.build(
                net, target="llvm", shape={"data": data_shape}, params=params)

    lib.export_library(os.path.join(target_dir, "deploy_lib.so"))
    with open(os.path.join(target_dir, "deploy_graph.json"), "w") as fo:
        fo.write(graph.json())
    with open(os.path.join(target_dir, "deploy_param.params"), "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))

    """
    module = graph_runtime.create(graph, lib, tvm.cpu(0))
    module.set_input(**params)

    input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))
    module.run(data=input_data)
    out = module.get_output(0, out=tvm.nd.empty(out_shape))
    # Print first 10 elements of output
    print(out.asnumpy()[0][0:10])
    """

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(-1)
    test_e2e(sys.argv[1])
