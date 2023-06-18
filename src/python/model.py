import onnx

# create onnx graph
graph = onnx.helper.make_graph(
    [
        onnx.helper.make_node(
            'MatMul',
            inputs=['a', 'b'],
            outputs=['c']
        )
    ],
    'model',
    [
        onnx.helper.make_tensor_value_info('a', onnx.TensorProto.FLOAT, [16, 32]),
        onnx.helper.make_tensor_value_info('b', onnx.TensorProto.FLOAT, [32, 64])
    ],
    [
        onnx.helper.make_tensor_value_info('c', onnx.TensorProto.FLOAT, [None, None])
    ]
)

model = onnx.helper.make_model(graph, producer_name='lumina')
model = onnx.shape_inference.infer_shapes(model)

onnx.checker.check_model(model)
onnx.save(model, 'model.onnx')