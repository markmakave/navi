import onnx

graph = onnx.helper.make_graph(
    nodes=[
        onnx.helper.make_node(
            "Split",
            inputs=["X"],
            outputs=["R", "G", "B"],
            axis=1,
            num_outputs=3
        ),
        onnx.helper.make_node(
            "Conv",
            inputs=["R", "W"],
            outputs=["RC"]
        ),
        onnx.helper.make_node(
            "Conv",
            inputs=["G", "W"],
            outputs=["GC"]
        ),
        onnx.helper.make_node(
            "Conv",
            inputs=["B", "W"],
            outputs=["BC"]
        ),
        onnx.helper.make_node(
            "Concat",
            inputs=["RC", "GC", "BC"],
            outputs=["Y"],
            axis=1
        ),
    ],
    name="onnx",
    inputs=[
        onnx.helper.make_tensor_value_info(
            "X",
            onnx.TensorProto.UINT8,
            [1, 3, 1080, 1920]
        ),
    ],
    outputs=[
        onnx.helper.make_tensor_value_info(
            "Y",
            onnx.TensorProto.UINT8,
            [None, None, None, None]
        ),
    ],
    initializer=[
        onnx.helper.make_tensor(
            "W",
            onnx.TensorProto.FLOAT,
            [1, 1, 3, 3],
            vals=[-1, -1, -1, -1, 8, -1, -1, -1, -1]
        )
    ]
)

model = onnx.helper.make_model(graph)
onnx.checker.check_model(model)
model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
onnx.save(model, "model.onnx")
