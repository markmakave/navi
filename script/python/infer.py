import tensorrt as trt
import numpy as np
import cv2
import sys

def main():
    logger = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(logger) as builder:
        builder.max_batch_size = 1
        builder.max_workspace_size = 1 << 30
        builder.fp16_mode = True
        builder.strict_type_constraints = True

        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        with open(sys.argv[1], 'rb') as model:
            parser.parse(model.read())

    pass

if __name__ == "__main__":
    main()
